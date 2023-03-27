import os
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from networks import get_network
from sources import get_source_velocity
from utils.diff_ops import jacobian, gradient
from utils.model_utils import sample_uniform_2D, sample_random_2D, sample_boundary_separate
from utils.vis_utils import draw_deformation_field2D, save_figure
from collision import collision_plane_force, collision_circle_force

class NeuralElasticity(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg.dt
        self.t_range = cfg.t_range
        self.max_n_iters = cfg.max_n_iters
        self.sample_resolution = cfg.sample_resolution
        self.vis_resolution = cfg.vis_resolution
        self.vis_resolution_time = cfg.vis_resolution_time
        self.timestep = 0
        self.tb = None
        self.sample_pattern = cfg.sample

        self.device = torch.device("cuda:0")
        self.train_step = 0

        # neural implicit network for density, velocity and pressure field
        n_spatial_dims = 2
        n_field_dims = 2
        self.field = get_network(cfg, n_spatial_dims + 1, n_field_dims).to(self.device)

        self.density = cfg.density
        self.ratio_arap = cfg.ratio_arap
        self.ratio_volume = cfg.ratio_volume
        self.gravity_g = cfg.gravity_g
        self.gravity = self.density * torch.tensor([0.0, self.gravity_g]).cuda()

        self.external_force = torch.tensor([cfg.external_force_x, cfg.external_force_y]).cuda()

        self.ratio_init = cfg.ratio_init
        self.ratio_vel_init = cfg.ratio_vel_init
        self.ratio_main = cfg.ratio_main
        self.ratio_bound = cfg.ratio_bound

        self.enable_collision_plane = cfg.enable_collision_plane
        self.ratio_collision = cfg.ratio_collision
        self.plane_height = cfg.plane_height

        self.enable_collision_circle = cfg.enable_collision_circle
        self.circle_center = torch.tensor([cfg.collision_circle_x, cfg.collision_circle_y]).cuda()
        self.circle_radius = cfg.collision_circle_r

        self.enable_bound_top = cfg.enable_bound_top
        self.enable_bound_bottom = cfg.enable_bound_bottom
        self.enable_bound_left = cfg.enable_bound_left
        self.enable_bound_right = cfg.enable_bound_right

    
    @property
    def _trainable_networks(self):
        return {'field': self.field}

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def create_optimizer(self, gamma=0.1, patience=500, min_lr=1e-8):
        param_list = []
        for net in self._trainable_networks.values():
            param_list.append({"params": net.parameters(), "lr": self.cfg.lr})
        self.optimizer = torch.optim.Adam(param_list)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma= 0.001 *  (1. / self.max_n_iters))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma= 0.95 ** 0.0001)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, 
        #     min_lr=min_lr, patience=patience, verbose=True) 

    def create_tb(self, name, overwrite=True):
        """create tensorboard log"""
        self.log_path = os.path.join(self.cfg.log_dir, name)
        if os.path.exists(self.log_path) and overwrite:
            shutil.rmtree(self.log_path, ignore_errors=True)
        return SummaryWriter(self.log_path)

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip > 0:
            param_list = []
            for net in self._trainable_networks.values():
                param_list = param_list + list(net.parameters())
            torch.nn.utils.clip_grad_norm_(param_list, self.cfg.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            # self.scheduler.step(loss_dict['main'])
            self.scheduler.step()

    def _set_require_grads(self, model, require_grad):
        for p in model.parameters():
            p.requires_grad_(require_grad)
    
    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.cfg.model_dir, f"ckpt_step_t{self.timestep:03d}.pth")
        else:
            save_path = os.path.join(self.cfg.model_dir, f"ckpt_{name}.pth")

        save_dict = {}
        for name, net in self._trainable_networks.items():
            save_dict.update({f'net_{name}': net.cpu().state_dict()})
            net.cuda()
        save_dict.update({'timestep': self.timestep})

        torch.save(save_dict, save_path)
    
    def load_ckpt(self, name=None, path=None):
        """load saved checkpoint"""
        if path is not None:
            load_path = path
        else:
            if type(name) is int:
                load_path = os.path.join(self.cfg.model_dir, f"ckpt_step_t{name:03d}.pth")
            else:
                load_path = os.path.join(self.cfg.model_dir, f"ckpt_{name}.pth")
        checkpoint = torch.load(load_path)

        for name, net in self._trainable_networks.items():
            net.load_state_dict(checkpoint[f'net_{name}'])
        self.timestep = checkpoint['timestep']

    def train(self):
        pbar = tqdm(range(self.max_n_iters))
        self.create_optimizer()
        self.tb = self.create_tb("train")
        for i in pbar:
            loss_dict = self._train_step()
            self.update_network(loss_dict)

            loss_value = {k: v.item() for k, v in loss_dict.items()}
            self.tb.add_scalars("loss", loss_value, global_step=i)
            pbar.set_postfix(loss_value)

            if i == 0 or (i + 1) % self.cfg.vis_frequency == 0:
                # visualze on tensorboard
                fig_list = self.visualize()
                for id, fig in enumerate(fig_list):
                    self.tb.add_figure(f"t{id:03d}", fig, global_step=self.train_step)
            
            if i % (self.max_n_iters // 10) == 0:
                self.save_ckpt(str(i))

            self.train_step += 1
            
        self.save_ckpt("final")

    def _train_step(self):
        # initial condition: initial position
        x_init, t_init = self.sample_in_training(is_init=True)
        t_init.requires_grad_(True)
        u_init = self.field(x_init, t_init)

        loss_init = torch.mean(u_init ** 2) * self.ratio_init

        # initial condition: initial velocity
        phi_init = u_init + x_init  # u_init is the initial deformation displacement
        phi_dot_init, _ = jacobian(phi_init, t_init)
        phi_dot_init = torch.squeeze(phi_dot_init)

        loss_vel_init = torch.mean(phi_dot_init ** 2) * self.ratio_vel_init

        # boundary condition 1: fix the points at the top
        if self.enable_bound_top:
            n_bc_samples = x_init.shape[0] // 100
            bc_sample_top = sample_boundary_separate(n_bc_samples, side='top', device=self.device).requires_grad_(True)
            t_bound_top = torch.rand(n_bc_samples, device=self.device).unsqueeze(-1) * self.t_range # (0, t_range)
            u_bound_top = self.field(bc_sample_top, t_bound_top)

            loss_bound_top = torch.mean(u_bound_top ** 2) * self.ratio_bound
        else:
            loss_bound_top = 0.0

        # boundary condition 2: fix the points at the bottom, and move them with offset (0, -2)
        if self.enable_bound_bottom:
            n_bc_samples = x_init.shape[0] // 100
            bc_sample_bottom = sample_boundary_separate(n_bc_samples, side='bottom', device=self.device).requires_grad_(True)
            t_bound_bottom = torch.rand(n_bc_samples, device=self.device).unsqueeze(-1) * self.t_range # (0, t_range)
            u_bound_bottom = self.field(bc_sample_bottom, t_bound_bottom)

            offset_bottom = torch.tensor([0.0, -2.0]).cuda().repeat(n_bc_samples, 1)
            loss_bound_bottom = torch.mean((u_bound_bottom - offset_bottom) ** 2) * self.ratio_bound
        else:
            loss_bound_bottom = 0.0

        # boundary condition 3: fix the points at the left, and move them with offset (-1, 0)
        if self.enable_bound_left:
            n_bc_samples = x_init.shape[0] // 100
            bc_sample_left = sample_boundary_separate(n_bc_samples, side='left', device=self.device).requires_grad_(True)
            t_bound_left = torch.rand(n_bc_samples, device=self.device).unsqueeze(-1) * self.t_range # (0, t_range)
            u_bound_left = self.field(bc_sample_left, t_bound_left)

            loss_bound_left = torch.mean(u_bound_left ** 2) * self.ratio_bound
        else:
            loss_bound_left = 0.0

        # boundary condition 4: fix the points at the right, and move them with offset (2, 0)
        if self.enable_bound_right:
            n_bc_samples = x_init.shape[0] // 100
            bc_sample_right = sample_boundary_separate(n_bc_samples, side='right', device=self.device).requires_grad_(True)
            t_bound_right = torch.rand(n_bc_samples, device=self.device).unsqueeze(-1) * self.t_range # (0, t_range)
            u_bound_right = self.field(bc_sample_right, t_bound_right)

            offset_right = torch.tensor([2.0, 0.0]).cuda().repeat(n_bc_samples, 1)
            loss_bound_right = torch.mean((u_bound_right - offset_right) ** 2) * self.ratio_bound
        else:
            loss_bound_right = 0.0

        # pde residual
        x_main, t_main = self.sample_in_training(is_init=False)
        x_main.requires_grad_(True)
        t_main.requires_grad_(True)
        u_main = self.field(x_main, t_main)

        phi = u_main + x_main  # u_main is the deformation displacement
        jac_x, _ = jacobian(phi, x_main) # (N, 2, 2)
        U_x, S_x, V_x = torch.svd(jac_x)
        psi = self.ratio_arap * torch.sum((S_x - 1.0) ** 2) + self.ratio_volume * torch.sum((torch.prod(S_x, dim=1) - 1.0) ** 2) 

        phi_dot, _ = jacobian(phi, t_main)
        phi_dot = torch.squeeze(phi_dot)

        phi_dot_dot, _ = jacobian(phi_dot, t_main)
        phi_dot_dot = torch.squeeze(phi_dot_dot)
        dpsi_dphi = gradient(psi, x_main)

        external_force = self.gravity.repeat(u_main.shape[0], 1) + self.external_force.repeat(u_main.shape[0], 1)
        if self.enable_collision_plane:
            collide_force = collision_plane_force(phi, self.ratio_collision, self.plane_height)
            external_force += collide_force

        if self.enable_collision_circle:
            collide_force = collision_circle_force(phi, self.ratio_collision, self.circle_center, self.circle_radius)
            external_force += collide_force

        # loss_main = torch.mean((self.density * phi_dot_dot) ** 2)
        # loss_main = torch.mean((self.density * phi_dot - self.density * external_force) ** 2)
        # loss_main = torch.mean((self.density * phi_dot_dot - self.density * external_force) ** 2)
        loss_main = torch.mean((self.density * phi_dot_dot + dpsi_dphi - self.density * external_force) ** 2) * self.ratio_main

        # # collision 
        # if self.enable_collision:
        #     loss_collision = collision_plane_loss(phi, phi_dot, self.ratio_collision, self.plane_height)
        # else:
        #     loss_collision = 0.0

        if self.enable_bound_top and self.enable_bound_bottom:
            loss_dict = {"init": loss_init, "init_vel": loss_vel_init, "bound": loss_bound_top, "bound_bottom": loss_bound_bottom, "main": loss_main}
        elif self.enable_bound_left and self.enable_bound_right:
            loss_dict = {"init": loss_init, "init_vel": loss_vel_init, "bound_left": loss_bound_left, "bound_right": loss_bound_right, "main": loss_main}
        else:
            loss_dict = {"init": loss_init, "init_vel": loss_vel_init, "main": loss_main}
        return loss_dict

    def sample_in_training(self, is_init=False):
        if self.sample_pattern == 'random':
            samples = sample_random_2D(self.sample_resolution ** 2, device=self.device).requires_grad_(True)
            if is_init:
                time = torch.zeros(self.sample_resolution ** 2, device=self.device).unsqueeze(-1)
            else:
                time = torch.rand(self.sample_resolution ** 2, device=self.device).unsqueeze(-1) * self.t_range # (0, t_range)
            return samples, time
        elif self.sample_pattern == 'fixed':
            n = 40960000
            if not hasattr(self, "pre_samples"):
                self.pre_samples = sample_random_2D(n, device=self.device)
                self.pre_samples_t = torch.rand(n, device=self.device).unsqueeze(-1) * self.t_range
                self.pre_samples_init = sample_random_2D(n, device=self.device)
            indices = torch.randint(0, n - 1, size=(self.sample_resolution ** 2, ), device=self.device)
            if is_init:
                return self.pre_samples_init[indices], None
            else:
                return self.pre_samples[indices], self.pre_samples_t[indices]
        else:
            raise NotImplementedError


###################################  Visualization ###################################
    def sample_in_visualization(self, resolution, t_count = 10, sample_boundary = True):
        samples = sample_uniform_2D(resolution, device=self.device).reshape(resolution**2, 2)
        time = torch.linspace(0, 1.0, t_count, device=self.device).unsqueeze(-1) * self.t_range
        return samples, time
    

    def visualize(self):
        x_vis, t_vis = self.sample_in_visualization(self.vis_resolution, self.vis_resolution_time)
        fig_list = []
        for i, t_i in enumerate(t_vis):
            t_vis_i = t_i * torch.ones(x_vis.shape[0], 1, device=self.device)
            u_vis = self.field(x_vis, t_vis_i)
            phi_vis = (u_vis + x_vis).detach().cpu().numpy()
            if self.enable_collision_plane:
                fig = draw_deformation_field2D(phi_vis, plane_height=self.plane_height, hide_axis=False)
            elif self.enable_collision_circle:
                fig = draw_deformation_field2D(phi_vis, circle_center=self.circle_center, circle_radius=self.circle_radius, hide_axis=False)
            else:
                fig = draw_deformation_field2D(phi_vis, hide_axis = False)
            self.write_output(fig, self.cfg.results_dir, t = t_i.detach().cpu().numpy()[0])
            fig_list.append(fig)
        return fig_list


    def write_output(self, fig, output_folder, t = 0):
        save_path = os.path.join(output_folder, f"t{t:03f}_deformation.png")
        save_figure(fig, save_path)

    def render_output(self, fig, output_folder, t = 0):
        save_path = os.path.join(output_folder, f"t{t:03f}_deformation.png")
        plt.savefig(save_path, dpi=450, transparent=True)
