import os
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import shutil
from tensorboardX import SummaryWriter
from networks import get_network
from sources import get_source_velocity
from utils.diff_ops import jacobian, gradient
from utils.model_utils import sample_uniform_2D, sample_random_2D, sample_boundary_separate
from utils.vis_utils import draw_deformation_field2D, save_figure


class NeuralElasticity(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg.dt
        self.t_range = cfg.t_range
        self.max_n_iters = cfg.max_n_iters
        self.sample_resolution = cfg.sample_resolution
        self.vis_resolution = cfg.vis_resolution
        self.timestep = 0
        self.tb = None
        self.sample_pattern = cfg.sample

        self.device = torch.device("cuda:0")

        # neural implicit network for density, velocity and pressure field
        n_spatial_dims = 2
        n_field_dims = 2
        self.field = get_network(cfg, n_spatial_dims + 1, n_field_dims).to(self.device)

        self.rho = 1.0
        self.ratio_arap = 1.0
        self.ratio_volume = 1.0
        self.external_force = torch.tensor([0.0, -1.0]).cuda()

    
    @property
    def _trainable_networks(self):
        return {'field': self.field}

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def create_optimizer(self):
        param_list = []
        for net in self._trainable_networks.values():
            param_list.append({"params": net.parameters(), "lr": self.cfg.lr})
        self.optimizer = torch.optim.Adam(param_list)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95 ** 0.0001)

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
            torch.nn.utils.clip_grad_norm_(param_list, 0.1)
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
            
        self.save_ckpt("final")

    def _train_step(self):
        # initial condition
        x_init, t_init = self.sample_in_training(is_init=True)
        u_init = self.field(x_init, t_init)
        loss_init = torch.mean(u_init ** 2)

        # boundary condition
        n_bc_samples = x_init.shape[0] // 100
        bc_sample_x = sample_boundary_separate(n_bc_samples, side='horizontal', device=self.device).requires_grad_(True)
        bc_sample_y = sample_boundary_separate(n_bc_samples, side='vertical', device=self.device).requires_grad_(True)
        t_bound = torch.rand(n_bc_samples, device=self.device).unsqueeze(-1) * self.t_range # (0, t_range)

        loss_bound = 0.0

        # pde residual
        x_main, t_main = self.sample_in_training(is_init=False)
        x_main.requires_grad_(True)
        t_main.requires_grad_(True)
        u_main = self.field(x_main, t_main)

        phi = u_main + x_main  # u_main is the deformation displacement
        jac_x, _ = jacobian(phi, x_main) # (N, 2, 2)
        U_x, S_x, V_x = torch.svd(jac_x)
        psi = self.ratio_arap * torch.sum((S_x - 1.0) ** 2) + self.ratio_volume * torch.sum((torch.prod(S_x, dim=1) - 1) ** 2) 

        phi_dot_dot = gradient(gradient(phi, t_main), t_main)
        dpsi_dphi = gradient(psi, x_main)

        loss_main = torch.sum((self.rho * phi_dot_dot + dpsi_dphi - self.rho * self.external_force) ** 2)

        loss_dict = {"init": loss_init, "bound": loss_bound, "main": loss_main}
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


################################### 
    def sample_in_visualization(self, resolution, sample_boundary = True):
        samples = sample_uniform_2D(resolution, device=self.device)
        time = torch.linspace(0, self.t_range, self.t_range+1, device=self.device).unsqueeze(-1).unsqueeze(-1)
        return samples, time


    def visualize(self):
        x_vis, t_vis = self.sample_in_visualization(self.vis_resolution)
        u_vis = self.field(x_vis, t_vis)
        fig_list = []
        for i, t_i in enumerate(t_vis):
            fig = draw_deformation_field2D(x_vis)
            self.write_output(fig, self.cfg.results_dir)
            fig_list.append(fig)
        return fig_list

    def write_output(self, fig, output_folder):
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_deformation.png")
        save_figure(fig, save_path)