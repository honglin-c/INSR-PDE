import os
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel, gradient, sample_random, sample_uniform, sample_boundary
from .examples import get_examples
from .visualize import scatter_signal1D, save_figure, draw_scalar_field2D, scatter_signal2D


class AdvectionNDModel(BaseModel):
    """advection equation with constant velocity"""
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.length = cfg.length
        self.dim = cfg.dim
        self.vel = cfg.vel * torch.ones(self.dim).cuda()
        self.vis_dim = cfg.vis_dim if cfg.vis_dim > 0 else cfg.dim
        self.vis_resolution = cfg.vis_resolution

        if self.dim == 2:        
            self.vis_samples = sample_uniform(self.vis_resolution, self.dim, device=self.device) * self.length / 2
        else:
            self.vis_samples = sample_random(self.vis_resolution ** self.dim, self.dim, device=self.device) * self.length / 2

        self.field = self._create_network(self.dim, 1)
        self.field_prev = self._create_network(self.dim, 1)
        self._set_require_grads(self.field_prev, False)

    @property
    def _trainable_networks(self):
        return {"field": self.field}
    
    def _sample_in_training(self):
        return sample_random(self.sample_resolution ** self.dim, self.dim, device=self.device).requires_grad_(True) * self.length / 2

    def sample_field(self, resolution, return_samples=False, to_squeeze = True, is_uniform = True, use_preset = False):
        """sample current field with uniform grid points"""
        if use_preset:
            grid_samples = self.vis_samples
        else:
            if self.dim > 3:
                is_uniform = False
            if is_uniform:
                grid_samples = sample_uniform(resolution, self.dim, device=self.device) * self.length / 2
            else:
                grid_samples = sample_random(self.sample_resolution ** self.dim, self.dim, device=self.device) * self.length / 2
            
        if to_squeeze:
            out = self.field(grid_samples).squeeze(-1)
        else:
            out = self.field(grid_samples)
        if return_samples:
            if to_squeeze:
                return out, grid_samples.squeeze(-1)
            else:
                return out, grid_samples
        return out

    @BaseModel._timestepping
    def initialize(self):
        if not hasattr(self, "init_cond_func"):
            self.init_cond_func = get_examples(self.cfg.init_cond, sdim=self.dim)
        self._initialize()

    @BaseModel._training_loop
    def _initialize(self):
        """forward computation for initialization"""
        samples = self._sample_in_training()
        ref = self.init_cond_func(samples)
        out = self.field(samples)
        loss_random = F.mse_loss(out, ref)

        loss_dict = {'main': loss_random}

        # # Dirichlet boundary constraint
        # # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
        # #        and fixed factor 1.0 for boundary loss
        # boundary_samples = sample_boundary(max(self.sample_resolution**self.dim // 100, 10), self.dim, device=self.device) * self.length / 2
        # bound_u = self.field(boundary_samples)
        # bc_loss = torch.mean(bound_u ** 2) * 1.
        # loss_dict.update({'bc': bc_loss})

        return loss_dict
    
    def _vis_initialize(self):
        """visualization on tb during training"""
        values, samples = self.sample_field(self.vis_resolution, return_samples=True, to_squeeze=False, use_preset=True)
        values = values.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        if self.vis_dim == 2:
            fig = scatter_signal2D(samples, color=values)
        else:
            fig = scatter_signal1D(samples[:,-1], values[:,-1], y_max=1.0)
        self.tb.add_figure("field", fig, global_step=self.train_step)

    @BaseModel._timestepping
    def step(self):
        """advection: dudt = -(vel \cdot grad)u"""
        self.field_prev.load_state_dict(self.field.state_dict())
        self._advect()

    @BaseModel._training_loop
    def _advect(self):
        """forward computation for advect"""
        samples = self._sample_in_training()

        prev_u = self.field_prev(samples).squeeze(-1)
        curr_u = self.field(samples).squeeze(-1)
        dudt = (curr_u - prev_u) / self.dt # (N, 1)

        # midpoint time integrator
        grad_u = gradient(curr_u, samples)
        grad_u0 = gradient(prev_u, samples).detach()

        # loss = torch.mean((dudt + torch.sum(self.vel * (grad_u + grad_u0) / 2., dim=-1, keepdim=True)) ** 2)
        loss = torch.mean((dudt + torch.sum(self.vel.unsqueeze(0) * (grad_u + grad_u0) / 2., dim=-1)) ** 2)
        loss_dict = {'main': loss}

        # Dirichlet boundary constraint
        # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
        #        and fixed factor 1.0 for boundary loss
        boundary_samples = sample_boundary(max(self.sample_resolution**self.dim // 100, 10), self.dim, device=self.device) * self.length / 2
        bound_u = self.field(boundary_samples)
        bc_loss = torch.mean(bound_u ** 2) * 1.
        loss_dict.update({'bc': bc_loss})

        return loss_dict


    def _compute_groundtruth(self, samples):
        '''compute the groundtruth using method of characteristics'''
        t = self.timestep * self.dt
        values_gt = self.init_cond_func(self.vis_samples - t * self.vel)
        return values_gt


    def _vis_advect(self):
        """visualization on tb during training"""
        values, samples = self.sample_field(self.vis_resolution, return_samples=True, to_squeeze=False, use_preset=True)

        values_gt = self._compute_groundtruth(samples).cpu().numpy()
        
        values = values.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        if self.vis_dim == 2:
            fig = scatter_signal2D(samples, color=values)
        elif self.vis_dim == 1:
            fig = scatter_signal1D(samples, values, y_max=1.0)
        else:
            fig = scatter_signal2D(samples[:,:-2], color=values[:,:-2])
        # else:
        #     fig = scatter_signal1D(samples[:,-1], values[:,-1], y_max=1.0)
        self.tb.add_figure("field", fig, global_step=self.train_step)

        values_error = np.linalg.norm(values-values_gt, axis=-1)
        if self.vis_dim == 2:
            fig = scatter_signal2D(samples, color=values_error)
        elif self.vis_dim == 1:
            fig = scatter_signal1D(samples, values_error, y_max=1.0)
        else:
            fig = scatter_signal2D(samples[:,:-2], color=values_error[:,:-2])
        # else:
        #     fig = scatter_signal1D(samples[:,-1], values_error[:,-1], y_max=1.0)
        self.tb.add_figure("error_field", fig, global_step=self.train_step)


    def write_output(self, output_folder):
        values, samples = self.sample_field(self.vis_resolution, return_samples=True, to_squeeze=False, use_preset=True)

        values_gt = self._compute_groundtruth(samples).cpu().numpy()
        values = values.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        if self.vis_dim == 2:
            fig = scatter_signal2D(samples, color=values)
        elif self.vis_dim == 1:
            fig = scatter_signal1D(samples, values, y_max=1.0)
        else:
            fig = scatter_signal2D(samples[:,:-2], color=values[:,:-2])

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}.png")
        save_figure(fig, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}.npy")
        np.savez(save_path, values)

        values_error = np.linalg.norm(values-values_gt, axis=-1)
        if self.vis_dim == 2:
            fig = scatter_signal2D(samples, color=values_error)
        elif self.vis_dim == 1:
            fig = scatter_signal1D(samples, values_error, y_max=1.0)
        else:
            fig = scatter_signal2D(samples[:,:-2], color=values_error[:,:-2])

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_error.png")
        save_figure(fig, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_samples.npy")
        np.savez(save_path, samples)
        
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_error.npy")
        np.savez(save_path, values_error)

        if self.vis_dim == 2:
            fig = scatter_signal2D(samples, color=values_gt)
        elif self.vis_dim == 1:
            fig = scatter_signal1D(samples, values_gt, y_max=1.0)
        else:
            fig = scatter_signal2D(samples[:,:-2], color=values_gt[:,:-2])

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_gt.png")
        save_figure(fig, save_path)

        