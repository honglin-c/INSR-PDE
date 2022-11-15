import torch
import torch.nn.functional as F
from .base import BaseModel
from .networks import get_network
from .sampling import sample_boundary
from utils.ops_utils import gradient


class Advection1DModel(BaseModel):
    """advection equation with constant velocity"""
    def __init__(self, cfg):
        super().__init__(cfg)

    def _define_networks(self):
        self.field = get_network(self.cfg, 1, 1).to(self.device)
        self.field_prev = get_network(self.cfg, 1, 1).to(self.device)
        self._set_require_grads(self.field_prev, False)

    @property
    def _trainable_networks(self):
        return {"field": self.field}
    
    def _sample_in_training(self):
        pass
    
    def initialize(self):
        self._create_tb("init")
        self._initialize()
        self.save_ckpt("init")

    @BaseModel._training_loop
    def _initialize(self):
        """forward computation for initialization"""
        samples = self._sample_in_training()
        ref = self.init_cond(samples)
        out = self.field(samples)
        loss_random = F.mse_loss(out, ref)

        loss_dict = {'main': loss_random}
        return loss_dict

    def step(self):
        """advection: dudt = -(vel \cdot grad)u"""
        self.timestep += 1
        self.field_prev.load_state_dict(self.field.state_dict())
        self._create_tb("step")
        self._advect()
        self.save_ckpt()

    @BaseModel._training_loop
    def _advect(self):
        """forward computation for advect"""
        samples = self._sample_in_training()

        prev_u = self.field_prev(samples)
        curr_u = self.field(samples)
        dudt = (curr_u - prev_u) / self.dt # (N, sdim)

        # midpoint time integrator
        grad_u = gradient(curr_u, samples)
        grad_u0 = gradient(prev_u, samples).detach()
        loss = torch.mean((dudt + self.vel * (grad_u + grad_u0) / 2.) ** 2)
        loss_dict = {'main': loss}

        # Dirichlet boundary constraint
        # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
        #        and fixed factor 1.0 for boundary loss
        boundary_samples = sample_boundary(max(self.sample_resolution ** self.sdim // 100, 10), self.sdim, self.length)
        bound_u = self.field(boundary_samples)
        bc_loss = torch.mean(bound_u ** 2) * 1.
        loss_dict.update({'bc': bc_loss})

        return loss_dict
