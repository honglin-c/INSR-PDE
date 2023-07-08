import os
from abc import ABC, abstractmethod
import torch
import numpy as np
import random
import time
from tqdm import tqdm
import shutil
from tensorboardX import SummaryWriter
from .networks import get_network


def seed_all(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BaseModel(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.exp_dir = cfg.exp_dir
        self.dt = cfg.dt
        self.max_n_iters = cfg.max_n_iters
        self.sample_resolution = cfg.sample_resolution
        self.vis_resolution = cfg.vis_resolution
        self.timestep = -1

        self.weights_extrapolate = cfg.weights_extrapolate
        self.loss_threshold = 0
        
        self.tb = None
        self.min_lr = 1.1e-8
        self.early_stop_plateau = 500
        self.train_step = 0

        self.device = torch.device("cuda:0")
        seed_all(0)
        self.time_records = []
        self.iter_records = []

    def _create_network(self, input_dim, output_dim):
        return get_network(self.cfg, input_dim, output_dim).to(self.device)

    @property
    @abstractmethod
    def _trainable_networks(self):
        """return a dict of trainable networks"""
        raise NotImplementedError

    @abstractmethod
    def _sample_in_training(self):
        """sampling points in each training step"""
        raise NotImplementedError

    @abstractmethod
    def initialize(self):
        """fit network to initial condition (timestep = 0). NOTE: warp with _timestepping."""
        raise NotImplementedError

    @abstractmethod
    def step(self):
        """step the system by one time step (timestep >= 1). NOTE: warp with _timestepping."""
        raise NotImplementedError
    
    def write_output(self, output_folder):
        """write visulized/discrete output"""
        pass

    def _reset_optimizer(self, max_n_iters, use_scheduler=True, gamma=0.1, patience=500, min_lr=1e-8):
        """create optimizer and scheduler"""
        param_list = []
        for net in self._trainable_networks.values():
            param_list.append({"params": net.parameters(), "lr": self.cfg.lr})
        self.optimizer = torch.optim.Adam(param_list)
        expo_gamma = 0.001 ** (1 / max_n_iters)
        if self.cfg.optim_type == "exp":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=expo_gamma)
        elif self.cfg.optim_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, 
                min_lr=min_lr, patience=patience, verbose=True) if use_scheduler else None

    def _create_tb(self, name, overwrite=True):
        """create tensorboard log"""
        self.log_path = os.path.join(self.cfg.log_dir, name)
        if os.path.exists(self.log_path) and overwrite:
            shutil.rmtree(self.log_path, ignore_errors=True)
        if self.tb is not None:
            self.tb.close()
        self.tb = SummaryWriter(self.log_path)

    def _update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        if self.scheduler is not None:
            if self.cfg.optim_type == "exp":
                self.scheduler.step()
            elif self.cfg.optim_type == "plateau":
                self.scheduler.step(loss_dict["main"])

    def _set_require_grads(self, model, require_grad):
        for p in model.parameters():
            p.requires_grad_(require_grad)
    
    @classmethod
    def _timestepping(cls, func):
        def warp(self):
            self.timestep += 1
            self._create_tb(f"t{self.timestep:03d}")
            func(self)
            self.save_ckpt()
        return warp

    @classmethod
    def _training_loop(cls, func):
        """a decorator function that warps a function inside a training loop

        Args:
            func (_type_): a function that returns a dict of losses, must have key "main".
        """
        tag = func.__name__
        def loop(self, *args, **kwargs):
            max_n_iters = 10000 if self.timestep == 0 else self.max_n_iters
            pbar = tqdm(range(max_n_iters), desc=f"{tag}[{self.timestep}]")
            self._reset_optimizer(max_n_iters)
            min_loss = float("inf")
            accum_steps = 0
            self.train_step = 0
            start = time.time()
            for i in pbar:
                # one gradient descent step
                loss_dict = func(self, *args, **kwargs)
                self._update_network(loss_dict)
                self.train_step += 1

                loss_value = {k: v.item() for k, v in loss_dict.items()}

                self.tb.add_scalars(tag, loss_value, global_step=i)
                pbar.set_postfix(loss_value)

                # optional visualization on tensorboard
                if (i == 0 or (i + 1) % self.cfg.vis_frequency == 0) and hasattr(self, f"_vis{tag}"):
                    vis_func = getattr(self, f"_vis{tag}")
                    vis_func()

                # early stop when converged
                if self.loss_threshold is None:
                    if loss_value["main"] < min_loss:
                        min_loss = loss_value["main"]
                        accum_steps = 0
                    else:
                        accum_steps += 1
                        
                    if self.cfg.early_stop and accum_steps >= 500:
                        tqdm.write(f"early stopping at iteration {i} for not dropping")
                        break
                else:
                    if loss_value["main"] < self.loss_threshold:
                        accum_steps += 1
                    else:
                        accum_steps = 0

                    if self.cfg.early_stop and accum_steps >= 10:
                        tqdm.write(f"early stopping at iteration {i}")
                        break
            end = time.time()
            self.time_records.append(end - start)
            self.iter_records.append(self.train_step)
        return loop

    def save_ckpt(self, name=None):
        """save checkpoint for future restore"""
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
    
    def load_ckpt(self, name):
        """load saved checkpoint"""
        if type(name) is int:
            load_path = os.path.join(self.cfg.model_dir, f"ckpt_step_t{name:03d}.pth")
        else:
            load_path = os.path.join(self.cfg.model_dir, f"ckpt_{name}.pth")
        checkpoint = torch.load(load_path)

        for name, net in self._trainable_networks.items():
            net.load_state_dict(checkpoint[f'net_{name}'])
        self.timestep = checkpoint['timestep']
