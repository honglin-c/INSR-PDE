import os
from tqdm import tqdm
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import Config
from base import gradient, sample_uniform, sample_boundary


cfg = Config("recap")

if cfg.pde == "advection":
    from advection import Advection1DModel as neuralModel
else:
    raise NotImplementedError


def compute_loss(model, uniform=False):
    if uniform:
        samples = sample_uniform(model.sample_resolution, 1, device=model.device) * model.length / 2
    else:
        samples = model._sample_in_training()
    samples.requires_grad_(True)

    prev_u = model.field_prev(samples)
    curr_u = model.field(samples)
    dudt = (curr_u - prev_u) / model.dt # (N, sdim)

    # midpoint time integrator
    grad_u = gradient(curr_u, samples)
    grad_u0 = gradient(prev_u, samples).detach()
    loss = torch.mean((dudt + model.vel * (grad_u + grad_u0) / 2.) ** 2)
    loss_dict = {'main': loss}

    boundary_samples = sample_boundary(max(model.sample_resolution // 100, 10), 1, device=model.device) * model.length / 2
    bound_u = model.field(boundary_samples)
    bc_loss = torch.mean(bound_u ** 2) * 1.
    loss_dict.update({'bc': bc_loss})
    return loss_dict


def compute_abs_pde_residual(model, resolution=500):
    samples = sample_uniform(model.sample_resolution, 1, device=model.device) * model.length / 2
    samples.requires_grad_(True)
    
    prev_u = model.field_prev(samples)
    curr_u = model.field(samples)
    dudt = (curr_u - prev_u) / model.dt # (N, sdim)

    # midpoint time integrator
    grad_u = gradient(curr_u, samples)
    grad_u0 = gradient(prev_u, samples).detach()

    pde_res = torch.abs((dudt + model.vel * (grad_u + grad_u0) / 2.)).mean().item()
    return pde_res


def extrapolate_weights(model, model_prev):
    weights_dict = {}
    for name, param in model_prev.field.named_parameters():
        weights_dict[name] = param.data

    for name, param in model.field.named_parameters():
        param.data = param.data * 2 - weights_dict[name]
    return model


def seed_all(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimize_one_timestep(model, max_n_iters):
    seed_all(0)
    model._reset_optimizer(expo_gamma=expo_gamma)
    pbar = tqdm(range(max_n_iters))
    loss_list = []
    pde_res_list = []
    for i in pbar:
        pde_res = compute_abs_pde_residual(model)
        pde_res_list.append(pde_res)

        loss_dict = compute_loss(model, uniform=False)
        model._update_network(loss_dict)
        
        loss_value = {k: v.item() for k, v in loss_dict.items()}
        pbar.set_postfix(loss_value)
        loss_list.append(loss_value["main"])

    return loss_list, pde_res_list


# test_ts = [2, 10, 50]
# init_lrs = [1e-4, 1e-5]
test_ts = [10]
init_lrs = [1e-4, 1e-5]
n_train_iters = 5000
expo_gamma = 0.999
save_dir = "extra_weights_adv2"
os.makedirs(save_dir, exist_ok=True)


for lr in init_lrs:
    cfg.lr = lr
    for t in test_ts:
        print(f"t={t}, lr={cfg.lr}")

        # baseline
        model = neuralModel(cfg)
        model.load_ckpt(t)
        model.field_prev.load_state_dict(model.field.state_dict())
        # loss = compute_loss(model, uniform=True)
        # print("original init loss:", loss.item())

        loss_list_base, pde_res_list_base = optimize_one_timestep(model, n_train_iters)
        print("baseline loss:", loss_list_base[0], loss_list_base[-1])
        print("baseline pde residual:", pde_res_list_base[0], pde_res_list_base[-1])


        # use extrapolation
        model = neuralModel(cfg)
        model.load_ckpt(t)
        model.field_prev.load_state_dict(model.field.state_dict())

        model_prev = neuralModel(cfg)
        model_prev.load_ckpt(t - 1)

        model = extrapolate_weights(model, model_prev)
        # loss = compute_loss(model, uniform=True)
        # print("extrapolate init loss:", loss.item())

        loss_list_ext, pde_res_list_ext = optimize_one_timestep(model, n_train_iters)
        print("extrapolate loss:", loss_list_ext[0], loss_list_ext[-1])
        print("extrapolate pde residual:", pde_res_list_ext[0], pde_res_list_ext[-1])


        fp = open(os.path.join(save_dir, f"t{t}_loss_base_lr{lr}.txt"), "w")
        for v in loss_list_base[::10]:
            print(v, file=fp)
        fp.close()
        
        fp = open(os.path.join(save_dir, f"t{t}_loss_ext_lr{lr}.txt"), "w")
        for v in loss_list_ext[::10]:
            print(v, file=fp)
        fp.close()

        fp = open(os.path.join(save_dir, f"t{t}_pde_base_lr{lr}.txt"), "w")
        for v in pde_res_list_base[::10]:
            print(v, file=fp)
        fp.close()
        
        fp = open(os.path.join(save_dir, f"t{t}_pde_ext_lr{lr}.txt"), "w")
        for v in pde_res_list_ext[::10]:
            print(v, file=fp)
        fp.close()
