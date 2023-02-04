import os
from tqdm import tqdm
import random
import numpy as np
import torch
from config import Config
from base import gradient, sample_uniform, sample_boundary2D_separate
from base import divergence, laplace


cfg = Config("recap")

if cfg.pde == "fluid":
    from fluid import Fluid2DModel as neuralModel
else:
    raise NotImplementedError


def advect_velocity(model):
    """velocity advection: dudt = -(u\cdot grad)u"""
    samples = model._sample_in_training()

    # dudt
    with torch.no_grad():
        prev_u = model.velocity_field_prev(samples).detach()
    curr_u = model.velocity_field(samples)

    # backtracking
    backtracked_position = samples - prev_u * model.cfg.dt
    backtracked_position = torch.clamp(backtracked_position, min=-1.0, max=1.0)
    
    with torch.no_grad():
        advected_u = model.velocity_field_prev(backtracked_position).detach()

    loss = torch.mean((curr_u - advected_u) ** 2)
    loss_dict = {'main': loss}

    # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
    #        and fixed factor 1.0 for boundary loss
    bc_sample_x = sample_boundary2D_separate(samples.shape[0] // 100, side='horizontal', device=model.device).requires_grad_(True)
    bc_sample_y = sample_boundary2D_separate(samples.shape[0] // 100, side='vertical', device=model.device).requires_grad_(True)
    vel_x = model.velocity_field(bc_sample_x)[..., 0]
    vel_y = model.velocity_field(bc_sample_y)[..., 1]
    bc_loss = (torch.mean(vel_x ** 2) + torch.mean(vel_y ** 2)) * 1.0
    loss_dict.update({"bc": bc_loss})

    return loss_dict


def solve_pressure(model):
    """solve pressure: div u = lap P."""
    samples = model._sample_in_training()

    out_u = model.velocity_field(samples)
    div_u = divergence(out_u, samples).detach()
    out_p = model.pressure_field(samples)
    lap_p = laplace(out_p, samples)

    loss = torch.mean((div_u - lap_p) ** 2) # FIXME: assume rho=1 here
    loss_dict = {'main': loss}

    # NOTE: neumann boundary condition, grad(p)\cdot norm(p) = 0
    bc_sample_x = sample_boundary2D_separate(model.sample_resolution ** 2 // 100, side='horizontal', device=model.device).requires_grad_(True)
    bc_sample_y = sample_boundary2D_separate(model.sample_resolution ** 2 // 100, side='vertical', device=model.device).requires_grad_(True)
    grad_px = gradient(model.pressure_field(bc_sample_x), bc_sample_x)[..., 0]
    grad_py = gradient(model.pressure_field(bc_sample_y), bc_sample_y)[..., 1]

    bc_loss = torch.mean(grad_px ** 2) + torch.mean(grad_py ** 2)
    loss_dict.update({'bc': bc_loss})

    return loss_dict


def projection(model):
    """velocity projection: u <- u - grad(p)"""
    samples = model._sample_in_training()

    with torch.no_grad():
        prev_u = model.velocity_field_prev(samples).detach()
    
    p = model.pressure_field(samples)
    grad_p = gradient(p, samples).detach()

    target_u = prev_u - grad_p
    curr_u = model.velocity_field(samples)
    loss = torch.mean((curr_u - target_u) ** 2)
    loss_dict = {'main': loss}

    # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
    #        and fixed factor 1.0 for boundary loss
    bc_sample_x = sample_boundary2D_separate(samples.shape[0] // 100, side='horizontal', device=model.device).requires_grad_(True)
    bc_sample_y = sample_boundary2D_separate(samples.shape[0] // 100, side='vertical', device=model.device).requires_grad_(True)
    vel_x = model.velocity_field(bc_sample_x)[..., 0]
    vel_y = model.velocity_field(bc_sample_y)[..., 1]
    bc_loss = (torch.mean(vel_x ** 2) + torch.mean(vel_y ** 2)) * 1.0
    loss_dict.update({"bc": bc_loss})
    return loss_dict


def seed_all(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimize_one_timestep(model, train_func, max_n_iters):
    seed_all(0)
    model._reset_optimizer()
    pbar = tqdm(range(max_n_iters))
    loss_list = []
    for i in pbar:
        loss_dict = train_func(model)
        model._update_network(loss_dict)
        
        loss_value = {k: v.item() for k, v in loss_dict.items()}
        pbar.set_postfix(loss_value)
        loss_list.append(loss_value["main"])
    return loss_list


def extrapolate_weights(net, ckpt_path, prev_ckpt_path, attr):
    checkpoint = torch.load(ckpt_path)
    checkpoint_prev = torch.load(prev_ckpt_path)
    if attr == "velocity":
        net_state = checkpoint["net_velocity"]
        net_prev_state = checkpoint_prev["net_velocity"]
    elif attr == "pressure":
        net_state = checkpoint["net_pressure"]
        net_prev_state = checkpoint_prev["net_pressure"]
    else:
        raise NotImplementedError

    # weights_dict = {}
    # for name, param in net_prev_state.items():
    #     weights_dict[name] = param.data
    net.cpu()
    for name, param in net.named_parameters():
        param.data = net_state[name] * 2 - net_prev_state[name]
        # param.data = param.data * 2 - net_prev_state[name]
    net.cuda()
    # return model


def run_one_timestep(cfg, t, max_n_iters, save_dir, extrapolation=False):
    model = neuralModel(cfg)
    model.load_ckpt(f"t{t:03d}_projection")

    model.velocity_field_prev.load_state_dict(model.velocity_field.state_dict())
    if extrapolation:
        ckpt_path = os.path.join(cfg.model_dir, f"ckpt_t{t:03d}_advect.pth")
        prev_ckpt_path = os.path.join(cfg.model_dir, f"ckpt_t{t - 1:03d}_advect.pth")
        extrapolate_weights(model.velocity_field, ckpt_path, prev_ckpt_path, "velocity")
    adv_loss_list = optimize_one_timestep(model, advect_velocity, max_n_iters)
    print("advection:", adv_loss_list[0], adv_loss_list[-1])

    if extrapolation:
        ckpt_path = os.path.join(cfg.model_dir, f"ckpt_t{t:03d}_pressure.pth")
        prev_ckpt_path = os.path.join(cfg.model_dir, f"ckpt_t{t - 1:03d}_pressure.pth")
        extrapolate_weights(model.pressure_field, ckpt_path, prev_ckpt_path, "pressure")
    pre_loss_list = optimize_one_timestep(model, solve_pressure, max_n_iters)
    print("pressure:", pre_loss_list[0], pre_loss_list[-1])

    model.velocity_field_prev.load_state_dict(model.velocity_field.state_dict())
    if extrapolation:
        ckpt_path = os.path.join(cfg.model_dir, f"ckpt_t{t:03d}_projection.pth")
        prev_ckpt_path = os.path.join(cfg.model_dir, f"ckpt_t{t - 1:03d}_projection.pth")
        extrapolate_weights(model.velocity_field, ckpt_path, prev_ckpt_path, "velocity")
    proj_loss_list = optimize_one_timestep(model, projection, max_n_iters)
    print("projection:", proj_loss_list[0], proj_loss_list[-1])

    for name, vlist in zip(["advect", "pressure", "projection"], [adv_loss_list, pre_loss_list, proj_loss_list]):
        save_path = os.path.join(save_dir, f"loss_{name}.txt")
        fp = open(save_path, "w")
        for v in vlist[::10]:
            print(v, file=fp)
        fp.close()


test_ts = [2, 10, 50]
init_lrs = [1e-4]
# test_ts = [10]
# init_lrs = [1e-4]
n_train_iters = 10000
save_dir = "extra_weights_fluid"
os.makedirs(save_dir, exist_ok=True)


for lr in init_lrs:
    cfg.lr = lr
    for t in test_ts:
        print(f"t={t}, lr={cfg.lr}")

        # baseline
        # model = neuralModel(cfg)
        # model.load_ckpt(t)
        
        run_save_dir = os.path.join(save_dir, f"t{t}lr{cfg.lr}_expo_base")
        os.makedirs(run_save_dir, exist_ok=True)
        # # run_one_timestep(model, n_train_iters, run_save_dir)
        run_one_timestep(cfg, t, n_train_iters, run_save_dir, extrapolation=False)


        # use extrapolation
        # model = neuralModel(cfg)
        # model.load_ckpt(t)

        # model_prev = neuralModel(cfg)
        # model_prev.load_ckpt(t - 1)

        # model = extrapolate_weights(model, model_prev)

        run_save_dir = os.path.join(save_dir, f"t{t}lr{cfg.lr}_expo_extra")
        os.makedirs(run_save_dir, exist_ok=True)
        # run_one_timestep(model, n_train_iters, run_save_dir)
        run_one_timestep(cfg, t, n_train_iters, run_save_dir, extrapolation=True)
