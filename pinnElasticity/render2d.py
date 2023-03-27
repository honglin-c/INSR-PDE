import os
import json
import torch
import torch.nn.functional as F
from config import Config
from model import NeuralElasticity
from utils.file_utils import ensure_dir
from utils.vis_utils import draw_deformation_field2D, save_figure
import imageio
import numpy as np

def main_deformation_test():
    # create experiment config containing all hyperparameters
    cfg = Config()

    # create network and training agent
    model = NeuralElasticity(cfg)

    def texture_mapping(points):
      im = imageio.imread('/home/honglinchen/projects/NeuralImplicitSimulation/neuralARAP/data/checkerboard-texture-trimmed.png')
      pts_max = np.max(points, axis=0)
      pts_min = np.min(points, axis=0)
      print(pts_max)
      print(pts_min)
      row_i = np.floor((points[:,0] - pts_min[0]) / (pts_max[0] - pts_min[0]) * (im.shape[0]-1)).astype(int)
      col_i = np.floor((points[:,1] - pts_min[1]) / (pts_max[1] - pts_min[1]) * (im.shape[1]-1)).astype(int)  
      colors = im[row_i, col_i, 0:3]
      colors = colors / 255
      return colors
    
    x_vis, t_vis = model.sample_in_visualization(model.vis_resolution, 1)

    samples_test = x_vis.detach().cpu().numpy()
    colors = texture_mapping(samples_test)

    model_dir = cfg.model_dir
    # filename = 'ckpt_2000.pth'
    filename = 'ckpt_final.pth'

    model_path = os.path.join(model_dir, filename)
    print(model_path)
    model.load_ckpt(path=model_path)

    output_dir = os.path.join(cfg.exp_dir, f'results_render_{filename}/')
    ensure_dir(output_dir)

    x_vis, t_vis = model.sample_in_visualization(model.vis_resolution, 6)
    fig_list = []
    for i, t_i in enumerate(t_vis):
        t_vis_i = t_i * torch.ones(x_vis.shape[0], 1, device=model.device)
        u_vis = model.field(x_vis, t_vis_i)
        phi_vis = (u_vis + x_vis).detach().cpu().numpy()
        if model.enable_collision_plane:
            fig = draw_deformation_field2D(phi_vis, color=colors, plane_height=model.plane_height, hide_axis=False)
        elif model.enable_collision_circle:
            fig = draw_deformation_field2D(phi_vis, color=colors, scatter_radius = 0.01, hide_axis=True, circle_center=model.circle_center, circle_radius=model.circle_radius)
        else:
            fig = draw_deformation_field2D(phi_vis, color=colors, hide_axis = False)
        model.render_output(fig, output_dir, t = t_i.detach().cpu().numpy()[0])
        fig_list.append(fig)
 

if __name__ == '__main__':
    main_deformation_test()
