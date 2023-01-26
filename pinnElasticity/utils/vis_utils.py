import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
# import cmasher as cmr

def draw_deformation_field2D(arr, vmin=None, vmax=None, color=None, plane_height=None, circle_center=None, circle_radius=None, scatter_radius=0.2, hide_axis = True, cmap = 'cool'):
    fig, ax = plt.subplots(figsize=(3, 3))
    if color is None:
        ax.scatter(arr[:,0], arr[:,1], c = arr[:,0]+arr[:,1], cmap=cmap, s=scatter_radius)
    else:
        ax.scatter(arr[:,0], arr[:,1], c = color, cmap=cmap, s=scatter_radius)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-3.5, 1.5)
    if plane_height is not None:
        plt.axhline(y=plane_height, color='green', linestyle='-', alpha=0.5)
    if circle_radius is not None and circle_center is not None:
        cir = plt.Circle(circle_center, circle_radius, color='orange',fill=False, linewidth=0.5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.add_patch(cir)
    if hide_axis:
        ax.set_axis_off()
    else:
        plt.axhline(y=1.0, color='orange', linestyle='-', alpha=0.5)
        plt.axhline(y=-1.0, color='orange', linestyle='-', alpha=0.5)
    return fig


def draw_scalar_field2D(arr, vmin=None, vmax=None, to_array=False, cmap=None):
    multi = max(arr.shape[0] // 512, 1)
    fig, ax = plt.subplots(figsize=(5 * multi, 5 * multi))
    if cmap is None:
        cax1 = ax.matshow(arr, vmin=vmin, vmax=vmax)
    else:
        cax1 = ax.matshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(cax1, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if not to_array:
        return fig
    return figure2array(fig)


def draw_vector_field2D(u, v, x=None, y=None, tag=None, to_array=False):
    assert u.shape == v.shape
    s = 5 * (u.shape[0] // 50 + 1)
    # fig, ax = plt.subplots(figsize=(s, s))
    fig, ax = plt.subplots(figsize=(5, 5))
    if x is None:
        # buggy
        raise NotImplementedError
        indices = np.indices(u.shape)
        # ax.quiver(indices[1], indices[0], u, v, scale=u.shape[0], scale_units='width')
        ax.quiver(indices[0], indices[1], u, v, scale=u.shape[0], scale_units='width')
    else:
        # ax.quiver(y, x, u, v, scale=u.shape[0], scale_units='width')
        ax.quiver(x, y, u, v, scale=u.shape[0], scale_units='width')
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
    if tag is not None:
        ax.text(-1, -1, tag, fontsize=12)
    fig.tight_layout()
    if not to_array:
        return fig
    return figure2array(fig)


# def draw_vorticity_field2D(curl, x, y, to_array=False):
#     fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
#     ax.contourf(
#         x,
#         y,
#         curl,
#         cmap=cmr.arctic,
#         levels=100,
#     )
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     fig.tight_layout()
#     if not to_array:
#         return fig
#     return figure2array(fig)


def figure2array(fig):
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def save_figure(fig, save_path, close=True, axis_off=False):
    if axis_off:
        plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    if close:
        plt.close(fig)


def frames2gif(src_dir, save_path, fps=24):
    filenames = sorted([x for x in os.listdir(src_dir) if x.endswith('.png')])
    img_list = [imageio.imread(os.path.join(src_dir, name)) for name in filenames]
    imageio.mimsave(save_path, img_list, fps=fps)
