import matplotlib.pyplot as plt


def draw_vector_field2D(vel, coords):
    u, v = vel[..., 0], vel[..., 1]
    x, y = coords[..., 0], coords[..., 1]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.quiver(x, y, u, v, scale=u.shape[0], scale_units='width')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    return fig


def draw_scalar_field2D(arr, vmin=None, vmax=None, cmap=None):
    multi = max(arr.shape[0] // 512, 1)
    fig, ax = plt.subplots(figsize=(5 * multi, 5 * multi))
    cax1 = ax.matshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(cax1, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
