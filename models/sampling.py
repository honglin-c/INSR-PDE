import torch


def sample_boundary(N, sdim=1, length=1.0, epsilon=1e-4):
    """sample boundary points within a small range"""
    if sdim == 1:
        coords_left = (torch.rand(N // 2, 1) * 2 - 1) * epsilon - 1.
        coords_right = (torch.rand(N // 2, 1) * 2 - 1) * epsilon + 1.
        coords = torch.cat([coords_left, coords_right], dim=0) * length
    elif sdim == 2:
        raise NotImplementedError
    else:
        raise NotImplementedError
    return coords
