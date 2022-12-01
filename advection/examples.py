import torch
from functools import partial


def get_examples(src, sdim=1, **kwargs):
    # select source function
    if src == 'example1':
        source_func = partial(gaussian_like, mu=-1.5)
    elif src == 'example2':
        source_func = partial(gaussianND_like, sdim=sdim, mu=-1.5)
    else:
        raise NotImplementedError
    return source_func


def gaussian_like(x, mu=0, sigma=0.1):
    """normalized gaussian distribution"""
    return torch.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))

def gaussianND_like(x, sdim=1, mu=0, sigma=0.1):
    """normalized gaussian distribution"""
    if sdim > 1:
        mu = mu * torch.ones(sdim).cuda()
    return torch.exp(-0.5 * torch.sum((x - mu) ** 2, dim=-1, keepdim=True) / (sigma ** 2))