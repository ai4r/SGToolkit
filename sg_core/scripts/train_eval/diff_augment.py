# code modified from https://github.com/mit-han-lab/data-efficient-gans/blob/master/DiffAugment_pytorch.py
# Differentiable Augmentation for Data-Efficient GAN Training, https://arxiv.org/pdf/2006.10738

import torch
import torch.nn.functional as F


def DiffAugment(x):
    for f in AUGMENT_FNS:
        x = f(x)
    x = x.contiguous()
    return x


def rand_gaussian(x):
    noise = torch.randn(x.size(0), 1, x.size(2), dtype=x.dtype, device=x.device)
    noise *= 0.15
    x = x + noise
    return x


AUGMENT_FNS = [rand_gaussian]
