# SPDX-License-Identifier: Apache-2.0
"""Differentiable training losses; torch is imported only by training callers."""

def inverse_depth_pearson(rendered, prior, mask):
    x = rendered[mask].clamp_min(1e-6).reciprocal()
    y = prior[mask].reciprocal()
    if x.numel() < 2:
        raise ValueError('Pearson loss needs at least two valid pixels')
    x, y = x-x.mean(), y-y.mean()
    return 1-(x*y).sum()/(x.square().sum()*y.square().sum()).sqrt().clamp_min(1e-12)


def anisotropy(log_scales, max_ratio=10.):
    axes = log_scales.exp()
    return (axes.max(dim=1).values/axes.min(dim=1).values-max_ratio).relu().mean()
