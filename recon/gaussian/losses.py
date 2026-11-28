# SPDX-License-Identifier: Apache-2.0
"""Differentiable training losses; torch is imported only by training callers."""


def inverse_depth_pearson(rendered, prior, mask):
    """1 - Pearson correlation of inverse depth over the mask (`train.depth_loss: pearson`, the default until 2026-11-27).

    Invariant to the scale of the prior and to an offset in inverse depth, so it also works when the prior and the
    poses disagree about the scale; for the same reason it cannot hold a frame at the right distance.
    """
    x = rendered[mask].clamp_min(1e-6).reciprocal()
    y = prior[mask].reciprocal()
    if x.numel() < 2:
        raise ValueError('Pearson loss needs at least two valid pixels')
    x, y = x - x.mean(), y - y.mean()
    return 1 - (x * y).sum() / (x.square().sum() * y.square().sum()).sqrt().clamp_min(1e-12)


def log_depth_l1(rendered, prior, mask):
    """Mean absolute difference of log depth over the mask (`train.depth_loss: log`, the default).

    The rendered depth has to equal the prior, in the units of the poses. The Pearson term above does not see a
    frame rendered at a different overall scale or with an offset in inverse depth, which lets the first frames
    of a forward walk collapse onto a shrunken copy of the scene in front of the camera (measured 2026-11-26).
    This term does, so it needs a prior whose scale agrees with the poses: the fusion stage fits the depth of
    every window to the poses, which is the prior the trainer reads.
    """
    return (rendered[mask].clamp_min(1e-6).log() - prior[mask].log()).abs().mean()


DEPTH_LOSSES = dict(pearson=inverse_depth_pearson, log=log_depth_l1)


def anisotropy(log_scales, max_ratio=10.0):
    axes = log_scales.exp()
    return (axes.max(dim=1).values / axes.min(dim=1).values - max_ratio).relu().mean()
