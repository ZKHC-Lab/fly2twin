# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
from gsplat import rasterization

def initialize(points, colors, scene_scale):
    from scipy.spatial import cKDTree
    distances, _ = cKDTree(points).query(points, k=4, workers=4)
    scales = np.log(np.sqrt((distances[:, 1:] ** 2).mean(1))).astype(np.float32)
    n = len(points)
    sh = torch.zeros(n, 16, 3)
    sh[:, 0] = (torch.from_numpy(colors) - 0.5) / 0.28209479177387814
    values = dict(means=torch.from_numpy(points), scales=torch.from_numpy(scales)[:, None].repeat(1, 3),
                  quats=torch.rand(n, 4), opacities=torch.logit(torch.full((n,), 0.1)),
                  sh0=sh[:, :1].contiguous(), shN=sh[:, 1:].contiguous())
    splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.cuda()) for k, v in values.items()})
    lrs = dict(means=1.6e-4 * scene_scale, scales=5e-3, quats=1e-3,
               opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    optimizers = {k: torch.optim.Adam([{'params': p, 'lr': lrs[k], 'name': k}],
                  eps=1e-15, betas=(0.9, 0.999), fused=True) for k, p in splats.items()}
    return splats, optimizers


def render(splats, c2w, K, width, height, degree=3, depth=False, near_plane=0.01):
    return rasterization(means=splats['means'], quats=splats['quats'], scales=splats['scales'].exp(),
        opacities=splats['opacities'].sigmoid(), colors=torch.cat([splats['sh0'], splats['shN']], 1),
        viewmats=torch.linalg.inv(c2w), Ks=K, width=width, height=height, sh_degree=degree,
        near_plane=near_plane, far_plane=1e10, packed=False, sparse_grad=False, absgrad=False,
        rasterize_mode='classic', camera_model='pinhole', render_mode='RGB+ED' if depth else 'RGB')
