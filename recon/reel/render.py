# SPDX-License-Identifier: Apache-2.0
"""GPU Gaussian and metric point rendering primitives."""
import numpy as np
import torch
from recon.gaussian.render import render
from .coverage import TIER_A, TIER_B

def tier(st):
    if st['unseen'] <= TIER_A['unseen'] and st['ge10'] >= TIER_A['ge10']:
        return 'A'
    return 'B' if st['unseen'] <= TIER_B['unseen'] and st['ge10'] >= TIER_B['ge10'] else 'C'


def intrinsics(sc, w, h, hfov=None):
    """Training camera K scaled to (w, h) (true principal point), or a centred pinhole of hfov."""
    K0 = sc.train_K[0].astype(np.float64)
    if hfov is None:
        sx, sy = w / sc.width, h / sc.height
        K = np.array([[K0[0, 0] * sx, 0, (K0[0, 2] + .5) * sx - .5], [0, K0[1, 1] * sy, (K0[1, 2] + .5) * sy - .5], [0, 0, 1]])
        return K.astype(np.float32), float(np.degrees(2 * np.arctan(sc.width / 2 / K0[0, 0])))
    f = w / 2 / np.tan(np.radians(hfov) / 2)
    return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]], np.float32), hfov


def point_splats(cloud, sc, size_m=0.012):
    """Fused cloud as small round Gaussians (world units), for point-cloud renders."""
    pts = cloud['points'].astype(np.float32)
    n = len(pts)
    rgb = torch.from_numpy(cloud['colors'].astype(np.float32))
    return dict(means=torch.from_numpy(pts).cuda(), quats=torch.tensor([[1., 0, 0, 0]]).repeat(n, 1).cuda(),
                scales=torch.full((n, 3), float(np.log(size_m))).cuda(),
                opacities=torch.full((n,), 6.0).cuda(),
                sh0=((rgb - 0.5) / 0.28209479177387814)[:, None].cuda(), shN=torch.zeros(n, 15, 3).cuda())


def draw(splats, pose, K, w, h, degree=3, bg=None):
    degree = min(degree, int(np.sqrt(1 + splats['shN'].shape[1])) - 1)
    c = torch.from_numpy(pose[None].astype(np.float32)).cuda()
    k = torch.from_numpy(K[None]).cuda()
    with torch.no_grad():
        colors, alphas, _ = render(splats, c, k, w, h, degree=degree)
    rgb = colors[0, ..., :3].clamp(0, 1)
    if bg is not None:
        rgb = rgb + (1 - alphas[0]) * torch.tensor(bg, device='cuda')
    return rgb.clamp(0, 1).cpu().numpy()

