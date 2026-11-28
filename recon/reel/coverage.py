# SPDX-License-Identifier: Apache-2.0
"""Per-pixel training-view coverage of a rendered frame.

For a query camera, render expected depth and alpha at low resolution, back-project the opaque
pixels, and count training cameras that see each 3D point: inside the frustum, in front, not occluded
(point depth <= training depth map * (1 + DEPTH_TOL)), and with the viewing rays from the query and
training centres to the point within ANGLE_DEG of each other, and the training camera no more than
RES_RATIO times farther from the point than the query (otherwise the query magnifies the texture). Training depth maps are rendered once.
"""

import numpy as np
import torch

from recon.geometry.transforms import resize_K
from recon.gaussian.render import render

TIER_A = dict(unseen=0.02, ge10=0.80)
TIER_B = dict(unseen=0.05, ge10=0.60)
LOW_W, LOW_H = 160, 90
DEPTH_TOL = 0.05
ANGLE_DEG = 30.0
ALPHA_OPAQUE = 0.5
RES_RATIO = 1.5  # a training view counts only if it was at most 1.5x farther from the point than the query


def low_K(K, w, h):
    return resize_K(K, LOW_W, LOW_H, w, h).astype(np.float32)


def render_depth(splats, c2w, K, w=LOW_W, h=LOW_H, degree=0, near_plane=0.01):
    """Batched expected depth and alpha; c2w [N,4,4], K [N,3,3] numpy at (w, h)."""
    out_d, out_a = [], []
    for i in range(0, len(c2w), 4):
        c = torch.from_numpy(np.ascontiguousarray(c2w[i : i + 4], np.float32)).cuda()
        k = torch.from_numpy(np.ascontiguousarray(K[i : i + 4], np.float32)).cuda()
        colors, alphas, _ = render(splats, c, k, w, h, degree=degree, depth=True, near_plane=near_plane)
        a = alphas[..., 0]
        out_d.append((colors[..., 3] / a.clamp_min(1e-6)).cpu())
        out_a.append(a.cpu())
    return torch.cat(out_d), torch.cat(out_a)


class Coverage:
    def __init__(self, scene, device='cuda'):
        self.scene = scene
        self.device = device
        self.unit = getattr(scene, 'meta', {}).get('world_from_normalized', {}).get('meters_per_unit', 1.0)
        self.splats = scene.torch_splats(device)
        self.K = np.stack([low_K(k, scene.width, scene.height) for k in scene.train_K])
        with torch.no_grad():
            depth, alpha = render_depth(self.splats, scene.train_c2w, self.K, near_plane=0.01 * self.unit)
        depth[alpha < ALPHA_OPAQUE] = float('inf')  # nothing there: never occludes
        self.depth = depth.to(self.device)
        self.w2c = torch.linalg.inv(torch.from_numpy(scene.train_c2w.astype(np.float32)).to(self.device))
        self.centers = torch.from_numpy(scene.train_c2w[:, :3, 3].astype(np.float32)).to(self.device)
        self.Kt = torch.from_numpy(self.K).to(self.device)

    @torch.no_grad()
    def frame(self, c2w, K, w, h):
        """Coverage counts [LOW_H, LOW_W] (-1 where the render is not opaque) and alpha map."""
        Kq = low_K(K, w, h)
        d, a = render_depth(self.splats, c2w[None], Kq[None], near_plane=0.01 * self.unit)
        d, a = d[0].to(self.device), a[0].to(self.device)
        v, u = torch.meshgrid(
            torch.arange(LOW_H, device=self.device), torch.arange(LOW_W, device=self.device), indexing='ij'
        )
        Kq = torch.from_numpy(Kq).to(self.device)
        rays = torch.stack([(u - Kq[0, 2]) / Kq[0, 0], (v - Kq[1, 2]) / Kq[1, 1], torch.ones_like(d)], -1)
        cq = torch.from_numpy(c2w.astype(np.float32)).to(self.device)
        mask = a >= ALPHA_OPAQUE
        world = (rays[mask] * d[mask][:, None]) @ cq[:3, :3].T + cq[:3, 3]
        qvec = world - cq[:3, 3]
        qdist = qvec.norm(dim=-1)
        qdir = qvec / qdist[:, None].clamp_min(1e-6)
        count = torch.zeros(len(world), device=self.device, dtype=torch.int32)
        cos_lim = np.cos(np.radians(ANGLE_DEG))
        for i in range(0, len(self.w2c), 64):
            R, t = self.w2c[i : i + 64, :3, :3], self.w2c[i : i + 64, :3, 3]
            cam = torch.einsum('nij,mj->nmi', R, world) + t[:, None]
            z = cam[..., 2]
            K = self.Kt[i : i + 64]
            uu = K[:, None, 0, 0] * cam[..., 0] / z.clamp_min(1e-6) + K[:, None, 0, 2]
            vv = K[:, None, 1, 1] * cam[..., 1] / z.clamp_min(1e-6) + K[:, None, 1, 2]
            ui, vi = uu.round().long(), vv.round().long()
            inside = (z > 0.02 * self.unit) & (ui >= 0) & (ui < LOW_W) & (vi >= 0) & (vi < LOW_H)
            dm = self.depth[i : i + 64]
            idx = torch.arange(dm.shape[0], device=self.device)[:, None].expand_as(ui)
            ref = dm[idx.clamp(0), vi.clamp(0, LOW_H - 1), ui.clamp(0, LOW_W - 1)]
            visible = inside & (z <= ref * (1 + DEPTH_TOL))
            tvec = world[None] - self.centers[i : i + 64, None]
            tdist = tvec.norm(dim=-1)
            agree = ((tvec * qdir[None]).sum(-1) > cos_lim * tdist) & (tdist <= RES_RATIO * qdist[None])
            count += (visible & agree).sum(0).int()
        out = torch.full((LOW_H, LOW_W), -1, device=self.device, dtype=torch.int32)
        out[mask] = count
        return out.cpu().numpy(), a.cpu().numpy()


def stats(counts):
    """Fractions over the whole frame: holes (not opaque), unseen (0 views), and >= 3 / >= 10 views."""
    n = counts.size
    return dict(
        hole=float((counts < 0).mean()),
        unseen=float((counts == 0).sum() / n),
        ge3=float((counts >= 3).sum() / n),
        ge10=float((counts >= 10).sum() / n),
        median_views=float(np.median(counts[counts >= 0])) if (counts >= 0).any() else 0.0,
    )
