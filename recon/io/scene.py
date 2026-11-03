# SPDX-License-Identifier: Apache-2.0
"""Load an exported scene (recon export directory) for downstream tools.

The export directory is the contract between reconstruction and everything that consumes it
(viewer, shot rendering, evaluation, simulation): scene.json + gaussians.ply + cloud.ply +
cameras.npz. Everything here is in the export's world frame and units (metres unless
scene.json says otherwise); no recon intermediate directories are read.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .ply import read_ply


@dataclass
class ExportedScene:
    path: Path
    meta: dict                      # scene.json
    gaussians: dict                 # numpy: means, quats(wxyz), scales(log), opacities(logit), sh0[N,1,3], shN[N,K,3]
    train_c2w: np.ndarray           # [T,4,4] training cameras, OpenCV axes
    train_K: np.ndarray             # [T,3,3] at (width, height)
    train_ids: np.ndarray
    width: int
    height: int
    world_up: np.ndarray            # unit vector in the export world
    bounds: dict                    # {'min': [3], 'max': [3]}
    test_c2w: np.ndarray = None     # capture cameras not used for training (if any)
    test_K: np.ndarray = None
    test_ids: np.ndarray = None
    cloud: dict = field(default_factory=dict)   # loaded lazily by points()

    def points(self):
        """Fused cloud as (points[N,3] float32, colors[N,3] float32 0..1)."""
        if not self.cloud:
            v = read_ply(self.path / self.meta.get('cloud', 'cloud.ply'))
            pts = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)
            rgb = np.stack([v['red'], v['green'], v['blue']], 1).astype(np.float32) / 255
            self.cloud.update(points=pts, colors=rgb)
        return self.cloud['points'], self.cloud['colors']

    def torch_splats(self, device='cuda'):
        """Gaussians as the torch dict expected by recon.gaussian.render.render."""
        import torch
        return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in self.gaussians.items()}


def read_gaussians(path):
    """INRIA Gaussian PLY -> numpy dict (log scales, logit opacities, wxyz quaternions, SH as [N,C,3])."""
    v = read_ply(path)
    names = v.dtype.names
    rest = sorted((n for n in names if n.startswith('f_rest_')), key=lambda n: int(n.split('_')[-1]))
    n = len(v)
    k = len(rest) // 3
    shN = (np.stack([v[c] for c in rest], 1).reshape(n, 3, k).transpose(0, 2, 1) if k
           else np.zeros((n, 0, 3), np.float32))
    return dict(means=np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32),
                quats=np.stack([v[f'rot_{i}'] for i in range(4)], 1).astype(np.float32),
                scales=np.stack([v[f'scale_{i}'] for i in range(3)], 1).astype(np.float32),
                opacities=np.asarray(v['opacity'], np.float32),
                sh0=np.stack([v[f'f_dc_{i}'] for i in range(3)], 1)[:, None].astype(np.float32),
                shN=np.ascontiguousarray(shN, np.float32))


def load_scene(path, width=None):
    """Read an export directory. width rescales training intrinsics (height follows the aspect)."""
    path = Path(path)
    meta = json.loads((path / 'scene.json').read_text())
    gaussians = read_gaussians(path / meta['preferred_model']['path'])
    with np.load(path / meta.get('cameras', 'cameras.npz'), allow_pickle=False) as z:
        ids, c2w, K, res = z['frame_ids'], z['c2w'].astype(np.float64), z['K'].astype(np.float64), z['resolution']
    sw, sh = int(res[0]), int(res[1])
    w = int(width or sw)
    h = int(round(sh * w / sw))
    sx, sy = w / sw, h / sh
    scale = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    K = scale @ K
    trained = np.isin(ids, meta['capture_corridor']['training_frame_ids'])
    up = np.asarray(meta.get('world_up', meta.get('coordinates', {}).get('world_up', [0, 0, 1])), float)
    bounds = meta.get('bounds')
    if bounds is None:   # older exports: capture cameras + 1 m
        c = c2w[:, :3, 3]
        bounds = dict(min=(c.min(0) - 1).tolist(), max=(c.max(0) + 1).tolist())
    scene = ExportedScene(path, meta, gaussians, c2w[trained].astype(np.float32), K[trained].astype(np.float32),
                          ids[trained], w, h, up / np.linalg.norm(up), bounds)
    if (~trained).any():
        scene.test_c2w, scene.test_K, scene.test_ids = (c2w[~trained].astype(np.float32),
                                                        K[~trained].astype(np.float32), ids[~trained])
    return scene
