"""Adapters from pinhole predictions to G4's first-camera depth gauge."""
import numpy as np
from .types import WindowGeometry


def from_prediction(window_id, frame_ids, timestamps_ns, depth, confidence, intrinsics, extrinsics,
                    shared_prefix=0, rgb_hashes=()):
    """extrinsics are world-to-camera [R|t]. Depth is axial camera z.

    The first-frame central 60% depth median defines the local length unit.
    No axis flip for a viewer or ROS is part of this geometry adapter.
    """
    d = np.asarray(depth)
    if d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 3 or len(d) != len(frame_ids):
        raise ValueError("depth must have shape NxHxW or NxHxWx1")
    h, w = d.shape[1:]
    central = d[0, int(h*.2):int(h*.8), int(w*.2):int(w*.8)]
    if not np.isfinite(central).all() or np.any(central <= 0):
        raise ValueError("first-camera depth gauge contains invalid depth")
    gauge = float(np.median(central))
    ex = np.asarray(extrinsics, float)
    if ex.shape != (len(d), 3, 4) or not np.isfinite(ex).all():
        raise ValueError("extrinsics must be finite Nx3x4 world-to-camera matrices")
    rc = ex[:, :, :3].transpose(0, 2, 1)
    cc = -np.einsum("nij,nj->ni", rc, ex[:, :, 3])
    centers = (cc@ex[0, :, :3].T+ex[0, :, 3])/gauge
    rotations = ex[0, :, :3][None]@rc
    return WindowGeometry(window_id, frame_ids, timestamps_ns, d/gauge, confidence,
                          intrinsics, rotations, centers, shared_prefix, rgb_hashes, gauge)


def resize_intrinsics(intrinsics, source_size, target_size):
    """Pinhole resize with pixel-center convention; sizes are (width, height)."""
    sw, sh = source_size
    tw, th = target_size
    if min(sw, sh, tw, th) <= 0:
        raise ValueError("image sizes must be positive")
    A = np.array([[tw/sw, 0, .5*(tw/sw-1)], [0, th/sh, .5*(th/sh-1)], [0, 0, 1.]])
    return A@np.asarray(intrinsics)
