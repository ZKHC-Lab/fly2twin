# SPDX-License-Identifier: Apache-2.0
"""T_AB maps B coordinates into A; optical camera axes are right/down/forward."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform(translation, quaternion_xyzw):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    out[:3, 3] = translation
    return out


def deduplicate(times, positions, quaternions):
    """Drop identical consecutive poses, never discard a pure rotation."""
    r = Rotation.from_quat(quaternions).as_matrix()
    keep = np.r_[True, np.any(np.diff(positions, axis=0) != 0, axis=1) |
                 np.any(np.abs(np.diff(r, axis=0)) > 1e-12, axis=(1, 2))]
    return times[keep], positions[keep], quaternions[keep]


def interpolate(query_ns, times_ns, positions, quaternions, body_from_camera=None, offset_ms=0):
    times_ns = np.asarray(times_ns, np.int64)
    if len(times_ns) < 2 or np.any(np.diff(times_ns) <= 0):
        raise ValueError('Need at least two strictly increasing pose timestamps')
    query = np.asarray(query_ns, np.int64) + int(round(offset_ms * 1e6))
    valid = (query >= times_ns[0]) & (query <= times_ns[-1])
    a = (times_ns - times_ns[0]) / 1e9
    b = (np.clip(query, times_ns[0], times_ns[-1]) - times_ns[0]) / 1e9
    poses = np.tile(np.eye(4), (len(query), 1, 1))
    poses[:, :3, :3] = Slerp(a, Rotation.from_quat(quaternions))(b).as_matrix()
    poses[:, :3, 3] = np.stack([np.interp(b, a, positions[:, k]) for k in range(3)], 1)
    if body_from_camera is not None:
        poses = poses @ np.asarray(body_from_camera)
    return poses, valid


def resize_K(K, width, height, source_width, source_height):
    sx, sy = width / source_width, height / source_height
    return np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]]) @ K


def normalization(poses, train, fixed=None):
    if fixed is not None:
        center = np.asarray(fixed['normalization_center'], dtype=np.float64)
        meters, scale = float(fixed['meters_per_unit']), float(fixed['scene_scale'])
    else:
        centers = poses[train, :3, 3]
        center = centers.mean(0)
        meters = float(np.linalg.norm(centers-center, axis=1).max())
        locations = (poses[:, :3, 3]-center)/meters if meters > 0 else centers
        scale = float(np.linalg.norm(locations-locations.mean(0), axis=1).max()) * 1.1
    if not np.isfinite(center).all() or not np.isfinite([meters, scale]).all() or min(meters, scale) <= 0:
        raise ValueError('Degenerate normalization; supply a valid fixed normalization')
    return center, meters, scale
