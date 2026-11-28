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
    keep = np.r_[
        True, np.any(np.diff(positions, axis=0) != 0, axis=1) | np.any(np.abs(np.diff(r, axis=0)) > 1e-12, axis=(1, 2))
    ]
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
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]]) @ K


def normalization(poses, train, fixed=None):
    if fixed is not None:
        center = np.asarray(fixed['normalization_center'], dtype=np.float64)
        meters, scale = float(fixed['meters_per_unit']), float(fixed['scene_scale'])
    else:
        centers = poses[train, :3, 3]
        center = centers.mean(0)
        meters = float(np.linalg.norm(centers - center, axis=1).max())
        locations = (poses[:, :3, 3] - center) / meters if meters > 0 else centers
        scale = float(np.linalg.norm(locations - locations.mean(0), axis=1).max()) * 1.1
    if not np.isfinite(center).all() or not np.isfinite([meters, scale]).all() or min(meters, scale) <= 0:
        raise ValueError('Degenerate normalization; supply a valid fixed normalization')
    return center, meters, scale


def robust_normalization(data):
    """Phone protocol normalisation: median centre, MAD-guarded radius, P95 scale.

    A handful of mis-registered cameras can sit orders of magnitude away; a plain
    P95 then describes those outliers instead of the room, and dividing by that
    radius compresses the real scene so far that training cannot recover.  Keep
    only cameras inside a robust bound first, then take the P95 of what remains.
    """
    centers = data['c2w'][data['train_indices'], :3, 3]
    if not len(centers):
        raise RuntimeError('No registered training cameras in cameras.npz')
    center = np.median(centers, axis=0)
    distance = np.linalg.norm(centers - center, axis=1)
    median = float(np.median(distance))
    if not np.isfinite(median) or median <= 0:
        raise RuntimeError('Invalid median camera distance')
    spread = float(np.median(np.abs(distance - median))) or 0.5 * median
    limit = max(median + 8.0 * spread, 1.5 * median)
    outlier = data['train_indices'][distance > limit]
    train = data['train_indices'][distance <= limit]
    scale = float(np.percentile(np.linalg.norm(data['c2w'][train, :3, 3] - center, axis=1), 95))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError('Invalid P95 radius for the guarded training centres')

    registered = data['registered_indices']
    spacing = np.linalg.norm(data['c2w'][registered, :3, 3] - center, axis=1)
    far = registered[spacing > 5 * scale]
    keep = registered[spacing <= 5 * scale]
    dropped = np.union1d(outlier, far)
    keep = np.setdiff1d(keep, outlier)
    result = dict(
        center=center.tolist(),
        p95_radius=scale,
        median_camera_distance=median,
        robust_limit=float(limit),
        rule='median centre; MAD-guarded bound median+8*MAD (>=1.5*median) for the '
        'normalisation scale; then drop cameras beyond 5*P95 of the kept set',
        registered_before=len(registered),
        registered_after=len(keep),
        dropped_count=int(len(dropped)),
        dropped_frames=data['frame_ids'][dropped].tolist(),
        outlier_frames=data['frame_ids'][outlier].tolist(),
        far_frames=data['frame_ids'][far].tolist(),
    )
    for split in ('train_indices', 'test_indices'):
        data[split] = data[split][np.isin(data[split], keep)]
        result[f'{split}_frames'] = len(data[split])
    if not result['train_indices_frames']:
        raise RuntimeError('Every training camera was rejected by the normalisation guard')
    data['registered_indices'] = keep
    return result
