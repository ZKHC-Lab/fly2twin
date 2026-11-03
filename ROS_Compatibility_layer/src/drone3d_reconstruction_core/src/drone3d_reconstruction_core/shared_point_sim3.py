"""Frozen A experiment: shared RGB pixels -> robust window-to-map Sim(3).

Ported from Data/experiments/_archive_gpt/testflight3_four_windows_20260910/scripts/common.py.
This module owns numerical computation only. The session owns frame identity,
reference lifetime, publication, and failure handling.

The threshold is 0.1 in the root-anchored map's units. Ten even shared-frame
indices fit the transform. The ten odd indices are reserved for validation.
"""

from __future__ import annotations

import numpy as np


ALGORITHM_VERSION = "shared_rgb_point_sim3_A_20260910"
SHARED_FRAMES = 20
TRAIN_POINTS_PER_FRAME = 2000
RANSAC_TRIALS = 1000
RANSAC_SEED = 42
INLIER_THRESHOLD_MAP = 0.1
MAX_REFITS = 5


def stats(values):
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("empty residual set")
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "max": float(np.max(values)),
    }


def so3(matrix):
    u, _singular, vt = np.linalg.svd(np.asarray(matrix, float))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def centers(extrinsic):
    return -np.einsum("nji,nj->ni", extrinsic[:, :, :3], extrinsic[:, :, 3])


def apply(points, transform):
    scale, rotation, translation = transform
    return scale * (points @ rotation.T) + translation


def inverse(transform):
    scale, rotation, translation = transform
    return 1 / scale, rotation.T, -rotation.T @ translation / scale


def compose(first, second):
    scale, rotation, translation = first
    other_scale, other_rotation, other_translation = second
    return (
        scale * other_scale,
        rotation @ other_rotation,
        scale * rotation @ other_translation + translation,
    )


def packed(transform):
    scale, rotation, translation = transform
    return {
        "scale": float(scale),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }


def unpack(payload):
    return (
        float(payload["scale"]),
        np.array(payload["rotation"]),
        np.array(payload["translation"]),
    )


def fit_pose(extrinsic, positions, camera_to_map):
    """Fit A's root transform. ROS quality thresholds are diagnostic only."""
    extrinsic = extrinsic.astype(float)
    camera_centers = centers(extrinsic)
    rotation = so3((camera_to_map @ extrinsic[:, :, :3]).sum(0))
    rotated = camera_centers @ rotation.T
    source_centered = rotated - rotated.mean(0)
    target_centered = positions - positions.mean(0)
    denominator = np.sum(source_centered * source_centered)
    if not denominator > 1e-10:
        raise ValueError("root model trajectory is degenerate")
    scale = float(np.sum(source_centered * target_centered) / denominator)
    if not np.isfinite(scale) or not scale > 0:
        raise ValueError("root scale is not finite/positive")
    translation = positions.mean(0) - scale * rotated.mean(0)
    if not np.isfinite(translation).all():
        raise ValueError("root translation is not finite")
    return scale, rotation, translation


def umeyama(source, target):
    source_centered = source - source.mean(0)
    target_centered = target - target.mean(0)
    u, singular, vt = np.linalg.svd(
        target_centered.T @ source_centered / len(source)
    )
    correction = np.ones(3)
    correction[-1] = np.linalg.det(u @ vt)
    rotation = (u * correction) @ vt
    denominator = np.mean(np.sum(source_centered * source_centered, axis=1))
    if denominator < 1e-12:
        raise ValueError("Zero source baseline")
    scale = float(singular @ correction / denominator)
    if scale <= 0:
        raise ValueError("Nonpositive scale")
    return scale, rotation, target.mean(0) - scale * rotation @ source.mean(0)


def shared_pairs(previous, current):
    """Pair equal preprocessed RGB pixels in the last/first twenty frames."""
    if len(previous["images"]) < SHARED_FRAMES or len(current["images"]) < SHARED_FRAMES:
        raise ValueError("twenty shared frames are required")
    if not np.array_equal(previous["images"][-20:], current["images"][:20]):
        raise ValueError("shared preprocessed RGB frames differ")
    pairs = []
    for index in range(SHARED_FRAMES):
        depth_a = previous["depth"][-20 + index, ..., 0]
        depth_b = current["depth"][index, ..., 0]
        confidence_a = previous["depth_conf"][-20 + index]
        confidence_b = current["depth_conf"][index]
        valid = (
            np.isfinite(depth_a) & np.isfinite(depth_b)
            & (depth_a > 0) & (depth_b > 0)
            & np.isfinite(confidence_a) & np.isfinite(confidence_b)
        )
        if not valid.any():
            raise ValueError("shared frame has no common valid depth/confidence")
        mask = (
            valid
            & (confidence_a >= np.percentile(confidence_a[valid], 50))
            & (confidence_b >= np.percentile(confidence_b[valid], 50))
        )
        yy, xx = np.nonzero(mask)
        a = previous["world_points_from_depth"][-20 + index][mask].astype(float)
        b = current["world_points_from_depth"][index][mask].astype(float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("shared world points are not finite")
        pairs.append((a, b, np.column_stack([xx, yy])))
    return pairs


def robust(pairs, previous_to_map):
    """Return the current local->map transform with A's exact fit parameters."""
    if len(pairs) != SHARED_FRAMES:
        raise ValueError("twenty shared frame pairs are required")
    rng = np.random.default_rng(RANSAC_SEED)
    xs, ys = [], []
    for index in range(0, SHARED_FRAMES, 2):
        previous, current, _uv = pairs[index]
        ids = rng.choice(
            len(previous), min(TRAIN_POINTS_PER_FRAME, len(previous)), replace=False
        )
        xs.append(current[ids])
        ys.append(apply(previous[ids], previous_to_map))
    source = np.concatenate(xs)
    target = np.concatenate(ys)
    if len(source) < 3:
        raise ValueError("fewer than three training point pairs")
    best = None
    best_score = (-1, -np.inf)
    valid_hypotheses = 0
    for _trial in range(RANSAC_TRIALS):
        ids = rng.choice(len(source), 3, replace=False)
        xx, yy = source[ids], target[ids]
        if min(
            np.linalg.svd(xx - xx.mean(0), compute_uv=False)[1],
            np.linalg.svd(yy - yy.mean(0), compute_uv=False)[1],
        ) < 1e-7:
            continue
        try:
            transform = umeyama(xx, yy)
        except ValueError:
            continue
        valid_hypotheses += 1
        error = np.linalg.norm(apply(source, transform) - target, axis=1)
        mask = error < INLIER_THRESHOLD_MAP
        count = int(mask.sum())
        score = (count, -float(np.mean(error[mask] ** 2)) if count else -np.inf)
        if score > best_score:
            best_score = score
            best = (transform, mask)
    if best is None or best_score[0] < 3:
        raise ValueError("No RANSAC consensus")
    transform, mask = best
    for iteration in range(MAX_REFITS):
        transform = umeyama(source[mask], target[mask])
        error = np.linalg.norm(apply(source, transform) - target, axis=1)
        new_mask = error < INLIER_THRESHOLD_MAP
        if new_mask.sum() < 3:
            raise ValueError("Refit consensus lost")
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    error = np.linalg.norm(apply(source, transform) - target, axis=1)
    mask = error < INLIER_THRESHOLD_MAP
    return transform, {
        "training_points": len(source),
        "inliers": int(mask.sum()),
        "inlier_ratio": float(mask.mean()),
        "threshold_map": INLIER_THRESHOLD_MAP,
        "valid_hypotheses": valid_hypotheses,
        "refit_iterations": iteration + 1,
        "all_train_error": stats(error),
        "inlier_error": stats(error[mask]),
        "transform": packed(transform),
    }


def held_out_metrics(pairs, previous_to_map, current_to_map):
    """Measure odd shared frames after fitting. These points do not tune the fit."""
    frames = []
    for index in range(1, SHARED_FRAMES, 2):
        previous, current, _uv = pairs[index]
        error = np.linalg.norm(
            apply(current, current_to_map) - apply(previous, previous_to_map), axis=1
        )
        frames.append({"shared_index": index, "point_distance_map": stats(error)})
    return {
        "per_frame": frames,
        "median_by_frame": stats([item["point_distance_map"]["median"] for item in frames]),
    }
