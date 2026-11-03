"""Window reference and diagnostics for the frozen shared-point A backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import shared_point_sim3 as algorithm
from .odom_local import (
    COORDINATE_ODOM_LOCAL_APPROX,
    _reference_camera_positions,
    _resolve_body_camera_transform,
)


BACKEND_A = "shared_point_sim3_A"
BACKEND_ODOM = "odom_pose_constrained_sim3"


@dataclass(frozen=True)
class SharedReference:
    window_index: int
    root_window_index: int
    map_id: str
    reference_frame_id: str
    frame_ids: tuple
    predictions: dict
    transform: tuple
    capture_generation: int


def _read_slice(path, selected):
    with np.load(str(path), allow_pickle=False) as archive:
        return {
            key: archive[key][selected].copy()
            for key in ("images", "depth", "depth_conf", "world_points_from_depth")
        }


def _pose_diagnostics(extrinsic, positions, rotations, transform, limits):
    predicted = algorithm.apply(algorithm.centers(extrinsic.astype(float)), transform)
    errors = np.linalg.norm(predicted - positions, axis=1)
    relative = rotations.transpose(0, 2, 1) @ (
        transform[1] @ extrinsic[:, :, :3].transpose(0, 2, 1)
    )
    angles = np.degrees(np.arccos(np.clip(
        (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0
    )))
    baseline = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
    position_rms = algorithm.stats(errors)["rms"]
    angle_rms = algorithm.stats(angles)["rms"]
    normalized = position_rms / baseline if baseline > 0 else None
    return {
        "residual_rms": position_rms,
        "orientation_residual_deg": angle_rms,
        "reference_baseline_m": baseline,
        "normalized_residual_rms": normalized,
        "odom_quality_diagnostic_only": True,
        "odom_quality_diagnostic_passed": (
            baseline >= limits[0] and angle_rms <= limits[1]
            and normalized is not None and normalized <= limits[2]
        ),
    }


def align(
    snapshot, predictions_path: Path, *, model_frame_ids, selected_indices,
    window_index, capture_generation, map_id, reference=None,
    odom_limits=(0.1, 15.0, 0.25), odom_jump_m=5.0,
):
    """Compute a candidate. The caller commits its reference after publication."""
    expected_frames = 100 if reference is None else 120
    frame_ids = tuple(model_frame_ids)
    if len(frame_ids) != expected_frames or len(set(frame_ids)) != expected_frames:
        raise ValueError("A requires exact unique 100/120 frame identities")
    if tuple(selected_indices or ()) != tuple(range(expected_frames)):
        raise ValueError("A forbids model input subsampling or reordering")
    with np.load(str(predictions_path), allow_pickle=False) as archive:
        extrinsic = archive["extrinsic"].astype(float)
        image_shape = archive["images"].shape
    if extrinsic.shape != (expected_frames, 3, 4) or image_shape[0] != expected_frames:
        raise ValueError("A prediction frame count does not match its input")
    if max(image_shape[-2:]) != 384:
        raise ValueError("A requires the frozen 384 max-size image resolution")

    indices = positions = rotations = frame_id = None
    odometry_error = ""
    try:
        body_camera = _resolve_body_camera_transform(
            tuple(snapshot.tf_entries or ()), snapshot.body_frame_id, snapshot.camera_frame_id
        )
        if body_camera is None:
            raise ValueError("missing body-camera transform")
        indices, positions, rotations, frame_id, _stamps = _reference_camera_positions(
            snapshot, *body_camera
        )
        for index in indices:
            if snapshot.odometry_payloads[index].get("child_frame_id") != snapshot.body_frame_id:
                raise ValueError("odometry child frame does not match body frame")
    except (ValueError, TypeError, KeyError) as error:
        odometry_error = str(error)
        indices = positions = rotations = frame_id = None

    diagnostics = {}
    if reference is None:
        if odometry_error:
            raise ValueError("root odometry unavailable: " + odometry_error)
        if not np.array_equal(indices, np.arange(100)):
            raise ValueError("root requires interpolated odometry for all 100 RGB frames")
        for index, payload in enumerate(snapshot.odometry_payloads):
            interpolation = payload.get("interpolation") or {}
            if interpolation.get("method") != "linear_position_slerp_orientation":
                raise ValueError("root requires explicit interpolated odometry provenance")
            stamp = snapshot.frame_stamps[index]
            if payload["header"]["stamp"] != {"sec": stamp.sec, "nsec": stamp.nsec}:
                raise ValueError("root interpolated pose stamp differs from RGB stamp")
        transform = algorithm.fit_pose(extrinsic, positions, rotations)
        root_window_index = window_index
        root_frame_id = frame_id
        method = BACKEND_A + "_root_odom"
        reference_window_index = None
    else:
        if reference.capture_generation != capture_generation:
            raise ValueError("capture timestamp generation changed across A windows")
        if frame_ids[:20] != reference.frame_ids:
            raise ValueError("shared frame IDs do not match the committed reference")
        current = _read_slice(predictions_path, slice(0, 20))
        pairs = algorithm.shared_pairs(reference.predictions, current)
        transform, diagnostics["ransac"] = algorithm.robust(pairs, reference.transform)
        diagnostics["held_out_shared_points"] = algorithm.held_out_metrics(
            pairs, reference.transform, transform
        )
        root_window_index = reference.root_window_index
        root_frame_id = reference.reference_frame_id
        map_id = reference.map_id
        method = BACKEND_A
        reference_window_index = reference.window_index

    if positions is not None and frame_id == root_frame_id:
        diagnostics.update(_pose_diagnostics(
            extrinsic[indices], positions, rotations, transform, odom_limits
        ))
        jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        diagnostics["odometry_jump_count"] = int((jumps > odom_jump_m).sum())
    elif frame_id is not None:
        odometry_error = "odometry frame changed; shared RGB map remains in its root frame"
    diagnostics["odometry_diagnostic_error"] = odometry_error
    diagnostics["observed_odometry_frame_id"] = frame_id or ""
    start = 0 if reference is None else 20
    payload = {
        "coordinate_state": COORDINATE_ODOM_LOCAL_APPROX,
        "method": method,
        "algorithm_version": algorithm.ALGORITHM_VERSION,
        "root_window_index": root_window_index,
        "reference_window_index": reference_window_index,
        "map_id": map_id,
        "reference_frame_id": root_frame_id,
        "metric_scale": True,
        "quality_passed": True,
        "pending": False,
        "unreliable": True,
        "reason": "",
        "correspondence_count": 100 if reference is None else diagnostics["ransac"]["training_points"],
        "merge_frame_indices": list(range(start, expected_frames)),
        "overlap_model_frames": start,
        "new_frame_count": 100,
        "model_frame_count": expected_frames,
        "shared_rgb_bitwise_equal": reference is not None,
        "inlier_count": diagnostics.get("ransac", {}).get("inliers"),
        "inlier_ratio": diagnostics.get("ransac", {}).get("inlier_ratio"),
        **algorithm.packed(transform),
        **diagnostics,
    }
    candidate = SharedReference(
        window_index=window_index, root_window_index=root_window_index,
        map_id=map_id, reference_frame_id=root_frame_id,
        frame_ids=frame_ids[-20:], predictions=_read_slice(predictions_path, slice(-20, None)),
        transform=transform, capture_generation=capture_generation,
    )
    return payload, candidate
