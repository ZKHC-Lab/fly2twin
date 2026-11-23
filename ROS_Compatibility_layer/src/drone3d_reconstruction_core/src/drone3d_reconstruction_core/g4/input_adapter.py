"""Read immutable, verified Omega outputs; never consume an old map placement."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit, unquote

import numpy as np
from g4_backend import Sim3, from_prediction


def local_path(uri):
    parsed = urlsplit(str(uri))
    if parsed.scheme not in ("", "file") or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("expected a plain local path or file URI")
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise ValueError("expected an absolute path")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class WindowInput:
    geometry: object
    metadata: dict
    reference_centers: np.ndarray
    reference_valid: np.ndarray
    # Camera->odometry rotations at RGB time; NaN where reference_valid is False.
    reference_rotations: np.ndarray = None


def _calibration(snapshot):
    """One CameraInfo K shared by every frame that has it, else None."""
    payloads = [p for p in getattr(snapshot, "camera_info_payloads", ()) or () if p]
    if not payloads:
        return None
    first = payloads[0]
    for p in payloads[1:]:
        if (p.get("width"), p.get("height")) != (first.get("width"), first.get("height")) or \
                not np.allclose(p.get("K"), first.get("K"), rtol=0, atol=1e-9):
            return None
    K = np.asarray(first["K"], float).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or min(first["width"], first["height"]) <= 0:
        return None
    return {"width": int(first["width"]), "height": int(first["height"]), "K": K.tolist()}


class WindowInputReader:
    """ROS serialization boundary. The optimization core never sees these files.

    Snapshot v4 is required for timestamp, RGB hash and camera/body extrinsic
    provenance. Raw sensor depth is not read. Only explicit interpolated RGB-time
    odometry is admitted as localization. Missing poses remain missing.
    """

    def load(self, record):
        # The legacy package __init__ imports ROS action servers. Keep that
        # dependency at the actual formal-record I/O boundary, not module import.
        from drone3d_reconstruction_core.snapshot_validator import SnapshotSafetyLimits, validate_snapshot
        from drone3d_reconstruction_core.odom_local import _reference_camera_positions
        from drone3d_reconstruction_core.static_camera import resolve_static_camera
        manifest_path = local_path(record["result_manifest_uri"])
        if sha256(manifest_path) != record["result_manifest_sha256"]:
            raise ValueError("worker manifest changed")
        payload = json.loads(manifest_path.read_text())
        backend = payload.get("backend", (record.get("effective_config") or {}).get("backend", "omega"))
        if backend not in ("omega", "vggt_omega", "vggt-omega"):
            raise ValueError("G4 adapter accepts Omega predictions only")
        artifact = payload["artifacts"]["predictions.npz"]
        prediction_path = local_path(artifact["uri"])
        if sha256(prediction_path) != artifact["sha256"]:
            raise ValueError("prediction digest changed")
        snapshot_path = local_path(record["snapshot_manifest_uri"])
        limits = SnapshotSafetyLimits(512, 10000, 32*1024**2,
                                      1024**3, 2*1024**3, 4096, 4096)
        snapshot = validate_snapshot(str(snapshot_path), record["snapshot_manifest_sha256"],
                                     [snapshot_path.parent], limits)
        selected = payload.get("selected_frames", [])
        indices = [int(item["index"]) for item in selected]
        if indices != list(range(snapshot.frame_count)):
            raise ValueError("frozen G4 adapter forbids model input subsampling or reordering")
        ids = record.get("model_frame_ids")
        if not ids or len(ids) != len(indices):
            raise ValueError("explicit model_frame_ids are required")
        timestamps = []
        hashes = []
        for k, index in enumerate(indices):
            stamp = snapshot.frame_stamps[index]
            if stamp is None:
                raise ValueError("RGB timestamp is missing")
            expected = {"sec": stamp.sec, "nsec": stamp.nsec}
            if selected[k].get("stamp") != expected:
                raise ValueError("worker RGB timestamp differs from snapshot")
            timestamps.append(stamp.sec * 10**9 + stamp.nsec)
            hashes.append(sha256(snapshot.image_paths[index]))
        with np.load(prediction_path, allow_pickle=False) as archive:
            images = archive["images"]
            # Omega max_size: both sides are patch (16) multiples, e.g. 384x288 / 512x384 for 4:3,
            # 384x224 / 512x288 for 16:9 sources.
            if (images.ndim != 4 or images.shape[:2] != (len(ids), 3)
                    or images.shape[2] % 16 or images.shape[3] % 16):
                raise ValueError("G4 requires Omega NCHW images with 16-pixel patch sides")
            if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError("Omega images must be finite RGB floats in [0,1]")
            if len(ids) != int(record["effective_config"]["window_frames"]):
                raise ValueError("G4 prediction count differs from configured total window size")
            ex = archive["extrinsic"]
            geometry = from_prediction(
                int(record["window_index"]), ids, np.array(timestamps, np.int64),
                archive["depth"], archive["depth_conf"], archive["intrinsic"], ex,
                shared_prefix=int(record.get("overlap_frame_count", 0)), rgb_hashes=hashes,
            )
        # Raw model -> first-camera normalized gauge. Do not apply old alignment.
        prediction_to_local = Sim3(1 / geometry.gauge, ex[0, :, :3],
                                   ex[0, :, 3] / geometry.gauge)
        refs = np.zeros((len(ids), 3), dtype=float)
        ref_rotations = np.full((len(ids), 3, 3), np.nan)
        valid = np.zeros(len(ids), bool)
        reference_frame = ""
        localization_reason = "no valid interpolated odometry"
        try:
            transform = resolve_static_camera(snapshot.tf_entries, snapshot.body_frame_id,
                                              snapshot.camera_frame_id)
            ext = None if transform is None else (transform[:3, :3], transform[:3, 3])
        except ValueError:
            ext = None
            localization_reason = "inconsistent static body-camera extrinsics"
        flags = getattr(snapshot, "pose_valid", None)
        has_pose = any(p and (flags is None or flags[k] is True)
                       for k, p in enumerate(snapshot.odometry_payloads))
        if ext is not None and has_pose:
            source_indices, positions, rotations, reference_frame, _ = _reference_camera_positions(snapshot, *ext)
            by_index = dict(zip(source_indices.tolist(), positions))
            rotation_by_index = dict(zip(source_indices.tolist(), rotations))
            for k, index in enumerate(indices):
                odom = snapshot.odometry_payloads[index]
                if index not in by_index or not odom:
                    continue
                stamp = snapshot.frame_stamps[index]
                if (odom.get("child_frame_id") == snapshot.body_frame_id
                    and (odom.get("interpolation") or {}).get("method") == "linear_position_slerp_orientation"
                    and odom["header"]["stamp"] == {"sec": stamp.sec, "nsec": stamp.nsec}):
                    refs[k] = by_index[index]
                    ref_rotations[k] = rotation_by_index[index]
                    valid[k] = True
            localization_reason = "" if valid.any() else localization_reason
        elif ext is None and localization_reason == "no valid interpolated odometry":
            localization_reason = "consistent static body-camera extrinsic unavailable"
        metadata = {
            "window_id": geometry.window_id, "frame_ids": list(ids),
            "timestamps_ns": timestamps, "shared_prefix": geometry.shared_prefix,
            "rgb_hashes": hashes, "gauge": geometry.gauge,
            "prediction_to_local": prediction_to_local.to_dict(),
            "prediction_uri": str(prediction_path), "prediction_sha256": artifact["sha256"],
            "result_manifest_sha256": record["result_manifest_sha256"],
            "snapshot_manifest_sha256": record["snapshot_manifest_sha256"],
            "camera_frame_id": snapshot.camera_frame_id,
            "camera_topic": snapshot.image_topic,
            "reference_frame_id": reference_frame or "",
            "capture_generation": record.get("capture_generation", 0),
            "body_from_camera": None if ext is None else {
                "rotation": ext[0].tolist(), "translation": ext[1].tolist()},
            "localization_reason": localization_reason,
            "localization_valid_frames": int(valid.sum()),
            "camera_info": _calibration(snapshot),
            "model_image_size": [int(images.shape[3]), int(images.shape[2])],
            "attention_backend": payload.get("profile", {}).get("attention_backend", "auto"),
        }
        offsets = {(p.get('interpolation', {}).get('pose_time_offset_stage', 'backend'),
                    p.get('interpolation', {}).get('pose_time_offset_ms', 0.0))
                   for k, p in enumerate(snapshot.odometry_payloads) if p and valid[k]}
        if len(offsets) > 1:
            raise ValueError('Mixed capture pose offsets in one window')
        if offsets and next(iter(offsets))[0] == 'capture':
            metadata['capture_pose_time_offset_ms'] = float(next(iter(offsets))[1])
        return WindowInput(geometry, metadata, refs, valid, ref_rotations)
