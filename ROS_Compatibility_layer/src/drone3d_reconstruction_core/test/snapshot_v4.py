"""Shared test fixture: write a capture snapshot in the only supported format (v4).

Not a test module (no ``test_`` prefix); test files import it from this directory.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

FRAME_COLUMNS = (
    "frame_index", "image_path", "stamp_sec", "stamp_nsec", "pose_valid",
    "pose_delta_ms", "odometry_gap_ms", "auxiliary_path", "camera_info_delta_ms",
    "odometry_delta_ms", "depth_path", "depth_delta_ms",
)


def odometry_payload(sec, nsec=0, position=(0.0, 0.0, 0.0), frame_id="odom", child_frame_id="body"):
    """A complete nav_msgs/Odometry payload as capture stores it."""
    x, y, z = (float(v) for v in position)
    return {
        "header": {"stamp": {"sec": int(sec), "nsec": int(nsec)}, "frame_id": frame_id},
        "child_frame_id": child_frame_id,
        "pose": {"position": {"x": x, "y": y, "z": z},
                 "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                 "covariance": [0.0] * 36},
        "twist": {"linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                  "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
                  "covariance": [0.0] * 36},
    }


def _png_bytes(index):
    import cv2
    import numpy as np
    image = ((np.indices((8, 8)).sum(axis=0) % 2) * 255).astype(np.uint8)
    encoded, png = cv2.imencode(".png", image)
    assert encoded
    return png.tobytes()


def write_v4_snapshot(
    snapshot_dir: Path,
    scene_id: str,
    frame_count: int,
    *,
    first_stamp_sec: int = 100,
    stamp_nsecs=None,
    camera_name: str = "front",
    image_topic: str = "/camera/color/image_raw",
    camera_role: str = "front",
    camera_frame_id: str = "camera_color_optical_frame",
    image_bytes=_png_bytes,
    missing_pose=(),
    positions=None,
    odometry_frame_id="odom",
    image_paths=None,
) -> str:
    """Write a valid v4 snapshot and return the files_sha256.csv SHA-256.

    Frame ``k`` is stamped ``first_stamp_sec + k`` seconds (plus ``stamp_nsecs[k]``).
    Every frame carries an odometry pose (position ``positions[k]``, default
    ``(0.1 k, 0, 0)``) except the indices in ``missing_pose``, whose
    ``pose_valid`` is false, as capture writes a frame it could not associate.
    """
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    missing = set(missing_pose)
    files, rows = [], []

    def put(relative, data):
        path = snapshot_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        files.append((relative, len(data), hashlib.sha256(data).hexdigest()))

    stamps = []
    for index in range(int(frame_count)):
        sec = first_stamp_sec + index
        nsec = int(stamp_nsecs[index]) if stamp_nsecs is not None else 0
        stamps.append((sec, nsec))
        image_relative = image_paths[index] if image_paths is not None else f"images/{index}.png"
        put(image_relative, image_bytes(index))
        row = dict(frame_index=str(index), image_path=image_relative, stamp_sec=str(sec),
                   stamp_nsec=str(nsec), pose_valid="false", pose_delta_ms="-1",
                   odometry_gap_ms="-1", auxiliary_path="", camera_info_delta_ms="-1",
                   odometry_delta_ms="-1", depth_path="", depth_delta_ms="-1")
        if index not in missing:
            position = positions[index] if positions is not None else (0.1 * index, 0.0, 0.0)
            aux_relative = f"auxiliary/frame_{index:06d}.json"
            aux = {"frame_index": index, "camera_info": None, "camera_info_delta_ms": None,
                   "odometry": odometry_payload(sec, nsec, position, frame_id=odometry_frame_id), "odometry_delta_ms": 0.0,
                   "depth": None}
            put(aux_relative, json.dumps(aux, sort_keys=True).encode("utf-8"))
            row.update(pose_valid="true", pose_delta_ms="0", odometry_gap_ms="0",
                       auxiliary_path=aux_relative, odometry_delta_ms="0")
        rows.append(row)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FRAME_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    put("frames.csv", stream.getvalue().encode("utf-8"))
    odometry_frames = int(frame_count) - len(missing & set(range(int(frame_count))))
    metadata = {
        "schema": "drone3d.reconstruction.snapshot", "version": 4, "status": "PASS",
        "scene_id": scene_id, "frame_count": int(frame_count),
        "camera_name": camera_name, "image_topic": image_topic,
        "first_stamp_sec": stamps[0][0], "first_stamp_nsec": stamps[0][1],
        "last_stamp_sec": stamps[-1][0], "last_stamp_nsec": stamps[-1][1],
        "has_camera_info": False, "has_odometry": odometry_frames > 0,
        "camera_role": camera_role, "camera_frame_id": camera_frame_id,
        "camera_info_topic": None, "odometry_topic": "/Odometry", "depth_topic": None,
        "tf_topic": None, "tf_static_topic": None, "save_sensor_depth": False,
        "body_frame_id": "body", "has_tf": False, "has_depth": False,
        "camera_info_digest": None,
        "auxiliary_summary": {
            "camera_info_samples": 0, "odometry_samples": odometry_frames,
            "depth_samples": 0, "tf_samples": 0, "camera_info_matched_frames": 0,
            "odometry_matched_frames": odometry_frames, "depth_matched_frames": 0,
            "tf_dropped": 0, "auxiliary_errors": 0,
        },
    }
    put("snapshot.json", json.dumps(metadata, sort_keys=True).encode("utf-8"))
    files.sort(key=lambda item: item[0])
    manifest = "path,size,sha256\n" + "".join(f"{p},{s},{d}\n" for p, s, d in files)
    manifest_bytes = manifest.encode("utf-8")
    (snapshot_dir / "files_sha256.csv").write_bytes(manifest_bytes)
    return hashlib.sha256(manifest_bytes).hexdigest()
