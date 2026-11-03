"""Frame storage shared by frame-pool selection and rolling overlap windows.

The module is ROS-independent. It owns:

- importing captured window frames into a per-session frame pool,
- per-frame confidence scores (sharpness proxy -> VGGT depth_conf + point
  count after a frame has been part of one reconstruction),
- quality-ranked selection with spatial, angular and temporal coverage,
- building synthetic snapshots while preserving the v4 sensor contract.

The pool itself never runs VGGT and never talks to actions/workers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

RECONSTRUCTION_MODE_PER_WINDOW = "per_window"
RECONSTRUCTION_MODE_FRAME_POOL = "frame_pool"
RECONSTRUCTION_MODES = (
    RECONSTRUCTION_MODE_PER_WINDOW,
    RECONSTRUCTION_MODE_FRAME_POOL,
)

_SNAPSHOT_SCHEMA = "drone3d.reconstruction.snapshot"
_JSON_INDENT = 2
_SCORE_EWMA_ALPHA = 0.7


class FramePoolError(ValueError):
    """Frame pool invariant or filesystem error."""


class FramePoolCapacityError(FramePoolError):
    """The session pool reached a hard frame or byte limit."""


@dataclass(frozen=True)
class FramePoolConfig:
    """Explicit pool bounds; the user target count is passed per select() call."""

    voxel_m: float = 0.5
    view_angle_deg: float = 30.0
    max_frames: int = 20000
    max_bytes: int = 20 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        if (
            isinstance(self.voxel_m, bool)
            or not isinstance(self.voxel_m, (int, float))
            or not math.isfinite(float(self.voxel_m))
            or self.voxel_m <= 0
        ):
            raise FramePoolError("voxel_m must be positive and finite")
        if (isinstance(self.view_angle_deg, bool)
                or not isinstance(self.view_angle_deg, (int, float))
                or not math.isfinite(self.view_angle_deg) or not 0 < self.view_angle_deg <= 180):
            raise FramePoolError("view_angle_deg must be finite and in (0, 180]")
        if isinstance(self.max_frames, bool) or not isinstance(self.max_frames, int) or self.max_frames <= 0:
            raise FramePoolError("max_frames must be a positive integer")
        if isinstance(self.max_bytes, bool) or not isinstance(self.max_bytes, int) or self.max_bytes <= 0:
            raise FramePoolError("max_bytes must be a positive integer")


@dataclass
class FramePoolEntry:
    """One retained frame and its evolving confidence metadata."""

    global_index: int
    window_index: int
    snapshot_frame_index: int
    image_relative_path: str
    stamp_sec: int
    stamp_nsec: int
    pose_valid: bool
    odom_position: tuple[float, float, float] | None
    score: float
    source: str
    odom_frame_id: str | None = None
    odom_orientation: tuple[float, float, float, float] | None = None
    frame_metadata: dict | None = None
    auxiliary: dict | None = None
    depth_relative_path: str | None = None
    conf_rank: float | None = None
    point_count_rank: float | None = None
    vggt_conf_median: float | None = None
    vggt_valid_points: int | None = None
    n_updates: int = 0
    last_updated_revision: int | None = None
    prediction_backend: str = ""

    def stamp_key(self) -> tuple[int, int]:
        return (self.stamp_sec, self.stamp_nsec)


class FramePool:
    """One bounded, session-scoped frame pool."""

    def __init__(self, root: Path, config: FramePoolConfig = FramePoolConfig()) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise FramePoolError("root must be an absolute Path")
        if not isinstance(config, FramePoolConfig):
            raise FramePoolError("config must be FramePoolConfig")
        self.root = root
        self.config = config
        self.entries: list[FramePoolEntry] = []
        self.sources: dict[str, dict] = {}
        self.next_global_index = 0
        self.total_bytes = 0
        self._loaded = False

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        manifest = self.root / "pool.json"
        if not manifest.exists():
            self._loaded = True
            return
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise FramePoolError(f"pool.json unreadable: {type(error).__name__}") from error
        if not isinstance(data, dict) or not isinstance(data.get("frames"), list):
            raise FramePoolError("pool.json schema is invalid")
        entries = []
        for item in data["frames"]:
            try:
                entries.append(_entry_from_json(item))
            except (KeyError, TypeError, ValueError) as error:
                raise FramePoolError(
                    f"pool.json entry is invalid: {type(error).__name__}"
                ) from error
        self.entries = entries
        self.sources = data.get("sources", {})
        self.next_global_index = int(data.get("next_global_index", len(entries)))
        self.total_bytes = int(data.get("total_bytes", 0))
        self._loaded = True

    def _persist(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "drone3d.frame_pool",
            "version": 1,
            "next_global_index": int(self.next_global_index),
            "total_bytes": int(self.total_bytes),
            "frames": [_entry_to_json(entry) for entry in self.entries],
            "sources": self.sources,
        }
        _atomic_write_json(self.root / "pool.json", payload)

    def clear(self) -> None:
        """Discard frames from an odometry epoch that can no longer be mixed."""
        self.load()
        next_global_index = self.next_global_index
        if self.root.is_symlink() or (self.root.exists() and not self.root.is_dir()):
            raise FramePoolError(f"frame pool root is unsafe: {self.root}")
        images_dir = self.root / "images"
        if images_dir.is_symlink() or (
            images_dir.exists() and not images_dir.is_dir()
        ):
            raise FramePoolError(f"frame pool images directory is unsafe: {images_dir}")
        if images_dir.exists():
            shutil.rmtree(str(images_dir))
        self.entries = []
        self.sources = {}
        self.next_global_index = next_global_index
        self.total_bytes = 0
        self._loaded = True
        self._persist()

    def retain(self, selected: Sequence[FramePoolEntry]) -> None:
        """Keep exactly ``selected`` frames, preserving global identities."""
        self.load()
        by_index = {entry.global_index: entry for entry in self.entries}
        wanted = [int(entry.global_index) for entry in selected]
        if len(wanted) != len(set(wanted)) or any(index not in by_index for index in wanted):
            raise FramePoolError("retained frames must be unique pool entries")
        keep = set(wanted)
        paths = {
            entry.global_index: self.root / entry.image_relative_path
            for entry in self.entries
        }
        sizes = {
            entry.global_index: _path_size(paths[entry.global_index]) + (
                _path_size(self.root / entry.depth_relative_path) if entry.depth_relative_path else 0
            ) for entry in self.entries
        }
        for entry in self.entries:
            if entry.global_index not in keep:
                paths[entry.global_index].unlink()
                if entry.depth_relative_path:
                    (self.root / entry.depth_relative_path).unlink()
        self.entries = [by_index[index] for index in wanted]
        self.total_bytes = sum(sizes[index] for index in wanted)
        retained_windows = {str(entry.window_index) for entry in self.entries}
        self.sources = {key: value for key, value in self.sources.items() if key in retained_windows}
        self._persist()

    # ------------------------------------------------------------------
    # import / scoring / selection
    # ------------------------------------------------------------------
    def import_window(
        self,
        snapshot,
        window_index: int,
        *,
        sharpness_reader=None,
        copier=shutil.copy2,
    ) -> list[int]:
        """Import all frames of a validated snapshot into the pool.

        Returns the new global indices in capture order. Caller must have
        validated the snapshot through snapshot_validator first.
        """
        self.load()
        image_paths = list(getattr(snapshot, "image_paths", ()) or ())
        frame_stamps = list(getattr(snapshot, "frame_stamps", ()) or ())
        pose_valid = list(getattr(snapshot, "pose_valid", ()) or ())
        odom_payloads = list(getattr(snapshot, "odometry_payloads", ()) or ())
        if not image_paths:
            raise FramePoolError("snapshot has no image paths")
        if len(frame_stamps) != len(image_paths):
            raise FramePoolError("snapshot stamps do not match image count")
        if len(pose_valid) != len(image_paths):
            raise FramePoolError("snapshot pose_valid does not match image count")
        if odom_payloads and len(odom_payloads) != len(image_paths):
            raise FramePoolError("snapshot odometry payloads do not match image count")
        if len(odom_payloads) < len(image_paths):
            odom_payloads = [None] * len(image_paths)

        sharpness = sharpness_reader or _sharpness_score
        rows = []
        source_info = None
        if getattr(snapshot, "snapshot_version", 0) == 4:
            with (snapshot.canonical_root / "frames.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            source_info = {
                "metadata": json.loads((snapshot.canonical_root / "snapshot.json").read_text()),
                "tf_entries": list(snapshot.tf_entries),
            }
        prepared = []
        incoming_bytes = 0
        for offset, (image_path, stamp, valid, payload) in enumerate(
            zip(image_paths, frame_stamps, pose_valid, odom_payloads)
        ):
            global_index = self.next_global_index + offset
            relative = f"images/frame_{global_index:08d}.png"
            destination = self.root / relative
            if destination.exists():
                raise FramePoolError(f"pool image already exists: {destination}")
            source = Path(image_path)
            if not source.is_file() or source.is_symlink():
                raise FramePoolError(f"snapshot image is not a real file: {source}")
            incoming_bytes += _path_size(source)
            if rows and rows[offset].get("depth_path"):
                incoming_bytes += _path_size(snapshot.canonical_root / rows[offset]["depth_path"])
            prepared.append(
                (offset, global_index, relative, source, destination, stamp, valid, payload)
            )
        if len(self.entries) + len(prepared) > self.config.max_frames:
            raise FramePoolCapacityError("frame pool exceeded max_frames")
        if self.total_bytes + incoming_bytes > self.config.max_bytes:
            raise FramePoolCapacityError("frame pool exceeded max_bytes")

        images_dir = self.root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        new_entries = []
        state_applied = False
        previous_source = self.sources.get(str(window_index))
        try:
            for (
                offset,
                global_index,
                relative,
                source,
                destination,
                stamp,
                valid,
                payload,
            ) in prepared:
                copied.append(destination)
                copier(str(source), str(destination))
                if _path_size(destination) != _path_size(source):
                    raise FramePoolError(f"pool image size changed during copy: {source}")
                stamp_sec = int(getattr(stamp, "sec", 0) or 0)
                stamp_nsec = int(getattr(stamp, "nsec", 0) or 0)
                position = _odom_position(payload)
                valid_flag = True if valid is None else bool(valid)
                score = sharpness(destination)
                auxiliary = None
                depth_relative = None
                if rows and rows[offset].get("auxiliary_path"):
                    auxiliary = json.loads((snapshot.canonical_root / rows[offset]["auxiliary_path"]).read_text())
                if rows and rows[offset].get("depth_path"):
                    depth_relative = f"images/frame_{global_index:08d}.bin"
                    depth_destination = self.root / depth_relative
                    copied.append(depth_destination)
                    copier(str(snapshot.canonical_root / rows[offset]["depth_path"]), str(depth_destination))
                new_entries.append(
                    FramePoolEntry(
                        global_index=global_index,
                        window_index=int(window_index),
                        snapshot_frame_index=offset,
                        image_relative_path=relative,
                        stamp_sec=stamp_sec,
                        stamp_nsec=stamp_nsec,
                        pose_valid=valid_flag,
                        odom_position=position,
                        score=float(score),
                        source="sharpness_proxy",
                        odom_frame_id=_odom_frame_id(payload),
                        odom_orientation=_odom_orientation(payload),
                        frame_metadata=rows[offset] if rows else None,
                        auxiliary=auxiliary,
                        depth_relative_path=depth_relative,
                    )
                )
            self.entries.extend(new_entries)
            self.next_global_index += len(new_entries)
            self.total_bytes += incoming_bytes
            if source_info is not None:
                self.sources[str(window_index)] = source_info
            state_applied = True
            self._persist()
        except BaseException:
            if state_applied:
                del self.entries[-len(new_entries):]
                self.next_global_index -= len(new_entries)
                self.total_bytes -= incoming_bytes
                if previous_source is None:
                    self.sources.pop(str(window_index), None)
                else:
                    self.sources[str(window_index)] = previous_source
            for destination in copied:
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass
            raise
        return [entry.global_index for entry in new_entries]

    def update_from_predictions(
        self, predictions_path: Path, selected_global_indices: Sequence[int], revision: int,
        *, backend: str = "omega"
    ) -> None:
        """Update scores for frames that participated in one global run.

        depth_conf and finite world-point count are each converted to a
        within-run percentile rank, combined 50/50, then EWMA-ed into the
        frame's existing score.
        """
        self.load()
        if backend not in ("omega",):
            raise FramePoolError("unknown prediction backend")
        if not selected_global_indices:
            return
        try:
            import numpy as np

            with np.load(str(predictions_path), allow_pickle=False) as archive:
                conf = archive["depth_conf"]
                world = archive["world_points_from_depth"]
        except Exception as error:
            raise FramePoolError(
                f"predictions.npz unusable: {type(error).__name__}"
            ) from error
        if conf.shape[0] != len(selected_global_indices):
            raise FramePoolError("predictions frame count does not match selection")
        conf_medians = []
        point_counts = []
        for index in range(conf.shape[0]):
            frame_conf = np.asarray(conf[index], dtype=np.float64).reshape(-1)
            valid_conf = frame_conf[np.isfinite(frame_conf)]
            conf_medians.append(
                float(np.median(valid_conf)) if valid_conf.size else 0.0
            )
            frame_world = np.asarray(world[index], dtype=np.float64)
            finite_points = (
                np.isfinite(frame_world).all(axis=-1).sum()
                if frame_world.ndim == 3 and frame_world.shape[-1] == 3
                else int(np.isfinite(frame_world).sum())
            )
            point_counts.append(int(finite_points))
        conf_ranks = _percentile_ranks(conf_medians)
        point_ranks = _percentile_ranks(point_counts)
        by_index = {entry.global_index: entry for entry in self.entries}
        for global_index, conf_rank, point_rank, conf_median, point_count in zip(
            selected_global_indices, conf_ranks, point_ranks, conf_medians, point_counts
        ):
            entry = by_index.get(int(global_index))
            if entry is None:
                continue
            vggt_score = 0.5 * conf_rank + 0.5 * point_rank
            entry.score = (
                _SCORE_EWMA_ALPHA * vggt_score
                + (1.0 - _SCORE_EWMA_ALPHA) * entry.score
                if entry.n_updates
                else vggt_score
            )
            entry.source = "vggt_depth_conf"
            entry.prediction_backend = backend
            entry.conf_rank = float(conf_rank)
            entry.point_count_rank = float(point_rank)
            entry.vggt_conf_median = float(conf_median)
            entry.vggt_valid_points = int(point_count)
            entry.n_updates += 1
            entry.last_updated_revision = int(revision)
        self._persist()

    def select(
        self,
        selection_size: int,
        *,
        voxel_m: float | None = None,
    ) -> list[FramePoolEntry]:
        """Select fresh from the full pool: quality first, then viewpoint coverage.

        Prefer distinct positions/orientations; fill remaining slots by quality
        rather than returning fewer frames due to hard spatial suppression.
        Missing poses fall back to temporal coverage. Return capture order.
        """
        self.load()
        if isinstance(selection_size, bool) or not isinstance(selection_size, int) or selection_size <= 0:
            raise FramePoolError("selection_size must be a positive integer")
        step = float(voxel_m if voxel_m is not None else self.config.voxel_m)
        if not math.isfinite(step) or step <= 0:
            raise FramePoolError("voxel_m must be positive and finite")
        candidates = [entry for entry in self.entries if math.isfinite(entry.score) and entry.score > 0]
        candidates.sort(key=lambda entry: (-entry.score, -entry.global_index))
        if not candidates:
            return []
        first_stamp = min(entry.stamp_sec + entry.stamp_nsec / 1e9 for entry in candidates)
        last_stamp = max(entry.stamp_sec + entry.stamp_nsec / 1e9 for entry in candidates)
        time_step = max((last_stamp - first_stamp) / selection_size, 1e-9)
        occupied = {}
        time_bins = set()
        chosen = []
        deferred = []
        for entry in candidates:
            if len(chosen) >= selection_size:
                break
            if entry.pose_valid and entry.odom_position is not None and entry.odom_orientation is not None:
                key = tuple(int(math.floor(axis / step)) for axis in entry.odom_position)
                orientations = occupied.setdefault(key, [])
                # Quaternion sign is arbitrary; use the configured angular separation.
                duplicate = any(abs(sum(a * b for a, b in zip(entry.odom_orientation, old)))
                                >= math.cos(math.radians(self.config.view_angle_deg / 2)) for old in orientations)
                if duplicate:
                    deferred.append(entry)
                    continue
                orientations.append(entry.odom_orientation)
            else:
                stamp = entry.stamp_sec + entry.stamp_nsec / 1e9
                key = min(selection_size - 1, int((stamp - first_stamp) / time_step))
                if key in time_bins:
                    deferred.append(entry)
                    continue
                time_bins.add(key)
            chosen.append(entry)
        if len(chosen) < selection_size:
            chosen.extend(deferred[:selection_size - len(chosen)])
        chosen.sort(key=lambda entry: (entry.stamp_key(), entry.global_index))
        return chosen

    def write_selection(
        self, revision: int, selected: Sequence[FramePoolEntry]
    ) -> Path:
        self.load()
        selections_dir = self.root / "selections"
        selections_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "revision": int(revision),
            "pool_size": len(self.entries),
            "voxel_m": float(self.config.voxel_m),
            "view_angle_deg": float(self.config.view_angle_deg),
            "selection_policy": "quality_spatial_angular_temporal",
            "candidate_count": sum(math.isfinite(e.score) and e.score > 0 for e in self.entries),
            "selected_count": len(selected),
            "selected_global_indices": [int(entry.global_index) for entry in selected],
            "scores": {str(entry.global_index): float(entry.score) for entry in selected},
            "confidence_summary": self.confidence_summary(selected),
            "selected_window_histogram": _histogram(
                entry.window_index for entry in selected
            ),
            "selected_source_histogram": _histogram(entry.source for entry in selected),
            "min_selected_odom_distance_m": _min_odom_distance(selected),
        }
        path = selections_dir / f"p{int(revision):06d}.json"
        _atomic_write_json(path, payload)
        return path

    @staticmethod
    def confidence_summary(selected: Sequence[FramePoolEntry]) -> dict[str, Any]:
        scores = [float(entry.score) for entry in selected]
        return {
            "selected_count": len(scores),
            "score_min": min(scores) if scores else None,
            "score_mean": sum(scores) / len(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "source_histogram": _histogram(entry.source for entry in selected),
        }

    # ------------------------------------------------------------------
    # synthetic snapshot for the existing ProductionPlanner path
    # ------------------------------------------------------------------
    def build_synthetic_snapshot(
        self,
        revision: int,
        selected: Sequence[FramePoolEntry],
        destination: Path,
        scene_id: str,
        *,
        camera_name: str,
        image_topic: str,
        camera_role: str,
        camera_frame_id: str,
        preserve_order: bool = False,
        copier=shutil.copy2,
        limits=None,
    ) -> tuple[str, str]:
        """Build a snapshot, preserving v4 auxiliary data when available.

        The destination directory must be an absolute, non-existing path under
        a caller-chosen trusted snapshot root.
        Legacy entries without auxiliary metadata use the v3 RGB contract.
        """
        if not destination.is_absolute():
            raise FramePoolError("synthetic snapshot destination must be absolute")
        if destination.exists():
            raise FramePoolError("synthetic snapshot destination already exists")
        if not selected:
            raise FramePoolError("cannot build synthetic snapshot from no frames")
        for value in (camera_name, image_topic, camera_role, camera_frame_id):
            if not isinstance(value, str) or not value:
                raise FramePoolError("camera identity fields must be nonempty strings")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
        )
        try:
            images_dir = staging / "images"
            images_dir.mkdir()
            rows = []
            manifest = {}
            ordered = list(selected) if preserve_order else sorted(
                selected, key=lambda entry: (entry.stamp_key(), entry.global_index)
            )
            if limits is not None and len(ordered) > limits.max_frame_count:
                raise FramePoolError("synthetic snapshot exceeds frame limit")
            v4 = any(entry.frame_metadata is not None for entry in ordered)
            auxiliaries = []
            for order, entry in enumerate(ordered):
                relative = f"images/frame_{order:06d}.png"
                source = self.root / entry.image_relative_path
                target = staging / relative
                if not source.is_file() or source.is_symlink():
                    raise FramePoolError(f"pool image missing: {source}")
                copier(str(source), str(target))
                size = _path_size(target)
                digest = _sha256_file(target)
                manifest[relative] = (size, digest)
                rows.append(
                    {
                        "frame_index": str(order),
                        "image_path": relative,
                        "stamp_sec": str(int(entry.stamp_sec)),
                        "stamp_nsec": str(int(entry.stamp_nsec)),
                        "pose_valid": "false",
                        "pose_delta_ms": "-1",
                        "odometry_gap_ms": "-1",
                    }
                )
                if v4:
                    row = rows[-1]
                    if entry.frame_metadata is not None:
                        row.update(entry.frame_metadata)
                    row.update(frame_index=str(order), image_path=relative,
                               auxiliary_path="", depth_path="")
                    for key in ("camera_info_delta_ms", "odometry_delta_ms", "depth_delta_ms"):
                        row.setdefault(key, "-1")
                    auxiliary = json.loads(json.dumps(entry.auxiliary)) if entry.auxiliary else None
                    auxiliaries.append(auxiliary)
                    if auxiliary is not None:
                        auxiliary["frame_index"] = order
                        if auxiliary.get("depth") is not None:
                            depth_relative = f"depth/frame_{order:06d}.bin"
                            (staging / "depth").mkdir(exist_ok=True)
                            copier(str(self.root / entry.depth_relative_path), str(staging / depth_relative))
                            auxiliary["depth"]["data_file"] = depth_relative
                            row["depth_path"] = depth_relative
                            manifest[depth_relative] = (_path_size(staging / depth_relative), _sha256_file(staging / depth_relative))
                        aux_relative = f"auxiliary/frame_{order:06d}.json"
                        (staging / "auxiliary").mkdir(exist_ok=True)
                        aux_bytes = json.dumps(auxiliary, allow_nan=False).encode()
                        _write_bytes(staging / aux_relative, aux_bytes)
                        row["auxiliary_path"] = aux_relative
                        manifest[aux_relative] = (len(aux_bytes), hashlib.sha256(aux_bytes).hexdigest())
            frames_bytes = _csv_bytes(
                (
                    "frame_index",
                    "image_path",
                    "stamp_sec",
                    "stamp_nsec",
                    "pose_valid",
                    "pose_delta_ms",
                    "odometry_gap_ms",
                ) + (("auxiliary_path", "camera_info_delta_ms", "odometry_delta_ms", "depth_path", "depth_delta_ms") if v4 else ()),
                rows,
            )
            snapshot = {
                "schema": _SNAPSHOT_SCHEMA,
                "version": 3,
                "status": "PASS",
                "scene_id": scene_id,
                "frame_count": len(rows),
                "camera_name": camera_name,
                "image_topic": image_topic,
                "first_stamp_sec": int(ordered[0].stamp_sec),
                "first_stamp_nsec": int(ordered[0].stamp_nsec),
                "last_stamp_sec": int(ordered[-1].stamp_sec),
                "last_stamp_nsec": int(ordered[-1].stamp_nsec),
                "has_camera_info": False,
                "has_odometry": False,
                "camera_role": camera_role,
                "camera_frame_id": camera_frame_id,
            }
            if v4:
                source_windows = dict.fromkeys(
                    str(entry.window_index) for entry in ordered
                    if entry.frame_metadata is not None
                )
                source_infos = [self.sources[key] for key in source_windows]
                if len({info["metadata"]["body_frame_id"] for info in source_infos}) > 1:
                    raise FramePoolError("body frame changed across synthetic snapshot")
                metadata = source_infos[-1]["metadata"]
                for key in ("camera_info_topic", "odometry_topic", "depth_topic", "tf_topic", "tf_static_topic", "body_frame_id"):
                    snapshot[key] = metadata[key]
                tf_entries = list({json.dumps(entry, sort_keys=True): entry
                                   for info in source_infos for entry in info["tf_entries"]}.values())
                tf_entries = _selected_tf_entries(tf_entries, ordered)
                if tf_entries:
                    tf_bytes = json.dumps(tf_entries, allow_nan=False).encode()
                    _write_bytes(staging / "tf.json", tf_bytes)
                    manifest["tf.json"] = (len(tf_bytes), hashlib.sha256(tf_bytes).hexdigest())
                summary = {"tf_samples": len(tf_entries), "tf_dropped": 0, "auxiliary_errors": 0}
                digest = hashlib.sha256()
                for name in ("camera_info", "odometry", "depth"):
                    payloads = [aux[name] for aux in auxiliaries if aux and aux[name] is not None]
                    summary[f"{name}_samples"] = summary[f"{name}_matched_frames"] = len(payloads)
                    snapshot[f"has_{name}"] = bool(payloads)
                    if name == "camera_info":
                        for payload in payloads:
                            digest.update(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode())
                            digest.update(b"\n")
                snapshot.update(version=4, has_tf=bool(tf_entries), auxiliary_summary=summary,
                                save_sensor_depth=any(info["metadata"]["save_sensor_depth"] for info in source_infos),
                                camera_info_digest=digest.hexdigest() if snapshot["has_camera_info"] else None)
            snapshot_bytes = json.dumps(
                snapshot, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            _write_bytes(staging / "snapshot.json", snapshot_bytes)
            _write_bytes(staging / "frames.csv", frames_bytes)
            manifest["snapshot.json"] = (
                len(snapshot_bytes),
                hashlib.sha256(snapshot_bytes).hexdigest(),
            )
            manifest["frames.csv"] = (
                len(frames_bytes),
                hashlib.sha256(frames_bytes).hexdigest(),
            )
            if limits is not None:
                sizes = [size for size, _digest in manifest.values()]
                if (len(manifest) > limits.max_manifest_entries or sum(sizes) > limits.max_total_bytes
                        or any(size > limits.max_file_bytes for size in sizes)):
                    raise FramePoolError("synthetic snapshot exceeds file/byte limits")
                if any(size > limits.max_metadata_bytes for path, (size, _) in manifest.items()
                       if path.endswith((".json", ".csv"))):
                    raise FramePoolError("synthetic snapshot exceeds metadata limit")
            manifest_rows = [
                {"path": path, "size": str(size), "sha256": digest}
                for path, (size, digest) in sorted(manifest.items())
            ]
            manifest_bytes = _csv_bytes(("path", "size", "sha256"), manifest_rows)
            if limits is not None and len(manifest_bytes) > limits.max_metadata_bytes:
                raise FramePoolError("synthetic manifest exceeds metadata limit")
            _write_bytes(staging / "files_sha256.csv", manifest_bytes)
            digest = hashlib.sha256(manifest_bytes).hexdigest()
            os.replace(staging, destination)
            return destination.as_uri(), digest
        except BaseException:
            if staging.exists() and not staging.is_symlink():
                shutil.rmtree(str(staging))
            raise


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _selected_tf_entries(entries, frames):
    """Keep static TF and dynamic samples bracketing the selected time span.

    Repeated rebasing must not recursively accumulate the anchor's entire
    dynamic TF history. Boundary samples remain available for interpolation.
    """
    first = min(frame.stamp_key() for frame in frames)
    last = max(frame.stamp_key() for frame in frames)
    retained, before, after = [], {}, {}
    for entry in entries:
        if entry["static"]:
            retained.append(entry)
            continue
        stamp = (entry["stamp"]["sec"], entry["stamp"]["nsec"])
        edge = (entry["parent_frame_id"], entry["child_frame_id"])
        if stamp < first:
            if edge not in before or stamp > before[edge][0]:
                before[edge] = (stamp, entry)
        elif stamp > last:
            if edge not in after or stamp < after[edge][0]:
                after[edge] = (stamp, entry)
        else:
            retained.append(entry)
    retained.extend(value[1] for value in before.values())
    retained.extend(value[1] for value in after.values())
    return sorted(retained, key=lambda entry: (
        entry["stamp"]["sec"], entry["stamp"]["nsec"],
        entry["parent_frame_id"], entry["child_frame_id"],
    ))


def _histogram(values) -> dict[str, int]:
    result = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def _min_odom_distance(entries: Sequence[FramePoolEntry]) -> float | None:
    positions = [entry.odom_position for entry in entries if entry.odom_position is not None]
    if len(positions) < 2:
        return None
    return min(
        math.dist(positions[left], positions[right])
        for left in range(len(positions))
        for right in range(left + 1, len(positions))
    )


def _entry_to_json(entry: FramePoolEntry) -> dict[str, Any]:
    return {
        "global_index": int(entry.global_index),
        "window_index": int(entry.window_index),
        "snapshot_frame_index": int(entry.snapshot_frame_index),
        "image_relative_path": entry.image_relative_path,
        "stamp_sec": int(entry.stamp_sec),
        "stamp_nsec": int(entry.stamp_nsec),
        "pose_valid": bool(entry.pose_valid),
        "odom_position": (
            [float(value) for value in entry.odom_position]
            if entry.odom_position is not None
            else None
        ),
        "score": float(entry.score),
        "source": entry.source,
        "odom_frame_id": entry.odom_frame_id,
        "frame_metadata": entry.frame_metadata,
        "auxiliary": entry.auxiliary,
        "depth_relative_path": entry.depth_relative_path,
        "odom_orientation": (
            [float(value) for value in entry.odom_orientation]
            if entry.odom_orientation is not None
            else None
        ),
        "conf_rank": entry.conf_rank,
        "point_count_rank": entry.point_count_rank,
        "vggt_conf_median": entry.vggt_conf_median,
        "vggt_valid_points": entry.vggt_valid_points,
        "n_updates": int(entry.n_updates),
        "last_updated_revision": entry.last_updated_revision,
        "prediction_backend": entry.prediction_backend,
    }


def _entry_from_json(item: dict[str, Any]) -> FramePoolEntry:
    position = item.get("odom_position")
    orientation = item.get("odom_orientation")
    return FramePoolEntry(
        global_index=int(item["global_index"]),
        window_index=int(item["window_index"]),
        snapshot_frame_index=int(item["snapshot_frame_index"]),
        image_relative_path=str(item["image_relative_path"]),
        stamp_sec=int(item["stamp_sec"]),
        stamp_nsec=int(item["stamp_nsec"]),
        pose_valid=bool(item["pose_valid"]),
        odom_position=(
            tuple(float(value) for value in position)
            if isinstance(position, (list, tuple))
            else None
        ),
        score=float(item["score"]),
        source=str(item["source"]),
        frame_metadata=item.get("frame_metadata"),
        auxiliary=item.get("auxiliary"),
        depth_relative_path=item.get("depth_relative_path"),
        odom_frame_id=(
            str(item["odom_frame_id"])
            if isinstance(item.get("odom_frame_id"), str)
            and item["odom_frame_id"]
            else None
        ),
        odom_orientation=(
            tuple(float(value) for value in orientation)
            if isinstance(orientation, (list, tuple)) and len(orientation) == 4
            else None
        ),
        conf_rank=item.get("conf_rank"),
        point_count_rank=item.get("point_count_rank"),
        vggt_conf_median=item.get("vggt_conf_median"),
        vggt_valid_points=item.get("vggt_valid_points"),
        n_updates=int(item.get("n_updates", 0)),
        last_updated_revision=item.get("last_updated_revision"),
        prediction_backend=item.get("prediction_backend", ""),
    )


def _odom_position(payload) -> tuple[float, float, float] | None:
    if not isinstance(payload, dict):
        return None
    try:
        position = payload.get("pose", {}).get("position", {})
        result = tuple(
            float(position[key]) for key in ("x", "y", "z")
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in result):
        return None
    return result


def _odom_frame_id(payload) -> str | None:
    if not isinstance(payload, dict):
        return None
    frame_id = payload.get("header", {}).get("frame_id")
    return frame_id if isinstance(frame_id, str) and frame_id else None


def _odom_orientation(payload) -> tuple[float, float, float, float] | None:
    if not isinstance(payload, dict):
        return None
    try:
        orientation = payload.get("pose", {}).get("orientation", {})
        result = tuple(float(orientation[key]) for key in ("x", "y", "z", "w"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    norm = math.sqrt(sum(value * value for value in result))
    if not all(math.isfinite(value) for value in result) or norm <= 0.0:
        return None
    return tuple(value / norm for value in result)


def _sharpness_score(path: Path, fallback: float = 0.0) -> float:
    try:
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            return fallback
        variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
        if not math.isfinite(variance):
            return fallback
        # 640x480 灰度经验标定：200 对应"清晰可用"，分数封顶到 1.0。
        return min(1.0, variance / 200.0)
    except BaseException:
        return fallback


def _percentile_ranks(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    result = []
    for value in values:
        less = 0
        for other in ordered:
            if other < value:
                less += 1
            elif other == value:
                continue
        result.append((less + 0.5) / count)
    return result


def _path_size(path: Path) -> int:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FramePoolError(f"path is not a trusted regular file: {path}")
    return int(info.st_size)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_bytes(columns: Sequence[str], rows: Sequence[dict[str, str]]) -> bytes:
    import io

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row[key] for key in columns})
    return stream.getvalue().encode("utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, sort_keys=True, indent=_JSON_INDENT).encode("utf-8")
    data += b"\n"
    _write_bytes(path, data)
