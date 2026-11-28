"""Incremental file ingestion and atomic, complete map revisions.

This module has no rospy dependency. Optimization consumes immutable local
geometry. Old placements are never fed back as model predictions. Each revision
re-solves every available window: frames are placed with the per-frame external
localization poses and VGGT supplies depth and a metric scale per window or per
frame (g4_backend.frame_pose).
"""
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import scipy
import g4_backend
from g4_backend import (FRAME_POSE_METHOD_ID, Sim3, FramePoseConfig, sim3,
                        interpolate_poses, resize_intrinsics, rotation_mean, shared_depth_ratio,
                        solve_window_scales, trajectory_prior, window_translation)
from .input_adapter import WindowInputReader, sha256
from .._common import fsync_directory as _sync_dir


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write(path, value):
    with open(path, "x") as stream:
        json.dump(_plain(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _commit_directory(staging, final):
    for path in staging.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    _sync_dir(staging)
    os.rename(staging, final)
    _sync_dir(final.parent)


def _binding(record):
    return _digest({key: record.get(key) for key in (
        "window_index", "result_manifest_sha256", "snapshot_manifest_sha256",
        "capture_generation", "model_frame_ids", "overlap_frame_count",
    )})


def _blank(value):
    """True for "no input given": None, an empty string or whitespace."""
    return value is None or (isinstance(value, str) and not value.strip())


def _check_intrinsics(value):
    """Calibrated source-image pinhole: {width, height, fx, fy, cx, cy}."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"width", "height", "fx", "fy", "cx", "cy"}:
        raise ValueError("camera_intrinsics needs exactly width, height, fx, fy, cx, cy")
    out = {k: float(value[k]) for k in ("fx", "fy", "cx", "cy")}
    out.update(width=int(value["width"]), height=int(value["height"]))
    if (not all(np.isfinite(v) for v in out.values()) or min(out["fx"], out["fy"]) <= 0
            or min(out["width"], out["height"]) <= 0):
        raise ValueError("camera_intrinsics must be finite with positive focal lengths and size")
    return out


# Keys that sessions written before graph placement and registered window plans
# were removed may still carry. Their neutral values are accepted; anything else
# means the session used a deleted feature and cannot be re-solved by this code.
_REMOVED_PROFILE_KEYS = {
    "window_frame_semantics": ("total_frames",),
    "window_plan_path": ("",),
    "threshold_table_path": ("",),
    "threshold_group": ("",),
    "fit_exclusion_source": ("",),
    "registration": ({},),
}


def build_backend(records_root, session_id, output_root, profile=None):
    """Construct the map backend from the session's backend_profile.json mapping."""
    profile = dict(profile or {})
    placement = profile.get("placement") or "graph"
    if placement != "frame_pose":
        raise ValueError(f"session uses placement {placement!r}; only frame_pose is supported "
                         "(graph placement was removed)")
    for key, neutral in _REMOVED_PROFILE_KEYS.items():
        value = profile.get(key)
        if value is not None and not (_blank(value) and "" in neutral) and value not in neutral:
            raise ValueError(f"session profile sets {key}={value!r}, a removed feature "
                             "(graph registration or registered window plans)")
    return G4MapBackend(
        records_root, session_id, output_root,
        method_id=profile.get("method_id") or None,
        frame_pose_config=FramePoseConfig(**profile.get("frame_pose", {})),
        camera_intrinsics=profile.get("camera_intrinsics") or None)


def _pack_cross(rows):
    return dict(window_i=np.array(rows["window_i"], np.int64), window_j=np.array(rows["window_j"], np.int64),
                frame_i=np.array(rows["frame_i"], dtype="U256"), frame_j=np.array(rows["frame_j"], dtype="U256"),
                value=np.array(rows["value"], np.float64), pixels=np.array(rows["pixels"], np.int64))


class G4MapBackend:
    def __init__(self, records_root, session_id, output_root, frame_id="g4_map",
                 *, reader=None, scanner=None, method_id=None, frame_pose_config=None,
                 camera_intrinsics=None):
        if not re.fullmatch(r"s[0-9a-f]{16}", session_id):
            raise ValueError("expected a formal session ID")
        if not frame_id or frame_id.startswith("/") or any(c.isspace() for c in frame_id):
            raise ValueError("invalid output frame_id")
        if method_id is None:
            method_id = FRAME_POSE_METHOD_ID
        if not isinstance(method_id, str) or not method_id:
            raise ValueError("method_id must be a nonempty string")
        self.frame_pose_config = frame_pose_config or FramePoseConfig()
        self.camera_intrinsics = _check_intrinsics(camera_intrinsics)
        self.records_root = Path(records_root).resolve()
        self.output_root = Path(output_root).resolve()
        if (self.output_root == self.records_root or self.records_root in self.output_root.parents
                or self.output_root in self.records_root.parents):
            raise ValueError("G4 output must be separate from immutable formal records")
        self.session_id, self.frame_id = session_id, frame_id
        self.method_id = method_id
        if scanner is None:
            from drone3d_reconstruction_core.formal_result import scan_session_windows
            scanner = scan_session_windows
        self.reader, self.scanner = reader or WindowInputReader(), scanner
        if (self.output_root.exists() and any(p.name not in {"runtime.log", "backend_profile.json"}
                                              for p in self.output_root.iterdir())
                and not (self.output_root / "run.json").exists()):
            raise ValueError("output directory is nonempty and is not an owned G4 run")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._lock = open(self.output_root / ".writer.lock", "a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            raise RuntimeError("another G4 writer owns this output directory")
        config = {"schema": "drone3d.g4.run.v1", "method_id": self.method_id,
                  "records_root": str(self.records_root), "session_id": session_id,
                  "frame_id": frame_id, "placement": "frame_pose",
                  "frame_pose": asdict(self.frame_pose_config), "camera_intrinsics": self.camera_intrinsics,
                  "formal_geometry_acceptance": False,
                  "runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
                  "core_sources": {p.name: sha256(p) for p in sorted(Path(g4_backend.__file__).parent.glob("*.py"))},
                  "adapter_sources": {name: sha256(Path(__file__).parent / name)
                                      for name in ("map_backend.py", "input_adapter.py")}}
        config_path = self.output_root / "run.json"
        if config_path.exists():
            if _read(config_path) != config:
                self.close()
                raise ValueError("output directory belongs to another session/configuration")
        else:
            _write(config_path, config)
        self.config_digest = _digest(config)
        (self.output_root / "windows").mkdir(exist_ok=True)
        (self.output_root / "revisions").mkdir(exist_ok=True)
        self._announced = None
        self._status = {"state": "waiting", "method_id": self.method_id, "session_id": session_id}

    def close(self):
        if not self._lock.closed:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()

    def status(self):
        return dict(self._status)

    def _window_path(self, index):
        return self.output_root / "windows" / ("window-%06d" % index)

    def _read_window(self, index):
        root = self._window_path(index)
        meta = _read(root / "window.json")
        if sha256(root / "window.json") != (root / "window.sha256").read_text().strip():
            raise ValueError("cached window metadata changed")
        for name, digest in meta["artifacts"].items():
            if sha256(root / name) != digest:
                raise ValueError("cached G4 window artifact changed: " + name)
        return meta

    def _cache_window(self, record, previous_record, earlier=()):
        """Cache one committed window. ``earlier``: metadata of the cached windows before it."""
        loaded = self.reader.load(record)
        w, meta = loaded.geometry, dict(loaded.metadata)
        # "edge" and "registration" stay in window.json: they are part of the cached
        # window format (no Sim(3) edge is estimated; scales come from shared depth).
        meta["edge"] = None
        if previous_record is None:
            # A preceding inference may have failed. These images have no
            # committed owner, so the first verified root owns its full input.
            meta["declared_shared_prefix"] = w.shared_prefix
            meta["shared_prefix"] = 0
            meta["registration"] = {"admitted": False, "reason":
                "root_after_missing_predecessor" if w.shared_prefix else "root_window"}
        else:
            previous = self.reader.load(previous_record)
            for key in ("capture_generation", "camera_topic", "camera_frame_id"):
                if previous.metadata[key] != meta[key]:
                    raise ValueError("camera/session identity changed: " + key)
            if previous.metadata.get("attention_backend", "auto") != meta.get("attention_backend", "auto"):
                raise ValueError("attention backend changed across windows; start a new session")
            old_ext, new_ext = previous.metadata.get("body_from_camera"), meta.get("body_from_camera")
            if old_ext is not None and new_ext is not None and any(
                    not np.allclose(old_ext[k], new_ext[k], atol=1e-8, rtol=0)
                    for k in ("rotation", "translation")):
                raise ValueError("fixed body-camera extrinsic changed across windows")
            if w.shared_prefix and all(f in previous.geometry.frame_ids for f in w.frame_ids[:w.shared_prefix]):
                # Only a relative depth scale is needed between the windows.
                ratio, used = shared_depth_ratio(previous.geometry, w)
                meta["scale_edge"] = None if ratio is None else {
                    "i": previous.geometry.window_id, "j": w.window_id, "scale": ratio, "frames": used}
                meta["registration"] = {"admitted": False, "reason": "frame_pose_uses_shared_depth_ratio"}
            else:
                meta["registration"] = {"admitted": False, "reason": "no_complete_shared_predecessor"}
                if w.shared_prefix and not any(
                        f in previous.geometry.frame_ids for f in w.frame_ids[:w.shared_prefix]):
                    # A failed middle window has no committed display ownership.
                    # Keep the unowned prefix while leaving its missing edge absent.
                    meta["declared_shared_prefix"] = w.shared_prefix
                    meta["shared_prefix"] = 0
            del previous
        meta.update(input_binding=_binding(record), geometry_accepted=False,
                    method_id=self.method_id)
        if loaded.reference_rotations is None:
            raise ValueError("frame_pose placement needs localization camera rotations")
        extra = {"reference_rotations": np.where(loaded.reference_valid[:, None, None],
                                                 loaded.reference_rotations, 0.)}
        meta["intrinsics"] = self._model_intrinsics(w, meta)
        cross = None
        if self._measures_rows():
            stride = self.frame_pose_config.cross_window_stride
            owned = np.arange(int(meta["shared_prefix"]), len(w.frame_ids))
            cam = dict(centers=w.centers, rotations=w.rotations, reference_centers=loaded.reference_centers,
                       reference_valid=loaded.reference_valid, **extra)
            cross = dict(owned=owned, depth=np.ascontiguousarray(w.depth[owned][:, ::stride, ::stride], np.float32))
            cross.update(self._measure_cross(w.window_id, meta, cam, cross, earlier))
        target = self._window_path(w.window_id)
        staging = Path(tempfile.mkdtemp(prefix=".window-", dir=target.parent))
        try:
            np.savez_compressed(staging / "cameras.npz", centers=w.centers, rotations=w.rotations,
                                timestamps_ns=w.timestamps_ns, reference_centers=loaded.reference_centers,
                                reference_valid=loaded.reference_valid, **extra)
            if cross is not None:
                np.savez_compressed(staging / "cross.npz", **cross)
            meta["artifacts"] = {p.name: sha256(p) for p in staging.iterdir()}
            _write(staging / "window.json", meta)
            (staging / "window.sha256").write_text(sha256(staging / "window.json") + "\n")
            _commit_directory(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return meta

    def _measures_rows(self):
        cfg = self.frame_pose_config
        return cfg.cross_window_weight > 0 or cfg.frame_scales

    def _scale_problem(self, entries):
        """Trajectory priors, adjacent edges and (window mode) weighted cross-window rows
        of (window_id, meta, cameras, cross arrays or None)."""
        cfg = self.frame_pose_config
        priors, edges = {}, []
        for i, meta, cam, cross in entries:
            priors[i] = trajectory_prior(cam["centers"], cam["rotations"], cam["reference_centers"],
                                         cam["reference_rotations"], cam["reference_valid"])
            if meta.get("scale_edge"):
                edges.append((meta["scale_edge"]["i"], meta["scale_edge"]["j"], float(meta["scale_edge"]["scale"])))
            if cfg.cross_window_weight > 0 and cross is not None and "window_i" in cross:
                # Row x_i - x_j = value, i.e. s_i = s_j * exp(value). The window being
                # measured has no rows yet.
                for wi, wj, value in zip(cross["window_i"], cross["window_j"], cross["value"]):
                    edges.append((int(wj), int(wi), float(np.exp(value)), cfg.cross_window_weight))
        return priors, edges

    def _frame_solve(self, entries):
        """Per-frame scales over ``entries`` (recon_core.scale.solve_frame_scales).

        Frames are each window's owned frames with a localization pose; temporal pairs
        link consecutive ones of one window; rows come from every window's cross.npz.
        Returns (window scales: id -> float, frame scales: id -> array over the window's
        model frames, NaN where a frame has no scale).
        """
        from recon_core.scale import solve_frame_scales
        priors, edges = self._scale_problem(entries)
        index, frame_window, temporal = {}, [], []
        for i, meta, cam, cross in entries:
            valid, previous = cam["reference_valid"].astype(bool), None
            for k in cross["owned"]:
                if valid[k]:
                    index[(i, str(meta["frame_ids"][k]))] = len(frame_window)
                    if previous is not None:
                        temporal.append((previous, len(frame_window)))
                    previous = len(frame_window)
                    frame_window.append(i)
        rows = []
        for _, _, _, cross in entries:
            if "window_i" not in cross:
                continue
            for wi, wj, fi, fj, value in zip(cross["window_i"], cross["window_j"], cross["frame_i"],
                                             cross["frame_j"], cross["value"]):
                a, b = index.get((int(wi), str(fi))), index.get((int(wj), str(fj)))
                if a is not None and b is not None:
                    rows.append((a, b, float(value)))
        window_scales, solved, _ = solve_frame_scales([e[0] for e in entries], priors, edges,
                                                      np.array(frame_window, np.int64), rows, temporal)
        frame_scales = {}
        for i, meta, _, _ in entries:
            scales = np.full(len(meta["frame_ids"]), np.nan)
            for k, fid in enumerate(meta["frame_ids"]):
                n = index.get((i, str(fid)))
                if n is not None:
                    scales[k] = solved[n]
            frame_scales[i] = scales
        return window_scales, frame_scales, len(rows)

    def _measure_cross(self, window_id, meta, cam, cross, earlier):
        """Depth rows between this window's frames and the cached windows before it (and,
        with frame_scales, between this window's own frames).

        Deterministic in the committed inputs: frame pairs are the fusion neighbours
        (recon_core.fusion.ranked_partners, symmetric) among the frames cached so far, and
        depths are linearised at the window scales solved over the same windows. Each row
        stores log s_i - log s_j - log(z / d_j) at those scales; later changes are small.
        """
        from scipy.spatial import cKDTree
        from recon_core.fusion import NEIGHBOR_RADIUS_M, ranked_partners
        from recon_core.scale import cross_view_log_ratio
        cfg = self.frame_pose_config
        entries, depths = [], {}
        for m in earlier:
            path = self._window_path(m["window_id"])
            if not (path / "cross.npz").exists():
                raise ValueError("cached window lacks cross-window depth; use a new output directory")
            with np.load(path / "cameras.npz", allow_pickle=False) as a:
                other = {k: a[k] for k in a.files}
            with np.load(path / "cross.npz", allow_pickle=False) as a:
                # Depth maps are read only for windows that end up in a pair.
                other_cross = {k: a[k] for k in a.files if k != "depth"}
            depths[m["window_id"]] = path / "cross.npz"
            entries.append((m["window_id"], m, other, other_cross))
        entries.append((window_id, meta, cam, cross))
        if cfg.frame_scales:
            scales = self._frame_solve(entries)[0]
        else:
            priors, edges = self._scale_problem(entries)
            scales, _, _ = solve_window_scales([e[0] for e in entries], priors, edges, cfg)
        rows = dict(window_i=[], window_j=[], frame_i=[], frame_j=[], value=[], pixels=[])
        frames = []  # (window position, owned row, model frame index)
        for position, (i, m, c, x) in enumerate(entries):
            if i not in scales:
                continue
            valid = c["reference_valid"].astype(bool)
            for row, k in enumerate(x["owned"]):
                if valid[k]:
                    frames.append((position, row, int(k)))
        new = [n for n, f in enumerate(frames) if f[0] == len(entries) - 1]
        if not new:
            return _pack_cross(rows)
        poses = np.repeat(np.eye(4)[None], len(frames), 0)
        for n, (position, _, k) in enumerate(frames):
            c = entries[position][2]
            poses[n, :3, :3], poses[n, :3, 3] = c["reference_rotations"][k], c["reference_centers"][k]
        centers, rotations = poses[:, :3, 3], poses[:, :3, :3]
        tree = cKDTree(centers)
        candidates = set(new)
        for a in new:
            candidates.update(tree.query_ball_point(centers[a], NEIGHBOR_RADIUS_M))
        ranked = {a: ranked_partners(a, centers, rotations, tree) for a in sorted(candidates)}
        new_set = set(new)
        pairs = {(min(a, b), max(a, b)) for a, picks in ranked.items() for b in picks
                 if (a in new_set or b in new_set) and (cfg.frame_scales or frames[a][0] != frames[b][0])}

        loaded_depth = {window_id: cross["depth"]}

        def metric(n):
            position, row, _ = frames[n]
            i, m, _, _ = entries[position]
            if i not in loaded_depth:
                with np.load(depths[i], allow_pickle=False) as a:
                    loaded_depth[i] = a["depth"]
            return loaded_depth[i][row] * scales[i], np.asarray(m["intrinsics"]["K"], float)

        for a, b in sorted(pairs):
            (depth_a, K_a), (depth_b, K_b) = metric(a), metric(b)
            for (i, j, depth_i, K_i, depth_j, K_j) in ((a, b, depth_a, K_a, depth_b, K_b),
                                                       (b, a, depth_b, K_b, depth_a, K_a)):
                value, pixels = cross_view_log_ratio(depth_i, K_i, poses[i], depth_j, K_j, poses[j],
                                                     stride=cfg.cross_window_stride)
                if value is None:
                    continue
                wi, wj = entries[frames[i][0]][0], entries[frames[j][0]][0]
                rows["window_i"].append(wi)
                rows["window_j"].append(wj)
                rows["frame_i"].append(str(entries[frames[i][0]][1]["frame_ids"][frames[i][2]]))
                rows["frame_j"].append(str(entries[frames[j][0]][1]["frame_ids"][frames[j][2]]))
                rows["value"].append(np.log(scales[wi]) - np.log(scales[wj]) - value)
                rows["pixels"].append(pixels)
        return _pack_cross(rows)

    def _model_intrinsics(self, window, meta):
        """Calibrated K at model resolution: profile > snapshot CameraInfo > VGGT median.

        Back-projecting VGGT depth with the calibrated K mattered outdoors
        (flight-out2, 40-frame windows: accuracy median 4.1 vs 7.6 cm).
        """
        h, w = window.depth.shape[1:]
        model_size = tuple(meta.get("model_image_size") or (w, h))
        if model_size != (w, h):
            raise ValueError("model image size differs from prediction depth")
        if self.camera_intrinsics is not None:
            c = self.camera_intrinsics
            K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.]])
            return {"K": resize_intrinsics(K, (c["width"], c["height"]), (w, h)).tolist(), "source": "profile"}
        info = meta.get("camera_info")
        if info:
            K = resize_intrinsics(np.asarray(info["K"]), (info["width"], info["height"]), (w, h))
            return {"K": K.tolist(), "source": "snapshot_camera_info"}
        return {"K": np.median(np.asarray(window.intrinsics), axis=0).tolist(), "source": "vggt_window_median"}

    def _next_revision(self):
        directories = sorted((self.output_root / "revisions").glob("revision-[0-9]*"))
        number = 1 + max([int(p.name.split("-")[1]) for p in directories] or [0])
        final = self.output_root / "revisions" / ("revision-%06d" % number)
        return number, final, Path(tempfile.mkdtemp(prefix=".revision-", dir=final.parent))

    def _commit_revision(self, staging, final, usable):
        _commit_directory(staging, final)
        # Commit the latest computation and the displayed complete map in
        # one pointer. A rejected update must not erase accepted history or
        # mix placements from different solves.
        current = {"path": str(final), "sha256": sha256(final / "revision.json")}
        display = current
        latest = self.output_root / "latest.json"
        if latest.exists():
            previous = _read(latest)
            previous = previous.get("display", previous)
            previous_path = Path(previous["path"])
            if (previous_path.parent != self.output_root / "revisions"
                    or sha256(previous_path / "revision.json") != previous["sha256"]):
                raise ValueError("display revision integrity failed")
            previous_data = _read(previous_path / "revision.json")
            if not set(previous_data["published_windows"]).issubset(usable):
                display = previous
        pointer = self.output_root / ".latest.tmp"
        if pointer.exists():
            pointer.unlink()
        _write(pointer, {**current, "display": display})
        os.replace(pointer, self.output_root / "latest.json")
        _sync_dir(self.output_root)
        return final

    def _publish_revision(self, metadata, signature):
        """Per-frame localization poses with window or per-frame VGGT depth scales."""
        cfg = self.frame_pose_config
        ids = [m["window_id"] for m in metadata]
        cams, entries = {}, []
        ref_by_stamp, frames = {}, set()
        for meta in metadata:
            i = meta["window_id"]
            with np.load(self._window_path(i) / "cameras.npz", allow_pickle=False) as a:
                cam = {k: a[k] for k in a.files}
            if "reference_rotations" not in cam:
                raise ValueError("cached window lacks reference rotations; use a new output directory")
            cams[i] = cam
            cross = None
            if self._measures_rows():
                if not (self._window_path(i) / "cross.npz").exists():
                    raise ValueError("cached window lacks cross-window depth; use a new output directory")
                with np.load(self._window_path(i) / "cross.npz", allow_pickle=False) as a:
                    cross = {k: a[k] for k in a.files if k != "depth"}
            entries.append((i, meta, cam, cross))
            valid = cam["reference_valid"]
            if valid.any():
                frames.add(meta["reference_frame_id"])
            for k in np.flatnonzero(valid):
                stamp = int(cam["timestamps_ns"][k])
                pose = (cam["reference_centers"][k], cam["reference_rotations"][k])
                if stamp in ref_by_stamp and not np.allclose(ref_by_stamp[stamp][0], pose[0], atol=1e-8, rtol=0):
                    raise ValueError("same RGB stamp has conflicting localization")
                ref_by_stamp[stamp] = pose
        if len(frames) > 1:
            raise ValueError("localization reference frame changed across windows")
        priors, edges = self._scale_problem(entries)
        scales, constrained, scale_diag = solve_window_scales(ids, priors, edges, cfg)
        frame_scales = None
        if cfg.frame_scales:
            scales, frame_scales, frame_rows = self._frame_solve(entries)
            for i in ids:
                scale_diag[i]["scale"] = scales.get(i)
                spread = np.log(frame_scales[i] / scales[i]) if i in scales else np.array([])
                spread = spread[np.isfinite(spread)]
                scale_diag[i]["frame_scale_log_deviation_p90"] = (
                    float(np.percentile(np.abs(spread), 90)) if len(spread) else None)
        if self._measures_rows():
            for i, _, _, _ in entries:
                scale_diag[i]["cross_window_rows"] = int(sum(
                    np.sum((c["window_i"] == i) | (c["window_j"] == i)) for _, _, _, c in entries))
        stamps = np.array(sorted(ref_by_stamp), np.int64)
        ref_c = np.array([ref_by_stamp[int(t)][0] for t in stamps]).reshape(-1, 3)
        ref_R = np.array([ref_by_stamp[int(t)][1] for t in stamps]).reshape(-1, 3, 3)
        offset = int(round(cfg.pose_time_offset_ms * 1e6))
        number, final, staging = self._next_revision()
        try:
            windows, usable, centers, rotations, owners, frame_stamps = [], [], [], [], [], []
            for meta in metadata:
                i, cam = meta["window_id"], cams[meta["window_id"]]
                applied = meta.get('capture_pose_time_offset_ms')
                if cfg.pose_time_offset_stage == 'capture':
                    if applied != cfg.pose_time_offset_ms:
                        raise ValueError('Capture pose offset marker differs from session profile')
                    c, R = cam['reference_centers'].copy(), cam['reference_rotations'].copy()
                    ok = cam['reference_valid'].astype(bool).copy()
                else:
                    if applied is not None:
                        raise ValueError('Capture already applied pose offset; refusing double shift')
                    c, R, ok = interpolate_poses(cam["timestamps_ns"] + offset, stamps, ref_c, ref_R,
                                                 int(cfg.max_pose_gap_ms * 1e6))
                # The time offset is a refinement: where no bracket exists (sparse
                # keyframes, session ends) keep the frame's own RGB-time pose.
                own = ~ok & cam["reference_valid"].astype(bool)
                c[own], R[own], ok = cam["reference_centers"][own], cam["reference_rotations"][own], ok | own
                shifted = int((ok & ~own).sum())
                s = scales.get(i)
                prior = priors[i]
                T = None
                if s is not None and ok.any():
                    # Window Sim(3) for frames without a pose and for consumers of
                    # prediction_to_map; frames with a pose use it directly.
                    Rw = prior["rotation"] if prior is not None else rotation_mean(
                        np.einsum("nij,nkj->nik", R[ok], cam["rotations"][ok]))
                    tw = window_translation(cam["centers"], s, Rw, c, ok)
                    T = Sim3(s, Rw, tw)
                is_usable = T is not None
                reason = "localized" if is_usable else (
                    "no_metric_scale" if s is None else "no_localization_pose")
                frame_poses = [{"rotation": R[k].tolist(), "center": c[k].tolist()} if ok[k] else None
                               for k in range(len(ok))]
                entry = {"window_id": i, "component": 0, "global_frame_status": reason,
                         "included_in_global_preview": is_usable, "source": meta, "geometry_accepted": False,
                         "placement": "frame_pose", "scale_diagnostics": scale_diag[i],
                         "frame_pose_valid": int(ok.sum()), "frame_pose_time_shifted": shifted,
                         # Pending windows keep their model-local gauge.
                         "prediction_to_map": meta["prediction_to_local"]}
                if is_usable:
                    usable.append(i)
                    entry.update(local_to_map=T.to_dict(),
                                 prediction_to_map=sim3.compose(T, Sim3.from_dict(meta["prediction_to_local"])).to_dict(),
                                 depth_scale=float(s / meta["gauge"]), intrinsics=meta["intrinsics"]["K"],
                                 intrinsics_source=meta["intrinsics"]["source"], frame_poses=frame_poses)
                    if frame_scales is not None:
                        # Same gauge as depth_scale; None for frames without a scale (shared
                        # prefix owned by the previous window, no pose).
                        entry["frame_depth_scales"] = [
                            float(v / meta["gauge"]) if np.isfinite(v) else None for v in frame_scales[i]]
                    sl = slice(meta["shared_prefix"], None)
                    placed = ok[sl]
                    centers.append(c[sl][placed]); rotations.append(R[sl][placed])
                    frame_stamps.append(cam["timestamps_ns"][sl][placed])
                    owners.append(np.full(int(placed.sum()), i, np.int64))
                windows.append(entry)
            np.savez_compressed(staging / "trajectory.npz",
                centers=np.concatenate(centers) if centers else np.empty((0, 3)),
                rotations=np.concatenate(rotations) if rotations else np.empty((0, 3, 3)),
                timestamps_ns=np.concatenate(frame_stamps) if frame_stamps else np.empty(0, np.int64),
                window_ids=np.concatenate(owners) if owners else np.empty(0, np.int64))
            # "graph", "loop_edges" and "orientation_factors" are part of the revision
            # format that the Web, export and earlier revisions share.
            revision = {"schema": "drone3d.g4.map_revision.v1", "revision": number,
                "method_id": self.method_id, "placement": "frame_pose", "session_id": self.session_id,
                "frame_id": self.frame_id, "reference_frame_id": next(iter(frames), ""),
                "input_signature": signature, "config_digest": self.config_digest,
                "replacement": "complete_map", "window_count": len(ids), "published_windows": usable,
                "pending_windows": [i for i in ids if i not in usable], "windows": windows,
                "graph": {"placement": "frame_pose", "constrained_windows": sorted(constrained),
                          "edges": len(edges), "pose_time_offset_ms": cfg.pose_time_offset_ms},
                "loop_edges": 0, "orientation_factors": 0,
                "formal_geometry_acceptance": False, "baseline_promoted": False,
                "localization_metric_interpretation": "frames placed by localization; not surface ground truth",
                "artifacts": {p.name: sha256(p) for p in staging.iterdir()}}
            _write(staging / "revision.json", revision)
            return self._commit_revision(staging, final, usable)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def update(self):
        """Return a new complete revision, or replay latest once after restart."""
        try:
            records = list(self.scanner(self.records_root, self.session_id))
            records.sort(key=lambda r: r["window_index"])
            ids = [r["window_index"] for r in records]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate formal window index")
            # The formal scanner excludes damaged records. Surface that exclusion.
            session = self.records_root / self.session_id
            disk_ids = {int(p.name[7:]) for p in session.glob("window-*") if re.fullmatch(r"window-[0-9]+", p.name)}
            if disk_ids and disk_ids != set(ids):
                raise ValueError("formal session contains missing/unverified windows")
            cached_ids = {int(p.name[7:]) for p in (self.output_root / "windows").glob("window-*")}
            if not cached_ids.issubset(ids):
                raise ValueError("previously committed source window disappeared")
            if sorted(cached_ids) != ids[:len(cached_ids)]:
                raise ValueError("historical window inserted before cached inputs; use a new output directory")
            if not records:
                self._status.update(state="waiting", window_count=0)
                return None
            metadata = []
            previous = None
            for record in records:
                path = self._window_path(record["window_index"])
                meta = (self._read_window(record["window_index"]) if path.exists()
                        else self._cache_window(record, previous, metadata))
                if meta["input_binding"] != _binding(record):
                    raise ValueError("committed input changed; use a new output directory")
                metadata.append(meta)
                previous = record
            signature = _digest([self.config_digest, [m["input_binding"] for m in metadata]])
            latest = self.output_root / "latest.json"
            output = None
            if latest.exists():
                pointer = _read(latest)
                output = Path(pointer["path"])
                if output.parent != self.output_root / "revisions" or sha256(output / "revision.json") != pointer["sha256"]:
                    raise ValueError("latest revision integrity failed")
                data = _read(output / "revision.json")
                for name, digest in data["artifacts"].items():
                    if sha256(output / name) != digest:
                        raise ValueError("latest revision artifact changed")
                if data["input_signature"] != signature:
                    output = None
            if output is None:
                self._status.update(state="optimizing", window_count=len(ids))
                output = self._publish_revision(metadata, signature)
            data = _read(output / "revision.json")
            self._status.update(state="ready", window_count=len(ids), revision=data["revision"],
                latest_revision_path=str(output), pending_windows=data["pending_windows"],
                published_windows=data["published_windows"], formal_geometry_acceptance=False, error="")
            if self._announced == output:
                return None
            self._announced = output
            return output
        except Exception as error:
            self._status.update(state="error", error=type(error).__name__ + ": " + str(error))
            raise
