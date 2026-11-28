"""Point-cloud building for the dashboard: select a session's placed windows and merge them.

Split from ``continuous_web``; :class:`WebDashboardCore` inherits :class:`PointCloudMixin`.
"""

from __future__ import annotations

import json
import math
from typing import Optional

from .odom_local import (
    COORDINATE_MODEL_LOCAL_UNALIGNED,
)
from .odom_local import valid_sim3_alignment
from .continuous_web_common import (
    _ALIGNED_STATES,
    WebDashboardError,
    _formal_records_for,
    _image_to_rgb_u8,
    _latest_window_record,
    _manifest_path_from_uri,
    _parse_session_id,
    _predictions_path_from_manifest,
    _record_alignment,
    _record_coordinate_state,
    _record_quality_passed,
)


def _aligned_transform(record):
    """从 provenance alignment 提取 (scale, R, t)；不可用返回 None。"""
    alignment = _record_alignment(record)
    if not valid_sim3_alignment(alignment):
        return None
    import numpy as np

    try:
        scale = float(alignment["scale"])
        rotation_matrix = np.asarray(alignment["rotation"], dtype=np.float64).reshape(3, 3)
        translation_vector = np.asarray(alignment["translation"], dtype=np.float64).reshape(3)
    except (TypeError, ValueError, OverflowError) as error:
        raise WebDashboardError("alignment transform is not finite") from error
    if not np.isfinite(rotation_matrix).all() or not np.isfinite(translation_vector).all():
        return None
    return scale, rotation_matrix, translation_vector


def _frame_placement(alignment, merge_indices, frame_count):
    """Validated per-frame localization placement of one G4 ``frame_pose`` window.

    Returns (depth_scales[F], K, centers[F,3], rotations[F,3,3], has_pose[F]) for the
    displayed (merged) frames, in the same order as the sliced prediction arrays. A
    frame's scale is its per-frame G4 scale when the revision has one, else the window's.
    """
    import numpy as np

    poses = alignment.get("frame_poses")
    if not isinstance(poses, list) or len(poses) != frame_count:
        raise WebDashboardError("frame_poses must list every model frame")
    try:
        scale = float(alignment["depth_scale"])
        K = np.asarray(alignment["intrinsics"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError) as error:
        raise WebDashboardError("frame_pose depth_scale/intrinsics are invalid") from error
    if not math.isfinite(scale) or scale <= 0 or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise WebDashboardError("frame_pose depth_scale/intrinsics are invalid")
    order = merge_indices if merge_indices is not None else list(range(frame_count))
    per_frame = alignment.get("frame_depth_scales")
    if per_frame is not None and (not isinstance(per_frame, list) or len(per_frame) != frame_count):
        raise WebDashboardError("frame_depth_scales must list every model frame")
    scales = np.full(len(order), scale)
    if per_frame is not None:
        for k, index in enumerate(order):
            if per_frame[index] is not None:
                value = float(per_frame[index])
                if not math.isfinite(value) or value <= 0:
                    raise WebDashboardError("frame_depth_scales entry is invalid")
                scales[k] = value
    centers = np.full((len(order), 3), np.nan)
    rotations = np.full((len(order), 3, 3), np.nan)
    has_pose = np.zeros(len(order), dtype=bool)
    for k, index in enumerate(order):
        pose = poses[index]
        if pose is None:
            continue
        try:
            rotation = np.asarray(pose["rotation"], dtype=np.float64).reshape(3, 3)
            center = np.asarray(pose["center"], dtype=np.float64).reshape(3)
        except (KeyError, TypeError, ValueError) as error:
            raise WebDashboardError("frame_poses entry is invalid") from error
        if not np.isfinite(rotation).all() or not np.isfinite(center).all():
            raise WebDashboardError("frame_poses entry is not finite")
        centers[k], rotations[k], has_pose[k] = center, rotation, True
    return scales, K, centers, rotations, has_pose


def _frame_pose_points(placement, depth, flat_indices):
    """Back-project sampled pixels with calibrated K and each frame's own pose."""
    import numpy as np

    scales, K, centers, rotations, has_pose = placement
    frames, pixels = np.divmod(flat_indices, depth.shape[1] * depth.shape[2])
    v, u = np.divmod(pixels, depth.shape[2])
    d = depth.reshape(-1)[flat_indices].astype(np.float64) * scales[frames]
    rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(d)), axis=1) * d[:, None]
    points = np.einsum("nij,nj->ni", np.nan_to_num(rotations[frames]), rays) + np.nan_to_num(centers[frames])
    points[~has_pose[frames]] = np.nan
    return points.astype(np.float32)


def _latest_map_id(records, include_rejected: bool = False) -> Optional[str]:
    candidates = []
    for record in records:
        if _record_coordinate_state(record) in _ALIGNED_STATES:
            if not include_rejected and not _record_quality_passed(record):
                continue
            map_id = _record_alignment(record).get("map_id")
            if isinstance(map_id, str) and map_id:
                candidates.append(map_id)
    if not candidates:
        return None
    return candidates[-1]


def _session_has_aligned_windows(records, include_rejected: bool = False) -> bool:
    return any(
        _record_coordinate_state(record) in _ALIGNED_STATES
        and (include_rejected or _record_quality_passed(record))
        and _aligned_transform(record) is not None
        for record in records
    )


class PointCloudMixin:
    """Point-cloud methods of WebDashboardCore (they use its config, lock and caches)."""

    # ------------------------------------------------------------------
    # 点云累计
    # ------------------------------------------------------------------
    def _select_point_cloud_records(
        self, session_id, map_id, max_points, include_rejected=False
    ):
        """Choose displayable records and compute the cache fingerprint."""
        records = _formal_records_for(session_id, self._config.formal_records_root)
        if not records:
            raise WebDashboardError("unknown session", 404)
        if not isinstance(map_id, str) or not map_id:
            raise WebDashboardError("map must be a nonempty string", 400)
        aligned_records = []
        for record in records:
            if _record_coordinate_state(record) not in _ALIGNED_STATES:
                continue
            if _record_alignment(record).get("map_id") != map_id:
                continue
            if not include_rejected and not _record_quality_passed(record):
                continue
            if _aligned_transform(record) is None:
                continue
            aligned_records.append(record)

        if aligned_records:
            selected_records = aligned_records
            aligned = True
            reason = _record_coordinate_state(aligned_records[-1])
        else:
            if _session_has_aligned_windows(records, include_rejected):
                raise WebDashboardError("unknown map for this session", 404)
            if any(_record_alignment(r).get("method") == "global_g4" for r in records):
                raise WebDashboardError("G4 processing: no global map revision yet", 409)
            latest = _latest_window_record(records)
            if latest is None or _record_coordinate_state(latest) != COORDINATE_MODEL_LOCAL_UNALIGNED:
                raise WebDashboardError("no displayable window for this session", 404)
            selected_records = [latest]
            aligned = False
            reason = _record_alignment(latest).get("reason", "") or "MODEL_LOCAL_UNALIGNED"

        fingerprint = tuple(
            (int(record.get("window_index", -1)), str(record.get("result_manifest_sha256", "")),
             str(_record_alignment(record).get("map_revision_sha256", "")))
            for record in selected_records
        )
        key = (
            session_id,
            map_id,
            int(max_points),
            bool(include_rejected),
            fingerprint,
        )
        return key, selected_records, aligned, reason

    def _point_cloud_cache_lookup(
        self, session_id, map_id, max_points, include_rejected=False
    ):
        """Fast cache path: no build lock, only the short cache lock."""
        key, selected, aligned, _reason = self._select_point_cloud_records(
            session_id, map_id, max_points, include_rejected
        )
        with self._pc_lock:
            cached = self._pc_cache.get(key)
            if cached is not None:
                if key in self._pc_order:
                    self._pc_order.remove(key)
                self._pc_order.append(key)
                result = dict(cached)
                result["_cached"] = True
                return result
        return None

    def build_point_cloud(
        self,
        session_id: str,
        map_id: str,
        max_points: int,
        include_rejected: bool = False,
    ) -> dict:
        session_id = _parse_session_id(session_id)
        cached = self._point_cloud_cache_lookup(
            session_id, map_id, max_points, include_rejected
        )
        if cached is not None:
            return cached
        with self._pc_build_lock:
            # double-check under the build lock; another request may have
            # populated the cache between the fast path and lock acquisition
            cached = self._point_cloud_cache_lookup(
                session_id, map_id, max_points, include_rejected
            )
            if cached is not None:
                return cached
            return self._build_point_cloud(
                session_id, map_id, max_points, include_rejected
            )

    def _build_point_cloud(
        self, session_id, map_id, max_points, include_rejected=False
    ) -> dict:
        import numpy as np

        key, selected_records, aligned, reason = self._select_point_cloud_records(
            session_id, map_id, max_points, include_rejected
        )
        fingerprint = key[4]
        incremental = None
        with self._pc_lock:
            cached = self._pc_cache.get(key)
            if cached is not None:
                if key in self._pc_order:
                    self._pc_order.remove(key)
                self._pc_order.append(key)
                result = dict(cached)
                result["_cached"] = True
                return result
            if aligned:
                for candidate_key in reversed(self._pc_order):
                    candidate = self._pc_cache.get(candidate_key)
                    candidate_fingerprint = (
                        candidate.get("_selected_fingerprint", ())
                        if candidate is not None
                        else ()
                    )
                    if (
                        candidate is not None
                        and candidate_key[:4] == key[:4]
                        and candidate.get("aligned") is True
                        and candidate_fingerprint
                        and len(candidate_fingerprint) < len(fingerprint)
                        and fingerprint[: len(candidate_fingerprint)]
                        == candidate_fingerprint
                    ):
                        incremental = candidate
                        break

        fusion_opt_in = False
        if self._config.frame_pose_fusion:
            from .g4.backend_process import output_root
            try:
                stored = json.loads((output_root(self._config.formal_records_root, session_id) / "backend_profile.json").read_text())
                fusion_opt_in = stored.get("frame_pose_fusion") is True
            except FileNotFoundError:
                pass  # Old sessions keep their original display algorithm.
        if (fusion_opt_in and aligned and selected_records
                and all(_record_alignment(r).get("placement") == "frame_pose" for r in selected_records)):
            from drone3d_onboard.session import SessionFusion
            from .g4.backend_process import output_root
            fuser = self._session_fusion.get(session_id)
            if fuser is None:
                fuser = SessionFusion(output_root(self._config.formal_records_root, session_id)
                                      / "fusion_cache" / "incremental")
                self._session_fusion = {session_id: fuser}  # keep only the session being viewed
            xyz, rgb, fusion = fuser.fuse(selected_records, max_points)
            result = dict(session_id=session_id, map_id=map_id, aligned=True,
                coordinate_state=_record_coordinate_state(selected_records[-1]), reason=reason,
                original_points=fusion['points_2cm'], displayed_points=len(xyz), equivalent_stride=1,
                sampling="multiview_adaptive_voxel", conf_percentile=50.,
                include_rejected=bool(include_rejected), windows=[r['window_index'] for r in selected_records],
                xyz=xyz.astype(np.float32), rgb=np.rint(rgb*255).clip(0,255).astype(np.uint8),
                fusion=fusion, _selected_fingerprint=fingerprint, _cached=False)
            with self._pc_lock:
                self._pc_cache[key] = result
                self._pc_order.append(key)
                while len(self._pc_order) > 2:
                    self._pc_cache.pop(self._pc_order.pop(0), None)
            return result

        if incremental is None:
            selected_to_load = selected_records
            total_valid = 0
            windows = []
            xyz = np.empty((0, 3), dtype=np.float32)
            rgb = np.empty((0, 3), dtype=np.uint8)
            priorities = np.empty(0, dtype=np.float64)
            rng = np.random.default_rng(0)
        else:
            selected_to_load = selected_records[
                len(incremental["_selected_fingerprint"]) :
            ]
            total_valid = int(incremental["original_points"])
            windows = list(incremental["windows"])
            xyz = incremental["xyz"].copy()
            rgb = incremental["rgb"].copy()
            priorities = incremental["_priorities"].copy()
            rng = np.random.default_rng()
            rng.bit_generator.state = incremental["_rng_state"]

        conf_percentile = float(self._config.conf_percentile)
        # ponytail: compressed NPZ is the atomic input, so peak RAM is one
        # window plus the bounded reservoir; chunk the artifact only if one
        # window itself stops fitting.
        for record in selected_to_load:
            manifest_path = _manifest_path_from_uri(record.get("result_manifest_uri"))
            predictions_path = _predictions_path_from_manifest(
                manifest_path, self._config.max_predictions_bytes
            )
            with np.load(str(predictions_path), allow_pickle=False) as data:
                if "world_points_from_depth" not in data:
                    raise WebDashboardError("predictions.npz is missing world_points_from_depth")
                if "depth_conf" not in data:
                    raise WebDashboardError("predictions.npz is missing depth_conf")
                if "images" not in data:
                    raise WebDashboardError("predictions.npz is missing images")
                world = np.asarray(data["world_points_from_depth"])
                conf = np.asarray(data["depth_conf"])
                images = np.asarray(data["images"])
                is_g4 = _record_alignment(record).get("method") == "global_g4"
                depth = np.asarray(data["depth"])[..., 0] if is_g4 else None
                g4_mask = None
                if is_g4:
                    from scipy.ndimage import maximum_filter, minimum_filter
                    # G4 computes its percentile on the complete model window,
                    # including the shared prefix, before excluding duplicates.
                    d = depth / float(_record_alignment(record)["gauge"])
                    span = maximum_filter(d, size=(1, 3, 3), mode="nearest") - minimum_filter(d, size=(1, 3, 3), mode="nearest")
                    g4_mask = ((conf >= np.percentile(conf, 50)) & np.isfinite(d)
                               & (d > 0) & (span / np.maximum(d, 1e-12) <= .03))
                if world.ndim != 4 or world.shape[3] != 3:
                    raise WebDashboardError("world_points_from_depth must be NxHxWx3")
                if conf.ndim != 3 or conf.shape != world.shape[:3]:
                    raise WebDashboardError("depth_conf shape must match world points")
                if images.ndim not in (3, 4):
                    raise WebDashboardError("images must be Nx3xHxW or NxHxWx3")
                height, width = world.shape[1], world.shape[2]
                if images.ndim == 4:
                    if images.shape[1] == 3 and images.shape[2] == height and images.shape[3] == width:
                        images = np.transpose(images, (0, 2, 3, 1))
                    elif images.shape[3] == 3 and images.shape[1] == height and images.shape[2] == width:
                        pass
                    else:
                        raise WebDashboardError(
                            "images layout must match world_points_from_depth HxW"
                        )
                else:
                    if images.shape[0] == 3 and images.shape[1] == height and images.shape[2] == width:
                        images = np.transpose(images, (1, 2, 0))[np.newaxis, ...]
                    elif images.shape[2] == 3 and images.shape[0] == height and images.shape[1] == width:
                        images = images[np.newaxis, ...]
                    else:
                        raise WebDashboardError(
                            "images layout must match world_points_from_depth HxW"
                        )
                if images.shape[0] != world.shape[0]:
                    raise WebDashboardError("images frame count must match world points")

                model_frame_count = int(world.shape[0])
                merge_indices = _record_alignment(record).get("merge_frame_indices")
                if merge_indices is not None:
                    if (
                        not isinstance(merge_indices, list)
                        or any(
                            isinstance(index, bool)
                            or not isinstance(index, int)
                            or index < 0
                            or index >= world.shape[0]
                            for index in merge_indices
                        )
                        or len(merge_indices) != len(set(merge_indices))
                    ):
                        raise WebDashboardError("merge_frame_indices is invalid")
                    world = world[merge_indices]
                    conf = conf[merge_indices]
                    images = images[merge_indices]
                    if depth is not None:
                        depth = depth[merge_indices]
                    if g4_mask is not None:
                        g4_mask = g4_mask[merge_indices]

                frame_placement = None
                if aligned and is_g4 and _record_alignment(record).get("placement") == "frame_pose":
                    frame_placement = _frame_placement(
                        _record_alignment(record), merge_indices, model_frame_count)

                flat_world = world.reshape(-1, 3)
                flat_conf = conf.reshape(-1)
                if flat_world.shape[0] != flat_conf.shape[0]:
                    raise WebDashboardError("world/conf size mismatch")

                finite_world = np.isfinite(flat_world).all(axis=1)
                if depth is not None:
                    flat_depth = depth.reshape(-1)
                    finite_world &= np.isfinite(flat_depth) & (flat_depth > 0)
                finite_conf = np.isfinite(flat_conf)
                conf_values = flat_conf[finite_world & finite_conf]
                if conf_values.size == 0:
                    threshold = float("inf")
                else:
                    threshold = float(
                        np.percentile(conf_values, conf_percentile)
                    )
                valid_mask = finite_world & finite_conf & (flat_conf >= threshold)
                if g4_mask is not None:
                    conf_percentile = 50.
                    valid_mask = finite_world & finite_conf & g4_mask.reshape(-1)
                valid_indices = np.flatnonzero(valid_mask)
                valid_count = int(valid_indices.size)
                if valid_count == 0:
                    continue
                total_valid += valid_count
                new_priorities = rng.random(valid_count)
                if valid_count > max_points:
                    keep = np.argpartition(new_priorities, max_points - 1)[:max_points]
                    keep = keep[
                        np.argsort(new_priorities[keep], kind="stable")
                    ]
                    valid_indices = valid_indices[keep]
                    new_priorities = new_priorities[keep]

                # Sample/gather before Sim(3) and color conversion.  The Web
                # preview is bounded, so transforming every source point only
                # creates large float64/float32 temporaries with no visible
                # benefit.
                new_xyz = np.asarray(flat_world[valid_indices], dtype=np.float32)
                if frame_placement is not None:
                    # Frames without a localization pose become NaN and are
                    # dropped by the finite filter below.
                    new_xyz = _frame_pose_points(frame_placement, depth, valid_indices)
                elif aligned:
                    scale, rotation, translation = _aligned_transform(record)
                    with np.errstate(over="ignore", invalid="ignore"):
                        new_xyz = (
                            np.float32(scale)
                            * (new_xyz @ np.asarray(rotation, dtype=np.float32).T)
                            + np.asarray(translation, dtype=np.float32)
                        )
                finite_preview = np.isfinite(new_xyz).all(axis=1)
                if not finite_preview.all():
                    valid_indices = valid_indices[finite_preview]
                    new_priorities = new_priorities[finite_preview]
                    new_xyz = new_xyz[finite_preview]
                if new_xyz.shape[0] == 0:
                    continue
                raw_rgb = images.reshape(-1, 3)[valid_indices]
                new_rgb = (
                    np.rint(raw_rgb * 255).clip(0, 255).astype(np.uint8)
                    if is_g4 else _image_to_rgb_u8(raw_rgb)
                )
                windows.append(int(record.get("window_index", 0)))
                priorities = np.concatenate((priorities, new_priorities))
                xyz = np.concatenate((xyz, new_xyz), axis=0)
                rgb = np.concatenate((rgb, new_rgb), axis=0)
                if priorities.size > max_points:
                    keep = np.argpartition(priorities, max_points - 1)[:max_points]
                    keep = keep[np.argsort(priorities[keep], kind="stable")]
                    priorities = priorities[keep].copy()
                    xyz = xyz[keep].copy()
                    rgb = rgb[keep].copy()

        if total_valid == 0 or not windows or xyz.shape[0] == 0:
            raise WebDashboardError("no points pass the confidence filter", 404)

        equivalent_stride = max(1, int(math.ceil(total_valid / float(max_points))))
        displayed = int(xyz.shape[0])

        result = {
            "session_id": session_id,
            "map_id": map_id if aligned else "",
            "aligned": aligned,
            "coordinate_state": (
                _record_coordinate_state(selected_records[-1])
                if aligned
                else COORDINATE_MODEL_LOCAL_UNALIGNED
            ),
            "reason": reason,
            "original_points": total_valid,
            "displayed_points": int(displayed),
            "equivalent_stride": equivalent_stride,
            "sampling": "uniform_reservoir" if total_valid > max_points else "none",
            "conf_percentile": conf_percentile,
            "include_rejected": bool(include_rejected),
            "windows": windows,
            "xyz": xyz,
            "rgb": rgb,
            "_priorities": priorities,
            "_rng_state": rng.bit_generator.state,
            "_selected_fingerprint": fingerprint,
            "_cached": False,
        }
        with self._pc_lock:
            self._pc_cache[key] = dict(result)
            if key in self._pc_order:
                self._pc_order.remove(key)
            self._pc_order.append(key)
            while len(self._pc_order) > 2:
                expired = self._pc_order.pop(0)
                self._pc_cache.pop(expired, None)
        return result

    def point_cloud_binary(self, result: dict) -> bytes:
        import struct

        xyz = result["xyz"]
        rgb = result["rgb"]
        count = int(xyz.shape[0])
        header = struct.pack("<I", count)
        return header + xyz.astype("<f4").tobytes() + rgb.astype("u1").tobytes()

    def point_cloud_meta(self, result: dict) -> dict:
        return {
            "session_id": result["session_id"],
            "map_id": result["map_id"],
            "aligned": result["aligned"],
            "coordinate_state": (
                result.get("coordinate_state", COORDINATE_MODEL_LOCAL_UNALIGNED)
            ),
            "reason": result["reason"],
            "original_points": result["original_points"],
            "displayed_points": result["displayed_points"],
            "sampling": result["sampling"],
            "equivalent_stride": result["equivalent_stride"],
            "conf_percentile": result["conf_percentile"],
            "include_rejected": bool(result.get("include_rejected", False)),
            "windows": result["windows"],
            "format": "uint32 count + float32 xyz[count*3] + uint8 rgb[count*3]",
        }
