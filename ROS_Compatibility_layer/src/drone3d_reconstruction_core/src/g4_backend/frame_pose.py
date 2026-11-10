"""Per-frame external-pose placement: VGGT supplies depth and a window scale only.

Each frame is placed with the localization camera pose at its RGB time. The
window's VGGT depth (local gauge units) is converted to metres by one scale per
window, solved as a log-scale least-squares problem over two constraint types:

* trajectory prior: the window's VGGT camera centres are rotated by the mean
  localization/VGGT orientation difference, then fitted to the localization
  centres for scale. Used only when the localization centres move more than
  ``min_extent_m``; weight = min(extent, ``extent_weight_cap``).
* adjacent edge: shared RGB frames observed by both windows give
  s_j = s_i * median(d_i / d_j) over pixels above both frames' confidence
  medians (``shared_depth_ratio``); weight ``edge_weight``. E1 registration is
  not needed for a scale; it also rejected several indoor seams that the depth
  ratio handled.

The numerics live in ``recon_core.scale`` and are shared with the offline pipeline.

Windows in a component without any trajectory prior have no metric scale and
stay pending. Experiment basis (testflight3 indoor, flight-out2 outdoor):
Data/experiments/接手诊断_20261008/REPORT.md.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from recon_core import scale as window_scale
from recon_core.scale import rotation_mean, trajectory_prior  # noqa: F401  re-exported

FRAME_POSE_METHOD_ID = "frame_pose_window_scale_20261008"


@dataclass(frozen=True)
class FramePoseConfig:
    min_extent_m: float = 0.3
    extent_weight_cap: float = 2.0
    edge_weight: float = 3.0
    # Localization pose is taken at RGB time + offset (camera timestamps lag the
    # odometry clock on the D435i/FAST-LIO rig: +70 ms indoor, +80 ms outdoor).
    pose_time_offset_ms: float = 0.0
    pose_time_offset_stage: str = "backend"
    max_pose_gap_ms: float = 500.0
    # Weight of one cross-window frame-pair row (recon_core.scale.cross_view_log_ratio);
    # 0 keeps the trajectory-prior + adjacent-edge solve. Needs capture-stage poses,
    # the poses frames are placed with.
    cross_window_weight: float = 0.0
    cross_window_stride: int = 4
    # One depth scale per frame (recon_core.scale.solve_frame_scales) from frame-pair rows
    # inside and across windows; the window scale stays the frame scales' common part.
    # Replaces the weighted cross-window rows, so cross_window_weight must be 0.
    frame_scales: bool = False

    def __post_init__(self):
        if self.pose_time_offset_stage not in ("capture", "backend"):
            raise ValueError("pose_time_offset_stage must be capture or backend")
        if not np.isfinite(self.cross_window_weight) or self.cross_window_weight < 0:
            raise ValueError("cross_window_weight must be finite and nonnegative")
        if type(self.frame_scales) is not bool:
            raise ValueError("frame_scales must be a bool")
        if self.frame_scales and self.cross_window_weight > 0:
            raise ValueError("frame_scales replaces cross_window_weight; set it to 0")
        if (self.cross_window_weight > 0 or self.frame_scales) and self.pose_time_offset_stage != "capture":
            raise ValueError("cross_window_weight / frame_scales need pose_time_offset_stage=capture")
        if type(self.cross_window_stride) is not int or not 1 <= self.cross_window_stride <= 16:
            raise ValueError("cross_window_stride must be an integer in 1..16")
        for name in ("min_extent_m", "extent_weight_cap", "edge_weight", "max_pose_gap_ms"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if not np.isfinite(self.pose_time_offset_ms) or abs(self.pose_time_offset_ms) > 1000:
            raise ValueError("pose_time_offset_ms must be finite and within +-1000 ms")


def solve_window_scales(window_ids, priors, edges, config=FramePoseConfig()):
    """``priors``: id -> trajectory_prior result or None; ``edges``: (i, j, r) with s_j = s_i * r.

    Returns (scales: id -> float, constrained: set, diagnostics: id -> dict).
    """
    return window_scale.solve_window_scales(window_ids, priors, edges, config.min_extent_m,
                                            config.extent_weight_cap, config.edge_weight)


def shared_depth_ratio(previous, current, min_pixels=500):
    """Relative scale d_prev / d_cur over the current window's shared prefix (WindowGeometry)."""
    position = {f: k for k, f in enumerate(previous.frame_ids)}
    frames = ((previous.depth[position[f]], previous.confidence[position[f]], current.depth[k], current.confidence[k])
              for k, f in enumerate(current.frame_ids[:current.shared_prefix]) if f in position)
    return window_scale.shared_depth_ratio(frames, min_pixels)


def interpolate_poses(query_ns, ref_ns, centers, rotations, max_gap_ns):
    """Linear centre / slerp rotation at ``query_ns``; no extrapolation, no wide gaps."""
    query_ns = np.asarray(query_ns, np.int64)
    ref_ns = np.asarray(ref_ns, np.int64)
    out_c = np.full((len(query_ns), 3), np.nan)
    out_R = np.full((len(query_ns), 3, 3), np.nan)
    valid = np.zeros(len(query_ns), bool)
    if len(ref_ns) < 2:
        exact = np.isin(query_ns, ref_ns)
        for k in np.flatnonzero(exact):
            m = int(np.flatnonzero(ref_ns == query_ns[k])[0])
            out_c[k], out_R[k], valid[k] = centers[m], rotations[m], True
        return out_c, out_R, valid
    order = np.argsort(ref_ns)
    t, c, R = ref_ns[order], np.asarray(centers)[order], np.asarray(rotations)[order]
    hi = np.searchsorted(t, query_ns, side="left")
    for k, q in enumerate(query_ns):
        h = hi[k]
        if h < len(t) and t[h] == q:
            out_c[k], out_R[k], valid[k] = c[h], R[h], True
            continue
        if h == 0 or h == len(t) or t[h] - t[h - 1] > max_gap_ns:
            continue
        lo = h - 1
        f = (q - t[lo]) / float(t[h] - t[lo])
        out_c[k] = c[lo] + f * (c[h] - c[lo])
        out_R[k] = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([R[lo], R[h]])))([f]).as_matrix()[0]
        valid[k] = True
    return out_c, out_R, valid


def window_translation(local_centers, scale, rotation, ref_centers, valid):
    """Median translation completing the window Sim(3) local -> map."""
    valid = np.asarray(valid, bool)
    if not valid.any():
        return None
    return np.median(np.asarray(ref_centers)[valid] - scale * np.asarray(local_centers)[valid] @ rotation.T, axis=0)
