"""Per-frame placement backend for VGGT-Omega windows; no ROS, inference or experiment I/O."""
from .types import Sim3, WindowGeometry
from .geometry import from_prediction, resize_intrinsics
from .frame_pose import (FRAME_POSE_METHOD_ID, FramePoseConfig, interpolate_poses, rotation_mean,
                         shared_depth_ratio, solve_window_scales, trajectory_prior, window_translation)
from . import sim3

__all__ = ["Sim3", "WindowGeometry", "from_prediction", "resize_intrinsics",
           "FRAME_POSE_METHOD_ID", "FramePoseConfig", "interpolate_poses", "rotation_mean", "shared_depth_ratio",
           "solve_window_scales", "trajectory_prior", "window_translation",
           "sim3"]
