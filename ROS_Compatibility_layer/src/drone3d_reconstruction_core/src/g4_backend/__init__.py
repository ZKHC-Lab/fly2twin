"""Independent VGGT-Omega backend; no ROS, inference or experiment I/O."""
from .types import (METHOD_ID, WINDOWED_METHOD_ID, THRESHOLD_RECOMPUTED, THRESHOLD_RECORDED,
                    Sim3, WindowGeometry, RelativeEdge, LocalizationAnchor, ThresholdPolicy,
                    RegistrationConfig, GraphConfig, LocalizationConfig, SharedPairs,
                    RegistrationResult, GraphResult, LocalizationSelection, edge_key, canonical_identifier)
from .geometry import from_prediction, unproject_pixels, resize_intrinsics
from .registration import build_shared_pairs, estimate_adjacent, point_masks
from .localization import select_localization_anchors
from .graph import build_initial_states, solve_g4, graph_residuals
from .window_plan import (SEMANTICS, SEMANTICS_NEW, SEMANTICS_TOTAL, WindowPlan, WindowPlanError,
                          WindowSpec, compare_plans, load_plan, plan_from_counts, plan_from_records)
from .frame_pose import (FRAME_POSE_METHOD_ID, FramePoseConfig, interpolate_poses, rotation_mean,
                         shared_depth_ratio, solve_window_scales, trajectory_prior, window_translation)
from . import sim3

__all__ = ["METHOD_ID", "WINDOWED_METHOD_ID", "THRESHOLD_RECOMPUTED", "THRESHOLD_RECORDED",
           "Sim3", "WindowGeometry", "RelativeEdge", "LocalizationAnchor", "ThresholdPolicy",
           "RegistrationConfig", "GraphConfig", "LocalizationConfig", "SharedPairs",
           "RegistrationResult", "GraphResult", "LocalizationSelection", "edge_key",
           "canonical_identifier",
           "from_prediction", "unproject_pixels", "resize_intrinsics", "build_shared_pairs",
           "estimate_adjacent", "point_masks", "select_localization_anchors",
           "build_initial_states", "solve_g4", "graph_residuals",
           "SEMANTICS", "SEMANTICS_NEW", "SEMANTICS_TOTAL", "WindowPlan", "WindowPlanError",
           "WindowSpec", "compare_plans", "load_plan", "plan_from_counts", "plan_from_records",
           "FRAME_POSE_METHOD_ID", "FramePoseConfig", "interpolate_poses", "rotation_mean", "shared_depth_ratio",
           "solve_window_scales", "trajectory_prior", "window_translation",
           "sim3"]
