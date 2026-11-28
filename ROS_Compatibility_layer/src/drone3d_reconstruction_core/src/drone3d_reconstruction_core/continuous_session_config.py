"""Configuration of the continuous session: ContinuousConfig, the per-window EffectiveConfig and
the value constants they accept.

Moved out of ``continuous_session``, which still re-exports every name.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .capture import CaptureError, validate_capture_goal
from .g4.backend_process import BACKEND_G4

from .action_types import (
    CameraRole,
)


QUEUE_LOSSLESS = "lossless_fifo"


RECONSTRUCTION_MODE_PER_WINDOW = "per_window"


RECONSTRUCTION_MODES = (RECONSTRUCTION_MODE_PER_WINDOW,)


class ContinuousConfigError(ValueError):
    """连续调度器配置非法。"""


@dataclass(frozen=True)
class ContinuousConfig:
    """连续 session 调度器的显式配置（不含运行时热调值）。"""

    roles: tuple[CameraRole, ...]
    reconstruct_action: str
    journal_root: str
    result_root: str
    service_name: str
    diagnostics_topic: str
    diagnostics_period_sec: float
    min_free_disk_gb: float
    action_connect_timeout_sec: float
    capture_wait_margin_sec: float
    reconstruct_wait_timeout_sec: float
    stop_timeout_sec: float
    reconstruct_frame_group_size: int
    initial_sample_fps: float
    initial_window_frames: int
    initial_queue_mode: str
    initial_queue_capacity: int
    initial_max_retries: int
    initial_keep_snapshot_on_success: bool
    initial_overlap_frames: int = 0
    # 窗口放置方式：只有 frame_pose（每帧用外部定位位姿，VGGT 只提供深度与尺度）。
    g4_placement: str = "frame_pose"
    # frame_pose 下位姿取在 RGB 时间 + 偏移（毫秒）。
    pose_time_offset_ms: float = 0.0
    pose_time_offset_stage: str = "backend"
    frame_pose_fusion: bool = False
    # frame_pose 窗口尺度：跨窗口重叠帧深度比的每行权重；0 = 只用轨迹先验与相邻共享帧。
    g4_cross_window_weight: float = 0.0
    # frame_pose：每帧一个深度尺度（窗内与跨窗口帧对联合求解）；与 g4_cross_window_weight 二选一。
    g4_frame_scales: bool = False
    # 源图像分辨率下的标定内参 {width, height, fx, fy, cx, cy}；为空时用快照 CameraInfo。
    camera_intrinsics: dict | None = None
    # 自动清理：保留最新的这么多个会话；更早的只删已导出完整交付包的（storage_retention）。
    retention_max_sessions: int = 3
    # 没有正式记录引用的 worker 结果目录、残留暂存目录，超过这么久才删。
    retention_orphan_grace_sec: float = 86400.0
    # 0 = inherit reconstruct profile (do not override the worker command)
    initial_inference_resolution: int = 0
    # 0.0 = inherit reconstruct profile worker timeout
    initial_worker_timeout_sec: float = 0.0
    # 重建模式：只有 per_window（逐窗增量）。只在 start_session 时冻结。
    initial_reconstruction_mode: str = RECONSTRUCTION_MODE_PER_WINDOW
    # 正式结果真实性门禁期望值。production 节点必须从 reconstruct profile
    # 注入（expected_reconstruction），core 允许为空以支撑注入假门禁的
    # 非正式单元测试；真实门禁在期望缺失时必然拒绝成功终态。
    expected_checkpoint_sha256: str = ""
    expected_checkpoint_path: str = ""
    expected_result_root: str = ""
    expected_algorithm_revision: str = ""
    expected_ros_revision: str = ""
    worker_python: str = ""
    cuda_visible_devices: str = ""
    alignment_backend: str = BACKEND_G4
    reconstruction_backend: str = "omega"

    @classmethod
    def from_params(cls, params: Mapping[str, Any], **values: Any) -> "ContinuousConfig":
        """Build from a flat parameter mapping (a ROS private namespace or a loaded YAML).

        Keys that are not fields are ignored; ``values`` supplies the fields the caller
        resolves itself (roles, paths, the expected reconstruction). A missing field
        without a default raises. Integers given for float fields become floats, so
        ``5`` and ``5.0`` freeze the same effective configuration. A ROS parameter
        cannot hold None: an empty ``camera_intrinsics`` ("" or {}) means none.
        """
        if not isinstance(params, Mapping):
            raise ContinuousConfigError("parameters must be a mapping")
        kwargs: dict[str, Any] = {}
        for item in fields(cls):
            if item.name in values:
                value = values[item.name]
            elif item.name in params:
                value = params[item.name]
            elif item.default is MISSING and item.default_factory is MISSING:
                raise ContinuousConfigError(f"missing required parameter: {item.name}")
            else:
                continue
            if item.type == "float" and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            kwargs[item.name] = value
        intrinsics = kwargs.get("camera_intrinsics")
        if intrinsics in ("", {}):
            kwargs["camera_intrinsics"] = None
        elif isinstance(intrinsics, Mapping):
            kwargs["camera_intrinsics"] = {
                str(key): float(value) if isinstance(value, int) and not isinstance(value, bool) else value
                for key, value in intrinsics.items()
            }
        return cls(**kwargs)

    def __post_init__(self) -> None:
        try:
            self.preflight()
        except ContinuousConfigError:
            raise
        except Exception as error:
            raise ContinuousConfigError(
                f"continuous config preflight failed: {type(error).__name__}: {error}"
            ) from error

    def preflight(self) -> None:
        from .backend_spec import validate_independent_request
        validate_independent_request(self.reconstruction_backend, self.initial_window_frames,
                                     self.initial_inference_resolution)
        if self.alignment_backend != BACKEND_G4:
            raise ContinuousConfigError(
                f"alignment_backend must be {BACKEND_G4!r}, got {self.alignment_backend!r}"
            )
        if self.reconstruction_backend != "omega":
            raise ContinuousConfigError("G4 requires the Omega reconstruction backend")
        if (self.initial_overlap_frames < 5 or self.initial_window_frames <= self.initial_overlap_frames
                or self.initial_inference_resolution <= 0 or self.initial_inference_resolution % 16
                or self.initial_queue_mode != QUEUE_LOSSLESS
                or not self.initial_keep_snapshot_on_success):
            raise ContinuousConfigError("G4 requires >=5 shared frames, a larger total window, a 16-multiple resolution "
                                        "(384/448/512), lossless_fifo and retained snapshots")
        for role in self.roles:
            if role.capture_config.odometry_association != "interpolate":
                raise ContinuousConfigError("G4 requires RGB-time interpolated localization")
        if not isinstance(self.roles, tuple) or not self.roles:
            raise ContinuousConfigError("roles must be a nonempty tuple")
        seen = set()
        for role in self.roles:
            if not isinstance(role, CameraRole):
                raise ContinuousConfigError("roles entries must be CameraRole")
            if role.role in seen:
                raise ContinuousConfigError(f"duplicate camera role: {role.role}")
            seen.add(role.role)
        for name in ("reconstruct_action", "service_name", "diagnostics_topic"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.startswith("/") or " " in value:
                raise ContinuousConfigError(f"{name} must be an absolute ROS name")
        if (
            isinstance(self.diagnostics_period_sec, bool)
            or not isinstance(self.diagnostics_period_sec, (int, float))
            or not _finite(self.diagnostics_period_sec)
            or self.diagnostics_period_sec <= 0
        ):
            raise ContinuousConfigError(
                "diagnostics_period_sec must be positive and finite"
            )
        result_root = Path(self.result_root)
        if not result_root.is_absolute() or not result_root.is_dir():
            raise ContinuousConfigError("result_root must be an existing directory")
        if (
            isinstance(self.reconstruct_frame_group_size, bool)
            or not isinstance(self.reconstruct_frame_group_size, int)
            or self.reconstruct_frame_group_size <= 0
        ):
            raise ContinuousConfigError(
                "reconstruct_frame_group_size must be a positive integer"
            )
        journal = Path(self.journal_root)
        if not journal.is_absolute() or not journal.is_dir():
            raise ContinuousConfigError("journal_root must be an existing directory")
        if journal != journal.resolve(strict=True) or journal.is_symlink():
            raise ContinuousConfigError("journal_root must be canonical and real")
        for name in (
            "min_free_disk_gb",
            "action_connect_timeout_sec",
            "capture_wait_margin_sec",
            "reconstruct_wait_timeout_sec",
            "stop_timeout_sec",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not _finite(value)
                or float(value) <= 0
            ):
                raise ContinuousConfigError(f"{name} must be positive and finite")
        if (
            isinstance(self.initial_sample_fps, bool)
            or not isinstance(self.initial_sample_fps, (int, float))
            or not _finite(self.initial_sample_fps)
            or self.initial_sample_fps <= 0
        ):
            raise ContinuousConfigError("initial_sample_fps must be positive and finite")
        _validate_window_frames(
            self.initial_window_frames, self.reconstruct_frame_group_size
        )
        _validate_overlap_frames(
            self.initial_overlap_frames, self.initial_window_frames
        )
        if type(self.frame_pose_fusion) is not bool:
            raise ContinuousConfigError("frame_pose_fusion must be a bool")
        if self.pose_time_offset_stage not in ("capture", "backend"):
            raise ContinuousConfigError("pose_time_offset_stage must be capture or backend")
        if self.g4_placement != "frame_pose":
            raise ContinuousConfigError(
                f"g4_placement must be 'frame_pose', got {self.g4_placement!r} (graph placement was removed)"
            )
        if (
            isinstance(self.pose_time_offset_ms, bool)
            or not isinstance(self.pose_time_offset_ms, (int, float))
            or not math.isfinite(self.pose_time_offset_ms)
            or abs(self.pose_time_offset_ms) > 1000
        ):
            raise ContinuousConfigError("pose_time_offset_ms must be finite and within +-1000")
        if (
            isinstance(self.g4_cross_window_weight, bool)
            or not isinstance(self.g4_cross_window_weight, (int, float))
            or not math.isfinite(self.g4_cross_window_weight)
            or self.g4_cross_window_weight < 0
        ):
            raise ContinuousConfigError("g4_cross_window_weight must be finite and nonnegative")
        if self.g4_cross_window_weight > 0 and self.pose_time_offset_stage != "capture":
            raise ContinuousConfigError(
                "g4_cross_window_weight needs pose_time_offset_stage=capture"
            )
        if type(self.g4_frame_scales) is not bool:
            raise ContinuousConfigError("g4_frame_scales must be a bool")
        if self.g4_frame_scales and (
            self.pose_time_offset_stage != "capture" or self.g4_cross_window_weight > 0
        ):
            raise ContinuousConfigError(
                "g4_frame_scales needs pose_time_offset_stage=capture and g4_cross_window_weight=0"
            )
        if self.camera_intrinsics is not None:
            keys = {"width", "height", "fx", "fy", "cx", "cy"}
            intrinsics = self.camera_intrinsics
            if not isinstance(intrinsics, dict) or set(intrinsics) != keys or any(
                isinstance(intrinsics[k], bool) or not isinstance(intrinsics[k], (int, float))
                or not math.isfinite(intrinsics[k]) for k in keys
            ) or min(intrinsics["fx"], intrinsics["fy"], intrinsics["width"], intrinsics["height"]) <= 0:
                raise ContinuousConfigError(
                    "camera_intrinsics needs finite width, height, fx, fy, cx, cy with positive size and focal length"
                )
        if (
            isinstance(self.initial_queue_capacity, bool)
            or not isinstance(self.initial_queue_capacity, int)
            or self.initial_queue_capacity <= 0
        ):
            raise ContinuousConfigError("initial_queue_capacity must be a positive integer")
        if (
            isinstance(self.initial_max_retries, bool)
            or not isinstance(self.initial_max_retries, int)
            or self.initial_max_retries < 0
        ):
            raise ContinuousConfigError("initial_max_retries must be a nonnegative integer")
        if type(self.initial_keep_snapshot_on_success) is not bool:
            raise ContinuousConfigError(
                "initial_keep_snapshot_on_success must be a bool"
            )
        if (
            isinstance(self.retention_max_sessions, bool)
            or not isinstance(self.retention_max_sessions, int)
            or self.retention_max_sessions <= 0
        ):
            raise ContinuousConfigError("retention_max_sessions must be a positive integer")
        if (
            isinstance(self.retention_orphan_grace_sec, bool)
            or not isinstance(self.retention_orphan_grace_sec, (int, float))
            or not math.isfinite(float(self.retention_orphan_grace_sec))
            or self.retention_orphan_grace_sec < 0
        ):
            raise ContinuousConfigError(
                "retention_orphan_grace_sec must be a nonnegative finite number"
            )
        _validate_inference_resolution(
            self.initial_inference_resolution, "initial_inference_resolution"
        )
        _validate_worker_timeout(
            self.initial_worker_timeout_sec, "initial_worker_timeout_sec"
        )
        if self.initial_reconstruction_mode not in RECONSTRUCTION_MODES:
            raise ContinuousConfigError(
                "initial_reconstruction_mode must be one of "
                f"{RECONSTRUCTION_MODES!r}"
            )
        for role in self.roles:
            _validate_capture_runtime(
                EffectiveConfig(
                    sample_fps=float(self.initial_sample_fps),
                    window_frames=int(self.initial_window_frames),
                    queue_mode=self.initial_queue_mode,
                    queue_capacity=int(self.initial_queue_capacity),
                    max_retries=int(self.initial_max_retries),
                    keep_snapshot_on_success=self.initial_keep_snapshot_on_success,
                    save_sensor_depth=role.capture_config.save_sensor_depth,
                    inference_resolution=int(self.initial_inference_resolution),
                    worker_timeout_sec=float(self.initial_worker_timeout_sec),
                    overlap_frames=int(self.initial_overlap_frames),
                ),
                role,
                self.capture_wait_margin_sec,
            )
        for name in ("expected_checkpoint_sha256",):
            value = getattr(self, name)
            if value and (not isinstance(value, str) or not _valid_sha256(value)):
                raise ContinuousConfigError(
                    f"{name} must be empty or a lowercase SHA-256 digest"
                )
        for name in (
            "expected_checkpoint_path",
            "expected_result_root",
            "expected_algorithm_revision",
            "expected_ros_revision",
            "worker_python",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ContinuousConfigError(f"{name} must be a string")
            if value and any(ord(char) < 32 for char in value):
                raise ContinuousConfigError(f"{name} contains control characters")
        if self.expected_checkpoint_path and not Path(self.expected_checkpoint_path).is_absolute():
            raise ContinuousConfigError(
                "expected_checkpoint_path must be absolute"
            )
        if self.expected_result_root and not Path(self.expected_result_root).is_absolute():
            raise ContinuousConfigError("expected_result_root must be absolute")
        if not isinstance(self.cuda_visible_devices, str):
            raise ContinuousConfigError("cuda_visible_devices must be a string")

    def role_by_name(self, role: str) -> CameraRole | None:
        for candidate in self.roles:
            if candidate.role == role:
                return candidate
        return None


def _validate_inference_resolution(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContinuousConfigError(f"{label} must be an integer")
    if value == 0:
        return
    if value < 32 or value > 4096:
        raise ContinuousConfigError(
            f"{label} must be 0 (inherit) or in [32, 4096]"
        )


def _validate_worker_timeout(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContinuousConfigError(f"{label} must be a finite number")
    converted = float(value)
    if not _finite(converted) or converted < 0:
        raise ContinuousConfigError(f"{label} must be 0.0 or positive finite")


def _validate_window_frames(window_frames: Any, frame_group_size: int) -> None:
    if (
        isinstance(window_frames, bool)
        or not isinstance(window_frames, int)
        or window_frames <= 0
    ):
        raise ContinuousConfigError("window_frames must be a positive integer")
    if window_frames > frame_group_size:
        raise ContinuousConfigError(
            f"window_frames={window_frames} exceeds reconstruct "
            f"frame_group_size={frame_group_size}; all captured frames must "
            "enter one inference"
        )


def _validate_overlap_frames(overlap_frames: Any, window_frames: int) -> None:
    if (
        isinstance(overlap_frames, bool)
        or not isinstance(overlap_frames, int)
        or overlap_frames < 0
        or overlap_frames >= window_frames
    ):
        raise ContinuousConfigError(
            "overlap_frames must be an integer in [0, window_frames)"
        )


@dataclass(frozen=True)
class EffectiveConfig:
    """一个窗口封存时冻结的实际生效配置。"""

    sample_fps: float
    window_frames: int
    queue_mode: str
    queue_capacity: int
    max_retries: int
    keep_snapshot_on_success: bool
    save_sensor_depth: bool
    inference_resolution: int
    worker_timeout_sec: float
    overlap_frames: int = 0

    def digest(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_fps": float(self.sample_fps),
            "window_frames": int(self.window_frames),
            "queue_mode": self.queue_mode,
            "queue_capacity": int(self.queue_capacity),
            "max_retries": int(self.max_retries),
            "keep_snapshot_on_success": bool(self.keep_snapshot_on_success),
            "save_sensor_depth": bool(self.save_sensor_depth),
            "inference_resolution": int(self.inference_resolution),
            "worker_timeout_sec": float(self.worker_timeout_sec),
            "overlap_frames": int(self.overlap_frames),
        }


def _validate_capture_runtime(
    candidate: EffectiveConfig,
    role: CameraRole,
    capture_wait_margin_sec: float,
) -> None:
    """Validate a full-window capture goal against the selected Capture profile."""
    # Motion keyframes can space written frames up to keyframe_max_interval_sec apart.
    max_duration = (
        float(candidate.window_frames)
        * role.capture_config.max_frame_interval_sec(float(candidate.sample_fps))
        + float(capture_wait_margin_sec)
    )
    goal = SimpleNamespace(
        scene_id="cs-s0000000000000000-w0",
        sample_fps=candidate.sample_fps,
        max_frames=candidate.window_frames,
        max_duration=SimpleNamespace(to_sec=lambda: max_duration),
        set_save_sensor_depth=True,
        save_sensor_depth=candidate.save_sensor_depth,
    )
    try:
        validate_capture_goal(goal, role.capture_config)
    except CaptureError as error:
        raise ContinuousConfigError(
            f"runtime params violate capture profile for role {role.role!r}: {error}"
        ) from error


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return float(value) == float(value) and float(value) not in (
            float("inf"),
            float("-inf"),
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except (TypeError, ValueError):
        return False
    return True
