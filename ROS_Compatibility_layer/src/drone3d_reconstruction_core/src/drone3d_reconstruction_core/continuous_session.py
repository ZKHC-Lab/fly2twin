"""ROS 无关的连续 session 调度器与磁盘 FIFO 核心。

本模块实现连续调度与磁盘 FIFO：

  - 复用现有 Capture/Reconstruct Action（由注入的客户端/后端适配）；
  - 连续窗口：首窗 N 帧，后续复用上一窗尾部 overlap_frames 帧；
  - 采集循环与单 worker 消费循环解耦；
  - 磁盘 FIFO：内存只存小状态，快照文件全部留在磁盘；
  - `lossless_fifo` 无损队列：满时暂停新采集，不丢窗口；
  - max_retries、pause/resume、stop_all、磁盘红线。

本模块不导入 rospy/actionlib；ROS 适配在 continuous_session_node.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import unquote, unquote_to_bytes, urlsplit

from .capture import CaptureConfig, CaptureError, validate_capture_goal
from .depth_metrics import compute_first_frame_depth_metrics
from .frame_pool import FramePool, FramePoolConfig, FramePoolError
from .formal_result import (
    FORMAL_RECORD_SCHEMA,
    FORMAL_RECORD_SCHEMA_VERSION,
    FormalResultError,
    publish_formal_record,
    verify_result_manifest,
)
from .odom_local import (
    COORDINATE_MODEL_LOCAL_UNALIGNED,
    COORDINATE_ODOM_LOCAL_APPROX,
)
from .snapshot_validator import validate_snapshot
from .g4.backend_process import BACKEND_G4, G4BackendProcess

from .action_types import (
    ACTION_SUCCEEDED,
    CameraRole,
    CaptureActionOutcome,
    ReconstructActionOutcome,
)

# 会话状态
SESSION_IDLE = "IDLE"
SESSION_RUNNING = "RUNNING"
SESSION_PAUSED = "PAUSED"
SESSION_DISK_PAUSED = "DISK_PAUSED"
SESSION_FATAL_OOM = "FATAL_OOM"
SESSION_MEMORY_PAUSED = "MEMORY_PAUSED"
SESSION_STOPPING = "STOPPING"
SESSION_STOPPED = "STOPPED"
# STOP with queued windows: capture has stopped, the queue is still being reconstructed.
SESSION_DRAINING = "DRAINING"
# After the queue drains, how long STOP waits for G4 to publish the complete map.
_DRAIN_G4_WAIT_SEC = 600.0

# CUDA OOM 的稳定文本模式（大小写不敏感；worker/run.log/result 消息共用）
_FATAL_OOM_PATTERN = re.compile(
    r"(?i)(\boom\b|out of memory|cuda[^\n]{0,64}out of memory)"
)

# 硬停（资源门/CUDA OOM）取消在飞采集时写入的记录文本；用于识别
# "被暂停取消、恢复后需要重新采集"的窗口，区别于普通采集失败。
_CANCELLED_CAPTURE_PATTERN = re.compile(
    r"(?i)(fatal oom|in-flight capture cancelled)"
)

_MEMORY_GUARD_PATTERN = re.compile(
    r"worker stopped: resource_guard|\b(?:LOW_MEM_AVAILABLE|HIGH_SWAP_USED|HIGH_SWAP_GROWTH)\b"
)

# 窗口状态
WINDOW_CAPTURING = "CAPTURING"
WINDOW_PENDING = "PENDING"
WINDOW_RUNNING = "RUNNING"
WINDOW_SUCCEEDED = "SUCCEEDED"
WINDOW_FAILED = "FAILED"
WINDOW_CANCELLED = "CANCELLED"

QUEUE_LOSSLESS = "lossless_fifo"
_DISK_REDLINE_ERROR = "disk redline reached; new capture paused"

RECONSTRUCTION_MODE_PER_WINDOW = "per_window"
RECONSTRUCTION_MODES = (RECONSTRUCTION_MODE_PER_WINDOW,)

# 一个窗口送去重建的帧里，没有关联上位姿（pose_valid 为假）的帧所占比例的上限；
# 严格大于时该窗口判失败、不提交重建。维护者 2026-11-23 决定：办公室那次飞行
# （118 窗、每窗 40 帧）里最差的两个窗口恰好缺 8 帧即 20%，没有窗口超过。
MAX_MISSING_POSE_FRACTION = 0.2

JOB_PENDING = "PENDING"
JOB_RUNNING = "RUNNING"
JOB_SUCCEEDED = "SUCCEEDED"
JOB_FAILED = "FAILED"
JOB_CANCELLED = "CANCELLED"
_JOB_TERMINAL = frozenset((JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED))

_SNAPSHOT_DIR_RE = re.compile(
    r"cs-s?[0-9a-f]{16}-w[0-9]+-[0-9a-f]{16}\Z"
)
_SCENE_ID_PREFIX_RE = re.compile(r"cs-s[0-9a-f]{16}-w[0-9]+\Z")
# 正式化期间检测到 stop_all 的专用返回值：消费路径必须按 CANCELLED 处理。
_FORMAL_SKIPPED_STOPPING = "formal record publication skipped: session is stopping"
_MAX_SNAPSHOT_MANIFEST_BYTES = 16 * 1024 * 1024
_NO_SENSOR_DEPTH_REASONS = frozenset(
    (
        "sensor depth saving is disabled",
        "snapshot has no first-frame sensor depth",
    )
)


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
        if not isinstance(self.reconstruct_action, str) or not self.reconstruct_action:
            raise ContinuousConfigError("reconstruct_action must be nonempty")
        for name in ("service_name", "diagnostics_topic"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or " " in value:
                raise ContinuousConfigError(f"{name} must be a nonempty ROS name")
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
    """Apply the selected Capture profile's existing goal boundary."""
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


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: str
    message: str = ""


class ReconstructionBackend:
    """P0-3 重建后端最小契约；V1 只有 LocalCudaBackend。"""

    def submit(
        self,
        snapshot_uri: str,
        manifest_sha256: str,
        window_id: str,
        inference_resolution: int = 0,
        worker_timeout_sec: float = 0.0,
    ) -> str:
        raise NotImplementedError

    def status(self, job_id: str) -> JobStatus:
        raise NotImplementedError

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError

    def collect_result(self, job_id: str) -> ReconstructActionOutcome:
        raise NotImplementedError


class ContinuousCaptureClient:
    """连续采集客户端（Action + CaptureControl pause/resume）。"""

    def wait_for_server(self, timeout_sec: float) -> bool:
        raise NotImplementedError

    def send_goal(
        self,
        scene_id: str,
        sample_fps: float,
        max_frames: int,
        max_duration_sec: float,
        feedback_cb,
        save_sensor_depth: bool | None = None,
        set_save_sensor_depth: bool = False,
    ) -> None:
        raise NotImplementedError

    def wait_for_result(self, timeout_sec: float) -> bool:
        raise NotImplementedError

    def cancel_goal(self) -> None:
        raise NotImplementedError

    def outcome(self) -> CaptureActionOutcome:
        raise NotImplementedError

    def pause(self) -> bool:
        raise NotImplementedError

    def resume(self) -> bool:
        raise NotImplementedError


@dataclass
class WindowRecord:
    """一个窗口在调度器内的元数据；图像内容永远只存在于磁盘。"""

    window_index: int
    scene_id: str
    role: CameraRole
    effective: EffectiveConfig
    state: str = WINDOW_CAPTURING
    snapshot_uri: str = ""
    manifest_sha256: str = ""
    capture_attempts: int = 0
    reconstruct_attempts: int = 0
    snapshot_bytes: int = 0
    captured_frames: int = 0
    capture_goal_frames: int = 0
    new_frame_count: int = 0
    overlap_frame_count: int = 0
    # 送去重建的帧（含复用的共享帧）里没有位姿的帧数；窗口判定用它。
    missing_pose_frames: int = 0
    # 其中本窗新采的帧。会话累计只加这一部分，共享帧不重复计数。
    new_missing_pose_frames: int = 0
    window_frame_ids: list[int] = field(default_factory=list)
    model_frame_ids: list[int] = field(default_factory=list)
    worker_result_manifest_uri: str = ""
    worker_result_manifest_sha256: str = ""
    queued_at_monotonic: float | None = None
    finished_monotonic: float | None = None
    error: str = ""
    source_stale: bool = False
    capture_cancel_pending: bool = False
    # True also means no Capture goal has been dispatched yet.
    capture_action_terminal: bool = True
    alignment: dict | None = None
    formal_result_uri: str = ""
    formal_provenance_sha256: str = ""
    capture_generation: int = 0
    captured_frame_ids: list[int] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)
    # 人工恢复历史：资源门拒绝后由操作者显式重试的次数与每次的拒绝原因。
    manual_retries: int = 0
    guard_failures: list[str] = field(default_factory=list)

    @property
    def attempts(self) -> int:
        """历史兼容：capture 与 reconstruct 阶段尝试次数之和。"""
        return self.capture_attempts + self.reconstruct_attempts


@dataclass(frozen=True)
class SessionStatus:
    """GET_STATUS 的机器可读快照。"""

    state: str
    session_id: str
    role: str
    window_index: int
    captured_frames: int
    capture_target_frames: int
    pending_windows: int
    running_window: int
    succeeded_windows: int
    failed_windows: int
    dropped_windows: int
    effective: EffectiveConfig
    last_error: str
    min_free_disk_gb: float
    fifo_bytes: int = 0
    fifo_oldest_wait_sec: float | None = None
    disk_free_gb: float = 0.0
    last_alignment_state: str = COORDINATE_MODEL_LOCAL_UNALIGNED
    formal_windows: int = 0
    reconstruction_mode: str = RECONSTRUCTION_MODE_PER_WINDOW
    alignment_backend: str = BACKEND_G4


class ContinuousSession:
    """连续 session 调度器核心。"""

    def __init__(
        self,
        config: ContinuousConfig,
        *,
        capture_client_factory: Callable[[CameraRole], ContinuousCaptureClient],
        backend_factory: Callable[[], ReconstructionBackend],
        disk_free_gb: Callable[[], float] | None = None,
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
        monotonic: Callable[[], float] = time.monotonic,
        result_gate: Callable[..., Any] | None = None,
        result_publisher: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._capture_client_factory = capture_client_factory
        self._backend_factory = backend_factory
        self._disk_free_gb = disk_free_gb or (lambda: float("inf"))
        self._result_gate = result_gate if result_gate is not None else verify_result_manifest
        self._result_publisher = (
            result_publisher if result_publisher is not None else publish_formal_record
        )
        self._worker_disk_blocked = False
        self._fifo_bytes = 0
        self._fatal_oom = False
        self._last_alignment_state = COORDINATE_MODEL_LOCAL_UNALIGNED
        self._formal_windows = 0
        self._last_worker_wall_time_sec: float | None = None
        self._last_service_wall_time_sec: float | None = None
        self._last_service_budget_sec: float | None = None
        self._reconstruction_mode = config.initial_reconstruction_mode
        self._pool: FramePool | None = None
        self._capture_generation = 0
        self._config_timeline: list[dict[str, Any]] = []
        self._thread_factory = thread_factory
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._goal_dispatch_lock = threading.Lock()
        self._selected_role: CameraRole | None = None
        self._runtime = EffectiveConfig(
            sample_fps=config.initial_sample_fps,
            window_frames=config.initial_window_frames,
            queue_mode=config.initial_queue_mode,
            queue_capacity=config.initial_queue_capacity,
            max_retries=config.initial_max_retries,
            keep_snapshot_on_success=config.initial_keep_snapshot_on_success,
            save_sensor_depth=False,
            inference_resolution=config.initial_inference_resolution,
            worker_timeout_sec=config.initial_worker_timeout_sec,
            overlap_frames=config.initial_overlap_frames,
        )
        self._state = SESSION_IDLE
        self._session_id = ""
        self._next_window = 0
        self._windows: list[WindowRecord] = []
        self._g4_process = None
        self._g4_start_error = ""
        self._fifo: list[WindowRecord] = []  # deque-like; only PENDING entries
        self._running_window: WindowRecord | None = None
        self._succeeded = 0
        self._failed = 0
        self._missing_pose_frames = 0
        # 失败历史分两层：failed_windows 只数**未恢复的终态失败**（外部健康
        # 判据）；attempt_failures 单调累加每次尝试级失败（人工重试后仍保留）。
        self._attempt_failures = 0
        self._manual_retries = 0
        self._last_error = ""
        self._capture_client: ContinuousCaptureClient | None = None
        self._backend: ReconstructionBackend | None = None
        self._active_job_id: str | None = None
        self._active_job_cancel_requested = False
        self._backend_cleanup_failure = ""
        self._stop_audit: dict[str, Any] | None = None
        self._stopping = False
        self._capture_stopped = False  # STOP latched for capture only (DRAINING)
        self._capture_thread: threading.Thread | None = None
        self._consume_thread: threading.Thread | None = None
        self._wake_capture = threading.Event()
        self._wake_consume = threading.Event()
        self._capture_done = threading.Event()
        self._consume_done = threading.Event()
        # 硬停的两个互斥原因，决定 RESUME 是否可人工恢复。
        self._oom_latched = False
        self._memory_guard_paused = False
        self._memory_failed_window: WindowRecord | None = None
        # 人工恢复是"校验 → 锁外等旧线程退出 → 重新入队并重启线程"的多步
        # 过程；该标志防止并发/重复 RESUME 重复入队同一窗口。
        self._resume_in_progress = False
        # 本次人工恢复事务的唯一令牌：锁外等待期间会话可能被 STOP 后重建，
        # 回来必须核对令牌/会话/记录三者未变，绝不能把重试打到新会话上。
        self._resume_token: object | None = None

    @property
    def config(self) -> ContinuousConfig:
        return self._config

    # ------------------------------------------------------------------
    def select_source(self, role_name: str) -> tuple[int, str]:
        role = self._resolve_role(role_name)
        if role is None:
            return 2, (
                f"unknown camera role {role_name!r}; configured roles: "
                f"{sorted(r.role for r in self._config.roles)}"
            )
        with self._lock:
            if self._state in (
                SESSION_RUNNING,
                SESSION_PAUSED,
                SESSION_DISK_PAUSED,
                SESSION_FATAL_OOM,
                SESSION_MEMORY_PAUSED,
                SESSION_STOPPING,
                SESSION_DRAINING,
            ):
                return 3, "cannot change source while a session is active"
            self._selected_role = role
            return 0, f"selected camera role {role.role!r}"

    def start_session(
        self, role_name: str = "", reconstruction_mode: str = ""
    ) -> tuple[int, str]:
        with self._lock:
            if self._state not in (SESSION_IDLE, SESSION_STOPPED):
                return 3, f"session is already {self._state}"
            role = self._resolve_role(role_name) if role_name else self._selected_role
            if role is None:
                return 4, "no camera role selected"
            selected_mode = (
                reconstruction_mode
                if reconstruction_mode
                else self._config.initial_reconstruction_mode
            )
            if selected_mode not in RECONSTRUCTION_MODES:
                return 5, (
                    "reconstruction_mode must be one of "
                    f"{RECONSTRUCTION_MODES!r}"
                )
            # 会话参数来自冻结 profile；只有是否保存传感器深度随所选角色。
            runtime = EffectiveConfig(
                sample_fps=self._config.initial_sample_fps,
                window_frames=self._config.initial_window_frames,
                queue_mode=self._config.initial_queue_mode,
                queue_capacity=self._config.initial_queue_capacity,
                max_retries=self._config.initial_max_retries,
                keep_snapshot_on_success=self._config.initial_keep_snapshot_on_success,
                save_sensor_depth=role.capture_config.save_sensor_depth,
                inference_resolution=self._config.initial_inference_resolution,
                worker_timeout_sec=self._config.initial_worker_timeout_sec,
                overlap_frames=self._config.initial_overlap_frames,
            )
            self._selected_role = role
            self._runtime = runtime
            self._reconstruction_mode = selected_mode
            self._pool = None
            self._capture_generation = 0
            self._session_id = "s" + secrets.token_hex(8)
            self._next_window = 0
            self._windows = []
            self._g4_start_error = ""
            self._fifo = []
            self._running_window = None
            self._succeeded = 0
            self._failed = 0
            self._missing_pose_frames = 0
            self._attempt_failures = 0
            self._manual_retries = 0
            self._last_error = ""
            self._stopping = False
            self._capture_stopped = False
            self._active_job_cancel_requested = False
            self._backend_cleanup_failure = ""
            self._stop_audit = None
            self._fatal_oom = False
            self._oom_latched = False
            self._memory_guard_paused = False
            self._memory_failed_window = None
            self._resume_in_progress = False
            self._resume_token = None
            self._worker_disk_blocked = False
            self._fifo_bytes = 0
            self._last_alignment_state = COORDINATE_MODEL_LOCAL_UNALIGNED
            self._formal_windows = 0
            self._last_worker_wall_time_sec = None
            self._last_service_wall_time_sec = None
            self._last_service_budget_sec = None
            self._config_timeline = [
                {
                    "index": 0,
                    "event": "session_start",
                    "config_digest": self._runtime.digest(),
                    "previous_config_digest": "",
                    "applies_from_window_index": 0,
                    "config": self._runtime.as_dict(),
                }
            ]
            self._capture_done.clear()
            self._consume_done.clear()
            self._backend = self._backend_factory()
            self._state = SESSION_RUNNING
            self._write_journal(
                {
                    "event": "session_start",
                    "session_id": self._session_id,
                    "role": role.role,
                    "reconstruction_mode": self._reconstruction_mode,
                }
            )
            # Thread publication is part of the START state transition.  STOP
            # must never observe RUNNING before both thread references exist.
            self._start_worker_threads_locked()
            session_id = self._session_id
        return 0, f"session {session_id} started for role {role.role!r}"

    def _start_worker_threads_locked(self, restart_capture: bool = True) -> None:
        """(重新)启动采集/消费线程；必须在持锁状态下调用。

        资源门暂停会让两个线程按硬停 latch 退出，因此人工恢复必须重新
        建线程——只把 state 改回 RUNNING 会得到一个没有 worker 的空转会话。
        调用方必须已经在锁外确认旧线程退出（见 resume()），否则会出现两个
        线程同时消费同一个 FIFO。会话已在排空（STOP 之后）时只重启消费线程：
        重试窗照样会被消费，不需要也不能再开新采集。
        """
        self._capture_done.clear()
        self._consume_done.clear()
        consume_thread = self._make_thread("continuous-consume", self._consume_loop)
        self._consume_thread = consume_thread
        if restart_capture:
            capture_thread = self._make_thread("continuous-capture", self._capture_loop)
            self._capture_thread = capture_thread
            capture_thread.start()
        else:
            self._capture_thread = None
        consume_thread.start()

    def pause(self) -> tuple[int, str]:
        with self._lock:
            if self._state not in (SESSION_RUNNING, SESSION_DISK_PAUSED):
                return 4, f"cannot pause session in state {self._state}"
            previous_state = self._state
            client = self._capture_client
        if client is not None:
            try:
                accepted = client.pause()
            except BaseException:
                accepted = False
            if not accepted:
                with self._lock:
                    if self._state == previous_state:
                        self._state = previous_state
                    self._last_error = (
                        "capture client pause request failed; session kept "
                        f"{previous_state}"
                    )
                return 3, self._last_error
        with self._lock:
            if self._state == previous_state:
                self._state = SESSION_PAUSED
        return 0, "capture paused; FIFO consumption continues"

    def resume(self) -> tuple[int, str]:
        binding = None
        with self._lock:
            if self._state == SESSION_MEMORY_PAUSED:
                if self._resume_in_progress:
                    return 4, "a manual retry of the resource-guard window is already in progress"
                code, message = self._memory_retry_precheck_locked()
                if code != 0:
                    return code, message
                token = object()
                self._resume_in_progress = True
                self._resume_token = token
                binding = (token, self._session_id, self._memory_failed_window)
            elif self._state != SESSION_PAUSED:
                return 4, f"cannot resume session in state {self._state}"
            client = self._capture_client
        if binding is not None:
            # 旧线程正按硬停 latch 退出：必须在**锁外**有界等待它们真的结束，
            # 否则会重启出第二个消费线程同时抢 FIFO（重复/丢失正式结果）。
            # 此期间 latch 仍为真，所以等待是有界且不会与业务竞争的死等。
            token, session_id, record = binding
            alive = self._join_threads(
                deadline=self._monotonic() + float(self._config.stop_timeout_sec)
            )
            with self._lock:
                try:
                    if self._resume_token is not token:
                        # 锁外等待期间会话被 STOP 后重建（或已被别的命令接管）：
                        # 这次 RESUME 的授权已经失效，什么都不做。
                        return 4, (
                            "manual retry was superseded while waiting for worker "
                            "threads; nothing was armed"
                        )
                    if self._stopping:
                        return 4, "cannot resume session while stop_all is converging"
                    if self._session_id != session_id or self._memory_failed_window is not record:
                        return 4, (
                            "session changed during manual retry "
                            f"(session {self._session_id!r}); nothing was armed"
                        )
                    if alive:
                        return 3, (
                            "cannot retry window yet: worker threads are still "
                            f"converging ({', '.join(alive)}); retry RESUME"
                        )
                    return self._arm_memory_retry_locked(token, session_id, record)
                finally:
                    if self._resume_token is token:
                        self._resume_in_progress = False
                        self._resume_token = None
        if client is not None:
            try:
                accepted = client.resume()
            except BaseException:
                accepted = False
            if not accepted:
                with self._lock:
                    if self._state == SESSION_PAUSED:
                        self._last_error = (
                            "capture client resume request failed; "
                            "session kept PAUSED"
                        )
                return 3, self._last_error
        with self._lock:
            if self._state == SESSION_PAUSED:
                if self._disk_low_locked():
                    self._state = SESSION_DISK_PAUSED
                else:
                    self._state = SESSION_RUNNING
                self._wake_capture.set()
        return 0, "capture resumed"

    # ------------------------------------------------------------------
    # MEMORY_PAUSED 的人工恢复（只重试被资源门拒绝的那个窗口）
    # ------------------------------------------------------------------
    def _memory_retry_precheck_locked(self) -> tuple[int, str]:
        """人工恢复的前置条件；不改变任何状态。"""
        if not self._memory_guard_paused or self._oom_latched:
            return 4, (
                f"cannot resume session in state {SESSION_MEMORY_PAUSED} without a "
                "recoverable resource-guard failure"
            )
        if self._stopping:
            return 4, "cannot resume session while stop_all is converging"
        if self._runtime.queue_mode != QUEUE_LOSSLESS:
            return 5, (
                "manual retry of a resource-guard window requires lossless_fifo; "
                f"{self._runtime.queue_mode} may evict the retried window or a "
                "sealed waiting window"
            )
        if self._active_job_id is not None or self._running_window is not None:
            return 3, (
                "cannot retry window while the backend still owns job "
                f"{self._active_job_id!r}"
            )
        record = self._memory_failed_window
        if record is None or record.state != WINDOW_FAILED:
            return 3, "no resource-guard failed window is available to retry"
        if record.window_index >= self._next_window:
            return 3, (
                f"window {record.window_index} is not sealed yet "
                f"(next capture window is {self._next_window})"
            )
        skipped = self._unfinished_capture_error_locked()
        if skipped is not None:
            return 3, skipped
        snapshot_error = self._retry_snapshot_error(record)
        if snapshot_error is not None:
            return 3, (
                f"cannot retry window {record.window_index}: {snapshot_error}; "
                "a resealed input cannot be reconstructed without re-capturing"
            )
        return 0, ""

    def _unfinished_capture_error_locked(self) -> str | None:
        """硬停取消掉的采集窗，恢复后是否真的还能被重新采到。

        已有后续封存记录的索引不再缺采。其余取消窗只有正好位于
        ``_next_window``、且会重启采集时才能补采，否则必须拒绝恢复。
        """
        target = self._memory_failed_window
        for position, record in enumerate(self._windows):
            if record is target or record.state != WINDOW_FAILED:
                continue
            if not _CANCELLED_CAPTURE_PATTERN.search(record.error or ""):
                continue  # 普通采集失败：既有语义就是跳过该窗并前进
            if any(
                replacement.window_index == record.window_index
                and replacement.scene_id == record.scene_id
                and replacement.window_index < self._next_window
                and replacement.capture_action_terminal
                and replacement.snapshot_uri
                and _valid_sha256(replacement.manifest_sha256)
                and (
                    replacement.state in (WINDOW_PENDING, WINDOW_RUNNING, WINDOW_SUCCEEDED)
                    or replacement is target
                )
                for replacement in self._windows[position + 1:]
            ):
                continue
            if record.window_index == self._next_window:
                continue  # 重启采集会从这个索引重新采
            return (
                f"window {record.window_index} was cancelled by the resource pause "
                f"but capture restarts from window {self._next_window}"
                "; that window cannot be recovered in place"
            )
        return None

    def _arm_memory_retry_locked(self, token, session_id, record) -> tuple[int, str]:
        """把被拒绝的窗口按原样重新入队，并重启已退出的 worker 线程。

        ``token``/``session_id``/``record`` 是本次事务在首检时绑定的三元组：
        锁外等待之后必须逐项复核，避免把重试打进另一个会话。
        """
        if (
            self._resume_token is not token
            or self._session_id != session_id
            or self._memory_failed_window is not record
        ):
            return 4, (
                "manual retry binding changed before arming; nothing was armed"
            )
        code, message = self._memory_retry_precheck_locked()
        if code != 0:
            return code, message
        # 失败历史保留在 record.guard_failures 与 _attempt_failures 里；
        # failed_windows 只表示"未恢复的终态失败"，重试入队后归零。
        record.manual_retries += 1
        self._manual_retries += 1
        record.state = WINDOW_PENDING
        record.finished_monotonic = None
        record.done.clear()
        record.queued_at_monotonic = self._monotonic()
        # 保留失败窗 + 原 FIFO：绝不为人工恢复删除任何已采集窗口。
        self._fifo.insert(0, record)
        self._fifo_bytes += record.snapshot_bytes
        self._failed = max(0, self._failed - 1)
        self._memory_guard_paused = False
        self._memory_failed_window = None
        self._fatal_oom = False
        self._oom_latched = False
        restart_capture = not self._capture_stopped
        if self._capture_stopped:
            self._state = SESSION_DRAINING
        else:
            self._state = (
                SESSION_DISK_PAUSED if self._disk_low_locked() else SESSION_RUNNING
            )
        self._last_error = ""
        self._start_worker_threads_locked(restart_capture=restart_capture)
        self._wake_capture.set()
        self._wake_consume.set()
        self._write_journal(
            {
                "event": "window_manual_retry",
                "session_id": self._session_id,
                "window_index": record.window_index,
                "scene_id": record.scene_id,
                "snapshot_uri": record.snapshot_uri,
                "snapshot_bytes": record.snapshot_bytes,
                "config_digest": record.effective.digest(),
                "manual_retries": record.manual_retries,
                "attempt_failures": self._attempt_failures,
                "capture_restarted": restart_capture,
                "reason": record.guard_failures[-1] if record.guard_failures else "",
                "resource_gate": (
                    "unchanged; the retry is re-submitted to the same worker launch gate"
                ),
            }
        )
        return 0, (
            f"manual retry armed for window {record.window_index}; the input snapshot "
            "is reused and the same resource gate applies"
        )

    def _retry_snapshot_error(self, record: WindowRecord) -> str | None:
        """人工重试前核对被保留的输入快照（含清单摘要校验）。

        只复用该窗口自己保留的输入，不复用任何其它窗口的采集；校验用的是
        与采集/重建完全相同的 ``validate_snapshot``。
        """
        uri = record.snapshot_uri
        if not uri:
            return "the failed window has no preserved input snapshot"
        if not _valid_sha256(record.manifest_sha256):
            return "the preserved input snapshot has no valid manifest sha256"
        role = record.role
        root = Path(role.capture_config.snapshot_root)
        try:
            split = urlsplit(uri)
            if split.scheme != "file" or split.netloc or split.query or split.fragment:
                return f"snapshot uri is not a local file uri: {uri!r}"
            path = Path(unquote(split.path))
            if not path.is_absolute() or not _is_relative_to(path, root):
                return f"snapshot {path} is outside the configured snapshot root"
            if not _SNAPSHOT_DIR_RE.fullmatch(path.name):
                return f"snapshot directory name is not scheduler-owned: {path.name}"
            if not path.is_dir() or path.is_symlink():
                return f"preserved input snapshot is missing: {path}"
            validate_snapshot(
                uri,
                record.manifest_sha256,
                (role.capture_config.snapshot_root,),
                role.capture_config.snapshot_limits,
            )
        except BaseException as error:
            return (
                "preserved input snapshot failed validation: "
                f"{type(error).__name__}: {error}"
            )
        return None

    def stop_all(self, discard_queue: bool = False) -> tuple[int, str]:
        """STOP. With windows queued or reconstructing (per_window), the first STOP
        only stops capture and the session reconstructs the queue before it stops
        (DRAINING); the NX reconstructs several times slower than it captures, so
        deleting the queue would lose most of a flight. STOP again while DRAINING,
        STOP with nothing queued, or ``discard_queue`` stops at once and discards
        unfinished windows.
        """
        drain = None if discard_queue else self._begin_drain()
        if drain is not None:
            return drain
        return self._stop_now()

    def _begin_drain(self) -> tuple[int, str] | None:
        with self._lock:
            if (
                self._stopping
                or self._capture_stopped
                or self._fatal_oom
                or self._state not in (SESSION_RUNNING, SESSION_PAUSED, SESSION_DISK_PAUSED)
                or not (self._fifo or self._running_window is not None or self._active_job_id is not None)
            ):
                return None
            self._capture_stopped = True
            self._state = SESSION_DRAINING
            client = self._capture_client
            session_id = self._session_id
            remaining = len(self._fifo) + int(self._running_window is not None)
            self._write_journal({
                "event": "session_drain_start",
                "session_id": session_id,
                "pending_before": len(self._fifo),
                "running_window_before": (
                    self._running_window.window_index if self._running_window is not None else None
                ),
            })
            self._wake_capture.set()
            self._wake_consume.set()
        if client is not None:
            try:
                client.cancel_goal()  # the unfinished window is discarded
            except BaseException:
                pass
        threading.Thread(
            target=self._drain_then_stop, args=(session_id,), name="drain", daemon=True
        ).start()
        return 0, (
            f"capture stopped; reconstructing {remaining} remaining window(s) before "
            "the session stops (STOP again to discard them and stop now)"
        )

    def _drain_then_stop(self, session_id: str) -> None:
        """Stop the session once the queue is empty. A memory-guard pause keeps waiting
        for the operator (RESUME retries, STOP discards); a hard STOP ends the watch."""
        while True:
            with self._lock:
                if self._session_id != session_id or self._stopping or not self._capture_stopped:
                    return
                if (
                    not self._fifo
                    and self._running_window is None
                    and self._active_job_id is None
                    and self._capture_done.is_set()
                ):
                    break
            time.sleep(0.5)  # not _wake_consume: the consume loop owns clearing it
        # Stopping closes the G4 process: first let it publish a revision with every
        # formal window, or the delivery export finds the map incomplete.
        deadline = time.monotonic() + _DRAIN_G4_WAIT_SEC
        while True:
            with self._lock:
                if self._session_id != session_id or self._stopping:
                    return
                g4, formal = self._g4_process, self._formal_windows
            if g4 is None:
                break
            status = g4.status()
            if status.get("state") == "ready" and int(status.get("window_count") or 0) >= formal:
                break
            if status.get("state") in ("error", "stopped") or time.monotonic() > deadline:
                with self._lock:
                    self._write_journal({
                        "event": "session_drain_g4_incomplete",
                        "session_id": session_id,
                        "formal_windows": formal,
                        "g4_state": status.get("state"),
                        "g4_window_count": status.get("window_count"),
                        "g4_error": status.get("error", ""),
                    })
                break
            time.sleep(1.0)
        with self._lock:
            if self._session_id != session_id or self._stopping:
                return
        self._stop_now()

    def _stop_now(self) -> tuple[int, str]:
        stop_deadline = self._monotonic() + float(self._config.stop_timeout_sec)
        # STOP 先原子地 latch _stopping，再进入有界收敛。不要先抢
        # _goal_dispatch_lock：consume 线程持该锁执行可能阻塞的
        # backend.submit，STOP 等锁会死锁。latch 后任何尚未派发的
        # Capture/Reconstruct 都会在锁内第二次检查时自行短路；已经越过
        # 检查的 in-flight dispatch 由下面的 cancel/join 有界收敛。
        with self._lock:
            if self._state in (SESSION_IDLE, SESSION_STOPPED):
                return 4, "no active session"
            self._stopping = True
            self._state = SESSION_STOPPING
            capture_client = self._capture_client
            backend = self._backend
            job_id = self._active_job_id
            active_capture = next(
                (
                    record
                    for record in reversed(self._windows)
                    if record.state == WINDOW_CAPTURING
                    and not record.capture_action_terminal
                    and not record.done.is_set()
                ),
                None,
            )
            if active_capture is not None:
                # Publish cancellation ownership synchronously with STOP;
                # do not rely on the capture thread winning a race before
                # the bounded join expires.
                active_capture.capture_cancel_pending = True
            if self._stop_audit is None:
                self._stop_audit = {
                    "running_before": bool(
                        self._running_window is not None or job_id is not None
                    ),
                    "running_window_before": (
                        self._running_window.window_index
                        if self._running_window is not None
                        else None
                    ),
                    "pending_before": len(self._fifo),
                    "backend_job_id_before": job_id,
                    "backend_cancel_requested": bool(
                        job_id is not None and self._active_job_cancel_requested
                    ),
                    "backend_terminal_collected": (
                        False if job_id is not None else None
                    ),
                    "backend_terminal_state": None,
                    "backend_terminal_message": "",
                    "snapshot_before": self._scheduler_snapshot_dir_count(),
                    "staging_before": self._formal_staging_dir_count(),
                }
            self._wake_capture.set()
            self._wake_consume.set()
            g4_process, self._g4_process = self._g4_process, None
        if g4_process is not None:
            g4_process.close()
        if capture_client is not None:
            try:
                capture_client.cancel_goal()
            except BaseException:
                pass
        if backend is not None and job_id is not None:
            try:
                backend.cancel(job_id)
            except BaseException as error:
                with self._lock:
                    self._stop_audit["backend_cancel_error"] = (
                        f"{type(error).__name__}: {error}"
                    )
            else:
                with self._lock:
                    self._update_stop_audit_backend_locked(
                        job_id, cancel_requested=True
                    )
        alive_threads = self._join_threads(deadline=stop_deadline)
        with self._lock:
            # 删除所有尚未开始重建的 PENDING snapshot。
            for record in list(self._fifo):
                self._remove_snapshot(record)
            self._fifo.clear()
            self._fifo_bytes = 0
            # 取消的 RUNNING 窗口同样不能留下未提交 snapshot。
            # 若消费线程 join 超时仍存活且 worker 尚未终态，先保留输入
            # snapshot（消费线程退出时会自行清理），避免与正在读盘的
            # worker 竞争。
            running = self._running_window
            consume_alive = (
                self._consume_thread is not None
                and self._consume_thread.is_alive()
            )
            if (
                running is not None
                and running.state != WINDOW_SUCCEEDED
                and running.snapshot_uri
            ):
                if not consume_alive or running.state in (
                    WINDOW_CANCELLED,
                    WINDOW_FAILED,
                ):
                    self._remove_snapshot(running)
                else:
                    self._last_error = (
                        "stop_all: worker cleanup still in progress; "
                        "running snapshot retained for consume thread cleanup"
                    )
            # stop 竞态中的 CAPTURING/CANCELLED 记录也可能已经拿到 snapshot。
            # 消费线程仍存活且 worker 尚未终态的 RUNNING 记录除外：删除其
            # 输入 snapshot 会与正在读盘的 worker 竞争，由消费线程自行清理。
            for record in self._windows:
                if (
                    record.state != WINDOW_SUCCEEDED
                    and record.snapshot_uri
                    and not (
                        record is running
                        and consume_alive
                        and record.state == WINDOW_RUNNING
                    )
                ):
                    self._remove_snapshot(record)
            for record in self._windows:
                if record.state == WINDOW_CAPTURING and not record.done.is_set():
                    record.state = WINDOW_CANCELLED
                    record.error = "stop_all"
                    record.finished_monotonic = self._monotonic()
                    record.done.set()
            pending_capture = next(
                (
                    record
                    for record in self._windows
                    if record.capture_cancel_pending
                    and not record.capture_action_terminal
                ),
                None,
            )
            capture_client = self._capture_client
            active_job_id = self._active_job_id
        # 1：capture Action 取消后必须证明其终态；超时保持 STOPPING。
        if pending_capture is not None and capture_client is not None:
            while True:
                try:
                    capture_client.cancel_goal()
                except BaseException:
                    pass
                if capture_client.wait_for_result(0.1):
                    with self._lock:
                        pending_capture.capture_action_terminal = True
                        pending_capture.capture_cancel_pending = False
                        if self._capture_client is capture_client:
                            self._capture_client = None
                    break
                if self._monotonic() >= stop_deadline:
                    with self._lock:
                        self._state = SESSION_STOPPING
                        self._stopping = True
                        self._last_error = (
                            "stop_all: capture action terminal was not "
                            "confirmed within "
                            f"{float(self._config.stop_timeout_sec)} s; "
                            "retry stop_all when the capture action reaches "
                            "terminal"
                        )
                        self._write_journal(
                            {
                                "event": "session_stop_pending",
                                "session_id": self._session_id,
                                **self._stop_audit,
                                "reason": self._last_error,
                                "terminal_state": self._state,
                                "running_after": int(
                                    self._running_window is not None
                                ),
                                "pending_after": len(self._fifo),
                                "worker_active_after": int(
                                    self._active_job_id is not None
                                ),
                                "threads_alive_after": len(alive_threads),
                                "alive_threads": list(alive_threads),
                                "capture_action_terminal": False,
                                "snapshot_after": (
                                    self._scheduler_snapshot_dir_count()
                                ),
                                "staging_after": self._formal_staging_dir_count(),
                            }
                        )
                    return 3, self._last_error
        with self._lock:
            pending_capture = next(
                (
                    record
                    for record in self._windows
                    if record.capture_cancel_pending
                    and not record.capture_action_terminal
                ),
                None,
            )
            active_job_id = self._active_job_id
        # 2：backend active job 在消费线程退出后仍未被收集时，证明终态并
        # 收集；只清已收集的 owner。
        if active_job_id is not None and not consume_alive:
            remaining = max(0.0, stop_deadline - self._monotonic())
            terminal, collected = self._cancel_job_best_effort(
                backend, active_job_id, timeout_sec=remaining
            )
            with self._lock:
                if collected and self._active_job_id == active_job_id:
                    self._active_job_id = None
                if not collected:
                    self._state = SESSION_STOPPING
                    self._stopping = True
                    self._last_error = (
                        "stop_all: backend job terminal/collect was not proven "
                        f"for {active_job_id}; retry stop_all when the worker "
                        "reaches terminal state"
                    )
        # Capture/backend convergence may have released a thread after the
        # first join.  Re-sample without extending the single STOP deadline.
        alive_threads = self._join_threads(deadline=stop_deadline)
        with self._lock:
            capture_unresolved = pending_capture is not None
            if not capture_unresolved:
                self._capture_client = None
            backend_unresolved = self._active_job_id is not None
            if alive_threads or self._backend_cleanup_failure or capture_unresolved or backend_unresolved:
                # join 超时/捕获终态未证明/worker 未收集：绝不宣称 STOPPED，
                # 也不允许在旧线程或旧 job 仍持有状态时启动新 session。
                self._state = SESSION_STOPPING
                self._stopping = True
                if self._backend_cleanup_failure:
                    self._last_error = (
                        "stop_all: backend process cleanup was not proven: "
                        + self._backend_cleanup_failure
                    )
                elif alive_threads:
                    self._last_error = (
                        "stop_all: threads still terminating: "
                        + ",".join(alive_threads)
                        + "; retry stop_all when worker reaches terminal state"
                    )
                elif capture_unresolved:
                    self._last_error = (
                        "stop_all: capture action terminal was not confirmed; "
                        "retry stop_all when the capture action reaches terminal"
                    )
                elif backend_unresolved:
                    # 保留上面设置的具体 collect/cancel 错误。
                    pass
                self._write_journal(
                    {
                        "event": "session_stop_pending",
                        "session_id": self._session_id,
                        **self._stop_audit,
                        "reason": self._last_error,
                        "terminal_state": self._state,
                        "running_after": int(self._running_window is not None),
                        "pending_after": len(self._fifo),
                        "worker_active_after": int(self._active_job_id is not None),
                        "threads_alive_after": len(alive_threads),
                        "alive_threads": list(alive_threads),
                        "capture_action_terminal": not capture_unresolved,
                        "snapshot_after": self._scheduler_snapshot_dir_count(),
                        "staging_after": self._formal_staging_dir_count(),
                    }
                )
                return 3, self._last_error
            if self._pool is not None:
                try:
                    self._pool.clear()
                except Exception as error:
                    self._last_error = f"overlap cache cleanup failed: {error}"
                    self._write_journal({
                        "event": "overlap_cache_cleanup_failed",
                        "session_id": self._session_id,
                        "reason": self._last_error,
                    })
            self._state = SESSION_STOPPED
            self._stopping = True
            self._write_journal(
                {
                    "event": "session_stop",
                    "session_id": self._session_id,
                    **self._stop_audit,
                    "terminal_state": self._state,
                    "running_after": int(self._running_window is not None),
                    "pending_after": len(self._fifo),
                    "worker_active_after": int(self._active_job_id is not None),
                    "threads_alive_after": 0,
                    "capture_action_terminal": True,
                    "snapshot_after": self._scheduler_snapshot_dir_count(),
                    "staging_after": self._formal_staging_dir_count(),
                }
            )
            return 0, f"session {self._session_id} stopped"

    def shutdown(self, timeout_sec: float) -> None:
        if self._state not in (SESSION_IDLE, SESSION_STOPPED):
            self.stop_all(discard_queue=True)  # the process is exiting; nothing would drain
        self._join_threads()

    def get_status(self) -> SessionStatus:
        with self._lock:
            pending = sum(
                1 for record in self._fifo if record.state == WINDOW_PENDING
            )
            running = (
                self._running_window.window_index
                if self._running_window is not None
                else 0
            )
            capturing = next(
                (
                    record
                    for record in reversed(self._windows)
                    if record.state == WINDOW_CAPTURING
                ),
                None,
            )
            captured_frames = capturing.captured_frames if capturing is not None else 0
            capture_target_frames = (
                capturing.capture_goal_frames if capturing is not None else 0
            )
            fifo_bytes = self._fifo_bytes
            fifo_oldest_wait_sec = self._fifo_oldest_wait_locked()
            return SessionStatus(
                state=self._state,
                session_id=self._session_id,
                role=(self._selected_role.role if self._selected_role else ""),
                window_index=self._next_window,
                captured_frames=captured_frames,
                capture_target_frames=capture_target_frames,
                pending_windows=pending,
                running_window=running,
                succeeded_windows=self._succeeded,
                failed_windows=self._failed,
                dropped_windows=0,
                effective=self._runtime,
                last_error=self._last_error,
                min_free_disk_gb=self._config.min_free_disk_gb,
                fifo_bytes=fifo_bytes,
                fifo_oldest_wait_sec=fifo_oldest_wait_sec,
                disk_free_gb=self._disk_free_locked(),
                last_alignment_state=self._last_alignment_state,
                formal_windows=self._formal_windows,
                reconstruction_mode=self._reconstruction_mode,
                alignment_backend=self._config.alignment_backend,
            )

    def _diagnostic_input_frames(self, status, window, kind: str):
        """诊断里的 新增/共享/模型输入 三个数（按窗口实际语义）。"""
        shared = int(window.overlap_frame_count) if window is not None else 0
        model = int((window.effective if window is not None else status.effective).window_frames)
        return {"new": model - shared, "model": model, "shared": shared}[kind]

    def diagnostics_snapshot(self) -> dict[str, Any]:
        status = self.get_status()
        with self._lock:
            waiting = [
                {
                    "window_index": record.window_index,
                    "scene_id": record.scene_id,
                    "attempts": record.attempts,
                    "queued_sec": (
                        self._monotonic() - record.queued_at_monotonic
                        if record.queued_at_monotonic is not None
                        else 0.0
                    ),
                }
                for record in self._fifo
            ]
            running = None
            if self._running_window is not None:
                record = self._running_window
                running = {
                    "window_index": record.window_index,
                    "scene_id": record.scene_id,
                    "attempts": record.attempts,
                }
            # Counts describe the window being collected, independently of GPU lag.
            display_window = next((item for item in reversed(self._windows)
                                   if item.state == WINDOW_CAPTURING), None)
            if display_window is None:
                display_window = self._running_window or (self._windows[-1] if self._windows else None)
            return {
                "state": status.state,
                "session_id": status.session_id,
                "alignment_backend": status.alignment_backend,
                "g4": self._g4_status_locked(),
                # 资源门暂停是可人工恢复的（RESUME 重试同一窗口）；CUDA OOM
                # 不可恢复。attempt_failures 是尝试级失败历史，单调累加。
                "memory_paused": self._memory_guard_paused,
                "oom_latched": self._oom_latched,
                "retry_window": (
                    self._memory_failed_window.window_index
                    if self._memory_failed_window is not None else None
                ),
                "attempt_failures": self._attempt_failures,
                "manual_retries": self._manual_retries,
                # 按当前（采集中或最近）窗口的实际共享帧数给出。
                "new_input_frames": self._diagnostic_input_frames(status, display_window, "new"),
                "model_input_frames": self._diagnostic_input_frames(status, display_window, "model"),
                "shared_input_frames": self._diagnostic_input_frames(status, display_window, "shared"),
                "role": status.role,
                "window_index": status.window_index,
                "captured_frames": status.captured_frames,
                "capture_target_frames": status.capture_target_frames,
                "pending_windows": status.pending_windows,
                "capture_queue_blocked": (
                    status.state == SESSION_RUNNING
                    and status.effective.queue_mode == QUEUE_LOSSLESS
                    and status.pending_windows >= status.effective.queue_capacity
                ),
                "running_window": running,
                "succeeded_windows": status.succeeded_windows,
                "failed_windows": status.failed_windows,
                "dropped_windows": status.dropped_windows,
                # 会话累计：采到的帧里没有位姿的帧数，每帧只算一次（含因此判失败的窗口）。
                "missing_pose_frames": self._missing_pose_frames,
                "effective": {
                    "sample_fps": status.effective.sample_fps,
                    "window_frames": status.effective.window_frames,
                    "queue_mode": status.effective.queue_mode,
                    "queue_capacity": status.effective.queue_capacity,
                    "max_retries": status.effective.max_retries,
                    "keep_snapshot_on_success": (
                        status.effective.keep_snapshot_on_success
                    ),
                    "save_sensor_depth": status.effective.save_sensor_depth,
                    "inference_resolution": (
                        status.effective.inference_resolution
                    ),
                    "worker_timeout_sec": (
                        status.effective.worker_timeout_sec
                    ),
                    "overlap_frames": status.effective.overlap_frames,
                    "alignment_backend": status.alignment_backend,
                    "g4_placement": self._config.g4_placement,
                    "pose_time_offset_ms": self._config.pose_time_offset_ms,
                },
                "fifo": waiting,
                "fifo_bytes": status.fifo_bytes,
                "fifo_oldest_wait_sec": status.fifo_oldest_wait_sec,
                "last_error": status.last_error,
                "min_free_disk_gb": status.min_free_disk_gb,
                "disk_free_gb": status.disk_free_gb,
                "disk_redline": self._disk_low_locked(),
                "worker_disk_blocked": self._worker_disk_blocked,
                "last_alignment_state": (COORDINATE_ODOM_LOCAL_APPROX
                    if self._g4_process is not None and self._g4_process.status().get("published_windows")
                    else status.last_alignment_state),
                "formal_windows": status.formal_windows,
                "reconstruction_mode": status.reconstruction_mode,
                "formal_records_root": str(self._formal_records_root()),
                "worker_wall_time_sec": self._last_worker_wall_time_sec,
                "last_result_wall_time_sec": self._last_worker_wall_time_sec,
                "service_wall_time_sec": self._last_service_wall_time_sec,
                "service_budget_sec": self._last_service_budget_sec,
                "capture_cancel_pending": any(
                    record.capture_cancel_pending
                    and not record.capture_action_terminal
                    for record in self._windows
                ),
                "capture_action_terminal": not any(
                    record.capture_cancel_pending
                    and not record.capture_action_terminal
                    for record in self._windows
                ),
            }

    # ------------------------------------------------------------------
    def _resolve_role(self, role_name: str) -> CameraRole | None:
        if not isinstance(role_name, str) or not role_name:
            return None
        return self._config.role_by_name(role_name)

    def _scene_id(self, window_index: int) -> str:
        return f"cs-{self._session_id}-w{window_index}"

    def _disk_low_locked(self) -> bool:
        free = self._disk_free_locked()
        if free != free:
            # 读不到剩余空间时 fail-closed：暂停新采集/新 worker 并告警。
            return True
        return free < float(self._config.min_free_disk_gb)

    def _disk_free_locked(self) -> float:
        try:
            free = float(self._disk_free_gb())
        except BaseException:
            return float("nan")
        if free != free or free == float("-inf"):
            return float("nan")
        # +inf 表示显式无界（开发测试缺省）；负值视为 0。
        if free == float("inf"):
            return float("inf")
        return max(0.0, free)

    @staticmethod
    def _snapshot_dir_size(path: Path) -> int:
        """计算调度器拥有 snapshot 目录的磁盘占用（不跟随符号链接）。"""
        try:
            if not path.is_dir():
                return 0
        except OSError:
            return 0
        total = 0
        try:
            for child in path.rglob("*"):
                try:
                    if child.is_symlink() or not child.is_file():
                        continue
                    total += child.stat().st_size
                except OSError:
                    continue
        except OSError:
            pass
        return total

    def _fifo_oldest_wait_locked(self) -> float | None:
        """计算 PENDING 窗口的最老等待秒数（O(FIFO 长度)，不碰文件系统）。"""
        oldest = None
        now = self._monotonic()
        for record in self._fifo:
            if record.queued_at_monotonic is not None:
                age = max(0.0, now - record.queued_at_monotonic)
                oldest = age if oldest is None else max(oldest, age)
        return oldest

    def _snapshot_bytes_from_uri(self, snapshot_uri: str) -> int:
        """把 file:// snapshot URI 换算为调度器可扫描的本地路径。"""
        if not snapshot_uri or not snapshot_uri.startswith("file://"):
            return 0
        try:
            path = Path(
                unquote_to_bytes(snapshot_uri[len("file://"):]).decode(
                    "utf-8", "strict"
                )
            )
        except BaseException:
            return 0
        return self._snapshot_dir_size(path)

    def _make_thread(self, name: str, target) -> threading.Thread:
        thread = self._thread_factory(target=target)
        if not isinstance(thread, threading.Thread):
            raise TypeError("thread_factory must return a Thread")
        thread.name = name
        thread.daemon = True
        return thread

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        try:
            while True:
                with self._lock:
                    if self._stopping or self._fatal_oom or self._capture_stopped:
                        break
                    if self._state in (SESSION_RUNNING, SESSION_DISK_PAUSED):
                        disk_low = self._disk_low_locked()
                        if self._state == SESSION_RUNNING and disk_low:
                            self._state = SESSION_DISK_PAUSED
                            self._last_error = _DISK_REDLINE_ERROR
                        elif self._state == SESSION_DISK_PAUSED and not disk_low:
                            self._state = SESSION_RUNNING
                            if self._last_error == _DISK_REDLINE_ERROR:
                                self._last_error = ""
                    if self._state != SESSION_RUNNING or (
                        self._runtime.queue_mode == QUEUE_LOSSLESS
                        and len(self._fifo) >= self._runtime.queue_capacity
                    ):
                        self._wake_capture.clear()
                        proceed = False
                    else:
                        proceed = True
                    if proceed:
                        role = self._selected_role
                        effective = self._runtime
                        window_index = self._next_window
                        record = WindowRecord(
                            window_index=window_index,
                            scene_id=self._scene_id(window_index),
                            role=role,
                            effective=effective,
                            capture_generation=self._capture_generation,
                        )
                        self._windows.append(record)
                        self._capture_client = None
                if not proceed:
                    self._wake_capture.wait(0.5)
                    continue

                # 同一窗口最多 (max_retries + 1) 次 capture 尝试。
                while True:
                    client = self._capture_client_factory(role)
                    with self._lock:
                        self._capture_client = client
                    error = self._attempt_capture_window(
                        record, client, role, effective
                    )
                    if error is None:
                        consecutive_failures = 0
                        with self._lock:
                            self._next_window = window_index + 1
                        break
                    with self._lock:
                        if self._stopping or self._fatal_oom:
                            break
                        if self._capture_stopped:
                            # STOP while draining: the unfinished window is cancelled, not failed.
                            if record.snapshot_uri and not record.done.is_set():
                                self._remove_snapshot(record)
                            self._window_terminal(record, WINDOW_CANCELLED, "stop_all: capture stopped")
                            break
                        record.capture_attempts += 1
                        self._attempt_failures += 1
                        retry = record.capture_attempts <= effective.max_retries
                    if retry:
                        continue
                    self._window_terminal(record, WINDOW_FAILED, error)
                    with self._lock:
                        self._next_window = window_index + 1
                        self._failed += 1
                        self._last_error = error
                        self._write_journal(
                            {
                                "event": "window_capture_failed",
                                "session_id": self._session_id,
                                "window_index": record.window_index,
                                "scene_id": record.scene_id,
                                "attempts": record.capture_attempts,
                                "config_digest": record.effective.digest(),
                                "reason": error,
                            }
                        )
                    # A source gap may invalidate one window. Repeated
                    # failures indicate a source/configuration fault.
                    consecutive_failures += 1
                    with self._lock:
                        if consecutive_failures >= 3 and not self._stopping and not self._fatal_oom:
                            self._state = SESSION_PAUSED
                            self._capture_client = None
                            self._write_journal({
                                "event": "capture_failure_paused",
                                "session_id": self._session_id,
                                "reason": error,
                                "recovery": "fix_source_then_resume",
                            })
                            consecutive_failures = 0
                    self._wake_capture.clear()
                    self._wake_capture.wait(0.5)
                    break
                with self._lock:
                    retain_client = (
                        self._stopping
                        and record.capture_cancel_pending
                        and not record.capture_action_terminal
                    )
                    if not retain_client:
                        self._capture_client = None
        finally:
            self._capture_done.set()

    def _attempt_capture_window(
        self,
        record: WindowRecord,
        client: ContinuousCaptureClient,
        role: CameraRole,
        effective: EffectiveConfig,
    ) -> str | None:
        """执行一次 Capture Goal；成功入队返回 None，否则返回错误文本。"""
        if not client.wait_for_server(self._config.action_connect_timeout_sec):
            return (
                "capture action server did not appear within "
                f"{self._config.action_connect_timeout_sec} s"
            )
        # 2a：pause/stop 可能落在 client 创建之后、send_goal 之前。此时绝
        # 不能启动新 Capture goal；等到 RUNNING（或 stopping/fatal）再继续。
        while True:
            with self._lock:
                if self._stopping or self._capture_stopped:
                    return "stopping"
                if self._fatal_oom:
                    return "fatal oom; capture goal suppressed"
                if self._state == SESSION_RUNNING:
                    break
            try:
                client.pause()
            except BaseException:
                pass
            self._wake_capture.wait(0.5)
        try:
            capture_goal_frames = self._capture_goal_frames(record)
        except Exception as error:
            return f"capture window planning failed: {type(error).__name__}: {error}"
        max_duration = (
            float(capture_goal_frames)
            * record.role.capture_config.max_frame_interval_sec(max(0.0001, float(effective.sample_fps)))
            + float(self._config.capture_wait_margin_sec)
        )
        with self._goal_dispatch_lock:
            with self._lock:
                if self._stopping or self._capture_stopped:
                    return "stopping"
                if self._fatal_oom:
                    return "fatal oom; capture goal suppressed"
            try:
                client.send_goal(
                    record.scene_id,
                    float(effective.sample_fps),
                    int(capture_goal_frames),
                    max_duration,
                    self._make_capture_feedback(record),
                    save_sensor_depth=effective.save_sensor_depth,
                    set_save_sensor_depth=True,
                )
            except BaseException as error:
                return f"capture goal could not be sent: {type(error).__name__}: {error}"
            with self._lock:
                record.capture_action_terminal = False

        # 2a：pause 也可能落在 send_goal 返回之后；立刻补发一次 pause，
        # 后续 wait loop 会冻结 scheduler deadline 直到 resume。
        with self._lock:
            if self._stopping or self._fatal_oom or self._capture_stopped:
                client.cancel_goal()
            elif self._state != SESSION_RUNNING:
                try:
                    client.pause()
                except BaseException:
                    pass

        deadline = self._monotonic() + max_duration + self._config.capture_wait_margin_sec
        allowance = max_duration + self._config.capture_wait_margin_sec
        previous_source_stale = False
        capture_cancel_deadline: float | None = None
        while True:
            with self._lock:
                stopping = self._stopping
                fatal_oom = self._fatal_oom
                capture_stopped = self._capture_stopped
                source_stale = record.source_stale
                paused = self._state == SESSION_PAUSED
                if stopping and capture_cancel_deadline is None:
                    record.capture_cancel_pending = True
                    capture_cancel_deadline = (
                        self._monotonic() + float(self._config.stop_timeout_sec)
                    )
            # SOURCE_STALE → SOURCE_RECOVERED：底层 Capture 的 deadline 在
            # stale 期间被冻结，恢复后应重新计时；否则恢复瞬间就可能被
            # 调度器 watchdog 误杀。
            if previous_source_stale and not source_stale:
                deadline = self._monotonic() + allowance
            previous_source_stale = source_stale
            # 2c：SESSION_PAUSED 期间 scheduler watchdog deadline 冻结，
            # 不得因暂停时长取消底层 Capture Action。
            if paused:
                deadline = self._monotonic() + allowance
            if stopping or fatal_oom or capture_stopped:
                client.cancel_goal()
            # stopping 时使用更短的轮询步长，让 bounded convergence
            # deadline 能及时被检查到，避免 stop_all join 已超时。
            wait_step = 0.1 if stopping else 0.5
            if client.wait_for_result(wait_step):
                with self._lock:
                    record.capture_action_terminal = True
                    record.capture_cancel_pending = False
                break
            if (
                stopping
                and capture_cancel_deadline is not None
                and self._monotonic() >= capture_cancel_deadline
            ):
                with self._lock:
                    self._stopping = True
                    self._state = SESSION_STOPPING
                    self._last_error = (
                        "stop_all: capture action terminal was not confirmed "
                        f"within {float(self._config.stop_timeout_sec)} s; "
                        "retry stop_all when the capture action reaches terminal"
                    )
                return self._last_error
            if (not stopping and not fatal_oom and not capture_stopped and not source_stale and not paused
                    and self._monotonic() >= deadline):
                client.cancel_goal()
                while not client.wait_for_result(0.5):
                    with self._lock:
                        if self._stopping:
                            break
                return (
                    "capture watchdog timeout; underlying action cancel requested"
                )
        outcome = client.outcome()
        record.captured_frames = outcome.frame_count
        record.new_frame_count = outcome.frame_count
        record.snapshot_uri = outcome.snapshot_uri
        record.manifest_sha256 = outcome.manifest_sha256
        with self._lock:
            if self._stopping or self._fatal_oom:
                if record.snapshot_uri:
                    self._remove_snapshot(record)
                if self._fatal_oom and not self._stopping:
                    self._window_terminal(record, WINDOW_FAILED, "fatal oom")
                    return "fatal oom; in-flight capture cancelled"
                self._window_terminal(record, WINDOW_CANCELLED, "stop_all")
                return "stopping"
        if outcome.action_state != ACTION_SUCCEEDED or outcome.error_code != 0:
            return (
                f"capture action ended as {outcome.action_state}: "
                f"code={outcome.error_code} message={outcome.message}"
            )
        identity_error = _capture_identity_error(record, outcome, role.capture_config)
        if identity_error is not None:
            return f"capture identity check failed: {identity_error}"
        if not record.snapshot_uri or not _valid_sha256(record.manifest_sha256):
            return (
                "capture succeeded without valid snapshot metadata "
                "(snapshot_uri or manifest_sha256)"
            )
        try:
            self._prepare_overlap_capture(record)
        except BaseException as error:
            return (
                "overlap window preparation failed: "
                f"{type(error).__name__}: {error}"
            )
        missing, total = record.missing_pose_frames, len(record.window_frame_ids)
        if missing / total > MAX_MISSING_POSE_FRACTION:
            # 重建救不了缺位姿：不提交、不重试；采集继续，共享帧照常复用。
            error = (
                f"missing poses: {missing} of {total} window frames have no pose "
                f"(limit {MAX_MISSING_POSE_FRACTION:.0%})"
            )
            with self._lock:
                self._missing_pose_frames += record.new_missing_pose_frames
                self._failed += 1
                self._window_terminal(record, WINDOW_FAILED, error)
                self._last_error = error
                self._write_journal(
                    {
                        "event": "window_failed",
                        "session_id": self._session_id,
                        "window_index": record.window_index,
                        "scene_id": record.scene_id,
                        "attempts": 0,
                        "config_digest": record.effective.digest(),
                        "reason": error,
                        "missing_pose_frames": missing,
                        "window_frames": total,
                    }
                )
                self._capture_client = None
            self._remove_snapshot(record)
            return None
        with self._lock:
            self._missing_pose_frames += record.new_missing_pose_frames
            record.queued_at_monotonic = self._monotonic()
            record.state = WINDOW_PENDING
            record.snapshot_bytes = self._snapshot_bytes_from_uri(
                record.snapshot_uri
            )
            self._fifo.append(record)
            self._fifo_bytes += record.snapshot_bytes
            self._write_journal(
                {
                    "event": "window_pending",
                    "session_id": self._session_id,
                    "window_index": record.window_index,
                    "scene_id": record.scene_id,
                    "snapshot_uri": record.snapshot_uri,
                    "manifest_sha256": record.manifest_sha256,
                    "new_frame_count": record.new_frame_count,
                    "overlap_frame_count": record.overlap_frame_count,
                    "missing_pose_frames": missing,
                    "config_digest": record.effective.digest(),
                }
            )
            self._wake_consume.set()
            self._capture_client = None
        return None

    def _consume_loop(self) -> None:
        backend = self._backend
        try:
            while True:
                with self._lock:
                    if self._stopping or self._fatal_oom:
                        break
                    if (
                        self._running_window is None
                        and self._active_job_id is None
                        and self._fifo
                    ):
                        if self._disk_low_locked():
                            # 磁盘红线：不启动新 CUDA worker，保持 PENDING
                            # 队列原样；空间恢复后自动继续。
                            if not self._worker_disk_blocked:
                                self._worker_disk_blocked = True
                                self._last_error = (
                                    "disk redline reached; worker startup delayed"
                                )
                            record = None
                        else:
                            self._worker_disk_blocked = False
                            if (
                                self._last_error
                                == "disk redline reached; worker startup delayed"
                            ):
                                self._last_error = ""
                            record = self._fifo.pop(0)
                            self._fifo_bytes -= record.snapshot_bytes
                            record.state = WINDOW_RUNNING
                            self._running_window = record
                            self._wake_capture.set()
                    else:
                        record = None
                if record is None:
                    self._wake_consume.wait(0.5)
                    self._wake_consume.clear()
                    continue
                service_started = self._monotonic()
                try:
                    self._consume_window(record, backend)
                except BaseException as error:
                    self._cancel_active_job_best_effort(backend)
                    self._handle_reconstruct_failure(
                        record,
                        "consume window raised unexpectedly: "
                        f"{type(error).__name__}: {error}",
                    )
                with self._lock:
                    self._last_service_wall_time_sec = max(
                        0.0, self._monotonic() - service_started
                    )
                    self._last_service_budget_sec = (
                        record.effective.window_frames - record.effective.overlap_frames
                    ) / record.effective.sample_fps
                    self._write_journal({
                        "event": "window_service_timing",
                        "session_id": self._session_id,
                        "window_index": record.window_index,
                        "attempt": record.reconstruct_attempts,
                        "state": record.state,
                        "config_digest": record.effective.digest(),
                        "service_wall_time_sec": self._last_service_wall_time_sec,
                        "service_budget_sec": self._last_service_budget_sec,
                    })
                    if (
                        self._stopping
                        and record.state != WINDOW_SUCCEEDED
                        and record.snapshot_uri
                    ):
                        # stop_all 期间 worker 已取消；消费线程负责清掉自己
                        # 正在处理的输入 snapshot，避免 join 竞态孤儿。
                        self._remove_snapshot(record)
                    self._running_window = None
                    self._wake_capture.set()
        finally:
            self._consume_done.set()

    def _cancel_job_best_effort(
        self,
        backend: ReconstructionBackend,
        job_id: str | None,
        timeout_sec: float | None = None,
    ) -> tuple[bool, bool]:
        """Cancel an exceptional-path job without collecting it while live.

        Returns ``(terminal, collected)``. ``timeout_sec`` bounds the wait
        when a caller (e.g. stop_all) must not block forever.
        """
        if not job_id:
            return True, False
        deadline = (
            self._monotonic() + float(timeout_sec)
            if timeout_sec is not None
            else None
        )
        while True:
            try:
                backend.cancel(job_id)
            except BaseException:
                pass
            else:
                with self._lock:
                    self._update_stop_audit_backend_locked(
                        job_id, cancel_requested=True
                    )
            try:
                status = backend.status(job_id)
            except BaseException:
                remaining = (
                    None if deadline is None else deadline - self._monotonic()
                )
                if remaining is not None and remaining <= 0:
                    return False, False
                time.sleep(0.2 if remaining is None else min(0.2, remaining))
                continue
            if status.state in _JOB_TERMINAL:
                with self._lock:
                    self._update_stop_audit_backend_locked(
                        job_id,
                        terminal_state=status.state,
                        terminal_message=status.message,
                    )
                try:
                    backend.collect_result(job_id)
                except BaseException:
                    return True, False
                return True, True
            if deadline is not None and self._monotonic() >= deadline:
                return False, False
            time.sleep(0.2)

    def _cancel_active_job_best_effort(self, backend: ReconstructionBackend) -> None:
        """Cancel the active job; clear ownership only after collect succeeds."""
        with self._lock:
            job_id = self._active_job_id
        terminal, collected = self._cancel_job_best_effort(backend, job_id)
        with self._lock:
            if collected and self._active_job_id == job_id:
                self._active_job_id = None
            if collected:
                self._update_stop_audit_backend_locked(
                    job_id, terminal_collected=True
                )

    def _backend_profile(self) -> dict[str, Any]:
        """交给后台地图后端的显式配置（写入 backend_profile.json）。"""
        from g4_backend import FRAME_POSE_METHOD_ID

        profile = {
            "method_id": FRAME_POSE_METHOD_ID,
            "window_frames": int(self._config.initial_window_frames),
            "overlap_frames": int(self._config.initial_overlap_frames),
            "placement": "frame_pose",
            "frame_pose": {"pose_time_offset_ms": float(self._config.pose_time_offset_ms)},
            "camera_intrinsics": (None if self._config.camera_intrinsics is None
                                  else dict(self._config.camera_intrinsics)),
        }
        if self._config.frame_pose_fusion:
            profile["frame_pose_fusion"] = True
        if self._config.pose_time_offset_stage == "capture":
            profile["frame_pose"]["pose_time_offset_stage"] = "capture"
        if self._config.g4_cross_window_weight > 0:
            profile["frame_pose"]["cross_window_weight"] = float(self._config.g4_cross_window_weight)
        if self._config.g4_frame_scales:
            profile["frame_pose"]["frame_scales"] = True
        return profile

    # ------------------------------------------------------------------
    # overlap window helpers (shared frames cached in FramePool)
    # ------------------------------------------------------------------
    def _capture_goal_frames(self, record: WindowRecord) -> int:
        """Plan only the new frames; the first window still captures all N."""
        pool = self._frame_pool()
        pool.load()
        record.overlap_frame_count = min(
            record.effective.overlap_frames, len(pool.entries)
        )
        record.capture_goal_frames = (
            record.effective.window_frames - record.overlap_frame_count
        )
        return record.capture_goal_frames

    def _prepare_overlap_capture(self, record: WindowRecord) -> None:
        """Build one N-frame snapshot from the previous tail plus new capture."""
        role = record.role
        snapshot = validate_snapshot(
            record.snapshot_uri,
            record.manifest_sha256,
            (role.capture_config.snapshot_root,),
            role.capture_config.snapshot_limits,
        )
        pool = self._frame_pool()
        pool.load()
        previous = list(pool.entries)
        if previous:
            first_stamp = snapshot.frame_stamps[0]
            if (first_stamp.sec, first_stamp.nsec) <= previous[-1].stamp_key():
                # Capture validates each individual window. A restart between
                # actions must also invalidate cached overlap across windows.
                pool.clear()
                previous = []
                with self._lock:
                    self._capture_generation += 1
                    record.capture_generation = self._capture_generation
                    self._write_journal({
                        "event": "capture_time_reset",
                        "session_id": self._session_id,
                        "window_index": record.window_index,
                        "capture_generation": record.capture_generation,
                    })
                if record.overlap_frame_count:
                    self._remove_snapshot(record)
                    raise FramePoolError(
                        "capture time reset: overlap cache cleared; next capture requires a full window"
                    )
        record.capture_generation = self._capture_generation
        synthetic_uri = ""
        original_uri = record.snapshot_uri
        original_digest = record.manifest_sha256
        try:
            new_ids = pool.import_window(snapshot, record.window_index)
            by_index = {entry.global_index: entry for entry in pool.entries}
            selected = previous[-record.overlap_frame_count :] if record.overlap_frame_count else []
            selected += [by_index[index] for index in new_ids]
            if len(selected) != record.effective.window_frames:
                raise FramePoolError(
                    "overlap window did not produce exactly window_frames inputs"
                )
            if record.overlap_frame_count:
                destination = (
                    Path(role.capture_config.snapshot_root)
                    / f"cs-{self._session_id}-w{record.window_index}-{secrets.token_hex(8)}"
                )
                synthetic_uri, digest = pool.build_synthetic_snapshot(
                    record.window_index,
                    selected,
                    destination,
                    record.scene_id,
                    camera_name=role.capture_config.camera_name,
                    image_topic=role.capture_config.image_topic,
                    camera_role=role.role,
                    camera_frame_id=role.capture_config.camera_frame_id,
                    preserve_order=True,
                    limits=role.capture_config.snapshot_limits,
                )
                # The builder enforces limits and hashes the bytes it just wrote.
                # Worker admission validates the immutable snapshot at the next
                # process boundary; do not re-read every file here as well.
                record.snapshot_uri = synthetic_uri
                record.manifest_sha256 = digest
            pool.retain(selected)
        except BaseException:
            try:
                pool.retain(previous)
            except BaseException:
                self._pool = None
            if synthetic_uri:
                self._remove_snapshot(
                    SimpleNamespace(snapshot_uri=synthetic_uri, role=role)
                )
            record.snapshot_uri = original_uri
            record.manifest_sha256 = original_digest
            raise
        record.window_frame_ids = [entry.global_index for entry in selected]
        record.missing_pose_frames = sum(1 for entry in selected if not entry.pose_valid)
        record.new_missing_pose_frames = sum(
            1 for index in new_ids if not by_index[index].pose_valid
        )
        record.captured_frame_ids = list(record.window_frame_ids)
        record.captured_frames = len(selected)
        if synthetic_uri:
            cleanup_error = self._remove_snapshot(
                SimpleNamespace(snapshot_uri=original_uri, role=role)
            )
            if cleanup_error is not None:
                self._write_journal(
                    {
                        "event": "overlap_source_cleanup_failed",
                        "session_id": self._session_id,
                        "window_index": record.window_index,
                        "reason": cleanup_error,
                    }
                )

    def _frame_pool(self) -> FramePool:
        if self._pool is None:
            root = (
                Path(self._config.result_root).parent
                / "frame_pool"
                / self._session_id
            )
            self._pool = FramePool(root, FramePoolConfig())
            self._pool.load()
        return self._pool

    def _consume_window(
        self, record: WindowRecord, backend: ReconstructionBackend
    ) -> None:
        with self._lock:
            record.reconstruct_attempts += 1
        with self._goal_dispatch_lock:
            with self._lock:
                if self._stopping or self._fatal_oom:
                    self._window_terminal(record, WINDOW_CANCELLED, "stop_all")
                    return
            try:
                job_id = backend.submit(
                    record.snapshot_uri,
                    record.manifest_sha256,
                    record.scene_id,
                    inference_resolution=record.effective.inference_resolution,
                    worker_timeout_sec=record.effective.worker_timeout_sec,
                )
            except BaseException as error:
                self._handle_reconstruct_failure(
                    record,
                    f"backend submit failed: {type(error).__name__}: {error}",
                )
                return
            with self._lock:
                self._active_job_id = job_id
                self._active_job_cancel_requested = False
                self._update_stop_audit_backend_locked(job_id)
        deadline = self._monotonic() + self._config.reconstruct_wait_timeout_sec
        watchdog_fired = False
        status_failure = ""
        while True:
            with self._lock:
                stopping = self._stopping
            try:
                status = backend.status(job_id)
            except BaseException as error:
                if not status_failure:
                    status_failure = (
                        f"backend status failed: {type(error).__name__}: {error}"
                    )
                # actionlib cancellation is asynchronous. Keep ownership and
                # polling until the lower action reports a terminal state;
                # collecting here would pop a still-live client.
                try:
                    backend.cancel(job_id)
                except BaseException as cancel_error:
                    with self._lock:
                        self._last_error = (
                            f"window {record.window_index} backend cancel failed "
                            "after status error: "
                            f"{type(cancel_error).__name__}: {cancel_error}"
                        )
                else:
                    with self._lock:
                        self._update_stop_audit_backend_locked(
                            job_id, cancel_requested=True
                        )
                now = self._monotonic()
                if stopping:
                    with self._lock:
                        self._window_terminal(
                            record, WINDOW_CANCELLED, "stop_all"
                        )
                    return
                if now >= deadline:
                    # The worker may still exist, so retain active-job
                    # ownership for stop_all to prove terminal+collect.  The
                    # scheduler must nevertheless fail closed instead of
                    # leaving an immortal consumer and a false RUNNING state.
                    error = (
                        f"{status_failure}; status unavailable through "
                        "reconstruction deadline"
                    )
                    with self._lock:
                        self._stopping = True
                        self._state = SESSION_STOPPING
                        self._last_error = error
                        self._wake_capture.set()
                        self._wake_consume.set()
                    self._handle_reconstruct_failure(record, error)
                    return
                time.sleep(min(0.2, max(0.0, deadline - now)))
                continue
            if status.state in _JOB_TERMINAL:
                break
            if stopping:
                try:
                    backend.cancel(job_id)
                except BaseException as error:
                    with self._lock:
                        self._last_error = (
                            f"window {record.window_index} backend cancel failed "
                            f"during stop: {type(error).__name__}: {error}"
                        )
                else:
                    with self._lock:
                        self._update_stop_audit_backend_locked(
                            job_id, cancel_requested=True
                        )
            if not stopping and not watchdog_fired and self._monotonic() >= deadline:
                watchdog_fired = True
                try:
                    backend.cancel(job_id)
                except BaseException as error:
                    with self._lock:
                        self._last_error = (
                            f"window {record.window_index} backend cancel failed "
                            f"during watchdog: {type(error).__name__}: {error}"
                        )
                else:
                    with self._lock:
                        self._update_stop_audit_backend_locked(
                            job_id, cancel_requested=True
                        )
                        self._last_error = (
                            f"window {record.window_index} reconstruction watchdog fired; "
                            "waiting for lower action terminal state"
                        )
            time.sleep(0.2)
        with self._lock:
            self._update_stop_audit_backend_locked(
                job_id,
                terminal_state=status.state,
                terminal_message=status.message,
            )
            cleanup_failure = (
                status.state == JOB_FAILED
                and "cleanup_failed" in str(status.message).lower()
            )
        try:
            outcome = backend.collect_result(job_id)
        except BaseException as error:
            with self._lock:
                self._last_error = (
                    f"window {record.window_index} backend collect_result "
                    f"failed for job {job_id}: "
                    f"{type(error).__name__}: {error}"
                )
                if self._stop_audit is not None:
                    self._stop_audit["backend_collect_error"] = self._last_error
            # Best-effort cancel/collect keeps the backend slot from staying
            # owned by an uncollectable job; _active_job_id is only cleared
            # after collect succeeds, so stop_all can observe the retained job.
            self._cancel_active_job_best_effort(backend)
            if cleanup_failure:
                self._handle_reconstruct_failure(
                    record, f"backend process cleanup failed: {status.message}"
                )
                return
            with self._lock:
                if self._stopping or self._fatal_oom:
                    self._window_terminal(record, WINDOW_CANCELLED, "stop_all")
                    return
            self._handle_reconstruct_failure(
                record,
                f"backend collect_result failed: {type(error).__name__}: {error}",
            )
            return
        with self._lock:
            self._active_job_id = None
            self._last_worker_wall_time_sec = (
                float(outcome.wall_time_sec)
                if _finite(outcome.wall_time_sec)
                else None
            )
            self._update_stop_audit_backend_locked(
                job_id, terminal_collected=True
            )
        if cleanup_failure:
            self._handle_reconstruct_failure(
                record, f"backend process cleanup failed: {status.message}"
            )
            return
        with self._lock:
            if self._stopping or self._fatal_oom:
                self._window_terminal(record, WINDOW_CANCELLED, "stop_all")
                return
        if status_failure:
            self._handle_reconstruct_failure(record, status_failure)
            return
        if outcome.action_state != ACTION_SUCCEEDED or outcome.error_code != 0:
            self._handle_reconstruct_failure(
                record,
                f"reconstruct action ended as {outcome.action_state}: "
                f"code={outcome.error_code} message={outcome.message}",
            )
            return
        if (
            not outcome.result_manifest_uri
            or not _valid_sha256(outcome.result_manifest_sha256)
            or isinstance(outcome.frame_count, bool)
            or outcome.frame_count <= 0
        ):
            self._handle_reconstruct_failure(
                record,
                "reconstruct action succeeded without valid result metadata "
                "(result_manifest_uri, result_manifest_sha256, frame_count)",
            )
            return
        identity_error = _reconstruct_identity_error(record, outcome)
        if identity_error is not None:
            self._handle_reconstruct_failure(
                record, f"reconstruct identity check failed: {identity_error}"
            )
            return
        record.worker_result_manifest_uri = outcome.result_manifest_uri
        record.worker_result_manifest_sha256 = outcome.result_manifest_sha256
        record.model_frame_ids = self._window_model_frame_ids(record, outcome)
        alignment = self._align_window()
        formal_error = self._formalize_window(record, outcome, alignment)
        if formal_error == _FORMAL_SKIPPED_STOPPING:
            with self._lock:
                self._window_terminal(record, WINDOW_CANCELLED, "stop_all")
            return
        if formal_error is not None:
            self._handle_reconstruct_failure(record, formal_error)
            return
        # Only committed, verified inputs can enter G4. The process scans
        # complete records and coalesces arrivals while a solve is running.
        # A map backend that cannot start, or that died, does not stop capture: the
        # recorded windows can be placed again after landing. It must be loud instead
        # (diagnostics ERROR, journal), because the window itself still succeeds.
        with self._lock:
            if not self._stopping:
                try:
                    if self._g4_process is not None and self._g4_process.process.poll() is not None:
                        exited = self._g4_process.status()
                        self._write_journal({
                            "event": "g4_backend_exited",
                            "session_id": self._session_id,
                            "window_index": record.window_index,
                            "exit_code": exited.get("exit_code"),
                            "error": exited.get("error", ""),
                        })
                        self._g4_process.close()
                        self._g4_process = None
                    if self._g4_process is None:
                        self._g4_process = G4BackendProcess(self._formal_records_root(), self._session_id,
                                                            profile=self._backend_profile())
                        self._g4_start_error = ""
                except Exception as error:
                    self._g4_start_error = f"G4 background start failed: {type(error).__name__}: {error}"
                    self._last_error = self._g4_start_error
                    self._write_journal({
                        "event": "g4_backend_start_failed",
                        "session_id": self._session_id,
                        "window_index": record.window_index,
                        "error": self._g4_start_error,
                    })
        with self._lock:
            record.alignment = alignment
            self._last_alignment_state = (
                COORDINATE_MODEL_LOCAL_UNALIGNED
                if alignment.get("quality_passed") is False
                else alignment["coordinate_state"]
            )
            self._succeeded += 1
            self._formal_windows += 1
            self._window_terminal(record, WINDOW_SUCCEEDED, "")
            self._write_journal(
                {
                    "event": "window_succeeded",
                    "session_id": self._session_id,
                    "window_index": record.window_index,
                    "scene_id": record.scene_id,
                    "result_manifest_uri": outcome.result_manifest_uri,
                    "result_manifest_sha256": outcome.result_manifest_sha256,
                    "config_digest": record.effective.digest(),
                    "formal_result_uri": record.formal_result_uri,
                    "formal_provenance_sha256": record.formal_provenance_sha256,
                    "missing_pose_frames": record.missing_pose_frames,
                }
            )
            self._write_journal(
                {
                    "event": "window_alignment",
                    "session_id": self._session_id,
                    "window_index": record.window_index,
                    "scene_id": record.scene_id,
                    **{
                        key: value
                        for key, value in alignment.items()
                        if key != "global_extrinsic"
                    },
                }
            )

    def _g4_status_locked(self) -> dict[str, Any]:
        """Map backend status for diagnostics; a failed start reports as an error, not as "waiting"."""
        if self._g4_process is not None:
            return self._g4_process.status()
        if self._g4_start_error:
            return {"state": "error", "error": self._g4_start_error}
        return {}

    def _align_window(self) -> dict[str, Any]:
        """G4 places windows in the background; each window waits for a map revision."""
        return {
            "coordinate_state": COORDINATE_MODEL_LOCAL_UNALIGNED,
            "map_id": "",
            "scale": None,
            "rotation": None,
            "translation": None,
            "residual_rms": None,
            "correspondence_count": 0,
            "reference_frame_id": "",
            "unreliable": True,
            "reason": "waiting_for_G4_map_revision",
            "method": BACKEND_G4,
            "pending": True,
            "quality_passed": False,
            "geometry_accepted": False,
        }

    def _window_model_frame_ids(self, record: WindowRecord, outcome) -> list[int]:
        if not record.window_frame_ids:
            return []
        selected = self._selected_frame_indices(outcome)
        if selected is None:
            return list(record.window_frame_ids)
        if any(index < 0 or index >= len(record.window_frame_ids) for index in selected):
            return []
        return [record.window_frame_ids[index] for index in selected]

    def _formal_records_root(self) -> Path:
        return Path(self._config.result_root) / "sessions"

    def _ensure_formal_session_dir(self, session_id: str) -> Path:
        records_root = self._formal_records_root()
        records_root.mkdir(parents=True, exist_ok=True)
        session_dir = records_root / session_id
        try:
            session_dir.mkdir()
        except FileExistsError as error:
            if not session_dir.is_dir() or session_dir.is_symlink():
                raise FormalResultError(
                    f"formal session dir exists but is unsafe: {session_dir}"
                ) from error
        return session_dir

    def _depth_metrics_window(
        self, record: WindowRecord, outcome
    ) -> tuple[dict | None, str | None]:
        """计算首帧传感器深度对照指标。

        返回 ``(depth_metrics, error_reason)``：
        - 计算成功：``(metrics, None)``；
        - 深度开关关闭或快照没有首帧传感器深度：``(None, None)``；
        - 深度开关打开但计算失败：``({"status": "unavailable", ...}, reason)``，
          让正式记录仍落盘 depth_metrics.json 并暴露失败原因。
        """
        if not bool(record.effective.save_sensor_depth):
            return None, None
        try:
            snapshot = validate_snapshot(
                record.snapshot_uri,
                record.manifest_sha256,
                (record.role.capture_config.snapshot_root,),
                record.role.capture_config.snapshot_limits,
            )
        except BaseException as error:
            reason = f"snapshot validation failed for depth metrics: {type(error).__name__}: {error}"
            return {"status": "unavailable", "reason": reason}, reason
        predictions_path = self._predictions_path(outcome)
        if predictions_path is None:
            reason = "result manifest uri has no local predictions.npz"
            return {"status": "unavailable", "reason": reason}, reason
        metrics, compute_reason = compute_first_frame_depth_metrics(
            snapshot, predictions_path
        )
        if metrics is not None:
            return metrics, None
        if compute_reason in _NO_SENSOR_DEPTH_REASONS:
            return None, None
        reason = compute_reason or "depth metrics unavailable"
        return {"status": "unavailable", "reason": reason}, reason

    @staticmethod
    def _read_snapshot_manifest_bytes(snapshot) -> bytes:
        """Read the validated snapshot manifest bytes from its canonical root."""
        root = getattr(snapshot, "canonical_root", None)
        if root is None:
            raise FormalResultError(
                "validated snapshot does not expose a canonical root"
            )
        path = Path(root) / "files_sha256.csv"
        try:
            info = os.lstat(str(path))
        except OSError as error:
            raise FormalResultError(
                "snapshot files_sha256.csv cannot be inspected"
            ) from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise FormalResultError(
                "snapshot files_sha256.csv is not a trusted regular file"
            )
        if info.st_size > _MAX_SNAPSHOT_MANIFEST_BYTES:
            raise FormalResultError(
                "snapshot files_sha256.csv exceeds size bound"
            )
        try:
            with open(str(path), "rb") as stream:
                data = stream.read(_MAX_SNAPSHOT_MANIFEST_BYTES + 1)
        except OSError as error:
            raise FormalResultError(
                "snapshot files_sha256.csv cannot be read"
            ) from error
        if len(data) > _MAX_SNAPSHOT_MANIFEST_BYTES:
            raise FormalResultError(
                "snapshot files_sha256.csv exceeds size bound"
            )
        return data

    def _formalize_window(
        self,
        record: WindowRecord,
        outcome: ReconstructActionOutcome,
        alignment: dict[str, Any],
    ) -> str | None:
        """执行真实性门禁并把通过门禁的窗口发布为正式记录。

        返回 None 表示窗口可以标记 SUCCEEDED；否则返回稳定失败原因。
        """
        expected_identity = {
            "scene_id": record.scene_id,
            "snapshot_manifest_sha256": record.manifest_sha256,
            "captured_frames": record.captured_frames,
            "camera_role": record.role.role,
            "camera_topic": record.role.capture_config.image_topic,
            "camera_frame_id": record.role.capture_config.camera_frame_id,
        }
        expected_authenticity = {
            "checkpoint_sha256": self._config.expected_checkpoint_sha256,
            "checkpoint_path": self._config.expected_checkpoint_path,
            "result_root": self._config.expected_result_root,
            "algorithm_revision": self._config.expected_algorithm_revision,
            "ros_revision": self._config.expected_ros_revision,
        }
        try:
            gate = self._result_gate(
                outcome.result_manifest_uri,
                outcome.result_manifest_sha256,
                expected_identity=expected_identity,
                expected_authenticity=expected_authenticity,
            )
        except BaseException as error:
            reason = f"result gate raised: {type(error).__name__}: {error}"
            self._write_journal(
                {
                    "event": "window_result_gate_failed",
                    "session_id": self._session_id,
                    "window_index": record.window_index,
                    "scene_id": record.scene_id,
                    "reason": reason,
                }
            )
            return f"formal result gate failed: {reason}"
        if not bool(getattr(gate, "ok", False)):
            reason = str(getattr(gate, "reason", "unknown"))
            self._write_journal(
                {
                    "event": "window_result_gate_failed",
                    "session_id": self._session_id,
                    "window_index": record.window_index,
                    "scene_id": record.scene_id,
                    "reason": reason,
                }
            )
            return f"formal result gate failed: {reason}"
        payload = getattr(gate, "payload", None)
        manifest_bytes = getattr(gate, "manifest_bytes", b"")
        if not isinstance(payload, dict) or not isinstance(manifest_bytes, bytes):
            reason = "formal result gate returned an invalid payload"
            self._write_journal(
                {
                    "event": "window_result_gate_failed",
                    "session_id": self._session_id,
                    "window_index": record.window_index,
                    "scene_id": record.scene_id,
                    "reason": reason,
                }
            )
            return f"formal result gate failed: {reason}"
        if not manifest_bytes:
            reason = "formal result gate returned empty manifest bytes"
            self._write_journal(
                {
                    "event": "window_result_gate_failed",
                    "session_id": self._session_id,
                    "window_index": record.window_index,
                    "scene_id": record.scene_id,
                    "reason": reason,
                }
            )
            return f"formal result gate failed: {reason}"
        with self._lock:
            if self._stopping:
                return _FORMAL_SKIPPED_STOPPING
        try:
            snapshot = validate_snapshot(
                record.snapshot_uri,
                record.manifest_sha256,
                (record.role.capture_config.snapshot_root,),
                record.role.capture_config.snapshot_limits,
            )
            snapshot_manifest_bytes = self._read_snapshot_manifest_bytes(snapshot)
        except BaseException as error:
            return (
                "window snapshot validation failed: "
                f"{type(error).__name__}: {error}"
            )
        snapshot_manifest_sha256 = hashlib.sha256(snapshot_manifest_bytes).hexdigest()
        if snapshot_manifest_sha256 != record.manifest_sha256:
            return (
                "window snapshot manifest digest mismatch after validation: "
                f"expected {record.manifest_sha256}, got {snapshot_manifest_sha256}"
            )
        depth_metrics, depth_metrics_error = self._depth_metrics_window(
            record, outcome
        )
        with self._lock:
            session_id = self._session_id
            timeline = [dict(item) for item in self._config_timeline]
            stopping = self._stopping
        if stopping:
            return _FORMAL_SKIPPED_STOPPING
        checkpoint = payload.get("checkpoint") or {}
        stage_timing = payload.get("stage_timing") or {}
        worker_exit = payload.get("worker_exit") or {}
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": session_id,
            "window_index": int(record.window_index),
            "scene_id": record.scene_id,
            "result_manifest_uri": outcome.result_manifest_uri,
            "result_manifest_sha256": outcome.result_manifest_sha256,
            "snapshot_manifest_uri": record.snapshot_uri,
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "captured_frames": int(record.captured_frames),
            "new_frame_count": int(
                getattr(record, "new_frame_count", record.captured_frames)
            ),
            "overlap_frame_count": int(
                getattr(record, "overlap_frame_count", 0)
            ),
            "window_frame_ids": list(getattr(record, "window_frame_ids", ())),
            "captured_frame_ids": list(getattr(record, "captured_frame_ids", ())),
            "capture_generation": getattr(record, "capture_generation", 0),
            "model_frame_ids": list(getattr(record, "model_frame_ids", ())),
            "camera_role": record.role.role,
            "camera_topic": record.role.capture_config.image_topic,
            "camera_frame_id": record.role.capture_config.camera_frame_id,
            "camera_name": record.role.capture_config.camera_name,
            "config_digest": record.effective.digest(),
            "effective_config": record.effective.as_dict(),
            "config_timeline": timeline,
            "checkpoint": {
                "path": checkpoint.get("path", ""),
                "sha256": expected_authenticity["checkpoint_sha256"],
            },
            "algorithm_revision": expected_authenticity["algorithm_revision"],
            "ros_revision": expected_authenticity["ros_revision"],
            "cuda_environment": {
                "worker_python": self._config.worker_python,
                "cuda_visible_devices": self._config.cuda_visible_devices,
            },
            "worker": {
                "wall_time_sec": stage_timing.get("worker_wall_time_sec"),
                "exit_returncode": worker_exit.get("returncode"),
            },
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            # pool_* 与 confidence_summary 属于正式记录格式，per_window 下恒为 None；
            # 保留这些键，正式记录的格式和字节都不变。
            "reconstruction_mode": self._reconstruction_mode,
            "pool_revision": None,
            "pool_epoch": None,
            "pool_size": None,
            "pool_selected_indices": None,
            "confidence_summary": None,
            "alignment": alignment,
            "depth_metrics": depth_metrics,
            "depth_metrics_error": depth_metrics_error,
        }
        try:
            self._ensure_formal_session_dir(session_id)
            final_path, provenance_sha256 = self._result_publisher(
                self._formal_records_root(),
                session_id,
                record.window_index,
                manifest_bytes,
                provenance,
                snapshot_manifest_bytes,
            )
        except BaseException as error:
            return (
                "formal record publication failed: "
                f"{type(error).__name__}: {error}"
            )
        if not _valid_sha256(provenance_sha256):
            return "formal record publication returned an invalid provenance sha256"
        record.formal_result_uri = str(final_path)
        record.formal_provenance_sha256 = provenance_sha256
        return None

    @staticmethod
    def _predictions_path(outcome) -> Path | None:
        try:
            if not outcome.result_manifest_uri.startswith("file://"):
                return None
            path = Path(
                unquote_to_bytes(
                    outcome.result_manifest_uri[len("file://"):]
                ).decode("utf-8", "strict")
            )
            return path.parent / "predictions.npz"
        except BaseException:
            return None

    @staticmethod
    def _selected_frame_indices(outcome) -> tuple[int, ...] | None:
        """从 result.json 读取每个模型帧对应的 snapshot 帧序号。

        Omega 在 frame_count > frame_group_size 时会均匀抽帧，模型帧序号
        不再等于 snapshot 帧序号；此时模型帧必须按 result.json 的
        ``selected_frames[].index`` 对应回窗口帧。读取失败或
        结构不符时返回 None，按模型帧序号即快照帧序号处理。
        """
        try:
            if not outcome.result_manifest_uri.startswith("file://"):
                return None
            manifest_path = Path(
                unquote_to_bytes(
                    outcome.result_manifest_uri[len("file://"):]
                ).decode("utf-8", "strict")
            )
            with open(manifest_path, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
            selected = payload.get("selected_frames")
            if not isinstance(selected, list):
                return None
            indices = []
            for entry in selected:
                if not isinstance(entry, dict):
                    return None
                index = entry.get("index")
                if isinstance(index, bool) or not isinstance(index, int):
                    return None
                indices.append(index)
            return tuple(indices)
        except BaseException:
            return None

    def _handle_reconstruct_failure(self, record: WindowRecord, error: str) -> None:
        # attempt_failures 是"失败尝试"总数：每次重建尝试失败各记一次，
        # 与终态 failed_windows（未恢复失败）分开，人工重试不清历史。
        with self._lock:
            self._attempt_failures += 1
        oom = bool(_FATAL_OOM_PATTERN.search(error or ""))
        memory_guard = bool(_MEMORY_GUARD_PATTERN.search(error or ""))
        terminal = False
        with self._lock:
            if oom or memory_guard:
                # Both failures stop capture and FIFO consumption. A watchdog
                # exit is distinct from an actual CUDA OOM; neither auto-retries.
                # CUDA OOM 优先：即使消息同时匹配资源门模式也不可人工恢复。
                self._fatal_oom = True
                self._oom_latched = oom
                self._memory_guard_paused = memory_guard and not oom
                self._memory_failed_window = record if self._memory_guard_paused else None
                self._state = SESSION_FATAL_OOM if oom else SESSION_MEMORY_PAUSED
                self._failed += 1
                if self._memory_guard_paused:
                    record.guard_failures.append(error)
                record.state = WINDOW_FAILED
                record.error = f"{self._state}: {error}"
                record.finished_monotonic = self._monotonic()
                record.done.set()
                self._last_error = record.error
                self._write_journal(
                    {
                        "event": "session_memory_paused" if memory_guard else "session_fatal_oom",
                        "session_id": self._session_id,
                        "window_index": record.window_index,
                        "scene_id": record.scene_id,
                        "config_digest": record.effective.digest(),
                        "reason": record.error,
                        "recoverable": self._memory_guard_paused,
                        "recovery": (
                            "RESUME retries this window through the same resource gate"
                            if self._memory_guard_paused
                            else "start a new session after the memory fault is cleared"
                        ),
                    }
                )
                self._wake_capture.set()
                self._wake_consume.set()
                terminal = True
            elif (
                record.reconstruct_attempts <= record.effective.max_retries
                and not self._stopping
                and self._active_job_id is None
            ):
                # 重新入队（下一次消费继续尝试同一 snapshot）。
                record.state = WINDOW_PENDING
                record.error = error
                record.queued_at_monotonic = self._monotonic()
                self._fifo.append(record)
                self._fifo_bytes += record.snapshot_bytes
                self._last_error = error
                self._write_journal(
                    {
                        "event": "window_retry",
                        "session_id": self._session_id,
                        "window_index": record.window_index,
                        "attempts": record.reconstruct_attempts,
                        "config_digest": record.effective.digest(),
                        "reason": error,
                    }
                )
                self._wake_consume.set()
                return
            else:
                self._failed += 1
                self._window_terminal(record, WINDOW_FAILED, error)
                self._last_error = error
                self._write_journal(
                    {
                        "event": "window_failed",
                        "session_id": self._session_id,
                        "window_index": record.window_index,
                        "scene_id": record.scene_id,
                        "attempts": record.reconstruct_attempts,
                        "config_digest": record.effective.digest(),
                        "reason": error,
                    }
                )
                terminal = True
        if terminal:
            cleanup_error = self._remove_snapshot(record)
            if cleanup_error is not None:
                with self._lock:
                    self._last_error = f"{record.error}; snapshot cleanup: {cleanup_error}"

    def _window_terminal(self, record: WindowRecord, state: str, error: str) -> None:
        if record.done.is_set():
            return
        record.state = state
        record.error = error
        record.finished_monotonic = self._monotonic()
        record.done.set()

    def _remove_snapshot(self, record: WindowRecord) -> str | None:
        if self._fatal_oom:
            # Fault evidence survives STOP. A new session uses a different root.
            return None
        uri = record.snapshot_uri
        if not uri:
            return None
        try:
            split = urlsplit(uri)
            if split.scheme != "file" or split.netloc or split.query or split.fragment:
                return f"snapshot uri is not a local file uri: {uri!r}"
            path = Path(unquote(split.path))
            root = Path(record.role.capture_config.snapshot_root)
        except BaseException as error:
            return f"snapshot uri decoding failed: {type(error).__name__}"
        try:
            if not path.is_absolute() or not _is_relative_to(path, root):
                return f"snapshot {path} is outside configured snapshot root"
            if not _SNAPSHOT_DIR_RE.fullmatch(path.name):
                return f"snapshot directory name is not scheduler-owned: {path.name}"
            if not path.exists() or path.is_symlink():
                return None
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode):
                return f"snapshot path is not a directory: {path}"
            import shutil
            shutil.rmtree(str(path))
            return None
        except BaseException as error:
            return f"snapshot removal failed: {type(error).__name__}: {error}"

    def _make_capture_feedback(self, record: WindowRecord):
        def _on_feedback(feedback):
            with self._lock:
                record.captured_frames = int(getattr(feedback, "frame_count", 0))
                stage = str(getattr(feedback, "stage", "") or "")
                record.source_stale = stage == "SOURCE_STALE"
        return _on_feedback

    def _scheduler_snapshot_dir_count(self) -> int | None:
        """Count scheduler-owned snapshot directories without following links."""
        total = 0
        roots = {role.capture_config.snapshot_root for role in self._config.roles}
        try:
            for root in roots:
                with os.scandir(root) as entries:
                    for entry in entries:
                        if (
                            _SNAPSHOT_DIR_RE.fullmatch(entry.name)
                            and entry.is_dir(follow_symlinks=False)
                        ):
                            total += 1
        except OSError:
            return None
        return total

    def _update_stop_audit_backend_locked(
        self,
        job_id: str,
        *,
        cancel_requested: bool = False,
        terminal_collected: bool = False,
        terminal_state: str | None = None,
        terminal_message: str = "",
    ) -> None:
        """Bind/update one backend job while ``self._lock`` is held."""
        message = str(terminal_message)
        if terminal_state == JOB_FAILED and "cleanup_failed" in message.lower():
            # This safety latch is independent of STOP_ALL timing: a worker
            # cleanup failure must stop capture/FIFO consumption immediately;
            # a later stop audit must also remain fail-closed.
            self._backend_cleanup_failure = message
            self._stopping = True
            self._state = SESSION_STOPPING
            self._last_error = (
                "backend process cleanup was not proven: " + message
            )
            self._wake_capture.set()
            self._wake_consume.set()
        audit = self._stop_audit
        if cancel_requested and self._active_job_id == job_id:
            self._active_job_cancel_requested = True
        if audit is None:
            return
        bound_job_id = audit.get("backend_job_id_before")
        if bound_job_id is None:
            audit["backend_job_id_before"] = job_id
            audit["backend_terminal_collected"] = False
        elif bound_job_id != job_id:
            return
        if cancel_requested:
            audit["backend_cancel_requested"] = True
        elif self._active_job_cancel_requested:
            audit["backend_cancel_requested"] = True
        if terminal_collected:
            audit["backend_terminal_collected"] = True
        if terminal_state is not None:
            audit["backend_terminal_state"] = terminal_state
            audit["backend_terminal_message"] = message

    def _formal_staging_dir_count(self) -> int | None:
        """Count current-session formal transaction staging directories."""
        session_dir = self._formal_records_root() / self._session_id
        try:
            if not session_dir.exists():
                return 0
            with os.scandir(session_dir) as entries:
                return sum(
                    1
                    for entry in entries
                    if entry.name.startswith(".drone3d-result-")
                    and entry.is_dir(follow_symlinks=False)
                )
        except OSError:
            return None

    def _join_threads(self, deadline: float | None = None) -> list[str]:
        """等待采集/消费线程退出；返回仍存活的线程名列表。

        ``deadline`` 让 STOP 的两个 join 共用一个总时限，避免串行
        等待将 Web 请求拖到配置时限的两倍。
        """
        alive = []
        if deadline is None:
            deadline = self._monotonic() + float(self._config.stop_timeout_sec)
        for name, thread in (
            ("capture", self._capture_thread),
            ("consume", self._consume_thread),
        ):
            if thread is not None and thread.is_alive():
                thread.join(max(0.0, deadline - self._monotonic()))
                if thread.is_alive():
                    alive.append(name)
        return alive

    def _write_journal(self, payload: dict[str, Any]) -> None:
        path = Path(self._config.journal_root) / f"{self._session_id}.jsonl"
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        try:
            with open(str(path), "a", encoding="utf-8") as stream:
                stream.write(line)
        except OSError:
            pass  # journal 是审计辅助，绝不影响主状态机


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


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _capture_identity_error(
    record: WindowRecord,
    outcome: CaptureActionOutcome,
    expected: CaptureConfig,
) -> str | None:
    if outcome.scene_id != record.scene_id:
        return (
            f"scene_id mismatch: expected {record.scene_id!r}, got {outcome.scene_id!r}"
        )
    if outcome.image_topic != expected.image_topic:
        return (
            f"image_topic mismatch: expected {expected.image_topic!r}, "
            f"got {outcome.image_topic!r}"
        )
    if outcome.camera_name != expected.camera_name:
        return (
            f"camera_name mismatch: expected {expected.camera_name!r}, "
            f"got {outcome.camera_name!r}"
        )
    expected_frames = record.capture_goal_frames or record.effective.window_frames
    if outcome.frame_count != expected_frames:
        return (
            f"capture returned {outcome.frame_count} frames but window required "
            f"{expected_frames}; partial windows are not admitted"
        )
    return None


def _reconstruct_identity_error(
    record: WindowRecord, outcome: ReconstructActionOutcome
) -> str | None:
    role = record.role
    if outcome.scene_id != record.scene_id:
        return (
            f"scene_id mismatch: expected {record.scene_id!r}, got {outcome.scene_id!r}"
        )
    expected_frame_count = record.captured_frames
    if outcome.frame_count != expected_frame_count:
        return (
            f"reconstruction frame_count {outcome.frame_count} does not match "
            f"expected frame_count {expected_frame_count}"
        )
    if outcome.camera_role != role.role:
        return (
            f"camera_role mismatch: expected {role.role!r}, got {outcome.camera_role!r}"
        )
    if outcome.camera_topic != role.capture_config.image_topic:
        return (
            f"camera_topic mismatch: expected {role.capture_config.image_topic!r}, "
            f"got {outcome.camera_topic!r}"
        )
    if outcome.camera_frame_id != role.capture_config.camera_frame_id:
        return (
            f"camera_frame_id mismatch: expected "
            f"{role.capture_config.camera_frame_id!r}, "
            f"got {outcome.camera_frame_id!r}"
        )
    if outcome.snapshot_manifest_sha256 != record.manifest_sha256:
        return (
            "snapshot manifest sha256 mismatch: expected "
            f"{record.manifest_sha256!r}, got {outcome.snapshot_manifest_sha256!r}"
        )
    return None


__all__ = [
    "ContinuousConfig",
    "ContinuousConfigError",
    "ContinuousSession",
    "EffectiveConfig",
    "JobStatus",
    "ReconstructionBackend",
    "SessionStatus",
    "WindowRecord",
    "QUEUE_LOSSLESS",
    "RECONSTRUCTION_MODE_PER_WINDOW",
    "RECONSTRUCTION_MODES",
    "SESSION_DISK_PAUSED",
    "SESSION_FATAL_OOM",
    "SESSION_MEMORY_PAUSED",
    "SESSION_IDLE",
    "SESSION_PAUSED",
    "SESSION_RUNNING",
    "SESSION_DRAINING",
    "SESSION_STOPPED",
    "SESSION_STOPPING",
]
