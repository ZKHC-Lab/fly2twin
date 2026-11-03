"""ROS 无关的连续 session 调度器与磁盘 FIFO 核心。

需求基线见
`docs/reports/single-nx-continuous-gap-audit/single-nx-continuous-gap-audit.md`。
本模块只实现其中的连续调度与磁盘 FIFO 子集：

  - 复用现有 Capture/Reconstruct Action（由注入的客户端/后端适配）；
  - 连续窗口：首窗 N 帧，后续复用上一窗尾部 overlap_frames 帧；
  - 采集循环与单 worker 消费循环解耦；
  - 磁盘 FIFO：内存只存小状态，快照文件全部留在磁盘；
  - `lossless_fifo` / `realtime_latest` 两种队列模式；
  - max_retries、pause/resume、stop_all、磁盘红线、重启清理。

本模块不导入 rospy/actionlib；ROS 适配在 continuous_session_node.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from typing import Any, Callable, Mapping
from urllib.parse import unquote, unquote_to_bytes, urlsplit

from .capture import CaptureConfig, CaptureError, validate_capture_goal
from .depth_metrics import compute_first_frame_depth_metrics
from .frame_pool import (
    RECONSTRUCTION_MODE_FRAME_POOL,
    RECONSTRUCTION_MODE_PER_WINDOW,
    RECONSTRUCTION_MODES,
    FramePool,
    FramePoolCapacityError,
    FramePoolConfig,
    FramePoolError,
)
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
    align_window,
)
from .snapshot_validator import validate_snapshot
from .storage_retention import prune_frame_pool_revisions
from .odom_local import valid_sim3_alignment
from .shared_point_backend import BACKEND_A, BACKEND_ODOM, align as align_shared_points
from .g4.backend_process import BACKEND_G4, G4BackendProcess

from .pipeline_service import (
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
SESSION_ALIGNMENT_PAUSED = "ALIGNMENT_PAUSED"
SESSION_STOPPING = "STOPPING"
SESSION_STOPPED = "STOPPED"

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
WINDOW_BUFFERED = "BUFFERED"

QUEUE_LOSSLESS = "lossless_fifo"
QUEUE_REALTIME_LATEST = "realtime_latest"
_QUEUE_MODES = (QUEUE_LOSSLESS, QUEUE_REALTIME_LATEST)
_DISK_REDLINE_ERROR = "disk redline reached; new capture paused"

# window_frames 的两种语义。total_frames 是既有增量语义；
# new_frames 是移植的长窗分组语义（模型输入 = 新增帧 + 共享帧）。
WINDOW_SEMANTICS_TOTAL = "total_frames"
WINDOW_SEMANTICS_NEW = "new_frames"
_WINDOW_SEMANTICS = (WINDOW_SEMANTICS_TOTAL, WINDOW_SEMANTICS_NEW)

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
    # 窗口帧数语义：total_frames 表示 window_frames 已含共享帧；
    # new_frames 表示 window_frames 只是新增帧，模型输入为 新增＋共享。
    # 新增分组只用中性命名，方法身份由后端 profile 显式给出。
    window_frame_semantics: str = WINDOW_SEMANTICS_TOTAL
    # 长窗分组的计算参数（仅 new_frames 语义使用）。
    minimum_total_frames: int = 100
    # 登记式窗口分组清单；给出时逐窗核对帧清单，缺失时按分组规则推导。
    window_plan_path: str = ""
    # 保留的非 fit 观测表（帧号→模型像素）与逐边历史阈值表。
    fit_exclusion_source: str = ""
    threshold_table_path: str = ""
    # 历史阈值表含多个分组（例如 G0/G1）时必须显式选择，不能让最后一条静默生效。
    threshold_group: str = ""
    # 全局配准的放置方式：graph = 原 G4 图优化（每窗一个 Sim(3)）；
    # frame_pose = 每帧用外部定位位姿，VGGT 只提供深度与每窗尺度。
    g4_placement: str = "graph"
    # frame_pose 下位姿取在 RGB 时间 + 偏移（毫秒）。
    pose_time_offset_ms: float = 0.0
    pose_time_offset_stage: str = "backend"
    frame_pose_fusion: bool = False
    # 源图像分辨率下的标定内参 {width, height, fx, fy, cx, cy}；为空时用快照 CameraInfo。
    camera_intrinsics: dict | None = None
    retention_max_sessions: int = 3
    retention_frame_pool_revisions: int = 2
    retention_orphan_grace_sec: float = 86400.0
    # 0 = inherit reconstruct profile (do not override the worker command)
    initial_inference_resolution: int = 0
    # 0.0 = inherit reconstruct profile worker timeout
    initial_worker_timeout_sec: float = 0.0
    # 重建模式：per_window（现有增量，默认）或 frame_pool（采样区全局重建）。
    # 只在 start_session 时冻结，运行中不可热调。
    initial_reconstruction_mode: str = RECONSTRUCTION_MODE_PER_WINDOW
    # 增量模式使用 ODOM_LOCAL_APPROX；帧数池模式始终跳过对齐。
    odom_local_enabled: bool = True
    odom_min_correspondences: int = 4
    odom_min_baseline_m: float = 0.1
    odom_max_orientation_residual_deg: float = 15.0
    odom_max_normalized_residual_rms: float = 0.25
    odom_reset_jump_m: float = 5.0
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
    alignment_backend: str = BACKEND_ODOM
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
        if self.alignment_backend not in (BACKEND_ODOM, BACKEND_A, BACKEND_G4):
            raise ContinuousConfigError("unknown alignment_backend")
        if self.alignment_backend == BACKEND_G4:
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
        if self.alignment_backend == BACKEND_A:
            profile_name = "A"
            if (
                self.initial_window_frames != 100 or self.initial_overlap_frames != 20
                or self.reconstruct_frame_group_size != 120
                or self.initial_inference_resolution != 384
                or self.initial_sample_fps != 5.0
                or self.initial_queue_mode != QUEUE_LOSSLESS
                or self.initial_max_retries != 0
            ):
                raise ContinuousConfigError(
                    f"{profile_name} fixes 5 Hz, 100 new frames, 20 shared frames, model limit 120, "
                    "resolution 384, lossless_fifo and zero automatic retries"
                )
            for role in self.roles:
                capture = role.capture_config
                if (
                    capture.odometry_association != "interpolate"
                    or capture.pose_interpolation_wait_ms != 500.0
                    or capture.pose_gap_max_ms != 500.0
                    or capture.snapshot_limits.max_frame_count < 120
                ):
                    raise ContinuousConfigError(f"{profile_name} requires 500 ms interpolation and 120-frame snapshots")
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
        if self.window_frame_semantics not in _WINDOW_SEMANTICS:
            raise ContinuousConfigError(
                f"window_frame_semantics must be one of {_WINDOW_SEMANTICS!r}"
            )
        if (
            isinstance(self.minimum_total_frames, bool)
            or not isinstance(self.minimum_total_frames, int)
            or self.minimum_total_frames <= 0
        ):
            raise ContinuousConfigError("minimum_total_frames must be a positive integer")
        if self.window_frame_semantics == WINDOW_SEMANTICS_NEW:
            if self.alignment_backend != BACKEND_G4:
                # 静默忽略一种显式给出的语义比拒绝更糟。
                raise ContinuousConfigError(
                    "new_frames semantics is defined for the global placement backend only"
                )
            if self.minimum_total_frames <= self.initial_overlap_frames:
                # 该下限只在 new_frames 语义下参与尾窗规则；total_frames 会话
                # 不受它约束，不能因为一个无效字段拒绝原本合法的配置。
                raise ContinuousConfigError(
                    "minimum_total_frames must exceed initial_overlap_frames under new_frames semantics"
                )
            # The model receives 新增＋共享 帧，因此上限必须容纳两者之和。
            if self.initial_window_frames + self.initial_overlap_frames > self.reconstruct_frame_group_size:
                raise ContinuousConfigError(
                    "new_frames semantics needs frame_group_size >= window_frames + overlap_frames; "
                    "no silent subsampling"
                )
            if self.initial_inference_resolution != 384:
                raise ContinuousConfigError("new_frames semantics is defined for the 384-wide model input")
        for name in ("window_plan_path", "fit_exclusion_source", "threshold_table_path", "threshold_group"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ContinuousConfigError(f"{name} must be a string path or empty")
        if type(self.frame_pose_fusion) is not bool:
            raise ContinuousConfigError("frame_pose_fusion must be a bool")
        if self.pose_time_offset_stage not in ("capture", "backend"):
            raise ContinuousConfigError("pose_time_offset_stage must be capture or backend")
        if self.g4_placement not in ("graph", "frame_pose"):
            raise ContinuousConfigError("g4_placement must be graph or frame_pose")
        if self.g4_placement == "frame_pose" and self.alignment_backend != BACKEND_G4:
            raise ContinuousConfigError("g4_placement=frame_pose needs alignment_backend global_g4")
        if (
            isinstance(self.pose_time_offset_ms, bool)
            or not isinstance(self.pose_time_offset_ms, (int, float))
            or not math.isfinite(self.pose_time_offset_ms)
            or abs(self.pose_time_offset_ms) > 1000
        ):
            raise ContinuousConfigError("pose_time_offset_ms must be finite and within +-1000")
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
        if self.initial_queue_mode not in _QUEUE_MODES:
            raise ContinuousConfigError(
                f"initial_queue_mode must be one of {_QUEUE_MODES!r}"
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
        for name in ("retention_max_sessions", "retention_frame_pool_revisions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContinuousConfigError(f"{name} must be a positive integer")
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
        if type(self.odom_local_enabled) is not bool:
            raise ContinuousConfigError("odom_local_enabled must be a bool")
        if (
            isinstance(self.odom_min_correspondences, bool)
            or not isinstance(self.odom_min_correspondences, int)
            or self.odom_min_correspondences < 4
        ):
            raise ContinuousConfigError(
                "odom_min_correspondences must be an integer >= 4"
            )
        for name in (
            "odom_min_baseline_m",
            "odom_max_orientation_residual_deg",
            "odom_max_normalized_residual_rms",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ContinuousConfigError(f"{name} must be positive and finite")
        if self.odom_max_orientation_residual_deg > 180:
            raise ContinuousConfigError(
                "odom_max_orientation_residual_deg must be <= 180"
            )
        if (
            isinstance(self.odom_reset_jump_m, bool)
            or not isinstance(self.odom_reset_jump_m, (int, float))
            or not math.isfinite(float(self.odom_reset_jump_m))
            or self.odom_reset_jump_m < 0
        ):
            raise ContinuousConfigError(
                "odom_reset_jump_m must be a nonnegative finite number"
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
    if converted != 0.0 and converted <= 0:
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
    # 长窗分组为该窗口计划的新增帧数与共享帧数（total_frames 语义下为 0）。
    planned_new_frames: int = 0
    planned_shared_frames: int = 0
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
    pool_imported: bool = False
    pool_epoch: int = -1
    pool_revision: int = -1
    pool_size: int = 0
    pool_selected_indices: list[int] = field(default_factory=list)
    pool_confidence_summary: dict[str, Any] = field(default_factory=dict)
    capture_generation: int = 0
    captured_frame_ids: list[int] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)
    shared_reference_candidate: Any = field(default=None, repr=False)
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
    map_epoch: int = 0
    last_alignment_state: str = COORDINATE_MODEL_LOCAL_UNALIGNED
    formal_windows: int = 0
    reconstruction_mode: str = RECONSTRUCTION_MODE_PER_WINDOW
    alignment_backend: str = BACKEND_ODOM


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
        if not isinstance(config, ContinuousConfig):
            raise TypeError("config must be ContinuousConfig")
        config.preflight()
        if not callable(capture_client_factory) or not callable(backend_factory):
            raise TypeError("client/backend factories must be callable")
        if result_gate is not None and not callable(result_gate):
            raise TypeError("result_gate must be callable")
        if result_publisher is not None and not callable(result_publisher):
            raise TypeError("result_publisher must be callable")
        self._config = config
        self._capture_client_factory = capture_client_factory
        self._backend_factory = backend_factory
        self._disk_free_gb = disk_free_gb or (lambda: float("inf"))
        self._result_gate = result_gate if result_gate is not None else verify_result_manifest
        self._result_publisher = (
            result_publisher if result_publisher is not None else publish_formal_record
        )
        self._worker_disk_blocked = False
        self._pool_capacity_blocked = False
        self._fifo_bytes = 0
        self._fatal_oom = False
        self._map_epoch = 0
        self._odom_reference_frame_id: str | None = None
        self._last_odom_position: tuple[float, float, float] | None = None
        self._last_odom_stamp: tuple[int, int] | None = None
        self._last_alignment_state = COORDINATE_MODEL_LOCAL_UNALIGNED
        self._formal_windows = 0
        self._last_worker_wall_time_sec: float | None = None
        self._last_service_wall_time_sec: float | None = None
        self._last_service_budget_sec: float | None = None
        self._reconstruction_mode = config.initial_reconstruction_mode
        self._pool: FramePool | None = None
        self._pool_revision = 0
        self._plan = None
        self._plan_path = ""
        self._plan_exhausted = False
        self._capture_generation = 0
        self._config_timeline: list[dict[str, Any]] = []
        self._thread_factory = thread_factory
        self._monotonic = monotonic
        self._pool_eligible_frames = 0
        self._pool_lock = threading.RLock()
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
            overlap_frames=(0 if config.initial_reconstruction_mode == RECONSTRUCTION_MODE_FRAME_POOL
                            else config.initial_overlap_frames),
        )
        self._state = SESSION_IDLE
        self._session_id = ""
        self._next_window = 0
        self._windows: list[WindowRecord] = []
        self._shared_reference = None
        self._g4_process = None
        self._alignment_fault = False
        self._fifo: list[WindowRecord] = []  # deque-like; only PENDING entries
        self._running_window: WindowRecord | None = None
        self._succeeded = 0
        self._failed = 0
        # 失败历史分两层：failed_windows 只数**未恢复的终态失败**（外部健康
        # 判据）；attempt_failures 单调累加每次尝试级失败（人工重试后仍保留）。
        self._attempt_failures = 0
        self._manual_retries = 0
        self._dropped = 0
        self._last_error = ""
        self._capture_client: ContinuousCaptureClient | None = None
        self._backend: ReconstructionBackend | None = None
        self._active_job_id: str | None = None
        self._active_job_cancel_requested = False
        self._backend_cleanup_failure = ""
        self._stop_audit: dict[str, Any] | None = None
        self._stopping = False
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
                SESSION_ALIGNMENT_PAUSED,
                SESSION_STOPPING,
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
            if (selected_mode == RECONSTRUCTION_MODE_PER_WINDOW
                    and not self._config.odom_local_enabled):
                return 5, "per_window requires odom_local_enabled=true"
            # 新 session 的角色相关采集值回到冻结 profile，避免上一
            # 角色的热调值穿透到边界不同的 Capture Action。
            runtime = EffectiveConfig(
                sample_fps=self._config.initial_sample_fps,
                window_frames=self._config.initial_window_frames,
                queue_mode=self._runtime.queue_mode,
                queue_capacity=self._runtime.queue_capacity,
                max_retries=self._runtime.max_retries,
                keep_snapshot_on_success=self._runtime.keep_snapshot_on_success,
                save_sensor_depth=role.capture_config.save_sensor_depth,
                inference_resolution=self._config.initial_inference_resolution,
                worker_timeout_sec=self._config.initial_worker_timeout_sec,
                overlap_frames=(0 if selected_mode == RECONSTRUCTION_MODE_FRAME_POOL
                                else self._config.initial_overlap_frames),
            )
            self._validate_runtime(runtime, role)
            self._selected_role = role
            self._runtime = runtime
            self._reconstruction_mode = selected_mode
            self._pool = None
            self._pool_eligible_frames = 0
            self._pool_revision = 0
            self._plan = None
            self._plan_path = ""
            self._plan_exhausted = False
            self._capture_generation = 0
            # 清空缓存之后再载入一次并校验：本次会话固定使用这份计划。
            # 顺序反了会先校验上一次会话的缓存，再让采集线程重新打开可能已被
            # 删除或改写的文件。
            if self._config.window_plan_path.strip():
                try:
                    self._window_plan()
                except Exception as error:
                    return 6, (
                        "grouping plan is unusable: "
                        f"{type(error).__name__}: {error}"
                    )
            self._session_id = "s" + secrets.token_hex(8)
            self._next_window = 0
            self._windows = []
            self._shared_reference = None
            self._alignment_fault = False
            self._fifo = []
            self._running_window = None
            self._succeeded = 0
            self._failed = 0
            self._attempt_failures = 0
            self._manual_retries = 0
            self._dropped = 0
            self._last_error = ""
            self._stopping = False
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
            self._pool_capacity_blocked = False
            self._fifo_bytes = 0
            self._map_epoch = 0
            self._odom_reference_frame_id = None
            self._last_odom_position = None
            self._last_odom_stamp = None
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
        线程同时消费同一个 FIFO。登记清单已耗尽时只重启消费线程：此时只剩
        排空，重试窗照样会被消费，不需要也不能再开新采集。
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

    def set_overlap_frames(self, count: int) -> tuple[int, str]:
        """Set reused tail frames: 0 captures independently; N reuses N frames.

        The first window always captures all frames. During a session the
        change applies to the next window; the active capture stays frozen.
        Returns (0, message) on success, otherwise an existing service error code.
        """
        with self._lock:
            if self._config.alignment_backend in (BACKEND_A, BACKEND_G4):
                return 5, "shared-window profiles freeze overlap; choose another configured profile"
            if (self._reconstruction_mode == RECONSTRUCTION_MODE_FRAME_POOL
                    and self._state not in (SESSION_IDLE, SESSION_STOPPED)):
                return 5, "frame_pool fixes overlap_frames at 0; it cannot be changed"
            try:
                _validate_overlap_frames(count, self._runtime.window_frames)
            except ContinuousConfigError as error:
                return 5, str(error)
            if self._state in (SESSION_IDLE, SESSION_STOPPED):
                try:
                    config = replace(self._config, initial_overlap_frames=count)
                except ContinuousConfigError as error:
                    return 5, str(error)
                self._config = config
                self._runtime = replace(self._runtime, overlap_frames=count)
                return 0, f"overlap_frames set to {count}"
        return self.set_params(overlap_frames=count)

    def set_params(
        self,
        *,
        sample_fps: float | None = None,
        window_frames: int | None = None,
        overlap_frames: int | None = None,
        queue_mode: str | None = None,
        queue_capacity: int | None = None,
        max_retries: int | None = None,
        keep_snapshot_on_success: bool | None = None,
        save_sensor_depth: bool | None = None,
        inference_resolution: int | None = None,
        worker_timeout_sec: float | None = None,
    ) -> tuple[int, str]:
        with self._lock:
            if self._uses_shared_A() or self._uses_g4():
                return 5, "shared-window runtime parameters are frozen; start another configured session"
            if self._state not in (
                SESSION_RUNNING,
                SESSION_PAUSED,
                SESSION_DISK_PAUSED,
            ):
                return 4, f"cannot set params in state {self._state}"
            if self._reconstruction_mode == RECONSTRUCTION_MODE_FRAME_POOL and overlap_frames is not None:
                return 5, "frame_pool fixes overlap_frames at 0; it cannot be changed"
            if save_sensor_depth is not None and type(save_sensor_depth) is not bool:
                return 5, "save_sensor_depth must be a bool"
            if inference_resolution is not None and (
                isinstance(inference_resolution, bool)
                or not isinstance(inference_resolution, int)
            ):
                return 5, "inference_resolution must be an integer"
            if worker_timeout_sec is not None and (
                isinstance(worker_timeout_sec, bool)
                or not isinstance(worker_timeout_sec, (int, float))
            ):
                return 5, "worker_timeout_sec must be a finite number"
            if overlap_frames is not None and (
                isinstance(overlap_frames, bool) or not isinstance(overlap_frames, int)
            ):
                return 5, "overlap_frames must be an integer"
            current = self._runtime
            candidate = EffectiveConfig(
                sample_fps=float(sample_fps)
                if sample_fps is not None
                else current.sample_fps,
                window_frames=int(window_frames)
                if window_frames is not None
                else current.window_frames,
                queue_mode=queue_mode if queue_mode is not None else current.queue_mode,
                queue_capacity=int(queue_capacity)
                if queue_capacity is not None
                else current.queue_capacity,
                max_retries=int(max_retries)
                if max_retries is not None
                else current.max_retries,
                keep_snapshot_on_success=bool(keep_snapshot_on_success)
                if keep_snapshot_on_success is not None
                else current.keep_snapshot_on_success,
                save_sensor_depth=bool(save_sensor_depth)
                if save_sensor_depth is not None
                else current.save_sensor_depth,
                inference_resolution=int(inference_resolution)
                if inference_resolution is not None
                else current.inference_resolution,
                worker_timeout_sec=float(worker_timeout_sec)
                if worker_timeout_sec is not None
                else current.worker_timeout_sec,
                overlap_frames=int(overlap_frames)
                if overlap_frames is not None
                else current.overlap_frames,
            )
            try:
                self._validate_runtime(candidate, self._selected_role)
            except ContinuousConfigError as error:
                return 5, str(error)
            previous_digest = current.digest()
            capturing_window = next(
                (
                    record.window_index
                    for record in reversed(self._windows)
                    if record.state == WINDOW_CAPTURING
                ),
                None,
            )
            self._runtime = candidate
            change_event = {
                "index": len(self._config_timeline),
                "event": "config_changed",
                "previous_config_digest": previous_digest,
                "config_digest": candidate.digest(),
                "applies_from_window_index": (
                    self._windows[-1].window_index + 1
                    if self._windows
                    else self._next_window
                ),
                "window_index_capturing": capturing_window,
                "config": candidate.as_dict(),
            }
            self._config_timeline.append(change_event)
            journal_payload = dict(change_event)
            journal_payload["session_id"] = self._session_id
            self._write_journal(journal_payload)
            return 0, "runtime params updated (next window applies)"

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
        if self._reconstruction_mode != RECONSTRUCTION_MODE_PER_WINDOW:
            return 5, (
                "manual retry of a resource-guard window is only supported for "
                f"per_window sessions, not {self._reconstruction_mode}: the "
                "frame_pool path re-selects pool inputs instead of reusing the "
                "preserved snapshot"
            )
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
            if not self._plan_exhausted and record.window_index == self._next_window:
                continue  # 重启采集会从这个索引重新采
            return (
                f"window {record.window_index} was cancelled by the resource pause "
                f"but capture restarts from window {self._next_window}"
                + (" and the registered plan is exhausted" if self._plan_exhausted else "")
                + "; that window cannot be recovered in place"
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
        # realtime_latest 下队列可能暂时多一条，由采集侧容量判断自然收敛。
        self._fifo.insert(0, record)
        self._fifo_bytes += record.snapshot_bytes
        self._failed = max(0, self._failed - 1)
        self._memory_guard_paused = False
        self._memory_failed_window = None
        self._fatal_oom = False
        self._oom_latched = False
        restart_capture = not self._plan_exhausted
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

    def stop_all(self) -> tuple[int, str]:
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
            if (self._reconstruction_mode == RECONSTRUCTION_MODE_PER_WINDOW
                    and self._pool is not None
                    and not (self._uses_shared_A() and (self._alignment_fault or self._fatal_oom))):
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
            self.stop_all()
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
                dropped_windows=self._dropped,
                effective=self._runtime,
                last_error=self._last_error,
                min_free_disk_gb=self._config.min_free_disk_gb,
                fifo_bytes=fifo_bytes,
                fifo_oldest_wait_sec=fifo_oldest_wait_sec,
                disk_free_gb=self._disk_free_locked(),
                map_epoch=self._map_epoch,
                last_alignment_state=self._last_alignment_state,
                formal_windows=self._formal_windows,
                reconstruction_mode=self._reconstruction_mode,
                alignment_backend=self._config.alignment_backend,
            )

    def _diagnostic_input_frames(self, status, window, kind: str):
        """诊断里的 新增/共享/模型输入 三个数（按窗口实际语义）。

        非分组后端（total_frames 语义）保持原样：模型输入就是
        effective.window_frames，共享为 effective.overlap_frames，新增不显示。
        """
        shared = int(window.overlap_frame_count) if window is not None else 0
        if self._uses_shared_A():
            values = {"new": 100, "model": 100 + shared, "shared": shared}
        elif self._uses_g4():
            if window is not None:
                model = int(window.effective.window_frames)
                planned_new = int(getattr(window, "planned_new_frames", 0) or 0)
                planned_shared = int(getattr(window, "planned_shared_frames", 0) or 0)
                if planned_new or planned_shared:
                    values = {"new": planned_new, "model": model, "shared": planned_shared}
                else:
                    values = {"new": model - shared, "model": model, "shared": shared}
            else:
                model = int(status.effective.window_frames)
                values = {"new": model - shared, "model": model, "shared": shared}
        else:
            values = {
                "new": None,
                "model": status.effective.window_frames,
                "shared": status.effective.overlap_frames,
            }
        return values[kind]

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
            shared_count = display_window.overlap_frame_count if display_window is not None else 0
            return {
                "state": status.state,
                "session_id": status.session_id,
                "alignment_backend": status.alignment_backend,
                "g4": self._g4_process.status() if self._g4_process is not None else {},
                "alignment_fault": self._alignment_fault,
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
                # 按**该窗口实际生效**的计划与配置给出：长窗分组下 window_frames 是
                # 模型输入总数，新增/共享必须取该窗口的 planned_*，否则会显示成
                # 总帧语义（例如 80/20/100 而非 100/20/120）。
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
                    "window_frame_semantics": (
                        "new_frames" if (self._uses_shared_A() or self._grouped_window_semantics())
                        else "total_frames"
                    ),
                    "steady_model_frames": (
                        120 if self._uses_shared_A()
                        else (self._config.initial_window_frames + self._config.initial_overlap_frames
                              if self._grouped_window_semantics() else status.effective.window_frames)
                    ),
                    "window_plan_source": (
                        None if not self._grouped_window_semantics()
                        else (self._config.window_plan_path or "derived_grouping_rule")
                    ),
                    "minimum_total_frames": (
                        self._config.minimum_total_frames if self._grouped_window_semantics() else None
                    ),
                    "grouping_plan_exhausted": bool(self._plan_exhausted),
                    "g4_placement": (
                        self._config.g4_placement if status.alignment_backend == BACKEND_G4 else None
                    ),
                    "pose_time_offset_ms": (
                        self._config.pose_time_offset_ms
                        if self._config.g4_placement == "frame_pose" else None
                    ),
                },
                "fifo": waiting,
                "fifo_bytes": status.fifo_bytes,
                "fifo_oldest_wait_sec": status.fifo_oldest_wait_sec,
                "last_error": status.last_error,
                "min_free_disk_gb": status.min_free_disk_gb,
                "disk_free_gb": status.disk_free_gb,
                "disk_redline": self._disk_low_locked(),
                "worker_disk_blocked": self._worker_disk_blocked,
                "pool_capacity_blocked": self._pool_capacity_blocked,
                "pool_eligible_frames": self._pool_eligible_frames,
                "pool_target_frames": status.effective.window_frames,
                "pool_waiting_for_frames": (
                    self._reconstruction_mode == RECONSTRUCTION_MODE_FRAME_POOL
                    and self._pool_eligible_frames < status.effective.window_frames
                ),
                "map_epoch": status.map_epoch,
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

    def _validate_runtime(
        self, candidate: EffectiveConfig, role: CameraRole | None
    ) -> None:
        _validate_window_frames(
            candidate.window_frames, self._config.reconstruct_frame_group_size
        )
        _validate_overlap_frames(candidate.overlap_frames, candidate.window_frames)
        if (
            isinstance(candidate.sample_fps, bool)
            or not isinstance(candidate.sample_fps, (int, float))
            or not _finite(candidate.sample_fps)
            or candidate.sample_fps <= 0
        ):
            raise ContinuousConfigError("sample_fps must be positive and finite")
        if candidate.queue_mode not in _QUEUE_MODES:
            raise ContinuousConfigError(
                f"queue_mode must be one of {_QUEUE_MODES!r}"
            )
        if (
            isinstance(candidate.queue_capacity, bool)
            or not isinstance(candidate.queue_capacity, int)
            or candidate.queue_capacity <= 0
        ):
            raise ContinuousConfigError("queue_capacity must be a positive integer")
        if (
            isinstance(candidate.max_retries, bool)
            or not isinstance(candidate.max_retries, int)
            or candidate.max_retries < 0
        ):
            raise ContinuousConfigError("max_retries must be a nonnegative integer")
        if type(candidate.keep_snapshot_on_success) is not bool:
            raise ContinuousConfigError("keep_snapshot_on_success must be a bool")
        if type(candidate.save_sensor_depth) is not bool:
            raise ContinuousConfigError("save_sensor_depth must be a bool")
        _validate_inference_resolution(
            candidate.inference_resolution, "inference_resolution"
        )
        from .backend_spec import validate_independent_request
        try:
            validate_independent_request(self._config.reconstruction_backend, candidate.window_frames,
                                         candidate.inference_resolution)
        except ValueError as error:
            raise ContinuousConfigError(str(error)) from error
        _validate_worker_timeout(candidate.worker_timeout_sec, "worker_timeout_sec")
        if role is None:
            raise ContinuousConfigError("no camera role selected")
        _validate_capture_runtime(
            candidate, role, self._config.capture_wait_margin_sec
        )

    def _scene_id(self, window_index: int) -> str:
        return f"cs-{self._session_id}-w{window_index}"

    def _disk_low_locked(self) -> bool:
        if self._pool_capacity_blocked:
            return True
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
                    if self._stopping or self._fatal_oom:
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
                    if self._state != SESSION_RUNNING or self._plan_exhausted or (
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
                        planned_new = planned_shared = 0
                        grouping_error = ""
                        if self._grouped_window_semantics():
                            # 计划读取本身也可能失败（文件被删/改写），必须和形状
                            # 解析一起转成可见状态，不能让采集线程退出。
                            try:
                                plan = self._window_plan()
                                if plan is not None and window_index >= len(plan):
                                    # 登记清单跑完是正常结束：停止新采集，让消费把
                                    # 已排队/在途窗口排空，不置故障、不暂停消费。
                                    self._plan_exhausted = True
                                    self._last_error = ""
                                    self._write_journal({
                                        "event": "grouping_plan_exhausted",
                                        "session_id": self._session_id,
                                        "window_index": window_index,
                                        "plan_windows": len(plan),
                                    })
                                    proceed = False
                                else:
                                    planned_new, planned_shared = self._planned_window_shape(window_index)
                                    effective = self._grouped_effective(window_index)
                            # WindowPlanError 是 ValueError 的子类，一并覆盖。
                            except (FramePoolError, ValueError, OSError) as error:
                                grouping_error = str(error)
                        if grouping_error:
                            # 分组与实到帧不符时暂停会话并保留现场；采集线程
                            # 不能因为配置/输入不符而退出，否则会变成静默停摆。
                            self._state = SESSION_ALIGNMENT_PAUSED
                            self._alignment_fault = True
                            self._last_error = "long-window grouping: " + grouping_error
                            self._write_journal({
                                "event": "long_window_grouping_paused",
                                "session_id": self._session_id,
                                "window_index": window_index,
                                "reason": grouping_error,
                                "recovery": "fix the grouping plan or start a new session",
                            })
                            proceed = False
                        elif proceed:
                            record = WindowRecord(
                                window_index=window_index,
                                scene_id=self._scene_id(window_index),
                                role=role,
                                effective=effective,
                                capture_generation=self._capture_generation,
                                planned_new_frames=planned_new,
                                planned_shared_frames=planned_shared,
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
                    if self._uses_shared_A():
                        self._pause_shared_failure(error)
                    elif self._uses_g4():
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
                if self._stopping:
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
                if self._stopping:
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
            if self._stopping or self._fatal_oom:
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
                source_stale = record.source_stale
                paused = self._state in (SESSION_PAUSED, SESSION_ALIGNMENT_PAUSED)
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
            if stopping or fatal_oom:
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
            if not stopping and not fatal_oom and not source_stale and not paused and self._monotonic() >= deadline:
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
        if self._reconstruction_mode == RECONSTRUCTION_MODE_FRAME_POOL:
            try:
                self._import_pool_capture(record)
            except FramePoolCapacityError as error:
                with self._lock:
                    self._pool_capacity_blocked = True
                    self._state = SESSION_DISK_PAUSED
                return f"frame pool capacity reached: {error}"
            except Exception as error:
                return f"frame pool import failed: {type(error).__name__}: {error}"
        if (
            self._reconstruction_mode == RECONSTRUCTION_MODE_PER_WINDOW
        ):
            try:
                self._prepare_overlap_capture(record)
            except BaseException as error:
                return (
                    "overlap window preparation failed: "
                    f"{type(error).__name__}: {error}"
                )
        with self._lock:
            record.queued_at_monotonic = self._monotonic()
            record.state = WINDOW_PENDING
            record.snapshot_bytes = self._snapshot_bytes_from_uri(
                record.snapshot_uri
            )
            self._fifo.append(record)
            self._fifo_bytes += record.snapshot_bytes
            if record.effective.queue_mode == QUEUE_REALTIME_LATEST:
                while len(self._fifo) > record.effective.queue_capacity:
                    dropped = self._fifo.pop(0)
                    self._fifo_bytes -= dropped.snapshot_bytes
                    self._remove_snapshot(dropped)
                    self._dropped += 1
                    dropped.state = WINDOW_CANCELLED
                    dropped.error = "realtime_latest dropped oldest waiting window"
                    dropped.done.set()
                    self._write_journal(
                        {
                            "event": "window_dropped",
                            "session_id": self._session_id,
                            "window_index": dropped.window_index,
                            "scene_id": dropped.scene_id,
                            "config_digest": dropped.effective.digest(),
                            "reason": dropped.error,
                        }
                    )
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
                        and not self._alignment_fault
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
                        100 if self._uses_shared_A()
                        else record.effective.window_frames - record.effective.overlap_frames
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
        """交给后台数值核对的显式配置。

        方法身份、分组清单、历史阈值表与保留观测表都在这里显式给出；
        未配置的项保持默认，既有 total_frames 会话因此逐位不变。
        """
        from g4_backend import FRAME_POSE_METHOD_ID, METHOD_ID, WINDOWED_METHOD_ID

        grouped = self._grouped_window_semantics()
        profile = {
            "method_id": WINDOWED_METHOD_ID if grouped else METHOD_ID,
            "window_plan_path": str(self._config.window_plan_path),
            "threshold_table_path": str(self._config.threshold_table_path),
            "threshold_group": str(self._config.threshold_group),
            "fit_exclusion_source": str(self._config.fit_exclusion_source),
            "window_frame_semantics": self._config.window_frame_semantics,
            "minimum_total_frames": int(self._config.minimum_total_frames),
            # 分组形状：后端据此核对登记清单，不自行推断。
            "window_frames": int(self._config.initial_window_frames),
            "overlap_frames": int(self._config.initial_overlap_frames),
            # 长窗分组按冻结算法复现：退化三点假设中止该边，而不是继续换假设。
            "registration": {"ransac_degenerate": "abort"} if grouped else {},
        }
        if self._config.g4_placement == "frame_pose":
            # 只有新放置方式才增加键，既有 graph 会话的 profile 逐字不变。
            profile.update(
                method_id=FRAME_POSE_METHOD_ID, placement="frame_pose",
                frame_pose={"pose_time_offset_ms": float(self._config.pose_time_offset_ms)},
                camera_intrinsics=(None if self._config.camera_intrinsics is None
                                   else dict(self._config.camera_intrinsics)),
            )
            if self._config.frame_pose_fusion:
                profile["frame_pose_fusion"] = True
            if self._config.pose_time_offset_stage == "capture":
                profile["frame_pose"]["pose_time_offset_stage"] = "capture"
        return profile

    # ------------------------------------------------------------------
    # 长窗分组（中性命名）：window_frames 只数新增帧，模型输入为新增＋共享
    # ------------------------------------------------------------------
    def _grouped_window_semantics(self) -> bool:
        """移植的长窗分组是否生效。

        只在全局配准后端上生效：window_frames 表示新增帧，模型实际输入
        为新增＋共享。既有 total_frames 语义与 A 模式都不受影响。
        """
        return (
            self._config.alignment_backend == BACKEND_G4
            and self._config.window_frame_semantics == WINDOW_SEMANTICS_NEW
        )

    def _window_plan(self):
        """登记的窗口分组清单；未登记时为 None（按分组规则推导）。"""
        path = self._config.window_plan_path
        if not path.strip():
            return None
        if self._plan is None or self._plan_path != path:
            from g4_backend import load_plan

            plan = load_plan(path)
            limit = int(self._config.reconstruct_frame_group_size)
            oversized = [spec.id for spec in plan if spec.input_frames > limit]
            if oversized:
                # 超限会被 worker 静默抽帧，因此在这里明确拒绝。
                raise FramePoolError(
                    "registered grouping plan exceeds reconstruct frame_group_size="
                    f"{limit}: " + ", ".join(oversized)
                )
            if plan.windows[0].shared_prefix:
                raise FramePoolError("a registered grouping plan must start with an unshared window")
            self._plan, self._plan_path = plan, path
        return self._plan

    def _planned_window_shape(self, window_index: int) -> tuple[int, int]:
        """返回 (新增帧数, 共享帧数)；登记清单优先，否则按分组规则推导。"""
        plan = self._window_plan()
        spec = None
        if plan is not None:
            if window_index >= len(plan):
                raise FramePoolError(
                    f"window {window_index} exceeds the registered grouping plan of {len(plan)} windows"
                )
            spec = plan.spec(window_index)
        if spec is not None:
            return spec.new_frames, spec.shared_prefix
        if window_index == 0:
            # 首窗没有可复用的前序帧：新增即整窗。
            return self._runtime.window_frames, 0
        return self._runtime.window_frames, self._runtime.overlap_frames

    def _grouped_effective(self, window_index: int) -> EffectiveConfig:
        """把新增帧语义解析成该窗口的实际模型输入总帧数与共享帧数。

        共享帧数必须一并写入：尾窗计划是"共享 26 ＋ 新增 74"，若只改总帧数，
        采集目标会按运行时 overlap（20）去凑共享帧并直接报错。
        """
        new_frames, shared = self._planned_window_shape(window_index)
        total = new_frames + shared
        limit = int(self._config.reconstruct_frame_group_size)
        if total > limit:
            raise FramePoolError(
                f"window {window_index} needs {total} model frames but frame_group_size is {limit}"
            )
        return replace(self._runtime, window_frames=total, overlap_frames=shared)

    # ------------------------------------------------------------------
    # frame_pool (采样区) helpers
    # ------------------------------------------------------------------
    def _capture_goal_frames(self, record: WindowRecord) -> int:
        """Plan only the new frames; the first window still captures all N."""
        if (
            self._reconstruction_mode != RECONSTRUCTION_MODE_PER_WINDOW
            or record.effective.overlap_frames <= 0
        ):
            record.capture_goal_frames = record.effective.window_frames
            record.overlap_frame_count = 0
            return record.capture_goal_frames
        pool = self._frame_pool()
        pool.load()
        record.overlap_frame_count = min(
            record.effective.overlap_frames, len(pool.entries)
        )
        if self._grouped_window_semantics():
            # 长窗分组：共享帧数由登记清单或分组规则给出，不随可用池静默缩小。
            required = record.planned_shared_frames
            if record.overlap_frame_count != required:
                raise FramePoolError(
                    "long-window grouping needs exactly "
                    f"{required} shared RGB frames but {record.overlap_frame_count} "
                    "are available; preserve inputs and start a new session"
                )
            record.capture_goal_frames = record.planned_new_frames
            return record.capture_goal_frames
        if self._uses_shared_A():
            if record.overlap_frame_count not in (0, 20):
                raise FramePoolError("shared-window profiles require exactly twenty cached shared RGB frames")
            record.capture_goal_frames = 100
            return 100
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
        if previous and snapshot.snapshot_version >= 2 and snapshot.frame_stamps:
            first_stamp = snapshot.frame_stamps[0]
            if (first_stamp.sec, first_stamp.nsec) <= previous[-1].stamp_key():
                if self._uses_shared_A():
                    raise FramePoolError("shared-window RGB time reset: preserve inputs and start a new session")
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
            new_ids = pool.import_window(
                snapshot, record.window_index, sharpness_reader=lambda _: 1.0
            )
            by_index = {entry.global_index: entry for entry in pool.entries}
            selected = previous[-record.overlap_frame_count :] if record.overlap_frame_count else []
            selected += [by_index[index] for index in new_ids]
            expected_frames = (
                100 + record.overlap_frame_count
                if self._uses_shared_A() else record.effective.window_frames
            )
            if len(selected) != expected_frames:
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

    def _import_pool_capture(self, record: WindowRecord) -> None:
        """Make every completed capture available before queueing reconstruction."""
        with self._pool_lock:
            self._import_pool_capture_locked(record)

    def _import_pool_capture_locked(self, record: WindowRecord) -> None:
        role = record.role
        pool = self._frame_pool()
        if not record.pool_imported:
            try:
                snapshot = validate_snapshot(
                    record.snapshot_uri,
                    record.manifest_sha256,
                    (role.capture_config.snapshot_root,),
                    role.capture_config.snapshot_limits,
                )
            except BaseException as error:
                raise FramePoolError(
                    f"frame pool snapshot validation failed: "
                    f"{type(error).__name__}: {error}"
                ) from error
            odom_span = self._frame_pool_odom_span(snapshot)
            planned = None
            if odom_span is not None:
                with self._lock:
                    planned = self._plan_map_epoch_locked(odom_span)
                    current_epoch = self._map_epoch
            try:
                if planned is not None and planned[0] != current_epoch:
                    pool.clear()
                    self._write_journal(
                        {
                            "event": "frame_pool_epoch_reset",
                            "session_id": self._session_id,
                            "window_index": record.window_index,
                            "previous_map_epoch": current_epoch,
                            "map_epoch": planned[0],
                        }
                    )
                pool.import_window(snapshot, record.window_index)
            except BaseException:
                self._pool = None
                raise
            if planned is not None:
                with self._lock:
                    self._apply_map_epoch_locked(
                        planned[1], planned[2], planned[3], planned[0]
                    )
                    record.pool_epoch = self._map_epoch
            else:
                with self._lock:
                    record.pool_epoch = self._map_epoch
            record.pool_imported = True
            with self._lock:
                self._pool_eligible_frames = sum(
                    math.isfinite(entry.score) and entry.score > 0 for entry in pool.entries
                )
        else:
            with self._lock:
                if record.pool_epoch != self._map_epoch:
                    raise FramePoolError(
                        "frame pool window belongs to a stale odometry epoch"
                    )

    def _prepare_pool_reconstruction(self, record: WindowRecord) -> bool:
        """Reselect the frozen target count from the complete pool on every attempt."""
        with self._pool_lock:
            return self._prepare_pool_reconstruction_locked(record)

    def _prepare_pool_reconstruction_locked(self, record: WindowRecord) -> bool:
        self._import_pool_capture_locked(record)
        role = record.role
        pool = self._frame_pool()
        original_uri = record.snapshot_uri

        selected = pool.select(record.effective.window_frames)
        record.pool_size = len(pool.entries)
        if len(selected) < record.effective.window_frames:
            # Inputs are safely retained in the pool. Keep capturing instead of
            # failing/retrying the same undersized input and blocking the FIFO.
            cleanup_error = self._remove_snapshot(record)
            if cleanup_error is not None:
                raise FramePoolError(f"buffered source cleanup failed: {cleanup_error}")
            with self._lock:
                self._window_terminal(record, WINDOW_BUFFERED, "")
                self._write_journal({
                    "event": "frame_pool_buffered", "session_id": self._session_id,
                    "window_index": record.window_index, "pool_size": record.pool_size,
                    "eligible_frames": len(selected), "required_frames": record.effective.window_frames,
                })
            return False
        revision = self._pool_revision
        pool.write_selection(revision, selected)

        scene_id = f"cs-{self._session_id}-p{revision}"
        destination = (
            Path(role.capture_config.snapshot_root)
            / f"cs-{self._session_id}-w{revision}-{secrets.token_hex(8)}"
        )
        uri, manifest_sha256 = pool.build_synthetic_snapshot(
            revision,
            selected,
            destination,
            scene_id,
            camera_name=role.capture_config.camera_name,
            image_topic=role.capture_config.image_topic,
            camera_role=role.role,
            camera_frame_id=role.capture_config.camera_frame_id,
        )
        try:
            validate_snapshot(
                uri, manifest_sha256, (role.capture_config.snapshot_root,),
                role.capture_config.snapshot_limits,
            )
        except BaseException:
            self._remove_snapshot(SimpleNamespace(snapshot_uri=uri, role=role))
            raise
        record.pool_revision = revision
        record.pool_size = len(pool.entries)
        record.pool_selected_indices = [
            int(entry.global_index) for entry in selected
        ]
        record.pool_confidence_summary = pool.confidence_summary(selected)
        record.snapshot_uri = uri
        record.manifest_sha256 = manifest_sha256
        record.scene_id = scene_id
        record.snapshot_bytes = self._snapshot_bytes_from_uri(uri)
        self._pool_revision += 1
        # 只有合成快照完整生成后才允许删除源快照；构建失败保留原物可重试。
        cleanup_error = self._remove_snapshot(
            SimpleNamespace(snapshot_uri=original_uri, role=role)
        )
        if cleanup_error is not None:
            self._write_journal(
                {
                    "event": "frame_pool_source_cleanup_failed",
                    "session_id": self._session_id,
                    "window_index": record.window_index,
                    "snapshot_uri": original_uri,
                    "reason": cleanup_error,
                }
            )
        self._write_journal(
            {
                "event": "frame_pool_selection",
                "window_index": record.window_index,
                "target_frames": record.effective.window_frames,
                "session_id": self._session_id,
                "pool_revision": revision,
                "pool_size": record.pool_size,
                "pool_epoch": record.pool_epoch,
                "selected_global_indices": record.pool_selected_indices,
                "confidence_summary": record.pool_confidence_summary,
                "synthetic_snapshot_uri": uri,
            }
        )

        return True

    def _update_pool_from_outcome(
        self, record: WindowRecord, outcome
    ) -> None:
        predictions_path = self._predictions_path(outcome)
        if predictions_path is None:
            self._write_journal(
                {
                    "event": "frame_pool_score_update_skipped",
                    "session_id": self._session_id,
                    "pool_revision": record.pool_revision,
                    "reason": "no local predictions.npz",
                }
            )
            return
        try:
            with self._pool_lock:
                if record.pool_epoch != self._map_epoch:
                    self._write_journal({
                        "event": "frame_pool_score_update_skipped",
                        "session_id": self._session_id,
                        "pool_revision": record.pool_revision,
                        "reason": "result belongs to an earlier pool epoch",
                    })
                    return
                self._frame_pool().update_from_predictions(
                    predictions_path,
                    record.pool_selected_indices,
                    record.pool_revision,
                    backend=self._config.reconstruction_backend,
                )
        except BaseException as error:
            self._write_journal(
                {
                    "event": "frame_pool_score_update_failed",
                    "session_id": self._session_id,
                    "pool_revision": record.pool_revision,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )


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
            if self._reconstruction_mode == RECONSTRUCTION_MODE_FRAME_POOL:
                try:
                    if not self._prepare_pool_reconstruction(record):
                        return
                except FramePoolCapacityError as error:
                    with self._lock:
                        self._pool_capacity_blocked = True
                        if not self._stopping:
                            self._state = SESSION_DISK_PAUSED
                    self._handle_reconstruct_failure(
                        record,
                        "frame pool capacity reached; start a new session: "
                        f"{error}",
                    )
                    return
                except BaseException as error:
                    self._handle_reconstruct_failure(
                        record,
                        "frame pool prepare failed: "
                        f"{type(error).__name__}: {error}",
                    )
                    return
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
        alignment = self._align_window(record, outcome)
        formal_error = self._formalize_window(record, outcome, alignment)
        if formal_error == _FORMAL_SKIPPED_STOPPING:
            with self._lock:
                self._window_terminal(record, WINDOW_CANCELLED, "stop_all")
            return
        if formal_error is not None:
            self._handle_reconstruct_failure(record, formal_error)
            return
        if self._uses_g4():
            # Only committed, verified inputs can enter G4. The process scans
            # complete records and coalesces arrivals while a solve is running.
            with self._lock:
                if not self._stopping:
                    try:
                        if self._g4_process is not None and self._g4_process.process.poll() is not None:
                            self._g4_process.close()
                            self._g4_process = None
                        if self._g4_process is None:
                            self._g4_process = G4BackendProcess(self._formal_records_root(), self._session_id,
                                                                profile=self._backend_profile())
                    except Exception as error:
                        self._last_error = "G4 background start failed: " + str(error)
        if self._uses_shared_A() and not valid_sim3_alignment(alignment):
            record.alignment = alignment
            with self._lock:
                self._formal_windows += 1
            self._handle_reconstruct_failure(record, "A alignment failed: " + alignment.get("reason", "unknown"))
            return
        if (
            self._reconstruction_mode == RECONSTRUCTION_MODE_FRAME_POOL
            and record.pool_revision >= 0
        ):
            self._update_pool_from_outcome(record, outcome)
        with self._lock:
            record.alignment = alignment
            if self._uses_shared_A():
                self._shared_reference = record.shared_reference_candidate
                record.shared_reference_candidate = None
            self._last_alignment_state = (
                COORDINATE_MODEL_LOCAL_UNALIGNED
                if alignment.get("quality_passed") is False
                else alignment["coordinate_state"]
            )
            self._succeeded += 1
            self._formal_windows += 1
            self._window_terminal(record, WINDOW_SUCCEEDED, "")
            keep = record.effective.keep_snapshot_on_success
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
        if not keep:
            error = self._remove_snapshot(record)
            if error is not None:
                with self._lock:
                    self._last_error = (
                        f"window {record.window_index} succeeded but snapshot "
                        f"cleanup failed: {error}"
                    )
        if (
            self._reconstruction_mode == RECONSTRUCTION_MODE_FRAME_POOL
            and self._config.expected_result_root
        ):
            cleanup = prune_frame_pool_revisions(
                result_root=self._config.result_root,
                worker_result_root=self._config.expected_result_root,
                session_id=self._session_id,
                keep_revisions=self._config.retention_frame_pool_revisions,
            )
            if cleanup["errors"]:
                with self._lock:
                    self._last_error = "frame pool retention failed: " + "; ".join(
                        cleanup["errors"]
                    )

    def _align_window(self, record: WindowRecord, outcome) -> dict[str, Any]:
        """Align each window using its odometry, independently of frame overlap."""
        base = {
            "coordinate_state": COORDINATE_MODEL_LOCAL_UNALIGNED,
            "map_id": "",
            "scale": None,
            "rotation": None,
            "translation": None,
            "residual_rms": None,
            "correspondence_count": 0,
            "reference_frame_id": "",
            "unreliable": True,
            "reason": "",
        }
        if self._reconstruction_mode == RECONSTRUCTION_MODE_FRAME_POOL:
            base["reason"] = (
                "frame_pool global reconstruction; per-window odom Sim(3) "
                "alignment is not applied"
            )
            return base
        predictions_path = self._predictions_path(outcome)
        odom_reason = ""
        if self._uses_g4():
            base.update(method=BACKEND_G4, pending=True, quality_passed=False,
                        geometry_accepted=False, reason="waiting_for_G4_map_revision")
            return base
        if self._uses_shared_A():
            base["method"] = BACKEND_A
            base["quality_passed"] = False
            try:
                if predictions_path is None:
                    raise ValueError("missing predictions.npz")
                snapshot = validate_snapshot(
                    record.snapshot_uri, record.manifest_sha256,
                    (record.role.capture_config.snapshot_root,), record.role.capture_config.snapshot_limits,
                )
                payload, reference = align_shared_points(
                    snapshot, predictions_path, model_frame_ids=record.model_frame_ids,
                    selected_indices=self._selected_frame_indices(outcome),
                    window_index=record.window_index, capture_generation=record.capture_generation,
                    map_id=f"{self._session_id}-m0", reference=self._shared_reference,
                    odom_limits=(self._config.odom_min_baseline_m,
                                 self._config.odom_max_orientation_residual_deg,
                                 self._config.odom_max_normalized_residual_rms),
                    odom_jump_m=self._config.odom_reset_jump_m,
                )
                record.shared_reference_candidate = reference
                return payload
            except Exception as error:
                base["reason"] = f"{type(error).__name__}: {error}"
                return base
        if (
            self._config.odom_local_enabled
            and predictions_path is not None
        ):
            try:
                snapshot = validate_snapshot(
                    record.snapshot_uri,
                    record.manifest_sha256,
                    (record.role.capture_config.snapshot_root,),
                    record.role.capture_config.snapshot_limits,
                )
                odom = align_window(
                    snapshot,
                    predictions_path,
                    min_correspondences=self._config.odom_min_correspondences,
                    selected_frame_indices=self._selected_frame_indices(outcome),
                    min_reference_baseline_m=self._config.odom_min_baseline_m,
                    max_orientation_residual_deg=(
                        self._config.odom_max_orientation_residual_deg
                    ),
                    max_normalized_residual_rms=(
                        self._config.odom_max_normalized_residual_rms
                    ),
                )
                if odom.coordinate_state == COORDINATE_ODOM_LOCAL_APPROX:
                    model_frame_ids = tuple(getattr(record, "model_frame_ids", ()))
                    payload = dict(base)
                    payload.update(
                        {
                            "coordinate_state": odom.coordinate_state,
                            "method": "odom_pose_constrained_sim3",
                            "reference_frame_id": odom.reference_frame_id,
                            "correspondence_count": odom.correspondence_count,
                            "scale": odom.scale,
                            "rotation": odom.rotation,
                            "translation": odom.translation,
                            "residual_rms": odom.residual_rms,
                            "normalized_residual_rms": odom.normalized_residual_rms,
                            "orientation_residual_deg": odom.orientation_residual_deg,
                            "reference_baseline_m": odom.reference_baseline_m,
                            "reference_first_position": (
                                list(odom.reference_first_position)
                                if odom.reference_first_position is not None
                                else None
                            ),
                            "reference_last_position": (
                                list(odom.reference_last_position)
                                if odom.reference_last_position is not None
                                else None
                            ),
                            "reference_first_stamp": (
                                list(odom.reference_first_stamp)
                                if odom.reference_first_stamp is not None
                                else None
                            ),
                            "reference_last_stamp": (
                                list(odom.reference_last_stamp)
                                if odom.reference_last_stamp is not None
                                else None
                            ),
                            "reference_samples": list(odom.reference_samples),
                            "metric_scale": True,
                            "quality_passed": True,
                            "pending": False,
                            "unreliable": True,
                            "merge_frame_indices": [
                                index
                                for index, frame_id in enumerate(model_frame_ids)
                            ],
                        }
                    )
                    payload["overlap_model_frames"] = len(model_frame_ids) - len(
                        payload["merge_frame_indices"]
                    )
                    return payload
                odom_reason = odom.reason
            except BaseException as error:
                odom_reason = f"odom alignment failed: {type(error).__name__}: {error}"

        if not self._config.odom_local_enabled:
            base["reason"] = "odom_local disabled by profile"
            return base
        if predictions_path is None:
            base["reason"] = "result manifest uri has no local predictions.npz"
            return base
        base["reason"] = odom_reason or "odom alignment unavailable"
        # map epoch 只在真实性门禁通过后推进（见 _formalize_window），
        # 门禁失败的窗口不得影响后续地图纪元。
        # 对齐 sidecar 不再写入 worker 已原子提交的结果目录；它随正式记录
        # 一起发布到 sessions/<session>/window-NNNN/（见 _formalize_window）。
        return base

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
            "captured_frames": (
                len(getattr(record, "pool_selected_indices", ()) or ())
                if getattr(record, "pool_revision", -1) >= 0
                else record.captured_frames
            ),
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
        planned_epoch = None
        planned_frame_id = None
        planned_last_position = None
        planned_last_stamp = None
        with self._lock:
            session_id = self._session_id
            timeline = [dict(item) for item in self._config_timeline]
            stopping = self._stopping
            if (
                not stopping
                and alignment.get("coordinate_state") == COORDINATE_ODOM_LOCAL_APPROX
                and alignment.get("quality_passed") is not False
                and not self._uses_shared_A()
            ):
                (
                    planned_epoch,
                    planned_frame_id,
                    planned_last_position,
                    planned_last_stamp,
                ) = self._plan_map_epoch_locked(alignment)
                alignment["map_id"] = f"{session_id}-m{planned_epoch}"
            model_frame_ids = tuple(getattr(record, "model_frame_ids", ()))
            if model_frame_ids and valid_sim3_alignment(alignment) and alignment.get("quality_passed") is not False:
                # A successful inference need not have contributed to this map.
                prior_ids = set()
                for candidate in self._windows:
                    prior = candidate.alignment or {}
                    if (
                        candidate is record or candidate.state != WINDOW_SUCCEEDED
                        or prior.get("map_id") != alignment.get("map_id")
                        or not valid_sim3_alignment(prior)
                        or prior.get("quality_passed") is False
                    ):
                        continue
                    indices = prior.get("merge_frame_indices", range(len(candidate.model_frame_ids)))
                    prior_ids.update(candidate.model_frame_ids[index] for index in indices)
                alignment["merge_frame_indices"] = [
                    index for index, frame_id in enumerate(model_frame_ids)
                    if frame_id not in prior_ids
                ]
                alignment["overlap_model_frames"] = len(model_frame_ids) - len(
                    alignment["merge_frame_indices"]
                )
        if stopping:
            return _FORMAL_SKIPPED_STOPPING
        checkpoint = payload.get("checkpoint") or {}
        stage_timing = payload.get("stage_timing") or {}
        worker_exit = payload.get("worker_exit") or {}
        frame_pool_mode = self._reconstruction_mode == RECONSTRUCTION_MODE_FRAME_POOL
        formal_window_index = (
            record.pool_revision if frame_pool_mode else record.window_index
        )
        formal_frame_count = (
            len(record.pool_selected_indices)
            if frame_pool_mode
            else record.captured_frames
        )
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": session_id,
            "window_index": int(formal_window_index),
            "scene_id": record.scene_id,
            "result_manifest_uri": outcome.result_manifest_uri,
            "result_manifest_sha256": outcome.result_manifest_sha256,
            "snapshot_manifest_uri": record.snapshot_uri,
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "captured_frames": int(formal_frame_count),
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
            "reconstruction_mode": self._reconstruction_mode,
            "pool_revision": (
                int(record.pool_revision)
                if frame_pool_mode
                else None
            ),
            "pool_epoch": (
                int(record.pool_epoch)
                if frame_pool_mode
                else None
            ),
            "pool_size": (
                int(record.pool_size)
                if frame_pool_mode
                else None
            ),
            "pool_selected_indices": (
                list(record.pool_selected_indices)
                if frame_pool_mode
                else None
            ),
            "confidence_summary": (
                dict(record.pool_confidence_summary)
                if frame_pool_mode
                else None
            ),
            "alignment": alignment,
            "depth_metrics": depth_metrics,
            "depth_metrics_error": depth_metrics_error,
        }
        if frame_pool_mode:
            provenance["source_window_index"] = int(record.window_index)
        try:
            self._ensure_formal_session_dir(session_id)
            final_path, provenance_sha256 = self._result_publisher(
                self._formal_records_root(),
                session_id,
                formal_window_index,
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
        # 只有原子发布成功后才提交地图纪元；发布失败/重试保持同一候选纪元。
        if planned_epoch is not None:
            with self._lock:
                self._apply_map_epoch_locked(
                    planned_frame_id,
                    planned_last_position,
                    planned_last_stamp,
                    planned_epoch,
                )
        record.formal_result_uri = str(final_path)
        record.formal_provenance_sha256 = provenance_sha256
        return None

    @staticmethod
    def _alignment_fields(alignment):
        """读取 OdomLocalAlignment 或同名字段 dict 的轨迹字段。"""
        if isinstance(alignment, Mapping):
            return (
                alignment.get("reference_frame_id"),
                alignment.get("reference_first_position"),
                alignment.get("reference_last_position"),
                alignment.get("reference_first_stamp"),
                alignment.get("reference_last_stamp"),
            )
        return (
            alignment.reference_frame_id,
            alignment.reference_first_position,
            alignment.reference_last_position,
            alignment.reference_first_stamp,
            alignment.reference_last_stamp,
        )

    def _plan_map_epoch_locked(
        self, alignment
    ) -> tuple[int, str, tuple | None, tuple | None]:
        """计算（不提交）该对齐窗口对应的地图纪元与 odom 状态。

        只读取当前状态；调用方在同一锁内持有快照，发布成功后用
        ``_apply_map_epoch_locked`` 提交。
        """
        (
            frame_id,
            first,
            last,
            first_stamp,
            last_stamp,
        ) = self._alignment_fields(alignment)
        epoch = self._map_epoch
        if self._odom_reference_frame_id is None:
            return epoch, frame_id, last, last_stamp
        if frame_id != self._odom_reference_frame_id:
            return epoch + 1, frame_id, last, last_stamp
        if (
            first_stamp is not None
            and self._last_odom_stamp is not None
            and first_stamp < self._last_odom_stamp
        ):
            # A rolling window starts in the past but must advance its end.
            if last_stamp is None or last_stamp <= self._last_odom_stamp:
                return epoch + 1, frame_id, last, last_stamp
            # Compare the same odom timestamp, never two ends of an overlap.
            samples = alignment.get("reference_samples", ()) if isinstance(alignment, Mapping) else ()
            first = next((sample["position"] for sample in samples
                          if tuple(sample["stamp"]) == tuple(self._last_odom_stamp)), None)
        if first is not None and self._last_odom_position is not None:
            dx = first[0] - self._last_odom_position[0]
            dy = first[1] - self._last_odom_position[1]
            dz = first[2] - self._last_odom_position[2]
            jump = math.sqrt(dx * dx + dy * dy + dz * dz)
            if jump > float(self._config.odom_reset_jump_m):
                return epoch + 1, frame_id, last, last_stamp
        return epoch, frame_id, last, last_stamp

    def _apply_map_epoch_locked(
        self,
        frame_id: str,
        last_position: tuple | None,
        last_stamp: tuple | None,
        epoch: int,
    ) -> None:
        self._map_epoch = int(epoch)
        self._odom_reference_frame_id = frame_id
        self._last_odom_position = last_position
        self._last_odom_stamp = last_stamp

    def _advance_map_epoch_locked(self, alignment) -> None:
        """立即提交该窗口的地图纪元（测试与历史调用路径）。"""
        epoch, frame_id, last_position, last_stamp = self._plan_map_epoch_locked(
            alignment
        )
        self._apply_map_epoch_locked(
            frame_id, last_position, last_stamp, epoch
        )

    def _uses_shared_A(self) -> bool:
        return (
            self._config.alignment_backend == BACKEND_A
            and self._reconstruction_mode == RECONSTRUCTION_MODE_PER_WINDOW
        )

    def _uses_g4(self) -> bool:
        return (self._config.alignment_backend == BACKEND_G4
                and self._reconstruction_mode == RECONSTRUCTION_MODE_PER_WINDOW)


    def _pause_shared_failure(self, reason):
        with self._lock:
            if self._stopping:
                return
            self._alignment_fault = True
            self._state = SESSION_ALIGNMENT_PAUSED
            self._last_error = reason
            client = self._capture_client
            self._write_journal({
                "event": "shared_alignment_paused", "session_id": self._session_id,
                "reason": reason, "recovery": "stop_and_start_new_session",
            })
            self._wake_capture.set()
            self._wake_consume.set()
        if client is not None:
            try:
                accepted = client.pause()
            except Exception:
                accepted = False
            if not accepted:
                with self._lock:
                    self._last_error += "; capture pause was not acknowledged; no new goals will start"

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
        不再等于 snapshot 帧序号；此时对齐必须按 result.json 的
        ``selected_frames[].index`` 匹配 odometry 参考位置。读取失败或
        结构不符时返回 None，由 align_window 回退旧行为。
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

    def _frame_pool_odom_span(self, snapshot) -> dict[str, Any] | None:
        samples = []
        for valid, payload in zip(
            getattr(snapshot, "pose_valid", ()) or (),
            getattr(snapshot, "odometry_payloads", ()) or (),
        ):
            if not valid or not isinstance(payload, dict):
                continue
            try:
                header = payload["header"]
                frame_id = header["frame_id"]
                stamp = header["stamp"]
                position = payload["pose"]["position"]
                point = tuple(float(position[key]) for key in ("x", "y", "z"))
                stamp_key = (int(stamp["sec"]), int(stamp["nsec"]))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not isinstance(frame_id, str) or not frame_id:
                continue
            if not all(math.isfinite(value) for value in point):
                continue
            samples.append((frame_id, stamp_key, point))
        if not samples:
            return None
        frame_ids = {sample[0] for sample in samples}
        if len(frame_ids) != 1:
            raise FramePoolError("odometry reference frame changed within snapshot")
        for previous, current in zip(samples, samples[1:]):
            if current[1] < previous[1]:
                raise FramePoolError("odometry time moved backwards within snapshot")
            if math.dist(previous[2], current[2]) > float(
                self._config.odom_reset_jump_m
            ):
                raise FramePoolError("odometry position jumped within snapshot")
        return {
            "reference_frame_id": samples[0][0],
            "reference_first_stamp": samples[0][1],
            "reference_last_stamp": samples[-1][1],
            "reference_first_position": samples[0][2],
            "reference_last_position": samples[-1][2],
        }

    def _handle_reconstruct_failure(self, record: WindowRecord, error: str) -> None:
        # attempt_failures 是"失败尝试"总数：每次重建尝试失败各记一次，
        # 与终态 failed_windows（未恢复失败）分开，人工重试不清历史。
        with self._lock:
            self._attempt_failures += 1
        oom = bool(_FATAL_OOM_PATTERN.search(error or ""))
        memory_guard = bool(_MEMORY_GUARD_PATTERN.search(error or ""))
        if self._uses_shared_A() and not self._stopping and not memory_guard and not oom:
            with self._lock:
                self._failed += 1
                record.shared_reference_candidate = None
                self._window_terminal(record, WINDOW_FAILED, error)
            self._pause_shared_failure(error)
            return
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
                if record.effective.queue_mode == QUEUE_REALTIME_LATEST:
                    while len(self._fifo) > record.effective.queue_capacity:
                        dropped = self._fifo.pop(0)
                        self._fifo_bytes -= dropped.snapshot_bytes
                        self._remove_snapshot(dropped)
                        self._dropped += 1
                        dropped.state = WINDOW_CANCELLED
                        dropped.error = "retry_overflow"
                        dropped.done.set()
                        self._write_journal(
                            {
                                "event": "window_dropped",
                                "session_id": self._session_id,
                                "window_index": dropped.window_index,
                                "scene_id": dropped.scene_id,
                                "config_digest": dropped.effective.digest(),
                                "reason": dropped.error,
                            }
                        )
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
        if self._fatal_oom or (self._uses_shared_A() and self._alignment_fault):
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
    expected_frame_count = (
        len(getattr(record, "pool_selected_indices", ()) or ())
        if getattr(record, "pool_revision", -1) >= 0
        else record.captured_frames
    )
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


def cleanup_stale_snapshots(config: ContinuousConfig) -> list[str]:
    """启动清理：删除所有 scheduler 命名的未完成 snapshot。

    只处理 capture profile snapshot_root 的直接子目录且 basename 匹配
    当前或旧版 scheduler 命名。返回被删除的路径列表。
    """
    if config.alignment_backend in (BACKEND_A, BACKEND_G4):
        # A preserves failure evidence.
        return []
    removed = []
    seen_roots = set()
    for role in config.roles:
        root = Path(role.capture_config.snapshot_root)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or entry.is_symlink():
                continue
            if _SNAPSHOT_DIR_RE.fullmatch(entry.name) is None:
                continue
            try:
                import shutil
                shutil.rmtree(str(entry))
                removed.append(str(entry))
            except OSError:
                continue
    return removed


__all__ = [
    "ContinuousConfig",
    "ContinuousConfigError",
    "ContinuousSession",
    "EffectiveConfig",
    "JobStatus",
    "ReconstructionBackend",
    "SessionStatus",
    "WindowRecord",
    "cleanup_stale_snapshots",
    "QUEUE_LOSSLESS",
    "QUEUE_REALTIME_LATEST",
    "SESSION_DISK_PAUSED",
    "SESSION_FATAL_OOM",
    "SESSION_MEMORY_PAUSED",
    "SESSION_IDLE",
    "SESSION_PAUSED",
    "SESSION_RUNNING",
    "SESSION_STOPPED",
    "SESSION_STOPPING",
]
