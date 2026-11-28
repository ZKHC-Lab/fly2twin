"""State constants, the window record and the status data classes of the continuous session,
plus the two interfaces it drives (reconstruction backend and capture client).

Moved out of ``continuous_session``, which still re-exports every name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading

from .odom_local import (
    COORDINATE_MODEL_LOCAL_UNALIGNED,
)
from .g4.backend_process import BACKEND_G4

from .action_types import (
    CameraRole,
    CaptureActionOutcome,
    ReconstructActionOutcome,
)
from .continuous_session_config import RECONSTRUCTION_MODE_PER_WINDOW, EffectiveConfig


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


# 窗口状态
WINDOW_CAPTURING = "CAPTURING"


WINDOW_PENDING = "PENDING"


WINDOW_RUNNING = "RUNNING"


WINDOW_SUCCEEDED = "SUCCEEDED"


WINDOW_FAILED = "FAILED"


WINDOW_CANCELLED = "CANCELLED"


JOB_PENDING = "PENDING"


JOB_RUNNING = "RUNNING"


JOB_SUCCEEDED = "SUCCEEDED"


JOB_FAILED = "FAILED"


JOB_CANCELLED = "CANCELLED"


_JOB_TERMINAL = frozenset((JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED))


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
