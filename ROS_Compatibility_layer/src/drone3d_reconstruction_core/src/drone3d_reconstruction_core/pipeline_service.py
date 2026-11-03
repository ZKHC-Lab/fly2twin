"""ROS 无关的相机管线上层服务核心。

本模块只做一件事：把已存在的 CaptureSegment / ReconstructSegment Action
编排成一个可观测的 camera-role 任务管线。

  - 数据源由 profile 中的 camera_role 配置，服务只引用这些已配置角色；
  - START_TASK 先通过选中角色的 Capture Action 封存一个不可变 snapshot，
    再把 snapshot_uri + manifest_sha256 原样提交给 Reconstruct Action；
  - 上层服务自身不订阅图像、不读取 rosbag、不启动 worker、不复制采集或
    重建实现；
  - 同一时刻只允许一个管线任务（max_running_cuda_jobs=1 的上层投影）。

模块不导入 rospy/actionlib：Action 客户端由 pipeline_service_node 注入，
核心逻辑可被 CPU 单元测试完整覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from .capture import (
    CaptureConfig,
    CaptureError,
    validate_capture_goal,
)

# 与 drone3d_reconstruction_msgs/PipelineCommand 响应常量保持一致
OK = 0
INVALID_COMMAND = 1
UNKNOWN_ROLE = 2
BUSY = 3
NO_TASK = 4
TASK_FAILED = 5
INTERNAL_ERROR = 6
NOT_READY = 7
INVALID_ARGUMENT = 8

# 管线状态机终态集合
STATE_IDLE = "IDLE"
STATE_CAPTURING = "CAPTURING"
STATE_RECONSTRUCTING = "RECONSTRUCTING"
STATE_STOPPING = "STOPPING"
STATE_SUCCEEDED = "SUCCEEDED"
STATE_FAILED = "FAILED"
STATE_CANCELLED = "CANCELLED"
_TERMINAL_STATES = frozenset((STATE_SUCCEEDED, STATE_FAILED, STATE_CANCELLED))

# 归一化后的 actionlib 终态名
ACTION_SUCCEEDED = "SUCCEEDED"
ACTION_CANCELLED = "CANCELLED"
ACTION_FAILED = "FAILED"


class PipelineConfigError(ValueError):
    """管线服务配置非法。"""


@dataclass(frozen=True)
class CameraRole:
    """一个已配置的相机角色：role + 现有 Capture Action 与 capture profile。"""

    role: str
    capture_action: str
    capture_config: CaptureConfig


@dataclass(frozen=True)
class PipelineConfig:
    """管线服务配置（不含任何运行时状态）。"""

    roles: tuple[CameraRole, ...]
    reconstruct_action: str
    service_name: str
    diagnostics_topic: str
    diagnostics_period_sec: float
    action_connect_timeout_sec: float
    capture_wait_margin_sec: float
    reconstruct_wait_timeout_sec: float
    reconstruct_frame_group_size: int

    def role_by_name(self, role: str) -> CameraRole | None:
        for candidate in self.roles:
            if candidate.role == role:
                return candidate
        return None

    def validate(self) -> None:
        seen = set()
        for role in self.roles:
            if role.role in seen:
                raise PipelineConfigError(f"duplicate camera role: {role.role}")
            seen.add(role.role)
        if not self.roles:
            raise PipelineConfigError("at least one camera role is required")
        for name in ("service_name", "diagnostics_topic", "reconstruct_action"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PipelineConfigError(f"{name} must be a nonempty string")
        for name in (
            "diagnostics_period_sec",
            "action_connect_timeout_sec",
            "capture_wait_margin_sec",
            "reconstruct_wait_timeout_sec",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not float(value) == float(value)
                or float(value) <= 0
                or float(value) in (float("inf"), float("-inf"))
            ):
                raise PipelineConfigError(f"{name} must be positive and finite")
        if (
            isinstance(self.reconstruct_frame_group_size, bool)
            or not isinstance(self.reconstruct_frame_group_size, int)
            or self.reconstruct_frame_group_size <= 0
        ):
            raise PipelineConfigError(
                "reconstruct_frame_group_size must be a positive integer"
            )


@dataclass(frozen=True)
class TaskParameters:
    """一次 START_TASK 的已校验参数。"""

    role: str
    scene_id: str
    sample_fps: float
    max_frames: int
    max_duration_sec: float


@dataclass(frozen=True)
class CaptureActionOutcome:
    """归一化的 CaptureSegment Action 终态。"""

    action_state: str
    error_code: int
    message: str
    scene_id: str
    camera_name: str
    image_topic: str
    snapshot_uri: str
    manifest_uri: str
    manifest_sha256: str
    frame_count: int


@dataclass(frozen=True)
class ReconstructActionOutcome:
    """归一化的 ReconstructSegment Action 终态。

    camera_role/camera_topic/camera_frame_id/snapshot_manifest_sha256 由
    ROS 层从正式 result.json 读取（本任务身份校验字段），读取失败时为空串，
    上层会据此拒绝成功终态。
    """

    action_state: str
    error_code: int
    message: str
    scene_id: str
    result_manifest_uri: str
    result_manifest_sha256: str
    scene_glb_uri: str
    frame_count: int
    wall_time_sec: float
    camera_role: str = ""
    camera_topic: str = ""
    camera_frame_id: str = ""
    snapshot_manifest_sha256: str = ""
    result_scene_id: str = ""
    result_actual_frame_count: int = 0


@dataclass(frozen=True)
class CommandOutcome:
    """服务响应负载（与 PipelineCommandResponse 字段对应）。"""

    error_code: int
    message: str
    role: str
    state: str
    task_id: int
    scene_id: str
    capture_action: str
    reconstruct_action: str
    snapshot_uri: str
    manifest_sha256: str
    result_manifest_uri: str
    result_manifest_sha256: str
    captured_frames: int
    reconstruction_frames: int
    reconstruction_progress: float
    reconstruction_stage: str
    last_error: str
    task_wall_time_sec: float

    @staticmethod
    def make(
        error_code: int,
        message: str,
        *,
        role: str = "",
        state: str = STATE_IDLE,
        task_id: int = 0,
        scene_id: str = "",
        capture_action: str = "",
        reconstruct_action: str = "",
        snapshot_uri: str = "",
        manifest_sha256: str = "",
        result_manifest_uri: str = "",
        result_manifest_sha256: str = "",
        captured_frames: int = 0,
        reconstruction_frames: int = 0,
        reconstruction_progress: float = 0.0,
        reconstruction_stage: str = "",
        last_error: str = "",
        task_wall_time_sec: float = 0.0,
    ) -> "CommandOutcome":
        return CommandOutcome(
            error_code=error_code,
            message=message,
            role=role,
            state=state,
            task_id=task_id,
            scene_id=scene_id,
            capture_action=capture_action,
            reconstruct_action=reconstruct_action,
            snapshot_uri=snapshot_uri,
            manifest_sha256=manifest_sha256,
            result_manifest_uri=result_manifest_uri,
            result_manifest_sha256=result_manifest_sha256,
            captured_frames=captured_frames,
            reconstruction_frames=reconstruction_frames,
            reconstruction_progress=reconstruction_progress,
            reconstruction_stage=reconstruction_stage,
            last_error=last_error,
            task_wall_time_sec=task_wall_time_sec,
        )


class CaptureActionClient:
    """选定角色的 Capture Action 客户端（由 ROS 层实现并注入）。"""

    def wait_for_server(self, timeout_sec: float) -> bool:
        raise NotImplementedError

    def send_goal(self, params: TaskParameters, feedback_cb) -> None:
        raise NotImplementedError

    def wait_for_result(self, timeout_sec: float) -> bool:
        raise NotImplementedError

    def cancel_goal(self) -> None:
        raise NotImplementedError

    def outcome(self) -> CaptureActionOutcome:
        raise NotImplementedError


class ReconstructActionClient:
    """Reconstruct Action 客户端（由 ROS 层实现并注入）。"""

    def wait_for_server(self, timeout_sec: float) -> bool:
        raise NotImplementedError

    def send_goal(
        self,
        snapshot_uri: str,
        manifest_sha256: str,
        feedback_cb,
        image_resolution: int = 0,
        set_image_resolution: bool = False,
        worker_timeout_sec: float = 0.0,
        set_worker_timeout: bool = False,
    ) -> None:
        raise NotImplementedError

    def wait_for_result(self, timeout_sec: float) -> bool:
        raise NotImplementedError

    def cancel_goal(self) -> None:
        raise NotImplementedError

    def outcome(self) -> ReconstructActionOutcome:
        raise NotImplementedError


@dataclass
class _TaskRecord:
    """一个活跃或最近完成的上层管线任务。"""

    task_id: int
    role: CameraRole
    params: TaskParameters
    started_monotonic: float
    state: str = STATE_CAPTURING
    finished_monotonic: float | None = None
    captured_frames: int = 0
    reconstruction_frames: int = 0
    reconstruction_progress: float = 0.0
    reconstruction_stage: str = ""
    snapshot_uri: str = ""
    manifest_sha256: str = ""
    result_manifest_uri: str = ""
    result_manifest_sha256: str = ""
    last_error: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    watchdog_error: str = ""
    slot_held: bool = False
    active_client: Any = None
    lower_goal_sent: bool = False
    capture_client: CaptureActionClient | None = None
    reconstruct_client: ReconstructActionClient | None = None


class CameraPipelineService:
    """上层管线服务：选源、启动、查询、取消与诊断快照。"""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        capture_client_factory: Callable[[CameraRole], CaptureActionClient],
        reconstruct_client_factory: Callable[[], ReconstructActionClient],
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be PipelineConfig")
        config.validate()
        if not callable(capture_client_factory) or not callable(
            reconstruct_client_factory
        ):
            raise TypeError("action client factories must be callable")
        self._config = config
        self._capture_client_factory = capture_client_factory
        self._reconstruct_client_factory = reconstruct_client_factory
        self._thread_factory = thread_factory
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._selected_role: CameraRole | None = None
        self._task: _TaskRecord | None = None
        self._next_task_id = 1
        self._shutting_down = False

    @property
    def config(self) -> PipelineConfig:
        return self._config

    def begin_shutdown(self) -> None:
        """进入关停态：立即拒绝新 START/SELECT，GET_STATUS/CANCEL 仍可用。"""
        with self._lock:
            self._shutting_down = True

    def select_source(self, role_name: str) -> CommandOutcome:
        """选择后续 START_TASK 使用的相机角色。"""
        role = self._resolve_role(role_name)
        if role is None:
            return CommandOutcome.make(
                UNKNOWN_ROLE,
                f"unknown camera role {role_name!r}; configured roles: "
                f"{sorted(candidate.role for candidate in self._config.roles)}",
            )
        with self._lock:
            if self._shutting_down:
                return self._status_locked(
                    INTERNAL_ERROR, "pipeline service is shutting down"
                )
            task = self._task
            if task is not None and (
                task.state not in _TERMINAL_STATES or task.slot_held
            ):
                return self._status_locked(
                    BUSY,
                    "cannot change source while a task is running or its "
                    "lower action has not reached a terminal state",
                )
            self._selected_role = role
            return self._status_locked(OK, f"selected camera role {role.role!r}")

    def start_task(
        self,
        role_name: str,
        scene_id: str,
        sample_fps: float,
        max_frames: int,
        max_duration_sec: float,
    ) -> CommandOutcome:
        """启动一次 Capture→Reconstruct 管线任务并立即返回。

        任务在线程中执行；GET_STATUS 查询进度与错误。同一时刻只允许一个
        管线任务（上层遵守 max_running_cuda_jobs=1）。
        """
        if role_name:
            role = self._resolve_role(role_name)
        else:
            with self._lock:
                role = self._selected_role
        if role is None:
            if role_name:
                message = (
                    f"unknown camera role {role_name!r}; configured roles: "
                    f"{sorted(candidate.role for candidate in self._config.roles)}"
                )
                return CommandOutcome.make(UNKNOWN_ROLE, message)
            return CommandOutcome.make(
                INVALID_ARGUMENT,
                "no camera role selected; call SELECT_SOURCE first or pass role",
            )

        try:
            params = self._validate_params(role, scene_id, sample_fps, max_frames, max_duration_sec)
        except ValueError as error:
            return CommandOutcome.make(INVALID_ARGUMENT, str(error), role=role.role)

        with self._lock:
            if self._shutting_down:
                return self._status_locked(
                    INTERNAL_ERROR, "pipeline service is shutting down"
                )
            task = self._task
            if task is not None and (
                task.state not in _TERMINAL_STATES or task.slot_held
            ):
                return self._status_locked(
                    BUSY,
                    f"task {task.task_id} is already {task.state}; "
                    "max_running_cuda_jobs=1; slot held until lower action "
                    "terminal state",
                )
            task_id = self._next_task_id
            self._next_task_id += 1
            record = _TaskRecord(
                task_id=task_id,
                role=role,
                params=params,
                started_monotonic=self._monotonic(),
            )
            self._task = record
            self._selected_role = role

        thread = None
        try:
            thread = self._thread_factory(target=self._run_task, args=(record,))
            if not isinstance(thread, threading.Thread):
                raise TypeError("thread_factory must return a Thread")
            if thread.ident is not None or thread.is_alive():
                raise RuntimeError("thread_factory returned an already-started Thread")
            thread.daemon = True
            thread.start()
        except BaseException as error:
            with self._lock:
                if self._task is record:
                    self._set_terminal_locked(
                        record,
                        STATE_FAILED,
                        f"task dispatch failed: {type(error).__name__}: {error}",
                    )
            return self._status_locked(
                INTERNAL_ERROR, f"task {record.task_id} could not be dispatched"
            )
        with self._lock:
            return self._status_locked(OK, f"task {task_id} started for role {role.role!r}")

    def get_status(self) -> CommandOutcome:
        """查询当前状态与最近错误。"""
        with self._lock:
            return self._status_locked(OK, "status")

    def cancel_task(self) -> CommandOutcome:
        """取消当前非终态任务并等待下层 Action 给出终态证明。

        上层槽位在下层 Action 到达 actionlib 终态之前保持占用：状态先进入
        STOPPING，新 START_TASK 会收到 BUSY。只有读取到下层终态后，任务才
        收敛为 CANCELLED/FAILED 并释放槽位。
        """
        with self._lock:
            task = self._task
            if task is None or task.state in _TERMINAL_STATES:
                message = (
                    "no active task to cancel"
                    if task is None
                    else f"task {task.task_id} already in terminal state {task.state}"
                )
                return self._status_locked(OK, message)
            task.cancel_event.set()
            task.state = STATE_STOPPING
            task.last_error = (
                "cancel requested; waiting for lower action terminal state"
            )
            capture_client = task.capture_client
            reconstruct_client = task.reconstruct_client
        if reconstruct_client is not None:
            reconstruct_client.cancel_goal()
        elif capture_client is not None:
            capture_client.cancel_goal()
        with self._lock:
            return self._status_locked(
                OK,
                f"cancel requested for task {task.task_id}; "
                "waiting for lower action terminal state",
            )

    def wait_for_task_terminal(self, timeout_sec: float) -> bool:
        """阻塞等待当前任务收敛到终态（最多 timeout_sec）。"""
        with self._lock:
            task = self._task
        if task is None:
            return True
        return task.done.wait(timeout_sec)

    def diagnostics_snapshot(self) -> dict[str, Any]:
        """线程安全的诊断快照。"""
        with self._lock:
            task = self._task
            return {
                "selected_role": (
                    self._selected_role.role if self._selected_role is not None else None
                ),
                "roles": [
                    {
                        "role": role.role,
                        "capture_action": role.capture_action,
                        "image_topic": role.capture_config.image_topic,
                        "camera_frame_id": role.capture_config.camera_frame_id,
                        "camera_name": role.capture_config.camera_name,
                    }
                    for role in self._config.roles
                ],
                "task": self._task_payload_locked(task),
            }

    # ------------------------------------------------------------------
    def _resolve_role(self, role_name: str) -> CameraRole | None:
        if not isinstance(role_name, str) or not role_name:
            return None
        return self._config.role_by_name(role_name)

    def _validate_params(
        self,
        role: CameraRole,
        scene_id: str,
        sample_fps: Any,
        max_frames: Any,
        max_duration_sec: Any,
    ) -> TaskParameters:
        goal_like = SimpleNamespace(
            scene_id=scene_id,
            sample_fps=sample_fps,
            max_frames=max_frames,
            max_duration=SimpleNamespace(to_sec=lambda: max_duration_sec),
        )
        try:
            parsed = validate_capture_goal(goal_like, role.capture_config)
        except CaptureError as error:
            raise ValueError(f"invalid task parameters: {error}") from error
        # C13/CAP-008：window_frames=N 的全部 N 帧必须进入一次 Omega 推理。
        # worker 的 frame_group_size 是硬上限，超出的 N 必须显式拒绝，
        # 不能依赖下层等距二次抽帧。
        if parsed.max_frames > self._config.reconstruct_frame_group_size:
            raise ValueError(
                f"max_frames={parsed.max_frames} exceeds reconstruct "
                f"frame_group_size={self._config.reconstruct_frame_group_size}; "
                "all captured frames must participate in one inference"
            )
        return TaskParameters(
            role=role.role,
            scene_id=parsed.scene_id,
            sample_fps=parsed.sample_fps,
            max_frames=parsed.max_frames,
            max_duration_sec=parsed.max_duration_sec,
        )

    def _run_task(self, record: _TaskRecord) -> None:
        """任务线程统一入口：任何未预期异常都尝试收敛为明确终态。"""
        try:
            self._run_task_inner(record)
        except BaseException as error:
            self._handle_task_thread_failure(record, error)

    def _run_task_inner(self, record: _TaskRecord) -> None:
        """任务主流程：Capture Action → Reconstruct Action → 终态。"""
        capture_client = self._capture_client_factory(record.role)
        with self._lock:
            if self._task is not record:
                return
            record.capture_client = capture_client

        if record.cancel_event.is_set():
            self._finish(record, STATE_CANCELLED, "cancelled before capture")
            return
        if not capture_client.wait_for_server(
            self._config.action_connect_timeout_sec
        ):
            self._finish(
                record,
                STATE_FAILED,
                "capture action server did not appear within "
                f"{self._config.action_connect_timeout_sec} s",
            )
            return

        try:
            with self._lock:
                if self._task is not record:
                    return
                record.active_client = capture_client
                record.lower_goal_sent = True
            capture_client.send_goal(
                record.params, self._make_capture_feedback(record)
            )
        except BaseException as error:
            self._clear_lower_goal(record)
            self._finish(
                record,
                STATE_FAILED,
                f"capture goal could not be sent: {type(error).__name__}: {error}",
            )
            return

        if not self._wait_for_action(
            record,
            capture_client,
            record.params.max_duration_sec + self._config.capture_wait_margin_sec,
            STATE_CAPTURING,
        ):
            return
        self._clear_lower_goal(record)
        capture_outcome = capture_client.outcome()
        self._apply_capture_outcome(record, capture_outcome)
        if record.state != STATE_RECONSTRUCTING:
            return

        # 竞态修复：capture 已终态但 reconstruct goal 尚未建立时取消，
        # 这里必须直接收敛 CANCELLED，不能带着 STOPPING 退出线程。
        reconstruct_client = self._reconstruct_client_factory()
        with self._lock:
            if self._task is not record:
                return
            if record.state != STATE_RECONSTRUCTING:
                if record.cancel_event.is_set():
                    self._set_terminal_locked(
                        record, STATE_CANCELLED, "cancelled before reconstruction"
                    )
                else:
                    self._set_terminal_locked(
                        record,
                        STATE_FAILED,
                        "pipeline exited before the reconstruction goal was created",
                    )
                return
            record.reconstruct_client = reconstruct_client

        if record.cancel_event.is_set():
            self._finish(record, STATE_CANCELLED, "cancelled before reconstruction")
            return
        if not reconstruct_client.wait_for_server(
            self._config.action_connect_timeout_sec
        ):
            self._finish(
                record,
                STATE_FAILED,
                "reconstruct action server did not appear within "
                f"{self._config.action_connect_timeout_sec} s",
            )
            return

        try:
            with self._lock:
                if self._task is not record:
                    return
                record.active_client = reconstruct_client
                record.lower_goal_sent = True
            reconstruct_client.send_goal(
                record.snapshot_uri,
                record.manifest_sha256,
                self._make_reconstruct_feedback(record),
            )
        except BaseException as error:
            self._clear_lower_goal(record)
            self._finish(
                record,
                STATE_FAILED,
                f"reconstruct goal could not be sent: {type(error).__name__}: {error}",
            )
            return

        if not self._wait_for_action(
            record,
            reconstruct_client,
            self._config.reconstruct_wait_timeout_sec,
            STATE_RECONSTRUCTING,
        ):
            return
        self._clear_lower_goal(record)
        self._apply_reconstruct_outcome(record, reconstruct_client.outcome())

    def _clear_lower_goal(self, record: _TaskRecord) -> None:
        with self._lock:
            if self._task is record:
                record.active_client = None
                record.lower_goal_sent = False

    def _handle_task_thread_failure(
        self, record: _TaskRecord, error: BaseException
    ) -> None:
        """处理任务线程未预期异常，保证线程出口收敛或显式保持槽位。"""
        with self._lock:
            if self._task is not record or record.state in _TERMINAL_STATES:
                return
            message = (
                f"pipeline task thread failed: {type(error).__name__}: {error}"
            )
            if not record.lower_goal_sent:
                self._set_terminal_locked(record, STATE_FAILED, message)
                return
            # 下层 Goal 可能仍活跃：先给出明确 FAILED 终态，但槽位保持到
            # 一个 reaper 线程确认下层 actionlib 终态为止。
            record.state = STATE_FAILED
            record.last_error = (
                message + "; slot held until lower action terminal state"
            )
            record.finished_monotonic = self._monotonic()
            record.slot_held = True
            client = record.active_client
        if client is None:
            return
        try:
            reaper = self._thread_factory(
                target=self._reap_lower_action, args=(record, client)
            )
            if not isinstance(reaper, threading.Thread):
                raise TypeError("thread_factory must return a Thread")
            reaper.daemon = True
            reaper.start()
        except BaseException:
            # 无法启动 reaper 时保持 slot_held（fail-safe）；进程重启前
            # 新任务持续收到 BUSY，绝不允许双 CUDA worker。
            return

    def _reap_lower_action(self, record: _TaskRecord, client) -> None:
        """等待下层 Action 终态后释放因线程异常而保持的槽位。"""
        try:
            while not client.wait_for_result(0.5):
                pass
        except BaseException:
            return  # 无法确认则保持 slot_held
        with self._lock:
            if self._task is record:
                record.slot_held = False
                if not record.done.is_set():
                    record.done.set()

    def _wait_for_action(self, record, client, timeout_sec, state) -> bool:
        """等待下层 Action 终态；取消/上层超时只请求停止，不提前终态化。

        上层槽位在下层 actionlib 终态到达前保持 STOPPING（非终态），因此
        CANCEL_TASK 或上层 watchdog 触发后新 START_TASK 仍会收到 BUSY。
        返回 True 表示已拿到下层终态结果，由调用方仲裁最终状态。
        """
        deadline = self._monotonic() + timeout_sec
        cancel_sent = False
        while True:
            if record.cancel_event.is_set() and not cancel_sent:
                cancel_sent = True
                client.cancel_goal()
                with self._lock:
                    if self._task is record and record.state not in _TERMINAL_STATES:
                        record.state = STATE_STOPPING
                        record.last_error = (
                            "cancel requested; waiting for lower action "
                            "terminal state"
                        )
            elif (
                not record.cancel_event.is_set()
                and not cancel_sent
                and self._monotonic() >= deadline
            ):
                # 上层 watchdog 只能请求下层取消；下层未证明终态前绝不释放槽位。
                cancel_sent = True
                client.cancel_goal()
                with self._lock:
                    if self._task is record and record.state not in _TERMINAL_STATES:
                        record.state = STATE_STOPPING
                        record.watchdog_error = (
                            f"upper service timeout after {timeout_sec} s in "
                            f"{state}; underlying action cancel requested and "
                            "slot held until its terminal state"
                        )
                        record.last_error = record.watchdog_error
            if client.wait_for_result(0.5):
                return True

    def _make_capture_feedback(self, record):
        def _on_feedback(feedback):
            with self._lock:
                if self._task is not record:
                    return
                record.captured_frames = int(getattr(feedback, "frame_count", 0))
        return _on_feedback

    def _make_reconstruct_feedback(self, record):
        def _on_feedback(feedback):
            with self._lock:
                if self._task is not record:
                    return
                record.reconstruction_stage = str(
                    getattr(feedback, "stage", record.reconstruction_stage)
                )
                record.reconstruction_progress = float(
                    getattr(feedback, "progress", record.reconstruction_progress)
                )
        return _on_feedback

    def _apply_capture_outcome(
        self, record: _TaskRecord, outcome: CaptureActionOutcome
    ) -> None:
        with self._lock:
            if self._task is not record:
                return
            record.captured_frames = outcome.frame_count
            record.snapshot_uri = outcome.snapshot_uri
            record.manifest_sha256 = outcome.manifest_sha256
            if record.cancel_event.is_set():
                self._set_terminal_locked(record, STATE_CANCELLED, "cancelled during capture")
                return
            if record.watchdog_error:
                self._set_terminal_locked(
                    record,
                    STATE_FAILED,
                    record.watchdog_error
                    + f"; lower capture action reached {outcome.action_state}",
                )
                return
            if outcome.action_state != ACTION_SUCCEEDED or outcome.error_code != 0:
                self._set_terminal_locked(
                    record,
                    STATE_FAILED,
                    f"capture action ended as {outcome.action_state}: "
                    f"code={outcome.error_code} message={outcome.message}",
                )
                return
            # 本任务身份门禁：禁止把其他窗口/数据源的 capture 结果串进任务。
            expected = record.role.capture_config
            identity_failure = None
            if outcome.scene_id != record.params.scene_id:
                identity_failure = (
                    f"scene_id mismatch: expected {record.params.scene_id!r}, "
                    f"got {outcome.scene_id!r}"
                )
            elif outcome.image_topic != expected.image_topic:
                identity_failure = (
                    f"image_topic mismatch: expected {expected.image_topic!r}, "
                    f"got {outcome.image_topic!r}"
                )
            elif outcome.camera_name != expected.camera_name:
                identity_failure = (
                    f"camera_name mismatch: expected {expected.camera_name!r}, "
                    f"got {outcome.camera_name!r}"
                )
            elif outcome.frame_count != record.params.max_frames:
                identity_failure = (
                    f"capture returned {outcome.frame_count} frames but task "
                    f"requested {record.params.max_frames}; partial windows are "
                    "not admitted because all N frames must enter one inference"
                )
            if identity_failure is not None:
                self._set_terminal_locked(
                    record, STATE_FAILED, f"capture identity check failed: {identity_failure}"
                )
                return
            if not outcome.snapshot_uri or not _valid_sha256(outcome.manifest_sha256):
                self._set_terminal_locked(
                    record,
                    STATE_FAILED,
                    "capture succeeded without valid snapshot metadata "
                    "(snapshot_uri or manifest_sha256)",
                )
                return
            record.state = STATE_RECONSTRUCTING

    def _apply_reconstruct_outcome(
        self, record: _TaskRecord, outcome: ReconstructActionOutcome
    ) -> None:
        with self._lock:
            if self._task is not record:
                return
            record.reconstruction_frames = outcome.frame_count
            record.result_manifest_uri = outcome.result_manifest_uri
            record.result_manifest_sha256 = outcome.result_manifest_sha256
            if record.cancel_event.is_set():
                self._set_terminal_locked(
                    record, STATE_CANCELLED, "cancelled during reconstruction"
                )
                return
            if record.watchdog_error:
                self._set_terminal_locked(
                    record,
                    STATE_FAILED,
                    record.watchdog_error
                    + f"; lower reconstruct action reached {outcome.action_state}",
                )
                return
            if outcome.action_state != ACTION_SUCCEEDED or outcome.error_code != 0:
                self._set_terminal_locked(
                    record,
                    STATE_FAILED,
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
                self._set_terminal_locked(
                    record,
                    STATE_FAILED,
                    "reconstruct action succeeded without valid result metadata "
                    "(result_manifest_uri, result_manifest_sha256, frame_count)",
                )
                return
            # 本任务身份门禁：正式 result.json 必须恰好来自本窗口/本角色/本 snapshot。
            expected = record.role.capture_config
            identity_failure = None
            if outcome.scene_id != record.params.scene_id:
                identity_failure = (
                    f"scene_id mismatch: expected {record.params.scene_id!r}, "
                    f"got {outcome.scene_id!r}"
                )
            elif outcome.frame_count != record.captured_frames:
                identity_failure = (
                    f"reconstruction frame_count {outcome.frame_count} does not "
                    f"match captured frame_count {record.captured_frames}"
                )
            elif outcome.camera_role != record.role.role:
                identity_failure = (
                    f"camera_role mismatch: expected {record.role.role!r}, "
                    f"got {outcome.camera_role!r}"
                )
            elif outcome.camera_topic != expected.image_topic:
                identity_failure = (
                    f"camera_topic mismatch: expected {expected.image_topic!r}, "
                    f"got {outcome.camera_topic!r}"
                )
            elif outcome.camera_frame_id != expected.camera_frame_id:
                identity_failure = (
                    f"camera_frame_id mismatch: expected "
                    f"{expected.camera_frame_id!r}, got {outcome.camera_frame_id!r}"
                )
            elif outcome.snapshot_manifest_sha256 != record.manifest_sha256:
                identity_failure = (
                    "snapshot manifest sha256 mismatch: expected "
                    f"{record.manifest_sha256!r}, got "
                    f"{outcome.snapshot_manifest_sha256!r}"
                )
            if identity_failure is not None:
                self._set_terminal_locked(
                    record,
                    STATE_FAILED,
                    f"reconstruct identity check failed: {identity_failure}",
                )
                return
            self._set_terminal_locked(record, STATE_SUCCEEDED, "")

    def _finish(self, record: _TaskRecord, state: str, error: str) -> None:
        with self._lock:
            if self._task is not record:
                return
            self._set_terminal_locked(record, state, error)

    def _set_terminal_locked(self, record: _TaskRecord, state: str, error: str) -> None:
        if record.state in _TERMINAL_STATES:
            return
        record.state = state
        record.last_error = error
        record.finished_monotonic = self._monotonic()
        record.slot_held = False
        record.active_client = None
        record.lower_goal_sent = False
        record.done.set()

    def _status_locked(self, error_code: int, message: str) -> CommandOutcome:
        """调用方必须持有 self._lock。"""
        role = self._selected_role.role if self._selected_role is not None else ""
        payload = self._task_payload_locked(self._task)
        return CommandOutcome.make(
            error_code,
            message,
            role=role,
            **payload,
        )

    def _task_payload_locked(self, task: _TaskRecord | None) -> dict[str, Any]:
        if task is None:
            return {
                "state": STATE_IDLE,
                "task_id": 0,
                "scene_id": "",
                "capture_action": "",
                "reconstruct_action": self._config.reconstruct_action,
                "snapshot_uri": "",
                "manifest_sha256": "",
                "result_manifest_uri": "",
                "result_manifest_sha256": "",
                "captured_frames": 0,
                "reconstruction_frames": 0,
                "reconstruction_progress": 0.0,
                "reconstruction_stage": "",
                "last_error": "",
                "task_wall_time_sec": 0.0,
            }
        finished = (
            task.finished_monotonic
            if task.finished_monotonic is not None
            else self._monotonic()
        )
        return {
            "state": task.state,
            "task_id": task.task_id,
            "scene_id": task.params.scene_id,
            "capture_action": task.role.capture_action,
            "reconstruct_action": self._config.reconstruct_action,
            "snapshot_uri": task.snapshot_uri,
            "manifest_sha256": task.manifest_sha256,
            "result_manifest_uri": task.result_manifest_uri,
            "result_manifest_sha256": task.result_manifest_sha256,
            "captured_frames": task.captured_frames,
            "reconstruction_frames": task.reconstruction_frames,
            "reconstruction_progress": task.reconstruction_progress,
            "reconstruction_stage": task.reconstruction_stage,
            "last_error": task.last_error,
            "task_wall_time_sec": max(0.0, finished - task.started_monotonic),
        }


def _valid_sha256(value: Any) -> bool:
    """校验值为 64 位小写十六进制 SHA-256 摘要。"""
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except (TypeError, ValueError):
        return False
    return True


def make_pipeline_config(
    roles: Mapping[str, Any] | list | tuple,
    *,
    reconstruct_action: str,
    service_name: str,
    diagnostics_topic: str,
    diagnostics_period_sec: float,
    action_connect_timeout_sec: float,
    capture_wait_margin_sec: float,
    reconstruct_wait_timeout_sec: float,
    reconstruct_frame_group_size: int,
    capture_config_builder,
    runtime_builder,
) -> PipelineConfig:
    """从 node 参数构造 PipelineConfig（注入 capture 配置构造器以便复用校验）。"""
    if not isinstance(roles, (Mapping, list, tuple)) or not roles:
        raise PipelineConfigError("roles must be a nonempty mapping or list")
    entries = roles.values() if isinstance(roles, Mapping) else roles
    parsed_roles = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise PipelineConfigError("each camera role entry must be a mapping")
        entry_dict = dict(entry)
        capture_config = capture_config_builder(entry_dict)
        runtime = runtime_builder(entry_dict)
        if (
            not isinstance(runtime.action_name, str)
            or not runtime.action_name.startswith("/")
        ):
            raise PipelineConfigError(
                f"role {capture_config.camera_role!r} action_name must be absolute"
            )
        parsed_roles.append(
            CameraRole(
                role=capture_config.camera_role,
                capture_action=runtime.action_name,
                capture_config=capture_config,
            )
        )
    config = PipelineConfig(
        roles=tuple(parsed_roles),
        reconstruct_action=reconstruct_action,
        service_name=service_name,
        diagnostics_topic=diagnostics_topic,
        diagnostics_period_sec=float(diagnostics_period_sec),
        action_connect_timeout_sec=float(action_connect_timeout_sec),
        capture_wait_margin_sec=float(capture_wait_margin_sec),
        reconstruct_wait_timeout_sec=float(reconstruct_wait_timeout_sec),
        reconstruct_frame_group_size=int(reconstruct_frame_group_size),
    )
    config.validate()
    return config


__all__ = [
    "ACTION_CANCELLED",
    "ACTION_FAILED",
    "ACTION_SUCCEEDED",
    "BUSY",
    "CameraPipelineService",
    "CameraRole",
    "CaptureActionClient",
    "CaptureActionOutcome",
    "CommandOutcome",
    "INVALID_ARGUMENT",
    "INVALID_COMMAND",
    "INTERNAL_ERROR",
    "NO_TASK",
    "NOT_READY",
    "OK",
    "PipelineConfig",
    "PipelineConfigError",
    "ReconstructActionClient",
    "ReconstructActionOutcome",
    "STATE_CANCELLED",
    "STATE_CAPTURING",
    "STATE_FAILED",
    "STATE_IDLE",
    "STATE_RECONSTRUCTING",
    "STATE_STOPPING",
    "STATE_SUCCEEDED",
    "TASK_FAILED",
    "TaskParameters",
    "UNKNOWN_ROLE",
    "make_pipeline_config",
]
