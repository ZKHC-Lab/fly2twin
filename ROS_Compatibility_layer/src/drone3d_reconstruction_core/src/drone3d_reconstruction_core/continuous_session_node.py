"""ROS1 连续 session 调度器节点。

把 ContinuousSession 核心挂到 rospy：
  - `ContinuousSessionCommand` 服务（选源/启动/暂停/恢复/停止/状态/热调）；
  - `/diagnostics` 周期发布连续 session 状态；
  - 只通过现有 Capture/Reconstruct Action 与 CaptureControl 服务工作，
    不订阅图像、不读 bag、不启动 worker。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import threading
from typing import Any, Callable, Mapping

import actionlib
from actionlib_msgs.msg import GoalStatus
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from drone3d_reconstruction_msgs.msg import (
    CaptureSegmentAction,
    CaptureSegmentGoal,
    CaptureSegmentResult,
)
from drone3d_reconstruction_msgs.srv import (
    CaptureControl,
    CaptureControlRequest,
    CaptureControlResponse,
    ContinuousSessionCommand,
    ContinuousSessionCommandRequest,
    ContinuousSessionCommandResponse,
)
import rospy

from .capture_node import (
    capture_control_service_name,
    load_capture_config,
    load_runtime_params,
)
from .continuous_session import (
    ContinuousConfig,
    ContinuousSession,
    JobStatus,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    QUEUE_LOSSLESS,
    RECONSTRUCTION_MODES,
    ReconstructionBackend,
    SESSION_FATAL_OOM,
    SESSION_MEMORY_PAUSED,
    SESSION_IDLE,
)
from .action_types import CameraRole
from .reconstruct_action_client import ActionlibReconstructClient

NODE_NAME = "continuous_session_node"
DEFAULT_SERVICE_NAME = "/continuous_session/command"
DEFAULT_DIAGNOSTICS_TOPIC = "/diagnostics"
DEFAULT_DIAGNOSTICS_PERIOD_SEC = 1.0
DEFAULT_QUEUE_MODE = QUEUE_LOSSLESS
DEFAULT_AUTO_START_DELAY_SEC = 2.0


class ContinuousNodeConfigError(ValueError):
    """连续调度器节点参数缺失或非法。"""


class ActionlibContinuousCaptureClient:
    """Capture Action + CaptureControl 服务的连续采集适配器。"""

    def __init__(self, role: CameraRole) -> None:
        self._role = role
        self._client = actionlib.SimpleActionClient(
            role.capture_action, CaptureSegmentAction
        )
        control_name = capture_control_service_name(role.capture_action)
        self._control = rospy.ServiceProxy(control_name, CaptureControl)

    def wait_for_server(self, timeout_sec: float) -> bool:
        return self._client.wait_for_server(rospy.Duration.from_sec(timeout_sec))

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
        goal = CaptureSegmentGoal()
        goal.scene_id = scene_id
        goal.sample_fps = float(sample_fps)
        goal.max_frames = int(max_frames)
        goal.max_duration = rospy.Duration.from_sec(float(max_duration_sec))
        if set_save_sensor_depth:
            if type(save_sensor_depth) is not bool:
                raise ValueError("save_sensor_depth must be a bool")
            goal.save_sensor_depth = bool(save_sensor_depth)
        goal.set_save_sensor_depth = bool(set_save_sensor_depth)
        self._client.send_goal(goal, feedback_cb=feedback_cb)

    def wait_for_result(self, timeout_sec: float) -> bool:
        return self._client.wait_for_result(rospy.Duration.from_sec(timeout_sec))

    def cancel_goal(self) -> None:
        self._client.cancel_goal()

    def outcome(self):
        from .action_types import (
            ACTION_CANCELLED,
            ACTION_FAILED,
            ACTION_SUCCEEDED,
            CaptureActionOutcome,
        )

        state = self._client.get_state()
        if state == GoalStatus.SUCCEEDED:
            action_state = ACTION_SUCCEEDED
        elif state in (GoalStatus.PREEMPTED, GoalStatus.RECALLED):
            action_state = ACTION_CANCELLED
        else:
            action_state = ACTION_FAILED
        result = self._client.get_result()
        if result is None:
            result = CaptureSegmentResult()
            result.error_code = CaptureSegmentResult.INTERNAL_ERROR
            result.message = "capture action has no result message"
        return CaptureActionOutcome(
            action_state=action_state,
            error_code=int(
                getattr(result, "error_code", CaptureSegmentResult.INTERNAL_ERROR)
            ),
            message=str(getattr(result, "message", "")),
            scene_id=str(getattr(result, "scene_id", "")),
            camera_name=str(getattr(result, "camera_name", "")),
            image_topic=str(getattr(result, "image_topic", "")),
            snapshot_uri=str(getattr(result, "snapshot_uri", "")),
            manifest_uri=str(getattr(result, "manifest_uri", "")),
            manifest_sha256=str(getattr(result, "manifest_sha256", "")),
            frame_count=int(getattr(result, "frame_count", 0)),
        )

    def _control_call(self, command: int) -> bool:
        try:
            response = self._control(command=command)
        except BaseException:
            return False
        return bool(response.error_code == CaptureControlResponse.OK)

    def pause(self) -> bool:
        return self._control_call(CaptureControlRequest.PAUSE)

    def resume(self) -> bool:
        return self._control_call(CaptureControlRequest.RESUME)


class LocalCudaBackend(ReconstructionBackend):
    """包装现有 ReconstructSegment Action 的本地 CUDA backend。"""

    def __init__(self, action_name: str) -> None:
        self._action_name = action_name
        self._jobs: dict[str, ActionlibReconstructClient] = {}

    def submit(
        self,
        snapshot_uri: str,
        manifest_sha256: str,
        window_id: str,
        inference_resolution: int = 0,
        worker_timeout_sec: float = 0.0,
    ) -> str:
        if window_id in self._jobs:
            # 拒绝重复 owner，但不代它取消或收集旧目标。
            raise RuntimeError(f"window_id {window_id!r} already has an active job")
        client = ActionlibReconstructClient(self._action_name)
        if not client.wait_for_server(30.0):
            raise RuntimeError("reconstruct action server did not appear")
        client.send_goal(
            snapshot_uri,
            manifest_sha256,
            lambda feedback: None,
            image_resolution=int(inference_resolution)
            if inference_resolution
            else 0,
            set_image_resolution=bool(inference_resolution != 0),
            worker_timeout_sec=float(worker_timeout_sec)
            if worker_timeout_sec
            else 0.0,
            set_worker_timeout=bool(worker_timeout_sec != 0.0),
        )
        self._jobs[window_id] = client
        return window_id

    def status(self, job_id: str) -> JobStatus:
        client = self._jobs.get(job_id)
        if client is None:
            return JobStatus(job_id, JOB_FAILED, "unknown job")
        state = client._client.get_state()
        if state == GoalStatus.SUCCEEDED:
            if client._client.get_result() is None:
                return JobStatus(
                    job_id, JOB_RUNNING, "waiting for action result message"
                )
            return JobStatus(job_id, JOB_SUCCEEDED, "")
        if state in (GoalStatus.PREEMPTED, GoalStatus.RECALLED):
            return JobStatus(job_id, JOB_CANCELLED, "cancelled")
        if state in (GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.LOST):
            result = client._client.get_result()
            message = (
                str(getattr(result, "message", "")) if result is not None else ""
            )
            return JobStatus(job_id, JOB_FAILED, message)
        return JobStatus(job_id, JOB_RUNNING, "")

    def cancel(self, job_id: str) -> None:
        client = self._jobs.get(job_id)
        if client is not None:
            client.cancel_goal()

    def collect_result(self, job_id: str):
        client = self._jobs.get(job_id)
        if client is None:
            raise RuntimeError(f"unknown job {job_id}")
        outcome = client.outcome()
        # 只有 outcome 成功取得后才释放 owner；异常时保留 job，stop_all
        # 必须观察到非空 active job 并保持 STOPPING 直到重试收敛。
        self._jobs.pop(job_id, None)
        return outcome


class ContinuousSessionNode:
    """连续 session 调度器 ROS 节点。"""

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        param_loader: Callable[[], Mapping[str, Any]] | None = None,
        config_builder: Callable[..., ContinuousConfig] = None,
        service_factory: Callable[..., Any] = rospy.Service,
        publisher_factory: Callable[..., Any] = rospy.Publisher,
        timer_factory: Callable[..., Any] = rospy.Timer,
        capture_client_factory: Callable[..., Any] = ActionlibContinuousCaptureClient,
        backend_factory: Callable[..., ReconstructionBackend] = None,
        auto_start: bool = False,
        start_guard=None,
        release_worker=None,
    ) -> None:
        if param_loader is not None:
            if params is not None:
                raise TypeError("provide params or param_loader, not both")
            params = param_loader()
        if params is None:
            raise TypeError("node parameters are required")
        self._params = dict(params)
        self._command_lock = threading.RLock()
        self._start_guard = start_guard
        self._release_worker = release_worker
        self._config = None
        self._core = None
        self._not_ready_error: str | None = None
        self._service = None
        self._publisher = None
        self._timer = None
        self._service_factory = service_factory
        self._publisher_factory = publisher_factory
        self._timer_factory = timer_factory
        self._capture_client_factory = capture_client_factory
        self._backend_factory = backend_factory

        config = None
        try:
            config = config_builder(params) if config_builder else load_continuous_config(
                params
            )
        except Exception as error:
            self._not_ready_error = f"{type(error).__name__}: {error}"
        else:
            self._config = config
            try:
                self._core = ContinuousSession(
                    config,
                    capture_client_factory=self._capture_client_factory,
                    backend_factory=lambda: (
                        self._backend_factory(config.reconstruct_action)
                        if self._backend_factory is not None
                        else LocalCudaBackend(config.reconstruct_action)
                    ),
                    disk_free_gb=lambda: _min_free_disk_gb(config),
                )
            except Exception as error:
                self._not_ready_error = f"{type(error).__name__}: {error}"

        service_name = (
            config.service_name
            if config is not None
            else params.get("service_name", DEFAULT_SERVICE_NAME)
        )
        self._service = self._service_factory(
            service_name, ContinuousSessionCommand, self._handle_command
        )
        self._auto_start_session = (
            type(params.get("auto_start_session")) is bool
            and params.get("auto_start_session")
        )
        self._initial_role = str(params.get("initial_role", "") or "")
        delay = params.get("auto_start_delay_sec", DEFAULT_AUTO_START_DELAY_SEC)
        try:
            delay = float(delay)
        except (TypeError, ValueError):
            delay = DEFAULT_AUTO_START_DELAY_SEC
        if not math.isfinite(delay) or delay < 0:
            delay = DEFAULT_AUTO_START_DELAY_SEC
        self._auto_start_delay_sec = delay
        if auto_start:
            self.start_publishing()
            if self._auto_start_session and self._core is not None:
                self._schedule_auto_start()

    @property
    def config(self):
        return self._config

    @property
    def core(self):
        return self._core

    def is_ready(self) -> bool:
        return self._core is not None

    def start_publishing(self) -> None:
        if self._publisher is not None:
            return
        if self._config is not None:
            topic = self._config.diagnostics_topic
            period = self._config.diagnostics_period_sec
        else:
            topic = _fallback_topic(self._params.get("diagnostics_topic"))
            period = _fallback_period(self._params.get("diagnostics_period_sec"))
        self._publisher = self._publisher_factory(
            topic, DiagnosticArray, queue_size=1
        )
        self._timer = self._timer_factory(
            rospy.Duration.from_sec(period), self._publish_callback
        )

    def _schedule_auto_start(self) -> None:
        """节点就绪后延迟自动 SELECT_SOURCE + START_SESSION。"""
        delay = self._auto_start_delay_sec

        def worker():
            threading.Event().wait(delay)
            self._auto_start_once()

        thread = threading.Thread(target=worker, name="continuous-auto-start")
        thread.daemon = True
        thread.start()

    def _auto_start_once(self) -> None:
        if self._core is None:
            return
        if self._start_guard:
            try:self._start_guard()
            except Exception as error:
                rospy.logerr("continuous auto-start rejected: %s", error)
                return
        code, message = self._core.select_source(self._initial_role)
        if code != 0:
            print(f"continuous auto-start failed to select source: {message}")
            return
        code, message = self._core.start_session("")
        if code != 0:
            print(f"continuous auto-start failed to start session: {message}")

    def publish_once(self) -> None:
        self._publish_callback(None)

    def _publish_callback(self, _event) -> None:
        if self._publisher is None:
            return
        message = assemble_continuous_diagnostics(
            core_snapshot=(
                self._core.diagnostics_snapshot() if self._core is not None else None
            ),
            config=self._config,
            not_ready_error=self._not_ready_error,
        )
        try:
            message.header.stamp = rospy.Time.now()
        except BaseException:
            message.header.stamp = rospy.Time(0)
        try:
            self._publisher.publish(message)
        except BaseException:
            pass

    def _handle_command(
        self, request: ContinuousSessionCommandRequest
    ) -> ContinuousSessionCommandResponse:
        if int(request.command) == ContinuousSessionCommandRequest.GET_STATUS:
            return self._dispatch_command(request)
        with self._command_lock:
            return self._dispatch_command(request)

    def _dispatch_command(
        self, request: ContinuousSessionCommandRequest
    ) -> ContinuousSessionCommandResponse:
        if self._core is None:
            response = _status_response(None, self._not_ready_error or "not ready")
            response.error_code = ContinuousSessionCommandResponse.INTERNAL_ERROR
            response.message = "continuous session service is not ready: " + (
                self._not_ready_error or "no config"
            )
            return response
        command = int(request.command)
        if command == ContinuousSessionCommandRequest.SELECT_SOURCE:
            code, message = self._core.select_source(str(request.role))
        elif command == ContinuousSessionCommandRequest.START_SESSION:
            if self._start_guard:
                try:self._start_guard()
                except Exception as error:
                    response = _status_response(self._core.get_status(), "")
                    response.error_code = 3
                    response.message = str(error)
                    return response
            code, message = self._core.start_session(
                str(request.role),
                str(getattr(request, "reconstruction_mode", "") or ""),
            )
        elif command == ContinuousSessionCommandRequest.PAUSE:
            code, message = self._core.pause()
        elif command == ContinuousSessionCommandRequest.RESUME:
            code, message = self._core.resume()
        elif command == ContinuousSessionCommandRequest.STOP_ALL:
            code, message = self._core.stop_all()
            if code == 0 and self._release_worker:
                try:
                    self._release_worker()
                    message += "; resident model weights released"
                except Exception as error:
                    code, message = 3, "session stopped but GPU release failed: " + str(error)
        elif command == ContinuousSessionCommandRequest.GET_STATUS:
            code, message = 0, "status"
        elif command == ContinuousSessionCommandRequest.SET_PARAMS:
            code, message = 5, (
                "session parameters are fixed; stop and start a session with another profile"
            )
        else:
            code, message = 1, f"unknown command {command}"
        response = _status_response(self._core.get_status(), "")
        response.error_code = int(code)
        response.message = message
        return response

    def shutdown(self, timeout_sec: float = 3.0) -> None:
        if self._timer is not None:
            try:
                self._timer.shutdown()
            except BaseException:
                pass
            self._timer = None
        error = None
        try:
            if self._core is not None:
                try:
                    self._core.shutdown(timeout_sec)
                except BaseException as shutdown_error:
                    error = shutdown_error
        finally:
            if self._service is not None:
                try:
                    self._service.shutdown("continuous session node shutting down")
                except BaseException:
                    pass
            if self._publisher is not None:
                try:
                    self._publisher.unregister()
                except BaseException:
                    pass
                self._publisher = None
        if error is not None:
            raise error


def _expected_reconstruction_field(params: Mapping[str, Any], key: str) -> str:
    """从 launch 注入的 reconstruct profile（expected_reconstruction）读取字段。"""
    block = params.get("expected_reconstruction")
    if not isinstance(block, Mapping):
        raise ContinuousNodeConfigError(
            "expected_reconstruction must be a mapping loaded from the "
            "reconstruct profile (continuous_session.launch rosparam)"
        )
    value = block.get(key)
    if not isinstance(value, str) or not value or not value.strip():
        raise ContinuousNodeConfigError(
            f"expected_reconstruction/{key} must be a nonempty string"
        )
    if any(ord(char) < 32 for char in value):
        raise ContinuousNodeConfigError(
            f"expected_reconstruction/{key} contains control characters"
        )
    return value


# minimum_total_frames only acted under new_frames and is ignored like any other extra key.
_REMOVED_KEYS = {
    "window_frame_semantics": "total_frames",
    "window_plan_path": "",
    "threshold_table_path": "",
    "threshold_group": "",
    "fit_exclusion_source": "",
}


def load_continuous_config(
    params: Mapping[str, Any],
    *,
    capture_config_builder: Callable[..., Any] = load_capture_config,
    capture_runtime_builder: Callable[..., Any] = load_runtime_params,
) -> ContinuousConfig:
    if not isinstance(params, Mapping):
        raise ContinuousNodeConfigError("node parameters must be a mapping")
    # 部署机上的旧配置可能还留着已删功能（分窗清单、图配准）的键：取中性值时忽略，
    # 取了别的值就报错，不能悄悄按另一套配置运行。
    for key, neutral in _REMOVED_KEYS.items():
        if key in params and params[key] != neutral:
            raise ContinuousNodeConfigError(
                f"{key}={params[key]!r} selects a removed feature; delete the key"
            )
    expected = params.get("expected_reconstruction") or {}
    alignment_backend = str(params.get("alignment_backend", "global_g4"))
    if alignment_backend != "global_g4":
        raise ContinuousNodeConfigError(
            f"alignment_backend must be 'global_g4', got {alignment_backend!r}"
        )
    for name, value in (("aggregator_frame_chunk_size", 10), ("load_mode", "max_size")):
        if expected.get(name) != value:
            raise ContinuousNodeConfigError(f"G4 requires expected_reconstruction/{name}={value}")
    resolution = expected.get("image_resolution")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0 or resolution % 16:
        raise ContinuousNodeConfigError("G4 requires expected_reconstruction/image_resolution to be a multiple of 16")
    total = _positive_integer(params.get("initial_window_frames", 16), "initial_window_frames")
    if expected.get("frame_group_size") != total:
        raise ContinuousNodeConfigError(
            "G4 frame_group_size must equal the model input frames "
            f"({total}; no subsampling)"
        )
    roles_param = _required(params, "roles")
    if not isinstance(roles_param, Mapping) or not roles_param:
        raise ContinuousNodeConfigError("roles must be a nonempty mapping")
    roles = []
    for name, entry in roles_param.items():
        if not isinstance(entry, Mapping):
            raise ContinuousNodeConfigError(f"roles/{name} must be a mapping")
        entry_dict = dict(entry)
        capture_config = capture_config_builder(entry_dict)
        runtime = capture_runtime_builder(entry_dict)
        if not runtime.action_name.startswith("/"):
            raise ContinuousNodeConfigError(
                f"roles/{name}/action_name must be an absolute action name"
            )
        roles.append(
            CameraRole(
                role=capture_config.camera_role,
                capture_action=runtime.action_name,
                capture_config=capture_config,
            )
        )
    journal_root = str(Path(_required(params, "journal_root")))
    if not Path(journal_root).is_absolute() or not Path(journal_root).is_dir():
        raise ContinuousNodeConfigError("journal_root must be an existing directory")
    config = ContinuousConfig(
        roles=tuple(roles),
        reconstruct_action=_absolute_ros_name(
            _required(params, "reconstruct_action"), "reconstruct_action"
        ),
        journal_root=journal_root,
        result_root=str(Path(_required(params, "result_root"))),
        service_name=_absolute_ros_name(
            _required(params, "service_name"), "service_name"
        ),
        diagnostics_topic=_absolute_ros_name(
            _required(params, "diagnostics_topic"), "diagnostics_topic"
        ),
        diagnostics_period_sec=_positive_finite(
            _required(params, "diagnostics_period_sec"),
            "diagnostics_period_sec",
        ),
        min_free_disk_gb=_positive_finite(
            _required(params, "min_free_disk_gb"), "min_free_disk_gb"
        ),
        action_connect_timeout_sec=_positive_finite(
            _required(params, "action_connect_timeout_sec"),
            "action_connect_timeout_sec",
        ),
        capture_wait_margin_sec=_positive_finite(
            _required(params, "capture_wait_margin_sec"), "capture_wait_margin_sec"
        ),
        reconstruct_wait_timeout_sec=_positive_finite(
            _required(params, "reconstruct_wait_timeout_sec"),
            "reconstruct_wait_timeout_sec",
        ),
        stop_timeout_sec=_positive_finite(
            _required(params, "stop_timeout_sec"), "stop_timeout_sec"
        ),
        reconstruct_frame_group_size=_positive_integer(
            _required(params, "reconstruct_frame_group_size"),
            "reconstruct_frame_group_size",
        ),
        initial_sample_fps=_positive_finite(
            _required(params, "initial_sample_fps"), "initial_sample_fps"
        ),
        initial_window_frames=_positive_integer(
            _required(params, "initial_window_frames"), "initial_window_frames"
        ),
        initial_overlap_frames=_nonnegative_integer(
            params.get("initial_overlap_frames", 0), "initial_overlap_frames"
        ),
        g4_placement=str(params.get("g4_placement", "frame_pose")),
        pose_time_offset_ms=_finite_number(params.get("pose_time_offset_ms", 0.0), "pose_time_offset_ms"),
        pose_time_offset_stage=str(params.get("pose_time_offset_stage", "backend")),
        frame_pose_fusion=_required_bool(params.get("frame_pose_fusion", False), "frame_pose_fusion"),
        g4_cross_window_weight=_finite_number(params.get("g4_cross_window_weight", 0.0), "g4_cross_window_weight"),
        g4_frame_scales=_required_bool(params.get("g4_frame_scales", False), "g4_frame_scales"),
        camera_intrinsics=_optional_intrinsics(params.get("camera_intrinsics")),
        initial_queue_mode=_queue_mode(_required(params, "initial_queue_mode")),
        initial_queue_capacity=_positive_integer(
            _required(params, "initial_queue_capacity"), "initial_queue_capacity"
        ),
        initial_max_retries=_nonnegative_integer(
            _required(params, "initial_max_retries"), "initial_max_retries"
        ),
        initial_keep_snapshot_on_success=_required_bool(
            _required(params, "initial_keep_snapshot_on_success"),
            "initial_keep_snapshot_on_success",
        ),
        initial_inference_resolution=_optional_inference_resolution(
            params.get("initial_inference_resolution", 0)
        ),
        initial_worker_timeout_sec=_optional_worker_timeout(
            params.get("initial_worker_timeout_sec", 0.0)
        ),
        initial_reconstruction_mode=_reconstruction_mode(
            params.get("initial_reconstruction_mode", "per_window")
        ),
        alignment_backend=alignment_backend,
        reconstruction_backend=str((params.get("expected_reconstruction") or {}).get("backend", "omega")),
        expected_checkpoint_sha256=_expected_reconstruction_field(
            params, "checkpoint_sha256"
        ),
        expected_checkpoint_path=_expected_reconstruction_field(
            params, "checkpoint_path"
        ),
        expected_result_root=str(
            Path(_expected_reconstruction_field(params, "result_root"))
        ),
        expected_algorithm_revision=_expected_reconstruction_field(
            params, "algorithm_revision"
        ),
        expected_ros_revision=_expected_reconstruction_field(params, "ros_revision"),
        worker_python=_expected_reconstruction_field(params, "worker_python"),
        cuda_visible_devices=str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
    )
    return config


def _status_response(status, fallback_error: str) -> ContinuousSessionCommandResponse:
    response = ContinuousSessionCommandResponse()
    if status is None:
        response.state = SESSION_IDLE
        response.session_id = ""
        response.role = ""
        response.window_index = 0
        response.captured_frames = 0
        response.capture_target_frames = 0
        response.pending_windows = 0
        response.running_window = 0
        response.succeeded_windows = 0
        response.failed_windows = 0
        response.dropped_windows = 0
        response.effective_sample_fps = 0.0
        response.effective_window_frames = 0
        response.effective_overlap_frames = 0
        response.effective_queue_mode = ""
        response.effective_queue_capacity = 0
        response.effective_max_retries = 0
        response.effective_keep_snapshot_on_success = False
        response.effective_save_sensor_depth = False
        response.effective_inference_resolution = 0
        response.effective_worker_timeout_sec = 0.0
        response.effective_reconstruction_mode = ""
        response.last_error = fallback_error
        response.min_free_disk_gb = 0.0
        return response
    response.state = status.state
    response.session_id = status.session_id
    response.role = status.role
    response.window_index = int(status.window_index)
    response.captured_frames = int(status.captured_frames)
    response.capture_target_frames = int(status.capture_target_frames)
    response.pending_windows = int(status.pending_windows)
    response.running_window = int(status.running_window)
    response.succeeded_windows = int(status.succeeded_windows)
    response.failed_windows = int(status.failed_windows)
    response.dropped_windows = int(status.dropped_windows)
    response.effective_sample_fps = float(status.effective.sample_fps)
    response.effective_window_frames = int(status.effective.window_frames)
    response.effective_overlap_frames = int(status.effective.overlap_frames)
    response.effective_queue_mode = status.effective.queue_mode
    response.effective_queue_capacity = int(status.effective.queue_capacity)
    response.effective_max_retries = int(status.effective.max_retries)
    response.effective_keep_snapshot_on_success = bool(
        status.effective.keep_snapshot_on_success
    )
    response.effective_save_sensor_depth = bool(
        status.effective.save_sensor_depth
    )
    response.effective_inference_resolution = int(
        status.effective.inference_resolution
    )
    response.effective_worker_timeout_sec = float(
        status.effective.worker_timeout_sec
    )
    response.effective_reconstruction_mode = str(status.reconstruction_mode)
    response.last_error = status.last_error
    response.min_free_disk_gb = float(status.min_free_disk_gb)
    return response


def assemble_continuous_diagnostics(
    *,
    core_snapshot: Mapping[str, Any] | None,
    config: ContinuousConfig | None,
    not_ready_error: str | None,
) -> DiagnosticArray:
    message = DiagnosticArray()
    message.status = [
        _server_status(config, core_snapshot, not_ready_error),
        _capture_status(core_snapshot, not_ready_error),
        _map_backend_status(core_snapshot, not_ready_error),
        _fifo_status(core_snapshot, not_ready_error),
        _worker_status(core_snapshot, not_ready_error),
    ]
    return message


def _server_status(config, core_snapshot, not_ready_error):
    status = DiagnosticStatus()
    status.name = "drone3d/session/server"
    if not_ready_error is not None or config is None:
        status.level = DiagnosticStatus.ERROR
        status.message = "not ready: " + (not_ready_error or "no config")
        status.values = [_kv("error", status.message)]
        return status
    state = core_snapshot.get("state", SESSION_IDLE) if core_snapshot else SESSION_IDLE
    if state in (SESSION_FATAL_OOM, SESSION_MEMORY_PAUSED):
        status.level = DiagnosticStatus.ERROR
    elif core_snapshot.get("disk_redline") or core_snapshot.get("worker_disk_blocked"):
        status.level = DiagnosticStatus.WARN
    else:
        status.level = DiagnosticStatus.OK
    status.message = state
    status.values = [
        _kv("state", core_snapshot.get("state", SESSION_IDLE) if core_snapshot else SESSION_IDLE),
        _kv("session_id", core_snapshot.get("session_id", "") if core_snapshot else ""),
        _kv("role", core_snapshot.get("role", "") if core_snapshot else ""),
        _kv("reconstruct_action", config.reconstruct_action),
        _kv("journal_root", config.journal_root),
        _kv("min_free_disk_gb", core_snapshot.get("min_free_disk_gb") if core_snapshot else None),
        _kv("disk_free_gb", core_snapshot.get("disk_free_gb") if core_snapshot else None),
        _kv("last_alignment_state", core_snapshot.get("last_alignment_state") if core_snapshot else None),
        _kv("formal_windows", core_snapshot.get("formal_windows") if core_snapshot else None),
        _kv("formal_records_root", core_snapshot.get("formal_records_root") if core_snapshot else None),
    ]
    return status


def _capture_status(core_snapshot, not_ready_error):
    status = DiagnosticStatus()
    status.name = "drone3d/session/capture"
    if not_ready_error is not None or core_snapshot is None:
        status.level = DiagnosticStatus.ERROR
        status.message = not_ready_error or "unavailable"
        return status
    status.level = DiagnosticStatus.OK
    status.message = f"window {core_snapshot.get('window_index', 0)}"
    status.values = [
        _kv("window_index", core_snapshot.get("window_index", 0)),
        _kv("captured_frames", core_snapshot.get("captured_frames", 0)),
        _kv("capture_target_frames", core_snapshot.get("capture_target_frames", 0)),
        _kv("capture_queue_blocked", core_snapshot.get("capture_queue_blocked", False)),
        _kv("effective_sample_fps", core_snapshot.get("effective", {}).get("sample_fps")),
        _kv("effective_window_frames", core_snapshot.get("effective", {}).get("window_frames")),
        _kv("effective_overlap_frames", core_snapshot.get("effective", {}).get("overlap_frames")),
        _kv("alignment_backend", core_snapshot.get("alignment_backend", "global_g4")),
        _kv("model_input_frames", core_snapshot.get("model_input_frames")),
        _kv("new_input_frames", core_snapshot.get("new_input_frames")),
        _kv("shared_input_frames", core_snapshot.get("shared_input_frames")),
    ]
    return status


def _map_backend_status(core_snapshot, not_ready_error):
    """地图后端（G4 后台进程）。出错时为 ERROR；采集和重建不因此停下。"""
    status = DiagnosticStatus()
    status.name = "drone3d/session/map_backend"
    if not_ready_error is not None or core_snapshot is None:
        status.level = DiagnosticStatus.ERROR
        status.message = not_ready_error or "unavailable"
        return status
    g4 = core_snapshot.get("g4", {})
    state = g4.get("state", "waiting")
    failed = state == "error" or g4.get("exit_code") not in (None, 0)
    status.level = DiagnosticStatus.ERROR if failed else DiagnosticStatus.OK
    status.message = (g4.get("error") or f"state {state}") if failed else str(state)
    status.values = [
        _kv("g4_state", state),
        _kv("g4_revision", g4.get("revision", 0)),
        _kv("g4_error", g4.get("error", "")),
        _kv("g4_exit_code", g4.get("exit_code")),
    ]
    return status


def _fifo_status(core_snapshot, not_ready_error):
    status = DiagnosticStatus()
    status.name = "drone3d/session/fifo"
    if not_ready_error is not None or core_snapshot is None:
        status.level = DiagnosticStatus.ERROR
        status.message = not_ready_error or "unavailable"
        return status
    fifo = core_snapshot.get("fifo", [])
    status.level = (
        DiagnosticStatus.WARN
        if fifo
        or core_snapshot.get("disk_redline")
        or core_snapshot.get("worker_disk_blocked")
        else DiagnosticStatus.OK
    )
    status.message = f"{len(fifo)} waiting windows"
    status.values = [
        _kv("pending_windows", core_snapshot.get("pending_windows", 0)),
        _kv("dropped_windows", core_snapshot.get("dropped_windows", 0)),
        _kv("fifo_bytes", core_snapshot.get("fifo_bytes", 0)),
        _kv(
            "fifo_oldest_wait_sec",
            core_snapshot.get("fifo_oldest_wait_sec"),
        ),
        _kv("queue_mode", core_snapshot.get("effective", {}).get("queue_mode")),
        _kv("queue_capacity", core_snapshot.get("effective", {}).get("queue_capacity")),
    ]
    return status


def _worker_status(core_snapshot, not_ready_error):
    status = DiagnosticStatus()
    status.name = "drone3d/session/worker"
    if not_ready_error is not None or core_snapshot is None:
        status.level = DiagnosticStatus.ERROR
        status.message = not_ready_error or "unavailable"
        return status
    running = core_snapshot.get("running_window")
    status.level = DiagnosticStatus.WARN if running else DiagnosticStatus.OK
    status.message = (
        f"window {running['window_index']} running" if running else "idle"
    )
    status.values = [
        _kv("running_window", running["window_index"] if running else 0),
        _kv("succeeded_windows", core_snapshot.get("succeeded_windows", 0)),
        _kv("failed_windows", core_snapshot.get("failed_windows", 0)),
        _kv("missing_pose_frames", core_snapshot.get("missing_pose_frames", 0)),
        _kv("memory_paused", core_snapshot.get("memory_paused", False)),
        _kv("oom_latched", core_snapshot.get("oom_latched", False)),
        _kv("retry_window", core_snapshot.get("retry_window")),
        _kv("attempt_failures", core_snapshot.get("attempt_failures", 0)),
        _kv("manual_retries", core_snapshot.get("manual_retries", 0)),
        _kv("reconstruction_mode", core_snapshot.get("reconstruction_mode", "")),
        _kv("last_error", core_snapshot.get("last_error", "")),
        _kv("service_wall_time_sec", core_snapshot.get("service_wall_time_sec")),
        _kv("service_budget_sec", core_snapshot.get("service_budget_sec")),
        _kv(
            "worker_wall_time_sec",
            core_snapshot.get("worker_wall_time_sec"),
        ),
        _kv(
            "last_result_wall_time_sec",
            core_snapshot.get("last_result_wall_time_sec"),
        ),
    ]
    return status


def _kv(key, value):
    item = KeyValue()
    item.key = key
    item.value = "unavailable" if value is None else str(value)
    return item


def _required(params, key):
    if key not in params:
        raise ContinuousNodeConfigError(f"missing required parameter: {key}")
    return params[key]


def _absolute_ros_name(value, label):
    if not isinstance(value, str) or not value.startswith("/") or " " in value:
        raise ContinuousNodeConfigError(f"{label} must be an absolute ROS name")
    return value


def _positive_finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContinuousNodeConfigError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ContinuousNodeConfigError(f"{label} must be positive and finite")
    return converted


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContinuousNodeConfigError(f"{label} must be a finite number")
    return float(value)


def _optional_intrinsics(value):
    """Calibrated source-image pinhole mapping, or None (use snapshot CameraInfo)."""
    if value is None or value == "" or value == {}:
        return None
    if not isinstance(value, Mapping):
        raise ContinuousNodeConfigError("camera_intrinsics must be a mapping {width, height, fx, fy, cx, cy}")
    return {str(k): (v if isinstance(v, bool) else float(v) if isinstance(v, (int, float)) else v)
            for k, v in value.items()}


def _positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContinuousNodeConfigError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContinuousNodeConfigError(f"{label} must be a nonnegative integer")
    return value


def _optional_inference_resolution(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContinuousNodeConfigError(
            "initial_inference_resolution must be an integer"
        )
    if value == 0:
        return 0
    if value < 32 or value > 4096:
        raise ContinuousNodeConfigError(
            "initial_inference_resolution must be 0 (inherit) or in [32, 4096]"
        )
    return value


def _optional_worker_timeout(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContinuousNodeConfigError(
            "initial_worker_timeout_sec must be a finite number"
        )
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ContinuousNodeConfigError(
            "initial_worker_timeout_sec must be 0.0 or positive finite"
        )
    return converted


def _queue_mode(value):
    if value != QUEUE_LOSSLESS:
        raise ContinuousNodeConfigError("queue mode must be lossless_fifo")
    return value


def _reconstruction_mode(value):
    if value not in RECONSTRUCTION_MODES:
        raise ContinuousNodeConfigError(
            f"reconstruction_mode must be one of {RECONSTRUCTION_MODES!r}"
        )
    return value


def _required_bool(value, label):
    if type(value) is not bool:
        raise ContinuousNodeConfigError(f"{label} must be a bool")
    return value


def _min_free_disk_gb(config: ContinuousConfig) -> float:
    minimum = float("inf")
    probed = 0
    roots = {role.capture_config.snapshot_root for role in config.roles}
    roots.add(config.journal_root)
    roots.add(config.result_root)
    expected_result_root = getattr(config, "expected_result_root", "")
    if expected_result_root:
        roots.add(expected_result_root)
    for root in roots:
        try:
            info = os.statvfs(root)
            free_gb = (info.f_bavail * info.f_frsize) / 1_000_000_000.0
        except OSError:
            continue
        probed += 1
        minimum = min(minimum, free_gb)
    if probed == 0:
        # fail-closed：任何已配置根目录的磁盘余量都不可读取时，节点不得
        # 以“无限空间”启动并放行采集/worker。
        raise RuntimeError(
            "disk free space could not be determined for any configured root"
        )
    return minimum


def _fallback_topic(value):
    return value if isinstance(value, str) and value.startswith("/") else DEFAULT_DIAGNOSTICS_TOPIC


def _fallback_period(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return DEFAULT_DIAGNOSTICS_PERIOD_SEC


def main(argv=None) -> int:
    rospy.init_node(NODE_NAME, argv=argv)
    params = _rospy_param_loader()
    def resource_callback(key):
        name = params.get(key)
        if not name:return None
        from std_srvs.srv import Trigger
        service = rospy.ServiceProxy(name, Trigger)
        def invoke():
            rospy.wait_for_service(name, timeout=2.)
            response = service()
            if not response.success:raise RuntimeError(response.message)
        return invoke
    node = ContinuousSessionNode(params=params, auto_start=True,
                                 start_guard=resource_callback("gpu_available_service"),
                                 release_worker=resource_callback("release_idle_worker_service"))
    if node.is_ready():
        rospy.loginfo("continuous session scheduler ready")
    else:
        rospy.logerr("continuous session scheduler is not ready; see /diagnostics")
    rospy.on_shutdown(lambda: node.shutdown(timeout_sec=10.0))
    rospy.spin()
    return 0


def _rospy_param_loader() -> Mapping[str, Any]:
    params = rospy.get_param("~", {})
    if not isinstance(params, Mapping):
        raise ContinuousNodeConfigError("rospy private parameters are not a mapping")
    return params


__all__ = [
    "ActionlibContinuousCaptureClient",
    "ContinuousNodeConfigError",
    "ContinuousSessionNode",
    "LocalCudaBackend",
    "assemble_continuous_diagnostics",
    "load_continuous_config",
    "main",
]
