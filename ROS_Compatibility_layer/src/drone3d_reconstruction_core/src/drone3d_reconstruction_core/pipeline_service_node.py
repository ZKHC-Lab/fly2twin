"""ROS1 上层相机管线服务节点。

把 CameraPipelineService 挂到 rospy：
  - `camera_pipeline_command` 服务：SELECT_SOURCE / START_TASK /
    GET_STATUS / CANCEL_TASK；
  - `/diagnostics` 周期发布：服务端、任务、已配置角色三组状态；
  - Action 客户端只调用现有 CaptureSegment / ReconstructSegment Action，
    本节点不订阅图像、不读 bag、不启动 worker。

配置通过私有参数提供。roles 是一个映射，每个条目就是一份完整 capture
profile（与对应 capture_production.launch 加载的是同一份 YAML），由
launch 文件以 `~roles/<role>/...` 形式加载。
"""

from __future__ import annotations

import json
import math
import os
import stat
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit

import actionlib
from actionlib_msgs.msg import GoalStatus
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from drone3d_reconstruction_msgs.msg import (
    CaptureSegmentAction,
    CaptureSegmentGoal,
    CaptureSegmentResult,
    ReconstructSegmentAction,
    ReconstructSegmentGoal,
    ReconstructSegmentResult,
)
from drone3d_reconstruction_msgs.srv import (
    PipelineCommand,
    PipelineCommandRequest,
    PipelineCommandResponse,
)
import rospy

from .capture_node import load_capture_config, load_runtime_params
from .pipeline_service import (
    ACTION_CANCELLED,
    ACTION_FAILED,
    ACTION_SUCCEEDED,
    INVALID_COMMAND,
    NOT_READY,
    STATE_CAPTURING,
    STATE_FAILED,
    STATE_RECONSTRUCTING,
    CameraPipelineService,
    CameraRole,
    CaptureActionClient,
    CaptureActionOutcome,
    CommandOutcome,
    PipelineConfig,
    PipelineConfigError,
    ReconstructActionClient,
    ReconstructActionOutcome,
    TaskParameters,
    make_pipeline_config,
)

NODE_NAME = "camera_pipeline_service"
DEFAULT_SERVICE_NAME = "/camera_pipeline/command"
DEFAULT_DIAGNOSTICS_TOPIC = "/diagnostics"
DEFAULT_DIAGNOSTICS_PERIOD_SEC = 1.0
DEFAULT_ACTION_CONNECT_TIMEOUT_SEC = 10.0
DEFAULT_CAPTURE_WAIT_MARGIN_SEC = 30.0
DEFAULT_RECONSTRUCT_WAIT_TIMEOUT_SEC = 3600.0


class PipelineNodeConfigError(ValueError):
    """管线节点参数缺失或非法。"""


class ActionlibCaptureClient(CaptureActionClient):
    """现有 CaptureSegment Action 的 SimpleActionClient 适配器。"""

    def __init__(self, role: CameraRole) -> None:
        self._action_name = role.capture_action
        self._client = actionlib.SimpleActionClient(
            self._action_name, CaptureSegmentAction
        )

    def wait_for_server(self, timeout_sec: float) -> bool:
        return self._client.wait_for_server(rospy.Duration.from_sec(timeout_sec))

    def send_goal(self, params: TaskParameters, feedback_cb) -> None:
        goal = CaptureSegmentGoal()
        goal.scene_id = params.scene_id
        goal.sample_fps = float(params.sample_fps)
        goal.max_frames = int(params.max_frames)
        goal.max_duration = rospy.Duration.from_sec(float(params.max_duration_sec))
        self._client.send_goal(goal, feedback_cb=feedback_cb)

    def wait_for_result(self, timeout_sec: float) -> bool:
        return self._client.wait_for_result(rospy.Duration.from_sec(timeout_sec))

    def cancel_goal(self) -> None:
        self._client.cancel_goal()

    def outcome(self) -> CaptureActionOutcome:
        action_state = _normalized_action_state(self._client.get_state())
        result = self._client.get_result()
        if result is None:
            result = CaptureSegmentResult()
            result.error_code = CaptureSegmentResult.INTERNAL_ERROR
            result.message = "capture action has no result message"
        return CaptureActionOutcome(
            action_state=action_state,
            error_code=int(getattr(result, "error_code", CaptureSegmentResult.INTERNAL_ERROR)),
            message=str(getattr(result, "message", "")),
            scene_id=str(getattr(result, "scene_id", "")),
            camera_name=str(getattr(result, "camera_name", "")),
            image_topic=str(getattr(result, "image_topic", "")),
            snapshot_uri=str(getattr(result, "snapshot_uri", "")),
            manifest_uri=str(getattr(result, "manifest_uri", "")),
            manifest_sha256=str(getattr(result, "manifest_sha256", "")),
            frame_count=int(getattr(result, "frame_count", 0)),
        )


_RESULT_IDENTITY_MAX_BYTES = 1024 * 1024


def read_result_identity(uri: str) -> dict[str, Any]:
    """从正式 result.json 读取任务身份字段（失败/超限时返回空字典）。

    这是上层成功门禁的输入来源：只有真实 Reconstruct Action 提交的
    file:// result.json 会被读取，任何异常都按身份不可证明处理。
    """
    if not isinstance(uri, str) or not uri.startswith("file://"):
        return {}
    try:
        split = urlsplit(uri)
    except ValueError:
        return {}
    if split.scheme != "file" or split.netloc or split.query or split.fragment:
        return {}
    path = unquote(split.path)
    if "\x00" in path or not os.path.isabs(path):
        return {}
    try:
        info = os.lstat(path)
    except OSError:
        return {}
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return {}
    if info.st_size <= 0 or info.st_size > _RESULT_IDENTITY_MAX_BYTES:
        return {}
    try:
        with open(path, "rb") as stream:
            data = stream.read(_RESULT_IDENTITY_MAX_BYTES + 1)
    except OSError:
        return {}
    if len(data) > _RESULT_IDENTITY_MAX_BYTES:
        return {}
    try:
        payload = json.loads(data.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    identity = {
        "scene_id": payload.get("scene_id", ""),
        "actual_frame_count": payload.get("actual_frame_count", 0),
        "camera_role": payload.get("camera_role", ""),
        "camera_topic": payload.get("camera_topic", ""),
        "camera_frame_id": payload.get("camera_frame_id", ""),
        "snapshot_sha256": snapshot.get("sha256", ""),
    }
    return {key: value for key, value in identity.items() if isinstance(value, (str, int))}


class ActionlibReconstructClient(ReconstructActionClient):
    """现有 ReconstructSegment Action 的 SimpleActionClient 适配器。"""

    def __init__(self, action_name: str) -> None:
        self._action_name = action_name
        self._client = actionlib.SimpleActionClient(
            action_name, ReconstructSegmentAction
        )

    def wait_for_server(self, timeout_sec: float) -> bool:
        return self._client.wait_for_server(rospy.Duration.from_sec(timeout_sec))

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
        goal = ReconstructSegmentGoal()
        goal.snapshot_uri = snapshot_uri
        goal.manifest_sha256 = manifest_sha256
        if set_image_resolution:
            if isinstance(image_resolution, bool) or not isinstance(
                image_resolution, int
            ) or image_resolution <= 0:
                raise ValueError("image_resolution must be a positive integer")
            goal.image_resolution = int(image_resolution)
        goal.set_image_resolution = bool(set_image_resolution)
        if set_worker_timeout:
            if isinstance(worker_timeout_sec, bool) or not isinstance(
                worker_timeout_sec, (int, float)
            ):
                raise ValueError("worker_timeout_sec must be a finite number")
            converted = float(worker_timeout_sec)
            if not math.isfinite(converted) or converted <= 0:
                raise ValueError("worker_timeout_sec must be positive and finite")
            goal.worker_timeout_sec = converted
        goal.set_worker_timeout = bool(set_worker_timeout)
        self._client.send_goal(goal, feedback_cb=feedback_cb)

    def wait_for_result(self, timeout_sec: float) -> bool:
        return self._client.wait_for_result(rospy.Duration.from_sec(timeout_sec))

    def cancel_goal(self) -> None:
        self._client.cancel_goal()

    def outcome(self) -> ReconstructActionOutcome:
        action_state = _normalized_action_state(self._client.get_state())
        result = self._client.get_result()
        if result is None:
            result = ReconstructSegmentResult()
            result.error_code = ReconstructSegmentResult.INTERNAL_ERROR
            result.message = "reconstruct action has no result message"
        try:
            wall_time_sec = float(result.wall_time.to_sec())
        except BaseException:
            wall_time_sec = 0.0
        identity = read_result_identity(
            str(getattr(result, "result_manifest_uri", ""))
        )
        return ReconstructActionOutcome(
            action_state=action_state,
            error_code=int(getattr(result, "error_code", ReconstructSegmentResult.INTERNAL_ERROR)),
            message=str(getattr(result, "message", "")),
            scene_id=str(getattr(result, "scene_id", "")),
            result_manifest_uri=str(getattr(result, "result_manifest_uri", "")),
            result_manifest_sha256=str(getattr(result, "result_manifest_sha256", "")),
            scene_glb_uri=str(getattr(result, "scene_glb_uri", "")),
            frame_count=int(getattr(result, "frame_count", 0)),
            wall_time_sec=wall_time_sec,
            camera_role=str(identity.get("camera_role", "")),
            camera_topic=str(identity.get("camera_topic", "")),
            camera_frame_id=str(identity.get("camera_frame_id", "")),
            snapshot_manifest_sha256=str(identity.get("snapshot_sha256", "")),
            result_scene_id=str(identity.get("scene_id", "")),
            result_actual_frame_count=int(identity.get("actual_frame_count", 0)),
        )


def _normalized_action_state(state: int) -> str:
    if state in (GoalStatus.SUCCEEDED,):
        return ACTION_SUCCEEDED
    if state in (GoalStatus.PREEMPTED, GoalStatus.RECALLED):
        return ACTION_CANCELLED
    return ACTION_FAILED


def load_pipeline_config(
    params: Mapping[str, Any],
    *,
    capture_config_builder: Callable[..., Any] = load_capture_config,
    capture_runtime_builder: Callable[..., Any] = load_runtime_params,
) -> PipelineConfig:
    """从 rospy 私有参数构造管线配置（复用 capture 配置/运行时校验）。"""
    if not isinstance(params, Mapping):
        raise PipelineNodeConfigError("node parameters must be a mapping")
    roles = _required(params, "roles")
    if not isinstance(roles, Mapping) or not roles:
        raise PipelineNodeConfigError("roles must be a nonempty mapping")
    for name, entry in roles.items():
        if not isinstance(entry, Mapping):
            raise PipelineNodeConfigError(f"roles/{name} must be a mapping")
        action_name = entry.get("action_name")
        if not isinstance(action_name, str) or not action_name.startswith("/"):
            raise PipelineNodeConfigError(
                f"roles/{name}/action_name must be an absolute action name"
            )

    reconstruct_action = _absolute_ros_name(
        _required(params, "reconstruct_action"), "reconstruct_action"
    )
    service_name = _absolute_ros_name(
        _required(params, "service_name"), "service_name"
    )
    diagnostics_topic = _absolute_ros_name(
        _required(params, "diagnostics_topic"), "diagnostics_topic"
    )
    diagnostics_period_sec = _positive_finite(
        _required(params, "diagnostics_period_sec"), "diagnostics_period_sec"
    )
    action_connect_timeout_sec = _positive_finite(
        _required(params, "action_connect_timeout_sec"),
        "action_connect_timeout_sec",
    )
    capture_wait_margin_sec = _positive_finite(
        _required(params, "capture_wait_margin_sec"), "capture_wait_margin_sec"
    )
    reconstruct_wait_timeout_sec = _positive_finite(
        _required(params, "reconstruct_wait_timeout_sec"),
        "reconstruct_wait_timeout_sec",
    )
    reconstruct_frame_group_size = _positive_integer(
        _required(params, "reconstruct_frame_group_size"),
        "reconstruct_frame_group_size",
    )
    try:
        return make_pipeline_config(
            roles,
            reconstruct_action=reconstruct_action,
            service_name=service_name,
            diagnostics_topic=diagnostics_topic,
            diagnostics_period_sec=diagnostics_period_sec,
            action_connect_timeout_sec=action_connect_timeout_sec,
            capture_wait_margin_sec=capture_wait_margin_sec,
            reconstruct_wait_timeout_sec=reconstruct_wait_timeout_sec,
            reconstruct_frame_group_size=reconstruct_frame_group_size,
            capture_config_builder=capture_config_builder,
            runtime_builder=capture_runtime_builder,
        )
    except (PipelineConfigError, ValueError, TypeError) as error:
        raise PipelineNodeConfigError(str(error)) from error




class PipelineServiceNode:
    """管线服务节点：配置 → CameraPipelineService → Service + /diagnostics。"""

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        param_loader: Callable[[], Mapping[str, Any]] | None = None,
        config_builder: Callable[..., PipelineConfig] = load_pipeline_config,
        service_factory: Callable[..., Any] = rospy.Service,
        publisher_factory: Callable[..., Any] = rospy.Publisher,
        timer_factory: Callable[..., Any] = rospy.Timer,
        capture_client_factory: Callable[..., CaptureActionClient] = ActionlibCaptureClient,
        reconstruct_client_factory: Callable[..., ReconstructActionClient] = ActionlibReconstructClient,
        auto_start: bool = False,
    ) -> None:
        if param_loader is not None:
            if params is not None:
                raise TypeError("provide params or param_loader, not both")
            params = param_loader()
        if params is None:
            raise TypeError("node parameters are required")
        self._params = dict(params)
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
        self._reconstruct_client_factory = reconstruct_client_factory

        config = None
        try:
            config = config_builder(params)
        except Exception as error:
            self._not_ready_error = f"{type(error).__name__}: {error}"
        else:
            self._config = config
            try:
                self._core = CameraPipelineService(
                    config,
                    capture_client_factory=self._capture_client_factory,
                    reconstruct_client_factory=lambda: self._reconstruct_client_factory(
                        config.reconstruct_action
                    ),
                )
            except Exception as error:
                self._not_ready_error = f"{type(error).__name__}: {error}"

        service_name = (
            config.service_name
            if config is not None
            else params.get("service_name", DEFAULT_SERVICE_NAME)
        )
        self._service = self._service_factory(
            service_name, PipelineCommand, self._handle_command
        )
        if auto_start:
            self.start_publishing()

    @property
    def config(self) -> PipelineConfig | None:
        return self._config

    @property
    def core(self) -> CameraPipelineService | None:
        return self._core

    def is_ready(self) -> bool:
        return self._core is not None

    def start_publishing(self) -> None:
        if self._publisher is not None:
            return
        # 配置失败也必须发布 /diagnostics：用 profile 参数（缺失时用常规默认）。
        if self._config is not None:
            topic = self._config.diagnostics_topic
            period = self._config.diagnostics_period_sec
        else:
            topic = _fallback_topic(
                self._params.get("diagnostics_topic"), "diagnostics_topic"
            )
            period = _fallback_period(
                self._params.get("diagnostics_period_sec"), "diagnostics_period_sec"
            )
        self._publisher = self._publisher_factory(
            topic, DiagnosticArray, queue_size=1
        )
        self._timer = self._timer_factory(
            rospy.Duration.from_sec(period),
            self._publish_callback,
        )

    def publish_once(self) -> None:
        self._publish_callback(None)

    def _publish_callback(self, _event) -> None:
        if self._publisher is None:
            return
        message = assemble_pipeline_diagnostics(
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
        except BaseException as error:
            try:
                rospy.logerr("pipeline diagnostics failed: %s", type(error).__name__)
            except BaseException:
                pass

    def _handle_command(self, request: PipelineCommandRequest) -> PipelineCommandResponse:
        if self._core is None:
            return _to_response(
                CommandOutcome.make(
                    NOT_READY,
                    "pipeline service is not ready: "
                    + (self._not_ready_error or "no config"),
                )
            )
        try:
            command = int(request.command)
        except BaseException:
            command = -1
        if command == PipelineCommandRequest.SELECT_SOURCE:
            outcome = self._core.select_source(str(request.role))
        elif command == PipelineCommandRequest.START_TASK:
            outcome = self._core.start_task(
                str(request.role),
                str(request.scene_id),
                float(request.sample_fps),
                int(request.max_frames),
                float(request.max_duration_sec),
            )
        elif command == PipelineCommandRequest.GET_STATUS:
            outcome = self._core.get_status()
        elif command == PipelineCommandRequest.CANCEL_TASK:
            outcome = self._core.cancel_task()
        else:
            outcome = CommandOutcome.make(INVALID_COMMAND, f"unknown command {command}")
        return _to_response(outcome)

    def shutdown(self, timeout_sec: float = 3.0) -> None:
        if self._timer is not None:
            try:
                self._timer.shutdown()
            except BaseException:
                pass
            self._timer = None
        wait_error = None
        try:
            if self._core is not None:
                # 立即进入关停态：新 START/SELECT 从此刻起被拒绝。
                self._core.begin_shutdown()
                # 请求取消并等待任务收敛到终态；槽位/下层 Action 未证明终态前
                # 不宣称已完成关停。
                try:
                    self._core.cancel_task()
                except BaseException:
                    pass
                if not self._core.wait_for_task_terminal(timeout_sec):
                    wait_error = RuntimeError(
                        "camera pipeline task did not reach a terminal state "
                        f"within {timeout_sec} s during shutdown"
                    )
        finally:
            # 无论任务是否收敛，Service/Publisher 都必须完成清理。
            if self._service is not None:
                try:
                    self._service.shutdown("pipeline service shutting down")
                except BaseException:
                    pass
            if self._publisher is not None:
                try:
                    self._publisher.unregister()
                except BaseException:
                    pass
                self._publisher = None
        if wait_error is not None:
            raise wait_error


def _to_response(outcome: CommandOutcome) -> PipelineCommandResponse:
    response = PipelineCommandResponse()
    response.error_code = int(outcome.error_code)
    response.message = outcome.message
    response.role = outcome.role
    response.state = outcome.state
    response.task_id = int(outcome.task_id)
    response.scene_id = outcome.scene_id
    response.capture_action = outcome.capture_action
    response.reconstruct_action = outcome.reconstruct_action
    response.snapshot_uri = outcome.snapshot_uri
    response.manifest_sha256 = outcome.manifest_sha256
    response.result_manifest_uri = outcome.result_manifest_uri
    response.result_manifest_sha256 = outcome.result_manifest_sha256
    response.captured_frames = int(outcome.captured_frames)
    response.reconstruction_frames = int(outcome.reconstruction_frames)
    response.reconstruction_progress = float(outcome.reconstruction_progress)
    response.reconstruction_stage = outcome.reconstruction_stage
    response.last_error = outcome.last_error
    response.task_wall_time = rospy.Duration.from_sec(
        max(0.0, float(outcome.task_wall_time_sec))
    )
    return response


def assemble_pipeline_diagnostics(
    *,
    core_snapshot: Mapping[str, Any] | None,
    config: PipelineConfig | None,
    not_ready_error: str | None,
) -> DiagnosticArray:
    """纯函数组装管线 /diagnostics 消息。"""
    message = DiagnosticArray()
    message.status = [
        _server_status(config, core_snapshot, not_ready_error),
        _task_status(core_snapshot, not_ready_error),
        _roles_status(config, core_snapshot, not_ready_error),
    ]
    return message


def _server_status(config, core_snapshot, not_ready_error):
    status = DiagnosticStatus()
    status.name = "drone3d/camera_pipeline/server"
    if not_ready_error is not None or config is None:
        status.level = DiagnosticStatus.ERROR
        status.message = "not ready: " + (not_ready_error or "no pipeline config")
        status.values = [_kv("error", status.message)]
        return status
    status.level = DiagnosticStatus.OK
    status.message = "ready"
    values = [
        _kv("service_name", config.service_name),
        _kv("reconstruct_action", config.reconstruct_action),
        _kv("configured_roles", len(config.roles)),
    ]
    if core_snapshot:
        values.append(_kv("selected_role", core_snapshot.get("selected_role")))
    status.values = values
    return status


def _task_status(core_snapshot, not_ready_error):
    status = DiagnosticStatus()
    status.name = "drone3d/camera_pipeline/task"
    if not_ready_error is not None:
        status.level = DiagnosticStatus.ERROR
        status.message = "pipeline not ready: " + not_ready_error
        status.values = [_kv("error", status.message)]
        return status
    task = core_snapshot.get("task", {}) if core_snapshot else {}
    state = str(task.get("state", "IDLE"))
    if state == STATE_FAILED:
        level = DiagnosticStatus.ERROR
    elif state in (STATE_CAPTURING, STATE_RECONSTRUCTING):
        level = DiagnosticStatus.WARN
    else:
        level = DiagnosticStatus.OK
    status.level = level
    status.message = state
    status.values = [
        _kv("state", state),
        _kv("task_id", task.get("task_id", 0)),
        _kv("scene_id", task.get("scene_id", "")),
        _kv("capture_action", task.get("capture_action", "")),
        _kv("reconstruct_action", task.get("reconstruct_action", "")),
        _kv("captured_frames", task.get("captured_frames", 0)),
        _kv("reconstruction_frames", task.get("reconstruction_frames", 0)),
        _kv("reconstruction_progress", task.get("reconstruction_progress", 0.0)),
        _kv("reconstruction_stage", task.get("reconstruction_stage", "")),
        _kv("last_error", task.get("last_error", "")),
        _kv("task_wall_time_sec", task.get("task_wall_time_sec", 0.0)),
        _kv("snapshot_uri", task.get("snapshot_uri", "")),
        _kv("result_manifest_uri", task.get("result_manifest_uri", "")),
    ]
    return status


def _roles_status(config, core_snapshot, not_ready_error):
    status = DiagnosticStatus()
    status.name = "drone3d/camera_pipeline/roles"
    if not_ready_error is not None or config is None:
        status.level = DiagnosticStatus.ERROR
        status.message = "roles unavailable: " + (not_ready_error or "no config")
        status.values = [_kv("error", status.message)]
        return status
    status.level = DiagnosticStatus.OK
    status.message = f"{len(config.roles)} configured camera roles"
    values = [_kv("role_count", len(config.roles))]
    roles = core_snapshot.get("roles", []) if core_snapshot else []
    for role in roles:
        prefix = f"role_{role['role']}"
        values.extend(
            [
                _kv(prefix + "/capture_action", role["capture_action"]),
                _kv(prefix + "/image_topic", role["image_topic"]),
                _kv(prefix + "/camera_frame_id", role["camera_frame_id"]),
                _kv(prefix + "/camera_name", role["camera_name"]),
            ]
        )
    status.values = values
    return status


def _fallback_topic(value: Any, label: str) -> str:
    if (
        isinstance(value, str)
        and value.startswith("/")
        and " " not in value
        and "\x00" not in value
    ):
        return value
    try:
        rospy.logwarn(
            "%s is unavailable or invalid; publishing diagnostics on %s",
            label,
            DEFAULT_DIAGNOSTICS_TOPIC,
        )
    except BaseException:
        pass
    return DEFAULT_DIAGNOSTICS_TOPIC


def _fallback_period(value: Any, label: str) -> float:
    if (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    ):
        return float(value)
    try:
        rospy.logwarn(
            "%s is unavailable or invalid; using %.1f s diagnostic period",
            label,
            DEFAULT_DIAGNOSTICS_PERIOD_SEC,
        )
    except BaseException:
        pass
    return DEFAULT_DIAGNOSTICS_PERIOD_SEC


def _kv(key: str, value: Any) -> KeyValue:
    item = KeyValue()
    item.key = key
    item.value = "unavailable" if value is None else str(value)
    return item


def _required(params: Mapping[str, Any], key: str) -> Any:
    if key not in params:
        raise PipelineNodeConfigError(f"missing required parameter: {key}")
    return params[key]


def _absolute_ros_name(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or " " in value
        or "\x00" in value
    ):
        raise PipelineNodeConfigError(f"{label} must be an absolute ROS name")
    return value


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineNodeConfigError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise PipelineNodeConfigError(f"{label} must be positive and finite")
    return converted


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PipelineNodeConfigError(f"{label} must be a positive integer")
    return value


def _rospy_param_loader() -> Mapping[str, Any]:
    params = rospy.get_param("~", {})
    if not isinstance(params, Mapping):
        raise PipelineNodeConfigError("rospy private parameters are not a mapping")
    return params


def main(argv=None) -> int:
    rospy.init_node(NODE_NAME, argv=argv)
    node = PipelineServiceNode(param_loader=_rospy_param_loader, auto_start=True)
    if node.is_ready():
        rospy.loginfo(
            "camera pipeline service ready with roles: %s",
            sorted(role.role for role in node.config.roles),
        )
    else:
        rospy.logerr("camera pipeline service is not ready; see /diagnostics")
    rospy.on_shutdown(lambda: node.shutdown(timeout_sec=3.0))
    rospy.spin()
    return 0


__all__ = [
    "ActionlibCaptureClient",
    "ActionlibReconstructClient",
    "NODE_NAME",
    "PipelineNodeConfigError",
    "PipelineServiceNode",
    "assemble_pipeline_diagnostics",
    "load_pipeline_config",
    "main",
]
