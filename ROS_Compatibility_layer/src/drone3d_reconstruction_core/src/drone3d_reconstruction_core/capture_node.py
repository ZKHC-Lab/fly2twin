"""生产级 ROS1 采集节点：显式配置 + /diagnostics 遥测。

本节点从私有参数（~）读取全部显式采集配置，构造 CaptureConfig 与
CaptureActionServer。任一配置缺失/非法都会让节点保持存活但 not-ready：
错误通过 /diagnostics 上报，而不是启动一个只会返回 CAMERA_UNAVAILABLE
的 Action 服务端。

主要功能：
  - load_capture_config：严格校验并构造不可变采集配置
  - CaptureNode：节点装配（服务端 + 诊断发布器）
  - /diagnostics 周期发布：服务端状态 + 输入新鲜度 + 会话计数

Production ROS1 capture node with /diagnostics telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Mapping

import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from drone3d_reconstruction_msgs.srv import (
    CaptureControl,
    CaptureControlRequest,
    CaptureControlResponse,
)

from .capture import CaptureConfig, CaptureConfigError
from .capture_action_server import CaptureActionServer
from .snapshot_validator import SnapshotSafetyLimits

NODE_NAME = "capture_production_node"
DEFAULT_ACTION_NAME = "capture_segment"
DEFAULT_DIAGNOSTICS_TOPIC = "/diagnostics"
DEFAULT_DIAGNOSTICS_PERIOD_SEC = 1.0
# SnapshotSafetyLimits 的全部字段名（用于严格校验参数键集合）
_SNAPSHOT_LIMIT_KEYS = frozenset(SnapshotSafetyLimits.__dataclass_fields__)

# 必填字符串参数
_REQUIRED_STR = (
    "snapshot_root",
    "image_topic",
    "camera_name",
    "camera_role",
    "camera_frame_id",
    "image_encoding",
)
# 必填浮点参数
_REQUIRED_FLOAT = (
    "min_sample_fps",
    "max_sample_fps",
    "max_goal_duration_sec",
    "pose_max_delta_ms",
    "pose_gap_max_ms",
    "max_sample_interval_overshoot_ms",
    "camera_info_max_delta_ms",
    "depth_max_delta_ms",
    "source_timeout_sec",
    "max_window_gap_sec",
)
# 必填整型参数
_REQUIRED_INT = (
    "max_goal_frames",
    "min_captured_frames",
    "max_image_width",
    "max_image_height",
    "max_auxiliary_samples",
    "max_tf_samples",
    "max_tf_bytes",
)


class CaptureNodeConfigError(ValueError):
    """节点参数缺失、非法或不安全。"""


@dataclass(frozen=True)
class CaptureRuntimeParams:
    """非安全性的运行时接线参数（Action 名/诊断话题/周期）。"""

    action_name: str
    diagnostics_topic: str
    diagnostics_period_sec: float


def load_runtime_params(params: Mapping[str, Any]) -> CaptureRuntimeParams:
    """读取运行时接线参数（仅使用常规接口默认值）。"""
    action_name = params.get("action_name", DEFAULT_ACTION_NAME)
    diagnostics_topic = params.get("diagnostics_topic", DEFAULT_DIAGNOSTICS_TOPIC)
    diagnostics_period_sec = params.get(
        "diagnostics_period_sec", DEFAULT_DIAGNOSTICS_PERIOD_SEC
    )
    if not isinstance(action_name, str) or not action_name or " " in action_name:
        raise CaptureNodeConfigError("action_name must be a ROS name")
    if (
        not isinstance(diagnostics_topic, str)
        or not diagnostics_topic.startswith("/")
        or " " in diagnostics_topic
    ):
        raise CaptureNodeConfigError("diagnostics_topic must be an absolute topic")
    if (
        isinstance(diagnostics_period_sec, bool)
        or not isinstance(diagnostics_period_sec, (int, float))
        or not _finite(diagnostics_period_sec)
        or diagnostics_period_sec <= 0
    ):
        raise CaptureNodeConfigError(
            "diagnostics_period_sec must be positive and finite"
        )
    return CaptureRuntimeParams(
        action_name=action_name,
        diagnostics_topic=diagnostics_topic,
        diagnostics_period_sec=float(diagnostics_period_sec),
    )


def load_capture_config(params: Mapping[str, Any]) -> CaptureConfig:
    """从节点参数构造严格校验后的不可变采集配置。

    所有必填键缺失、类型错误、snapshot_limits 键集不符都会抛
    CaptureNodeConfigError，保证服务端不会带着坏配置启动。
    """
    if not isinstance(params, Mapping):
        raise CaptureNodeConfigError("node parameters must be a mapping")
    values = {}
    for key in _REQUIRED_STR:
        values[key] = _required_string(params, key)
    for key in _REQUIRED_FLOAT:
        values[key] = _finite_number(_required(params, key), key)
    for key in _REQUIRED_INT:
        values[key] = _integer(_required(params, key), key)
    values["odometry_topic"] = _optional_topic(params, "odometry_topic")
    values["camera_info_topic"] = _optional_topic(params, "camera_info_topic")
    values["depth_topic"] = _optional_topic(params, "depth_topic")
    values["tf_topic"] = _optional_topic(params, "tf_topic")
    values["tf_static_topic"] = _optional_topic(params, "tf_static_topic")
    values["save_sensor_depth"] = _required_bool(params, "save_sensor_depth")
    # Optional keys: an absent key leaves the CaptureConfig field default in force.
    if "odometry_association" in params:
        values["odometry_association"] = params["odometry_association"]
    for key in ("pose_time_offset_ms", "pose_interpolation_wait_ms", "keyframe_min_translation_m",
                "keyframe_min_rotation_deg", "keyframe_max_interval_sec"):
        if key in params:
            values[key] = _finite_number(params[key], key)
    for key in ("handoff_buffer_frames", "handoff_buffer_max_bytes"):
        if key in params:
            values[key] = _integer(params[key], key)
    body_frame_id = _optional_string(params, "body_frame_id")
    values["body_frame_id"] = body_frame_id
    values["tf_relevant_frames"] = tuple(
        _frame_id_list(_required(params, "tf_relevant_frames"))
    )

    snapshot_limits = _required(params, "snapshot_limits")
    if not isinstance(snapshot_limits, Mapping):
        raise CaptureNodeConfigError("snapshot_limits must be a mapping")
    # 键集必须与 SnapshotSafetyLimits 字段完全一致（不允许缺键/多余键）
    if set(snapshot_limits) != _SNAPSHOT_LIMIT_KEYS:
        raise CaptureNodeConfigError(
            "snapshot_limits must contain exactly "
            f"{sorted(_SNAPSHOT_LIMIT_KEYS)!r}"
        )
    values["snapshot_limits"] = SnapshotSafetyLimits(
        **{
            key: _integer(value, f"snapshot_limits/{key}")
            for key, value in snapshot_limits.items()
        }
    )
    try:
        return CaptureConfig(**values)
    except CaptureConfigError as error:
        raise CaptureNodeConfigError(str(error)) from error
    except TypeError as error:
        raise CaptureNodeConfigError(
            f"capture config missing a field: {error}"
        ) from error


class CaptureDiagnosticsState:
    """线程安全的诊断状态（配置 + not-ready 错误），供 /diagnostics 使用。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.not_ready_error: str | None = None
        self.config: CaptureConfig | None = None

    def set_config(self, config: CaptureConfig) -> None:
        """登记有效配置并清除 not-ready 错误。"""
        with self._lock:
            self.config = config
            self.not_ready_error = None

    def set_not_ready(self, error: BaseException) -> None:
        """登记 not-ready 错误（节点保持存活但不可用）。"""
        with self._lock:
            self.not_ready_error = f"{type(error).__name__}: {error}"

    def snapshot(self):
        """线程安全的诊断快照。"""
        with self._lock:
            return {"config": self.config, "not_ready_error": self.not_ready_error}


class CaptureNode:
    """采集监督节点：参数 → 配置 → 服务端 → 诊断发布器。"""

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        param_loader: Callable[[], Mapping[str, Any]] | None = None,
        config_builder: Callable[..., CaptureConfig] = load_capture_config,
        runtime_builder: Callable[..., CaptureRuntimeParams] = load_runtime_params,
        server_factory: Callable[..., CaptureActionServer] = CaptureActionServer,
        publisher_factory: Callable[..., Any] = rospy.Publisher,
        timer_factory: Callable[..., Any] = rospy.Timer,
        control_service_factory: Callable[..., Any] | None = None,
        auto_start: bool = False,
    ) -> None:
        """构造节点；任何装配失败都进入 not-ready 状态而非崩溃。

        Args:
            params: 节点参数字典（与 param_loader 二选一）
            param_loader: 参数加载函数（如 rospy 私有参数）
            config_builder: 配置构造器（测试注入点）
            runtime_builder: 运行时参数构造器（测试注入点）
            server_factory: Action 服务端工厂（测试注入点）
            publisher_factory / timer_factory: ROS 发布器/定时器工厂
            control_service_factory: CaptureControl 服务工厂（生产为
                rospy.Service；CPU 测试可注入 fake）
            auto_start: 构造后立即开始发布诊断
        """
        if param_loader is not None:
            if params is not None:
                raise TypeError("provide params or param_loader, not both")
            params = param_loader()
        if params is None:
            raise TypeError("node parameters are required")
        self._params = dict(params)
        self._state = CaptureDiagnosticsState()
        self._server = None
        self._runtime = None
        self._config = None
        self._publisher = None
        self._timer = None
        self._control_service = None
        self._publisher_factory = publisher_factory
        self._timer_factory = timer_factory
        self._control_service_factory = control_service_factory

        try:
            self._runtime = runtime_builder(params)
            self._config = config_builder(params)
            self._state.set_config(self._config)
            self._server = server_factory(
                action_name=self._runtime.action_name,
                config=self._config,
                terminal_observer=self._on_terminal,
            )
            if self._control_service_factory is not None:
                self._control_service = self._control_service_factory(
                    capture_control_service_name(self._runtime.action_name),
                    CaptureControl,
                    self._handle_capture_control,
                )
        except Exception as error:
            # 配置或装配失败：记录错误，节点保持存活供诊断
            self._state.set_not_ready(error)
        if auto_start:
            self.start_publishing()

    @property
    def server(self) -> CaptureActionServer | None:
        return self._server

    def is_ready(self) -> bool:
        """服务端是否已成功构造。"""
        return self._server is not None

    def start_publishing(self) -> None:
        """创建诊断发布器与周期定时器。"""
        if self._publisher is not None:
            return
        if self._runtime is None:
            return
        self._publisher = self._publisher_factory(
            self._runtime.diagnostics_topic, DiagnosticArray, queue_size=1
        )
        self._timer = self._timer_factory(
            rospy.Duration.from_sec(self._runtime.diagnostics_period_sec),
            self._publish_callback,
        )

    def publish_once(self) -> None:
        """立即发布一次诊断（供测试）。"""
        self._publish_callback(None)

    def _handle_capture_control(
        self, request: CaptureControlRequest
    ) -> CaptureControlResponse:
        response = CaptureControlResponse()
        if self._server is None:
            response.error_code = CaptureControlResponse.INTERNAL_ERROR
            response.message = "capture server is not ready"
            return response
        try:
            if request.command == CaptureControlRequest.PAUSE:
                accepted = self._server.pause()
            elif request.command == CaptureControlRequest.RESUME:
                accepted = self._server.resume()
            else:
                response.error_code = CaptureControlResponse.INVALID_COMMAND
                response.message = f"unknown capture control command {request.command}"
                return response
        except BaseException as error:
            response.error_code = CaptureControlResponse.INTERNAL_ERROR
            response.message = f"capture control failed: {type(error).__name__}"
            return response
        if not accepted:
            response.error_code = CaptureControlResponse.NO_ACTIVE_SESSION
            response.message = "no active capture session"
            return response
        response.error_code = CaptureControlResponse.OK
        response.message = "ok"
        response.paused = bool(self._server.is_paused())
        return response

    def _on_terminal(self, method, result, text):
        """终态观察者：终态到达即代表已恢复 ready。"""
        with self._state._lock:
            self._state.not_ready_error = None

    def _publish_callback(self, _event):
        """周期发布回调：组装并发布 DiagnosticArray。"""
        if self._publisher is None:
            return
        state = self._state.snapshot()
        server_snapshot = (
            self._server.diagnostics_snapshot()
            if self._server is not None
            else {"running": False, "stage": "not_ready"}
        )
        message = assemble_diagnostics_array(state, server_snapshot)
        try:
            message.header.stamp = rospy.Time.now()
        except BaseException:
            message.header.stamp = rospy.Time(0)
        self._publisher.publish(message)

    def shutdown(self, timeout_sec: float = 3.0) -> None:
        """关闭节点：停止定时器、关闭服务端、注销发布器。"""
        if self._timer is not None:
            try:
                self._timer.shutdown()
            except BaseException:
                pass
        if self._server is not None:
            self._server.shutdown(timeout_sec)
        if self._control_service is not None:
            try:
                self._control_service.shutdown("capture node shutting down")
            except BaseException:
                pass
        if self._publisher is not None:
            try:
                self._publisher.unregister()
            except BaseException:
                pass


def assemble_diagnostics_array(state, server) -> DiagnosticArray:
    """纯函数组装一条 /diagnostics 消息（服务端 + 输入两个状态）。"""
    message = DiagnosticArray()
    config = state.get("config")
    not_ready = state.get("not_ready_error")
    message.status = [
        _server_status(config, not_ready),
        _input_status(config, server, not_ready),
    ]
    return message


def _server_status(config, not_ready):
    """服务端状态：not-ready 时 ERROR，否则 OK + 配置摘要。"""
    status = DiagnosticStatus()
    role = config.camera_role if config is not None else "unknown"
    status.name = f"drone3d/capture/{role}/server"
    if not_ready is not None or config is None:
        status.level = DiagnosticStatus.ERROR
        status.message = "not ready: " + (not_ready or "no capture config")
        status.values = [kv("error", status.message)]
        return status
    status.level = DiagnosticStatus.OK
    status.message = "ready"
    status.values = [
        kv("camera_name", config.camera_name),
        kv("camera_role", config.camera_role),
        kv("camera_frame_id", config.camera_frame_id),
        kv("image_topic", config.image_topic),
        kv("image_encoding", config.image_encoding),
        kv("odometry_topic", config.odometry_topic or "unavailable"),
        kv("snapshot_root", config.snapshot_root),
    ]
    return status


def _input_status(config, server, not_ready):
    """输入状态：运行中的会话计数、新鲜度与最近终态。"""
    status = DiagnosticStatus()
    role = config.camera_role if config is not None else "unknown"
    status.name = f"drone3d/capture/{role}/input"
    if not_ready is not None:
        status.level = DiagnosticStatus.ERROR
        status.message = "capture server not constructed"
        status.values = [kv("error", status.message)]
        return status
    running = bool(server.get("running", False))
    status.level = DiagnosticStatus.OK
    status.message = "stage " + str(server.get("stage", "idle")) if running else "idle"
    values = [
        kv("running", running),
        kv("stage", server.get("stage", "idle")),
        kv("input_age_image_sec", server.get("latest_image_wall_age_sec", "unavailable")),
        kv("handoff_queued_frames", server.get("handoff_queued_frames", 0)),
        kv("handoff_queued_bytes", server.get("handoff_queued_bytes", 0)),
        kv("handoff_evicted_frames", server.get("handoff_evicted_frames", 0)),
        kv("handoff_replayed_frames", server.get("handoff_replayed_frames", 0)),
        kv("handoff_decimated_frames", server.get("handoff_decimated_frames", 0)),
        kv("input_age_odometry_sec", server.get("latest_odometry_wall_age_sec", "unavailable")),
        kv("last_result", _last_result_text(server)),
    ]
    session = server.get("session")
    if session:
        values.extend(
            [
                kv("received_images", session["received_images"]),
                kv("skipped_images", session["skipped_images"]),
                kv("selected_frames", session["selected_frames"]),
                kv("odometry_gaps", session["odometry_gaps"]),
                kv("odometry_regressions", session["odometry_regressions"]),
                kv("auxiliary_errors", session.get("auxiliary_errors", 0)),
                kv("auxiliary_dropped", session.get("auxiliary_dropped", 0)),
                kv("tf_dropped", session.get("tf_dropped", 0)),
                kv("last_auxiliary_error", session.get("last_auxiliary_error", "")),
                kv("source_stale", session["source_stale"]),
                kv("source_stale_entries", session["source_stale_entries"]),
                kv("zero_stamp_count", session["zero_stamp_count"]),
                kv("duplicate_stamp_count", session["duplicate_stamp_count"]),
                kv("out_of_order_stamp_count", session["out_of_order_stamp_count"]),
                kv(
                    "last_image_wall_age_sec",
                    session.get("last_image_wall_age_sec", "unavailable"),
                ),
                kv(
                    "aux_coverage",
                    "ci{ci} odom{odom} depth{depth} tf{tf}".format(
                        ci="+" if session.get("has_camera_info") else "-",
                        odom="+" if session.get("has_odometry") else "-",
                        depth="+" if session.get("has_depth") else "-",
                        tf="+" if session.get("has_tf") else "-",
                    ),
                ),
            ]
        )
    status.values = values
    return status


def _last_result_text(server):
    """把最近终态格式化为 method/error_code/message 文本。"""
    last = server.get("last_terminal")
    if not last:
        return "unavailable"
    return f"{last['method']}/{last['error_code']}/{last['message']}"


def kv(key, value):
    """构造 KeyValue 项（None 值显示为 unavailable）。"""
    item = KeyValue()
    item.key = key
    item.value = "unavailable" if value is None else str(value)
    return item


def capture_control_service_name(action_name: str) -> str:
    """从 Capture Action 名推导同命名空间下的 control 服务名。"""
    if not isinstance(action_name, str) or not action_name.startswith("/"):
        raise CaptureNodeConfigError("action_name must be an absolute action name")
    parts = action_name.rstrip("/").split("/")
    parts[-1] = "capture_control"
    return "/" + "/".join(parts[1:])


def _required(params, key):
    """读取必填参数，缺失即抛配置错误。"""
    if key not in params:
        raise CaptureNodeConfigError(f"missing required parameter: {key}")
    return params[key]


def _required_string(params, key):
    """读取必填非空字符串（拒绝 NUL）。"""
    value = _required(params, key)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CaptureNodeConfigError(f"{key} must be a nonempty string")
    return value


def _optional_topic(params, key):
    """读取可选话题：空串/None → None，否则必须是绝对话题。"""
    value = _required(params, key)
    if value in ("", None):
        return None
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise CaptureNodeConfigError(f"{key} must be empty or an absolute topic")
    return value


def _optional_string(params, key):
    """读取可选字符串：空串/None → None，否则必须是非空无 NUL 字符串。"""
    value = _required(params, key)
    if value in ("", None):
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise CaptureNodeConfigError(f"{key} must be empty or a string")
    return value


def _finite(value):
    """校验数值有限（拒绝 NaN/无穷）。"""
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    import math

    return math.isfinite(converted)


def _finite_number(value, key):
    """要求值为有限数值（拒绝 bool）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptureNodeConfigError(f"{key} must be a finite number")
    converted = float(value)
    if converted != converted or converted in (float("inf"), float("-inf")):
        raise CaptureNodeConfigError(f"{key} must be finite")
    return converted


def _integer(value, key):
    """要求值为整数（拒绝 bool）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaptureNodeConfigError(f"{key} must be an integer")
    return value


def _required_bool(params, key):
    """要求值为 bool。"""
    value = _required(params, key)
    if type(value) is not bool:
        raise CaptureNodeConfigError(f"{key} must be a bool")
    return value


def _frame_id_list(value):
    """要求为字符串列表，每项都是安全 frame id。"""
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item or "\x00" in item for item in value
    ):
        raise CaptureNodeConfigError(
            "tf_relevant_frames must be a list of frame id strings"
        )
    return value


def _rospy_param_loader() -> Mapping[str, Any]:
    """从 rospy 私有参数命名空间 (~) 读取全部参数。"""
    params = rospy.get_param("~", {})
    if not isinstance(params, Mapping):
        raise CaptureNodeConfigError("rospy private parameters are not a mapping")
    return params


def main(argv=None) -> int:
    """rospy 入口：scripts/capture_production_node.py 的实际执行体。"""
    rospy.init_node(NODE_NAME, argv=argv)
    node = CaptureNode(
        param_loader=_rospy_param_loader,
        control_service_factory=rospy.Service,
        auto_start=True,
    )
    if node.is_ready():
        rospy.loginfo("capture server ready for %s", node._config.image_topic)
    else:
        rospy.logerr("capture server is not ready; see /diagnostics")
    rospy.on_shutdown(lambda: node.shutdown(timeout_sec=3.0))
    rospy.spin()
    return 0


__all__ = [
    "CaptureNode",
    "CaptureNodeConfigError",
    "assemble_diagnostics_array",
    "capture_control_service_name",
    "load_capture_config",
    "load_runtime_params",
]
