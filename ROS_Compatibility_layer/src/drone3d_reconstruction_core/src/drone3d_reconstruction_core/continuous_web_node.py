#!/usr/bin/env python3
"""ROS1 目标 Web 节点（WP8）。

把 ROS-free 的 :class:`~drone3d_reconstruction_core.continuous_web.WebDashboardCore`
挂到 rospy：

- 参数加载：``session_command_service``、``result_root``、``roles``
  （与 continuous_session_node 相同的 front/down capture profile rosparam 布局）、
  ``web_host``、``web_port``、``web_preview_fps``、``max_point_cloud_points``
  以及可选的 ``max_session_windows``、``conf_percentile``、``diagnostics_topic``；
- get_status()：调用 ``ContinuousSessionCommand.GET_STATUS`` 服务，并合并
  ``/diagnostics`` 上最新一帧 DiagnosticArray（不发明 GPU/温度等字段）；
- send_command()：通过 service proxy 把 Web 控制命令映射到
  ``ContinuousSessionCommandRequest``；
- read_image()/read_depth()：订阅两个角色的 image_topic / depth_topic，
  cv_bridge 只负责把 ROS Image 转成 numpy（JPEG 编码在 Web 核心的 MJPEG
  路径中按需执行）；深度图在本节点用与 Web 核心相同的本地 turbo-like
  色带着色为 BGR，绝不伪造深度源；
- 不读 bag、不启动 roscore、不启动 worker。

本节点测试只做 CPU/假 proxy，见 test/test_continuous_web_node.py。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from drone3d_reconstruction_msgs.srv import (
    ContinuousSessionCommand,
    ContinuousSessionCommandRequest,
    ContinuousSessionCommandResponse,
)
import rospy

from .capture_node import load_capture_config
from .continuous_web import (
    DEFAULT_CONF_PERCENTILE,
    DEFAULT_MAX_PREDICTIONS_BYTES,
    DEFAULT_MAX_POINT_CLOUD_POINTS,
    DEFAULT_MAX_SESSION_WINDOWS,
    WebDashboardConfig,
    WebDashboardCore,
    WebDashboardError,
    WebRoleInfo,
    colorize_depth,
)
from .storage_retention import storage_usage

NODE_NAME = "continuous_web_node"
DEFAULT_DIAGNOSTICS_TOPIC = "/diagnostics"


class ContinuousWebNodeError(ValueError):
    """Web 节点参数缺失或非法。"""


def _required(params: Mapping[str, Any], key: str) -> Any:
    if key not in params:
        raise ContinuousWebNodeError(f"missing required parameter: {key}")
    return params[key]


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContinuousWebNodeError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ContinuousWebNodeError(f"{label} must be positive and finite")
    return converted


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContinuousWebNodeError(f"{label} must be a positive integer")
    return value


def _positive_byte_limit(value: Any, label: str) -> int:
    # ROS XML-RPC cannot carry integers above 2**31-1. Byte limits use decimal
    # strings in YAML, matching the capture/reconstruction configuration contract.
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        value = int(value)
    return _positive_integer(value, label)


def _optional_string(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key, "")
    if not isinstance(value, str):
        raise ContinuousWebNodeError(f"{key} must be a string")
    return value


def load_web_dashboard_config(
    params: Mapping[str, Any],
    *,
    capture_config_builder: Callable[..., Any] = load_capture_config,
) -> WebDashboardConfig:
    """从节点参数构造严格校验后的 WebDashboardConfig（CPU-only 可测）。"""
    if not isinstance(params, Mapping):
        raise ContinuousWebNodeError("node parameters must be a mapping")

    roles_param = _required(params, "roles")
    if not isinstance(roles_param, Mapping) or not roles_param:
        raise ContinuousWebNodeError("roles must be a nonempty mapping")
    roles = {}
    for name, entry in roles_param.items():
        if not isinstance(entry, Mapping):
            raise ContinuousWebNodeError(f"roles/{name} must be a mapping")
        capture_config = capture_config_builder(entry)
        roles[str(name)] = WebRoleInfo(
            image_topic=capture_config.image_topic,
            depth_topic=capture_config.depth_topic,
            camera_name=capture_config.camera_name,
            camera_frame_id=capture_config.camera_frame_id,
            snapshot_root=getattr(capture_config, "snapshot_root", None),
        )

    result_root = str(Path(_required(params, "result_root")))
    if not Path(result_root).is_absolute() or not Path(result_root).is_dir():
        raise ContinuousWebNodeError("result_root must be an existing directory")
    expected_reconstruction = params.get("expected_reconstruction", {})
    if not isinstance(expected_reconstruction, Mapping):
        raise ContinuousWebNodeError("expected_reconstruction must be a mapping")
    worker_result_root = str(expected_reconstruction.get("result_root", "") or "")
    journal_root = str(params.get("journal_root", "") or "")

    session_command_service = params.get(
        "session_command_service",
        params.get("service_name", ""),
    )
    if not isinstance(session_command_service, str) or not session_command_service:
        raise ContinuousWebNodeError(
            "session_command_service must be a nonempty string"
        )

    web_host = params.get("web_host", "127.0.0.1")
    if not isinstance(web_host, str) or not web_host.strip():
        raise ContinuousWebNodeError("web_host must be a nonempty string")
    if any(ord(char) < 32 for char in web_host):
        raise ContinuousWebNodeError("web_host contains control characters")
    web_port = params.get("web_port", 8080)
    if (
        isinstance(web_port, bool)
        or not isinstance(web_port, int)
        or web_port < 0
        or web_port > 65535
    ):
        raise ContinuousWebNodeError("web_port must be an integer in [0, 65535]")
    conf_percentile = params.get("conf_percentile", DEFAULT_CONF_PERCENTILE)
    if isinstance(conf_percentile, bool) or not isinstance(
        conf_percentile, (int, float)
    ):
        raise ContinuousWebNodeError("conf_percentile must be a finite number")
    conf_percentile = float(conf_percentile)
    if (
        not math.isfinite(conf_percentile)
        or conf_percentile < 0.0
        or conf_percentile > 100.0
    ):
        raise ContinuousWebNodeError("conf_percentile must be in [0, 100]")

    try:
        config = WebDashboardConfig(
            host=web_host,
            port=web_port,
            session_command_service=str(session_command_service),
            result_root=result_root,
            worker_result_root=worker_result_root,
            journal_root=journal_root,
            roles=roles,
            web_preview_fps=_positive_finite(
                params.get("web_preview_fps", 5.0), "web_preview_fps"
            ),
            max_session_windows=_positive_integer(
                params.get("max_session_windows", DEFAULT_MAX_SESSION_WINDOWS),
                "max_session_windows",
            ),
            max_point_cloud_points=_positive_integer(
                params.get("max_point_cloud_points", DEFAULT_MAX_POINT_CLOUD_POINTS),
                "max_point_cloud_points",
            ),
            max_predictions_bytes=_positive_byte_limit(
                params.get("max_predictions_bytes", DEFAULT_MAX_PREDICTIONS_BYTES),
                "max_predictions_bytes",
            ),
            conf_percentile=conf_percentile,
            frame_pose_fusion=params.get("frame_pose_fusion", False),
        )
    except (TypeError, ValueError, WebDashboardError) as error:
        raise ContinuousWebNodeError(str(error)) from error
    return config


def build_continuous_session_request(
    command_dict: Mapping[str, Any],
) -> ContinuousSessionCommandRequest:
    """把 Web 控制命令严格映射为 ContinuousSessionCommandRequest（CPU-only 可测）。

    Web 核心已经做过严格类型校验；这里仍保持防御式校验，避免绕过核心时
    构造坏请求。未知命令/字段抛 :class:`ContinuousWebNodeError`。
    """
    if not isinstance(command_dict, Mapping):
        raise ContinuousWebNodeError("command must be a mapping")
    command = command_dict.get("command")
    if not isinstance(command, str):
        raise ContinuousWebNodeError("command must be a string")
    request = ContinuousSessionCommandRequest()
    if command == "pause":
        request.command = ContinuousSessionCommandRequest.PAUSE
    elif command == "resume":
        request.command = ContinuousSessionCommandRequest.RESUME
    elif command == "stop_all":
        request.command = ContinuousSessionCommandRequest.STOP_ALL
    elif command == "get_status":
        request.command = ContinuousSessionCommandRequest.GET_STATUS
    elif command == "select_source":
        request.command = ContinuousSessionCommandRequest.SELECT_SOURCE
        request.role = _string_arg(command_dict, "role")
    elif command == "start_session":
        request.command = ContinuousSessionCommandRequest.START_SESSION
        request.role = _string_arg(command_dict, "role")
    else:
        raise ContinuousWebNodeError(f"unsupported web command: {command!r}")
    return request


def _string_arg(command_dict: Mapping[str, Any], key: str) -> str:
    value = command_dict.get(key)
    if not isinstance(value, str) or not value:
        raise ContinuousWebNodeError(f"{key} must be a nonempty string")
    return value


def command_response_to_dict(response) -> dict:
    """把 ContinuousSessionCommandResponse 转成 Web 核心可用的 dict。"""
    error_code = int(response.error_code)
    return {
        "ok": error_code == int(ContinuousSessionCommandResponse.OK),
        "error_code": error_code,
        "error": str(response.message or ""),
        "state": str(response.state or ""),
        "session_id": str(response.session_id or ""),
        "role": str(response.role or ""),
        "window_index": int(response.window_index),
        "captured_frames": int(response.captured_frames),
        "capture_target_frames": int(response.capture_target_frames),
        "pending_windows": int(response.pending_windows),
        "running_window": int(response.running_window),
        "succeeded_windows": int(response.succeeded_windows),
        "failed_windows": int(response.failed_windows),
        "dropped_windows": int(response.dropped_windows),
        "effective": {
            "sample_fps": float(response.effective_sample_fps),
            "window_frames": int(response.effective_window_frames),
            "overlap_frames": int(response.effective_overlap_frames),
            "queue_mode": str(response.effective_queue_mode or ""),
            "queue_capacity": int(response.effective_queue_capacity),
            "max_retries": int(response.effective_max_retries),
            "keep_snapshot_on_success": bool(
                response.effective_keep_snapshot_on_success
            ),
            "save_sensor_depth": bool(response.effective_save_sensor_depth),
            "inference_resolution": int(
                response.effective_inference_resolution
            ),
            "worker_timeout_sec": float(
                response.effective_worker_timeout_sec
            ),
            "reconstruction_mode": str(
                response.effective_reconstruction_mode or ""
            ),
        },
        "last_error": str(response.last_error or ""),
        "min_free_disk_gb": float(response.min_free_disk_gb),
    }


class ContinuousWebNode:
    """ROS Web 节点：配置 → 服务/订阅适配器 → WebDashboardCore 线程。"""

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        param_loader: Callable[[], Mapping[str, Any]] | None = None,
        config_builder: Callable[..., WebDashboardConfig] = load_web_dashboard_config,
        service_proxy_factory: Callable[..., Any] | None = None,
        subscriber_factory: Callable[..., Any] | None = None,
        core_factory: Callable[..., WebDashboardCore] = WebDashboardCore,
    ) -> None:
        if params is None and param_loader is not None:
            params = param_loader()
        if params is None:
            params = {}
        self._params = dict(params)
        self._config_builder = config_builder
        self._service_proxy_factory = service_proxy_factory
        self._subscriber_factory = subscriber_factory
        self._core_factory = core_factory

        self._config: WebDashboardConfig | None = None
        self._core: WebDashboardCore | None = None
        self._not_ready_error: str | None = None
        self._service = None
        self._diagnostics_sub = None
        self._image_subs = {}
        self._depth_subs = {}
        self._latest_images = {}
        self._latest_depths = {}
        self._last_image_preview = {}
        self._last_depth_preview = {}
        self._cv_bridge = None
        self._frames_lock = threading.Lock()
        self._diagnostics_lock = threading.Lock()
        self._diagnostics_by_name: dict = {}
        self._storage_disk_lock = threading.Lock()
        self._storage_disk_cached_at = 0.0
        self._storage_disk_cached = {}
        self._available_roles_lock = threading.Lock()
        self._available_roles_cached_at = 0.0
        self._available_roles_cached: list[str] = []

        try:
            self._config = self._config_builder(self._params)
        except BaseException as error:
            self._not_ready_error = f"{type(error).__name__}: {error}"
            return

        if self._service_proxy_factory is not None:
            self._service = self._service_proxy_factory(
                self._config.session_command_service, ContinuousSessionCommand
            )
        if self._subscriber_factory is not None:
            self._start_subscribers()
        self._core = self._core_factory(
            self._config,
            get_status=self._collect_status,
            send_command=self._send_service_command,
            read_image=self._read_image,
            read_depth=self._read_depth,
        )

    @property
    def config(self):
        return self._config

    @property
    def core(self):
        return self._core

    def is_ready(self) -> bool:
        return self._core is not None

    def start(self) -> None:
        if self._core is not None:
            self._core.start()

    def shutdown(self, timeout_sec: float = 2.0) -> None:
        if self._core is not None:
            self._core.stop(timeout_sec)
        for sub in list(self._image_subs.values()) + list(self._depth_subs.values()) + (
            [self._diagnostics_sub] if self._diagnostics_sub is not None else []
        ):
            if sub is None:
                continue
            try:
                sub.unregister()
            except BaseException:
                pass
        self._image_subs.clear()
        self._depth_subs.clear()
        self._diagnostics_sub = None

    # ------------------------------------------------------------------
    def _start_subscribers(self) -> None:
        diagnostics_topic = self._params.get(
            "diagnostics_topic", DEFAULT_DIAGNOSTICS_TOPIC
        )
        if isinstance(diagnostics_topic, str) and diagnostics_topic:
            self._diagnostics_sub = self._subscriber_factory(
                diagnostics_topic,
                DiagnosticArray,
                self._handle_diagnostics,
            )
        for role_name, role in self._config.roles.items():
            self._image_subs[role_name] = self._subscriber_factory(
                role.image_topic,
                _image_message_type(),
                lambda message, name=role_name: self._handle_image(name, message),
            )
            if role.depth_topic is not None:
                self._depth_subs[role_name] = self._subscriber_factory(
                    role.depth_topic,
                    _image_message_type(),
                    lambda message, name=role_name: self._handle_depth(name, message),
                )

    def _handle_diagnostics(self, message) -> None:
        for status in message.status:
            values = {}
            for item in status.values:
                values[str(item.key)] = str(item.value)
            item = {
                "name": str(status.name),
                "level": int(status.level),
                "message": str(status.message),
                "values": values,
            }
            with self._diagnostics_lock:
                self._diagnostics_by_name[str(status.name)] = item

    def _handle_image(self, role_name: str, message) -> None:
        received = time.monotonic()
        fps = (
            self._core.web_preview_fps
            if self._core is not None
            else self._config.web_preview_fps
        )
        with self._frames_lock:
            last_preview = self._last_image_preview.get(role_name)
            if last_preview is not None and received - last_preview < 1.0 / fps:
                return
            self._last_image_preview[role_name] = received
        try:
            frame = _image_to_bgr(message, self._bridge())
        except BaseException:
            return
        if frame is None:
            return
        stamp = _stamp_seconds(message)
        with self._frames_lock:
            self._latest_images[role_name] = (stamp, frame)

    def _handle_depth(self, role_name: str, message) -> None:
        received = time.monotonic()
        fps = (
            self._core.web_preview_fps
            if self._core is not None
            else self._config.web_preview_fps
        )
        with self._frames_lock:
            last_preview = self._last_depth_preview.get(role_name)
            if last_preview is not None and received - last_preview < 1.0 / fps:
                return
            self._last_depth_preview[role_name] = received
        try:
            raw = _image_to_numpy(message, self._bridge())
        except BaseException:
            return
        if raw is None:
            return
        try:
            colored = colorize_depth(raw)
        except BaseException:
            return
        # cv2.imencode 期望 BGR；colorize_depth 返回 RGB。
        bgr = colored[..., ::-1].copy()
        stamp = _stamp_seconds(message)
        with self._frames_lock:
            self._latest_depths[role_name] = (stamp, bgr)

    def _bridge(self):
        with self._frames_lock:
            if self._cv_bridge is None:
                self._cv_bridge = _new_cv_bridge()
            return self._cv_bridge

    def _read_image(self, role_name: str):
        with self._frames_lock:
            item = self._latest_images.get(role_name)
        return item

    def _read_depth(self, role_name: str):
        with self._frames_lock:
            item = self._latest_depths.get(role_name)
        return item

    def _storage_disk_usage(self) -> dict:
        now = time.monotonic()
        with self._storage_disk_lock:
            if now - self._storage_disk_cached_at < 10.0:
                return dict(self._storage_disk_cached)
            config = self._config
            usage = storage_usage(
                result_root=config.result_root if config is not None else "",
                worker_result_root=(
                    config.worker_result_root if config is not None else ""
                ),
                journal_root=config.journal_root if config is not None else "",
                snapshot_roots=(
                    tuple(
                        role.snapshot_root
                        for role in config.roles.values()
                        if role.snapshot_root
                    )
                    if config is not None
                    else ()
                ),
                runtime_log_root=(
                    str(Path(config.result_root).parent)
                    if config is not None
                    else ""
                ),
                ros_log_root=(
                    str(
                        os.environ.get("ROS_LOG_DIR")
                        or (Path.home() / ".ros" / "log")
                    )
                    if config is not None
                    else ""
                ),
            )
            self._storage_disk_cached_at = now
            self._storage_disk_cached = usage
            return dict(usage)

    def _available_roles(self) -> list[str]:
        """返回当前有图像发布者的相机角色（2 秒缓存）。"""
        now = time.monotonic()
        with self._available_roles_lock:
            if now - self._available_roles_cached_at < 2.0:
                return list(self._available_roles_cached)
            published: set[str] = set()
            try:
                published = {
                    str(topic)
                    for topic, _message_type in rospy.get_published_topics()
                }
            except BaseException:
                published = set()
            available = []
            if self._config is not None:
                for name, role in self._config.roles.items():
                    if role.image_topic in published:
                        available.append(str(name))
            self._available_roles_cached_at = now
            self._available_roles_cached = available
            return list(available)

    def _collect_status(self) -> dict:
        status = self._service_get_status()
        usage = self._storage_disk_usage()
        status.update(usage)
        status["capture_disk_bytes"] = usage["storage_capture_bytes"]
        status["available_roles"] = self._available_roles()
        with self._diagnostics_lock:
            diagnostics_list = list(self._diagnostics_by_name.values())
        diagnostics_list.sort(key=lambda item: str(item.get("name", "")))
        status["diagnostics"] = diagnostics_list
        worker_running_active = False
        for item in diagnostics_list:
            if not isinstance(item, dict):
                continue
            if item.get("name") == "drone3d/session/worker":
                try:
                    worker_window = int(
                        item.get("values", {}).get("running_window", 0)
                    )
                except (TypeError, ValueError):
                    worker_window = 0
                # window 0 and idle both serialize running_window as 0; the
                # producer's WARN/OK level carries the missing active bit.
                worker_running_active = int(item.get("level", 0)) == int(
                    DiagnosticStatus.WARN
                )
                if worker_running_active:
                    status["running_window"] = worker_window
            for key, value in item.get("values", {}).items():
                if key not in status:
                    status[key] = str(value)
        status["worker_running_active"] = worker_running_active
        return status

    def _service_get_status(self) -> dict:
        if self._service is None:
            return {"state": "UNAVAILABLE", "error": "no session command service"}
        request = build_continuous_session_request({"command": "get_status"})
        try:
            response = self._service(request)
        except BaseException as error:
            return {
                "state": "UNAVAILABLE",
                "error": f"{type(error).__name__}: {error}",
            }
        return command_response_to_dict(response)

    def _send_service_command(self, command_dict: dict) -> dict:
        if self._service is None:
            return {"ok": False, "error": "no session command service"}
        try:
            request = build_continuous_session_request(command_dict)
            response = self._service(request)
        except BaseException as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}"}
        return command_response_to_dict(response)


def _image_message_type():
    from sensor_msgs.msg import Image

    return Image


def _new_cv_bridge():
    from cv_bridge import CvBridge

    return CvBridge()


def _image_to_bgr(message, bridge=None):
    bridge = bridge if bridge is not None else _new_cv_bridge()
    return bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")


def _image_to_numpy(message, bridge=None):
    bridge = bridge if bridge is not None else _new_cv_bridge()
    return bridge.imgmsg_to_cv2(message)


def _stamp_seconds(message) -> float:
    try:
        return float(message.header.stamp.secs) + float(
            message.header.stamp.nsecs
        ) * 1e-9
    except BaseException:
        return 0.0


def _rospy_param_loader() -> Mapping[str, Any]:
    params = rospy.get_param("~", {})
    if not isinstance(params, Mapping):
        raise ContinuousWebNodeError("rospy private parameters are not a mapping")
    return params


def main(argv=None) -> int:
    rospy.init_node(NODE_NAME, argv=argv)
    node = ContinuousWebNode(
        param_loader=_rospy_param_loader,
        service_proxy_factory=rospy.ServiceProxy,
        subscriber_factory=rospy.Subscriber,
    )
    if node.is_ready():
        node.start()
        rospy.loginfo("continuous web dashboard ready")
        try:
            rospy.spin()
        finally:
            node.shutdown()
    else:
        rospy.logerr("continuous web dashboard is not ready: %s", node._not_ready_error)
        return 1
    return 0


if __name__ == "__main__":
    main()
