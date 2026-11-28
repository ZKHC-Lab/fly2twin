"""目标 Web 显示、状态和控制（ROS 无关核心）。

本模块是 WP8 的 ROS-free 核心。它提供：

- :class:`WebDashboardConfig`：不可变 Web 配置，构造时严格 preflight；
- :class:`WebDashboardCore`：可注入回调的仪表盘核心，拥有线程安全状态机
  （follow_latest / selected_session / web_preview_fps / last_command_result），
  并基于 ``http.server.ThreadingHTTPServer`` 提供完整的生产 Web 路由；
- 正式结果只读访问：不修改 worker result、不修改正式记录；点云读取遵循
  ``formal_result.py`` 的严格路径风格（file:// only、绝对路径、不跟随符号
  链接），路径参数全部经正则/整数白名单校验；
- 点云累计：同一 ``map_id`` 的已对齐窗口按 provenance 里的
  Sim(3)（``scale * R @ x + t``）合并；没有对齐窗口时回退到所选 session
  最新一个 ``MODEL_LOCAL_UNALIGNED`` 窗口且不合并。原始点数、显示点数、
  采样方式、等效采样倍率、置信度百分位始终通过响应头或
  ``/api/pointcloud_meta`` 暴露。

本模块不导入 rospy，不导入 trimesh/gradio/flask/tornado。numpy 与 PIL
仅在点云/帧编码路径惰性使用；cv2 仅在 MJPEG 编码路径惰性使用。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer
from typing import Any, Callable, Mapping, Optional

from .odom_local import (  # noqa: F401  re-exported for callers of this module
    COORDINATE_MODEL_LOCAL_UNALIGNED,
    COORDINATE_ODOM_LOCAL_APPROX,
)

# Shared helpers, point-cloud building and the HTTP server live in their own modules;
# WebDashboardCore combines them and this module keeps the original import path.
from .continuous_web_common import (  # noqa: F401  colorize_depth, _formal_records_for re-exported
    _SAFE_FRAME_ID_RE,
    _SAFE_ROLE_RE,
    _SESSION_ID_RE,
    _WINDOW_DIR_RE,
    WebDashboardError,
    _dashboard_html,
    _find_window_record,
    _formal_records_for,
    _latest_window_record,
    _load_frame_image,
    _parse_bool,
    _parse_finite_positive,
    _parse_nonnegative_int,
    _parse_session_id,
    _png_bytes,
    _record_alignment,
    _record_coordinate_state,
    colorize_depth,
)
from .continuous_web_http import HttpServerMixin
from .continuous_web_pointcloud import PointCloudMixin

DEFAULT_MAX_PREDICTIONS_BYTES = 2 * 1024 * 1024 * 1024
_FRAME_CACHE_MAX_ENTRIES = 32

DEFAULT_MAX_POINT_CLOUD_POINTS = 1_000_000
DEFAULT_MAX_SESSION_WINDOWS = 64
DEFAULT_CONF_PERCENTILE = 20.0
MAX_WEB_PREVIEW_FPS = 30.0

_COMMAND_KEYS = {
    "pause": frozenset(),
    "resume": frozenset(),
    "stop_all": frozenset(),
    "select_source": frozenset(("role",)),
    "start_session": frozenset(("role",)),
    "follow_latest": frozenset(("value",)),
    "set_preview_fps": frozenset(("value",)),
    "select_session": frozenset(("session_id",)),
}
_COMMANDS_THAT_FORWARD = frozenset(
    ("pause", "resume", "stop_all", "select_source", "start_session")
)


@dataclass(frozen=True)
class WebRoleInfo:
    """一个相机角色的 Web 展示/订阅信息。"""

    image_topic: str
    camera_name: str
    camera_frame_id: str
    depth_topic: Optional[str] = None
    snapshot_root: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_topic, str) or not self.image_topic.startswith(
            "/"
        ) or any(ord(char) < 32 for char in self.image_topic):
            raise WebDashboardError(
                "role image_topic must be an absolute ROS topic without control chars"
            )
        if not isinstance(self.camera_name, str) or not _SAFE_ROLE_RE.fullmatch(
            self.camera_name
        ):
            raise WebDashboardError("role camera_name must be a safe identifier")
        if not isinstance(self.camera_frame_id, str) or not _SAFE_FRAME_ID_RE.fullmatch(
            self.camera_frame_id
        ):
            raise WebDashboardError("role camera_frame_id must be a safe frame id")
        if self.depth_topic is not None:
            if not isinstance(self.depth_topic, str) or not self.depth_topic.startswith(
                "/"
            ) or any(ord(char) < 32 for char in self.depth_topic):
                raise WebDashboardError(
                    "role depth_topic must be an absolute ROS topic or None"
                )
        if self.snapshot_root is not None:
            if (
                not isinstance(self.snapshot_root, str)
                or not self.snapshot_root
                or any(ord(char) < 32 for char in self.snapshot_root)
            ):
                raise WebDashboardError(
                    "role snapshot_root must be a nonempty path string or None"
                )


@dataclass(frozen=True)
class WebDashboardConfig:
    """Web 仪表盘配置；preflight 失败即拒绝启动。"""

    host: str
    port: int
    session_command_service: str
    result_root: str
    roles: Mapping[str, WebRoleInfo]
    web_preview_fps: float
    worker_result_root: str = ""
    journal_root: str = ""
    max_session_windows: int = DEFAULT_MAX_SESSION_WINDOWS
    max_point_cloud_points: int = DEFAULT_MAX_POINT_CLOUD_POINTS
    max_predictions_bytes: int = DEFAULT_MAX_PREDICTIONS_BYTES
    conf_percentile: float = DEFAULT_CONF_PERCENTILE
    frame_pose_fusion: bool = False

    def __post_init__(self) -> None:
        if type(self.frame_pose_fusion) is not bool:
            raise WebDashboardError("frame_pose_fusion must be a bool")
        if not isinstance(self.host, str) or not self.host.strip():
            raise WebDashboardError("host must be a nonempty string")
        if any(ord(char) < 32 for char in self.host):
            raise WebDashboardError("host contains control characters")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise WebDashboardError("port must be an integer")
        if self.port < 0 or self.port > 65535:
            raise WebDashboardError("port must be in [0, 65535]")
        if not isinstance(self.session_command_service, str) or not (
            self.session_command_service.startswith("/")
        ) or " " in self.session_command_service:
            raise WebDashboardError(
                "session_command_service must be an absolute ROS service name"
            )
        if not isinstance(self.result_root, str) or not self.result_root:
            raise WebDashboardError("result_root must be a nonempty string")
        root = Path(self.result_root)
        if not root.is_absolute():
            raise WebDashboardError("result_root must be an absolute path")
        if not root.is_dir() or root.is_symlink():
            raise WebDashboardError("result_root must be an existing directory")
        for name in ("worker_result_root", "journal_root"):
            value = getattr(self, name)
            if value:
                path = Path(value)
                if not path.is_absolute() or not path.is_dir() or path.is_symlink():
                    raise WebDashboardError(
                        f"{name} must be empty or an existing real directory"
                    )
        if not isinstance(self.roles, Mapping) or not self.roles:
            raise WebDashboardError("roles must be a nonempty mapping")
        normalized_roles = {}
        for name, entry in self.roles.items():
            if not isinstance(name, str) or not _SAFE_ROLE_RE.fullmatch(name):
                raise WebDashboardError("role name must be a safe identifier")
            if isinstance(entry, WebRoleInfo):
                normalized_roles[name] = entry
            elif isinstance(entry, Mapping):
                try:
                    normalized_roles[name] = WebRoleInfo(**dict(entry))
                except TypeError as error:
                    raise WebDashboardError(
                        f"roles/{name} has invalid fields: {error}"
                    ) from error
            else:
                raise WebDashboardError(f"roles/{name} must be a mapping or WebRoleInfo")
        object.__setattr__(self, "roles", normalized_roles)
        fps = self.web_preview_fps
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise WebDashboardError("web_preview_fps must be a finite number")
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0 or fps > MAX_WEB_PREVIEW_FPS:
            raise WebDashboardError(
                f"web_preview_fps must be in (0, {MAX_WEB_PREVIEW_FPS:g}]"
            )
        object.__setattr__(self, "web_preview_fps", fps)
        if (
            isinstance(self.max_session_windows, bool)
            or not isinstance(self.max_session_windows, int)
            or self.max_session_windows <= 0
        ):
            raise WebDashboardError("max_session_windows must be a positive integer")
        if (
            isinstance(self.max_point_cloud_points, bool)
            or not isinstance(self.max_point_cloud_points, int)
            or self.max_point_cloud_points <= 0
        ):
            raise WebDashboardError("max_point_cloud_points must be a positive integer")
        if (
            isinstance(self.max_predictions_bytes, bool)
            or not isinstance(self.max_predictions_bytes, int)
            or self.max_predictions_bytes <= 0
        ):
            raise WebDashboardError("max_predictions_bytes must be a positive integer")
        if isinstance(self.conf_percentile, bool) or not isinstance(
            self.conf_percentile, (int, float)
        ):
            raise WebDashboardError("conf_percentile must be a finite number")
        conf = float(self.conf_percentile)
        if not math.isfinite(conf) or conf < 0.0 or conf > 100.0:
            raise WebDashboardError("conf_percentile must be in [0, 100]")
        object.__setattr__(self, "conf_percentile", conf)

    @property
    def formal_records_root(self) -> Path:
        """正式记录根：``<result_root>/sessions``。"""
        return Path(self.result_root) / "sessions"

    def role_by_name(self, role: str) -> WebRoleInfo:
        return self.roles[role]


class WebDashboardCore(PointCloudMixin, HttpServerMixin):
    """ROS 无关的 Web 仪表盘核心。

    所有回调均可注入：测试注入假回调，ROS 节点注入真实服务/订阅适配器。
    """

    def __init__(
        self,
        config: WebDashboardConfig,
        *,
        get_status: Callable[[], dict] | None = None,
        send_command: Callable[[dict], dict] | None = None,
        list_sessions: Callable[[], list] | None = None,
        read_image: Callable[[str], Any] | None = None,
        read_depth: Callable[[str], Any] | None = None,
        now: Callable[[], float] = time.monotonic,
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
    ) -> None:
        if not isinstance(config, WebDashboardConfig):
            raise TypeError("config must be WebDashboardConfig")
        if get_status is not None and not callable(get_status):
            raise TypeError("get_status must be callable")
        if send_command is not None and not callable(send_command):
            raise TypeError("send_command must be callable")
        if list_sessions is not None and not callable(list_sessions):
            raise TypeError("list_sessions must be callable")
        if read_image is not None and not callable(read_image):
            raise TypeError("read_image must be callable")
        if read_depth is not None and not callable(read_depth):
            raise TypeError("read_depth must be callable")
        if not callable(now) or not callable(thread_factory):
            raise TypeError("now/thread_factory must be callable")
        self._config = config
        self._get_status = get_status if get_status is not None else (lambda: {})
        self._send_command = send_command if send_command is not None else (
            lambda _cmd: {"ok": False, "error": "no command transport configured"}
        )
        self._list_sessions = list_sessions if list_sessions is not None else (
            self._scan_sessions
        )
        self._read_image = read_image
        self._read_depth = read_depth
        self._now = now
        self._thread_factory = thread_factory
        self._lock = threading.Lock()
        self._follow_latest = True
        self._selected_session_id = ""
        self._web_preview_fps = float(config.web_preview_fps)
        self._last_command_result: dict = {}
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_cache: dict[tuple, bytes] = {}
        self._frame_cache_order: list[tuple] = []
        self._pc_lock = threading.Lock()
        # ponytail: one cold-build lock; use per-key locks if distinct-map
        # point-cloud build throughput ever matters.
        self._pc_build_lock = threading.Lock()
        self._pc_cache: dict[tuple, dict] = {}
        self._pc_order: list[tuple] = []
        # session id -> drone3d_onboard.session.SessionFusion (re-fuses only frames whose inputs changed)
        self._session_fusion: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 状态机
    # ------------------------------------------------------------------
    def _status_diagnostics(self) -> dict:
        try:
            status = self._get_status()
        except BaseException as error:
            status = {"state": "UNAVAILABLE", "error": f"{type(error).__name__}: {error}"}
        if not isinstance(status, dict):
            status = {"state": "INVALID", "error": "get_status() did not return a dict"}
        return status

    def _scan_sessions(self) -> list:
        """Build a cheap directory index without validating formal artifacts.

        The 1 Hz state path must not hash manifests or materialize prediction
        files.  Candidate window names are only a freshness/count hint; the
        detailed records endpoint still uses :func:`scan_session_windows` and
        is the sole source of trusted records.
        """
        sessions_root = self._config.formal_records_root
        sessions = []
        try:
            if not sessions_root.is_dir() or sessions_root.is_symlink():
                return []
            for entry in sessions_root.iterdir():
                if not entry.is_dir() or entry.is_symlink():
                    continue
                if not _SESSION_ID_RE.fullmatch(entry.name):
                    continue
                window_indexes = []
                try:
                    window_entries = entry.iterdir()
                    for window_entry in window_entries:
                        match = _WINDOW_DIR_RE.fullmatch(window_entry.name)
                        if match is None:
                            continue
                        if not window_entry.is_dir() or window_entry.is_symlink():
                            continue
                        window_indexes.append(int(match.group(1)))
                except OSError:
                    continue
                if not window_indexes:
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    mtime = 0.0
                sessions.append(
                    {
                        "session_id": entry.name,
                        "window_count": len(window_indexes),
                        "latest_window_index": max(window_indexes),
                        "mtime": mtime,
                    }
                )
        except OSError:
            return []
        sessions.sort(key=lambda item: (item.get("mtime", 0.0), item["session_id"]), reverse=True)
        sessions = sessions[: self._config.max_session_windows]
        for item in sessions:
            item.pop("mtime", None)
        return sessions

    def _refresh_selected_session_locked(
        self, sessions: list, diagnostics: dict
    ) -> None:
        """只做内存状态选择；不得调用任何外部回调（必须在持锁时调用）。"""
        if not self._follow_latest:
            return
        preferred = ""
        current_session = diagnostics.get("session_id")
        if isinstance(current_session, str) and _SESSION_ID_RE.fullmatch(current_session):
            preferred = current_session
        if not preferred and sessions:
            preferred = str(sessions[0].get("session_id", ""))
        self._selected_session_id = preferred

    def select_session(self, session_id: str) -> dict:
        session_id = _parse_session_id(session_id)
        sessions = self._safe_list_sessions()
        known = {str(item.get("session_id")) for item in sessions}
        if session_id not in known:
            return {"ok": False, "error": "unknown session"}
        with self._lock:
            self._selected_session_id = session_id
            self._follow_latest = False
        return {
            "ok": True,
            "error": "",
            "response": {
                "selected_session": session_id,
                "follow_latest": False,
            },
        }

    def set_follow_latest(self, value: bool) -> dict:
        if type(value) is not bool:
            return {"ok": False, "error": "value must be a bool"}
        sessions = []
        diagnostics = {}
        if value:
            # 外部回调必须在锁外执行；只把快照带进锁内。
            diagnostics = self._status_diagnostics()
            sessions = self._safe_list_sessions()
        with self._lock:
            self._follow_latest = value
            if value:
                self._refresh_selected_session_locked(sessions, diagnostics)
            selected = self._selected_session_id
        return {
            "ok": True,
            "error": "",
            "response": {
                "follow_latest": value,
                "selected_session": selected,
            },
        }

    def set_preview_fps(self, value: float) -> dict:
        try:
            value = _parse_finite_positive(value, "value")
        except WebDashboardError as error:
            return {"ok": False, "error": str(error)}
        if value > MAX_WEB_PREVIEW_FPS:
            return {
                "ok": False,
                "error": f"value must be <= {MAX_WEB_PREVIEW_FPS:g}",
            }
        with self._lock:
            self._web_preview_fps = value
        return {"ok": True, "error": "", "response": {"preview_fps": value}}

    def _safe_list_sessions(self) -> list:
        try:
            sessions = self._list_sessions()
        except BaseException as error:
            return [
                {
                    "session_id": "",
                    "window_count": 0,
                    "latest_window_index": -1,
                    "error": f"{type(error).__name__}: {error}",
                }
            ]
        if not isinstance(sessions, list):
            return []
        return [item for item in sessions if isinstance(item, dict)]

    # ------------------------------------------------------------------
    # 命令校验与分发
    # ------------------------------------------------------------------
    def _validate_command_payload(self, payload: dict) -> dict:
        command = payload.get("command")
        if not isinstance(command, str) or command not in _COMMAND_KEYS:
            raise WebDashboardError(
                f"command must be one of {sorted(_COMMAND_KEYS)!r}"
            )
        allowed = _COMMAND_KEYS[command]
        unknown = sorted(set(payload.keys()) - {"command"} - allowed)
        if unknown:
            raise WebDashboardError(f"unknown fields for command {command!r}: {unknown!r}")
        args = {}
        if command in ("select_source", "start_session"):
            role = payload.get("role", "")
            if not isinstance(role, str) or not role:
                raise WebDashboardError("role must be a nonempty string")
            if role not in self._config.roles:
                raise WebDashboardError(f"unknown role {role!r}")
            args["role"] = role
        elif command == "follow_latest":
            args["value"] = _parse_bool(payload.get("value"), "value")
        elif command == "set_preview_fps":
            args["value"] = _parse_finite_positive(payload.get("value"), "value")
        elif command == "select_session":
            args["session_id"] = _parse_session_id(payload.get("session_id"))
        full = {"command": command}
        full.update(args)
        return full

    def handle_command(self, payload: dict) -> dict:
        """校验并执行一条控制命令；返回结构化 {ok, error, response}。"""
        try:
            command = self._validate_command_payload(payload)
        except WebDashboardError as error:
            return {"ok": False, "error": str(error), "response": {}}
        if command["command"] == "follow_latest":
            result = self.set_follow_latest(command["value"])
        elif command["command"] == "set_preview_fps":
            result = self.set_preview_fps(command["value"])
        elif command["command"] == "select_session":
            result = self.select_session(command["session_id"])
        else:
            result = self._forward_command(command)
        with self._lock:
            self._last_command_result = dict(result)
        return result

    def _forward_command(self, command: dict) -> dict:
        try:
            returned = self._send_command(dict(command))
        except BaseException as error:
            return {"ok": False, "error": f"{type(error).__name__}: {error}", "response": {}}
        if not isinstance(returned, dict):
            return {
                "ok": False,
                "error": "send_command did not return a dict",
                "response": {},
            }
        if "ok" in returned:
            ok = bool(returned.get("ok"))
        elif "error_code" in returned:
            try:
                ok = int(returned["error_code"]) == 0
            except (TypeError, ValueError):
                ok = False
        else:
            ok = not bool(returned.get("error"))
        error = returned.get("error", "")
        if not isinstance(error, str):
            error = str(error)
        return {"ok": ok, "error": error, "response": returned}

    # ------------------------------------------------------------------
    # 状态 JSON
    # ------------------------------------------------------------------
    def build_state(
        self,
        *,
        include_records: bool = True,
        view_session: str | None = None,
    ) -> dict:
        """Build one dashboard snapshot.

        ``include_records=False`` is the cheap 1 Hz polling path.  A caller
        may request records for a browser-local historical selection without
        mutating the live control target kept by the scheduler.
        """
        if type(include_records) is not bool:
            raise WebDashboardError("include_records must be a bool")
        if view_session is not None:
            view_session = _parse_session_id(view_session)
        # 外部回调（get_status / list_sessions）一律在锁外执行，
        # 锁内只做内存状态快照与选择。
        diagnostics = self._status_diagnostics()
        sessions = self._safe_list_sessions()
        with self._lock:
            self._refresh_selected_session_locked(sessions, diagnostics)
            selected = self._selected_session_id
            preview_fps = self._web_preview_fps
            follow_latest = self._follow_latest
            last_command_result = dict(self._last_command_result)
        if selected and not any(
            item.get("session_id") == selected for item in sessions
        ):
            sessions = [
                {
                    "session_id": selected,
                    "window_count": 0,
                    "latest_window_index": -1,
                }
            ] + sessions
        # Revision identity refreshes historical maps even without new windows.
        from .g4.backend_process import output_root
        sessions = [dict(item) for item in sessions]
        for item in sessions:
            try:
                pointer = json.loads((output_root(self._config.formal_records_root, item["session_id"]) / "latest.json").read_text())
                item["map_revision_sha256"] = str(pointer.get("sha256", ""))
            except (OSError, ValueError, KeyError):
                item["map_revision_sha256"] = ""
        display_session = view_session or selected
        records = (
            self._formal_records(display_session)
            if include_records and display_session
            else []
        )
        compact_status = self._compact_status(diagnostics)
        return {
            "follow_latest": follow_latest,
            "selected_session": selected,
            "view_session": display_session,
            "sessions": sessions,
            "roles": [str(name) for name in self._config.roles],
            "status": compact_status,
            # The polling response stays bounded even when /diagnostics has
            # hundreds of MAVROS entries.  Detailed, on-demand state preserves
            # the raw diagnostics object for backwards compatibility.
            "diagnostics": diagnostics if include_records else compact_status,
            "preview_fps": preview_fps,
            "last_command_result": last_command_result if include_records else {},
            "formal_records": records,
        }

    def readiness(self) -> tuple[bool, dict]:
        """Return HTTP readiness without changing capture/reconstruction state."""
        root = Path(self._config.result_root)
        root_ready = root.is_dir() and not root.is_symlink()
        diagnostics = self._status_diagnostics()
        state = str(diagnostics.get("state", ""))
        ros_ready = state not in ("", "UNAVAILABLE", "INVALID")
        payload = {
            "ok": bool(root_ready and ros_ready),
            "service": "drone3d-continuous-web",
            "result_root_ready": bool(root_ready),
            "session_service_ready": bool(ros_ready),
            "state": state,
        }
        error = diagnostics.get("error") or diagnostics.get("last_error")
        if error:
            payload["error"] = str(error)
        return bool(payload["ok"]), payload

    def _compact_status(self, diagnostics: dict) -> dict:
        raw_effective = diagnostics.get("effective") if isinstance(
            diagnostics.get("effective"), dict
        ) else {}
        effective_keys = (
            "sample_fps",
            "window_frames",
            "overlap_frames",
            "queue_mode",
            "queue_capacity",
            "max_retries",
            "keep_snapshot_on_success",
            "save_sensor_depth",
            "inference_resolution",
            "worker_timeout_sec",
            "reconstruction_mode",
        )
        effective = {
            key: raw_effective[key]
            for key in effective_keys
            if key in raw_effective
            and isinstance(raw_effective[key], (str, int, float, bool, type(None)))
        }
        for key, value in list(effective.items()):
            if isinstance(value, str):
                effective[key] = value[:1024]

        def bounded_text(key: str) -> str:
            value = diagnostics.get(key, "")
            return str(value)[:4096] if value is not None else ""

        def diagnostic_int(key: str):
            value = diagnostics.get(key)
            if type(value) is int:
                return value
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    pass
            return None

        g4 = diagnostics.get("g4")
        if not isinstance(g4, dict) or not g4:
            # ROS DiagnosticArray carries scalar KeyValue entries. The test
            # harness may provide the already structured backend status.
            g4 = {"state": bounded_text("g4_state"),
                  "revision": bounded_text("g4_revision"),
                  "error": bounded_text("g4_error"),
                  "exit_code": diagnostic_int("g4_exit_code")}
        return {
            "state": bounded_text("state"),
            "alignment_backend": bounded_text("alignment_backend"),
            "g4": g4,
            "new_input_frames": diagnostics.get("new_input_frames"),
            "shared_input_frames": diagnostics.get("shared_input_frames"),
            "model_input_frames": diagnostics.get("model_input_frames"),
            "session_id": bounded_text("session_id"),
            "role": bounded_text("role"),
            "window_index": diagnostics.get("window_index", 0),
            "captured_frames": diagnostics.get("captured_frames", 0),
            "capture_target_frames": diagnostics.get("capture_target_frames", 0),
            "pending_windows": diagnostics.get("pending_windows", 0),
            "running_window": diagnostics.get("running_window"),
            "worker_running_active": bool(diagnostics.get("worker_running_active")),
            "succeeded_windows": diagnostics.get("succeeded_windows", 0),
            "failed_windows": diagnostics.get("failed_windows", 0),
            "missing_pose_frames": diagnostic_int("missing_pose_frames"),
            "memory_paused": bounded_text("memory_paused").lower() == "true",
            "oom_latched": bounded_text("oom_latched").lower() == "true",
            "retry_window": diagnostic_int("retry_window"),
            "attempt_failures": diagnostic_int("attempt_failures"),
            "manual_retries": diagnostic_int("manual_retries"),
            "dropped_windows": diagnostics.get("dropped_windows", 0),
            "effective": effective,
            "reconstruction_mode": bounded_text(
                "reconstruction_mode"
            ) or str(effective.get("reconstruction_mode", "")),
            "fifo_bytes": diagnostics.get("fifo_bytes", 0),
            "fifo_oldest_wait_sec": diagnostics.get("fifo_oldest_wait_sec"),
            "capture_queue_blocked": diagnostics.get("capture_queue_blocked"),
            "capture_disk_bytes": diagnostics.get("capture_disk_bytes", 0),
            "storage_total_bytes": diagnostics.get("storage_total_bytes", 0),
            "storage_capture_bytes": diagnostics.get("storage_capture_bytes", 0),
            "storage_worker_bytes": diagnostics.get("storage_worker_bytes", 0),
            "storage_formal_bytes": diagnostics.get("storage_formal_bytes", 0),
            "storage_frame_pool_bytes": diagnostics.get("storage_frame_pool_bytes", 0),
            "storage_journal_bytes": diagnostics.get("storage_journal_bytes", 0),
            "storage_runtime_log_bytes": diagnostics.get("storage_runtime_log_bytes", 0),
            "storage_ros_log_bytes": diagnostics.get("storage_ros_log_bytes", 0),
            "storage_capture_roots": bounded_text("storage_capture_roots"),
            "storage_worker_root": bounded_text("storage_worker_root"),
            "storage_formal_root": bounded_text("storage_formal_root"),
            "storage_frame_pool_root": bounded_text("storage_frame_pool_root"),
            "storage_journal_root": bounded_text("storage_journal_root"),
            "storage_runtime_log_root": bounded_text("storage_runtime_log_root"),
            "storage_ros_log_root": bounded_text("storage_ros_log_root"),
            "storage_scan_error": bounded_text("storage_scan_error"),
            "available_roles": diagnostics.get("available_roles", []),
            "last_error": bounded_text("last_error"),
            "min_free_disk_gb": diagnostics.get("min_free_disk_gb"),
            "disk_free_gb": diagnostics.get("disk_free_gb"),
            "disk_redline": diagnostics.get("disk_redline"),
            "memory_mb": diagnostics.get("memory_mb"),
            "mem_available_bytes": diagnostics.get("mem_available_bytes"),
            "swap_used_bytes": diagnostics.get("swap_used_bytes"),
            "gpu_name": bounded_text("gpu_name"),
            "gpu_util": diagnostics.get("gpu_util"),
            "temp_c": diagnostics.get("temp_c"),
            "worker_wall_time_sec": diagnostics.get("worker_wall_time_sec"),
            "service_wall_time_sec": diagnostics.get("service_wall_time_sec"),
            "service_budget_sec": diagnostics.get("service_budget_sec"),
            "aux_coverage": bounded_text("aux_coverage"),
            "last_alignment_state": diagnostics.get(
                "last_alignment_state", COORDINATE_MODEL_LOCAL_UNALIGNED
            ),
            "formal_windows": diagnostics.get("formal_windows", 0),
        }

    def _formal_records(self, session_id: str) -> list:
        if not _SESSION_ID_RE.fullmatch(session_id):
            return []
        try:
            records = _formal_records_for(
                session_id, self._config.formal_records_root
            )
        except BaseException:
            return []
        summaries = []
        for record in records:
            alignment = _record_alignment(record)
            effective = record.get("effective_config")
            summaries.append(
                {
                    "session_id": record.get("session_id", session_id),
                    "window_index": record.get("window_index", -1),
                    "scene_id": record.get("scene_id", ""),
                    "camera_role": record.get("camera_role", ""),
                    "camera_name": record.get("camera_name", ""),
                    "coordinate_state": _record_coordinate_state(record),
                    "map_id": alignment.get("map_id", ""),
                    "alignment_method": alignment.get("method", ""),
                    "map_revision": alignment.get("map_revision", 0),
                    "map_revision_sha256": alignment.get("map_revision_sha256", ""),
                    "map_stale": alignment.get("map_stale", False),
                    "geometry_accepted": alignment.get("geometry_accepted", False),
                    "root_window_index": alignment.get("root_window_index"),
                    "reference_window_index": alignment.get("reference_window_index"),
                    "new_frame_count": alignment.get("new_frame_count"),
                    "model_frame_count": alignment.get("model_frame_count"),
                    "reason": alignment.get("reason", ""),
                    "unreliable": bool(alignment.get("unreliable", True)),
                    "scale": alignment.get("scale"),
                    "residual_rms": alignment.get("residual_rms"),
                    "normalized_residual_rms": alignment.get("normalized_residual_rms"),
                    "inlier_count": alignment.get("inlier_count"),
                    "inlier_ratio": alignment.get("inlier_ratio"),
                    "quality_passed": alignment.get(
                        "quality_passed",
                        not bool(alignment.get("unreliable", True)),
                    ),
                    "component_anchor": bool(
                        alignment.get("component_anchor", False)
                    ),
                    "overlap_frame_count": record.get("overlap_frame_count", 0),
                    "pending": bool(alignment.get("pending", False)),
                    "config_digest": record.get("config_digest", ""),
                    "reconstruction_mode": record.get("reconstruction_mode", ""),
                    "effective_config": effective if isinstance(effective, dict) else {},
                    "worker": record.get("worker") if isinstance(record.get("worker"), dict) else {},
                    "result_manifest_sha256": record.get("result_manifest_sha256", ""),
                    "depth_metrics": record.get("depth_metrics")
                    if isinstance(record.get("depth_metrics"), dict)
                    else None,
                    "depth_metrics_error": record.get("depth_metrics_error", ""),
                }
            )
        return summaries

    # ------------------------------------------------------------------
    # 帧缓存
    # ------------------------------------------------------------------
    def _cached_frame(self, key: tuple, loader: Callable[[], bytes]) -> bytes:
        with self._lock:
            cached = self._frame_cache.get(key)
            if cached is not None:
                return cached
        data = loader()
        with self._lock:
            if key not in self._frame_cache:
                self._frame_cache[key] = data
                self._frame_cache_order.append(key)
                while len(self._frame_cache_order) > _FRAME_CACHE_MAX_ENTRIES:
                    oldest = self._frame_cache_order.pop(0)
                    self._frame_cache.pop(oldest, None)
        return data

    def window_frame_png(self, session_id: str, window_index: int, frame_index: int, kind: str) -> bytes:
        session_id = _parse_session_id(session_id)
        window_index = _parse_nonnegative_int(window_index, "window")
        frame_index = _parse_nonnegative_int(frame_index, "frame")
        if kind not in ("rgb", "depth"):
            raise WebDashboardError("kind must be rgb or depth", 400)
        records = _formal_records_for(
            session_id, self._config.formal_records_root
        )
        record = _find_window_record(records, window_index)
        if record is None:
            raise WebDashboardError("window not found", 404)
        key = (
            session_id,
            window_index,
            str(record.get("result_manifest_sha256", "")),
            frame_index,
            kind,
        )
        return self._cached_frame(
            key,
            lambda: _png_bytes(
                _load_frame_image(
                    record,
                    frame_index,
                    kind,
                    self._config.max_predictions_bytes,
                )
            ),
        )

    def latest_first_depth_png(self, session_id: str):
        session_id = _parse_session_id(session_id)
        records = _formal_records_for(session_id, self._config.formal_records_root)
        if not records:
            raise WebDashboardError("unknown session", 404)
        latest = _latest_window_record(records)
        if latest is None:
            raise WebDashboardError("no formal window for this session", 404)
        window_index = int(latest.get("window_index", 0))
        key = (
            session_id,
            window_index,
            str(latest.get("result_manifest_sha256", "")),
            0,
            "depth",
        )
        return self._cached_frame(
            key,
            lambda: _png_bytes(
                _load_frame_image(
                    latest,
                    0,
                    "depth",
                    self._config.max_predictions_bytes,
                )
            ),
        ), session_id, window_index

    @property
    def web_preview_fps(self) -> float:
        with self._lock:
            return self._web_preview_fps

    @property
    def follow_latest(self) -> bool:
        with self._lock:
            return self._follow_latest

    @property
    def selected_session_id(self) -> str:
        with self._lock:
            return self._selected_session_id

    @property
    def last_command_result(self) -> dict:
        with self._lock:
            return dict(self._last_command_result)

