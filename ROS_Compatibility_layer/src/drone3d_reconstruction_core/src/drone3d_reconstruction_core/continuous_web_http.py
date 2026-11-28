"""The dashboard's HTTP server: routes, MJPEG preview streams, start and stop.

Split from ``continuous_web``; :class:`WebDashboardCore` inherits :class:`HttpServerMixin`.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit

from ._common import reject_duplicate_json_fields as _reject_duplicate_json_fields
from .continuous_web_common import (
    _MAX_JSON_BODY_BYTES,
    WebDashboardError,
    _dashboard_html,
    _formal_records_for,
    _json_bytes,
    _parse_nonnegative_int,
    _parse_positive_int,
    _parse_session_id,
    _valid_stream_item,
)
from .continuous_web_pointcloud import _latest_map_id


class HttpServerMixin:
    """HTTP server methods of WebDashboardCore."""

    # ------------------------------------------------------------------
    # 服务器
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._server is not None:
            return
        core = self
        config = self._config

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                return

            def send_json(self, payload, status=200, extra_headers=None):
                body = _json_bytes(payload)
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    if extra_headers:
                        for key, value in extra_headers.items():
                            self.send_header(key, str(value))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                    pass

            def send_error_json(self, status, message):
                self.send_json({"ok": False, "error": message}, status=status)

            def do_GET(self):
                try:
                    self._route_get()
                except WebDashboardError as error:
                    status = 400
                    if len(error.args) > 1 and isinstance(error.args[1], int):
                        status = error.args[1]
                    self.send_error_json(status, str(error))
                except BrokenPipeError:
                    pass
                except ConnectionAbortedError:
                    pass
                except BaseException as error:
                    self.send_error_json(500, f"{type(error).__name__}: {error}")

            def do_POST(self):
                try:
                    self._route_post()
                except WebDashboardError as error:
                    status = 400
                    if len(error.args) > 1 and isinstance(error.args[1], int):
                        status = error.args[1]
                    self.send_error_json(status, str(error))
                except BrokenPipeError:
                    pass
                except ConnectionAbortedError:
                    pass
                except BaseException as error:
                    self.send_error_json(500, f"{type(error).__name__}: {error}")

            def _route_get(self):
                split = urlsplit(self.path)
                path = unquote(split.path)
                query = parse_qs(split.query, keep_blank_values=True)
                if path in ("/", "/index.html"):
                    body = _dashboard_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/healthz":
                    self.send_json(
                        {
                            "ok": True,
                            "service": "drone3d-continuous-web",
                            "time": core._now(),
                        }
                    )
                    return
                if path == "/readyz":
                    ready, payload = core.readiness()
                    self.send_json(payload, status=200 if ready else 503)
                    return
                if path == "/api/state":
                    include_raw = query.get("include_records", ["1"])
                    if len(include_raw) != 1 or include_raw[0] not in (
                        "0", "1", "false", "true"
                    ):
                        raise WebDashboardError(
                            "include_records must be 0/1/false/true"
                        )
                    requested = query.get("session")
                    if requested is not None and len(requested) != 1:
                        raise WebDashboardError("session must occur once")
                    view_session = requested[0] if requested else None
                    self.send_json(
                        core.build_state(
                            include_records=include_raw[0] in ("1", "true"),
                            view_session=view_session,
                        )
                    )
                    return
                if path == "/api/pointcloud":
                    self._handle_pointcloud(query, meta_only=False)
                    return
                if path == "/api/pointcloud_meta":
                    self._handle_pointcloud(query, meta_only=True)
                    return
                if path == "/api/latest_first_depth":
                    session_id = _parse_session_id(query.get("session", [""])[0])
                    png, actual_session, actual_window = core.latest_first_depth_png(session_id)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(png)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Session", actual_session)
                    self.send_header("X-Window", str(actual_window))
                    self.send_header("X-Frame", "0")
                    self.end_headers()
                    self.wfile.write(png)
                    return
                if path.startswith("/api/window/"):
                    self._handle_window_frame(path, query)
                    return
                if path.startswith("/stream/"):
                    self._handle_stream(path)
                    return
                self.send_error_json(404, "not found")

            def _route_post(self):
                split = urlsplit(self.path)
                path = unquote(split.path)
                if path != "/api/command":
                    self.send_error_json(404, "not found")
                    return
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length <= 0 or length > _MAX_JSON_BODY_BYTES:
                    self.send_error_json(400, "JSON body must be between 1 and 64 KiB")
                    return
                body = self.rfile.read(length)
                try:
                    payload = json.loads(
                        body.decode("utf-8", "strict"),
                        object_pairs_hook=_reject_duplicate_json_fields,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    self.send_error_json(400, f"invalid JSON: {error}")
                    return
                if not isinstance(payload, dict):
                    self.send_error_json(400, "JSON body must be an object")
                    return
                result = core.handle_command(payload)
                status = 200 if result.get("ok") else 400
                self.send_json(result, status=status)

            def _handle_window_frame(self, path, query):
                parts = path.strip("/").split("/")
                if len(parts) != 6 or parts[0] != "api" or parts[1] != "window" or parts[4] != "frames":
                    self.send_error_json(400, "invalid window frame path")
                    return
                session_id = _parse_session_id(parts[2])
                window_index = _parse_nonnegative_int(parts[3], "window")
                frame_index = _parse_nonnegative_int(parts[5], "frame")
                kind = query.get("kind", ["rgb"])[0]
                png = core.window_frame_png(session_id, window_index, frame_index, kind)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Session", session_id)
                self.send_header("X-Window", str(window_index))
                self.send_header("X-Frame", str(frame_index))
                self.send_header("X-Kind", kind)
                self.end_headers()
                self.wfile.write(png)

            def _handle_pointcloud(self, query, meta_only):
                session_id = _parse_session_id(query.get("session", [""])[0])
                include_values = query.get("include_rejected", ["0"])
                if len(include_values) != 1 or include_values[0] not in (
                    "0",
                    "1",
                    "false",
                    "true",
                ):
                    raise WebDashboardError(
                        "include_rejected must be 0/1/false/true"
                    )
                include_rejected = include_values[0] in ("1", "true")
                map_id = query.get("map", ["latest"])[0]
                if map_id == "latest":
                    records = _formal_records_for(
                        session_id, config.formal_records_root
                    )
                    if not records:
                        raise WebDashboardError("unknown session", 404)
                    map_id = (
                        _latest_map_id(records, include_rejected) or "latest"
                    )
                max_points = config.max_point_cloud_points
                raw_max = query.get("max_points", [None])[0]
                if raw_max is not None:
                    max_points = _parse_positive_int(raw_max, "max_points")
                    if max_points > config.max_point_cloud_points:
                        max_points = config.max_point_cloud_points
                result = core.build_point_cloud(
                    session_id, map_id, max_points, include_rejected
                )
                meta = core.point_cloud_meta(result)
                if meta_only:
                    self.send_json(meta)
                    return
                body = core.point_cloud_binary(result)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Point-Cloud-Cache", "HIT" if result.get("_cached") else "MISS")
                self.send_header("X-Point-Count", str(meta["displayed_points"]))
                self.send_header("X-Original-Points", str(meta["original_points"]))
                self.send_header("X-Displayed-Points", str(meta["displayed_points"]))
                self.send_header("X-Sampling", meta["sampling"])
                self.send_header(
                    "X-Equivalent-Stride", str(meta["equivalent_stride"])
                )
                self.send_header("X-Conf-Percentile", str(meta["conf_percentile"]))
                self.send_header("X-Reason", quote(str(meta["reason"]), safe=""))
                self.send_header("X-Aligned", "1" if meta["aligned"] else "0")
                self.send_header(
                    "X-Include-Rejected",
                    "1" if meta["include_rejected"] else "0",
                )
                self.send_header("X-Coordinate-State", meta["coordinate_state"])
                self.send_header("X-Session", session_id)
                self.send_header("X-Map-Id", quote(str(meta["map_id"]), safe=""))
                self.send_header("X-Windows", ",".join(str(item) for item in meta["windows"]))
                self.end_headers()
                self.wfile.write(body)

            def _handle_stream(self, path):
                parts = path.strip("/").split("/")
                if len(parts) != 3 or parts[0] != "stream":
                    self.send_error_json(404, "not found")
                    return
                role = parts[1]
                kind = parts[2]
                if role not in config.roles:
                    self.send_error_json(404, f"unknown role {role!r}")
                    return
                if kind not in ("rgb", "depth"):
                    self.send_error_json(404, "stream kind must be rgb or depth")
                    return
                if kind == "depth" and config.roles[role].depth_topic is None:
                    self.send_error_json(404, "sensor depth unavailable")
                    return
                producer = core._read_depth if kind == "depth" else core._read_image
                if producer is None:
                    self.send_error_json(404, "sensor depth unavailable" if kind == "depth" else "image unavailable")
                    return
                core._stream_mjpeg(self, role, kind, producer)

        self._server = ThreadingHTTPServer((config.host, config.port), Handler)
        self._server.daemon_threads = True
        ready = threading.Event()

        def _serve() -> None:
            ready.set()
            self._server.serve_forever(poll_interval=0.1)

        self._server_thread = self._thread_factory(target=_serve)
        self._server_thread.daemon = True
        self._server_thread.start()
        ready.wait(timeout=2.0)

    def _preview_interval(self) -> float:
        with self._lock:
            fps = self._web_preview_fps
        return 1.0 / float(fps) if fps > 0 else 0.2

    def _stream_mjpeg(self, handler, role: str, kind: str, producer) -> None:
        import cv2

        interval = self._preview_interval()
        first_deadline = self._now() + max(2.0 * interval, 1.0)
        first_item = None
        while self._now() < first_deadline and not self._stop_event.is_set():
            item = producer(role)
            if _valid_stream_item(item):
                first_item = item
                break
            time.sleep(0.05)
        if first_item is None:
            message = "sensor depth unavailable" if kind == "depth" else "image unavailable"
            handler.send_json({"ok": False, "error": message}, status=404)
            return
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        handler.end_headers()
        boundary = b"--frame\r\n"
        item = first_item
        while item is not None and not self._stop_event.is_set():
            if not _valid_stream_item(item):
                break
            # 热调 web_preview_fps 必须对已存在的 MJPEG 连接生效：
            # 每轮都读当前值，而不是连接建立时读一次。
            interval = self._preview_interval()
            stamp, frame = item
            try:
                ok, encoded = cv2.imencode(".jpg", frame)
                if ok:
                    payload = encoded.tobytes()
                    handler.wfile.write(boundary)
                    handler.wfile.write(
                        b"Content-Type: image/jpeg\r\n"
                        + ("X-Stamp: %s\r\n" % str(stamp)).encode("utf-8")
                        + ("Content-Length: %d\r\n\r\n" % len(payload)).encode("utf-8")
                    )
                    handler.wfile.write(payload)
                    handler.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return
            except BaseException:
                pass
            deadline = self._now() + interval
            while self._now() < deadline and not self._stop_event.is_set():
                time.sleep(0.02)
            if self._stop_event.is_set():
                return
            item = producer(role)
        try:
            handler.wfile.write(b"--frame--\r\n")
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass

    def stop(self, timeout_sec: float = 2.0) -> None:
        self._stop_event.set()
        server = self._server
        thread = self._server_thread
        self._server = None
        self._server_thread = None
        if server is not None:
            try:
                server.shutdown()
            except BaseException:
                pass
            try:
                server.server_close()
            except BaseException:
                pass
        if thread is not None and thread.is_alive():
            thread.join(timeout_sec)

    @property
    def server_port(self) -> int:
        server = self._server
        if server is None:
            return 0
        return int(server.server_address[1])
