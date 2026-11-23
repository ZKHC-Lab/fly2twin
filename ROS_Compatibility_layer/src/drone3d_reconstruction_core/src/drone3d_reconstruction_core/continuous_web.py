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
import os
from pathlib import Path
import re
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Optional
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .formal_result import (
    scan_session_windows,
)
from .odom_local import (
    COORDINATE_MODEL_LOCAL_UNALIGNED,
    COORDINATE_ODOM_LOCAL_APPROX,
)
from .odom_local import valid_sim3_alignment
from .g4.map_snapshot import apply_map_snapshot

_SESSION_ID_RE = re.compile(r"s[0-9a-f]{16}\Z")
_WINDOW_DIR_RE = re.compile(r"window-([0-9]{4,})\Z")
_SAFE_ROLE_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_SAFE_FRAME_ID_RE = re.compile(r"[A-Za-z0-9_/-]{1,128}\Z")
_MAX_JSON_BODY_BYTES = 64 * 1024
_MAX_RESULT_MANIFEST_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_PREDICTIONS_BYTES = 2 * 1024 * 1024 * 1024
_FRAME_CACHE_MAX_ENTRIES = 32

DEFAULT_MAX_POINT_CLOUD_POINTS = 1_000_000
DEFAULT_MAX_SESSION_WINDOWS = 64
DEFAULT_CONF_PERCENTILE = 20.0
MAX_WEB_PREVIEW_FPS = 30.0

_COORDINATE_STATES = frozenset(
    (
        COORDINATE_MODEL_LOCAL_UNALIGNED,
        COORDINATE_ODOM_LOCAL_APPROX,
    )
)
_ALIGNED_STATES = frozenset(
    (COORDINATE_ODOM_LOCAL_APPROX,)
)
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


class WebDashboardError(ValueError):
    """Web 配置或请求非法；第二个位置参数是 HTTP 状态码（默认 400）。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message, status)
        self.message = message
        self.status = status

    def __str__(self) -> str:
        return str(self.message)


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


def _reject_duplicate_json_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def _json_bytes(payload: Any, status: int = 200) -> bytes:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return body


def _finite(value: Any) -> bool:
    try:
        converted = float(value)
        return math.isfinite(converted)
    except (TypeError, ValueError, OverflowError):
        return False


def _parse_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise WebDashboardError(f"{label} must be a nonnegative integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise WebDashboardError(f"{label} must be a nonnegative integer") from error
    if converted < 0 or str(value).strip() != str(converted):
        # int("1.2") -> 1, so reject non-integer strings explicitly.
        raise WebDashboardError(f"{label} must be a nonnegative integer")
    return converted


def _parse_positive_int(value: Any, label: str) -> int:
    parsed = _parse_nonnegative_int(value, label)
    if parsed <= 0:
        raise WebDashboardError(f"{label} must be a positive integer")
    return parsed


def _parse_finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WebDashboardError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise WebDashboardError(f"{label} must be positive and finite")
    return converted


def _parse_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise WebDashboardError(f"{label} must be a bool")
    return value


def _trusted_no_symlink_file(path: Path, maximum_bytes: int) -> None:
    """严格文件检查：绝对路径、路径各组件均非符号链接、最终为常规文件。"""
    if not path.is_absolute():
        raise WebDashboardError("path is not absolute")
    current = Path(path.anchor)
    parts = path.parts
    if not parts:
        raise WebDashboardError("path is empty")
    for part in parts[1:]:
        current = current / part
        try:
            info = os.lstat(str(current))
        except OSError as error:
            raise WebDashboardError("path component cannot be inspected") from error
        if stat.S_ISLNK(info.st_mode):
            raise WebDashboardError("symbolic links are not allowed")
    if not stat.S_ISREG(info.st_mode):
        raise WebDashboardError("not a regular file")
    if info.st_size <= 0 or info.st_size > maximum_bytes:
        raise WebDashboardError("file size is outside the trusted bound")


def _manifest_path_from_uri(uri: Any) -> Path:
    if not isinstance(uri, str) or not uri.startswith("file://"):
        raise WebDashboardError("result manifest uri is not a local file uri")
    try:
        split = urlsplit(uri)
    except ValueError as error:
        raise WebDashboardError("result manifest uri is not parseable") from error
    if split.scheme != "file" or split.netloc or split.query or split.fragment:
        raise WebDashboardError("result manifest uri is not a plain file uri")
    path = Path(unquote(split.path))
    if not path.is_absolute():
        raise WebDashboardError("result manifest path is not absolute")
    _trusted_no_symlink_file(path, _MAX_RESULT_MANIFEST_BYTES)
    if path.name != "result.json":
        raise WebDashboardError("result manifest path must end with result.json")
    return path


def _predictions_path_from_manifest(manifest_path: Path, max_bytes: int) -> Path:
    predictions = manifest_path.parent / "predictions.npz"
    _trusted_no_symlink_file(predictions, max_bytes)
    if predictions.parent != manifest_path.parent:
        raise WebDashboardError("predictions.npz must sit next to result.json")
    return predictions


def _valid_stream_item(item: Any) -> bool:
    """MJPEG 生产者帧校验：必须是 (stamp, ndarray) 二元组。"""
    import numpy as np

    if item is None:
        return False
    if not isinstance(item, (tuple, list)) or len(item) != 2:
        return False
    return isinstance(item[1], np.ndarray) and item[1].size > 0


def _turbo_lut() -> Any:
    """本地 turbo-like 色带（多项式近似；不依赖 matplotlib）。"""
    import numpy as np

    red = (0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943)
    green = (0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604)
    blue = (0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973)

    def channel(coeffs, x):
        value = coeffs[0]
        power = x
        for coeff in coeffs[1:]:
            value += coeff * power
            power = power * x
        return np.clip(value, 0.0, 1.0)

    x = np.linspace(0.0, 1.0, 256)
    return np.stack(
        (
            channel(red, x),
            channel(green, x),
            channel(blue, x),
        ),
        axis=-1,
    )


def colorize_depth(depth: Any) -> Any:
    """把单帧深度图（2D 或 HxWx1）着色为 RGB uint8（本地 turbo-like 色带）。"""
    import numpy as np

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise WebDashboardError("depth array must be 2D or HxWx1")
    valid = np.isfinite(depth) & (depth > 0)
    colored = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not valid.any():
        return colored
    finite = depth[valid]
    low, high = np.percentile(finite, [2.0, 98.0])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        high = float(low) + 1.0
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    lut = _turbo_lut()
    indices = (normalized[valid] * 255.0).astype(np.int32)
    indices = np.clip(indices, 0, 255)
    colored[valid] = (lut[indices] * 255.0).astype(np.uint8)
    return colored


def _image_to_rgb_u8(image: Any) -> Any:
    """把 predictions.npz 的 images[n] 规范化为 HxWx3 uint8 RGB。

    兼容实际 Omega 输出的 CHW 布局 (3,H,W) 与契约文档的 HWC 布局 (H,W,3)。
    """
    import numpy as np

    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[0] == 3 and arr.shape[2] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[2] != 3:
            raise WebDashboardError("images[n] is neither CHW nor HWC")
    elif arr.ndim != 2:
        raise WebDashboardError("images[n] must be a 2D/3D array")
    if arr.shape[-1] != 3:
        raise WebDashboardError("images[n] must have 3 color channels")
    if arr.dtype != np.uint8:
        max_value = float(np.max(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
    return arr.astype(np.uint8)


def _load_frame_image(
    record: dict,
    frame_index: int,
    kind: str,
    max_predictions_bytes: int,
):
    """加载一个正式窗口帧；返回 rgb_u8_hwc。"""
    import numpy as np

    manifest_path = _manifest_path_from_uri(record.get("result_manifest_uri"))
    predictions_path = _predictions_path_from_manifest(
        manifest_path, max_predictions_bytes
    )
    with np.load(str(predictions_path), allow_pickle=False) as data:
        if "images" not in data or "depth" not in data:
            raise WebDashboardError("predictions.npz is missing images/depth")
        images = data["images"]
        frame_count = images.shape[0] if images.ndim >= 3 else 0
        if frame_index >= frame_count:
            raise WebDashboardError("frame index out of range", 404)
        if kind == "rgb":
            image = images[frame_index]
            return _image_to_rgb_u8(image)
        depth = np.asarray(data["depth"][frame_index])
        return colorize_depth(depth)


def _find_window_record(records, window_index: int):
    for record in records:
        if record.get("window_index") == window_index:
            return record
    return None


def _latest_window_record(records):
    if not records:
        return None

    return max(records, key=lambda record: int(record.get("window_index", 0)))


def _record_alignment(record) -> dict:
    alignment = record.get("alignment")
    if not isinstance(alignment, dict):
        return {}
    return alignment


def _record_coordinate_state(record) -> str:
    state = _record_alignment(record).get("coordinate_state", COORDINATE_MODEL_LOCAL_UNALIGNED)
    if state not in _COORDINATE_STATES:
        return COORDINATE_MODEL_LOCAL_UNALIGNED
    return state


def _record_quality_passed(record) -> bool:
    alignment = _record_alignment(record)
    if alignment.get("quality_passed") is False:
        return False
    return True


def _aligned_transform(record):
    """从 provenance alignment 提取 (scale, R, t)；不可用返回 None。"""
    alignment = _record_alignment(record)
    if not valid_sim3_alignment(alignment):
        return None
    import numpy as np

    try:
        scale = float(alignment["scale"])
        rotation_matrix = np.asarray(alignment["rotation"], dtype=np.float64).reshape(3, 3)
        translation_vector = np.asarray(alignment["translation"], dtype=np.float64).reshape(3)
    except (TypeError, ValueError, OverflowError) as error:
        raise WebDashboardError("alignment transform is not finite") from error
    if not np.isfinite(rotation_matrix).all() or not np.isfinite(translation_vector).all():
        return None
    return scale, rotation_matrix, translation_vector


def _frame_placement(alignment, merge_indices, frame_count):
    """Validated per-frame localization placement of one G4 ``frame_pose`` window.

    Returns (depth_scales[F], K, centers[F,3], rotations[F,3,3], has_pose[F]) for the
    displayed (merged) frames, in the same order as the sliced prediction arrays. A
    frame's scale is its per-frame G4 scale when the revision has one, else the window's.
    """
    import numpy as np

    poses = alignment.get("frame_poses")
    if not isinstance(poses, list) or len(poses) != frame_count:
        raise WebDashboardError("frame_poses must list every model frame")
    try:
        scale = float(alignment["depth_scale"])
        K = np.asarray(alignment["intrinsics"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError) as error:
        raise WebDashboardError("frame_pose depth_scale/intrinsics are invalid") from error
    if not math.isfinite(scale) or scale <= 0 or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise WebDashboardError("frame_pose depth_scale/intrinsics are invalid")
    order = merge_indices if merge_indices is not None else list(range(frame_count))
    per_frame = alignment.get("frame_depth_scales")
    if per_frame is not None and (not isinstance(per_frame, list) or len(per_frame) != frame_count):
        raise WebDashboardError("frame_depth_scales must list every model frame")
    scales = np.full(len(order), scale)
    if per_frame is not None:
        for k, index in enumerate(order):
            if per_frame[index] is not None:
                value = float(per_frame[index])
                if not math.isfinite(value) or value <= 0:
                    raise WebDashboardError("frame_depth_scales entry is invalid")
                scales[k] = value
    centers = np.full((len(order), 3), np.nan)
    rotations = np.full((len(order), 3, 3), np.nan)
    has_pose = np.zeros(len(order), dtype=bool)
    for k, index in enumerate(order):
        pose = poses[index]
        if pose is None:
            continue
        try:
            rotation = np.asarray(pose["rotation"], dtype=np.float64).reshape(3, 3)
            center = np.asarray(pose["center"], dtype=np.float64).reshape(3)
        except (KeyError, TypeError, ValueError) as error:
            raise WebDashboardError("frame_poses entry is invalid") from error
        if not np.isfinite(rotation).all() or not np.isfinite(center).all():
            raise WebDashboardError("frame_poses entry is not finite")
        centers[k], rotations[k], has_pose[k] = center, rotation, True
    return scales, K, centers, rotations, has_pose


def _frame_pose_points(placement, depth, flat_indices):
    """Back-project sampled pixels with calibrated K and each frame's own pose."""
    import numpy as np

    scales, K, centers, rotations, has_pose = placement
    frames, pixels = np.divmod(flat_indices, depth.shape[1] * depth.shape[2])
    v, u = np.divmod(pixels, depth.shape[2])
    d = depth.reshape(-1)[flat_indices].astype(np.float64) * scales[frames]
    rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(d)), axis=1) * d[:, None]
    points = np.einsum("nij,nj->ni", np.nan_to_num(rotations[frames]), rays) + np.nan_to_num(centers[frames])
    points[~has_pose[frames]] = np.nan
    return points.astype(np.float32)


def _latest_map_id(records, include_rejected: bool = False) -> Optional[str]:
    candidates = []
    for record in records:
        if _record_coordinate_state(record) in _ALIGNED_STATES:
            if not include_rejected and not _record_quality_passed(record):
                continue
            map_id = _record_alignment(record).get("map_id")
            if isinstance(map_id, str) and map_id:
                candidates.append(map_id)
    if not candidates:
        return None
    return candidates[-1]


def _session_has_aligned_windows(records, include_rejected: bool = False) -> bool:
    return any(
        _record_coordinate_state(record) in _ALIGNED_STATES
        and (include_rejected or _record_quality_passed(record))
        and _aligned_transform(record) is not None
        for record in records
    )


def _parse_session_id(value: Any) -> str:
    if not isinstance(value, str) or not _SESSION_ID_RE.fullmatch(value):
        raise WebDashboardError("session must match s<32 hex>")
    return value


def _png_bytes(rgb_u8: Any) -> bytes:
    import io

    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(rgb_u8, mode="RGB").save(stream, format="PNG")
    return stream.getvalue()


class WebDashboardCore:
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
    # 点云累计
    # ------------------------------------------------------------------
    def _select_point_cloud_records(
        self, session_id, map_id, max_points, include_rejected=False
    ):
        """Choose displayable records and compute the cache fingerprint."""
        records = _formal_records_for(session_id, self._config.formal_records_root)
        if not records:
            raise WebDashboardError("unknown session", 404)
        if not isinstance(map_id, str) or not map_id:
            raise WebDashboardError("map must be a nonempty string", 400)
        aligned_records = []
        for record in records:
            if _record_coordinate_state(record) not in _ALIGNED_STATES:
                continue
            if _record_alignment(record).get("map_id") != map_id:
                continue
            if not include_rejected and not _record_quality_passed(record):
                continue
            if _aligned_transform(record) is None:
                continue
            aligned_records.append(record)

        if aligned_records:
            selected_records = aligned_records
            aligned = True
            reason = _record_coordinate_state(aligned_records[-1])
        else:
            if _session_has_aligned_windows(records, include_rejected):
                raise WebDashboardError("unknown map for this session", 404)
            if any(_record_alignment(r).get("method") == "global_g4" for r in records):
                raise WebDashboardError("G4 processing: no global map revision yet", 409)
            latest = _latest_window_record(records)
            if latest is None or _record_coordinate_state(latest) != COORDINATE_MODEL_LOCAL_UNALIGNED:
                raise WebDashboardError("no displayable window for this session", 404)
            selected_records = [latest]
            aligned = False
            reason = _record_alignment(latest).get("reason", "") or "MODEL_LOCAL_UNALIGNED"

        fingerprint = tuple(
            (int(record.get("window_index", -1)), str(record.get("result_manifest_sha256", "")),
             str(_record_alignment(record).get("map_revision_sha256", "")))
            for record in selected_records
        )
        key = (
            session_id,
            map_id,
            int(max_points),
            bool(include_rejected),
            fingerprint,
        )
        return key, selected_records, aligned, reason

    def _point_cloud_cache_lookup(
        self, session_id, map_id, max_points, include_rejected=False
    ):
        """Fast cache path: no build lock, only the short cache lock."""
        key, selected, aligned, _reason = self._select_point_cloud_records(
            session_id, map_id, max_points, include_rejected
        )
        with self._pc_lock:
            cached = self._pc_cache.get(key)
            if cached is not None:
                if key in self._pc_order:
                    self._pc_order.remove(key)
                self._pc_order.append(key)
                result = dict(cached)
                result["_cached"] = True
                return result
        return None

    def build_point_cloud(
        self,
        session_id: str,
        map_id: str,
        max_points: int,
        include_rejected: bool = False,
    ) -> dict:
        session_id = _parse_session_id(session_id)
        cached = self._point_cloud_cache_lookup(
            session_id, map_id, max_points, include_rejected
        )
        if cached is not None:
            return cached
        with self._pc_build_lock:
            # double-check under the build lock; another request may have
            # populated the cache between the fast path and lock acquisition
            cached = self._point_cloud_cache_lookup(
                session_id, map_id, max_points, include_rejected
            )
            if cached is not None:
                return cached
            return self._build_point_cloud(
                session_id, map_id, max_points, include_rejected
            )

    def _build_point_cloud(
        self, session_id, map_id, max_points, include_rejected=False
    ) -> dict:
        import numpy as np

        key, selected_records, aligned, reason = self._select_point_cloud_records(
            session_id, map_id, max_points, include_rejected
        )
        fingerprint = key[4]
        incremental = None
        with self._pc_lock:
            cached = self._pc_cache.get(key)
            if cached is not None:
                if key in self._pc_order:
                    self._pc_order.remove(key)
                self._pc_order.append(key)
                result = dict(cached)
                result["_cached"] = True
                return result
            if aligned:
                for candidate_key in reversed(self._pc_order):
                    candidate = self._pc_cache.get(candidate_key)
                    candidate_fingerprint = (
                        candidate.get("_selected_fingerprint", ())
                        if candidate is not None
                        else ()
                    )
                    if (
                        candidate is not None
                        and candidate_key[:4] == key[:4]
                        and candidate.get("aligned") is True
                        and candidate_fingerprint
                        and len(candidate_fingerprint) < len(fingerprint)
                        and fingerprint[: len(candidate_fingerprint)]
                        == candidate_fingerprint
                    ):
                        incremental = candidate
                        break

        fusion_opt_in = False
        if self._config.frame_pose_fusion:
            from .g4.backend_process import output_root
            try:
                stored = json.loads((output_root(self._config.formal_records_root, session_id) / "backend_profile.json").read_text())
                fusion_opt_in = stored.get("frame_pose_fusion") is True
            except FileNotFoundError:
                pass  # Old sessions keep their original display algorithm.
        if (fusion_opt_in and aligned and selected_records
                and all(_record_alignment(r).get("placement") == "frame_pose" for r in selected_records)):
            from drone3d_onboard.session import SessionFusion
            from .g4.backend_process import output_root
            fuser = self._session_fusion.get(session_id)
            if fuser is None:
                fuser = SessionFusion(output_root(self._config.formal_records_root, session_id)
                                      / "fusion_cache" / "incremental")
                self._session_fusion = {session_id: fuser}  # keep only the session being viewed
            xyz, rgb, fusion = fuser.fuse(selected_records, max_points)
            result = dict(session_id=session_id, map_id=map_id, aligned=True,
                coordinate_state=_record_coordinate_state(selected_records[-1]), reason=reason,
                original_points=fusion['points_2cm'], displayed_points=len(xyz), equivalent_stride=1,
                sampling="multiview_adaptive_voxel", conf_percentile=50.,
                include_rejected=bool(include_rejected), windows=[r['window_index'] for r in selected_records],
                xyz=xyz.astype(np.float32), rgb=np.rint(rgb*255).clip(0,255).astype(np.uint8),
                fusion=fusion, _selected_fingerprint=fingerprint, _cached=False)
            with self._pc_lock:
                self._pc_cache[key] = result
                self._pc_order.append(key)
                while len(self._pc_order) > 2:
                    self._pc_cache.pop(self._pc_order.pop(0), None)
            return result

        if incremental is None:
            selected_to_load = selected_records
            total_valid = 0
            windows = []
            xyz = np.empty((0, 3), dtype=np.float32)
            rgb = np.empty((0, 3), dtype=np.uint8)
            priorities = np.empty(0, dtype=np.float64)
            rng = np.random.default_rng(0)
        else:
            selected_to_load = selected_records[
                len(incremental["_selected_fingerprint"]) :
            ]
            total_valid = int(incremental["original_points"])
            windows = list(incremental["windows"])
            xyz = incremental["xyz"].copy()
            rgb = incremental["rgb"].copy()
            priorities = incremental["_priorities"].copy()
            rng = np.random.default_rng()
            rng.bit_generator.state = incremental["_rng_state"]

        conf_percentile = float(self._config.conf_percentile)
        # ponytail: compressed NPZ is the atomic input, so peak RAM is one
        # window plus the bounded reservoir; chunk the artifact only if one
        # window itself stops fitting.
        for record in selected_to_load:
            manifest_path = _manifest_path_from_uri(record.get("result_manifest_uri"))
            predictions_path = _predictions_path_from_manifest(
                manifest_path, self._config.max_predictions_bytes
            )
            with np.load(str(predictions_path), allow_pickle=False) as data:
                if "world_points_from_depth" not in data:
                    raise WebDashboardError("predictions.npz is missing world_points_from_depth")
                if "depth_conf" not in data:
                    raise WebDashboardError("predictions.npz is missing depth_conf")
                if "images" not in data:
                    raise WebDashboardError("predictions.npz is missing images")
                world = np.asarray(data["world_points_from_depth"])
                conf = np.asarray(data["depth_conf"])
                images = np.asarray(data["images"])
                is_g4 = _record_alignment(record).get("method") == "global_g4"
                depth = np.asarray(data["depth"])[..., 0] if is_g4 else None
                g4_mask = None
                if is_g4:
                    from scipy.ndimage import maximum_filter, minimum_filter
                    # G4 computes its percentile on the complete model window,
                    # including the shared prefix, before excluding duplicates.
                    d = depth / float(_record_alignment(record)["gauge"])
                    span = maximum_filter(d, size=(1, 3, 3), mode="nearest") - minimum_filter(d, size=(1, 3, 3), mode="nearest")
                    g4_mask = ((conf >= np.percentile(conf, 50)) & np.isfinite(d)
                               & (d > 0) & (span / np.maximum(d, 1e-12) <= .03))
                if world.ndim != 4 or world.shape[3] != 3:
                    raise WebDashboardError("world_points_from_depth must be NxHxWx3")
                if conf.ndim != 3 or conf.shape != world.shape[:3]:
                    raise WebDashboardError("depth_conf shape must match world points")
                if images.ndim not in (3, 4):
                    raise WebDashboardError("images must be Nx3xHxW or NxHxWx3")
                height, width = world.shape[1], world.shape[2]
                if images.ndim == 4:
                    if images.shape[1] == 3 and images.shape[2] == height and images.shape[3] == width:
                        images = np.transpose(images, (0, 2, 3, 1))
                    elif images.shape[3] == 3 and images.shape[1] == height and images.shape[2] == width:
                        pass
                    else:
                        raise WebDashboardError(
                            "images layout must match world_points_from_depth HxW"
                        )
                else:
                    if images.shape[0] == 3 and images.shape[1] == height and images.shape[2] == width:
                        images = np.transpose(images, (1, 2, 0))[np.newaxis, ...]
                    elif images.shape[2] == 3 and images.shape[0] == height and images.shape[1] == width:
                        images = images[np.newaxis, ...]
                    else:
                        raise WebDashboardError(
                            "images layout must match world_points_from_depth HxW"
                        )
                if images.shape[0] != world.shape[0]:
                    raise WebDashboardError("images frame count must match world points")

                model_frame_count = int(world.shape[0])
                merge_indices = _record_alignment(record).get("merge_frame_indices")
                if merge_indices is not None:
                    if (
                        not isinstance(merge_indices, list)
                        or any(
                            isinstance(index, bool)
                            or not isinstance(index, int)
                            or index < 0
                            or index >= world.shape[0]
                            for index in merge_indices
                        )
                        or len(merge_indices) != len(set(merge_indices))
                    ):
                        raise WebDashboardError("merge_frame_indices is invalid")
                    world = world[merge_indices]
                    conf = conf[merge_indices]
                    images = images[merge_indices]
                    if depth is not None:
                        depth = depth[merge_indices]
                    if g4_mask is not None:
                        g4_mask = g4_mask[merge_indices]

                frame_placement = None
                if aligned and is_g4 and _record_alignment(record).get("placement") == "frame_pose":
                    frame_placement = _frame_placement(
                        _record_alignment(record), merge_indices, model_frame_count)

                flat_world = world.reshape(-1, 3)
                flat_conf = conf.reshape(-1)
                if flat_world.shape[0] != flat_conf.shape[0]:
                    raise WebDashboardError("world/conf size mismatch")

                finite_world = np.isfinite(flat_world).all(axis=1)
                if depth is not None:
                    flat_depth = depth.reshape(-1)
                    finite_world &= np.isfinite(flat_depth) & (flat_depth > 0)
                finite_conf = np.isfinite(flat_conf)
                conf_values = flat_conf[finite_world & finite_conf]
                if conf_values.size == 0:
                    threshold = float("inf")
                else:
                    threshold = float(
                        np.percentile(conf_values, conf_percentile)
                    )
                valid_mask = finite_world & finite_conf & (flat_conf >= threshold)
                if g4_mask is not None:
                    conf_percentile = 50.
                    valid_mask = finite_world & finite_conf & g4_mask.reshape(-1)
                valid_indices = np.flatnonzero(valid_mask)
                valid_count = int(valid_indices.size)
                if valid_count == 0:
                    continue
                total_valid += valid_count
                new_priorities = rng.random(valid_count)
                if valid_count > max_points:
                    keep = np.argpartition(new_priorities, max_points - 1)[:max_points]
                    keep = keep[
                        np.argsort(new_priorities[keep], kind="stable")
                    ]
                    valid_indices = valid_indices[keep]
                    new_priorities = new_priorities[keep]

                # Sample/gather before Sim(3) and color conversion.  The Web
                # preview is bounded, so transforming every source point only
                # creates large float64/float32 temporaries with no visible
                # benefit.
                new_xyz = np.asarray(flat_world[valid_indices], dtype=np.float32)
                if frame_placement is not None:
                    # Frames without a localization pose become NaN and are
                    # dropped by the finite filter below.
                    new_xyz = _frame_pose_points(frame_placement, depth, valid_indices)
                elif aligned:
                    scale, rotation, translation = _aligned_transform(record)
                    with np.errstate(over="ignore", invalid="ignore"):
                        new_xyz = (
                            np.float32(scale)
                            * (new_xyz @ np.asarray(rotation, dtype=np.float32).T)
                            + np.asarray(translation, dtype=np.float32)
                        )
                finite_preview = np.isfinite(new_xyz).all(axis=1)
                if not finite_preview.all():
                    valid_indices = valid_indices[finite_preview]
                    new_priorities = new_priorities[finite_preview]
                    new_xyz = new_xyz[finite_preview]
                if new_xyz.shape[0] == 0:
                    continue
                raw_rgb = images.reshape(-1, 3)[valid_indices]
                new_rgb = (
                    np.rint(raw_rgb * 255).clip(0, 255).astype(np.uint8)
                    if is_g4 else _image_to_rgb_u8(raw_rgb)
                )
                windows.append(int(record.get("window_index", 0)))
                priorities = np.concatenate((priorities, new_priorities))
                xyz = np.concatenate((xyz, new_xyz), axis=0)
                rgb = np.concatenate((rgb, new_rgb), axis=0)
                if priorities.size > max_points:
                    keep = np.argpartition(priorities, max_points - 1)[:max_points]
                    keep = keep[np.argsort(priorities[keep], kind="stable")]
                    priorities = priorities[keep].copy()
                    xyz = xyz[keep].copy()
                    rgb = rgb[keep].copy()

        if total_valid == 0 or not windows or xyz.shape[0] == 0:
            raise WebDashboardError("no points pass the confidence filter", 404)

        equivalent_stride = max(1, int(math.ceil(total_valid / float(max_points))))
        displayed = int(xyz.shape[0])

        result = {
            "session_id": session_id,
            "map_id": map_id if aligned else "",
            "aligned": aligned,
            "coordinate_state": (
                _record_coordinate_state(selected_records[-1])
                if aligned
                else COORDINATE_MODEL_LOCAL_UNALIGNED
            ),
            "reason": reason,
            "original_points": total_valid,
            "displayed_points": int(displayed),
            "equivalent_stride": equivalent_stride,
            "sampling": "uniform_reservoir" if total_valid > max_points else "none",
            "conf_percentile": conf_percentile,
            "include_rejected": bool(include_rejected),
            "windows": windows,
            "xyz": xyz,
            "rgb": rgb,
            "_priorities": priorities,
            "_rng_state": rng.bit_generator.state,
            "_selected_fingerprint": fingerprint,
            "_cached": False,
        }
        with self._pc_lock:
            self._pc_cache[key] = dict(result)
            if key in self._pc_order:
                self._pc_order.remove(key)
            self._pc_order.append(key)
            while len(self._pc_order) > 2:
                expired = self._pc_order.pop(0)
                self._pc_cache.pop(expired, None)
        return result

    def point_cloud_binary(self, result: dict) -> bytes:
        import struct

        xyz = result["xyz"]
        rgb = result["rgb"]
        count = int(xyz.shape[0])
        header = struct.pack("<I", count)
        return header + xyz.astype("<f4").tobytes() + rgb.astype("u1").tobytes()

    def point_cloud_meta(self, result: dict) -> dict:
        return {
            "session_id": result["session_id"],
            "map_id": result["map_id"],
            "aligned": result["aligned"],
            "coordinate_state": (
                result.get("coordinate_state", COORDINATE_MODEL_LOCAL_UNALIGNED)
            ),
            "reason": result["reason"],
            "original_points": result["original_points"],
            "displayed_points": result["displayed_points"],
            "sampling": result["sampling"],
            "equivalent_stride": result["equivalent_stride"],
            "conf_percentile": result["conf_percentile"],
            "include_rejected": bool(result.get("include_rejected", False)),
            "windows": result["windows"],
            "format": "uint32 count + float32 xyz[count*3] + uint8 rgb[count*3]",
        }

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


def _formal_records_for(session_id: str, records_root: Path) -> list:
    records = scan_session_windows(records_root, session_id)
    if not records:
        raise WebDashboardError("unknown session", 404)
    return apply_map_snapshot(records, records_root, session_id)


def _dashboard_html() -> str:
    """内嵌 CSS/JS 的单页中文仪表盘；不依赖任何外部资源。"""
    document = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>无人机三维重建地面站软件</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--fg:#e6edf3;--muted:#8b949e;--accent:#58a6ff;--danger:#f85149;--warn:#d29922;--ok:#3fb950}
[hidden]{display:none!important}*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif;background:var(--bg);color:var(--fg)}
header{padding:10px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;flex-wrap:wrap}
header h1{font-size:18px;margin:0}
.badge{font-size:12px;padding:2px 8px;border-radius:10px;border:1px solid var(--border);color:var(--muted)}
.badge.approx{color:var(--warn);border-color:var(--warn)}
.badge.unreliable{color:var(--danger);border-color:var(--danger)}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:12px;padding:12px;align-items:start}
.panel{min-width:0;background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:10px}
.panel h2{font-size:14px;margin:0 0 8px;color:var(--fg);display:flex;justify-content:space-between;align-items:center}
.panel h2 .hint{font-size:12px;color:var(--muted);font-weight:normal}
#viewer{position:relative;height:calc(100vh - 150px);min-height:420px;background:#000;border-radius:6px;overflow:hidden}
#viewer canvas{width:100%;height:100%;display:block}
#viewer-toolbar{position:absolute;left:8px;right:8px;top:8px;display:flex;flex-direction:column;align-items:flex-start;gap:6px;pointer-events:none}
#viewer .pc-badge{max-width:100%;overflow-wrap:anywhere;font-size:12px;background:rgba(0,0,0,.72);padding:4px 8px;border-radius:6px;color:var(--fg)}
#viewer-controls{display:flex;flex-wrap:wrap;gap:5px;align-items:center;align-self:flex-end;max-width:100%;background:rgba(0,0,0,.72);padding:4px;border-radius:6px;pointer-events:auto}
#viewer-controls label{display:flex;gap:4px;align-items:center;min-width:0;max-width:100%;font-size:12px}
#viewer-controls select{min-width:0;max-width:220px}
#viewer-controls input[type=range]{width:128px;max-width:100%}
#axis-legend{position:absolute;left:8px;bottom:8px;font:12px ui-monospace,monospace;background:rgba(0,0,0,.72);padding:4px 6px;border-radius:4px}
.three-col{display:grid;grid-template-rows:auto auto auto;gap:10px}
.live{width:100%;height:190px;background:#000;border-radius:6px;object-fit:contain;border:1px solid var(--border)}
.live-label{font-size:12px;color:var(--muted);margin:4px 0}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{border:1px solid var(--border);padding:3px 6px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{color:var(--muted);font-weight:normal;white-space:nowrap}
.mono{font-family:ui-monospace,monospace}
.muted{color:var(--muted)}
.ok{color:var(--ok)} .warn{color:var(--warn)} .err{color:var(--danger)}
select,input[type=text],input[type=number]{background:#0d1117;color:var(--fg);border:1px solid var(--border);border-radius:4px;padding:4px 6px;font-size:12px;max-width:100%}
button{background:#21262d;color:var(--fg);border:1px solid var(--border);border-radius:4px;padding:4px 10px;font-size:12px;cursor:pointer}
button:hover{background:#30363d}
button.danger{background:#3d1519;border-color:var(--danger);color:#ffb3b8}
button.primary{background:#0c2d6b;border-color:var(--accent)}
form label{font-size:12px;color:var(--muted)}
.msg{margin-top:6px;font-size:12px;min-height:16px}
button:disabled{opacity:.45;cursor:default}
@media(max-width:960px){.layout{grid-template-columns:minmax(0,1fr)!important}#viewer{height:65vh;min-height:380px}.three-col{grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:auto}.live{height:150px}}
@media(max-width:620px){.three-col{grid-template-columns:minmax(0,1fr)}header h1{font-size:16px}#viewer-controls select{max-width:140px}}
</style>
</head>
<body>
<header>
  <h1>无人机三维重建地面站软件</h1>
  <span class="badge" id="alignment-badge">对齐状态: --</span>
  <span class="badge unreliable" id="g4-badge" hidden>地图后端出错，采集继续</span>
  <span class="badge" id="view-session-badge">查看: --</span>
  <span class="badge" id="live-session-badge">控制: --</span>
</header>
<div class="layout">
  <div class="panel">
    <h2><span id="pointcloud-title">累计点云</span>
      <span class="hint">只读预览 · 正式结果保留</span>
    </h2>
    <div id="viewer">
      <canvas id="pointcloud-canvas"></canvas>
      <div id="viewer-toolbar">
      <div class="pc-badge" id="pc-badge">等待加载…</div>
      <div id="viewer-controls">
        <button id="btn-fit" type="button" title="主体视野；只改变相机，不删除点">Fit</button>
        <button id="btn-reset" type="button" title="完整范围（包含异常窗口）">Reset/全景</button>
        <label class="muted">地图分量 <select id="map-select"></select></label>
        <label class="muted"><input id="include-rejected" type="checkbox"> 显示未过配准门限窗口</label>
        <label class="muted">点径 <input id="point-size" type="range" min="1" max="6" step="0.5" value="2"></label>
      </div>
      </div>
      <div id="axis-legend"><span style="color:#ff5d5d">X</span> <span style="color:#57d163">Y</span> <span style="color:#5f8fff">Z</span> · <span id="axis-convention">--</span></div>
    </div>
  </div>
  <div class="panel three-col">
    <div>
      <h2>实时 RGB</h2>
      <img class="live" id="live-rgb" hidden alt="实时 RGB">
      <div class="live-label" id="live-rgb-label">等待流…</div>
    </div>
    <div>
      <h2>实时传感器深度</h2>
      <img class="live" id="live-depth" hidden alt="实时传感器深度">
      <div class="live-label" id="live-depth-label">等待流…</div>
    </div>
    <div>
      <h2>Omega 首帧深度 <span class="hint" id="omega-window-label">--</span></h2>
      <img class="live" id="omega-first-depth" hidden alt="Omega 首帧深度">
      <div class="live-label" id="omega-depth-label">等待最新窗口…</div>
    </div>
  </div>
</div>
<div class="layout" style="grid-template-columns:repeat(3,minmax(0,1fr))">
  <div class="panel">
    <h2>运行状态</h2>
    <table id="status-table"></table>
  </div>
  <div class="panel">
    <h2>会话与历史 <span class="hint">历史会话 · 仅改变本浏览器的查看对象</span></h2>
    <div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">
      <select id="session-select"></select>
      <label><input type="checkbox" id="follow-latest"> 查看时跟随 live</label>
    </div>
    <div class="msg" id="session-msg"></div>
    <table id="formal-table"></table>
  </div>
  <div class="panel">
    <h2>控制 <span class="hint" id="control-target">live session: --</span></h2>
    <div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap">
      <select id="start-role" title="新重建使用的相机角色"></select>
      <button id="btn-start" class="primary command-control" title="在 IDLE/STOPPED 时开始新重建">开始新重建</button>
      <button id="btn-pause-resume" class="command-control" title="运行中暂停；暂停后恢复">暂停/恢复</button>
      <button id="btn-stop" class="danger command-control" title="停止采集；已采集的窗口继续重建完再结束">停止</button>
    </div>
    <form id="preview-form" style="margin-top:8px">
      <label>Web 预览帧率 <span class="muted">web_preview_fps</span> <input type="number" id="p-preview-fps" step="0.5" min="0.5" max="30"></label>
      <button type="submit" class="command-control">应用预览帧率</button>
    </form>
    <div class="msg" id="command-msg"></div>
  </div>
</div>
<script>
"use strict";
var state = {selected_session:"", view_session:"", sessions:[], roles:[], preview_fps:5, formal_records:[], status:{}, diagnostics:{}};
var stateTimer=null, stateInFlight=false, stateFailures=0, lastStateOk=0, recordsKey="", recordsInFlight=false;
var omegaFailures=0,pcFailures=0,omegaRetryTimer=null,pcRetryTimer=null,omegaInFlight="",pcInFlight="";
var commandInFlight=false, previewDirty=false, mapDirty=false, rendererState="new";
var pc = {count:0, positions:null, colors:null, positionBuffer:null, colorBuffer:null, axisPositionBuffer:null, axisColorBuffer:null, bounds:null, zUp:false};
var gl = null, glProgram=null, omegaObjectUrl="";
var viewSession=storageGet("drone3d.view_session","");
var viewFollowLatest=storageGet("drone3d.follow_live","1")!=="0";
var viewMap="",includeRejected=false;
function $(id){return document.getElementById(id)}
function storageGet(key,fallback){try{var value=window.localStorage.getItem(key);return value===null?fallback:value}catch(_e){return fallback}}
function storageSet(key,value){try{window.localStorage.setItem(key,value)}catch(_e){}}
function fetchWithTimeout(url,options,timeoutMs){var controller=new AbortController(),timedOut=false;var timer=setTimeout(function(){timedOut=true;controller.abort()},timeoutMs);var request=Object.assign({},options||{},{signal:controller.signal});return fetch(url,request).then(function(response){return response.arrayBuffer().then(function(body){return new Response(response.status===204?null:body,{status:response.status,statusText:response.statusText,headers:response.headers})})}).catch(function(error){if(timedOut){var timeoutError=new Error("请求超时");timeoutError.code="REQUEST_TIMEOUT";throw timeoutError}throw error}).finally(function(){clearTimeout(timer)})}
function headerText(response,name){var value=response.headers.get(name)||"";try{return decodeURIComponent(value)}catch(_e){return value}}
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
function fmt(s){return s==null||s===""?"未接入":esc(s)}
function fmtBytes(n){if(n===undefined||n===null||n==="")return "未接入"; var v=Number(n); if(!Number.isFinite(v)||v<0)return esc(n); if(v<1024)return v+" B"; var units=["KB","MB","GB","TB"]; var u=-1; do{v/=1024;u++}while(v>=1024&&u<units.length-1); return v.toFixed(v>=100?0:1)+" "+units[u]}
function parseDiagValues(d){var out={}; if(!d||typeof d!=="object")return out;
 Object.keys(d).forEach(function(k){var v=d[k]; if(typeof v!=="object"||v===null){out[k]=v}});
 if(Array.isArray(d.diagnostics)){d.diagnostics.forEach(function(item){if(item&&item.values){Object.keys(item.values).forEach(function(k){out[k]=item.values[k]})}})}
 return out}
function vget(vals){for(var i=0;i<vals.length;i++){var v=vals[i]; if(v!==undefined&&v!==null&&v!=="")return v} return undefined}
function fmtOrNA(s){return (s===undefined||s===null||s==="")?"未接入":esc(s)}
function boolLabel(v){if(v===undefined||v===null||v==="")return "未接入"; if(v===true||v==="true"||v==="True"||v==="1")return "是"; if(v===false||v==="false"||v==="False"||v==="0")return "否"; return esc(v)}
function runningActive(st){return !!st.worker_running_active || (st.running_window!==undefined&&st.running_window!==null&&typeof st.running_window==="object")}
function runningLabel(rw,active){if(rw!==undefined&&rw!==null&&typeof rw==="object"){if(rw.window_index!==undefined&&rw.window_index!==null)return "窗口 "+esc(rw.window_index); return "空闲"} if(active)return "窗口 "+esc(rw); if(rw===undefined||rw===null||rw===""||rw===0)return "空闲"; return "窗口 "+esc(rw)}
function depthMetricsLabel(records){for(var i=(records||[]).length-1;i>=0;i--){var r=records[i]||{}; var dm=r.depth_metrics; if(dm&&typeof dm==="object"){if(dm.status==="unavailable")return "不可用: "+esc(dm.reason||r.depth_metrics_error||"计算失败"); if(dm.coverage!==undefined&&dm.abs_rel!==undefined&&dm.rmse!==undefined)return esc(dm.coverage)+" / "+esc(dm.abs_rel)+" / "+esc(dm.rmse)} if(r.depth_metrics_error)return "不可用: "+esc(r.depth_metrics_error)} return "无深度对照指标"}
function g4Failed(g4s){return !!g4s&&(g4s.state==="error"||(g4s.exit_code!=null&&Number(g4s.exit_code)!==0))}
function updateStatus(compact,raw){var d=(raw&&typeof raw==="object")?raw:compact; state.status=compact||{}; state.diagnostics=raw||d||{}; var v=parseDiagValues(d); var rows=[];
 function row(a,b){rows.push("<tr><th>"+a+"</th><td class='mono'>"+b+"</td></tr>")}
 var eff=(d&&d.effective)||{}; var st=d||{};
 var rw=vget([st.running_window, v.running_window]);
 var workerTime=vget([v.worker_wall_time_sec, v.wall_time_sec, st.worker_wall_time_sec]);
 var mem=vget([v.memory_mb, v.memory, v.mem]); var gpuName=vget([v.gpu_name, v.gpu]); var gpuUtil=vget([v.gpu_util]); var temp=vget([v.temp_c, v.temp, v.temperature]);
 row("live session", fmt(st.session_id)); row("运行/暂停", fmt(st.state));
 row("本次新增采集帧 k/N", fmt(st.captured_frames)+" / "+fmt(st.capture_target_frames));
 if(st.alignment_backend==="global_g4"){
  row("配准方法", "G4 全局配准");
  row("G4 状态 / 地图版本",fmt((st.g4||{}).state)+" / "+fmt((st.g4||{}).revision));
  var g4s=(st.g4||{}); var g4Bad=(g4s.state==="error"||g4s.state==="stopped"||g4s.exit_code!=null);
  var g4Badge=$("g4-badge"); g4Badge.hidden=!g4Failed(g4s); g4Badge.title=g4Failed(g4s)?String(g4s.error||("状态 "+g4s.state+" 退出码 "+g4s.exit_code))+"。窗口照常采集和重建，但没有被放进地图；落地后可以重算。":"";
  row("G4 提示",fmt(g4s.error || (g4Bad ? ("状态 "+fmtOrNA(g4s.state)+" 退出码 "+fmtOrNA(g4s.exit_code)) : (g4s.state ? "无错误" : "等待后台启动"))));
  row("每批新增 / 共享 / 模型输入",fmt(st.new_input_frames)+" / "+fmt(st.shared_input_frames)+" / "+fmt(st.model_input_frames));
 }else{row("重建窗口总帧", fmt(vget([(eff&&eff.window_frames), st.window_frames])));}
 row("已采集窗口", fmt(vget([st.window_index, v.window_index])));
 row("实际采样 fps", fmt(vget([(eff&&eff.sample_fps), st.sample_fps])));
 row("窗口重叠帧", fmt(vget([(eff&&eff.overlap_frames), st.overlap_frames])));
 row("队列模式", fmt(vget([(eff&&eff.queue_mode), v.queue_mode]))); row("队列长度", fmt(st.pending_windows));
 row("项目存储总占用", fmtBytes(st.storage_total_bytes));
 row("采集快照", fmtBytes(st.storage_capture_bytes)+" · "+fmt(st.storage_capture_roots));
 row("Worker 结果", fmtBytes(st.storage_worker_bytes)+" · "+fmt(st.storage_worker_root));
 row("正式记录", fmtBytes(st.storage_formal_bytes)+" · "+fmt(st.storage_formal_root));
 row("共享帧缓存", fmtBytes(st.storage_frame_pool_bytes)+" · "+fmt(st.storage_frame_pool_root));
 row("会话日志", fmtBytes(st.storage_journal_bytes)+" · "+fmt(st.storage_journal_root));
 row("运行日志", fmtBytes(st.storage_runtime_log_bytes)+" · "+fmt(st.storage_runtime_log_root)+"/*.log*");
 row("ROS 日志", fmtBytes(st.storage_ros_log_bytes)+" · "+fmt(st.storage_ros_log_root));
 if(st.storage_scan_error)row("存储统计错误",fmt(st.storage_scan_error));
 row("FIFO 字节", fmtBytes(vget([v.fifo_bytes, st.fifo_bytes]))); row("最老等待(秒)", fmt(vget([v.fifo_oldest_wait_sec, st.fifo_oldest_wait_sec])));
 row("队列满暂停采集", boolLabel(st.capture_queue_blocked));
 row("整窗处理耗时(秒)", fmtOrNA(st.service_wall_time_sec)); row("名义产窗间隔(秒)", fmtOrNA(st.service_budget_sec));
 row("worker 窗口", runningLabel(rw, runningActive(st))); row("worker 状态", runningActive(st)?"运行中":"空闲"); row("worker 耗时", fmtOrNA(workerTime));
 row("持续失败", fmt(vget([st.last_error, v.last_error]))); row("未恢复失败窗口数", fmt(st.failed_windows)); row("缺位姿帧（累计）", fmt(st.missing_pose_frames));
 row("失败尝试 / 人工重试",fmt(st.attempt_failures)+" / "+fmt(st.manual_retries));
 if(st.memory_paused)row("资源失败窗口",fmt(st.retry_window));
 row("辅助覆盖", fmt(vget([v.aux_coverage, v.aux])));
 row("磁盘红线", boolLabel(vget([v.disk_redline, st.disk_redline])));
 row("磁盘剩余 GB", fmt(vget([v.disk_free_gb, st.disk_free_gb])));
 row("系统可用内存", fmtBytes(vget([v.mem_available_bytes, st.mem_available_bytes])));
 row("逻辑 swap", fmtBytes(vget([v.swap_used_bytes, st.swap_used_bytes])));
 row("GPU", gpuName!==undefined?esc(gpuName)+(gpuUtil!==undefined?" @ "+esc(gpuUtil):""):"未接入");
 row("温度", temp!==undefined?esc(temp)+" ℃":"未接入");
 row("最新结果", fmt(st.succeeded_windows)+" 成功 / "+fmt(st.formal_windows)+" 正式"); row("对齐状态", fmt(vget([st.last_alignment_state, v.last_alignment_state])));
 if((vget([st.last_alignment_state, v.last_alignment_state])||"MODEL_LOCAL_UNALIGNED")==="MODEL_LOCAL_UNALIGNED" && String(vget([v.aux_coverage, st.aux_coverage])||"").indexOf("odom-")>=0)row("点云叠加","未接入里程计 · 仅显示最新单窗口");
 row("深度对照", depthMetricsLabel(state.formal_records));
 refreshControls(st);
 $("status-table").innerHTML=rows.join("");
}
function qualityPassed(r){return r.quality_passed!==false}
function qualityLabel(r){if(r.alignment_method==="global_g4")return r.pending?"G4 待定":"G4 地图 v"+r.map_revision+(r.map_stale?" · 旧版本，等待新解":" · 数值可放置");if(r.component_anchor)return "锚点";var ratio=r.inlier_ratio==null?"--":Number(r.inlier_ratio).toFixed(3),rms=r.normalized_residual_rms==null?"--":Number(r.normalized_residual_rms).toFixed(3);return (qualityPassed(r)?"通过 ":"拒绝 ")+ratio+" / "+rms}
function buildFormalTable(){var rows=[]; rows.push("<tr><th>窗口</th><th>状态</th><th>门限</th><th>方法 / 根 / 参考窗</th><th>map</th><th>原因</th></tr>");
 state.formal_records.forEach(function(r){rows.push("<tr><td>"+esc(r.window_index)+"</td><td>"+esc(r.coordinate_state)+"</td><td>"+esc(qualityLabel(r))+"</td><td>"+esc(r.alignment_method)+" / "+esc(r.root_window_index)+" / "+esc(r.reference_window_index)+"</td><td class='mono'>"+esc(r.map_id)+"</td><td class='muted'>"+esc(r.reason)+"</td></tr>")});
 $("formal-table").innerHTML=rows.join("")||"<tr><td class='muted'>无正式窗口</td></tr>"
}
var streamState={initialized:false,lastRole:""};
function refreshStreams(){var role=currentRole(); if(streamState.initialized&&streamState.lastRole===role)return; streamState.initialized=true; streamState.lastRole=role;
 var q="?t="+Date.now();
 $("live-rgb").src="stream/"+encodeURIComponent(role)+"/rgb"+q;
 $("live-rgb").onload=function(){$("live-rgb").hidden=false;$("live-rgb-label").textContent="最近收到的 RGB 帧"};
 $("live-depth").onload=function(){$("live-depth").hidden=false;$("live-depth-label").textContent="最近收到的传感器深度"};
 $("live-depth").src="stream/"+encodeURIComponent(role)+"/depth"+q;
 $("live-rgb").onerror=function(){$("live-rgb").hidden=true;$("live-rgb-label").textContent="实时 RGB 不可用"};
 $("live-depth").onerror=function(){$("live-depth").hidden=true;$("live-depth-label").textContent="实时传感器深度不可用"};
}
function currentRole(){return (state.status&&state.status.role)||"front"}
function latestRecord(records){var latest=null; (records||[]).forEach(function(r){if(!latest){latest=r;return}var current=Number(r.window_index),previous=Number(latest.window_index);if(!Number.isFinite(current))current=-1;if(!Number.isFinite(previous))previous=-1;if(current>previous)latest=r});return latest}
function omegaSpec(){var r=latestRecord(state.formal_records); if(!viewSession||!r)return null; var w=String(r.window_index); return {session:viewSession,window:w,key:viewSession+"|"+w+"|"+(r.result_manifest_sha256||"")}}
function clearOmega(){if(omegaObjectUrl){URL.revokeObjectURL(omegaObjectUrl);omegaObjectUrl=""} $("omega-first-depth").hidden=true;$("omega-first-depth").removeAttribute("src"); $("omega-window-label").textContent="--"; $("omega-depth-label").textContent="等待最新窗口…"}
function refreshOmega(spec){if(!spec)return; fetchWithTimeout("api/latest_first_depth?session="+encodeURIComponent(spec.session)+"&t="+Date.now(),{},15000).then(function(r){if(!r.ok)throw new Error("HTTP "+r.status); if(r.headers.get("X-Session")!==spec.session||r.headers.get("X-Window")!==spec.window)throw new Error("窗口已变更"); return r.blob()}).then(function(b){if(window.__omegaKey!==spec.key)return; $("omega-window-label").textContent="session/window: "+esc(spec.session)+" / "+esc(spec.window); $("omega-depth-label").textContent="已更新"; if(omegaObjectUrl)URL.revokeObjectURL(omegaObjectUrl);omegaObjectUrl=URL.createObjectURL(b);$("omega-first-depth").onload=function(){$("omega-first-depth").hidden=false};$("omega-first-depth").src=omegaObjectUrl}).catch(function(e){if(window.__omegaKey===spec.key){window.__omegaKey=""; $("omega-depth-label").textContent="不可用: "+e.message}})}
function renderRecords(){buildFormalTable();updateAlignmentBadge();updateStatus(state.status,state.diagnostics);updateMapOptions();var omega=omegaSpec(),omegaKey=omega?omega.key:"";if(omegaKey!==window.__omegaKey){clearTimeout(omegaRetryTimer);omegaFailures=0;omegaInFlight="";window.__omegaKey=omegaKey;clearOmega();if(omega)refreshOmega(omega)}var pcSpec=pointCloudSpec();if(pcSpec.key!==window.__pcKey){clearTimeout(pcRetryTimer);pcFailures=0;pcInFlight="";window.__pcKey=pcSpec.key; clearPointCloud(); if(pcSpec.hasRecords)refreshPointCloud(pcSpec)}}
function refreshRecords(key){if(recordsInFlight||!viewSession)return; recordsInFlight=true; var requested=viewSession; fetchWithTimeout("api/state?include_records=1&session="+encodeURIComponent(requested),{},8000).then(function(r){if(!r.ok)throw new Error("HTTP "+r.status);return r.json()}).then(function(detail){if(requested!==viewSession)return; state.formal_records=detail.formal_records||[];recordsKey=key;renderRecords();$("session-msg").textContent="已读取 "+state.formal_records.length+" 个正式窗口"}).catch(function(e){if(requested===viewSession)$("session-msg").textContent="历史记录请求失败: "+e.message}).finally(function(){recordsInFlight=false})}
function refreshState(){if(stateInFlight)return; stateInFlight=true; fetchWithTimeout("api/state?include_records=0",{},3000).then(function(r){if(!r.ok)throw new Error("HTTP "+r.status);return r.json()}).then(function(s){state.selected_session=s.selected_session||"";state.sessions=s.sessions||[];state.roles=s.roles||[];state.preview_fps=s.preview_fps||5;state.status=s.status||{};state.diagnostics=s.diagnostics||{}; var live=(state.status&&state.status.session_id)||""; var known=state.sessions.some(function(x){return x.session_id===viewSession}); if(viewFollowLatest||!viewSession||!known)viewSession=live||(state.sessions[0]&&state.sessions[0].session_id)||""; state.view_session=viewSession;storageSet("drone3d.view_session",viewSession); var sel=$("session-select");sel.innerHTML="";state.sessions.forEach(function(x){var o=document.createElement("option");o.value=x.session_id;o.textContent=x.session_id+" ("+x.window_count+" 窗口)";sel.appendChild(o)});sel.value=viewSession;$("follow-latest").checked=viewFollowLatest;if(!previewDirty)$("p-preview-fps").value=state.preview_fps;$("view-session-badge").textContent="查看: "+(viewSession||"--");$("live-session-badge").textContent="控制: "+(live||"--");$("control-target").textContent="live session: "+(live||"--");var available=(state.status&&Array.isArray(state.status.available_roles))?state.status.available_roles:null;var roleSel=$("start-role"),oldRole=roleSel.value||"";roleSel.innerHTML="";(state.roles.length?state.roles:["front"]).forEach(function(name){var o=document.createElement("option");var ok=!available||available.indexOf(name)>=0;o.value=name;o.textContent=name+(ok?"":"（无图像源）");o.disabled=!ok;roleSel.appendChild(o)});roleSel.value=(oldRole&&state.roles.indexOf(oldRole)>=0&&(!available||available.indexOf(oldRole)>=0))?oldRole:((available&&available[0])||(state.roles[0])||"front");updateStatus(state.status,state.diagnostics);refreshStreams(); var item=state.sessions.filter(function(x){return x.session_id===viewSession})[0]||{}; var nextKey=viewSession+"|"+(item.window_count||0)+"|"+(item.latest_window_index==null?-1:item.latest_window_index)+"|"+(item.map_revision_sha256||"");if(nextKey!==recordsKey)refreshRecords(nextKey); if(stateFailures&&$("session-msg").textContent.indexOf("state 请求失败: ")===0)$("session-msg").textContent=""; lastStateOk=Date.now();stateFailures=0}).catch(function(e){stateFailures++;$("session-msg").textContent="state 请求失败: "+e.message}).finally(function(){stateInFlight=false;scheduleState()})}
function scheduleState(){if(stateTimer)return;var delay=stateFailures?Math.min(5000,1000*Math.pow(2,Math.min(3,stateFailures-1))):1000;stateTimer=setTimeout(function(){stateTimer=null;refreshState()},delay)}
function updateAlignmentBadge(){var rec=null; for(var i=state.formal_records.length-1;i>=0;i--){if(state.formal_records[i].coordinate_state==="ODOM_LOCAL_APPROX"){rec=state.formal_records[i];break}} var badge=$("alignment-badge"); if(rec){var passed=qualityPassed(rec);badge.textContent=(passed?rec.coordinate_state:"配准未过门限")+" "+rec.map_id; badge.className=passed?"badge approx":"badge unreliable"} else {badge.textContent="MODEL_LOCAL_UNALIGNED"; badge.className="badge unreliable"} }
function deletePointBuffers(){if(!gl)return;[pc.positionBuffer,pc.colorBuffer,pc.axisPositionBuffer,pc.axisColorBuffer].forEach(function(buffer){if(buffer)gl.deleteBuffer(buffer)})}
function clearPointCloud(){deletePointBuffers();pc={count:0,positions:null,colors:null,positionBuffer:null,colorBuffer:null,axisPositionBuffer:null,axisColorBuffer:null,bounds:null,zUp:false};if(gl){gl.clearColor(0.04,0.05,0.07,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT)}if(rendererState==="ready")$("pc-badge").textContent="等待正式窗口…"}
function updateMapOptions(){var maps=[];(state.formal_records||[]).forEach(function(r){var aligned=r.coordinate_state==="ODOM_LOCAL_APPROX";if(aligned&&r.map_id&&(includeRejected||qualityPassed(r))&&maps.indexOf(r.map_id)<0)maps.push(r.map_id)});if(!mapDirty||maps.indexOf(viewMap)<0)viewMap=maps.length?maps[maps.length-1]:"";var sel=$("map-select");sel.innerHTML="";if(!maps.length){var empty=document.createElement("option");empty.value="";empty.textContent="无可显示分量";sel.appendChild(empty)}else maps.forEach(function(map){var o=document.createElement("option");o.value=map;o.textContent=map;sel.appendChild(o)});sel.value=viewMap;sel.disabled=maps.length<2}
function pointCloudSpec(){var records=state.formal_records||[],map=viewMap||"latest",selected=[];if(viewMap){selected=records.filter(function(r){return (r.coordinate_state==="ODOM_LOCAL_APPROX")&&r.map_id===viewMap&&(includeRejected||qualityPassed(r))})}else{var latest=latestRecord(records);if(latest)selected=[latest]}var fingerprint=selected.map(function(r){return r.window_index+":"+(r.result_manifest_sha256||"")+":"+(r.map_revision_sha256||"")}).join(","); return {session:viewSession,map:map,includeRejected:includeRejected,key:viewSession+"|"+map+"|"+(includeRejected?"all":"passed")+"|"+fingerprint,hasRecords:selected.some(function(r){return r.alignment_method!=="global_g4"||!r.pending})}}
function refreshPointCloud(spec){if(!spec||!spec.hasRecords||window.__pcKey!==spec.key||pcInFlight===spec.key)return;pcInFlight=spec.key;$("pc-badge").textContent="点云加载中…";var url="api/pointcloud?session="+encodeURIComponent(spec.session)+"&map="+encodeURIComponent(spec.map)+"&include_rejected="+(spec.includeRejected?"1":"0")+"&max_points="+(window.PC_MAX_POINTS||1000000);fetchWithTimeout(url,{},30000).then(function(r){if(!r.ok)throw new Error("HTTP "+r.status); if(r.headers.get("X-Session")!==spec.session)throw new Error("session 已变更"); var aligned=r.headers.get("X-Aligned")==="1"; var original=Number(r.headers.get("X-Original-Points")||0); var displayed=Number(r.headers.get("X-Displayed-Points")||0); var sampling=r.headers.get("X-Sampling")||""; var equivalentStride=Number(r.headers.get("X-Equivalent-Stride")||0); var conf=Number(r.headers.get("X-Conf-Percentile")||0); var mapId=headerText(r,"X-Map-Id"); var reason=headerText(r,"X-Reason"); var stateName=r.headers.get("X-Coordinate-State")||""; var rejected=r.headers.get("X-Include-Rejected")==="1"; return r.arrayBuffer().then(function(buf){if(window.__pcKey===spec.key){pcFailures=0;loadPointCloud(buf,aligned,mapId,original,displayed,sampling,equivalentStride,conf,stateName,reason,rejected)}})}).catch(function(e){if(window.__pcKey===spec.key){pcFailures++;var delay=Math.min(10000,1000*Math.pow(2,Math.min(3,pcFailures-1)));$("pc-badge").textContent="点云加载失败: "+e.message+" · "+(delay/1000)+"秒后重试";clearTimeout(pcRetryTimer);pcRetryTimer=setTimeout(function(){refreshPointCloud(spec)},delay)}}).finally(function(){if(pcInFlight===spec.key)pcInFlight=""})}
function loadPointCloud(buf,aligned,mapId,original,displayed,sampling,equivalentStride,conf,stateName,reason,rejected){if(buf.byteLength<4)throw new Error("点云响应过短");var dv=new DataView(buf),count=dv.getUint32(0,true),expected=4+count*15;if(count<1||expected!==buf.byteLength)throw new Error("点云响应长度非法");var positions=new Float32Array(count*3),colors=new Uint8Array(count*3),colorOffset=4+count*12;for(var i=0;i<count*3;i++){positions[i]=dv.getFloat32(4+i*4,true);colors[i]=dv.getUint8(colorOffset+i)}deletePointBuffers();var zUp=stateName==="ODOM_LOCAL_APPROX";pc={count:count,positions:positions,colors:colors,positionBuffer:null,colorBuffer:null,axisPositionBuffer:null,axisColorBuffer:null,bounds:null,zUp:zUp};uploadPointCloud();orbit.yaw=0.6;orbit.pitch=0.5;fitPointCloud();var sample="sampling "+esc(sampling)+" · 等效倍率 ×"+equivalentStride;var badge=$("pc-badge"),quality=stateName==="ODOM_LOCAL_APPROX"?"<span class='err'>不可靠·未经真值验证</span>":"相对尺度·非公制";if(rejected)quality+=" · <span class='err'>含未过门限窗口</span>";if(aligned){badge.innerHTML=esc(stateName)+" + <span class='mono'>"+esc(mapId)+"</span> · "+quality+" · 原始 "+original+" · 显示 "+displayed+" · "+sample+" · conf P"+conf+" · Fit=主体 / Reset=含异常全景"}else{badge.innerHTML=esc(stateName)+"（不累计）· Omega/OpenCV model-local · 原因: "+esc(reason)+" · 原始 "+original+" · 显示 "+displayed+" · "+sample+" · conf P"+conf+" · Fit=主体 / Reset=含异常全景"}$("axis-convention").textContent=zUp?"ROS odom · Z-up":"Omega/OpenCV session · Y-down"}
function initGL(){var canvas=$("pointcloud-canvas");gl=canvas.getContext("webgl",{antialias:true,preserveDrawingBuffer:true});if(!gl){rendererState="unavailable";$("pc-badge").textContent="WebGL 不可用；请启用浏览器硬件加速";return}var vs="attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uProj; uniform mat4 uView; uniform float uPointSize; varying vec3 vCol; void main(){gl_Position=uProj*uView*vec4(aPos,1.0); gl_PointSize=uPointSize; vCol=aCol;}";var fs="precision mediump float; varying vec3 vCol; void main(){gl_FragColor=vec4(vCol,1.0);}";function sh(type,src){var s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error("shader compile: "+gl.getShaderInfoLog(s));return s}var program=gl.createProgram();gl.attachShader(program,sh(gl.VERTEX_SHADER,vs));gl.attachShader(program,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error("program link: "+gl.getProgramInfoLog(program));glProgram={program:program,uProj:gl.getUniformLocation(program,"uProj"),uView:gl.getUniformLocation(program,"uView"),uPointSize:gl.getUniformLocation(program,"uPointSize"),aPos:gl.getAttribLocation(program,"aPos"),aCol:gl.getAttribLocation(program,"aCol")};if(glProgram.aPos<0||glProgram.aCol<0||!glProgram.uProj||!glProgram.uView||!glProgram.uPointSize)throw new Error("shader location missing");gl.useProgram(program);gl.enable(gl.DEPTH_TEST);rendererState="ready"}
function uploadPointCloud(){if(!gl||rendererState!=="ready"||!pc.count)return;pc.positionBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pc.positionBuffer);gl.bufferData(gl.ARRAY_BUFFER,pc.positions,gl.STATIC_DRAW);pc.colorBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pc.colorBuffer);gl.bufferData(gl.ARRAY_BUFFER,pc.colors,gl.STATIC_DRAW)}
var orbit={yaw:0.6,pitch:0.5,dist:6,target:[0,0,0],radius:1};
function quantileBounds(){var stride=Math.max(1,Math.floor(pc.count/10000)),bounds=[];for(var k=0;k<3;k++){var values=[];for(var i=0;i<pc.count;i+=stride){var value=pc.positions[i*3+k];if(Number.isFinite(value))values.push(value)}values.sort(function(a,b){return a-b});if(!values.length)return null;var lo=values[Math.floor((values.length-1)*0.05)],hi=values[Math.ceil((values.length-1)*0.95)];if(!(hi>lo)){lo=values[0];hi=values[values.length-1]}bounds.push([lo,hi])}return bounds}
function fullBounds(){var bounds=[[Infinity,-Infinity],[Infinity,-Infinity],[Infinity,-Infinity]];for(var i=0;i<pc.count;i++){for(var k=0;k<3;k++){var value=pc.positions[i*3+k];if(Number.isFinite(value)){bounds[k][0]=Math.min(bounds[k][0],value);bounds[k][1]=Math.max(bounds[k][1],value)}}}return bounds[0][0]===Infinity?null:bounds}
function updateAxisBuffers(){if(!gl||!pc.bounds)return;[pc.axisPositionBuffer,pc.axisColorBuffer].forEach(function(b){if(b)gl.deleteBuffer(b)});var t=[0,0,0],s=Math.max(orbit.radius*0.28,0.01),positions=new Float32Array([t[0],t[1],t[2],t[0]+s,t[1],t[2],t[0],t[1],t[2],t[0],t[1]+s,t[2],t[0],t[1],t[2],t[0],t[1],t[2]+s]),colors=new Uint8Array([255,70,70,255,70,70,70,220,90,70,220,90,80,120,255,80,120,255]);pc.axisPositionBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pc.axisPositionBuffer);gl.bufferData(gl.ARRAY_BUFFER,positions,gl.STATIC_DRAW);pc.axisColorBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,pc.axisColorBuffer);gl.bufferData(gl.ARRAY_BUFFER,colors,gl.STATIC_DRAW)}
function applyViewBounds(b){if(!b)return;pc.bounds=b;orbit.target=[(b[0][0]+b[0][1])/2,(b[1][0]+b[1][1])/2,(b[2][0]+b[2][1])/2];orbit.radius=Math.hypot(b[0][1]-b[0][0],b[1][1]-b[1][0],b[2][1]-b[2][0])/2||1;orbit.dist=Math.max(orbit.radius*2.6,0.01);updateAxisBuffers();drawPointCloud()}
function fitPointCloud(){if(pc.count)applyViewBounds(quantileBounds())}
function resetView(){orbit.yaw=0.6;orbit.pitch=0.5;if(pc.count)applyViewBounds(fullBounds())}
function drawPointCloud(){if(!gl||!glProgram||rendererState!=="ready")return;var canvas=$("pointcloud-canvas"),width=Math.max(1,canvas.clientWidth),height=Math.max(1,canvas.clientHeight);if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height}gl.viewport(0,0,width,height);gl.clearColor(0.04,0.05,0.07,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);if(!pc.count||!pc.positionBuffer)return;var radius=orbit.radius||1,near=Math.max(0.0001,radius/500),far=Math.max(radius*100,orbit.dist+radius*4),target=orbit.target,eye,up;if(pc.zUp){eye=[target[0]+orbit.dist*Math.cos(orbit.pitch)*Math.cos(orbit.yaw),target[1]+orbit.dist*Math.cos(orbit.pitch)*Math.sin(orbit.yaw),target[2]+orbit.dist*Math.sin(orbit.pitch)];up=[0,0,1]}else{eye=[target[0]+orbit.dist*Math.cos(orbit.pitch)*Math.sin(orbit.yaw),target[1]+orbit.dist*Math.sin(orbit.pitch),target[2]+orbit.dist*Math.cos(orbit.pitch)*Math.cos(orbit.yaw)];up=[0,-1,0]}var proj=matPerspective(45*Math.PI/180,width/height,near,far),view=matLookAt(eye,target,up),p=glProgram;gl.useProgram(p.program);gl.uniformMatrix4fv(p.uProj,false,proj);gl.uniformMatrix4fv(p.uView,false,view);gl.uniform1f(p.uPointSize,Number($("point-size").value)||2);gl.bindBuffer(gl.ARRAY_BUFFER,pc.positionBuffer);gl.enableVertexAttribArray(p.aPos);gl.vertexAttribPointer(p.aPos,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,pc.colorBuffer);gl.enableVertexAttribArray(p.aCol);gl.vertexAttribPointer(p.aCol,3,gl.UNSIGNED_BYTE,true,0,0);gl.drawArrays(gl.POINTS,0,pc.count);if(pc.axisPositionBuffer){gl.uniform1f(p.uPointSize,1);gl.bindBuffer(gl.ARRAY_BUFFER,pc.axisPositionBuffer);gl.vertexAttribPointer(p.aPos,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,pc.axisColorBuffer);gl.vertexAttribPointer(p.aCol,3,gl.UNSIGNED_BYTE,true,0,0);gl.drawArrays(gl.LINES,0,6)}canvas.dataset.drawCount=String(pc.count);canvas.dataset.glError=String(gl.getError());canvas.dataset.orbit=JSON.stringify({target:target,radius:radius,dist:orbit.dist,eye:eye,near:near,far:far})}
function matPerspective(fov,aspect,near,far){var f=1/Math.tan(fov/2),nf=1/(near-far); return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]}
function matLookAt(eye,center,up){var z=[eye[0]-center[0],eye[1]-center[1],eye[2]-center[2]]; var l=Math.hypot(z[0],z[1],z[2])||1; z=[z[0]/l,z[1]/l,z[2]/l]; var x=[up[1]*z[2]-up[2]*z[1],up[2]*z[0]-up[0]*z[2],up[0]*z[1]-up[1]*z[0]]; l=Math.hypot(x[0],x[1],x[2])||1; x=[x[0]/l,x[1]/l,x[2]/l]; var y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]]; return [x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -(x[0]*eye[0]+x[1]*eye[1]+x[2]*eye[2]), -(y[0]*eye[0]+y[1]*eye[1]+y[2]*eye[2]), -(z[0]*eye[0]+z[1]*eye[1]+z[2]*eye[2]), 1]}
function bindViewer(){var c=$("pointcloud-canvas"),drag=false,lx=0,ly=0;c.addEventListener("webglcontextlost",function(e){e.preventDefault();rendererState="lost";gl=null;glProgram=null;$("pc-badge").textContent="WebGL context 已丢失，等待自动恢复…"});c.addEventListener("webglcontextrestored",function(){try{initGL();uploadPointCloud();updateAxisBuffers();drawPointCloud()}catch(e){rendererState="error";$("pc-badge").textContent="WebGL 恢复失败: "+e.message}});c.addEventListener("mousedown",function(e){drag=true;lx=e.clientX;ly=e.clientY});window.addEventListener("mouseup",function(){drag=false});window.addEventListener("mousemove",function(e){if(!drag)return;orbit.yaw+=(e.clientX-lx)*0.01;orbit.pitch=Math.max(-1.4,Math.min(1.4,orbit.pitch+(e.clientY-ly)*0.01));lx=e.clientX;ly=e.clientY;drawPointCloud()});c.addEventListener("wheel",function(e){e.preventDefault();orbit.dist=Math.max(0.01,orbit.dist*Math.exp(e.deltaY*0.001));drawPointCloud()},{passive:false});window.addEventListener("resize",drawPointCloud);$("btn-fit").onclick=fitPointCloud;$("btn-reset").onclick=resetView;$("point-size").oninput=drawPointCloud}
function refreshControls(st){var s=(st&&st.state)||""; var startable=(s==="IDLE"||s==="STOPPED"); var pausable=(s==="RUNNING"||s==="DISK_PAUSED"); var retryable=(s==="MEMORY_PAUSED"); var resumable=(s==="PAUSED"||retryable); var stoppable=(s==="RUNNING"||s==="PAUSED"||s==="DISK_PAUSED"||s==="FATAL_OOM"||s==="MEMORY_PAUSED"||s==="DRAINING");var roleSel=$("start-role"),roleOk=!(roleSel.selectedOptions&&roleSel.selectedOptions[0]&&roleSel.selectedOptions[0].disabled);
 $("btn-start").disabled=commandInFlight||!startable||!roleOk;
 $("btn-pause-resume").disabled=commandInFlight||!(pausable||resumable);
 $("btn-pause-resume").textContent=retryable?"重试资源失败窗":(resumable?"恢复":(pausable?"暂停":"暂停/恢复"));
  $("btn-pause-resume").title=retryable?"只重试被资源门拒绝的那一个窗口：复用已保留输入，阈值与门禁不变；已完成的窗口不重采。":(resumable?"恢复采集":"运行中暂停；暂停后恢复");
 $("btn-stop").disabled=commandInFlight||!stoppable;
 $("btn-stop").textContent=s==="DRAINING"?"立即停止":"停止";
 $("btn-stop").title=s==="DRAINING"?"丢弃还没重建的窗口并立即停止":"停止采集；已采集的窗口继续重建完再结束";
}
function setCommandsDisabled(value){["#btn-pause-resume",".command-control"].forEach(function(selector){document.querySelectorAll(selector).forEach(function(button){button.disabled=value})})}
function postCommand(cmd,args,timeoutMs){if(commandInFlight)return Promise.resolve({ok:false,error:"已有命令执行中"});commandInFlight=true;setCommandsDisabled(true);$("command-msg").textContent="执行中: "+cmd;var commandTimeout=timeoutMs||{"start_session":30000,"stop_all":30000,"select_source":15000,"pause":15000,"resume":15000}[cmd]||15000;return fetchWithTimeout("api/command",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(Object.assign({command:cmd},args||{}))},commandTimeout).then(function(response){return response.json().catch(function(){throw new Error("HTTP "+response.status+" 且响应不是 JSON")}).then(function(payload){if(!response.ok||!payload.ok)throw new Error(payload.error||("HTTP "+response.status));return payload})}).then(function(result){$("command-msg").textContent="OK: "+cmd+" · "+JSON.stringify(result.response||{});refreshState();return result}).catch(function(error){$("command-msg").textContent="ERR: "+cmd+" · "+error.message;return {ok:false,error:error.message}}).finally(function(){commandInFlight=false;setCommandsDisabled(false);refreshControls(state.status)})}
function wire(){var canvas=$("pointcloud-canvas");bindViewer();try{initGL()}catch(e){rendererState="error";gl=null;$("pc-badge").textContent="WebGL 初始化失败: "+e.message}
 $("follow-latest").addEventListener("change",function(){viewFollowLatest=this.checked;storageSet("drone3d.follow_live",viewFollowLatest?"1":"0");recordsKey="";refreshState()});
 $("map-select").addEventListener("change",function(){viewMap=this.value;mapDirty=true;renderRecords()});
 $("include-rejected").addEventListener("change",function(){includeRejected=this.checked;renderRecords()});
 $("session-select").addEventListener("change",function(){if(!this.value)return;viewFollowLatest=false;$("follow-latest").checked=false;storageSet("drone3d.follow_live","0");viewSession=this.value;viewMap="";mapDirty=false;storageSet("drone3d.view_session",viewSession);state.formal_records=[];recordsKey="";window.__pcKey="";window.__omegaKey="";clearPointCloud();clearOmega();refreshRecords(viewSession+"|manual")});
 $("btn-start").onclick=function(){var role=$("start-role").value; if(!role){$("command-msg").textContent="ERR: start_session · 未选择相机角色";return} postCommand("start_session",{role:role})};
 $("btn-pause-resume").onclick=function(){var s=(state.status&&state.status.state)||""; postCommand((s==="PAUSED"||s==="MEMORY_PAUSED")?"resume":"pause",{})};
 $("btn-stop").onclick=function(){var st=state.status||{},n=(st.pending_windows||0)+(runningActive(st)?1:0);var text=st.state==="DRAINING"?"立即停止：还没重建的 "+n+" 个窗口将被丢弃，确定吗？":"停止采集：正在采的窗口丢弃，已采集的窗口继续重建完再结束会话。确定吗？";if(window.confirm(text))postCommand("stop_all",{})};
 $("p-preview-fps").addEventListener("input",function(){previewDirty=true});
 $("preview-form").addEventListener("submit",function(e){e.preventDefault();var value=Number($("p-preview-fps").value);if(Number.isFinite(value)&&value>=0.5&&value<=30)postCommand("set_preview_fps",{value:value}).then(function(result){if(result.ok)previewDirty=false});else $("command-msg").textContent="ERR: web_preview_fps 必须在 [0.5, 30]"});
 refreshState()}
var controlPanel=$("preview-form").closest(".panel"),side=document.querySelector(".three-col");
var advanced=document.createElement("details"),summary=document.createElement("summary");summary.textContent="预览参数";summary.style.cssText="cursor:pointer;color:var(--muted);font-size:12px;margin:8px 0";advanced.append(summary);controlPanel.append(advanced);advanced.append($("preview-form"));side.prepend(controlPanel);
if(window.top!==window){var nav=document.querySelector("header nav");if(nav)nav.hidden=true}
wire();
</script>
</body>
</html>"""

    return document
