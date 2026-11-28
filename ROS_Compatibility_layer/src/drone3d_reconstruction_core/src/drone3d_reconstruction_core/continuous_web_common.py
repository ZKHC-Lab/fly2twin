"""Shared pieces of the dashboard: errors, request parsing, trusted file access,
record helpers, frame images and the page itself.

Split from ``continuous_web``, which still re-exports the names other modules use.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import unquote, urlsplit

from .formal_result import (
    scan_session_windows,
)
from .odom_local import (
    COORDINATE_MODEL_LOCAL_UNALIGNED,
    COORDINATE_ODOM_LOCAL_APPROX,
)
from .g4.map_snapshot import apply_map_snapshot


_SESSION_ID_RE = re.compile(r"s[0-9a-f]{16}\Z")


_WINDOW_DIR_RE = re.compile(r"window-([0-9]{4,})\Z")


_SAFE_ROLE_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


_SAFE_FRAME_ID_RE = re.compile(r"[A-Za-z0-9_/-]{1,128}\Z")


_MAX_JSON_BODY_BYTES = 64 * 1024


_MAX_RESULT_MANIFEST_BYTES = 16 * 1024 * 1024


_COORDINATE_STATES = frozenset(
    (
        COORDINATE_MODEL_LOCAL_UNALIGNED,
        COORDINATE_ODOM_LOCAL_APPROX,
    )
)


_ALIGNED_STATES = frozenset(
    (COORDINATE_ODOM_LOCAL_APPROX,)
)


class WebDashboardError(ValueError):
    """Web 配置或请求非法；第二个位置参数是 HTTP 状态码（默认 400）。"""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message, status)
        self.message = message
        self.status = status

    def __str__(self) -> str:
        return str(self.message)


def _json_bytes(payload: Any, status: int = 200) -> bytes:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return body


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


def _formal_records_for(session_id: str, records_root: Path) -> list:
    records = scan_session_windows(records_root, session_id)
    if not records:
        raise WebDashboardError("unknown session", 404)
    return apply_map_snapshot(records, records_root, session_id)


@lru_cache(maxsize=1)
def _dashboard_html() -> str:
    """内嵌 CSS/JS 的单页中文仪表盘；不依赖任何外部资源。

    页面在包内 ``static/dashboard.html``；源码树、catkin devel 和安装后的布局里
    它都在本模块旁边（安装规则见 setup.py 的 package_data）。
    """
    with open(Path(__file__).with_name("static") / "dashboard.html", encoding="utf-8", newline="") as stream:
        return stream.read()
