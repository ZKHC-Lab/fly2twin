"""Capture error types, the per-frame record and identifier patterns shared by the capture modules.

Split from ``capture``, which still re-exports the public names.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .preparation import INTERNAL_ERROR, INVALID_GOAL, IO_ERROR
from .snapshot_validator import FrameStamp


CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"


# scene_id / camera_name / 目录 token 的安全标识符正则
_SCENE_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?\Z")
_CAMERA_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_CAMERA_FRAME_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_/.-]{0,126}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
# 深度原始字节的元数据安全编码（ROS sensor_msgs/Image.encoding 值）
_DEPTH_ENCODING = re.compile(r"[A-Za-z0-9_]{1,32}\Z")


class CaptureError(RuntimeError):
    """类型化采集失败，映射到冻结的 CaptureSegmentResult 错误码。

    Attributes:
        code: 稳定错误码（INVALID_GOAL / CAMERA_UNAVAILABLE / IO_ERROR / INTERNAL_ERROR）
    """

    def __init__(self, code: str, message: str) -> None:
        if code not in (
            INVALID_GOAL,
            CAMERA_UNAVAILABLE,
            IO_ERROR,
            INTERNAL_ERROR,
        ):
            raise ValueError("unsupported capture error code")
        if not isinstance(message, str) or not message:
            raise ValueError("capture error message must be nonempty")
        self.code = code
        super().__init__(message)


class CaptureStreamError(CaptureError):
    """图像/里程计流违反 stamp 或编码契约。

    这类失败统一映射为 CAMERA_UNAVAILABLE：流不再可信，相当于相机不可用。
    """

    def __init__(self, message: str) -> None:
        super().__init__(CAMERA_UNAVAILABLE, message)


@dataclass(frozen=True)
class CapturedFrame:
    """一帧已写入暂存目录的采集结果元数据。"""

    index: int
    stamp: FrameStamp
    image_relative_path: str
    pose_valid: bool
    pose_delta_ms: float | None
    odometry_gap_ms: float | None
