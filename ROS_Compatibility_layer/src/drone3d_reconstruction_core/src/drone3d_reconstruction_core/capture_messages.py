"""ROS messages to plain, validated dictionaries for capture snapshots.

CameraInfo, Odometry, depth Image and TF messages become the JSON payloads stored in a
snapshot's auxiliary files. Split from ``capture``, which still re-exports the public names.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any



from .snapshot_validator import (
    FrameStamp,
)
from ._common import is_finite_number as _finite_number
from .capture_types import _CAMERA_FRAME_ID, _DEPTH_ENCODING, CaptureStreamError


@dataclass(frozen=True)
class AuxiliaryCameraInfo:
    """一帧匹配到的完整 CameraInfo（JSON 安全的全量 payload）。"""

    stamp: FrameStamp
    frame_id: str
    payload: dict
    byte_size: int


@dataclass(frozen=True)
class AuxiliaryOdometry:
    """一帧匹配到的完整 Odometry（JSON 安全的全量 payload）。"""

    stamp: FrameStamp
    frame_id: str
    child_frame_id: str
    payload: dict
    byte_size: int


@dataclass(frozen=True)
class AuxiliaryDepth:
    """一帧匹配到的对齐传感器深度原始数据。"""

    stamp: FrameStamp
    frame_id: str
    encoding: str
    width: int
    height: int
    step: int
    is_bigendian: int
    data: bytes


@dataclass(frozen=True)
class AuxiliaryFrameData:
    """一帧图像的全部辅助数据关联结果（v4 契约）。"""

    camera_info: AuxiliaryCameraInfo | None = None
    camera_info_delta_ms: float | None = None
    odometry: AuxiliaryOdometry | None = None
    odometry_delta_ms: float | None = None
    depth: AuxiliaryDepth | None = None
    depth_delta_ms: float | None = None


def _message_stamp(message: Any) -> FrameStamp:
    """从消息提取并校验 header stamp（兼容 secs/nsecs 与 sec/nsec 两种命名）。"""
    header = getattr(message, "header", None)
    if header is None:
        raise CaptureStreamError("message has no header")
    stamp = getattr(header, "stamp", header)
    sec = getattr(stamp, "secs", getattr(stamp, "sec", None))
    nsec = getattr(stamp, "nsecs", getattr(stamp, "nsec", None))
    if (
        isinstance(sec, bool)
        or not isinstance(sec, int)
        or sec < 0
        or isinstance(nsec, bool)
        or not isinstance(nsec, int)
        or nsec < 0
        or nsec > 999999999
    ):
        raise CaptureStreamError("message header stamp is invalid")
    # 零 stamp 不能作为帧身份（ROS 未填充时间的信号）
    if sec == 0 and nsec == 0:
        raise CaptureStreamError("zero ROS header stamps are not valid frame identity")
    return FrameStamp(sec=sec, nsec=nsec)


def _watchdog_image_stamp(message: Any) -> FrameStamp:
    """提取图像 stamp 供 source watchdog 分类。

    与 _message_stamp 相同，但允许 (0,0)：零 stamp 由 StampSampler 分类
    计数后跳过，不再直接把整窗打死。
    """
    header = getattr(message, "header", None)
    if header is None:
        raise CaptureStreamError("message has no header")
    stamp = getattr(header, "stamp", header)
    sec = getattr(stamp, "secs", getattr(stamp, "sec", None))
    nsec = getattr(stamp, "nsecs", getattr(stamp, "nsec", None))
    if (
        isinstance(sec, bool)
        or not isinstance(sec, int)
        or sec < 0
        or isinstance(nsec, bool)
        or not isinstance(nsec, int)
        or nsec < 0
        or nsec > 999999999
    ):
        raise CaptureStreamError("message header stamp is invalid")
    return FrameStamp(sec=sec, nsec=nsec)


def _message_frame_id(message: Any) -> str:
    """提取并校验消息 header.frame_id（必须是非空字符串）。"""
    header = getattr(message, "header", None)
    if header is None:
        raise CaptureStreamError("message has no header")
    frame_id = getattr(header, "frame_id", None)
    if (
        not isinstance(frame_id, str)
        or not frame_id
        or "\x00" in frame_id
        or not _CAMERA_FRAME_ID.fullmatch(frame_id)
    ):
        raise CaptureStreamError("message header frame_id is invalid")
    return frame_id


def _aux_finite(value: Any, label: str) -> float:
    """辅助 payload 的有限数值检查（失败抛 ValueError，只计辅助错误）。"""
    if not _finite_number(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _aux_header(message: Any, label: str) -> dict[str, Any]:
    """提取完整 ROS header（stamp + frame_id，保留 seq）为 JSON 安全对象。"""
    header = getattr(message, "header", None)
    if header is None:
        raise ValueError(f"{label} has no header")
    stamp = _message_stamp(message)
    frame_id = _message_frame_id(message)
    seq = getattr(header, "seq", None)
    if seq is None:
        seq = 0
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ValueError(f"{label} header seq is invalid")
    return {
        "seq": int(seq),
        "stamp": {"sec": stamp.sec, "nsec": stamp.nsec},
        "frame_id": frame_id,
    }


def _aux_vector(value: Any, label: str, length: int) -> list[float]:
    """把长度固定的数值数组转换为有限 float 列表。"""
    try:
        items = list(value)
    except TypeError as error:
        raise ValueError(f"{label} is not an array") from error
    if len(items) != length:
        raise ValueError(f"{label} must have length {length}")
    return [_aux_finite(item, f"{label}[{index}]") for index, item in enumerate(items)]


def _camera_info_payload(message: Any) -> AuxiliaryCameraInfo:
    """提取完整 CameraInfo 为 JSON 安全 payload（含 K/D/R/P/frame/stamp）。"""
    if getattr(message, "header", None) is None:
        raise ValueError("camera_info has no header")
    stamp = _message_stamp(message)
    frame_id = _message_frame_id(message)
    height = int(message.height)
    width = int(message.width)
    if height <= 0 or width <= 0:
        raise ValueError("camera_info dimensions must be positive")
    distortion_model = str(message.distortion_model)
    if not distortion_model or "\x00" in distortion_model:
        raise ValueError("camera_info distortion_model is invalid")
    d = [_aux_finite(item, f"camera_info.D[{index}]") for index, item in enumerate(list(message.D))]
    if len(d) > 16:
        raise ValueError("camera_info.D has too many coefficients")
    k = _aux_vector(message.K, "camera_info.K", 9)
    r = _aux_vector(message.R, "camera_info.R", 9)
    p = _aux_vector(message.P, "camera_info.P", 12)
    binning_x = int(message.binning_x)
    binning_y = int(message.binning_y)
    if binning_x < 0 or binning_y < 0:
        raise ValueError("camera_info binning must be nonnegative")
    roi = getattr(message, "roi", None)
    roi_payload = None
    if roi is not None:
        for name in ("x_offset", "y_offset", "width", "height"):
            value = int(getattr(roi, name))
            if value < 0:
                raise ValueError(f"camera_info roi {name} must be nonnegative")
        do_rectify = bool(getattr(roi, "do_rectify", False))
        roi_payload = {
            "x_offset": int(roi.x_offset),
            "y_offset": int(roi.y_offset),
            "width": int(roi.width),
            "height": int(roi.height),
            "do_rectify": do_rectify,
        }
    payload = {
        "header": _aux_header(message, "camera_info"),
        "height": height,
        "width": width,
        "distortion_model": distortion_model,
        "D": d,
        "K": k,
        "R": r,
        "P": p,
        "binning_x": binning_x,
        "binning_y": binning_y,
        "roi": roi_payload,
    }
    byte_size = len(
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True
        ).encode("utf-8")
    )
    return AuxiliaryCameraInfo(
        stamp=stamp, frame_id=frame_id, payload=payload, byte_size=byte_size
    )


def _covariance_payload(value: Any, label: str) -> list[float]:
    """提取 ROS Pose/TwistWithCovariance 的固定 36 项数组。"""
    try:
        covariance = list(value.covariance)
    except (AttributeError, TypeError) as error:
        raise ValueError(f"{label}.covariance is missing") from error
    if len(covariance) != 36:
        raise ValueError(f"{label}.covariance must have exactly 36 entries")
    return [
        _aux_finite(item, f"{label}.covariance[{index}]")
        for index, item in enumerate(covariance)
    ]


def _pose_payload(value: Any, label: str) -> dict[str, Any]:
    """提取 ROS PoseWithCovariance 的完整数值字段。

    nav_msgs/Odometry.pose 是 PoseWithCovariance（内含 pose）。
    nav_msgs/Odometry.twist 是 TwistWithCovariance，必须使用
    :func:`_twist_payload`，其 linear/angular 不是 position/orientation。
    """
    if value is None:
        raise ValueError(f"{label} is missing")
    inner = getattr(value, "pose", None)
    if inner is None:
        inner = value
    position = getattr(inner, "position", None)
    orientation = getattr(inner, "orientation", None)
    if position is None or orientation is None:
        raise ValueError(f"{label} is missing position/orientation")
    position_payload = {
        "x": _aux_finite(getattr(position, "x"), f"{label}.position.x"),
        "y": _aux_finite(getattr(position, "y"), f"{label}.position.y"),
        "z": _aux_finite(getattr(position, "z"), f"{label}.position.z"),
    }
    orientation_payload = {
        "x": _aux_finite(getattr(orientation, "x"), f"{label}.orientation.x"),
        "y": _aux_finite(getattr(orientation, "y"), f"{label}.orientation.y"),
        "z": _aux_finite(getattr(orientation, "z"), f"{label}.orientation.z"),
        "w": _aux_finite(getattr(orientation, "w"), f"{label}.orientation.w"),
    }
    return {
        "position": position_payload,
        "orientation": orientation_payload,
        "covariance": _covariance_payload(value, label),
    }


def _twist_payload(value: Any, label: str) -> dict[str, Any]:
    """提取 ROS TwistWithCovariance 的完整数值字段。

    nav_msgs/Odometry.twist 内含 twist（linear/angular），没有
    position/orientation；若误按 pose 解析会丢失线/角速度。
    """
    if value is None:
        raise ValueError(f"{label} is missing")
    inner = getattr(value, "twist", None)
    if inner is None:
        inner = value
    linear = getattr(inner, "linear", None)
    angular = getattr(inner, "angular", None)
    if linear is None or angular is None:
        raise ValueError(f"{label} is missing linear/angular")
    linear_payload = {
        "x": _aux_finite(getattr(linear, "x"), f"{label}.linear.x"),
        "y": _aux_finite(getattr(linear, "y"), f"{label}.linear.y"),
        "z": _aux_finite(getattr(linear, "z"), f"{label}.linear.z"),
    }
    angular_payload = {
        "x": _aux_finite(getattr(angular, "x"), f"{label}.angular.x"),
        "y": _aux_finite(getattr(angular, "y"), f"{label}.angular.y"),
        "z": _aux_finite(getattr(angular, "z"), f"{label}.angular.z"),
    }
    return {
        "linear": linear_payload,
        "angular": angular_payload,
        "covariance": _covariance_payload(value, label),
    }


def _odometry_payload(message: Any) -> AuxiliaryOdometry:
    """提取完整 nav_msgs/Odometry 为 JSON 安全 payload。"""
    stamp = _message_stamp(message)
    frame_id = _message_frame_id(message)
    child_frame_id = getattr(message, "child_frame_id", None)
    if (
        not isinstance(child_frame_id, str)
        or not child_frame_id
        or not _CAMERA_FRAME_ID.fullmatch(child_frame_id)
    ):
        raise ValueError("odometry child_frame_id is invalid")
    pose = _pose_payload(getattr(message, "pose", None), "odometry.pose")
    twist = _twist_payload(getattr(message, "twist", None), "odometry.twist")
    payload = {
        "header": _aux_header(message, "odometry"),
        "child_frame_id": child_frame_id,
        "pose": pose,
        "twist": twist,
    }
    byte_size = len(
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, sort_keys=True
        ).encode("utf-8")
    )
    return AuxiliaryOdometry(
        stamp=stamp,
        frame_id=frame_id,
        child_frame_id=child_frame_id,
        payload=payload,
        byte_size=byte_size,
    )


def _depth_payload(message: Any) -> AuxiliaryDepth:
    """提取原始传感器深度消息（保存原始字节，不做有损转换）。"""
    stamp = _message_stamp(message)
    frame_id = _message_frame_id(message)
    encoding = getattr(message, "encoding", None)
    if (
        not isinstance(encoding, str)
        or not _DEPTH_ENCODING.fullmatch(encoding)
        or len(encoding.encode("ascii")) > 32
    ):
        raise ValueError("depth encoding is invalid")
    width = int(getattr(message, "width"))
    height = int(getattr(message, "height"))
    step = int(getattr(message, "step"))
    if width <= 0 or height <= 0 or step <= 0:
        raise ValueError("depth dimensions/step must be positive")
    is_bigendian = int(getattr(message, "is_bigendian", 0))
    if is_bigendian not in (0, 1):
        raise ValueError("depth is_bigendian must be 0 or 1")
    try:
        data = bytes(message.data)
    except BaseException as error:
        raise ValueError("depth payload is not a byte buffer") from error
    if len(data) != step * height:
        raise ValueError("depth payload length does not match step*height")
    return AuxiliaryDepth(
        stamp=stamp,
        frame_id=frame_id,
        encoding=encoding,
        width=width,
        height=height,
        step=step,
        is_bigendian=is_bigendian,
        data=data,
    )


def _tf_payloads(message: Any, static: bool) -> list[dict[str, Any]]:
    """提取 tf/tf_static 消息中的全部变换（逐条校验数值有限）。

    `/tf_static` 是 ROS 常见的 latched 静态变换，其 stamp 通常为 (0,0)；
    只有 static=True 时允许零 stamp 入库，动态 /tf 的零 stamp 仍按流契约
    拒绝。
    """
    transforms = getattr(message, "transforms", None)
    if transforms is None:
        raise ValueError("tf message has no transforms")
    entries = []
    for transform in list(transforms):
        stamp = _tf_message_stamp(transform, static)
        parent = _message_frame_id(transform)
        child = getattr(transform, "child_frame_id", None)
        if (
            not isinstance(child, str)
            or not child
            or not _CAMERA_FRAME_ID.fullmatch(child)
        ):
            raise ValueError("tf child_frame_id is invalid")
        inner = getattr(transform, "transform", None)
        if inner is None:
            raise ValueError("tf transform is missing transform")
        translation = getattr(inner, "translation", None)
        rotation = getattr(inner, "rotation", None)
        if translation is None or rotation is None:
            raise ValueError("tf transform is missing translation/rotation")
        entries.append(
            {
                "stamp": {"sec": stamp.sec, "nsec": stamp.nsec},
                "parent_frame_id": parent,
                "child_frame_id": child,
                "translation": {
                    "x": _aux_finite(getattr(translation, "x"), "tf.translation.x"),
                    "y": _aux_finite(getattr(translation, "y"), "tf.translation.y"),
                    "z": _aux_finite(getattr(translation, "z"), "tf.translation.z"),
                },
                "rotation": {
                    "x": _aux_finite(getattr(rotation, "x"), "tf.rotation.x"),
                    "y": _aux_finite(getattr(rotation, "y"), "tf.rotation.y"),
                    "z": _aux_finite(getattr(rotation, "z"), "tf.rotation.z"),
                    "w": _aux_finite(getattr(rotation, "w"), "tf.rotation.w"),
                },
                "static": bool(static),
            }
        )
    return entries


def _tf_message_stamp(message: Any, static: bool) -> FrameStamp:
    """提取 TF 消息 stamp；static 变换允许 (0,0)，其余必须是非零有效 stamp。"""
    header = getattr(message, "header", None)
    if header is None:
        raise ValueError("tf transform has no header")
    stamp = getattr(header, "stamp", header)
    sec = getattr(stamp, "secs", getattr(stamp, "sec", None))
    nsec = getattr(stamp, "nsecs", getattr(stamp, "nsec", None))
    if (
        isinstance(sec, bool)
        or not isinstance(sec, int)
        or sec < 0
        or isinstance(nsec, bool)
        or not isinstance(nsec, int)
        or nsec < 0
        or nsec > 999999999
    ):
        raise ValueError("tf transform header stamp is invalid")
    if static and sec == 0 and nsec == 0:
        return FrameStamp(sec=0, nsec=0)
    if sec == 0 and nsec == 0:
        raise ValueError("zero tf stamps are only valid for /tf_static")
    return FrameStamp(sec=sec, nsec=nsec)
