"""采集核心模块：单摄像头不可变快照的 ROS 无关实现。

本模块实现"订阅回调内确定性采样 + 事务暂存 + 原子发布"的采集流水线：

  - StampSampler 以 Image.header.stamp 为帧身份做确定性采样：首个有效
    stamp 被选中，之后每到达 sample_period 就选中下一帧；到达时刻的
    墙上时间永远不会成为帧身份。
  - ImageCodec 在订阅回调内把图像转为 RGB PNG 字节，写入监督进程拥有
    的隐藏暂存目录。
  - SnapshotWriter 在目标结束（帧数达标或超时）时封存 frames.csv 与
    snapshot.json，并通过 ResultTransaction 的原子 rename 线性化点发布。

The sampler is deterministic on ``Image.header.stamp`` inside the image
callback: the first valid stamp is selected and each subsequent frame is
selected when its stamp reaches the previous selection plus the configured
sample period.  Arrival timestamps are never used as frame identity.  Images
are converted to RGB PNG while still inside the subscription callback and
written into a supervisor-owned hidden staging directory; ``frames.csv`` and
``snapshot.json`` are sealed only when the goal finishes and the whole staging
directory is published through the existing ``ResultTransaction`` atomic
rename linearization point.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import tempfile
import threading
import time
from typing import Any, Deque

import cv2
import numpy as np

from .preparation import INTERNAL_ERROR, INVALID_GOAL, IO_ERROR
from .odometry_interpolation import interpolate_payload


CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
from .result_transaction import (
    ResultTransaction,
    ResultTransactionError,
    allocate_result_transaction,
)
from .snapshot_validator import (
    MAX_TF_ENTRIES,
    SNAPSHOT_SCHEMA,
    SNAPSHOT_VERSION,
    FrameStamp,
    SnapshotSafetyLimits,
)

# scene_id / camera_name / 目录 token 的安全标识符正则
_SCENE_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?\Z")
_CAMERA_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_CAMERA_FRAME_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_/.-]{0,126}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
# 支持的 ROS 图像编码（cv2 以 BGR 语义编码 PNG）
_SUPPORTED_ENCODINGS = ("bgr8", "rgb8")
# frames.csv 的列定义（与快照 v2/v3 schema 对应）
_FRAME_COLUMNS = (
    "frame_index",
    "image_path",
    "stamp_sec",
    "stamp_nsec",
    "pose_valid",
    "pose_delta_ms",
    "odometry_gap_ms",
)
# v4 frames.csv 在 v2/v3 列之后增加完整辅助数据文件与关联时间差
_FRAME_COLUMNS_V4 = _FRAME_COLUMNS + (
    "auxiliary_path",
    "camera_info_delta_ms",
    "odometry_delta_ms",
    "depth_path",
    "depth_delta_ms",
)
# 深度原始字节的元数据安全编码（ROS sensor_msgs/Image.encoding 值）
_DEPTH_ENCODING = re.compile(r"[A-Za-z0-9_]{1,32}\Z")


class CaptureConfigError(ValueError):
    """采集配置非法：必须在构造前失败（fail-fast）。"""


@dataclass(frozen=True)
class CaptureConfig:
    """显式采集配置；不含任何内嵌默认值。

    所有路径/话题/阈值都在 __post_init__ 中 preflight 校验：
    snapshot_root 必须是规范化的真实目录（拒绝符号链接遍历），
    话题必须是绝对 ROS 话题名，帧率/帧数/姿态阈值均有严格边界。
    """

    snapshot_root: str
    image_topic: str
    camera_name: str
    camera_role: str
    camera_frame_id: str
    image_encoding: str
    odometry_topic: str | None
    camera_info_topic: str | None
    depth_topic: str | None
    tf_topic: str | None
    tf_static_topic: str | None
    save_sensor_depth: bool
    body_frame_id: str | None
    tf_relevant_frames: tuple[str, ...]
    camera_info_max_delta_ms: float
    depth_max_delta_ms: float
    max_auxiliary_samples: int
    max_tf_samples: int
    max_tf_bytes: int
    source_timeout_sec: float
    max_window_gap_sec: float
    min_sample_fps: float
    max_sample_fps: float
    max_goal_frames: int
    max_goal_duration_sec: float
    min_captured_frames: int
    max_image_width: int
    max_image_height: int
    pose_max_delta_ms: float
    pose_gap_max_ms: float
    max_sample_interval_overshoot_ms: float
    snapshot_limits: SnapshotSafetyLimits
    odometry_association: str = "previous"
    pose_interpolation_wait_ms: float = 500.0
    pose_time_offset_ms: float = 0.0
    # Motion keyframes (interpolate association only): a sampled frame is written
    # only after the body moved >= keyframe_min_translation_m or turned
    # >= keyframe_min_rotation_deg since the last written frame, or after
    # keyframe_max_interval_sec. Both thresholds 0 disables the gate.
    keyframe_min_translation_m: float = 0.0
    keyframe_min_rotation_deg: float = 0.0
    keyframe_max_interval_sec: float = 2.0
    handoff_buffer_frames: int = 0
    handoff_buffer_max_bytes: int = 128 * 1024 * 1024

    def __post_init__(self) -> None:
        try:
            self.preflight()
        except CaptureConfigError:
            raise
        except Exception as error:
            raise CaptureConfigError(
                f"capture preflight failed: {type(error).__name__}: {error}"
            ) from error

    @property
    def keyframe_gating(self) -> bool:
        return self.keyframe_min_translation_m > 0 or self.keyframe_min_rotation_deg > 0

    def max_frame_interval_sec(self, sample_fps: float) -> float:
        """Longest expected gap between written frames for a goal at sample_fps."""
        period = 1.0 / sample_fps
        return max(period, self.keyframe_max_interval_sec) if self.keyframe_gating else period

    def preflight(self) -> None:
        """对全部显式配置做一次完整校验。"""
        if (isinstance(self.pose_time_offset_ms, bool) or not isinstance(self.pose_time_offset_ms, (int, float))
                or not math.isfinite(self.pose_time_offset_ms) or abs(self.pose_time_offset_ms) > 1000):
            raise CaptureConfigError("pose_time_offset_ms must be finite and within +-1000")
        if self.pose_time_offset_ms and self.odometry_association != "interpolate":
            raise CaptureConfigError("pose offset requires interpolated odometry")
        if type(self.handoff_buffer_frames) is not int or self.handoff_buffer_frames < 0:
            raise CaptureConfigError("handoff_buffer_frames must be a nonnegative integer")
        if type(self.handoff_buffer_max_bytes) is not int or self.handoff_buffer_max_bytes <= 0:
            raise CaptureConfigError("handoff_buffer_max_bytes must be a positive integer")
        if self.handoff_buffer_frames and self.odometry_association != "interpolate":
            raise CaptureConfigError("handoff buffering requires interpolated odometry")
        if self.odometry_association not in ("previous", "interpolate"):
            raise CaptureConfigError("odometry_association must be previous or interpolate")
        if (
            isinstance(self.pose_interpolation_wait_ms, bool)
            or not isinstance(self.pose_interpolation_wait_ms, (int, float))
            or not math.isfinite(self.pose_interpolation_wait_ms)
            or self.pose_interpolation_wait_ms <= 0
        ):
            raise CaptureConfigError("pose_interpolation_wait_ms must be positive and finite")
        for name in ("keyframe_min_translation_m", "keyframe_min_rotation_deg", "keyframe_max_interval_sec"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise CaptureConfigError(f"{name} must be finite and nonnegative")
        if self.keyframe_gating:
            if self.odometry_association != "interpolate":
                raise CaptureConfigError("motion keyframes need odometry_association=interpolate")
            if self.keyframe_max_interval_sec <= 0:
                raise CaptureConfigError("keyframe_max_interval_sec must be positive when keyframes are enabled")
        object.__setattr__(
            self,
            "snapshot_root",
            _canonical_directory(self.snapshot_root, "snapshot_root"),
        )
        if (
            not isinstance(self.image_topic, str)
            or not self.image_topic.startswith("/")
            or "\x00" in self.image_topic
        ):
            raise CaptureConfigError("image_topic must be an absolute ROS topic")
        if (
            not isinstance(self.camera_name, str)
            or not _CAMERA_NAME.fullmatch(self.camera_name)
        ):
            raise CaptureConfigError("camera_name must be a safe identifier")
        if (
            not isinstance(self.camera_role, str)
            or not _CAMERA_NAME.fullmatch(self.camera_role)
            or len(self.camera_role.encode("ascii")) > 64
        ):
            raise CaptureConfigError("camera_role must be a safe identifier")
        if (
            not isinstance(self.camera_frame_id, str)
            or not _CAMERA_FRAME_ID.fullmatch(self.camera_frame_id)
        ):
            raise CaptureConfigError("camera_frame_id must be a safe frame id")
        if self.image_encoding not in _SUPPORTED_ENCODINGS:
            raise CaptureConfigError(
                f"image_encoding must be one of {_SUPPORTED_ENCODINGS!r}"
            )
        for label, value in (
            ("odometry_topic", self.odometry_topic),
            ("camera_info_topic", self.camera_info_topic),
            ("depth_topic", self.depth_topic),
            ("tf_topic", self.tf_topic),
            ("tf_static_topic", self.tf_static_topic),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.startswith("/") or "\x00" in value
            ):
                raise CaptureConfigError(
                    f"{label} must be None or an absolute ROS topic"
                )
        if type(self.save_sensor_depth) is not bool:
            raise CaptureConfigError("save_sensor_depth must be a bool")
        if self.save_sensor_depth and self.depth_topic is None:
            raise CaptureConfigError(
                "save_sensor_depth=true requires depth_topic to be configured"
            )
        if self.body_frame_id is not None and (
            not isinstance(self.body_frame_id, str)
            or not _CAMERA_FRAME_ID.fullmatch(self.body_frame_id)
        ):
            raise CaptureConfigError("body_frame_id must be None or a safe frame id")
        if not isinstance(self.tf_relevant_frames, tuple) or any(
            not isinstance(frame, str) or not _CAMERA_FRAME_ID.fullmatch(frame)
            for frame in self.tf_relevant_frames
        ):
            raise CaptureConfigError(
                "tf_relevant_frames must be a tuple of safe frame ids"
            )
        if len(self.tf_relevant_frames) > 64:
            raise CaptureConfigError("tf_relevant_frames must not exceed 64 entries")
        for name, value in (
            ("min_sample_fps", self.min_sample_fps),
            ("max_sample_fps", self.max_sample_fps),
            ("max_goal_duration_sec", self.max_goal_duration_sec),
            ("pose_max_delta_ms", self.pose_max_delta_ms),
            ("pose_gap_max_ms", self.pose_gap_max_ms),
            ("max_sample_interval_overshoot_ms", self.max_sample_interval_overshoot_ms),
            ("camera_info_max_delta_ms", self.camera_info_max_delta_ms),
            ("depth_max_delta_ms", self.depth_max_delta_ms),
            ("source_timeout_sec", self.source_timeout_sec),
            ("max_window_gap_sec", self.max_window_gap_sec),
        ):
            if not _finite_number(value):
                raise CaptureConfigError(f"{name} must be a finite number")
        if self.source_timeout_sec <= 0 or self.max_window_gap_sec <= 0:
            raise CaptureConfigError(
                "source_timeout_sec and max_window_gap_sec must be positive"
            )
        if self.camera_info_max_delta_ms < 0 or self.depth_max_delta_ms < 0:
            raise CaptureConfigError("auxiliary delta thresholds must be nonnegative")
        if self.min_sample_fps <= 0 or self.max_sample_fps < self.min_sample_fps:
            raise CaptureConfigError(
                "sample fps range must be positive and ordered"
            )
        if self.max_goal_duration_sec <= 0:
            raise CaptureConfigError("max_goal_duration_sec must be positive")
        if self.pose_max_delta_ms < 0 or self.pose_gap_max_ms < 0:
            raise CaptureConfigError("pose thresholds must be nonnegative")
        if self.max_sample_interval_overshoot_ms < 0:
            raise CaptureConfigError(
                "max_sample_interval_overshoot_ms must be nonnegative"
            )
        if self.pose_gap_max_ms < self.pose_max_delta_ms:
            raise CaptureConfigError(
                "pose_gap_max_ms must be at least pose_max_delta_ms"
            )
        for name, value in (
            ("max_goal_frames", self.max_goal_frames),
            ("min_captured_frames", self.min_captured_frames),
            ("max_image_width", self.max_image_width),
            ("max_image_height", self.max_image_height),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CaptureConfigError(f"{name} must be a positive integer")
        if self.min_captured_frames > self.max_goal_frames:
            raise CaptureConfigError(
                "min_captured_frames must not exceed max_goal_frames"
            )
        for name, value in (
            ("max_auxiliary_samples", self.max_auxiliary_samples),
            ("max_tf_samples", self.max_tf_samples),
            ("max_tf_bytes", self.max_tf_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CaptureConfigError(f"{name} must be a positive integer")
        if self.max_tf_samples > MAX_TF_ENTRIES:
            raise CaptureConfigError(
                f"max_tf_samples must not exceed {MAX_TF_ENTRIES}"
            )
        if not isinstance(self.snapshot_limits, SnapshotSafetyLimits):
            raise CaptureConfigError(
                "snapshot_limits must be SnapshotSafetyLimits"
            )
        try:
            self.snapshot_limits.validate()
        except Exception as error:
            raise CaptureConfigError(f"snapshot_limits invalid: {error}") from error
        if self.snapshot_limits.max_frame_count < self.max_goal_frames:
            raise CaptureConfigError(
                "snapshot_limits.max_frame_count must cover max_goal_frames"
            )
        if self.max_tf_bytes > self.snapshot_limits.max_file_bytes:
            raise CaptureConfigError(
                "max_tf_bytes must not exceed snapshot_limits.max_file_bytes"
            )


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
class CaptureGoalParameters:
    """一次采集目标的解析后参数（已通过配置边界校验）。

    Attributes:
        scene_id: 场景标识符
        sample_fps: 采样帧率（目标允许范围内）
        max_frames: 最大帧数上限
        max_duration_sec: 最长采集时长（秒）
    """

    scene_id: str
    sample_fps: float
    max_frames: int
    max_duration_sec: float

    @property
    def sample_period_ns(self) -> int:
        """由采样帧率换算的采样周期（纳秒）。"""
        return int(round(1_000_000_000.0 / self.sample_fps))


@dataclass(frozen=True)
class CapturedFrame:
    """一帧已写入暂存目录的采集结果元数据。"""

    index: int
    stamp: FrameStamp
    image_relative_path: str
    pose_valid: bool
    pose_delta_ms: float | None
    odometry_gap_ms: float | None


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


@dataclass(frozen=True)
class SealedSnapshot:
    """封存完成的快照摘要（供 Action 结果字段填充）。"""

    scene_id: str
    snapshot_uri: str
    manifest_uri: str
    manifest_sha256: str
    frame_count: int
    first_stamp: FrameStamp
    last_stamp: FrameStamp
    camera_role: str
    camera_frame_id: str
    has_camera_info: bool
    has_odometry: bool
    has_depth: bool = False
    has_tf: bool = False
    camera_info_digest: str | None = None
    auxiliary_summary: dict | None = None


def validate_capture_goal(goal: Any, config: CaptureConfig) -> CaptureGoalParameters:
    """校验采集目标并解析为类型化参数（所有字段都有配置边界）。

    Args:
        goal: CaptureSegmentGoal 消息（scene_id / sample_fps / max_frames / max_duration）
        config: 采集配置（提供帧率范围、帧数/时长上限）

    Returns:
        解析后的 CaptureGoalParameters

    Raises:
        CaptureError: 任一字段缺失、类型非法或超出配置边界时抛 INVALID_GOAL
    """
    if not isinstance(goal, object):
        raise CaptureError(INVALID_GOAL, "goal is missing")
    try:
        scene_id = goal.scene_id
        sample_fps = goal.sample_fps
        max_frames = goal.max_frames
        max_duration = goal.max_duration
    except AttributeError as error:
        raise CaptureError(INVALID_GOAL, "goal is missing capture fields") from error
    if (
        not isinstance(scene_id, str)
        or _SCENE_ID.fullmatch(scene_id) is None
        or len(scene_id.encode("ascii")) > 128
    ):
        raise CaptureError(INVALID_GOAL, "scene_id is unsafe")
    if not _finite_number(sample_fps):
        raise CaptureError(INVALID_GOAL, "sample_fps must be a finite number")
    sample_fps = float(sample_fps)
    if sample_fps < config.min_sample_fps or sample_fps > config.max_sample_fps:
        raise CaptureError(
            INVALID_GOAL,
            "sample_fps is outside the configured range "
            f"[{config.min_sample_fps}, {config.max_sample_fps}]",
        )
    if (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames <= 0
    ):
        raise CaptureError(INVALID_GOAL, "max_frames must be a positive integer")
    if max_frames > config.max_goal_frames:
        raise CaptureError(
            INVALID_GOAL, f"max_frames exceeds configured cap {config.max_goal_frames}"
        )
    try:
        max_duration_sec = float(max_duration.to_sec())
    except AttributeError as error:
        raise CaptureError(INVALID_GOAL, "max_duration is not a ROS duration") from error
    if not math.isfinite(max_duration_sec) or max_duration_sec <= 0:
        raise CaptureError(INVALID_GOAL, "max_duration must be positive and finite")
    if max_duration_sec > config.max_goal_duration_sec:
        raise CaptureError(
            INVALID_GOAL,
            f"max_duration exceeds configured cap {config.max_goal_duration_sec}",
        )
    set_save_sensor_depth = getattr(goal, "set_save_sensor_depth", False)
    if type(set_save_sensor_depth) is not bool:
        raise CaptureError(INVALID_GOAL, "set_save_sensor_depth must be a bool")
    if set_save_sensor_depth:
        save_sensor_depth = getattr(goal, "save_sensor_depth", False)
        if type(save_sensor_depth) is not bool:
            raise CaptureError(INVALID_GOAL, "save_sensor_depth must be a bool")
        if save_sensor_depth and config.depth_topic is None:
            raise CaptureError(
                INVALID_GOAL,
                "save_sensor_depth=true requires depth_topic to be configured",
            )
    return CaptureGoalParameters(
        scene_id=scene_id,
        sample_fps=sample_fps,
        max_frames=max_frames,
        max_duration_sec=max_duration_sec,
    )


class StampSampler:
    """基于 ROS header stamp 的确定性帧采样器。

    首个有效 stamp 必选；之后当当前 stamp 到达上一选中 stamp + 采样周期
    时选中下一帧。零/重复 stamp 被拒绝并分类计数（不推进选帧时钟）；
    回退 stamp 视为流契约违反并抛 `CaptureStreamError`（由上层放弃当前
    未封存窗口并从新基准开始）。
    """

    def __init__(self, goal: CaptureGoalParameters) -> None:
        self._goal = goal
        self._last_seen: tuple[int, int] | None = None
        self._last_selected: tuple[int, int] | None = None
        self._next_selection_ns = 0
        self.zero_stamp_count = 0
        self.duplicate_stamp_count = 0
        self.out_of_order_stamp_count = 0

    def classify(self, stamp: FrameStamp) -> str:
        """对 stamp 做流契约分类，不改变选帧状态。

        Returns:
            "zero" / "duplicate" / "backward" / "ok"
        """
        if stamp.sec == 0 and stamp.nsec == 0:
            return "zero"
        current = (stamp.sec, stamp.nsec)
        if self._last_seen is None:
            return "ok"
        if current == self._last_seen:
            return "duplicate"
        if current < self._last_seen:
            return "backward"
        return "ok"

    def select(self, stamp: FrameStamp) -> bool:
        """决定给定 stamp 的帧是否应被选中。

        零/重复 stamp 拒绝并计数后跳过；回退 stamp 计数后抛
        CaptureStreamError。
        """
        kind = self.classify(stamp)
        if kind == "zero":
            self.zero_stamp_count += 1
            return False
        if kind == "duplicate":
            self.duplicate_stamp_count += 1
            return False
        if kind == "backward":
            self.out_of_order_stamp_count += 1
            raise CaptureStreamError(
                "image header stamps must be strictly increasing during capture"
            )
        current = (stamp.sec, stamp.nsec)
        self._last_seen = current
        if self._last_selected is None:
            self._last_selected = current
            self._next_selection_ns = _stamp_ns(current) + self._goal.sample_period_ns
            return True
        if _stamp_ns(current) < self._next_selection_ns:
            return False
        self._last_selected = current
        self._next_selection_ns = _stamp_ns(current) + self._goal.sample_period_ns
        return True

    @property
    def last_seen(self) -> tuple[int, int] | None:
        """最近一次通过流契约的 stamp（供 source watchdog 使用）。"""
        return self._last_seen

    @property
    def last_selected(self) -> tuple[int, int] | None:
        """最近一次选中的 stamp（供诊断使用）。"""
        return self._last_selected


class OdometryAssociator:
    """里程计关联器：最近前置（nearest-before）姿态关联，带间隙纪元处理。

    维护最多 512 条里程计采样的滑动窗口；当出现 stamp 回退或超过
    pose_gap_max_ms 的间隙时递增"纪元"（epoch），同一纪元内的采样才参与
    关联，从而隔离回退/间隙造成的污染。record() 可携带完整 Odometry
    payload；associate_full() 在姿态有效时返回该 payload，供 v4 快照
    保存逐帧完整里程计数据。

    FAST-LIO 以约 100 Hz 发布 /Odometry，但位姿只在每帧激光（10 Hz）后更新，
    其余消息重复上一位姿、只换时间戳。距上一条保留采样不到 150 ms 的重复
    消息不入窗口，否则插值退化为取最近位姿保持 100 ms；更久的重复（静止或
    里程计停更）照常记录。
    """

    def __init__(self, config: CaptureConfig) -> None:
        self._config = config
        self._samples: Deque[
            tuple[tuple[int, int], int, AuxiliaryOdometry | None]
        ] = deque(maxlen=512)
        self._epoch = 0
        self._last_stamp: tuple[int, int] | None = None
        self.received_count = 0
        self.regression_count = 0
        self.gap_count = 0
        self.repeated_count = 0

    def record(
        self,
        stamp: FrameStamp,
        payload: AuxiliaryOdometry | None = None,
    ) -> None:
        """记录一条里程计采样（检测回退与间隙并切换纪元）。"""
        current = (stamp.sec, stamp.nsec)
        self.received_count += 1
        if payload is not None and self._samples:
            last = self._samples[-1][2]
            if last is not None and last.payload["pose"]["position"] == payload.payload["pose"]["position"] \
                    and last.payload["pose"]["orientation"] == payload.payload["pose"]["orientation"] \
                    and current > self._last_stamp \
                    and _stamp_delta_ms(self._last_stamp, current) < 150.0:
                self.repeated_count += 1
                return
        if self._last_stamp is None:
            self._epoch += 1
        elif current <= self._last_stamp:
            self._epoch += 1
            self.regression_count += 1
        elif _stamp_delta_ms(self._last_stamp, current) > self._config.pose_gap_max_ms:
            self._epoch += 1
            self.gap_count += 1
        self._samples.append((current, self._epoch, payload))
        self._last_stamp = current

    def associate(
        self, image_stamp: FrameStamp
    ) -> tuple[bool, float | None, float | None]:
        """与 associate_full 兼容的轻量接口（不要求完整 payload）。

        该接口保留 v1–v3 时代的历史语义：只要时间关联与间隙满足即
        pose_valid；完整 payload 是否缺失不影响这里的返回值。
        """
        valid, delta_ms, gap_ms, _payload = self._associate(image_stamp, False)
        return valid, delta_ms, gap_ms

    def associate_full(
        self, image_stamp: FrameStamp
    ) -> tuple[bool, float | None, float | None, AuxiliaryOdometry | None]:
        """为图像 stamp 关联最近的（同一纪元、时间 <= 图像）里程计采样。

        Returns:
            (pose_valid, delta_ms, max_gap_ms, payload)：
            pose_valid 为 True 当且仅当关联存在且 delta_ms <= pose_max_delta_ms
            且关联后的采样间最大间隙 <= pose_gap_max_ms 且该采样带有完整
            payload（v4 语义：没有 payload 就不能宣称姿态有效）。
        """
        valid, delta_ms, gap_ms, payload = self._associate(image_stamp, True)
        return valid, delta_ms, gap_ms, payload if valid else None

    def associate_interpolated(self, image_stamp: FrameStamp):
        """Use a same-epoch bracket. Missing brackets remain missing."""
        query_ns = _stamp_ns((image_stamp.sec, image_stamp.nsec)) + int(round(self._config.pose_time_offset_ms * 1e6))
        query_stamp = dict(zip(("sec", "nsec"), divmod(query_ns, 10**9)))
        before = after = None
        for stamp, epoch, payload in self._samples:
            if epoch != self._epoch or payload is None:
                continue
            sample_ns = _stamp_ns(stamp)
            if sample_ns <= query_ns:
                before = payload
            if sample_ns >= query_ns:
                after = payload
                break
        if before is None or after is None:
            return False, None, None, None
        try:
            payload = interpolate_payload(
                before.payload, after.payload,
                query_stamp,
                self._config.pose_gap_max_ms,
            )
        except (ValueError, KeyError, TypeError):
            return False, None, None, None
        if self._config.pose_time_offset_ms:
            payload["interpolation"].update(pose_time_offset_ms=self._config.pose_time_offset_ms,
                pose_time_offset_stage="capture", query_stamp=query_stamp)
            # Association header remains the RGB stamp; actual LIO query is explicit.
            payload["header"]["stamp"] = {"sec": image_stamp.sec, "nsec": image_stamp.nsec}
        odometry = AuxiliaryOdometry(
            stamp=image_stamp, frame_id=before.frame_id,
            child_frame_id=before.child_frame_id, payload=payload,
            byte_size=len(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        )
        return True, 0.0, payload["interpolation"]["bracket_gap_ms"], odometry

    def _associate(
        self, image_stamp: FrameStamp, require_payload: bool
    ) -> tuple[bool, float | None, float | None, AuxiliaryOdometry | None]:
        if not self._samples:
            return False, None, None, None
        current_epoch = self._samples[-1][1]
        image_ns = _stamp_ns((image_stamp.sec, image_stamp.nsec))
        chosen_index = None
        for index in range(len(self._samples) - 1, -1, -1):
            stamp, epoch, _payload = self._samples[index]
            if epoch != current_epoch:
                continue
            if _stamp_ns(stamp) <= image_ns:
                chosen_index = index
                break
        if chosen_index is None:
            return False, None, None, None
        chosen_stamp, _, chosen_payload = self._samples[chosen_index]
        delta_ms = (image_ns - _stamp_ns(chosen_stamp)) / 1_000_000.0
        # 计算关联点到当前窗口末尾之间的最大采样间隙
        max_gap_ms = 0.0
        previous = chosen_stamp
        for index in range(chosen_index + 1, len(self._samples)):
            stamp, _epoch, _payload = self._samples[index]
            gap = _stamp_delta_ms(previous, stamp)
            max_gap_ms = max(max_gap_ms, gap)
            previous = stamp
        valid = (
            delta_ms <= self._config.pose_max_delta_ms
            and max_gap_ms <= self._config.pose_gap_max_ms
        )
        if require_payload and chosen_payload is None:
            valid = False
        return valid, delta_ms, max_gap_ms, chosen_payload


class AuxiliaryAssociator:
    """辅助数据（CameraInfo/深度）的最近前置关联器。

    与里程计不同，辅助源没有 pose 间隙纪元语义：只要求时间不超过图像
    stamp 且时间差不超过配置阈值。异常/缺失只表现为关联失败，绝不阻塞
    RGB 主链。
    """

    def __init__(self, max_delta_ms: float, max_samples: int) -> None:
        self._max_delta_ms = max_delta_ms
        self._samples: Deque[
            tuple[tuple[int, int], AuxiliaryCameraInfo | AuxiliaryDepth]
        ] = deque(maxlen=max_samples)
        self.received_count = 0

    def record(self, stamp: FrameStamp, payload) -> None:
        """记录一条辅助采样（丢弃最旧的超出窗口样本由 deque 自动完成）。"""
        self.received_count += 1
        self._samples.append(((stamp.sec, stamp.nsec), payload))

    def associate(
        self, image_stamp: FrameStamp
    ) -> tuple[AuxiliaryCameraInfo | AuxiliaryDepth | None, float | None]:
        image_ns = _stamp_ns((image_stamp.sec, image_stamp.nsec))
        for index in range(len(self._samples) - 1, -1, -1):
            stamp, payload = self._samples[index]
            stamp_ns = _stamp_ns(stamp)
            if stamp_ns > image_ns:
                continue
            delta_ms = (image_ns - stamp_ns) / 1_000_000.0
            if delta_ms > self._max_delta_ms:
                return None, None
            return payload, delta_ms
        return None, None


class ImageCodec:
    """解码配置的 ROS 图像编码并输出 RGB PNG 字节。"""

    def __init__(self, config: CaptureConfig) -> None:
        self._config = config

    def to_rgb_png(self, message: Any) -> bytes:
        return self.encode_png(self.check(message), message.encoding)

    def check(self, message: Any):
        """校验 ROS Image 消息并返回 HxWx3 uint8 视图（不复制、不编码）。

        Raises:
            CaptureStreamError: 编码不匹配、尺寸越界、step 带填充或负载长度不一致
        """
        if getattr(message, "is_bigendian", 0) != 0:
            raise CaptureStreamError("big-endian images are not supported")
        if message.encoding != self._config.image_encoding:
            raise CaptureStreamError(
                f"image encoding {message.encoding!r} does not match configured "
                f"{self._config.image_encoding!r}"
            )
        width = int(message.width)
        height = int(message.height)
        step = int(message.step)
        if (
            width <= 0
            or height <= 0
            or width > self._config.max_image_width
            or height > self._config.max_image_height
        ):
            raise CaptureStreamError("image dimensions are outside configured bounds")
        channels = 3
        if step != width * channels:
            raise CaptureStreamError("image step must be width*3 without padding")
        data = message.data
        try:
            data_length = len(data)
        except TypeError as error:
            raise CaptureStreamError("image payload is not a byte buffer") from error
        if data_length != step * height:
            raise CaptureStreamError("image payload length does not match step*height")
        return np.frombuffer(data, dtype=np.uint8).reshape((height, width, channels))

    def encode_png(self, array, encoding: str) -> bytes:
        if encoding == "rgb8":
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(".png", array)
        if not success:
            raise CaptureStreamError("image PNG encoding failed")
        return bytes(encoded.tobytes())


class SnapshotWriter:
    """在事务暂存目录内写帧并封存不可变快照。

    帧 PNG 直接写入暂存目录；seal() 时生成 frames.csv、snapshot.json 与
    files_sha256.csv 清单，全部满足 snapshot_limits 的字节/条目上限。

    v4 契约在每帧 PNG 之外落盘完整辅助数据：每帧一个 auxiliary/frame_*.json
    （完整 CameraInfo + Odometry payload 与关联时间差）、可选的
    depth/frame_*.bin 原始传感器深度、以及窗口级 tf.json。辅助数据遵循
    “尽量保存，但不阻塞 RGB”：超预算/超条目时逐项丢弃并计数，绝不让
    辅助源造成整窗失败。
    """

    def __init__(
        self,
        config: CaptureConfig,
        goal: CaptureGoalParameters,
        transaction: ResultTransaction,
    ) -> None:
        self._config = config
        self._goal = goal
        self._transaction = transaction
        self._frames: list[CapturedFrame] = []
        self._auxiliary: list[AuxiliaryFrameData] = []
        self._aux_file_count = 0
        self._tf_entries: list[dict] = []
        self._tf_bytes = 0
        self.tf_dropped = 0
        self.auxiliary_dropped = 0
        self.camera_info_samples = 0
        self.odometry_samples = 0
        self.depth_samples = 0
        self.auxiliary_errors = 0
        self.camera_info_matched_frames = 0
        self.odometry_matched_frames = 0
        self.depth_matched_frames = 0
        self._bytes_written = 0
        self._metadata_reserve = (
            3 * config.snapshot_limits.max_metadata_bytes + config.max_tf_bytes
        )

    @property
    def staging_path(self) -> Path:
        """事务暂存目录（worker/帧数据的唯一写入目标）。"""
        return self._transaction.staging_path

    @property
    def frame_count(self) -> int:
        """已写入的帧数。"""
        return len(self._frames)

    @property
    def has_camera_info(self) -> bool:
        """是否存在任何一帧保存了完整 CameraInfo payload。"""
        return self.camera_info_matched_frames > 0

    @property
    def has_odometry(self) -> bool:
        """是否存在任何一帧保存了完整 Odometry payload。"""
        return self.odometry_matched_frames > 0

    @property
    def has_depth(self) -> bool:
        return self.depth_matched_frames > 0

    @property
    def has_tf(self) -> bool:
        return bool(self._tf_entries)

    def _can_reserve(self, size: int, entries: int) -> bool:
        """判断辅助数据在总字节与 manifest 条目预算内是否仍可落盘。

        RGB 图像不属于本预算：图像超限是真正的采集失败。这里只保证辅助
        数据永远不会把整窗推到 total/entry 上限之外。
        """
        limits = self._config.snapshot_limits
        if self._bytes_written + size > limits.max_total_bytes - self._metadata_reserve:
            return False
        used_entries = (
            len(self._frames)
            + 3
            + (1 if self._tf_entries else 0)
            + self._aux_file_count
            + entries
        )
        return used_entries < limits.max_manifest_entries

    def _drop_auxiliary(self) -> None:
        self.auxiliary_dropped += 1

    def _canonical_json(self, value: Any) -> bytes:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    def add_transform(self, entry: dict) -> bool:
        """记录一条与相机/机体相关的 TF 变换。

        超过 max_tf_samples 或 max_tf_bytes 时拒绝并计数，不抛异常。
        """
        entry_bytes = len(self._canonical_json(entry))
        if entry_bytes > self._config.max_tf_bytes:
            self.tf_dropped += 1
            return False
        if len(self._tf_entries) >= self._config.max_tf_samples:
            self.tf_dropped += 1
            return False
        if self._tf_bytes + entry_bytes > self._config.max_tf_bytes:
            self.tf_dropped += 1
            return False
        self._tf_entries.append(entry)
        self._tf_bytes += entry_bytes
        return True

    def write_frame(
        self,
        stamp: FrameStamp,
        image_bytes: bytes,
        *,
        pose_valid: bool,
        pose_delta_ms: float | None,
        odometry_gap_ms: float | None,
        auxiliary: AuxiliaryFrameData | None = None,
    ) -> CapturedFrame:
        """原子写入一帧 PNG、辅助 JSON/深度并登记帧元数据。"""
        index = len(self._frames)
        relative = f"images/frame_{index:06d}.png"
        path = self.staging_path.joinpath(*Path(relative).parts)
        _write_file_atomically(
            path,
            image_bytes,
            self._config.snapshot_limits.max_file_bytes,
            sync=False,
        )
        self._bytes_written += len(image_bytes)
        frame = CapturedFrame(
            index=index,
            stamp=stamp,
            image_relative_path=relative,
            pose_valid=pose_valid,
            pose_delta_ms=pose_delta_ms,
            odometry_gap_ms=odometry_gap_ms,
        )
        self._frames.append(frame)
        auxiliary = auxiliary or AuxiliaryFrameData()
        self._store_auxiliary(index, auxiliary)
        self._auxiliary.append(auxiliary)
        return frame

    def _store_auxiliary(self, index: int, auxiliary: AuxiliaryFrameData) -> None:
        """按预算写入一帧的辅助 JSON 与原始深度文件（超预算逐项丢弃）。

        写入顺序保证一致性：先写 auxiliary JSON（其中声明深度文件），再写
        depth bin；任何一步失败都会把这一帧的辅助数据整体降级为不落盘，
        staging 中绝不残留未被 frames.csv 引用的孤儿文件。
        """
        depth_relative = f"depth/frame_{index:06d}.bin"
        depth_fits = False
        if auxiliary.depth is not None:
            size = len(auxiliary.depth.data)
            depth_fits = (
                size <= self._config.snapshot_limits.max_file_bytes
                and self._can_reserve(size, 1)
            )

        def build_payload(with_depth: bool) -> dict:
            return {
                "frame_index": index,
                "camera_info": (
                    auxiliary.camera_info.payload
                    if auxiliary.camera_info is not None
                    else None
                ),
                "camera_info_delta_ms": auxiliary.camera_info_delta_ms,
                "odometry": (
                    auxiliary.odometry.payload
                    if auxiliary.odometry is not None
                    else None
                ),
                "odometry_delta_ms": auxiliary.odometry_delta_ms,
                "depth": (
                    {
                        "stamp": {
                            "sec": auxiliary.depth.stamp.sec,
                            "nsec": auxiliary.depth.stamp.nsec,
                        },
                        "delta_ms": auxiliary.depth_delta_ms,
                        "frame_id": auxiliary.depth.frame_id,
                        "encoding": auxiliary.depth.encoding,
                        "width": auxiliary.depth.width,
                        "height": auxiliary.depth.height,
                        "step": auxiliary.depth.step,
                        "is_bigendian": auxiliary.depth.is_bigendian,
                        "data_file": depth_relative,
                    }
                    if with_depth and auxiliary.depth is not None
                    else None
                ),
            }

        if (
            auxiliary.camera_info is None
            and auxiliary.odometry is None
            and not depth_fits
        ):
            return

        relative = f"auxiliary/frame_{index:06d}.json"
        path = self.staging_path.joinpath(*Path(relative).parts)
        json_bytes = self._canonical_json(build_payload(depth_fits))
        if not (
            len(json_bytes) <= self._config.snapshot_limits.max_file_bytes
            and self._can_reserve(len(json_bytes), 1)
        ):
            self._drop_auxiliary()
            return
        try:
            _write_file_atomically(
                path,
                json_bytes,
                self._config.snapshot_limits.max_file_bytes,
                sync=False,
            )
            self._bytes_written += len(json_bytes)
            self._aux_file_count += 1
        except BaseException:
            self._drop_auxiliary()
            return

        # auxiliary JSON 已成功落盘：CameraInfo/Odometry 匹配帧计数在此定格。
        ci_matched = auxiliary.camera_info is not None
        odom_matched = auxiliary.odometry is not None
        if ci_matched:
            self.camera_info_matched_frames += 1
        if odom_matched:
            self.odometry_matched_frames += 1

        if depth_fits:
            depth_path_obj = self.staging_path.joinpath(*Path(depth_relative).parts)
            try:
                _write_file_atomically(
                    depth_path_obj,
                    auxiliary.depth.data,
                    self._config.snapshot_limits.max_file_bytes,
                    sync=False,
                )
                self._bytes_written += len(auxiliary.depth.data)
                self._aux_file_count += 1
                self.depth_matched_frames += 1
            except BaseException:
                # 深度写失败：把已写 JSON 改写为无深度版本，保持两者一致。
                self._drop_auxiliary()
                try:
                    json_without_depth = self._canonical_json(build_payload(False))
                    _write_file_atomically(
                        path,
                        json_without_depth,
                        self._config.snapshot_limits.max_file_bytes,
                        sync=False,
                    )
                    self._bytes_written += len(json_without_depth) - len(json_bytes)
                except BaseException:
                    # 连改写都失败：本帧辅助整体放弃，删除已写文件。
                    if ci_matched:
                        self.camera_info_matched_frames -= 1
                    if odom_matched:
                        self.odometry_matched_frames -= 1
                    self._remove_aux_files(path, depth_path_obj, len(json_bytes))

    def _remove_aux_files(self, json_path: Path, depth_path: Path, json_bytes: int) -> None:
        """辅助写盘完全失败时清理本帧已写文件（尽力而为，不抛异常）。"""
        for path in (json_path, depth_path):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass
        self._bytes_written = max(0, self._bytes_written - json_bytes)
        self._aux_file_count = max(0, self._aux_file_count - 1)

    def _aux_path_for(self, index: int) -> str | None:
        """返回已成功落盘的 auxiliary JSON 相对路径（不存在返回 None）。"""
        relative = f"auxiliary/frame_{index:06d}.json"
        path = self.staging_path.joinpath(*Path(relative).parts)
        if path.is_file():
            return relative
        return None

    def _camera_info_digest(self) -> str | None:
        """由全部已保存 CameraInfo payload 计算确定性 SHA-256。

        只有保存成功（auxiliary JSON 落盘）的 payload 参与摘要，顺序为帧序。
        """
        if self.camera_info_matched_frames == 0:
            return None
        digest = hashlib.sha256()
        counted = 0
        for index, auxiliary in enumerate(self._auxiliary):
            if auxiliary.camera_info is None or self._aux_path_for(index) is None:
                continue
            payload = json.dumps(
                auxiliary.camera_info.payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(payload)
            digest.update(b"\n")
            counted += 1
        if counted != self.camera_info_matched_frames:
            return None
        return digest.hexdigest()

    def seal(self) -> SealedSnapshot:
        """封存快照：写 frames.csv / snapshot.json / tf.json / 清单并返回摘要。

        Raises:
            CaptureError: 帧数不足本次 Goal 的有效下限时抛 CAMERA_UNAVAILABLE
        """
        config = self._config
        minimum_frames = min(config.min_captured_frames, self._goal.max_frames)
        if len(self._frames) < minimum_frames:
            raise CaptureError(
                CAMERA_UNAVAILABLE,
                "capture ended before effective min_captured_frames was reached "
                f"({len(self._frames)} < {minimum_frames})",
            )
        first = self._frames[0].stamp
        last = self._frames[-1].stamp
        camera_info_digest = self._camera_info_digest()
        auxiliary_summary = {
            "camera_info_samples": self.camera_info_samples,
            "odometry_samples": self.odometry_samples,
            "depth_samples": self.depth_samples,
            "tf_samples": len(self._tf_entries),
            "camera_info_matched_frames": self.camera_info_matched_frames,
            "odometry_matched_frames": self.odometry_matched_frames,
            "depth_matched_frames": self.depth_matched_frames,
            "tf_dropped": self.tf_dropped,
            "auxiliary_errors": self.auxiliary_errors,
        }
        snapshot_metadata = {
            "schema": SNAPSHOT_SCHEMA,
            "version": SNAPSHOT_VERSION,
            "status": "PASS",
            "scene_id": self._goal.scene_id,
            "frame_count": len(self._frames),
            "camera_name": config.camera_name,
            "image_topic": config.image_topic,
            "camera_role": config.camera_role,
            "camera_frame_id": config.camera_frame_id,
            "first_stamp_sec": first.sec,
            "first_stamp_nsec": first.nsec,
            "last_stamp_sec": last.sec,
            "last_stamp_nsec": last.nsec,
            "has_camera_info": self.has_camera_info,
            "has_odometry": self.has_odometry,
            # v4 辅助数据契约
            "camera_info_topic": config.camera_info_topic,
            "odometry_topic": config.odometry_topic,
            "depth_topic": config.depth_topic,
            "tf_topic": config.tf_topic,
            "tf_static_topic": config.tf_static_topic,
            "save_sensor_depth": config.save_sensor_depth,
            "body_frame_id": config.body_frame_id,
            "has_tf": self.has_tf,
            "has_depth": self.has_depth,
            "camera_info_digest": camera_info_digest,
            "auxiliary_summary": auxiliary_summary,
        }
        frame_rows = []
        for index, frame in enumerate(self._frames):
            auxiliary = (
                self._auxiliary[index] if index < len(self._auxiliary) else None
            )
            aux_path = self._aux_path_for(index)
            depth_path = self._depth_path_for(index)
            row = {
                "frame_index": str(frame.index),
                "image_path": frame.image_relative_path,
                "stamp_sec": str(frame.stamp.sec),
                "stamp_nsec": str(frame.stamp.nsec),
                "pose_valid": "true" if frame.pose_valid else "false",
                "pose_delta_ms": _format_optional_ms(frame.pose_delta_ms),
                "odometry_gap_ms": _format_optional_ms(frame.odometry_gap_ms),
            }
            if aux_path is not None:
                row.update(
                    {
                        "auxiliary_path": aux_path,
                        "camera_info_delta_ms": _format_optional_ms(
                            auxiliary.camera_info_delta_ms
                        ),
                        "odometry_delta_ms": _format_optional_ms(
                            auxiliary.odometry_delta_ms
                        ),
                        "depth_path": depth_path or "",
                        "depth_delta_ms": (
                            _format_optional_ms(auxiliary.depth_delta_ms)
                            if depth_path is not None
                            else "-1"
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "auxiliary_path": "",
                        "camera_info_delta_ms": "-1",
                        "odometry_delta_ms": "-1",
                        "depth_path": "",
                        "depth_delta_ms": "-1",
                    }
                )
            frame_rows.append(row)
        _write_csv(
            self.staging_path / "frames.csv",
            _FRAME_COLUMNS_V4,
            frame_rows,
            config.snapshot_limits.max_metadata_bytes,
        )
        if self._tf_entries:
            tf_bytes = self._canonical_json(self._tf_entries)
            try:
                _write_file_atomically(
                    self.staging_path / "tf.json",
                    tf_bytes,
                    config.snapshot_limits.max_file_bytes,
                )
                self._bytes_written += len(tf_bytes)
            except BaseException:
                # TF 是辅助数据：写不下只能丢弃，不能让整窗失败。
                self.tf_dropped += len(self._tf_entries)
                self._tf_entries = []
                auxiliary_summary["tf_samples"] = 0
                auxiliary_summary["tf_dropped"] = self.tf_dropped
                snapshot_metadata["has_tf"] = False
        snapshot_bytes = (
            json.dumps(
                snapshot_metadata,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        _write_file_atomically(
            self.staging_path / "snapshot.json",
            snapshot_bytes,
            config.snapshot_limits.max_metadata_bytes,
        )
        # 帧/辅助文件写盘时 sync=False（避免逐帧 fsync 阻塞采集）；这里在
        # 封窗前一次性补 fsync，保证清单引用到的每个文件内容也已持久化。
        _fsync_staging_files(self.staging_path)
        _write_manifest(self.staging_path, config.snapshot_limits)
        _fsync_directory(self.staging_path)
        manifest_path = self.staging_path / "files_sha256.csv"
        manifest_sha256 = _sha256_file(manifest_path)
        snapshot_uri = self._transaction.final_path.as_uri()
        manifest_uri = (
            self._transaction.final_path / "files_sha256.csv"
        ).as_uri()
        return SealedSnapshot(
            scene_id=self._goal.scene_id,
            snapshot_uri=snapshot_uri,
            manifest_uri=manifest_uri,
            manifest_sha256=manifest_sha256,
            frame_count=len(self._frames),
            first_stamp=first,
            last_stamp=last,
            camera_role=config.camera_role,
            camera_frame_id=config.camera_frame_id,
            has_camera_info=self.has_camera_info,
            has_odometry=self.has_odometry,
            has_depth=self.has_depth,
            has_tf=bool(self._tf_entries),
            camera_info_digest=camera_info_digest,
            auxiliary_summary=auxiliary_summary,
        )

    def _depth_path_for(self, index: int) -> str | None:
        """返回已成功落盘的深度文件相对路径（不存在返回 None）。"""
        auxiliary = self._auxiliary[index] if index < len(self._auxiliary) else None
        if auxiliary is None or auxiliary.depth is None:
            return None
        relative = f"depth/frame_{index:06d}.bin"
        path = self.staging_path.joinpath(*Path(relative).parts)
        if path.is_file():
            return relative
        return None


class CaptureSession:
    """一次活跃采集目标，绑定其事务与订阅回调。

    所有回调入口（on_image / on_odometry / on_camera_info / check_deadline）
    都持有内部锁，保证采集状态（帧数、错误、封存结果）跨线程一致。
    封存只发生一次：要么帧数达标（_sealed_by_frames），要么超时
    （_sealed_by_deadline），要么失败（error 非空）。
    """

    def __init__(
        self,
        config: CaptureConfig,
        goal: CaptureGoalParameters,
        transaction: ResultTransaction,
        cancellation_event: threading.Event,
        *,
        codec: ImageCodec | None = None,
        sampler: StampSampler | None = None,
        monotonic: Any = time.monotonic,
        feedback_callback: Any = None,
        source_state_callback: Any = None,
        defer_seal: bool = False,
    ) -> None:
        """构造采集会话。

        Args:
            config: 采集配置
            goal: 已校验的采集目标参数
            transaction: 监督进程拥有的结果事务（暂存目录写入目标）
            cancellation_event: 取消事件（Action 服务端注入）
            codec/sampler: 注入的编解码器/采样器（测试注入点）
            monotonic: 单调时钟
            feedback_callback: 每选中一帧后调用的反馈回调
            source_state_callback: source watchdog 状态迁移回调
                （stage: "SOURCE_STALE"/"SOURCE_RECOVERED"，
                 stamp: 最近通过契约的图像 stamp；异常被忽略）
            defer_seal: 帧收齐时只置 done 和 final_stamp_ns，封存（fsync、清单）
                留给调用方在传感器回调线程之外调用 finalize()。封存在磁盘繁忙时
                可能耗时数秒，放在回调里会让图像/里程计队列溢出丢帧。
        """
        self.config = config
        self.goal = goal
        self.transaction = transaction
        self.cancellation_event = cancellation_event
        self.codec = codec or ImageCodec(config)
        self.sampler = sampler or StampSampler(goal)
        self.writer = SnapshotWriter(config, goal, transaction)
        self.odometry = OdometryAssociator(config)
        self.camera_info = AuxiliaryAssociator(
            config.camera_info_max_delta_ms, config.max_auxiliary_samples
        )
        self.depth = AuxiliaryAssociator(
            config.depth_max_delta_ms, config.max_auxiliary_samples
        )
        self._relevant_tf_frames = set(config.tf_relevant_frames)
        self._relevant_tf_frames.add(config.camera_frame_id)
        if config.body_frame_id is not None:
            self._relevant_tf_frames.add(config.body_frame_id)
        self.tf_received = 0
        self.auxiliary_errors = 0
        self.last_auxiliary_error: str | None = None
        self._lock = threading.Lock()
        self._monotonic = monotonic
        self._feedback_callback = feedback_callback
        self.deadline = monotonic() + goal.max_duration_sec
        self.done = threading.Event()
        self.sealed: SealedSnapshot | None = None
        self._defer_seal = defer_seal
        self.frames_complete = False
        # Stamp (ns) of the last written frame once frames are complete; later
        # images belong to the next window.
        self.final_stamp_ns: int | None = None
        self.error: CaptureError | None = None
        self.received_images = 0
        self.skipped_images = 0
        self.superseded_images = 0
        self.paused = False
        self._pause_remaining_sec: float | None = None
        self._sealed_by_frames = False
        self._sealed_by_deadline = False
        self._source_state_callback = source_state_callback
        self.source_stale = False
        self.source_stale_entries = 0
        self.recovered_from_stale = False
        self._last_image_monotonic: float | None = None
        self._started_monotonic = self._monotonic()
        self._stale_remaining_sec: float | None = None
        # Sample RGB immediately. Wait only for its right odometry bracket.
        self._pending_frames = deque()
        self.keyframe_skipped = 0
        self._last_keyframe = None  # (stamp_ns, position, quaternion_xyzw) of the last written posed frame

    def pause(self) -> None:
        """暂停向当前窗口加入帧；保留窗口与 stamp 基准，deadline 同步冻结。"""
        with self._lock:
            if self.done.is_set() or self.paused:
                return
            self._pause_remaining_sec = max(0.0, self.deadline - self._monotonic())
            self.paused = True

    def resume(self) -> None:
        """恢复当前窗口采集，deadline 顺延暂停时长。

        暂停是人工控制而非 source 故障：恢复时同步重置 source watchdog 的
        "最后有效 RGB" 时钟，避免暂停时长被误判成 SOURCE_STALE。
        """
        with self._lock:
            if self.done.is_set() or not self.paused:
                return
            remaining = self._pause_remaining_sec
            self.paused = False
            self._pause_remaining_sec = None
            if remaining is not None:
                self.deadline = self._monotonic() + remaining
            self._last_image_monotonic = self._monotonic()

    def is_paused(self) -> bool:
        with self._lock:
            return self.paused

    def on_image(self, message: Any) -> None:
        """图像订阅回调：采样→姿态关联→编码→写帧→（可能）封存。"""
        with self._lock:
            self._on_image_locked(message)

    def _on_image_locked(self, message: Any) -> None:
        if self.done.is_set():
            return
        if self.cancellation_event.is_set():
            self.done.set()
            return
        self.received_images += 1
        if self.paused:
            # 暂停只停止加帧；窗口、stamp 基准和 deadline 都保留。
            return
        try:
            if self.config.odometry_association == "interpolate":
                self._flush_pending_frames_locked()
                if (
                    self.done.is_set() or self._sealed_by_deadline
                    or self.writer.frame_count + len(self._pending_frames) >= self.goal.max_frames
                ):
                    return
            stamp = _watchdog_image_stamp(message)
            # frame 身份必须与 profile 配置一致，禁止错流混入窗口
            frame_id = _message_frame_id(message)
            if frame_id != self.config.camera_frame_id:
                raise CaptureStreamError(
                    "image header frame_id does not match configured "
                    f"camera frame: expected {self.config.camera_frame_id!r}, "
                    f"got {frame_id!r}"
                )
            # 零/重复 stamp：拒绝并分类计数（由 sampler.select 完成），
            # 不推进选帧时钟、不影响当前窗口。它们也不刷新 source watchdog
            # 的"最后有效 RGB"时钟。
            kind = self.sampler.classify(stamp)
            if kind in ("zero", "duplicate"):
                self.sampler.select(stamp)
                self.skipped_images += 1
                return
            if kind == "backward":
                # 回退属于基准变化：只放弃窗口，绝不能先发布 SOURCE_RECOVERED。
                self.sampler.select(stamp)
            self._last_image_monotonic = self._monotonic()
            last_seen = self.sampler.last_seen
            if last_seen is not None:
                gap_ms = _stamp_delta_ms(last_seen, (stamp.sec, stamp.nsec))
                if gap_ms > self.config.max_window_gap_sec * 1000.0:
                    raise CaptureStreamError(
                        "source gap exceeded configured max_window_gap: "
                        f"{gap_ms} ms > {self.config.max_window_gap_sec * 1000.0} ms"
                    )
            if self.source_stale:
                self._recover_source_locked(stamp)
                self.recovered_from_stale = True
            # 采样器决定是否选中本帧（基于 stamp 的确定性选择）。
            # 未选中的帧绝不能消费 recovered_from_stale：恢复后的第一个
            # 选中帧才允许包含 stale 期间的更大间隔。
            if not self.sampler.select(stamp):
                self.skipped_images += 1
                return
            recovered_from_stale = self.recovered_from_stale
            self.recovered_from_stale = False
            pose_valid, delta_ms, gap_ms, odometry = self.odometry.associate_full(
                stamp
            )
            camera_info, camera_info_delta_ms = self.camera_info.associate(stamp)
            depth, depth_delta_ms = self.depth.associate(stamp)
            auxiliary = AuxiliaryFrameData(
                camera_info=camera_info,
                camera_info_delta_ms=camera_info_delta_ms,
                odometry=odometry,
                odometry_delta_ms=delta_ms if odometry is not None else None,
                depth=depth,
                depth_delta_ms=depth_delta_ms,
            )
            if self.config.odometry_association == "interpolate":
                # Keep the message; PNG encoding (~45 ms on Orin NX) happens only
                # for frames that survive the keyframe gate. Validate now so a bad
                # stream still fails at arrival.
                self.codec.check(message)
                self._pending_frames.append((
                    stamp, message, auxiliary, recovered_from_stale,
                    self._monotonic() + self.config.pose_interpolation_wait_ms / 1000.0,
                ))
                self._flush_pending_frames_locked()
            else:
                self._write_selected_frame_locked(
                    stamp, self.codec.to_rgb_png(message), auxiliary, recovered_from_stale,
                    pose_valid, delta_ms, gap_ms,
                )
        except BaseException as error:
            self.fail_locked(_capture_exception(error))

    def _write_selected_frame_locked(
        self, stamp, image_bytes, auxiliary, recovered_from_stale,
        pose_valid, delta_ms, gap_ms,
    ):
        frame = self.writer.write_frame(
            stamp, image_bytes, pose_valid=pose_valid, pose_delta_ms=delta_ms,
            odometry_gap_ms=gap_ms, auxiliary=auxiliary,
        )
        if self.writer.frame_count > 1 and not recovered_from_stale:
            previous = self.writer._frames[-2].stamp
            interval_ms = _stamp_delta_ms(
                (previous.sec, previous.nsec), (stamp.sec, stamp.nsec)
            )
            allowed_ms = (
                1000.0 * self.config.max_frame_interval_sec(self.goal.sample_fps)
                + self.config.max_sample_interval_overshoot_ms
            )
            if interval_ms > allowed_ms:
                raise CaptureStreamError(
                    "selected image interval exceeded configured tolerance: "
                    f"{interval_ms} ms > {allowed_ms} ms"
                )
        if self._feedback_callback is not None:
            try:
                self._feedback_callback(frame, self._monotonic())
            except BaseException:
                pass
        if self.writer.frame_count >= self.goal.max_frames:
            self._sealed_by_frames = True
            self._finish_success_locked()

    def _flush_pending_frames_locked(self):
        while self._pending_frames and not self.done.is_set():
            stamp, message, auxiliary, recovered, deadline = self._pending_frames[0]
            valid, delta, gap, odometry = self.odometry.associate_interpolated(stamp)
            if not valid and self._monotonic() < deadline:
                break
            self._pending_frames.popleft()
            if valid and self._skip_as_non_keyframe_locked(stamp, odometry):
                continue
            # Explicitly discard the earlier-sample association on timeout.
            auxiliary = replace(
                auxiliary, odometry=odometry,
                odometry_delta_ms=delta if valid else None,
            )
            self._write_selected_frame_locked(
                stamp, self.codec.encode_png(self.codec.check(message), message.encoding),
                auxiliary, recovered, valid, delta, gap,
            )
        if self._sealed_by_deadline and not self._pending_frames and not self.done.is_set():
            self._finish_success_locked()

    def _skip_as_non_keyframe_locked(self, stamp, odometry) -> bool:
        """Motion gate on the interpolated body pose; frames without a pose are kept."""
        if not self.config.keyframe_gating or odometry is None:
            return False
        try:
            pose = odometry.payload["pose"]
            position = [float(pose["position"][k]) for k in "xyz"]
            quaternion = [float(pose["orientation"][k]) for k in "xyzw"]
        except (KeyError, TypeError, ValueError):
            return False
        now = _stamp_ns((stamp.sec, stamp.nsec))
        last = self._last_keyframe
        if last is not None:
            moved = math.dist(position, last[1])
            dot = min(1.0, abs(sum(a * b for a, b in zip(quaternion, last[2]))))
            turned = math.degrees(2.0 * math.acos(dot))
            if (moved < self.config.keyframe_min_translation_m
                    and turned < self.config.keyframe_min_rotation_deg
                    and now - last[0] < self.config.keyframe_max_interval_sec * 1e9):
                self.keyframe_skipped += 1
                return True
        self._last_keyframe = (now, position, quaternion)
        return False

    def on_odometry(self, message: Any) -> None:
        """里程计订阅回调：解析并记录完整采样用于后续姿态关联。

        坏消息只计辅助错误，绝不导致整窗失败（辅助源不阻塞 RGB）。
        """
        with self._lock:
            if self.done.is_set():
                return
            if self.cancellation_event.is_set():
                self.done.set()
                return
            self._on_odometry_locked(message)

    def _on_odometry_locked(self, message: Any) -> None:
        try:
            payload = _odometry_payload(message)
            self.odometry.record(payload.stamp, payload)
            self.writer.odometry_samples += 1
        except BaseException as error:
            self._on_auxiliary_error_locked("odometry", error)
            return
        if self.config.odometry_association == "interpolate":
            try:
                self._flush_pending_frames_locked()
            except BaseException as error:
                self.fail_locked(_capture_exception(error))

    def on_camera_info(self, message: Any) -> None:
        """CameraInfo 订阅回调：解析完整内参 payload 并登记关联窗口。"""
        with self._lock:
            if self.done.is_set():
                return
            if self.cancellation_event.is_set():
                self.done.set()
                return
            self._on_camera_info_locked(message)

    def _on_camera_info_locked(self, message: Any) -> None:
        try:
            payload = _camera_info_payload(message)
            self.camera_info.record(payload.stamp, payload)
            self.writer.camera_info_samples += 1
        except BaseException as error:
            self._on_auxiliary_error_locked("camera_info", error)

    def on_depth(self, message: Any) -> None:
        """对齐传感器深度订阅回调：仅在启用保存时记录原始数据。"""
        with self._lock:
            if self.done.is_set():
                return
            if self.cancellation_event.is_set():
                self.done.set()
                return
            self._on_depth_locked(message)

    def _on_depth_locked(self, message: Any) -> None:
        if not self.config.save_sensor_depth:
            return
        try:
            payload = _depth_payload(message)
            self.depth.record(payload.stamp, payload)
            self.writer.depth_samples += 1
        except BaseException as error:
            self._on_auxiliary_error_locked("depth", error)

    def on_tf(self, message: Any, static: bool = False) -> None:
        """保存有界静态 TF 图及相机/机体相关动态变换。"""
        with self._lock:
            if self.done.is_set():
                return
            if self.cancellation_event.is_set():
                self.done.set()
                return
            self._on_tf_locked(message, static)

    def _on_tf_locked(self, message: Any, static: bool) -> None:
        try:
            entries = _tf_payloads(message, static)
        except BaseException as error:
            self._on_auxiliary_error_locked("tf", error)
            return
        for entry in entries:
            self.tf_received += 1
            if (
                static
                or entry["parent_frame_id"] in self._relevant_tf_frames
                or entry["child_frame_id"] in self._relevant_tf_frames
            ):
                self.writer.add_transform(entry)

    def _on_auxiliary_error_locked(self, source: str, error: BaseException) -> None:
        """登记辅助源错误：只计数，绝不转成整窗失败。"""
        self.auxiliary_errors += 1
        self.writer.auxiliary_errors += 1
        message = f"{source}: {type(error).__name__}: {error}"
        self.last_auxiliary_error = message[:512]

    def check_deadline(self) -> None:
        """周期检查 source watchdog 与目标时长上限。

        - 运行中超过 source_timeout_sec 没有有效 RGB → 进入 SOURCE_STALE：
          冻结 deadline、暂停窗口计数，等待时间连续的图像恢复；
        - SOURCE_STALE 期间不因 max_duration 封存；
        - 恢复由 _on_image_locked 完成；长间隙/回退在那里放弃窗口。
        """
        with self._lock:
            if self.done.is_set():
                return
            if self.cancellation_event.is_set():
                self.done.set()
                return
            if self.paused:
                return
            if self.config.odometry_association == "interpolate":
                try:
                    self._flush_pending_frames_locked()
                except BaseException as error:
                    self.fail_locked(_capture_exception(error))
                if self.done.is_set():
                    return
            if not self.source_stale:
                if self.sampler.last_seen is None:
                    stale_elapsed = self._monotonic() - self._started_monotonic
                else:
                    stale_elapsed = (
                        self._monotonic() - self._last_image_monotonic
                        if self._last_image_monotonic is not None
                        else 0.0
                    )
                if stale_elapsed >= self.config.source_timeout_sec:
                    self._enter_source_stale_locked()
                elif self._monotonic() >= self.deadline:
                    self._sealed_by_deadline = True
                    self._finish_success_locked()

    def _enter_source_stale_locked(self) -> None:
        """进入 SOURCE_STALE：冻结 deadline 并通知上层（反馈/诊断）。"""
        self.source_stale = True
        self.source_stale_entries += 1
        self._stale_remaining_sec = max(0.0, self.deadline - self._monotonic())
        self._notify_source_state_locked("SOURCE_STALE", self.sampler.last_seen)

    def _recover_source_locked(self, stamp: FrameStamp) -> None:
        """从 SOURCE_STALE 恢复：顺延 deadline 并通知上层。"""
        self.source_stale = False
        if self._stale_remaining_sec is not None:
            self.deadline = self._monotonic() + self._stale_remaining_sec
            self._stale_remaining_sec = None
        self._notify_source_state_locked("SOURCE_RECOVERED", stamp)

    def _notify_source_state_locked(
        self, stage: str, stamp: FrameStamp | None
    ) -> None:
        if self._source_state_callback is None:
            return
        try:
            self._source_state_callback(stage, stamp)
        except BaseException:
            pass  # 状态回调异常绝不影响采集主流程


    def _finish_success_locked(self) -> None:
        """成功收齐：立即封存，或（defer_seal）只标记完成、由 finalize() 封存。"""
        if self._pending_frames:
            return
        if self.writer.frame_count:
            last = self.writer._frames[-1].stamp
            self.final_stamp_ns = last.sec * 10**9 + last.nsec
        if self._defer_seal:
            self.frames_complete = True
            self.done.set()
            return
        try:
            self.sealed = self.writer.seal()
        except BaseException as error:
            self.fail_locked(_capture_exception(error))
            return
        self.done.set()

    def finalize(self) -> None:
        """封存 defer_seal 会话；在传感器回调线程之外调用，且只调用一次。

        done 已在锁内置位，所有回调入口拿到锁后都会直接返回，writer 不再变化，
        因此这里不持会话锁：封存期间回调和诊断快照都不会被阻塞。
        """
        if not self.frames_complete or self.sealed is not None or self.error is not None:
            return
        try:
            self.sealed = self.writer.seal()
        except BaseException as error:
            self.error = _capture_exception(error)

    def fail(self, error: CaptureError) -> None:
        """从外部线程注入失败（幂等）。"""
        with self._lock:
            self.fail_locked(error)

    def fail_locked(self, error: CaptureError) -> None:
        """登记失败并置位完成事件（幂等）。"""
        if self.done.is_set():
            return
        self.error = error
        self.done.set()

    @property
    def terminal_reason(self) -> str:
        """本次采集的终态原因：max_frames / max_duration / error / cancelled。"""
        if self.sealed is not None:
            return "max_frames" if self._sealed_by_frames else "max_duration"
        return "error" if self.error is not None else "cancelled"

    def snapshot(self) -> dict[str, Any]:
        """线程安全的采集诊断快照（供 /diagnostics 使用）。"""
        last = self.sampler.last_selected
        return {
            "stage": "active",
            "received_images": self.received_images,
            "skipped_images": self.skipped_images,
            "selected_frames": self.writer.frame_count,
            "keyframe_skipped": self.keyframe_skipped,
            "paused": self.paused,
            "last_selected_stamp_sec": last[0] if last is not None else 0,
            "last_selected_stamp_nsec": last[1] if last is not None else 0,
            "odometry_received": self.odometry.received_count,
            "odometry_regressions": self.odometry.regression_count,
            "odometry_gaps": self.odometry.gap_count,
            "camera_info_received": self.camera_info.received_count,
            "depth_received": self.depth.received_count,
            "tf_received": self.tf_received,
            "has_camera_info": self.writer.has_camera_info,
            "has_odometry": self.writer.has_odometry,
            "has_depth": self.writer.has_depth,
            "has_tf": self.writer.has_tf,
            "auxiliary_dropped": self.writer.auxiliary_dropped,
            "tf_dropped": self.writer.tf_dropped,
            "auxiliary_errors": self.auxiliary_errors,
            "last_auxiliary_error": self.last_auxiliary_error or "",
            "source_stale": self.source_stale,
            "source_stale_entries": self.source_stale_entries,
            "zero_stamp_count": self.sampler.zero_stamp_count,
            "duplicate_stamp_count": self.sampler.duplicate_stamp_count,
            "out_of_order_stamp_count": self.sampler.out_of_order_stamp_count,
            "last_image_wall_age_sec": (
                self._monotonic() - self._last_image_monotonic
                if self._last_image_monotonic is not None
                else None
            ),
        }


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


def _capture_exception(error: BaseException) -> CaptureError:
    """把任意异常归一化为类型化 CaptureError（映射到稳定错误码）。"""
    if isinstance(error, CaptureError):
        return error
    if isinstance(error, (OSError, ResultTransactionError)):
        return CaptureError(IO_ERROR, f"capture I/O failed: {type(error).__name__}: {error}")
    return CaptureError(
        INTERNAL_ERROR, f"capture failed: {type(error).__name__}: {error}"
    )


def snapshot_final_name(
    scene_id: str, *, token_source=None
) -> str:
    """生成快照最终目录名：{scene_id}-{16 位十六进制 token}。"""
    if not _SCENE_ID.fullmatch(scene_id):
        raise CaptureError(INVALID_GOAL, "scene_id is unsafe")
    token = (token_source or secrets_token)()
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise CaptureError(INTERNAL_ERROR, "snapshot name token is unsafe")
    return f"{scene_id}-{token}"


def allocate_snapshot_transaction(
    config: CaptureConfig,
    scene_id: str,
    *,
    token_source=None,
) -> ResultTransaction:
    """分配监督进程拥有的快照暂存/发布事务。

    Args:
        config: 采集配置（提供 snapshot_root）
        scene_id: 场景标识符（用于生成最终目录名）
        token_source: 随机 token 生成器（测试注入点）

    Returns:
        已分配的结果事务（暂存目录位于 snapshot_root 下）
    """
    return allocate_result_transaction(
        config.snapshot_root,
        snapshot_final_name(scene_id, token_source=token_source),
    )


def _stamp_ns(stamp: tuple[int, int]) -> int:
    """(sec, nsec) 元组 → 纳秒整数。"""
    return stamp[0] * 1_000_000_000 + stamp[1]


def _stamp_delta_ms(left: tuple[int, int], right: tuple[int, int]) -> float:
    """两个 stamp 的间隔（毫秒），钳制为非负。"""
    return max(0, (_stamp_ns(right) - _stamp_ns(left))) / 1_000_000.0


def _finite_number(value: Any) -> bool:
    """校验值为有限数值（拒绝 bool/NaN/无穷）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _canonical_directory(value: str, label: str) -> str:
    """校验并规范化目录：绝对路径、无符号链接遍历、真实存在。"""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CaptureConfigError(f"{label} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute():
        raise CaptureConfigError(f"{label} must be absolute")
    # 逐组件检查符号链接，杜绝经祖先软链的伪装路径
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            info = current.lstat()
        except OSError as error:
            raise CaptureConfigError(f"{label} cannot be inspected") from error
        if stat.S_ISLNK(info.st_mode):
            raise CaptureConfigError(f"{label} must not traverse symlinks")
    try:
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise CaptureConfigError(f"{label} must exist and resolve") from error
    if canonical != path or path.is_symlink() or not path.is_dir():
        raise CaptureConfigError(f"{label} must be a canonical real directory")
    return str(canonical)


def _write_file_atomically(
    path: Path, data: bytes, maximum: int, *, sync: bool = True
) -> None:
    """原子写文件：临时文件 + os.replace（sync=True 时 fsync 文件与父目录）。

    帧/辅助数据写在事务暂存目录内，seal() 生成清单时统一 fsync；这些
    中间文件不逐帧 fsync，避免 worker 加载 checkpoint / 写 predictions
    时磁盘竞争把采集回调阻塞到秒级，导致 ROS 图像队列丢帧、选中帧
    stamp 间隔超过 tolerance。
    """
    if len(data) > maximum:
        raise CaptureError(IO_ERROR, f"capture asset exceeds bound: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + "-", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            if sync:
                os.fsync(stream.fileno())
        os.replace(temporary, str(path))
        temporary = None
    except OSError as error:
        raise CaptureError(IO_ERROR, f"capture asset could not be written: {path.name}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if sync:
        _fsync_directory(path.parent)


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
    maximum: int,
) -> None:
    """以规范 CSV 格式原子写出表（LF 行终止符）。"""
    import csv
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_file_atomically(
        path, output.getvalue().encode("utf-8"), maximum
    )


def _write_manifest(staging: Path, limits: SnapshotSafetyLimits) -> None:
    """生成 files_sha256.csv 清单：遍历暂存目录并逐文件计算大小与 SHA-256。"""
    contents = {}
    for path in sorted(staging.rglob("*")):
        # 暂存目录内只包含本 writer 写入的文件（快照封存前提）
        relative = path.relative_to(staging).as_posix()
        if not path.is_file() or path.name == "files_sha256.csv":
            continue
        contents[relative] = path
    if len(contents) > limits.max_manifest_entries:
        raise CaptureError(IO_ERROR, "snapshot manifest has too many entries")
    total = 0
    rows = []
    for relative in sorted(contents):
        path = contents[relative]
        size = path.stat().st_size
        if size > limits.max_file_bytes:
            raise CaptureError(IO_ERROR, f"snapshot file exceeds bound: {relative}")
        total += size
        if total > limits.max_total_bytes:
            raise CaptureError(IO_ERROR, "snapshot total exceeds configured bound")
        rows.append(
            {
                "path": relative,
                "size": str(size),
                "sha256": _sha256_file(path),
            }
        )
    _write_csv(
        staging / "files_sha256.csv",
        ("path", "size", "sha256"),
        rows,
        limits.max_metadata_bytes,
    )


def _sha256_file(path: Path) -> str:
    """流式计算文件的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _fsync_staging_files(staging: Path) -> None:
    """批量 fsync 暂存目录内全部普通文件（不跟随符号链接）。

    快照帧文件在采集时以 sync=False 写入以减少磁盘竞争；封窗时调用本
    函数一次性补齐持久性，再写 manifest，保证清单引用的内容与目录条目
    在断电后一致。
    """
    for path in sorted(staging.rglob("*")):
        try:
            if not path.is_file() or path.is_symlink():
                continue
            with open(path, "rb") as stream:
                os.fsync(stream.fileno())
        except OSError:
            # 单文件 fsync 失败由后续 manifest/校验路径暴露；这里不改变
            # 封存控制流，避免辅助文件问题拖死主窗口。
            continue


def _fsync_directory(path: Path) -> None:
    """fsync 目录条目（确保 rename/创建操作持久化）。"""
    flags = os.O_RDONLY | os.O_DIRECTORY
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _format_optional_ms(value: float | None) -> str:
    """把可选毫秒值格式化为 CSV 字段（None → -1）。"""
    return "-1" if value is None else repr(float(value))


_token_sequence = [0]


def secrets_token() -> str:
    """生成 16 位十六进制随机 token（快照目录名后缀）。"""
    return secrets.token_hex(8)


__all__ = [
    "CAMERA_UNAVAILABLE",
    "AuxiliaryAssociator",
    "AuxiliaryCameraInfo",
    "AuxiliaryDepth",
    "AuxiliaryFrameData",
    "AuxiliaryOdometry",
    "CaptureConfig",
    "CaptureConfigError",
    "CaptureError",
    "CaptureGoalParameters",
    "CapturedFrame",
    "CaptureSession",
    "CaptureStreamError",
    "FrameStamp",
    "ImageCodec",
    "OdometryAssociator",
    "SealedSnapshot",
    "SnapshotWriter",
    "StampSampler",
    "allocate_snapshot_transaction",
    "snapshot_final_name",
    "validate_capture_goal",
]
