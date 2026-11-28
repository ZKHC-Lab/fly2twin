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
import json
import math
from pathlib import Path
import stat
import threading
import time
from typing import Any, Deque

import cv2
import numpy as np

from .preparation import INTERNAL_ERROR, INVALID_GOAL, IO_ERROR
from .odometry_interpolation import interpolate_payload
from .result_transaction import (
    ResultTransaction,
    ResultTransactionError,
)
from .snapshot_validator import (
    MAX_TF_ENTRIES,
    FrameStamp,
    SnapshotSafetyLimits,
)
from ._common import is_finite_number as _finite_number

# The message converters, file writers and snapshot writer live in their own modules;
# this one keeps the session, the associators and the original import path.
from .capture_messages import (
    AuxiliaryCameraInfo,
    AuxiliaryDepth,
    AuxiliaryFrameData,
    AuxiliaryOdometry,
    _camera_info_payload,
    _depth_payload,
    _message_frame_id,
    _odometry_payload,
    _tf_payloads,
    _watchdog_image_stamp,
)
from .capture_snapshot import (
    SealedSnapshot,
    SnapshotWriter,
    allocate_snapshot_transaction,
    snapshot_final_name,
)
from .capture_types import (
    _CAMERA_FRAME_ID,
    _CAMERA_NAME,
    _SCENE_ID,
    CAMERA_UNAVAILABLE,
    CaptureError,
    CapturedFrame,
    CaptureStreamError,
)

# 支持的 ROS 图像编码（cv2 以 BGR 语义编码 PNG）
_SUPPORTED_ENCODINGS = ("bgr8", "rgb8")


class CaptureConfigError(ValueError):
    """采集配置非法：必须在构造前失败（fail-fast）。"""


@dataclass(frozen=True)
class CaptureConfig:
    """显式采集配置。只有末尾几个可选项带默认值，默认值只写在这里。

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


class CaptureSession:
    """一次活跃采集目标，绑定其事务与订阅回调。

    所有回调入口（on_image / on_odometry / on_camera_info / check_deadline）
    都持有内部锁，保证采集状态（帧数、错误、封存结果）跨线程一致。
    封存只发生一次：要么帧数达标，要么超时（_sealed_by_deadline），
    要么失败（error 非空）。
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


def _capture_exception(error: BaseException) -> CaptureError:
    """把任意异常归一化为类型化 CaptureError（映射到稳定错误码）。"""
    if isinstance(error, CaptureError):
        return error
    if isinstance(error, (OSError, ResultTransactionError)):
        return CaptureError(IO_ERROR, f"capture I/O failed: {type(error).__name__}: {error}")
    return CaptureError(
        INTERNAL_ERROR, f"capture failed: {type(error).__name__}: {error}"
    )


def _stamp_ns(stamp: tuple[int, int]) -> int:
    """(sec, nsec) 元组 → 纳秒整数。"""
    return stamp[0] * 1_000_000_000 + stamp[1]


def _stamp_delta_ms(left: tuple[int, int], right: tuple[int, int]) -> float:
    """两个 stamp 的间隔（毫秒），钳制为非负。"""
    return max(0, (_stamp_ns(right) - _stamp_ns(left))) / 1_000_000.0


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
