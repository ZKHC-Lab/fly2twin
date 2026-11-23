"""ReconstructSegment Action 的 actionlib 客户端适配器。

把 SimpleActionClient 的终态和结果消息转成 ``ReconstructActionOutcome``，并从正式
result.json 读回任务身份字段，供调度器核对这个结果确实属于它提交的那个窗口。
"""

from __future__ import annotations

import json
import math
import os
import stat
from typing import Any
from urllib.parse import unquote, urlsplit

import actionlib
from actionlib_msgs.msg import GoalStatus
from drone3d_reconstruction_msgs.msg import (
    ReconstructSegmentAction,
    ReconstructSegmentGoal,
    ReconstructSegmentResult,
)
import rospy

from .action_types import (
    ACTION_CANCELLED,
    ACTION_FAILED,
    ACTION_SUCCEEDED,
    ReconstructActionClient,
    ReconstructActionOutcome,
)

_RESULT_IDENTITY_MAX_BYTES = 1024 * 1024


def read_result_identity(uri: str) -> dict[str, Any]:
    """从正式 result.json 读取任务身份字段（失败/超限时返回空字典）。

    这是上层成功门禁的输入来源：只有真实 Reconstruct Action 提交的
    file:// result.json 会被读取，任何异常都按身份不可证明处理。
    """
    if not isinstance(uri, str) or not uri.startswith("file://"):
        return {}
    try:
        split = urlsplit(uri)
    except ValueError:
        return {}
    if split.scheme != "file" or split.netloc or split.query or split.fragment:
        return {}
    path = unquote(split.path)
    if "\x00" in path or not os.path.isabs(path):
        return {}
    try:
        info = os.lstat(path)
    except OSError:
        return {}
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        return {}
    if info.st_size <= 0 or info.st_size > _RESULT_IDENTITY_MAX_BYTES:
        return {}
    try:
        with open(path, "rb") as stream:
            data = stream.read(_RESULT_IDENTITY_MAX_BYTES + 1)
    except OSError:
        return {}
    if len(data) > _RESULT_IDENTITY_MAX_BYTES:
        return {}
    try:
        payload = json.loads(data.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict):
        snapshot = {}
    identity = {
        "scene_id": payload.get("scene_id", ""),
        "actual_frame_count": payload.get("actual_frame_count", 0),
        "camera_role": payload.get("camera_role", ""),
        "camera_topic": payload.get("camera_topic", ""),
        "camera_frame_id": payload.get("camera_frame_id", ""),
        "snapshot_sha256": snapshot.get("sha256", ""),
    }
    return {key: value for key, value in identity.items() if isinstance(value, (str, int))}


class ActionlibReconstructClient(ReconstructActionClient):
    """现有 ReconstructSegment Action 的 SimpleActionClient 适配器。"""

    def __init__(self, action_name: str) -> None:
        self._action_name = action_name
        self._client = actionlib.SimpleActionClient(
            action_name, ReconstructSegmentAction
        )

    def wait_for_server(self, timeout_sec: float) -> bool:
        return self._client.wait_for_server(rospy.Duration.from_sec(timeout_sec))

    def send_goal(
        self,
        snapshot_uri: str,
        manifest_sha256: str,
        feedback_cb,
        image_resolution: int = 0,
        set_image_resolution: bool = False,
        worker_timeout_sec: float = 0.0,
        set_worker_timeout: bool = False,
    ) -> None:
        goal = ReconstructSegmentGoal()
        goal.snapshot_uri = snapshot_uri
        goal.manifest_sha256 = manifest_sha256
        if set_image_resolution:
            if isinstance(image_resolution, bool) or not isinstance(
                image_resolution, int
            ) or image_resolution <= 0:
                raise ValueError("image_resolution must be a positive integer")
            goal.image_resolution = int(image_resolution)
        goal.set_image_resolution = bool(set_image_resolution)
        if set_worker_timeout:
            if isinstance(worker_timeout_sec, bool) or not isinstance(
                worker_timeout_sec, (int, float)
            ):
                raise ValueError("worker_timeout_sec must be a finite number")
            converted = float(worker_timeout_sec)
            if not math.isfinite(converted) or converted <= 0:
                raise ValueError("worker_timeout_sec must be positive and finite")
            goal.worker_timeout_sec = converted
        goal.set_worker_timeout = bool(set_worker_timeout)
        self._client.send_goal(goal, feedback_cb=feedback_cb)

    def wait_for_result(self, timeout_sec: float) -> bool:
        return self._client.wait_for_result(rospy.Duration.from_sec(timeout_sec))

    def cancel_goal(self) -> None:
        self._client.cancel_goal()

    def outcome(self) -> ReconstructActionOutcome:
        action_state = _normalized_action_state(self._client.get_state())
        result = self._client.get_result()
        if result is None:
            result = ReconstructSegmentResult()
            result.error_code = ReconstructSegmentResult.INTERNAL_ERROR
            result.message = "reconstruct action has no result message"
        try:
            wall_time_sec = float(result.wall_time.to_sec())
        except BaseException:
            wall_time_sec = 0.0
        identity = read_result_identity(
            str(getattr(result, "result_manifest_uri", ""))
        )
        return ReconstructActionOutcome(
            action_state=action_state,
            error_code=int(getattr(result, "error_code", ReconstructSegmentResult.INTERNAL_ERROR)),
            message=str(getattr(result, "message", "")),
            scene_id=str(getattr(result, "scene_id", "")),
            result_manifest_uri=str(getattr(result, "result_manifest_uri", "")),
            result_manifest_sha256=str(getattr(result, "result_manifest_sha256", "")),
            scene_glb_uri=str(getattr(result, "scene_glb_uri", "")),
            frame_count=int(getattr(result, "frame_count", 0)),
            wall_time_sec=wall_time_sec,
            camera_role=str(identity.get("camera_role", "")),
            camera_topic=str(identity.get("camera_topic", "")),
            camera_frame_id=str(identity.get("camera_frame_id", "")),
            snapshot_manifest_sha256=str(identity.get("snapshot_sha256", "")),
            result_scene_id=str(identity.get("scene_id", "")),
            result_actual_frame_count=int(identity.get("actual_frame_count", 0)),
        )


def _normalized_action_state(state: int) -> str:
    if state in (GoalStatus.SUCCEEDED,):
        return ACTION_SUCCEEDED
    if state in (GoalStatus.PREEMPTED, GoalStatus.RECALLED):
        return ACTION_CANCELLED
    return ACTION_FAILED
