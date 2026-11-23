"""Capture / Reconstruct Action 的归一化终态与客户端接口；不导入 rospy/actionlib。

连续会话的调度核心只认这里的类型：ROS 层把 actionlib 的终态和结果消息转成
这些不可变对象，再交给调度器。
"""

from __future__ import annotations

from dataclasses import dataclass

from .capture import CaptureConfig

# 归一化后的 actionlib 终态名
ACTION_SUCCEEDED = "SUCCEEDED"
ACTION_CANCELLED = "CANCELLED"
ACTION_FAILED = "FAILED"


@dataclass(frozen=True)
class CameraRole:
    """一个已配置的相机角色：role + 现有 Capture Action 与 capture profile。"""

    role: str
    capture_action: str
    capture_config: CaptureConfig


@dataclass(frozen=True)
class CaptureActionOutcome:
    """归一化的 CaptureSegment Action 终态。"""

    action_state: str
    error_code: int
    message: str
    scene_id: str
    camera_name: str
    image_topic: str
    snapshot_uri: str
    manifest_uri: str
    manifest_sha256: str
    frame_count: int


@dataclass(frozen=True)
class ReconstructActionOutcome:
    """归一化的 ReconstructSegment Action 终态。

    camera_role/camera_topic/camera_frame_id/snapshot_manifest_sha256 由
    ROS 层从正式 result.json 读取（本任务身份校验字段），读取失败时为空串，
    上层会据此拒绝成功终态。
    """

    action_state: str
    error_code: int
    message: str
    scene_id: str
    result_manifest_uri: str
    result_manifest_sha256: str
    scene_glb_uri: str
    frame_count: int
    wall_time_sec: float
    camera_role: str = ""
    camera_topic: str = ""
    camera_frame_id: str = ""
    snapshot_manifest_sha256: str = ""
    result_scene_id: str = ""
    result_actual_frame_count: int = 0


class ReconstructActionClient:
    """Reconstruct Action 客户端（由 ROS 层实现并注入）。"""

    def wait_for_server(self, timeout_sec: float) -> bool:
        raise NotImplementedError

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
        raise NotImplementedError

    def wait_for_result(self, timeout_sec: float) -> bool:
        raise NotImplementedError

    def cancel_goal(self) -> None:
        raise NotImplementedError

    def outcome(self) -> ReconstructActionOutcome:
        raise NotImplementedError
