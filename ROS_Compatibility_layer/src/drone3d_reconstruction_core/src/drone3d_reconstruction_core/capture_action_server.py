"""ROS1 CaptureSegment Action 服务端：固定单摄像头角色的采集调度。

服务端拥有话题订阅与一个活跃采集槽位。采集进行中到达的第二个目标在
接受前就被以 CAMERA_UNAVAILABLE 拒绝——既不抢占也不排队。采集与重建
可以并发运行，因为它们是独立的 ROS 节点，且采集服务端从不触碰重建事务。

主要功能：
  - 目标预接受仲裁（单槽位，忙时拒绝）与取消处理
  - 采集会话生命周期：starting → accepting → active → terminalizing
  - 快照事务的分配/提交/清理（监督进程所有）
  - /diagnostics 遥测快照与终态观察者

The server owns the topic subscriptions and one active capture slot.  A second
goal while a capture is running is rejected pre-accept with
``CAMERA_UNAVAILABLE``; it is never preempted and never queued.  Capture and
Reconstruct may run concurrently because they are separate ROS nodes and the
capture server never touches the reconstruction transaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from collections import deque
from bisect import bisect_right
import threading
import time
from typing import Any, Callable

import actionlib
from actionlib_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
import rospy
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

from .action_terminal import TerminalCall, converge_terminal, query_goal_status
from .capture_handoff import ImageHandoff, consecutive, stamp_ns

from drone3d_reconstruction_msgs.msg import (
    CaptureSegmentAction,
    CaptureSegmentFeedback,
    CaptureSegmentResult,
)

from .capture import (
    CAMERA_UNAVAILABLE,
    INTERNAL_ERROR,
    INVALID_GOAL,
    IO_ERROR,
    CaptureConfig,
    CaptureConfigError,
    CaptureError,
    CaptureSession,
    CapturedFrame,
    FrameStamp,
    _tf_payloads,
    allocate_snapshot_transaction,
    validate_capture_goal,
)

# 采集错误码 → 冻结的 CaptureSegmentResult 错误码映射
_CAPTURE_CODE_TO_RESULT = {
    INVALID_GOAL: CaptureSegmentResult.INVALID_GOAL,
    CAMERA_UNAVAILABLE: CaptureSegmentResult.CAMERA_UNAVAILABLE,
    IO_ERROR: CaptureSegmentResult.IO_ERROR,
    INTERNAL_ERROR: CaptureSegmentResult.INTERNAL_ERROR,
}

# 图像回调偶尔会被 worker 磁盘/内存竞争拖慢；queue_size=1 会让 15Hz
# 相机的中间帧被 ROS 丢弃，选中帧 stamp 跳变数秒，最终触发 interval
# tolerance 整窗失败。给图像/深度回调保留约 1 秒的应用层缓冲。
_CAPTURE_IMAGE_QUEUE_SIZE = 16
_CAPTURE_DEPTH_QUEUE_SIZE = 16


@dataclass
class _CaptureRecord:
    """一次采集目标在仲裁期间的内部状态（受服务端锁保护）。

    生命周期：``starting`` → ``accepting`` → ``active`` → ``terminalizing``；
    除 actionlib 自身线程安全的 ``handle`` 与两个 ``threading.Event`` 外，
    所有可变字段只在持有 ``self._lock`` 时读写。
    """

    # -- 目标身份 -------------------------------------------------

    handle: Any
    # actionlib GoalHandle：接受/拒绝目标、发布反馈、执行终态过渡的
    # 唯一通道。注意取消回调拿到的 handle 可能是重新包装后的对象，
    # 身份比对一律走 ``goal_id`` 而非 ``is``（见 _cancel_callback）。

    goal_id: str | None
    # actionlib 目标号（``get_goal_id().id``）：取消请求按它比对，终态日志用它定位。
    # 在 goal 回调里、服务端锁外读出缓存：get_goal_id 要拿 actionlib 的锁，服务端锁内
    # 不调用句柄方法。读不到时为 None。

    # -- 配置与校验结果 -------------------------------------------

    config: CaptureConfig | None = None
    # 生效的运行时采集配置。``None`` 表示目标尚未校验通过；校验成功后
    # 写入（已应用 per-goal 的 save_sensor_depth 覆盖），供会话构造、
    # 结果回填与错误路径兜底（``record.config or self._config``）使用。

    # -- 状态机 ---------------------------------------------------

    phase: str = "starting"
    # 状态机阶段，取值：starting / accepting / active / committing /
    # terminalizing；/diagnostics 遥测中作为 ``stage`` 上报。

    cancel_requested: bool = False
    # 是否已受理取消请求。取消先于目标回调到达时（初始状态即 RECALLING）
    # 也预置为 True，保证“接受前取消”绝不进入采集。

    accepted: bool = False
    # ``set_accepted`` 是否已成功执行；接受失败按 INTERNAL_ERROR 收敛。

    terminal_claimed: bool = False
    # 终态认领标志（幂等闸门）：保证终态过渡（发布结果 + 释放槽位）在
    # 取消、错误、正常完成等并发路径下只执行一次。

    # -- 运行时资源 -----------------------------------------------

    session: CaptureSession | None = None
    # 活跃采集会话。仅在重放全部 latched 静态 TF 之后、与
    # ``record.phase = "active"`` 同一临界区内注册；注册之前订阅回调
    # 看不到它，从而关闭“先收到首帧并封存、后重放 TF”的竞态窗口。

    transaction: Any = None
    # 快照事务（暂存目录的分配/提交/清理所有权）；终态时若未提交
    # （``committed=False``）则调用 cleanup 回滚暂存。

    sealed: Any = None
    # 预留字段：当前实现中从未被赋值——终态结果直接取自
    # ``session.sealed``。保留作状态扩展点，勿在未落实现有语义前依赖。

    handoff_replaying: bool = False

    committed: bool = False
    # 快照事务是否已提交。renameat2 发布成功即视为已提交；之后防御性
    # 检查/fsync 失败不会把已发布快照当作失败重试同一最终名。

    # -- 并发原语 -------------------------------------------------

    done: threading.Event = field(default_factory=threading.Event)
    # 目标整体收敛事件：``_release_terminal`` 置位；``shutdown()`` 据此
    # 等待活跃目标在超时内终止。

    cancellation_event: threading.Event = field(default_factory=threading.Event)
    # 取消信号事件：注入 CaptureSession，使传感器回调与 ``_run_goal``
    # 的等待循环都能及时观察到取消请求；``_cancel_callback`` 置位。
    # PAUSE+立即 STOP 场景下还会把 ``session.done`` 一并置位，避免
    # 已暂停会话因等不到回调而永远停在 PREEMPTING。

    started_monotonic: float = field(default_factory=time.monotonic)
    # 目标到达时刻的 monotonic 时间戳：反馈消息 ``elapsed`` 字段的
    # 计时基准（避免系统时钟跳变影响耗时计算）。


class CaptureActionServer:
    """预接受单槽位采集服务端，发布权归监督进程所有。"""

    def __init__(
        self,
        *,
        action_name: str,
        config: CaptureConfig,
        action_server_factory=actionlib.ActionServer,
        subscriber_factory=rospy.Subscriber,
        transaction_factory=allocate_snapshot_transaction,
        session_factory=CaptureSession,
        launch_dispatch: Callable | None = None,
        terminal_observer: Callable[[str, CaptureSegmentResult, str], None] | None = None,
    ) -> None:
        """构造采集服务端并注册订阅与 Action 服务。

        Args:
            action_name: Action 服务名（如 capture_segment）
            config: 已 preflight 的采集配置
            action_server_factory: actionlib 工厂（测试注入点）
            subscriber_factory: 话题订阅工厂（测试注入点）
            transaction_factory: 快照事务工厂（测试注入点）
            session_factory: 采集会话工厂（测试注入点）
            launch_dispatch: 目标派发函数（默认守护线程）
            terminal_observer: 最佳努力终态观察者，异常被忽略
        """
        if not isinstance(config, CaptureConfig):
            raise TypeError("config must be CaptureConfig")
        if terminal_observer is not None and not callable(terminal_observer):
            raise TypeError("terminal_observer must be callable or None")
        self._config = config
        self._transaction_factory = transaction_factory
        self._session_factory = session_factory
        self._terminal_observer = terminal_observer
        self._launch_dispatch = launch_dispatch or _make_daemon_thread
        self._lock = threading.Lock()
        self._active: _CaptureRecord | None = None
        self._shutdown = False
        self._accepted_count = 0
        self._busy_rejection_count = 0
        self._cancel_request_count = 0
        self._terminal_attempt_count = 0
        self._last_terminal: dict[str, Any] | None = None
        self._subscribers = []
        self._server_stopped = False
        # 最近消息时间戳（供 /diagnostics 输入新鲜度诊断）
        self._latest_image_stamp = None
        self._latest_image_wall_sec = None
        self._latest_odometry_stamp = None
        self._latest_odometry_wall_sec = None
        self._latest_depth_stamp = None
        self._latest_depth_wall_sec = None
        self._latest_tf_stamp = None
        self._latest_tf_wall_sec = None
        # Keyed by (parent, child): two latched publishers may give the same child
        # different parents (RealSense camera_aligned_depth_to_color_frame and the
        # body->camera extrinsic both parent camera_color_optical_frame).
        self._tf_static_transforms: dict[tuple[str, str], Any] = {}
        self._handoff = ImageHandoff(config.handoff_buffer_frames, config.handoff_buffer_max_bytes)
        self._handoff_predecessor = None
        self._odometry_history = deque(maxlen=max(512, config.handoff_buffer_frames * 32))
        self._camera_info_history = deque(maxlen=max(1, config.handoff_buffer_frames * 2))

        # 按配置注册图像/里程计/CameraInfo 订阅
        self._subscribers.append(
            subscriber_factory(
                config.image_topic,
                Image,
                self._image_callback,
                queue_size=_CAPTURE_IMAGE_QUEUE_SIZE,
                buff_size=config.max_image_width * config.max_image_height * 3 + 1024,
            )
        )
        if config.odometry_topic is not None:
            self._subscribers.append(
                subscriber_factory(
                    config.odometry_topic,
                    Odometry,
                    self._odometry_callback,
                    queue_size=8,
                )
            )
        if config.camera_info_topic is not None:
            self._subscribers.append(
                subscriber_factory(
                    config.camera_info_topic,
                    CameraInfo,
                    self._camera_info_callback,
                    queue_size=1,
                )
            )
        if config.depth_topic is not None:
            self._subscribers.append(
                subscriber_factory(
                    config.depth_topic,
                    Image,
                    self._depth_callback,
                    queue_size=_CAPTURE_DEPTH_QUEUE_SIZE,
                )
            )
        if config.tf_topic is not None:
            self._subscribers.append(
                subscriber_factory(
                    config.tf_topic,
                    TFMessage,
                    self._tf_callback,
                    queue_size=1,
                )
            )
        if config.tf_static_topic is not None:
            self._subscribers.append(
                subscriber_factory(
                    config.tf_static_topic,
                    TFMessage,
                    self._tf_static_callback,
                    queue_size=1,
                )
            )
        try:
            self._server = action_server_factory(
                action_name,
                CaptureSegmentAction,
                self._goal_callback,
                self._cancel_callback,
                auto_start=False,
            )
            self._server.start()
        except BaseException:
            # 构造失败必须注销已注册的订阅，避免泄漏
            self._unregister_subscribers()
            raise

    @property
    def accepted_count(self) -> int:
        """已接受目标累计数。"""
        with self._lock:
            return self._accepted_count

    @property
    def busy_rejection_count(self) -> int:
        """忙时拒绝目标累计数。"""
        with self._lock:
            return self._busy_rejection_count

    @property
    def cancel_request_count(self) -> int:
        """受理的取消请求累计数。"""
        with self._lock:
            return self._cancel_request_count

    @property
    def terminal_attempt_count(self) -> int:
        """终态过渡尝试累计数。"""
        with self._lock:
            return self._terminal_attempt_count

    def is_busy(self) -> bool:
        """当前是否有活跃采集目标。"""
        with self._lock:
            return self._active is not None

    def pause(self) -> bool:
        """暂停当前活跃采集会话的加帧；无活跃会话返回 False。"""
        with self._lock:
            record = self._active
            session = record.session if record is not None else None
        if session is None:
            return False
        session.pause()
        return True

    def resume(self) -> bool:
        """恢复当前活跃采集会话；无活跃会话返回 False。"""
        with self._lock:
            record = self._active
            session = record.session if record is not None else None
        if session is None:
            return False
        session.resume()
        return True

    def is_paused(self) -> bool:
        """当前活跃会话是否处于暂停状态。"""
        with self._lock:
            record = self._active
            session = record.session if record is not None else None
        return bool(session is not None and session.is_paused())

    def diagnostics_snapshot(self) -> dict[str, Any]:
        """线程安全的遥测快照（供监督节点 /diagnostics 发布）。"""
        with self._lock:
            record = self._active
            session_snapshot = (
                record.session.snapshot()
                if record is not None and record.session is not None
                else None
            )
            return {
                "running": record is not None,
                "stage": record.phase if record is not None else "idle",
                "accepted_count": self._accepted_count,
                "busy_rejection_count": self._busy_rejection_count,
                "cancel_request_count": self._cancel_request_count,
                "terminal_attempt_count": self._terminal_attempt_count,
                "handoff_queued_frames": len(self._handoff.queue),
                "handoff_queued_bytes": self._handoff.bytes,
                "handoff_evicted_frames": self._handoff.evicted,
                "handoff_replayed_frames": self._handoff.replayed,
                "handoff_decimated_frames": self._handoff.decimated,
                "session": session_snapshot,
                "latest_image_stamp": self._latest_image_stamp,
                "latest_image_wall_age_sec": (
                    time.monotonic() - self._latest_image_wall_sec
                    if self._latest_image_wall_sec is not None
                    else None
                ),
                "latest_odometry_stamp": self._latest_odometry_stamp,
                "latest_odometry_wall_age_sec": (
                    time.monotonic() - self._latest_odometry_wall_sec
                    if self._latest_odometry_wall_sec is not None
                    else None
                ),
                "latest_depth_stamp": self._latest_depth_stamp,
                "latest_depth_wall_age_sec": (
                    time.monotonic() - self._latest_depth_wall_sec
                    if self._latest_depth_wall_sec is not None
                    else None
                ),
                "latest_tf_stamp": self._latest_tf_stamp,
                "latest_tf_wall_age_sec": (
                    time.monotonic() - self._latest_tf_wall_sec
                    if self._latest_tf_wall_sec is not None
                    else None
                ),
                "last_terminal": dict(self._last_terminal)
                if self._last_terminal is not None
                else None,
            }

    def shutdown(self, timeout_sec: float = 3.0) -> None:
        """关闭服务端：取消活跃目标、等待收敛、注销订阅。"""
        deadline = time.monotonic() + timeout_sec
        with self._lock:
            self._shutdown = True
            record = self._active
            if record is not None:
                record.cancel_requested = True
                record.cancellation_event.set()
        if record is not None:
            remaining = max(0.0, deadline - time.monotonic())
            if not record.done.wait(remaining):
                raise RuntimeError("capture goal did not terminate during shutdown")
        self._unregister_subscribers()
        self._stop_action_server()

    def _stop_action_server(self) -> None:
        """Stop actionlib timers and ROS endpoints exactly once."""
        with self._lock:
            if self._server_stopped:
                return
            self._server_stopped = True
        self._server.stop()

    def _image_callback(self, message: Image) -> None:
        """图像订阅回调：记录新鲜度并转发给活跃会话。"""
        with self._lock:
            stamp = getattr(message.header, "stamp", message.header)
            incoming = (
                getattr(stamp, "secs", getattr(stamp, "sec", 0)),
                getattr(stamp, "nsecs", getattr(stamp, "nsec", 0)),
            )
            if self._latest_image_stamp is not None and incoming < self._latest_image_stamp:
                self._handoff.clear()
                self._handoff_predecessor = None
            self._latest_image_stamp = incoming
            self._latest_image_wall_sec = time.monotonic()
            record = self._active
            session = record.session if record is not None else None
            if session is None or session.done.is_set() or record.handoff_replaying:
                self._handoff.push(message)
                return
        session.on_image(message)
        # An odometry callback can complete the window while this callback is in
        # flight. Preserve this later image rather than silently losing it.
        final = session.final_stamp_ns
        if final is not None and stamp_ns(message) > final:
            with self._lock:
                active = self._active
                target = active.session if active is not None else None
                if (active is not record and target is not None and not target.done.is_set()
                        and not active.handoff_replaying
                        and consecutive(session.goal.scene_id, target.goal.scene_id)):
                    pass  # The successor became live before this callback returned.
                else:
                    self._handoff.push(message)
                    target = None
            if target is not None:
                target.on_image(message)

    def _odometry_callback(self, message: Odometry) -> None:
        """里程计订阅回调：记录新鲜度并转发给活跃会话。"""
        with self._lock:
            stamp = getattr(message.header, "stamp", message.header)
            incoming = (
                getattr(stamp, "secs", getattr(stamp, "sec", 0)),
                getattr(stamp, "nsecs", getattr(stamp, "nsec", 0)),
            )
            if self._latest_odometry_stamp is not None and incoming < self._latest_odometry_stamp:
                self._odometry_history.clear()
            self._latest_odometry_stamp = incoming
            self._latest_odometry_wall_sec = time.monotonic()
            if self._config.odometry_association == "interpolate":
                self._odometry_history.append(message)
            record = self._active
            session = record.session if record is not None else None
        if session is not None and not record.handoff_replaying:
            session.on_odometry(message)

    def _camera_info_callback(self, message: CameraInfo) -> None:
        """CameraInfo 订阅回调：转发给活跃会话。"""
        with self._lock:
            self._camera_info_history.append(message)
            record = self._active
            session = record.session if record is not None else None
        if session is not None and not record.handoff_replaying:
            session.on_camera_info(message)

    def _depth_callback(self, message: Image) -> None:
        """深度订阅回调：记录新鲜度并转发给活跃会话。"""
        with self._lock:
            stamp = getattr(message.header, "stamp", message.header)
            self._latest_depth_stamp = (
                getattr(stamp, "secs", getattr(stamp, "sec", 0)),
                getattr(stamp, "nsecs", getattr(stamp, "nsec", 0)),
            )
            self._latest_depth_wall_sec = time.monotonic()
            record = self._active
            session = record.session if record is not None else None
        if session is not None:
            session.on_depth(message)

    def _tf_callback(self, message: TFMessage) -> None:
        """TF 订阅回调：记录新鲜度并转发给活跃会话。"""
        with self._lock:
            self._update_latest_tf_locked(message)
            record = self._active
            session = record.session if record is not None else None
        if session is not None:
            session.on_tf(message, static=False)

    def _tf_static_callback(self, message: TFMessage) -> None:
        """TF static 订阅回调：记录新鲜度、缓存并转发给活跃会话。"""
        with self._lock:
            self._update_latest_tf_locked(message)
            try:
                transforms = tuple(getattr(message, "transforms", ()))
            except TypeError:
                transforms = ()
            for transform in transforms:
                child = getattr(transform, "child_frame_id", None)
                parent = getattr(getattr(transform, "header", None), "frame_id", None)
                if not isinstance(child, str) or not child or not isinstance(parent, str):
                    continue
                key = (parent, child)
                # Intermediate static frames may connect body and camera.
                # Keep the bounded static graph, including separate publishers.
                try:
                    _tf_payloads(TFMessage(transforms=[transform]), static=True)
                except Exception:
                    # A malformed update must not replace a last-known-good
                    # latched transform replayed into future sessions.
                    continue
                if (
                    key in self._tf_static_transforms
                    or len(self._tf_static_transforms) < self._config.max_tf_samples
                ):
                    self._tf_static_transforms[key] = transform
            record = self._active
            session = record.session if record is not None else None
        if session is not None:
            # Isolate each transform: one malformed sibling must not discard
            # valid static extrinsics in the same TFMessage. Invalid entries
            # still reach CaptureSession individually and increment its
            # auxiliary error counter.
            for transform in transforms:
                session.on_tf(TFMessage(transforms=[transform]), static=True)

    def _update_latest_tf_locked(self, message: TFMessage) -> None:
        transforms = getattr(message, "transforms", [])
        if not transforms:
            return
        latest = None
        for transform in transforms:
            stamp = getattr(transform.header, "stamp", transform.header)
            current = (
                getattr(stamp, "secs", getattr(stamp, "sec", 0)),
                getattr(stamp, "nsecs", getattr(stamp, "nsec", 0)),
            )
            if latest is None or current > latest:
                latest = current
        self._latest_tf_stamp = latest
        self._latest_tf_wall_sec = time.monotonic()

    def _goal_callback(self, goal_handle: Any) -> None:
        """新目标回调：单槽位仲裁（忙/关闭 → 预接受拒绝）。"""
        record = _CaptureRecord(goal_handle, _goal_id(goal_handle))
        rejection = None
        try:
            initial_status = goal_handle.get_goal_status().status
        except BaseException:
            initial_status = None
        with self._lock:
            if self._shutdown:
                rejection = _result(
                    CaptureSegmentResult.INTERNAL_ERROR,
                    "capture server is shutting down",
                    config=self._config,
                )
            elif self._active is not None:
                # 单槽位：忙时直接拒绝，绝不抢占或排队
                self._busy_rejection_count += 1
                rejection = _result(
                    CaptureSegmentResult.CAMERA_UNAVAILABLE,
                    "another capture goal is active",
                    config=self._config,
                )
            else:
                self._active = record
                # RECALLING 状态说明取消先于本回调到达
                record.cancel_requested = initial_status == GoalStatus.RECALLING
        if rejection is not None:
            _reject_goal(goal_handle, rejection)
            return

        try:
            thread = self._launch_dispatch(self._run_goal, (record,))
            if not isinstance(thread, threading.Thread):
                raise TypeError("launch_dispatch must return an unstarted Thread")
            if thread.ident is not None or thread.is_alive():
                raise RuntimeError("launch_dispatch returned an already-started Thread")
            thread.daemon = True
            thread.start()
        except BaseException as error:
            # 派发失败：按内部错误收敛并释放槽位
            self._terminalize(
                record,
                "rejected",
                _result(
                    CaptureSegmentResult.INTERNAL_ERROR,
                    f"capture dispatch failed: {type(error).__name__}",
                    config=self._config,
                ),
            )

    def _cancel_callback(self, goal_handle: Any) -> None:
        """取消请求回调：仅受理当前活跃目标并置位取消事件。

        PAUSE 后立即 STOP 的真实复现中，仅置 ``cancellation_event`` 还不够：
        已暂停的 CaptureSession 可能等不到下一个传感器回调，导致 action
        永远停在 PREEMPTING。因此这里同时把 session.done 置位，让
        ``_run_goal`` 的等待循环无条件立刻退出并发布 PREEMPTED。
        """
        # actionlib 传给 cancel 回调的 goal handle 可能不是 goal 回调里保存的
        # 同一个对象（ActionServer 会重新包一层）；按目标号比对，避免 PAUSE+STOP
        # 时取消请求被身份比较误拒。读不到目标号时不认领：别的目标的取消不能
        # 打断当前采集。目标号在服务端锁外读（get_goal_id 要拿 actionlib 的锁）。
        cancel_id = _goal_id(goal_handle)
        with self._lock:
            record = self._active
            if record is None:
                return
            record_id = record.goal_id
            if record_id is None or record_id != cancel_id:
                rospy.loginfo(
                    "capture cancel ignored: goal id mismatch record=%r cancel=%r",
                    record_id,
                    cancel_id,
                )
                return
            record.cancel_requested = True
            rospy.loginfo("capture cancel accepted: goal=%s", record_id)
            record.cancellation_event.set()
            session = record.session
            if session is not None:
                try:
                    session.done.set()
                except BaseException:
                    pass
            self._cancel_request_count += 1

    def _run_goal(self, record: _CaptureRecord) -> None:
        """目标主流程（派发线程）：校验→事务→会话→接受→等待→终态化。"""
        goal_handle = record.handle
        try:
            goal_message = goal_handle.get_goal()
            set_save_sensor_depth = getattr(
                goal_message, "set_save_sensor_depth", False
            )
            if type(set_save_sensor_depth) is not bool:
                raise CaptureError(
                    INVALID_GOAL, "set_save_sensor_depth must be a bool"
                )
            runtime_config = self._config
            if set_save_sensor_depth:
                if type(getattr(goal_message, "save_sensor_depth", False)) is not bool:
                    raise CaptureError(
                        INVALID_GOAL, "save_sensor_depth must be a bool"
                    )
                try:
                    runtime_config = replace(
                        self._config,
                        save_sensor_depth=goal_message.save_sensor_depth,
                    )
                except CaptureConfigError as error:
                    raise CaptureError(INVALID_GOAL, str(error)) from error
            goal = validate_capture_goal(goal_message, runtime_config)
            with self._lock:
                if self._active is record and not record.terminal_claimed:
                    record.config = runtime_config
        except CaptureError as error:
            self._terminalize_rejected(record, error)
            return
        except BaseException as error:
            self._terminalize(
                record,
                "rejected",
                _result(
                    CaptureSegmentResult.INTERNAL_ERROR,
                    f"capture goal could not be read: {type(error).__name__}",
                    config=record.config or self._config,
                ),
            )
            return

        runtime_config = record.config or self._config
        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return
            cancelled = record.cancel_requested
        if cancelled:
            # 接受前取消：直接收敛为已取消，绝不接受
            self._terminalize(
                record,
                "canceled",
                _result(
                    CaptureSegmentResult.CANCELLED,
                    "cancelled before capture",
                    config=runtime_config,
                ),
            )
            return

        try:
            transaction = self._transaction_factory(
                runtime_config, goal.scene_id
            )
        except CaptureError as error:
            self._terminalize_rejected(record, error)
            return
        except BaseException as error:
            self._terminalize(
                record,
                "rejected",
                _result(
                    CaptureSegmentResult.IO_ERROR,
                    f"snapshot transaction allocation failed: {type(error).__name__}",
                    config=runtime_config,
                ),
            )
            return

        owns_record = False
        with self._lock:
            if self._active is record and not record.terminal_claimed:
                owns_record = True
                record.transaction = transaction
                cancelled_before_accept = record.cancel_requested
                if not cancelled_before_accept:
                    record.phase = "accepting"
        if not owns_record:
            # 所有权已变化（取消/关闭）：回滚事务并退出
            transaction.cleanup()
            return

        if cancelled_before_accept:
            self._terminalize(
                record,
                "canceled",
                _result(CaptureSegmentResult.CANCELLED, "cancelled before capture"),
            )
            return

        try:
            session = self._session_factory(
                runtime_config,
                goal,
                transaction,
                record.cancellation_event,
                feedback_callback=lambda frame, _now: self._publish_feedback(
                    record, frame
                ),
                source_state_callback=lambda stage, stamp: (
                    self._publish_source_state(record, stage, stamp)
                ),
                # Sealing fsyncs every frame; keep it off the sensor callback threads.
                defer_seal=True,
            )
        except BaseException as error:
            self._terminalize(
                record,
                "rejected",
                _result(
                    CaptureSegmentResult.INTERNAL_ERROR,
                    f"capture session construction failed: {type(error).__name__}",
                    config=runtime_config,
                ),
            )
            return

        try:
            goal_handle.set_accepted("capture started")
        except BaseException as error:
            # set_accepted can fail before or after actionlib moved the goal to ACTIVE;
            # only the status says which, and only an accepted goal may be aborted.
            status, _ = query_goal_status(goal_handle)
            accepted = status in (GoalStatus.ACTIVE, GoalStatus.PREEMPTING)
            with self._lock:
                cancelled = record.cancel_requested
                if accepted and self._active is record and not record.terminal_claimed:
                    record.accepted = True
                    self._accepted_count += 1
                    record.phase = "active"
            method = _legal_failure_method(status, accepted, cancelled)
            self._terminalize(
                record,
                method,
                _result(
                    CaptureSegmentResult.CANCELLED
                    if method == "canceled"
                    else CaptureSegmentResult.INTERNAL_ERROR,
                    f"failed to accept capture goal: {type(error).__name__}",
                    config=runtime_config,
                ),
            )
            return

        try:
            status = goal_handle.get_goal_status().status
        except BaseException:
            status = None
        if status not in (GoalStatus.ACTIVE, GoalStatus.PREEMPTING):
            self._terminalize(
                record,
                "aborted",
                _result(
                    CaptureSegmentResult.INTERNAL_ERROR,
                    f"failed to accept capture goal: unexpected status {status}",
                    config=runtime_config,
                ),
            )
            return

        owns_record = False
        handoff_cutoff = None
        with self._lock:
            if self._active is record and not record.terminal_claimed:
                owns_record = True
                record.accepted = True
                self._accepted_count += 1
                record.phase = "active"
                self._handoff.min_interval_ns = goal.sample_period_ns
                predecessor = self._handoff_predecessor
                if (runtime_config.handoff_buffer_frames and not runtime_config.save_sensor_depth
                        and predecessor and consecutive(predecessor[0], goal.scene_id)):
                    handoff_cutoff = predecessor[1]
                    record.handoff_replaying = True
                else:
                    # A new session, nonconsecutive goal, or depth-recording job
                    # must start live; never replay a previous task's images.
                    self._handoff.clear()
                    if runtime_config.odometry_association == "interpolate":
                        for message in self._odometry_history:
                            session.on_odometry(message)
                # 在 session 对订阅回调可见前重放全部 latched 静态变换；持有
                # 服务端锁可关闭“先收到首帧并封存、后重放 TF”的竞态窗口。
                if self._tf_static_transforms:
                    session.on_tf(
                        TFMessage(
                            transforms=list(self._tf_static_transforms.values())
                        ),
                        static=True,
                    )
                record.session = session
        if not owns_record:
            transaction.cleanup()
            return

        if handoff_cutoff is not None:
            self._replay_handoff(record, handoff_cutoff)

        # 等待会话完成（每 100ms 检查一次截止时间）。取消请求在这里直接
        # 中断等待：即使 session.done 因暂停/竞态尚未置位，也不能让
        # action 永远停在 PREEMPTING（PAUSE + 立即 STOP 的真机复现）。
        while True:
            with self._lock:
                if (
                    record.cancel_requested
                    and self._active is record
                    and not record.terminal_claimed
                ):
                    rospy.loginfo("capture wait loop breaking on cancel_requested")
                    break
            done = session.done.wait(0.1)
            if done:
                break
            session.check_deadline()

        terminal = None
        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return
            if record.cancel_requested:
                record.phase = "terminalizing"
                terminal = (
                    "preempted",
                    _result(
                        CaptureSegmentResult.CANCELLED,
                        "capture cancelled",
                        config=record.config or self._config,
                    ),
                    "capture cancelled",
                )
            elif session.error is not None:
                record.phase = "terminalizing"
                code = _CAPTURE_CODE_TO_RESULT.get(
                    session.error.code, CaptureSegmentResult.INTERNAL_ERROR
                )
                message = f"capture failed: {session.error}"
                terminal = (
                    "aborted",
                    _result(code, message, config=record.config or self._config),
                    message,
                )
            elif session.sealed is not None or session.frames_complete:
                # Seal and publish below without the server lock: every sensor
                # callback takes it, and fsync can stall for seconds on a busy disk.
                record.phase = "committing"
            else:
                record.phase = "terminalizing"
                terminal = (
                    "aborted",
                    _result(
                        CaptureSegmentResult.INTERNAL_ERROR,
                        "capture ended without a terminal outcome",
                        config=record.config or self._config,
                    ),
                    "capture ended without a terminal outcome",
                )
            if terminal is not None:
                self._claim_terminal_locked(record)

        if terminal is None:
            terminal = self._seal_and_commit(record, session, transaction)
            with self._lock:
                if self._active is not record or record.terminal_claimed:
                    return
                self._claim_terminal_locked(record)
        self._perform_terminal(record, *terminal)

    def _seal_and_commit(self, record, session, transaction):
        """Seal the completed window and publish it; runs without the server lock."""
        config = record.config or self._config
        session.finalize()
        if session.error is not None:
            code = _CAPTURE_CODE_TO_RESULT.get(session.error.code, CaptureSegmentResult.INTERNAL_ERROR)
            message = f"capture failed: {session.error}"
            return "aborted", _result(code, message, config=config), message
        text = "capture snapshot committed"
        message = "capture sealed snapshot committed"
        try:
            transaction.commit()
        except BaseException as error:
            if not transaction.committed:
                message = f"snapshot publication failed: {type(error).__name__}"
                return "aborted", _result(CaptureSegmentResult.IO_ERROR, message, config=config), message
            # renameat2 已经发布成功；之后的防御性检查/fsync 失败绝不把已发布
            # 快照当失败重试同一最终名。
            message = text = f"capture sealed snapshot committed with warning: {type(error).__name__}"
        with self._lock:
            record.committed = True
        return "succeeded", _result(CaptureSegmentResult.OK, message, sealed=session.sealed, config=config), text

    def _replay_handoff(self, record, cutoff):
        """Drain RGB with its own pose brackets; don't rescan history per image."""
        session = record.session
        last_odom = None
        odom_times, odom_messages, camera_info = [], [], []
        while not session.done.is_set() and not record.cancel_requested:
            with self._lock:
                item = self._handoff.pop_after(cutoff)
                if item is None:
                    record.handoff_replaying = False
                    return
                refresh = not odom_times or item[0] > odom_times[-1]
                if refresh:
                    odometry = list(self._odometry_history)
                    camera_info = list(self._camera_info_history)
            stamp, image = item
            if refresh:
                entries = []
                for message in odometry:
                    try:
                        entries.append((stamp_ns(message), message))
                    except (AttributeError, TypeError, ValueError):
                        continue
                entries.sort(key=lambda x: x[0])
                odom_times = [t for t, _ in entries]
                odom_messages = [m for _, m in entries]
            right = bisect_right(odom_times, stamp)
            first = max(0, right - 1) if last_odom is None else bisect_right(odom_times, last_odom)
            for k in range(first, min(len(odom_times), right + 1)):
                session.on_odometry(odom_messages[k])
                last_odom = odom_times[k]
            if session.done.is_set():
                with self._lock:
                    self._handoff.push(image, front=True)
                break
            valid_info = []
            for message in camera_info:
                try:
                    valid_info.append((abs(stamp_ns(message) - stamp), message))
                except (AttributeError, TypeError, ValueError):
                    continue
            if valid_info:
                session.on_camera_info(min(valid_info, key=lambda x: x[0])[1])
            session.on_image(image)
            cutoff = stamp
            with self._lock:
                self._handoff.replayed += 1
        with self._lock:
            record.handoff_replaying = False

    def _publish_feedback(self, record: _CaptureRecord, frame: CapturedFrame) -> None:
        """每选中一帧发布一次反馈（帧计数/最新 stamp/耗时）。"""
        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return
        feedback = CaptureSegmentFeedback()
        feedback.stage = "capturing"
        feedback.frame_count = frame.index + 1
        feedback.latest_stamp = rospy.Time(frame.stamp.sec, frame.stamp.nsec)
        feedback.elapsed = rospy.Duration.from_sec(
            max(0.0, time.monotonic() - record.started_monotonic)
        )
        feedback.message = "selected frame written"
        try:
            record.handle.publish_feedback(feedback)
        except BaseException:
            pass  # 反馈失败绝不影响采集主流程

    def _publish_source_state(
        self,
        record: _CaptureRecord,
        stage: str,
        stamp: FrameStamp | None,
    ) -> None:
        """source watchdog 状态迁移反馈（SOURCE_STALE/SOURCE_RECOVERED）。"""
        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return
            frame_count = (
                record.session.writer.frame_count
                if record.session is not None
                else 0
            )
        feedback = CaptureSegmentFeedback()
        feedback.stage = stage
        feedback.frame_count = frame_count
        if stamp is not None:
            feedback.latest_stamp = rospy.Time(stamp.sec, stamp.nsec)
        feedback.elapsed = rospy.Duration.from_sec(
            max(0.0, time.monotonic() - record.started_monotonic)
        )
        feedback.message = (
            "source watchdog: no RGB within source_timeout_sec"
            if stage == "SOURCE_STALE"
            else "source watchdog: RGB recovered with continuous stamp"
        )
        try:
            record.handle.publish_feedback(feedback)
        except BaseException:
            pass  # 反馈失败绝不影响采集主流程

    def _terminalize_rejected(self, record: _CaptureRecord, error: CaptureError) -> None:
        """把类型化采集错误映射为拒绝终态。"""
        code = _CAPTURE_CODE_TO_RESULT.get(
            error.code, CaptureSegmentResult.INTERNAL_ERROR
        )
        message = f"capture goal rejected: {error}"
        self._terminalize(
            record,
            "rejected",
            _result(code, message, config=record.config or self._config),
        )

    def _claim_terminal_locked(self, record: _CaptureRecord) -> None:
        """认领终态（持有锁）：保证终态过渡只执行一次。"""
        record.terminal_claimed = True
        record.phase = "terminalizing"
        self._terminal_attempt_count += 1

    def _set_goal_terminal(self, handle: Any, method: str, result: Any, text: str) -> None:
        """按 actionlib 实际实现调用终态方法。

        生产环境使用 ``actionlib.ActionServer``：它的 GoalHandle 没有
        ``set_preempted``，取消终态是 ``set_canceled``；SimpleActionServer /
        测试桩才有 ``set_preempted``。按能力回退，避免 AttributeError 导致
        目标永远停在 PREEMPTING。
        """
        preferred = "set_" + method
        if method == "preempted" and not hasattr(handle, preferred):
            preferred = "set_canceled"
        getattr(handle, preferred)(result, text)
        try:
            rospy.loginfo(
                "capture terminal called method=%s status=%s",
                preferred,
                getattr(handle.get_goal_status(), "status", None),
            )
        except BaseException:
            pass

    def _perform_terminal(
        self,
        record: _CaptureRecord,
        method: str,
        result: CaptureSegmentResult,
        text: str,
    ) -> None:
        """执行终态过渡：清理未提交事务，再交给共用执行器（观察者 → set_<method> → 确认，至多回退一次）。"""
        with self._lock:
            committed = record.committed
            transaction = record.transaction
        if transaction is not None and not committed:
            # 未提交的事务必须清理暂存目录
            try:
                transaction.cleanup()
            except BaseException as cleanup_error:
                status, _ = query_goal_status(record.handle)
                with self._lock:
                    accepted = record.accepted
                    cancelled = record.cancel_requested
                method = _legal_failure_method(status, accepted, cancelled)
                result = _result(
                    CaptureSegmentResult.IO_ERROR,
                    f"capture staging cleanup failed: {type(cleanup_error).__name__}",
                    config=self._config,
                )
                text = result.message

        def fallback(primary, status, _error):
            # 原方法在当前状态下仍合法（或状态读不到）就重试原方法；不合法就换成当前状态
            # 允许的失败方法，否则 actionlib 会再次拒绝、槽位永远占着。
            method = primary.method
            if status is not None and status not in _LEGAL_FROM.get(method, ()):
                with self._lock:
                    accepted = record.accepted
                    cancelled = record.cancel_requested
                method = _legal_failure_method(status, accepted, cancelled)
            return TerminalCall(method, primary.result, primary.result.message)

        observer = None
        if self._terminal_observer is not None:
            def observer(call):
                self._terminal_observer(call.method, call.result, call.text)

        outcome = converge_terminal(
            record.handle,
            TerminalCall(method, result, text),
            invoke=lambda handle, call: self._set_goal_terminal(
                handle, call.method, call.result, call.text
            ),
            fallback=fallback,
            observer=observer,
            log_context=f"capture goal {record.goal_id}",
        )
        # 除非 actionlib 证明已到终态，否则绝不释放容量：已提交快照仍可能归属 ACTIVE 目标。
        if outcome.converged:
            self._release_terminal(record, outcome.final_call.method, outcome.final_call.result)

    def _release_terminal(self, record, method, result):
        """记录终态摘要并释放槽位、置位完成事件。"""
        with self._lock:
            self._last_terminal = {
                "method": method,
                "error_code": int(result.error_code),
                "message": result.message,
                "scene_id": result.scene_id,
                "snapshot_uri": result.snapshot_uri,
                "frame_count": result.frame_count,
            }
            sealed = record.session.sealed if record.session is not None else None
            self._handoff_predecessor = (
                (result.scene_id, sealed.last_stamp.sec * 10**9 + sealed.last_stamp.nsec)
                if method == "succeeded" and sealed is not None else None
            )
            if self._active is record:
                self._active = None
        record.done.set()

    def _terminalize(
        self,
        record: _CaptureRecord,
        method: str,
        result: CaptureSegmentResult,
    ) -> None:
        """统一终态化入口（幂等，并发安全）。"""
        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return
            self._claim_terminal_locked(record)
        self._perform_terminal(record, method, result, result.message)

    def _unregister_subscribers(self) -> None:
        """注销全部话题订阅（关闭路径）。"""
        for subscriber in self._subscribers:
            try:
                subscriber.unregister()
            except BaseException:
                pass
        self._subscribers = []


def _result(
    error_code: int,
    message: str,
    sealed: Any = None,
    config: CaptureConfig | None = None,
) -> CaptureSegmentResult:
    """构造 CaptureSegmentResult 消息；sealed 非空时填充快照元数据。"""
    result = CaptureSegmentResult()
    result.error_code = error_code
    result.message = message
    if config is not None:
        result.camera_name = config.camera_name
        result.image_topic = config.image_topic
    if sealed is not None:
        result.scene_id = sealed.scene_id
        result.snapshot_uri = sealed.snapshot_uri
        result.manifest_uri = sealed.manifest_uri
        result.manifest_sha256 = sealed.manifest_sha256
        result.frame_count = sealed.frame_count
        result.first_stamp = rospy.Time(
            sealed.first_stamp.sec, sealed.first_stamp.nsec
        )
        result.last_stamp = rospy.Time(
            sealed.last_stamp.sec, sealed.last_stamp.nsec
        )
        result.has_camera_info = sealed.has_camera_info
        result.has_odometry = sealed.has_odometry
    return result


def _reject_goal(goal_handle: Any, result: CaptureSegmentResult) -> bool:
    """拒绝一个没有占到槽位的目标，并确认它到了终态。

    这个目标没有记录，拒绝失败后不会再有别的路径处理它，所以与占槽目标一样交给共用
    执行器：确认终态，没证实就再试一次（rejected 在当前状态下不合法时换成合法的失败方法）。
    在服务端锁外调用。
    """

    def fallback(primary, status, _error):
        method = primary.method
        if status is not None and status not in _LEGAL_FROM[method]:
            method = _legal_failure_method(status, False, False)
        return TerminalCall(method, primary.result, primary.result.message)

    outcome = converge_terminal(
        goal_handle,
        TerminalCall("rejected", result, result.message),
        invoke=lambda handle, call: getattr(handle, "set_" + call.method)(
            call.result, call.text
        ),
        fallback=fallback,
        log_context=f"capture contender {_goal_id(goal_handle)}",
    )
    return outcome.converged


def _goal_id(goal_handle: Any) -> str | None:
    """actionlib 目标号 ``get_goal_id().id``（与重建服务端相同的来源）；读不到时返回 None。"""
    try:
        return str(goal_handle.get_goal_id().id)
    except BaseException:
        return None


# actionlib ServerGoalHandle 各终态方法允许的起始状态（Noetic server_goal_handle.py）。
_LEGAL_FROM = {
    "rejected": (GoalStatus.PENDING, GoalStatus.RECALLING),
    "canceled": (GoalStatus.PENDING, GoalStatus.RECALLING, GoalStatus.ACTIVE, GoalStatus.PREEMPTING),
    "preempted": (GoalStatus.PENDING, GoalStatus.RECALLING, GoalStatus.ACTIVE, GoalStatus.PREEMPTING),
    "aborted": (GoalStatus.ACTIVE, GoalStatus.PREEMPTING),
    "succeeded": (GoalStatus.ACTIVE, GoalStatus.PREEMPTING),
}


def _legal_failure_method(status, accepted: bool, cancel_requested: bool) -> str:
    """按 actionlib 当前状态选一个合法的失败终态方法（与重建服务端同一取法）。

    ACTIVE/PREEMPTING 或已确认接受 → aborted；RECALLING → canceled；PENDING → 已请求取消
    则 canceled，否则 rejected；状态读不到且未接受 → 同 PENDING。
    """
    if status in (GoalStatus.ACTIVE, GoalStatus.PREEMPTING) or (status is None and accepted):
        return "aborted"
    if status == GoalStatus.RECALLING or cancel_requested:
        return "canceled"
    return "rejected"


def _make_daemon_thread(target, args) -> threading.Thread:
    """构造尚未启动的守护派发线程。"""
    return threading.Thread(target=target, args=args, name="capture-goal-task")


__all__ = ["CaptureActionServer"]
