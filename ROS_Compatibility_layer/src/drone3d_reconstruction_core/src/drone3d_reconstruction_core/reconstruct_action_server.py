"""底层 ROS Action 适配层：把单个重建 worker 桥接为 ActionServer。

本模块只负责 Action 目标仲裁与 worker 终态映射这两件事：
  - 快照校验、资源门禁、命令构造由调用方注入（command_builder）
  - worker 的进程生命周期由 JobController 托管，本模块只消费其
    JobFinished 回调并映射为 Action 的终态（succeeded/aborted/...）

约定与边界:
  - actionlib 会把"在对应 ActionGoal 送达前到达的取消"解析为 RECALLED
    并附带默认 Result，这类目标不会被派发到本适配器
  - 同一时刻只仲裁一个目标（单槽位）：第二个目标收到 BUSY 拒绝
  - 所有目标状态流转都必须收敛到某个终态（terminal state），
    且终态过渡最多尝试一次主方法 + 一次安全回退

设计要点:
  - 目标生命周期阶段（phase）驱动仲裁逻辑：starting → accepting →
    active → terminalizing，配合 cancel_requested / buffered_finished
    等标志处理"取消与完成竞争"的竞态
  - 所有共享状态在 self._lock 保护下读写；worker 终态回调可能来自
    任意线程，因此一律通过锁 + 阶段判断做幂等收敛
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import math
import re
import threading
import time
from typing import Any, Callable

import actionlib
from actionlib_msgs.msg import GoalStatus
import rospy

from drone3d_reconstruction_msgs.msg import (
    ReconstructSegmentAction,
    ReconstructSegmentFeedback,
    ReconstructSegmentResult,
)

from .action_terminal import TerminalCall, converge_terminal, is_terminal, query_goal_status
from .controller import BusyError, JobController, JobFinished
from .warm_worker_pool import WarmWorkerUnavailableError
from .preparation import (
    ALGORITHM_FAILED,
    INPUT_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_GOAL,
    IO_ERROR,
    CompletionError,
    PreparationError,
)
from .resource_gate import ResourceBusyError
from .result_transaction import ResultTransaction


_SCENE_ID_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?\Z"
)
_LOWER_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_URI_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*:[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+\Z"
)
_BAD_PERCENT_ESCAPE_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_MAX_FRAME_COUNT = (1 << 32) - 1

_PREPARATION_CODE_TO_RESULT = {
    INVALID_GOAL: ReconstructSegmentResult.INVALID_GOAL,
    INPUT_NOT_FOUND: ReconstructSegmentResult.INPUT_NOT_FOUND,
    IO_ERROR: ReconstructSegmentResult.IO_ERROR,
    INTERNAL_ERROR: ReconstructSegmentResult.INTERNAL_ERROR,
}
_COMPLETION_CODE_TO_RESULT = {
    ALGORITHM_FAILED: ReconstructSegmentResult.ALGORITHM_FAILED,
    IO_ERROR: ReconstructSegmentResult.IO_ERROR,
    INTERNAL_ERROR: ReconstructSegmentResult.INTERNAL_ERROR,
}


@dataclass(frozen=True)
class VerifiedCompletion:
    """一次成功重建的已校验、不可变元数据。

    该对象由完成校验器（ProductionCompletionVerifier）产出，最终映射到
    ReconstructSegmentResult 的成功字段。所有字段在 __post_init__ 中严格
    校验：scene_id 必须是安全标识符，URI 必须是绝对 ASCII URI，
    result_manifest_sha256 必须是 64 位小写十六进制摘要。

    Attributes:
        scene_id: 场景标识符
        result_manifest_uri: 最终 result.json 的 file:// URI
        result_manifest_sha256: result.json 的 SHA-256 摘要
        scene_glb_uri: 最终 scene.glb 的 file:// URI（未要求 GLB 时为空串）
        frame_count: 重建帧数
    """

    scene_id: str
    result_manifest_uri: str
    result_manifest_sha256: str
    scene_glb_uri: str
    frame_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.scene_id, str):
            raise TypeError("scene_id must be a string")
        if _SCENE_ID_RE.fullmatch(self.scene_id) is None:
            raise ValueError(
                "scene_id must be a safe identifier of at most 128 characters"
            )
        _validate_uri("result_manifest_uri", self.result_manifest_uri)
        if not isinstance(self.result_manifest_sha256, str):
            raise TypeError("result_manifest_sha256 must be a string")
        if _LOWER_SHA256_RE.fullmatch(self.result_manifest_sha256) is None:
            raise ValueError("result_manifest_sha256 must be a lowercase SHA-256 digest")
        _validate_uri("scene_glb_uri", self.scene_glb_uri, allow_empty=True)
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise TypeError("frame_count must be an integer")
        if not 1 <= self.frame_count <= _MAX_FRAME_COUNT:
            raise ValueError("frame_count must be in the uint32 positive range")


def _validate_uri(name: str, value: str, *, allow_empty: bool = False) -> None:
    """校验绝对 ASCII URI（不查询任何外部状态）。

    允许空串（仅当 allow_empty=True，用于未要求 GLB 时的 scene_glb_uri）；
    拒绝错误百分号转义与相对 URI。
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value == "" and allow_empty:
        return
    if _URI_RE.fullmatch(value) is None or _BAD_PERCENT_ESCAPE_RE.search(value):
        suffix = " or empty" if allow_empty else ""
        raise ValueError(f"{name} must be an absolute ASCII URI{suffix}")


@dataclass(frozen=True)
class PreparedLaunch:
    """不可变 worker 命令与其绑定的完成校验器。

    由调用方（ProductionPlanner）构造：command 是一次性 worker 的精确
    argv（warm 不可用时的回退路径），warm_command + warm_job 为驻留
    warm worker 的可选启动路径；result_transaction 是监督进程拥有的结果
    事务，verify_completion 是校验回调
    (JobFinished, 取消事件) → VerifiedCompletion。
    """

    command: tuple[str, ...]
    result_transaction: ResultTransaction
    verify_completion: Callable[
        [JobFinished, threading.Event], VerifiedCompletion
    ]
    warm_command: tuple[str, ...] | None = None
    warm_job: dict[str, Any] | None = None
    preserve_uncommitted_staging: bool = False
    # Seconds spent in each preparation step, logged once per goal.
    timings: dict[str, float] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if type(self.preserve_uncommitted_staging) is not bool:
            raise TypeError("preserve_uncommitted_staging must be a bool")
        if not isinstance(self.command, tuple) or not self.command:
            raise TypeError("command must be a nonempty tuple")
        if any(not isinstance(argument, str) or not argument for argument in self.command):
            raise ValueError("command entries must be nonempty strings")
        if (self.warm_command is None) != (self.warm_job is None):
            raise ValueError("warm_command and warm_job must be provided together")
        if self.warm_command is not None:
            if not isinstance(self.warm_command, tuple) or not self.warm_command:
                raise TypeError("warm_command must be a nonempty tuple")
            if any(
                not isinstance(argument, str) or not argument
                for argument in self.warm_command
            ):
                raise ValueError("warm_command entries must be nonempty strings")
            if not isinstance(self.warm_job, dict) or "job_id" in self.warm_job:
                raise TypeError(
                    "warm_job must be a dict without a job_id entry"
                )
            try:
                json.dumps(self.warm_job, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise ValueError("warm_job must be JSON serializable") from error
        if not isinstance(self.result_transaction, ResultTransaction):
            raise TypeError("result_transaction must be a ResultTransaction")
        if not callable(self.verify_completion):
            raise TypeError("verify_completion must be callable")


@dataclass
class _GoalRecord:
    """单个 Action 目标在仲裁期间的完整状态记录。

    Attributes:
        handle: actionlib 的目标句柄（goal handle）
        key: 目标唯一键 (goal_id.id, stamp.secs, stamp.nsecs)
        phase: 目标所处的仲裁阶段，取值 starting / accepting / active /
            accept_failed / verifying / committing / terminalizing
        cancel_requested: 是否已收到取消请求
        buffered_finished: 在目标被接受前 worker 就结束时的缓冲结果
        accept_error: 接受目标失败时的错误描述
        terminal_claimed: 是否已认领终态（保证终态过渡只执行一次）
        accepted: 目标是否已被接受（set_accepted 成功）
        verification_cancel: 仅属于该目标的验证取消事件
        done: 终态收敛后置位的事件，供 shutdown 阻塞等待
    """

    handle: Any
    key: tuple[str, int, int]
    phase: str = "starting"
    cancel_requested: bool = False
    buffered_finished: JobFinished | None = None
    accept_error: str | None = None
    terminal_claimed: bool = False
    completion_claimed: bool = False
    result_committed: bool = False
    staging_cleaned: bool = False
    finished_observed: bool = False
    prepared_launch: PreparedLaunch | None = None
    controller_job_id: int | None = None
    image_resolution_override: int | None = None
    worker_timeout_sec_override: float | None = None
    accepted: bool = False
    launch_permitted: bool = False
    started_monotonic: float = field(default_factory=time.monotonic)
    launch_decided: threading.Event = field(default_factory=threading.Event)
    launch_waiting: threading.Event = field(default_factory=threading.Event)
    launch_finished: threading.Event = field(default_factory=threading.Event)
    verification_cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)


class ReconstructActionServer:
    """把底层 ActionServer 桥接到一个注入的 worker 命令。

    仲裁规则（单槽位）：
      - 新目标到来时，若已有活跃目标则返回 BUSY 拒绝
      - 目标被接受前到达的取消 → 目标直接以 CANCELLED 收敛，绝不接受
      - 目标被接受后到达的取消 → 转发给 JobController.cancel()
      - worker 结束（JobFinished）按 cause/returncode 映射到终态

    使用方式::

        server = ReconstructActionServer(
            action_name="reconstruct_segment",
            command_builder=build_prepared_launch,       # goal -> PreparedLaunch
            worker_timeout_sec=600.0,
            term_grace_sec=5.0,
        )
    """

    def __init__(
        self,
        *,
        action_name: str,
        command_builder: Callable[[Any], PreparedLaunch],
        worker_timeout_sec: float,
        term_grace_sec: float,
        controller_factory=JobController,
        action_server_factory=actionlib.ActionServer,
        launch_dispatch=None,
        terminal_observer: Callable[[str, ReconstructSegmentResult, str], None] | None = None,
        finished_observer: Callable[[JobFinished], None] | None = None,
    ) -> None:
        """构造 Action 服务端并启动（auto_start=False 后手动 start）。

        Args:
            action_name: Action 服务端注册的名称（如 "reconstruct_segment"）
            command_builder: 由 goal 对象构造不可变 PreparedLaunch 的回调
            worker_timeout_sec: worker 最长运行时长（秒），透传给 JobController
            term_grace_sec: SIGTERM 后的优雅退出宽限时长（秒）
            controller_factory: JobController 工厂（测试注入点）
            action_server_factory: actionlib ActionServer 工厂（测试注入点）
            launch_dispatch: 目标派发函数，默认在守护线程中执行
            terminal_observer: 最佳努力终态观察者 (method, result, text)，异常被
                记录后忽略，绝不影响 Action 收敛
            finished_observer: 最佳努力 worker 完成观察者 (JobFinished)，每个
                匹配的 worker 完成最多调用一次，异常被记录后忽略
        """
        for name, observer in (
            ("terminal_observer", terminal_observer),
            ("finished_observer", finished_observer),
        ):
            if observer is not None and not callable(observer):
                raise TypeError(f"{name} must be callable or None")
        self._command_builder = command_builder
        self._terminal_observer = terminal_observer
        self._finished_observer = finished_observer
        self._launch_dispatch = launch_dispatch or _make_daemon_thread
        self._lock = threading.Lock()
        self._active: _GoalRecord | None = None
        self._shutdown = False
        self._accepted_count = 0
        self._terminal_attempt_count = 0
        self._busy_rejection_count = 0
        self._warm_fallback_count = 0
        self._cancel_request_count = 0
        self._server_stopped = False
        self._controller = controller_factory(
            worker_timeout_sec=worker_timeout_sec,
            term_grace_sec=term_grace_sec,
            on_finished=self._on_finished,
        )
        try:
            self._server = action_server_factory(
                action_name,
                ReconstructSegmentAction,
                self._goal_callback,
                self._cancel_callback,
                auto_start=False,
            )
            self._server.start()
        except BaseException:
            try:
                self._controller.shutdown()
            except BaseException as shutdown_error:
                rospy.logerr(
                    "Controller shutdown failed during ActionServer construction: %s",
                    type(shutdown_error).__name__,
                )
            raise

    @property
    def accepted_count(self) -> int:
        """已成功接受（set_accepted）的目标累计数。"""
        with self._lock:
            return self._accepted_count

    @property
    def terminal_attempt_count(self) -> int:
        """已发起终态过渡尝试的累计次数。"""
        with self._lock:
            return self._terminal_attempt_count

    @property
    def busy_rejection_count(self) -> int:
        """因单槽位忙而拒绝的目标累计数。"""
        with self._lock:
            return self._busy_rejection_count

    @property
    def cancel_request_count(self) -> int:
        """已受理的取消请求累计数。"""
        with self._lock:
            return self._cancel_request_count

    def is_busy(self) -> bool:
        """当前是否正有目标在仲裁（含启动/运行/终态化阶段）。

        Returns:
            存在活跃目标时为 True
        """
        with self._lock:
            return self._active is not None

    def release_idle_worker(self):
        with self._lock:
            if self._active is not None:
                raise BusyError("reconstruction goal is still active")
            release = getattr(self._controller, "release_idle_worker", None)
            if release is not None:
                release()

    def resident_worker_pid(self):
        method = getattr(self._controller, "resident_worker_pid", None)
        return method() if method else None

    def diagnostics_snapshot(self) -> dict:
        """Thread-safe telemetry snapshot for the supervisor diagnostics node."""
        with self._lock:
            record = self._active
            return {
                "running": record is not None,
                "stage": record.phase if record is not None else "idle",
                "job_id": (
                    record.controller_job_id if record is not None else None
                ),
                "cancel_requested": bool(
                    record.cancel_requested if record is not None else False
                ),
                "accepted": bool(record.accepted if record is not None else False),
                "accepted_count": self._accepted_count,
                "busy_rejection_count": self._busy_rejection_count,
                "warm_fallback_count": self._warm_fallback_count,
                "cancel_request_count": self._cancel_request_count,
                "terminal_attempt_count": self._terminal_attempt_count,
            }

    def _publish_feedback(
        self, record: _GoalRecord, stage: str, progress: float
    ) -> None:
        """最佳努力发布阶段 feedback；绝不依赖其成功，也绝不抛异常。

        progress 按 Action 契约是 best effort，不是 ETA。本适配器没有真实
        推理进度源，因此只发布稳定的 phase 标签，progress 恒为 0。
        """
        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return
            elapsed = max(0.0, time.monotonic() - record.started_monotonic)
        feedback = ReconstructSegmentFeedback()
        feedback.stage = stage
        feedback.progress = float(progress)
        feedback.elapsed = rospy.Duration.from_sec(elapsed)
        feedback.message = stage
        try:
            record.handle.publish_feedback(feedback)
        except BaseException:
            pass  # feedback 失败绝不影响 Action 收敛

    def shutdown(self, timeout_sec: float = 3.0) -> None:
        """关闭服务端：先停止 worker，再等待活跃目标收敛到终态。

        Args:
            timeout_sec: 等待目标收敛的最长时间（秒）

        Raises:
            RuntimeError: 目标未在期限内达到终态
        """
        deadline = time.monotonic() + timeout_sec
        with self._lock:
            self._shutdown = True
            record = self._active
            if record is not None and not record.terminal_claimed:
                record.cancel_requested = True
                if record.phase == "verifying":
                    record.verification_cancel.set()
                record.launch_permitted = False
                record.launch_decided.set()

        # A controller completion is the only proof that a published job id is
        # quiescent.  Stop it before allowing shutdown to claim any terminal.
        self._controller.shutdown()

        if record is None:
            self._stop_action_server()
            return
        remaining = max(0.0, deadline - time.monotonic())
        if not record.launch_finished.wait(remaining):
            raise RuntimeError("Action launch task did not stop during shutdown")

        terminal = None
        with self._lock:
            if self._active is record and not record.terminal_claimed:
                if record.controller_job_id is not None:
                    # Never fail open: the matching JobFinished callback must
                    # prove process-group quiescence and own terminalization.
                    terminal = None
                else:
                    terminal = self._claim_cancel_terminal_locked(
                        record,
                        "reconstruction server is shutting down",
                    )
        if terminal is not None:
            self._perform_terminal_transition(record, *terminal)

        remaining = max(0.0, deadline - time.monotonic())
        if not record.done.wait(remaining):
            raise RuntimeError("Action goal did not reach a terminal state during shutdown")
        self._stop_action_server()

    def _stop_action_server(self) -> None:
        """Stop actionlib timers and ROS endpoints exactly once."""
        with self._lock:
            if self._server_stopped:
                return
            self._server_stopped = True
        self._server.stop()

    def _goal_callback(self, goal_handle: Any) -> None:
        """新目标到达回调：登记记录并做单槽位仲裁。

        处理流程：
          1. 关闭中 → INTERNAL_ERROR 拒绝；已有活跃目标 → BUSY 拒绝
          2. 否则把记录登记为活跃目标，并检查目标是否已处于
             RECALLING（说明取消先于本回调到达，需登记 cancel_requested）
          3. 通过 launch_dispatch 派发启动流程（默认守护线程）

        Args:
            goal_handle: actionlib 目标句柄
        """
        record = _GoalRecord(goal_handle, _goal_key(goal_handle))
        rejection = None
        try:
            initial_status = goal_handle.get_goal_status().status
        except BaseException:
            initial_status = None
        with self._lock:
            if self._shutdown:
                rejection = _result(
                    ReconstructSegmentResult.INTERNAL_ERROR,
                    "reconstruction server is shutting down",
                )
            elif self._active is not None:
                self._busy_rejection_count += 1
                rejection = _result(
                    ReconstructSegmentResult.BUSY,
                    "another reconstruction worker is active",
                )
            else:
                self._active = record
                record.cancel_requested = initial_status == GoalStatus.RECALLING

        if rejection is not None:
            self._reject_nonowned_goal(goal_handle, rejection)
            return

        try:
            launch_thread = self._launch_dispatch(
                self._launch_goal_task, (record,)
            )
            if not isinstance(launch_thread, threading.Thread):
                raise TypeError("launch_dispatch must return an unstarted Thread")
            if launch_thread.ident is not None or launch_thread.is_alive():
                raise RuntimeError("launch_dispatch returned an already-started Thread")
            launch_thread.daemon = True
            launch_thread.start()
        except BaseException as error:
            with self._lock:
                if self._active is record:
                    record.launch_permitted = False
                    record.launch_decided.set()
            record.launch_finished.set()
            self._terminalize_dispatch_failure(record, error)
            return

        with self._lock:
            if self._active is record and not record.terminal_claimed:
                record.launch_permitted = True
            record.launch_decided.set()

    @staticmethod
    def _reject_nonowned_goal(
        goal_handle: Any, result: ReconstructSegmentResult
    ) -> bool:
        """把 BUSY/关闭竞争的未归属目标收敛到终态，不触碰活跃槽位。

        先尝试 set_rejected；若目标处于 RECALLING（取消先到）则改用
        set_canceled 收敛；最后查询状态确认是否已到达终态。

        Returns:
            True 表示目标已收敛到终态
        """
        try:
            goal_handle.set_rejected(result, result.message)
        except BaseException:
            pass
        try:
            status = goal_handle.get_goal_status().status
        except BaseException:
            status = None
        if is_terminal(status):
            return True

        if status == GoalStatus.RECALLING:
            fallback = "canceled"
            fallback_result = _result(
                ReconstructSegmentResult.CANCELLED,
                "contending goal was cancelled before ownership",
            )
        else:
            fallback = "rejected"
            fallback_result = result
        try:
            getattr(goal_handle, f"set_{fallback}")(
                fallback_result, fallback_result.message
            )
        except BaseException:
            pass
        try:
            converged = _is_terminal(goal_handle)
        except BaseException:
            converged = False
        if not converged:
            rospy.logerr("Action rejection failed for non-owned goal")
        return converged

    def _terminalize_dispatch_failure(
        self, record: _GoalRecord, error: BaseException
    ) -> None:
        """派发线程创建失败时的兜底：把目标收敛为拒绝/取消。

        若取消请求恰好先到，则按已取消收敛（CANCELLED），
        否则按内部错误拒绝（INTERNAL_ERROR），并释放容量。

        Args:
            record: 目标记录
            error: 派发时抛出的异常（仅用于构造错误消息）
        """
        with self._lock:
            terminal = self._claim_dispatch_failure_terminal_locked(record, error)
        if terminal is not None:
            self._perform_terminal_transition(record, *terminal)

    def _claim_dispatch_failure_terminal_locked(
        self, record: _GoalRecord, error: BaseException
    ) -> tuple[str, ReconstructSegmentResult, str] | None:
        """Choose and claim dispatch failure versus a preceding cancellation."""
        if self._active is not record or record.terminal_claimed:
            return None
        message = f"worker launch dispatch failed: {type(error).__name__}"
        if record.cancel_requested:
            terminal = (
                "canceled",
                _result(ReconstructSegmentResult.CANCELLED, message),
                message,
            )
        else:
            terminal = (
                "rejected",
                _result(ReconstructSegmentResult.INTERNAL_ERROR, message),
                message,
            )
        self._claim_terminal_locked(record)
        return terminal

    def _launch_goal_task(self, record: _GoalRecord) -> None:
        try:
            self._launch_goal(record)
        except BaseException as error:
            converged = self._converge_unexpected_launch_exception(record, error)
            if not isinstance(error, Exception) and converged:
                raise
        finally:
            record.launch_finished.set()

    def _converge_unexpected_launch_exception(
        self, record: _GoalRecord, error: BaseException
    ) -> bool:
        """Cancel a published job id and wait for its matching completion."""
        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return record.done.is_set()
            has_job = record.controller_job_id is not None
        if has_job:
            self._begin_post_start_failure(
                record,
                f"worker launch task failed: {type(error).__name__}",
            )
            return record.done.is_set()

        message = f"worker launch task failed: {type(error).__name__}"
        self._terminalize(
            record,
            "rejected",
            _result(ReconstructSegmentResult.INTERNAL_ERROR, message),
            message,
        )
        return record.done.is_set()

    def _begin_post_start_failure(self, record: _GoalRecord, message: str) -> None:
        """Wait for JobFinished after any fault occurring after job-id publication."""
        try:
            status = record.handle.get_goal_status().status
        except BaseException:
            status = None
        terminal = None
        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return
            if record.controller_job_id is None:
                return
            record.accept_error = message
            was_accepted = record.accepted
            record.accepted = record.accepted or status in (
                GoalStatus.ACTIVE,
                GoalStatus.PREEMPTING,
            )
            if record.accepted and not was_accepted:
                self._accepted_count += 1
            record.phase = "accept_failed"
            buffered_finished = _matching_buffered_finished(record)
            if buffered_finished is not None:
                terminal = self._claim_accept_failure_terminal_locked(
                    record, buffered_finished.elapsed_sec
                )
        try:
            self._controller.cancel()
        except BaseException as cancel_error:
            rospy.logerr(
                "Controller cancellation failed after launch task error: %s",
                type(cancel_error).__name__,
            )
        if terminal is not None:
            self._perform_terminal_transition(record, *terminal)

    def _launch_goal(self, record: _GoalRecord) -> None:
        """目标启动流程（在派发线程中执行）：构造命令并启动 worker。

        步骤：
          1. command_builder(goal) 构造命令，失败 → 拒绝/取消
          2. 启动前复查取消请求，有则直接以 CANCELLED 收敛
          3. JobController.start 启动 worker（BusyError → BUSY 拒绝）
          4. 无取消则 set_accepted，校验状态后进入 active 阶段
          5. 处理启动期间可能已缓冲的 worker 结束回调

        Args:
            record: 目标记录
        """
        record.launch_waiting.set()
        record.launch_decided.wait()
        with self._lock:
            if (
                self._active is not record
                or record.terminal_claimed
                or not record.launch_permitted
            ):
                return
        goal_handle = record.handle
        try:
            goal = goal_handle.get_goal()
            image_resolution, worker_timeout_sec = _parse_goal_overrides(goal)
            with self._lock:
                if self._active is record and not record.terminal_claimed:
                    record.image_resolution_override = image_resolution
                    record.worker_timeout_sec_override = worker_timeout_sec
            builder_kwargs = {}
            if image_resolution is not None:
                builder_kwargs["image_resolution"] = image_resolution
            if worker_timeout_sec is not None:
                builder_kwargs["worker_timeout_sec"] = worker_timeout_sec
            prepare_started = time.monotonic()
            prepared_launch = self._command_builder(goal, **builder_kwargs)
            prepare_sec = time.monotonic() - prepare_started
            if not isinstance(prepared_launch, PreparedLaunch):
                raise TypeError("command_builder must return PreparedLaunch")
        except ResourceBusyError as error:
            self._terminalize_resource_busy(record, error)
            return
        except BaseException as error:
            self._terminalize_launch_failure(record, error)
            return

        with self._lock:
            owns_record = self._active is record and not record.terminal_claimed
            if owns_record:
                record.prepared_launch = prepared_launch
            launch_allowed = owns_record and record.launch_permitted
            cancelled_before_launch = launch_allowed and record.cancel_requested
        if not launch_allowed:
            if owns_record:
                self._terminalize(
                    record,
                    "canceled",
                    _result(
                        ReconstructSegmentResult.CANCELLED,
                        "reconstruction server is shutting down",
                    ),
                    "reconstruction server is shutting down",
                )
            else:
                cleanup_error = self._cleanup_result_staging(prepared_launch)
                if cleanup_error is not None:
                    rospy.logerr(
                        "Result staging cleanup failed after ownership changed for goal %s: %s",
                        record.key[0],
                        type(cleanup_error).__name__,
                    )
            return
        if cancelled_before_launch:
            self._terminalize(
                record,
                "canceled",
                _result(ReconstructSegmentResult.CANCELLED, "cancelled before launch"),
                "cancelled before launch",
            )
            return

        def record_job_id(job_id: int) -> None:
            with self._lock:
                if self._active is record and not record.terminal_claimed:
                    record.controller_job_id = job_id

        try:
            controller_kwargs = {"on_started": record_job_id}
            if record.worker_timeout_sec_override is not None:
                controller_kwargs["worker_timeout_sec"] = (
                    record.worker_timeout_sec_override
                )
            warm_job = prepared_launch.warm_job
            start_warm = getattr(self._controller, "start_warm", None)
            if warm_job is not None and callable(start_warm):
                try:
                    controller_job_id = start_warm(
                        prepared_launch.warm_command,
                        warm_job,
                        **controller_kwargs,
                    )
                except WarmWorkerUnavailableError as error:
                    # warm 进程同步 spawn 失败发生在 job_id 发布之前，
                    # 暂存目录仍洁净：安全退回一次性 worker 路径。
                    # 一次性 worker 每窗重新加载模型，更慢、峰值内存也不同，所以要出声。
                    with self._lock:
                        self._warm_fallback_count += 1
                    rospy.logwarn(
                        "warm worker unavailable (%s); this window uses a one-shot worker "
                        "that reloads the model", error)
                    controller_job_id = self._controller.start(
                        prepared_launch.command,
                        **controller_kwargs,
                    )
            else:
                controller_job_id = self._controller.start(
                    prepared_launch.command,
                    **controller_kwargs,
                )
            with self._lock:
                if self._active is record and not record.terminal_claimed:
                    record.controller_job_id = controller_job_id
            rospy.loginfo(
                "reconstruct launch timing: prepare %.2f s (%s), worker start %.2f s",
                prepare_sec,
                ", ".join(f"{k} {v:.2f}" for k, v in prepared_launch.timings.items()),
                time.monotonic() - prepare_started - prepare_sec,
            )
        except BusyError as error:
            with self._lock:
                published_job = record.controller_job_id is not None
            if published_job:
                self._begin_post_start_failure(
                    record,
                    f"worker launch failed: {type(error).__name__}",
                )
            else:
                self._terminalize_launch_failure(record, error)
            return
        except BaseException as error:
            with self._lock:
                published_job = record.controller_job_id is not None
            if published_job:
                self._begin_post_start_failure(
                    record,
                    f"worker launch failed: {type(error).__name__}",
                )
            else:
                self._terminalize_launch_failure(record, error)
            return

        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return
            cancel_before_accept = record.cancel_requested
            if not cancel_before_accept:
                record.phase = "accepting"

        if not cancel_before_accept:
            self._publish_feedback(record, "accepting", 0.0)

        if cancel_before_accept:
            # Cancellation is asynchronous.  Capacity and staging remain owned
            # until the matching completion proves process-group quiescence.
            terminal = None
            with self._lock:
                buffered_finished = _matching_buffered_finished(record)
                if buffered_finished is not None:
                    terminal = self._claim_cancel_terminal_locked(
                        record,
                        "cancelled during launch",
                        buffered_finished.elapsed_sec,
                    )
            if terminal is not None:
                self._perform_terminal_transition(record, *terminal)
            else:
                self._controller.cancel()
            return

        try:
            goal_handle.set_accepted("worker started")
        except BaseException as error:
            self._begin_accept_failure(
                record,
                f"failed to accept goal: {type(error).__name__}",
            )
            return

        # A successful set_accepted call is authoritative even if the following
        # defensive status query fails.  Never later downgrade this fact.
        with self._lock:
            if self._active is record and not record.terminal_claimed:
                if not record.accepted:
                    record.accepted = True
                    self._accepted_count += 1

        status = goal_handle.get_goal_status().status
        if status not in (GoalStatus.ACTIVE, GoalStatus.PREEMPTING):
            self._begin_accept_failure(
                record,
                f"failed to accept goal: unexpected status {status}",
            )
            return

        terminal = None
        verification = None
        cancel_requested = False
        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return
            record.phase = "active"
            buffered_finished = _matching_buffered_finished(record)
            if buffered_finished is not None:
                terminal, verification = self._claim_finished_locked(
                    record, buffered_finished
                )
            else:
                cancel_requested = record.cancel_requested

        if verification is None and terminal is None:
            self._publish_feedback(record, "active", 0.0)

        # Acceptance and a buffered completion are arbitrated under one lock:
        # cancellation already recorded wins, while a later cancellation cannot
        # revoke a claimed completion or run its verifier twice.
        if terminal is not None:
            self._perform_terminal_transition(record, *terminal)
            return
        if verification is not None:
            self._run_completion_verifier(record, buffered_finished, verification)
            return
        if cancel_requested:
            self._controller.cancel()

    def _begin_accept_failure(
        self, record: _GoalRecord, message: str
    ) -> None:
        """set_accepted 失败或状态异常的兜底：尝试取消 worker 并终态化。

        先记录 accept_error 与是否已处于活跃状态（此时目标在 actionlib
        侧实际已被接受），再请求控制器取消；若 worker 恰好已结束，
        则使用其运行时长收敛终态。

        Args:
            record: 目标记录
            message: 接受失败的错误描述
        """
        try:
            status = record.handle.get_goal_status().status
        except BaseException:
            status = None
        with self._lock:
            if self._active is not record or record.terminal_claimed:
                return
            record.accept_error = message
            was_accepted = record.accepted
            record.accepted = record.accepted or status in (
                GoalStatus.ACTIVE,
                GoalStatus.PREEMPTING,
            )
            if record.accepted and not was_accepted:
                self._accepted_count += 1
            record.phase = "accept_failed"
            buffered_finished = _matching_buffered_finished(record)
        if self._controller.cancel():
            return
        if buffered_finished is not None:
            self._terminalize_accept_failure(record, buffered_finished.elapsed_sec)

    def _cancel_callback(self, goal_handle: Any) -> None:
        """取消请求回调：登记取消意图并按阶段决定是否转发给控制器。

        只受理当前活跃目标（按 key 匹配，避免陈旧的取消误伤新目标）；
        active 阶段在锁外请求控制器停止 worker，verifying 阶段在锁内
        置位该目标的验证取消事件，其余阶段由启动流程消费取消标记。

        Args:
            goal_handle: 目标句柄（用于匹配当前活跃目标）
        """
        should_cancel = False
        cancel_key = _goal_key(goal_handle)
        with self._lock:
            record = self._active
            if (
                record is None
                or record.terminal_claimed
                or record.key != cancel_key
            ):
                return
            if record.completion_claimed and record.phase != "verifying":
                return
            record.cancel_requested = True
            self._cancel_request_count += 1
            if record.phase == "verifying":
                record.verification_cancel.set()
            should_cancel = record.phase == "active" and not record.completion_claimed
        if should_cancel:
            self._controller.cancel()

    def _on_finished(self, finished: JobFinished) -> None:
        """JobController 终态回调：按目标所处阶段路由收敛路径。

        三种情形：
          - accept_failed 阶段 → 按接受失败收敛（终态化）
          - starting 阶段且已请求取消 → 按"启动期间取消"收敛
          - 尚未接受（starting/accepting）→ 缓冲结果，待接受后处理
          - 已接受（active）→ 立即终态化

        Args:
            finished: worker 结束摘要
        """
        terminal = None
        verification = None
        observed_finished = None
        buffered = False
        with self._lock:
            record = self._active
            if (
                record is None
                or record.terminal_claimed
                or record.controller_job_id is None
                or finished.job_id != record.controller_job_id
            ):
                return
            if not record.finished_observed:
                record.finished_observed = True
                observed_finished = finished
            if record.phase == "accept_failed":
                terminal = self._claim_accept_failure_terminal_locked(
                    record, finished.elapsed_sec
                )
            elif record.phase in ("starting", "accepting"):
                if record.phase == "starting" and record.cancel_requested:
                    terminal = self._claim_cancel_terminal_locked(
                        record,
                        "cancelled during launch",
                        finished.elapsed_sec,
                    )
                else:
                    # JobController promises exactly one completion. At this
                    # boundary, retain the first matching event so a duplicate
                    # cannot rewrite an authoritative pre-accept outcome.
                    if record.buffered_finished is None:
                        record.buffered_finished = finished
                    buffered = True
            else:
                terminal, verification = self._claim_finished_locked(record, finished)
        if observed_finished is not None and self._finished_observer is not None:
            try:
                self._finished_observer(observed_finished)
            except BaseException as observer_error:
                rospy.logerr(
                    "finished observer failed: %s",
                    type(observer_error).__name__,
                )
        if buffered:
            return
        if terminal is not None:
            self._perform_terminal_transition(record, *terminal)
        elif verification is not None:
            self._run_completion_verifier(record, finished, verification)

    def _terminalize_resource_busy(
        self, record: _GoalRecord, error: ResourceBusyError
    ) -> None:
        """Atomically choose cancellation or resource rejection and claim it."""
        with self._lock:
            terminal = self._claim_resource_busy_terminal_locked(record, error)
        if terminal is not None:
            self._perform_terminal_transition(record, *terminal)

    def _claim_resource_busy_terminal_locked(
        self, record: _GoalRecord, error: ResourceBusyError
    ) -> tuple[str, ReconstructSegmentResult, str] | None:
        """Choose and claim the resource-denial terminal while holding the lock."""
        if self._active is not record or record.terminal_claimed:
            return None
        if record.cancel_requested:
            method = "canceled"
            result = _result(
                ReconstructSegmentResult.CANCELLED,
                "cancelled during preparation",
            )
            text = "cancelled during preparation"
        else:
            decision = error.decision
            text = f"resource gate denied launch: {decision.denial_code}"
            method = "rejected"
            result = _result(ReconstructSegmentResult.BUSY, text)
        self._claim_terminal_locked(record)
        return method, result, text

    def _terminalize_launch_failure(
        self, record: _GoalRecord, error: BaseException
    ) -> None:
        """worker 启动失败（命令构造或 spawn 异常）的终态收敛。

        若取消请求先到则按已取消收敛，否则按内部错误拒绝。

        Args:
            record: 目标记录
            error: 启动异常（仅用于构造错误消息）
        """
        with self._lock:
            terminal = self._claim_launch_failure_terminal_locked(record, error)
        if terminal is not None:
            self._perform_terminal_transition(record, *terminal)

    def _claim_launch_failure_terminal_locked(
        self, record: _GoalRecord, error: BaseException
    ) -> tuple[str, ReconstructSegmentResult, str] | None:
        """Choose and claim launch failure versus a preceding cancellation."""
        if self._active is not record or record.terminal_claimed:
            return None
        if record.cancel_requested:
            terminal = self._cancel_terminal(
                "cancelled during launch"
            )
        else:
            if isinstance(error, BusyError):
                message = "worker capacity is busy"
                error_code = ReconstructSegmentResult.BUSY
            elif isinstance(error, PreparationError):
                error_code = _PREPARATION_CODE_TO_RESULT.get(
                    error.code, ReconstructSegmentResult.INTERNAL_ERROR
                )
                message = f"goal preparation failed: {error}"
            else:
                message = f"worker launch failed: {type(error).__name__}"
                error_code = ReconstructSegmentResult.INTERNAL_ERROR
            terminal = (
                "rejected",
                _result(error_code, message),
                message,
            )
        self._claim_terminal_locked(record)
        return terminal

    def _terminalize_accept_failure(
        self, record: _GoalRecord, elapsed_sec: float
    ) -> None:
        """接受失败后的终态收敛：按已接受/已取消/未接受三分支处理。

        - 已接受（actionlib 侧已是 ACTIVE）→ aborted
        - 已请求取消 → canceled
        - 其余 → rejected

        Args:
            record: 目标记录
            elapsed_sec: worker 已运行时长（秒），写入结果
        """
        with self._lock:
            terminal = self._claim_accept_failure_terminal_locked(
                record, elapsed_sec
            )
        if terminal is not None:
            self._perform_terminal_transition(record, *terminal)

    def _claim_accept_failure_terminal_locked(
        self, record: _GoalRecord, elapsed_sec: float
    ) -> tuple[str, ReconstructSegmentResult, str] | None:
        """Choose and claim an accept-failure terminal while holding the lock."""
        if self._active is not record or record.terminal_claimed:
            return None
        message = record.accept_error or "failed to accept goal"
        # Once actionlib accepted the goal, the accept contract is broken and
        # ABORTED remains authoritative even if cancellation was recorded later.
        if record.accepted:
            method = "aborted"
            error_code = ReconstructSegmentResult.INTERNAL_ERROR
        elif record.cancel_requested:
            method = "canceled"
            error_code = ReconstructSegmentResult.CANCELLED
        else:
            method = "rejected"
            error_code = ReconstructSegmentResult.INTERNAL_ERROR
        self._claim_terminal_locked(record)
        return method, _result(error_code, message, elapsed_sec), message


    def _claim_finished_locked(
        self, record: _GoalRecord, finished: JobFinished
    ) -> tuple[
        tuple[str, ReconstructSegmentResult, str] | None,
        Callable[
            [JobFinished, threading.Event], VerifiedCompletion
        ] | None,
    ]:
        """持有锁时认领匹配的完成身份，并选择终态或进入验证阶段。

        取消通常优先；但进程组静默校验失败必须保留为失败，不能伪装成
        CANCELLED。worker 正常退出（exit 0）→ 返回验证回调进入 verifying
        阶段；否则直接映射失败终态。
        """
        if (
            self._active is not record
            or record.terminal_claimed
            or record.completion_claimed
            or record.controller_job_id is None
            or finished.job_id != record.controller_job_id
        ):
            return None, None

        record.completion_claimed = True
        if record.cancel_requested and finished.cause != "cleanup_failed":
            terminal = self._cancel_terminal(
                "reconstruction cancelled", finished.elapsed_sec
            )
            self._claim_terminal_locked(record)
            return terminal, None

        if finished.cause == "worker_exit" and finished.returncode == 0:
            prepared_launch = record.prepared_launch
            if (
                not isinstance(prepared_launch, PreparedLaunch)
                or not callable(prepared_launch.verify_completion)
            ):
                terminal = self._validation_failure_terminal(
                    TypeError, finished.elapsed_sec
                )
                self._claim_terminal_locked(record)
                return terminal, None
            record.phase = "verifying"
            return None, prepared_launch.verify_completion

        terminal = self._finished_terminal(finished, record.prepared_launch)
        self._claim_terminal_locked(record)
        return terminal, None

    def _run_completion_verifier(
        self,
        record: _GoalRecord,
        finished: JobFinished,
        verifier: Callable[
            [JobFinished, threading.Event], VerifiedCompletion
        ],
    ) -> None:
        """在锁外运行校验器，再回到锁内选择取消或提交。

        校验器运行期间持有该目标的 verification_cancel 事件；校验完成后
        回到锁内统一仲裁：取消优先，其次校验失败，最后才原子提交。
        只有非 Exception 的进程控制类异常会在收敛后继续向上抛出。
        """
        validation_error = None
        verified_completion = None
        self._publish_feedback(record, "verifying", 0.0)
        try:
            candidate = verifier(finished, record.verification_cancel)
            if not isinstance(candidate, VerifiedCompletion):
                raise TypeError("verify_completion must return VerifiedCompletion")
            verified_completion = candidate
        except BaseException as error:
            validation_error = error

        with self._lock:
            terminal = self._claim_verified_terminal_locked(
                record, finished, verified_completion, validation_error
            )

        converged = True
        if terminal is not None:
            converged = self._perform_terminal_transition(record, *terminal)
        # 普通校验/提交失败由 Action 结果表示；进程控制类异常只在收敛后传播
        if (
            validation_error is not None
            and not isinstance(validation_error, Exception)
            and converged
        ):
            raise validation_error

    def _claim_verified_terminal_locked(
        self,
        record: _GoalRecord,
        finished: JobFinished,
        verified_completion: VerifiedCompletion | None,
        validation_error: BaseException | None,
    ) -> tuple[str, ReconstructSegmentResult, str] | None:
        """持有锁时选择取消/失败终态，或原子发布已验证结果。

        优先级：取消 > 校验错误 > 非 VerifiedCompletion > 提交。提交是
        适配器唯一的发布/线性化系统调用；rename 后的防御性检查失败仍按
        成功收敛（发布不可回滚）。
        """
        if (
            self._active is not record
            or record.terminal_claimed
            or not record.completion_claimed
            or record.controller_job_id is None
            or finished.job_id != record.controller_job_id
        ):
            return None

        if record.cancel_requested:
            terminal = self._cancel_terminal(
                "reconstruction cancelled during verification", finished.elapsed_sec
            )
        elif validation_error is not None:
            terminal = self._verification_failure_terminal(
                validation_error, finished.elapsed_sec
            )
        elif not isinstance(verified_completion, VerifiedCompletion):
            terminal = self._validation_failure_terminal(
                TypeError, finished.elapsed_sec
            )
        else:
            prepared_launch = record.prepared_launch
            if not isinstance(prepared_launch, PreparedLaunch):
                terminal = self._commit_failure_terminal(
                    TypeError, finished.elapsed_sec
                )
            else:
                record.phase = "committing"
                try:
                    # 事务拥有全部文件系统检查；这里发出适配器唯一的
                    # 发布/线性化系统调用
                    prepared_launch.result_transaction.commit()
                except BaseException as error:
                    if prepared_launch.result_transaction.committed:
                        # 发布不可回滚：rename 后的防御性检查失败
                        # 也必须按真实的成功收敛
                        record.result_committed = True
                        terminal = self._committed_success_terminal(
                            finished.elapsed_sec, verified_completion
                        )
                    else:
                        terminal = self._commit_failure_terminal(
                            type(error), finished.elapsed_sec
                        )
                else:
                    record.result_committed = True
                    terminal = self._committed_success_terminal(
                        finished.elapsed_sec, verified_completion
                    )
        self._claim_terminal_locked(record)
        return terminal

    @staticmethod
    def _committed_success_terminal(
        elapsed_sec: float,
        verified_completion: VerifiedCompletion,
    ) -> tuple[str, ReconstructSegmentResult, str]:
        """构造"结果已提交"的成功终态（携带已验证元数据）。"""
        return (
            "succeeded",
            _result(
                ReconstructSegmentResult.OK,
                "reconstruction worker completed and result committed",
                elapsed_sec,
                verified_completion,
            ),
            "worker result committed",
        )

    @staticmethod
    def _commit_failure_terminal(
        error_type: type[BaseException], elapsed_sec: float
    ) -> tuple[str, ReconstructSegmentResult, str]:
        """构造提交失败终态（IO_ERROR，携带异常类型名）。"""
        message = f"result commit failed: {error_type.__name__}"
        return (
            "aborted",
            _result(ReconstructSegmentResult.IO_ERROR, message, elapsed_sec),
            message,
        )

    @staticmethod
    def _validation_failure_terminal(
        error_type: type[BaseException], elapsed_sec: float
    ) -> tuple[str, ReconstructSegmentResult, str]:
        """构造校验失败终态（ALGORITHM_FAILED）。"""
        message = f"result validation failed: {error_type.__name__}"
        return (
            "aborted",
            _result(
                ReconstructSegmentResult.ALGORITHM_FAILED,
                message,
                elapsed_sec,
            ),
            message,
        )

    @staticmethod
    def _verification_failure_terminal(
        error: BaseException, elapsed_sec: float
    ) -> tuple[str, ReconstructSegmentResult, str]:
        """映射绑定校验器的失败，保留类型化完成错误码。"""
        if isinstance(error, CompletionError):
            error_code = _COMPLETION_CODE_TO_RESULT.get(
                error.code, ReconstructSegmentResult.INTERNAL_ERROR
            )
            message = f"result validation failed: {error}"
            return (
                "aborted",
                _result(error_code, message, elapsed_sec),
                message,
            )
        return ReconstructActionServer._validation_failure_terminal(
            type(error), elapsed_sec
        )

    def _claim_cancel_terminal_locked(
        self,
        record: _GoalRecord,
        message: str,
        elapsed_sec: float = 0.0,
    ) -> tuple[str, ReconstructSegmentResult, str] | None:
        """持有锁时认领固定取消终态（幂等）。"""
        if self._active is not record or record.terminal_claimed:
            return None
        terminal = self._cancel_terminal(message, elapsed_sec)
        self._claim_terminal_locked(record)
        return terminal

    @staticmethod
    def _cancel_terminal(
        message: str, elapsed_sec: float = 0.0
    ) -> tuple[str, ReconstructSegmentResult, str]:
        """构造取消终态（CANCELLED）。"""
        return (
            "canceled",
            _result(ReconstructSegmentResult.CANCELLED, message, elapsed_sec),
            message,
        )

    _OOM_RUN_LOG_PATTERN = re.compile(
        r"(?i)(\boom\b|out of memory|cuda[^\n]{0,64}out of memory)"
    )
    _MAX_RUN_LOG_TAIL_BYTES = 1024 * 1024

    @classmethod
    def _oom_annotated_message(
        cls, message: str, prepared_launch: Any
    ) -> str:
        """worker 非零退出时读取暂存 run.log，识别真实 CUDA OOM 文本。

        正常失败保持原文；只有日志明确出现 OOM 时才附加上下文，使上层
        连续调度器的 FATAL_OOM 状态机能从 Action 终态文本稳定触发。
        """
        try:
            transaction = getattr(prepared_launch, "result_transaction", None)
            staging = getattr(transaction, "staging_path", None)
            if staging is None:
                return message
            path = Path(staging) / "run.log"
            info = os.lstat(str(path))
            if not (info.st_mode and os.path.isfile(str(path))):
                return message
            tail_size = min(info.st_size, cls._MAX_RUN_LOG_TAIL_BYTES)
            with open(str(path), "rb") as stream:
                stream.seek(max(0, info.st_size - tail_size))
                data = stream.read(tail_size)
            text = data.decode("utf-8", errors="replace")
            match = cls._OOM_RUN_LOG_PATTERN.search(text)
            if match is None:
                return message
            excerpt = text[max(0, match.start() - 256): match.end() + 512]
            return (
                "CUDA out of memory detected in worker run.log "
                f"({message}): {excerpt.strip()}"
            )
        except BaseException:
            return message

    @staticmethod
    def _finished_terminal(
        finished: JobFinished,
        prepared_launch: Any = None,
    ) -> tuple[str, ReconstructSegmentResult, str]:
        """把 worker 完成摘要映射到 Action 终态过渡。"""
        if finished.cause == "worker_exit" and finished.returncode == 0:
            # 正常退出但没有校验器路径（防御性兜底）
            return ReconstructActionServer._validation_failure_terminal(
                RuntimeError, finished.elapsed_sec
            )
        if finished.cause == "worker_exit":
            message = f"worker exited with code {finished.returncode}"
            message = ReconstructActionServer._oom_annotated_message(
                message, prepared_launch
            )
            return (
                "aborted",
                _result(
                    ReconstructSegmentResult.ALGORITHM_FAILED,
                    message,
                    finished.elapsed_sec,
                ),
                message,
            )
        if finished.cause == "cancelled":
            return ReconstructActionServer._cancel_terminal(
                "reconstruction cancelled", finished.elapsed_sec
            )
        if finished.cause == "timeout":
            return (
                "aborted",
                _result(
                    ReconstructSegmentResult.ALGORITHM_FAILED,
                    "worker timed out",
                    finished.elapsed_sec,
                ),
                "worker timed out",
            )
        message = f"worker stopped: {finished.cause}"
        return (
            "aborted",
            _result(
                ReconstructSegmentResult.INTERNAL_ERROR,
                message,
                finished.elapsed_sec,
            ),
            message,
        )

    def _terminalize(
        self,
        record: _GoalRecord,
        method: str,
        result: ReconstructSegmentResult,
        text: str,
    ) -> bool:
        """对目标执行一次终态过渡（幂等，最多一次主尝试 + 一次安全回退）。

        流程：
          1. 认领终态（terminal_claimed），确保并发下只执行一次
          2. 调用 handle 的 set_<method>(result, text)
          3. 若已达终态则释放容量并返回 True
          4. 否则按状态选择安全回退方法再次尝试；仍不达终态则记录
             rospy 错误日志并返回 False（容量保持占用）

        Args:
            record: 目标记录
            method: 终态方法名（succeeded / aborted / canceled / rejected）
            result: 要写入的目标结果
            text: 终态描述文本

        Returns:
            True 表示目标已收敛到终态；False 表示终态过渡失败
        """
        with self._lock:
            if (
                self._active is not record
                or record.terminal_claimed
            ):
                return False
            self._claim_terminal_locked(record)
        return self._perform_terminal_transition(record, method, result, text)

    def _claim_terminal_locked(self, record: _GoalRecord) -> None:
        """Claim one terminal transition; caller must hold ``self._lock``."""
        record.terminal_claimed = True
        record.phase = "terminalizing"
        self._terminal_attempt_count += 1

    def _perform_terminal_transition(
        self,
        record: _GoalRecord,
        method: str,
        result: ReconstructSegmentResult,
        text: str,
    ) -> bool:
        """Clean staging, then hand the call to the shared executor (at most one fallback)."""
        method, result, text = self._cleanup_before_terminal_callback(
            record, method, result, text
        )

        def fallback(primary, status, error):
            method, result = self._terminal_fallback(record, error, primary.result, status)
            return TerminalCall(method, result, result.message)

        observer = None
        if self._terminal_observer is not None:
            def observer(call):
                self._terminal_observer(call.method, call.result, call.text)

        outcome = converge_terminal(
            record.handle,
            TerminalCall(method, result, text),
            invoke=lambda handle, call: getattr(handle, f"set_{call.method}")(
                call.result, call.text
            ),
            fallback=fallback,
            observer=observer,
            log_context=f"reconstruct goal {record.key[0]}",
        )
        if outcome.converged:
            self._release_terminal(record)
        return outcome.converged

    def _cleanup_before_terminal_callback(
        self,
        record: _GoalRecord,
        method: str,
        result: ReconstructSegmentResult,
        text: str,
    ) -> tuple[str, ReconstructSegmentResult, str]:
        """在服务端锁与 Action 锁之外清理归本目标所有的未提交暂存。

        清理失败且目标仍可变时，把终态改写为 IO_ERROR 的合法失败方法
        （按状态/接受情况选择）；否则只记录日志、保留原终态。
        """
        with self._lock:
            prepared_launch = record.prepared_launch
            needs_cleanup = (
                isinstance(prepared_launch, PreparedLaunch)
                and not record.result_committed
                and not record.staging_cleaned
            )
        if not needs_cleanup:
            return method, result, text

        if prepared_launch.preserve_uncommitted_staging:
            # Keep failed evidence outside the formal result namespace. The
            # terminal path runs after worker/process-group convergence.
            path = prepared_launch.result_transaction.staging_path
            message = f"{text}; uncommitted staging preserved: {path}"
            result.message = message
            rospy.logwarn("%s", message)
            with self._lock:
                if self._active is record and not record.result_committed:
                    record.staging_cleaned = True  # Retention policy is resolved.
            return method, result, message

        cleanup_error = self._cleanup_result_staging(prepared_launch)
        if cleanup_error is None:
            with self._lock:
                if self._active is record and not record.result_committed:
                    record.staging_cleaned = True
            return method, result, text

        status, _ = query_goal_status(record.handle)
        with self._lock:
            mutable = (
                self._active is record
                and record.terminal_claimed
                and not record.result_committed
            )
            accepted = record.accepted
        if mutable:
            message = f"result cleanup failed: {type(cleanup_error).__name__}"
            cleanup_method = _legal_failure_method(status, accepted)
            return (
                cleanup_method,
                _result(
                    ReconstructSegmentResult.IO_ERROR,
                    message,
                    _result_elapsed_sec(result),
                ),
                message,
            )
        rospy.logerr(
            "Result staging cleanup failed for terminal goal %s: %s",
            record.key[0],
            type(cleanup_error).__name__,
        )
        return method, result, text

    @staticmethod
    def _cleanup_result_staging(
        prepared_launch: PreparedLaunch,
    ) -> BaseException | None:
        """在服务端锁之外执行唯一一次未提交暂存清理调用。

        Returns:
            清理成功返回 None；失败返回异常对象（由调用方决定如何处理）
        """
        try:
            prepared_launch.result_transaction.cleanup()
        except BaseException as error:
            return error
        return None

    def _terminal_fallback(
        self,
        record: _GoalRecord,
        error: BaseException | None,
        primary_result: ReconstructSegmentResult,
        status: int | None,
    ) -> tuple[str, ReconstructSegmentResult]:
        """选择合法回退终态（不持锁查询 actionlib）。

        已提交 → 一律回退为 succeeded（发布不可回滚）；
        RECALLING/PENDING+取消 → canceled；PENDING → rejected；
        status 未知且未接受 → 按取消/拒绝；其余 → aborted。
        """
        with self._lock:
            committed = record.result_committed
            cancel_requested = record.cancel_requested
            accepted = record.accepted
        if committed:
            return "succeeded", primary_result
        message = (
            f"terminal transition failed: {type(error).__name__}"
            if error is not None
            else "terminal transition did not reach a terminal status"
        )
        elapsed_sec = _result_elapsed_sec(primary_result)
        if status == GoalStatus.RECALLING or (
            status == GoalStatus.PENDING and cancel_requested
        ):
            return (
                "canceled",
                _result(ReconstructSegmentResult.CANCELLED, message, elapsed_sec),
            )
        if status == GoalStatus.PENDING:
            return (
                "rejected",
                _result(ReconstructSegmentResult.INTERNAL_ERROR, message, elapsed_sec),
            )
        if status is None and not accepted:
            method = "canceled" if cancel_requested else "rejected"
            code = (
                ReconstructSegmentResult.CANCELLED
                if cancel_requested
                else ReconstructSegmentResult.INTERNAL_ERROR
            )
            return method, _result(code, message, elapsed_sec)
        return (
            "aborted",
            _result(ReconstructSegmentResult.INTERNAL_ERROR, message, elapsed_sec),
        )

    def _release_terminal(self, record: _GoalRecord) -> None:
        """释放目标占用的容量并置位完成事件（幂等）。

        Args:
            record: 已收敛到终态的目标记录
        """
        with self._lock:
            if self._active is record:
                self._active = None
            record.done.set()


def _matching_buffered_finished(record: _GoalRecord) -> JobFinished | None:
    """取出与当前作业 ID 匹配的缓冲完成；不匹配时返回 None。"""
    finished = record.buffered_finished
    if (
        finished is None
        or record.controller_job_id is None
        or finished.job_id != record.controller_job_id
    ):
        return None
    return finished


def _legal_failure_method(status: int | None, accepted: bool) -> str:
    """按目标当前状态与接受情况选择合法的失败终态方法。"""
    if status == GoalStatus.RECALLING:
        return "canceled"
    if status == GoalStatus.PENDING:
        return "rejected"
    if status in (GoalStatus.ACTIVE, GoalStatus.PREEMPTING) or accepted:
        return "aborted"
    return "rejected"


def _result_elapsed_sec(result: ReconstructSegmentResult) -> float:
    """从结果消息提取耗时（秒），异常时返回 0。"""
    try:
        return max(0.0, float(result.wall_time.to_sec()))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0.0


def _is_terminal(goal_handle: Any) -> bool:
    """判断目标句柄当前是否已处于 actionlib 终态。"""
    status, error = query_goal_status(goal_handle)
    if error is not None:
        raise error
    return is_terminal(status)


def _make_daemon_thread(target, args) -> threading.Thread:
    """构造尚未启动的守护线程，由适配器统一启动并确认所有权。"""
    return threading.Thread(target=target, args=args, daemon=True)


def _parse_goal_overrides(goal: Any) -> tuple[int | None, float | None]:
    """解析 ReconstructSegmentGoal 的每目标运行时覆盖。

    返回 (image_resolution, worker_timeout_sec)；未设置 flag 时对应值为
    None。任何非 bool flag 或非法覆盖值都抛 PreparationError(INVALID_GOAL)。
    """
    set_image_resolution = getattr(goal, "set_image_resolution", False)
    set_worker_timeout = getattr(goal, "set_worker_timeout", False)
    if type(set_image_resolution) is not bool:
        raise PreparationError(INVALID_GOAL, "set_image_resolution must be a bool")
    if type(set_worker_timeout) is not bool:
        raise PreparationError(INVALID_GOAL, "set_worker_timeout must be a bool")
    image_resolution = None
    if set_image_resolution:
        value = goal.image_resolution
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PreparationError(
                INVALID_GOAL, "image_resolution must be a positive integer"
            )
        image_resolution = int(value)
    worker_timeout_sec = None
    if set_worker_timeout:
        value = goal.worker_timeout_sec
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PreparationError(
                INVALID_GOAL, "worker_timeout_sec must be a finite number"
            )
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise PreparationError(
                INVALID_GOAL, "worker_timeout_sec must be positive and finite"
            )
        worker_timeout_sec = value
    return image_resolution, worker_timeout_sec


def _goal_key(goal_handle: Any) -> tuple[str, int, int]:
    """从目标句柄提取用于身份匹配的键。

    Args:
        goal_handle: actionlib 目标句柄

    Returns:
        (goal_id.id, stamp.secs, stamp.nsecs) 三元组
    """
    goal_id = goal_handle.get_goal_id()
    return (goal_id.id, int(goal_id.stamp.secs), int(goal_id.stamp.nsecs))


def _result(
    error_code: int,
    message: str,
    elapsed_sec: float = 0.0,
    verified_completion: VerifiedCompletion | None = None,
):
    """构造一个 ReconstructSegmentResult 消息。

    Args:
        error_code: 结果错误码（ReconstructSegmentResult.OK 等常量）
        message: 人类可读的结果描述
        elapsed_sec: 耗时（秒），写入 wall_time 字段
        verified_completion: 仅成功提交后发布的已验证结果元数据

    Returns:
        填充完成的 ReconstructSegmentResult 消息
    """
    result = ReconstructSegmentResult()
    result.error_code = error_code
    result.message = message
    result.wall_time = rospy.Duration.from_sec(max(0.0, float(elapsed_sec)))
    if verified_completion is not None:
        result.scene_id = verified_completion.scene_id
        result.result_manifest_uri = verified_completion.result_manifest_uri
        result.result_manifest_sha256 = verified_completion.result_manifest_sha256
        result.scene_glb_uri = verified_completion.scene_glb_uri
        result.frame_count = verified_completion.frame_count
    return result
