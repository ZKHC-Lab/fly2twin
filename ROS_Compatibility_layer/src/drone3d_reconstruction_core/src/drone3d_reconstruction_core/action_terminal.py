"""把一个 actionlib 目标送进终态：主调用、确认、至多一次回退、再确认。

采集和重建两个 Action 服务端共用这一段。调用前，调用方必须已经：
- 在自己的服务端锁内认领了这个目标的终态（只认领一次）；
- 在服务端锁外做完业务清理（未提交的暂存、清理失败时改写的终态）。
调用时不得持有服务端锁：invoke 和状态查询都会进入 actionlib，而 actionlib 是持着自己的锁调用
goal / cancel 回调的，锁顺序只能是 actionlib 锁 → 服务端锁。

本模块不持锁、不释放槽位。终态是否证实由返回值告诉调用方；没证实时槽位必须保持占用。
选哪个回退方法、结果怎么构造、观察者看到什么，都由调用方传入。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import rospy
from actionlib_msgs.msg import GoalStatus

TERMINAL_STATUSES = frozenset(
    (
        GoalStatus.PREEMPTED,
        GoalStatus.SUCCEEDED,
        GoalStatus.ABORTED,
        GoalStatus.REJECTED,
        GoalStatus.RECALLED,
        GoalStatus.LOST,
    )
)


@dataclass(frozen=True)
class TerminalCall:
    """一次终态调用：set_<method>(result, text)。"""

    method: str
    result: Any
    text: str


@dataclass(frozen=True)
class TerminalOutcome:
    """converged：actionlib 报告的状态已是终态。final_call：最后一次实际发出的调用。"""

    converged: bool
    final_call: TerminalCall
    observed_status: int | None


def query_goal_status(goal_handle: Any) -> tuple[int | None, BaseException | None]:
    """查询目标状态；查询失败时返回 (None, error)，视为“尚未证实”。"""
    try:
        return goal_handle.get_goal_status().status, None
    except BaseException as error:
        return None, error


def is_terminal(status: int | None) -> bool:
    return status in TERMINAL_STATUSES


def converge_terminal(
    goal_handle: Any,
    primary: TerminalCall,
    *,
    invoke: Callable[[Any, TerminalCall], None],
    fallback: Callable[[TerminalCall, int | None, BaseException | None], TerminalCall],
    observer: Callable[[TerminalCall], None] | None = None,
    log_context: str = "action",
) -> TerminalOutcome:
    """主调用一次，查状态；未证实终态则按 fallback 选的调用再试一次，再查状态。

    observer 只在主调用前通知一次，异常被记录后忽略。invoke 抛出的异常只影响回退的选择，
    不向外抛。fallback(primary, 主调用后查到的状态, 主调用或查询的异常) 返回第二次调用。
    """
    if observer is not None:
        try:
            observer(primary)
        except BaseException as observer_error:
            rospy.logerr(
                "%s terminal observer failed: %s", log_context, type(observer_error).__name__
            )
    call_error = None
    try:
        invoke(goal_handle, primary)
    except BaseException as error:
        call_error = error
    status, status_error = query_goal_status(goal_handle)
    if is_terminal(status):
        return TerminalOutcome(True, primary, status)

    second = fallback(primary, status, call_error or status_error)
    second_error = None
    try:
        invoke(goal_handle, second)
    except BaseException as error:
        second_error = error
    second_status, second_status_error = query_goal_status(goal_handle)
    if is_terminal(second_status):
        return TerminalOutcome(True, second, second_status)

    final_error = second_error or second_status_error or status_error or call_error
    rospy.logerr(
        "%s terminal did not converge: %s",
        log_context,
        type(final_error).__name__ if final_error is not None else "nonterminal status",
    )
    return TerminalOutcome(False, second, second_status)
