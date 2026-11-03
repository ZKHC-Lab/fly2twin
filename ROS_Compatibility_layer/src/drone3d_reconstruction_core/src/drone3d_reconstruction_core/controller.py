"""Worker 进程生命周期控制器模块（不依赖 ROS）。

本模块负责托管最多一个重建 worker 子进程：启动、超时、取消、优雅终止、
进程组清理与一次性完成回调。核心保证：

  - 同一时刻至多持有一个活跃子进程（单槽位，BusyError 拒绝并发启动）；
  - 每个 job 的完成回调（on_finished）至多收敛一次，携带 cause 与 returncode；
  - 终止时先 SIGTERM（宽限期内轮询），再 SIGKILL 兜底，并等待进程组
    完全静默（quiescence）后才交付完成回调；
  - worker 以 start_new_session 启动（独立进程组），保证整组可被信号控制。

ROS-independent lifecycle control for one reconstruction worker process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import signal
import math
import subprocess
import threading
import time
from typing import Callable, Sequence

from .warm_worker_pool import (
    WarmJobFailedError,
    WarmWorkerPool,
    WarmWorkerUnavailableError,
)


class BusyError(RuntimeError):
    """在容量已被占用时尝试启动 worker 所抛出的异常。"""


@dataclass(frozen=True)
class JobFinished:
    """一次 worker 完成的不可变摘要。

    Attributes:
        cause: 结束原因（worker_exit / timeout / cancelled / shutdown / …）
        returncode: 进程退出码
        elapsed_sec: 从启动到结束的耗时（秒）
        job_id: 本次作业的单调递增编号
    """

    cause: str
    returncode: int
    elapsed_sec: float
    job_id: int


@dataclass
class _ActiveJob:
    """一个活跃作业的内部可变状态（受控制器锁保护）。"""

    job_id: int
    process: subprocess.Popen
    process_group: int
    started: float
    worker_timeout_sec: float
    requested_cause: str | None = None
    finished: bool = False
    stop_claimed: bool = False
    started_notified: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)


@dataclass
class _WarmActiveJob:
    """一个活跃 warm 作业的内部可变状态（受控制器锁保护）。"""

    job_id: int
    started: float
    worker_timeout_sec: float
    finished: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)


class JobController:
    """托管最多一个子进程，并把每次运行收敛到唯一一次完成回调。"""

    def __init__(
        self,
        *,
        worker_timeout_sec: float,
        term_grace_sec: float,
        on_finished: Callable[[JobFinished], None],
        popen_factory=subprocess.Popen,
        thread_factory=threading.Thread,
        group_quiescence_waiter=None,
        monotonic=time.monotonic,
        poll_interval_sec: float = 0.02,
    ) -> None:
        """构造控制器并校验所有时长参数为正且有限。

        Args:
            worker_timeout_sec: worker 最长运行时长（秒），超时触发 SIGTERM
            term_grace_sec: SIGTERM 后的优雅退出宽限时长（秒）
            on_finished: 唯一完成回调 (JobFinished)
            popen_factory: 子进程工厂（测试注入点）
            thread_factory: 监控/终止线程工厂（测试注入点）
            group_quiescence_waiter: 进程组静默等待函数（测试注入点）
            monotonic: 单调时钟
            poll_interval_sec: 轮询间隔（秒）
        """
        if not _positive_finite(worker_timeout_sec):
            raise ValueError("worker_timeout_sec must be positive and finite")
        if not _positive_finite(term_grace_sec):
            raise ValueError("term_grace_sec must be positive and finite")
        if not _positive_finite(poll_interval_sec):
            raise ValueError("poll_interval_sec must be positive and finite")
        self._worker_timeout_sec = float(worker_timeout_sec)
        self._term_grace_sec = float(term_grace_sec)
        self._on_finished = on_finished
        self._popen_factory = popen_factory
        self._thread_factory = thread_factory
        self._group_quiescence_waiter = (
            group_quiescence_waiter or _wait_for_group_quiescence
        )
        self._monotonic = monotonic
        self._poll_interval_sec = float(poll_interval_sec)
        self._lock = threading.Lock()
        self._active: _ActiveJob | None = None
        self._starting = False
        self._shutdown = False
        self._next_job_id = 1

    def is_busy(self) -> bool:
        """当前是否有作业在启动中或活跃中。"""
        with self._lock:
            return self._starting or self._active is not None

    def start(
        self,
        command: Sequence[str],
        *,
        on_started: Callable[[int], None] | None = None,
        worker_timeout_sec: float | None = None,
    ) -> int:
        """启动一个新 worker 作业并返回其 job_id。

        Args:
            command: 非空字符串 argv
            on_started: 作业 ID 登记回调（在 worker 启动后立即调用）
            worker_timeout_sec: 每作业超时覆盖；None 时使用控制器默认值

        Returns:
            新作业的 job_id

        Raises:
            BusyError: 已有活跃/启动中的作业
            RuntimeError: 控制器已关闭
        """
        if worker_timeout_sec is not None:
            if isinstance(worker_timeout_sec, bool) or not isinstance(
                worker_timeout_sec, (int, float)
            ):
                raise ValueError("worker_timeout_sec must be positive and finite")
            worker_timeout_sec = float(worker_timeout_sec)
            if not math.isfinite(worker_timeout_sec) or worker_timeout_sec <= 0:
                raise ValueError("worker_timeout_sec must be positive and finite")
        else:
            worker_timeout_sec = self._worker_timeout_sec
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("command must contain nonempty string arguments")
        with self._lock:
            if self._shutdown:
                raise RuntimeError("controller is shut down")
            if self._starting or self._active is not None:
                raise BusyError("a reconstruction worker is already active")
            self._starting = True

        try:
            # 独立进程组：便于整组发送 SIGTERM/SIGKILL
            process = self._popen_factory(list(command), start_new_session=True)
        except BaseException:
            with self._lock:
                self._starting = False
            raise

        with self._lock:
            job_id = self._next_job_id
            self._next_job_id += 1
        job = _ActiveJob(
            job_id=job_id,
            process=process,
            process_group=process.pid,
            started=self._monotonic(),
            worker_timeout_sec=float(worker_timeout_sec),
        )
        stop_for_shutdown = False
        with self._lock:
            self._starting = False
            if self._shutdown:
                # 启动与关闭竞态：登记立即停止请求
                job.requested_cause = "shutdown"
                self._active = job
                stop_for_shutdown = True
            else:
                self._active = job

        if on_started is not None:
            try:
                on_started(job_id)
            except BaseException:
                job.started_notified.set()
                self._stop_after_dispatch_failure(job, "dispatch_failed")
                raise
        job.started_notified.set()

        with self._lock:
            stop_after_started = (
                job is self._active
                and not job.finished
                and job.requested_cause is not None
            )
        if stop_for_shutdown or stop_after_started:
            self._start_stopper(job)
            return job_id
        self._start_monitor(job)
        return job_id

    def cancel(self) -> bool:
        """请求取消当前作业（SIGTERM 终止流程），返回是否受理。"""
        return self._request_stop("cancelled")

    def shutdown(self) -> None:
        """关闭控制器：等待启动完成、请求停止并等待作业终止。"""
        wait_limit = 2.0 * self._term_grace_sec + 1.0
        deadline = self._monotonic() + wait_limit
        while True:
            with self._lock:
                self._shutdown = True
                starting = self._starting
                job = self._active
            if not starting:
                break
            if self._monotonic() >= deadline:
                raise RuntimeError("worker launch did not finish during controller shutdown")
            time.sleep(self._poll_interval_sec)

        self._request_stop("shutdown")
        if job is not None and not job.done.wait(wait_limit):
            raise RuntimeError("worker did not terminate during controller shutdown")

    def _request_stop(self, cause: str) -> bool:
        """登记停止原因；若已完成通知则立即启动终止线程。"""
        with self._lock:
            job = self._active
            if job is None or job.finished:
                return False
            if job.requested_cause is not None:
                return False
            job.requested_cause = cause
            started_notified = job.started_notified.is_set()
        if started_notified:
            self._start_stopper(job)
        return True

    def _start_monitor(self, job: _ActiveJob) -> None:
        """启动监控线程（超时检测 + 进程退出检测）。"""
        try:
            self._start_thread(self._monitor, job)
        except BaseException:
            self._stop_after_dispatch_failure(job, "dispatch_failed")
            raise

    def _start_stopper(self, job: _ActiveJob) -> None:
        """启动终止线程；若线程创建失败则同步执行终止。"""
        if not self._claim_stop(job):
            return
        try:
            self._start_thread(self._stop, job)
        except Exception:
            self._stop(job)
        except BaseException:
            try:
                self._stop(job)
            finally:
                raise

    def _claim_stop(self, job: _ActiveJob) -> bool:
        """认领停止权（确保终止只执行一次）。"""
        with self._lock:
            if job is not self._active or job.finished or job.stop_claimed:
                return False
            job.stop_claimed = True
            return True

    def _start_thread(self, target: Callable[[_ActiveJob], None], job: _ActiveJob) -> None:
        thread = self._thread_factory(target=target, args=(job,), daemon=True)
        thread.start()

    def _stop_after_dispatch_failure(self, job: _ActiveJob, cause: str) -> None:
        """派发线程失败后的兜底：登记原因并同步终止。"""
        with self._lock:
            if job is not self._active or job.finished:
                return
            if job.requested_cause is None:
                job.requested_cause = cause
            if job.stop_claimed:
                return
            job.stop_claimed = True
        self._stop(job)

    def _monitor(self, job: _ActiveJob) -> None:
        """监控线程：检测进程退出或超时，触发对应终止流程。"""
        deadline = job.started + job.worker_timeout_sec
        while True:
            returncode = job.process.poll()
            if returncode is not None:
                with self._lock:
                    if job is not self._active or job.finished:
                        return
                    if job.requested_cause is not None:
                        return  # 终止线程负责清理与终态交付
                    job.requested_cause = "worker_exit"
                    job.stop_claimed = True
                self._stop(job)
                return
            if self._monotonic() >= deadline:
                stop_for_timeout = False
                with self._lock:
                    if job is not self._active or job.finished:
                        return
                    if job.requested_cause is None:
                        job.requested_cause = "timeout"
                        stop_for_timeout = True
                if stop_for_timeout:
                    self._start_stopper(job)
                return
            time.sleep(self._poll_interval_sec)

    def _stop(self, job: _ActiveJob) -> None:
        """终止流程：SIGTERM → 宽限期 → SIGKILL → 等待进程组静默。"""
        cause = job.requested_cause or "shutdown"
        _signal_group(job.process_group, signal.SIGTERM)
        deadline = self._monotonic() + self._term_grace_sec
        while self._monotonic() < deadline:
            job.process.poll()  # 先回收已退出的组长进程，再检测进程组
            if not _group_exists(job.process_group):
                break
            time.sleep(self._poll_interval_sec)
        if _group_exists(job.process_group):
            _signal_group(job.process_group, signal.SIGKILL)

        try:
            returncode = job.process.wait(timeout=self._term_grace_sec)
        except subprocess.TimeoutExpired:
            _signal_group(job.process_group, signal.SIGKILL)
            returncode = job.process.wait()
        cleanup_error = None
        try:
            # 等待进程组完全静默，确保没有残留成员
            self._group_quiescence_waiter(
                job.process_group,
                timeout_sec=self._term_grace_sec,
                monotonic=self._monotonic,
                poll_interval_sec=self._poll_interval_sec,
            )
        except Exception as error:
            cleanup_error = error
        if cleanup_error is not None:
            cause = "cleanup_failed"
        self._finish(job, cause, returncode)

    def _finish(self, job: _ActiveJob, cause: str, returncode: int) -> None:
        """交付唯一完成回调并置位 done 事件（幂等）。"""
        with self._lock:
            if job is not self._active or job.finished:
                return
            job.finished = True
            self._active = None
            finished = JobFinished(
                cause=cause,
                returncode=int(returncode),
                elapsed_sec=max(0.0, self._monotonic() - job.started),
                job_id=job.job_id,
            )
        try:
            self._on_finished(finished)
        except Exception:
            pass  # 观察者异常绝不阻断终止收敛
        finally:
            job.done.set()


class WarmWorkerController:
    """单槽位控制器：优先驻留 warm worker，失败时退回一次性 worker。

    与 :class:`JobController` 相同的仲裁/回调契约：

      - 同一时刻至多一个作业（warm 作业与一次性作业共享一个槽位）；
      - 每个 job 的 ``on_finished`` 至多收敛一次，携带 cause 与 returncode；
      - ``start_warm`` 在派发前同步确保 warm 进程可 spawn；该步骤失败抛
        :class:`WarmWorkerUnavailableError`，调用方随后可用 ``start`` 走
        一次性 worker 路径（此时尚未发布 job_id，回退安全）；
      - warm 进程持久存活：job 成功不代表进程退出，shutdown 时统一终止。
    """

    def __init__(
        self,
        *,
        worker_timeout_sec: float,
        term_grace_sec: float,
        on_finished: Callable[[JobFinished], None],
        warm_command: Sequence[str],
        pool_log_path: str,
        popen_factory=subprocess.Popen,
        thread_factory=threading.Thread,
        monotonic=time.monotonic,
    ) -> None:
        if not _positive_finite(worker_timeout_sec):
            raise ValueError("worker_timeout_sec must be positive and finite")
        if not _positive_finite(term_grace_sec):
            raise ValueError("term_grace_sec must be positive and finite")
        self._worker_timeout_sec = float(worker_timeout_sec)
        self._term_grace_sec = float(term_grace_sec)
        self._monotonic = monotonic
        self._thread_factory = thread_factory
        self._lock = threading.Lock()
        self._shutdown = False
        self._starting_warm = False
        self._warm_active: _WarmActiveJob | None = None
        self._next_job_id = 1
        self._pool = WarmWorkerPool(
            warm_command,
            pool_log_path=pool_log_path,
            popen_factory=popen_factory,
            thread_factory=thread_factory,
            monotonic=monotonic,
        )
        # 一次性 worker 回退路径（start_warm 同步失败后由调用方 start 触发）
        self._oneshot = JobController(
            worker_timeout_sec=worker_timeout_sec,
            term_grace_sec=term_grace_sec,
            on_finished=on_finished,
            popen_factory=popen_factory,
            thread_factory=thread_factory,
            monotonic=monotonic,
        )
        self._on_finished = on_finished

    def is_busy(self) -> bool:
        with self._lock:
            return (
                self._starting_warm
                or self._warm_active is not None
                or self._oneshot.is_busy()
            )

    def start(
        self,
        command: Sequence[str],
        *,
        on_started: Callable[[int], None] | None = None,
        worker_timeout_sec: float | None = None,
    ) -> int:
        """一次性 worker 回退路径（与 JobController.start 契约一致）。"""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("controller is shut down")
            if self._starting_warm or self._warm_active is not None:
                raise BusyError("a warm reconstruction worker is already active")
        return self._oneshot.start(
            command,
            on_started=on_started,
            worker_timeout_sec=worker_timeout_sec,
        )

    def resident_worker_pid(self):
        return self._pool.worker_pid()

    def release_idle_worker(self):
        with self._lock:
            if self._starting_warm or self._warm_active is not None or self._oneshot.is_busy():
                raise BusyError("cannot release a worker while a job is active")
            self._pool.shutdown()

    def start_warm(
        self,
        warm_command: Sequence[str],
        job_payload: dict,
        *,
        on_started: Callable[[int], None] | None = None,
        worker_timeout_sec: float | None = None,
    ) -> int:
        """启动一个 warm worker 作业并返回其 job_id。

        Args:
            warm_command: 驻留进程 argv（覆盖控制器默认 warm_command）
            job_payload: 重建 job JSON（不得含 job_id，由本控制器写入）
            on_started: 作业 ID 登记回调（warm 进程 spawn 成功后立即调用）
            worker_timeout_sec: 每作业超时覆盖；None 时使用控制器默认值

        Returns:
            新作业的 job_id

        Raises:
            BusyError: 已有活跃/启动中的作业
            RuntimeError: 控制器已关闭
            WarmWorkerUnavailableError: warm 进程同步 spawn 失败（尚未发布
                job_id，调用方可安全退回一次性 worker）
        """
        if worker_timeout_sec is not None:
            if isinstance(worker_timeout_sec, bool) or not isinstance(
                worker_timeout_sec, (int, float)
            ):
                raise ValueError("worker_timeout_sec must be positive and finite")
            worker_timeout_sec = float(worker_timeout_sec)
            if not math.isfinite(worker_timeout_sec) or worker_timeout_sec <= 0:
                raise ValueError("worker_timeout_sec must be positive and finite")
        else:
            worker_timeout_sec = self._worker_timeout_sec
        if not warm_command or any(
            not isinstance(part, str) or not part or "\x00" in part
            for part in warm_command
        ):
            raise ValueError("warm_command must contain nonempty strings without NUL")
        if not isinstance(job_payload, dict):
            raise TypeError("job_payload must be a dict")
        if "job_id" in job_payload:
            raise ValueError("job_payload must not contain job_id")

        with self._lock:
            if self._shutdown:
                raise RuntimeError("controller is shut down")
            if self._starting_warm or self._warm_active is not None:
                raise BusyError("a reconstruction worker is already active")
            if self._oneshot.is_busy():
                raise BusyError("a reconstruction worker is already active")
            self._starting_warm = True

        try:
            # 同步 spawn 失败发生在 job_id 发布之前：调用方可安全回退
            self._pool.prepare()
        except BaseException:
            with self._lock:
                self._starting_warm = False
            raise

        with self._lock:
            job_id = self._next_job_id
            self._next_job_id += 1
        payload = dict(job_payload)
        payload["job_id"] = job_id
        try:
            json.dumps(payload, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            with self._lock:
                self._starting_warm = False
            raise TypeError("job_payload must be JSON serializable") from error

        job = _WarmActiveJob(
            job_id=job_id,
            started=self._monotonic(),
            worker_timeout_sec=float(worker_timeout_sec),
        )
        stop_for_shutdown = False
        with self._lock:
            self._starting_warm = False
            if self._shutdown:
                job.cancel_event.set()
                stop_for_shutdown = True
            self._warm_active = job

        if on_started is not None:
            try:
                on_started(job_id)
            except BaseException:
                job.done.set()
                self._finish_warm(job, "dispatch_failed", -9, 0.0)
                raise
        if stop_for_shutdown:
            self._finish_warm(
                job,
                "shutdown",
                -signal.SIGKILL,
                self._monotonic() - job.started,
            )
            return job_id

        try:
            thread = self._thread_factory(target=self._run_warm, args=(job, payload))
            thread.daemon = True
            thread.start()
        except BaseException:
            self._finish_warm(job, "dispatch_failed", -9, 0.0)
            raise
        return job_id

    def _run_warm(self, job: _WarmActiveJob, payload: dict) -> None:
        """在后台执行 warm job 并收敛到唯一一次 JobFinished。"""
        started = self._monotonic()
        try:
            response = self._pool.submit(
                payload,
                timeout_sec=job.worker_timeout_sec,
                cancel_event=job.cancel_event,
            )
        except WarmJobFailedError:
            self._finish_warm(job, "worker_exit", 1, self._monotonic() - started)
        except WarmWorkerUnavailableError as error:
            elapsed = self._monotonic() - started
            if error.reason == "timeout":
                cause = "timeout"
            elif error.reason == "cancelled":
                cause = "cancelled"
            else:
                # 进程死亡/管道失败一律按 worker_exit（非零退出码）交付，
                # 让 Action 层走 ALGORITHM_FAILED 而不是误跑完成校验器。
                cause = "worker_exit"
            returncode = (
                error.returncode
                if error.returncode is not None
                else -signal.SIGKILL
            )
            self._finish_warm(job, cause, int(returncode), elapsed)
        except BaseException:
            self._finish_warm(job, "dispatch_failed", -9, self._monotonic() - started)
        else:
            elapsed = _response_elapsed(response, self._monotonic() - started)
            self._finish_warm(job, "worker_exit", 0, elapsed)

    def _finish_warm(
        self, job: _WarmActiveJob, cause: str, returncode: int, elapsed_sec: float
    ) -> None:
        """交付唯一完成回调并置位 done 事件（幂等）。"""
        with self._lock:
            if job is not self._warm_active or job.finished:
                return
            job.finished = True
            self._warm_active = None
            finished = JobFinished(
                cause=cause,
                returncode=int(returncode),
                elapsed_sec=max(0.0, float(elapsed_sec)),
                job_id=job.job_id,
            )
        try:
            self._on_finished(finished)
        except Exception:
            pass  # 观察者异常绝不阻断终止收敛
        finally:
            job.done.set()

    def cancel(self) -> bool:
        """请求取消当前作业，返回是否受理。"""
        with self._lock:
            job = self._warm_active
            if job is not None and not job.finished:
                job.cancel_event.set()
                return True
        return self._oneshot.cancel()

    def shutdown(self) -> None:
        """关闭控制器：终止 warm 进程与回退 worker，等待作业收敛。"""
        with self._lock:
            self._shutdown = True
            job = self._warm_active
            if job is not None and not job.finished:
                job.cancel_event.set()

        # 先收敛可能活跃的一次性 worker，再停驻留进程。
        self._oneshot.shutdown()

        try:
            if job is not None:
                wait_limit = 2.0 * self._term_grace_sec + 6.0
                if not job.done.wait(wait_limit):
                    raise RuntimeError(
                        "warm worker did not terminate during controller shutdown"
                    )
        finally:
            self._pool.shutdown()


def _response_elapsed(response: dict, fallback_sec: float) -> float:
    """读取 warm job 响应中的耗时字段，非法时退回墙钟时间。"""
    try:
        value = response.get("elapsed_sec")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return fallback_sec
        number = float(value)
        if not math.isfinite(number) or number < 0:
            return fallback_sec
        return number
    except (TypeError, ValueError, OverflowError):
        return fallback_sec


def _wait_for_group_quiescence(
    process_group: int,
    *,
    timeout_sec: float,
    monotonic: Callable[[], float],
    poll_interval_sec: float,
) -> None:
    """轮询 /proc 直到进程组内无存活成员（僵尸除外）或超时。"""
    deadline = monotonic() + timeout_sec
    while _group_has_live_members(process_group):
        if monotonic() >= deadline:
            raise RuntimeError(
                f"worker process group {process_group} still has live members after SIGKILL"
            )
        time.sleep(poll_interval_sec)


def _group_has_live_members(process_group: int) -> bool:
    """扫描 /proc，检查进程组内是否存在非僵尸的存活成员。"""
    proc_root = "/proc"
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return _group_exists(process_group)
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, entry, "stat"), "rb") as stream:
                stat = stream.read()
            closing_paren = stat.rfind(b")")
            fields = stat[closing_paren + 2 :].split()
            state = fields[0]
            process_group_id = int(fields[2])
        except (OSError, ValueError, IndexError):
            continue
        if process_group_id == process_group and state != b"Z":
            return True
    return False


def _positive_finite(value: float) -> bool:
    """校验值为正且有限的数值。"""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return number > 0.0 and number != float("inf")


def _signal_group(process_group: int, sig: signal.Signals) -> None:
    """向进程组发信号；组不存在时静默忽略。"""
    try:
        os.killpg(process_group, sig)
    except ProcessLookupError:
        pass


def _group_exists(process_group: int) -> bool:
    """用 killpg(pid, 0) 探测进程组是否存在。"""
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
