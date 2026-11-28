"""Waiting for, cancelling and collecting one submitted reconstruction job (S2 step 4).

The scheduler submits the job and binds its ownership under its own locks; this module
only runs the state machine that follows: poll status until a terminal state, cancel on
STOP, on the reconstruction watchdog or after a status error, and collect the result.
Every backend call happens with no session lock held. Everything the scheduler must see
or latch is done through ``control``, whose methods take the session lock themselves and
are called at exactly the points the scheduler used to do the same work inline:

- ``stopping()``: read the STOP latch.
- ``cancel_failed(message)``: record a cancel that raised (last_error).
- ``cancel_requested(job_id)``: record a successful cancel request in the stop audit.
- ``watchdog_fired(job_id, message)``: record the watchdog's cancel request and last_error.
- ``status_deadline(error)``: fail closed when status stayed unavailable through the
  deadline (latch STOPPING; the job stays owned for STOP to prove terminal + collect).
- ``terminal(job_id, status)``: record the terminal state in the stop audit (this latches
  the cleanup_failed safety stop synchronously) and return whether it is a cleanup failure.
- ``collect_failed(job_id, message)``: record a collect_result failure (last_error, audit).
- ``collected(job_id, outcome)``: clear ownership, keep the worker wall time, audit.

The result says what happened; the scheduler alone decides terminal states, retries and
failure counts, so a window is never failed twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .continuous_session_types import _JOB_TERMINAL, JobStatus, ReconstructionBackend

# The job was collected; ``outcome`` is the backend's result.
COLLECTED = "collected"
# STOP arrived while status was failing: the cancel was requested, the job is still owned
# and STOP converges it (terminal + collect).
STOP_CANCELLED_WITH_OWNER_RETAINED = "stop_cancelled_with_owner_retained"
# Status stayed unavailable through the deadline: the session is latched STOPPING and the
# job is still owned; ``error`` is the failure to record for the window.
STATUS_DEADLINE_WITH_OWNER_RETAINED = "status_deadline_with_owner_retained"
# The job reached a terminal state but collect_result raised; the job is still owned.
COLLECT_ERROR = "collect_error"


@dataclass(frozen=True)
class JobExecutionResult:
    kind: str
    outcome: Any = None
    status: JobStatus | None = None
    # First status error seen while polling, even if a later status succeeded.
    status_failure: str = ""
    error: str = ""
    # The terminal status reports that the worker's process group was not cleaned up.
    cleanup_failure: bool = False


def wait_collect_job(
    backend: ReconstructionBackend,
    job_id: str,
    *,
    window_label: str,
    wait_timeout_sec: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
    control: Any,
) -> JobExecutionResult:
    """Poll, cancel when required and collect one submitted job; see the module docstring."""
    deadline = monotonic() + wait_timeout_sec
    watchdog_fired = False
    status_failure = ""
    while True:
        stopping = control.stopping()
        try:
            status = backend.status(job_id)
        except BaseException as error:
            if not status_failure:
                status_failure = (
                    f"backend status failed: {type(error).__name__}: {error}"
                )
            # actionlib cancellation is asynchronous. Keep ownership and
            # polling until the lower action reports a terminal state;
            # collecting here would pop a still-live client.
            try:
                backend.cancel(job_id)
            except BaseException as cancel_error:
                control.cancel_failed(
                    f"{window_label} backend cancel failed "
                    "after status error: "
                    f"{type(cancel_error).__name__}: {cancel_error}"
                )
            else:
                control.cancel_requested(job_id)
            now = monotonic()
            if stopping:
                return JobExecutionResult(
                    STOP_CANCELLED_WITH_OWNER_RETAINED, status_failure=status_failure
                )
            if now >= deadline:
                # The worker may still exist, so retain active-job
                # ownership for stop_all to prove terminal+collect.  The
                # scheduler must nevertheless fail closed instead of
                # leaving an immortal consumer and a false RUNNING state.
                error = (
                    f"{status_failure}; status unavailable through "
                    "reconstruction deadline"
                )
                control.status_deadline(error)
                return JobExecutionResult(
                    STATUS_DEADLINE_WITH_OWNER_RETAINED,
                    status_failure=status_failure,
                    error=error,
                )
            sleep(min(0.2, max(0.0, deadline - now)))
            continue
        if status.state in _JOB_TERMINAL:
            break
        if stopping:
            try:
                backend.cancel(job_id)
            except BaseException as error:
                control.cancel_failed(
                    f"{window_label} backend cancel failed "
                    f"during stop: {type(error).__name__}: {error}"
                )
            else:
                control.cancel_requested(job_id)
        if not stopping and not watchdog_fired and monotonic() >= deadline:
            watchdog_fired = True
            try:
                backend.cancel(job_id)
            except BaseException as error:
                control.cancel_failed(
                    f"{window_label} backend cancel failed "
                    f"during watchdog: {type(error).__name__}: {error}"
                )
            else:
                control.watchdog_fired(
                    job_id,
                    f"{window_label} reconstruction watchdog fired; "
                    "waiting for lower action terminal state",
                )
        sleep(0.2)
    cleanup_failure = control.terminal(job_id, status)
    try:
        outcome = backend.collect_result(job_id)
    except BaseException as error:
        control.collect_failed(
            job_id,
            f"{window_label} backend collect_result "
            f"failed for job {job_id}: "
            f"{type(error).__name__}: {error}",
        )
        return JobExecutionResult(
            COLLECT_ERROR,
            status=status,
            status_failure=status_failure,
            error=f"backend collect_result failed: {type(error).__name__}: {error}",
            cleanup_failure=cleanup_failure,
        )
    control.collected(job_id, outcome)
    return JobExecutionResult(
        COLLECTED,
        outcome=outcome,
        status=status,
        status_failure=status_failure,
        cleanup_failure=cleanup_failure,
    )


__all__ = [
    "COLLECTED",
    "COLLECT_ERROR",
    "STATUS_DEADLINE_WITH_OWNER_RETAINED",
    "STOP_CANCELLED_WITH_OWNER_RETAINED",
    "JobExecutionResult",
    "wait_collect_job",
]
