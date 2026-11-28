"""Unit tests for wait_collect_job: the job state machine without a session."""

import unittest

from drone3d_reconstruction_core.continuous_job import (
    COLLECTED,
    COLLECT_ERROR,
    STATUS_DEADLINE_WITH_OWNER_RETAINED,
    STOP_CANCELLED_WITH_OWNER_RETAINED,
    wait_collect_job,
)
from drone3d_reconstruction_core.continuous_session import (
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    JobStatus,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Control:
    """Records every control call in order; ``stop`` is the STOP latch."""

    def __init__(self):
        self.stop = False
        self.calls = []

    def stopping(self):
        return self.stop

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name,) + args)
            if name == "terminal":
                status = args[1]
                return status.state == JOB_FAILED and "cleanup_failed" in status.message
        return record

    def names(self):
        return [call[0] for call in self.calls]


class Backend:
    """status(n) returns the n-th scripted state (the last one repeats); entries may be exceptions."""

    def __init__(self, states, cancel_error=None, collect_errors=0):
        self.states, self.calls = list(states), []
        self.cancel_error, self.collect_errors = cancel_error, collect_errors

    def status(self, job_id):
        self.calls.append("status")
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        if isinstance(state, Exception):
            raise state
        return JobStatus(job_id, state, "cleanup_failed: alive" if state == JOB_FAILED else "")

    def cancel(self, job_id):
        self.calls.append("cancel")
        if self.cancel_error is not None:
            raise self.cancel_error

    def collect_result(self, job_id):
        self.calls.append("collect")
        if self.collect_errors:
            self.collect_errors -= 1
            raise RuntimeError("collect broke")
        return "outcome"


def run(backend, control=None, timeout=1.0, clock=None):
    clock = clock or Clock()
    control = control or Control()
    result = wait_collect_job(backend, "job", window_label="window 3", wait_timeout_sec=timeout,
                              monotonic=clock.monotonic, sleep=clock.sleep, control=control)
    return result, control


class WaitCollectJobTest(unittest.TestCase):
    def test_collects_once_after_a_terminal_state(self):
        backend = Backend([JOB_RUNNING, JOB_RUNNING, JOB_SUCCEEDED])
        result, control = run(backend)
        self.assertEqual((result.kind, result.outcome, result.status.state), (COLLECTED, "outcome", JOB_SUCCEEDED))
        self.assertEqual(backend.calls, ["status", "status", "status", "collect"])
        self.assertEqual(control.names(), ["terminal", "collected"])

    def test_stop_cancels_every_poll_and_never_collects_a_live_job(self):
        control = Control()
        control.stop = True
        backend = Backend([JOB_RUNNING, JOB_RUNNING, JOB_CANCELLED])
        result, control = run(backend, control)
        self.assertEqual(backend.calls, ["status", "cancel", "status", "cancel", "status", "collect"])
        self.assertEqual(result.kind, COLLECTED)
        self.assertEqual(control.names(), ["cancel_requested", "cancel_requested", "terminal", "collected"])

    def test_watchdog_cancels_once_and_reports_it(self):
        backend = Backend([JOB_RUNNING] * 12 + [JOB_CANCELLED])
        result, control = run(backend, timeout=1.0)
        self.assertEqual(backend.calls.count("cancel"), 1)
        self.assertEqual(result.kind, COLLECTED)
        watchdog = [call for call in control.calls if call[0] == "watchdog_fired"]
        self.assertEqual(watchdog, [("watchdog_fired", "job",
                                     "window 3 reconstruction watchdog fired; waiting for lower action terminal state")])

    def test_watchdog_cancel_error_is_recorded_and_not_retried(self):
        backend = Backend([JOB_RUNNING] * 12 + [JOB_CANCELLED], cancel_error=RuntimeError("no"))
        result, control = run(backend, timeout=1.0)
        self.assertEqual(backend.calls.count("cancel"), 1)
        self.assertIn(("cancel_failed", "window 3 backend cancel failed during watchdog: RuntimeError: no"),
                      control.calls)

    def test_status_error_cancels_and_keeps_polling_then_remembers_the_first_failure(self):
        backend = Backend([RuntimeError("down"), JOB_RUNNING, JOB_SUCCEEDED])
        result, control = run(backend, timeout=5.0)
        self.assertEqual(backend.calls, ["status", "cancel", "status", "status", "collect"])
        self.assertEqual(result.kind, COLLECTED)
        self.assertEqual(result.status_failure, "backend status failed: RuntimeError: down")

    def test_status_error_while_stopping_returns_with_the_owner_retained(self):
        control = Control()
        control.stop = True
        backend = Backend([RuntimeError("down")])
        result, control = run(backend, control)
        self.assertEqual(result.kind, STOP_CANCELLED_WITH_OWNER_RETAINED)
        self.assertEqual(backend.calls, ["status", "cancel"])
        self.assertNotIn("collected", control.names())

    def test_status_unavailable_through_the_deadline_fails_closed(self):
        backend = Backend([RuntimeError("down")])
        result, control = run(backend, timeout=1.0)
        self.assertEqual(result.kind, STATUS_DEADLINE_WITH_OWNER_RETAINED)
        self.assertEqual(result.error, "backend status failed: RuntimeError: down; "
                                       "status unavailable through reconstruction deadline")
        self.assertEqual(control.names().count("status_deadline"), 1)
        self.assertEqual(control.names()[-1], "status_deadline")
        self.assertNotIn("collect", backend.calls)

    def test_collect_error_returns_the_terminal_status_and_cleanup_failure(self):
        backend = Backend([JOB_FAILED], collect_errors=1)
        result, control = run(backend)
        self.assertEqual(result.kind, COLLECT_ERROR)
        self.assertTrue(result.cleanup_failure)
        self.assertEqual(result.status.state, JOB_FAILED)
        self.assertEqual(result.error, "backend collect_result failed: RuntimeError: collect broke")
        self.assertEqual(control.names(), ["terminal", "collect_failed"])
        self.assertEqual(control.calls[-1][2],
                         "window 3 backend collect_result failed for job job: RuntimeError: collect broke")


if __name__ == "__main__":
    unittest.main()
