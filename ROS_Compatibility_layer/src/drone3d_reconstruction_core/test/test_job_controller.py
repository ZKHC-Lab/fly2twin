"""Unit tests for single-worker lifecycle and terminal-state convergence."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from drone3d_reconstruction_core import (
    BusyError,
    JobController,
    WarmWorkerController,
)
from drone3d_reconstruction_core.warm_worker_pool import WarmWorkerUnavailableError


FAKE_WORKER = Path(__file__).resolve().with_name("fake_worker.py")


class Recorder:
    def __init__(self):
        self.items = []
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)

    def __call__(self, item):
        with self.changed:
            self.items.append(item)
            self.changed.notify_all()

    def wait_one(self, timeout=3.0):
        return self.wait_count(1, timeout)[0]

    def wait_count(self, count, timeout=3.0, quiet_sec=0.1):
        deadline = time.monotonic() + timeout
        with self.changed:
            while len(self.items) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"worker did not produce {count} terminal callbacks"
                    )
                self.changed.wait(remaining)
            quiet_deadline = min(deadline, time.monotonic() + quiet_sec)
            while time.monotonic() < quiet_deadline:
                self.changed.wait(quiet_deadline - time.monotonic())
            if len(self.items) != count:
                raise AssertionError(
                    f"expected {count} callbacks, got {len(self.items)}"
                )
            return list(self.items)
    def assert_exact_count(self, count, quiet_sec=0.1):
        time.sleep(quiet_sec)
        with self.lock:
            if len(self.items) != count:
                raise AssertionError(
                    f"expected exactly {count} callbacks, got {len(self.items)}"
                )


def command(mode, *extra):
    return [sys.executable, str(FAKE_WORKER), mode, *extra]


class OneShotThreadStartFailure:
    def __init__(self, target_name):
        self.target_name = target_name
        self.failed = False

    def __call__(self, *, target, args, daemon):
        thread = threading.Thread(target=target, args=args, daemon=daemon)
        if target.__name__ != self.target_name or self.failed:
            return thread
        self.failed = True

        def fail_start():
            raise RuntimeError(f"injected {self.target_name} dispatch failure")

        thread.start = fail_start
        return thread


class OneShotThreadBaseFailure:
    def __init__(self, target_name, exception):
        self.target_name = target_name
        self.exception = exception
        self.failed = False

    def __call__(self, *, target, args, daemon):
        thread = threading.Thread(target=target, args=args, daemon=daemon)
        if target.__name__ != self.target_name or self.failed:
            return thread
        self.failed = True

        def fail_start():
            raise self.exception

        thread.start = fail_start
        return thread


class TrackingPopen:
    def __init__(self):
        self.processes = []

    def __call__(self, *args, **kwargs):
        process = subprocess.Popen(*args, **kwargs)
        self.processes.append(process)
        return process


class JobControllerTest(unittest.TestCase):
    def controller(self, recorder, *, timeout=2.0, grace=0.1, **kwargs):
        return JobController(
            worker_timeout_sec=timeout,
            term_grace_sec=grace,
            on_finished=recorder,
            **kwargs,
        )

    def test_success_and_failure_exit_codes(self):
        for mode, expected in (("success", 0), ("fail", 23)):
            recorder = Recorder()
            controller = self.controller(recorder)
            controller.start(command(mode))
            result = recorder.wait_one()
            self.assertEqual(result.cause, "worker_exit")
            self.assertEqual(result.returncode, expected)
            self.assertFalse(controller.is_busy())

    def test_started_precedes_immediate_completion_with_matching_identity(self):
        events = []

        def on_finished(item):
            events.append(("finished", item.job_id))
            recorder(item)

        recorder = Recorder()
        controller = JobController(
            worker_timeout_sec=2.0,
            term_grace_sec=0.05,
            on_finished=on_finished,
        )

        returned_job_id = controller.start(
            command("success"),
            on_started=lambda job_id: events.append(("started", job_id)),
        )
        result = recorder.wait_one()

        self.assertEqual(returned_job_id, 1)
        self.assertEqual(result.job_id, returned_job_id)
        self.assertEqual(events, [("started", 1), ("finished", 1)])
        recorder.assert_exact_count(1)

    def test_busy_rejects_second_start_without_queueing(self):
        recorder = Recorder()
        controller = self.controller(recorder)
        controller.start(command("sleep", "--seconds", "10"))
        with self.assertRaises(BusyError):
            controller.start(command("success"))
        self.assertTrue(controller.cancel())
        self.assertEqual(recorder.wait_one().cause, "cancelled")

    def test_cancel_terminates_and_reaps_worker(self):
        recorder = Recorder()
        controller = self.controller(recorder)
        controller.start(command("sleep", "--seconds", "10"))
        self.assertTrue(controller.cancel())
        result = recorder.wait_one()
        self.assertEqual(result.cause, "cancelled")
        self.assertLess(result.elapsed_sec, 2.0)
        self.assertFalse(controller.cancel())

    def test_timeout_kills_term_ignoring_worker(self):
        recorder = Recorder()
        controller = self.controller(recorder, timeout=0.5, grace=0.05)
        with tempfile.TemporaryDirectory() as directory:
            ready_file = Path(directory) / "ready"
            controller.start(
                command(
                    "ignore-term",
                    "--seconds",
                    "10",
                    "--ready-file",
                    str(ready_file),
                )
            )
            _wait_for_path(ready_file)
            result = recorder.wait_one()
        self.assertEqual(result.cause, "timeout")
        self.assertEqual(result.returncode, -signal.SIGKILL)
        self.assertFalse(controller.is_busy())

    def test_per_job_timeout_override_fires_before_default(self):
        recorder = Recorder()
        controller = self.controller(recorder, timeout=10.0, grace=0.05)
        with tempfile.TemporaryDirectory() as directory:
            ready_file = Path(directory) / "ready"
            controller.start(
                command(
                    "ignore-term",
                    "--seconds",
                    "10",
                    "--ready-file",
                    str(ready_file),
                ),
                worker_timeout_sec=0.5,
            )
            _wait_for_path(ready_file)
            result = recorder.wait_one()
        self.assertEqual(result.cause, "timeout")
        self.assertEqual(result.returncode, -signal.SIGKILL)
        self.assertFalse(controller.is_busy())

    def test_per_job_timeout_override_validation(self):
        recorder = Recorder()
        controller = self.controller(recorder, timeout=10.0, grace=0.05)
        with self.assertRaises(ValueError):
            controller.start(command("success"), worker_timeout_sec=0.0)
        with self.assertRaises(ValueError):
            controller.start(command("success"), worker_timeout_sec=-1.0)
        with self.assertRaises(ValueError):
            controller.start(command("success"), worker_timeout_sec=True)

    def test_cancel_kills_descendant_process_group(self):
        recorder = Recorder()
        controller = self.controller(recorder, grace=0.05)
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            ready_file = Path(directory) / "child.ready"
            controller.start(
                command(
                    "spawn-child",
                    "--seconds",
                    "10",
                    "--child-pid-file",
                    str(pid_file),
                    "--child-ready-file",
                    str(ready_file),
                )
            )
            _wait_for_path(pid_file)
            _wait_for_path(ready_file)
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            self.assertTrue(controller.cancel())
            self.assertEqual(recorder.wait_one().cause, "cancelled")

            deadline = time.monotonic() + 2.0
            while _pid_exists(child_pid) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(_pid_exists(child_pid))

    def test_normal_leader_exit_cleans_descendant_before_callback(self):
        callback_observation = {}

        def record_after_group_cleanup(item):
            child_pid = callback_observation["child_pid"]
            callback_observation["child_live_in_callback"] = _pid_is_live(child_pid)
            recorder(item)

        recorder = Recorder()
        controller = JobController(
            worker_timeout_sec=2.0,
            term_grace_sec=0.1,
            on_finished=record_after_group_cleanup,
        )
        with tempfile.TemporaryDirectory() as directory:
            pid_file = Path(directory) / "child.pid"
            ready_file = Path(directory) / "child.ready"
            controller.start(
                command(
                    "spawn-child-exit",
                    "--seconds",
                    "10",
                    "--child-pid-file",
                    str(pid_file),
                    "--child-ready-file",
                    str(ready_file),
                )
            )
            _wait_for_path(pid_file)
            _wait_for_path(ready_file)
            child_pid = int(pid_file.read_text(encoding="utf-8"))
            callback_observation["child_pid"] = child_pid
            result = recorder.wait_one()
            self.assertEqual(result.cause, "worker_exit")
            self.assertEqual(result.returncode, 0)
            self.assertFalse(callback_observation["child_live_in_callback"])

            deadline = time.monotonic() + 2.0
            while _pid_exists(child_pid) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(_pid_exists(child_pid))
            self.assertFalse(controller.is_busy())

    def test_spawn_failure_releases_capacity_without_callback(self):
        recorder = Recorder()

        def fail_spawn(*args, **kwargs):
            raise OSError("exec failed")

        controller = self.controller(recorder, popen_factory=fail_spawn)
        with self.assertRaisesRegex(OSError, "exec failed"):
            controller.start(command("success"))
        self.assertFalse(controller.is_busy())
        self.assertEqual(recorder.items, [])

    def test_on_started_failure_reaps_and_releases_capacity(self):
        recorder = Recorder()
        popen = TrackingPopen()
        controller = self.controller(
            recorder,
            grace=0.05,
            popen_factory=popen,
        )

        def fail_started(job_id):
            self.assertEqual(job_id, 1)
            raise RuntimeError("injected on_started failure")

        with self.assertRaisesRegex(RuntimeError, "injected on_started failure"):
            controller.start(
                command("sleep", "--seconds", "10"),
                on_started=fail_started,
            )

        self.assertEqual(popen.processes[0].poll(), -signal.SIGTERM)
        result = recorder.wait_one()
        self.assertEqual(result.cause, "dispatch_failed")
        self.assertEqual(result.job_id, 1)
        self.assertFalse(controller.is_busy())
        controller.start(command("success"))
        second = recorder.wait_count(2)[1]
        self.assertEqual(second.returncode, 0)
        self.assertEqual(second.job_id, 2)

    def test_on_started_can_cancel_when_stopper_dispatch_fails(self):
        recorder = Recorder()
        controller = self.controller(
            recorder,
            grace=0.05,
            thread_factory=OneShotThreadStartFailure("_stop"),
        )

        def cancel_from_started(job_id):
            self.assertEqual(job_id, 1)
            self.assertTrue(controller.cancel())

        controller.start(
            command("sleep", "--seconds", "10"),
            on_started=cancel_from_started,
        )

        self.assertEqual(recorder.wait_one().cause, "cancelled")
        self.assertFalse(controller.is_busy())

    def test_shutdown_blocks_until_term_ignoring_worker_is_reaped(self):
        recorder = Recorder()
        controller = self.controller(recorder, grace=0.05)
        with tempfile.TemporaryDirectory() as directory:
            ready_file = Path(directory) / "ready"
            controller.start(
                command(
                    "ignore-term",
                    "--seconds",
                    "10",
                    "--ready-file",
                    str(ready_file),
                )
            )
            _wait_for_path(ready_file)
            controller.shutdown()
        result = recorder.wait_one()
        self.assertEqual(result.cause, "shutdown")
        self.assertEqual(result.returncode, -signal.SIGKILL)
        self.assertFalse(controller.is_busy())
        controller.shutdown()
        time.sleep(0.05)
        self.assertEqual(len(recorder.items), 1)

    def test_monitor_dispatch_failure_reaps_and_releases_capacity(self):
        recorder = Recorder()
        popen = TrackingPopen()
        controller = self.controller(
            recorder,
            grace=0.05,
            popen_factory=popen,
            thread_factory=OneShotThreadStartFailure("_monitor"),
        )

        with self.assertRaisesRegex(RuntimeError, "injected _monitor"):
            controller.start(command("sleep", "--seconds", "10"))

        self.assertEqual(popen.processes[0].poll(), -signal.SIGTERM)
        self.assertEqual(recorder.wait_one().cause, "dispatch_failed")
        self.assertFalse(controller.is_busy())
        controller.start(command("success"))
        self.assertEqual(recorder.wait_count(2)[1].returncode, 0)

    def test_stopper_dispatch_failure_falls_back_synchronously_for_cancel(self):
        recorder = Recorder()
        popen = TrackingPopen()
        controller = self.controller(
            recorder,
            grace=0.05,
            popen_factory=popen,
            thread_factory=OneShotThreadStartFailure("_stop"),
        )
        controller.start(command("sleep", "--seconds", "10"))

        self.assertTrue(controller.cancel())

        self.assertEqual(popen.processes[0].poll(), -signal.SIGTERM)
        self.assertEqual(recorder.wait_one().cause, "cancelled")
        self.assertFalse(controller.is_busy())
        controller.start(command("success"))
        self.assertEqual(recorder.wait_count(2)[1].returncode, 0)

    def test_stopper_base_exception_cleans_up_and_propagates(self):
        recorder = Recorder()
        popen = TrackingPopen()
        controller = self.controller(
            recorder,
            grace=0.05,
            popen_factory=popen,
            thread_factory=OneShotThreadBaseFailure(
                "_stop", KeyboardInterrupt("injected interrupt")
            ),
        )
        controller.start(command("sleep", "--seconds", "10"))

        with self.assertRaisesRegex(KeyboardInterrupt, "injected interrupt"):
            controller.cancel()

        self.assertEqual(popen.processes[0].poll(), -signal.SIGTERM)
        self.assertEqual(recorder.wait_one().cause, "cancelled")
        self.assertFalse(controller.is_busy())

    def test_stopper_dispatch_failure_falls_back_synchronously_for_timeout(self):
        recorder = Recorder()
        popen = TrackingPopen()
        controller = self.controller(
            recorder,
            timeout=0.05,
            grace=0.05,
            popen_factory=popen,
            thread_factory=OneShotThreadStartFailure("_stop"),
        )
        controller.start(command("sleep", "--seconds", "10"))

        result = recorder.wait_one()

        self.assertEqual(result.cause, "timeout")
        self.assertEqual(popen.processes[0].poll(), -signal.SIGTERM)
        self.assertFalse(controller.is_busy())

    def test_stopper_dispatch_failure_falls_back_synchronously_for_shutdown(self):
        recorder = Recorder()
        popen = TrackingPopen()
        controller = self.controller(
            recorder,
            grace=0.05,
            popen_factory=popen,
            thread_factory=OneShotThreadStartFailure("_stop"),
        )
        controller.start(command("sleep", "--seconds", "10"))

        controller.shutdown()

        self.assertEqual(popen.processes[0].poll(), -signal.SIGTERM)
        self.assertEqual(recorder.wait_one().cause, "shutdown")
        self.assertFalse(controller.is_busy())

    def test_cleanup_failure_still_releases_capacity(self):
        recorder = Recorder()

        def fail_quiescence(*args, **kwargs):
            raise RuntimeError("injected quiescence failure")

        controller = self.controller(
            recorder,
            grace=0.05,
            group_quiescence_waiter=fail_quiescence,
        )
        controller.start(command("sleep", "--seconds", "10"))

        self.assertTrue(controller.cancel())

        result = recorder.wait_one()
        self.assertEqual(result.cause, "cleanup_failed")
        self.assertFalse(controller.is_busy())
        recorder.assert_exact_count(1)

    def test_terminal_callback_exception_is_contained(self):
        event = threading.Event()

        def fail_callback(item):
            event.set()
            raise ValueError("injected terminal callback failure")

        controller = JobController(
            worker_timeout_sec=2.0,
            term_grace_sec=0.05,
            on_finished=fail_callback,
            thread_factory=OneShotThreadStartFailure("_stop"),
        )
        controller.start(command("sleep", "--seconds", "10"))

        self.assertTrue(controller.cancel())

        self.assertTrue(event.wait(2.0))
        self.assertFalse(controller.is_busy())

    def test_shutdown_waits_for_on_started_before_terminal_callback(self):
        events = []
        started_entered = threading.Event()
        started_release = threading.Event()
        finished = threading.Event()

        def on_finished(item):
            events.append(("finished", item.job_id, item.cause))
            finished.set()

        controller = JobController(
            worker_timeout_sec=2.0,
            term_grace_sec=0.05,
            on_finished=on_finished,
            thread_factory=OneShotThreadStartFailure("_stop"),
        )

        def on_started(job_id):
            started_entered.set()
            self.assertTrue(started_release.wait(2.0))
            events.append(("started", job_id))

        start_thread = threading.Thread(
            target=controller.start,
            args=(command("sleep", "--seconds", "10"),),
            kwargs={"on_started": on_started},
        )
        start_thread.start()
        self.assertTrue(started_entered.wait(1.0))
        shutdown_thread = threading.Thread(target=controller.shutdown)
        shutdown_thread.start()
        time.sleep(0.05)
        self.assertFalse(finished.is_set())

        started_release.set()
        start_thread.join(2.0)
        shutdown_thread.join(2.0)

        self.assertEqual(events[0], ("started", 1))
        self.assertEqual(events[1], ("finished", 1, "shutdown"))
        self.assertFalse(controller.is_busy())

    def test_invalid_configuration_and_command_are_rejected(self):
        recorder = Recorder()
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.controller(recorder, timeout=value)
        controller = self.controller(recorder)
        for invalid in ([], [""], [None]):
            with self.assertRaises(ValueError):
                controller.start(invalid)


WARM_PROTOCOL_WORKER = r"""
import json
import sys

for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        job = json.loads(raw)
    except ValueError:
        continue
    if job.get("action") == "shutdown":
        break
    if job.get("slow"):
        import time
        time.sleep(60)
    ok = not job.get("fail")
    print(json.dumps({
        "job_id": int(job.get("job_id", -1)),
        "ok": ok,
        "elapsed_sec": 0.25,
        "error": None if ok else "injected warm failure",
    }), flush=True)
"""


def warm_command():
    return [sys.executable, "-c", WARM_PROTOCOL_WORKER]


class FailOnceThenRealPopen:
    """One injected spawn failure, then normal subprocess.Popen."""

    def __init__(self):
        self.calls = 0
        self.processes = []

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise OSError("injected warm spawn failure")
        process = subprocess.Popen(*args, **kwargs)
        self.processes.append(process)
        return process


class WarmWorkerControllerTest(unittest.TestCase):
    def test_idle_release_allows_fresh_worker_without_shutting_controller(self):
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(recorder, directory)
            try:
                controller.start_warm(warm_command(), {"images_dir": "/tmp/first"})
                recorder.wait_one()
                first_pid = controller.resident_worker_pid()
                self.assertIsNotNone(first_pid)
                controller.release_idle_worker()
                self.assertIsNone(controller.resident_worker_pid())
                controller.start_warm(warm_command(), {"images_dir": "/tmp/second"})
                recorder.wait_count(2)
                self.assertNotEqual(controller.resident_worker_pid(), first_pid)
            finally:
                controller.shutdown()

    def controller(self, recorder, directory, *, timeout=3.0, grace=0.1, **kwargs):
        return WarmWorkerController(
            worker_timeout_sec=timeout,
            term_grace_sec=grace,
            on_finished=recorder,
            warm_command=warm_command(),
            pool_log_path=str(Path(directory) / "warm-worker.log"),
            **kwargs,
        )

    def test_success_reuses_one_persistent_process(self):
        recorder = Recorder()
        tracker = TrackingPopen()
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(recorder, directory, popen_factory=tracker)
            try:
                first = controller.start_warm(
                    warm_command(), {"images_dir": "/tmp/first"}
                )
                first_result = recorder.wait_one()
                self.assertEqual(first, 1)
                self.assertEqual(first_result.job_id, 1)
                self.assertEqual(first_result.cause, "worker_exit")
                self.assertEqual(first_result.returncode, 0)
                self.assertGreater(first_result.elapsed_sec, 0)
                self.assertFalse(controller.is_busy())

                second = controller.start_warm(
                    warm_command(), {"images_dir": "/tmp/second"}
                )
                first_result, second_result = recorder.wait_count(2)
                self.assertEqual(second, 2)
                self.assertEqual(second_result.job_id, 2)
                self.assertEqual(second_result.cause, "worker_exit")
                self.assertEqual(second_result.returncode, 0)
                self.assertEqual(len(tracker.processes), 1)
            finally:
                controller.shutdown()
        self.assertFalse(controller.is_busy())

    def test_job_failure_reports_worker_exit_with_nonzero_code(self):
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(recorder, directory)
            try:
                controller.start_warm(warm_command(), {"fail": True})
                result = recorder.wait_one()
                self.assertEqual(result.cause, "worker_exit")
                self.assertEqual(result.returncode, 1)
            finally:
                controller.shutdown()

    def test_spawn_failure_raises_before_dispatch_then_oneshot_fallback_works(self):
        recorder = Recorder()
        popen = FailOnceThenRealPopen()
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(recorder, directory, popen_factory=popen)
            try:
                with self.assertRaises(WarmWorkerUnavailableError) as caught:
                    controller.start_warm(warm_command(), {"images_dir": "/tmp/x"})
                self.assertEqual(caught.exception.reason, "spawn_failed")
                self.assertEqual(recorder.items, [])
                self.assertFalse(controller.is_busy())

                # 退回路径必须仍是可用的一次性 JobController。
                controller.start(command("success"))
                result = recorder.wait_one()
                self.assertEqual(result.cause, "worker_exit")
                self.assertEqual(result.returncode, 0)
            finally:
                controller.shutdown()

    def test_busy_rejects_second_warm_job_and_cancel_converges(self):
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(recorder, directory)
            try:
                controller.start_warm(warm_command(), {"slow": True})
                with self.assertRaises(BusyError):
                    controller.start_warm(warm_command(), {"images_dir": "/tmp/y"})
                self.assertTrue(controller.is_busy())
                self.assertTrue(controller.cancel())
                result = recorder.wait_one()
                self.assertEqual(result.cause, "cancelled")
            finally:
                controller.shutdown()

    def test_timeout_kills_warm_process_group(self):
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(recorder, directory, timeout=0.2)
            try:
                controller.start_warm(warm_command(), {"slow": True})
                result = recorder.wait_one()
                self.assertEqual(result.cause, "timeout")
            finally:
                controller.shutdown()

    def test_job_id_injection_and_bad_commands_are_rejected(self):
        recorder = Recorder()
        with tempfile.TemporaryDirectory() as directory:
            controller = self.controller(recorder, directory)
            try:
                with self.assertRaises(ValueError):
                    controller.start_warm(
                        warm_command(), {"job_id": 99, "images_dir": "/tmp/z"}
                    )
                with self.assertRaises(ValueError):
                    controller.start_warm(
                        ["", warm_command()[-1]], {"images_dir": "/tmp/z"}
                    )
                self.assertEqual(recorder.items, [])
            finally:
                controller.shutdown()


def _wait_for_path(path, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not path.exists():
        raise AssertionError(f"path did not appear: {path}")


def _pid_is_live(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return len(stat) > 2 and stat[2] != "Z"


def _pid_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


if __name__ == "__main__":
    unittest.main()
