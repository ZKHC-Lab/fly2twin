"""Pin tests for reconstruction job execution: stop, asynchronous cancel and resource pauses.

They hold the scheduler's backend calls at Event gates instead of hoping a sleep hits a race,
and check what an operator or the journal can observe: public state and counters, the
backend call trajectory, and one terminal per window. Written before the wait/cancel/collect
state machine is moved out of ContinuousSession (S2 step 4), to keep its protocol intact.
"""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_continuous_session import (  # noqa: E402  (helpers only, no TestCase classes)
    BlockingCaptureClient,
    FakeBackend,
    FakeCaptureClient,
    _patch_g4_process,
    make_capture_config,
)
from drone3d_reconstruction_core.action_types import (  # noqa: E402
    ACTION_FAILED,
    CameraRole,
    ReconstructActionOutcome,
)
from drone3d_reconstruction_core.continuous_session import (  # noqa: E402
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    QUEUE_LOSSLESS,
    SESSION_DRAINING,
    SESSION_MEMORY_PAUSED,
    SESSION_RUNNING,
    SESSION_STOPPED,
    SESSION_STOPPING,
    ContinuousConfig,
    ContinuousSession,
    JobStatus,
)

GUARD_TEXT = "worker stopped: resource_guard HIGH_SWAP_USED"


def lock_free(session) -> bool:
    """True when another thread can take the session lock right now (get_status takes it)."""
    probe = threading.Thread(target=session.get_status, daemon=True)
    probe.start()
    probe.join(2.0)
    return not probe.is_alive()


def failed_outcome(job, message):
    return ReconstructActionOutcome(
        action_state=ACTION_FAILED, error_code=3, message=message, scene_id=job["scene_id"],
        result_manifest_uri="", result_manifest_sha256="", scene_glb_uri="", frame_count=1,
        wall_time_sec=0.1, camera_role="front", camera_topic="/front/image_raw",
        camera_frame_id="front_camera_optical_frame", snapshot_manifest_sha256=job["manifest_sha256"],
        result_scene_id=job["scene_id"], result_actual_frame_count=1,
    )


class ScriptedBackend(FakeBackend):
    """Every call is logged and checks the session lock is free; any call can be held at a gate.

    Jobs start RUNNING. ``status_fn(job_id, n)``, ``cancel_fn(job_id, n)`` and
    ``collect_fn(job_id, n)`` (n counts that operation across jobs, from 1) override the
    default behaviour; they may raise.
    """

    def __init__(self):
        super().__init__()
        self.session = None
        self.calls = []
        self.counts = Counter()
        self.entered = {op: threading.Event() for op in ("submit", "status", "cancel", "collect")}
        self.gates = {}
        self.lock_probes = []
        self.status_fn = self.cancel_fn = self.collect_fn = None
        self._log = threading.Lock()

    def _enter(self, op, job_id):
        with self._log:
            self.counts[op] += 1
            n = self.counts[op]
            self.calls.append((op, job_id))
        if self.session is not None:
            self.lock_probes.append((op, lock_free(self.session)))
        self.entered[op].set()
        gate = self.gates.get(op)
        if gate is not None:
            assert gate.wait(10.0), f"{op} gate never released"
        return n

    def submit(self, snapshot_uri, manifest_sha256, window_id, inference_resolution=0, worker_timeout_sec=0.0):
        self._enter("submit", window_id)
        job_id = super().submit(snapshot_uri, manifest_sha256, window_id, inference_resolution, worker_timeout_sec)
        job_id = f"{window_id}#{self.counts['submit']}"
        self.jobs[job_id] = {"snapshot_uri": snapshot_uri, "manifest_sha256": manifest_sha256,
                             "scene_id": window_id, "state": JOB_RUNNING}
        self.submitted[-1] = job_id
        with self._log:  # the submit entry was logged under the window id
            index = max(i for i, call in enumerate(self.calls) if call == ("submit", window_id))
            self.calls[index] = ("submit", job_id)
        return job_id

    def status(self, job_id):
        n = self._enter("status", job_id)
        if self.status_fn is not None:
            status = self.status_fn(job_id, n)
            self.jobs[job_id]["state"] = status.state  # the outcome follows the reported state
            return status
        return JobStatus(job_id, self.jobs[job_id]["state"], "")

    def cancel(self, job_id):
        n = self._enter("cancel", job_id)
        if self.cancel_fn is not None:
            return self.cancel_fn(job_id, n)
        if self.jobs[job_id]["state"] == JOB_RUNNING:
            self.jobs[job_id]["state"] = JOB_CANCELLED

    def collect_result(self, job_id):
        n = self._enter("collect", job_id)
        if self.collect_fn is not None:
            return self.collect_fn(job_id, n)
        return self.result(job_id)

    def result(self, job_id):
        """FakeBackend's outcome for this attempt, under the window's scene id."""
        scene = self.jobs[job_id]["scene_id"]
        outcome = FakeBackend.collect_result(self, job_id)
        return replace(outcome, scene_id=scene, result_scene_id=scene)

    def ops(self, job_id):
        return [op for op, job in self.calls if job == job_id]


class JobExecutionPinTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.journal = root / "journal"
        self.journal.mkdir()
        results = root / "results"
        results.mkdir()
        role = CameraRole("front", "/front/capture_segment",
                          replace(make_capture_config(root, "front"), odometry_association="interpolate"))
        self.role, self.front_root = role, Path(role.capture_config.snapshot_root)
        self.config = ContinuousConfig(
            roles=(role,), reconstruct_action="/reconstruct/reconstruct_segment",
            journal_root=str(self.journal), result_root=str(results),
            service_name="/continuous_session/command", diagnostics_topic="/diagnostics",
            diagnostics_period_sec=1.0, min_free_disk_gb=0.01, action_connect_timeout_sec=1.0,
            capture_wait_margin_sec=1.0, reconstruct_wait_timeout_sec=5.0, stop_timeout_sec=0.5,
            reconstruct_frame_group_size=6, initial_sample_fps=5.0, initial_window_frames=6,
            initial_overlap_frames=5, initial_inference_resolution=384,
            initial_queue_mode=QUEUE_LOSSLESS, initial_queue_capacity=8, initial_max_retries=0,
            initial_keep_snapshot_on_success=True,
        )
        _patch_g4_process(self)
        self.service = None
        self.captures = []

    def tearDown(self):
        backend = getattr(self, "backend", None)
        if backend is not None:
            for gate in backend.gates.values():
                gate.set()
        if self.service is not None:
            try:
                self.service.shutdown(2.0)
            except BaseException:
                pass

    # ------------------------------------------------------------------
    def start(self, backend, windows=1, config=None, capture_clients=None):
        """Start a session whose first ``windows`` captures succeed; later ones never finish."""
        self.backend = backend

        def factory(_role):
            if capture_clients is not None and len(self.captures) < len(capture_clients):
                client = capture_clients[len(self.captures)]
            elif len(self.captures) < windows:
                client = FakeCaptureClient(self.role, self.front_root)
            else:
                client = UnfinishedCaptureClient(self.role, self.front_root)
            self.captures.append(client)
            return client

        self.service = ContinuousSession(config or self.config, capture_client_factory=factory,
                                         backend_factory=lambda: backend,
                                         disk_free_gb=lambda: 100.0,
                                         result_gate=_ok_gate, result_publisher=_fake_publisher)
        backend.session = self.service
        code, message = self.service.start_session("front")
        self.assertEqual(code, 0, message)
        return self.service

    def wait_until(self, predicate, timeout=8.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        raise AssertionError("condition not reached")

    def events(self, name=None):
        lines = []
        for path in sorted(self.journal.glob("*.jsonl")):
            lines += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [e for e in lines if name is None or e.get("event") == name]

    def last_stop_event(self):
        return [e for e in self.events() if e.get("event") in ("session_stop", "session_stop_pending")][-1]

    def assert_one_terminal_per_window(self):
        terminal = Counter(e["window_index"] for e in self.events()
                           if e.get("event") in ("window_failed", "window_succeeded", "session_fatal_oom"))
        self.assertTrue(all(count == 1 for count in terminal.values()), terminal)

    def assert_lock_never_held_during_backend_calls(self):
        held = [op for op, free in self.backend.lock_probes if not free]
        self.assertEqual(held, [])

    def stop_until_stopped(self):
        """Hard STOP until the session proves every job terminal and collected."""
        return self.wait_until(lambda: self.service.stop_all(discard_queue=True)[0] == 0)

    # ------------------------------------------------------------------
    # 1. A submit still blocked when STOP arrives
    # ------------------------------------------------------------------
    def test_1_hard_stop_during_blocked_submit_binds_and_collects_the_late_job(self):
        backend = ScriptedBackend()
        backend.gates["submit"] = threading.Event()
        service = self.start(backend)
        self.assertTrue(backend.entered["submit"].wait(5.0))
        started = time.monotonic()
        code, message = service.stop_all(discard_queue=True)
        self.assertLess(time.monotonic() - started, self.config.stop_timeout_sec + 1.0)
        self.assertEqual(code, 3, message)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)
        backend.gates["submit"].set()
        self.wait_until(lambda: backend.counts["collect"] == 1)
        self.stop_until_stopped()
        job = backend.submitted[0]
        trajectory = backend.ops(job)
        self.assertEqual(len(backend.submitted), 1)
        self.assertIn("cancel", trajectory)
        self.assertLess(trajectory.index("cancel"), trajectory.index("collect"))
        self.assertEqual(trajectory.count("collect"), 1)
        stop = self.last_stop_event()
        self.assertEqual((stop["event"], stop["backend_job_id_before"]), ("session_stop", job))
        self.assertTrue(stop["backend_cancel_requested"])
        self.assertTrue(stop["backend_terminal_collected"])
        self.assertEqual(service.get_status().failed_windows, 0)
        self.assert_lock_never_held_during_backend_calls()

    # ------------------------------------------------------------------
    # 2. status / cancel / collect never hold the lock; no collect before terminal
    # ------------------------------------------------------------------
    def test_2_blocked_poll_keeps_the_session_responsive_and_collect_waits_for_terminal(self):
        backend = ScriptedBackend()
        backend.gates["status"] = threading.Event()
        backend.cancel_fn = lambda job_id, n: None  # cancellation stays pending
        backend.collect_fn = lambda job_id, n: (_raise(RuntimeError("collect failed")) if n <= 2
                                                else backend.result(job_id))
        service = self.start(backend)
        self.assertTrue(backend.entered["status"].wait(5.0))
        self.assertTrue(lock_free(service))
        started = time.monotonic()
        code, message = service.stop_all(discard_queue=True)
        self.assertLess(time.monotonic() - started, self.config.stop_timeout_sec + 1.0)
        self.assertEqual(code, 3, message)
        _release(backend, "status")
        job = backend.submitted[0]
        self.wait_until(lambda: backend.counts["cancel"] >= 1)
        polls = backend.counts["status"]
        self.wait_until(lambda: backend.counts["status"] >= polls + 3)
        self.assertEqual(backend.counts["collect"], 0)  # cancel requested, not terminal yet
        self.assertEqual(len(backend.submitted), 1)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)
        backend.jobs[job]["state"] = JOB_CANCELLED
        self.wait_until(lambda: backend.counts["collect"] >= 2)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)
        self.stop_until_stopped()
        self.assertEqual(backend.counts["collect"], 3)
        self.assertEqual(len(backend.submitted), 1)
        stop = self.last_stop_event()
        self.assertEqual(stop["event"], "session_stop")
        self.assertTrue(stop["backend_terminal_collected"])
        self.assertEqual(stop["backend_terminal_state"], JOB_CANCELLED)
        self.assertEqual(service.get_status().failed_windows, 0)
        self.assert_lock_never_held_during_backend_calls()

    # ------------------------------------------------------------------
    # 3. status keeps failing
    # ------------------------------------------------------------------
    def test_3a_status_failing_until_the_deadline_fails_once_and_retains_the_job(self):
        backend = ScriptedBackend()
        broken = threading.Event()
        broken.set()
        backend.status_fn = lambda job_id, n: (_raise(RuntimeError("status down")) if broken.is_set()
                                               else JobStatus(job_id, backend.jobs[job_id]["state"], ""))
        backend.cancel_fn = lambda job_id, n: None
        service = self.start(backend, config=replace(self.config, reconstruct_wait_timeout_sec=0.6))
        # STOPPING is latched before the failure is recorded (a separate lock section): wait
        # for the window's failure event, the point where the counters are committed.
        self.wait_until(lambda: self.events("window_failed"))
        status = service.get_status()
        self.assertEqual(status.state, SESSION_STOPPING)
        self.assertEqual(status.failed_windows, 1)
        self.assertEqual(service.diagnostics_snapshot()["attempt_failures"], 1)
        self.assertIn("status unavailable through reconstruction deadline", status.last_error)
        self.assertEqual(backend.counts["collect"], 0)  # the job may still be alive
        self.assertEqual(service.stop_all(discard_queue=True)[0], 3)
        broken.clear()
        backend.jobs[backend.submitted[0]]["state"] = JOB_CANCELLED
        self.stop_until_stopped()
        self.assertEqual(backend.counts["collect"], 1)
        self.assertEqual(len(self.events("window_failed")), 1)
        self.assertEqual(service.get_status().failed_windows, 1)
        self.assert_one_terminal_per_window()
        self.assert_lock_never_held_during_backend_calls()

    def test_3b_stop_while_status_fails_cancels_without_counting_a_failure(self):
        backend = ScriptedBackend()
        broken = threading.Event()
        broken.set()
        backend.status_fn = lambda job_id, n: (_raise(RuntimeError("status down")) if broken.is_set()
                                               else JobStatus(job_id, backend.jobs[job_id]["state"], ""))
        backend.cancel_fn = lambda job_id, n: None
        service = self.start(backend)
        self.wait_until(lambda: backend.counts["status"] >= 2)
        self.assertEqual(service.stop_all(discard_queue=True)[0], 3)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)
        broken.clear()
        backend.jobs[backend.submitted[0]]["state"] = JOB_CANCELLED
        self.stop_until_stopped()
        self.assertEqual(backend.counts["collect"], 1)
        self.assertEqual(service.get_status().failed_windows, 0)
        self.assertEqual(service.diagnostics_snapshot()["attempt_failures"], 0)
        self.assertEqual(self.events("window_failed"), [])
        self.assert_lock_never_held_during_backend_calls()

    # ------------------------------------------------------------------
    # 4. watchdog
    # ------------------------------------------------------------------
    def test_4a_watchdog_cancels_once_and_collects_only_after_a_late_terminal(self):
        backend = ScriptedBackend()
        cancelled_at = []
        backend.cancel_fn = lambda job_id, n: cancelled_at.append(backend.counts["status"])

        def status(job_id, n):
            if cancelled_at and n >= cancelled_at[0] + 4:
                backend.jobs[job_id]["state"] = JOB_CANCELLED
            return JobStatus(job_id, backend.jobs[job_id]["state"], "")

        backend.status_fn = status
        service = self.start(backend, config=replace(self.config, reconstruct_wait_timeout_sec=0.6))
        self.wait_until(lambda: service.get_status().failed_windows == 1)
        job = backend.submitted[0]
        trajectory = backend.ops(job)
        self.assertEqual(trajectory.count("cancel"), 1)
        self.assertEqual(trajectory.count("collect"), 1)
        self.assertGreaterEqual(trajectory[trajectory.index("cancel"):trajectory.index("collect")].count("status"), 3)
        self.assertEqual(service.get_status().state, SESSION_RUNNING)
        self.assertEqual(service.diagnostics_snapshot()["attempt_failures"], 1)
        self.assert_one_terminal_per_window()
        self.assert_lock_never_held_during_backend_calls()

    def test_4b_watchdog_cancel_raising_is_not_repeated_and_a_later_stop_converges(self):
        backend = ScriptedBackend()
        raising = threading.Event()
        raising.set()

        def cancel(job_id, n):
            if raising.is_set():
                raise RuntimeError("cancel refused")
            backend.jobs[job_id]["state"] = JOB_CANCELLED

        backend.cancel_fn = cancel
        service = self.start(backend, config=replace(self.config, reconstruct_wait_timeout_sec=0.6))
        self.wait_until(lambda: backend.counts["cancel"] == 1)
        polls = backend.counts["status"]
        self.wait_until(lambda: backend.counts["status"] >= polls + 3)
        self.assertEqual(backend.counts["cancel"], 1)  # no second automatic watchdog cancel
        self.assertIn("cancel failed during watchdog", service.get_status().last_error)
        self.assertEqual(backend.counts["collect"], 0)
        raising.clear()
        self.stop_until_stopped()
        job = backend.submitted[0]
        self.assertEqual(backend.ops(job).count("collect"), 1)
        stop = self.last_stop_event()
        self.assertEqual(stop["backend_job_id_before"], job)
        self.assertTrue(stop["backend_cancel_requested"])
        self.assertTrue(stop["backend_terminal_collected"])
        self.assertEqual(service.get_status().failed_windows, 0)
        self.assert_lock_never_held_during_backend_calls()

    # ------------------------------------------------------------------
    # 5. First STOP drains, the second cancels; resource pause while draining
    # ------------------------------------------------------------------
    def _first_stop_drains(self, phase):
        backend = ScriptedBackend()
        backend.gates[phase] = threading.Event()
        backend.status_fn = lambda job_id, n: JobStatus(job_id, JOB_SUCCEEDED, "")
        service = self.start(backend)
        self.assertTrue(backend.entered[phase].wait(5.0))
        code, message = service.stop_all()
        self.assertEqual(code, 0, message)
        self.assertEqual(service.get_status().state, SESSION_DRAINING)
        _release(backend, phase)
        self.wait_until(lambda: service.get_status().state == SESSION_STOPPED, timeout=10.0)
        self.assertEqual(service.get_status().succeeded_windows, 1)
        self.assertEqual(backend.counts["cancel"], 0)
        self.assertEqual(len(backend.submitted), 1)
        self.assert_lock_never_held_during_backend_calls()

    def test_5a_first_stop_during_submit_drains(self):
        self._first_stop_drains("submit")

    def test_5b_first_stop_during_poll_drains(self):
        self._first_stop_drains("status")

    def test_5c_first_stop_during_collect_drains(self):
        self._first_stop_drains("collect")

    def test_5d_second_stop_while_draining_cancels(self):
        backend = ScriptedBackend()
        service = self.start(backend)
        self.wait_until(lambda: backend.counts["status"] >= 1)
        self.assertEqual(service.stop_all()[0], 0)
        self.assertEqual(service.get_status().state, SESSION_DRAINING)
        self.stop_until_stopped()
        job = backend.submitted[0]
        self.assertIn("cancel", backend.ops(job))
        self.assertEqual(backend.ops(job).count("collect"), 1)
        self.assertEqual(service.get_status().succeeded_windows, 0)
        self.assertEqual(service.get_status().failed_windows, 0)

    def test_5e_resource_pause_while_draining_resumes_the_same_window_then_the_next(self):
        backend = ScriptedBackend()
        backend.gates["collect"] = threading.Event()
        backend.status_fn = lambda job_id, n: JobStatus(job_id, JOB_SUCCEEDED, "")
        backend.collect_fn = lambda job_id, n: (failed_outcome(backend.jobs[job_id], GUARD_TEXT) if n == 1
                                                else backend.result(job_id))
        service = self.start(backend, windows=2)
        self.assertTrue(backend.entered["collect"].wait(5.0))
        self.wait_until(lambda: service.get_status().pending_windows == 1)
        self.assertEqual(service.stop_all()[0], 0)
        self.assertEqual(service.get_status().state, SESSION_DRAINING)
        captures_before = len(self.captures)
        _release(backend, "collect")
        self.wait_until(lambda: service.get_status().state == SESSION_MEMORY_PAUSED)
        self.assertEqual(service.get_status().failed_windows, 1)
        self.assertEqual(service.get_status().pending_windows, 1)
        self.assertEqual(len(backend.submitted), 1)
        # Hold the retried submit so the state right after RESUME can be observed.
        backend.gates["submit"] = threading.Event()
        backend.entered["submit"].clear()
        code, message = service.resume()
        self.assertEqual(code, 0, message)
        self.assertTrue(backend.entered["submit"].wait(5.0))
        # Documented: back to DRAINING without new capture, then the stop runs by itself.
        self.assertEqual(service.get_status().state, SESSION_DRAINING)
        _release(backend, "submit")
        self.wait_until(lambda: service.get_status().state == SESSION_STOPPED, timeout=10.0)
        status = service.get_status()
        self.assertEqual((status.succeeded_windows, status.failed_windows), (2, 0))
        self.assertEqual(service.diagnostics_snapshot()["attempt_failures"], 1)
        scenes = [job.split("#")[0] for job in backend.submitted]
        self.assertEqual(len(scenes), 3)
        self.assertEqual(scenes[0], scenes[1])  # the paused window first, then the queued one
        self.assertNotEqual(scenes[1], scenes[2])
        self.assertEqual(len(self.captures), captures_before)  # draining: no new capture
        self.assertEqual(len(self.events("window_succeeded")), 2)

    # ------------------------------------------------------------------
    # 6. Failure priorities
    # ------------------------------------------------------------------
    def test_6a_status_failure_wins_over_a_later_successful_result(self):
        backend = ScriptedBackend()
        backend.status_fn = lambda job_id, n: (_raise(RuntimeError("status hiccup")) if n == 1
                                               else JobStatus(job_id, JOB_SUCCEEDED, ""))
        backend.cancel_fn = lambda job_id, n: None
        service = self.start(backend)
        self.wait_until(lambda: service.get_status().failed_windows == 1)
        self.assertTrue(service.get_status().last_error.startswith("backend status failed"))
        self.assertEqual(self.events("window_succeeded"), [])
        self.assertEqual(backend.counts["collect"], 1)
        self.assert_one_terminal_per_window()

    def test_6b_cleanup_failure_with_stop_stays_fail_closed(self):
        backend = ScriptedBackend()
        backend.gates["status"] = threading.Event()
        backend.status_fn = lambda job_id, n: JobStatus(job_id, JOB_FAILED, "cleanup_failed: group alive")
        service = self.start(backend)
        self.assertTrue(backend.entered["status"].wait(5.0))
        self.assertEqual(service.stop_all(discard_queue=True)[0], 3)
        _release(backend, "status")
        self.wait_until(lambda: backend.counts["collect"] == 1)
        self.wait_until(lambda: service.get_status().failed_windows == 1)
        failed = self.events("window_failed")
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0]["reason"].startswith("backend process cleanup failed"))
        code, message = service.stop_all(discard_queue=True)
        self.assertEqual(code, 3, message)
        self.assertIn("cleanup_failed", message)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)

    def test_6c_collect_error_wins_over_a_resource_rejection_returned_by_the_retry(self):
        backend = ScriptedBackend()
        backend.status_fn = lambda job_id, n: JobStatus(job_id, JOB_FAILED, "")
        backend.collect_fn = lambda job_id, n: (_raise(RuntimeError("collect broke")) if n == 1
                                                else failed_outcome(backend.jobs[job_id], GUARD_TEXT))
        service = self.start(backend)
        self.wait_until(lambda: service.get_status().failed_windows == 1)
        self.assertEqual(service.get_status().state, SESSION_RUNNING)
        failed = self.events("window_failed")
        self.assertEqual(len(failed), 1)
        self.assertTrue(failed[0]["reason"].startswith("backend collect_result failed"))
        self.assertEqual(backend.counts["collect"], 2)
        self.assert_one_terminal_per_window()

    # ------------------------------------------------------------------
    # 7. A window being sealed when a resource rejection arrives
    # ------------------------------------------------------------------
    def test_7_window_sealed_during_a_resource_rejection_is_not_lost(self):
        backend = ScriptedBackend()
        backend.gates["collect"] = threading.Event()
        backend.status_fn = lambda job_id, n: JobStatus(job_id, JOB_SUCCEEDED, "")
        backend.collect_fn = lambda job_id, n: (failed_outcome(backend.jobs[job_id], GUARD_TEXT) if n == 1
                                                else backend.result(job_id))
        sealing = SealGateCaptureClient(self.role, self.front_root)
        clients = [FakeCaptureClient(self.role, self.front_root), sealing,
                   FakeCaptureClient(self.role, self.front_root)]
        service = self.start(backend, capture_clients=clients)
        self.assertTrue(backend.entered["collect"].wait(5.0))
        self.assertTrue(sealing.sealing.wait(5.0))  # window 1 is inside outcome()
        _release(backend, "collect")
        self.wait_until(lambda: service.get_status().state == SESSION_MEMORY_PAUSED)
        sealing.release_outcome.set()
        self.assertEqual(len(backend.submitted), 1)  # nothing passed the failed window
        code, message = service.resume()
        self.assertEqual(code, 0, message)
        self.wait_until(lambda: service.get_status().succeeded_windows == 2, timeout=10.0)
        succeeded = [e["window_index"] for e in self.events("window_succeeded")]
        self.assertEqual(succeeded, [0, 1])
        scenes = [job.split("#")[0] for job in backend.submitted]
        self.assertEqual(scenes[0], scenes[1])
        self.assertEqual(service.get_status().failed_windows, 0)
        self.assert_one_terminal_per_window()

    # ------------------------------------------------------------------
    # 8. One failure per attempt, one terminal per window
    # ------------------------------------------------------------------
    def test_8_retry_counts_each_attempt_once_and_ends_with_one_terminal(self):
        backend = ScriptedBackend()
        backend.status_fn = lambda job_id, n: JobStatus(job_id, JOB_SUCCEEDED, "")
        backend.collect_fn = lambda job_id, n: (failed_outcome(backend.jobs[job_id], "worker crashed") if n == 1
                                                else backend.result(job_id))
        service = self.start(backend, config=replace(self.config, initial_max_retries=1))
        self.wait_until(lambda: service.get_status().succeeded_windows == 1)
        self.assertEqual(service.get_status().failed_windows, 0)
        self.assertEqual(service.diagnostics_snapshot()["attempt_failures"], 1)
        self.assertEqual(len(self.events("window_retry")), 1)
        self.assertEqual(self.events("window_failed"), [])
        first, second = backend.submitted[:2]
        self.assertEqual(first.split("#")[0], second.split("#")[0])
        for job in (first, second):
            trajectory = backend.ops(job)
            self.assertEqual(trajectory[0], "submit")
            self.assertEqual(trajectory[-1], "collect")
            self.assertEqual(trajectory.count("collect"), 1)
            self.assertNotIn("cancel", trajectory)
        self.assert_one_terminal_per_window()
        self.assert_lock_never_held_during_backend_calls()


class UnfinishedCaptureClient(BlockingCaptureClient):
    """A capture still collecting frames; cancelling it ends the goal without a snapshot,
    as the real capture action does."""

    def outcome(self):
        self.fail = True
        return super().outcome()


class SealGateCaptureClient(FakeCaptureClient):
    """Capture whose outcome (the sealed snapshot) is held until the test releases it."""

    def __init__(self, role, snapshot_root):
        super().__init__(role, snapshot_root)
        self.sealing = threading.Event()
        self.release_outcome = threading.Event()

    def outcome(self):
        self.sealing.set()
        assert self.release_outcome.wait(10.0)
        return super().outcome()


def _raise(error):
    raise error


def _release(backend, op):
    gate = backend.gates.pop(op, None)
    if gate is not None:
        gate.set()


def _ok_gate(uri, expected_sha256, *, expected_identity, expected_authenticity):
    from types import SimpleNamespace
    return SimpleNamespace(ok=True, reason="", payload={}, manifest_bytes=b"{}")


def _fake_publisher(records_root, session_id, window_index, manifest_bytes, provenance,
                    snapshot_manifest_bytes=b""):
    return Path("/tmp/fake-formal-window"), "a" * 64


if __name__ == "__main__":
    unittest.main()
