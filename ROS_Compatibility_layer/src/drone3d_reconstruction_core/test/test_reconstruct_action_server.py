"""Deterministic tests for Action goal arbitration and callback races."""

from types import SimpleNamespace
import shutil
import sys
import tempfile
import threading
import weakref
import time
import unittest
from pathlib import Path
from unittest import mock

from actionlib_msgs.msg import GoalStatus
from drone3d_reconstruction_msgs.msg import ReconstructSegmentResult

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actionlib_goal_handle import RecordingGoalHandle, assert_no_refused_transitions  # noqa: E402
from drone3d_reconstruction_core import VerifiedCompletion as PublicVerifiedCompletion
from drone3d_reconstruction_core.controller import JobFinished
from drone3d_reconstruction_core.warm_worker_pool import WarmWorkerUnavailableError
from drone3d_reconstruction_core.preparation import (
    ALGORITHM_FAILED,
    INPUT_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_GOAL,
    IO_ERROR,
    CompletionError,
    PreparationError,
)
from drone3d_reconstruction_core.resource_gate import (
    ResourceBusyError,
    ResourceDecision,
    ResourceGateError,
    ResourceLimits,
    ResourceSample,
)
from drone3d_reconstruction_core.result_transaction import (
    ResultCollisionError,
    ResultSafetyError,
    ResultTransaction,
)
from drone3d_reconstruction_core.reconstruct_action_server import (
    PreparedLaunch,
    ReconstructActionServer,
    VerifiedCompletion,
)


class FakeResultTransaction(ResultTransaction):
    """In-memory transaction double that still satisfies the typed boundary."""

    def __init__(
        self,
        *,
        events=None,
        commit_error=None,
        cleanup_error=None,
        on_commit=None,
        on_cleanup=None,
    ):
        self.events = events if events is not None else []
        self.commit_error = commit_error
        self.cleanup_error = cleanup_error
        self.on_commit = on_commit
        self.on_cleanup = on_cleanup
        self.commit_calls = 0
        self.cleanup_calls = 0
        self._committed = False

    @property
    def committed(self):
        return self._committed

    def commit(self):
        self.commit_calls += 1
        self.events.append("commit")
        if self.on_commit is not None:
            self.on_commit()
        if self.commit_error is not None:
            raise self.commit_error
        self._committed = True
        return "/results/final"

    def cleanup(self):
        self.cleanup_calls += 1
        self.events.append("cleanup")
        if self.on_cleanup is not None:
            self.on_cleanup()
        if self.cleanup_error is not None:
            raise self.cleanup_error


def verified_completion(**overrides):
    values = {
        "scene_id": "scene-001",
        "result_manifest_uri": "file:///results/scene-001/manifest.json",
        "result_manifest_sha256": "a" * 64,
        "scene_glb_uri": "file:///results/scene-001/scene.glb",
        "frame_count": 12,
    }
    values.update(overrides)
    return VerifiedCompletion(**values)


def verify_success(
    _finished: JobFinished, _cancellation_event: threading.Event
) -> VerifiedCompletion:
    return verified_completion()


def prepared_launch(
    command=("fake-worker",), verifier=verify_success, transaction=None
):
    if transaction is None:
        transaction = FakeResultTransaction()
    return PreparedLaunch(
        command=command,
        result_transaction=transaction,
        verify_completion=verifier,
    )


def resource_busy_error():
    limits = ResourceLimits(1, 1, 1, 1, 1, 1.0)
    sample = ResourceSample(0, 0, 0, 0, 0, 1, 1, 1.0)
    return ResourceBusyError(
        ResourceDecision(False, "LOW_MEM_AVAILABLE", "low memory", sample, limits)
    )


class FakeActionServer:
    """Stands in for actionlib.ActionServer. Tests deliver goals and cancels through the
    callbacks the server registered here, the way actionlib calls them."""

    by_server = weakref.WeakKeyDictionary()

    def __init__(self, name, action_type, goal_cb, cancel_cb, auto_start):
        self.goal_cb = goal_cb
        self.cancel_cb = cancel_cb
        FakeActionServer.by_server[goal_cb.__self__] = self
        self.started = False
        self.stop_calls = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stop_calls += 1

    @classmethod
    def of(cls, server):
        """The fake actionlib endpoint a server registered its callbacks with."""
        return cls.by_server[server]


class FakeGoalHandle(RecordingGoalHandle):
    """actionlib's real ServerGoalHandle: illegal transitions are refused as on the robot."""

    def __init__(self, identifier, status=GoalStatus.PENDING):
        super().__init__(
            identifier, SimpleNamespace(snapshot_uri="", manifest_sha256=""), status
        )


class OneShotStatusFaultGoalHandle(FakeGoalHandle):
    def __init__(self, identifier, fault):
        super().__init__(identifier)
        self.fault = fault
        self.status_calls = 0

    def get_goal_status(self):
        self.status_calls += 1
        # The first call is initial arbitration; fault after worker acceptance.
        if self.status_calls == 2:
            raise self.fault
        return super().get_goal_status()


class StatusFaultGoalHandle(FakeGoalHandle):
    def __init__(self, identifier, fault_calls, status=GoalStatus.PENDING):
        super().__init__(identifier, status=status)
        self.fault_calls = set(fault_calls)
        self.status_calls = 0

    def get_goal_status(self):
        self.status_calls += 1
        if self.status_calls in self.fault_calls:
            raise RuntimeError("injected status failure")
        return super().get_goal_status()


class FailingGoalHandle(FakeGoalHandle):
    def __init__(self, identifier, failures, status=GoalStatus.PENDING):
        super().__init__(identifier, status=status)
        self.failures = dict(failures)

    def _fail(self, method):
        remaining = self.failures.get(method, 0)
        if remaining:
            self.failures[method] = remaining - 1
            raise RuntimeError(f"injected {method} failure")

    def set_accepted(self, text):
        self._fail("accepted")
        super().set_accepted(text)

    def set_rejected(self, result, text):
        self._fail("rejected")
        super().set_rejected(result, text)

    def set_succeeded(self, result, text):
        self._fail("succeeded")
        super().set_succeeded(result, text)

    def set_aborted(self, result, text):
        self._fail("aborted")
        super().set_aborted(result, text)

    def set_canceled(self, result, text):
        self._fail("canceled")
        super().set_canceled(result, text)


class FakeController:
    def __init__(self, *, on_start=None, on_cancel=None, warm_error=None):
        self.on_start = on_start
        self.on_cancel = on_cancel
        self.warm_error = warm_error
        self.on_finished = None
        self.start_calls = 0
        self.start_commands = []
        self.start_worker_timeouts = []
        self.start_warm_calls = 0
        self.start_warm_commands = []
        self.start_warm_jobs = []
        self.start_warm_worker_timeouts = []
        self.cancel_calls = 0
        self.shutdown_calls = 0
        self.next_job_id = 1
        self.active_job_id = None

    def factory(self, **kwargs):
        self.on_finished = kwargs["on_finished"]
        return self

    def start(self, command, *, on_started=None, worker_timeout_sec=None):
        if self.active_job_id is not None:
            raise RuntimeError("fake controller already has an active job")
        self.start_calls += 1
        self.start_commands.append(command)
        self.start_worker_timeouts.append(worker_timeout_sec)
        job_id = self.next_job_id
        self.next_job_id += 1
        self.active_job_id = job_id
        if on_started is not None:
            on_started(job_id)
        try:
            if self.on_start is not None:
                self.on_start(self)
        except BaseException:
            if self.active_job_id == job_id:
                self.active_job_id = None
            raise
        return job_id

    def start_warm(self, command, job_payload, *, on_started=None, worker_timeout_sec=None):
        self.start_warm_calls += 1
        self.start_warm_commands.append(command)
        self.start_warm_jobs.append(job_payload)
        self.start_warm_worker_timeouts.append(worker_timeout_sec)
        if self.warm_error is not None:
            raise self.warm_error
        if self.active_job_id is not None:
            raise RuntimeError("fake controller already has an active job")
        job_id = self.next_job_id
        self.next_job_id += 1
        self.active_job_id = job_id
        if on_started is not None:
            on_started(job_id)
        try:
            if self.on_start is not None:
                self.on_start(self)
        except BaseException:
            if self.active_job_id == job_id:
                self.active_job_id = None
            raise
        return job_id

    def cancel(self):
        self.cancel_calls += 1
        if self.on_cancel is not None:
            return self.on_cancel(self)
        return False

    def shutdown(self):
        self.shutdown_calls += 1

    def finish(self, cause="worker_exit", returncode=0, elapsed=0.1):
        if self.active_job_id is None:
            raise RuntimeError("fake controller has no active job")
        job_id = self.active_job_id
        self.active_job_id = None
        finished = JobFinished(cause, returncode, elapsed, job_id)
        self.on_finished(finished)
        return finished

    def abandon_active_job_for_fault_injection(self):
        self.active_job_id = None

    def inject_completion(self, job_id, cause="worker_exit", returncode=0, elapsed=0.1):
        self.on_finished(JobFinished(cause, returncode, elapsed, job_id))


def _capture_exception(callback, errors):
    try:
        callback()
    except BaseException as error:
        errors.append(error)


class ReconstructActionServerTest(unittest.TestCase):
    def setUp(self):
        RecordingGoalHandle.created = []

    def tearDown(self):
        assert_no_refused_transitions(self)

    def wait_events(self, goal, count, timeout=1.0):
        deadline = time.monotonic() + timeout
        while len(goal.events) < count and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertGreaterEqual(len(goal.events), count)

    def wait_until_idle(self, server, timeout=1.0):
        deadline = time.monotonic() + timeout
        while server.is_busy() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(server.is_busy())

    def assert_empty_result_metadata(self, result):
        self.assertEqual(result.scene_id, "")
        self.assertEqual(result.result_manifest_uri, "")
        self.assertEqual(result.result_manifest_sha256, "")
        self.assertEqual(result.scene_glb_uri, "")
        self.assertEqual(result.frame_count, 0)

    def server(self, controller, *, launch_dispatch=None, verifier=verify_success):
        if launch_dispatch is None:
            launch_dispatch = lambda target, args: threading.Thread(
                target=target, args=args
            )
        return ReconstructActionServer(
            action_name="test_reconstruct",
            command_builder=lambda goal: prepared_launch(verifier=verifier),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
            launch_dispatch=launch_dispatch,
        )

    def test_shutdown_stops_action_server_once(self):
        server = self.server(FakeController())
        action_server = FakeActionServer.of(server)
        server.shutdown()
        server.shutdown()
        self.assertEqual(action_server.stop_calls, 1)

    def test_verified_completion_strictly_validates_immutable_metadata(self):
        self.assertIs(PublicVerifiedCompletion, VerifiedCompletion)
        valid = verified_completion()
        self.assertEqual(valid.frame_count, 12)
        with self.assertRaises(AttributeError):
            valid.scene_id = "replacement"

        invalid_values = (
            ({"scene_id": "../escape"}, ValueError),
            ({"scene_id": ".hidden"}, ValueError),
            ({"scene_id": "scene/child"}, ValueError),
            ({"scene_id": "scene."}, ValueError),
            ({"scene_id": 3}, TypeError),
            ({"result_manifest_uri": ""}, ValueError),
            ({"result_manifest_uri": "/results/manifest.json"}, ValueError),
            ({"result_manifest_uri": "file:///bad path"}, ValueError),
            ({"result_manifest_uri": "file:///bad%escape"}, ValueError),
            ({"result_manifest_uri": None}, TypeError),
            ({"result_manifest_sha256": "A" * 64}, ValueError),
            ({"result_manifest_sha256": "a" * 63}, ValueError),
            ({"result_manifest_sha256": None}, TypeError),
            ({"scene_glb_uri": "/results/scene.glb"}, ValueError),
            ({"scene_glb_uri": None}, TypeError),
            ({"frame_count": True}, TypeError),
            ({"frame_count": 1.0}, TypeError),
            ({"frame_count": 0}, ValueError),
            ({"frame_count": -1}, ValueError),
            ({"frame_count": 1 << 32}, ValueError),
        )
        for overrides, error_type in invalid_values:
            with self.subTest(overrides=overrides):
                with self.assertRaises(error_type):
                    verified_completion(**overrides)

        self.assertEqual(
            verified_completion(
                scene_glb_uri="", frame_count=(1 << 32) - 1
            ).scene_glb_uri,
            "",
        )

    def test_prepared_launch_rejects_malformed_values(self):
        transaction = FakeResultTransaction()
        invalid = (
            lambda: PreparedLaunch([], transaction, verify_success),
            lambda: PreparedLaunch((), transaction, verify_success),
            lambda: PreparedLaunch(("",), transaction, verify_success),
            lambda: PreparedLaunch(("worker", 3), transaction, verify_success),
            lambda: PreparedLaunch(("worker",), object(), verify_success),
            lambda: PreparedLaunch(("worker",), transaction, None),
            lambda: PreparedLaunch(
                ("worker",),
                transaction,
                verify_success,
                warm_command=("warm",),
            ),
            lambda: PreparedLaunch(
                ("worker",),
                transaction,
                verify_success,
                warm_job={"images_dir": "/tmp/images"},
            ),
            lambda: PreparedLaunch(
                ("worker",),
                transaction,
                verify_success,
                warm_command=("warm",),
                warm_job={"job_id": 99},
            ),
            lambda: PreparedLaunch(
                ("worker",),
                transaction,
                verify_success,
                warm_command=("",),
                warm_job={},
            ),
            lambda: PreparedLaunch(
                ("worker",),
                transaction,
                verify_success,
                warm_command=("warm",),
                warm_job={"nan": float("nan")},
            ),
        )
        for construct in invalid:
            with self.assertRaises((TypeError, ValueError)):
                construct()
        plan = prepared_launch()
        with self.assertRaises(AttributeError):
            plan.command = ("replacement",)

    def test_constructor_shuts_down_controller_when_action_server_setup_fails(self):
        for failure_point in ("factory", "start"):
            controller = FakeController()

            class StartFailingServer(FakeActionServer):
                def start(self):
                    raise RuntimeError("start failed")

            def action_factory(*args, **kwargs):
                if failure_point == "factory":
                    raise RuntimeError("factory failed")
                return StartFailingServer(*args, **kwargs)

            with self.subTest(failure_point=failure_point):
                with self.assertRaises(RuntimeError):
                    ReconstructActionServer(
                        action_name="test_constructor_rollback",
                        command_builder=lambda goal: prepared_launch(),
                        worker_timeout_sec=1.0,
                        term_grace_sec=0.1,
                        controller_factory=controller.factory,
                        action_server_factory=action_factory,
                    )
                self.assertEqual(controller.shutdown_calls, 1)

    def test_bare_argv_or_missing_verifier_never_starts_worker(self):
        malformed = (["fake-worker"], SimpleNamespace(command=("fake-worker",)))
        for index, value in enumerate(malformed):
            controller = FakeController()
            server = ReconstructActionServer(
                action_name=f"test_malformed_{index}",
                command_builder=lambda goal, result=value: result,
                worker_timeout_sec=1.0,
                term_grace_sec=0.1,
                controller_factory=controller.factory,
                action_server_factory=FakeActionServer,
            )
            goal = FakeGoalHandle(f"malformed-{index}")
            FakeActionServer.of(server).goal_cb(goal)
            self.wait_events(goal, 1)
            self.assertEqual(goal.events[0][0], "rejected")
            self.assertEqual(controller.start_calls, 0)
            self.assertFalse(server.is_busy())

    def test_exact_plan_is_retained_and_only_tuple_command_reaches_controller(self):
        controller = FakeController()
        verified = []
        transaction = FakeResultTransaction()

        def verify_exact_finished(finished, cancellation_event):
            verified.append((finished, cancellation_event))
            return verified_completion()

        plan = PreparedLaunch(
            ("fake-worker", "--identity"),
            transaction,
            verify_exact_finished,
        )
        server = ReconstructActionServer(
            action_name="test_plan_identity",
            command_builder=lambda goal: plan,
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("plan-identity")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        self.assertEqual(controller.start_commands, [plan.command])
        finished = controller.finish(elapsed=0.375)
        self.wait_events(goal, 2)
        self.assertEqual(len(verified), 1)
        verified_finished, cancellation_event = verified[0]
        self.assertIs(verified_finished, finished)
        # The goal's own cancellation event, untouched by a goal nobody cancelled.
        self.assertIsInstance(cancellation_event, threading.Event)
        self.assertFalse(cancellation_event.is_set())
        self.assertEqual(verified_finished.job_id, 1)
        self.assertEqual(verified_finished.cause, "worker_exit")
        self.assertEqual(verified_finished.returncode, 0)
        self.assertAlmostEqual(verified_finished.elapsed_sec, 0.375)
        self.assertEqual(goal.events[-1][0], "succeeded")

    def test_warm_job_dispatches_to_start_warm_and_finishes_through_verifier(self):
        controller = FakeController()
        transaction = FakeResultTransaction()
        warm_job = {
            "images_dir": "/tmp/images",
            "output_dir": "/tmp/output",
            "input_images": ["/tmp/images/000000.png"],
        }
        plan = PreparedLaunch(
            ("one-shot-fallback",),
            transaction,
            verify_success,
            warm_command=("warm-worker", "--checkpoint", "checkpoint.pt"),
            warm_job=warm_job,
        )
        server = ReconstructActionServer(
            action_name="test_warm_dispatch",
            command_builder=lambda goal: plan,
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("warm-dispatch")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        self.assertEqual(goal.events[0][0], "accepted")
        self.assertEqual(controller.start_warm_calls, 1)
        self.assertEqual(controller.start_calls, 0)
        self.assertEqual(controller.start_warm_commands, [plan.warm_command])
        self.assertEqual(controller.start_warm_jobs, [warm_job])
        finished = controller.finish(elapsed=0.25)
        self.wait_events(goal, 2)
        self.assertEqual(goal.events[-1][0], "succeeded")
        self.assertFalse(server.is_busy())
        self.assertEqual(finished.job_id, 1)

    def test_warm_spawn_failure_falls_back_to_oneshot_start(self):
        controller = FakeController(
            warm_error=WarmWorkerUnavailableError("spawn_failed")
        )
        transaction = FakeResultTransaction()
        plan = PreparedLaunch(
            ("one-shot-fallback",),
            transaction,
            verify_success,
            warm_command=("warm-worker",),
            warm_job={"images_dir": "/tmp/images"},
        )
        server = ReconstructActionServer(
            action_name="test_warm_fallback",
            command_builder=lambda goal: plan,
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("warm-fallback")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        self.assertEqual(controller.start_warm_calls, 1)
        self.assertEqual(controller.start_calls, 1)
        self.assertEqual(server.diagnostics_snapshot()["warm_fallback_count"], 1)
        self.assertEqual(controller.start_commands, [plan.command])
        controller.finish()
        self.wait_events(goal, 2)
        self.assertEqual(goal.events[-1][0], "succeeded")

    def test_exit_zero_fails_closed_when_bound_verifier_raises(self):
        controller = FakeController()

        def reject_result(_finished, _cancellation_event):
            raise ValueError("secret /result/path must not escape")

        server = ReconstructActionServer(
            action_name="test_validation_failure",
            command_builder=lambda goal: prepared_launch(verifier=reject_result),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("validation-failure")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish()
        self.wait_events(goal, 2)

        self.assertEqual([event[0] for event in goal.events], ["accepted", "aborted"])
        result = goal.events[-1][1]
        self.assertEqual(result.error_code, ReconstructSegmentResult.ALGORITHM_FAILED)
        self.assertEqual(result.message, "result validation failed: ValueError")
        self.assertNotIn("secret", result.message)
        self.assertEqual(server.terminal_attempt_count, 1)

    def test_completion_error_codes_map_without_weakening_validation_failures(self):
        cases = (
            (ALGORITHM_FAILED, ReconstructSegmentResult.ALGORITHM_FAILED),
            (IO_ERROR, ReconstructSegmentResult.IO_ERROR),
            (INTERNAL_ERROR, ReconstructSegmentResult.INTERNAL_ERROR),
        )
        for index, (code, expected_code) in enumerate(cases):
            with self.subTest(code=code):
                controller = FakeController()
                transaction = FakeResultTransaction()

                def reject_result(_finished, _cancellation_event):
                    raise CompletionError(code, f"typed verifier failure {code}")

                server = ReconstructActionServer(
                    action_name=f"test_completion_error_{index}",
                    command_builder=lambda goal: prepared_launch(
                        verifier=reject_result,
                        transaction=transaction,
                    ),
                    worker_timeout_sec=1.0,
                    term_grace_sec=0.1,
                    controller_factory=controller.factory,
                    action_server_factory=FakeActionServer,
                )
                goal = FakeGoalHandle(f"completion-error-{index}")
                FakeActionServer.of(server).goal_cb(goal)
                self.wait_events(goal, 1)
                controller.finish()
                self.wait_events(goal, 2)

                self.assertEqual(
                    [event[0] for event in goal.events], ["accepted", "aborted"]
                )
                result = goal.events[-1][1]
                self.assertEqual(result.error_code, expected_code)
                self.assertEqual(
                    result.message,
                    f"result validation failed: typed verifier failure {code}",
                )
                self.assertEqual(transaction.commit_calls, 0)
                self.assertEqual(transaction.cleanup_calls, 1)
                self.assert_empty_result_metadata(result)

    def test_exit_zero_fails_closed_for_missing_wrong_or_malformed_metadata(self):
        cases = (
            (lambda _finished, _cancellation_event: None, "TypeError"),
            (lambda _finished, _cancellation_event: object(), "TypeError"),
            (lambda _finished, _cancellation_event: verified_completion(frame_count=0), "ValueError"),
        )
        for index, (verifier, expected_error) in enumerate(cases):
            with self.subTest(index=index):
                controller = FakeController()
                transaction = FakeResultTransaction()
                server = ReconstructActionServer(
                    action_name=f"test_invalid_completion_{index}",
                    command_builder=lambda goal, verifier=verifier: prepared_launch(
                        verifier=verifier,
                        transaction=transaction,
                    ),
                    worker_timeout_sec=1.0,
                    term_grace_sec=0.1,
                    controller_factory=controller.factory,
                    action_server_factory=FakeActionServer,
                )
                goal = FakeGoalHandle(f"invalid-completion-{index}")

                FakeActionServer.of(server).goal_cb(goal)
                self.wait_events(goal, 1)
                controller.finish()
                self.wait_events(goal, 2)

                self.assertEqual(
                    [event[0] for event in goal.events], ["accepted", "aborted"]
                )
                result = goal.events[-1][1]
                self.assertEqual(
                    result.message,
                    f"result validation failed: {expected_error}",
                )
                self.assert_empty_result_metadata(result)
                self.assertEqual(transaction.commit_calls, 0)
                self.assertEqual(transaction.cleanup_calls, 1)

    def test_observers_receive_one_finished_and_one_terminal_event(self):
        controller = FakeController()
        finished_events = []
        terminal_events = []
        metadata = verified_completion()
        server = ReconstructActionServer(
            action_name="test_observers",
            command_builder=lambda goal: prepared_launch(
                verifier=lambda _finished, _cancellation_event: metadata
            ),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
            finished_observer=finished_events.append,
            terminal_observer=lambda method, result, text: terminal_events.append(
                (method, result, text)
            ),
        )
        goal = FakeGoalHandle("observer-success")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        finished = controller.finish(elapsed=0.25)
        self.wait_events(goal, 2)

        self.assertEqual(len(finished_events), 1)
        self.assertIs(finished_events[0], finished)
        self.assertEqual(len(terminal_events), 1)
        method, result, text = terminal_events[0]
        self.assertEqual(method, "succeeded")
        self.assertEqual(result.error_code, ReconstructSegmentResult.OK)
        self.assertEqual(text, "worker result committed")
        self.assertEqual(result.scene_id, metadata.scene_id)

    def test_observer_exceptions_never_disturb_terminal_convergence(self):
        controller = FakeController()
        server = ReconstructActionServer(
            action_name="test_observer_faults",
            command_builder=lambda goal: prepared_launch(),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
            finished_observer=lambda _finished: (_ for _ in ()).throw(
                RuntimeError("finished observer fault")
            ),
            terminal_observer=lambda _method, _result, _text: (_ for _ in ()).throw(
                RuntimeError("terminal observer fault")
            ),
        )
        goal = FakeGoalHandle("observer-faults")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish()
        self.wait_events(goal, 2)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.OK)

    def test_finished_observer_rejects_duplicate_and_stale_events(self):
        controller = FakeController()
        finished_events = []
        server = ReconstructActionServer(
            action_name="test_observer_duplicates",
            command_builder=lambda goal: prepared_launch(),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
            finished_observer=finished_events.append,
        )
        goal = FakeGoalHandle("observer-duplicates")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        finished = controller.finish()
        self.wait_events(goal, 2)
        controller.on_finished(finished)
        controller.on_finished(JobFinished("worker_exit", 0, 0.1, 99))
        self.assertEqual(len(finished_events), 1)

    def test_diagnostics_snapshot_exposes_stage_and_busy_transitions(self):
        controller = FakeController()
        server = ReconstructActionServer(
            action_name="test_diagnostics_snapshot",
            command_builder=lambda goal: prepared_launch(),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        idle = server.diagnostics_snapshot()
        self.assertFalse(idle["running"])
        self.assertEqual(idle["stage"], "idle")
        self.assertIsNone(idle["job_id"])

        goal = FakeGoalHandle("diagnostics-active")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        active = server.diagnostics_snapshot()
        self.assertTrue(active["running"])
        self.assertEqual(active["stage"], "active")
        self.assertEqual(active["job_id"], 1)
        self.assertEqual(active["accepted_count"], 1)

        controller.finish(elapsed=0.25)
        self.wait_events(goal, 2)
        terminal = server.diagnostics_snapshot()
        self.assertFalse(terminal["running"])
        self.assertEqual(terminal["stage"], "idle")
        self.assertIsNone(terminal["job_id"])
        self.assertEqual(terminal["terminal_attempt_count"], 1)

    def test_success_result_contains_verified_metadata(self):
        metadata = verified_completion(
            scene_id="warehouse-7",
            result_manifest_uri="s3://reconstruction/warehouse-7/manifest.json",
            result_manifest_sha256="0123456789abcdef" * 4,
            scene_glb_uri="s3://reconstruction/warehouse-7/scene.glb",
            frame_count=847,
        )
        controller = FakeController()
        server = self.server(controller, verifier=lambda _finished, _cancellation_event: metadata)
        goal = FakeGoalHandle("metadata-success")

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish(elapsed=1.75)
        self.wait_events(goal, 2)

        result = goal.events[-1][1]
        self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])
        self.assertEqual(result.error_code, ReconstructSegmentResult.OK)
        self.assertEqual(result.scene_id, metadata.scene_id)
        self.assertEqual(result.result_manifest_uri, metadata.result_manifest_uri)
        self.assertEqual(result.result_manifest_sha256, metadata.result_manifest_sha256)
        self.assertEqual(result.scene_glb_uri, metadata.scene_glb_uri)
        self.assertEqual(result.frame_count, metadata.frame_count)
        self.assertAlmostEqual(result.wall_time.to_sec(), 1.75)

    def test_empty_glb_uri_is_preserved_on_success(self):
        metadata = verified_completion(scene_glb_uri="")
        controller = FakeController()
        server = self.server(controller, verifier=lambda _finished, _cancellation_event: metadata)
        goal = FakeGoalHandle("empty-glb-success")

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish()
        self.wait_events(goal, 2)

        result = goal.events[-1][1]
        self.assertEqual(goal.events[-1][0], "succeeded")
        self.assertEqual(result.scene_id, metadata.scene_id)
        self.assertEqual(result.scene_glb_uri, "")
        self.assertEqual(result.frame_count, metadata.frame_count)

    def test_commit_runs_after_verifier_and_is_the_only_success_linearization(self):
        events = []
        controller = FakeController()

        def assert_commit_holds_server_lock():
            acquired = server._lock.acquire(blocking=False)
            if acquired:
                server._lock.release()
            self.assertFalse(acquired)

        transaction = FakeResultTransaction(
            events=events,
            on_commit=assert_commit_holds_server_lock,
        )
        server = ReconstructActionServer(
            action_name="test_commit_order",
            command_builder=lambda goal: prepared_launch(
                verifier=lambda _finished, _cancellation_event: (
                    events.append("verify"), verified_completion()
                )[1],
                transaction=transaction,
            ),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("commit-order")

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish()
        self.wait_events(goal, 2)
        FakeActionServer.of(server).cancel_cb(goal)
        controller.inject_completion(1)

        self.assertEqual(events, ["verify", "commit"])
        self.assertEqual(transaction.commit_calls, 1)
        self.assertEqual(transaction.cleanup_calls, 0)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.OK)
        self.assertEqual(server.cancel_request_count, 0)
        self.assertEqual(server.terminal_attempt_count, 1)

    def test_real_transaction_post_publication_error_remains_truthful_success(self):
        temporary = tempfile.mkdtemp(prefix="action-real-transaction-")
        try:
            root = Path(temporary) / "results"
            root.mkdir()
            transaction = ResultTransaction(root, "published")
            staging = transaction.staging_path
            (staging / "mesh.txt").write_text("mesh", encoding="utf-8")
            controller = FakeController()
            server = ReconstructActionServer(
                action_name="test_real_post_publish_error",
                command_builder=lambda goal: prepared_launch(transaction=transaction),
                worker_timeout_sec=1.0,
                term_grace_sec=0.1,
                controller_factory=controller.factory,
                action_server_factory=FakeActionServer,
            )
            goal = FakeGoalHandle("real-post-publish")

            from drone3d_reconstruction_core import result_transaction as transaction_module

            original_lstat_at = transaction_module._lstat_at

            def fail_only_after_publication(directory_fd, name):
                info = original_lstat_at(directory_fd, name)
                if name == "published" and info is not None:
                    raise ResultSafetyError(
                        "COMMIT_IDENTITY_CHANGED", "injected post-publication fault"
                    )
                return info

            FakeActionServer.of(server).goal_cb(goal)
            self.wait_events(goal, 1)
            with mock.patch.object(
                transaction_module, "_lstat_at", side_effect=fail_only_after_publication
            ):
                controller.finish()
            self.wait_events(goal, 2)

            self.assertTrue(transaction.committed)
            self.assertIsNone(transaction._root_fd)
            self.assertFalse(staging.exists())
            self.assertEqual(
                (root / "published" / "mesh.txt").read_text(encoding="utf-8"),
                "mesh",
            )
            self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])
            result = goal.events[-1][1]
            expected_metadata = verified_completion()
            self.assertEqual(result.error_code, ReconstructSegmentResult.OK)
            self.assertEqual(result.scene_id, expected_metadata.scene_id)
            self.assertEqual(
                result.result_manifest_uri, expected_metadata.result_manifest_uri
            )
            self.assertEqual(
                result.result_manifest_sha256,
                expected_metadata.result_manifest_sha256,
            )
            self.assertEqual(result.scene_glb_uri, expected_metadata.scene_glb_uri)
            self.assertEqual(result.frame_count, expected_metadata.frame_count)
            self.assertFalse(server.is_busy())
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def test_commit_collision_aborts_io_error_and_cleans_staging(self):
        events = []
        transaction = FakeResultTransaction(
            events=events,
            commit_error=ResultCollisionError("FINAL_EXISTS", "occupied"),
        )
        controller = FakeController()
        server = ReconstructActionServer(
            action_name="test_commit_collision",
            command_builder=lambda goal: prepared_launch(
                verifier=lambda _finished, _cancellation_event: (
                    events.append("verify"), verified_completion()
                )[1],
                transaction=transaction,
            ),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("commit-collision")

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish()
        self.wait_events(goal, 2)

        self.assertEqual(events, ["verify", "commit", "cleanup"])
        self.assertEqual([event[0] for event in goal.events], ["accepted", "aborted"])
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.IO_ERROR)
        self.assertEqual(
            goal.events[-1][1].message,
            "result commit failed: ResultCollisionError",
        )

    def test_opt_in_keeps_failed_staging_without_publishing_success(self):
        from dataclasses import replace
        with tempfile.TemporaryDirectory(prefix="action-preserve-failure-") as tmp:
            root = Path(tmp)
            for cause, code in (("resource_guard", -15), ("worker_exit", 1), ("cancelled", -15)):
                with self.subTest(cause=cause):
                    transaction = ResultTransaction(root, cause)
                    staging = transaction.staging_path
                    (staging / "run.log").write_text("failure evidence\n")
                    controller = FakeController()
                    server = ReconstructActionServer(
                        action_name="test_preserve_" + cause,
                        command_builder=lambda goal: replace(
                            prepared_launch(transaction=transaction),
                            preserve_uncommitted_staging=True,
                        ),
                        worker_timeout_sec=1.0, term_grace_sec=0.1,
                        controller_factory=controller.factory,
                        action_server_factory=FakeActionServer,
                    )
                    goal = FakeGoalHandle("preserve-" + cause)
                    FakeActionServer.of(server).goal_cb(goal)
                    self.wait_events(goal, 1)
                    controller.finish(cause, code)
                    self.wait_events(goal, 2)
                    self.assertEqual((staging / "run.log").read_text(), "failure evidence\n")
                    self.assertFalse(transaction.committed)
                    self.assertFalse(transaction.final_path.exists())
                    self.assertNotEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.OK)
                    self.assertIn(str(staging), goal.events[-1][1].message)
                    self.assertFalse(server.is_busy())
                    transaction.cleanup()

    def test_validation_failure_cleans_before_terminal_callback(self):
        events = []
        controller = FakeController()
        goal = FakeGoalHandle("validation-cleanup-order")
        original_aborted = goal.set_aborted

        def assert_cleanup_runs_outside_server_lock():
            acquired = server._lock.acquire(blocking=False)
            self.assertTrue(acquired)
            if acquired:
                server._lock.release()

        transaction = FakeResultTransaction(
            events=events,
            on_cleanup=assert_cleanup_runs_outside_server_lock,
        )

        def reject_result(_finished, _cancellation_event):
            events.append("verify")
            raise ValueError("invalid staging")

        def record_aborted(result, text):
            events.append("terminal")
            original_aborted(result, text)

        goal.set_aborted = record_aborted
        server = ReconstructActionServer(
            action_name="test_validation_cleanup_order",
            command_builder=lambda current_goal: prepared_launch(
                verifier=reject_result,
                transaction=transaction,
            ),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish()
        self.wait_events(goal, 2)

        self.assertEqual(events, ["verify", "cleanup", "terminal"])
        self.assertEqual(transaction.commit_calls, 0)
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.ALGORITHM_FAILED)

    def test_cleanup_failure_overrides_mutable_outcome_with_io_error(self):
        transaction = FakeResultTransaction(
            cleanup_error=ResultSafetyError("CLEANUP_REFUSED", "unsafe")
        )
        controller = FakeController()
        server = ReconstructActionServer(
            action_name="test_cleanup_failure",
            command_builder=lambda goal: prepared_launch(transaction=transaction),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("cleanup-failure")

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish("worker_exit", 7)
        self.wait_events(goal, 2)

        self.assertEqual(transaction.commit_calls, 0)
        self.assertEqual(transaction.cleanup_calls, 1)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "aborted"])
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.IO_ERROR)
        self.assertEqual(
            goal.events[-1][1].message,
            "result cleanup failed: ResultSafetyError",
        )

    def test_non_success_completion_never_invokes_verifier(self):
        cases = (("worker_exit", 8), ("timeout", -9), ("cancelled", -15), ("shutdown", -15))
        for index, (cause, returncode) in enumerate(cases):
            calls = []
            controller = FakeController()
            server = ReconstructActionServer(
                action_name=f"test_unverified_{index}",
                command_builder=lambda goal, calls=calls: prepared_launch(
                    verifier=lambda _finished, _cancellation_event: (
                        calls.append(1), verified_completion()
                    )[1]
                ),
                worker_timeout_sec=1.0,
                term_grace_sec=0.1,
                controller_factory=controller.factory,
                action_server_factory=FakeActionServer,
            )
            goal = FakeGoalHandle(f"unverified-{index}")
            FakeActionServer.of(server).goal_cb(goal)
            self.wait_events(goal, 1)
            controller.finish(cause, returncode)
            self.wait_events(goal, 2)
            self.assertEqual(calls, [])

    def test_blocked_verifier_runs_outside_lock_and_reserves_slot(self):
        entered = threading.Event()
        release = threading.Event()
        observed_events = []
        controller = FakeController()

        def blocked_verifier(_finished, cancellation_event):
            observed_events.append(cancellation_event)
            entered.set()
            self.assertTrue(release.wait(2.0))
            return verified_completion()

        server = ReconstructActionServer(
            action_name="test_blocked_verifier",
            command_builder=lambda goal: prepared_launch(verifier=blocked_verifier),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("blocked-verifier")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        finish_thread = threading.Thread(target=controller.finish)
        finish_thread.start()
        self.assertTrue(entered.wait(1.0))
        # The verifier holds this goal's cancellation event while the server reports it verifying.
        self.assertEqual(server.diagnostics_snapshot()["stage"], "verifying")
        self.assertEqual(len(observed_events), 1)
        verification_cancel = observed_events[0]
        self.assertFalse(verification_cancel.is_set())

        contender = FakeGoalHandle("verifier-contender")
        FakeActionServer.of(server).goal_cb(contender)
        self.assertEqual(contender.events[0][1].error_code, ReconstructSegmentResult.BUSY)
        FakeActionServer.of(server).cancel_cb(goal)
        self.assertEqual(server.cancel_request_count, 1)
        self.assertTrue(verification_cancel.is_set())
        self.assertEqual(controller.cancel_calls, 0)
        started = time.monotonic()
        controller.inject_completion(1)
        self.assertLess(time.monotonic() - started, 0.2)
        with self.assertRaises(RuntimeError):
            server.shutdown(timeout_sec=0.02)

        release.set()
        finish_thread.join(1.0)
        self.assertFalse(finish_thread.is_alive())
        self.wait_events(goal, 2)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "canceled"])
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.CANCELLED)
        self.assert_empty_result_metadata(goal.events[-1][1])

    def test_cooperative_verifier_cancel_discards_metadata_without_commit(self):
        for outcome in ("return", "raise"):
            with self.subTest(outcome=outcome):
                entered = threading.Event()
                exited = threading.Event()
                observed_events = []
                transaction = FakeResultTransaction()
                controller = FakeController()

                def cooperative_verifier(_finished, cancellation_event):
                    observed_events.append(cancellation_event)
                    entered.set()
                    self.assertTrue(cancellation_event.wait(2.0))
                    try:
                        if outcome == "raise":
                            raise ValueError("verification cancelled")
                        return verified_completion(scene_id="must-not-be-published")
                    finally:
                        exited.set()

                server = ReconstructActionServer(
                    action_name=f"test_cooperative_verifier_cancel_{outcome}",
                    command_builder=lambda goal: prepared_launch(
                        verifier=cooperative_verifier,
                        transaction=transaction,
                    ),
                    worker_timeout_sec=1.0,
                    term_grace_sec=0.1,
                    controller_factory=controller.factory,
                    action_server_factory=FakeActionServer,
                )
                goal = FakeGoalHandle(f"cooperative-verifier-cancel-{outcome}")
                FakeActionServer.of(server).goal_cb(goal)
                self.wait_events(goal, 1)

                finish_thread = threading.Thread(target=controller.finish)
                finish_thread.start()
                self.assertTrue(entered.wait(1.0))
                self.assertEqual(server.diagnostics_snapshot()["stage"], "verifying")
                self.assertEqual(len(observed_events), 1)
                verification_cancel = observed_events[0]
                self.assertFalse(verification_cancel.is_set())

                FakeActionServer.of(server).cancel_cb(goal)
                finish_thread.join(1.0)

                self.assertFalse(finish_thread.is_alive())
                self.assertTrue(exited.is_set())
                self.assertTrue(verification_cancel.is_set())
                self.wait_events(goal, 2)
                self.assertEqual(
                    [event[0] for event in goal.events], ["accepted", "canceled"]
                )
                result = goal.events[-1][1]
                self.assertEqual(result.error_code, ReconstructSegmentResult.CANCELLED)
                self.assert_empty_result_metadata(result)
                self.assertEqual(transaction.commit_calls, 0)
                self.assertEqual(transaction.cleanup_calls, 1)
                self.assertEqual(controller.cancel_calls, 0)
                self.assertEqual(server.cancel_request_count, 1)
                self.assertEqual(server.terminal_attempt_count, 1)
                self.assertFalse(server.is_busy())

    def test_verifier_process_control_exception_aborts_then_propagates(self):
        controller = FakeController()

        def interrupt(_finished, _cancellation_event):
            raise KeyboardInterrupt("stop")

        server = ReconstructActionServer(
            action_name="test_verifier_interrupt",
            command_builder=lambda goal: prepared_launch(verifier=interrupt),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("verifier-interrupt")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        with self.assertRaises(KeyboardInterrupt):
            controller.finish()
        self.wait_events(goal, 2)
        self.assertEqual(goal.events[-1][0], "aborted")
        self.assertEqual(
            goal.events[-1][1].message,
            "result validation failed: KeyboardInterrupt",
        )
        self.assertFalse(server.is_busy())

    def test_dispatch_rejects_non_thread_result(self):
        controller = FakeController()
        def invalid_dispatch(*_):
            return None

        server = self.server(
            controller,
            launch_dispatch=invalid_dispatch,
        )
        goal = FakeGoalHandle("invalid-dispatch")

        FakeActionServer.of(server).goal_cb(goal)

        self.assertEqual([event[0] for event in goal.events], ["rejected"])
        self.assertEqual(controller.start_calls, 0)
        self.assertFalse(server.is_busy())

    def test_dispatch_rejects_already_started_thread(self):
        controller = FakeController()
        release = threading.Event()
        started = threading.Thread(target=release.wait)
        started.start()

        def invalid_dispatch(*_):
            return started

        server = self.server(controller, launch_dispatch=invalid_dispatch)
        goal = FakeGoalHandle("started-dispatch")
        FakeActionServer.of(server).goal_cb(goal)
        release.set()
        started.join(1.0)

        self.assertEqual([event[0] for event in goal.events], ["rejected"])
        self.assertEqual(controller.start_calls, 0)
        self.assertFalse(server.is_busy())

    def test_launch_dispatch_failure_rejects_and_releases_capacity(self):
        controller = FakeController()

        dispatch_fails = [True]

        def dispatch(target, args):
            if dispatch_fails[0]:
                raise RuntimeError("thread creation failed")
            return threading.Thread(target=target, args=args)

        server = self.server(controller, launch_dispatch=dispatch)
        failed = FakeGoalHandle("dispatch-failure")
        FakeActionServer.of(server).goal_cb(failed)

        self.assertEqual([event[0] for event in failed.events], ["rejected"])
        self.assertEqual(
            failed.events[0][1].error_code,
            ReconstructSegmentResult.INTERNAL_ERROR,
        )
        self.assertIn("dispatch failed", failed.events[0][1].message)
        self.assertEqual(controller.start_calls, 0)
        self.assertFalse(server.is_busy())

        dispatch_fails[0] = False  # Thread creation works again for the next goal.
        replacement = FakeGoalHandle("replacement")
        FakeActionServer.of(server).goal_cb(replacement)
        self.wait_events(replacement, 1)
        controller.finish()
        self.wait_events(replacement, 2)
        self.assertEqual(
            [event[0] for event in replacement.events],
            ["accepted", "succeeded"],
        )

    def test_cancelled_dispatch_failure_returns_cancelled_result(self):
        controller = FakeController()
        goal = FakeGoalHandle("cancelled-dispatch")

        def cancel_then_fail(*_):
            FakeActionServer.of(server).cancel_cb(goal)
            raise RuntimeError("thread creation failed")

        server = self.server(controller, launch_dispatch=cancel_then_fail)
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)

        self.assertEqual([event[0] for event in goal.events], ["canceled"])
        self.assertEqual(
            goal.events[0][1].error_code,
            ReconstructSegmentResult.CANCELLED,
        )
        self.assertEqual(controller.start_calls, 0)
        self.assertFalse(server.is_busy())

    def test_shutdown_cancels_blocked_command_build_without_worker_start(self):
        controller = FakeController()
        entered = threading.Event()
        release = threading.Event()
        server = ReconstructActionServer(
            action_name="test_reconstruct",
            command_builder=lambda goal: (
                entered.set(), release.wait(2.0), prepared_launch()
            )[-1],
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("shutdown-build")
        FakeActionServer.of(server).goal_cb(goal)
        self.assertTrue(entered.wait(1.0))

        shutdown_error = []

        def run_shutdown():
            try:
                server.shutdown(timeout_sec=1.0)
            except BaseException as error:
                shutdown_error.append(error)

        shutdown_thread = threading.Thread(target=run_shutdown)
        shutdown_thread.start()
        time.sleep(0.05)
        self.assertTrue(shutdown_thread.is_alive())
        release.set()
        shutdown_thread.join(1.0)
        self.assertFalse(shutdown_thread.is_alive())
        self.assertEqual(shutdown_error, [])
        self.assertTrue(goal._goal_status.status in (GoalStatus.RECALLED, GoalStatus.PREEMPTED))
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.CANCELLED)
        self.assertEqual(controller.start_calls, 0)
        self.assertFalse(server.is_busy())

    def test_resource_denial_rejects_busy_without_start_and_reuses_slot(self):
        controller = FakeController()
        calls = []

        def builder(goal):
            calls.append(goal)
            if len(calls) == 1:
                raise resource_busy_error()
            return prepared_launch()

        server = ReconstructActionServer(
            action_name="test_resource_busy",
            command_builder=builder,
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        denied = FakeGoalHandle("resource-denied")
        FakeActionServer.of(server).goal_cb(denied)
        self.wait_events(denied, 1)

        self.assertEqual([event[0] for event in denied.events], ["rejected"])
        self.assertEqual(denied.events[0][1].error_code, ReconstructSegmentResult.BUSY)
        self.assertIn("LOW_MEM_AVAILABLE", denied.events[0][1].message)
        self.assertEqual(controller.start_calls, 0)
        self.assertFalse(server.is_busy())

        replacement = FakeGoalHandle("resource-replacement")
        FakeActionServer.of(server).goal_cb(replacement)
        self.wait_events(replacement, 1)
        self.assertEqual([event[0] for event in replacement.events], ["accepted"])
        self.assertEqual(controller.start_calls, 1)
        controller.finish()
        self.wait_events(replacement, 2)
        self.assertEqual(replacement.events[-1][0], "succeeded")

    def test_blocked_resource_preparation_reserves_slot_and_cancel_wins(self):
        controller = FakeController()
        entered = threading.Event()
        release = threading.Event()

        def builder(goal):
            entered.set()
            if not release.wait(2.0):
                raise RuntimeError("test did not release resource gate")
            raise resource_busy_error()

        server = ReconstructActionServer(
            action_name="test_resource_race",
            command_builder=builder,
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        first = FakeGoalHandle("blocked-resource")
        second = FakeGoalHandle("blocked-contender")
        FakeActionServer.of(server).goal_cb(first)
        self.assertTrue(entered.wait(1.0))

        FakeActionServer.of(server).goal_cb(second)
        self.assertEqual(second.events[0][0], "rejected")
        self.assertEqual(second.events[0][1].error_code, ReconstructSegmentResult.BUSY)
        FakeActionServer.of(server).cancel_cb(first)
        release.set()
        self.wait_events(first, 1)

        self.assertEqual([event[0] for event in first.events], ["canceled"])
        self.assertEqual(first.events[0][1].error_code, ReconstructSegmentResult.CANCELLED)
        self.assertEqual(controller.start_calls, 0)
        self.assertFalse(server.is_busy())

    def test_resource_gate_and_unrelated_errors_remain_internal_error(self):
        for index, error in enumerate((ResourceGateError("bad probe"), ValueError("bad command"))):
            controller = FakeController()

            def builder(goal, failure=error):
                raise failure

            server = ReconstructActionServer(
                action_name=f"test_resource_error_{index}",
                command_builder=builder,
                worker_timeout_sec=1.0,
                term_grace_sec=0.1,
                controller_factory=controller.factory,
                action_server_factory=FakeActionServer,
            )
            goal = FakeGoalHandle(f"resource-error-{index}")
            FakeActionServer.of(server).goal_cb(goal)
            self.wait_events(goal, 1)
            self.assertEqual(goal.events[0][0], "rejected")
            self.assertEqual(
                goal.events[0][1].error_code,
                ReconstructSegmentResult.INTERNAL_ERROR,
            )
            self.assertEqual(controller.start_calls, 0)
            self.assertFalse(server.is_busy())

    def test_preparation_error_codes_map_without_weakening_resource_busy(self):
        cases = (
            (INVALID_GOAL, ReconstructSegmentResult.INVALID_GOAL),
            (INPUT_NOT_FOUND, ReconstructSegmentResult.INPUT_NOT_FOUND),
            (IO_ERROR, ReconstructSegmentResult.IO_ERROR),
            (INTERNAL_ERROR, ReconstructSegmentResult.INTERNAL_ERROR),
        )
        for index, (code, expected_code) in enumerate(cases):
            with self.subTest(code=code):
                controller = FakeController()

                def builder(goal, failure=PreparationError(code, f"typed preparation failure {code}")):
                    raise failure

                server = ReconstructActionServer(
                    action_name=f"test_preparation_error_{index}",
                    command_builder=builder,
                    worker_timeout_sec=1.0,
                    term_grace_sec=0.1,
                    controller_factory=controller.factory,
                    action_server_factory=FakeActionServer,
                )
                goal = FakeGoalHandle(f"preparation-error-{index}")
                FakeActionServer.of(server).goal_cb(goal)
                self.wait_events(goal, 1)
                self.assertEqual(goal.events[0][0], "rejected")
                result = goal.events[0][1]
                self.assertEqual(result.error_code, expected_code)
                self.assertEqual(
                    result.message,
                    f"goal preparation failed: typed preparation failure {code}",
                )
                self.assertEqual(controller.start_calls, 0)
                self.assertFalse(server.is_busy())
                self.assert_empty_result_metadata(result)

    def test_preparation_error_loses_to_earlier_cancellation(self):
        controller = FakeController()
        entered = threading.Event()
        release = threading.Event()

        def builder(_goal):
            # Fail only after the test has delivered the cancel: the cancel is earlier.
            entered.set()
            self.assertTrue(release.wait(2.0))
            raise PreparationError(IO_ERROR, "cannot prepare snapshot")

        server = ReconstructActionServer(
            action_name="test_preparation_cancel",
            command_builder=builder,
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("preparation-cancel")
        FakeActionServer.of(server).goal_cb(goal)
        self.assertTrue(entered.wait(1.0))
        FakeActionServer.of(server).cancel_cb(goal)
        release.set()
        self.wait_events(goal, 1)

        self.assertEqual([event[0] for event in goal.events], ["canceled"])
        self.assertEqual(
            goal.events[0][1].error_code, ReconstructSegmentResult.CANCELLED
        )
        self.assertEqual(controller.start_calls, 0)
        self.assertFalse(server.is_busy())

    def test_immediate_completion_is_buffered_until_after_acceptance(self):
        controller = FakeController(on_start=lambda current: current.finish())
        verified = []
        goal = FakeGoalHandle("immediate")

        def verify_completion(_finished, _cancellation_event):
            self.assertEqual([event[0] for event in goal.events], ["accepted"])
            verified.append(1)
            return verified_completion()

        transaction = FakeResultTransaction()
        server = ReconstructActionServer(
            action_name="test_immediate_duplicate",
            command_builder=lambda current_goal: prepared_launch(
                verifier=verify_completion,
                transaction=transaction,
            ),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        FakeActionServer.of(server).goal_cb(goal)
        controller.inject_completion(1)  # A duplicate controller callback must be harmless.
        self.wait_events(goal, 2)

        self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])
        self.assertEqual(verified, [1])
        self.assertEqual(transaction.commit_calls, 1)
        self.assertEqual(transaction.cleanup_calls, 0)
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.OK)
        self.assertEqual(server.accepted_count, 1)
        self.assertEqual(server.terminal_attempt_count, 1)
        self.assertFalse(server.is_busy())

    def test_first_buffered_completion_cannot_be_overwritten_by_duplicate(self):
        verified = []

        def emit_failure_then_duplicate_success(controller):
            controller.finish("worker_exit", 9)
            controller.inject_completion(1, "worker_exit", 0)

        controller = FakeController(on_start=emit_failure_then_duplicate_success)
        server = self.server(
            controller,
            verifier=lambda _finished, _cancellation_event: (
                verified.append(1), verified_completion()
            )[1],
        )
        goal = FakeGoalHandle("buffered-first-wins")

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 2)

        self.assertEqual([event[0] for event in goal.events], ["accepted", "aborted"])
        self.assertEqual(
            goal.events[-1][1].error_code,
            ReconstructSegmentResult.ALGORITHM_FAILED,
        )
        self.assertEqual(verified, [])
        self.assertEqual(server.terminal_attempt_count, 1)
        self.assertFalse(server.is_busy())

    def test_nonowned_busy_and_shutdown_rejections_use_one_safe_fallback(self):
        for mode in ("busy-exception", "busy-noop", "shutdown-exception", "shutdown-noop"):
            controller = FakeController()
            server = self.server(controller)
            active = None
            if mode.startswith("busy"):
                active = FakeGoalHandle("active-{}".format(mode))
                FakeActionServer.of(server).goal_cb(active)
                self.wait_events(active, 1)
            else:
                server.shutdown()

            contender = FakeGoalHandle("contender-{}".format(mode))
            original_rejected = contender.set_rejected
            calls = []

            def fail_or_noop_once(result, text):
                calls.append(1)
                if len(calls) == 1:
                    if mode.endswith("exception"):
                        raise RuntimeError("injected rejection failure")
                    return
                original_rejected(result, text)

            contender.set_rejected = fail_or_noop_once
            FakeActionServer.of(server).goal_cb(contender)

            self.assertEqual(len(calls), 2)
            self.assertEqual(contender._goal_status.status, GoalStatus.REJECTED)
            self.assertEqual([event[0] for event in contender.events], ["rejected"])
            expected_code = (
                ReconstructSegmentResult.BUSY
                if mode.startswith("busy")
                else ReconstructSegmentResult.INTERNAL_ERROR
            )
            self.assertEqual(contender.events[0][1].error_code, expected_code)
            if active is not None:
                self.assertTrue(server.is_busy())
                self.assertEqual([event[0] for event in active.events], ["accepted"])
                controller.finish()
                self.wait_events(active, 2)

    def test_unexpected_process_control_fault_after_worker_start_converges_then_raises(self):
        def finish_cancel(current):
            current.finish("cancelled", -15)
            return True

        controller = FakeController(on_cancel=finish_cancel)
        transaction = FakeResultTransaction()
        launches = []

        def dispatch(target, args):
            launches.append((target, args))
            return threading.Thread(target=lambda: None)

        server = ReconstructActionServer(
            action_name="test_launch_status_fault",
            command_builder=lambda goal: prepared_launch(transaction=transaction),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
            launch_dispatch=dispatch,
        )
        # The goal arrives through actionlib (first status query); the launch task the
        # server hands to its dispatcher then runs on this thread so the fault surfaces here.
        goal = OneShotStatusFaultGoalHandle("status-fault", KeyboardInterrupt("stop"))
        FakeActionServer.of(server).goal_cb(goal)
        self.assertEqual(len(launches), 1)
        self.assertTrue(server.is_busy())
        target, args = launches[0]
        with self.assertRaises(KeyboardInterrupt):
            target(*args)

        self.assertEqual(controller.start_calls, 1)
        self.assertEqual(controller.cancel_calls, 1)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "aborted"])
        self.assertEqual(transaction.cleanup_calls, 1)
        self.assertEqual(server.terminal_attempt_count, 1)
        self.assertFalse(server.is_busy())

    def test_acceptance_remains_authoritative_when_status_queries_fail(self):
        controller = FakeController()
        server = self.server(controller)
        goal = StatusFaultGoalHandle("accepted-status-faults", {2, 3})

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)

        self.assertEqual([event[0] for event in goal.events], ["accepted"])
        self.assertEqual(server.accepted_count, 1)
        self.assertTrue(server.is_busy())
        self.assertEqual(controller.cancel_calls, 1)

        controller.finish("cancelled", -15, elapsed=0.4)
        self.wait_events(goal, 2)

        self.assertEqual([event[0] for event in goal.events], ["accepted", "aborted"])
        self.assertEqual(
            goal.events[-1][1].error_code,
            ReconstructSegmentResult.INTERNAL_ERROR,
        )
        self.assertAlmostEqual(goal.events[-1][1].wall_time.to_sec(), 0.4)
        self.assertFalse(server.is_busy())

    def test_second_goal_is_rejected_while_first_is_starting(self):
        entered = threading.Event()
        release = threading.Event()

        def block_start(controller):
            entered.set()
            if not release.wait(2.0):
                raise RuntimeError("test did not release worker start")

        controller = FakeController(on_start=block_start)
        server = self.server(controller)
        first = FakeGoalHandle("first")
        second = FakeGoalHandle("second")
        thread = threading.Thread(target=FakeActionServer.of(server).goal_cb, args=(first,))
        thread.start()
        self.assertTrue(entered.wait(1.0))

        FakeActionServer.of(server).goal_cb(second)
        release.set()
        thread.join(2.0)
        self.wait_events(first, 1)
        self.assertFalse(thread.is_alive())

        self.assertEqual(second.events[0][0], "rejected")
        self.assertEqual(second.events[0][1].error_code, ReconstructSegmentResult.BUSY)
        self.assertNotIn("accepted", [event[0] for event in second.events])
        self.assertEqual(server.busy_rejection_count, 1)
        self.assertEqual([event[0] for event in first.events], ["accepted"])

        controller.finish()
        self.assertEqual(first.events[-1][0], "succeeded")

    def test_cancel_during_start_never_accepts_goal(self):
        entered = threading.Event()
        release = threading.Event()

        def block_start(controller):
            entered.set()
            if not release.wait(2.0):
                raise RuntimeError("test did not release worker start")

        def finish_cancel(controller):
            controller.finish("cancelled", -15)
            return True

        controller = FakeController(on_start=block_start, on_cancel=finish_cancel)
        server = self.server(controller)
        goal = FakeGoalHandle("cancel-start")
        thread = threading.Thread(target=FakeActionServer.of(server).goal_cb, args=(goal,))
        thread.start()
        self.assertTrue(entered.wait(1.0))

        FakeActionServer.of(server).cancel_cb(goal)
        release.set()
        thread.join(2.0)
        self.wait_events(goal, 1)
        self.assertFalse(thread.is_alive())

        self.assertEqual([event[0] for event in goal.events], ["canceled"])
        self.assertEqual(goal.events[0][1].error_code, ReconstructSegmentResult.CANCELLED)
        self.assertEqual(controller.cancel_calls, 1)
        self.assertEqual(server.accepted_count, 0)
        self.assertEqual(server.terminal_attempt_count, 1)

    def test_cancel_during_acceptance_overrides_buffered_success(self):
        controller = FakeController(on_cancel=lambda current: False)
        server = self.server(controller)
        goal = FakeGoalHandle("cancel-accepting")
        accept_entered = threading.Event()
        accept_release = threading.Event()
        original_accept = goal.set_accepted

        def blocked_accept(text):
            accept_entered.set()
            if not accept_release.wait(1.0):
                raise RuntimeError("test did not release acceptance")
            original_accept(text)

        goal.set_accepted = blocked_accept
        FakeActionServer.of(server).goal_cb(goal)
        self.assertTrue(accept_entered.wait(1.0))
        controller.finish("worker_exit", 0)
        FakeActionServer.of(server).cancel_cb(goal)
        accept_release.set()
        self.wait_events(goal, 2)

        self.assertEqual([event[0] for event in goal.events], ["accepted", "canceled"])
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.CANCELLED)
        self.assertFalse(server.is_busy())

    def test_cancel_before_delayed_success_callback_wins(self):
        controller = FakeController(on_cancel=lambda current: False)
        verifier_calls = []
        server = self.server(
            controller,
            verifier=lambda _finished, _cancellation_event: (
                verifier_calls.append(1), verified_completion()
            )[1],
        )
        goal = FakeGoalHandle("cancel-before-callback")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)

        FakeActionServer.of(server).cancel_cb(goal)
        controller.finish("worker_exit", 0)
        self.wait_events(goal, 2)

        self.assertEqual([event[0] for event in goal.events], ["accepted", "canceled"])
        self.assertEqual(
            goal.events[-1][1].error_code,
            ReconstructSegmentResult.CANCELLED,
        )
        self.assertEqual(verifier_calls, [])
        self.assertFalse(server.is_busy())

    def test_unbound_stale_completion_cannot_claim_replacement(self):
        old_id = 41
        replacement_id = 42
        builder_entered = threading.Event()
        builder_release = threading.Event()
        controller = FakeController(on_cancel=lambda current: False)
        controller.next_job_id = replacement_id
        server = ReconstructActionServer(
            action_name="test_reconstruct_identity",
            command_builder=lambda goal: (
                builder_entered.set(), builder_release.wait(2.0), prepared_launch()
            )[-1],
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        replacement = FakeGoalHandle("replacement-unbound")

        FakeActionServer.of(server).goal_cb(replacement)
        self.assertTrue(builder_entered.wait(1.0))
        controller.inject_completion(old_id)
        controller.inject_completion(999)
        time.sleep(0.02)

        # Neither stale completion binds to the replacement, which is still preparing.
        diagnostics = server.diagnostics_snapshot()
        self.assertTrue(diagnostics["running"])
        self.assertEqual(diagnostics["stage"], "starting")
        self.assertIsNone(diagnostics["job_id"])
        self.assertEqual(replacement.events, [])

        builder_release.set()
        self.wait_events(replacement, 1)
        controller.inject_completion(old_id)
        time.sleep(0.02)
        self.assertEqual([event[0] for event in replacement.events], ["accepted"])

        controller.finish()
        self.wait_events(replacement, 2)
        self.assertEqual(replacement.events[-1][0], "succeeded")
        self.assertEqual(controller.active_job_id, None)

    def test_mismatched_buffer_is_revalidated_before_consumption(self):
        record = SimpleNamespace(
            buffered_finished=JobFinished("worker_exit", 0, 0.1, 5),
            controller_job_id=6,
        )

        from drone3d_reconstruction_core.reconstruct_action_server import (
            _matching_buffered_finished,
        )

        self.assertIsNone(_matching_buffered_finished(record))

    def test_delayed_old_completion_cannot_terminate_replacement(self):
        controller = FakeController(on_cancel=lambda current: False)
        server = self.server(controller)
        old = FakeGoalHandle("old-delayed")
        FakeActionServer.of(server).goal_cb(old)
        self.wait_events(old, 1)
        old_job_id = 1

        FakeActionServer.of(server).cancel_cb(old)
        controller.inject_completion(old_job_id, "cancelled", -15)
        controller.abandon_active_job_for_fault_injection()
        self.wait_events(old, 2)

        replacement = FakeGoalHandle("replacement-delayed")
        FakeActionServer.of(server).goal_cb(replacement)
        self.wait_events(replacement, 1)

        controller.inject_completion(old_job_id, "worker_exit", 0)
        time.sleep(0.02)
        self.assertEqual([event[0] for event in replacement.events], ["accepted"])

        controller.finish("worker_exit", 0)
        self.wait_events(replacement, 2)
        self.assertEqual(replacement.events[-1][0], "succeeded")

    def test_stale_cancel_cannot_cancel_replacement_goal(self):
        controller = FakeController()
        server = self.server(controller)
        first = FakeGoalHandle("old")
        second = FakeGoalHandle("new")

        FakeActionServer.of(server).goal_cb(first)
        self.wait_events(first, 1)
        controller.finish()
        self.wait_events(first, 2)
        FakeActionServer.of(server).goal_cb(second)
        self.wait_events(second, 1)
        FakeActionServer.of(server).cancel_cb(first)

        self.assertEqual(controller.cancel_calls, 0)
        self.assertEqual([event[0] for event in second.events], ["accepted"])
        controller.finish()
        self.assertEqual(second.events[-1][0], "succeeded")

    def test_launch_failure_is_rejected_before_acceptance(self):
        start_failed = threading.Event()

        def fail_start(controller):
            start_failed.set()
            raise OSError("exec failed")

        controller = FakeController(on_start=fail_start)
        server = self.server(controller)
        goal = FakeGoalHandle("spawn-failure")

        FakeActionServer.of(server).goal_cb(goal)
        self.assertTrue(start_failed.wait(1.0))
        deadline = time.monotonic() + 1.0
        while controller.cancel_calls < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(controller.cancel_calls, 1)
        self.assertEqual(goal.events, [])
        self.assertTrue(server.is_busy())
        controller.inject_completion(1, "cancelled", -15, elapsed=0.4)
        self.wait_events(goal, 1)

        self.assertEqual([event[0] for event in goal.events], ["rejected"])
        self.assertEqual(
            goal.events[0][1].error_code,
            ReconstructSegmentResult.INTERNAL_ERROR,
        )
        self.assertAlmostEqual(goal.events[0][1].wall_time.to_sec(), 0.4)
        self.assertEqual(server.accepted_count, 0)
        self.assertFalse(server.is_busy())

        replacement = FakeGoalHandle("replacement-after-spawn-failure")
        controller.on_start = None
        FakeActionServer.of(server).goal_cb(replacement)
        self.wait_events(replacement, 1)
        self.assertEqual([event[0] for event in replacement.events], ["accepted"])
        controller.finish()
        self.wait_events(replacement, 2)
        self.assertEqual(replacement.events[-1][0], "succeeded")

    def test_accept_exception_stops_worker_and_rejects_goal(self):
        def finish_cancel(controller):
            controller.finish("cancelled", -15)
            return True

        controller = FakeController(on_cancel=finish_cancel)
        verifier_calls = []
        server = self.server(
            controller,
            verifier=lambda _finished, _cancellation_event: (
                verifier_calls.append(1), verified_completion()
            )[1],
        )
        goal = FailingGoalHandle("accept-failure", {"accepted": 1})

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)

        self.assertEqual([event[0] for event in goal.events], ["rejected"])
        self.assertEqual(
            goal.events[0][1].error_code,
            ReconstructSegmentResult.INTERNAL_ERROR,
        )
        self.assertEqual(controller.start_calls, 1)
        self.assertEqual(controller.cancel_calls, 1)
        self.assertEqual(verifier_calls, [])
        self.assertFalse(server.is_busy())

    def test_committed_success_terminal_exception_retries_truthful_success(self):
        controller = FakeController()
        transaction = FakeResultTransaction()
        server = ReconstructActionServer(
            action_name="test_committed_terminal_retry",
            command_builder=lambda goal: prepared_launch(transaction=transaction),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FailingGoalHandle("terminal-fallback", {"succeeded": 1})

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish()
        self.wait_events(goal, 2)

        self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.OK)
        self.assertEqual(transaction.commit_calls, 1)
        self.assertEqual(transaction.cleanup_calls, 0)
        self.assertFalse(server.is_busy())
        self.assertEqual(server.terminal_attempt_count, 1)

    def test_failed_committed_success_retry_keeps_capacity_reserved(self):
        controller = FakeController()
        server = self.server(controller)
        goal = FailingGoalHandle(
            "terminal-wedge",
            {"succeeded": 2},
        )

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish()
        time.sleep(0.02)
        contender = FakeGoalHandle("contender")
        FakeActionServer.of(server).goal_cb(contender)

        self.assertEqual([event[0] for event in goal.events], ["accepted"])
        self.assertTrue(server.is_busy())
        self.assertEqual(contender.events[0][0], "rejected")
        self.assertEqual(
            contender.events[0][1].error_code,
            ReconstructSegmentResult.BUSY,
        )

    def test_failure_timeout_and_shutdown_have_one_coherent_terminal(self):
        cases = (
            (JobFinished("worker_exit", 23, 0.2, 1), "aborted", ReconstructSegmentResult.ALGORITHM_FAILED),
            (JobFinished("timeout", -9, 0.3, 1), "aborted", ReconstructSegmentResult.ALGORITHM_FAILED),
            (JobFinished("shutdown", -15, 0.1, 1), "aborted", ReconstructSegmentResult.INTERNAL_ERROR),
        )
        for index, (finished, method, error_code) in enumerate(cases):
            controller = FakeController()
            server = self.server(controller)
            goal = FakeGoalHandle(f"mapping-{index}")
            FakeActionServer.of(server).goal_cb(goal)
            self.wait_events(goal, 1)
            controller.on_finished(finished)
            controller.on_finished(finished)
            self.wait_events(goal, 2)
            self.assertEqual([event[0] for event in goal.events], ["accepted", method])
            self.assertEqual(goal.events[-1][1].error_code, error_code)
            self.assertEqual(server.terminal_attempt_count, 1)

    def test_cancel_does_not_hide_process_group_cleanup_failure(self):
        controller = FakeController()
        server = self.server(controller)
        goal = FakeGoalHandle("cancel-cleanup-failed")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)

        FakeActionServer.of(server).cancel_cb(goal)
        controller.finish("cleanup_failed", -9, elapsed=0.2)
        self.wait_events(goal, 2)

        self.assertEqual([event[0] for event in goal.events], ["accepted", "aborted"])
        result = goal.events[-1][1]
        self.assertEqual(result.error_code, ReconstructSegmentResult.INTERNAL_ERROR)
        self.assertIn("cleanup_failed", result.message)

    def test_worker_exit_annotates_real_cuda_oom_from_run_log(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            (staging / "run.log").write_text(
                "torch.cuda.OutOfMemoryError: CUDA out of memory. "
                "Tried to allocate 20.00 MiB\n",
                encoding="utf-8",
            )
            launch = SimpleNamespace(
                result_transaction=SimpleNamespace(staging_path=staging)
            )
            finished = JobFinished("worker_exit", 137, 0.2, 1)
            _, result, message = ReconstructActionServer._finished_terminal(
                finished, launch
            )
            self.assertEqual(result.error_code, ReconstructSegmentResult.ALGORITHM_FAILED)
            self.assertIn("CUDA out of memory detected", message)
            self.assertIn("worker exited with code 137", message)
        # 普通非 OOM 失败保持原文，不得误触发 FATAL_OOM。
        _, _, message = ReconstructActionServer._finished_terminal(
            JobFinished("worker_exit", 2, 0.2, 1), None
        )
        self.assertEqual(message, "worker exited with code 2")

    def test_late_cancel_cannot_revoke_claimed_buffered_completion(self):
        controller = FakeController(on_cancel=lambda current: False)
        server = self.server(controller)
        goal = FakeGoalHandle("buffered-completion-claimed")
        accept_entered = threading.Event()
        accept_release = threading.Event()
        terminal_entered = threading.Event()
        terminal_release = threading.Event()
        original_accept = goal.set_accepted
        original_succeeded = goal.set_succeeded

        def blocked_accept(text):
            accept_entered.set()
            self.assertTrue(accept_release.wait(1.0))
            original_accept(text)

        def blocked_succeeded(result, text):
            terminal_entered.set()
            self.assertTrue(terminal_release.wait(1.0))
            original_succeeded(result, text)

        goal.set_accepted = blocked_accept
        goal.set_succeeded = blocked_succeeded
        FakeActionServer.of(server).goal_cb(goal)
        self.assertTrue(accept_entered.wait(1.0))
        controller.finish("worker_exit", 0)
        accept_release.set()
        self.assertTrue(terminal_entered.wait(1.0))
        FakeActionServer.of(server).cancel_cb(goal)
        terminal_release.set()
        self.wait_events(goal, 2)

        self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])
        self.assertEqual(server.cancel_request_count, 0)
        self.assertEqual(server.terminal_attempt_count, 1)

    def test_post_start_failure_consumes_already_buffered_completion(self):
        def finish_then_fail(controller):
            controller.finish("worker_exit", 9, elapsed=0.35)
            raise RuntimeError("monitor dispatch failed")

        transaction = FakeResultTransaction()
        controller = FakeController(on_start=finish_then_fail)
        server = ReconstructActionServer(
            action_name="test_buffered_launch_failure",
            command_builder=lambda goal: prepared_launch(transaction=transaction),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("buffered-launch-failure")

        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)

        self.assertEqual([event[0] for event in goal.events], ["rejected"])
        self.assertEqual(
            goal.events[0][1].error_code,
            ReconstructSegmentResult.INTERNAL_ERROR,
        )
        self.assertAlmostEqual(goal.events[0][1].wall_time.to_sec(), 0.35)
        self.assertEqual(transaction.cleanup_calls, 1)
        self.assertFalse(server.is_busy())

    def test_asynchronous_cancel_waits_for_matching_completion_before_cleanup(self):
        start_entered = threading.Event()
        start_release = threading.Event()
        transaction = FakeResultTransaction()

        def blocked_start(controller):
            start_entered.set()
            self.assertTrue(start_release.wait(2.0))

        controller = FakeController(
            on_start=blocked_start,
            on_cancel=lambda current: True,
        )
        server = ReconstructActionServer(
            action_name="test_async_cancel",
            command_builder=lambda goal: prepared_launch(transaction=transaction),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("async-cancel")
        FakeActionServer.of(server).goal_cb(goal)
        self.assertTrue(start_entered.wait(1.0))

        FakeActionServer.of(server).cancel_cb(goal)
        start_release.set()
        deadline = time.monotonic() + 1.0
        while controller.cancel_calls < 1 and time.monotonic() < deadline:
            time.sleep(0.005)

        self.assertEqual(controller.cancel_calls, 1)
        self.assertEqual(goal.events, [])
        self.assertEqual(transaction.cleanup_calls, 0)
        self.assertTrue(server.is_busy())

        controller.finish("cancelled", -15, elapsed=0.6)
        self.wait_events(goal, 1)

        self.assertEqual([event[0] for event in goal.events], ["canceled"])
        self.assertEqual(goal.events[0][1].error_code, ReconstructSegmentResult.CANCELLED)
        self.assertAlmostEqual(goal.events[0][1].wall_time.to_sec(), 0.6)
        self.assertEqual(transaction.cleanup_calls, 1)
        self.assertFalse(server.is_busy())

    def test_action_status_queries_never_hold_adapter_lock(self):
        controller = FakeController()
        server = self.server(controller)
        goal = FakeGoalHandle("status-lock-order")
        original_status = goal.get_goal_status
        lock_observations = []

        def checked_status():
            acquired = server._lock.acquire(blocking=False)
            lock_observations.append(acquired)
            if acquired:
                server._lock.release()
            return original_status()

        goal.get_goal_status = checked_status
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish()
        self.wait_events(goal, 2)

        self.assertGreaterEqual(len(lock_observations), 3)
        self.assertTrue(all(lock_observations))

    def test_terminal_status_faults_after_primary_and_fallback_retain_slot(self):
        # An accepted goal whose worker fails: the primary set_aborted raises, the status
        # query after it and the one after the fallback both fail (status calls 3 and 4;
        # 1 is arbitration, 2 the post-accept check). Capacity must stay owned.
        controller = FakeController()
        server = self.server(controller)
        goal = StatusFaultGoalHandle("status-query-faults", {3, 4})
        original_aborted = goal.set_aborted
        abort_calls = []

        def fail_once(result, text):
            abort_calls.append(1)
            if len(abort_calls) == 1:
                raise RuntimeError("primary failed")
            original_aborted(result, text)

        goal.set_aborted = fail_once
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish("worker_exit", 2, elapsed=0.7)

        self.assertEqual(len(abort_calls), 2)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "aborted"])
        self.assertAlmostEqual(goal.events[-1][1].wall_time.to_sec(), 0.7)
        self.assertEqual(server.terminal_attempt_count, 1)
        self.assertTrue(server.is_busy())

    def test_cleanup_refusal_uses_legal_method_and_preserves_elapsed(self):
        # Accepted: the worker fails, the staging cleanup is refused -> aborted, IO_ERROR,
        # with the worker's elapsed time kept.
        controller = FakeController()
        transaction = FakeResultTransaction(
            cleanup_error=ResultSafetyError("CLEANUP_REFUSED", "unsafe")
        )
        server = ReconstructActionServer(
            action_name="test_cleanup_legality_accepted",
            command_builder=lambda goal: prepared_launch(transaction=transaction),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("cleanup-legality-accepted")
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish("worker_exit", 2, elapsed=1.25)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "aborted"])
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.IO_ERROR)
        self.assertAlmostEqual(goal.events[-1][1].wall_time.to_sec(), 1.25)
        self.assertEqual(transaction.cleanup_calls, 1)
        self.assertFalse(server.is_busy())

        # Never accepted: cancelled while the launch is being prepared, so the prepared
        # staging must be cleaned and is refused. The server picks rejected for PENDING
        # and canceled for RECALLING (actionlib would also accept rejected for RECALLING).
        for index, (status, expected_method) in enumerate((
            (GoalStatus.PENDING, "rejected"),
            (GoalStatus.RECALLING, "canceled"),
        )):
            with self.subTest(status=status):
                controller = FakeController()
                transaction = FakeResultTransaction(
                    cleanup_error=ResultSafetyError("CLEANUP_REFUSED", "unsafe")
                )
                entered = threading.Event()
                release = threading.Event()

                def builder(_goal, transaction=transaction, entered=entered, release=release):
                    entered.set()
                    self.assertTrue(release.wait(2.0))
                    return prepared_launch(transaction=transaction)

                server = ReconstructActionServer(
                    action_name=f"test_cleanup_legality_{index}",
                    command_builder=builder,
                    worker_timeout_sec=1.0,
                    term_grace_sec=0.1,
                    controller_factory=controller.factory,
                    action_server_factory=FakeActionServer,
                )
                goal = FakeGoalHandle(f"cleanup-legality-{index}", status=status)
                FakeActionServer.of(server).goal_cb(goal)
                self.assertTrue(entered.wait(1.0))
                FakeActionServer.of(server).cancel_cb(goal)
                release.set()
                self.wait_events(goal, 1)
                self.assertEqual([event[0] for event in goal.events], [expected_method])
                self.assertEqual(goal.events[0][1].error_code, ReconstructSegmentResult.IO_ERROR)
                self.assertEqual(transaction.cleanup_calls, 1)
                self.assertEqual(controller.start_calls, 0)
                self.wait_until_idle(server)

    def test_recalling_nonowned_contender_fallback_returns_cancelled(self):
        controller = FakeController()
        server = self.server(controller)
        active = FakeGoalHandle("owned-active")
        FakeActionServer.of(server).goal_cb(active)
        self.wait_events(active, 1)
        contender = FailingGoalHandle(
            "recalling-contender",
            {"rejected": 1},
            status=GoalStatus.RECALLING,
        )

        FakeActionServer.of(server).goal_cb(contender)

        self.assertEqual([event[0] for event in contender.events], ["canceled"])
        self.assertEqual(
            contender.events[0][1].error_code,
            ReconstructSegmentResult.CANCELLED,
        )
        controller.finish()
        self.wait_events(active, 2)

    def test_shutdown_waits_for_builder_and_cleanup_refusal_fails_closed(self):
        controller = FakeController()
        entered = threading.Event()
        release = threading.Event()
        transaction = FakeResultTransaction(
            cleanup_error=ResultSafetyError("CLEANUP_REFUSED", "unsafe")
        )

        def builder(goal):
            entered.set()
            self.assertTrue(release.wait(2.0))
            return prepared_launch(transaction=transaction)

        server = ReconstructActionServer(
            action_name="test_shutdown_refused_cleanup",
            command_builder=builder,
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("shutdown-refused-cleanup")
        FakeActionServer.of(server).goal_cb(goal)
        self.assertTrue(entered.wait(1.0))
        shutdown_errors = []
        shutdown_thread = threading.Thread(
            target=lambda: _capture_exception(
                lambda: server.shutdown(timeout_sec=1.0), shutdown_errors
            )
        )
        shutdown_thread.start()
        time.sleep(0.05)
        self.assertEqual(goal.events, [])
        self.assertTrue(server.is_busy())

        release.set()
        shutdown_thread.join(1.0)

        self.assertEqual(shutdown_errors, [])
        self.assertEqual([event[0] for event in goal.events], ["rejected"])
        self.assertEqual(goal.events[0][1].error_code, ReconstructSegmentResult.IO_ERROR)
        self.assertEqual(transaction.cleanup_calls, 1)
        self.assertEqual(controller.start_calls, 0)
        self.assertFalse(server.is_busy())

    def test_shutdown_claims_starting_cancel_before_builder_failure(self):
        controller = FakeController()
        entered = threading.Event()
        release = threading.Event()

        def failing_builder(goal):
            entered.set()
            self.assertTrue(release.wait(2.0))
            raise OSError("builder failed during shutdown")

        server = ReconstructActionServer(
            action_name="test_shutdown_builder_failure",
            command_builder=failing_builder,
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
        )
        goal = FakeGoalHandle("shutdown-builder-failure")
        FakeActionServer.of(server).goal_cb(goal)
        self.assertTrue(entered.wait(1.0))
        shutdown_errors = []

        def run_shutdown():
            try:
                server.shutdown(timeout_sec=1.0)
            except BaseException as error:
                shutdown_errors.append(error)

        shutdown_thread = threading.Thread(target=run_shutdown)
        shutdown_thread.start()
        time.sleep(0.05)
        self.assertEqual(goal.events, [])
        self.assertTrue(shutdown_thread.is_alive())
        release.set()
        shutdown_thread.join(1.0)

        self.assertFalse(shutdown_thread.is_alive())
        self.assertEqual(shutdown_errors, [])
        self.assertEqual([event[0] for event in goal.events], ["canceled"])
        self.assertEqual(goal.events[0][1].error_code, ReconstructSegmentResult.CANCELLED)
        self.assertEqual(server.terminal_attempt_count, 1)
        self.assertEqual(controller.start_calls, 0)
        self.assertFalse(server.is_busy())

    def test_per_goal_overrides_forwarded_to_builder_and_controller(self):
        controller = FakeController()
        builder_calls = []

        def command_builder(goal, **kwargs):
            builder_calls.append(kwargs)
            return prepared_launch(verifier=verify_success)

        server = ReconstructActionServer(
            action_name="test_reconstruct_overrides",
            command_builder=command_builder,
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
            launch_dispatch=lambda target, args: threading.Thread(
                target=target, args=args
            ),
        )
        goal = FakeGoalHandle("overrides")
        goal._goal = SimpleNamespace(
            snapshot_uri="file:///snapshots/scene",
            manifest_sha256="a" * 64,
            image_resolution=128,
            set_image_resolution=True,
            worker_timeout_sec=9.5,
            set_worker_timeout=True,
        )
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1, timeout=3.0)
        self.assertEqual(len(builder_calls), 1)
        self.assertEqual(builder_calls[0]["image_resolution"], 128)
        self.assertEqual(builder_calls[0]["worker_timeout_sec"], 9.5)
        self.assertEqual(controller.start_worker_timeouts, [9.5])
        controller.finish()
        self.wait_events(goal, 2, timeout=3.0)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])

    def test_per_goal_override_flags_must_be_bool(self):
        controller = FakeController()
        server = ReconstructActionServer(
            action_name="test_reconstruct_bad_flags",
            command_builder=lambda goal: prepared_launch(verifier=verify_success),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=FakeActionServer,
            launch_dispatch=lambda target, args: threading.Thread(
                target=target, args=args
            ),
        )
        goal = FakeGoalHandle("bad-flags")
        goal._goal = SimpleNamespace(
            snapshot_uri="file:///snapshots/scene",
            manifest_sha256="a" * 64,
            image_resolution=128,
            set_image_resolution="yes",
            worker_timeout_sec=9.5,
            set_worker_timeout=True,
        )
        FakeActionServer.of(server).goal_cb(goal)
        self.wait_events(goal, 1, timeout=3.0)
        self.assertEqual(goal.events[0][0], "rejected")
        self.assertEqual(
            goal.events[0][1].error_code, ReconstructSegmentResult.INVALID_GOAL
        )


if __name__ == "__main__":
    unittest.main()
