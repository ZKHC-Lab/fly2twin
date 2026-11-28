"""Pins for "every goal reaches exactly one terminal state" in both Action servers.

The capture and the reconstruction server each implement terminal convergence on their own.
These tests drive both only through what actionlib and their injected collaborators see:
goal and cancel callbacks registered with the Action server, the goal handle's set_* calls,
the reconstruction worker controller and the capture sensor subscriptions. Races are staged
with Events; nothing relies on a sleep winning or losing.

Where the two servers answer the same situation differently, both answers are pinned as
they are today and the test says so; a shared implementation must keep or consciously
change each of them.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import threading
import unittest

from actionlib_msgs.msg import GoalStatus
from drone3d_reconstruction_msgs.msg import CaptureSegmentResult, ReconstructSegmentResult

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Modules, not classes: importing the TestCase classes would run their tests twice.
import test_capture_action_server as capture_tests  # noqa: E402
import test_reconstruct_action_server as reconstruct_tests  # noqa: E402
from actionlib_goal_handle import RecordingGoalHandle, assert_no_refused_transitions  # noqa: E402

from drone3d_reconstruction_core.reconstruct_action_server import (  # noqa: E402
    ReconstructActionServer,
)

FakeGoal = reconstruct_tests.FakeGoalHandle
Actionlib = reconstruct_tests.FakeActionServer
CaptureGoal = capture_tests.FakeGoalHandle
CaptureActionlib = capture_tests.FakeActionServer


def call_bounded(test, function, timeout=2.0):
    """Run a server entry point on another thread; it must return, not wait for a lock."""
    done = threading.Event()
    thread = threading.Thread(target=lambda: (function(), done.set()))
    thread.start()
    thread.join(timeout)
    test.assertTrue(done.is_set(), "blocked on a lock held by another callback")


def deliver_as_actionlib(actionlib_lock, callback, handle):
    """actionlib calls goal and cancel callbacks while holding its own (reentrant) lock."""
    with actionlib_lock:
        callback(handle)


def goal_id_blocking_inside_actionlib_lock(handle, actionlib_lock):
    """Make handle.get_goal_id take the actionlib lock, as ServerGoalHandle does, and stop
    inside it until released. Returns (inside, release) events."""
    inside, release = threading.Event(), threading.Event()
    goal_id = handle._goal_id

    def get_goal_id():
        with actionlib_lock:
            inside.set()
            assert release.wait(5.0), "test never released get_goal_id"
            return goal_id

    handle.get_goal_id = get_goal_id
    return inside, release


class BlockingTerminalGoal(FakeGoal):
    """set_<method> enters, then waits for the test before it takes effect."""

    def __init__(self, identifier, method):
        super().__init__(identifier)
        self.entered = threading.Event()
        self.release = threading.Event()
        original = getattr(self, "set_" + method)

        def blocked(result, text):
            self.entered.set()
            assert self.release.wait(5.0), "test never released the terminal call"
            original(result, text)

        setattr(self, "set_" + method, blocked)


class ReconstructTerminalPins(unittest.TestCase):
    wait_events = reconstruct_tests.ReconstructActionServerTest.wait_events

    def setUp(self):
        RecordingGoalHandle.created = []

    def tearDown(self):
        assert_no_refused_transitions(self)

    def make(self, controller, **kwargs):
        server = ReconstructActionServer(
            action_name="test_terminal_pins",
            command_builder=kwargs.pop(
                "command_builder", lambda goal: reconstruct_tests.prepared_launch()
            ),
            worker_timeout_sec=1.0,
            term_grace_sec=0.1,
            controller_factory=controller.factory,
            action_server_factory=Actionlib,
            **kwargs,
        )
        return server, Actionlib.of(server)

    @staticmethod
    def finish_on_cancel(current):
        current.finish("cancelled", -15, elapsed=0.4)
        return True

    def test_cancel_terminates_once_and_only_after_the_worker_proves_it_stopped(self):
        controller = reconstruct_tests.FakeController()
        server, actionlib = self.make(controller)
        goal = FakeGoal("pin-cancel")
        actionlib.goal_cb(goal)
        self.wait_events(goal, 1)
        actionlib.cancel_cb(goal)
        # Cancellation is forwarded, but the slot stays owned until the worker's completion.
        self.assertEqual(controller.cancel_calls, 1)
        self.assertEqual([event[0] for event in goal.events], ["accepted"])
        self.assertTrue(server.is_busy())
        controller.finish("cancelled", -15, elapsed=0.4)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "canceled"])
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.CANCELLED)
        self.assertAlmostEqual(goal.events[-1][1].wall_time.to_sec(), 0.4)
        self.assertFalse(server.is_busy())
        # A repeated cancel and a duplicate completion after the terminal change nothing.
        actionlib.cancel_cb(goal)
        controller.inject_completion(1, cause="cancelled", returncode=-15)
        self.assertEqual(len(goal.events), 2)
        self.assertEqual(server.cancel_request_count, 1)
        self.assertEqual(server.terminal_attempt_count, 1)

    def test_worker_timeout_aborts_once_with_its_elapsed_time(self):
        controller = reconstruct_tests.FakeController()
        server, actionlib = self.make(controller)
        goal = FakeGoal("pin-timeout")
        actionlib.goal_cb(goal)
        self.wait_events(goal, 1)
        controller.finish("timeout", -9, elapsed=1.0)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "aborted"])
        result = goal.events[-1][1]
        self.assertEqual(result.error_code, ReconstructSegmentResult.ALGORITHM_FAILED)
        self.assertEqual(result.message, "worker timed out")
        self.assertAlmostEqual(result.wall_time.to_sec(), 1.0)
        self.assertEqual(server.terminal_attempt_count, 1)
        self.assertFalse(server.is_busy())

    def test_contending_goal_is_rejected_busy_and_its_cancel_never_reaches_the_owner(self):
        controller = reconstruct_tests.FakeController()
        server, actionlib = self.make(controller)
        owner = FakeGoal("pin-owner")
        contender = FakeGoal("pin-contender")
        actionlib.goal_cb(owner)
        self.wait_events(owner, 1)
        actionlib.goal_cb(contender)
        self.assertEqual([event[0] for event in contender.events], ["rejected"])
        self.assertEqual(contender.events[0][1].error_code, ReconstructSegmentResult.BUSY)
        actionlib.cancel_cb(contender)
        self.assertEqual(server.cancel_request_count, 0)
        self.assertEqual(controller.cancel_calls, 0)
        controller.finish()
        self.wait_events(owner, 2)
        self.assertEqual([event[0] for event in owner.events], ["accepted", "succeeded"])
        # Rejecting a goal that never owned the slot is not a terminal attempt of the owner.
        self.assertEqual(server.terminal_attempt_count, 1)
        self.assertEqual(server.busy_rejection_count, 1)

    def test_cancel_and_duplicate_completion_during_the_terminal_call_are_ignored(self):
        controller = reconstruct_tests.FakeController()
        server, actionlib = self.make(controller)
        goal = BlockingTerminalGoal("pin-duplicate", "succeeded")
        actionlib.goal_cb(goal)
        self.wait_events(goal, 1)
        finisher = threading.Thread(target=controller.finish)
        finisher.start()
        self.assertTrue(goal.entered.wait(2.0))
        # The success is claimed and its set_succeeded is in progress.
        call_bounded(self, lambda: actionlib.cancel_cb(goal))
        call_bounded(self, lambda: controller.inject_completion(1))
        self.assertTrue(server.is_busy())
        goal.release.set()
        finisher.join(2.0)
        self.assertFalse(finisher.is_alive())
        self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])
        self.assertEqual(server.cancel_request_count, 0)
        self.assertEqual(controller.cancel_calls, 0)
        self.assertEqual(server.terminal_attempt_count, 1)
        self.assertFalse(server.is_busy())

    def test_cancel_delivered_while_the_server_is_inside_set_accepted(self):
        controller = reconstruct_tests.FakeController(on_cancel=self.finish_on_cancel)
        server, actionlib = self.make(controller)
        goal = FakeGoal("pin-accept-interleave")
        accept = goal.set_accepted

        def accept_with_cancel(text):
            actionlib.cancel_cb(goal)  # actionlib's cancel arrives mid-call
            accept(text)

        goal.set_accepted = accept_with_cancel
        actionlib.goal_cb(goal)
        self.wait_events(goal, 2)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "canceled"])
        self.assertEqual(goal.events[-1][1].error_code, ReconstructSegmentResult.CANCELLED)
        self.assertEqual(controller.cancel_calls, 1)
        self.assertEqual(server.terminal_attempt_count, 1)
        self.assertFalse(server.is_busy())

    def test_cancel_before_the_goal_callback_never_starts_a_worker(self):
        controller = reconstruct_tests.FakeController()
        server, actionlib = self.make(controller)
        goal = FakeGoal("pin-recalling", status=GoalStatus.RECALLING)
        actionlib.goal_cb(goal)
        self.wait_events(goal, 1)
        self.assertEqual([event[0] for event in goal.events], ["canceled"])
        self.assertEqual(goal.events[0][1].error_code, ReconstructSegmentResult.CANCELLED)
        self.assertEqual(controller.start_calls, 0)
        self.assertEqual(server.terminal_attempt_count, 1)
        self.assertFalse(server.is_busy())

    def test_cancel_reads_the_goal_id_outside_the_server_lock(self):
        # While actionlib's cancel callback sits inside get_goal_id (holding the actionlib
        # lock), everything that takes the server lock must still go through.
        controller = reconstruct_tests.FakeController(on_cancel=self.finish_on_cancel)
        server, actionlib = self.make(controller)
        actionlib_lock = threading.RLock()
        goal = FakeGoal("pin-lock-order")
        deliver_as_actionlib(actionlib_lock, actionlib.goal_cb, goal)
        self.wait_events(goal, 1)
        cancel = FakeGoal("pin-lock-order")  # actionlib re-wraps the same goal id
        inside, release = goal_id_blocking_inside_actionlib_lock(cancel, actionlib_lock)
        canceller = threading.Thread(
            target=deliver_as_actionlib, args=(actionlib_lock, actionlib.cancel_cb, cancel))
        canceller.start()
        self.assertTrue(inside.wait(2.0))
        call_bounded(self, server.is_busy)
        call_bounded(self, server.diagnostics_snapshot)
        release.set()
        canceller.join(2.0)
        self.assertFalse(canceller.is_alive())
        self.assertEqual([event[0] for event in goal.events], ["accepted", "canceled"])
        self.assertEqual(server.cancel_request_count, 1)

    def test_accept_failure_honours_an_earlier_cancel_unless_actionlib_accepted(self):
        # Replaces the white-box test of _claim_accept_failure_terminal_locked. Both cases
        # deliver the cancel from inside set_accepted, then the acceptance fails.
        for accepted, expected in ((False, ["canceled"]), (True, ["accepted", "aborted"])):
            with self.subTest(accepted=accepted):
                controller = reconstruct_tests.FakeController(on_cancel=self.finish_on_cancel)
                server, actionlib = self.make(controller)
                goal = FakeGoal(f"pin-accept-failure-{accepted}")

                real_accept = goal.set_accepted

                def failing_accept(text, goal=goal, accepted=accepted, actionlib=actionlib,
                                   real_accept=real_accept):
                    actionlib.cancel_cb(goal)
                    if accepted:
                        # actionlib moved the goal to ACTIVE, then the call failed
                        # (for example while publishing the status).
                        real_accept(text)
                    raise RuntimeError("set_accepted failed")

                goal.set_accepted = failing_accept
                actionlib.goal_cb(goal)
                self.wait_events(goal, len(expected))
                self.assertEqual([event[0] for event in goal.events], expected)
                self.assertEqual(
                    goal.events[-1][1].error_code,
                    ReconstructSegmentResult.INTERNAL_ERROR if accepted
                    else ReconstructSegmentResult.CANCELLED,
                )
                self.assertEqual(controller.cancel_calls, 1)
                self.assertEqual(server.terminal_attempt_count, 1)
                self.assertFalse(server.is_busy())


class CaptureTerminalPins(unittest.TestCase):
    setUp = capture_tests.CaptureActionServerTest.setUp
    tearDown = capture_tests.CaptureActionServerTest.tearDown
    make_server = capture_tests.CaptureActionServerTest.make_server
    subscriber_factory = capture_tests.CaptureActionServerTest.subscriber_factory
    image_message = capture_tests.CaptureActionServerTest.image_message
    wait_events = capture_tests.CaptureActionServerTest.wait_events
    sensor = capture_tests.CaptureActionServerTest.sensor
    wait_session_visible = capture_tests.CaptureActionServerTest.wait_session_visible

    def image(self, sec, nsec=0):
        self.sensor(self.server, "/camera4/image_raw")(self.image_message(sec, nsec))

    @property
    def actionlib(self):
        return CaptureActionlib.of(self.server)

    def start(self, goal):
        self.actionlib.goal_cb(goal)
        self.wait_events(goal, 1)
        self.wait_session_visible(self.server)

    def test_cancel_terminates_once_and_a_repeated_cancel_is_ignored(self):
        goal = CaptureGoal("pin-capture-cancel")
        self.start(goal)
        self.image(10)
        self.actionlib.cancel_cb(goal)
        self.wait_events(goal, 2)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "canceled"])
        self.assertEqual(goal.events[-1][1].error_code, CaptureSegmentResult.CANCELLED)
        self.assertFalse(self.server.is_busy())
        self.actionlib.cancel_cb(goal)
        self.assertEqual(len(goal.events), 2)
        self.assertEqual(self.server.cancel_request_count, 1)
        self.assertEqual(self.server.terminal_attempt_count, 1)
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_duration_deadline_seals_what_was_captured_once(self):
        # Capture has no timeout failure: at max_duration the window is sealed with the
        # frames it has (here 1 of 2), unlike the reconstruction worker timeout (aborted).
        goal = CaptureGoal("pin-capture-deadline", goal=SimpleNamespace(
            scene_id="scene-capture", sample_fps=2.0, max_frames=2,
            max_duration=SimpleNamespace(to_sec=lambda: 0.6)))
        self.start(goal)
        self.image(10)
        self.wait_events(goal, 2, timeout=5.0)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])
        self.assertEqual(goal.events[-1][1].frame_count, 1)
        self.assertEqual(self.server.terminal_attempt_count, 1)
        self.assertFalse(self.server.is_busy())

    def test_contending_goal_is_rejected_and_its_cancel_never_reaches_the_owner(self):
        owner = CaptureGoal("pin-capture-owner")
        contender = CaptureGoal("pin-capture-contender")
        self.start(owner)
        self.actionlib.goal_cb(contender)
        self.assertEqual([event[0] for event in contender.events], ["rejected"])
        self.assertEqual(contender.events[0][1].error_code, CaptureSegmentResult.CAMERA_UNAVAILABLE)
        self.actionlib.cancel_cb(contender)
        self.assertEqual(self.server.cancel_request_count, 0)
        self.image(10)
        self.image(10, 500000000)
        self.wait_events(owner, 2)
        self.assertEqual([event[0] for event in owner.events], ["accepted", "succeeded"])
        self.assertEqual(self.server.terminal_attempt_count, 1)
        self.assertEqual(self.server.busy_rejection_count, 1)

    def test_a_contender_whose_first_rejection_fails_still_gets_a_result(self):
        # The contender never owns the slot, so nothing else would ever answer it: the
        # rejection itself must be confirmed and retried. Here set_rejected fails once
        # before actionlib moved the goal (the next call works).
        owner = CaptureGoal("pin-capture-reject-owner")
        self.start(owner)
        contender = CaptureGoal("pin-capture-reject-contender")
        real_reject = contender.set_rejected
        calls = []

        def flaky_reject(result=None, text=""):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("set_rejected failed before the transition")
            real_reject(result, text)

        contender.set_rejected = flaky_reject
        self.actionlib.goal_cb(contender)
        self.assertEqual(len(calls), 2)
        self.assertEqual([event[0] for event in contender.events], ["rejected"])
        self.assertEqual(contender.events[0][1].error_code, CaptureSegmentResult.CAMERA_UNAVAILABLE)
        self.assertEqual(contender.get_goal_status().status, GoalStatus.REJECTED)
        # The owner keeps its slot and finishes normally.
        self.assertEqual([event[0] for event in owner.events], ["accepted"])
        self.assertTrue(self.server.is_busy())
        self.actionlib.cancel_cb(owner)
        self.wait_events(owner, 2)
        self.assertEqual(self.server.terminal_attempt_count, 1)

    def test_cancel_during_the_success_call_does_not_add_a_terminal(self):
        goal = CaptureGoal("pin-capture-duplicate")
        entered, release = threading.Event(), threading.Event()
        succeeded = goal.set_succeeded

        def blocked(result, text):
            entered.set()
            assert release.wait(5.0)
            succeeded(result, text)

        goal.set_succeeded = blocked
        self.start(goal)
        self.image(10)
        self.image(10, 500000000)
        self.assertTrue(entered.wait(3.0))
        call_bounded(self, lambda: self.actionlib.cancel_cb(goal))
        release.set()
        self.wait_events(goal, 2)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "succeeded"])
        self.assertEqual(self.server.terminal_attempt_count, 1)
        self.assertFalse(self.server.is_busy())
        # Differs from the reconstruction server, which ignores a cancel once the terminal
        # is claimed: the capture server still counts it (the goal is still its active one).
        self.assertEqual(self.server.cancel_request_count, 1)

    def test_cancel_delivered_while_the_server_is_inside_set_accepted(self):
        goal = CaptureGoal("pin-capture-accept-interleave")
        accept = goal.set_accepted

        def accept_with_cancel(text):
            self.actionlib.cancel_cb(goal)
            accept(text)

        goal.set_accepted = accept_with_cancel
        self.actionlib.goal_cb(goal)
        self.wait_events(goal, 2)
        self.assertEqual([event[0] for event in goal.events], ["accepted", "canceled"])
        self.assertEqual(goal.events[-1][1].error_code, CaptureSegmentResult.CANCELLED)
        self.assertEqual(self.server.terminal_attempt_count, 1)
        self.assertFalse(self.server.is_busy())
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_cancel_before_the_goal_callback_never_accepts(self):
        goal = CaptureGoal("pin-capture-recalling", status=GoalStatus.RECALLING)
        self.actionlib.goal_cb(goal)
        self.wait_events(goal, 1)
        self.assertEqual([event[0] for event in goal.events], ["canceled"])
        self.assertEqual(goal.events[0][1].error_code, CaptureSegmentResult.CANCELLED)
        self.assertEqual(self.server.accepted_count, 0)
        self.assertEqual(self.server.terminal_attempt_count, 1)
        self.assertFalse(self.server.is_busy())
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_cancel_reads_the_goal_id_outside_the_server_lock(self):
        # While actionlib's cancel callback sits inside get_goal_id (holding the actionlib
        # lock), sensor callbacks and status queries, which take the server lock, must run.
        actionlib_lock = threading.RLock()
        owner = CaptureGoal("pin-capture-lock-order")
        deliver_as_actionlib(actionlib_lock, self.actionlib.goal_cb, owner)
        self.wait_events(owner, 1)
        self.wait_session_visible(self.server)
        cancel = CaptureGoal("pin-capture-lock-order")  # actionlib re-wraps the same goal id
        inside, release = goal_id_blocking_inside_actionlib_lock(cancel, actionlib_lock)
        canceller = threading.Thread(
            target=deliver_as_actionlib, args=(actionlib_lock, self.actionlib.cancel_cb, cancel))
        canceller.start()
        self.assertTrue(inside.wait(2.0))
        call_bounded(self, lambda: self.image(10))
        call_bounded(self, self.server.diagnostics_snapshot)
        release.set()
        canceller.join(2.0)
        self.assertFalse(canceller.is_alive())
        self.wait_events(owner, 2)
        self.assertEqual([event[0] for event in owner.events], ["accepted", "canceled"])
        self.assertEqual(self.server.cancel_request_count, 1)

    def assert_released_once(self, goal, expected, code):
        self.wait_events(goal, len(expected))
        self.assertEqual([event[0] for event in goal.events], expected)
        self.assertEqual(goal.events[-1][1].error_code, code)
        self.assertTrue(goal.get_goal_status().status in (
            GoalStatus.REJECTED, GoalStatus.RECALLED, GoalStatus.ABORTED))
        self.assertEqual(self.server.terminal_attempt_count, 1)
        self.assertFalse(self.server.is_busy())
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_accept_failure_picks_a_terminal_the_goal_can_take(self):
        # From the reviewer's real-handle probe: set_accepted can fail before or after
        # actionlib moved the goal to ACTIVE. Only an accepted goal may be aborted; a
        # PENDING one is cancelled if a cancel came first, otherwise rejected. Before the
        # fix the server aborted twice, actionlib refused both, and the slot stayed taken.
        cases = (
            ("before-transition", False, False, ["rejected"], CaptureSegmentResult.INTERNAL_ERROR),
            ("cancel-then-fail", False, True, ["canceled"], CaptureSegmentResult.CANCELLED),
            ("after-transition", True, False, ["accepted", "aborted"],
             CaptureSegmentResult.INTERNAL_ERROR),
        )
        for name, transitions, cancel_first, expected, code in cases:
            with self.subTest(case=name):
                self.server.shutdown()
                self.server = self.make_server()
                goal = CaptureGoal(f"pin-capture-accept-failure-{name}")
                real_accept = goal.set_accepted

                def failing_accept(text, goal=goal, transitions=transitions,
                                   cancel_first=cancel_first, real_accept=real_accept):
                    if cancel_first:
                        self.actionlib.cancel_cb(goal)
                    if transitions:
                        real_accept(text)
                    raise RuntimeError("set_accepted failed")

                goal.set_accepted = failing_accept
                self.actionlib.goal_cb(goal)
                self.assert_released_once(goal, expected, code)
                self.assertEqual(self.server.accepted_count, int(transitions))

    def test_staging_cleanup_failure_before_accept_still_terminates(self):
        # From the reviewer's probe: the session cannot be built and the staging cleanup
        # then fails, all before acceptance. The IO_ERROR must go out through a method the
        # goal can take (PENDING: rejected; RECALLING after a cancel: canceled) instead of
        # aborted, which actionlib refuses there.
        for name, cancel_first, expected in (("pending", False, ["rejected"]),
                                             ("recalling", True, ["canceled"])):
            with self.subTest(case=name):
                goal = CaptureGoal(f"pin-capture-cleanup-failure-{name}")

                def transaction_factory(config, scene_id):
                    transaction = capture_tests.allocate_snapshot_transaction(config, scene_id)
                    real_cleanup = transaction.cleanup

                    def failing_cleanup():
                        real_cleanup()
                        raise OSError("injected cleanup failure")

                    transaction.cleanup = failing_cleanup
                    return transaction

                def session_factory(*args, goal=goal, cancel_first=cancel_first, **kwargs):
                    if cancel_first:
                        # As actionlib does for a cancel: mark the goal, then call back.
                        if goal.set_cancel_requested():
                            self.actionlib.cancel_cb(goal)
                    raise RuntimeError("injected session construction failure")

                self.server.shutdown()
                self.server = self.make_server(
                    transaction_factory=transaction_factory, session_factory=session_factory)
                self.actionlib.goal_cb(goal)
                self.assert_released_once(goal, expected, CaptureSegmentResult.IO_ERROR)
                self.assertEqual(self.server.accepted_count, 0)


if __name__ == "__main__":
    unittest.main()
