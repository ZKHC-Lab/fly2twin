"""Contract of the shared terminal executor, on actionlib's real ServerGoalHandle.

The servers' own behaviour stays pinned in test_action_terminal_pins.py and the two server
test files; these tests fix what every caller may rely on from converge_terminal itself.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

from actionlib_msgs.msg import GoalStatus

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actionlib_goal_handle import RecordingGoalHandle  # noqa: E402

from drone3d_reconstruction_core.action_terminal import (  # noqa: E402
    TerminalCall,
    converge_terminal,
)


def invoke(handle, call):
    getattr(handle, "set_" + call.method)(call.result, call.text)


class ConvergeTerminalContract(unittest.TestCase):
    def setUp(self):
        self.observed = []
        self.fallbacks = []

    def handle(self, status=GoalStatus.ACTIVE):
        return RecordingGoalHandle("contract", SimpleNamespace(), status)

    def fallback_to(self, method):
        def fallback(primary, status, error):
            self.fallbacks.append((primary, status, error))
            return TerminalCall(method, primary.result, "fallback")
        return fallback

    def converge(self, handle, primary, fallback_method="aborted", invoke=invoke, observer=None):
        return converge_terminal(
            handle, primary, invoke=invoke, fallback=self.fallback_to(fallback_method),
            observer=observer or self.observed.append, log_context="contract",
        )

    def test_a_legal_primary_call_converges_without_fallback(self):
        handle = self.handle()
        primary = TerminalCall("succeeded", "result", "done")
        outcome = self.converge(handle, primary)
        self.assertTrue(outcome.converged)
        self.assertIs(outcome.final_call, primary)
        self.assertEqual(outcome.observed_status, GoalStatus.SUCCEEDED)
        self.assertEqual([event[0] for event in handle.events], ["succeeded"])
        self.assertEqual(self.observed, [primary])
        self.assertEqual(self.fallbacks, [])

    def test_a_refused_primary_call_gets_one_fallback_with_the_observed_status(self):
        handle = self.handle(GoalStatus.PENDING)  # aborted is illegal before acceptance
        primary = TerminalCall("aborted", "result", "failed")
        outcome = self.converge(handle, primary, fallback_method="rejected")
        self.assertTrue(outcome.converged)
        self.assertEqual(outcome.final_call.method, "rejected")
        self.assertEqual(outcome.observed_status, GoalStatus.REJECTED)
        self.assertEqual(self.fallbacks, [(primary, GoalStatus.PENDING, None)])
        self.assertEqual(handle.refused, [("aborted", GoalStatus.PENDING)])
        # The observer heard of the primary choice only.
        self.assertEqual(self.observed, [primary])

    def test_a_raising_primary_call_passes_its_error_to_the_fallback(self):
        handle = self.handle()
        calls = []

        def failing_once(goal, call):
            calls.append(call.method)
            if len(calls) == 1:
                raise RuntimeError("primary failed")
            invoke(goal, call)

        primary = TerminalCall("succeeded", "result", "done")
        outcome = self.converge(handle, primary, fallback_method="succeeded", invoke=failing_once)
        self.assertTrue(outcome.converged)
        self.assertEqual(calls, ["succeeded", "succeeded"])
        (seen_primary, status, error), = self.fallbacks
        self.assertIs(seen_primary, primary)
        self.assertEqual(status, GoalStatus.ACTIVE)
        self.assertIsInstance(error, RuntimeError)

    def test_unconfirmed_status_never_counts_as_converged_and_stops_after_two_calls(self):
        handle = self.handle()
        calls = []
        real_status = handle.get_goal_status

        def unreadable():
            real_status()
            raise RuntimeError("status unavailable")

        handle.get_goal_status = unreadable

        def counting(goal, call):
            calls.append(call.method)
            invoke(goal, call)

        outcome = self.converge(handle, TerminalCall("succeeded", "result", "done"),
                                fallback_method="succeeded", invoke=counting)
        # actionlib did reach SUCCEEDED, but nothing proved it: the caller keeps the slot.
        self.assertFalse(outcome.converged)
        self.assertIsNone(outcome.observed_status)
        self.assertEqual(calls, ["succeeded", "succeeded"])
        self.assertEqual(handle._goal_status.status, GoalStatus.SUCCEEDED)
        (_, status, error), = self.fallbacks
        self.assertIsNone(status)
        self.assertIsInstance(error, RuntimeError)

    def test_an_observer_failure_does_not_stop_convergence(self):
        handle = self.handle()

        def broken(_call):
            raise RuntimeError("observer fault")

        outcome = self.converge(handle, TerminalCall("succeeded", "result", "done"),
                                observer=broken)
        self.assertTrue(outcome.converged)
        self.assertEqual([event[0] for event in handle.events], ["succeeded"])


if __name__ == "__main__":
    unittest.main()
