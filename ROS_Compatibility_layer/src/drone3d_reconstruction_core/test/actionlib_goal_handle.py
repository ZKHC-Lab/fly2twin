"""Test goal handles built on actionlib's real ServerGoalHandle.

The state machine is actionlib's own: a set_* call that is illegal in the current status
logs an error and changes nothing, exactly as on the robot. Each handle records the calls
that took effect in ``events`` and the refused ones in ``refused``; the test classes fail any
test that leaves a refused transition behind (``assert_no_refused_transitions``).
"""

from types import SimpleNamespace
import threading

import rospy
from actionlib.server_goal_handle import ServerGoalHandle
from actionlib_msgs.msg import GoalID, GoalStatus

# Terminal transitions stamp rospy.Time.now(); tests run without a ROS node.
rospy.rostime.set_rostime_initialized(True)


class RecordingGoalHandle(ServerGoalHandle):
    created = []  # every handle, so a test can check none was driven illegally

    def __init__(self, identifier, goal, status=GoalStatus.PENDING):
        status_message = GoalStatus(
            goal_id=GoalID(id=identifier, stamp=rospy.Time(1, len(identifier))),
            status=status,
        )
        tracker = SimpleNamespace(
            status=status_message,
            goal=SimpleNamespace(goal=goal),
            handle_destruction_time=None,
        )
        self.events = []
        self.refused = []
        self.feedbacks = []
        endpoint = SimpleNamespace(
            lock=threading.RLock(),
            publish_status=lambda: None,
            publish_result=lambda status, result: None,
            publish_feedback=lambda status, feedback: self.feedbacks.append(feedback),
            ActionResultType=SimpleNamespace,
        )
        super().__init__(tracker, endpoint, None)
        # Direct views used by the tests.
        self._goal_status = status_message
        self._goal_id = status_message.goal_id
        RecordingGoalHandle.created.append(self)

    @property
    def _goal(self):
        return self.goal.goal

    @_goal.setter
    def _goal(self, value):
        self.goal.goal = value

    def _record(self, name, before, result, text):
        if self._goal_status.status != before:
            self.events.append((name, result, text))
        else:
            self.refused.append((name, before))

    def set_accepted(self, text=""):
        before = self._goal_status.status
        super().set_accepted(text)
        self._record("accepted", before, None, text)

    def set_rejected(self, result=None, text=""):
        before = self._goal_status.status
        super().set_rejected(result, text)
        self._record("rejected", before, result, text)

    def set_canceled(self, result=None, text=""):
        before = self._goal_status.status
        super().set_canceled(result, text)
        self._record("canceled", before, result, text)

    def set_aborted(self, result=None, text=""):
        before = self._goal_status.status
        super().set_aborted(result, text)
        self._record("aborted", before, result, text)

    def set_succeeded(self, result=None, text=""):
        before = self._goal_status.status
        super().set_succeeded(result, text)
        self._record("succeeded", before, result, text)

    # ServerGoalHandle compares and hashes by goal id; tests need identity semantics.
    __eq__ = object.__eq__
    __ne__ = object.__ne__
    __hash__ = object.__hash__


def assert_no_refused_transitions(test):
    """Every goal of the test was only driven through transitions actionlib accepts."""
    refused = [(handle._goal_id.id, handle.refused)
               for handle in RecordingGoalHandle.created if handle.refused]
    RecordingGoalHandle.created = []
    test.assertEqual(refused, [], "illegal actionlib transitions were attempted")
