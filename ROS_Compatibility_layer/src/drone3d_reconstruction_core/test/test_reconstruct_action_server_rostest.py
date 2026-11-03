#!/usr/bin/env python3
"""ROS transport test for the production low-level Action adapter."""

from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import actionlib
from actionlib_msgs.msg import GoalStatus
import rospy
import rostest

from drone3d_reconstruction_msgs.msg import (
    ReconstructSegmentAction,
    ReconstructSegmentGoal,
    ReconstructSegmentResult,
)

from drone3d_reconstruction_core.controller import JobFinished
from drone3d_reconstruction_core.reconstruct_action_server import (
    PreparedLaunch,
    ReconstructActionServer,
    VerifiedCompletion,
)
from drone3d_reconstruction_core.result_transaction import (
    allocate_result_transaction,
)


FAKE_WORKER = Path(__file__).resolve().with_name("fake_worker.py")
ACTION_NAME = "/test_reconstruct_segment"


class ReconstructActionServerRostest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.ready_file = Path(self.temporary_directory.name) / "worker.ready"
        self.result_root = Path(self.temporary_directory.name) / "results"
        self.result_root.mkdir()
        self.result_sequence = 0
        self.transactions = []
        self.completion_metadata = []
        self.builder_entered = threading.Event()
        self.builder_release = threading.Event()
        self.server = ReconstructActionServer(
            action_name=ACTION_NAME,
            command_builder=self.command_builder,
            worker_timeout_sec=0.5,
            term_grace_sec=0.05,
        )
        self.client_a = actionlib.SimpleActionClient(
            ACTION_NAME, ReconstructSegmentAction
        )
        self.client_b = actionlib.SimpleActionClient(
            ACTION_NAME, ReconstructSegmentAction
        )
        self.assertTrue(self.client_a.wait_for_server(rospy.Duration(3.0)))
        self.assertTrue(self.client_b.wait_for_server(rospy.Duration(3.0)))

    def tearDown(self):
        self.builder_release.set()
        self.server.shutdown()
        self.temporary_directory.cleanup()

    def command_builder(self, goal):
        prefix = "test://"
        if not goal.snapshot_uri.startswith(prefix):
            raise ValueError("test goal must use test:// mode")
        mode = goal.snapshot_uri[len(prefix) :]
        if mode == "block-start":
            self.builder_entered.set()
            if not self.builder_release.wait(3.0):
                raise RuntimeError("test did not release command builder")
            mode = "success"
        command = [sys.executable, str(FAKE_WORKER), mode]
        if mode in ("sleep", "ignore-term"):
            command.extend(("--seconds", "10"))
        if mode == "ignore-term":
            command.extend(("--ready-file", str(self.ready_file)))
        self.result_sequence += 1
        transaction = allocate_result_transaction(
            self.result_root, "result-{}".format(self.result_sequence)
        )
        self.transactions.append(transaction)
        metadata = VerifiedCompletion(
            scene_id="result-{}".format(self.result_sequence),
            result_manifest_uri=(transaction.final_path / "manifest.json").as_uri(),
            result_manifest_sha256="b" * 64,
            scene_glb_uri=(transaction.final_path / "scene.glb").as_uri(),
            frame_count=24,
        )
        self.completion_metadata.append(metadata)

        def verify_completion(
            _finished: JobFinished, _cancellation_event: threading.Event
        ) -> VerifiedCompletion:
            return metadata

        return PreparedLaunch(tuple(command), transaction, verify_completion)

    def test_busy_cancel_success_failure_and_timeout(self):
        self.client_a.send_goal(
            ReconstructSegmentGoal(snapshot_uri="test://block-start")
        )
        self.assertTrue(self.builder_entered.wait(2.0))

        pending_contender_was_active = threading.Event()
        self.client_b.send_goal(
            ReconstructSegmentGoal(snapshot_uri="test://success"),
            active_cb=pending_contender_was_active.set,
        )
        self.assertTrue(self.client_b.wait_for_result(rospy.Duration(2.0)))
        self.assertEqual(self.client_b.get_state(), GoalStatus.REJECTED)
        self.assertEqual(
            self.client_b.get_result().error_code,
            ReconstructSegmentResult.BUSY,
        )
        self.assertFalse(pending_contender_was_active.is_set())

        self.client_a.cancel_goal()
        self.assertTrue(
            _wait_for_condition(lambda: self.server.cancel_request_count == 1, 2.0)
        )
        self.builder_release.set()
        self.assertTrue(self.client_a.wait_for_result(rospy.Duration(2.0)))
        self.assertEqual(self.client_a.get_state(), GoalStatus.RECALLED)
        self.assertEqual(
            self.client_a.get_result().error_code,
            ReconstructSegmentResult.CANCELLED,
        )

        self.assertTrue(
            _wait_for_condition(lambda: not self.server.is_busy(), 3.0)
        )
        self.client_a.send_goal(ReconstructSegmentGoal(snapshot_uri="test://sleep"))
        self.assertTrue(_wait_for_state(self.client_a, GoalStatus.ACTIVE, 5.0))

        second_was_active = threading.Event()
        self.client_b.send_goal(
            ReconstructSegmentGoal(snapshot_uri="test://success"),
            active_cb=second_was_active.set,
        )
        self.assertTrue(self.client_b.wait_for_result(rospy.Duration(2.0)))
        self.assertEqual(self.client_b.get_state(), GoalStatus.REJECTED)
        self.assertEqual(
            self.client_b.get_result().error_code,
            ReconstructSegmentResult.BUSY,
        )
        self.assertFalse(second_was_active.is_set())
        self.assertEqual(self.client_a.get_state(), GoalStatus.ACTIVE)

        self.client_a.cancel_goal()
        self.assertTrue(self.client_a.wait_for_result(rospy.Duration(3.0)))
        self.assertEqual(self.client_a.get_state(), GoalStatus.PREEMPTED)
        self.assertEqual(
            self.client_a.get_result().error_code,
            ReconstructSegmentResult.CANCELLED,
        )

        self.assertTrue(
            _wait_for_condition(lambda: not self.server.is_busy(), 3.0)
        )
        self.client_b.send_goal(ReconstructSegmentGoal(snapshot_uri="test://success"))
        self.assertTrue(self.client_b.wait_for_result(rospy.Duration(5.0)))
        successful_transaction = self.transactions[-1]
        successful_metadata = self.completion_metadata[-1]
        self.assertEqual(self.client_b.get_state(), GoalStatus.SUCCEEDED)
        result = self.client_b.get_result()
        self.assertEqual(
            result.error_code,
            ReconstructSegmentResult.OK,
        )
        self.assertEqual(result.scene_id, successful_metadata.scene_id)
        self.assertEqual(
            result.result_manifest_uri, successful_metadata.result_manifest_uri
        )
        self.assertEqual(
            result.result_manifest_sha256,
            successful_metadata.result_manifest_sha256,
        )
        self.assertEqual(result.scene_glb_uri, successful_metadata.scene_glb_uri)
        self.assertEqual(result.frame_count, successful_metadata.frame_count)
        self.assertTrue(successful_transaction.committed)
        self.assertTrue(successful_transaction.final_path.is_dir())
        self.assertFalse(successful_transaction.staging_path.exists())

        self.assertTrue(
            _wait_for_condition(lambda: not self.server.is_busy(), 3.0)
        )
        self.client_b.send_goal(ReconstructSegmentGoal(snapshot_uri="test://fail"))
        self.assertTrue(self.client_b.wait_for_result(rospy.Duration(3.0)))
        self.assertEqual(self.client_b.get_state(), GoalStatus.ABORTED)
        self.assertEqual(
            self.client_b.get_result().error_code,
            ReconstructSegmentResult.ALGORITHM_FAILED,
        )
        self.assertIn("23", self.client_b.get_result().message)

        self.assertTrue(
            _wait_for_condition(lambda: not self.server.is_busy(), 3.0)
        )
        self.client_b.send_goal(
            ReconstructSegmentGoal(snapshot_uri="test://ignore-term")
        )
        self.assertTrue(_wait_for_path(self.ready_file, 2.0))
        self.assertTrue(self.client_b.wait_for_result(rospy.Duration(3.0)))
        self.assertEqual(self.client_b.get_state(), GoalStatus.ABORTED)
        self.assertEqual(
            self.client_b.get_result().error_code,
            ReconstructSegmentResult.ALGORITHM_FAILED,
        )
        self.assertIn("timed out", self.client_b.get_result().message)

        self.assertEqual(self.server.busy_rejection_count, 2)
        self.assertEqual(self.server.cancel_request_count, 2)
        self.assertEqual(self.server.accepted_count, 4)
        self.assertEqual(self.server.terminal_attempt_count, 5)
        self.assertTrue(
            _wait_for_condition(lambda: not self.server.is_busy(), 3.0)
        )


def _wait_for_condition(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _wait_for_state(client, expected, timeout):
    deadline = time.monotonic() + timeout
    terminal = {
        GoalStatus.PREEMPTED,
        GoalStatus.SUCCEEDED,
        GoalStatus.ABORTED,
        GoalStatus.REJECTED,
        GoalStatus.RECALLED,
        GoalStatus.LOST,
    }
    while time.monotonic() < deadline:
        state = client.get_state()
        if state == expected:
            return True
        if state in terminal:
            return False
        time.sleep(0.01)
    return False


def _wait_for_path(path, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return path.exists()


if __name__ == "__main__":
    rospy.init_node("test_reconstruct_action_server")
    rostest.rosrun(
        "drone3d_reconstruction_core",
        "test_reconstruct_action_server",
        ReconstructActionServerRostest,
    )
