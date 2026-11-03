"""CPU-only tests for the CaptureSegment action server arbitration."""

from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from actionlib_msgs.msg import GoalStatus

from drone3d_reconstruction_msgs.msg import CaptureSegmentResult

from drone3d_reconstruction_core.capture import (
    CaptureConfig,
    CaptureSession,
    FrameStamp,
)
from drone3d_reconstruction_core.capture_action_server import (
    _CaptureRecord,
    CaptureActionServer,
)
from drone3d_reconstruction_core.snapshot_validator import SnapshotSafetyLimits


class FakeSubscriber:
    def __init__(self, topic, message_type, callback, queue_size, buff_size=None):
        self.topic = topic
        self.callback = callback
        self.queue_size = queue_size
        self.buff_size = buff_size
        self.unregistered = False

    def unregister(self):
        self.unregistered = True


class FakeActionServer:
    def __init__(self, name, action_type, goal_cb, cancel_cb, auto_start):
        self.goal_cb = goal_cb
        self.cancel_cb = cancel_cb
        self.started = False
        self.stop_calls = 0

    def start(self):
        self.started = True

    def stop(self):
        self.stop_calls += 1


class FakeGoalHandle:
    def __init__(self, identifier, goal=None, status=GoalStatus.PENDING):
        self._goal_id = SimpleNamespace(
            id=identifier,
            stamp=SimpleNamespace(secs=1, nsecs=len(identifier)),
        )
        self._goal = goal or SimpleNamespace(
            scene_id="scene-capture",
            sample_fps=2.0,
            max_frames=2,
            max_duration=SimpleNamespace(to_sec=lambda: 5.0),
        )
        self._goal_status = SimpleNamespace(status=status)
        self.events = []
        self.feedbacks = []

    def publish_feedback(self, feedback):
        self.feedbacks.append(feedback)

    def get_goal_status(self):
        return self._goal_status

    def get_goal(self):
        return self._goal

    def set_accepted(self, text):
        self._goal_status.status = GoalStatus.ACTIVE
        self.events.append(("accepted", None, text))

    def set_rejected(self, result, text):
        self._goal_status.status = GoalStatus.REJECTED
        self.events.append(("rejected", result, text))

    def set_succeeded(self, result, text):
        self._goal_status.status = GoalStatus.SUCCEEDED
        self.events.append(("succeeded", result, text))

    def set_aborted(self, result, text):
        self._goal_status.status = GoalStatus.ABORTED
        self.events.append(("aborted", result, text))

    def set_preempted(self, result, text):
        self._goal_status.status = GoalStatus.PREEMPTED
        self.events.append(("preempted", result, text))

    def set_canceled(self, result, text):
        if self._goal_status.status in (GoalStatus.PENDING, GoalStatus.RECALLING):
            self._goal_status.status = GoalStatus.RECALLED
        else:
            self._goal_status.status = GoalStatus.PREEMPTED
        self.events.append(("canceled", result, text))


class ActionServerStyleGoalHandle:
    """Minimal ``actionlib.ActionServer`` GoalHandle 形状。

    与 SimpleActionServer 的关键差异：goal id 通过 ``get_goal_id()`` 提供，
    取消终态只有 ``set_canceled``、没有 ``set_preempted``。
    """

    def __init__(self, identifier, goal=None):
        self._goal_id = SimpleNamespace(id=identifier)
        self._goal = goal or SimpleNamespace(
            scene_id="scene-capture",
            sample_fps=2.0,
            max_frames=2,
            max_duration=SimpleNamespace(to_sec=lambda: 5.0),
        )
        self._goal_status = SimpleNamespace(status=GoalStatus.PENDING)
        self.events = []
        self.feedbacks = []

    def publish_feedback(self, feedback):
        self.feedbacks.append(feedback)

    def get_goal_id(self):
        return self._goal_id

    def get_goal_status(self):
        return self._goal_status

    def get_goal(self):
        return self._goal

    def set_accepted(self, text):
        self._goal_status.status = GoalStatus.ACTIVE
        self.events.append(("accepted", None, text))

    def set_rejected(self, result, text):
        self._goal_status.status = GoalStatus.REJECTED
        self.events.append(("rejected", result, text))

    def set_canceled(self, result, text):
        self._goal_status.status = GoalStatus.PREEMPTED
        self.events.append(("canceled", result, text))


class FailingSuccessGoalHandle(FakeGoalHandle):
    def __init__(self, identifier, goal=None):
        super().__init__(identifier, goal=goal)
        self.fail_first = True

    def set_succeeded(self, result, text):
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError("injected set_succeeded failure")
        super().set_succeeded(result, text)


class CaptureActionServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot_root = self.root / "snapshots"
        self.snapshot_root.mkdir()
        self.subscribers = []
        self.config = CaptureConfig(
            snapshot_root=str(self.snapshot_root),
            image_topic="/camera4/image_raw",
            camera_name="camera4",
            camera_role="forward",
            camera_frame_id="camera4_optical_frame",
            image_encoding="bgr8",
            odometry_topic="/Odometry",
            camera_info_topic=None,
            depth_topic=None,
            tf_topic=None,
            tf_static_topic=None,
            save_sensor_depth=False,
            body_frame_id="body",
            tf_relevant_frames=(),
            camera_info_max_delta_ms=100.0,
            depth_max_delta_ms=100.0,
            max_auxiliary_samples=8,
            max_tf_samples=64,
            max_tf_bytes=256 * 1024,
            source_timeout_sec=5.0,
            max_window_gap_sec=30.0,
            min_sample_fps=0.5,
            max_sample_fps=30.0,
            max_goal_frames=4,
            max_goal_duration_sec=10.0,
            min_captured_frames=1,
            max_image_width=640,
            max_image_height=480,
            pose_max_delta_ms=50.0,
            pose_gap_max_ms=200.0,
            max_sample_interval_overshoot_ms=250.0,
            snapshot_limits=SnapshotSafetyLimits(
                max_frame_count=4,
                max_manifest_entries=64,
                max_metadata_bytes=1024 * 1024,
                max_file_bytes=1024 * 1024,
                max_total_bytes=8 * 1024 * 1024,
                max_scene_id_bytes=128,
                max_relative_path_bytes=1024,
            ),
        )
        self.server = self.make_server()

    def tearDown(self):
        self.server.shutdown()

    def make_server(self, **kwargs):
        return CaptureActionServer(
            action_name="/test_capture",
            config=self.config,
            action_server_factory=FakeActionServer,
            subscriber_factory=self.subscriber_factory,
            **kwargs,
        )

    def subscriber_factory(
        self, topic, message_type, callback, queue_size, buff_size=None
    ):
        subscriber = FakeSubscriber(
            topic, message_type, callback, queue_size, buff_size
        )
        self.subscribers.append(subscriber)
        return subscriber

    def image_message(self, sec, nsec=0):
        import numpy as np

        data = np.zeros((3, 4, 3), dtype=np.uint8)
        return SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=sec, nsec=nsec),
                frame_id="camera4_optical_frame",
            ),
            encoding="bgr8",
            width=4,
            height=3,
            step=12,
            data=bytes(data.tobytes()),
            is_bigendian=0,
        )

    def wait_events(self, goal, count, timeout=2.0):
        deadline = time.monotonic() + timeout
        while len(goal.events) < count and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertGreaterEqual(len(goal.events), count)

    def image_callback(self):
        return [s.callback for s in self.subscribers if s.topic == "/camera4/image_raw"][0]

    def _handoff_server(self):
        server = CaptureActionServer(
            action_name="/test_handoff", config=replace(self.config,
                odometry_association="interpolate", pose_gap_max_ms=600.0,
                handoff_buffer_frames=8),
            action_server_factory=FakeActionServer, subscriber_factory=self.subscriber_factory,
        )
        self.addCleanup(server.shutdown)
        return server

    def _handoff_goal(self, server, scene):
        goal = FakeGoalHandle(scene, SimpleNamespace(scene_id=scene, sample_fps=2.0,
            max_frames=2, max_duration=SimpleNamespace(to_sec=lambda: 5.0)))
        server._goal_callback(goal)
        self.wait_events(goal, 1)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with server._lock:
                if server._active and server._active.session is not None:
                    break
            time.sleep(.005)
        return goal

    def _handoff_image(self, server, sec, nsec=0):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_capture import CaptureTestBase
        server._odometry_callback(CaptureTestBase.odometry_message(sec, nsec))
        server._image_callback(self.image_message(sec, nsec))

    def test_handoff_replays_gap_frames_with_their_pose_and_without_duplicates(self):
        import csv
        server = self._handoff_server()
        first = self._handoff_goal(server, "cs-s0123456789abcdef-w0")
        self._handoff_image(server, 10)
        self._handoff_image(server, 10, 500000000)
        self.wait_events(first, 2)
        self.assertFalse(server.is_busy())
        self._handoff_image(server, 11)
        self._handoff_image(server, 11, 500000000)
        self.assertEqual(server.diagnostics_snapshot()["handoff_queued_frames"], 2)
        second = self._handoff_goal(server, "cs-s0123456789abcdef-w1")
        self.wait_events(second, 2)
        self.assertEqual(second.events[-1][0], "succeeded")
        from urllib.parse import unquote, urlsplit
        root = Path(unquote(urlsplit(second.events[-1][1].snapshot_uri).path))
        rows = list(csv.DictReader((root / "frames.csv").open()))
        self.assertEqual([(int(r["stamp_sec"]), int(r["stamp_nsec"])) for r in rows],
                         [(11, 0), (11, 500000000)])
        self.assertEqual([r["pose_valid"] for r in rows], ["true", "true"])
        self.assertEqual(server.diagnostics_snapshot()["handoff_replayed_frames"], 2)

    def test_handoff_never_replays_frames_into_a_new_session(self):
        server = self._handoff_server()
        first = self._handoff_goal(server, "cs-s0123456789abcdef-w0")
        self._handoff_image(server, 10)
        self._handoff_image(server, 10, 500000000)
        self.wait_events(first, 2)
        self._handoff_image(server, 11)
        second = self._handoff_goal(server, "cs-s1123456789abcdef-w1")
        with server._lock:
            self.assertEqual(server._active.session.writer.frame_count, 0)
        self._handoff_image(server, 12)
        self._handoff_image(server, 12, 500000000)
        self.wait_events(second, 2)
        self.assertEqual(server.diagnostics_snapshot()["handoff_replayed_frames"], 0)

    def test_handoff_buffer_is_bounded_and_reports_eviction(self):
        from drone3d_reconstruction_core.capture_handoff import ImageHandoff
        queue = ImageHandoff(3, 72)  # tiny fixtures are 36 bytes per frame
        for sec in (10, 11, 12):
            queue.push(self.image_message(sec))
        self.assertEqual(queue.bytes, 72)
        self.assertEqual(queue.evicted, 1)
        self.assertEqual(queue.pop_after(0)[0], 11 * 10**9)
        self.assertEqual(queue.pop_after(0)[0], 12 * 10**9)
        self.assertEqual(queue.bytes, 0)
        self.assertIsNone(queue.pop_after(0))

    def test_cancel_breaks_handoff_continuity(self):
        server = self._handoff_server()
        first = self._handoff_goal(server, "cs-s0123456789abcdef-w0")
        self._handoff_image(server, 10)
        server._cancel_callback(first)
        self.wait_events(first, 2)
        self._handoff_image(server, 11)
        second = self._handoff_goal(server, "cs-s0123456789abcdef-w1")
        with server._lock:
            self.assertEqual(server._active.session.writer.frame_count, 0)
        self._handoff_image(server, 12)
        self._handoff_image(server, 12, 500000000)
        self.wait_events(second, 2)
        self.assertEqual(server.diagnostics_snapshot()["handoff_replayed_frames"], 0)

    def test_image_time_reset_discards_handoff_history(self):
        server = self._handoff_server()
        server._handoff_predecessor = ("cs-s0123456789abcdef-w0", 10 * 10**9)
        server._image_callback(self.image_message(11))
        server._image_callback(self.image_message(9))
        self.assertIsNone(server._handoff_predecessor)
        self.assertEqual(len(server._handoff.queue), 1)
        self.assertEqual(server._handoff.pop_after(0)[0], 9 * 10**9)

    def test_image_subscriber_buffer_covers_maximum_raw_frame(self):
        image_subscriber = [
            subscriber
            for subscriber in self.subscribers
            if subscriber.topic == self.config.image_topic
        ][0]
        self.assertGreater(
            image_subscriber.buff_size,
            self.config.max_image_width * self.config.max_image_height * 3,
        )
        # 图像回调被磁盘/worker 竞争拖慢时，queue_size=1 会丢帧造成
        # 选中帧 stamp 跳变；至少保留约 1 秒 15Hz 缓冲。
        self.assertGreaterEqual(image_subscriber.queue_size, 16)

    def test_shutdown_stops_action_server_once(self):
        action_server = self.server._server
        self.server.shutdown()
        self.server.shutdown()
        self.assertEqual(action_server.stop_calls, 1)

    def test_valid_goal_seals_commits_and_publishes_snapshot_metadata(self):
        goal = FakeGoalHandle("valid")
        self.server._goal_callback(goal)
        self.wait_events(goal, 1)
        self.image_callback()(self.image_message(10, 0))
        self.image_callback()(self.image_message(10, 500000000))
        self.wait_events(goal, 2, timeout=3.0)
        self.assertEqual(goal._goal_status.status, GoalStatus.SUCCEEDED)
        result = goal.events[-1][1]
        self.assertEqual(result.error_code, CaptureSegmentResult.OK)
        self.assertEqual(result.scene_id, "scene-capture")
        self.assertEqual(result.camera_name, "camera4")
        self.assertEqual(result.image_topic, "/camera4/image_raw")
        self.assertEqual(result.frame_count, 2)
        self.assertTrue(result.snapshot_uri.startswith("file://"))
        self.assertTrue((self.snapshot_root / Path(result.snapshot_uri[7:]).name).is_dir())

    def test_valid_goal_fsync_failure_still_succeeds_with_warning(self):
        goal = FakeGoalHandle("valid-fsync")
        self.server._goal_callback(goal)
        self.wait_events(goal, 1)
        import drone3d_reconstruction_core.result_transaction as rt

        original_commit = rt.ResultTransaction.commit

        def commit_then_fsync_fail(transaction):
            original_commit(transaction)
            raise rt.ResultCommitError(
                "PUBLISHED_FSYNC_FAILED",
                "published result directory could not be fsynced",
            )

        with mock.patch.object(rt.ResultTransaction, "commit", commit_then_fsync_fail):
            self.image_callback()(self.image_message(10, 0))
            self.image_callback()(self.image_message(10, 500000000))
            self.wait_events(goal, 2, timeout=3.0)
        self.assertEqual(goal._goal_status.status, GoalStatus.SUCCEEDED)
        result = goal.events[-1][1]
        self.assertEqual(result.error_code, CaptureSegmentResult.OK)
        self.assertIn("warning", result.message)
        self.assertTrue(
            (self.snapshot_root / Path(result.snapshot_uri[7:]).name).is_dir()
        )

    def test_dispatch_failure_rejects_goal_and_releases_slot(self):
        server = self.make_server(
            launch_dispatch=lambda target, args: object()
        )
        goal = FakeGoalHandle("dispatch-fail")
        server._goal_callback(goal)
        self.wait_events(goal, 1)
        self.assertEqual(goal._goal_status.status, GoalStatus.REJECTED)
        self.assertEqual(
            goal.events[-1][1].error_code, CaptureSegmentResult.INTERNAL_ERROR
        )
        self.assertFalse(server.is_busy())
        server.shutdown()

    def test_terminal_primary_failure_retries_and_converges(self):
        goal = FailingSuccessGoalHandle("terminal-retry")
        self.server._goal_callback(goal)
        self.wait_events(goal, 1)
        self.image_callback()(self.image_message(10, 0))
        self.image_callback()(self.image_message(10, 500000000))
        self.wait_events(goal, 2, timeout=3.0)
        self.assertEqual(goal._goal_status.status, GoalStatus.SUCCEEDED)
        self.assertFalse(self.server.is_busy())

    def test_second_goal_is_rejected_busy_before_accept(self):
        first = FakeGoalHandle("first")
        self.server._goal_callback(first)
        self.wait_events(first, 1)
        second = FakeGoalHandle("second")
        self.server._goal_callback(second)
        self.wait_events(second, 1)
        self.assertEqual(second._goal_status.status, GoalStatus.REJECTED)
        self.assertEqual(
            second.events[-1][1].error_code,
            CaptureSegmentResult.CAMERA_UNAVAILABLE,
        )
        # Release the first slot for teardown.
        self.server._cancel_callback(first)
        self.wait_events(first, 2, timeout=3.0)

    def test_invalid_goal_rejected_pre_accept(self):
        goal = FakeGoalHandle(
            "invalid",
            goal=SimpleNamespace(
                scene_id="bad scene",
                sample_fps=2.0,
                max_frames=2,
                max_duration=SimpleNamespace(to_sec=lambda: 5.0),
            ),
        )
        self.server._goal_callback(goal)
        self.wait_events(goal, 1)
        self.assertEqual(goal._goal_status.status, GoalStatus.REJECTED)
        self.assertEqual(
            goal.events[-1][1].error_code, CaptureSegmentResult.INVALID_GOAL
        )

    def test_image_stamp_regression_aborts_with_camera_unavailable(self):
        goal = FakeGoalHandle("regress")
        self.server._goal_callback(goal)
        self.wait_events(goal, 1)
        self.image_callback()(self.image_message(10, 0))
        self.image_callback()(self.image_message(9, 0))
        self.wait_events(goal, 2, timeout=3.0)
        self.assertEqual(goal._goal_status.status, GoalStatus.ABORTED)
        self.assertEqual(
            goal.events[-1][1].error_code,
            CaptureSegmentResult.CAMERA_UNAVAILABLE,
        )
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_cancel_after_accept_leaves_no_snapshot(self):
        goal = FakeGoalHandle("cancel")
        self.server._goal_callback(goal)
        self.wait_events(goal, 1)
        self.image_callback()(self.image_message(10, 0))
        self.server._cancel_callback(goal)
        self.wait_events(goal, 2, timeout=3.0)
        self.assertEqual(goal._goal_status.status, GoalStatus.PREEMPTED)
        self.assertEqual(goal.events[-1][0], "preempted")
        self.assertEqual(
            goal.events[-1][1].error_code, CaptureSegmentResult.CANCELLED
        )
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_cancel_after_accept_without_further_frames_terminates(self):
        # PAUSE + STOP 复现：已接受的 capture 在没有任何后续传感器回调时
        # 收到 cancel，也必须立刻 PREEMPTED，而不是停在 PREEMPTING。
        goal = FakeGoalHandle("cancel-no-frames")
        self.server._goal_callback(goal)
        self.wait_events(goal, 1)
        self.server._cancel_callback(goal)
        self.wait_events(goal, 2, timeout=3.0)
        self.assertEqual(goal._goal_status.status, GoalStatus.PREEMPTED)
        self.assertEqual(goal.events[-1][0], "preempted")
        self.assertEqual(
            goal.events[-1][1].error_code, CaptureSegmentResult.CANCELLED
        )
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_cancel_with_action_server_style_handle_terminates(self):
        # 生产环境用 actionlib.ActionServer：cancel 回调收到的是另一个
        # GoalHandle 包装，且没有 set_preempted。必须按 goal id 匹配目标
        # 并回退到 set_canceled，否则 PAUSE+STOP 永远停在 PREEMPTING。
        goal = ActionServerStyleGoalHandle("nx-action-server")
        self.server._goal_callback(goal)
        self.wait_events(goal, 1)
        cancel_handle = ActionServerStyleGoalHandle(
            "nx-action-server", goal=goal._goal
        )
        self.server._cancel_callback(cancel_handle)
        self.wait_events(goal, 2, timeout=3.0)
        self.assertEqual(goal._goal_status.status, GoalStatus.PREEMPTED)
        self.assertEqual(goal.events[-1][0], "canceled")
        self.assertEqual(
            goal.events[-1][1].error_code, CaptureSegmentResult.CANCELLED
        )
        self.assertEqual(list(self.snapshot_root.iterdir()), [])

    def test_session_factory_failure_rejects_and_cleans_staging(self):
        server = self.make_server(
            session_factory=mock.Mock(side_effect=RuntimeError("session boom"))
        )
        goal = FakeGoalHandle("session-fail")
        server._goal_callback(goal)
        self.wait_events(goal, 1)
        self.assertEqual(goal._goal_status.status, GoalStatus.REJECTED)
        self.assertEqual(
            goal.events[-1][1].error_code, CaptureSegmentResult.INTERNAL_ERROR
        )
        self.assertEqual(list(self.snapshot_root.iterdir()), [])
        server.shutdown()

    def test_source_state_feedback_publishes_stage_and_stops_after_terminal(self):
        handle = FakeGoalHandle("source-state")
        record = _CaptureRecord(handle, ("source-state", 1, 1))
        record.session = mock.Mock()
        record.session.writer.frame_count = 3
        with self.server._lock:
            self.server._active = record
        try:
            self.server._publish_source_state(
                record, "SOURCE_STALE", FrameStamp(10, 0)
            )
            self.assertEqual(handle.feedbacks[-1].stage, "SOURCE_STALE")
            self.assertEqual(handle.feedbacks[-1].frame_count, 3)
            self.server._publish_source_state(
                record, "SOURCE_RECOVERED", FrameStamp(11, 0)
            )
            self.assertEqual(handle.feedbacks[-1].stage, "SOURCE_RECOVERED")
            count = len(handle.feedbacks)
            record.terminal_claimed = True
            self.server._publish_source_state(
                record, "SOURCE_STALE", FrameStamp(12, 0)
            )
            self.assertEqual(len(handle.feedbacks), count)
        finally:
            with self.server._lock:
                self.server._active = None

    def test_diagnostics_snapshot_exposes_stage_and_input_counters(self):
        goal = FakeGoalHandle("diag")
        self.server._goal_callback(goal)
        self.wait_events(goal, 1)
        deadline = time.monotonic() + 2.0
        while (
            self.server.diagnostics_snapshot().get("stage") != "active"
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        self.assertEqual(self.server.diagnostics_snapshot()["stage"], "active")
        self.image_callback()(self.image_message(10, 0))
        snapshot = self.server.diagnostics_snapshot()
        self.assertTrue(snapshot["running"])
        self.assertEqual(snapshot["stage"], "active")
        self.assertEqual(snapshot["session"]["selected_frames"], 1)
        self.assertEqual(snapshot["latest_image_stamp"], (10, 0))
        self.image_callback()(self.image_message(10, 500000000))
        self.wait_events(goal, 2, timeout=3.0)
        snapshot = self.server.diagnostics_snapshot()
        self.assertFalse(snapshot["running"])
        self.assertEqual(snapshot["last_terminal"]["method"], "succeeded")


    def tf_static_message(self, parent="body", child="camera4_optical_frame"):
        return SimpleNamespace(
            transforms=[
                SimpleNamespace(
                    header=SimpleNamespace(
                        stamp=SimpleNamespace(sec=0, nsec=0),
                        frame_id=parent,
                    ),
                    child_frame_id=child,
                    transform=SimpleNamespace(
                        translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    ),
                )
            ]
        )

    def test_tf_static_latched_messages_from_all_publishers_are_replayed(self):
        config = replace(self.config, tf_static_topic="/tf_static")
        server = CaptureActionServer(
            action_name="/test_capture_tf_static_replay",
            config=config,
            action_server_factory=FakeActionServer,
            subscriber_factory=self.subscriber_factory,
        )
        try:
            static_callback = [
                subscriber.callback
                for subscriber in self.subscribers
                if subscriber.topic == "/tf_static"
            ][0]
            # 没有活跃 session 时收到 latched static transform：必须缓存并在
            # 下一个 session 接受后重放，否则 /tf_static 要等下一次 latch。
            # An intermediate static edge must survive before body/camera
            # neighbors arrive from other latched publishers.
            static_callback(self.tf_static_message(parent="rig", child="mount"))
            static_callback(self.tf_static_message())
            static_callback(self.tf_static_message(parent="odom", child="body"))
            goal = FakeGoalHandle("tf-static-replay")
            server._goal_callback(goal)
            self.wait_events(goal, 1)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                with server._lock:
                    record = server._active
                    session = record.session if record is not None else None
                if session is not None and session.tf_received >= 3:
                    break
                if session is not None and session.done.is_set():
                    break
                time.sleep(0.005)
            with server._lock:
                record = server._active
                session = record.session if record is not None else None
            self.assertIsNotNone(session)
            self.assertEqual(session.tf_received, 3)
            self.assertTrue(session.writer.has_tf)
            self.assertEqual(session.snapshot()["tf_received"], 3)
        finally:
            server.shutdown()

    def test_tf_static_publishers_sharing_a_child_frame_are_both_kept(self):
        # RealSense publishes camera_aligned_depth_to_color_frame -> optical frame;
        # the rig publishes body -> optical frame. Neither may overwrite the other.
        config = replace(self.config, tf_static_topic="/tf_static")
        server = CaptureActionServer(
            action_name="/test_capture_tf_static_shared_child",
            config=config,
            action_server_factory=FakeActionServer,
            subscriber_factory=self.subscriber_factory,
        )
        try:
            static_callback = [
                subscriber.callback
                for subscriber in self.subscribers
                if subscriber.topic == "/tf_static"
            ][0]
            static_callback(self.tf_static_message())
            static_callback(self.tf_static_message(parent="camera_aligned_depth", child="camera4_optical_frame"))
            self.assertEqual(set(server._tf_static_transforms),
                             {("body", "camera4_optical_frame"),
                              ("camera_aligned_depth", "camera4_optical_frame")})
        finally:
            server.shutdown()

    def test_malformed_tf_static_update_does_not_poison_replay_cache(self):
        config = replace(self.config, tf_static_topic="/tf_static")
        server = CaptureActionServer(
            action_name="/test_capture_tf_static_valid_cache",
            config=config,
            action_server_factory=FakeActionServer,
            subscriber_factory=self.subscriber_factory,
        )
        try:
            static_callback = [
                subscriber.callback
                for subscriber in self.subscribers
                if subscriber.topic == "/tf_static"
            ][0]
            static_callback(self.tf_static_message())
            malformed = self.tf_static_message()
            malformed.transforms[0].transform.translation.x = float("nan")
            static_callback(malformed)

            cached = server._tf_static_transforms[("body", "camera4_optical_frame")]
            self.assertEqual(cached.transform.translation.x, 1.0)
            goal = FakeGoalHandle("tf-static-valid-cache")
            server._goal_callback(goal)
            self.wait_events(goal, 1)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                with server._lock:
                    session = (
                        server._active.session
                        if server._active is not None
                        else None
                    )
                if session is not None and session.tf_received == 1:
                    break
                time.sleep(0.005)
            with server._lock:
                session = server._active.session
            self.assertEqual(session.tf_received, 1)
            self.assertTrue(session.writer.has_tf)
            self.assertEqual(session.writer.auxiliary_errors, 0)
        finally:
            server.shutdown()

    def test_malformed_tf_static_sibling_does_not_drop_valid_active_transform(self):
        config = replace(self.config, tf_static_topic="/tf_static")
        server = CaptureActionServer(
            action_name="/test_capture_tf_static_mixed_active",
            config=config,
            action_server_factory=FakeActionServer,
            subscriber_factory=self.subscriber_factory,
        )
        try:
            static_callback = [
                subscriber.callback
                for subscriber in self.subscribers
                if subscriber.topic == "/tf_static"
            ][0]
            goal = FakeGoalHandle("tf-static-mixed-active")
            server._goal_callback(goal)
            self.wait_events(goal, 1)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                with server._lock:
                    session = (
                        server._active.session
                        if server._active is not None
                        else None
                    )
                if session is not None:
                    break
                time.sleep(0.005)

            valid = self.tf_static_message().transforms[0]
            malformed = self.tf_static_message(
                parent="body", child="camera4_bad_optical_frame"
            ).transforms[0]
            malformed.transform.translation.x = float("nan")
            static_callback(SimpleNamespace(transforms=[valid, malformed]))

            self.assertEqual(session.tf_received, 1)
            self.assertTrue(session.writer.has_tf)
            self.assertEqual(session.writer.auxiliary_errors, 1)
        finally:
            server.shutdown()

    def test_tf_static_replay_finishes_before_session_is_visible(self):
        config = replace(self.config, tf_static_topic="/tf_static")
        replay_started = threading.Event()
        release_replay = threading.Event()
        sessions = []

        def session_factory(config, goal, transaction, cancellation_event, **kwargs):
            session = CaptureSession(
                config, goal, transaction, cancellation_event, **kwargs
            )
            original_on_tf = session.on_tf

            def blocking_on_tf(message, static=False):
                replay_started.set()
                release_replay.wait(3.0)
                original_on_tf(message, static=static)

            session.on_tf = blocking_on_tf
            sessions.append(session)
            return session

        local_subscribers = []

        def subscriber_factory(topic, message_type, callback, queue_size, buff_size=None):
            subscriber = FakeSubscriber(
                topic, message_type, callback, queue_size, buff_size
            )
            local_subscribers.append(subscriber)
            return subscriber

        server = CaptureActionServer(
            action_name="/test_capture_tf_static_attach_order",
            config=config,
            action_server_factory=FakeActionServer,
            subscriber_factory=subscriber_factory,
            session_factory=session_factory,
        )
        image_thread = None
        try:
            callbacks = {item.topic: item.callback for item in local_subscribers}
            callbacks["/tf_static"](self.tf_static_message())
            goal = FakeGoalHandle(
                "tf-static-attach-order",
                goal=SimpleNamespace(
                    scene_id="scene-capture",
                    sample_fps=2.0,
                    max_frames=1,
                    max_duration=SimpleNamespace(to_sec=lambda: 5.0),
                ),
            )
            server._goal_callback(goal)
            self.assertTrue(replay_started.wait(2.0))
            image_done = threading.Event()

            def send_image():
                callbacks[config.image_topic](self.image_message(10, 0))
                image_done.set()

            image_thread = threading.Thread(target=send_image)
            image_thread.start()
            self.assertFalse(image_done.wait(0.5))
            self.assertEqual(sessions[0].writer.frame_count, 0)
            release_replay.set()
            image_thread.join(2.0)
            self.assertTrue(image_done.is_set())
            self.wait_events(goal, 2, timeout=3.0)
            self.assertTrue(sessions[0].writer.has_tf)
        finally:
            release_replay.set()
            if image_thread is not None:
                image_thread.join(2.0)
            server.shutdown()

    def test_auxiliary_subscriptions_follow_profile(self):
        config = replace(
            self.config,
            depth_topic="/depth",
            save_sensor_depth=True,
            tf_topic="/tf",
            tf_static_topic="/tf_static",
        )
        # make_server 使用 self.config；这里直接构造新配置的服务端
        server = CaptureActionServer(
            action_name="/test_capture_aux",
            config=config,
            action_server_factory=FakeActionServer,
            subscriber_factory=self.subscriber_factory,
        )
        try:
            topics = {subscriber.topic for subscriber in self.subscribers}
            self.assertIn("/camera4/image_raw", topics)
            self.assertIn("/Odometry", topics)
            self.assertIn("/depth", topics)
            self.assertIn("/tf", topics)
            self.assertIn("/tf_static", topics)
        finally:
            server.shutdown()

        config_no_depth = replace(
            self.config,
            depth_topic="/depth",
            save_sensor_depth=False,
        )
        local_subscribers = []

        def local_subscriber_factory(topic, message_type, callback, queue_size, buff_size=None):
            subscriber = self.subscriber_factory(
                topic, message_type, callback, queue_size, buff_size
            )
            local_subscribers.append(subscriber)
            return subscriber

        server = CaptureActionServer(
            action_name="/test_capture_aux_off",
            config=config_no_depth,
            action_server_factory=FakeActionServer,
            subscriber_factory=local_subscriber_factory,
        )
        try:
            # Depth subscription must follow depth_topic, not the initial
            # save flag: a later hot-tuned window may enable depth saving
            # without restarting the node.
            self.assertIn("/depth", {subscriber.topic for subscriber in local_subscribers})
        finally:
            server.shutdown()


    def test_capture_goal_save_sensor_depth_override_reaches_session(self):
        config = replace(
            self.config, depth_topic="/depth", save_sensor_depth=False
        )
        seen = []

        def session_factory(config, goal, transaction, cancellation_event, **kwargs):
            seen.append(config)
            return self.server._session_factory(
                config,
                goal,
                transaction,
                cancellation_event,
                **kwargs,
            )

        local_subscribers = []

        def local_subscriber_factory(topic, message_type, callback, queue_size, buff_size=None):
            subscriber = self.subscriber_factory(
                topic, message_type, callback, queue_size, buff_size
            )
            local_subscribers.append(subscriber)
            return subscriber

        server = CaptureActionServer(
            action_name="/test_capture_override",
            config=config,
            action_server_factory=FakeActionServer,
            subscriber_factory=local_subscriber_factory,
            session_factory=session_factory,
        )
        try:
            goal_message = SimpleNamespace(
                scene_id="scene-capture",
                sample_fps=2.0,
                max_frames=2,
                max_duration=SimpleNamespace(to_sec=lambda: 5.0),
                save_sensor_depth=True,
                set_save_sensor_depth=True,
            )
            handle = FakeGoalHandle("override", goal=goal_message)
            server._goal_callback(handle)
            self.wait_events(handle, 1, timeout=3.0)
            image_cb = [
                subscriber.callback
                for subscriber in local_subscribers
                if subscriber.topic == "/camera4/image_raw"
            ][0]
            image_cb(self.image_message(10, 0))
            image_cb(self.image_message(10, 500000000))
            self.wait_events(handle, 2, timeout=3.0)
            self.assertEqual(handle.events[-1][0], "succeeded")
            self.assertEqual(len(seen), 1)
            self.assertTrue(seen[0].save_sensor_depth)
            self.assertFalse(config.save_sensor_depth)
        finally:
            server.shutdown()

    def test_capture_goal_override_without_depth_topic_rejected(self):
        goal_message = SimpleNamespace(
            scene_id="scene-capture",
            sample_fps=2.0,
            max_frames=2,
            max_duration=SimpleNamespace(to_sec=lambda: 5.0),
            save_sensor_depth=True,
            set_save_sensor_depth=True,
        )
        handle = FakeGoalHandle("override-no-depth", goal=goal_message)
        self.server._goal_callback(handle)
        self.wait_events(handle, 1, timeout=3.0)
        self.assertEqual(handle.events[0][0], "rejected")
        self.assertEqual(
            handle.events[0][1].error_code, CaptureSegmentResult.INVALID_GOAL
        )

    def test_capture_goal_set_save_sensor_depth_must_be_bool(self):
        goal_message = SimpleNamespace(
            scene_id="scene-capture",
            sample_fps=2.0,
            max_frames=2,
            max_duration=SimpleNamespace(to_sec=lambda: 5.0),
            save_sensor_depth=True,
            set_save_sensor_depth="yes",
        )
        handle = FakeGoalHandle("override-bad-flag", goal=goal_message)
        self.server._goal_callback(handle)
        self.wait_events(handle, 1, timeout=3.0)
        self.assertEqual(handle.events[0][0], "rejected")
        self.assertEqual(
            handle.events[0][1].error_code, CaptureSegmentResult.INVALID_GOAL
        )


if __name__ == "__main__":
    unittest.main()
