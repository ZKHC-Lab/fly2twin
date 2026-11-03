"""CPU-only tests for the camera pipeline upper-layer service core."""

from pathlib import Path
import tempfile
import threading
import time
import unittest

from drone3d_reconstruction_core.capture import CaptureConfig
from drone3d_reconstruction_core.snapshot_validator import SnapshotSafetyLimits
from drone3d_reconstruction_core.pipeline_service import (
    ACTION_CANCELLED,
    ACTION_FAILED,
    ACTION_SUCCEEDED,
    BUSY,
    INVALID_ARGUMENT,
    INTERNAL_ERROR,
    OK,
    STATE_CAPTURING,
    STATE_CANCELLED,
    STATE_FAILED,
    STATE_IDLE,
    STATE_STOPPING,
    STATE_SUCCEEDED,
    UNKNOWN_ROLE,
    CameraPipelineService,
    CameraRole,
    CaptureActionClient,
    CaptureActionOutcome,
    PipelineConfig,
    ReconstructActionClient,
    ReconstructActionOutcome,
)


def make_capture_config(root: Path, role: str) -> CaptureConfig:
    snapshots = root / f"{role}_snapshots"
    snapshots.mkdir()
    return CaptureConfig(
        snapshot_root=str(snapshots),
        image_topic=f"/{role}/image_raw",
        camera_name=f"{role}_camera",
        camera_role=role,
        camera_frame_id=f"{role}_camera_optical_frame",
        image_encoding="bgr8",
        odometry_topic=None,
        camera_info_topic=None,
        depth_topic=None,
        tf_topic=None,
        tf_static_topic=None,
        save_sensor_depth=False,
        body_frame_id=f"{role}_body",
        tf_relevant_frames=(),
        camera_info_max_delta_ms=100.0,
        depth_max_delta_ms=100.0,
        max_auxiliary_samples=8,
        max_tf_samples=64,
        max_tf_bytes=256 * 1024,
        source_timeout_sec=5.0,
        max_window_gap_sec=30.0,
        min_sample_fps=0.5,
        max_sample_fps=10.0,
        max_goal_frames=16,
        max_goal_duration_sec=30.0,
        min_captured_frames=1,
        max_image_width=640,
        max_image_height=480,
        pose_max_delta_ms=50.0,
        pose_gap_max_ms=200.0,
        max_sample_interval_overshoot_ms=250.0,
        snapshot_limits=SnapshotSafetyLimits(
            max_frame_count=16,
            max_manifest_entries=128,
            max_metadata_bytes=1024 * 1024,
            max_file_bytes=1024 * 1024,
            max_total_bytes=64 * 1024 * 1024,
            max_scene_id_bytes=128,
            max_relative_path_bytes=1024,
        ),
    )


class FakeCaptureClient(CaptureActionClient):
    def __init__(self, outcome=None, server_present=True):
        self.goal_sent = threading.Event()
        self.feedback = []
        self._ready = threading.Event()
        self._outcome = outcome or CaptureActionOutcome(
            action_state=ACTION_SUCCEEDED,
            error_code=0,
            message="ok",
            scene_id="scene-front-01",
            camera_name="front_camera",
            image_topic="/front/image_raw",
            snapshot_uri="file:///tmp/front/snapshot",
            manifest_uri="file:///tmp/front/snapshot/files_sha256.csv",
            manifest_sha256="a" * 64,
            frame_count=2,
        )
        self.server_present = server_present
        self.cancel_calls = 0

    def wait_for_server(self, timeout_sec):
        return self.server_present

    def send_goal(self, params, feedback_cb):
        self.goal_sent.set()
        if feedback_cb is not None:
            feedback_cb(type("Feedback", (), {"frame_count": 2})())
        self._ready.set()

    def wait_for_result(self, timeout_sec):
        return self._ready.wait(timeout_sec)

    def cancel_goal(self):
        self.cancel_calls += 1
        self._ready.set()

    def outcome(self):
        return self._outcome


class BlockingCaptureClient(FakeCaptureClient):
    def __init__(self):
        super().__init__()
        self._ready = threading.Event()

    def send_goal(self, params, feedback_cb):
        self.goal_sent.set()
        if feedback_cb is not None:
            feedback_cb(type("Feedback", (), {"frame_count": 1})())


class IgnoreCancelCaptureClient(BlockingCaptureClient):
    """下层 Action 收到 cancel 后不立即收敛（复现复核提出的竞态）。"""

    def __init__(self):
        super().__init__()
        self.cancel_calls = 0

    def cancel_goal(self):
        self.cancel_calls += 1
        # 不设置 _ready：模拟下层仍在 TERM/KILL/validator 收敛。

    def release_terminal(self):
        self._ready.set()


class WatchdogCaptureClient(IgnoreCancelCaptureClient):
    """取消前永远不收敛，取消后由测试显式释放终态的下层 Action。"""

    def wait_for_result(self, timeout_sec):
        if self.cancel_calls == 0:
            return False
        return self._ready.wait(timeout_sec)


class FakeReconstructClient(ReconstructActionClient):
    def __init__(self, outcome=None, server_present=True):
        self.goal_sent = threading.Event()
        self.feedback = []
        self._ready = threading.Event()
        self._outcome = outcome or ReconstructActionOutcome(
            action_state=ACTION_SUCCEEDED,
            error_code=0,
            message="ok",
            scene_id="scene-front-01",
            result_manifest_uri="file:///tmp/front/result.json",
            result_manifest_sha256="b" * 64,
            scene_glb_uri="",
            frame_count=2,
            wall_time_sec=1.5,
            camera_role="front",
            camera_topic="/front/image_raw",
            camera_frame_id="front_camera_optical_frame",
            snapshot_manifest_sha256="a" * 64,
            result_scene_id="scene-front-01",
            result_actual_frame_count=2,
        )
        self.server_present = server_present
        self.cancel_calls = 0

    def wait_for_server(self, timeout_sec):
        return self.server_present

    def send_goal(
        self,
        snapshot_uri,
        manifest_sha256,
        feedback_cb,
        image_resolution=0,
        set_image_resolution=False,
        worker_timeout_sec=0.0,
        set_worker_timeout=False,
    ):
        self.snapshot_uri = snapshot_uri
        self.manifest_sha256 = manifest_sha256
        self.goal_sent.set()
        if feedback_cb is not None:
            feedback_cb(
                type("Feedback", (), {"stage": "predicting", "progress": 0.5})()
            )
        self._ready.set()

    def wait_for_result(self, timeout_sec):
        return self._ready.wait(timeout_sec)

    def cancel_goal(self):
        self.cancel_calls += 1
        self._ready.set()

    def outcome(self):
        return self._outcome


class BlockingReconstructClient(FakeReconstructClient):
    def __init__(self):
        super().__init__()
        self._ready = threading.Event()


class CameraPipelineServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        front_role = CameraRole(
            role="front",
            capture_action="/front/capture_segment",
            capture_config=make_capture_config(root, "front"),
        )
        down_role = CameraRole(
            role="down",
            capture_action="/down/capture_segment",
            capture_config=make_capture_config(root, "down"),
        )
        self.config = PipelineConfig(
            roles=(front_role, down_role),
            reconstruct_action="/reconstruct/reconstruct_segment",
            service_name="/camera_pipeline/command",
            diagnostics_topic="/diagnostics",
            diagnostics_period_sec=1.0,
            action_connect_timeout_sec=1.0,
            capture_wait_margin_sec=1.0,
            reconstruct_wait_timeout_sec=5.0,
            reconstruct_frame_group_size=8,
        )
        self.capture_clients = []
        self.reconstruct_clients = []

    def make_service(
        self,
        capture_factory=None,
        reconstruct_factory=None,
        config=None,
    ):
        self.service = CameraPipelineService(
            config or self.config,
            capture_client_factory=capture_factory
            or self._capture_factory,
            reconstruct_client_factory=reconstruct_factory
            or self._reconstruct_factory,
            thread_factory=threading.Thread,
        )
        return self.service

    def _capture_factory(self, role):
        client = FakeCaptureClient()
        self.capture_clients.append(client)
        return client

    def _reconstruct_factory(self):
        client = FakeReconstructClient()
        self.reconstruct_clients.append(client)
        return client

    def wait_state(self, state, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.service.get_status()
            if status.state == state:
                return status
            time.sleep(0.01)
        raise AssertionError(f"state {state!r} not reached; got {status.state!r}")

    def test_select_source_and_unknown_role(self):
        service = self.make_service()
        outcome = service.select_source("down")
        self.assertEqual(outcome.error_code, OK)
        self.assertEqual(outcome.role, "down")
        self.assertEqual(outcome.state, STATE_IDLE)
        unknown = service.select_source("side")
        self.assertEqual(unknown.error_code, UNKNOWN_ROLE)

    def test_start_task_runs_capture_then_reconstruct(self):
        service = self.make_service()
        service.select_source("front")
        outcome = service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        self.assertEqual(outcome.error_code, OK)
        self.assertIn(outcome.state, (STATE_CAPTURING, STATE_SUCCEEDED))
        status = self.wait_state(STATE_SUCCEEDED, timeout=3.0)
        self.assertEqual(status.captured_frames, 2)
        self.assertEqual(status.reconstruction_frames, 2)
        self.assertEqual(status.snapshot_uri, "file:///tmp/front/snapshot")
        self.assertEqual(status.result_manifest_uri, "file:///tmp/front/result.json")
        self.assertTrue(self.reconstruct_clients[0].goal_sent.is_set())
        self.assertEqual(
            self.reconstruct_clients[0].snapshot_uri, "file:///tmp/front/snapshot"
        )
        self.assertEqual(self.reconstruct_clients[0].manifest_sha256, "a" * 64)

    def test_second_start_is_busy_while_running(self):
        capture = BlockingCaptureClient()
        service = self.make_service(
            capture_factory=lambda role: capture,
            reconstruct_factory=self._reconstruct_factory,
        )
        service.select_source("front")
        first = service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        self.assertEqual(first.error_code, OK)
        self.assertTrue(capture.goal_sent.wait(1.0))
        second = service.start_task("down", "scene-down-01", 1.0, 2, 30.0)
        self.assertEqual(second.error_code, BUSY)
        service.cancel_task()
        self.wait_state(STATE_CANCELLED, timeout=3.0)
        # After a terminal state the slot is free again.
        third = service.start_task("down", "scene-down-02", 1.0, 2, 30.0)
        self.assertEqual(third.error_code, OK)

    def test_invalid_start_arguments_are_rejected(self):
        service = self.make_service()
        service.select_source("front")
        for args in (
            ("", "bad scene", 1.0, 2, 30.0),
            ("", "scene-front-01", 0.0, 2, 30.0),
            ("", "scene-front-01", 1.0, 99, 30.0),
            ("", "scene-front-01", 1.0, 2, -1.0),
        ):
            with self.subTest(args=args):
                outcome = service.start_task(*args)
                self.assertEqual(outcome.error_code, INVALID_ARGUMENT)

    def test_capture_failure_marks_failed_and_keeps_snapshot_metadata(self):
        failed = FakeCaptureClient(
            outcome=CaptureActionOutcome(
                action_state=ACTION_FAILED,
                error_code=2,
                message="camera unavailable",
                scene_id="scene-front-01",
                camera_name="front_camera",
                image_topic="/front/image_raw",
                snapshot_uri="",
                manifest_uri="",
                manifest_sha256="",
                frame_count=0,
            )
        )
        service = self.make_service(capture_factory=lambda role: failed)
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        status = self.wait_state(STATE_FAILED)
        self.assertIn("camera unavailable", status.last_error)
        self.assertEqual(len(self.reconstruct_clients), 0)

    def test_reconstruct_failure_marks_failed_with_result_error(self):
        failed = FakeReconstructClient(
            outcome=ReconstructActionOutcome(
                action_state=ACTION_FAILED,
                error_code=3,
                message="another reconstruction worker is active",
                scene_id="scene-front-01",
                result_manifest_uri="",
                result_manifest_sha256="",
                scene_glb_uri="",
                frame_count=0,
                wall_time_sec=0.1,
            )
        )
        service = self.make_service(
            reconstruct_factory=lambda: failed,
        )
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        status = self.wait_state(STATE_FAILED)
        self.assertIn("another reconstruction worker is active", status.last_error)

    def test_cancel_during_capture_sets_cancelled(self):
        capture = BlockingCaptureClient()
        capture._outcome = CaptureActionOutcome(
            action_state=ACTION_CANCELLED,
            error_code=4,
            message="capture cancelled",
            scene_id="scene-front-01",
            camera_name="front_camera",
            image_topic="/front/image_raw",
            snapshot_uri="",
            manifest_uri="",
            manifest_sha256="",
            frame_count=0,
        )
        service = self.make_service(capture_factory=lambda role: capture)
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        self.assertTrue(capture.goal_sent.wait(1.0))
        service.cancel_task()
        status = self.wait_state(STATE_CANCELLED, timeout=3.0)
        self.assertGreater(capture.cancel_calls, 0)
        self.assertEqual(status.state, STATE_CANCELLED)

    def test_cancel_holds_slot_until_lower_action_terminal(self):
        capture = IgnoreCancelCaptureClient()
        service = self.make_service(capture_factory=lambda role: capture)
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        self.assertTrue(capture.goal_sent.wait(1.0))
        outcome = service.cancel_task()
        self.assertEqual(outcome.error_code, OK)
        stopping = self.wait_state(STATE_STOPPING, timeout=3.0)
        self.assertIn("waiting for lower action", stopping.last_error)
        # 下层尚未收敛：上层绝不释放槽位。
        second = service.start_task("down", "scene-down-01", 1.0, 2, 30.0)
        self.assertEqual(second.error_code, BUSY)
        self.assertEqual(capture.cancel_calls, 1)
        # 下层给出终态后才允许新任务。
        capture.release_terminal()
        self.wait_state(STATE_CANCELLED, timeout=3.0)
        third = service.start_task("down", "scene-down-02", 1.0, 2, 30.0)
        self.assertEqual(third.error_code, OK)

    def test_watchdog_holds_slot_until_lower_action_terminal(self):
        capture = WatchdogCaptureClient()
        short_config = PipelineConfig(
            roles=self.config.roles,
            reconstruct_action=self.config.reconstruct_action,
            service_name=self.config.service_name,
            diagnostics_topic=self.config.diagnostics_topic,
            diagnostics_period_sec=self.config.diagnostics_period_sec,
            action_connect_timeout_sec=self.config.action_connect_timeout_sec,
            capture_wait_margin_sec=0.2,
            reconstruct_wait_timeout_sec=self.config.reconstruct_wait_timeout_sec,
            reconstruct_frame_group_size=self.config.reconstruct_frame_group_size,
        )
        service = self.make_service(
            capture_factory=lambda role: capture,
            reconstruct_factory=self._reconstruct_factory,
            config=short_config,
        )
        # 目标时长 0.2s + 0.2s 余量，上层 watchdog 约 0.4s 后触发。
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 0.2)
        self.assertTrue(capture.goal_sent.wait(1.0))
        stopping = self.wait_state(STATE_STOPPING, timeout=3.0)
        self.assertIn("upper service timeout", stopping.last_error)
        second = service.start_task("down", "scene-down-01", 1.0, 2, 30.0)
        self.assertEqual(second.error_code, BUSY)
        capture.release_terminal()
        failed = self.wait_state(STATE_FAILED, timeout=3.0)
        self.assertIn("upper service timeout", failed.last_error)
        third = service.start_task("down", "scene-down-02", 1.0, 2, 30.0)
        self.assertEqual(third.error_code, OK)

    def test_reconstruct_success_with_empty_result_metadata_is_failed(self):
        empty = FakeReconstructClient(
            outcome=ReconstructActionOutcome(
                action_state=ACTION_SUCCEEDED,
                error_code=0,
                message="ok",
                scene_id="scene-front-01",
                result_manifest_uri="",
                result_manifest_sha256="",
                scene_glb_uri="",
                frame_count=0,
                wall_time_sec=0.1,
            )
        )
        service = self.make_service(reconstruct_factory=lambda: empty)
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        status = self.wait_state(STATE_FAILED)
        self.assertIn("without valid result metadata", status.last_error)

    def test_cancel_after_capture_before_reconstruct_goal_converges(self):
        """复核竞态：capture 已终态、reconstruct goal 尚未建立时取消不得永久 STOPPING。"""
        service_holder = {}

        def capture_factory(role):
            return FakeCaptureClient()

        def reconstruct_factory():
            # 在 reconstruct client 尚未登记到 record 的窗口内请求取消。
            service_holder["service"].cancel_task()
            return FakeReconstructClient()

        service = self.make_service(
            capture_factory=capture_factory,
            reconstruct_factory=reconstruct_factory,
        )
        service_holder["service"] = service
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        final = self.wait_state(STATE_CANCELLED, timeout=3.0)
        self.assertEqual(final.state, STATE_CANCELLED)
        self.assertFalse(service.get_status().state == STATE_STOPPING)
        next_start = service.start_task("down", "scene-down-02", 1.0, 2, 30.0)
        self.assertEqual(next_start.error_code, OK)

    def test_capture_factory_failure_converges_and_releases_slot(self):
        def broken_factory(role):
            raise RuntimeError("capture client factory broken")

        service = self.make_service(capture_factory=broken_factory)
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        self.wait_state(STATE_FAILED, timeout=3.0)
        self.assertTrue(service.wait_for_task_terminal(2.0))
        again = service.start_task("down", "scene-down-02", 1.0, 2, 30.0)
        self.assertEqual(again.error_code, OK)

    def test_reconstruct_factory_failure_converges_and_releases_slot(self):
        def broken_factory():
            raise RuntimeError("reconstruct client factory broken")

        service = self.make_service(
            reconstruct_factory=broken_factory,
        )
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        self.wait_state(STATE_FAILED, timeout=3.0)
        self.assertTrue(service.wait_for_task_terminal(2.0))
        again = service.start_task("down", "scene-down-02", 1.0, 2, 30.0)
        self.assertEqual(again.error_code, OK)

    def test_thread_dispatch_failure_returns_internal_error_and_releases_slot(self):
        attempts = []

        def flaky_thread_factory(target, args):
            if not attempts:
                attempts.append(1)
                return object()  # not a Thread, first dispatch fails
            thread = threading.Thread(target=target, args=args)
            thread.daemon = True
            return thread

        service = CameraPipelineService(
            self.config,
            capture_client_factory=self._capture_factory,
            reconstruct_client_factory=self._reconstruct_factory,
            thread_factory=flaky_thread_factory,
        )
        service.select_source("front")
        outcome = service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        self.assertEqual(outcome.error_code, INTERNAL_ERROR)
        self.assertTrue(service.wait_for_task_terminal(2.0))
        again = service.start_task("down", "scene-down-02", 1.0, 2, 30.0)
        self.assertEqual(again.error_code, OK)

    def test_shutdown_rejects_new_start_and_select(self):
        service = self.make_service()
        service.select_source("front")
        service.begin_shutdown()
        select = service.select_source("down")
        self.assertEqual(select.error_code, INTERNAL_ERROR)
        start = service.start_task("down", "scene-down-02", 1.0, 2, 30.0)
        self.assertEqual(start.error_code, INTERNAL_ERROR)
        # GET_STATUS/CANCEL 仍可用。
        self.assertEqual(service.get_status().error_code, OK)
        self.assertEqual(service.cancel_task().error_code, OK)

    def test_capture_identity_mismatch_is_rejected(self):
        wrong = FakeCaptureClient(
            outcome=CaptureActionOutcome(
                action_state=ACTION_SUCCEEDED,
                error_code=0,
                message="ok",
                scene_id="other-scene",
                camera_name="front_camera",
                image_topic="/front/image_raw",
                snapshot_uri="file:///tmp/front/snapshot",
                manifest_uri="file:///tmp/front/snapshot/files_sha256.csv",
                manifest_sha256="a" * 64,
                frame_count=2,
            )
        )
        service = self.make_service(capture_factory=lambda role: wrong)
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        status = self.wait_state(STATE_FAILED)
        self.assertIn("capture identity check failed", status.last_error)
        self.assertIn("scene_id mismatch", status.last_error)

    def test_reconstruct_identity_mismatch_is_rejected(self):
        wrong = FakeReconstructClient(
            outcome=ReconstructActionOutcome(
                action_state=ACTION_SUCCEEDED,
                error_code=0,
                message="ok",
                scene_id="scene-front-01",
                result_manifest_uri="file:///tmp/front/result.json",
                result_manifest_sha256="b" * 64,
                scene_glb_uri="",
                frame_count=2,
                wall_time_sec=1.5,
                camera_role="down",
                camera_topic="/front/image_raw",
                camera_frame_id="front_camera_optical_frame",
                snapshot_manifest_sha256="a" * 64,
                result_scene_id="scene-front-01",
                result_actual_frame_count=2,
            )
        )
        service = self.make_service(reconstruct_factory=lambda: wrong)
        service.select_source("front")
        service.start_task("", "scene-front-01", 1.0, 2, 30.0)
        status = self.wait_state(STATE_FAILED)
        self.assertIn("reconstruct identity check failed", status.last_error)
        self.assertIn("camera_role mismatch", status.last_error)

    def test_frames_above_reconstruct_group_are_rejected(self):
        service = self.make_service()
        service.select_source("front")
        outcome = service.start_task("", "scene-front-01", 1.0, 9, 30.0)
        self.assertEqual(outcome.error_code, INVALID_ARGUMENT)
        self.assertIn("frame_group_size", outcome.message)

    def test_diagnostics_snapshot_reports_roles_and_selected(self):
        service = self.make_service()
        service.select_source("down")
        snapshot = service.diagnostics_snapshot()
        self.assertEqual(snapshot["selected_role"], "down")
        self.assertEqual(
            {role["role"] for role in snapshot["roles"]}, {"front", "down"}
        )
        self.assertEqual(snapshot["task"]["state"], STATE_IDLE)


if __name__ == "__main__":
    unittest.main()
