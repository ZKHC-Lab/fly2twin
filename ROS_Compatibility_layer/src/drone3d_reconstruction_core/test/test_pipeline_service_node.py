"""CPU-only tests for the camera pipeline service node boundary."""

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from diagnostic_msgs.msg import DiagnosticStatus
from drone3d_reconstruction_msgs.srv import PipelineCommandRequest

from drone3d_reconstruction_core.pipeline_service import (
    STATE_IDLE,
)
from drone3d_reconstruction_core.pipeline_service_node import (
    PipelineServiceNode,
    assemble_pipeline_diagnostics,
    load_pipeline_config,
    read_result_identity,
)


class FakeService:
    def __init__(self, name, srv_type, callback):
        self.name = name
        self.srv_type = srv_type
        self.callback = callback
        self.shutdown_calls = []

    def shutdown(self, reason=""):
        self.shutdown_calls.append(reason)


class FakePublisher:
    def __init__(self, *args, **kwargs):
        self.messages = []
        self.unregistered = False

    def publish(self, message):
        self.messages.append(message)

    def unregister(self):
        self.unregistered = True


class FakeTimer:
    def __init__(self, *args, **kwargs):
        pass

    def shutdown(self):
        pass


class NeverCaptureClient:
    """下层 Capture Action 永不收敛的测试替身（仅用于关停超时清理测试）。"""

    def wait_for_server(self, timeout_sec):
        return True

    def send_goal(self, params, feedback_cb):
        pass

    def wait_for_result(self, timeout_sec):
        return False

    def cancel_goal(self):
        pass

    def outcome(self):
        raise AssertionError("not expected to be called")


class PipelineServiceNodeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _capture_entry(self, role):
        snapshots = self.root / f"{role}_snapshots"
        snapshots.mkdir(exist_ok=True)
        return {
            "snapshot_root": str(snapshots),
            "image_topic": f"/{role}/image_raw",
            "camera_name": f"{role}_camera",
            "camera_role": role,
            "camera_frame_id": f"{role}_camera_optical_frame",
            "image_encoding": "bgr8",
            "odometry_topic": "",
            "camera_info_topic": "",
            "depth_topic": "",
            "tf_topic": "",
            "tf_static_topic": "",
            "save_sensor_depth": False,
            "body_frame_id": "",
            "tf_relevant_frames": [],
            "camera_info_max_delta_ms": 100.0,
            "depth_max_delta_ms": 100.0,
            "max_auxiliary_samples": 8,
            "max_tf_samples": 64,
            "max_tf_bytes": 256 * 1024,
            "source_timeout_sec": 5.0,
            "max_window_gap_sec": 30.0,
            "min_sample_fps": 0.5,
            "max_sample_fps": 10.0,
            "max_goal_frames": 8,
            "max_goal_duration_sec": 30.0,
            "min_captured_frames": 1,
            "max_image_width": 640,
            "max_image_height": 480,
            "pose_max_delta_ms": 50.0,
            "pose_gap_max_ms": 200.0,
            "max_sample_interval_overshoot_ms": 250.0,
            "snapshot_limits": {
                "max_frame_count": 8,
                "max_manifest_entries": 128,
                "max_metadata_bytes": 1024 * 1024,
                "max_file_bytes": 1024 * 1024,
                "max_total_bytes": 64 * 1024 * 1024,
                "max_scene_id_bytes": 128,
                "max_relative_path_bytes": 1024,
            },
            "action_name": f"/{role}/capture_segment",
        }

    def make_params(self):
        return {
            "roles": {
                "front": self._capture_entry("front"),
                "down": self._capture_entry("down"),
            },
            "reconstruct_action": "/reconstruct/reconstruct_segment",
            "service_name": "/camera_pipeline/command",
            "diagnostics_topic": "/diagnostics",
            "diagnostics_period_sec": 1.0,
            "action_connect_timeout_sec": 5.0,
            "capture_wait_margin_sec": 10.0,
            "reconstruct_wait_timeout_sec": 600.0,
            "reconstruct_frame_group_size": 8,
        }

    def test_load_pipeline_config_builds_roles_from_capture_profiles(self):
        config = load_pipeline_config(self.make_params())
        self.assertEqual({role.role for role in config.roles}, {"front", "down"})
        self.assertEqual(config.roles[0].capture_action, "/front/capture_segment")
        self.assertEqual(
            config.roles[0].capture_config.camera_frame_id,
            "front_camera_optical_frame",
        )
        self.assertEqual(config.reconstruct_frame_group_size, 8)

    def test_load_pipeline_config_requires_absolute_actions(self):
        for field in ("reconstruct_action", "service_name", "diagnostics_topic"):
            params = self.make_params()
            params[field] = "relative_name"
            with self.subTest(field=field):
                with self.assertRaises(Exception):
                    load_pipeline_config(params)

    def test_node_not_ready_still_publishes_error_diagnostics(self):
        publisher = FakePublisher()
        bad = PipelineServiceNode(
            {"service_name": "bad"},
            service_factory=lambda *a, **k: FakeService(*a, **k),
            publisher_factory=lambda *a, **k: publisher,
            timer_factory=lambda *a, **k: FakeTimer(*a, **k),
            auto_start=True,
        )
        self.assertFalse(bad.is_ready())
        self.assertIsNotNone(bad._service)
        bad.publish_once()
        self.assertEqual(len(publisher.messages), 1)
        statuses = {
            status.name: status for status in publisher.messages[-1].status
        }
        self.assertEqual(
            statuses["drone3d/camera_pipeline/server"].level, DiagnosticStatus.ERROR
        )
        self.assertIn(
            "not ready", statuses["drone3d/camera_pipeline/server"].message
        )

    def test_node_not_ready_and_ready_paths(self):
        service_holder = []
        publisher = FakePublisher()
        node = PipelineServiceNode(
            self.make_params(),
            service_factory=lambda *a, **k: service_holder.append(
                FakeService(*a, **k)
            )
            or service_holder[-1],
            publisher_factory=lambda *a, **k: publisher,
            timer_factory=lambda *a, **k: FakeTimer(*a, **k),
            auto_start=True,
        )
        self.assertTrue(node.is_ready())
        self.assertEqual(service_holder[0].name, "/camera_pipeline/command")
        node.publish_once()
        self.assertEqual(len(publisher.messages), 1)
        statuses = {
            status.name: status for status in publisher.messages[-1].status
        }
        self.assertEqual(
            statuses["drone3d/camera_pipeline/server"].level, DiagnosticStatus.OK
        )
        self.assertEqual(
            statuses["drone3d/camera_pipeline/task"].level, DiagnosticStatus.OK
        )
        node.shutdown()
        self.assertTrue(publisher.unregistered)

    def test_load_pipeline_config_requires_frame_group_size(self):
        params = self.make_params()
        del params["reconstruct_frame_group_size"]
        with self.assertRaises(Exception):
            load_pipeline_config(params)

    def test_shutdown_timeout_still_cleans_service_and_publisher(self):
        service_holder = []
        publisher = FakePublisher()
        node = PipelineServiceNode(
            self.make_params(),
            service_factory=lambda *a, **k: service_holder.append(
                FakeService(*a, **k)
            )
            or service_holder[-1],
            publisher_factory=lambda *a, **k: publisher,
            timer_factory=lambda *a, **k: FakeTimer(*a, **k),
            capture_client_factory=lambda role: NeverCaptureClient(),
            auto_start=True,
        )
        service = service_holder[0]
        select = PipelineCommandRequest()
        select.command = PipelineCommandRequest.SELECT_SOURCE
        select.role = "front"
        node._service.callback(select)
        start = PipelineCommandRequest()
        start.command = PipelineCommandRequest.START_TASK
        start.role = "front"
        start.scene_id = "scene-front-01"
        start.sample_fps = 1.0
        start.max_frames = 2
        start.max_duration_sec = 30.0
        node._service.callback(start)
        with self.assertRaises(RuntimeError):
            node.shutdown(timeout_sec=0.2)
        self.assertTrue(publisher.unregistered)
        self.assertTrue(service.shutdown_calls)

    def test_read_result_identity_reads_local_file_and_rejects_missing(self):
        result = self.root / "result.json"
        result.write_text(
            '{"scene_id":"scene-front-01","actual_frame_count":2,'
            '"camera_role":"front","camera_topic":"/front/image_raw",'
            '"camera_frame_id":"front_camera_optical_frame",'
            '"snapshot":{"sha256":"' + "a" * 64 + '"}}',
            encoding="utf-8",
        )
        identity = read_result_identity(result.as_uri())
        self.assertEqual(identity["scene_id"], "scene-front-01")
        self.assertEqual(identity["camera_role"], "front")
        self.assertEqual(identity["snapshot_sha256"], "a" * 64)
        self.assertEqual(
            read_result_identity(str(self.root / "missing.json")), {}
        )


class PipelineDiagnosticsAssemblyTest(unittest.TestCase):
    def test_idle_snapshot_assembles_ok_statuses(self):
        config = mock.Mock()
        config.service_name = "/camera_pipeline/command"
        config.reconstruct_action = "/reconstruct/reconstruct_segment"
        config.roles = [mock.Mock()]
        core_snapshot = {
            "selected_role": "front",
            "roles": [
                {
                    "role": "front",
                    "capture_action": "/front/capture_segment",
                    "image_topic": "/front/image_raw",
                    "camera_frame_id": "front_camera_optical_frame",
                    "camera_name": "front_camera",
                }
            ],
            "task": {"state": STATE_IDLE},
        }
        message = assemble_pipeline_diagnostics(
            core_snapshot=core_snapshot, config=config, not_ready_error=None
        )
        self.assertEqual(len(message.status), 3)
        self.assertEqual(
            message.status[0].name, "drone3d/camera_pipeline/server"
        )
        self.assertEqual(message.status[1].name, "drone3d/camera_pipeline/task")
        self.assertEqual(message.status[2].name, "drone3d/camera_pipeline/roles")

    def test_not_ready_snapshot_reports_error(self):
        message = assemble_pipeline_diagnostics(
            core_snapshot=None, config=None, not_ready_error="bad config"
        )
        self.assertTrue(
            all(status.level == DiagnosticStatus.ERROR for status in message.status)
        )


if __name__ == "__main__":
    unittest.main()
