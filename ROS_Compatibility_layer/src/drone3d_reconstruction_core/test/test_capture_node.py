"""CPU-only tests for the capture production node boundary."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from diagnostic_msgs.msg import DiagnosticStatus

from drone3d_reconstruction_core.capture import CaptureConfig
from drone3d_reconstruction_core.capture_node import (
    CaptureNode,
    CaptureNodeConfigError,
    assemble_diagnostics_array,
    load_capture_config,
    load_runtime_params,
)


class FakePublisher:
    def __init__(self):
        self.messages = []
        self.unregistered = False

    def publish(self, message):
        self.messages.append(message)

    def unregister(self):
        self.unregistered = True


class FakeTimer:
    def shutdown(self):
        pass


class FakeServer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.shutdown_calls = 0

    def diagnostics_snapshot(self):
        return {
            "running": False,
            "stage": "idle",
            "session": None,
            "latest_image_wall_age_sec": None,
            "latest_odometry_wall_age_sec": None,
            "last_terminal": None,
        }

    def shutdown(self, timeout_sec):
        self.shutdown_calls += 1


class CaptureNodeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshots_root = self.root / "snapshots"
        self.snapshots_root.mkdir()

    def make_params(self, **overrides):
        params = {
            "snapshot_root": str(self.snapshots_root),
            "image_topic": "/camera4/image_raw",
            "camera_name": "camera4",
            "camera_role": "forward",
            "camera_frame_id": "camera4_optical_frame",
            "image_encoding": "bgr8",
            "odometry_topic": "/Odometry",
            "camera_info_topic": "",
            "depth_topic": "",
            "tf_topic": "",
            "tf_static_topic": "",
            "save_sensor_depth": False,
            "body_frame_id": "body",
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
            "max_goal_frames": 16,
            "max_goal_duration_sec": 30.0,
            "min_captured_frames": 1,
            "max_image_width": 640,
            "max_image_height": 480,
            "pose_max_delta_ms": 50.0,
            "pose_gap_max_ms": 200.0,
            "max_sample_interval_overshoot_ms": 250.0,
            "snapshot_limits": {
                "max_frame_count": 16,
                "max_manifest_entries": 128,
                "max_metadata_bytes": 1024 * 1024,
                "max_file_bytes": 1024 * 1024,
                "max_total_bytes": 64 * 1024 * 1024,
                "max_scene_id_bytes": 128,
                "max_relative_path_bytes": 1024,
            },
        }
        params.update(overrides)
        return params

    def test_valid_params_build_explicit_capture_config(self):
        config = load_capture_config(self.make_params())
        self.assertIsInstance(config, CaptureConfig)
        self.assertEqual(config.image_topic, "/camera4/image_raw")
        self.assertEqual(config.camera_role, "forward")
        self.assertEqual(config.camera_frame_id, "camera4_optical_frame")
        self.assertEqual(config.max_sample_interval_overshoot_ms, 250.0)

    def test_every_safety_parameter_is_required(self):
        params = self.make_params()
        for key in list(params) + ["odometry_topic", "camera_info_topic"]:
            with self.subTest(key=key):
                broken = dict(params)
                broken.pop(key, None)
                with self.assertRaises(CaptureNodeConfigError):
                    load_capture_config(broken)

    def test_invalid_values_and_snapshot_limits_fail_fast(self):
        cases = (
            {"image_topic": "camera4"},
            {"camera_name": "bad name"},
            {"camera_role": "bad role"},
            {"camera_frame_id": ""},
            {"min_sample_fps": 0},
            {"min_sample_fps": 20, "max_sample_fps": 10},
            {"pose_max_delta_ms": -1},
            {"pose_gap_max_ms": 10, "pose_max_delta_ms": 50},
            {"max_sample_interval_overshoot_ms": -1},
            {"max_goal_frames": True},
            {"max_image_width": 0},
            {"odometry_topic": "Odometry"},
            {"save_sensor_depth": True, "depth_topic": ""},
            {"tf_relevant_frames": "body"},
            {"max_auxiliary_samples": 0},
            {"source_timeout_sec": 0},
            {"max_window_gap_sec": -1},
            {"snapshot_limits": {"max_frame_count": 16}},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(CaptureNodeConfigError):
                    load_capture_config(self.make_params(**overrides))

    def test_runtime_wiring_rejects_nonfinite_period(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                with self.assertRaises(CaptureNodeConfigError):
                    load_runtime_params({"diagnostics_period_sec": value})

    def test_input_status_publishes_aux_coverage(self):
        state = {
            "config": SimpleNamespace(
                camera_role="forward",
                camera_name="camera4",
                camera_frame_id="camera4_optical_frame",
                image_topic="/camera4/image_raw",
                image_encoding="bgr8",
                odometry_topic="/Odometry",
                snapshot_root="/tmp/snapshots",
            ),
            "not_ready_error": None,
        }
        server = {
            "running": True,
            "stage": "capturing",
            "latest_image_wall_age_sec": None,
            "latest_odometry_wall_age_sec": None,
            "last_terminal": None,
            "session": {
                "received_images": 3,
                "skipped_images": 0,
                "selected_frames": 2,
                "odometry_gaps": 0,
                "odometry_regressions": 0,
                "source_stale": False,
                "source_stale_entries": 0,
                "zero_stamp_count": 0,
                "duplicate_stamp_count": 0,
                "out_of_order_stamp_count": 0,
                "last_image_wall_age_sec": None,
                "has_camera_info": True,
                "has_odometry": False,
                "has_depth": True,
                "has_tf": True,
            },
        }
        message = assemble_diagnostics_array(state, server)
        input_status = next(
            status
            for status in message.status
            if status.name == "drone3d/capture/forward/input"
        )
        values = {item.key: item.value for item in input_status.values}
        self.assertEqual(values["aux_coverage"], "ci+ odom- depth+ tf+")

    def test_node_not_ready_and_ready_paths(self):
        server_factory = mock.Mock()
        bad = CaptureNode(
            {"snapshot_root": "missing"},
            server_factory=server_factory,
            auto_start=False,
        )
        self.assertFalse(bad.is_ready())
        server_factory.assert_not_called()

        fake_server = FakeServer()
        publisher = FakePublisher()
        node = CaptureNode(
            self.make_params(),
            server_factory=lambda **kwargs: fake_server,
            publisher_factory=lambda *args, **kwargs: publisher,
            timer_factory=lambda *args, **kwargs: FakeTimer(),
            auto_start=False,
        )
        self.assertTrue(node.is_ready())
        node.start_publishing()
        node.publish_once()
        self.assertEqual(len(publisher.messages), 1)
        statuses = {status.name: status for status in publisher.messages[-1].status}
        self.assertEqual(
            statuses["drone3d/capture/forward/server"].level, DiagnosticStatus.OK
        )
        self.assertEqual(
            statuses["drone3d/capture/forward/input"].level, DiagnosticStatus.OK
        )
        node.shutdown()
        self.assertTrue(publisher.unregistered)


if __name__ == "__main__":
    unittest.main()
