"""CPU-only tests for the ROS target-Web node boundary (WP8)."""

from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from drone3d_reconstruction_msgs.srv import (
    ContinuousSessionCommand,
    ContinuousSessionCommandRequest,
    ContinuousSessionCommandResponse,
)

from drone3d_reconstruction_core.continuous_web import WebDashboardConfig
from drone3d_reconstruction_core.continuous_web_node import (
    ContinuousWebNode,
    ContinuousWebNodeError,
    build_continuous_session_request,
    command_response_to_dict,
    load_web_dashboard_config,
)


class FakeServiceProxy:
    def __init__(self, name, srv_type):
        self.name = name
        self.srv_type = srv_type
        self.calls = []

    def __call__(self, request):
        self.calls.append(request)
        response = ContinuousSessionCommandResponse()
        response.error_code = ContinuousSessionCommandResponse.OK
        response.message = "ok"
        response.state = "RUNNING"
        response.session_id = "s" + "a" * 16
        response.role = "front"
        response.window_index = 1
        response.captured_frames = 2
        response.capture_target_frames = 6
        response.pending_windows = 3
        response.running_window = 4
        response.succeeded_windows = 5
        response.failed_windows = 6
        response.dropped_windows = 7
        response.effective_sample_fps = 1.0
        response.effective_window_frames = 8
        response.effective_queue_mode = "lossless_fifo"
        response.effective_queue_capacity = 9
        response.effective_max_retries = 0
        response.effective_keep_snapshot_on_success = False
        response.last_error = ""
        response.min_free_disk_gb = 2.0
        return response


class ContinuousWebNodeTest(unittest.TestCase):
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
            "depth_topic": f"/{role}/depth" if role == "front" else "",
            "tf_topic": "",
            "tf_static_topic": "",
            "save_sensor_depth": role == "front",
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
        results = self.root / "results"
        worker_results = self.root / "worker_results"
        journal = self.root / "journal"
        results.mkdir(exist_ok=True)
        worker_results.mkdir(exist_ok=True)
        journal.mkdir(exist_ok=True)
        return {
            "roles": {
                "front": self._capture_entry("front"),
                "down": self._capture_entry("down"),
            },
            "result_root": str(results),
            "journal_root": str(journal),
            "expected_reconstruction": {"result_root": str(worker_results)},
            "session_command_service": "/continuous_session/command",
            "web_host": "127.0.0.1",
            "web_port": 8080,
            "web_preview_fps": 5.0,
            "max_point_cloud_points": 500000,
            "max_session_windows": 32,
            "conf_percentile": 20.0,
        }

    def test_load_web_dashboard_config(self):
        config = load_web_dashboard_config(self.make_params())
        self.assertIsInstance(config, WebDashboardConfig)
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8080)
        self.assertEqual(config.web_preview_fps, 5.0)
        self.assertEqual(config.max_point_cloud_points, 500000)
        self.assertEqual(set(config.roles.keys()), {"front", "down"})
        self.assertEqual(config.roles["front"].image_topic, "/front/image_raw")
        self.assertEqual(config.roles["front"].depth_topic, "/front/depth")
        self.assertIsNone(config.roles["down"].depth_topic)
        self.assertEqual(
            config.roles["front"].snapshot_root,
            str(self.root / "front_snapshots"),
        )
        self.assertEqual(
            config.roles["front"].camera_frame_id, "front_camera_optical_frame"
        )
        self.assertEqual(config.worker_result_root, str(self.root / "worker_results"))
        self.assertEqual(config.journal_root, str(self.root / "journal"))

    def test_ros_large_byte_limit_uses_decimal_string(self):
        params = self.make_params()
        params["max_predictions_bytes"] = "2147483648"
        self.assertEqual(load_web_dashboard_config(params).max_predictions_bytes, 2147483648)
        for value in ("2GB", "1.5", "-1", True, "0"):
            params["max_predictions_bytes"] = value
            with self.assertRaises(ContinuousWebNodeError):
                load_web_dashboard_config(params)

    def test_load_web_dashboard_config_rejects_bad_params(self):
        invalid = [
            {"web_preview_fps": 0.0},
            {"web_preview_fps": -1.0},
            {"web_preview_fps": "fast"},
            {"web_port": 70000},
            {"web_port": True},
            {"conf_percentile": True},
            {"max_point_cloud_points": 0},
            {"max_session_windows": 0},
            {"max_predictions_bytes": 0},
            {"max_predictions_bytes": True},
            {"conf_percentile": 100.1},
            {"result_root": str(self.root / "missing")},
            {"session_command_service": ""},
        ]
        base = self.make_params()
        for patch in invalid:
            params = dict(base)
            params.update(patch)
            with self.subTest(patch=patch):
                with self.assertRaises(ContinuousWebNodeError):
                    load_web_dashboard_config(params)

        params = dict(base)
        params["roles"] = {}
        with self.assertRaises(ContinuousWebNodeError):
            load_web_dashboard_config(params)

    def test_build_request_maps_commands(self):
        request = build_continuous_session_request({"command": "pause"})
        self.assertEqual(request.command, ContinuousSessionCommandRequest.PAUSE)
        request = build_continuous_session_request({"command": "resume"})
        self.assertEqual(request.command, ContinuousSessionCommandRequest.RESUME)
        request = build_continuous_session_request({"command": "stop_all"})
        self.assertEqual(request.command, ContinuousSessionCommandRequest.STOP_ALL)
        request = build_continuous_session_request(
            {"command": "select_source", "role": "front"}
        )
        self.assertEqual(
            request.command, ContinuousSessionCommandRequest.SELECT_SOURCE
        )
        self.assertEqual(request.role, "front")
        request = build_continuous_session_request(
            {"command": "start_session", "role": "front"}
        )
        self.assertEqual(
            request.command, ContinuousSessionCommandRequest.START_SESSION
        )
        self.assertEqual(request.role, "front")
        self.assertEqual(request.reconstruction_mode, "")

    def test_build_request_rejects_set_params(self):
        with self.assertRaisesRegex(ContinuousWebNodeError, "unsupported web command"):
            build_continuous_session_request({"command": "set_params", "sample_fps": 2.0})

    def test_command_response_to_dict(self):
        response = ContinuousSessionCommandResponse()
        response.error_code = ContinuousSessionCommandResponse.OK
        response.message = "status"
        response.state = "RUNNING"
        response.session_id = "s" + "b" * 16
        response.role = "front"
        response.captured_frames = 3
        response.capture_target_frames = 4
        response.effective_sample_fps = 1.0
        response.effective_window_frames = 4
        response.effective_overlap_frames = 2
        response.effective_queue_mode = "lossless_fifo"
        response.effective_queue_capacity = 8
        response.effective_max_retries = 0
        response.effective_keep_snapshot_on_success = False
        payload = command_response_to_dict(response)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["state"], "RUNNING")
        self.assertEqual(payload["captured_frames"], 3)
        self.assertEqual(payload["capture_target_frames"], 4)
        self.assertEqual(payload["effective"]["window_frames"], 4)
        self.assertEqual(payload["effective"]["overlap_frames"], 2)

    def test_node_command_mapping_with_fake_service_proxy(self):
        proxies = []
        node = ContinuousWebNode(
            self.make_params(),
            service_proxy_factory=lambda *a, **k: proxies.append(
                FakeServiceProxy(*a, **k)
            )
            or proxies[-1],
            subscriber_factory=None,
        )
        self.assertTrue(node.is_ready())
        self.assertEqual(len(proxies), 1)
        self.assertEqual(
            proxies[0].name, "/continuous_session/command"
        )
        self.assertEqual(proxies[0].srv_type, ContinuousSessionCommand)

        result = node._send_service_command({"command": "pause"})
        self.assertTrue(result["ok"])
        self.assertEqual(
            proxies[0].calls[-1].command, ContinuousSessionCommandRequest.PAUSE
        )

        status = node._collect_status()
        self.assertEqual(status["state"], "RUNNING")
        self.assertEqual(
            proxies[0].calls[-1].command, ContinuousSessionCommandRequest.GET_STATUS
        )
        node.shutdown(timeout_sec=1.0)

    def test_collect_status_includes_storage_bytes_and_paths(self):
        front_dir = self.root / "front_snapshots" / "cs-live-w0"
        front_dir.mkdir(parents=True)
        (front_dir / "frames.csv").write_bytes(b"x" * 123)
        down_dir = self.root / "down_snapshots" / "cs-live-w0"
        down_dir.mkdir(parents=True)
        (down_dir / "frames.csv").write_bytes(b"y" * 45)
        node = ContinuousWebNode(
            self.make_params(),
            service_proxy_factory=FakeServiceProxy,
            subscriber_factory=None,
        )
        status = node._collect_status()
        self.assertGreaterEqual(status["capture_disk_bytes"], 123 + 45)
        self.assertEqual(
            status["storage_capture_bytes"], status["capture_disk_bytes"]
        )
        self.assertEqual(
            status["storage_worker_root"], str(self.root / "worker_results")
        )
        self.assertEqual(status["storage_formal_root"], str(self.root / "results"))
        self.assertEqual(status["storage_journal_root"], str(self.root / "journal"))
        self.assertGreaterEqual(
            status["storage_total_bytes"], status["storage_capture_bytes"]
        )
        node.shutdown(timeout_sec=1.0)

    def test_collect_status_reports_roles_with_live_image_publisher(self):
        node = ContinuousWebNode(
            self.make_params(),
            service_proxy_factory=FakeServiceProxy,
            subscriber_factory=None,
        )
        with mock.patch(
            "drone3d_reconstruction_core.continuous_web_node.rospy.get_published_topics",
            return_value=[
                ("/front/image_raw", "sensor_msgs/Image"),
                ("/camera/color/image_raw", "sensor_msgs/Image"),
            ],
        ):
            status = node._collect_status()
        self.assertEqual(status["available_roles"], ["front"])
        node.shutdown(timeout_sec=1.0)

    def test_diagnostics_merge_by_name_and_worker_running_active(self):
        params = self.make_params()
        params["diagnostics_topic"] = "/diagnostics"
        subscribers = []

        def subscriber_factory(topic, msg_type, callback):
            subscribers.append((topic, msg_type, callback))
            return SimpleNamespace(unregister=lambda: None)

        node = ContinuousWebNode(
            params,
            config_builder=load_web_dashboard_config,
            service_proxy_factory=FakeServiceProxy,
            subscriber_factory=subscriber_factory,
            core_factory=lambda *args, **kwargs: SimpleNamespace(),
        )
        diagnostics_callback = next(
            callback for topic, _type, callback in subscribers if topic == "/diagnostics"
        )

        def diagnostic_message(
            name, values, message="ok", level=DiagnosticStatus.OK
        ):
            msg = DiagnosticArray()
            status = DiagnosticStatus()
            status.name = name
            status.level = level
            status.message = message
            for key, value in values.items():
                item = KeyValue()
                item.key = key
                item.value = str(value)
                status.values.append(item)
            msg.status.append(status)
            return msg

        diagnostics_callback(diagnostic_message("drone3d/a", {"a": "1"}))
        diagnostics_callback(
            diagnostic_message(
                "drone3d/session/worker", {"running_window": 0}
            )
        )
        diagnostics_callback(diagnostic_message("drone3d/b", {"b": "2"}))
        diagnostics_callback(
            diagnostic_message(
                "drone3d/session/worker",
                {"running_window": 2},
                "window 2 running",
                DiagnosticStatus.WARN,
            )
        )
        status = node._collect_status()
        names = {item["name"] for item in status["diagnostics"]}
        self.assertIn("drone3d/a", names)
        self.assertIn("drone3d/b", names)
        self.assertEqual(status["running_window"], 2)
        self.assertTrue(status["worker_running_active"])

        diagnostics_callback(
            diagnostic_message(
                "drone3d/session/worker",
                {"running_window": 0},
                "window 0 running",
                DiagnosticStatus.WARN,
            )
        )
        status = node._collect_status()
        self.assertEqual(status["running_window"], 0)
        self.assertTrue(status["worker_running_active"])

    def test_node_not_ready_with_bad_params(self):
        node = ContinuousWebNode(
            {"web_host": "127.0.0.1"},
            subscriber_factory=None,
        )
        self.assertFalse(node.is_ready())


if __name__ == "__main__":
    unittest.main()
