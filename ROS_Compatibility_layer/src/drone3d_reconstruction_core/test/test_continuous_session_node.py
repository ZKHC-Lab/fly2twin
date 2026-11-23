"""CPU-only tests for the continuous session node boundary."""

from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from actionlib_msgs.msg import GoalStatus
from diagnostic_msgs.msg import DiagnosticStatus
from drone3d_reconstruction_msgs.srv import (
    ContinuousSessionCommandRequest,
    ContinuousSessionCommandResponse,
)

from drone3d_reconstruction_core.continuous_session import (
    JOB_RUNNING,
    JOB_SUCCEEDED,
    QUEUE_LOSSLESS,
    ContinuousConfigError,
    SESSION_FATAL_OOM,
    SESSION_RUNNING,
)
from drone3d_reconstruction_core.continuous_session_node import (
    ContinuousNodeConfigError,
    ContinuousSessionNode,
    LocalCudaBackend,
    _min_free_disk_gb,
    assemble_continuous_diagnostics,
    load_continuous_config,
)


class BlockingCaptureClient:
    """auto-start 测试用：阻塞 Capture Action，使 core 停留在 RUNNING。"""

    def __init__(self, role):
        self.role = role
        self.goal_sent = False
        self.pause_calls = 0
        self.resume_calls = 0

    def wait_for_server(self, timeout_sec):
        return True

    def send_goal(
        self,
        scene_id,
        sample_fps,
        max_frames,
        max_duration_sec,
        feedback_cb,
        save_sensor_depth=None,
        set_save_sensor_depth=False,
    ):
        self.goal_sent = True

    def wait_for_result(self, timeout_sec):
        return False

    def cancel_goal(self):
        pass

    def outcome(self):
        raise AssertionError("blocked capture should not be collected")

    def pause(self):
        self.pause_calls += 1
        return True

    def resume(self):
        self.resume_calls += 1
        return True


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


class FakeCaptureClient:
    def __init__(self, role):
        self.role = role
        self.pause_calls = 0
        self.resume_calls = 0

    def wait_for_server(self, timeout_sec):
        return True

    def send_goal(self, *args, **kwargs):
        pass

    def wait_for_result(self, timeout_sec):
        return True

    def cancel_goal(self):
        pass

    def outcome(self):
        from drone3d_reconstruction_core.action_types import (
            ACTION_FAILED,
            CaptureActionOutcome,
        )

        return CaptureActionOutcome(
            action_state=ACTION_FAILED,
            error_code=2,
            message="fake capture outcome",
            scene_id="",
            camera_name="",
            image_topic="",
            snapshot_uri="",
            manifest_uri="",
            manifest_sha256="",
            frame_count=0,
        )

    def pause(self):
        self.pause_calls += 1
        return True

    def resume(self):
        self.resume_calls += 1
        return True


class ContinuousSessionNodeTest(unittest.TestCase):
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
            "odometry_association": "interpolate",
            "action_name": f"/{role}/capture_segment",
        }

    def make_params(self):
        journal = self.root / "journal"
        results = self.root / "results"
        journal.mkdir()
        results.mkdir()
        return {
            "roles": {
                "front": self._capture_entry("front"),
                "down": self._capture_entry("down"),
            },
            "alignment_backend": "global_g4",
            "reconstruct_action": "/reconstruct/reconstruct_segment",
            "service_name": "/continuous_session/command",
            "diagnostics_topic": "/diagnostics",
            "diagnostics_period_sec": 1.0,
            "journal_root": str(journal),
            "result_root": str(results),
            "min_free_disk_gb": 0.01,
            "action_connect_timeout_sec": 1.0,
            "capture_wait_margin_sec": 1.0,
            "reconstruct_wait_timeout_sec": 5.0,
            "stop_timeout_sec": 2.0,
            "reconstruct_frame_group_size": 6,
            "initial_sample_fps": 1.0,
            "initial_window_frames": 6,
            "initial_overlap_frames": 5,
            "initial_queue_mode": QUEUE_LOSSLESS,
            "initial_queue_capacity": 8,
            "initial_max_retries": 0,
            "initial_keep_snapshot_on_success": True,
            "initial_inference_resolution": 384,
            "initial_worker_timeout_sec": 0.0,
            "expected_reconstruction": {
                "checkpoint_sha256": "b" * 64,
                "checkpoint_path": "/checkpoint/model.pt",
                "result_root": str(results),
                "algorithm_revision": "algo-test",
                "ros_revision": "ros-test",
                "worker_python": "/usr/bin/python3",
                "image_resolution": 384,
                "aggregator_frame_chunk_size": 10,
                "load_mode": "max_size",
                "frame_group_size": 6,
            },
        }

    def make_large_window_params(self):
        """100 帧窗（含 20 帧共享）的节点参数。"""
        params = self.make_params()
        params.update({
            "alignment_backend": "global_g4",
            "reconstruct_frame_group_size": 100,
            "initial_window_frames": 100,
            "initial_overlap_frames": 20,
            "initial_inference_resolution": 384,
            "initial_keep_snapshot_on_success": True,
        })
        params["expected_reconstruction"].update({
            "backend": "omega",
            "image_resolution": 384,
            "aggregator_frame_chunk_size": 10,
            "load_mode": "max_size",
            "frame_group_size": 100,
        })
        for entry in params["roles"].values():
            entry["max_goal_frames"] = 128
            entry["max_goal_duration_sec"] = 300.0
            entry["odometry_association"] = "interpolate"
            entry["snapshot_limits"] = dict(entry["snapshot_limits"], max_frame_count=128)
        return params

    def test_removed_grouping_keys_are_ignored_when_neutral_and_rejected_otherwise(self):
        neutral = {"window_frame_semantics": "total_frames", "minimum_total_frames": 100,
                   "window_plan_path": "", "fit_exclusion_source": "",
                   "threshold_table_path": "", "threshold_group": ""}
        params = self.make_large_window_params()
        reference = load_continuous_config(params)
        self.assertEqual(load_continuous_config(dict(params, **neutral)), reference)
        for key, value in (("window_frame_semantics", "new_frames"),
                           ("window_plan_path", "/plans/candidate_windows.json"),
                           ("fit_exclusion_source", "/plans/retained_observations.json"),
                           ("threshold_table_path", "/plans/seams.json"),
                           ("threshold_group", "G0")):
            with self.subTest(key=key), self.assertRaisesRegex(
                ContinuousNodeConfigError, "removed feature"
            ):
                load_continuous_config(dict(params, **{key: value}))

    def test_frame_pose_placement_params_reach_the_session_config(self):
        import copy
        params = self.make_large_window_params()
        default = load_continuous_config(copy.deepcopy(params))
        self.assertEqual(default.g4_placement, "frame_pose")
        self.assertEqual(default.pose_time_offset_ms, 0.0)
        self.assertIsNone(default.camera_intrinsics)
        params.update({"g4_placement": "frame_pose", "pose_time_offset_ms": 70,
                       "camera_intrinsics": {"width": 640, "height": 480, "fx": 604.2,
                                             "fy": 604, "cx": 319, "cy": 247.5}})
        config = load_continuous_config(params)
        self.assertEqual(config.g4_placement, "frame_pose")
        self.assertEqual(config.pose_time_offset_ms, 70.0)
        self.assertEqual(config.camera_intrinsics["fy"], 604.0)
        params["camera_intrinsics"] = "604,604,319,247"
        with self.assertRaisesRegex(Exception, "camera_intrinsics"):
            load_continuous_config(params)
        params.update(camera_intrinsics=None, g4_placement="graph")
        with self.assertRaisesRegex(Exception, "graph placement was removed"):
            load_continuous_config(params)

    def test_total_frames_profile_still_requires_matching_group_size(self):
        params = self.make_params()
        params["alignment_backend"] = "global_g4"
        params["initial_inference_resolution"] = 384
        params["initial_keep_snapshot_on_success"] = True
        params["initial_window_frames"] = 16
        params["initial_overlap_frames"] = 5
        params["reconstruct_frame_group_size"] = 16
        for entry in params["roles"].values():
            entry["odometry_association"] = "interpolate"
            entry["max_goal_frames"] = 16
            entry["max_goal_duration_sec"] = 60.0
            entry["snapshot_limits"] = dict(entry["snapshot_limits"], max_frame_count=16)
        params["expected_reconstruction"].update({
            "image_resolution": 384, "aggregator_frame_chunk_size": 10,
            "load_mode": "max_size", "frame_group_size": 16,
        })
        load_continuous_config(params)
        params["expected_reconstruction"]["frame_group_size"] = 8
        with self.assertRaises(ContinuousNodeConfigError):
            load_continuous_config(params)

    def test_g4_default_requires_matching_worker_resolution_and_total_size(self):
        params = self.make_params()
        params.pop("alignment_backend")
        params.update(initial_window_frames=16, initial_overlap_frames=5,
                      initial_inference_resolution=384, reconstruct_frame_group_size=16,
                      initial_keep_snapshot_on_success=True)
        params["expected_reconstruction"].update(image_resolution=384,frame_group_size=16,
                                                aggregator_frame_chunk_size=10,load_mode="max_size")
        for role in params["roles"].values():
            role.update(max_goal_frames=16,odometry_association="interpolate")
            role["snapshot_limits"]["max_frame_count"]=16
        config=load_continuous_config(params)
        self.assertEqual(config.alignment_backend, "global_g4")
        params["expected_reconstruction"]["frame_group_size"]=8
        with self.assertRaisesRegex(ContinuousNodeConfigError, "no subsampling"):
            load_continuous_config(params)

    def test_load_continuous_config(self):
        config = load_continuous_config(self.make_params())
        self.assertEqual({role.role for role in config.roles}, {"front", "down"})
        self.assertEqual(config.reconstruct_frame_group_size, 6)
        self.assertEqual(config.initial_window_frames, 6)
        self.assertEqual(config.initial_overlap_frames, 5)
        self.assertEqual(config.alignment_backend, "global_g4")
        self.assertEqual(config.expected_checkpoint_sha256, "b" * 64)
        self.assertEqual(config.expected_checkpoint_path, "/checkpoint/model.pt")
        self.assertEqual(
            config.expected_result_root, str(self.root / "results")
        )
        self.assertEqual(config.expected_algorithm_revision, "algo-test")
        self.assertEqual(config.expected_ros_revision, "ros-test")
        self.assertEqual(config.worker_python, "/usr/bin/python3")
        self.assertEqual(config.initial_inference_resolution, 384)
        self.assertEqual(config.initial_worker_timeout_sec, 0.0)




    def test_load_continuous_config_rejects_missing_or_bad_reconstruction(self):
        base = self.make_params()
        for missing in ("checkpoint_sha256", "checkpoint_path", "result_root", "algorithm_revision", "ros_revision", "worker_python"):
            params = dict(base)
            expected = dict(params["expected_reconstruction"])
            expected.pop(missing)
            params["expected_reconstruction"] = expected
            with self.subTest(missing=missing):
                with self.assertRaises(ContinuousNodeConfigError):
                    load_continuous_config(params)
        params = dict(base)
        params["expected_reconstruction"] = {"checkpoint_sha256": "not-a-sha"}
        with self.assertRaises((ContinuousNodeConfigError, ContinuousConfigError)):
            load_continuous_config(params)

    def test_alignment_backend_other_than_g4_is_a_config_error(self):
        base = self.make_params()
        for backend in ("odom_pose_constrained_sim3", "shared_point_sim3_A"):
            params = dict(base, alignment_backend=backend)
            with self.subTest(backend=backend), self.assertRaisesRegex(
                ContinuousNodeConfigError, "alignment_backend must be 'global_g4'"
            ):
                load_continuous_config(params)

    def test_load_continuous_config_rejects_invalid_params(self):
        invalid = [
            {"initial_queue_mode": "realtime_latest"},
            {"initial_keep_snapshot_on_success": False},
            {"initial_overlap_frames": 0},
            {"initial_inference_resolution": True},
            {"initial_inference_resolution": 31},
            {"initial_inference_resolution": 4097},
            {"initial_worker_timeout_sec": True},
            {"initial_worker_timeout_sec": -1.0},
        ]
        base = self.make_params()
        for patch in invalid:
            params = dict(base)
            params.update(patch)
            with self.subTest(patch=patch):
                with self.assertRaises(
                    (ContinuousNodeConfigError, ContinuousConfigError)
                ):
                    load_continuous_config(params)

    def test_node_service_select_start_status_stop(self):
        service_holder = []
        params = self.make_params()
        params["roles"]["front"]["depth_topic"] = "/front/depth"
        node = ContinuousSessionNode(
            params,
            service_factory=lambda *a, **k: service_holder.append(
                FakeService(*a, **k)
            )
            or service_holder[-1],
            publisher_factory=lambda *a, **k: FakePublisher(*a, **k),
            timer_factory=lambda *a, **k: FakeTimer(*a, **k),
            capture_client_factory=lambda role: FakeCaptureClient(role),
            backend_factory=lambda action: None,  # 本次测试不真正消费
            auto_start=True,
        )
        self.assertTrue(node.is_ready())
        service = service_holder[0]
        select = ContinuousSessionCommandRequest(
            command=ContinuousSessionCommandRequest.SELECT_SOURCE, role="front"
        )
        response = service.callback(select)
        self.assertEqual(response.error_code, ContinuousSessionCommandResponse.OK)
        start = ContinuousSessionCommandRequest(
            command=ContinuousSessionCommandRequest.START_SESSION, role=""
        )
        response = service.callback(start)
        self.assertEqual(response.error_code, ContinuousSessionCommandResponse.OK)
        self.assertEqual(response.state, SESSION_RUNNING)
        mode_change = ContinuousSessionCommandRequest(
            command=ContinuousSessionCommandRequest.SET_PARAMS, reconstruction_mode="per_window"
        )
        response = service.callback(mode_change)
        self.assertEqual(response.error_code, ContinuousSessionCommandResponse.INVALID_ARGUMENT)
        self.assertEqual(response.effective_reconstruction_mode, "per_window")
        tune = ContinuousSessionCommandRequest(
            command=ContinuousSessionCommandRequest.SET_PARAMS, sample_fps=10.0, window_frames=3,
        )
        response = service.callback(tune)
        self.assertEqual(response.error_code, ContinuousSessionCommandResponse.INVALID_ARGUMENT)
        self.assertIn("session parameters are fixed", response.message)
        self.assertEqual(response.state, SESSION_RUNNING)
        self.assertEqual(response.effective_sample_fps, 1.0)
        self.assertEqual(response.effective_window_frames, 6)
        status = ContinuousSessionCommandRequest(
            command=ContinuousSessionCommandRequest.GET_STATUS
        )
        response = service.callback(status)
        self.assertEqual(response.error_code, ContinuousSessionCommandResponse.OK)
        self.assertTrue(response.session_id)
        node.shutdown(timeout_sec=2.0)
        self.assertTrue(service.shutdown_calls)

    def test_auto_start_session_selects_role_and_starts(self):
        params = self.make_params()
        params.update(
            {
                "auto_start_session": True,
                "initial_role": "front",
                "auto_start_delay_sec": 0.0,
            }
        )
        node = ContinuousSessionNode(
            params,
            service_factory=lambda *a, **k: FakeService(*a, **k),
            publisher_factory=lambda *a, **k: FakePublisher(*a, **k),
            timer_factory=lambda *a, **k: FakeTimer(*a, **k),
            capture_client_factory=lambda role: BlockingCaptureClient(role),
            backend_factory=lambda action: None,
            auto_start=True,
        )
        self.assertTrue(node.is_ready())
        deadline = time.monotonic() + 3.0
        while (
            node.core.get_status().state != SESSION_RUNNING
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertEqual(node.core.get_status().state, SESSION_RUNNING)
        self.assertEqual(node.core.get_status().role, "front")
        node.shutdown(timeout_sec=2.0)

    def test_not_ready_diagnostics(self):
        publisher = FakePublisher()
        node = ContinuousSessionNode(
            {"service_name": "/continuous_session/command"},
            service_factory=lambda *a, **k: FakeService(*a, **k),
            publisher_factory=lambda *a, **k: publisher,
            timer_factory=lambda *a, **k: FakeTimer(*a, **k),
            auto_start=True,
        )
        self.assertFalse(node.is_ready())
        node.publish_once()
        statuses = {
            status.name: status for status in publisher.messages[-1].status
        }
        self.assertEqual(
            statuses["drone3d/session/server"].level, DiagnosticStatus.ERROR
        )


class ContinuousDiagnosticsAssemblyTest(unittest.TestCase):
    def test_memory_retry_and_backend_exit_survive_ros_diagnostics(self):
        for state, paused, retry, failures in (
            ("MEMORY_PAUSED", True, 30, 1),
            ("RUNNING", False, None, 0),
        ):
            with self.subTest(state=state):
                message = assemble_continuous_diagnostics(
                    core_snapshot={
                        "state": state, "effective": {}, "fifo": [],
                        "memory_paused": paused, "oom_latched": False,
                        "retry_window": retry, "failed_windows": failures,
                        "attempt_failures": 1, "manual_retries": int(not paused),
                        "g4": {"state": "error", "exit_code": -9},
                    },
                    config=None, not_ready_error=None,
                )
                statuses = {item.name: item for item in message.status}
                worker = {item.key: item.value for item in
                          statuses["drone3d/session/worker"].values}
                self.assertEqual(worker["memory_paused"], str(paused))
                self.assertEqual(worker["oom_latched"], "False")
                self.assertEqual(worker["retry_window"],
                                 "unavailable" if retry is None else "30")
                self.assertEqual(worker["failed_windows"], str(failures))
                self.assertEqual(worker["attempt_failures"], "1")
                self.assertEqual(worker["manual_retries"], str(int(not paused)))
                map_backend = statuses["drone3d/session/map_backend"]
                self.assertEqual(
                    {item.key: item.value for item in map_backend.values}["g4_exit_code"], "-9"
                )
                self.assertEqual(map_backend.level, DiagnosticStatus.ERROR)

    def test_idle_snapshot_assembles_five_statuses(self):
        message = assemble_continuous_diagnostics(
            core_snapshot={
                "state": "IDLE",
                "session_id": "",
                "role": "",
                "window_index": 0,
                "captured_frames": 0,
                "pending_windows": 0,
                "running_window": None,
                "succeeded_windows": 0,
                "failed_windows": 0,
                "dropped_windows": 0,
                "effective": {
                    "sample_fps": 1.0,
                    "window_frames": 2,
                    "queue_mode": QUEUE_LOSSLESS,
                    "queue_capacity": 8,
                    "max_retries": 0,
                    "keep_snapshot_on_success": False,
                },
                "fifo": [],
                "last_error": "",
                "min_free_disk_gb": 0.01,
            },
            config=None,
            not_ready_error=None,
        )
        self.assertEqual(
            {status.name for status in message.status},
            {
                "drone3d/session/server",
                "drone3d/session/capture",
                "drone3d/session/map_backend",
                "drone3d/session/fifo",
                "drone3d/session/worker",
            },
        )

    def test_map_backend_status_is_error_only_when_the_backend_failed(self):
        ok, error = DiagnosticStatus.OK, DiagnosticStatus.ERROR
        cases = (
            ({}, ok, "waiting"),                      # 还没有成功的窗口，后台未启动
            ({"state": "optimizing", "revision": 2}, ok, "optimizing"),
            ({"state": "ready", "revision": 3}, ok, "ready"),
            ({"state": "stopped", "exit_code": 0}, ok, "stopped"),
            # 进程没起来、求解抛错、进程异常退出：都必须是 ERROR，并带上原因。
            ({"state": "error", "error": "G4 background start failed: OSError: x"}, error,
             "G4 background start failed: OSError: x"),
            ({"state": "error", "error": "ValueError: bad window"}, error, "ValueError: bad window"),
            ({"state": "error", "exit_code": -9}, error, "state error"),
        )
        for g4, level, text in cases:
            with self.subTest(g4=g4):
                message = assemble_continuous_diagnostics(
                    core_snapshot={"state": "RUNNING", "effective": {}, "fifo": [], "g4": g4},
                    config=None, not_ready_error=None,
                )
                statuses = {item.name: item for item in message.status}
                status = statuses["drone3d/session/map_backend"]
                self.assertEqual((status.level, status.message), (level, text))
                # 地图后端出错不改变会话其它状态项的等级：采集和重建照常。
                self.assertEqual(statuses["drone3d/session/capture"].level, ok)

    def test_fatal_oom_snapshot_marks_server_error(self):
        message = assemble_continuous_diagnostics(
            core_snapshot={
                "state": SESSION_FATAL_OOM,
                "session_id": "s-test",
                "role": "front",
                "window_index": 1,
                "captured_frames": 0,
                "pending_windows": 1,
                "running_window": None,
                "succeeded_windows": 0,
                "failed_windows": 1,
                "dropped_windows": 0,
                "effective": {
                    "sample_fps": 1.0,
                    "window_frames": 2,
                    "queue_mode": QUEUE_LOSSLESS,
                    "queue_capacity": 8,
                    "max_retries": 0,
                    "keep_snapshot_on_success": False,
                },
                "fifo": [],
                "last_error": "FATAL_OOM: CUDA out of memory",
                "min_free_disk_gb": 0.01,
            },
            config=SimpleNamespace(
                reconstruct_action="/reconstruct/reconstruct_segment",
                journal_root="/tmp/journal",
            ),
            not_ready_error=None,
        )
        statuses = {status.name: status for status in message.status}
        self.assertEqual(
            statuses["drone3d/session/server"].level, DiagnosticStatus.ERROR
        )
        self.assertEqual(
            statuses["drone3d/session/server"].message, SESSION_FATAL_OOM
        )
        server_keys = {
            item.key for item in statuses["drone3d/session/server"].values
        }
        self.assertIn("last_alignment_state", server_keys)

    def test_min_free_disk_fails_closed_when_all_roots_unreadable(self):
        config = SimpleNamespace(
            roles=(
                SimpleNamespace(
                    capture_config=SimpleNamespace(snapshot_root="/tmp/snaps")
                ),
            ),
            journal_root="/tmp/journal",
            result_root="/tmp/results",
        )
        with mock.patch(
            "drone3d_reconstruction_core.continuous_session_node.os.statvfs",
            side_effect=OSError("statvfs failed"),
        ):
            with self.assertRaises(RuntimeError):
                _min_free_disk_gb(config)

    def test_worker_status_publishes_worker_wall_time_sec(self):
        message = assemble_continuous_diagnostics(
            core_snapshot={
                "state": SESSION_RUNNING,
                "session_id": "s-test",
                "role": "front",
                "window_index": 1,
                "captured_frames": 3,
                "pending_windows": 0,
                "running_window": None,
                "succeeded_windows": 1,
                "failed_windows": 0,
                "dropped_windows": 0,
                "effective": {
                    "sample_fps": 1.0,
                    "window_frames": 2,
                    "queue_mode": QUEUE_LOSSLESS,
                    "queue_capacity": 8,
                    "max_retries": 0,
                    "keep_snapshot_on_success": False,
                },
                "fifo": [],
                "last_error": "",
                "min_free_disk_gb": 0.01,
                "worker_wall_time_sec": 12.5,
                "last_result_wall_time_sec": 12.5,
            },
            config=SimpleNamespace(
                reconstruct_action="/reconstruct/reconstruct_segment",
                journal_root="/tmp/journal",
                result_root="/tmp/results",
            ),
            not_ready_error=None,
        )
        statuses = {status.name: status for status in message.status}
        worker_values = {
            item.key: item.value
            for item in statuses["drone3d/session/worker"].values
        }
        self.assertEqual(worker_values["worker_wall_time_sec"], "12.5")
        self.assertEqual(worker_values["last_result_wall_time_sec"], "12.5")

    def test_min_free_disk_includes_expected_result_root(self):
        config = SimpleNamespace(
            roles=(
                SimpleNamespace(
                    capture_config=SimpleNamespace(snapshot_root="/tmp/snaps")
                ),
            ),
            journal_root="/tmp/journal",
            result_root="/tmp/results",
            expected_result_root="/tmp/expected_results",
        )

        def fake_statvfs(path):
            free = {
                "/tmp/snaps": 10.0,
                "/tmp/journal": 20.0,
                "/tmp/results": 30.0,
                "/tmp/expected_results": 5.0,
            }[path]
            return SimpleNamespace(f_bavail=free, f_frsize=1_000_000_000)

        with mock.patch(
            "drone3d_reconstruction_core.continuous_session_node.os.statvfs",
            side_effect=fake_statvfs,
        ):
            self.assertEqual(_min_free_disk_gb(config), 5.0)

    def test_min_free_disk_uses_other_roots_when_expected_result_root_unreadable(self):
        config = SimpleNamespace(
            roles=(
                SimpleNamespace(
                    capture_config=SimpleNamespace(snapshot_root="/tmp/snaps")
                ),
            ),
            journal_root="/tmp/journal",
            result_root="/tmp/results",
            expected_result_root="/tmp/expected_results",
        )

        def fake_statvfs(path):
            if path == "/tmp/expected_results":
                raise OSError("unreadable expected result root")
            free = {
                "/tmp/snaps": 10.0,
                "/tmp/journal": 20.0,
                "/tmp/results": 30.0,
            }[path]
            return SimpleNamespace(f_bavail=free, f_frsize=1_000_000_000)

        with mock.patch(
            "drone3d_reconstruction_core.continuous_session_node.os.statvfs",
            side_effect=fake_statvfs,
        ):
            self.assertEqual(_min_free_disk_gb(config), 10.0)

    def test_local_cuda_backend_collect_result_removes_client(self):
        backend = LocalCudaBackend("/reconstruct/reconstruct_segment")
        backend._jobs["job-0"] = SimpleNamespace(outcome=lambda: "ok")
        self.assertEqual(backend.collect_result("job-0"), "ok")
        self.assertNotIn("job-0", backend._jobs)
        with self.assertRaises(RuntimeError):
            backend.collect_result("job-0")

    def test_local_cuda_backend_waits_for_success_result_message(self):
        backend = LocalCudaBackend("/reconstruct/reconstruct_segment")
        result = [None]
        action_client = SimpleNamespace(
            get_state=lambda: GoalStatus.SUCCEEDED,
            get_result=lambda: result[0],
        )
        backend._jobs["job-0"] = SimpleNamespace(_client=action_client)
        self.assertEqual(backend.status("job-0").state, JOB_RUNNING)
        result[0] = SimpleNamespace()
        self.assertEqual(backend.status("job-0").state, JOB_SUCCEEDED)

    def test_local_cuda_backend_collect_result_retains_client_on_outcome_error(self):
        backend = LocalCudaBackend("/reconstruct/reconstruct_segment")
        stale = SimpleNamespace(outcome=mock.Mock(side_effect=RuntimeError("boom")))
        backend._jobs["job-0"] = stale
        with self.assertRaises(RuntimeError):
            backend.collect_result("job-0")
        self.assertIs(backend._jobs.get("job-0"), stale)

    def test_local_cuda_backend_submit_rejects_existing_window_id(self):
        backend = LocalCudaBackend("/reconstruct/reconstruct_segment")
        stale = SimpleNamespace(cancel_goal=mock.Mock(), outcome=lambda: "stale")
        backend._jobs["job-0"] = stale
        with self.assertRaises(RuntimeError):
            backend.submit("file:///tmp/snapshot", "a" * 64, "job-0")
        self.assertIs(backend._jobs["job-0"], stale)
        stale.cancel_goal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
