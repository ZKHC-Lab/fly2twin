"""CPU-only tests for the production ROS node config/diagnostics boundary."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from diagnostic_msgs.msg import DiagnosticStatus

from drone3d_reconstruction_core.controller import JobFinished
from drone3d_reconstruction_core.production import ProductionConfig
from drone3d_reconstruction_core.production_node import (
    DiagnosticsState,
    ProductionNode,
    ProductionNodeConfigError,
    ProductionRuntimeParams,
    assemble_diagnostic_array,
    load_production_config,
    load_runtime_params,
    sample_board_resources,
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class FakePublisher:
    def __init__(self):
        self.messages = []
        self.unregistered = False

    def publish(self, message):
        self.messages.append(message)

    def unregister(self):
        self.unregistered = True


class FakeTimer:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


class FakeServer:
    def __init__(self):
        self.constructor_kwargs = None
        self.shutdown_calls = 0
        self.snapshot = {
            "running": True,
            "stage": "verifying",
            "job_id": 3,
            "cancel_requested": False,
            "accepted": True,
            "accepted_count": 4,
            "busy_rejection_count": 2,
            "cancel_request_count": 1,
            "terminal_attempt_count": 5,
        }

    def diagnostics_snapshot(self):
        return dict(self.snapshot)

    def shutdown(self, timeout_sec):
        self.shutdown_calls += 1


class ProductionNodeTestBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.python = str(Path(sys.executable).resolve())
        self.snapshots_root = self.root / "snapshots"
        self.snapshots_root.mkdir()
        self.result_root = self.root / "results"
        self.result_root.mkdir()
        self.checkpoint = self.root / "checkpoint.pt"
        checkpoint_bytes = b"explicit checkpoint bytes"
        self.checkpoint.write_bytes(checkpoint_bytes)
        self.reconstruct_script = self.root / "reconstruct.py"
        self.reconstruct_script.write_text("import sys\nprint(sys.argv)\n")
        self.validator_script = self.root / "validator.py"
        self.validator_script.write_text("import sys\nprint(sys.argv)\n")
        self.checkpoint_sha256 = sha256_bytes(checkpoint_bytes)

    def make_params(self, **overrides):
        params = {
            "supervisor_python": self.python,
            "worker_python": self.python,
            "reconstruct_script": str(self.reconstruct_script),
            "validator_script": str(self.validator_script),
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "result_root": str(self.result_root),
            "allowed_snapshot_roots": [str(self.snapshots_root)],
            "snapshot_limits": {
                "max_frame_count": 16,
                "max_manifest_entries": 64,
                "max_metadata_bytes": 1024 * 1024,
                "max_file_bytes": 1024 * 1024,
                "max_total_bytes": 8 * 1024 * 1024,
                "max_scene_id_bytes": 128,
                "max_relative_path_bytes": 1024,
            },
            "resource_limits": {
                "min_mem_available_bytes": 1000,
                "max_swap_used_bytes": 1000,
                "max_swap_growth_bytes": 1000,
                "min_snapshot_free_bytes": 1000,
                "min_result_free_bytes": 1000,
                "swap_sample_window_sec": 0.01,
            },
            "image_resolution": 4,
            "frame_group_size": 3,
            "max_result_pixels": 64,
            "max_result_file_bytes": 256 * 1024,
            "max_result_total_bytes": 1024 * 1024,
            "max_log_bytes": 64 * 1024,
            "verifier_timeout_sec": 2.0,
            "worker_timeout_sec": 3.0,
            "term_grace_sec": 0.05,
            "load_mode": "max_size",
            "require_glb": False,
            "show_cameras": True,
            "conf_thres": 50.0,
            "max_points_k": 5,
            "profile_name": "local_docker_test_candidate",
            "required_device_sku": "docker_test_candidate",
            "device_sku": "docker_test_candidate",
            "algorithm_revision": "test-algorithm-revision",
            "ros_revision": "test-ros-revision",
        }
        params.update(overrides)
        return params


class ProductionNodeConfigTest(ProductionNodeTestBase):
    def test_failed_staging_retention_profile_is_strict_and_optional(self):
        self.assertFalse(load_production_config(self.make_params()).preserve_failed_worker_staging)
        config = load_production_config(self.make_params(preserve_failed_worker_staging=True))
        self.assertTrue(config.preserve_failed_worker_staging)
        with self.assertRaises(ProductionNodeConfigError):
            load_production_config(self.make_params(preserve_failed_worker_staging="true"))

    def test_valid_params_build_explicit_production_config(self):
        config = load_production_config(self.make_params())
        self.assertIsInstance(config, ProductionConfig)
        self.assertEqual(config.profile_name, "local_docker_test_candidate")
        self.assertEqual(config.required_device_sku, "docker_test_candidate")
        self.assertEqual(config.image_resolution, 4)

    def test_runtime_memory_guard_is_optional_and_profile_controlled(self):
        from drone3d_reconstruction_core.production_node import _controller_factory_for
        from drone3d_reconstruction_core.runtime_memory_guard import MemoryGuardedController
        from drone3d_reconstruction_core.controller import JobController
        legacy = load_production_config(self.make_params())
        self.assertIs(_controller_factory_for(legacy), JobController)
        guarded = load_production_config(self.make_params(runtime_memory_guard_enabled=True))
        controller = _controller_factory_for(guarded)(worker_timeout_sec=10., term_grace_sec=.1,
                                                    on_finished=lambda _: None)
        self.addCleanup(controller.shutdown)
        self.assertIsInstance(controller, MemoryGuardedController)
        with self.assertRaises(ProductionNodeConfigError):
            load_production_config(self.make_params(runtime_memory_guard_enabled="true"))

    def test_large_byte_values_accept_ros_safe_decimal_strings(self):
        params = self.make_params(
            max_result_file_bytes="8589934592",
            max_result_total_bytes="17179869184",
        )
        params["snapshot_limits"]["max_total_bytes"] = "4294967296"
        params["resource_limits"]["min_result_free_bytes"] = "8589934592"
        config = load_production_config(params)
        self.assertEqual(config.max_result_file_bytes, 8 * 1024 ** 3)
        self.assertEqual(config.max_result_total_bytes, 16 * 1024 ** 3)
        self.assertEqual(config.snapshot_limits.max_total_bytes, 4 * 1024 ** 3)
        self.assertEqual(
            config.resource_limits.min_result_free_bytes, 8 * 1024 ** 3
        )

    def test_decimal_strings_are_only_allowed_for_canonical_byte_values(self):
        cases = (
            {"image_resolution": "256"},
            {"max_result_file_bytes": "08"},
            {"max_result_file_bytes": "-1"},
            {"max_result_file_bytes": "1.5"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ProductionNodeConfigError):
                    load_production_config(self.make_params(**overrides))

    def test_every_safety_parameter_is_required(self):
        params = self.make_params()
        required_keys = [
            "supervisor_python",
            "worker_python",
            "reconstruct_script",
            "validator_script",
            "checkpoint_path",
            "checkpoint_sha256",
            "result_root",
            "allowed_snapshot_roots",
            "snapshot_limits",
            "resource_limits",
            "image_resolution",
            "frame_group_size",
            "max_result_pixels",
            "max_result_file_bytes",
            "max_result_total_bytes",
            "max_log_bytes",
            "verifier_timeout_sec",
            "worker_timeout_sec",
            "term_grace_sec",
            "load_mode",
            "require_glb",
            "show_cameras",
            "conf_thres",
            "max_points_k",
            "profile_name",
            "required_device_sku",
            "device_sku",
            "algorithm_revision",
            "ros_revision",
        ]
        for key in required_keys:
            with self.subTest(key=key):
                broken = dict(params)
                del broken[key]
                with self.assertRaises(ProductionNodeConfigError):
                    load_production_config(broken)

    def test_wrong_parameter_types_fail_before_action_server(self):
        cases = (
            {"image_resolution": True},
            {"frame_group_size": 1.5},
            {"require_glb": 1},
            {"show_cameras": 0},
            {"allowed_snapshot_roots": []},
            {"allowed_snapshot_roots": "not-a-list"},
            {"snapshot_limits": {"max_frame_count": 16}},
            {"resource_limits": {"min_mem_available_bytes": 1}},
            {"resource_limits": {
                "min_mem_available_bytes": 1000,
                "max_swap_used_bytes": 1000,
                "max_swap_growth_bytes": 1000,
                "min_snapshot_free_bytes": 1000,
                "min_result_free_bytes": 1000,
                "swap_sample_window_sec": 0.01,
                "extra": 1,
            }},
            {"device_sku": "bad sku"},
            {"required_device_sku": ""},
            {"verifier_timeout_sec": float("nan")},
            {"conf_thres": True},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ProductionNodeConfigError):
                    load_production_config(self.make_params(**overrides))

    def test_device_sku_mismatch_is_a_deployment_error(self):
        params = self.make_params(required_device_sku="orin_nx_16gb_unsigned")
        with self.assertRaises(ProductionNodeConfigError) as context:
            load_production_config(params)
        self.assertIn("device SKU mismatch", str(context.exception))

    def test_injected_sku_probe_result_is_compared(self):
        params = self.make_params()
        with self.assertRaises(ProductionNodeConfigError):
            load_production_config(
                params, sku_probe=lambda _params: "different_device"
            )

    def test_runtime_wiring_defaults_are_interface_only(self):
        runtime = load_runtime_params({})
        self.assertEqual(runtime.action_name, "reconstruct_segment")
        self.assertEqual(runtime.diagnostics_topic, "/diagnostics")
        self.assertEqual(runtime.diagnostics_period_sec, 1.0)
        self.assertIsInstance(runtime, ProductionRuntimeParams)
        for overrides in (
            {"diagnostics_period_sec": 0},
            {"diagnostics_period_sec": True},
            {"action_name": ""},
            {"diagnostics_topic": "bad topic"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ProductionNodeConfigError):
                    load_runtime_params(overrides)


class DiagnosticsAssemblyTest(ProductionNodeTestBase):
    def test_diagnostic_array_contains_phase1_required_fields(self):
        state = DiagnosticsState()
        config = load_production_config(self.make_params())
        state.set_config(config, "docker_test_candidate")
        state.record_finished(JobFinished("worker_exit", 137, 12.5, 8))
        terminal = SimpleNamespace(
            error_code=4,
            message="worker exited with code 137",
            wall_time=SimpleNamespace(to_sec=lambda: 12.5),
            scene_id="",
            result_manifest_uri="",
            result_manifest_sha256="",
            scene_glb_uri="",
            frame_count=0,
        )
        state.record_terminal("aborted", terminal, "worker exited with code 137")

        message = assemble_diagnostic_array(
            state=state.snapshot(),
            server_snapshot={
                "running": False,
                "stage": "idle",
                "job_id": None,
                "accepted_count": 1,
                "busy_rejection_count": 2,
                "cancel_request_count": 1,
                "terminal_attempt_count": 1,
            },
            board_sample={
                "mem_available_bytes": 123456,
                "swap_total_bytes": 100,
                "swap_used_bytes": 10,
                "result_free_bytes": 200000,
                "snapshot_free_bytes": 300000,
            },
        )
        names = [status.name for status in message.status]
        self.assertEqual(
            names,
            [
                "drone3d/reconstruct/supervisor",
                "drone3d/reconstruct/job",
                "drone3d/reconstruct/resources",
                "drone3d/reconstruct/capture",
            ],
        )
        supervisor = message.status[0]
        self.assertEqual(supervisor.level, DiagnosticStatus.OK)
        supervisor_values = {item.key: item.value for item in supervisor.values}
        self.assertEqual(supervisor_values["profile_name"], "local_docker_test_candidate")
        self.assertEqual(supervisor_values["actual_device_sku"], "docker_test_candidate")
        self.assertEqual(supervisor_values["checkpoint_sha256"], config.checkpoint_sha256)

        job = message.status[1]
        job_values = {item.key: item.value for item in job.values}
        self.assertEqual(job_values["running"], "False")
        self.assertEqual(job_values["stage"], "idle")
        self.assertEqual(job_values["worker_exit_cause"], "worker_exit")
        self.assertEqual(job_values["worker_exit_returncode"], "137")
        self.assertEqual(job_values["last_result_method"], "aborted")
        self.assertEqual(job_values["last_result_error"], "ALGORITHM_FAILED")
        self.assertEqual(job_values["last_result_message"], "worker exited with code 137")
        self.assertEqual(job_values["last_result_wall_time_sec"], "12.5")
        self.assertEqual(job_values["worker_wall_time_sec"], "12.5")

        resources = message.status[2]
        resource_values = {item.key: item.value for item in resources.values}
        self.assertEqual(resource_values["mem_available_bytes"], "123456")
        self.assertEqual(resource_values["swap_used_bytes"], "10")
        self.assertEqual(resource_values["result_free_bytes"], "200000")

        capture = message.status[3]
        self.assertEqual(capture.level, DiagnosticStatus.STALE)
        capture_values = {item.key: item.value for item in capture.values}
        self.assertEqual(capture_values["input_age"], "unavailable")
        self.assertEqual(capture_values["capture_gap"], "unavailable")

    def test_not_ready_state_reports_error_instead_of_busy(self):
        state = DiagnosticsState()
        state.set_not_ready(ProductionNodeConfigError("checkpoint digest mismatch"))
        message = assemble_diagnostic_array(
            state=state.snapshot(),
            server_snapshot={"running": False, "stage": "not_ready"},
            board_sample={"error": "no config"},
        )
        self.assertEqual(message.status[0].level, DiagnosticStatus.ERROR)
        self.assertIn("not ready", message.status[0].message)
        self.assertIn("checkpoint digest mismatch", message.status[0].message)
        self.assertEqual(message.status[1].level, DiagnosticStatus.ERROR)
        self.assertIn("not constructed", message.status[1].message)
        self.assertEqual(message.status[2].level, DiagnosticStatus.ERROR)

    def test_resource_breaches_are_reported_with_warn(self):
        state = DiagnosticsState()
        config = load_production_config(self.make_params())
        state.set_config(config, "docker_test_candidate")
        message = assemble_diagnostic_array(
            state=state.snapshot(),
            server_snapshot={"running": False, "stage": "idle"},
            board_sample={
                "mem_available_bytes": 10,
                "swap_total_bytes": 100,
                "swap_used_bytes": 90,
                "result_free_bytes": 1,
                "snapshot_free_bytes": 1,
            },
        )
        resources = message.status[2]
        self.assertEqual(resources.level, DiagnosticStatus.WARN)
        self.assertIn("MemAvailable", resources.message)

    def test_board_sampler_parses_bounded_proc_meminfo_and_disks(self):
        config = load_production_config(self.make_params())
        meminfo = (
            "MemTotal:       1000 kB\n"
            "MemAvailable:   400 kB\n"
            "SwapTotal:      200 kB\n"
            "SwapFree:       50 kB\n"
        )
        fake_statvfs = mock.Mock(
            side_effect=lambda path: SimpleNamespace(
                f_bavail=500 if str(path) == str(self.result_root) else 700,
                f_frsize=4096,
            )
        )
        board = sample_board_resources(
            config,
            meminfo_reader=lambda: meminfo,
            statvfs=fake_statvfs,
        )
        self.assertEqual(board["mem_available_bytes"], 400 * 1024)
        self.assertEqual(board["swap_used_bytes"], 150 * 1024)
        self.assertEqual(board["swap_total_bytes"], 200 * 1024)
        self.assertEqual(board["result_free_bytes"], 500 * 4096)
        self.assertEqual(board["snapshot_free_bytes"], 700 * 4096)

    def test_board_sampler_failure_is_recorded_not_raised(self):
        config = load_production_config(self.make_params())
        board = sample_board_resources(
            config, meminfo_reader=lambda: "MemAvailable: 1 kB\n"
        )
        self.assertIn("error", board)
        self.assertIn("meminfo", board["error"])


class ProductionNodeTest(ProductionNodeTestBase):
    def make_node(self, params=None, **kwargs):
        if params is None:
            params = self.make_params()
        return ProductionNode(params, **kwargs)

    def test_not_ready_node_never_constructs_action_server(self):
        params = self.make_params(checkpoint_sha256="a" * 64)
        publisher = FakePublisher()
        server_factory = mock.Mock()
        node = ProductionNode(
            params,
            server_factory=server_factory,
            publisher_factory=lambda *args, **kwargs: publisher,
            timer_factory=lambda *args, **kwargs: FakeTimer(),
            auto_start=True,
        )
        node.publish_once()
        server_factory.assert_not_called()
        self.assertFalse(node.is_ready())
        message = publisher.messages[-1]
        self.assertEqual(message.status[0].level, DiagnosticStatus.ERROR)
        self.assertIn("checkpoint digest mismatch", message.status[0].message)
        node.shutdown()

    def test_ready_node_constructs_server_with_explicit_timeouts_and_observers(self):
        fake_server = FakeServer()

        def server_factory(**kwargs):
            fake_server.constructor_kwargs = kwargs
            return fake_server

        publisher = FakePublisher()
        node = ProductionNode(
            self.make_params(),
            server_factory=server_factory,
            publisher_factory=lambda *args, **kwargs: publisher,
            timer_factory=lambda *args, **kwargs: FakeTimer(),
            auto_start=False,
        )
        self.assertTrue(node.is_ready())
        kwargs = fake_server.constructor_kwargs
        self.assertEqual(kwargs["action_name"], "reconstruct_segment")
        self.assertEqual(kwargs["worker_timeout_sec"], 3.0)
        self.assertEqual(kwargs["term_grace_sec"], 0.05)
        self.assertTrue(callable(kwargs["command_builder"]))
        self.assertTrue(callable(kwargs["terminal_observer"]))
        self.assertTrue(callable(kwargs["finished_observer"]))

        node.start_publishing()
        node.publish_once()
        self.assertEqual(publisher.messages[-1].status[0].level, DiagnosticStatus.OK)
        self.assertEqual(publisher.messages[-1].status[1].values[0].value, "True")
        self.assertEqual(publisher.messages[-1].status[1].values[1].value, "verifying")
        node.shutdown(timeout_sec=0.5)
        self.assertEqual(fake_server.shutdown_calls, 1)
        self.assertTrue(publisher.unregistered)

    def test_runtime_planner_and_server_failures_stay_not_ready(self):
        server_factory = mock.Mock()
        invalid_runtime = ProductionNode(
            {"diagnostics_period_sec": 0},
            server_factory=server_factory,
            auto_start=False,
        )
        self.assertFalse(invalid_runtime.is_ready())
        self.assertIsNone(invalid_runtime.config)
        server_factory.assert_not_called()

        planner_failed = ProductionNode(
            self.make_params(),
            planner_factory=mock.Mock(side_effect=RuntimeError("planner fail")),
            server_factory=server_factory,
            auto_start=False,
        )
        self.assertFalse(planner_failed.is_ready())
        server_factory.assert_not_called()

        server_failed = ProductionNode(
            self.make_params(),
            server_factory=mock.Mock(side_effect=RuntimeError("server fail")),
            auto_start=False,
        )
        self.assertFalse(server_failed.is_ready())

    def test_observers_feed_state_without_raising(self):
        state = DiagnosticsState()
        finished = JobFinished("worker_exit", 0, 1.0, 9)
        state.record_finished(finished)
        state.record_finished(None)
        snapshot = state.snapshot()
        self.assertEqual(snapshot["last_finished"]["returncode"], 0)
        self.assertIsNone(snapshot["last_terminal"])
        self.assertEqual(state.record_terminal("succeeded", object(), "x"), None)
        snapshot = state.snapshot()
        self.assertEqual(snapshot["last_terminal"]["error_code"], 0)


if __name__ == "__main__":
    unittest.main()
