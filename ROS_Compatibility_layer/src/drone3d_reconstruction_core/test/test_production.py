"""CPU-only tests for the production planner and completion verifier."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from drone3d_reconstruction_core import (
    JobFinished,
    ResourceBusyError,
    ResourceDecision,
    ResourceLimits,
    ResourceSample,
    ResultCollisionError,
)
from drone3d_reconstruction_core.preparation import (
    ALGORITHM_FAILED,
    INPUT_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_GOAL,
    IO_ERROR,
    CompletionError,
    PreparationError,
    VerificationCancelled,
)
from drone3d_reconstruction_core.production import (
    COORDINATE_STATE,
    RESULT_SCHEMA,
    RESULT_SCHEMA_VERSION,
    ProductionConfig,
    ProductionConfigError,
    ProductionPlanner,
    _selected_indices,
    warm_worker_command,
)
from drone3d_reconstruction_core.snapshot_validator import SnapshotSafetyLimits
from drone3d_reconstruction_core.worker_runner import WORKER_RUN_LOG_MARKER


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def passing_resource_enforcer(limits, snapshot_root, result_root):
    sample = ResourceSample(0, 0, 0, 0, 0, 0, 0, limits.swap_sample_window_sec)
    return ResourceDecision(True, None, "injected pass", sample, limits)


def make_snapshot(
    snapshots_root,
    scene_id="scene-001",
    frame_count=3,
    *,
    duplicate_basenames=False,
):
    scene = snapshots_root / scene_id
    (scene / "images").mkdir(parents=True, exist_ok=True)
    image_relatives = []
    for index in range(frame_count):
        if duplicate_basenames:
            directory = "a" if index == 0 else "b"
            (scene / "images" / directory).mkdir(exist_ok=True)
            relative = f"images/{directory}/same.png"
        else:
            relative = f"images/frame_{index:03d}.png"
        path = scene.joinpath(*Path(relative).parts)
        path.write_bytes(f"image-{scene_id}-{index}".encode("utf-8"))
        image_relatives.append(relative)

    snapshot_bytes = json.dumps(
        {
            "schema": "drone3d.reconstruction.snapshot",
            "version": 1,
            "status": "PASS",
            "scene_id": scene_id,
            "frame_count": frame_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    (scene / "snapshot.json").write_bytes(snapshot_bytes)

    frame_lines = ["frame_index,image_path"]
    for index, relative in enumerate(image_relatives):
        frame_lines.append(f"{index},{relative}")
    frames_bytes = ("\n".join(frame_lines) + "\n").encode("utf-8")
    (scene / "frames.csv").write_bytes(frames_bytes)

    manifest_rows = {
        "snapshot.json": snapshot_bytes,
        "frames.csv": frames_bytes,
    }
    for relative in image_relatives:
        manifest_rows[relative] = (
            scene.joinpath(*Path(relative).parts).read_bytes()
        )
    lines = ["path,size,sha256"]
    for relative in sorted(manifest_rows):
        data = manifest_rows[relative]
        lines.append(f"{relative},{len(data)},{sha256_bytes(data)}")
    manifest_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    (scene / "files_sha256.csv").write_bytes(manifest_bytes)
    return str(scene), sha256_bytes(manifest_bytes)


def make_snapshot_v2(
    snapshots_root,
    scene_id="scene-v2",
    frame_count=3,
):
    scene = snapshots_root / scene_id
    (scene / "images").mkdir(parents=True, exist_ok=True)
    image_relatives = []
    for index in range(frame_count):
        relative = f"images/frame_{index:03d}.png"
        path = scene.joinpath(*Path(relative).parts)
        path.write_bytes(f"image-{scene_id}-{index}".encode("utf-8"))
        image_relatives.append(relative)
    stamp_secs = [100 + index for index in range(frame_count)]
    snapshot_bytes = json.dumps(
        {
            "schema": "drone3d.reconstruction.snapshot",
            "version": 2,
            "status": "PASS",
            "scene_id": scene_id,
            "frame_count": frame_count,
            "camera_name": "camera4",
            "image_topic": "/camera4/image_raw",
            "first_stamp_sec": stamp_secs[0],
            "first_stamp_nsec": 0,
            "last_stamp_sec": stamp_secs[-1],
            "last_stamp_nsec": 0,
            "has_camera_info": False,
            "has_odometry": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    (scene / "snapshot.json").write_bytes(snapshot_bytes)
    frame_lines = [
        "frame_index,image_path,stamp_sec,stamp_nsec,pose_valid,"
        "pose_delta_ms,odometry_gap_ms"
    ]
    for index, relative in enumerate(image_relatives):
        frame_lines.append(
            f"{index},{relative},{stamp_secs[index]},0,"
            f"{'true' if index != 2 else 'false'},"
            f"{'10.0' if index != 2 else '-1'},"
            f"{'0.0' if index != 2 else '250.0'}"
        )
    frames_bytes = ("\n".join(frame_lines) + "\n").encode("utf-8")
    (scene / "frames.csv").write_bytes(frames_bytes)
    manifest_rows = {
        "snapshot.json": snapshot_bytes,
        "frames.csv": frames_bytes,
    }
    for relative in image_relatives:
        manifest_rows[relative] = (
            scene.joinpath(*Path(relative).parts).read_bytes()
        )
    lines = ["path,size,sha256"]
    for relative in sorted(manifest_rows):
        data = manifest_rows[relative]
        lines.append(f"{relative},{len(data)},{sha256_bytes(data)}")
    manifest_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    (scene / "files_sha256.csv").write_bytes(manifest_bytes)
    return str(scene), hashlib.sha256(manifest_bytes).hexdigest()


def make_fake_validator(behavior="success", probe_path=None, frame_count_from_argv=True):
    if behavior == "success":
        print_lines = (
            "import json,sys\n"
            "args=sys.argv[1:]\n"
            "expected=[]\n"
            "for i,a in enumerate(args):\n"
            "    if a=='--expected_image_basename' and i+1<len(args):\n"
            "        expected.append(args[i+1])\n"
            f"count=len(expected) if {frame_count_from_argv!r} else 3\n"
            "print(json.dumps({'ok':True,'frame_count':count,'width':4,'height':4,"
            "'glb_validated':False},separators=(',',':')))\n"
        )
        exit_code = 0
    elif behavior == "reject":
        print_lines = (
            "import json\n"
            "print(json.dumps({'error':'predictions.npz is corrupt',"
            "'error_type':'ResultValidationError','ok':False},"
            "separators=(',',':')))\n"
        )
        exit_code = 1
    elif behavior == "malformed":
        print_lines = "print('line one')\nprint('line two')\n"
        exit_code = 0
    elif behavior == "sleep":
        print_lines = "import time\nprint('about to sleep')\ntime.sleep(60)\n"
        exit_code = 0
    elif behavior == "loud_stderr":
        print_lines = (
            "import json,sys\n"
            "sys.stderr.write('x'*4000)\n"
            "sys.stderr.flush()\n"
            "print(json.dumps({'ok':True,'frame_count':3,'width':4,'height':4,"
            "'glb_validated':False},separators=(',',':')))\n"
        )
        exit_code = 0
    elif behavior == "loud":
        print_lines = (
            "import json,sys\n"
            "print(json.dumps({'ok':True,'frame_count':3,'width':4,'height':4,"
            "'glb_validated':False}) + 'x'*4000)\n"
        )
        exit_code = 0
    else:
        raise ValueError(behavior)

    if probe_path is not None:
        probe = (
            "import json,sys\n"
            f"with open({probe_path!r},'w') as f:\n"
            "    json.dump(sys.argv, f)\n"
        )
        script = probe + print_lines
    else:
        script = print_lines
    script += f"raise SystemExit({exit_code})\n"
    return script


class ProductionTestBase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.transactions = []
        self.python = str(Path(sys.executable).resolve())
        self.snapshots_root = self.root / "snapshots"
        self.snapshots_root.mkdir()
        self.result_root = self.root / "results"
        self.result_root.mkdir()

    def tearDown(self):
        for transaction in reversed(self.transactions):
            try:
                transaction.cleanup()
            except BaseException:
                pass

    def make_config(self, **overrides):
        self.checkpoint = self.root / "checkpoint.pt"
        checkpoint_bytes = b"checkpoint-bytes-for-testing"
        self.checkpoint.write_bytes(checkpoint_bytes)
        self.reconstruct_script = self.root / "reconstruct.py"
        self.reconstruct_script.write_text("import sys\nprint(sys.argv)\n")
        self.validator_script = self.root / "validator.py"
        self.validator_script.write_text(make_fake_validator("success"))
        values = dict(
            supervisor_python=self.python,
            worker_python=self.python,
            reconstruct_script=str(self.reconstruct_script),
            validator_script=str(self.validator_script),
            checkpoint_path=str(self.checkpoint),
            checkpoint_sha256=sha256_bytes(checkpoint_bytes),
            result_root=str(self.result_root),
            allowed_snapshot_roots=(str(self.snapshots_root),),
            snapshot_limits=SnapshotSafetyLimits(
                max_frame_count=16,
                max_manifest_entries=64,
                max_metadata_bytes=1024 * 1024,
                max_file_bytes=1024 * 1024,
                max_total_bytes=8 * 1024 * 1024,
                max_scene_id_bytes=128,
                max_relative_path_bytes=1024,
            ),
            resource_limits=ResourceLimits(
                min_mem_available_bytes=1000,
                max_swap_used_bytes=1000,
                max_swap_growth_bytes=1000,
                min_snapshot_free_bytes=1000,
                min_result_free_bytes=1000,
                swap_sample_window_sec=0.01,
            ),
            image_resolution=4,
            frame_group_size=3,
            max_result_pixels=64,
            max_result_file_bytes=256 * 1024,
            max_result_total_bytes=1024 * 1024,
            max_log_bytes=64 * 1024,
            verifier_timeout_sec=2.0,
            worker_timeout_sec=3.0,
            term_grace_sec=0.05,
            load_mode="max_size",
            require_glb=False,
            show_cameras=True,
            conf_thres=50.0,
            max_points_k=5,
            profile_name="local_docker_test_candidate",
            required_device_sku="docker_test_candidate",
            algorithm_revision="test-algorithm-revision",
            ros_revision="test-ros-revision",
        )
        values.update(overrides)
        return ProductionConfig(**values)

    def make_goal(self, snapshot_uri=None, manifest_sha256=None):
        if snapshot_uri is None:
            snapshot_uri, manifest_sha256 = make_snapshot(self.snapshots_root)
        return SimpleNamespace(
            snapshot_uri=snapshot_uri, manifest_sha256=manifest_sha256
        )

    def make_planner(self, config=None, token="tok00000001", **kwargs):
        if config is None:
            config = self.make_config()
        if token is not None:
            kwargs.setdefault("token_source", lambda token=token: token)
        kwargs.setdefault("resource_enforcer", passing_resource_enforcer)
        return ProductionPlanner(config, **kwargs)

    def track(self, transaction):
        self.transactions.append(transaction)
        return transaction

    def write_valid_staging(self, staging, basenames, *, glb=False, **overrides):
        run_log = WORKER_RUN_LOG_MARKER + b"worker output line\n"
        manifest = ("\n".join(basenames) + "\n").encode("utf-8")
        predictions = b"predictions.npz bytes for testing"
        values = {
            "run.log": run_log,
            "manifest.txt": manifest,
            "predictions.npz": predictions,
        }
        if glb:
            values["scene.glb"] = b"scene.glb bytes for testing"
        values.update(overrides)
        for name, data in values.items():
            (staging / name).write_bytes(data)
        if glb is False and "scene.glb" in values:
            pass

    def successful_prepared_launch(
        self, *, frame_count=3, config=None, token="tok00000001", **planner_kwargs
    ):
        if config is None:
            config = self.make_config(frame_group_size=max(3, frame_count))
        snapshot_uri, manifest_sha256 = make_snapshot(
            self.snapshots_root, frame_count=frame_count
        )
        planner = self.make_planner(config, token=token, **planner_kwargs)
        prepared = planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.track(prepared.result_transaction)
        return prepared


class ProductionConfigTest(ProductionTestBase):
    def test_valid_config_preflights_and_is_immutable(self):
        config = self.make_config()
        self.assertEqual(config.profile_name, "local_docker_test_candidate")
        self.assertEqual(config.image_resolution, 4)
        with self.assertRaises(AttributeError):
            config.image_resolution = 8

    def test_preflight_rejects_bool_as_int_and_nonfinite_values(self):
        cases = (
            {"image_resolution": True},
            {"frame_group_size": False},
            {"aggregator_frame_chunk_size": True},
            {"aggregator_frame_chunk_size": -1},
            {"max_result_pixels": 1},
            {"max_result_file_bytes": 0},
            {"max_result_total_bytes": -1},
            {"max_log_bytes": float("nan")},
            {"max_points_k": True},
            {"verifier_timeout_sec": float("inf")},
            {"verifier_timeout_sec": 0},
            {"worker_timeout_sec": 0},
            {"term_grace_sec": float("nan")},
            {"conf_thres": 2.0},
            {"conf_thres": 101},
            {"conf_thres": float("inf")},
            {"load_mode": "balanced_plus"},
            {"require_glb": 1},
            {"show_cameras": 0},
            {"profile_name": "bad name"},
            {"required_device_sku": ""},
            {"required_device_sku": "bad sku"},
            {"required_device_sku": True},
            {"algorithm_revision": ""},
            {"ros_revision": None},
            {"allowed_snapshot_roots": ()},
            {"allowed_snapshot_roots": [self.snapshots_root]},
        )
        for index, overrides in enumerate(cases):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ProductionConfigError):
                    self.make_config(**overrides)

    def test_preflight_rejects_missing_or_unsafe_runtime_paths(self):
        self.make_config()
        missing_script = self.root / "missing.py"
        symlink = self.root / "reconstruct-link"
        symlink.symlink_to(self.reconstruct_script)
        symlink_root = self.root / "result-link"
        symlink_root.symlink_to(self.result_root)
        cases = (
            {"reconstruct_script": str(missing_script)},
            {"validator_script": str(missing_script)},
            {"checkpoint_path": str(missing_script)},
            {"supervisor_python": str(missing_script)},
            {"worker_python": str(missing_script)},
            {"reconstruct_script": str(symlink)},
            {"checkpoint_path": str(symlink)},
            {"result_root": str(symlink_root)},
            {"result_root": str(self.root / "absent-results")},
            {"checkpoint_path": str(self.checkpoint) + "/../checkpoint.pt"},
        )
        for index, overrides in enumerate(cases):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ProductionConfigError):
                    self.make_config(**overrides)

    def test_preflight_rejects_checkpoint_digest_mismatch(self):
        with self.assertRaises(ProductionConfigError) as context:
            self.make_config(checkpoint_sha256="b" * 64)
        self.assertIn("checkpoint digest mismatch", str(context.exception))

    def test_preflight_rejects_frame_cap_and_pixel_inconsistency(self):
        tight_snapshot_limits = SnapshotSafetyLimits(
            max_frame_count=2,
            max_manifest_entries=64,
            max_metadata_bytes=1024 * 1024,
            max_file_bytes=1024 * 1024,
            max_total_bytes=8 * 1024 * 1024,
            max_scene_id_bytes=128,
            max_relative_path_bytes=1024,
        )
        cases = (
            {"snapshot_limits": tight_snapshot_limits, "frame_group_size": 3},
            {"max_result_pixels": 31},
            {
                "max_result_file_bytes": 64 * 1024,
                "max_result_total_bytes": 32 * 1024,
            },
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ProductionConfigError):
                    self.make_config(**overrides)

    def test_preflight_rejects_invalid_resource_limits(self):
        bad_limits = ResourceLimits(
            min_mem_available_bytes=1000,
            max_swap_used_bytes=1000,
            max_swap_growth_bytes=-1,
            min_snapshot_free_bytes=1000,
            min_result_free_bytes=1000,
            swap_sample_window_sec=0.01,
        )
        with self.assertRaises(ProductionConfigError):
            self.make_config(resource_limits=bad_limits)

    def test_planner_requires_config_type(self):
        with self.assertRaises(TypeError):
            ProductionPlanner(object())


class ProductionPlannerTest(ProductionTestBase):
    def test_failed_staging_retention_is_opt_in_and_forwarded(self):
        self.assertFalse(self.successful_prepared_launch().preserve_uncommitted_staging)
        enabled = self.make_config(preserve_failed_worker_staging=True)
        self.assertTrue(self.successful_prepared_launch(config=enabled).preserve_uncommitted_staging)
        with self.assertRaises(ProductionConfigError):
            self.make_config(preserve_failed_worker_staging="true")

    def test_attention_backend_reaches_cold_and_warm_workers(self):
        self.assertNotIn("--attention_backend", self.successful_prepared_launch().command)
        config = self.make_config(attention_backend="flash_attn_v1")
        command = self.successful_prepared_launch(config=config).command
        self.assertEqual(command[command.index("--attention_backend") + 1], "flash_attn_v1")
        config = self.make_config(attention_backend="flash_attn_v1", warm_worker_enabled=True,
                                  warm_worker_script=str(self.reconstruct_script))
        command = warm_worker_command(config)
        self.assertEqual(command[command.index("--attention-backend") + 1], "flash_attn_v1")
        with self.assertRaisesRegex(ProductionConfigError, "attention_backend"):
            self.make_config(attention_backend="unknown")

    def test_chunk_parameter_reaches_cold_and_warm_workers(self):
        config = self.make_config(aggregator_frame_chunk_size=10)
        prepared = self.successful_prepared_launch(config=config)
        command = prepared.command
        self.assertEqual(command[command.index("--aggregator_frame_chunk_size") + 1], "10")
        warm_config = self.make_config(
            aggregator_frame_chunk_size=10, warm_worker_enabled=True,
            warm_worker_script=str(self.reconstruct_script),
        )
        warm = warm_worker_command(warm_config)
        self.assertEqual(warm[warm.index("--aggregator-frame-chunk-size") + 1], "10")

    def test_prepared_launch_command_binds_explicit_worker_argv_without_shell(self):
        prepared = self.successful_prepared_launch(frame_count=3)
        command = prepared.command
        self.assertIsInstance(command, tuple)
        self.assertEqual(command[0], self.python)
        self.assertTrue(command[1].endswith("worker_runner.py"))
        self.assertEqual(command[2], "--log-path")
        self.assertEqual(
            command[3],
            str(prepared.result_transaction.staging_path / "run.log"),
        )
        self.assertEqual(command[4], "--")
        self.assertEqual(command[5], self.python)
        self.assertEqual(command[6], str(self.reconstruct_script))
        self.assertIn("--supervisor_staging", command)
        self.assertEqual(
            command[command.index("--output_dir") + 1],
            str(prepared.result_transaction.staging_path),
        )
        self.assertEqual(command[command.index("--image_resolution") + 1], "4")
        self.assertEqual(command[command.index("--frame_group_size") + 1], "3")
        self.assertEqual(
            command[command.index("--checkpoint_sha256") + 1],
            sha256_bytes(self.checkpoint.read_bytes()),
        )
        self.assertIn("--no_glb", command)
        self.assertEqual(command.count("--input_image"), 3)
        inputs = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--input_image"
        ]
        self.assertEqual(
            inputs,
            [
                str(self.snapshots_root / "scene-001" / "images" / f"frame_{i:03d}.png")
                for i in range(3)
            ],
        )

    def test_oversized_snapshot_uses_deterministic_ordered_selection(self):
        cases = (
            (7, 3, (0, 3, 6)),
            (6, 4, (0, 2, 3, 5)),
            (5, 2, (0, 4)),
            (10, 1, (0,)),
        )
        for frame_count, group_size, expected in cases:
            with self.subTest(frame_count=frame_count, group_size=group_size):
                snapshot_uri, manifest_sha256 = make_snapshot(
                    self.snapshots_root,
                    scene_id=f"sel-{frame_count}-{group_size}",
                    frame_count=frame_count,
                )
                config = self.make_config(frame_group_size=group_size)
                planner = self.make_planner(config)
                prepared = planner(self.make_goal(snapshot_uri, manifest_sha256))
                self.track(prepared.result_transaction)
                inputs = [
                    prepared.command[index + 1]
                    for index, value in enumerate(prepared.command)
                    if value == "--input_image"
                ]
                self.assertEqual(
                    [Path(path).name for path in inputs],
                    [f"frame_{index:03d}.png" for index in expected],
                )
                self.assertEqual(
                    _selected_indices(frame_count, group_size), expected
                )

    def test_duplicate_selected_basenames_are_rejected_before_allocation(self):
        snapshot_uri, manifest_sha256 = make_snapshot(
            self.snapshots_root, scene_id="dup", frame_count=2, duplicate_basenames=True
        )
        planner = self.make_planner()
        with self.assertRaises(PreparationError) as context:
            planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.assertEqual(context.exception.code, INVALID_GOAL)
        self.assertIn("duplicate basenames", str(context.exception))

    def test_snapshot_and_resource_error_mappings_are_stable(self):
        missing_uri = str(self.snapshots_root / "absent")
        bad_digest_uri, _ = make_snapshot(self.snapshots_root, scene_id="bad-digest")
        outside = self.root / "outside"
        outside.mkdir()
        cases = (
            (
                self.make_goal(missing_uri, "a" * 64),
                PreparationError,
                INPUT_NOT_FOUND,
            ),
            (
                self.make_goal(bad_digest_uri, "b" * 64),
                PreparationError,
                INVALID_GOAL,
            ),
            (
                self.make_goal(str(outside), "a" * 64),
                PreparationError,
                INVALID_GOAL,
            ),
        )
        for index, (goal, error_type, code) in enumerate(cases):
            with self.subTest(index=index):
                planner = self.make_planner()
                with self.assertRaises(error_type) as context:
                    planner(goal)
                self.assertEqual(context.exception.code, code)

    def test_resource_busy_propagates_untouched(self):
        def busy(limits, snapshot_root, result_root):
            limits = ResourceLimits(1, 1, 1, 1, 1, 1.0)
            sample = ResourceSample(0, 0, 0, 0, 0, 1, 1, 1.0)
            raise ResourceBusyError(
                ResourceDecision(False, "LOW_MEM_AVAILABLE", "low", sample, limits)
            )

        planner = self.make_planner(resource_enforcer=busy)
        snapshot_uri, manifest_sha256 = make_snapshot(self.snapshots_root)
        with self.assertRaises(ResourceBusyError):
            planner(self.make_goal(snapshot_uri, manifest_sha256))

    def test_resource_probe_failure_maps_to_internal_error(self):
        from drone3d_reconstruction_core.resource_gate import ResourceGateError

        def broken(limits, snapshot_root, result_root):
            raise ResourceGateError("probe failed")

        planner = self.make_planner(resource_enforcer=broken)
        snapshot_uri, manifest_sha256 = make_snapshot(self.snapshots_root)
        with self.assertRaises(PreparationError) as context:
            planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.assertEqual(context.exception.code, INTERNAL_ERROR)

    def test_transaction_collision_maps_to_io_error(self):
        def colliding(trusted_root, final_name):
            raise ResultCollisionError("FINAL_EXISTS", "already there")

        planner = self.make_planner(transaction_factory=colliding)
        snapshot_uri, manifest_sha256 = make_snapshot(self.snapshots_root)
        with self.assertRaises(PreparationError) as context:
            planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.assertEqual(context.exception.code, IO_ERROR)
        self.assertIn("FINAL_EXISTS", str(context.exception))

    def test_post_allocation_failure_cleans_staging_identity_safely(self):
        planner = self.make_planner()
        snapshot_uri, manifest_sha256 = make_snapshot(self.snapshots_root)
        with mock.patch(
            "drone3d_reconstruction_core.production.ProductionCompletionVerifier",
            side_effect=ValueError("injected construction failure"),
        ):
            with self.assertRaises(ValueError):
                planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.assertEqual(
            [path.name for path in self.result_root.iterdir()], []
        )

    def test_unique_final_names_use_scene_digest_selection_and_token(self):
        prepared = self.successful_prepared_launch(frame_count=3, token="tok00000001")
        name = prepared.result_transaction.final_path.name
        self.assertTrue(name.startswith("scene-001-"))
        self.assertIn("tok00000001", name)
        self.assertNotIn("..", name)
        self.assertNotIn("/", name)


    def test_per_goal_runtime_overrides_flow_to_command_and_verifier(self):
        config = self.make_config(image_resolution=4, worker_timeout_sec=100.0)
        snapshot_uri, manifest_sha256 = make_snapshot(
            self.snapshots_root, frame_count=3
        )
        planner = self.make_planner(config)
        prepared = planner(
            self.make_goal(snapshot_uri, manifest_sha256),
            image_resolution=5,
            worker_timeout_sec=42.5,
        )
        self.assertIn("--image_resolution", prepared.command)
        self.assertEqual(
            prepared.command[prepared.command.index("--image_resolution") + 1],
            "5",
        )
        self.assertEqual(
            prepared.verify_completion._config.image_resolution, 5
        )
        self.assertEqual(
            prepared.verify_completion._config.worker_timeout_sec, 42.5
        )
        self.track(prepared.result_transaction)

    def test_per_goal_runtime_override_invalid_is_rejected_as_invalid_goal(self):
        config = self.make_config()
        snapshot_uri, manifest_sha256 = make_snapshot(
            self.snapshots_root, frame_count=3
        )
        planner = self.make_planner(config)
        from drone3d_reconstruction_core.preparation import PreparationError

        with self.assertRaises(PreparationError):
            planner(
                self.make_goal(snapshot_uri, manifest_sha256),
                image_resolution=True,
            )

class CompletionVerifierTest(ProductionTestBase):
    def make_successful_verifier(self, *, require_glb=False, **config_overrides):
        config = self.make_config(require_glb=require_glb, **config_overrides)
        snapshot_uri, manifest_sha256 = make_snapshot(
            self.snapshots_root, frame_count=3
        )
        planner = self.make_planner(config)
        prepared = planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.track(prepared.result_transaction)
        verifier = prepared.verify_completion
        return config, prepared, verifier

    def test_success_writes_atomic_result_json_and_verified_completion(self):
        config, prepared, verifier = self.make_successful_verifier(require_glb=False)
        staging = prepared.result_transaction.staging_path
        self.write_valid_staging(staging, ["frame_000.png", "frame_001.png", "frame_002.png"])

        finished = JobFinished("worker_exit", 0, 2.5, 7)
        completed = verifier(finished, threading.Event())

        self.assertEqual(completed.scene_id, "scene-001")
        self.assertEqual(completed.frame_count, 3)
        self.assertEqual(
            completed.result_manifest_uri,
            (prepared.result_transaction.final_path / "result.json").as_uri(),
        )
        self.assertEqual(completed.scene_glb_uri, "")
        self.assertRegex(completed.result_manifest_sha256, r"^[0-9a-f]{64}$")

        result_path = staging / "result.json"
        self.assertTrue(result_path.is_file())
        result_bytes = result_path.read_bytes()
        self.assertEqual(sha256_bytes(result_bytes), completed.result_manifest_sha256)
        payload = json.loads(result_bytes)
        self.assertEqual(payload["schema"], RESULT_SCHEMA)
        self.assertEqual(payload["schema_version"], RESULT_SCHEMA_VERSION)
        self.assertEqual(payload["scene_id"], "scene-001")
        self.assertEqual(payload["job_id"], 7)
        self.assertEqual(payload["patch_id"], prepared.result_transaction.final_path.name)
        self.assertEqual(payload["snapshot"]["frame_count"], 3)
        self.assertRegex(payload["snapshot"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(payload["actual_frame_count"], 3)
        self.assertFalse(payload["frame_stamps_available"])
        self.assertEqual(payload["selected_frames"][0]["basename"], "frame_000.png")
        self.assertIsNone(payload["selected_frames"][0]["stamp"])
        self.assertEqual(payload["image_geometry"], {"width": 4, "height": 4})
        self.assertEqual(payload["checkpoint"]["sha256"], config.checkpoint_sha256)
        self.assertEqual(payload["algorithm_revision"], "test-algorithm-revision")
        self.assertEqual(payload["ros_revision"], "test-ros-revision")
        self.assertEqual(payload["profile"]["name"], "local_docker_test_candidate")
        self.assertEqual(payload["profile"]["required_device_sku"], "docker_test_candidate")
        self.assertEqual(payload["profile"]["frame_group_size"], 3)
        self.assertEqual(payload["worker_exit"], {"cause": "worker_exit", "returncode": 0})
        self.assertEqual(payload["stage_timing"]["worker_wall_time_sec"], 2.5)
        self.assertGreaterEqual(payload["stage_timing"]["verifier_wall_time_sec"], 0.0)
        self.assertIsNone(payload["stage_timing"]["commit_wall_time_sec"])
        self.assertEqual(payload["coordinate_state"], COORDINATE_STATE)
        self.assertFalse(payload["metric_scale_valid"])
        self.assertFalse(payload["map_alignment_valid"])
        self.assertIn("run.log", payload["artifacts"])
        self.assertIn("manifest.txt", payload["artifacts"])
        self.assertIn("predictions.npz", payload["artifacts"])
        self.assertNotIn("result.json", payload["artifacts"])
        self.assertNotIn("scene.glb", payload["artifacts"])
        for name in ("run.log", "manifest.txt", "predictions.npz"):
            artifact = payload["artifacts"][name]
            self.assertEqual(
                artifact["size"], (staging / name).stat().st_size
            )
            self.assertEqual(
                artifact["sha256"],
                sha256_bytes((staging / name).read_bytes()),
            )
            self.assertEqual(
                artifact["uri"],
                (prepared.result_transaction.final_path / name).as_uri(),
            )
        self.assertEqual(
            sorted(path.name for path in staging.iterdir()),
            sorted(["run.log", "manifest.txt", "predictions.npz", "result.json"]),
        )

    def test_result_json_counts_toward_max_result_total_bytes(self):
        config = self.make_config(
            max_result_file_bytes=8192,
            max_result_total_bytes=9000,
        )
        snapshot_uri, manifest_sha256 = make_snapshot(
            self.snapshots_root, frame_count=3
        )
        planner = self.make_planner(config)
        prepared = planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.track(prepared.result_transaction)
        staging = prepared.result_transaction.staging_path
        self.write_valid_staging(
            staging,
            ["frame_000.png", "frame_001.png", "frame_002.png"],
            **{"predictions.npz": b"x" * 6500},
        )
        verifier = prepared.verify_completion
        with self.assertRaises(CompletionError) as context:
            verifier(JobFinished("worker_exit", 0, 0.1, 1), threading.Event())
        self.assertEqual(context.exception.code, ALGORITHM_FAILED)
        self.assertIn("max_result_total_bytes", str(context.exception))
        self.assertFalse(prepared.result_transaction.committed)

    def test_v2_snapshot_stamp_pose_metadata_reaches_result_json(self):
        config = self.make_config(frame_group_size=3)
        snapshot_uri, manifest_sha256 = make_snapshot_v2(
            self.snapshots_root, frame_count=3
        )
        planner = self.make_planner(config)
        prepared = planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.track(prepared.result_transaction)
        staging = prepared.result_transaction.staging_path
        self.write_valid_staging(
            staging, ["frame_000.png", "frame_001.png", "frame_002.png"]
        )
        verifier = prepared.verify_completion
        verifier(JobFinished("worker_exit", 0, 1.0, 1), threading.Event())
        payload = json.loads((staging / "result.json").read_text())
        self.assertTrue(payload["frame_stamps_available"])
        self.assertEqual(payload["snapshot"]["schema_version"], 2)
        self.assertEqual(payload["snapshot"]["camera_name"], "camera4")
        self.assertEqual(
            payload["snapshot"]["first_stamp"], {"sec": 100, "nsec": 0}
        )
        self.assertTrue(payload["snapshot"]["has_odometry"])
        self.assertEqual(
            [frame["stamp"] for frame in payload["selected_frames"]],
            [{"sec": 100, "nsec": 0}, {"sec": 101, "nsec": 0}, {"sec": 102, "nsec": 0}],
        )
        self.assertEqual(
            [frame["pose_valid"] for frame in payload["selected_frames"]],
            [True, True, False],
        )

    def test_v2_subsampled_snapshot_keeps_original_stamp_identity(self):
        config = self.make_config(frame_group_size=2)
        snapshot_uri, manifest_sha256 = make_snapshot_v2(
            self.snapshots_root, frame_count=3
        )
        planner = self.make_planner(config)
        prepared = planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.track(prepared.result_transaction)
        staging = prepared.result_transaction.staging_path
        self.write_valid_staging(staging, ["frame_000.png", "frame_002.png"])
        verifier = prepared.verify_completion
        verifier(JobFinished("worker_exit", 0, 1.0, 1), threading.Event())
        payload = json.loads((staging / "result.json").read_text())
        self.assertEqual(
            [frame["index"] for frame in payload["selected_frames"]],
            [0, 2],
        )
        self.assertEqual(
            [frame["stamp"] for frame in payload["selected_frames"]],
            [{"sec": 100, "nsec": 0}, {"sec": 102, "nsec": 0}],
        )
        self.assertEqual(
            [frame["basename"] for frame in payload["selected_frames"]],
            ["frame_000.png", "frame_002.png"],
        )

    def test_glb_success_publishes_glb_uri_and_artifact(self):
        config, prepared, verifier = self.make_successful_verifier(require_glb=True)
        staging = prepared.result_transaction.staging_path
        self.write_valid_staging(
            staging,
            ["frame_000.png", "frame_001.png", "frame_002.png"],
            glb=True,
        )
        completed = verifier(JobFinished("worker_exit", 0, 1.0, 2), threading.Event())
        self.assertEqual(
            completed.scene_glb_uri,
            (prepared.result_transaction.final_path / "scene.glb").as_uri(),
        )
        payload = json.loads((staging / "result.json").read_bytes())
        self.assertIn("scene.glb", payload["artifacts"])

    def test_verifier_cli_argv_and_popen_boundary(self):
        config, prepared, verifier = self.make_successful_verifier()
        probe = self.root / "validator-argv.json"
        self.validator_script.write_text(
            make_fake_validator("success", probe_path=str(probe))
        )
        staging = prepared.result_transaction.staging_path
        self.write_valid_staging(staging, ["frame_000.png", "frame_001.png", "frame_002.png"])
        verifier(JobFinished("worker_exit", 0, 0.1, 1), threading.Event())

        argv = json.loads(probe.read_text())
        self.assertEqual(argv[0], str(self.validator_script))
        self.assertEqual(argv[1], "--output_dir")
        self.assertEqual(argv[2], str(staging))
        self.assertIn("--no_require_glb", argv)
        self.assertEqual(argv[argv.index("--max_frames") + 1], "3")
        self.assertEqual(argv[argv.index("--max_pixels") + 1], "64")
        expected = ["frame_000.png", "frame_001.png", "frame_002.png"]
        got = []
        for index, value in enumerate(argv):
            if value == "--expected_image_basename":
                got.append(argv[index + 1])
        self.assertEqual(got, expected)

    def test_validator_spawn_sanitizes_supervisor_python_environment(self):
        config, prepared, verifier = self.make_successful_verifier()
        calls = []

        def recording_popen(argv, **kwargs):
            calls.append((list(argv), dict(kwargs)))
            return mock.Mock()

        verifier._popen_factory = recording_popen
        with mock.patch.dict(
            "os.environ",
            {
                "PYTHONPATH": "/bad/pythonpath",
                "PYTHONHOME": "/bad/pythonhome",
                "CONDA_PREFIX": "/bad/conda",
                "KEEP_ME": "kept",
            },
        ):
            verifier._spawn_validator(["validator"], io.BytesIO())
        self.assertEqual(len(calls), 1)
        _argv, kwargs = calls[0]
        env = kwargs["env"]
        self.assertIsInstance(env, dict)
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertNotIn("CONDA_PREFIX", env)
        self.assertEqual(env["KEEP_ME"], "kept")

    def test_validator_rejection_and_malformed_output_fail_closed(self):
        behaviors = ("reject", "malformed")
        for behavior in behaviors:
            with self.subTest(behavior=behavior):
                config, prepared, verifier = self.make_successful_verifier()
                self.validator_script.write_text(make_fake_validator(behavior))
                staging = prepared.result_transaction.staging_path
                self.write_valid_staging(
                    staging,
                    ["frame_000.png", "frame_001.png", "frame_002.png"],
                )
                with self.assertRaises(CompletionError) as context:
                    verifier(JobFinished("worker_exit", 0, 0.1, 1), threading.Event())
                self.assertEqual(context.exception.code, ALGORITHM_FAILED)
                self.assertFalse((staging / "result.json").exists())
                self.assertFalse((staging / ".verifier-stderr.txt").exists())
                self.assertFalse(prepared.result_transaction.committed)

    def test_validator_spawn_failure_maps_to_io_error(self):
        config, prepared, verifier = self.make_successful_verifier()
        verifier._popen_factory = mock.Mock(side_effect=OSError("no exec"))
        with self.assertRaises(CompletionError) as context:
            verifier(JobFinished("worker_exit", 0, 0.1, 1), threading.Event())
        self.assertEqual(context.exception.code, IO_ERROR)
        self.assertFalse((prepared.result_transaction.staging_path / "result.json").exists())

    def test_validator_timeout_terminates_and_fails_algorithm(self):
        config = self.make_config(verifier_timeout_sec=0.1, term_grace_sec=0.05)
        self.validator_script.write_text(make_fake_validator("sleep"))
        snapshot_uri, manifest_sha256 = make_snapshot(self.snapshots_root, frame_count=3)
        planner = self.make_planner(config)
        prepared = planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.track(prepared.result_transaction)
        verifier = prepared.verify_completion
        verifier._poll_interval_sec = 0.005

        started = time.monotonic()
        with self.assertRaises(CompletionError) as context:
            verifier(JobFinished("worker_exit", 0, 0.1, 1), threading.Event())
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(context.exception.code, ALGORITHM_FAILED)
        self.assertIn("verifier_timeout_sec", str(context.exception))
        self.assertFalse((prepared.result_transaction.staging_path / "result.json").exists())

    def test_validator_cancellation_reaps_before_raising(self):
        config, prepared, verifier = self.make_successful_verifier()
        self.validator_script.write_text(make_fake_validator("sleep"))
        verifier._poll_interval_sec = 0.005
        cancellation_event = threading.Event()
        errors = []

        def run():
            try:
                verifier(JobFinished("worker_exit", 0, 0.1, 1), cancellation_event)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.05)
        cancellation_event.set()
        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], VerificationCancelled)
        self.assertFalse((prepared.result_transaction.staging_path / "result.json").exists())

    def test_verifier_stderr_bound_after_exit_fails_closed(self):
        config = self.make_config(max_log_bytes=1024)
        self.validator_script.write_text(make_fake_validator("loud_stderr"))
        snapshot_uri, manifest_sha256 = make_snapshot(
            self.snapshots_root, frame_count=3
        )
        planner = self.make_planner(config)
        prepared = planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.track(prepared.result_transaction)
        staging = prepared.result_transaction.staging_path
        self.write_valid_staging(
            staging, ["frame_000.png", "frame_001.png", "frame_002.png"]
        )
        verifier = prepared.verify_completion
        with self.assertRaises(CompletionError) as context:
            verifier(JobFinished("worker_exit", 0, 0.1, 1), threading.Event())
        self.assertEqual(context.exception.code, ALGORITHM_FAILED)
        self.assertIn("stderr exceeded", str(context.exception))
        self.assertFalse((staging / "result.json").exists())
        self.assertFalse((staging / ".verifier-stderr.txt").exists())

    def test_verifier_stdout_bound_fails_closed(self):
        config = self.make_config(max_log_bytes=1024)
        self.validator_script.write_text(make_fake_validator("loud"))
        snapshot_uri, manifest_sha256 = make_snapshot(self.snapshots_root, frame_count=3)
        planner = self.make_planner(config)
        prepared = planner(self.make_goal(snapshot_uri, manifest_sha256))
        self.track(prepared.result_transaction)
        verifier = prepared.verify_completion
        verifier._poll_interval_sec = 0.005
        with self.assertRaises(CompletionError) as context:
            verifier(JobFinished("worker_exit", 0, 0.1, 1), threading.Event())
        self.assertEqual(context.exception.code, ALGORITHM_FAILED)
        self.assertIn("stdout exceeded", str(context.exception))
        self.assertFalse((prepared.result_transaction.staging_path / "result.json").exists())

    def test_asset_corruption_matrix_fails_without_manifest(self):
        cases = (
            ("missing predictions", {"predictions.npz": None}),
            ("missing run.log", {"run.log": None}),
            ("bad run.log prefix", {"run.log": b"not the wrapper\n"}),
            ("empty predictions", {"predictions.npz": b""}),
            ("manifest mismatch", {"manifest.txt": b"other.png\n"}),
            ("oversized predictions", {"predictions.npz": b"x" * 4096}),
        )
        for label, overrides in cases:
            with self.subTest(label=label):
                config = self.make_config(max_result_file_bytes=1024)
                snapshot_uri, manifest_sha256 = make_snapshot(
                    self.snapshots_root, frame_count=3
                )
                planner = self.make_planner(config)
                prepared = planner(self.make_goal(snapshot_uri, manifest_sha256))
                self.track(prepared.result_transaction)
                staging = prepared.result_transaction.staging_path
                self.write_valid_staging(
                    staging, ["frame_000.png", "frame_001.png", "frame_002.png"]
                )
                for name, replacement in overrides.items():
                    if replacement is None:
                        (staging / name).unlink()
                    else:
                        (staging / name).write_bytes(replacement)
                verifier = prepared.verify_completion
                with self.assertRaises(CompletionError) as context:
                    verifier(JobFinished("worker_exit", 0, 0.1, 1), threading.Event())
                self.assertEqual(context.exception.code, ALGORITHM_FAILED)
                self.assertFalse((staging / "result.json").exists())

    def test_symlink_asset_and_unexpected_glb_fail_closed(self):
        config, prepared, verifier = self.make_successful_verifier(require_glb=False)
        staging = prepared.result_transaction.staging_path
        self.write_valid_staging(staging, ["frame_000.png", "frame_001.png", "frame_002.png"])
        target = staging / "run.log"
        link = staging / "predictions.npz"
        link.unlink()
        link.symlink_to(target)
        with self.assertRaises(CompletionError) as context:
            verifier(JobFinished("worker_exit", 0, 0.1, 1), threading.Event())
        self.assertEqual(context.exception.code, ALGORITHM_FAILED)

        config, prepared, verifier = self.make_successful_verifier(require_glb=False)
        staging = prepared.result_transaction.staging_path
        self.write_valid_staging(
            staging,
            ["frame_000.png", "frame_001.png", "frame_002.png"],
            glb=True,
        )
        with self.assertRaises(CompletionError) as context:
            verifier(JobFinished("worker_exit", 0, 0.1, 1), threading.Event())
        self.assertEqual(context.exception.code, ALGORITHM_FAILED)
        self.assertIn("scene.glb", str(context.exception))


if __name__ == "__main__":
    unittest.main()
