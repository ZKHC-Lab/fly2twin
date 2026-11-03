#!/usr/bin/env python3
"""ROS transport error matrix for the production planner and verifier.

The real Omega model cannot run in the ROS1 container (no torch/checkpoint), so
the worker executable is a synthetic Omega CLI run through the real
worker_runner and JobController.  The completion verifier is real: scenarios
that reach exit-zero verification use the actual ``third_party/vggt-omega/result_validator.py``
CLI, which validates a real float32 NPZ schema and rejects corrupted artifacts.
"""

from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
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

from drone3d_reconstruction_core import (
    ProductionConfig,
    ProductionPlanner,
    ResourceBusyError,
    ResourceDecision,
    ResourceLimits,
    ResourceSample,
    SnapshotSafetyLimits,
)
from drone3d_reconstruction_core.result_transaction import (
    ResultCollisionError,
)

ACTION_NAME = "/test_reconstruct_production_segment"
REAL_VALIDATOR = (
    Path(__file__).resolve().parents[4] / "third_party" / "vggt-omega" / "result_validator.py"
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def make_snapshot(snapshots_root, scene_id="prod-e2e", frame_count=3):
    scene = snapshots_root / scene_id
    images = scene / "images"
    images.mkdir(parents=True)
    image_relatives = []
    for index in range(frame_count):
        relative = f"images/frame_{index:03d}.png"
        path = scene.joinpath(*Path(relative).parts)
        path.write_bytes(f"image bytes {scene_id} {index}".encode("utf-8"))
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
    manifest_rows = {"snapshot.json": snapshot_bytes, "frames.csv": frames_bytes}
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


def fake_omega_script(script_path):
    script_path.write_text(
        r'''import json, os, signal, sys
from pathlib import Path
mode = os.environ.get("DRONE3D_FAKE_OMEGA_MODE", "success")
if mode == "exit137":
    os._exit(137)
if mode == "exit23":
    sys.exit(23)
output_dir = None
images = []
resolution = 4
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--output_dir":
        output_dir = Path(args[i+1]); i += 2
    elif args[i] == "--input_image":
        images.append(Path(args[i+1])); i += 2
    elif args[i] == "--image_resolution":
        resolution = int(args[i+1]); i += 2
    else:
        i += 1
if mode == "sleep":
    import time
    time.sleep(60)
    sys.exit(0)
if mode == "slow_valid":
    import time
    time.sleep(2)
(output_dir / "manifest.txt").write_text(
    "\n".join(p.name for p in images) + "\n", encoding="utf-8")
if mode == "bad_npz":
    (output_dir / "predictions.npz").write_bytes(b"corrupt npz payload")
    print("wrote corrupt predictions.npz")
    sys.exit(0)
import numpy as np
frames = len(images)
height = width = resolution
depth = np.full((frames, height, width, 1), 0.5, dtype=np.float32)
depth_conf = np.full((frames, height, width), 1.0, dtype=np.float32)
rgb = np.full((frames, 3, height, width), 0.25, dtype=np.float32)
extrinsic = np.zeros((frames, 3, 4), dtype=np.float32)
extrinsic[:, :3, :3] = np.eye(3, dtype=np.float32)
extrinsic[:, :3, 3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
intrinsic = np.zeros((frames, 3, 3), dtype=np.float32)
intrinsic[:, 0, 0] = 2.0
intrinsic[:, 1, 1] = 2.0
intrinsic[:, 0, 2] = width / 2.0
intrinsic[:, 1, 2] = height / 2.0
intrinsic[:, 2, 2] = 1.0
y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
world_points = np.zeros((frames, height, width, 3), dtype=np.float32)
for frame in range(frames):
    frame_depth = depth[frame, ..., 0]
    rotation_t = extrinsic[frame, :3, :3].T
    translation = extrinsic[frame, :3, 3]
    fx, fy = intrinsic[frame, 0, 0], intrinsic[frame, 1, 1]
    cx, cy = intrinsic[frame, 0, 2], intrinsic[frame, 1, 2]
    camera_points = np.stack(
        ((x - cx) / fx * frame_depth,
         (y - cy) / fy * frame_depth,
         frame_depth), axis=-1)
    world_points[frame] = np.einsum(
        "ij,hwj->hwi", rotation_t,
        (camera_points - translation).astype(np.float32)).astype(np.float32)
np.savez(str(output_dir / "predictions.npz"),
         depth=depth, depth_conf=depth_conf, images=rgb,
         extrinsic=extrinsic, intrinsic=intrinsic,
         world_points_from_depth=world_points)
print("fake omega wrote valid schema")
''',
        encoding="utf-8",
    )


def fake_sleep_validator_script(script_path):
    script_path.write_text(
        "import time\nprint('fake validator sleeping')\ntime.sleep(60)\n",
        encoding="utf-8",
    )


def passing_resource_enforcer(limits, snapshot_root, result_root):
    sample = ResourceSample(0, 0, 0, 0, 0, 0, 0, limits.swap_sample_window_sec)
    return ResourceDecision(True, None, "injected pass", sample, limits)


def busy_result_disk_enforcer(limits, snapshot_root, result_root):
    limits = ResourceLimits(1, 1, 1, 1, 1, 1.0)
    sample = ResourceSample(0, 0, 0, 0, 0, 1, 1, 1.0)
    raise ResourceBusyError(
        ResourceDecision(False, "LOW_RESULT_DISK", "low result disk", sample, limits)
    )


class ProductionActionServerRostest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.snapshots_root = self.root / "snapshots"
        self.snapshots_root.mkdir()
        self.result_root = self.root / "results"
        self.result_root.mkdir()
        self.checkpoint = self.root / "checkpoint.pt"
        checkpoint_bytes = b"e2e explicit checkpoint bytes"
        self.checkpoint.write_bytes(checkpoint_bytes)
        self.checkpoint_sha256 = sha256_bytes(checkpoint_bytes)
        self.fake_omega = self.root / "fake_omega.py"
        fake_omega_script(self.fake_omega)
        self.fake_validator = self.root / "fake_validator.py"
        fake_sleep_validator_script(self.fake_validator)
        self.snapshot_uri, self.manifest_sha256 = make_snapshot(
            self.snapshots_root, frame_count=3
        )
        self.planner = self.make_planner(REAL_VALIDATOR)
        self.server = None
        self.client = None
        self.env_restore = {
            "DRONE3D_FAKE_OMEGA_MODE": os.environ.get("DRONE3D_FAKE_OMEGA_MODE")
        }
        self._start_server()

    def tearDown(self):
        for key, value in self.env_restore.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self.server is not None:
            self.server.shutdown()
        self.temporary_directory.cleanup()

    def make_config(self, validator_script):
        return ProductionConfig(
            supervisor_python=str(Path(sys.executable).resolve()),
            worker_python=str(Path(sys.executable).resolve()),
            reconstruct_script=str(self.fake_omega),
            validator_script=str(validator_script),
            checkpoint_path=str(self.checkpoint),
            checkpoint_sha256=self.checkpoint_sha256,
            result_root=str(self.result_root),
            allowed_snapshot_roots=(str(self.snapshots_root),),
            snapshot_limits=SnapshotSafetyLimits(
                max_frame_count=8,
                max_manifest_entries=64,
                max_metadata_bytes=1024 * 1024,
                max_file_bytes=1024 * 1024,
                max_total_bytes=8 * 1024 * 1024,
                max_scene_id_bytes=128,
                max_relative_path_bytes=1024,
            ),
            resource_limits=ResourceLimits(
                min_mem_available_bytes=1,
                max_swap_used_bytes=sys.maxsize,
                max_swap_growth_bytes=sys.maxsize,
                min_snapshot_free_bytes=1,
                min_result_free_bytes=1,
                swap_sample_window_sec=0.01,
            ),
            image_resolution=4,
            frame_group_size=3,
            max_result_pixels=64,
            max_result_file_bytes=256 * 1024,
            max_result_total_bytes=1024 * 1024,
            max_log_bytes=64 * 1024,
            verifier_timeout_sec=2.0,
            worker_timeout_sec=4.0,
            term_grace_sec=0.1,
            load_mode="max_size",
            require_glb=False,
            show_cameras=True,
            conf_thres=50.0,
            max_points_k=5,
            profile_name="local_docker_test_candidate",
            required_device_sku="docker_test_candidate",
            algorithm_revision="e2e-algorithm-revision",
            ros_revision="e2e-ros-revision",
        )

    def make_planner(self, validator_script, *, token=None, **planner_kwargs):
        config = self.make_config(validator_script)
        if token is None:
            token = secrets_token()
        kwargs = dict(
            token_source=lambda token=token: token,
            resource_enforcer=passing_resource_enforcer,
        )
        kwargs.update(planner_kwargs)
        return ProductionPlanner(config, **kwargs)

    def _start_server(self):
        if self.server is not None:
            self.server.shutdown()
        from drone3d_reconstruction_core.reconstruct_action_server import (
            ReconstructActionServer,
        )

        self.action_name = unique_action_name()
        self.server = ReconstructActionServer(
            action_name=self.action_name,
            command_builder=self.planner,
            worker_timeout_sec=4.0,
            term_grace_sec=0.1,
        )
        self.client = actionlib.SimpleActionClient(
            self.action_name, ReconstructSegmentAction
        )
        self.assertTrue(self.client.wait_for_server(rospy.Duration(3.0)))

    def _restart_with_planner(self, planner):
        if self.server is not None:
            self.server.shutdown()
        from drone3d_reconstruction_core.reconstruct_action_server import (
            ReconstructActionServer,
        )

        self.planner = planner
        self.action_name = unique_action_name()
        self.server = ReconstructActionServer(
            action_name=self.action_name,
            command_builder=planner,
            worker_timeout_sec=4.0,
            term_grace_sec=0.1,
        )
        self.client = actionlib.SimpleActionClient(
            self.action_name, ReconstructSegmentAction
        )
        self.assertTrue(self.client.wait_for_server(rospy.Duration(5.0)))

    def _send(self, mode, goal=None):
        if goal is None:
            goal = ReconstructSegmentGoal(
                snapshot_uri=self.snapshot_uri,
                manifest_sha256=self.manifest_sha256,
            )
        self.assertTrue(
            _wait_for_stage(self.server, "idle", 3.0)
            or not self.server.is_busy(),
            "server did not release the previous goal slot",
        )
        os.environ["DRONE3D_FAKE_OMEGA_MODE"] = mode
        self.client.send_goal(goal)
        self.assertTrue(self.client.wait_for_result(rospy.Duration(6.0)))
        return self.client.get_state(), self.client.get_result()

    def _assert_result_root_empty(self):
        self.assertEqual(sorted(path.name for path in self.result_root.iterdir()), [])

    def test_error_matrix_reaches_unique_terminals_without_half_products(self):
        # Missing snapshot -> REJECTED + INPUT_NOT_FOUND.
        state, result = self._send(
            "success",
            ReconstructSegmentGoal(
                snapshot_uri=str(self.snapshots_root / "missing"),
                manifest_sha256=self.manifest_sha256,
            ),
        )
        self.assertEqual(state, GoalStatus.REJECTED)
        self.assertEqual(result.error_code, ReconstructSegmentResult.INPUT_NOT_FOUND)
        self._assert_result_root_empty()

        # Digest mismatch -> REJECTED + INVALID_GOAL.
        state, result = self._send(
            "success",
            ReconstructSegmentGoal(
                snapshot_uri=self.snapshot_uri,
                manifest_sha256="a" * 64,
            ),
        )
        self.assertEqual(state, GoalStatus.REJECTED)
        self.assertEqual(result.error_code, ReconstructSegmentResult.INVALID_GOAL)
        self._assert_result_root_empty()

        # Disk admission denial -> REJECTED + BUSY.
        self._restart_with_planner(
            self.make_planner(
                REAL_VALIDATOR, resource_enforcer=busy_result_disk_enforcer
            )
        )
        state, result = self._send("success")
        self.assertEqual(state, GoalStatus.REJECTED)
        self.assertEqual(result.error_code, ReconstructSegmentResult.BUSY)
        self.assertIn("LOW_RESULT_DISK", result.message)
        self._assert_result_root_empty()

        # Result-name collision -> REJECTED + IO_ERROR.
        self._restart_with_planner(
            self.make_planner(
                REAL_VALIDATOR,
                transaction_factory=lambda root, name: (_ for _ in ()).throw(
                    ResultCollisionError("FINAL_EXISTS", "final exists")
                ),
            )
        )
        state, result = self._send("success")
        self.assertEqual(state, GoalStatus.REJECTED)
        self.assertEqual(result.error_code, ReconstructSegmentResult.IO_ERROR)
        self.assertIn("FINAL_EXISTS", result.message)
        self._assert_result_root_empty()

        # Normal crash -> ABORTED + ALGORITHM_FAILED.
        self._restart_with_planner(self.make_planner(REAL_VALIDATOR))
        state, result = self._send("exit23")
        self.assertEqual(state, GoalStatus.ABORTED)
        self.assertEqual(result.error_code, ReconstructSegmentResult.ALGORITHM_FAILED)
        self.assertIn("23", result.message)
        self._assert_result_root_empty()

        # OOM-style exit 137 -> ABORTED + ALGORITHM_FAILED.
        state, result = self._send("exit137")
        self.assertEqual(state, GoalStatus.ABORTED)
        self.assertEqual(result.error_code, ReconstructSegmentResult.ALGORITHM_FAILED)
        self.assertIn("137", result.message)
        self._assert_result_root_empty()

        # Corrupt NPZ rejected by the real validator -> ABORTED + ALGORITHM_FAILED.
        state, result = self._send("bad_npz")
        self.assertEqual(state, GoalStatus.ABORTED)
        self.assertEqual(result.error_code, ReconstructSegmentResult.ALGORITHM_FAILED)
        self.assertIn("result validator rejected", result.message)
        self._assert_result_root_empty()

        # Concurrent contender while worker is running -> REJECTED + BUSY,
        # first goal succeeds with verified metadata.
        self._restart_with_planner(self.make_planner(REAL_VALIDATOR, token="tok00000001"))
        contender = actionlib.SimpleActionClient(
            self.action_name, ReconstructSegmentAction
        )
        self.assertTrue(contender.wait_for_server(rospy.Duration(3.0)))
        # A slow-but-valid worker keeps the slot occupied deterministically.
        os.environ["DRONE3D_FAKE_OMEGA_MODE"] = "slow_valid"
        first_goal = ReconstructSegmentGoal(
            snapshot_uri=self.snapshot_uri,
            manifest_sha256=self.manifest_sha256,
        )
        # slow_valid mode falls back to valid output after a short pause.
        self.client.send_goal(first_goal)
        self.assertTrue(_wait_for_stage(self.server, "active", 2.0))
        contender.send_goal(
            ReconstructSegmentGoal(
                snapshot_uri=self.snapshot_uri,
                manifest_sha256=self.manifest_sha256,
            )
        )
        self.assertTrue(contender.wait_for_result(rospy.Duration(2.0)))
        self.assertEqual(contender.get_state(), GoalStatus.REJECTED)
        self.assertEqual(
            contender.get_result().error_code, ReconstructSegmentResult.BUSY
        )
        self.assertTrue(self.client.wait_for_result(rospy.Duration(6.0)))
        self.assertEqual(self.client.get_state(), GoalStatus.SUCCEEDED)
        first_result = self.client.get_result()
        self.assertEqual(first_result.error_code, ReconstructSegmentResult.OK)
        self.assertEqual(first_result.scene_id, "prod-e2e")
        self.assertEqual(first_result.frame_count, 3)
        self.assertRegex(first_result.result_manifest_sha256, r"^[0-9a-f]{64}$")
        committed_dirs = list(self.result_root.iterdir())
        self.assertEqual(len(committed_dirs), 1)
        final_result_json = committed_dirs[0] / "result.json"
        self.assertTrue(final_result_json.is_file())
        payload = json.loads(final_result_json.read_text(encoding="utf-8"))
        self.assertEqual(payload["scene_id"], "prod-e2e")
        self.assertEqual(payload["coordinate_state"], "MODEL_LOCAL_UNALIGNED")
        self.assertFalse(payload["metric_scale_valid"])
        self.assertFalse(payload["map_alignment_valid"])

        # Cancel during verification -> PREEMPTED + CANCELLED, no commit.  The
        # previously committed successful result must remain untouched and no
        # additional final directory may appear.
        self._restart_with_planner(
            self.make_planner(self.fake_validator, token="tok00000002")
        )
        committed_before_cancel = {
            path.name for path in self.result_root.iterdir()
        }
        state, result = self._send_and_cancel_during_verifying()
        self.assertEqual(state, GoalStatus.PREEMPTED)
        self.assertEqual(result.error_code, ReconstructSegmentResult.CANCELLED)
        self.assertEqual(
            {path.name for path in self.result_root.iterdir()},
            committed_before_cancel,
        )

    def _send_and_cancel_during_verifying(self):
        os.environ["DRONE3D_FAKE_OMEGA_MODE"] = "success"
        goal = ReconstructSegmentGoal(
            snapshot_uri=self.snapshot_uri,
            manifest_sha256=self.manifest_sha256,
        )
        self.client.send_goal(goal)
        self.assertTrue(_wait_for_stage(self.server, "verifying", 3.0))
        self.client.cancel_goal()
        self.assertTrue(self.client.wait_for_result(rospy.Duration(6.0)))
        return self.client.get_state(), self.client.get_result()


_token_sequence = [0]


def secrets_token():
    _token_sequence[0] += 1
    return f"e2etoken{_token_sequence[0]:08d}"


def unique_action_name():
    _token_sequence[0] += 1
    return f"{ACTION_NAME}_{_token_sequence[0]:04d}"


def _wait_for_stage(server, stage, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = server.diagnostics_snapshot()
        if snapshot.get("stage") == stage:
            return True
        time.sleep(0.01)
    return False


if __name__ == "__main__":
    rospy.init_node("test_reconstruct_production_action_server")
    rostest.rosrun(
        "drone3d_reconstruction_core",
        "test_reconstruct_production_action_server",
        ProductionActionServerRostest,
    )
