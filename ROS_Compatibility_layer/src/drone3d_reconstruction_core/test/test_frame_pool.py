"""Unit tests for the frame-pool core (import, scoring, NMS, synthetic snapshot)."""

from pathlib import Path
from types import SimpleNamespace
import shutil
import tempfile
import unittest

import numpy as np

from drone3d_reconstruction_core.frame_pool import (
    FramePoolCapacityError,
    FramePool,
    FramePoolConfig,
    FramePoolError,
    RECONSTRUCTION_MODE_FRAME_POOL,
    RECONSTRUCTION_MODE_PER_WINDOW,
)
from drone3d_reconstruction_core.snapshot_validator import (
    SnapshotSafetyLimits,
    validate_snapshot,
)


def _limits():
    return SnapshotSafetyLimits(
        max_frame_count=128,
        max_manifest_entries=1024,
        max_metadata_bytes=8 * 1024 * 1024,
        max_file_bytes=4 * 1024 * 1024,
        max_total_bytes=1024 * 1024 * 1024,
        max_scene_id_bytes=128,
        max_relative_path_bytes=1024,
    )


def _png(path: Path, value: int):
    import cv2
    image = ((np.indices((16, 16)).sum(axis=0) % 2) * 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("could not write test PNG")


def _snapshot(root: Path, count=4, positions=None):
    images = root / "images"
    images.mkdir(parents=True)
    paths = []
    stamps = []
    valid = []
    payloads = []
    for index in range(count):
        path = images / f"frame_{index:06d}.png"
        _png(path, index + 1)
        paths.append(path)
        stamps.append(SimpleNamespace(sec=100 + index, nsec=index))
        valid.append(True)
        if positions is not None:
            payloads.append(
                {
                    "header": {"frame_id": "map", "stamp": {"sec": 100 + index, "nsec": 0}},
                    "pose": {
                        "position": {
                            "x": float(positions[index][0]),
                            "y": float(positions[index][1]),
                            "z": float(positions[index][2]),
                        },
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                }
            )
        else:
            payloads.append(None)
    return SimpleNamespace(
        image_paths=tuple(paths),
        frame_stamps=tuple(stamps),
        pose_valid=tuple(valid),
        odometry_payloads=tuple(payloads),
    )


class FramePoolTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "pool"
        self.pool = FramePool(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_import_persist_and_reload(self):
        snapshot_root = Path(self.temp.name) / "snap"
        snapshot = _snapshot(snapshot_root)
        indices = self.pool.import_window(snapshot, 0)
        self.assertEqual(indices, [0, 1, 2, 3])
        self.assertEqual(len(self.pool.entries), 4)
        self.assertTrue((self.root / "images" / "frame_00000000.png").is_file())

        reloaded = FramePool(self.root)
        reloaded.load()
        self.assertEqual(len(reloaded.entries), 4)
        self.assertEqual(reloaded.entries[0].global_index, 0)
        self.assertEqual(reloaded.entries[-1].stamp_sec, 103)

    def test_import_persists_minimal_odometry_pose_for_overlap_alignment(self):
        snapshot = _snapshot(
            Path(self.temp.name) / "snap",
            positions=[(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)],
            count=2,
        )
        self.pool.import_window(snapshot, 0)
        reloaded = FramePool(self.root)
        reloaded.load()
        self.assertEqual(reloaded.entries[1].odom_frame_id, "map")
        self.assertEqual(reloaded.entries[1].odom_position, (1.0, 2.0, 3.0))
        self.assertEqual(
            reloaded.entries[1].odom_orientation, (0.0, 0.0, 0.0, 1.0)
        )

    def test_v4_auxiliary_survives_pool_reload_and_source_cleanup(self):
        import sys

        test_dir = str(Path(__file__).resolve().parent)
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)
        from test_capture import AuxiliaryContractV4Test

        context = AuxiliaryContractV4Test()
        context.setUp()
        self.addCleanup(context.doCleanups)
        self.addCleanup(context.tearDown)
        config = context.make_aux_config()
        from drone3d_reconstruction_core.capture import validate_capture_goal
        capture = context.make_session(
            config=config, goal=validate_capture_goal(context.make_goal(max_frames=2), config)
        )
        capture.on_tf(context.tf_message(0, parent="odom", child="body"), static=True)
        for stamp in (8, 9, 10):
            capture.on_tf(context.tf_message(stamp, parent="odom", child="body"))
        for nsec in (0, 500000000):
            capture.on_camera_info(context.camera_info_message(10, nsec))
            capture.on_odometry(context.odometry_message(10, nsec))
            depth = context.depth_message(10, nsec)
            depth.step, depth.data = 4, b"\x01\x00\x02\x00"
            capture.on_depth(depth)
            capture.on_image(context.image_message(10, nsec))
        self.assertIsNone(capture.error)
        capture.transaction.commit()
        snapshot = validate_snapshot(capture.sealed.snapshot_uri, capture.sealed.manifest_sha256,
                                     (config.snapshot_root,), config.snapshot_limits)
        self.pool.import_window(snapshot, 0)
        self.pool.retain(self.pool.entries[-1:])
        reloaded = FramePool(self.root)
        reloaded.load()
        shutil.rmtree(snapshot.canonical_root)
        uri, digest = reloaded.build_synthetic_snapshot(
            1, reloaded.entries, Path(self.temp.name) / "combined", "combined",
            camera_name=config.camera_name, image_topic=config.image_topic,
            camera_role=config.camera_role, camera_frame_id=config.camera_frame_id,
        )
        combined = validate_snapshot(uri, digest, (self.temp.name,), _limits())
        self.assertEqual(combined.snapshot_version, 4)
        self.assertEqual(combined.odometry_payloads, (snapshot.odometry_payloads[1],))
        self.assertEqual(combined.camera_info_payloads, (snapshot.camera_info_payloads[1],))
        self.assertEqual(combined.pose_valid, (snapshot.pose_valid[1],))
        self.assertEqual(combined.body_frame_id, snapshot.body_frame_id)
        self.assertTrue(combined.tf_entries)
        expected_tf = tuple(entry for entry in snapshot.tf_entries
                            if entry["static"] or entry["stamp"]["sec"] == 10)
        self.assertEqual(combined.tf_entries, expected_tf)
        self.assertEqual((combined.canonical_root / combined.depth_paths[0]).read_bytes(), b"\x01\x00\x02\x00")

    def test_selection_is_score_ranked_with_odom_voxel_nms(self):
        snapshot_root = Path(self.temp.name) / "snap"
        positions = [
            (0.0, 0.0, 0.0),   # voxel 0
            (0.1, 0.1, 0.0),   # voxel 0 (NMS removes lower score)
            (1.0, 0.0, 0.0),   # voxel 2
            (1.1, 0.0, 0.0),   # voxel 2
        ]
        snapshot = _snapshot(snapshot_root, positions=positions)
        self.pool.import_window(snapshot, 0)
        self.pool.entries[0].score = 0.9
        self.pool.entries[1].score = 0.8
        self.pool.entries[2].score = 0.6
        self.pool.entries[3].score = 0.4
        selected = self.pool.select(2, voxel_m=0.5)
        self.assertEqual([entry.global_index for entry in selected], [0, 2])
        self.assertTrue(selected[0].stamp_key() < selected[1].stamp_key())

    def test_selection_keeps_different_angles_at_same_position_and_fills_count(self):
        snapshot = _snapshot(Path(self.temp.name) / "views", count=120,
                             positions=[(0, 0, 0)] * 120)
        self.pool.import_window(snapshot, 0)
        for entry in self.pool.entries:
            entry.score = 1.0 - entry.global_index / 1000
        self.pool.entries[1].odom_orientation = (0.0, 0.0, 1.0, 0.0)
        selected = self.pool.select(2)
        self.assertEqual([e.global_index for e in selected], [0, 1])
        self.assertEqual(len(self.pool.select(100)), 100)
        self.pool.entries[-1].score = 2.0
        self.assertIn(119, [e.global_index for e in self.pool.select(100)])

    def test_bad_images_excluded_and_missing_pose_uses_time_coverage(self):
        snapshot = _snapshot(Path(self.temp.name) / "no-pose", count=6)
        snapshot.pose_valid = (False,) * 6
        snapshot.image_paths[0].write_bytes(b"broken PNG")
        self.pool.import_window(snapshot, 0)
        selected = self.pool.select(3)
        self.assertEqual(len(selected), 3)
        self.assertNotIn(0, [e.global_index for e in selected])
        self.assertGreaterEqual(selected[-1].stamp_sec - selected[0].stamp_sec, 3)

    def test_import_failure_is_atomic_and_retryable(self):
        oversized = _snapshot(Path(self.temp.name) / "oversized", count=2)
        pool = FramePool(self.root, FramePoolConfig(max_frames=1))
        with self.assertRaisesRegex(FramePoolCapacityError, "max_frames"):
            pool.import_window(oversized, 0)
        self.assertEqual(pool.entries, [])
        self.assertFalse((self.root / "images").exists())

        retry = _snapshot(Path(self.temp.name) / "retry", count=1)
        self.assertEqual(pool.import_window(retry, 0), [0])

        failing_pool = FramePool(Path(self.temp.name) / "failing-pool")
        calls = []

        def fail_second_copy(source, destination):
            calls.append(destination)
            shutil.copy2(source, destination)
            if len(calls) == 2:
                raise OSError("injected copy failure")

        with self.assertRaisesRegex(OSError, "injected copy failure"):
            failing_pool.import_window(oversized, 0, copier=fail_second_copy)
        self.assertEqual(failing_pool.entries, [])
        self.assertEqual(list((failing_pool.root / "images").glob("*.png")), [])

    def test_clear_discards_epoch_without_reusing_indices(self):
        first = _snapshot(Path(self.temp.name) / "first", count=2)
        self.pool.import_window(first, 0)
        self.pool.write_selection(0, self.pool.select(2))
        self.pool.clear()
        self.assertTrue((self.root / "selections" / "p000000.json").is_file())
        second = _snapshot(Path(self.temp.name) / "second", count=1)
        self.assertEqual(self.pool.import_window(second, 1), [2])
        self.assertEqual([entry.global_index for entry in self.pool.entries], [2])

    def test_retain_bounds_pool_without_reusing_global_indices(self):
        first = _snapshot(Path(self.temp.name) / "first", count=4)
        indices = self.pool.import_window(first, 0)
        removed = [self.root / entry.image_relative_path for entry in self.pool.entries[:-2]]
        self.pool.retain(self.pool.entries[-2:])
        self.assertEqual(
            [entry.global_index for entry in self.pool.entries], indices[-2:]
        )
        self.assertTrue(all(not path.exists() for path in removed))
        second = _snapshot(Path(self.temp.name) / "second", count=1)
        self.assertEqual(self.pool.import_window(second, 1), [4])

    def test_update_from_predictions_scores_conf_and_points(self):
        snapshot_root = Path(self.temp.name) / "snap"
        snapshot = _snapshot(snapshot_root, count=2)
        self.pool.import_window(snapshot, 0)
        predictions = Path(self.temp.name) / "predictions.npz"
        conf = np.array([[[0.9, 1.0], [1.1, 1.2]], [[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
        world = np.zeros((2, 2, 2, 3), dtype=np.float32)
        world[0, :, :, :] = 0.5
        world[1, :, :, :] = 0.5
        world[1, 1, 1, 1] = float("nan")  # fewer valid points
        np.savez(
            predictions,
            depth_conf=conf,
            world_points_from_depth=world,
        )
        self.pool.update_from_predictions(predictions, [0, 1], revision=1)
        self.assertGreater(self.pool.entries[0].score, self.pool.entries[1].score)
        self.assertEqual(self.pool.entries[0].source, "vggt_depth_conf")
        self.assertEqual(self.pool.entries[0].vggt_valid_points, 4)
        self.assertEqual(self.pool.entries[1].vggt_valid_points, 3)
        self.assertAlmostEqual(self.pool.entries[0].score, 0.75, places=4)
        self.assertAlmostEqual(self.pool.entries[1].score, 0.25, places=4)

    def test_synthetic_snapshot_passes_existing_validator(self):
        snapshot_root = Path(self.temp.name) / "snap"
        snapshot = _snapshot(snapshot_root, count=3)
        self.pool.import_window(snapshot, 0)
        selected = self.pool.select(3)
        destination = Path(self.temp.name) / "cs-s0123456789abcdef-w1-0123456789abcdef"
        uri, digest = self.pool.build_synthetic_snapshot(
            1,
            selected,
            destination,
            "cs-s0123456789abcdef-p1",
            camera_name="front_camera",
            image_topic="/front/image_raw",
            camera_role="front",
            camera_frame_id="front_camera_optical_frame",
        )
        self.assertTrue(destination.is_dir())
        self.assertEqual(uri, destination.as_uri())
        validated = validate_snapshot(uri, digest, (destination.parent,), _limits())
        self.assertEqual(validated.snapshot_version, 3)
        self.assertEqual(validated.frame_count, 3)
        self.assertEqual(len(validated.image_paths), 3)
        self.assertEqual(validated.camera_role, "front")

    def test_synthetic_snapshot_checks_limits_without_rereading_its_files(self):
        from dataclasses import replace
        snapshot = _snapshot(Path(self.temp.name) / "snap", count=3)
        self.pool.import_window(snapshot, 0)
        for key, value in (("max_frame_count", 2), ("max_total_bytes", 1),
                           ("max_manifest_entries", 1), ("max_metadata_bytes", 32)):
            with self.subTest(limit=key):
                destination = Path(self.temp.name) / ("limited-" + key)
                with self.assertRaises(FramePoolError):
                    self.pool.build_synthetic_snapshot(0, self.pool.select(3), destination, "scene",
                        camera_name="front", image_topic="/rgb", camera_role="front",
                        camera_frame_id="camera", limits=replace(_limits(), **{key: value}))
                self.assertFalse(destination.exists())
                self.assertEqual(list(destination.parent.glob("." + destination.name + ".*")), [])

    def test_synthetic_snapshot_failure_leaves_no_partial_destination(self):
        snapshot = _snapshot(Path(self.temp.name) / "snap", count=2)
        self.pool.import_window(snapshot, 0)
        destination = Path(self.temp.name) / "cs-s0123456789abcdef-w0-0123456789abcdef"
        calls = []

        def fail_second_copy(source, target):
            calls.append(target)
            shutil.copy2(source, target)
            if len(calls) == 2:
                raise OSError("injected synthetic copy failure")

        with self.assertRaisesRegex(OSError, "injected synthetic copy failure"):
            self.pool.build_synthetic_snapshot(
                0,
                self.pool.select(2),
                destination,
                "cs-s0123456789abcdef-p0",
                camera_name="front_camera",
                image_topic="/front/image_raw",
                camera_role="front",
                camera_frame_id="front_camera_optical_frame",
                copier=fail_second_copy,
            )
        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.glob(f".{destination.name}.*")), [])

    def test_invalid_config_and_selection_size_are_rejected(self):
        with self.assertRaises(FramePoolError):
            FramePoolConfig(voxel_m=0)
        with self.assertRaises(FramePoolError):
            FramePoolConfig(max_frames=0)
        with self.assertRaises(FramePoolError):
            self.pool.select(0)
        with self.assertRaises(FramePoolError):
            self.pool.build_synthetic_snapshot(
                1,
                [],
                Path(self.temp.name) / "dst",
                "scene",
                camera_name="front_camera",
                image_topic="/front/image_raw",
                camera_role="front",
                camera_frame_id="front_camera_optical_frame",
            )

    def test_mode_constants(self):
        self.assertEqual(RECONSTRUCTION_MODE_PER_WINDOW, "per_window")
        self.assertEqual(RECONSTRUCTION_MODE_FRAME_POOL, "frame_pool")


if __name__ == "__main__":
    unittest.main()
