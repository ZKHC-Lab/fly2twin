"""Window-level tests for shared identities and odometry diagnostic isolation."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from drone3d_reconstruction_core import shared_point_sim3 as math3d
from drone3d_reconstruction_core.shared_point_backend import align
from drone3d_reconstruction_core.snapshot_validator import FrameStamp


class SharedPointBackendTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(928)
        self.root_images = rng.random((100, 3, 1, 384), dtype=np.float32)
        self.root_points = rng.random((100, 1, 384, 3)).astype(np.float32) + 1
        self.identity = (1.0, np.eye(3), np.zeros(3))
        angle = 0.2
        self.transform = (
            1.4,
            np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]),
            np.array([0.2, -0.1, 0.0]),
        )
        self.a = self.root / "a.npz"
        self.b = self.root / "b.npz"
        self.write(self.a, range(100), self.root_images, self.root_points, self.identity)
        images = rng.random((120, 3, 1, 384), dtype=np.float32)
        images[:20] = self.root_images[-20:]
        points = rng.random((120, 1, 384, 3)).astype(np.float32) + 1
        points[:20] = math3d.apply(self.root_points[-20:], math3d.inverse(self.transform))
        self.write(self.b, range(80, 200), images, points, self.transform)

    @staticmethod
    def positions(ids):
        return np.array([[i / 10.0, np.sin(i / 10.0), 0] for i in ids])

    def write(self, path, ids, images, points, transform):
        centers = math3d.apply(self.positions(ids), math3d.inverse(transform))
        rotations = np.tile(transform[1], (len(centers), 1, 1))
        extrinsic = np.concatenate(
            (rotations, -np.einsum("nij,nj->ni", rotations, centers)[:, :, None]), axis=2
        )
        np.savez(path, images=images, world_points_from_depth=points,
                 depth=np.ones(images.shape[:1] + images.shape[2:] + (1,)),
                 depth_conf=np.ones(images.shape[:1] + images.shape[2:]), extrinsic=extrinsic)

    def snapshot(self, ids, frame_id="map"):
        ids = list(ids)
        stamps = [FrameStamp(10 + i // 5, i % 5 * 200_000_000) for i in ids]
        poses = []
        for stamp, position in zip(stamps, self.positions(ids)):
            poses.append({
                "header": {"frame_id": frame_id, "stamp": {"sec": stamp.sec, "nsec": stamp.nsec}},
                "child_frame_id": "body",
                "pose": {"position": dict(zip("xyz", position.tolist())),
                         "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
                "interpolation": {"method": "linear_position_slerp_orientation"},
            })
        return SimpleNamespace(
            body_frame_id="body", camera_frame_id="camera", frame_stamps=tuple(stamps),
            pose_valid=tuple(True for _ in ids), odometry_payloads=tuple(poses),
            tf_entries=({"static": True, "parent_frame_id": "body", "child_frame_id": "camera",
                         "translation": {"x": 0, "y": 0, "z": 0},
                         "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}},),
        )

    def root_alignment(self, **kwargs):
        return align(self.snapshot(range(100)), self.a, model_frame_ids=range(100),
                     selected_indices=range(100), window_index=0, capture_generation=0,
                     map_id="s123-m0", **kwargs)

    def test_root_keeps_old_geometry_thresholds_diagnostic_only(self):
        payload, reference = self.root_alignment(odom_limits=(1000.0, 15.0, 0.25))
        self.assertTrue(payload["quality_passed"])
        self.assertFalse(payload["odom_quality_diagnostic_passed"])
        self.assertEqual(reference.frame_ids, tuple(range(80, 100)))

    def test_shared_edge_recovers_transform_and_excludes_prefix(self):
        _root, reference = self.root_alignment()
        payload, candidate = align(
            self.snapshot(range(80, 200)), self.b, model_frame_ids=range(80, 200),
            selected_indices=range(120), window_index=1, capture_generation=0,
            map_id="unused", reference=reference,
        )
        for actual, expected in zip(math3d.unpack(payload), self.transform):
            np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertEqual(payload["merge_frame_indices"], list(range(20, 120)))
        self.assertEqual(payload["map_id"], reference.map_id)
        self.assertEqual(candidate.frame_ids, tuple(range(180, 200)))
        self.assertEqual(reference.frame_ids, tuple(range(80, 100)))

    def test_odom_frame_reset_does_not_reset_rgb_map(self):
        _root, reference = self.root_alignment()
        payload, candidate = align(
            self.snapshot(range(80, 200), "new_map"), self.b, model_frame_ids=range(80, 200),
            selected_indices=range(120), window_index=1, capture_generation=0,
            map_id="unused", reference=reference,
        )
        self.assertTrue(payload["quality_passed"])
        self.assertEqual(candidate.map_id, reference.map_id)
        self.assertEqual(payload["reference_frame_id"], "map")
        self.assertIn("frame changed", payload["odometry_diagnostic_error"])

    def test_wrong_reference_ids_and_subsampling_are_rejected(self):
        _root, reference = self.root_alignment()
        for ids, selected in ((range(81, 201), range(120)), (range(80, 200), range(100))):
            with self.subTest(ids=ids, selected=selected), self.assertRaises(ValueError):
                align(self.snapshot(range(80, 200)), self.b, model_frame_ids=ids,
                      selected_indices=selected, window_index=1, capture_generation=0,
                      map_id="unused", reference=reference)

    def test_root_refuses_missing_pose(self):
        snapshot = self.snapshot(range(100))
        snapshot.pose_valid = (False,) + snapshot.pose_valid[1:]
        with self.assertRaisesRegex(ValueError, "all 100"):
            align(snapshot, self.a, model_frame_ids=range(100), selected_indices=range(100),
                  window_index=0, capture_generation=0, map_id="s123-m0")


if __name__ == "__main__":
    unittest.main()
