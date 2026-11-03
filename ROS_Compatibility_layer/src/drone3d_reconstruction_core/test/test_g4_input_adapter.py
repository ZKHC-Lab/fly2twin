"""I/O boundary tests: identity, time, coordinate gauges and full-cloud export."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from g4_backend import Sim3, sim3
from drone3d_reconstruction_core.g4.input_adapter import WindowInputReader, sha256


class InputAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        n = 3
        self.images = np.full((n, 3, 288, 384), .5, np.float32)
        self.images[:, 0] = .25
        K = np.repeat(np.array([[200., 0, 191.5], [0, 200., 143.5], [0, 0, 1.]])[None], n, axis=0)
        self.ex = np.repeat(np.c_[np.eye(3), [1., 2., 3.]][None], n, axis=0)
        self.ex[:, 0, 3] += np.arange(n)*.1
        path = self.root / "predictions.npz"
        np.savez_compressed(path, images=self.images, intrinsic=K, extrinsic=self.ex,
                            depth=np.full((n, 288, 384, 1), 2., np.float32),
                            depth_conf=np.ones((n, 288, 384), np.float32))
        stamps = [SimpleNamespace(sec=k+1, nsec=7) for k in range(n)]
        image_paths = []
        for k in range(n):
            p = self.root / (str(k)+".png")
            p.write_bytes(b"verified-snapshot-image-"+str(k).encode())
            image_paths.append(p)
        self.snapshot = SimpleNamespace(snapshot_version=4, frame_count=n, frame_stamps=stamps,
            image_paths=image_paths, tf_entries=({"static": True, "parent_frame_id": "body", "child_frame_id": "camera", "translation": {"x": 0., "y": 0., "z": 0.}, "rotation": {"x": 0., "y": 0., "z": 0., "w": 1.}},), body_frame_id="body", camera_frame_id="camera",
            image_topic="/rgb", odometry_payloads=[{
                "child_frame_id": "body", "header": {"stamp": {"sec": s.sec, "nsec": s.nsec}},
                "interpolation": {"method": "linear_position_slerp_orientation"},
            } for s in stamps])
        self.payload = {"backend": "omega", "artifacts": {"predictions.npz": {
            "uri": str(path), "sha256": sha256(path)}},
            "selected_frames": [{"index": k, "stamp": {"sec": s.sec, "nsec": s.nsec}} for k, s in enumerate(stamps)]}
        self.record = {"result_manifest_uri": str(self.root/"result.json"),
            "snapshot_manifest_uri": str(self.root/"snapshot"), "snapshot_manifest_sha256": "0"*64,
            "effective_config": {"window_frames": n}, "model_frame_ids": ["a", "b", "c"], "window_index": 0, "overlap_frame_count": 0}
        self.write_manifest()
    def write_manifest(self):
        path = Path(self.record["result_manifest_uri"])
        path.write_text(json.dumps(self.payload))
        self.record["result_manifest_sha256"] = sha256(path)

    def load(self):
        with patch("drone3d_reconstruction_core.snapshot_validator.validate_snapshot", return_value=self.snapshot), \
             patch("drone3d_reconstruction_core.odom_local._reference_camera_positions", return_value=(
                 np.arange(3), np.arange(9).reshape(3, 3), np.tile(np.eye(3), (3, 1, 1)), "map", None)):
            return WindowInputReader().load(self.record)

    def test_prediction_gauge_and_exact_localization(self):
        loaded = self.load()
        self.assertTrue(loaded.reference_valid.all())
        T = Sim3.from_dict(loaded.metadata["prediction_to_local"])
        raw = np.array([[0., 0., 2.]]) - self.ex[0, :, 3]
        np.testing.assert_allclose(sim3.apply(T, raw), [[0, 0, 1.]])
        np.testing.assert_allclose(loaded.geometry.centers[0], [0, 0, 0])

    def test_rotations_camera_info_and_model_size_are_carried(self):
        K = [600., 0, 320, 0, 600, 240, 0, 0, 1]
        self.snapshot.camera_info_payloads = [{"width": 640, "height": 480, "K": list(K)} for _ in range(3)]
        loaded = self.load()
        np.testing.assert_allclose(loaded.reference_rotations, np.tile(np.eye(3), (3, 1, 1)))
        self.assertEqual(loaded.metadata["camera_info"], {"width": 640, "height": 480,
                                                          "K": np.reshape(K, (3, 3)).tolist()})
        self.assertEqual(loaded.metadata["model_image_size"], [384, 288])
        # CameraInfo that disagrees across frames is not trusted as a calibration.
        self.snapshot.camera_info_payloads[2]["K"][0] = 500.
        self.assertIsNone(self.load().metadata["camera_info"])

    def test_model_image_sizes_are_patch_multiples(self):
        def write(shape):
            path = self.root / ("predictions_%dx%d.npz" % shape[2:])
            n, _, h, w = shape
            K = np.repeat(np.array([[200., 0, w / 2 - .5], [0, 200., h / 2 - .5], [0, 0, 1.]])[None], n, axis=0)
            np.savez_compressed(path, images=np.full(shape, .5, np.float32), intrinsic=K, extrinsic=self.ex,
                                depth=np.full((n, h, w, 1), 2., np.float32), depth_conf=np.ones((n, h, w), np.float32))
            self.payload["artifacts"]["predictions.npz"] = {"uri": str(path), "sha256": sha256(path)}
            self.write_manifest()
        for shape in ((3, 3, 224, 384), (3, 3, 288, 512), (3, 3, 384, 512)):  # 16:9 and 4:3 at 384/512
            write(shape)
            self.assertEqual(self.load().metadata["model_image_size"], [shape[3], shape[2]])
        write((3, 3, 216, 384))
        with self.assertRaisesRegex(ValueError, "16-pixel patch"):
            self.load()

    def test_missing_pose_keeps_rotation_undefined(self):
        self.snapshot.odometry_payloads[1]["interpolation"] = {"method": "nearest"}
        loaded = self.load()
        self.assertTrue(np.isnan(loaded.reference_rotations[1]).all())
        np.testing.assert_allclose(loaded.reference_rotations[0], np.eye(3))

    def test_nearest_pose_is_excluded_instead_of_relabelled_as_interpolated(self):
        self.snapshot.odometry_payloads[1]["interpolation"] = {"method": "nearest"}
        loaded = self.load()
        np.testing.assert_array_equal(loaded.reference_valid, [True, False, True])

    def test_subsampling_is_rejected(self):
        self.payload["selected_frames"] = self.payload["selected_frames"][::2]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "subsampling"):
            self.load()
    def test_rgb_timestamp_mismatch_is_rejected(self):
        self.payload["selected_frames"][1]["stamp"]["nsec"] = 8
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "timestamp differs"):
            self.load()

    def test_absent_localization_keeps_geometry_pending(self):
        self.snapshot.odometry_payloads = [None] * 3
        loaded = self.load()
        self.assertFalse(loaded.reference_valid.any())
        self.assertEqual(len(loaded.geometry.frame_ids), 3)

    def test_dynamic_extrinsic_is_not_applied_as_a_constant_lever_arm(self):
        self.snapshot.tf_entries = ({"static": False},)
        loaded = self.load()
        self.assertFalse(loaded.reference_valid.any())
        self.assertIn("static body-camera", loaded.metadata["localization_reason"])

    def test_static_chain_matches_direct_transform(self):
        import copy
        first = copy.deepcopy(self.snapshot.tf_entries[0])
        second = copy.deepcopy(first)
        first["child_frame_id"] = "camera_link"
        first["translation"]["x"] = .1
        second["parent_frame_id"] = "camera_link"
        second["translation"]["y"] = .2
        self.snapshot.tf_entries = (first, second)
        loaded = self.load()
        self.assertTrue(loaded.reference_valid.all())
        np.testing.assert_allclose(loaded.metadata["body_from_camera"]["translation"], [.1, .2, 0.])
        # A conflicting direct path must remain unplaced, rather than picking an edge.
        direct = copy.deepcopy(first)
        direct["child_frame_id"] = "camera"
        self.snapshot.tf_entries += (direct,)
        loaded = self.load()
        self.assertFalse(loaded.reference_valid.any())
        self.assertIn("inconsistent", loaded.metadata["localization_reason"])

    def test_prediction_tamper_is_rejected(self):
        (self.root/"predictions.npz").write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "prediction digest changed"):
            self.load()


if __name__ == "__main__":
    unittest.main()
