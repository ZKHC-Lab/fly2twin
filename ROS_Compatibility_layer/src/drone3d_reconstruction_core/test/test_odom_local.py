"""CPU-only tests for ODOM_LOCAL_APPROX Sim(3) alignment."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from drone3d_reconstruction_core.odom_local import (
    COORDINATE_MODEL_LOCAL_UNALIGNED,
    COORDINATE_ODOM_LOCAL_APPROX,
    _finite_float,
    align_window,
)


def _quat_identity():
    return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def _make_snapshot(points, frame_id="odom", tf=True):
    payloads = []
    for index, point in enumerate(points):
        payloads.append(
            {
                "header": {
                    "stamp": {"sec": 10 + index, "nsec": 0},
                    "frame_id": frame_id,
                },
                "pose": {
                    "position": {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])},
                    "orientation": _quat_identity(),
                },
            }
        )
    tf_entries = []
    if tf:
        tf_entries.append(
            {
                "stamp": {"sec": 0, "nsec": 0},
                "parent_frame_id": "body",
                "child_frame_id": "camera_optical",
                "translation": {"x": 0.1, "y": 0.0, "z": 0.0},
                "rotation": _quat_identity(),
                "static": True,
            }
        )
    return SimpleNamespace(
        odometry_payloads=payloads,
        tf_entries=tuple(tf_entries),
        body_frame_id="body",
        camera_frame_id="camera_optical",
    )


def _write_predictions(path: Path, camera_centers, world_to_camera=None):
    rotation = np.asarray(
        world_to_camera if world_to_camera is not None else np.eye(3),
        dtype=np.float64,
    )
    rotations = np.repeat(rotation[None, ...], len(camera_centers), axis=0)
    translations = -np.einsum(
        "nij,nj->ni", rotations, np.asarray(camera_centers, dtype=np.float64)
    )
    extrinsic = np.concatenate(
        [rotations, translations[:, :, None]],
        axis=2,
    ).astype(np.float32)
    np.savez(str(path), extrinsic=extrinsic)


class OdomLocalAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.predictions = Path(self.temp.name) / "predictions.npz"

    def _reference(self, n=6):
        # 空间曲线，避免退化
        return np.array(
            [[float(i), np.sin(float(i)), float(i % 3)] for i in range(n)]
        )

    def test_recovers_known_sim3(self):
        reference = self._reference()
        scale = 2.0
        angle = 0.3
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.array([1.0, -2.0, 0.5])
        model = scale * (rotation @ reference.T).T + translation
        _write_predictions(self.predictions, model, rotation.T)
        result = align_window(
            _make_snapshot(reference), self.predictions, min_correspondences=4
        )
        self.assertEqual(result.coordinate_state, COORDINATE_ODOM_LOCAL_APPROX)
        self.assertEqual(result.correspondence_count, 6)
        # 对齐方向是 model → odom：scale = 1/model_scale，
        # rotation = model_rotation^T，translation = -(1/s) R^T t + t_body_camera。
        self.assertAlmostEqual(result.scale, 1.0 / scale, places=5)
        expected_rotation = rotation.T
        expected_translation = -(1.0 / scale) * (rotation.T @ translation)
        expected_translation += np.array([0.1, 0.0, 0.0])
        self.assertTrue(np.allclose(result.rotation, expected_rotation, atol=1e-5))
        self.assertTrue(
            np.allclose(result.translation, expected_translation, atol=1e-5)
        )
        self.assertLess(result.residual_rms, 1e-5)

    def test_coplanar_noncollinear_square_recovers_identity_scale(self):
        reference = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        _write_predictions(self.predictions, reference)
        result = align_window(
            _make_snapshot(reference), self.predictions, min_correspondences=4
        )
        self.assertEqual(result.coordinate_state, COORDINATE_ODOM_LOCAL_APPROX)
        self.assertAlmostEqual(result.scale, 1.0, places=5)
        self.assertLess(result.residual_rms, 1e-5)

    def test_missing_body_camera_tf_falls_back(self):
        reference = self._reference()
        _write_predictions(self.predictions, reference)
        result = align_window(
            _make_snapshot(reference, tf=False), self.predictions
        )
        self.assertEqual(result.coordinate_state, COORDINATE_MODEL_LOCAL_UNALIGNED)
        self.assertIn("T_body_camera", result.reason)

    def test_near_static_reference_trajectory_falls_back(self):
        reference = np.array([[0.001 * i, 0.0, 0.0] for i in range(6)])
        _write_predictions(self.predictions, reference)
        result = align_window(
            _make_snapshot(reference), self.predictions, min_correspondences=4
        )
        self.assertEqual(result.coordinate_state, COORDINATE_MODEL_LOCAL_UNALIGNED)
        self.assertIn("baseline", result.reason)

    def test_straight_trajectory_uses_camera_attitude_for_rotation(self):
        reference = np.array([[float(i), 0.0, 0.0] for i in range(6)])
        angle = 0.7
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        model = 2.0 * (rotation @ reference.T).T + [1.0, -2.0, 0.5]
        _write_predictions(self.predictions, model, rotation.T)
        result = align_window(
            _make_snapshot(reference), self.predictions, min_correspondences=4
        )
        self.assertEqual(result.coordinate_state, COORDINATE_ODOM_LOCAL_APPROX)
        self.assertAlmostEqual(result.scale, 0.5, places=5)
        self.assertTrue(np.allclose(result.rotation, rotation.T, atol=1e-5))
        self.assertLess(result.residual_rms, 1e-5)

    def test_frame_change_inside_window_falls_back(self):
        reference = self._reference()
        _write_predictions(self.predictions, reference)
        payloads = _make_snapshot(reference).odometry_payloads
        payloads[-1]["header"]["frame_id"] = "odom_new"
        snapshot = SimpleNamespace(
            odometry_payloads=payloads,
            tf_entries=_make_snapshot(reference).tf_entries,
            body_frame_id="body",
            camera_frame_id="camera_optical",
        )
        result = align_window(snapshot, self.predictions)
        self.assertEqual(result.coordinate_state, COORDINATE_MODEL_LOCAL_UNALIGNED)
        self.assertIn("frame changed", result.reason)

    def test_insufficient_correspondences_fall_back(self):
        reference = self._reference()
        _write_predictions(self.predictions, reference[:3])
        result = align_window(
            _make_snapshot(reference), self.predictions, min_correspondences=4
        )
        self.assertEqual(result.coordinate_state, COORDINATE_MODEL_LOCAL_UNALIGNED)
        self.assertIn("not enough model frames", result.reason)

    def test_selected_frame_indices_map_subsampled_model_frames(self):
        reference = self._reference()
        selected = [0, 2, 4]
        model = reference[selected]
        _write_predictions(self.predictions, model)
        result = align_window(
            _make_snapshot(reference),
            self.predictions,
            min_correspondences=3,
            selected_frame_indices=selected,
        )
        self.assertEqual(result.coordinate_state, COORDINATE_ODOM_LOCAL_APPROX)
        self.assertEqual(result.correspondence_count, 3)
        self.assertAlmostEqual(result.scale, 1.0, places=5)
        self.assertLess(result.residual_rms, 1e-5)

    def test_selected_frame_indices_length_mismatch_falls_back(self):
        reference = self._reference()
        _write_predictions(self.predictions, reference)
        result = align_window(
            _make_snapshot(reference),
            self.predictions,
            selected_frame_indices=[0, 1],
        )
        self.assertEqual(result.coordinate_state, COORDINATE_ODOM_LOCAL_APPROX)
        self.assertEqual(result.correspondence_count, 6)

    def test_pose_invalid_odometry_frames_are_excluded_from_reference(self):
        reference = self._reference()
        _write_predictions(self.predictions, reference)
        base = _make_snapshot(reference)
        snapshot = SimpleNamespace(
            odometry_payloads=base.odometry_payloads,
            tf_entries=base.tf_entries,
            body_frame_id="body",
            camera_frame_id="camera_optical",
            pose_valid=(True, True, False, True, True, True),
        )
        result = align_window(snapshot, self.predictions, min_correspondences=4)
        self.assertEqual(result.coordinate_state, COORDINATE_ODOM_LOCAL_APPROX)
        self.assertEqual(result.correspondence_count, 5)
        self.assertLess(result.residual_rms, 1e-5)

    def test_nonfinite_odometry_falls_back(self):
        reference = self._reference()
        _write_predictions(self.predictions, reference)
        payloads = _make_snapshot(reference).odometry_payloads
        payloads[2]["pose"]["position"]["x"] = float("nan")
        snapshot = SimpleNamespace(
            odometry_payloads=payloads,
            tf_entries=_make_snapshot(reference).tf_entries,
            body_frame_id="body",
            camera_frame_id="camera_optical",
        )
        result = align_window(snapshot, self.predictions)
        self.assertEqual(result.coordinate_state, COORDINATE_MODEL_LOCAL_UNALIGNED)
        self.assertIn("invalid position", result.reason)

    def test_odom_stamp_rollback_inside_window_falls_back(self):
        reference = self._reference()
        _write_predictions(self.predictions, reference)
        payloads = _make_snapshot(reference).odometry_payloads
        payloads[3]["header"]["stamp"] = {"sec": 9, "nsec": 0}
        snapshot = SimpleNamespace(
            odometry_payloads=payloads,
            tf_entries=_make_snapshot(reference).tf_entries,
            body_frame_id="body",
            camera_frame_id="camera_optical",
        )
        result = align_window(snapshot, self.predictions)
        self.assertEqual(result.coordinate_state, COORDINATE_MODEL_LOCAL_UNALIGNED)
        self.assertIn("stamp reset", result.reason)

    def test_inverted_body_camera_tf_is_resolved(self):
        reference = self._reference()
        model = reference
        _write_predictions(self.predictions, model)
        # tf.json 记录 camera→body；解析器应求逆得到 body→camera。
        snapshot = SimpleNamespace(
            odometry_payloads=_make_snapshot(reference).odometry_payloads,
            tf_entries=(
                {
                    "stamp": {"sec": 0, "nsec": 0},
                    "parent_frame_id": "camera_optical",
                    "child_frame_id": "body",
                    "translation": {"x": -0.1, "y": 0.0, "z": 0.0},
                    "rotation": _quat_identity(),
                    "static": True,
                },
            ),
            body_frame_id="body",
            camera_frame_id="camera_optical",
        )
        result = align_window(snapshot, self.predictions)
        self.assertEqual(result.coordinate_state, COORDINATE_ODOM_LOCAL_APPROX)
        self.assertAlmostEqual(result.scale, 1.0, places=5)
        self.assertTrue(
            np.allclose(result.translation, [0.1, 0.0, 0.0], atol=1e-5)
        )

    def test_static_miss_falls_back_to_dynamic_body_camera(self):
        reference = self._reference()
        _write_predictions(self.predictions, reference)
        base = _make_snapshot(reference)
        snapshot = SimpleNamespace(
            odometry_payloads=base.odometry_payloads,
            tf_entries=(
                {
                    "stamp": {"sec": 0, "nsec": 0},
                    "parent_frame_id": "map",
                    "child_frame_id": "odom",
                    "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "rotation": _quat_identity(),
                    "static": True,
                },
                {
                    "stamp": {"sec": 0, "nsec": 0},
                    "parent_frame_id": "body",
                    "child_frame_id": "camera_optical",
                    "translation": {"x": 0.1, "y": 0.0, "z": 0.0},
                    "rotation": _quat_identity(),
                    "static": False,
                },
            ),
            body_frame_id="body",
            camera_frame_id="camera_optical",
        )
        result = align_window(snapshot, self.predictions)
        self.assertEqual(result.coordinate_state, COORDINATE_ODOM_LOCAL_APPROX)
        self.assertTrue(
            np.allclose(result.translation, [0.1, 0.0, 0.0], atol=1e-5)
        )

    def test_unrelated_malformed_tf_entry_is_ignored(self):
        reference = self._reference()
        _write_predictions(self.predictions, reference)
        base = _make_snapshot(reference)
        entries = (
            {
                "stamp": {"sec": 0, "nsec": 0},
                "parent_frame_id": "map",
                "child_frame_id": "odom",
                "translation": {"x": "bad", "y": 0.0, "z": 0.0},
                "rotation": _quat_identity(),
                "static": True,
            },
        ) + base.tf_entries
        snapshot = SimpleNamespace(
            odometry_payloads=base.odometry_payloads,
            tf_entries=entries,
            body_frame_id="body",
            camera_frame_id="camera_optical",
        )
        result = align_window(snapshot, self.predictions)
        self.assertEqual(result.coordinate_state, COORDINATE_ODOM_LOCAL_APPROX)

    def test_float_stamp_components_fall_back(self):
        reference = self._reference()
        _write_predictions(self.predictions, reference)
        payloads = _make_snapshot(reference).odometry_payloads
        payloads[1]["header"]["stamp"] = {"sec": 10.5, "nsec": 0}
        snapshot = SimpleNamespace(
            odometry_payloads=payloads,
            tf_entries=_make_snapshot(reference).tf_entries,
            body_frame_id="body",
            camera_frame_id="camera_optical",
        )
        result = align_window(snapshot, self.predictions)
        self.assertEqual(result.coordinate_state, COORDINATE_MODEL_LOCAL_UNALIGNED)
        self.assertIn("invalid stamp", result.reason)

    def test_finite_float_rejects_bool_string_and_nan(self):
        for bad in (True, False, "1.0", b"1.0", None, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _finite_float(bad, "bad")
        self.assertEqual(_finite_float(1, "ok"), 1.0)


if __name__ == "__main__":
    unittest.main()
