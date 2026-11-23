"""CPU-only tests for the odometry trajectory helpers used by G4 input."""

import unittest
from types import SimpleNamespace

import numpy as np

from drone3d_reconstruction_core.odom_local import (
    _finite_float,
    _reference_camera_positions,
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


class ReferenceCameraPositionsTest(unittest.TestCase):
    BODY_CAMERA = (np.eye(3), np.array([0.1, 0.0, 0.0]))

    def _reference(self, n=6):
        # 空间曲线，避免退化
        return np.array(
            [[float(i), np.sin(float(i)), float(i % 3)] for i in range(n)]
        )

    def _positions(self, snapshot):
        return _reference_camera_positions(snapshot, *self.BODY_CAMERA)

    def test_camera_positions_apply_the_body_camera_offset(self):
        reference = self._reference()
        indices, positions, rotations, frame_id, stamps = self._positions(
            _make_snapshot(reference)
        )
        self.assertEqual(indices.tolist(), list(range(6)))
        self.assertTrue(np.allclose(positions, reference + [0.1, 0.0, 0.0]))
        self.assertTrue(np.allclose(rotations, np.eye(3)))
        self.assertEqual(frame_id, "odom")
        self.assertEqual(stamps[0], (10, 0))

    def test_frame_change_inside_window_is_rejected(self):
        snapshot = _make_snapshot(self._reference())
        snapshot.odometry_payloads[-1]["header"]["frame_id"] = "odom_new"
        with self.assertRaisesRegex(ValueError, "frame changed"):
            self._positions(snapshot)

    def test_pose_invalid_odometry_frames_are_excluded_from_reference(self):
        snapshot = _make_snapshot(self._reference())
        snapshot.pose_valid = (True, True, False, True, True, True)
        indices, positions, _rotations, _frame_id, _stamps = self._positions(snapshot)
        self.assertEqual(indices.tolist(), [0, 1, 3, 4, 5])
        self.assertEqual(len(positions), 5)

    def test_nonfinite_odometry_is_rejected(self):
        snapshot = _make_snapshot(self._reference())
        snapshot.odometry_payloads[2]["pose"]["position"]["x"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid position"):
            self._positions(snapshot)

    def test_float_stamp_components_are_rejected(self):
        snapshot = _make_snapshot(self._reference())
        snapshot.odometry_payloads[1]["header"]["stamp"] = {"sec": 10.5, "nsec": 0}
        with self.assertRaisesRegex(ValueError, "invalid stamp"):
            self._positions(snapshot)

    def test_finite_float_rejects_bool_string_and_nan(self):
        for bad in (True, False, "1.0", b"1.0", None, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _finite_float(bad, "bad")
        self.assertEqual(_finite_float(1, "ok"), 1.0)


if __name__ == "__main__":
    unittest.main()
