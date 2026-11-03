"""CPU checks for A's direction, outlier rejection, and held-out isolation."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

# The package __init__ imports actionlib. Load this ROS-independent numerical
# module directly so these checks also run in the experiment's NumPy runtime.
_source = Path(__file__).resolve().parents[1] / "src/drone3d_reconstruction_core/shared_point_sim3.py"
_spec = importlib.util.spec_from_file_location("shared_point_sim3_under_test", _source)
registration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(registration)


class SharedPointSim3Test(unittest.TestCase):
    def setUp(self):
        angle = 0.47
        self.transform = (
            1.7,
            np.array([[np.cos(angle), -np.sin(angle), 0.0],
                      [np.sin(angle), np.cos(angle), 0.0],
                      [0.0, 0.0, 1.0]]),
            np.array([2.0, -0.7, 0.4]),
        )
        self.identity = (1.0, np.eye(3), np.zeros(3))

    def pairs(self):
        rng = np.random.default_rng(831)
        pairs = []
        for _index in range(20):
            current = rng.normal(size=(250, 3))
            previous = registration.apply(current, self.transform)
            previous += rng.normal(scale=0.001, size=previous.shape)
            previous[:75] = rng.uniform(-10.0, 10.0, size=(75, 3))
            pairs.append((previous, current, np.zeros((250, 2))))
        return pairs

    def test_recovers_current_to_map_with_thirty_percent_outliers(self):
        pairs = self.pairs()
        transform, diagnostics = registration.robust(pairs, self.identity)
        probe = np.random.default_rng(87).normal(size=(100, 3))
        np.testing.assert_allclose(
            registration.apply(probe, transform),
            registration.apply(probe, self.transform),
            atol=0.001,
        )
        self.assertAlmostEqual(diagnostics["inlier_ratio"], 0.7, delta=0.005)
        self.assertEqual(diagnostics["training_points"], 2500)
        self.assertGreater(np.linalg.det(transform[1]), 0.99999)

    def test_held_out_points_do_not_change_fit(self):
        pairs = self.pairs()
        original, original_diagnostics = registration.robust(pairs, self.identity)
        for index in range(1, 20, 2):
            a, b, uv = pairs[index]
            pairs[index] = (a + 1000.0, b * 10.0, uv)
        changed, changed_diagnostics = registration.robust(pairs, self.identity)
        for before, after in zip(original, changed):
            np.testing.assert_array_equal(before, after)
        self.assertEqual(original_diagnostics, changed_diagnostics)

    def test_root_fit_uses_camera_attitudes_and_centers(self):
        source = np.random.default_rng(21).normal(size=(100, 3))
        extrinsic = np.concatenate(
            (np.tile(np.eye(3), (100, 1, 1)), -source[:, :, None]), axis=2
        )
        positions = registration.apply(source, self.transform)
        orientations = np.tile(self.transform[1], (100, 1, 1))
        fitted = registration.fit_pose(extrinsic, positions, orientations)
        for actual, expected in zip(fitted, self.transform):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_root_rejects_stationary_scale_degeneracy(self):
        extrinsic = np.concatenate(
            (np.tile(np.eye(3), (100, 1, 1)), np.zeros((100, 3, 1))), axis=2
        )
        with self.assertRaisesRegex(ValueError, "degenerate"):
            registration.fit_pose(
                extrinsic, np.zeros((100, 3)), np.tile(np.eye(3), (100, 1, 1))
            )

    def test_collinear_shared_points_fail(self):
        points = np.column_stack([np.arange(10), np.zeros(10), np.zeros(10)])
        pairs = [(points, points, np.zeros((10, 2))) for _ in range(20)]
        with self.assertRaisesRegex(ValueError, "No RANSAC consensus"):
            registration.robust(pairs, self.identity)

    def test_shared_rgb_identity_is_required(self):
        a = {"images": np.zeros((20, 3, 2, 2))}
        b = {"images": np.ones((120, 3, 2, 2))}
        with self.assertRaisesRegex(ValueError, "RGB frames differ"):
            registration.shared_pairs(a, b)


if __name__ == "__main__":
    unittest.main()
