import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from g4_backend import (FramePoseConfig, WindowGeometry, interpolate_poses, shared_depth_ratio,
                        solve_window_scales, trajectory_prior, window_translation)


def window(scale, R_map_local, t, n=40, stationary=False, seed=0):
    """Local camera centres/rotations and their localization counterparts."""
    rng = np.random.default_rng(seed)
    local_c = np.zeros((n, 3)) if stationary else np.cumsum(rng.normal(0, .05, (n, 3)), axis=0)
    local_R = Rotation.from_rotvec(rng.normal(0, .3, (n, 3))).as_matrix()
    ref_c = scale * local_c @ R_map_local.T + t
    ref_R = R_map_local @ local_R
    return local_c, local_R, ref_c, ref_R


class TrajectoryPriorTest(unittest.TestCase):
    def test_recovers_scale_rotation_and_translation(self):
        R = Rotation.from_rotvec([.2, -1.3, .4]).as_matrix()
        lc, lR, rc, rR = window(4.2, R, np.array([1., 2., 3.]))
        p = trajectory_prior(lc, lR, rc, rR, np.ones(len(lc), bool))
        self.assertAlmostEqual(p["scale"], 4.2, places=9)
        np.testing.assert_allclose(p["rotation"], R, atol=1e-9)
        self.assertLess(p["residual_rms_m"], 1e-9)
        np.testing.assert_allclose(window_translation(lc, p["scale"], p["rotation"], rc, np.ones(len(lc), bool)),
                                   [1., 2., 3.], atol=1e-9)

    def test_stationary_window_keeps_rotation_but_has_no_extent(self):
        R = Rotation.from_rotvec([0, 0, 2.9]).as_matrix()  # near-180 deg yaw is a valid frame change
        lc, lR, rc, rR = window(3., R, np.zeros(3), stationary=True)
        p = trajectory_prior(lc, lR, rc, rR, np.ones(len(lc), bool))
        np.testing.assert_allclose(p["rotation"], R, atol=1e-9)
        self.assertEqual(p["extent_m"], 0.)

    def test_needs_two_valid_frames(self):
        lc, lR, rc, rR = window(1., np.eye(3), np.zeros(3))
        valid = np.zeros(len(lc), bool); valid[3] = True
        self.assertIsNone(trajectory_prior(lc, lR, rc, rR, valid))


class SolveScalesTest(unittest.TestCase):
    def prior(self, scale, extent):
        return {"scale": scale, "extent_m": extent, "residual_rms_m": 0., "rotation": np.eye(3), "frames": 10}

    def test_edges_carry_scale_into_stationary_windows(self):
        # Window 0 stationary, 1 and 2 moving; s_j = s_i * rel.
        truth = {0: 2., 1: 3., 2: 4.5}
        priors = {0: self.prior(9., 0.), 1: self.prior(3., 1.), 2: self.prior(4.5, 1.)}
        edges = [(0, 1, truth[1] / truth[0]), (1, 2, truth[2] / truth[1])]
        scales, constrained, diag = solve_window_scales([0, 1, 2], priors, edges)
        self.assertEqual(constrained, {0, 1, 2})
        for w, s in truth.items():
            self.assertAlmostEqual(scales[w], s, places=9)
        self.assertFalse(diag[0]["trajectory_prior_used"])

    def test_component_without_prior_stays_pending(self):
        priors = {0: self.prior(2., 1.), 1: self.prior(2., 1.), 2: None, 3: self.prior(5., .1)}
        edges = [(0, 1, 1.), (2, 3, 1.5)]
        scales, constrained, _ = solve_window_scales([0, 1, 2, 3], priors, edges)
        self.assertEqual(constrained, {0, 1})
        self.assertNotIn(2, scales)
        self.assertNotIn(3, scales)

    def test_ignores_nonpositive_edges_and_scales(self):
        priors = {0: self.prior(-1., 2.), 1: self.prior(2., 2.)}
        scales, constrained, _ = solve_window_scales([0, 1], priors, [(0, 1, 0.)])
        self.assertEqual(constrained, {1})
        self.assertAlmostEqual(scales[1], 2.)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            FramePoseConfig(pose_time_offset_ms=5000)
        with self.assertRaises(ValueError):
            FramePoseConfig(edge_weight=0)


class SharedDepthRatioTest(unittest.TestCase):
    @staticmethod
    def geometry(wid, ids, depth, prefix):
        n = len(ids)
        conf = np.ones_like(depth)
        conf[:, :, :16] = 2.  # half the pixels reach the frame median, half exceed it
        K = np.repeat(np.array([[20., 0, 15.5], [0, 20., 15.5], [0, 0, 1.]])[None], n, axis=0)
        return WindowGeometry(wid, ids, np.arange(n, dtype=np.int64) + 10 * wid, depth, conf, K,
                              np.repeat(np.eye(3)[None], n, axis=0), np.zeros((n, 3)), prefix)

    def test_ratio_of_shared_frame_depths(self):
        rng = np.random.default_rng(0)
        d0 = (1 + rng.random((4, 32, 32))) * 3.
        d1 = (1 + rng.random((4, 32, 32)))
        d1[:2] = d0[2:] / 3.  # same images, a third of the depth unit
        previous = self.geometry(0, [0, 1, 2, 3], d0, 0)
        current = self.geometry(1, [2, 3, 4, 5], d1, 2)
        ratio, used = shared_depth_ratio(previous, current, min_pixels=100)
        self.assertAlmostEqual(ratio, 3.0, places=6)
        self.assertEqual(used, 2)

    def test_no_shared_identity_gives_no_edge(self):
        previous = self.geometry(0, [0, 1, 2], np.ones((3, 32, 32)), 0)
        current = self.geometry(1, [7, 8, 9], np.ones((3, 32, 32)), 2)
        self.assertEqual(shared_depth_ratio(previous, current), (None, 0))


class InterpolatePosesTest(unittest.TestCase):
    def test_interpolates_without_extrapolating_or_bridging_gaps(self):
        t = np.array([0, 100, 200, 2000], np.int64)
        c = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        R = Rotation.from_rotvec([[0, 0, 0], [0, 0, .2], [0, 0, .4], [0, 0, .6]]).as_matrix()
        q = np.array([50, 100, 150, -10, 1000, 2500], np.int64)
        out_c, out_R, valid = interpolate_poses(q, t, c, R, max_gap_ns=500)
        np.testing.assert_array_equal(valid, [True, True, True, False, False, False])
        np.testing.assert_allclose(out_c[0], [.5, 0, 0])
        np.testing.assert_allclose(Rotation.from_matrix(out_R[2]).as_rotvec(), [0, 0, .3], atol=1e-12)
        np.testing.assert_allclose(out_c[1], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
