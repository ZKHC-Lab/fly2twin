import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from recon_core.scale import batch_window_scales, shared_depth_ratio, solve_window_scales


def window(wid, frames, scale, poses, depth, rotation):
    """VGGT-like window: provider trajectory seen in a rotated gauge, shrunk by ``scale``."""
    idx = np.asarray(frames)
    R = rotation.as_matrix()
    c = poses[idx, :3, 3]
    vc = (c - c[0]) @ R / scale
    vR = np.einsum('ji,njk->nik', R, poses[idx, :3, :3])
    conf = np.tile(np.arange(1024, dtype=np.float32).reshape(32, 32), (len(idx), 1, 1))
    return dict(id=wid, idx=idx, vc=vc, vR=vR, depth=depth[idx] / scale, conf=conf)


class BatchWindowScalesTest(unittest.TestCase):
    def setUp(self):
        n = 12
        self.ids = np.arange(n)
        self.poses = np.tile(np.eye(4), (n, 1, 1))
        self.poses[:, 0, 3] = np.linspace(0, 2.2, n)
        self.poses[:, :3, :3] = Rotation.from_rotvec(np.c_[np.zeros(n), np.zeros(n), np.linspace(0, .3, n)]).as_matrix()
        self.depth = (1 + np.random.default_rng(0).random((n, 32, 32))).astype(np.float32)

    def test_recovers_window_scales_from_trajectory_and_shared_depth(self):
        truth = [2., 3.5, .8]
        windows = [window('K%d' % k, range(3 * k, 3 * k + 6), s, self.poses, self.depth,
                          Rotation.from_rotvec([0, .1 * k, .2]))
                   for k, s in enumerate(truth)]
        scales, trajectory, edges = batch_window_scales(windows, self.ids, self.poses)
        np.testing.assert_allclose(scales, truth, rtol=1e-5)
        np.testing.assert_allclose([t['scale'] for t in trajectory], truth, rtol=1e-9)
        self.assertEqual([e['shared_frames'] for e in edges], [3, 3])
        np.testing.assert_allclose([e['depth_ratio'] for e in edges], [3.5 / 2, .8 / 3.5], rtol=1e-5)

    def test_unanchored_windows_raise(self):
        still = np.tile(np.eye(4), (6, 1, 1))
        win = window('K0', range(6), 1., still, self.depth, Rotation.identity())
        with self.assertRaises(ValueError):
            batch_window_scales([win], np.arange(6), still)


class SharedDepthRatioTest(unittest.TestCase):
    def test_uses_pixels_strictly_above_both_confidence_medians(self):
        d = np.ones((8, 8), np.float32)
        c = np.arange(64, dtype=np.float32).reshape(8, 8)
        self.assertEqual(shared_depth_ratio([(3 * d, c, d, c)], min_pixels=32), (3.0, 1))
        self.assertEqual(shared_depth_ratio([(3 * d, c, d, c)], min_pixels=33), (None, 0))


if __name__ == '__main__':
    unittest.main()


class CrossViewRatioTest(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[40., 0, 31.5], [0, 40., 23.5], [0, 0, 1.]])
        self.pose_i, self.pose_j = np.eye(4), np.eye(4)
        self.pose_j[0, 3] = .3  # pure sideways baseline in front of a fronto-parallel wall

    def test_scaled_source_depth_gives_its_log_ratio(self):
        from recon_core.scale import cross_view_log_ratio
        wall = np.full((48, 64), 2.)
        for stride in (1, 2, 4):
            value, pixels = cross_view_log_ratio(wall[::stride, ::stride] * 1.1, self.K, self.pose_i,
                                                 wall[::stride, ::stride], self.K, self.pose_j, stride=stride)
            self.assertAlmostEqual(value, np.log(1.1), places=12)
            self.assertGreater(pixels * stride * stride, 500)

    def test_too_few_pixels_or_out_of_gate_give_none(self):
        from recon_core.scale import cross_view_log_ratio
        wall = np.full((48, 64), 2.)
        self.assertIsNone(cross_view_log_ratio(wall * 1.5, self.K, self.pose_i, wall, self.K, self.pose_j)[0])
        far = self.pose_j.copy()
        far[0, 3] = 5.
        self.assertIsNone(cross_view_log_ratio(wall, self.K, self.pose_i, wall, self.K, far)[0])

    def test_weighted_edges_change_the_solve(self):
        prior = {"scale": 2.0, "extent_m": 1.0, "residual_rms_m": 0.0}
        priors = {0: prior, 1: dict(prior, scale=2.4)}
        plain, _, _ = solve_window_scales([0, 1], priors, [])
        weighted, _, _ = solve_window_scales([0, 1], priors, [(0, 1, 1.0, 1e6)])
        self.assertAlmostEqual(plain[1], 2.4)
        self.assertAlmostEqual(weighted[0], weighted[1], places=6)


class FrameScaleSolveTest(unittest.TestCase):
    def test_rows_move_frames_off_their_window_scale_and_unanchored_stay_nan(self):
        from recon_core.scale import solve_frame_scales
        prior = {"scale": 2.0, "extent_m": 1.0, "residual_rms_m": 0.0}
        # Windows 0 and 1, three frames each; window 2 has no prior and no rows.
        frame_window = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        temporal = [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7)]
        # Frame 5 of window 1 is 10 % deeper than the frame it overlaps (frame 2).
        rows = [(5, 2, np.log(1.1)), (2, 5, -np.log(1.1))]
        windows, frames, constrained = solve_frame_scales([0, 1, 2], {0: prior, 1: prior}, [(0, 1, 1.0)],
                                                          frame_window, rows, temporal)
        self.assertEqual(constrained, {0, 1})
        self.assertEqual(set(windows), {0, 1})
        self.assertTrue(np.all(np.isnan(frames[6:])))
        log = np.log(frames[:6] / 2.0)
        self.assertGreater(log[5] - log[2], .02)            # the row pulls frame 5 up, frame 2 down
        self.assertLess(abs(log[0]), abs(log[2]) + 1e-12)   # far from the row the deviation fades
        self.assertTrue(np.all(np.abs(log) < np.log(1.1)))

    def test_without_rows_frames_equal_the_window_solve(self):
        from recon_core.scale import solve_frame_scales
        priors = {0: {"scale": 2.0, "extent_m": 1.0, "residual_rms_m": 0.0},
                  1: {"scale": 2.4, "extent_m": 1.0, "residual_rms_m": 0.0}}
        windows, frames, _ = solve_frame_scales([0, 1], priors, [(0, 1, 1.0)], np.array([0, 0, 1, 1]), [],
                                                [(0, 1), (2, 3)])
        plain, _, _ = solve_window_scales([0, 1], priors, [(0, 1, 1.0)], edge_weight=1.0)
        for w in (0, 1):
            self.assertAlmostEqual(windows[w], plain[w], places=8)
        np.testing.assert_allclose(frames, [windows[0], windows[0], windows[1], windows[1]], rtol=1e-8)
