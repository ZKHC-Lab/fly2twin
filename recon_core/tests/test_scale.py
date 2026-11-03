import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from recon_core.scale import batch_window_scales, shared_depth_ratio


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
