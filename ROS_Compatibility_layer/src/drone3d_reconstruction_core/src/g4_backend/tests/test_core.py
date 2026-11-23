"""Synthetic tests for the Sim(3) and window geometry conventions."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from g4_backend import Sim3, WindowGeometry, from_prediction, resize_intrinsics
from g4_backend import sim3


def window(wid, ids, prefix=0, shift=0.):
    n, h, w = len(ids), 12, 16
    K = np.tile([[12., 0., 7.5], [0., 12., 5.5], [0., 0., 1.]], (n, 1, 1))
    return WindowGeometry(wid, ids, np.asarray(ids, dtype=np.int64)*200_000_000,
                          np.ones((n, h, w))*2, np.ones((n, h, w)), K,
                          np.tile(np.eye(3), (n, 1, 1)), np.tile([shift, 0, 0], (n, 1)),
                          prefix, tuple("rgb_"+str(i) for i in ids))


class AlgebraTests(unittest.TestCase):
    def test_compose_applies_right_then_left(self):
        x = np.random.default_rng(1).normal(size=(100, 3))
        a = Sim3(2., Rotation.from_rotvec([.1, -.2, .05]).as_matrix(), [.2, -.1, .5])
        b = Sim3(.5, Rotation.from_rotvec([-.3, .1, .2]).as_matrix(), [1., 0., -2.])
        np.testing.assert_allclose(sim3.apply(sim3.compose(a, b), x), sim3.apply(a, sim3.apply(b, x)), atol=1e-12)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            Sim3(-1)
        with self.assertRaises(ValueError):
            Sim3(rotation=np.diag([1, 1, -1]))


class GeometryTests(unittest.TestCase):
    def test_first_camera_gauge_and_halfpixel(self):
        R = Rotation.from_rotvec([.1, .2, .3]).as_matrix()
        t = np.array([2., 3., 4.])
        extrinsics = np.tile(np.c_[R, t], (2, 1, 1))
        extrinsics[1, :, 3] += [1, 0, 0]
        original = window(0, [0, 1])
        result = from_prediction(0, [0, 1], original.timestamps_ns, original.depth,
                                 original.confidence, original.intrinsics, extrinsics)
        np.testing.assert_allclose(result.rotations[0], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(result.centers[0], 0, atol=1e-12)
        np.testing.assert_allclose(result.depth, 1.)
        np.testing.assert_allclose(result.centers[1], [-.5, 0, 0], atol=1e-12)
        K = np.array([[600., 0, 319.], [0, 601., 239.], [0, 0, 1]])
        resized = resize_intrinsics(K, (640, 480), (384, 288))
        np.testing.assert_allclose(resized[0, 2], (319+.5)*.6-.5)


if __name__ == "__main__":
    unittest.main()
