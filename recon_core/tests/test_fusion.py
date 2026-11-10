import unittest

import numpy as np

from recon_core.fusion import _PACK_BITS, voxel_down_sample


def reference_voxel(points, colors, voxel):
    """Row-wise unique + add.at, the 2026-10-29 implementation."""
    points, colors = np.asarray(points, np.float64), np.asarray(colors, np.float64)
    indices = np.floor((points - (points.min(axis=0) - voxel * .5)) / voxel).astype(np.int64)
    _, inverse, counts = np.unique(indices, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    xyz, rgb = np.zeros((len(counts), 3)), np.zeros((len(counts), 3))
    np.add.at(xyz, inverse, points)
    np.add.at(rgb, inverse, colors)
    return xyz / counts[:, None], rgb / counts[:, None]


class VoxelDownSampleTest(unittest.TestCase):
    def assert_bitwise(self, points, colors, voxel):
        xyz, rgb = voxel_down_sample(points, colors, voxel)
        ref_xyz, ref_rgb = reference_voxel(points, colors, voxel)
        self.assertEqual(xyz.tobytes(), ref_xyz.tobytes())
        self.assertEqual(rgb.tobytes(), ref_rgb.tobytes())

    def test_packed_keys_match_rowwise_unique_bitwise(self):
        rng = np.random.default_rng(7)
        for scale, voxel in ((3.0, .02), (40.0, .05), (.1, .02), (500.0, .3)):
            points = rng.normal(size=(200000, 3)) * scale + rng.normal(size=3) * 100
            points[::17] = points[::17].round(2)  # many points exactly on voxel faces
            colors = rng.random((200000, 3))
            self.assert_bitwise(points, colors, voxel)

    def test_extent_beyond_packed_range_falls_back(self):
        points = np.array([[0, 0, 0], [0, 0, 1e-3], [(1 << _PACK_BITS) * .02 + 5, 0, 0]])
        colors = np.eye(3)
        self.assert_bitwise(points, colors, .02)
        xyz, _ = voxel_down_sample(points, colors, .02)
        self.assertEqual(len(xyz), 2)


if __name__ == '__main__':
    unittest.main()
