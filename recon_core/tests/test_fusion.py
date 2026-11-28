import unittest

import numpy as np

from recon_core.fusion import (_PACK_BITS, confidence_mask, footprint_cloud, footprint_levels, frame_points,
                               gaussian_init_cloud, partners, voxel_down_sample)


def reference_voxel(points, colors, voxel):
    """Row-wise unique + add.at, the 2026-10-29 implementation."""
    points, colors = np.asarray(points, np.float64), np.asarray(colors, np.float64)
    indices = np.floor((points - (points.min(axis=0) - voxel * 0.5)) / voxel).astype(np.int64)
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
        for scale, voxel in ((3.0, 0.02), (40.0, 0.05), (0.1, 0.02), (500.0, 0.3)):
            points = rng.normal(size=(200000, 3)) * scale + rng.normal(size=3) * 100
            points[::17] = points[::17].round(2)  # many points exactly on voxel faces
            colors = rng.random((200000, 3))
            self.assert_bitwise(points, colors, voxel)

    def test_extent_beyond_packed_range_falls_back(self):
        points = np.array([[0, 0, 0], [0, 0, 1e-3], [(1 << _PACK_BITS) * 0.02 + 5, 0, 0]])
        colors = np.eye(3)
        self.assert_bitwise(points, colors, 0.02)
        xyz, _ = voxel_down_sample(points, colors, 0.02)
        self.assertEqual(len(xyz), 2)



class FootprintCloudTest(unittest.TestCase):
    def wall(self, n, width, seed):
        rng = np.random.default_rng(seed)
        return np.c_[rng.random(n) * width, rng.random(n) * 3, np.zeros(n)]

    def cloud(self, points, sizes, cap=0):
        xyz, rgb, info = footprint_cloud(points, np.full((len(points), 3), 100, np.uint8), sizes, cap)
        return xyz, rgb, info

    def test_cell_follows_the_point_size(self):
        near, far = self.wall(200000, 4, 1), self.wall(200000, 4, 2) + [10, 0, 0]
        points = np.r_[near, far]
        sizes = np.r_[np.full(len(near), 0.01), np.full(len(far), 0.08)]
        levels, base = footprint_levels(sizes)
        self.assertEqual(sorted(set(levels.tolist())), [int(levels.min()), int(levels.min()) + 3])
        xyz, rgb, _ = self.cloud(points, sizes)
        fine, coarse = base * 2.0 ** levels.min(), base * 2.0 ** levels.max()
        left = int((xyz[:, 0] < 5).sum())
        right = len(xyz) - left
        full = 12 / fine**2  # cells of the 4 m x 3 m near wall
        self.assertGreater(left, 0.9 * full * (1 - np.exp(-200000 / full)))
        self.assertLess(right, 1.1 * 12 / coarse**2 + 200)
        self.assertGreater(right, 0.9 * 12 / coarse**2)
        np.testing.assert_allclose(rgb, 100)

    def test_far_view_adds_nothing_where_a_near_view_has_points(self):
        near, far = self.wall(300000, 4, 3), self.wall(50000, 4, 4)
        alone, _, _ = self.cloud(near, np.full(len(near), 0.01))
        both, _, _ = self.cloud(np.r_[near, far], np.r_[np.full(len(near), 0.01), np.full(len(far), 0.08)])
        self.assertLess(len(both) - len(alone), 0.02 * len(alone))

    def test_the_unit_of_the_scene_does_not_matter(self):
        wall = self.wall(300000, 4, 5)
        sizes = np.full(len(wall), 0.0055)
        reference, _, _ = self.cloud(wall, sizes)
        for unit in (1.2, 1.41, 7.3, 0.01):
            scaled, _, _ = self.cloud(wall * unit, sizes * unit)
            self.assertEqual(len(scaled), len(reference))
            np.testing.assert_allclose(np.sort(scaled[:, 0]) / unit, np.sort(reference[:, 0]), atol=1e-9)

    def test_one_far_outlier_costs_one_point(self):
        wall = self.wall(300000, 4, 6)
        sizes = np.full(len(wall), 0.0055)
        alone, _, _ = self.cloud(wall, sizes)
        for distance in (1e4, 1e6):
            with_outlier, _, _ = self.cloud(np.r_[wall, [[distance, 0, 0]]], np.r_[sizes, [distance * 0.0055]])
            self.assertEqual(len(with_outlier), len(alone) + 1)

    def test_cap_is_met_without_overshooting(self):
        rng = np.random.default_rng(7)
        ground = np.c_[rng.random(1500000) * 150, rng.random(1500000) * 150, np.zeros(1500000)]
        sizes = 0.275 * (1 + 0.02 * rng.standard_normal(len(ground)))  # one flight height: every size nearly the same
        for cap in (50000, 150000):
            xyz, _, info = self.cloud(ground, sizes, cap)
            self.assertLessEqual(len(xyz), cap)
            self.assertGreater(len(xyz), 0.8 * cap)
            self.assertGreater(info['footprint_factor'], 1.0)
        free, _, plain = self.cloud(ground, sizes)
        self.assertEqual(plain['footprint_factor'], 1.0)
        self.assertEqual(plain['candidates'], len(ground))

    def test_empty_and_single_input(self):
        xyz, rgb, info = self.cloud(np.empty((0, 3)), np.empty(0))
        self.assertEqual((xyz.shape, rgb.shape, info['points']), ((0, 3), (0, 3), 0))
        xyz, rgb, _ = self.cloud(np.array([[1.0, 2.0, 3.0]]), np.array([0.0]))
        np.testing.assert_allclose(xyz, [[1.0, 2.0, 3.0]])


class InitMaskAndRangeTest(unittest.TestCase):
    def test_percentile_mask_contains_the_median_mask(self):
        rng = np.random.default_rng(11)
        depth, confidence = np.ones((40, 60), np.float32), rng.random((40, 60)).astype(np.float32)
        strict, loose = confidence_mask(depth, confidence), confidence_mask(depth, confidence, 20.0)
        np.testing.assert_array_equal(strict, confidence > np.median(confidence))
        self.assertTrue(loose[strict].all())
        self.assertAlmostEqual(float(loose.mean()), 0.8, delta=0.01)

    def scene(self, distance):
        """Five cameras 0.2 m apart looking along +z at a fronto-parallel plane ``distance`` away."""
        H, W, n = 24, 32, 5
        K = np.tile(np.array([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1]]), (n, 1, 1))
        poses = np.tile(np.eye(4), (n, 1, 1))
        poses[:, 0, 3] = np.arange(n) * 0.2
        depth = np.full((n, H, W), distance, np.float32)
        return poses, K, depth, np.ones((n, H, W), bool)

    def test_depth_range_applies_only_when_given(self):
        for distance, inside in ((2.0, True), (12.0, False)):
            poses, K, depth, good = self.scene(distance)
            neighbors = partners(poses)
            _, keep = frame_points(2, poses, K, depth, good, neighbors)
            _, free = frame_points(2, poses, K, depth, good, neighbors, depth_range=None)
            self.assertEqual(bool(keep.any()), inside)
            self.assertTrue(free.any())

    def test_init_cloud_keeps_a_far_plane(self):
        import os
        import tempfile

        try:
            import cv2
        except ImportError:
            self.skipTest('OpenCV is needed to read the frames for the point colours')

        poses, K, depth, good = self.scene(12.0)
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for i in range(len(poses)):
                paths[str(i)] = os.path.join(folder, '%d.png' % i)
                cv2.imwrite(paths[str(i)], np.full((24, 32, 3), 100, np.uint8))
            xyz, rgb, info = gaussian_init_cloud(np.arange(len(poses)), poses, K, depth, good, paths, max_points=0)
        self.assertGreater(len(xyz), 0)
        np.testing.assert_allclose(xyz[:, 2], 12.0, atol=1e-5)
        np.testing.assert_allclose(rgb, 100 / 255)
        self.assertEqual(info['points'], len(xyz))

if __name__ == '__main__':
    unittest.main()
