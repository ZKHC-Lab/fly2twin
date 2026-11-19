import unittest

import numpy as np

from recon_core.fusion import frame_points, partners, supported_points
from recon_core.revisit import frame_log_scale_corrections

W, H, F = 64, 48, 50.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
ROOM = np.array([4.0, 3.0, 2.5])                                   # inner box [0, ROOM]
FORWARD_X = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])        # camera looks along +x, image y down


def pose(x, y, yaw=0.0, origin=(0.0, 0.0, 0.0)):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]) @ FORWARD_X
    T[:3, 3] = np.array([x, y, 1.2]) + origin
    return T


def box_depth(T, origin=(0.0, 0.0, 0.0)):
    """Exact z-depth of the box interior seen from pose T, and the axis of the face each pixel hits."""
    v, u = np.mgrid[0:H, 0:W]
    ray = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones((H, W))], -1) @ T[:3, :3].T
    c = T[:3, 3] - origin
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(ray > 0, (ROOM - c) / ray, np.where(ray < 0, -c / ray, np.inf))
    return t.min(-1).astype(np.float32), t.argmin(-1)


def capture(first, second, second_origin=(0.0, 0.0, 0.0)):
    """Two windows of ten frames each; returns poses, depth, face axis, window labels."""
    poses = [pose(*p) for p in first] + [pose(*p, origin=second_origin) for p in second]
    maps = [box_depth(T) for T in poses[:len(first)]] + [box_depth(T, second_origin) for T in poses[len(first):]]
    return (np.array(poses), np.stack([m[0] for m in maps]), np.stack([m[1] for m in maps]),
            np.r_[np.zeros(len(first), int), np.ones(len(second), int)])


FAR = [(1.2, 0.9 + .12 * k, .04 * (k - 4.5)) for k in range(10)]   # 2.8 m from the wall x = 4
NEAR = [(2.9, 0.9 + .12 * k, .04 * (4.5 - k)) for k in range(10)]  # 1.1 m from it, 1.7 m from FAR: no fusion partners


def solve(poses, depth, window):
    Ks = np.repeat(K[None], len(poses), 0)
    return frame_log_scale_corrections(poses, Ks, depth, depth > 0, window)


def front_wall_x(poses, depth, axis, corrections, frames):
    """Median world x of the pixels of ``frames`` that see the wall x = ROOM[0], after the corrections."""
    v, u = np.mgrid[0:H, 0:W]
    ray = np.stack([(u - K[0, 2]) / F, (v - K[1, 2]) / F, np.ones((H, W))], -1)
    xs = []
    for f in frames:
        world = (ray * (depth[f] * np.exp(corrections[f]))[..., None]) @ poses[f, :3, :3].T + poses[f, :3, 3]
        xs.append(world[..., 0][(axis[f] == 0) & (world[..., 0] > 2.)])
    return float(np.median(np.concatenate(xs)))


class RevisitTest(unittest.TestCase):
    def test_pixel_subset_takes_the_decisions_of_frame_points(self):
        poses, depth, _, _ = capture(FAR, [(1.5 + .02 * k, 1.0 + .1 * k, .05 * k) for k in range(10)])
        depth[3, 10:20, 5:30] *= 1.2                                 # some pixels without support
        good = depth > 0
        good[5, ::3] = False
        Ks = np.repeat(K[None], len(poses), 0)
        neighbors = partners(poses)
        rows, cols = (a.reshape(-1) for a in np.mgrid[0:H:2, 0:W:4])
        kept = 0
        for i in range(len(poses)):
            xyz, keep = frame_points(i, poses, Ks, depth, good, neighbors)
            sub_xyz, sub_keep = supported_points(i, poses, Ks, depth, good, neighbors, rows, cols)
            sub_keep = sub_keep.reshape(H // 2, -1)
            np.testing.assert_array_equal(sub_keep, keep[:, ::2])
            world = np.full(keep.shape + (3,), np.nan)
            world[keep] = xyz
            np.testing.assert_allclose(sub_xyz, world[:, ::2][sub_keep], rtol=0, atol=1e-12)
            kept += int(sub_keep.sum())
        self.assertGreater(kept, 1000)
        self.assertLess(kept, len(poses) * len(rows))

    def test_consistent_capture_is_left_alone(self):
        poses, depth, _, window = capture(FAR, NEAR)
        corrections, info = solve(poses, depth, window)
        self.assertGreater(info['rounds'][0]['matches'], 500)
        self.assertLess(np.abs(corrections).max(), 2e-3)

    def test_window_with_a_scale_error_is_pulled_onto_the_other(self):
        poses, depth, axis, window = capture(FAR, NEAR)
        depth[window == 1] *= np.float32(np.exp(.04))             # the near window is 4 % too deep: its wall is 4.4 cm behind
        zero = np.zeros(len(poses))
        far, near = np.flatnonzero(window == 0), np.flatnonzero(window == 1)
        before = front_wall_x(poses, depth, axis, zero, near) - front_wall_x(poses, depth, axis, zero, far)
        corrections, info = solve(poses, depth, window)
        after = front_wall_x(poses, depth, axis, corrections, near) - front_wall_x(poses, depth, axis, corrections, far)
        self.assertGreater(before, .04)
        self.assertLess(abs(after), .01)
        self.assertLess(corrections[near].mean(), 0)              # the deep window shrinks ...
        self.assertGreater(corrections[far].mean(), 0)            # ... and the other one grows: only the gap is observed
        self.assertLess(np.abs(corrections).max(), .04)
        self.assertLess(np.abs(np.diff(corrections[near])).max(), .01)   # frames of a window move together
        self.assertEqual(len(info['rounds']), 3)

    def test_without_revisits_nothing_changes(self):
        poses, depth, _, window = capture(FAR, NEAR, second_origin=(20.0, 0.0, 0.0))   # the second window is in another room
        depth[window == 1] *= np.float32(np.exp(.04))
        corrections, info = solve(poses, depth, window)
        self.assertEqual(info['rounds'], [])
        self.assertEqual(info['correction_abs_max'], 0.)
        self.assertTrue(np.all(corrections == 0))

    def test_two_faces_of_a_thin_wall_are_not_pulled_together(self):
        # Two rooms share a 10 cm wall (x = 4.0 .. 4.1); each window sees one face from 1.45 m. Depths are exact.
        first = [(2.55, 0.9 + .12 * k, .04 * (k - 4.5)) for k in range(10)]
        second = [(1.45, 2.1 - .12 * k, np.pi + .04 * (k - 4.5)) for k in range(10)]   # in the room at x = 4.1, looking back at the wall
        poses, depth, _, window = capture(first, second, second_origin=(4.1, 0.0, 0.0))
        corrections, _ = solve(poses, depth, window)
        self.assertLess(np.abs(corrections).max(), 2e-3)

    def test_consecutive_windows_on_one_pass(self):
        # One straight pass cut into two windows: the neighbours see the same wall from almost the same place.
        first = [(1.2, 0.5 + .08 * k, 0.) for k in range(10)]
        second = [(1.2, 1.3 + .08 * k, 0.) for k in range(10)]
        poses, depth, axis, window = capture(first, second)
        corrections, info = solve(poses, depth, window)
        self.assertGreater(info['rounds'][0]['matches'], 500)
        self.assertLess(np.abs(corrections).max(), 2e-3)                   # consistent: left alone
        depth[window == 1] *= np.float32(np.exp(.03))
        zero = np.zeros(len(poses))
        a, b = np.flatnonzero(window == 0), np.flatnonzero(window == 1)
        before = front_wall_x(poses, depth, axis, zero, b) - front_wall_x(poses, depth, axis, zero, a)
        corrections, _ = solve(poses, depth, window)
        after = front_wall_x(poses, depth, axis, corrections, b) - front_wall_x(poses, depth, axis, corrections, a)
        self.assertGreater(before, .07)
        self.assertLess(abs(after), .015)
        self.assertLess(np.abs(corrections).max(), .03)                    # nobody moves by more than the error

    def test_single_window_and_repeat_calls(self):
        poses, depth, _, window = capture(FAR, NEAR)
        corrections, _ = solve(poses, depth, np.zeros(len(poses), int))
        self.assertTrue(np.all(corrections == 0))
        depth[window == 1] *= np.float32(np.exp(.03))
        a, _ = solve(poses, depth, window)
        b, _ = solve(poses, depth, window)
        self.assertEqual(a.tobytes(), b.tobytes())


if __name__ == '__main__':
    unittest.main()
