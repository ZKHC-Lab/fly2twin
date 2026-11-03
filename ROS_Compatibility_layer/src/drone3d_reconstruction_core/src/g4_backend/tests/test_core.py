"""Synthetic tests for geometry conventions and G4 failure handling."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from g4_backend import (Sim3, WindowGeometry, RelativeEdge, LocalizationAnchor, RegistrationConfig, LocalizationConfig, SharedPairs, from_prediction,
                        resize_intrinsics, build_shared_pairs, estimate_adjacent,
                        build_initial_states, solve_g4, graph_residuals, select_localization_anchors)
from g4_backend import sim3
from g4_backend.graph import robust_block
from g4_backend.registration import point_masks


def window(wid, ids, prefix=0, shift=0.):
    n, h, w = len(ids), 12, 16
    K = np.tile([[12., 0., 7.5], [0., 12., 5.5], [0., 0., 1.]], (n, 1, 1))
    return WindowGeometry(wid, ids, np.asarray(ids, dtype=np.int64)*200_000_000,
                          np.ones((n, h, w))*2, np.ones((n, h, w)), K,
                          np.tile(np.eye(3), (n, 1, 1)), np.tile([shift, 0, 0], (n, 1)),
                          prefix, tuple("rgb_"+str(i) for i in ids))


class AlgebraTests(unittest.TestCase):
    def test_exponential_against_matrix_exponential(self):
        rng = np.random.default_rng(50)
        for _ in range(15):
            v = rng.normal(size=7)*.25
            t = sim3.exp(v)
            a = np.zeros((4, 4))
            a[:3, :3] = sim3.skew(v[3:6])+v[6]*np.eye(3)
            a[:3, 3] = v[:3]
            m = expm(a)
            np.testing.assert_allclose(t.scale*t.rotation, m[:3, :3], atol=1e-11)
            np.testing.assert_allclose(t.translation, m[:3, 3], atol=1e-11)
            np.testing.assert_allclose(sim3.log(t), v, atol=1e-11)

    def test_direction_and_known_transform(self):
        x = np.random.default_rng(1).normal(size=(100, 3))
        expected = sim3.exp(np.array([.2, -.1, .5, .1, -.2, .05, .3]))
        observed = sim3.fit(x, sim3.apply(expected, x))
        np.testing.assert_allclose(sim3.log(sim3.compose(sim3.inverse(expected), observed)), 0, atol=1e-11)
        np.testing.assert_allclose(sim3.apply(sim3.inverse(expected), sim3.apply(expected, x)), x, atol=1e-11)
        self.assertTrue(sim3.support(np.c_[x[:, :2], np.zeros(len(x))])[0])
        self.assertFalse(sim3.support(np.c_[x[:, 0], np.zeros((len(x), 2))])[0])

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            Sim3(-1)
        with self.assertRaises(ValueError):
            Sim3(rotation=np.diag([1, 1, -1]))
        with self.assertRaises(ValueError):
            RelativeEdge(1, 1, Sim3(), np.eye(7), 1.)
        with self.assertRaises(np.linalg.LinAlgError):
            RelativeEdge(1, 2, Sim3(), -np.eye(7), 1.)


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

    def test_shared_identity_and_fit_exclusion(self):
        a, b = window(10, list(range(10))), window(11, list(range(5, 15)), 5, shift=.2)
        config = RegistrationConfig(points_per_cell=3)
        pairs = build_shared_pairs(a, b, config, exclude_fit_pixels={5: [[0., 0.]]})
        np.testing.assert_allclose(pairs.source-pairs.target, np.tile([.2, 0, 0], (len(pairs.source), 1)), atol=1e-12)
        self.assertEqual(set(pairs.split), {0, 1, 2})
        self.assertGreater(pairs.diagnostics["excluded_fit_neighborhood_points"], 0)
        self.assertEqual(pairs.diagnostics["samples_by_split"]["1"], 48)
        bad = window(12, list(range(5, 15)), 5)
        object.__setattr__(bad, "rgb_hashes", tuple("changed" for _ in bad.frame_ids))
        with self.assertRaises(ValueError):
            build_shared_pairs(a, bad)

    def test_invalid_depth_is_not_retained(self):
        a = window(0, list(range(3)))
        d = a.depth.copy()
        d[:, 4, 4] = np.nan
        d[:, 2, 2] = -1
        b = WindowGeometry(0, a.frame_ids, a.timestamps_ns, d, a.confidence, a.intrinsics,
                           a.rotations, a.centers)
        mask = point_masks(b)
        self.assertFalse(mask[:, 4, 4].any())
        self.assertFalse(mask[:, 2, 2].any())


class RegistrationTests(unittest.TestCase):
    def test_empty_observations_are_reportable(self):
        import json
        pairs = SharedPairs(0, 1, np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0),
                            np.empty(0, int), np.empty(0, int), np.empty(0, int),
                            np.empty((0, 2)), (), np.nan)
        result = estimate_adjacent(pairs)
        self.assertIsNone(result.edge)
        self.assertEqual(result.diagnostics["reason"], "empty_fit_pool")
        json.dumps(result.diagnostics, allow_nan=False)

    def test_consensus_and_degenerate_failure(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(180, 3))
        expected = sim3.exp(np.array([.3, .1, -.2, .1, .1, -.1, .2]))
        y = sim3.apply(expected, x)
        y[:20] += 10
        pairs = SharedPairs(0, 1, x, y, np.ones(180), np.ones(180), np.zeros(180, int),
                            np.arange(180)//6, np.arange(180)//96, np.zeros((180, 2)),
                            tuple(range(180)), .01)
        cfg = RegistrationConfig(ransac_trials=100, bootstrap_replicates=20, bootstrap_minimum_valid=16)
        result = estimate_adjacent(pairs, config=cfg)
        self.assertIsNotNone(result.edge)
        np.testing.assert_allclose(sim3.log(sim3.compose(sim3.inverse(expected), result.edge.measurement)), 0, atol=1e-10)
        self.assertFalse(result.diagnostics["geometry_accepted"])
        x[:, 1:] = 0
        pairs.source = x
        pairs.target = x
        rejected = estimate_adjacent(pairs, config=cfg)
        self.assertIsNone(rejected.edge)
        self.assertEqual(rejected.diagnostics["reason"], "no_noncollinear_hypothesis")


class GraphTests(unittest.TestCase):
    def test_tree_and_source_units(self):
        rng = np.random.default_rng(7)
        truth = [sim3.exp(rng.normal(size=7)*.1) for _ in range(4)]
        edges = [RelativeEdge(i, i+1, sim3.compose(sim3.inverse(truth[i]), truth[i+1]), np.eye(7)*.01, 1.) for i in range(3)]
        initial, components = build_initial_states(edges, range(4))
        result = solve_g4(initial, edges, [])
        for i in initial:
            np.testing.assert_allclose(sim3.log(sim3.compose(sim3.inverse(initial[i]), result.states[i])), 0, atol=1e-8)
        self.assertFalse(result.geometry_accepted)
        self.assertEqual(set(components.values()), {0})

    def test_known_localization_and_weak_baseline(self):
        truth = sim3.exp(np.array([.2, -.4, .6, .03, -.1, .2, .25]))
        points = np.random.default_rng(4).normal(size=(12, 3))
        anchors = [LocalizationAnchor("window", x, sim3.apply(truth, x)) for x in points]
        result = solve_g4({"window": Sim3()}, [], anchors)
        np.testing.assert_allclose(sim3.log(sim3.compose(sim3.inverse(truth), result.states["window"])), 0, atol=1e-8)
        self.assertTrue(result.diagnostics["publishable_numerical"])
        weak = solve_g4({"window": Sim3()}, [], [LocalizationAnchor("window", np.zeros(3), np.zeros(3)) for _ in range(3)])
        self.assertFalse(weak.diagnostics["publishable_numerical"])
        self.assertGreaterEqual(weak.diagnostics["weak_directions"], 4)

    def test_legacy_huber_order_is_explicit(self):
        anchor = LocalizationAnchor(0, np.zeros(3), [10., 0, 0], weight=.1)
        residual = graph_residuals({0: Sim3()}, [], [anchor])
        np.testing.assert_allclose(residual, robust_block(np.array([-10., 0, 0])))
        corrected_order = robust_block(np.array([-100., 0, 0]))*.1
        self.assertGreater(np.linalg.norm(residual-corrected_order), 1)

    def test_unknown_window_is_rejected(self):
        with self.assertRaises(ValueError):
            solve_g4({0: Sim3()}, [], [LocalizationAnchor(1, np.zeros(3), np.ones(3))])

    def test_component_rank_keeps_observable_component_separate(self):
        truth = sim3.exp(np.array([.2, -.4, .6, .03, -.1, .2, .25]))
        points = np.random.default_rng(9).normal(size=(12, 3))
        anchors = [LocalizationAnchor("good", x, sim3.apply(truth, x)) for x in points]
        anchors += [LocalizationAnchor("weak", np.zeros(3), np.zeros(3)) for _ in range(3)]
        result = solve_g4({"good": Sim3(), "weak": Sim3(), "local_only": Sim3()}, [], anchors)
        detail = {d["nodes"][0]: d for d in result.diagnostics["component_diagnostics"]}
        self.assertFalse(result.diagnostics["publishable_numerical"])
        self.assertTrue(detail["good"]["publishable_numerical"])
        self.assertEqual(detail["good"]["jacobian_rank"], 7)
        self.assertGreaterEqual(detail["weak"]["weak_directions"], 4)
        self.assertFalse(detail["weak"]["publishable_numerical"])
        self.assertEqual(detail["local_only"]["variable_count"], 0)
        self.assertFalse(detail["local_only"]["publishable_numerical"])


class LocalizationTests(unittest.TestCase):
    def test_lightweight_records_interpolation_ownership(self):
        frames = np.arange(200)
        ts = frames.astype(np.int64)*200_000_000
        path = np.c_[frames*.01, np.sin(frames*.05), frames*.001]
        windows = [SimpleNamespace(window_id=0, frame_ids=tuple(frames[:100]), timestamps_ns=ts[:100], centers=path[:100], shared_prefix=0),
                   SimpleNamespace(window_id=1, frame_ids=tuple(frames[80:]), timestamps_ns=ts[80:], centers=path[80:], shared_prefix=20)]
        selection = select_localization_anchors(windows, ts, path,
                                                config=LocalizationConfig(fit_fraction=1))
        self.assertEqual(len(selection.frame_ids), 200)
        np.testing.assert_allclose(selection.reference_centers, path)
        self.assertTrue(all(a.frame_id >= 100 for a in selection.anchors if a.window_id == 1))
        self.assertEqual(len(selection.anchors), 38)
        self.assertTrue(all(a.frame_id % 5 == 0 for a in selection.anchors))
        for block in set(a.block for a in selection.anchors):
            self.assertAlmostEqual(sum(a.weight*a.weight for a in selection.anchors if a.block == block), 1.)

    def test_body_to_camera_lever_arm(self):
        w = SimpleNamespace(window_id=0, frame_ids=(0, 1), timestamps_ns=np.array([0, 1_000_000_000]), centers=np.zeros((2, 3)), shared_prefix=0)
        ext = np.eye(4)
        ext[:3, 3] = [1, 0, 0]
        rotations = np.tile(Rotation.from_rotvec([0, 0, np.pi/2]).as_matrix(), (2, 1, 1))
        selection = select_localization_anchors([w], w.timestamps_ns, np.zeros((2, 3)), rotations, ext)
        np.testing.assert_allclose(selection.reference_centers, np.tile([0, 1, 0], (2, 1)), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
