"""窗口分组计划与输入策略（阈值来源、排除池、方法身份）的合成测试。"""
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from g4_backend import (METHOD_ID, WINDOWED_METHOD_ID, THRESHOLD_RECOMPUTED, THRESHOLD_RECORDED,
                        RegistrationConfig, SharedPairs, ThresholdPolicy, WindowGeometry,
                        WindowPlanError, build_shared_pairs, estimate_adjacent, load_plan,
                        plan_from_counts, plan_from_records, compare_plans, solve_g4)
from g4_backend.window_plan import (FIRST_WINDOW_FRAMES, MINIMUM_TOTAL_FRAMES, SEMANTICS_NEW,
                                    STEADY_NEW_FRAMES, STEADY_SHARED_FRAMES, WindowSpec)

# 冻结登记清单是可选的本地材料：缺失时相关用例跳过，而不是失败。
# 仓库根由本文件位置推导，可用 DRONE3D_REPO_ROOT 覆盖。
REPO_ROOT = Path(os.environ.get("DRONE3D_REPO_ROOT", Path(__file__).resolve().parents[6]))
FROZEN_PLAN = REPO_ROOT / "Data/experiments/testflight3_full_route_20260915/inputs/candidate_windows.json"


def window(wid, ids, prefix=0, shift=0., height=24, width=32):
    n = len(ids)
    K = np.tile([[16., 0., 15.5], [0., 16., 11.5], [0., 0., 1.]], (n, 1, 1))
    depth = np.ones((n, height, width))*2
    return WindowGeometry(wid, ids, np.asarray(ids, dtype=np.int64)*200_000_000,
                          depth, np.ones((n, height, width)), K,
                          np.tile(np.eye(3), (n, 1, 1)), np.tile([shift, 0, 0], (n, 1)),
                          prefix, tuple("rgb_"+str(i) for i in ids))


class WindowPlanTests(unittest.TestCase):
    def test_derived_grouping_rules(self):
        plan = plan_from_counts(1000)
        self.assertEqual(plan.windows[0].new_frames, FIRST_WINDOW_FRAMES)
        self.assertEqual(plan.windows[0].shared_prefix, 0)
        for spec in plan.windows[1:]:
            self.assertEqual(spec.new_frames, STEADY_NEW_FRAMES)
            self.assertEqual(spec.shared_prefix, STEADY_SHARED_FRAMES)
            self.assertEqual(spec.input_frames, STEADY_NEW_FRAMES + STEADY_SHARED_FRAMES)
        self.assertEqual(plan.total_frames, 1000)
        self.assertEqual([s.new_start for s in plan.windows[1:]], list(range(100, 1000, 100)))

    def test_short_final_window_borrows_shared_frames(self):
        plan = plan_from_counts(374)
        tail = plan.windows[-1]
        self.assertEqual(tail.new_frames, 74)
        self.assertEqual(tail.shared_prefix, MINIMUM_TOTAL_FRAMES - 74)
        self.assertEqual(tail.input_frames, MINIMUM_TOTAL_FRAMES)
        # 常规窗仍保持 20 共享＋100 新增。
        self.assertEqual(plan.windows[1].shared_prefix, STEADY_SHARED_FRAMES)

    def test_final_window_never_shrinks_model_input(self):
        for total in (101, 174, 199, 201, 299):
            plan = plan_from_counts(total)
            self.assertLessEqual(plan.windows[-1].shared_prefix,
                                 max(STEADY_SHARED_FRAMES, MINIMUM_TOTAL_FRAMES - plan.windows[-1].new_frames))
            self.assertGreater(plan.windows[-1].input_frames, 0)

    def test_single_short_window(self):
        plan = plan_from_counts(37)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan.windows[0].new_frames, 37)
        self.assertEqual(plan.windows[0].shared_prefix, 0)

    def test_validation_rejects_contradictions(self):
        with self.assertRaises(WindowPlanError):
            WindowSpec(0, "F000", 0, 10, 5, 10, 5)
        with self.assertRaises(WindowPlanError):
            WindowSpec(0, "F000", 0, 0, 0, 0, 0)
        with self.assertRaises(WindowPlanError):
            WindowSpec(0, "F000", 10, 20, 9, 20, 5)
        # 帧覆盖的连续性属于计划级约束，由 WindowPlan 校验。
        with self.assertRaises(WindowPlanError):
            plan_from_records([
                {"id": "F000", "new_start": 0, "new_end_exclusive": 10,
                 "input_start": 0, "input_end_exclusive": 10, "shared_prefix": 0},
                {"id": "F001", "new_start": 15, "new_end_exclusive": 20,
                 "input_start": 10, "input_end_exclusive": 20, "shared_prefix": 5},
            ])
        with self.assertRaises(WindowPlanError):
            plan_from_counts(1000, shared=100, minimum_total=100)
        records = plan_from_counts(300).to_records()
        with self.assertRaises(WindowPlanError):
            plan_from_records(records, total_frames=299)

    def test_registered_plan_round_trip(self):
        plan = plan_from_counts(374)
        again = plan_from_records(plan.to_records(), source="round_trip")
        self.assertTrue(compare_plans(plan, again)["equal"])
        # digest 只描述分组本身，与声明来源无关。
        self.assertEqual(plan.digest(), again.digest())
        self.assertEqual(plan.semantics, SEMANTICS_NEW)

    @unittest.skipUnless(FROZEN_PLAN.exists(), "frozen candidate list is not present")
    def test_derivation_reproduces_the_registered_baseline_grouping(self):
        registered = load_plan(FROZEN_PLAN)
        derived = plan_from_counts(registered.total_frames)
        report = compare_plans(derived, registered)
        self.assertTrue(report["equal"], report["differences"][:5])
        self.assertEqual(len(registered), 38)
        self.assertEqual(registered.windows[-1].shared_prefix, 26)
        self.assertEqual(registered.windows[-1].new_frames, 74)
        self.assertEqual(registered.model_frames[-1], 100)
        self.assertEqual(registered.model_frames[1], 120)
        self.assertEqual(registered.windows[23].new_start, 2300)

    @unittest.skipUnless(FROZEN_PLAN.exists(), "frozen candidate list is not present")
    def test_registered_plan_digest_is_stable(self):
        first = load_plan(FROZEN_PLAN)
        second = load_plan(FROZEN_PLAN)
        self.assertEqual(first.digest(), second.digest())
        self.assertEqual(first.digest(), plan_from_counts(3774).digest())


class ThresholdPolicyTests(unittest.TestCase):
    def test_recorded_value_is_reused_and_labelled(self):
        older, newer = window(0, [0, 1], prefix=0), window(1, [0, 1, 2], prefix=2, shift=0.5)
        policy = ThresholdPolicy({(0, 1): 0.25}, source="registered_table.json")
        pairs = build_shared_pairs(older, newer, exclude_fit_pixels=None, threshold_policy=policy)
        self.assertAlmostEqual(pairs.threshold, 0.25)
        self.assertEqual(pairs.threshold_source, "registered_table.json")
        self.assertEqual(pairs.diagnostics["threshold_source"], "registered_table.json")
        self.assertAlmostEqual(pairs.diagnostics["threshold_recomputed"],
                               RegistrationConfig().threshold_depth_fraction*np.median(older.depth))

    def test_recomputed_value_is_labelled_as_recomputed(self):
        older, newer = window(0, [0, 1], prefix=0), window(1, [0, 1, 2], prefix=2, shift=0.5)
        pairs = build_shared_pairs(older, newer)
        self.assertAlmostEqual(pairs.threshold,
                               RegistrationConfig().threshold_depth_fraction*np.median(older.depth))
        self.assertEqual(pairs.threshold_source, THRESHOLD_RECOMPUTED)
        self.assertEqual(pairs.diagnostics["threshold_source"], THRESHOLD_RECOMPUTED)

    def test_unused_recorded_edge_is_not_substituted(self):
        older, newer = window(0, [0, 1], prefix=0), window(1, [0, 1, 2], prefix=2, shift=0.5)
        policy = ThresholdPolicy({(7, 8): 0.25})
        pairs = build_shared_pairs(older, newer, threshold_policy=policy)
        self.assertEqual(pairs.threshold_source, THRESHOLD_RECOMPUTED)

    def test_explicit_argument_wins_and_is_labelled(self):
        older, newer = window(0, [0, 1], prefix=0), window(1, [0, 1, 2], prefix=2, shift=0.5)
        pairs = build_shared_pairs(older, newer, threshold=0.125)
        self.assertAlmostEqual(pairs.threshold, 0.125)
        self.assertEqual(pairs.threshold_source, "explicit_argument")

    def test_conflicting_alias_keys_are_rejected_in_both_orders(self):
        """0|1 与 F000|F001 规范化成同一边：值不同必须在加载时拒绝，不能 last-wins。"""
        with self.assertRaises(ValueError):
            ThresholdPolicy({"0|1": 0.03, "F000|F001": 3.0})
        with self.assertRaises(ValueError):
            ThresholdPolicy({"F000|F001": 3.0, "0|1": 0.03})

    def test_identical_alias_keys_collapse_to_one_edge(self):
        """相同值的重复拼写是同一策略：合并成一条，且与单键写法摘要一致。"""
        duplicated = ThresholdPolicy({"0|1": 0.03, "F000|F001": 0.03})
        single = ThresholdPolicy({"0|1": 0.03})
        self.assertEqual(len(duplicated), 2)          # 原始键仍逐条记录
        self.assertEqual(duplicated.canonical, single.canonical)
        self.assertAlmostEqual(duplicated.lookup("F000", "F001")[0], 0.03)
        self.assertAlmostEqual(duplicated.lookup(0, 1)[0], 0.03)

    def test_from_records_rejects_duplicate_edges_with_different_values(self):
        records = [{"i": 0, "j": 1, "threshold": 0.03},
                   {"i": "F000", "j": "F001", "threshold": 3.0}]
        with self.assertRaises(ValueError):
            ThresholdPolicy.from_records(records)
        duplicate_raw = [{"i": 0, "j": 1, "threshold": 0.03},
                         {"i": 0, "j": 1, "threshold": 3.0}]
        with self.assertRaises(ValueError):
            ThresholdPolicy.from_records(duplicate_raw)

    def test_from_records_collapses_identical_duplicates(self):
        policy = ThresholdPolicy.from_records([{"i": 0, "j": 1, "threshold": 0.03},
                                               {"i": 0, "j": 1, "threshold": 0.03}])
        self.assertEqual(len(policy), 1)
        self.assertAlmostEqual(policy.lookup(0, 1)[0], 0.03)

    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            ThresholdPolicy({(0, 1): -1})
        with self.assertRaises(ValueError):
            ThresholdPolicy({(0,): 1})
        policy = ThresholdPolicy.from_records([{"from": "F000", "to": "F001", "value": 0.5}],
                                              i_field="from", j_field="to", value_field="value")
        self.assertAlmostEqual(policy.lookup("F000", "F001")[0], 0.5)
        self.assertEqual(policy.lookup("F000", "F002"), (None, THRESHOLD_RECOMPUTED))
        self.assertEqual(ThresholdPolicy().source, THRESHOLD_RECORDED)


class FitExclusionTests(unittest.TestCase):
    def test_only_fit_samples_are_excluded_and_counted(self):
        older = window(0, [0, 1, 2, 3, 4, 5], prefix=0)
        newer = window(1, [0, 1, 2, 3, 4, 5, 6], prefix=6, shift=0.5)
        pixels = np.array([[1., 1.], [2., 2.]])
        plain = build_shared_pairs(older, newer)
        # k%5<3 才是 fit；k=3 属于检查划分，不应被排除也不改变计数。
        excluded = build_shared_pairs(older, newer, exclude_fit_pixels={0: pixels, 1: pixels, 3: pixels},
                                      exclusion_source="registered_report_table.json")
        self.assertEqual(excluded.diagnostics["exclusion_source"], "registered_report_table.json")
        self.assertGreater(excluded.diagnostics["excluded_fit_neighborhood_points"], 0)
        self.assertEqual(excluded.diagnostics["excluded_fit_frames"], 2)
        # 只影响 fit（k%5<3），检查/留出样本数不变。
        self.assertEqual(plain.diagnostics["samples_by_split"]["1"], excluded.diagnostics["samples_by_split"]["1"])
        self.assertEqual(plain.diagnostics["samples_by_split"]["2"], excluded.diagnostics["samples_by_split"]["2"])

    def test_exclusion_keeps_seed_reproducible(self):
        older = window(0, [0, 1, 2, 3, 4, 5], prefix=0)
        newer = window(1, [0, 1, 2, 3, 4, 5, 6], prefix=6, shift=0.5)
        first = build_shared_pairs(older, newer, exclude_fit_pixels={0: [[1., 1.]]})
        second = build_shared_pairs(older, newer, exclude_fit_pixels={0: [[1., 1.]]})
        np.testing.assert_array_equal(first.model_pixels, second.model_pixels)
        np.testing.assert_array_equal(first.source, second.source)

    def test_mixed_identifier_types_still_match(self):
        """ROS 记录用整数帧号，JSON 表用字符串键；两者必须能对上。"""
        older = window(0, [0, 1, 2, 3, 4, 5], prefix=0)
        newer = window(1, [0, 1, 2, 3, 4, 5, 6], prefix=6, shift=0.5)
        pixels = np.array([[1., 1.], [2., 2.]])
        integer_keys = build_shared_pairs(older, newer, exclude_fit_pixels={0: pixels})
        string_keys = build_shared_pairs(older, newer, exclude_fit_pixels={"0": pixels})
        self.assertEqual(integer_keys.diagnostics["excluded_fit_neighborhood_points"],
                         string_keys.diagnostics["excluded_fit_neighborhood_points"])
        self.assertGreater(integer_keys.diagnostics["excluded_fit_neighborhood_points"], 0)
        self.assertEqual(integer_keys.diagnostics["excluded_fit_frames"], 1)

    def test_invalid_exclusions_are_rejected(self):
        older, newer = window(0, [0, 1], prefix=0), window(1, [0, 1, 2], prefix=2, shift=0.5)
        with self.assertRaises(ValueError):
            build_shared_pairs(older, newer, exclude_fit_pixels={0: np.zeros((2, 3))})
        with self.assertRaises(ValueError):
            build_shared_pairs(older, newer, exclude_fit_pixels={0: np.array([[np.nan, 0.]])})

    def test_identifier_spelling_is_normalized_both_ways(self):
        from g4_backend import canonical_identifier
        self.assertEqual(canonical_identifier("F000"), canonical_identifier(0))
        self.assertEqual(canonical_identifier("000"), canonical_identifier("0"))
        self.assertNotEqual(canonical_identifier("F000"), canonical_identifier("F001"))
        policy = ThresholdPolicy({"F000|F001": 0.5})
        self.assertAlmostEqual(policy.lookup(0, 1)[0], 0.5)
        self.assertAlmostEqual(policy.lookup("000", "001")[0], 0.5)
        self.assertEqual(policy.lookup(0, 2), (None, THRESHOLD_RECOMPUTED))
        reverse = ThresholdPolicy({(0, 1): 0.25})
        self.assertAlmostEqual(reverse.lookup("F000", "F001")[0], 0.25)

    def test_records_without_a_threshold_are_skipped_not_zeroed(self):
        records = [{"i": 0, "j": 1, "threshold": 0.5},
                   {"i": 1, "j": 2},
                   {"i": 2, "j": 3, "threshold": None}]
        policy = ThresholdPolicy.from_records(records)
        self.assertEqual(len(policy), 1)
        self.assertEqual(policy.skipped_records, 2)
        self.assertAlmostEqual(policy.lookup(0, 1)[0], 0.5)
        self.assertEqual(policy.lookup(1, 2)[0], None)


class NonFiniteDepthTests(unittest.TestCase):
    def test_nonfinite_depth_is_counted_in_diagnostics(self):
        older = window(0, [0, 1], prefix=0)
        newer = window(1, [0, 1, 2], prefix=2, shift=0.5)
        clean = build_shared_pairs(older, newer)
        self.assertEqual(clean.diagnostics["nonfinite_depth_values"], 0)
        depth = np.array(newer.depth, copy=True)
        depth[0, 3, 3] = np.nan
        depth[0, 4, 4] = np.inf
        holed = WindowGeometry(newer.window_id, newer.frame_ids, newer.timestamps_ns, depth,
                               newer.confidence, newer.intrinsics, newer.rotations, newer.centers,
                               newer.shared_prefix, newer.rgb_hashes)
        pairs = build_shared_pairs(older, holed)
        self.assertEqual(pairs.diagnostics["nonfinite_depth_values"], 2)


class RansacDegenerateTests(unittest.TestCase):
    def test_degenerate_hypothesis_semantics_are_selectable(self):
        from unittest import mock
        from g4_backend import registration, sim3 as sim3_module
        x = np.random.default_rng(3).normal(size=(60, 3))
        y = sim3_module.apply(sim3_module.exp(np.array([.1, .2, .3, .01, .02, .03, .05])), x)
        real_fit = registration.sim3.fit

        def make_flaky():
            calls = {"n": 0}

            def flaky(source, target, weights=None):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise ValueError("zero source spread")
                return real_fit(source, target, weights)
            return flaky

        with mock.patch.object(registration.sim3, "fit", side_effect=make_flaky()):
            transform, _ = registration.ransac(x, y, 0.5, 20260915,
                                               RegistrationConfig(ransac_degenerate="skip"))
            self.assertTrue(np.isfinite(transform.scale))
        with mock.patch.object(registration.sim3, "fit", side_effect=make_flaky()):
            with self.assertRaises(ValueError):
                registration.ransac(x, y, 0.5, 20260915, RegistrationConfig(ransac_degenerate="abort"))
        with self.assertRaises(ValueError):
            RegistrationConfig(ransac_degenerate="ignore")


class MethodIdentityTests(unittest.TestCase):
    def test_identity_travels_with_the_graph_result(self):
        from g4_backend import Sim3, LocalizationAnchor, RelativeEdge
        edge = RelativeEdge(0, 1, Sim3(1., np.eye(3), [1., 0, 0]), np.eye(7)*1e-4, 1., 40)
        anchor = LocalizationAnchor(0, np.zeros(3), np.zeros(3))
        result = solve_g4({0: Sim3(), 1: Sim3()}, [edge], [anchor], method_id=WINDOWED_METHOD_ID)
        self.assertEqual(result.method_id, WINDOWED_METHOD_ID)
        self.assertEqual(result.diagnostics["method_id"], WINDOWED_METHOD_ID)
        default = solve_g4({0: Sim3(), 1: Sim3()}, [edge], [anchor])
        self.assertEqual(default.method_id, METHOD_ID)
        self.assertNotEqual(METHOD_ID, WINDOWED_METHOD_ID)


if __name__ == "__main__":
    unittest.main()
