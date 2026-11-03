"""Deterministic CPU-only tests for reconstruction resource admission."""

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import sys
import unittest

from drone3d_reconstruction_core.resource_gate import (
    HIGH_SWAP_GROWTH,
    HIGH_SWAP_USED,
    LOW_MEM_AVAILABLE,
    LOW_RESULT_DISK,
    LOW_SNAPSHOT_DISK,
    ResourceBusyError,
    ResourceGateError,
    ResourceLimits,
    ResourceSample,
    enforce_resources,
    evaluate_resources,
    parse_meminfo,
)


class Sequence:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class ResourceGateTest(unittest.TestCase):
    def limits(self, **changes):
        values = dict(
            min_mem_available_bytes=100,
            max_swap_used_bytes=100,
            max_swap_growth_bytes=100,
            min_snapshot_free_bytes=100,
            min_result_free_bytes=100,
            swap_sample_window_sec=1.0,
        )
        values.update(changes)
        return ResourceLimits(**values)

    def sample(self, **changes):
        values = dict(
            first_mem_available_bytes=100,
            second_mem_available_bytes=100,
            first_swap_used_bytes=0,
            second_swap_used_bytes=100,
            swap_growth_bytes=100,
            snapshot_free_bytes=100,
            result_free_bytes=100,
            observed_window_sec=1.0,
        )
        values.update(changes)
        return ResourceSample(**values)

    def meminfo(self, available_kb, total_kb, free_kb):
        return (
            f"MemTotal: 999 kB\nMemAvailable: {available_kb} kB\n"
            f"SwapTotal: {total_kb} kB\nSwapFree: {free_kb} kB\n"
        )

    def test_contracts_are_frozen_and_limits_have_no_defaults(self):
        with self.assertRaises(TypeError):
            ResourceLimits()
        with self.assertRaises(FrozenInstanceError):
            self.limits().max_swap_used_bytes = 2
        with self.assertRaises(FrozenInstanceError):
            self.sample().snapshot_free_bytes = 2

    def test_strict_meminfo_parser(self):
        self.assertEqual(parse_meminfo(self.meminfo(7, 5, 2)), (7 * 1024, 3 * 1024))
        invalid = (
            "MemAvailable: 01 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
            "MemAvailable: +1 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
            "MemAvailable: -1 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
            "MemAvailable: 1 KB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
            "MemAvailable: 1 kB extra\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
            "MemAvailable: 1 kB\nSwapTotal: 0 kB\n",
            "MemAvailable: 1 kB\nMemAvailable: 2 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
            "MemAvailable: 1 kB\nSwapTotal: 1 kB\nSwapFree: 2 kB\n",
            f"MemAvailable: {sys.maxsize // 1024 + 1} kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
        )
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ResourceGateError):
                parse_meminfo(text)

    def test_stable_denial_order_and_equality_passes(self):
        equality = evaluate_resources(self.sample(), self.limits())
        self.assertTrue(equality.allowed)
        self.assertIsNone(equality.denial_code)
        cases = (
            (dict(first_mem_available_bytes=99), LOW_MEM_AVAILABLE),
            (dict(second_mem_available_bytes=99), LOW_MEM_AVAILABLE),
            (dict(first_swap_used_bytes=101, second_swap_used_bytes=100, swap_growth_bytes=0), HIGH_SWAP_USED),
            (dict(second_swap_used_bytes=101, swap_growth_bytes=101), HIGH_SWAP_USED),
            (dict(first_swap_used_bytes=1, second_swap_used_bytes=102, swap_growth_bytes=101), HIGH_SWAP_USED),
            (dict(first_swap_used_bytes=0, second_swap_used_bytes=100, swap_growth_bytes=100,
                  snapshot_free_bytes=99), LOW_SNAPSHOT_DISK),
            (dict(result_free_bytes=99), LOW_RESULT_DISK),
        )
        for changes, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(evaluate_resources(self.sample(**changes), self.limits()).denial_code, expected)
        growth = self.sample(first_swap_used_bytes=0, second_swap_used_bytes=10, swap_growth_bytes=10)
        decision = evaluate_resources(growth, self.limits(max_swap_growth_bytes=9))
        self.assertEqual(decision.denial_code, HIGH_SWAP_GROWTH)
        all_bad = self.sample(
            second_mem_available_bytes=0, second_swap_used_bytes=200,
            swap_growth_bytes=200, snapshot_free_bytes=0, result_free_bytes=0,
        )
        self.assertEqual(evaluate_resources(all_bad, self.limits()).denial_code, LOW_MEM_AVAILABLE)

    def test_enforce_samples_twice_without_real_sleep_and_clamps_swap_growth(self):
        meminfos = Sequence((self.meminfo(10, 8, 2), self.meminfo(9, 8, 4)))
        sleeps = []
        statvfs_calls = []

        def fake_statvfs(root):
            statvfs_calls.append(root)
            return SimpleNamespace(f_bavail=10, f_frsize=20)

        decision = enforce_resources(
            self.limits(
                min_mem_available_bytes=9 * 1024,
                max_swap_used_bytes=6 * 1024,
                max_swap_growth_bytes=0,
                min_snapshot_free_bytes=200,
                min_result_free_bytes=200,
            ),
            "/snapshot", "/result",
            meminfo_reader=meminfos,
            monotonic=Sequence((4.0, 5.0)),
            sleeper=sleeps.append,
            statvfs=fake_statvfs,
            stat=lambda root: SimpleNamespace(st_dev=7),
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.sample.swap_growth_bytes, 0)
        self.assertEqual(sleeps, [1.0])
        self.assertEqual(statvfs_calls, ["/snapshot"])
        self.assertEqual(decision.sample.snapshot_free_bytes, 200)
        self.assertEqual(decision.sample.result_free_bytes, 200)

    def test_shared_device_uses_one_sample_but_independent_floors(self):
        with self.assertRaises(ResourceBusyError) as caught:
            enforce_resources(
                self.limits(
                    min_mem_available_bytes=0, max_swap_used_bytes=0,
                    max_swap_growth_bytes=0, min_snapshot_free_bytes=100,
                    min_result_free_bytes=101,
                ),
                "/snapshot", "/result",
                meminfo_reader=Sequence((self.meminfo(1, 0, 0), self.meminfo(1, 0, 0))),
                monotonic=Sequence((0.0, 1.0)), sleeper=lambda _: None,
                statvfs=lambda root: SimpleNamespace(f_bavail=10, f_frsize=10),
                stat=lambda root: SimpleNamespace(st_dev=12),
            )
        self.assertEqual(caught.exception.decision.denial_code, LOW_RESULT_DISK)
        self.assertEqual(caught.exception.decision.sample.snapshot_free_bytes, 100)
        self.assertEqual(caught.exception.decision.sample.result_free_bytes, 100)

    def test_distinct_devices_are_sampled_independently(self):
        calls = []
        free = {"/snapshot": 3, "/result": 7}

        decision = enforce_resources(
            self.limits(
                min_mem_available_bytes=0, max_swap_used_bytes=0,
                max_swap_growth_bytes=0, min_snapshot_free_bytes=30,
                min_result_free_bytes=70,
            ),
            "/snapshot", "/result",
            meminfo_reader=Sequence((self.meminfo(1, 0, 0), self.meminfo(1, 0, 0))),
            monotonic=Sequence((0.0, 1.0)), sleeper=lambda _: None,
            statvfs=lambda root: (calls.append(root) or SimpleNamespace(f_bavail=free[root], f_frsize=10)),
            stat=lambda root: SimpleNamespace(st_dev={"/snapshot": 1, "/result": 2}[root]),
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(calls, ["/snapshot", "/result"])

    def test_invalid_policy_and_probe_fail_permanently(self):
        invalid_limits = (
            replace(self.limits(), min_mem_available_bytes=-1),
            replace(self.limits(), max_swap_used_bytes=True),
            replace(self.limits(), swap_sample_window_sec=0),
            replace(self.limits(), swap_sample_window_sec=float("inf")),
        )
        for limits in invalid_limits:
            with self.subTest(limits=limits), self.assertRaises(ResourceGateError):
                evaluate_resources(self.sample(), limits)

        common = dict(
            limits=self.limits(min_mem_available_bytes=0, max_swap_used_bytes=0,
                               max_swap_growth_bytes=0, min_snapshot_free_bytes=0,
                               min_result_free_bytes=0),
            snapshot_root="/snapshot", result_root="/result",
            meminfo_reader=Sequence((self.meminfo(1, 0, 0), self.meminfo(1, 0, 0))),
            sleeper=lambda _: None,
            statvfs=lambda root: SimpleNamespace(f_bavail=1, f_frsize=1),
            stat=lambda root: SimpleNamespace(st_dev=1),
        )
        with self.assertRaises(ResourceGateError):
            enforce_resources(monotonic=Sequence((2.0, 1.0)), **common)

        common["meminfo_reader"] = Sequence((self.meminfo(1, 0, 0), self.meminfo(1, 0, 0)))
        with self.assertRaises(ResourceGateError):
            enforce_resources(monotonic=Sequence((1.0, 1.5)), **common)

        malformed_sample = replace(self.sample(), swap_growth_bytes=99)
        with self.assertRaises(ResourceGateError):
            evaluate_resources(malformed_sample, self.limits())

    def test_monotonic_rejects_invalid_values_and_normalizes_overflow(self):
        common = dict(
            limits=self.limits(
                min_mem_available_bytes=0, max_swap_used_bytes=0,
                max_swap_growth_bytes=0, min_snapshot_free_bytes=0,
                min_result_free_bytes=0,
            ),
            snapshot_root="/snapshot", result_root="/result",
            meminfo_reader=Sequence((self.meminfo(1, 0, 0), self.meminfo(1, 0, 0))),
            sleeper=lambda _: None,
            statvfs=lambda root: SimpleNamespace(f_bavail=1, f_frsize=1),
            stat=lambda root: SimpleNamespace(st_dev=1),
        )
        for value in (True, "1", 10 ** 10000, float("inf"), float("nan")):
            common["meminfo_reader"] = Sequence(
                (self.meminfo(1, 0, 0), self.meminfo(1, 0, 0))
            )
            with self.subTest(value=type(value).__name__), self.assertRaises(ResourceGateError):
                enforce_resources(monotonic=Sequence((value, 1.0)), **common)

    def test_timing_validation_normalizes_invalid_values(self):
        huge = 10 ** 10000
        for value in (True, "1", huge):
            with self.subTest(configured=type(value).__name__), self.assertRaises(ResourceGateError):
                evaluate_resources(
                    self.sample(), self.limits(swap_sample_window_sec=value)
                )
            with self.subTest(observed=type(value).__name__), self.assertRaises(ResourceGateError):
                evaluate_resources(
                    self.sample(observed_window_sec=value), self.limits()
                )

    def test_zero_fragment_size_is_invalid_even_with_zero_floors(self):
        with self.assertRaises(ResourceGateError):
            enforce_resources(
                self.limits(
                    min_mem_available_bytes=0, max_swap_used_bytes=0,
                    max_swap_growth_bytes=0, min_snapshot_free_bytes=0,
                    min_result_free_bytes=0,
                ),
                "/snapshot", "/result",
                meminfo_reader=Sequence((self.meminfo(1, 0, 0), self.meminfo(1, 0, 0))),
                monotonic=Sequence((0.0, 1.0)), sleeper=lambda _: None,
                statvfs=lambda root: SimpleNamespace(f_bavail=0, f_frsize=0),
                stat=lambda root: SimpleNamespace(st_dev=1),
            )

    def test_malformed_device_ids_are_resource_gate_errors(self):
        for device in (-1, True, [], 10 ** 10000):
            with self.subTest(device_type=type(device).__name__), self.assertRaises(ResourceGateError):
                enforce_resources(
                    self.limits(
                        min_mem_available_bytes=0, max_swap_used_bytes=0,
                        max_swap_growth_bytes=0, min_snapshot_free_bytes=0,
                        min_result_free_bytes=0,
                    ),
                    "/snapshot", "/result",
                    meminfo_reader=Sequence((self.meminfo(1, 0, 0), self.meminfo(1, 0, 0))),
                    monotonic=Sequence((0.0, 1.0)), sleeper=lambda _: None,
                    statvfs=lambda root: SimpleNamespace(f_bavail=1, f_frsize=1),
                    stat=lambda root, value=device: SimpleNamespace(st_dev=value),
                )

    def test_process_control_exceptions_propagate_from_resource_probes(self):
        process_control = (KeyboardInterrupt, SystemExit, GeneratorExit)

        def raising(exception_type):
            def probe(*_):
                raise exception_type()
            return probe

        for exception_type in process_control:
            probes = (
                dict(sleeper=raising(exception_type)),
                dict(meminfo_reader=raising(exception_type)),
                dict(monotonic=raising(exception_type)),
                dict(stat=raising(exception_type)),
                dict(statvfs=raising(exception_type)),
            )
            for override in probes:
                defaults = dict(
                    meminfo_reader=Sequence((self.meminfo(1, 0, 0), self.meminfo(1, 0, 0))),
                    monotonic=Sequence((0.0, 1.0)), sleeper=lambda _: None,
                    statvfs=lambda root: SimpleNamespace(f_bavail=1, f_frsize=1),
                    stat=lambda root: SimpleNamespace(st_dev=1),
                )
                defaults.update(override)
                with self.subTest(exception=exception_type.__name__, probe=tuple(override)):
                    with self.assertRaises(exception_type):
                        enforce_resources(
                            self.limits(
                                min_mem_available_bytes=0, max_swap_used_bytes=0,
                                max_swap_growth_bytes=0, min_snapshot_free_bytes=0,
                                min_result_free_bytes=0,
                            ),
                            "/snapshot", "/result", **defaults,
                        )

    def test_ordinary_probe_exceptions_become_resource_gate_errors(self):
        def fail(*_):
            raise OSError("injected probe failure")

        probes = ("sleeper", "meminfo_reader", "monotonic", "stat", "statvfs")
        for probe in probes:
            defaults = dict(
                meminfo_reader=Sequence(
                    (self.meminfo(1, 0, 0), self.meminfo(1, 0, 0))
                ),
                monotonic=Sequence((0.0, 1.0)),
                sleeper=lambda _: None,
                statvfs=lambda root: SimpleNamespace(f_bavail=1, f_frsize=1),
                stat=lambda root: SimpleNamespace(st_dev=1),
            )
            defaults[probe] = fail
            with self.subTest(probe=probe), self.assertRaises(ResourceGateError):
                enforce_resources(
                    self.limits(
                        min_mem_available_bytes=0,
                        max_swap_used_bytes=0,
                        max_swap_growth_bytes=0,
                        min_snapshot_free_bytes=0,
                        min_result_free_bytes=0,
                    ),
                    "/snapshot",
                    "/result",
                    **defaults,
                )

    def test_busy_error_requires_and_carries_denied_decision(self):
        denied = evaluate_resources(
            self.sample(second_mem_available_bytes=0), self.limits()
        )
        error = ResourceBusyError(denied)
        self.assertIs(error.decision, denied)
        with self.assertRaises(ResourceGateError):
            ResourceBusyError(evaluate_resources(self.sample(), self.limits()))


if __name__ == "__main__":
    unittest.main()
