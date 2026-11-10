"""Watchdog tests use tiny real workers and injected memory readings."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from drone3d_reconstruction_core.controller import JobController, WarmWorkerController
from drone3d_reconstruction_core.runtime_memory_guard import GIB, MemoryGuardedController


class RuntimeMemoryGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / 'guard.jsonl'
        self.events = []
        self.done = threading.Event()
        self.low = threading.Event()
        self.swap = threading.Event()

    def make(self, probe=None):
        def memory():
            return {'total_bytes': 30 * GIB,
                    'available_bytes': GIB - 1 if self.low.is_set() else 4 * GIB,
                    'swap_used_bytes': 2 * GIB + 1 if self.swap.is_set() else GIB}
        def finished(event):
            self.events.append(event)
            self.done.set()
        factory = lambda **kw: JobController(worker_timeout_sec=5., term_grace_sec=.1, **kw)
        guard = MemoryGuardedController(factory, on_finished=finished, audit_path=self.log,
                                        probe=probe or memory, interval_sec=.02)
        self.addCleanup(guard.shutdown)
        return guard

    def test_low_memory_terminates_real_worker_and_records_exit(self):
        guard = self.make()
        guard.start([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.low.set()
        self.assertTrue(self.done.wait(4))
        self.assertEqual(self.events[0].cause, 'resource_guard')
        self.assertNotEqual(self.events[0].returncode, 0)
        rows = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertTrue(any(r.get('cancel_accepted') is True for r in rows))
        self.assertEqual(rows[-1]['cause'], 'resource_guard')

    def test_swap_growth_stops_worker_when_available_ram_is_short(self):
        guard = self.make()  # 4 of 30 GiB available: below a quarter
        guard.start([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.swap.set()
        self.assertTrue(self.done.wait(4))
        self.assertEqual(self.events[0].cause, 'resource_guard')
        self.assertIn('job_swap_growth_above_1_gib', self.log.read_text())

    def test_swap_growth_with_plenty_available_is_ignored(self):
        # Heavy buffered writes make the kernel swap idle pages of other
        # processes while most memory is still available.
        guard = self.make(lambda: {'total_bytes': 30 * GIB, 'available_bytes': 16 * GIB,
                                   'swap_used_bytes': 2 * GIB + 1 if self.swap.is_set() else GIB})
        self.swap.set()
        guard.start([sys.executable, '-c', 'import time; time.sleep(.3)'])
        self.assertTrue(self.done.wait(4))
        self.assertEqual(self.events[0].cause, 'worker_exit')

    def test_clean_exit_remains_success_and_old_watchdog_leaves_new_job_alone(self):
        guard = self.make()
        for _ in range(2):
            self.done.clear()
            guard.start([sys.executable, '-c', 'pass'])
            self.assertTrue(self.done.wait(4))
        self.assertEqual([(e.cause, e.returncode) for e in self.events], [('worker_exit',0)] * 2)

    def test_warm_worker_is_terminated_on_low_memory(self):
        command = [sys.executable, '-u', '-c', 'import sys,time; sys.stdin.readline(); time.sleep(30)']
        def finished(event):
            self.events.append(event)
            self.done.set()
        def factory(**kw):
            return WarmWorkerController(worker_timeout_sec=30., term_grace_sec=.1,
                                        warm_command=command, pool_log_path=str(self.log.with_suffix('.pool.log')), **kw)
        guard = MemoryGuardedController(factory, on_finished=finished, audit_path=self.log,
                    probe=lambda: {'total_bytes': 30*GIB, 'available_bytes': GIB-1 if self.low.is_set() else 4*GIB,
                                   'swap_used_bytes': 0}, interval_sec=.02)
        self.addCleanup(guard.shutdown)
        guard.start_warm(command, {'example': True})
        self.low.set()
        self.assertTrue(self.done.wait(4))
        self.assertEqual(self.events[0].cause, 'resource_guard')
        self.assertNotEqual(self.events[0].returncode, 0)
        self.assertIsNotNone(guard._controller._pool._process.poll())

    def test_equal_thresholds_allow_completion(self):
        count = [0]
        def probe():
            count[0] += 1
            return {'total_bytes': 30 * GIB, 'available_bytes': GIB,
                    'swap_used_bytes': 0 if count[0] == 1 else GIB}
        guard = self.make(probe)
        guard.start([sys.executable, '-c', 'import time; time.sleep(.15)'])
        self.assertTrue(self.done.wait(4))
        self.assertEqual(self.events[0].cause, 'worker_exit')
        self.assertEqual(self.events[0].returncode, 0)

    def test_probe_failure_before_dispatch_does_not_start_worker(self):
        marker = Path(self.temp.name) / 'started'
        def broken():
            raise OSError('injected unreadable meminfo')
        guard = self.make(broken)
        with self.assertRaises(OSError):
            guard.start([sys.executable, '-c', 'open(%r,"w").close()' % str(marker)])
        self.assertFalse(marker.exists())
        self.assertFalse(guard.is_busy())


if __name__ == '__main__':
    unittest.main()
