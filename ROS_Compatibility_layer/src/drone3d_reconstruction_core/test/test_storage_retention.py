"""CPU-only checks for bounded continuous-reconstruction storage."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from drone3d_reconstruction_core.storage_retention import storage_usage


class StorageRetentionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.formal = self.root / "results"
        self.worker = self.root / "worker_results"
        self.journal = self.root / "journal"
        self.snapshots = self.root / "snapshots"
        for path in (
            self.formal,
            self.worker,
            self.journal,
            self.snapshots,
        ):
            path.mkdir()

    def test_storage_usage_reports_allocated_bytes_and_paths(self):
        (self.worker / "payload").write_bytes(b"x")
        (self.snapshots / "payload").write_bytes(b"y")
        (self.root / "service.log").write_bytes(b"z")
        ros_logs = self.root / "ros_logs"
        ros_logs.mkdir()
        (ros_logs / "rosout.log").write_bytes(b"r")
        usage = storage_usage(
            result_root=str(self.formal),
            worker_result_root=str(self.worker),
            journal_root=str(self.journal),
            snapshot_roots=(None, str(self.snapshots)),
            runtime_log_root=str(self.root),
            ros_log_root=str(ros_logs),
        )
        self.assertEqual(usage["storage_capture_roots"], str(self.snapshots))
        self.assertEqual(usage["storage_worker_root"], str(self.worker))
        self.assertEqual(usage["storage_scan_error"], "")
        self.assertGreater(usage["storage_capture_bytes"], 0)
        self.assertGreater(usage["storage_worker_bytes"], 0)
        self.assertGreater(usage["storage_runtime_log_bytes"], 0)
        self.assertGreater(usage["storage_ros_log_bytes"], 0)
        self.assertEqual(
            usage["storage_total_bytes"],
            sum(
                usage[key]
                for key in (
                    "storage_capture_bytes",
                    "storage_worker_bytes",
                    "storage_formal_bytes",
                    "storage_frame_pool_bytes",
                    "storage_journal_bytes",
                    "storage_runtime_log_bytes",
                    "storage_ros_log_bytes",
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
