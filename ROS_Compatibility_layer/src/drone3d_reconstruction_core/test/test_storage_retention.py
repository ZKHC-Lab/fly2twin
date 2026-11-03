"""CPU-only checks for bounded continuous-reconstruction storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from drone3d_reconstruction_core.storage_retention import (
    StoragePolicy,
    prune_frame_pool_revisions,
    prune_storage,
    storage_usage,
)


class StorageRetentionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.formal = self.root / "results"
        self.worker = self.root / "worker_results"
        self.journal = self.root / "journal"
        self.snapshots = self.root / "snapshots"
        self.pool = self.root / "frame_pool"
        for path in (
            self.formal,
            self.worker,
            self.journal,
            self.snapshots,
            self.pool,
        ):
            path.mkdir()

    def _session(self, char: str, timestamp: float) -> str:
        session_id = "s" + char * 16
        worker = self.worker / f"cs-{session_id}-w0-job"
        worker.mkdir()
        (worker / "result.json").write_text("{}", encoding="utf-8")
        window = self.formal / "sessions" / session_id / "window-0000"
        window.mkdir(parents=True)
        (window / "provenance.json").write_text(
            json.dumps({"result_manifest_uri": (worker / "result.json").as_uri()}),
            encoding="utf-8",
        )
        journal = self.journal / f"{session_id}.jsonl"
        journal.write_text("{}\n", encoding="utf-8")
        pool = self.pool / session_id
        pool.mkdir()
        snapshot = self.snapshots / f"cs-{session_id}-w0-{'f' * 16}"
        snapshot.mkdir()
        for path in (window.parent, worker, journal, pool, snapshot):
            os.utime(path, (timestamp, timestamp))
        return session_id

    def test_prune_keeps_protected_and_newest_completed_sessions(self):
        oldest = self._session("a", 100.0)
        removed = self._session("b", 200.0)
        kept = [self._session("c", 300.0), self._session("d", 400.0)]
        orphan = self.worker / f"cs-s{'e' * 16}-w9-orphan"
        orphan.mkdir()
        (orphan / "predictions.npz").write_bytes(b"x" * 8192)
        os.utime(orphan, (100.0, 100.0))
        staging = self.worker / ".drone3d-result-deadbeef"
        staging.mkdir()
        os.utime(staging, (100.0, 100.0))

        report = prune_storage(
            result_root=str(self.formal),
            worker_result_root=str(self.worker),
            journal_root=str(self.journal),
            snapshot_roots=(str(self.snapshots),),
            policy=StoragePolicy(
                max_sessions=2,
                frame_pool_keep_revisions=2,
                orphan_grace_sec=10.0,
            ),
            protected_session_ids=(oldest,),
            now=1000.0,
        )

        self.assertEqual(report["errors"], [])
        self.assertEqual(set(report["kept_sessions"]), {oldest, *kept})
        self.assertTrue((self.formal / "sessions" / oldest).is_dir())
        self.assertFalse((self.formal / "sessions" / removed).exists())
        self.assertFalse(any(removed in path.name for path in self.worker.iterdir()))
        self.assertFalse(any(removed in path.name for path in self.snapshots.iterdir()))
        self.assertFalse((self.journal / f"{removed}.jsonl").exists())
        self.assertFalse((self.pool / removed).exists())
        self.assertFalse(orphan.exists())
        self.assertFalse(staging.exists())
        self.assertGreater(report["freed_bytes"], 0)

    def test_frame_pool_prune_keeps_newest_two_revisions(self):
        session_id = "s" + "a" * 16
        session = self.formal / "sessions" / session_id
        for revision in range(4):
            worker = self.worker / f"cs-{session_id}-p{revision}-job"
            worker.mkdir()
            (worker / "result.json").write_text("{}", encoding="utf-8")
            window = session / f"window-{revision:04d}"
            window.mkdir(parents=True)
            (window / "provenance.json").write_text(
                json.dumps(
                    {
                        "reconstruction_mode": "frame_pool",
                        "pool_revision": revision,
                        "result_manifest_uri": (worker / "result.json").as_uri(),
                    }
                ),
                encoding="utf-8",
            )

        report = prune_frame_pool_revisions(
            result_root=str(self.formal),
            worker_result_root=str(self.worker),
            session_id=session_id,
            keep_revisions=2,
        )

        self.assertEqual(report["errors"], [])
        self.assertEqual(
            sorted(path.name for path in session.iterdir()),
            ["window-0002", "window-0003"],
        )
        self.assertEqual(
            sorted(path.name for path in self.worker.iterdir()),
            [f"cs-{session_id}-p2-job", f"cs-{session_id}-p3-job"],
        )

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
