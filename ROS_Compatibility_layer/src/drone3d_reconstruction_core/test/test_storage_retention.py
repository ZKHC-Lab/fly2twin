"""CPU-only checks for bounded continuous-reconstruction storage."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from drone3d_reconstruction_core.storage_retention import (
    StoragePolicy,
    export_marker_path,
    export_verification,
    formal_window_digests,
    prune_storage,
    storage_usage,
    verified_export,
    write_export_marker,
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
        self.deliveries = self.root / "deliveries"
        for path in (
            self.formal,
            self.worker,
            self.journal,
            self.snapshots,
            self.pool,
            self.deliveries,
        ):
            path.mkdir()

    def _session(self, char: str, timestamp: float, exported=True, delivery=None) -> str:
        """All six kinds of session data, dated ``timestamp``; optionally exported."""
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
        g4 = self.formal / "g4" / session_id
        (g4 / "revisions").mkdir(parents=True)
        (g4 / ".writer.lock").write_text("", encoding="utf-8")
        journal = self.journal / f"{session_id}.jsonl"
        journal.write_text("{}\n", encoding="utf-8")
        pool = self.pool / session_id
        pool.mkdir()
        snapshot = self.snapshots / f"cs-{session_id}-w0-{'f' * 16}"
        snapshot.mkdir()
        for path in (window.parent, worker, journal, pool, snapshot):
            os.utime(path, (timestamp, timestamp))
        if exported:
            self._export(session_id, delivery or self.deliveries / session_id)
        return session_id

    def _export(self, session_id, delivery):
        """A delivery package as drone3d_onboard.export leaves it, plus its marker."""
        (delivery / "windows").mkdir(parents=True)
        files = {"cloud.ply": b"cloud " + session_id.encode(), "windows/000000.npz": b"window 0"}
        for relative, data in files.items():
            (delivery / relative).write_bytes(data)
        (delivery / "manifest.json").write_text(json.dumps({
            "session_id": session_id,
            "hashes": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        }))
        write_export_marker(self.formal, session_id, delivery,
                            formal_window_digests(self.formal, session_id))

    def _session_paths(self, session_id):
        return [
            self.formal / "sessions" / session_id,
            self.formal / "g4" / session_id,
            self.pool / session_id,
            self.journal / f"{session_id}.jsonl",
            export_marker_path(self.formal, session_id),
            *[path for path in self.worker.iterdir() if session_id in path.name],
            *[path for path in self.snapshots.iterdir() if session_id in path.name],
        ]

    def _prune(self, max_sessions=1, protected=(), now=1000.0, remove_staging=False):
        return prune_storage(
            result_root=str(self.formal),
            worker_result_root=str(self.worker),
            journal_root=str(self.journal),
            snapshot_roots=(str(self.snapshots),),
            policy=StoragePolicy(max_sessions=max_sessions, orphan_grace_sec=10.0),
            protected_session_ids=protected,
            remove_staging=remove_staging,
            now=now,
        )

    def test_prune_removes_all_six_kinds_of_old_exported_sessions(self):
        removed = self._session("b", 200.0)
        kept = self._session("c", 300.0)
        orphan = self.worker / f"cs-s{'e' * 16}-w9-orphan"
        orphan.mkdir()
        (orphan / "predictions.npz").write_bytes(b"x" * 8192)
        os.utime(orphan, (100.0, 100.0))
        staging = self.worker / ".drone3d-result-deadbeef"
        staging.mkdir()
        os.utime(staging, (100.0, 100.0))

        report = self._prune(remove_staging=True)  # node start: no session is running

        self.assertEqual(report["errors"], [])
        self.assertEqual(report["kept_sessions"], [kept])
        self.assertEqual(report["kept_unexported"], [])
        self.assertEqual([path for path in self._session_paths(removed) if path.exists()], [])
        self.assertEqual(list((self.formal / "sessions").iterdir()), [self.formal / "sessions" / kept])
        self.assertEqual(list((self.formal / "g4").iterdir()), [self.formal / "g4" / kept])
        self.assertTrue(all(path.exists() for path in self._session_paths(kept)))
        self.assertFalse(orphan.exists())
        self.assertFalse(staging.exists())
        self.assertTrue((self.deliveries / removed / "manifest.json").is_file())
        self.assertGreater(report["freed_bytes"], 0)

    def test_sessions_within_the_retention_count_are_kept(self):
        sessions = [self._session(char, 100.0 * (index + 1)) for index, char in enumerate("abc")]
        report = self._prune(max_sessions=3)
        self.assertEqual(report["removed"], [])
        self.assertEqual(report["kept_sessions"], sorted(sessions))
        for session_id in sessions:
            self.assertTrue(all(path.exists() for path in self._session_paths(session_id)))

    def test_unexported_old_session_is_kept_and_reported(self):
        unexported = self._session("a", 100.0, exported=False)
        self._session("c", 300.0)
        report = self._prune()
        self.assertEqual(report["kept_unexported"], [unexported])
        self.assertIn(unexported, report["kept_sessions"])
        self.assertTrue(all(path.exists() for path in self._session_paths(unexported)
                            if path != export_marker_path(self.formal, unexported)))

    def test_window_published_after_the_export_keeps_the_session(self):
        session_id = self._session("a", 100.0)
        self._session("c", 300.0)
        late = self.formal / "sessions" / session_id / "window-0001"
        late.mkdir()
        (late / "provenance.json").write_text("{}", encoding="utf-8")
        os.utime(late.parent, (100.0, 100.0))  # still the oldest session
        report = self._prune()
        self.assertEqual(report["kept_unexported"], [session_id])
        self.assertEqual(report["kept_reasons"][session_id], "formal windows differ from the exported ones")
        self.assertTrue(late.exists())
        self.assertTrue(all(path.exists() for path in self._session_paths(session_id)))

    def test_long_session_marker_stays_small_and_grants_deletion(self):
        session_id = self._session("a", 100.0, exported=False)
        records = self.formal / "sessions" / session_id
        for index in range(1, 1000):  # window-0000 exists already: 1000 windows in all
            window = records / f"window-{index:04d}"
            window.mkdir()
            (window / "provenance.json").write_text(json.dumps({"window_index": index}), encoding="utf-8")
        os.utime(records, (100.0, 100.0))
        self._export(session_id, self.deliveries / session_id)
        self.assertLess(export_marker_path(self.formal, session_id).stat().st_size, 4096)
        self.assertEqual(export_verification(self.formal, session_id), "")
        self._session("c", 300.0)
        report = self._prune()
        self.assertEqual(report["removed"] and report["kept_unexported"], [])
        self.assertFalse(records.exists())

    def test_incomplete_delivery_keeps_the_session(self):
        cases = {
            "a": (lambda d: (d / "cloud.ply").unlink(), "delivery file cloud.ply is missing"),
            "b": (lambda d: (d / "windows/000000.npz").write_bytes(b"other"),
                  "delivery file windows/000000.npz changed"),
        }
        damaged = {}
        for char, (damage, reason) in cases.items():
            session_id = self._session(char, 100.0)
            damage(self.deliveries / session_id)
            damaged[session_id] = reason
        self._session("c", 300.0)
        report = self._prune()
        self.assertEqual(report["kept_reasons"], damaged)
        for session_id in damaged:
            self.assertTrue(all(path.exists() for path in self._session_paths(session_id)))

    def test_manifest_must_parse_and_name_the_session(self):
        session_id = self._session("a", 100.0)
        manifest = self.deliveries / session_id / "manifest.json"
        for content, reason in ((b"{", "export cannot be verified: JSONDecodeError"),
                                (json.dumps({"session_id": "s" + "b" * 16, "hashes": {"x": "0"}}).encode(),
                                 "delivery manifest.json names another session")):
            manifest.write_bytes(content)
            write_export_marker(self.formal, session_id, self.deliveries / session_id,
                                formal_window_digests(self.formal, session_id))
            self.assertEqual(export_verification(self.formal, session_id), reason)

    def test_delivery_inside_the_sessions_own_data_keeps_the_session(self):
        inside = {"a": lambda sid: self.formal / "g4" / sid / "delivery",
                  "b": lambda sid: self.pool / sid / "delivery"}
        expected = {}
        for char, where in inside.items():
            session_id = "s" + char * 16
            self._session(char, 100.0, delivery=where(session_id))
            os.utime(self.pool / session_id, (100.0, 100.0))  # writing the delivery touched it
            expected[session_id] = where(session_id)
        self._session("c", 300.0)
        report = self._prune()
        self.assertEqual(sorted(report["kept_unexported"]), sorted(expected))
        for session_id, delivery in expected.items():
            self.assertIn("lies inside the session's own data", report["kept_reasons"][session_id])
            self.assertTrue((delivery / "manifest.json").is_file())

    def test_delivery_inside_another_sessions_data_survives(self):
        a, b = "s" + "a" * 16, "s" + "b" * 16
        self._session("b", 200.0)  # exported to its own delivery elsewhere
        inside_b = self.formal / "g4" / b / "delivery-for-a"
        self._session("a", 100.0, delivery=inside_b)
        os.utime(self.formal / "g4" / b, (200.0, 200.0))
        self._session("c", 300.0)
        report = self._prune()
        self.assertTrue((inside_b / "manifest.json").is_file())
        self.assertEqual(sorted(report["kept_unexported"]), [a, b])
        self.assertIn("reclaimable storage", report["kept_reasons"][a])
        self.assertIn(f"holds the delivery of session {a}", report["kept_reasons"][b])
        report = self._prune()  # and again on the next run
        self.assertTrue((inside_b / "manifest.json").is_file())

    def test_nested_storage_roots_skip_the_whole_cleanup(self):
        self._session("a", 100.0)
        self._session("c", 300.0)
        nested = self.formal / "sessions" / ("s" + "a" * 16) / "journal"
        nested.mkdir()
        report = prune_storage(
            result_root=str(self.formal), worker_result_root=str(self.worker),
            journal_root=str(nested), snapshot_roots=(str(self.snapshots),),
            policy=StoragePolicy(max_sessions=1, orphan_grace_sec=10.0), now=1000.0)
        self.assertEqual(report["removed"], [])
        self.assertEqual(len(report["errors"]), 1)
        self.assertIn("storage roots are nested", report["errors"][0])
        self.assertTrue(nested.is_dir())

    def test_staging_is_kept_unless_no_session_can_be_running(self):
        current = self._session("a", 900.0)
        staging = [self.worker / ".drone3d-result-live", self.snapshots / ".drone3d-result-capture"]
        for path in staging:
            path.mkdir()
            os.utime(path, (100.0, 100.0))
        report = self._prune(protected=(current,))  # after STOP_ALL, possibly still draining
        self.assertEqual(report["kept_staging"], 2)
        self.assertTrue(all(path.is_dir() for path in staging))
        report = self._prune(remove_staging=True)  # node start
        self.assertEqual(report["kept_staging"], 0)
        self.assertFalse(any(path.exists() for path in staging))

    def test_failed_hide_removes_nothing_of_the_session(self):
        self._session("c", 300.0)
        for blocked in ("sessions", "g4"):
            with self.subTest(blocked=blocked):
                session_id = self._session("a" if blocked == "sessions" else "b", 100.0)
                obstacle = self.formal / blocked / (".deleting-" + session_id)
                obstacle.mkdir()
                report = self._prune()
                self.assertTrue(any("could not be hidden" in e for e in report["errors"]))
                # Both entry points visible again, every dependency still there.
                self.assertTrue(all(path.exists() for path in self._session_paths(session_id)))
                self.assertFalse((self.formal / ("g4" if blocked == "sessions" else "sessions")
                                  / (".deleting-" + session_id)).exists())
                obstacle.rmdir()
                self._prune()  # the obstacle gone, the next run removes the session
                self.assertFalse(any(path.exists() for path in self._session_paths(session_id)))

    def test_failed_hide_also_protects_the_sessions_unreferenced_worker_results(self):
        session_id = self._session("a", 100.0)
        self._session("c", 300.0)
        attempt = self.worker / f"cs-{session_id}-w9-unpublished"  # a failed attempt, no formal record
        attempt.mkdir()
        os.utime(attempt, (0.0, 0.0))
        (self.formal / "sessions" / (".deleting-" + session_id)).mkdir()
        report = self._prune()
        self.assertTrue(any("could not be hidden" in e for e in report["errors"]))
        self.assertIn(session_id, report["kept_sessions"])
        self.assertTrue(attempt.is_dir())

    def test_moved_or_changed_delivery_keeps_the_session(self):
        moved = self._session("a", 100.0)
        changed = self._session("b", 200.0)
        self._session("c", 300.0)
        shutil.move(str(self.deliveries / moved), str(self.root / "elsewhere"))
        (self.deliveries / changed / "manifest.json").write_text("{}")
        self.assertFalse(verified_export(self.formal, moved))
        self.assertFalse(verified_export(self.formal, changed))
        report = self._prune()
        self.assertEqual(report["kept_unexported"], [moved, changed])
        for session_id in (moved, changed):
            self.assertTrue(all(path.exists() for path in self._session_paths(session_id)))

    def test_current_and_locked_sessions_are_never_removed(self):
        current = self._session("a", 100.0)
        locked = self._session("b", 200.0)
        self._session("c", 300.0)
        with open(self.formal / "g4" / locked / ".writer.lock", "a+") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = self._prune(protected=(current,))
        self.assertEqual(report["kept_locked"], [locked])
        for session_id in (current, locked):
            self.assertIn(session_id, report["kept_sessions"])
            self.assertTrue(all(path.exists() for path in self._session_paths(session_id)))

    def test_interrupted_removal_is_finished_by_the_next_run(self):
        session_id = self._session("a", 100.0)
        self._session("c", 300.0)
        # A power cut after the formal records were hidden: the rest is still on disk.
        sessions = self.formal / "sessions"
        os.rename(sessions / session_id, sessions / (".deleting-" + session_id))
        report = self._prune()
        self.assertEqual(report["errors"], [])
        self.assertEqual([path for path in self._session_paths(session_id) if path.exists()], [])
        self.assertFalse((sessions / (".deleting-" + session_id)).exists())

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
