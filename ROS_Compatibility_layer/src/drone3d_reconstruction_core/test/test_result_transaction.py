"""Focused filesystem-safety tests for result directory transactions."""

import os
import shutil
import stat
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from drone3d_reconstruction_core.result_transaction import (
    ResultCollisionError,
    ResultCommitError,
    ResultNameError,
    ResultRootError,
    ResultSafetyError,
    ResultStateError,
    ResultTransaction,
)


class ResultTransactionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.mkdtemp(prefix="result-transaction-test-")
        self.root = Path(self.temporary).resolve() / "results"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.temporary, ignore_errors=True)

    def test_allocates_unique_hidden_staging_under_canonical_root(self):
        first = ResultTransaction(self.root, "job-1")
        second = ResultTransaction(self.root, "job-2")
        self.assertEqual(first.canonical_root, self.root)
        self.assertEqual(first.final_path, self.root / "job-1")
        self.assertNotEqual(first.staging_path, second.staging_path)
        self.assertEqual(first.staging_path.parent, self.root)
        self.assertTrue(first.staging_path.name.startswith("."))
        self.assertTrue(first.staging_path.is_dir())
        first.cleanup()
        second.cleanup()

    def test_rejects_unsafe_roots_and_names(self):
        with self.assertRaises(ResultRootError):
            ResultTransaction(Path("relative"), "job")
        with self.assertRaises(ResultRootError):
            ResultTransaction(self.root / "missing", "job")
        with self.assertRaises(ResultRootError):
            ResultTransaction(self.root / ".." / "results", "job")

        linked = Path(self.temporary) / "linked-results"
        linked.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ResultRootError):
            ResultTransaction(linked, "job")

        for name in ("", ".", "..", "a/b", "a\\b", ".hidden", "white space"):
            with self.subTest(name=name), self.assertRaises(ResultNameError):
                ResultTransaction(self.root, name)

    def test_constructor_failure_rolls_back_only_identity_proven_allocation(self):
        original_lstat = Path.lstat
        allocated = []

        def unsafe_staging_lstat(path):
            info = original_lstat(path)
            if path.name.startswith(".drone3d-result-"):
                allocated.append(path)
                values = list(info)
                values[0] = stat.S_IFREG | 0o600
                return os.stat_result(values)
            return info

        with mock.patch.object(Path, "lstat", unsafe_staging_lstat):
            with self.assertRaises(ResultSafetyError):
                ResultTransaction(self.root, "job")

        self.assertEqual(len(allocated), 1)
        self.assertFalse(allocated[0].exists())

    def test_constructor_rollback_never_deletes_substituted_entry(self):
        original_lstat = Path.lstat
        replacement = []

        def substitute_then_fail(path):
            info = original_lstat(path)
            if path.name.startswith(".drone3d-result-") and not replacement:
                parked = self.root / ".parked-constructor-allocation"
                path.rename(parked)
                path.mkdir()
                (path / "keep").write_text("replacement", encoding="utf-8")
                replacement.append((path, parked))
                values = list(info)
                values[0] = stat.S_IFREG | 0o600
                return os.stat_result(values)
            return info

        with mock.patch.object(Path, "lstat", substitute_then_fail):
            with self.assertRaises(ResultSafetyError):
                ResultTransaction(self.root, "job")

        path, parked = replacement[0]
        self.assertEqual((path / "keep").read_text(encoding="utf-8"), "replacement")
        self.assertTrue(parked.is_dir())

    def test_rejects_preexisting_final_at_allocation(self):
        final = self.root / "job"
        final.mkdir()
        with self.assertRaises(ResultCollisionError) as caught:
            ResultTransaction(self.root, "job")
        self.assertEqual(caught.exception.code, "FINAL_EXISTS")
        self.assertTrue(final.is_dir())

    def test_commit_publishes_once_and_preserves_contents(self):
        transaction = ResultTransaction(self.root, "job")
        payload = transaction.staging_path / "mesh.txt"
        payload.write_text("mesh", encoding="utf-8")
        staging_inode = transaction.staging_path.stat().st_ino

        published = transaction.commit()

        self.assertEqual(published, self.root / "job")
        self.assertFalse(transaction.staging_path.exists())
        self.assertEqual((published / "mesh.txt").read_text(encoding="utf-8"), "mesh")
        self.assertEqual(published.stat().st_ino, staging_inode)
        self.assertTrue(transaction.committed)
        self.assertIsNone(transaction._root_fd)
        with self.assertRaises(ResultStateError):
            transaction.commit()
        transaction.cleanup()
        self.assertTrue(published.exists())

    def test_commit_fsync_failure_keeps_committed_truth(self):
        transaction = ResultTransaction(self.root, "job")
        payload = transaction.staging_path / "mesh.txt"
        payload.write_text("mesh", encoding="utf-8")

        with mock.patch(
            "drone3d_reconstruction_core.result_transaction.os.fsync",
            side_effect=OSError("injected fsync failure"),
        ):
            with self.assertRaises(ResultCommitError) as caught:
                transaction.commit()
        self.assertEqual(caught.exception.code, "PUBLISHED_FSYNC_FAILED")
        self.assertTrue(transaction.committed)
        self.assertTrue((self.root / "job" / "mesh.txt").is_file())
        self.assertIsNone(transaction._root_fd)

    def test_atomic_collision_preserves_both_final_and_staging(self):
        transaction = ResultTransaction(self.root, "job")
        (transaction.staging_path / "mine").write_text("staging", encoding="utf-8")
        final = self.root / "job"
        final.mkdir()
        (final / "owner").write_text("other", encoding="utf-8")

        with self.assertRaises(ResultCollisionError):
            transaction.commit()

        self.assertEqual((final / "owner").read_text(encoding="utf-8"), "other")
        self.assertEqual(
            (transaction.staging_path / "mine").read_text(encoding="utf-8"),
            "staging",
        )
        with self.assertRaises(ResultStateError):
            transaction.commit()
        transaction.cleanup()
        self.assertTrue(final.exists())
        self.assertFalse(transaction.staging_path.exists())

    def test_cleanup_is_recursive_idempotent_and_never_removes_final(self):
        transaction = ResultTransaction(self.root, "job")
        nested = transaction.staging_path / "nested"
        nested.mkdir()
        (nested / "value").write_text("data", encoding="utf-8")
        transaction.cleanup()
        transaction.cleanup()
        self.assertIsNone(transaction._root_fd)
        self.assertFalse(transaction.staging_path.exists())
        self.assertFalse(transaction.final_path.exists())
        with self.assertRaises(ResultStateError):
            transaction.commit()

    def test_cleanup_unlinks_child_symlink_without_following_it(self):
        outside = Path(self.temporary) / "outside-sentinel"
        outside.write_text("sentinel", encoding="utf-8")
        transaction = ResultTransaction(self.root, "job")
        (transaction.staging_path / "link").symlink_to(outside)

        transaction.cleanup()

        self.assertEqual(outside.read_text(encoding="utf-8"), "sentinel")

    def test_cleanup_of_child_hardlink_preserves_outside_sentinel(self):
        outside = Path(self.temporary) / "outside-sentinel"
        outside.write_text("sentinel", encoding="utf-8")
        transaction = ResultTransaction(self.root, "job")
        os.link(str(outside), str(transaction.staging_path / "hardlink"))

        transaction.cleanup()

        self.assertEqual(outside.read_text(encoding="utf-8"), "sentinel")
        self.assertEqual(outside.stat().st_nlink, 1)

    def test_cleanup_refuses_mount_boundary_and_retains_fd_for_retry(self):
        transaction = ResultTransaction(self.root, "job")
        nested = transaction.staging_path / "mounted-child"
        nested.mkdir()
        (nested / "keep").write_text("data", encoding="utf-8")
        root_inode = self.root.stat().st_ino
        staging_inode = transaction.staging_path.stat().st_ino

        def injected_mount_identity(directory_fd):
            inode = os.fstat(directory_fd).st_ino
            return 10 if inode in (root_inode, staging_inode) else 20

        with mock.patch(
            "drone3d_reconstruction_core.result_transaction._mount_identity",
            side_effect=injected_mount_identity,
        ):
            with self.assertRaises(ResultSafetyError) as caught:
                transaction.cleanup()

        self.assertEqual(caught.exception.code, "MOUNT_BOUNDARY")
        self.assertEqual((nested / "keep").read_text(encoding="utf-8"), "data")
        self.assertIsNotNone(transaction._root_fd)

        with mock.patch(
            "drone3d_reconstruction_core.result_transaction._mount_identity",
            return_value=10,
        ):
            transaction.cleanup()
        self.assertIsNone(transaction._root_fd)
        self.assertFalse(transaction.staging_path.exists())

    def test_cleanup_refuses_staging_identity_substitution(self):
        outside = Path(self.temporary) / "outside-sentinel"
        outside.write_text("sentinel", encoding="utf-8")
        transaction = ResultTransaction(self.root, "job")
        original = self.root / ".parked-original"
        transaction.staging_path.rename(original)
        transaction.staging_path.mkdir()
        (transaction.staging_path / "replacement").write_text("keep", encoding="utf-8")

        with self.assertRaises(ResultSafetyError) as caught:
            transaction.cleanup()

        self.assertEqual(caught.exception.code, "STAGING_IDENTITY_CHANGED")
        self.assertTrue(original.exists())
        self.assertTrue((transaction.staging_path / "replacement").exists())
        self.assertEqual(outside.read_text(encoding="utf-8"), "sentinel")

    def test_cleanup_refuses_staging_symlink_and_hardlink_substitution(self):
        outside = Path(self.temporary) / "outside-sentinel"
        outside.write_text("sentinel", encoding="utf-8")
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                transaction = ResultTransaction(self.root, "job-{}".format(kind))
                parked = self.root / ".parked-{}".format(kind)
                transaction.staging_path.rename(parked)
                if kind == "symlink":
                    transaction.staging_path.symlink_to(outside)
                else:
                    os.link(str(outside), str(transaction.staging_path))
                with self.assertRaises(ResultSafetyError):
                    transaction.cleanup()
                self.assertEqual(outside.read_text(encoding="utf-8"), "sentinel")
                self.assertTrue(parked.exists())

    def test_thread_safe_commit_has_one_winner(self):
        transaction = ResultTransaction(self.root, "job")
        outcomes = []

        def invoke():
            try:
                outcomes.append(("ok", transaction.commit()))
            except ResultStateError as exc:
                outcomes.append(("state", exc.code))

        threads = [threading.Thread(target=invoke) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([kind for kind, _ in outcomes].count("ok"), 1)
        self.assertEqual([kind for kind, _ in outcomes].count("state"), 1)

    def test_fails_closed_when_renameat2_is_unavailable(self):
        transaction = ResultTransaction(self.root, "job")
        with mock.patch(
            "drone3d_reconstruction_core.result_transaction._RENAMEAT2", None
        ):
            with self.assertRaises(ResultCommitError) as caught:
                transaction.commit()
        self.assertEqual(caught.exception.code, "RENAMEAT2_UNAVAILABLE")
        self.assertTrue(transaction.staging_path.exists())
        self.assertFalse(transaction.final_path.exists())
        transaction.cleanup()


if __name__ == "__main__":
    unittest.main()
