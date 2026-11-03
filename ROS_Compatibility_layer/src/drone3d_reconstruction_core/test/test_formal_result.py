"""CPU-only tests for the formal result gate and session/window records."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import drone3d_reconstruction_core.formal_result as formal_result_module
from drone3d_reconstruction_core.result_transaction import ResultCommitError
from drone3d_reconstruction_core.formal_result import (
    FORMAL_RECORD_SCHEMA,
    FORMAL_RECORD_SCHEMA_VERSION,
    FormalResultError,
    publish_formal_record,
    scan_session_windows,
    verify_result_manifest,
)


def snapshot_manifest_bytes() -> bytes:
    """构造一条合法且足够小的 snapshot manifest（测试夹具用）。"""
    return (
        b"path,size,sha256\nsnapshot.json,2,"
        + hashlib.sha256(b"{}").hexdigest().encode("ascii")
        + b"\n"
    )


def snapshot_manifest_sha() -> str:
    return hashlib.sha256(snapshot_manifest_bytes()).hexdigest()


def make_result_manifest(
    *,
    scene_id="cs-s0123456789abcdef-w0",
    manifest_sha256="a" * 64,
    captured=4,
    camera_role="front",
    camera_topic="/front/image_raw",
    camera_frame_id="front_camera_optical_frame",
    checkpoint_sha256="b" * 64,
    checkpoint_path="/checkpoint/model.pt",
    algorithm_revision="algo-0123456789abcdef",
    ros_revision="ros-0123456789abcdef",
    coordinate_state="MODEL_LOCAL_UNALIGNED",
    worker_returncode=0,
    wall_time=1.5,
    artifact_root: Path = None,
):
    artifacts = {}
    for artifact_name in ("predictions.npz", "manifest.txt", "run.log"):
        artifact_content = f"{artifact_name} artifact bytes for tests\n".encode()
        if artifact_root is None:
            artifact_path = Path("/results") / artifact_name
        else:
            artifact_path = Path(artifact_root) / artifact_name
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(artifact_content)
        artifacts[artifact_name] = {
            "uri": artifact_path.as_uri(),
            "size": len(artifact_content),
            "sha256": hashlib.sha256(artifact_content).hexdigest(),
        }
    return {
        "schema": "drone3d.reconstruction.result",
        "schema_version": 1,
        "scene_id": scene_id,
        "snapshot": {
            "sha256": manifest_sha256,
            "frame_count": captured,
        },
        "actual_frame_count": captured,
        "camera_role": camera_role,
        "camera_topic": camera_topic,
        "camera_frame_id": camera_frame_id,
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256,
        },
        "algorithm_revision": algorithm_revision,
        "ros_revision": ros_revision,
        "coordinate_state": coordinate_state,
        "worker_exit": {"returncode": worker_returncode, "cause": "ok"},
        "stage_timing": {
            "wall_time_sec": wall_time,
            "worker_wall_time_sec": wall_time,
        },
        "artifacts": artifacts,
    }


class FormalResultTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session_id = "s0123456789abcdef"
        self.session_dir = self.root / "sessions" / self.session_id
        self.session_dir.mkdir(parents=True)

    def _expected_identity(self, **overrides):
        identity = {
            "scene_id": "cs-s0123456789abcdef-w0",
            "snapshot_manifest_sha256": "a" * 64,
            "captured_frames": 4,
            "camera_role": "front",
            "camera_topic": "/front/image_raw",
            "camera_frame_id": "front_camera_optical_frame",
        }
        identity.update(overrides)
        return identity

    def _expected_authenticity(self):
        return {
            "checkpoint_sha256": "b" * 64,
            "checkpoint_path": "/checkpoint/model.pt",
            "result_root": str(self.root),
            "algorithm_revision": "algo-0123456789abcdef",
            "ros_revision": "ros-0123456789abcdef",
        }

    def test_valid_manifest_passes_and_returns_original_bytes(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload).encode("utf-8")
        path = self.root / "result.json"
        path.write_bytes(data)
        gate = verify_result_manifest(
            path.as_uri(),
            hashlib.sha256(data).hexdigest(),
            expected_identity=self._expected_identity(),
            expected_authenticity=self._expected_authenticity(),
        )
        self.assertTrue(gate.ok)
        self.assertEqual(gate.reason, "")
        self.assertEqual(gate.manifest_bytes, data)
        self.assertEqual(gate.payload["scene_id"], payload["scene_id"])

    def test_sha_mismatch_is_rejected(self):
        payload = make_result_manifest(artifact_root=self.root)
        path = self.root / "result.json"
        path.write_bytes(json.dumps(payload).encode("utf-8"))
        gate = verify_result_manifest(
            path.as_uri(),
            "f" * 64,
            expected_identity=self._expected_identity(),
            expected_authenticity=self._expected_authenticity(),
        )
        self.assertFalse(gate.ok)
        self.assertIn("sha256 mismatch", gate.reason)

    def test_identity_mismatches_are_rejected(self):
        cases = [
            ("scene_id", {"scene_id": "other-window"}),
            ("snapshot_manifest_sha256", {"manifest_sha256": "d" * 64}),
            ("captured_frames", {"captured": 3}),
            ("camera_role", {"camera_role": "down"}),
            ("camera_topic", {"camera_topic": "/down/image_raw"}),
            ("camera_frame_id", {"camera_frame_id": "other_optical"}),
        ]
        for label, overrides in cases:
            with self.subTest(label=label):
                payload = make_result_manifest(artifact_root=self.root, **overrides)
                if label == "snapshot_manifest_sha256":
                    payload["snapshot"]["sha256"] = "d" * 64
                    payload.pop("_", None)
                data = json.dumps(payload).encode("utf-8")
                path = self.root / "result.json"
                path.write_bytes(data)
                gate = verify_result_manifest(
                    path.as_uri(),
                    hashlib.sha256(data).hexdigest(),
                    expected_identity=self._expected_identity(),
                    expected_authenticity=self._expected_authenticity(),
                )
                self.assertFalse(gate.ok, label)
                self.assertIn("does not match", gate.reason)

    def test_authenticity_mismatches_are_rejected(self):
        cases = [
            (
                "checkpoint_sha256",
                {"checkpoint_sha256": "e" * 64},
                "reconstruct profile",
            ),
            (
                "checkpoint_path",
                {"checkpoint_path": "/checkpoint/other.pt"},
                "checkpoint path",
            ),
            ("algorithm_revision", {"algorithm_revision": "other"}, "reconstruct profile"),
            ("ros_revision", {"ros_revision": "other"}, "reconstruct profile"),
        ]
        for label, overrides, reason_fragment in cases:
            with self.subTest(label=label):
                payload = make_result_manifest(artifact_root=self.root, **overrides)
                data = json.dumps(payload).encode("utf-8")
                path = self.root / "result.json"
                path.write_bytes(data)
                gate = verify_result_manifest(
                    path.as_uri(),
                    hashlib.sha256(data).hexdigest(),
                    expected_identity=self._expected_identity(),
                    expected_authenticity=self._expected_authenticity(),
                )
                self.assertFalse(gate.ok)
                self.assertIn(reason_fragment, gate.reason)

    def test_gate_requires_expected_reconstruction_wiring(self):
        payload = make_result_manifest(artifact_root=self.root)
        path = self.root / "result.json"
        path.write_bytes(json.dumps(payload).encode("utf-8"))
        gate = verify_result_manifest(
            path.as_uri(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_identity=self._expected_identity(),
            expected_authenticity={"checkpoint_sha256": "", "algorithm_revision": "", "ros_revision": ""},
        )
        self.assertFalse(gate.ok)
        self.assertIn("expected checkpoint sha256", gate.reason)

    def test_bad_schema_coordinate_and_worker_are_rejected(self):
        cases = [
            ("schema", lambda p: p.update({"schema": "fake.schema"})),
            ("coordinate", lambda p: p.update({"coordinate_state": "GPS_ABSOLUTE"})),
            ("worker returncode", lambda p: p.update({"worker_exit": {"returncode": 3}})),
            ("worker wall", lambda p: p.update({"stage_timing": {"worker_wall_time_sec": 0.0}})),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                payload = make_result_manifest(artifact_root=self.root)
                mutate(payload)
                data = json.dumps(payload).encode("utf-8")
                path = self.root / "result.json"
                path.write_bytes(data)
                gate = verify_result_manifest(
                    path.as_uri(),
                    hashlib.sha256(data).hexdigest(),
                    expected_identity=self._expected_identity(),
                    expected_authenticity=self._expected_authenticity(),
                )
                self.assertFalse(gate.ok)

    def test_non_file_uri_and_symlink_are_rejected(self):
        payload = make_result_manifest(artifact_root=self.root)
        path = self.root / "result.json"
        path.write_bytes(json.dumps(payload).encode("utf-8"))
        gate = verify_result_manifest(
            "http://example/result.json",
            "f" * 64,
            expected_identity=self._expected_identity(),
            expected_authenticity=self._expected_authenticity(),
        )
        self.assertFalse(gate.ok)
        link = self.root / "link.json"
        link.symlink_to(path)
        gate = verify_result_manifest(
            link.as_uri(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_identity=self._expected_identity(),
            expected_authenticity=self._expected_authenticity(),
        )
        self.assertFalse(gate.ok)
        self.assertIn("trusted regular file", gate.reason)

    def test_manifest_outside_expected_worker_root_is_rejected(self):
        payload = make_result_manifest(artifact_root=self.root)
        path = self.root / "result.json"
        path.write_bytes(json.dumps(payload).encode("utf-8"))
        other_root = self.root / "other-worker-root"
        other_root.mkdir()
        expected = self._expected_authenticity()
        expected["result_root"] = str(other_root)
        gate = verify_result_manifest(
            path.as_uri(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_identity=self._expected_identity(),
            expected_authenticity=expected,
        )
        self.assertFalse(gate.ok)
        self.assertIn("outside the configured worker result root", gate.reason)

    def test_duplicate_json_fields_are_rejected(self):
        path = self.root / "result.json"
        path.write_text(
            '{"schema": "drone3d.reconstruction.result", '
            '"schema": "drone3d.reconstruction.result"}',
            encoding="utf-8",
        )
        gate = verify_result_manifest(
            path.as_uri(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_identity=self._expected_identity(),
            expected_authenticity=self._expected_authenticity(),
        )
        self.assertFalse(gate.ok)
        self.assertIn("strict JSON", gate.reason)

    def test_publish_formal_record_is_atomic_and_no_overwrite(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": snapshot_manifest_sha(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
            "alignment": {
                "coordinate_state": "MODEL_LOCAL_UNALIGNED",
                "unreliable": True,
                "reason": "odom_local disabled by profile",
            },
            "depth_metrics": {
                "frame_index": 0,
                "coverage": 0.9,
                "abs_rel": 0.01,
                "rmse": 0.02,
            },
        }
        final, prov_sha = publish_formal_record(
            self.root / "sessions",
            self.session_id,
            0,
            data,
            provenance,
            snapshot_manifest_bytes(),
        )
        self.assertEqual(final.name, "window-0000")
        self.assertEqual(
            (final / "result.json").read_bytes(),
            data,
        )
        alignment = json.loads(
            (final / "odom_local_alignment.json").read_text(encoding="utf-8")
        )
        self.assertTrue(alignment["unreliable"])
        self.assertTrue((final / "depth_metrics.json").is_file())
        prov_path = final / "provenance.json"
        self.assertEqual(prov_sha, hashlib.sha256(prov_path.read_bytes()).hexdigest())
        with self.assertRaises(Exception):
            publish_formal_record(
                self.root / "sessions",
                self.session_id,
                0,
                data,
                provenance,
                snapshot_manifest_bytes(),
            )

    def test_publish_formal_record_fsync_failure_is_published_with_warning(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        worker_result = self.root / "result.json"
        worker_result.write_bytes(data)
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_uri": worker_result.as_uri(),
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": snapshot_manifest_sha(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        original_commit = formal_result_module.ResultTransaction.commit

        def commit_then_fsync_fail(transaction):
            original_commit(transaction)
            raise ResultCommitError(
                "PUBLISHED_FSYNC_FAILED",
                "published result directory could not be fsynced",
            )

        with mock.patch.object(
            formal_result_module.ResultTransaction, "commit", commit_then_fsync_fail
        ):
            final, prov_sha = publish_formal_record(
                self.root / "sessions",
                self.session_id,
                0,
                data,
                provenance,
                snapshot_manifest_bytes(),
            )
        self.assertTrue(final.is_dir())
        self.assertEqual(
            hashlib.sha256((final / "provenance.json").read_bytes()).hexdigest(),
            prov_sha,
        )
        warning = json.loads(
            (final / "publication_warning.json").read_text(encoding="utf-8")
        )
        self.assertEqual(warning["code"], "PUBLISHED_FSYNC_FAILED")
        self.assertFalse(warning["fatal"])
        # 不能重试同一最终名：目录已发布，重试会命中 FINAL_EXISTS。
        with mock.patch.object(
            formal_result_module.ResultTransaction, "commit", commit_then_fsync_fail
        ):
            with self.assertRaises(Exception):
                publish_formal_record(
                    self.root / "sessions",
                    self.session_id,
                    0,
                    data,
                    provenance,
                    snapshot_manifest_bytes(),
                )
        records = scan_session_windows(self.root / "sessions", self.session_id)
        self.assertEqual(len(records), 1)

    def test_odom_sim3_sidecar_rejects_scaled_rotation(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": snapshot_manifest_sha(),
            "alignment": {
                "coordinate_state": "ODOM_LOCAL_APPROX",
                "method": "odom_pose_constrained_sim3",
                "scale": 1.2,
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "translation": [0.1, 0.2, 0.3],
            },
        }
        final, _ = publish_formal_record(
            self.root / "sessions",
            self.session_id,
            0,
            data,
            provenance,
            snapshot_manifest_bytes(),
        )
        self.assertTrue((final / "odom_local_alignment.json").is_file())

        bad = dict(provenance)
        bad["window_index"] = 1
        bad["alignment"] = dict(provenance["alignment"])
        bad["alignment"]["rotation"] = [
            [1.2, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.0, 0.0, 1.2],
        ]
        with self.assertRaisesRegex(FormalResultError, "valid Sim"):
            publish_formal_record(
                self.root / "sessions",
                self.session_id,
                1,
                data,
                bad,
                snapshot_manifest_bytes(),
            )

    def test_scan_session_windows_only_returns_formal_records(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload).encode("utf-8")
        worker_result = self.root / "result.json"
        worker_result.write_bytes(data)
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_uri": worker_result.as_uri(),
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": snapshot_manifest_sha(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        publish_formal_record(
            self.root / "sessions",
            self.session_id,
            0,
            data,
            provenance,
            snapshot_manifest_bytes(),
        )
        publish_formal_record(
            self.root / "sessions",
            self.session_id,
            1,
            data,
            dict(provenance, window_index=1),
            snapshot_manifest_bytes(),
        )
        foreign = self.session_dir / "foreign-dir"
        foreign.mkdir()
        (foreign / "provenance.json").write_text("{}", encoding="utf-8")
        bad_window = self.session_dir / "window-0002"
        bad_window.mkdir()
        (bad_window / "provenance.json").write_text(
            json.dumps({"schema": "fake"}), encoding="utf-8"
        )
        bad_index = self.session_dir / "window-0003"
        bad_index.mkdir()
        (bad_index / "result.json").write_bytes(data)
        (bad_index / "provenance.json").write_text(
            json.dumps(
                {
                    "schema": FORMAL_RECORD_SCHEMA,
                    "window_index": "bad",
                }
            ),
            encoding="utf-8",
        )
        tampered = self.session_dir / "window-0004"
        tampered.mkdir()
        (tampered / "result.json").write_bytes(b"{}")
        (tampered / "provenance.json").write_text(
            json.dumps(
                {
                    "schema": FORMAL_RECORD_SCHEMA,
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "window_index": 4,
                    "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
                    "gate": {
                        "result_manifest_digest_verified": True,
                        "identity_verified": True,
                        "authenticity_verified": True,
                        "schema_verified": True,
                    },
                    "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
                }
            ),
            encoding="utf-8",
        )
        records = scan_session_windows(self.root / "sessions", self.session_id)
        self.assertEqual(
            [item["window_index"] for item in records], [0, 1]
        )
        self.assertEqual(scan_session_windows(self.root / "sessions", "s" + "0" * 16), [])
        self.assertEqual(scan_session_windows(self.root / "sessions", "../bad"), [])

    def test_window_10000_publishes_and_scans(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        worker_result = self.root / "result.json"
        worker_result.write_bytes(data)
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 10000,
            "result_manifest_uri": worker_result.as_uri(),
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": snapshot_manifest_sha(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        final, _provenance_sha = publish_formal_record(
            self.root / "sessions",
            self.session_id,
            10000,
            data,
            provenance,
            snapshot_manifest_bytes(),
        )
        self.assertEqual(final.name, "window-10000")
        records = scan_session_windows(self.root / "sessions", self.session_id)
        self.assertIn(10000, [record["window_index"] for record in records])

    def test_nonexistent_artifact_is_rejected(self):
        payload = make_result_manifest(artifact_root=self.root)
        payload["artifacts"]["predictions.npz"]["uri"] = (
            self.root / "missing-predictions.npz"
        ).as_uri()
        data = json.dumps(payload).encode("utf-8")
        path = self.root / "result.json"
        path.write_bytes(data)
        gate = verify_result_manifest(
            path.as_uri(),
            hashlib.sha256(data).hexdigest(),
            expected_identity=self._expected_identity(),
            expected_authenticity=self._expected_authenticity(),
        )
        self.assertFalse(gate.ok)
        self.assertIn("cannot be inspected", gate.reason)

    def test_all_required_artifacts_must_be_declared(self):
        for missing in ("predictions.npz", "manifest.txt", "run.log"):
            with self.subTest(missing=missing):
                payload = make_result_manifest(artifact_root=self.root)
                payload["artifacts"].pop(missing)
                data = json.dumps(payload).encode("utf-8")
                path = self.root / "result.json"
                path.write_bytes(data)
                gate = verify_result_manifest(
                    path.as_uri(),
                    hashlib.sha256(data).hexdigest(),
                    expected_identity=self._expected_identity(),
                    expected_authenticity=self._expected_authenticity(),
                )
                self.assertFalse(gate.ok)
                self.assertIn("missing required artifacts", gate.reason)

    def test_nonlocal_and_relative_artifact_uris_are_rejected(self):
        for uri in ("https://example.invalid/predictions.npz", "predictions.npz"):
            with self.subTest(uri=uri):
                payload = make_result_manifest(artifact_root=self.root)
                payload["artifacts"]["predictions.npz"]["uri"] = uri
                data = json.dumps(payload).encode("utf-8")
                path = self.root / "result.json"
                path.write_bytes(data)
                gate = verify_result_manifest(
                    path.as_uri(),
                    hashlib.sha256(data).hexdigest(),
                    expected_identity=self._expected_identity(),
                    expected_authenticity=self._expected_authenticity(),
                )
                self.assertFalse(gate.ok)
                self.assertIn("local absolute path", gate.reason)

    def test_artifact_uri_must_match_its_sibling_name(self):
        payload = make_result_manifest(artifact_root=self.root)
        payload["artifacts"]["predictions.npz"] = dict(
            payload["artifacts"]["run.log"]
        )
        data = json.dumps(payload).encode("utf-8")
        path = self.root / "result.json"
        path.write_bytes(data)
        gate = verify_result_manifest(
            path.as_uri(),
            hashlib.sha256(data).hexdigest(),
            expected_identity=self._expected_identity(),
            expected_authenticity=self._expected_authenticity(),
        )
        self.assertFalse(gate.ok)
        self.assertIn("must sit next to", gate.reason)

    def test_wrong_artifact_size_is_rejected(self):
        payload = make_result_manifest(artifact_root=self.root)
        entry = payload["artifacts"]["predictions.npz"]
        entry["size"] = entry["size"] + 5
        data = json.dumps(payload).encode("utf-8")
        path = self.root / "result.json"
        path.write_bytes(data)
        gate = verify_result_manifest(
            path.as_uri(),
            hashlib.sha256(data).hexdigest(),
            expected_identity=self._expected_identity(),
            expected_authenticity=self._expected_authenticity(),
        )
        self.assertFalse(gate.ok)
        self.assertIn("size mismatch", gate.reason)

    def test_wrong_required_artifact_digests_are_rejected(self):
        for artifact_name in ("predictions.npz", "manifest.txt", "run.log"):
            with self.subTest(artifact_name=artifact_name):
                payload = make_result_manifest(artifact_root=self.root)
                payload["artifacts"][artifact_name]["sha256"] = "d" * 64
                data = json.dumps(payload).encode("utf-8")
                path = self.root / "result.json"
                path.write_bytes(data)
                gate = verify_result_manifest(
                    path.as_uri(),
                    hashlib.sha256(data).hexdigest(),
                    expected_identity=self._expected_identity(),
                    expected_authenticity=self._expected_authenticity(),
                )
                self.assertFalse(gate.ok)
                self.assertIn("sha256 mismatch", gate.reason)

    def test_tampered_provenance_after_publication_is_skipped_by_scanner(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload).encode("utf-8")
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": snapshot_manifest_sha(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        final, _ = publish_formal_record(
            self.root / "sessions",
            self.session_id,
            0,
            data,
            provenance,
            snapshot_manifest_bytes(),
        )
        self.assertTrue((final / "provenance.json.sha256").is_file())
        tampered = dict(provenance)
        tampered["worker"] = dict(tampered["worker"], wall_time_sec=99.0)
        (final / "provenance.json").write_text(
            json.dumps(tampered, sort_keys=True), encoding="utf-8"
        )
        self.assertEqual(
            scan_session_windows(self.root / "sessions", self.session_id), []
        )

    def test_snapshot_manifest_csv_preserved_and_verified(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload).encode("utf-8")
        worker_result = self.root / "result.json"
        worker_result.write_bytes(data)
        snapshot_manifest_bytes = (
            b"path,size,sha256\nsnapshot.json,2,"
            + hashlib.sha256(b"{}").hexdigest().encode("ascii")
            + b"\n"
        )
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_uri": worker_result.as_uri(),
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": hashlib.sha256(
                snapshot_manifest_bytes
            ).hexdigest(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        final, _ = publish_formal_record(
            self.root / "sessions",
            self.session_id,
            0,
            data,
            provenance,
            snapshot_manifest_bytes,
        )
        self.assertEqual(
            (final / "snapshot_manifest.csv").read_bytes(),
            snapshot_manifest_bytes,
        )
        records = scan_session_windows(self.root / "sessions", self.session_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["provenance_integrity"], "verified")
        self.assertEqual(
            records[0]["snapshot_manifest_sha256"],
            hashlib.sha256(snapshot_manifest_bytes).hexdigest(),
        )
        (final / "snapshot_manifest.csv").write_bytes(b"tampered")
        self.assertEqual(
            scan_session_windows(self.root / "sessions", self.session_id), []
        )

    def test_current_v2_without_snapshot_manifest_is_skipped(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload).encode("utf-8")
        worker_result = self.root / "result.json"
        worker_result.write_bytes(data)
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_uri": worker_result.as_uri(),
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": snapshot_manifest_sha(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        final, _ = publish_formal_record(
            self.root / "sessions",
            self.session_id,
            0,
            data,
            provenance,
            snapshot_manifest_bytes(),
        )
        (final / "snapshot_manifest.csv").unlink()
        self.assertEqual(
            scan_session_windows(self.root / "sessions", self.session_id), []
        )

    def test_rejected_digest_is_rechecked_after_restore_with_same_metadata(self):
        path = self.root / "restored.npz"
        valid = b"original"
        path.write_bytes(b"modified")
        info = path.stat()
        expected = hashlib.sha256(valid).hexdigest()
        arguments = (
            str(path), *formal_result_module._stat_fingerprint(info), expected
        )
        cached_hash = formal_result_module._sha256_file_for_metadata
        cached_hash.cache_clear()
        with mock.patch.object(formal_result_module.os, "lstat", return_value=info):
            with self.assertRaisesRegex(FormalResultError, "sha256 mismatch"):
                cached_hash(*arguments)
            path.write_bytes(valid)
            self.assertEqual(cached_hash(*arguments), expected)
            self.assertEqual(cached_hash.cache_info().misses, 2)
            self.assertEqual(cached_hash(*arguments), expected)
            self.assertEqual(cached_hash.cache_info().hits, 1)
        cached_hash.cache_clear()

    def test_current_scanner_revalidates_worker_files_without_rehashing(self):
        worker_dir = self.root / "worker"
        payload = make_result_manifest(artifact_root=worker_dir)
        data = json.dumps(payload).encode("utf-8")
        worker_result = worker_dir / "result.json"
        worker_result.write_bytes(data)
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_uri": worker_result.as_uri(),
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": snapshot_manifest_sha(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        formal_result_module._sha256_file_for_metadata.cache_clear()
        original_hash = formal_result_module._sha256_file_bounded
        with mock.patch.object(
            formal_result_module,
            "_sha256_file_bounded",
            wraps=original_hash,
        ) as hash_file:
            gate = verify_result_manifest(
                worker_result.as_uri(),
                provenance["result_manifest_sha256"],
                expected_identity=self._expected_identity(),
                expected_authenticity=self._expected_authenticity(),
            )
            self.assertTrue(gate.ok, gate.reason)
            self.assertEqual(hash_file.call_count, 3)
            publish_formal_record(
                self.root / "sessions",
                self.session_id,
                0,
                data,
                provenance,
                snapshot_manifest_bytes(),
            )
            self.assertEqual(
                len(scan_session_windows(self.root / "sessions", self.session_id)),
                1,
            )
            self.assertEqual(hash_file.call_count, 4)
            self.assertEqual(
                len(scan_session_windows(self.root / "sessions", self.session_id)),
                1,
            )
            self.assertEqual(hash_file.call_count, 4)

            predictions = worker_dir / "predictions.npz"
            original_predictions = predictions.read_bytes()
            predictions.write_bytes(b"X" * len(original_predictions))
            self.assertEqual(
                scan_session_windows(self.root / "sessions", self.session_id), []
            )
            self.assertEqual(hash_file.call_count, 5)

            predictions.write_bytes(original_predictions)
            self.assertEqual(
                len(scan_session_windows(self.root / "sessions", self.session_id)),
                1,
            )
            (worker_dir / "run.log").unlink()
            self.assertEqual(
                scan_session_windows(self.root / "sessions", self.session_id), []
            )

    def test_current_scanner_binds_original_worker_result(self):
        worker_dir = self.root / "worker"
        payload = make_result_manifest(artifact_root=worker_dir)
        data = json.dumps(payload).encode("utf-8")
        worker_result = worker_dir / "result.json"
        worker_result.write_bytes(data)
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_uri": worker_result.as_uri(),
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": snapshot_manifest_sha(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        publish_formal_record(
            self.root / "sessions",
            self.session_id,
            0,
            data,
            provenance,
            snapshot_manifest_bytes(),
        )
        self.assertEqual(
            len(scan_session_windows(self.root / "sessions", self.session_id)), 1
        )
        worker_result.write_bytes(b"X" * len(data))
        self.assertEqual(
            scan_session_windows(self.root / "sessions", self.session_id), []
        )

    def test_current_provenance_without_sidecar_is_skipped(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload).encode("utf-8")
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": snapshot_manifest_sha(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        final, _ = publish_formal_record(
            self.root / "sessions",
            self.session_id,
            0,
            data,
            provenance,
            snapshot_manifest_bytes(),
        )
        (final / "provenance.json.sha256").unlink()
        self.assertEqual(
            scan_session_windows(self.root / "sessions", self.session_id), []
        )

    def test_v1_snapshot_record_without_sidecar_is_not_downgraded(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload).encode("utf-8")
        snapshot_manifest_bytes = b"legacy transition snapshot manifest\n"
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": 1,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "snapshot_manifest_sha256": hashlib.sha256(
                snapshot_manifest_bytes
            ).hexdigest(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        final = self.session_dir / "window-0000"
        final.mkdir()
        (final / "result.json").write_bytes(data)
        (final / "snapshot_manifest.csv").write_bytes(snapshot_manifest_bytes)
        (final / "provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        self.assertEqual(
            scan_session_windows(self.root / "sessions", self.session_id), []
        )

    def test_legacy_provenance_without_sidecar_still_scanned(self):
        payload = make_result_manifest(artifact_root=self.root)
        data = json.dumps(payload).encode("utf-8")
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": 1,
            "session_id": self.session_id,
            "window_index": 0,
            "result_manifest_sha256": hashlib.sha256(data).hexdigest(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        final = self.session_dir / "window-0000"
        final.mkdir()
        (final / "result.json").write_bytes(data)
        (final / "provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        records = scan_session_windows(self.root / "sessions", self.session_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["window_index"], 0)
        self.assertEqual(records[0]["provenance_integrity"], "legacy")

        alias = self.session_dir / "window-00000"
        alias.mkdir()
        (alias / "result.json").write_bytes(data)
        (alias / "provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        records = scan_session_windows(self.root / "sessions", self.session_id)
        self.assertEqual(len(records), 1)


if __name__ == "__main__":
    unittest.main()
