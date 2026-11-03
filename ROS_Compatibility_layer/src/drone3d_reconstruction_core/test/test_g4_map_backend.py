"""Real-core tests for ingestion, immutable revisions and isolated failures."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from g4_backend import Sim3, WindowGeometry, solve_g4
from drone3d_reconstruction_core.g4.input_adapter import WindowInput
from drone3d_reconstruction_core.g4.map_backend import G4MapBackend


SESSION = "s0123456789abcdef"


def fixture(window_id, start, count=30, shared=0, valid_reference=True, stationary=False):
    frame_ids = tuple("rgb-%d" % k for k in range(start, start+count))
    f = np.arange(start, start+count, dtype=float)
    global_centers = np.column_stack((f*.025, np.sin(f*.19)*.1, np.cos(f*.13)*.06))
    if stationary:
        global_centers[:] = [1.5, .4, .2]
    origin = global_centers[0]
    local = global_centers-origin
    timestamps = (np.arange(start, start+count, dtype=np.int64)+100)*1000000000
    depth = np.full((count, 32, 32), 3.)
    K = np.repeat(np.array([[20., 0, 15.5], [0, 20., 15.5], [0, 0, 1.]])[None], count, axis=0)
    geometry = WindowGeometry(window_id, frame_ids, timestamps, depth, np.ones_like(depth),
                              K, np.repeat(np.eye(3)[None], count, axis=0), local,
                              shared, tuple("sha-"+key for key in frame_ids))
    metadata = {
        "window_id": window_id, "frame_ids": list(frame_ids), "timestamps_ns": timestamps.tolist(),
        "shared_prefix": shared, "rgb_hashes": list(geometry.rgb_hashes), "gauge": 1.,
        "prediction_to_local": Sim3().to_dict(), "prediction_uri": "/synthetic/unused.npz",
        "prediction_sha256": "0"*64, "result_manifest_sha256": str(window_id)*64,
        "snapshot_manifest_sha256": "1"*64, "camera_frame_id": "camera_optical",
        "camera_topic": "/camera/rgb", "reference_frame_id": "odom" if valid_reference else "",
        "capture_generation": 0, "localization_reason": "",
        "localization_valid_frames": count if valid_reference else 0,
    }
    loaded = WindowInput(geometry, metadata,
                          global_centers*2 + [4., 3., 2.], np.full(count, valid_reference))
    record = {"window_index": window_id, "result_manifest_sha256": str(window_id)*64,
              "snapshot_manifest_sha256": "1"*64, "capture_generation": 0,
              "model_frame_ids": list(frame_ids), "overlap_frame_count": shared}
    return record, loaded


class Reader:
    def __init__(self, loaded):
        self.loaded = loaded

    def load(self, record):
        return self.loaded[record["window_index"]]


class MapSnapshotTest(unittest.TestCase):
    def setUp(self):
        from drone3d_reconstruction_core.g4.backend_process import output_root
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.records_root = Path(self.temp.name) / "sessions"
        self.root = output_root(self.records_root, SESSION)
        self.records = [{
            "session_id": SESSION, "window_index": i,
            "result_manifest_sha256": hashlib.sha256(("result-%d" % i).encode()).hexdigest(),
            "snapshot_manifest_sha256": hashlib.sha256(("snapshot-%d" % i).encode()).hexdigest(),
            "alignment": {"method": "global_g4", "pending": True,
                          "coordinate_state": "MODEL_LOCAL_UNALIGNED",
                          "quality_passed": False, "geometry_accepted": False,
                          "reason": "waiting_for_G4_map_revision"},
        } for i in range(30)]

    def publish(self, count, number=1, display=None, usable=True):
        directory = self.root / "revisions" / ("revision-%06d" % number)
        directory.mkdir(parents=True)
        windows = []
        for record in self.records[:count]:
            i = record["window_index"]
            source = {key: record[key] for key in
                      ("result_manifest_sha256", "snapshot_manifest_sha256")}
            source.update(frame_ids=list(range(max(0, i * 100 - 20), (i + 1) * 100)),
                          shared_prefix=20 if i else 0, gauge=1.)
            windows.append({"window_id": i, "source": source,
                            "included_in_global_preview": usable,
                            "prediction_to_map": {"scale": 1., "rotation": np.eye(3).tolist(),
                                                  "translation": [float(number), 0., 0.]},
                            "global_frame_status": "localized" if usable else "pending"})
        revision = {"session_id": SESSION, "revision": number, "replacement": "complete_map",
                    "reference_frame_id": "odom", "window_count": count, "windows": windows,
                    "published_windows": list(range(count)) if usable else [],
                    "pending_windows": [] if usable else list(range(count))}
        raw = json.dumps(revision).encode()
        (directory / "revision.json").write_bytes(raw)
        pointer = {"path": str(directory), "sha256": hashlib.sha256(raw).hexdigest()}
        (self.root / "latest.json").write_text(json.dumps({**pointer, "display": display or pointer}))
        return pointer

    def overlay(self, records=None):
        from drone3d_reconstruction_core.g4.map_snapshot import apply_map_snapshot
        return apply_map_snapshot(self.records if records is None else records,
                                  self.records_root, SESSION)

    def test_30_formal_windows_make_22_window_display_stale_without_claiming_new_windows(self):
        pointer = self.publish(22)
        before = json.dumps(self.records, sort_keys=True)
        overlaid = self.overlay()
        self.assertEqual(len(overlaid), 30)
        for record in overlaid[:22]:
            self.assertTrue(record["alignment"]["map_stale"])
            self.assertEqual(record["alignment"]["map_revision"], 1)
            self.assertEqual(record["alignment"]["map_revision_sha256"], pointer["sha256"])
        for original, record in zip(self.records[22:], overlaid[22:]):
            self.assertEqual(record, original)
            self.assertTrue(record["alignment"]["pending"])
            self.assertNotIn("map_revision", record["alignment"])
        self.assertEqual(json.dumps(self.records, sort_keys=True), before)

    def test_same_complete_input_set_is_not_stale(self):
        self.publish(30)
        self.assertTrue(all(not r["alignment"]["map_stale"] for r in self.overlay()))

    def test_rejected_new_solution_retains_display_and_sha_staleness(self):
        previous = self.publish(30)
        rejected = self.publish(30, number=2, display=previous, usable=False)
        self.assertNotEqual(previous["sha256"], rejected["sha256"])
        for record in self.overlay():
            alignment = record["alignment"]
            self.assertTrue(alignment["map_stale"])
            self.assertEqual(alignment["map_revision"], 1)
            self.assertEqual(alignment["map_revision_sha256"], previous["sha256"])
            self.assertEqual(alignment["translation"], [1., 0., 0.])
            self.assertFalse(alignment["pending"])


class G4BackendProcessTest(unittest.TestCase):
    def test_parent_bootstrap_and_unknown_directory_rejection(self):
        from drone3d_reconstruction_core.g4.backend_process import G4BackendProcess, output_root
        from g4_backend import WINDOWED_METHOD_ID

        grouped = {"method_id": WINDOWED_METHOD_ID, "window_frame_semantics": "new_frames",
                   "window_frames": 100, "overlap_frames": 20, "minimum_total_frames": 100,
                   "window_plan_path": "", "threshold_table_path": "", "fit_exclusion_source": ""}
        for profile, unknown in ((None, False), (grouped, False), (grouped, True)):
            with self.subTest(profile=profile, unknown=unknown), tempfile.TemporaryDirectory() as directory:
                records = Path(directory) / "sessions"
                records.mkdir()
                root = output_root(records, SESSION)
                if unknown:
                    root.mkdir(parents=True)
                    (root / "unrelated.txt").write_text("preserve")
                process = G4BackendProcess(records, SESSION, profile=profile)
                try:
                    deadline = time.monotonic() + 20
                    while time.monotonic() < deadline:
                        status = process.status()
                        if status["state"] == "waiting" or process.process.poll() is not None:
                            break
                        time.sleep(0.05)
                    if unknown:
                        self.assertEqual(status["state"], "error")
                        self.assertIn("exited with code", status["error"])
                        self.assertIn("runtime.log", status["error"])
                        self.assertEqual((root / "unrelated.txt").read_text(), "preserve")
                        self.assertFalse((root / "run.json").exists())
                    else:
                        self.assertEqual(status["state"], "waiting", (root / "runtime.log").read_text())
                        self.assertTrue((root / "run.json").is_file())
                        self.assertTrue((root / "status.json").is_file())
                        self.assertIsNone(process.process.poll())
                finally:
                    process.close()


class G4MapBackendTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = []
        self.loaded = {}
        self.services = []

    def tearDown(self):
        for service in self.services:
            service.close()

    def add_window(self, window_id, start, **kwargs):
        record, loaded = fixture(window_id, start, **kwargs)
        self.records.append(record)
        self.loaded[window_id] = loaded
        return record

    def service(self):
        service = G4MapBackend(self.root/"records", SESSION, self.root/"output",
            reader=Reader(self.loaded),
            scanner=lambda root, session: list(self.records))
        self.services.append(service)
        return service

    @staticmethod
    def revision(path):
        return json.loads((path/"revision.json").read_text())

    def test_first_verified_window_after_failed_inference_owns_all_frames(self):
        self.add_window(1, 11, count=16, shared=5)
        path = self.service().update()
        revision = self.revision(path)
        source = revision["windows"][0]["source"]
        self.assertEqual(source["shared_prefix"], 0)
        self.assertEqual(source["declared_shared_prefix"], 5)
        with np.load(path/"trajectory.npz") as z:
            self.assertEqual(len(z["centers"]), 16)

    def test_complete_replacement_keeps_old_revision_and_avoids_shared_camera_duplicates(self):
        self.add_window(0, 0)
        service = self.service()
        first = service.update()
        before = (first/"revision.json").read_bytes()
        self.assertIsNone(service.update())
        self.add_window(1, 20, shared=10)
        second = service.update()
        self.assertNotEqual(first, second)
        self.assertEqual((first/"revision.json").read_bytes(), before)
        revision = self.revision(second)
        self.assertEqual(revision["replacement"], "complete_map")
        self.assertEqual(revision["published_windows"], [0, 1])
        self.assertFalse(revision["formal_geometry_acceptance"])
        self.assertEqual(revision["loop_edges"], 0)
        with np.load(second/"trajectory.npz") as data:
            self.assertEqual(len(data["timestamps_ns"]), 50)
            self.assertEqual(len(np.unique(data["timestamps_ns"])), 50)

    def test_restart_reannounces_committed_revision_without_duplicate_solve(self):
        self.add_window(0, 0)
        first = self.service()
        revision = first.update()
        first.close()
        second = self.service()
        self.assertEqual(second.update(), revision)
        self.assertIsNone(second.update())
        self.assertEqual(len(list((self.root/"output/revisions").glob("revision-*"))), 1)

    def test_pending_graph_retains_one_complete_display_then_recovers(self):
        self.add_window(0, 0)
        service = self.service()
        first = service.update()
        self.add_window(1, 20, shared=10)
        def pending(*args, **kwargs):
            result = solve_g4(*args, **kwargs)
            for component in result.diagnostics["component_diagnostics"]:
                component["publishable_numerical"] = False
            return result
        with patch("drone3d_reconstruction_core.g4.map_backend.solve_g4", side_effect=pending):
            second = service.update()
        pointer = json.loads((self.root/"output/latest.json").read_text())
        self.assertEqual(pointer["path"], str(second))
        self.assertEqual(pointer["display"]["path"], str(first))
        self.assertEqual(self.revision(second)["published_windows"], [])
        service.close()
        restarted = self.service()
        self.assertEqual(restarted.update(), second)
        self.add_window(2, 40, shared=10)
        third = restarted.update()
        pointer = json.loads((self.root/"output/latest.json").read_text())
        self.assertEqual(pointer["display"]["path"], str(third))
        self.assertEqual(self.revision(third)["published_windows"], [0, 1, 2])

    def test_changed_source_binding_preserves_latest(self):
        record = self.add_window(0, 0)
        service = self.service()
        first = service.update()
        pointer = (self.root/"output/latest.json").read_bytes()
        record["snapshot_manifest_sha256"] = "9"*64
        with self.assertRaisesRegex(ValueError, "committed input changed"):
            service.update()
        self.assertEqual((self.root/"output/latest.json").read_bytes(), pointer)
        self.assertEqual(service.status()["latest_revision_path"], str(first))

    def test_nonconvergent_identical_inputs_do_not_repeat_expensive_solve(self):
        self.add_window(0, 0)
        service = self.service()
        service.update()
        pointer = (self.root/"output/latest.json").read_bytes()
        self.add_window(1, 20, shared=10)
        def fail(*args, **kwargs):
            result = solve_g4(*args, **kwargs)
            result.diagnostics["status"] = 0
            return result
        with patch("drone3d_reconstruction_core.g4.map_backend.solve_g4", side_effect=fail) as solve:
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                service.update()
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                service.update()
            self.assertEqual(solve.call_count, 1)
        self.assertEqual((self.root/"output/latest.json").read_bytes(), pointer)
        self.add_window(2, 40, shared=10)
        self.assertEqual(self.revision(service.update())["window_count"], 3)

    def test_duplicate_and_disappeared_window_fail_without_replacement(self):
        record = self.add_window(0, 0)
        service = self.service()
        first = service.update()
        self.records.append(dict(record))
        with self.assertRaisesRegex(ValueError, "duplicate formal"):
            service.update()
        self.records.clear()
        with self.assertRaisesRegex(ValueError, "source window disappeared"):
            service.update()
        self.assertTrue(first.exists())

    def test_mismatched_shared_rgb_fails_and_keeps_previous_revision(self):
        self.add_window(0, 0)
        service = self.service()
        first = service.update()
        self.add_window(1, 20, shared=10)
        old = self.loaded[1].geometry
        # Same identities with altered RGB hashes must never produce an edge.
        self.loaded[1].geometry = WindowGeometry(old.window_id, old.frame_ids, old.timestamps_ns,
            old.depth, old.confidence, old.intrinsics, old.rotations, old.centers,
            old.shared_prefix, tuple("changed-"+s for s in old.rgb_hashes))
        with self.assertRaises(ValueError):
            service.update()
        pointer = json.loads((self.root/"output/latest.json").read_text())
        self.assertEqual(pointer["path"], str(first))

    def test_second_writer_is_rejected(self):
        self.service()
        with self.assertRaisesRegex(RuntimeError, "another G4 writer"):
            self.service()

    def test_corrupt_revision_artifact_is_rejected(self):
        self.add_window(0, 0)
        service = self.service()
        first = service.update()
        (first/"trajectory.npz").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            service.update()

    def test_unanchored_new_component_does_not_erase_valid_main_component(self):
        self.add_window(0, 0)
        service = self.service()
        service.update()
        self.add_window(1, 40, valid_reference=False)
        second = service.update()
        revision = self.revision(second)
        self.assertEqual(revision["published_windows"], [0])
        self.assertEqual(revision["pending_windows"], [1])

    def test_weakly_anchored_component_does_not_erase_valid_main_component(self):
        self.add_window(0, 0)
        service = self.service()
        service.update()
        self.add_window(1, 40, stationary=True)
        second = service.update()
        revision = self.revision(second)
        self.assertEqual(revision["published_windows"], [0])
        self.assertEqual(revision["pending_windows"], [1])

    def test_missing_window_creates_gap_and_late_insert_is_explicit(self):
        self.add_window(0, 0)
        service = self.service()
        service.update()
        self.add_window(2, 80, valid_reference=False)
        second = service.update()
        self.assertEqual(self.revision(second)["pending_windows"], [2])
        pointer = (self.root/"output/latest.json").read_bytes()
        self.add_window(1, 40, valid_reference=False)
        with self.assertRaisesRegex(ValueError, "historical window inserted"):
            service.update()
        self.assertEqual((self.root/"output/latest.json").read_bytes(), pointer)

    def test_total_16_shared_5_preserves_source_ownership(self):
        self.add_window(0, 0, count=16)
        service = self.service()
        service.update()
        self.add_window(1, 11, count=16, shared=5)
        latest = service.update()
        revision = self.revision(latest)
        self.assertEqual(revision["graph"]["edge_count"], 1)
        self.assertEqual(revision["published_windows"], [0, 1])
        with np.load(latest/"trajectory.npz") as data:
            self.assertEqual(len(data["timestamps_ns"]), 27)
            self.assertEqual(len(np.unique(data["timestamps_ns"])), 27)

    def test_failed_middle_window_does_not_delete_unowned_shared_frames(self):
        self.add_window(0, 0, count=30)
        service = self.service()
        service.update()
        self.add_window(2, 80, count=30, shared=5)
        latest = service.update()
        revision = self.revision(latest)
        self.assertEqual(revision["published_windows"], [0, 2])
        self.assertEqual(revision["graph"]["edge_count"], 0)
        source = revision["windows"][1]["source"]
        self.assertEqual(source["declared_shared_prefix"], 5)
        self.assertEqual(source["shared_prefix"], 0)
        with np.load(latest/"trajectory.npz") as data:
            self.assertEqual(len(data["timestamps_ns"]), 60)
            self.assertEqual(len(np.unique(data["timestamps_ns"])), 60)


if __name__ == "__main__":
    unittest.main()

class BackendProfileTests(unittest.TestCase):
    """D-0/D-5：会话给出的 profile 必须能直接构造后端，包括所有默认空值。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "records").mkdir()
        (self.root / "out").mkdir()

    def test_default_profile_with_empty_paths_builds_a_backend(self):
        from drone3d_reconstruction_core.g4.map_backend import build_backend
        profile = {"method_id": "legacy_g4_20260916", "window_plan_path": "",
                   "threshold_table_path": "", "fit_exclusion_source": "",
                   "window_frame_semantics": "total_frames", "minimum_total_frames": 100,
                   "registration": {}}
        backend = build_backend(self.root / "records", SESSION, self.root / "out", profile)
        try:
            self.assertEqual(len(backend.threshold_policy), 0)
            self.assertEqual(backend.exclusions, {})
            self.assertIsNone(backend.plan)
            self.assertEqual(backend.method_id, "legacy_g4_20260916")
        finally:
            backend.close()

    def test_grouped_profile_with_empty_paths_builds_a_backend(self):
        from drone3d_reconstruction_core.g4.map_backend import build_backend
        profile = {"method_id": "windowed_sim3_graph_20261008", "window_plan_path": "",
                   "threshold_table_path": "", "threshold_group": "", "fit_exclusion_source": "",
                   "window_frame_semantics": "new_frames", "minimum_total_frames": 100,
                   "window_frames": 100, "overlap_frames": 20,
                   "registration": {"ransac_degenerate": "abort"}}
        backend = build_backend(self.root / "records", SESSION, self.root / "out", profile)
        try:
            self.assertEqual(backend.reg_config.ransac_degenerate, "abort")
            self.assertEqual(backend.method_id, "windowed_sim3_graph_20261008")
        finally:
            backend.close()

    def test_grouped_profile_with_a_plan_and_a_real_threshold_table(self):
        # 冻结材料是可选的本地输入：按 DRONE3D_REPO_ROOT 约定定位，缺失时跳过。
        root = Path(os.environ.get("DRONE3D_REPO_ROOT", Path(__file__).resolve().parents[4]))
        frozen = root / "Data/experiments/testflight3_full_route_20260915"
        seams, plan = frozen / "reports/seams.json", frozen / "inputs/candidate_windows.json"
        if not seams.exists() or not plan.exists():
            self.skipTest("frozen threshold table or plan is not present")
        from drone3d_reconstruction_core.g4.map_backend import build_backend
        profile = {"method_id": "windowed_sim3_graph_20261008", "window_plan_path": str(plan),
                   "threshold_table_path": str(seams), "threshold_group": "G0",
                   "fit_exclusion_source": "", "window_frame_semantics": "new_frames",
                   "minimum_total_frames": 100, "window_frames": 100, "overlap_frames": 20,
                   "registration": {"ransac_degenerate": "abort"}}
        backend = build_backend(self.root / "records", SESSION, self.root / "out", profile)
        try:
            self.assertEqual(len(backend.plan), 38)
            # 冻结表按 F0NN 命名；运行时窗口号是整数，必须能对上。
            value, source = backend.threshold_policy.lookup(0, 1)
            self.assertAlmostEqual(value, 0.014139644801616669, places=15)
            self.assertTrue(source.endswith("seams.json"))
            # 末段缺记录的两条边必须落回重算，而不是被当成 0。
            self.assertGreaterEqual(backend.threshold_policy.skipped_records, 1)
        finally:
            backend.close()


    def test_plan_shape_is_checked_against_the_declared_window(self):
        from drone3d_reconstruction_core.g4.map_backend import build_backend, check_plan_shape
        from g4_backend import plan_from_records
        profile = {"window_frame_semantics": "new_frames", "window_frames": 100,
                   "overlap_frames": 20, "minimum_total_frames": 100}
        good = plan_from_records([
            {"id": "F000", "new_start": 0, "new_end_exclusive": 100,
             "input_start": 0, "input_end_exclusive": 100, "shared_prefix": 0},
            {"id": "F001", "new_start": 100, "new_end_exclusive": 200,
             "input_start": 80, "input_end_exclusive": 200, "shared_prefix": 20},
            {"id": "F002", "new_start": 200, "new_end_exclusive": 274,
             "input_start": 174, "input_end_exclusive": 274, "shared_prefix": 26},
        ])
        check_plan_shape(good, profile)  # 尾窗 26 共享 / 100 输入：接受
        # 短窗借用不足：模型输入低于下限
        starved = plan_from_records([
            {"id": "F000", "new_start": 0, "new_end_exclusive": 100,
             "input_start": 0, "input_end_exclusive": 100, "shared_prefix": 0},
            {"id": "F001", "new_start": 100, "new_end_exclusive": 174,
             "input_start": 80, "input_end_exclusive": 174, "shared_prefix": 20},
        ])
        with self.assertRaises(ValueError):
            check_plan_shape(starved, profile)
        # 常规窗与会话声明不一致
        drifted = plan_from_records([
            {"id": "F000", "new_start": 0, "new_end_exclusive": 100,
             "input_start": 0, "input_end_exclusive": 100, "shared_prefix": 0},
            {"id": "F001", "new_start": 100, "new_end_exclusive": 200,
             "input_start": 70, "input_end_exclusive": 200, "shared_prefix": 30},
        ])
        with self.assertRaises(ValueError):
            check_plan_shape(drifted, profile)
        # total_frames 语义下不做该校验
        check_plan_shape(drifted, {"window_frame_semantics": "total_frames"})

    def test_threshold_table_group_ambiguity_is_rejected(self):
        root = Path(os.environ.get("DRONE3D_REPO_ROOT", Path(__file__).resolve().parents[4]))
        seams = root / "Data/experiments/testflight3_full_route_20260915/reports/seams.json"
        if not seams.exists():
            self.skipTest("frozen threshold table is not present")
        from drone3d_reconstruction_core.g4.map_backend import load_threshold_table
        with self.assertRaises(ValueError):
            load_threshold_table(seams)

class PortedArtifactTests(unittest.TestCase):
    """移植字段必须真的落到消费者读取的产物里（run.json / revision.json / 窗口元数据）。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records, self.loaded, self.services = [], {}, []
        from g4_backend import ThresholdPolicy, plan_from_records
        self.policy = ThresholdPolicy({"0|1": 0.02}, source="registered_table.json")
        self.plan = plan_from_records([
            {"id": "F000", "new_start": 0, "new_end_exclusive": 30,
             "input_start": 0, "input_end_exclusive": 30, "shared_prefix": 0},
            {"id": "F001", "new_start": 30, "new_end_exclusive": 50,
             "input_start": 20, "input_end_exclusive": 50, "shared_prefix": 10},
        ], source="registered_plan.json")

    def tearDown(self):
        for service in self.services:
            service.close()

    def add_window(self, window_id, start, **kwargs):
        record, loaded = fixture(window_id, start, **kwargs)
        self.records.append(record)
        self.loaded[window_id] = loaded
        return record

    def service(self):
        from drone3d_reconstruction_core.g4.map_backend import G4MapBackend
        from g4_backend import WINDOWED_METHOD_ID
        service = G4MapBackend(self.root/"records", SESSION, self.root/"output",
            reader=Reader(self.loaded), scanner=lambda root, session: list(self.records),
            threshold_policy=self.policy, plan=self.plan, method_id=WINDOWED_METHOD_ID,
            exclusion_source={"rgb-20": [[1., 1.], [2., 2.]]})
        self.services.append(service)
        return service

    def test_run_and_revision_record_the_ported_input_policy(self):
        from g4_backend import WINDOWED_METHOD_ID
        self.add_window(0, 0, count=30)
        self.add_window(1, 20, count=30, shared=10)
        service = self.service()
        path = service.update()
        run = json.loads((self.root/"output"/"run.json").read_text())
        self.assertEqual(run["method_id"], WINDOWED_METHOD_ID)
        self.assertEqual(run["input_policy"]["recorded_threshold_count"], 1)
        self.assertEqual(run["input_policy"]["plan_digest"], self.plan.digest())
        self.assertEqual(run["input_policy"]["exclusion_frame_count"], 1)
        revision = json.loads((path/"revision.json").read_text())
        self.assertEqual(revision["method_id"], WINDOWED_METHOD_ID)
        self.assertEqual(revision["graph"]["method_id"], WINDOWED_METHOD_ID)
        self.assertEqual(revision["edge_thresholds"],
                         [{"i": 0, "j": 1, "value": 0.02, "source": "registered_table.json"}])
        self.assertEqual(revision["input_policy"]["plan_source"], "registered_plan.json")
        second = revision["windows"][1]["source"]
        self.assertEqual(second["method_id"], WINDOWED_METHOD_ID)
        self.assertEqual(second["plan"]["digest"], self.plan.digest())
        self.assertEqual(second["threshold"], {"value": 0.02, "source": "registered_table.json"})
        self.assertEqual(second["registration"]["exclusion_source"], "inline_mapping")

    def test_plan_mismatch_stops_ingestion_without_a_revision(self):
        from drone3d_reconstruction_core.g4.map_backend import G4MapBackend
        self.add_window(0, 0, count=30)
        self.add_window(1, 20, count=30, shared=5)  # 计划要求 10 个共享帧
        service = self.service()
        with self.assertRaises(ValueError):
            service.update()
        self.assertFalse((self.root/"output"/"revisions").exists() and
                         any((self.root/"output"/"revisions").iterdir()))

class InputPolicyContentBindingTests(unittest.TestCase):
    """输入策略必须按**内容**绑定：同路径、同条目数但数值变化不得复用旧输出目录。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "records").mkdir()
        self.out = self.root / "output"

    def open_backend(self, thresholds, exclusions):
        from g4_backend import ThresholdPolicy
        return G4MapBackend(self.root / "records", SESSION, self.out,
                            threshold_policy=ThresholdPolicy(thresholds, source="registered_table.json"),
                            exclusion_source=exclusions)

    def test_changed_threshold_values_reject_reuse(self):
        backend = self.open_backend({"0|1": 0.03}, {"123": [[10.0, 20.0]]})
        digest = backend.config_digest
        backend.close()
        with self.assertRaises(ValueError):
            self.open_backend({"0|1": 3.0}, {"123": [[10.0, 20.0]]})

    def test_changed_exclusion_content_rejects_reuse(self):
        backend = self.open_backend({"0|1": 0.03}, {"123": [[10.0, 20.0]]})
        backend.close()
        with self.assertRaises(ValueError):
            self.open_backend({"0|1": 0.03}, {"123": [[110.0, 120.0]]})

    def test_identical_content_reuses_the_output_directory(self):
        backend = self.open_backend({"0|1": 0.03}, {"123": [[10.0, 20.0]]})
        digest = backend.config_digest
        backend.close()
        again = self.open_backend({"0|1": 0.03}, {"123": [[10.0, 20.0]]})
        try:
            self.assertEqual(again.config_digest, digest)
        finally:
            again.close()

    def test_content_is_bound_even_when_only_order_differs(self):
        backend = self.open_backend({"0|1": 0.03, "1|2": 0.05}, {"123": [[10.0, 20.0]]})
        backend.close()
        reordered = self.open_backend({"1|2": 0.05, "0|1": 0.03}, {"123": [[10.0, 20.0]]})
        reordered.close()

    def test_alias_spellings_of_the_same_edge_share_one_digest(self):
        """同一策略的两种拼写必须得到同一摘要；冲突拼写则在构造时就拒绝。"""
        from g4_backend import ThresholdPolicy
        from drone3d_reconstruction_core.g4.map_backend import _input_policy
        single = _input_policy(ThresholdPolicy({"0|1": 0.03}, source="t"), {}, "none", None)
        alias = _input_policy(ThresholdPolicy({"F000|F001": 0.03}, source="t"), {}, "none", None)
        self.assertEqual(single, alias)
        with self.assertRaises(ValueError):
            ThresholdPolicy({"0|1": 0.03, "F000|F001": 3.0})


def pose_fixture(window_id, start, **kwargs):
    """fixture() plus localization camera rotations (identity, matching the local frame).

    Confidence varies over the image like VGGT's, so half of each frame lies above its median.
    """
    import dataclasses
    record, loaded = fixture(window_id, start, **kwargs)
    rotations = np.repeat(np.eye(3)[None], len(loaded.reference_valid), axis=0)
    confidence = np.broadcast_to(1 + np.arange(32*32).reshape(32, 32) / 1024., loaded.geometry.depth.shape).copy()
    geometry = dataclasses.replace(loaded.geometry, confidence=confidence)
    return record, WindowInput(geometry, loaded.metadata, loaded.reference_centers,
                               loaded.reference_valid, rotations)


class FramePosePlacementTest(unittest.TestCase):
    """placement=frame_pose：逐帧外部位姿 + 每窗 VGGT 深度尺度。fixture 的定位 = 2×局部 + 平移。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records, self.loaded, self.services = [], {}, []

    def tearDown(self):
        for service in self.services:
            service.close()

    def add_window(self, window_id, start, **kwargs):
        record, loaded = pose_fixture(window_id, start, **kwargs)
        self.records.append(record)
        self.loaded[window_id] = loaded

    def service(self, **kwargs):
        service = G4MapBackend(self.root / "records", SESSION, self.root / "output",
                               reader=Reader(self.loaded), scanner=lambda root, session: list(self.records),
                               placement="frame_pose", **kwargs)
        self.services.append(service)
        return service

    @staticmethod
    def revision(path):
        return json.loads((path / "revision.json").read_text())

    def test_frames_sit_on_localization_poses_with_window_depth_scale(self):
        self.add_window(0, 0)
        self.add_window(1, 30)
        path = self.service().update()
        revision = self.revision(path)
        self.assertEqual(revision["placement"], "frame_pose")
        from g4_backend import FRAME_POSE_METHOD_ID
        self.assertEqual(revision["method_id"], FRAME_POSE_METHOD_ID)
        self.assertEqual(revision["published_windows"], [0, 1])
        for w in revision["windows"]:
            self.assertAlmostEqual(w["depth_scale"], 2.0, places=9)
            self.assertEqual(w["intrinsics_source"], "vggt_window_median")
            np.testing.assert_allclose([p["center"] for p in w["frame_poses"]],
                                       self.loaded[w["window_id"]].reference_centers, atol=1e-9)
            local_to_map = Sim3.from_dict(w["local_to_map"])
            self.assertAlmostEqual(local_to_map.scale, 2.0, places=9)
        with np.load(path / "trajectory.npz") as z:
            self.assertEqual(len(z["centers"]), 60)
        self.assertEqual(json.loads((self.root / "output" / "run.json").read_text())["placement"], "frame_pose")

    def test_pose_time_offset_interpolates_and_never_extrapolates(self):
        from g4_backend import FramePoseConfig
        self.add_window(0, 0)
        self.add_window(1, 30)
        # fixture frames are 1 s apart; +500 ms lands midway between consecutive poses.
        path = self.service(frame_pose_config=FramePoseConfig(pose_time_offset_ms=500,
                                                              max_pose_gap_ms=1500)).update()
        windows = self.revision(path)["windows"]
        refs0 = self.loaded[0].reference_centers
        np.testing.assert_allclose(windows[0]["frame_poses"][0]["center"], (refs0[0] + refs0[1]) / 2, atol=1e-9)
        # Window 0's last frame brackets with window 1's first frame. The session's last
        # frame has no bracket and keeps its own RGB-time pose instead of being dropped.
        self.assertEqual(windows[0]["frame_pose_time_shifted"], 30)
        np.testing.assert_allclose(windows[1]["frame_poses"][-1]["center"],
                                   self.loaded[1].reference_centers[-1], atol=1e-12)
        self.assertEqual(windows[1]["frame_pose_valid"], 30)
        self.assertEqual(windows[1]["frame_pose_time_shifted"], 29)

    def test_shared_frames_give_a_depth_ratio_edge_without_e1(self):
        self.add_window(0, 0)
        self.add_window(1, 25, shared=5)
        revision = self.revision(self.service().update())
        source = revision["windows"][1]["source"]
        self.assertEqual(source["scale_edge"], {"i": 0, "j": 1, "scale": 1.0, "frames": 5})
        self.assertIsNone(source["edge"])
        self.assertEqual(source["registration"]["reason"], "frame_pose_uses_shared_depth_ratio")
        self.assertEqual(revision["graph"]["edges"], 1)
        self.assertEqual(revision["published_windows"], [0, 1])

    def test_window_without_scale_constraint_stays_pending_with_local_gauge(self):
        self.add_window(0, 0)
        self.add_window(1, 40, valid_reference=False)
        revision = self.revision(self.service().update())
        self.assertEqual(revision["published_windows"], [0])
        pending = revision["windows"][1]
        self.assertFalse(pending["included_in_global_preview"])
        self.assertEqual(pending["global_frame_status"], "no_metric_scale")
        self.assertIn("prediction_to_map", pending)
        self.assertNotIn("frame_poses", pending)

    def test_profile_intrinsics_are_resized_to_model_pixels(self):
        self.add_window(0, 0)
        intrinsics = {"width": 64, "height": 64, "fx": 40., "fy": 44., "cx": 31.5, "cy": 30.}
        window = self.revision(self.service(camera_intrinsics=intrinsics).update())["windows"][0]
        self.assertEqual(window["intrinsics_source"], "profile")
        K = np.array(window["intrinsics"])
        np.testing.assert_allclose(K[[0, 1, 0, 1], [0, 1, 2, 2]], [20., 22., 15.5, 14.75])

    def test_graph_runs_keep_their_historical_run_file(self):
        self.add_window(0, 0)
        service = G4MapBackend(self.root / "records", SESSION, self.root / "output",
                               reader=Reader(self.loaded), scanner=lambda root, session: list(self.records))
        self.services.append(service)
        stored = json.loads((self.root / "output" / "run.json").read_text())
        for key in ("placement", "frame_pose", "camera_intrinsics"):
            self.assertNotIn(key, stored)

    def test_frame_pose_profile_builds_a_backend_with_its_method_identity(self):
        from drone3d_reconstruction_core.g4.map_backend import build_backend
        from g4_backend import FRAME_POSE_METHOD_ID
        (self.root / "records").mkdir()
        backend = build_backend(self.root / "records", SESSION, self.root / "out", {
            "window_plan_path": "", "threshold_table_path": "", "fit_exclusion_source": "",
            "window_frame_semantics": "total_frames", "registration": {}, "placement": "frame_pose",
            "frame_pose": {"pose_time_offset_ms": 70.0}, "camera_intrinsics": None})
        try:
            self.assertEqual(backend.method_id, FRAME_POSE_METHOD_ID)
            self.assertEqual(backend.frame_pose_config.pose_time_offset_ms, 70.0)
        finally:
            backend.close()
        with self.assertRaises(ValueError):
            G4MapBackend(self.root / "records", SESSION, self.root / "out2", placement="bundle")
