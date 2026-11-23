"""Real-core tests for ingestion, immutable revisions and isolated failures."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from g4_backend import Sim3, WindowGeometry
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
                    "placement": "frame_pose", "reference_frame_id": "odom", "window_count": count, "windows": windows,
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

        frame_pose = {"placement": "frame_pose", "frame_pose": {"pose_time_offset_ms": 40.0},
                      "window_frames": 16, "overlap_frames": 5, "camera_intrinsics": None}
        for profile, unknown in ((frame_pose, False), (frame_pose, True)):
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
        record, loaded = pose_fixture(window_id, start, **kwargs)
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

    def test_pending_solve_retains_one_complete_display_then_recovers(self):
        from drone3d_reconstruction_core.g4 import map_backend
        self.add_window(0, 0)
        service = self.service()
        first = service.update()
        self.add_window(1, 20, shared=10)
        real = map_backend.solve_window_scales
        def pending(*args, **kwargs):
            _scales, constrained, diagnostics = real(*args, **kwargs)
            return {}, constrained, diagnostics  # no window gets a metric scale
        with patch("drone3d_reconstruction_core.g4.map_backend.solve_window_scales", side_effect=pending):
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
        self.assertEqual(revision["graph"]["edges"], 1)
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
        self.assertEqual(revision["graph"]["edges"], 0)
        source = revision["windows"][1]["source"]
        self.assertEqual(source["declared_shared_prefix"], 5)
        self.assertEqual(source["shared_prefix"], 0)
        with np.load(latest/"trajectory.npz") as data:
            self.assertEqual(len(data["timestamps_ns"]), 60)
            self.assertEqual(len(np.unique(data["timestamps_ns"])), 60)


class BackendProfileTests(unittest.TestCase):
    """会话写的 backend_profile.json 直接构造后端；老会话的旧键取中性值时照常，取已删功能的值时报错。"""

    # Profile written before graph placement and registered plans were removed
    # (the reference flight of 2026-11-20 carries exactly these keys).
    OLD_FRAME_POSE = {
        "camera_intrinsics": None, "fit_exclusion_source": "",
        "frame_pose": {"frame_scales": True, "pose_time_offset_ms": 40.0, "pose_time_offset_stage": "capture"},
        "frame_pose_fusion": True, "method_id": "frame_pose_window_scale_20261008",
        "minimum_total_frames": 100, "overlap_frames": 5, "placement": "frame_pose", "registration": {},
        "threshold_group": "", "threshold_table_path": "", "window_frame_semantics": "total_frames",
        "window_frames": 40, "window_plan_path": "",
    }

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "records").mkdir()

    def build(self, profile):
        from drone3d_reconstruction_core.g4.map_backend import build_backend
        return build_backend(self.root / "records", SESSION, self.root / "out", profile)

    def test_old_frame_pose_profile_with_neutral_removed_keys_builds(self):
        backend = self.build(self.OLD_FRAME_POSE)
        try:
            self.assertEqual(backend.method_id, "frame_pose_window_scale_20261008")
            self.assertTrue(backend.frame_pose_config.frame_scales)
            self.assertEqual(backend.frame_pose_config.pose_time_offset_ms, 40.0)
        finally:
            backend.close()

    def test_profiles_that_used_removed_features_are_rejected(self):
        for change, message in (
            ({"placement": "graph"}, "graph placement was removed"),
            ({"placement": None}, "graph placement was removed"),
            ({"window_frame_semantics": "new_frames"}, "removed feature"),
            ({"window_plan_path": "/plans/candidate_windows.json"}, "removed feature"),
            ({"threshold_table_path": "/reports/seams.json"}, "removed feature"),
            ({"threshold_group": "G0"}, "removed feature"),
            ({"fit_exclusion_source": "/reports/exclusions.json"}, "removed feature"),
            ({"registration": {"ransac_degenerate": "abort"}}, "removed feature"),
        ):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.build({**self.OLD_FRAME_POSE, **change})
        self.assertFalse((self.root / "out").exists())


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
                               **kwargs)
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

    def test_cross_window_rows_pull_a_too_deep_window_and_are_deterministic(self):
        import dataclasses
        from g4_backend import FramePoseConfig
        from drone3d_reconstruction_core.g4.input_adapter import WindowInput
        self.add_window(0, 0)
        self.add_window(1, 30)
        # 3 m walls (cross rows count 0.3-4 m depths). Window 1's depth is 20 % too deep for its
        # own trajectory; only the overlap shows it.
        for index, factor in ((0, .5), (1, .6)):
            loaded = self.loaded[index]
            self.loaded[index] = WindowInput(dataclasses.replace(loaded.geometry, depth=loaded.geometry.depth * factor),
                                             loaded.metadata, loaded.reference_centers, loaded.reference_valid,
                                             loaded.reference_rotations)

        def scales(output, weight):
            config = FramePoseConfig(pose_time_offset_stage="capture", cross_window_weight=weight)
            for loaded in self.loaded.values():
                loaded.metadata["capture_pose_time_offset_ms"] = 0.0
            service = G4MapBackend(self.root / "records", SESSION, self.root / output, reader=Reader(self.loaded),
                                   scanner=lambda root, session: list(self.records),
                                   frame_pose_config=config)
            self.services.append(service)
            windows = self.revision(service.update())["windows"]
            return [w["depth_scale"] for w in windows], windows

        plain, _ = scales("plain", 0.0)
        self.assertFalse((self.root / "plain" / "windows" / "window-000001" / "cross.npz").exists())
        crossed, windows = scales("cross", 1.0)
        again, _ = scales("cross_again", 1.0)
        self.assertEqual(crossed, again)
        self.assertLess(crossed[1], plain[1] * 0.97)
        self.assertGreater(crossed[0], plain[0])
        self.assertGreater(windows[1]["scale_diagnostics"]["cross_window_rows"], 0)
        with np.load(self.root / "cross" / "windows" / "window-000001" / "cross.npz") as rows:
            self.assertTrue(set(rows["window_i"]) | set(rows["window_j"]) <= {0, 1})
            self.assertTrue(np.all(rows["window_i"] != rows["window_j"]))
        with np.load(self.root / "cross" / "windows" / "window-000000" / "cross.npz") as rows:
            self.assertEqual(len(rows["value"]), 0)  # the first window has nothing before it

    def test_frame_scales_follow_rows_inside_and_across_windows(self):
        import dataclasses
        from g4_backend import FramePoseConfig
        from drone3d_reconstruction_core.g4.input_adapter import WindowInput
        self.add_window(0, 0)
        self.add_window(1, 25, shared=5)
        # 1.5 m walls. The second half of window 1 is 15 % too deep: a shape error a window
        # scale cannot fix, visible to rows against its own first half and window 0.
        factors = {0: np.full(30, .5), 1: np.r_[np.full(15, .5), np.full(15, .575)]}
        for index, factor in factors.items():
            loaded = self.loaded[index]
            depth = loaded.geometry.depth * factor[:, None, None]
            self.loaded[index] = WindowInput(dataclasses.replace(loaded.geometry, depth=depth),
                                             dict(loaded.metadata, capture_pose_time_offset_ms=0.0),
                                             loaded.reference_centers, loaded.reference_valid,
                                             loaded.reference_rotations)

        def solve(output):
            config = FramePoseConfig(pose_time_offset_stage="capture", frame_scales=True)
            service = G4MapBackend(self.root / "records", SESSION, self.root / output, reader=Reader(self.loaded),
                                   scanner=lambda root, session: list(self.records),
                                   frame_pose_config=config)
            self.services.append(service)
            return self.revision(service.update())["windows"]

        windows = solve("frames")
        self.assertEqual(windows, solve("frames_again"))
        w0, w1 = windows
        self.assertEqual(len(w1["frame_depth_scales"]), 30)
        self.assertTrue(all(v is None for v in w1["frame_depth_scales"][:5]))  # owned by window 0
        self.assertTrue(all(v is not None for v in w0["frame_depth_scales"]))
        late, early = np.mean(w1["frame_depth_scales"][-5:]), np.mean(w1["frame_depth_scales"][5:10])
        self.assertLess(late / early, 0.95)  # the too-deep half gets a smaller scale
        with np.load(self.root / "frames" / "windows" / "window-000001" / "cross.npz") as rows:
            same = rows["window_i"] == rows["window_j"]
            self.assertTrue(same.any() and (~same).any())

    def test_frame_scales_replace_weighted_cross_rows(self):
        from g4_backend import FramePoseConfig
        with self.assertRaises(ValueError):
            FramePoseConfig(pose_time_offset_stage="capture", frame_scales=True, cross_window_weight=.3)

    def test_cross_window_rows_need_capture_stage_poses(self):
        from g4_backend import FramePoseConfig
        with self.assertRaises(ValueError):
            FramePoseConfig(cross_window_weight=.3)

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

    def test_frame_pose_profile_builds_a_backend_with_its_method_identity(self):
        from drone3d_reconstruction_core.g4.map_backend import build_backend
        from g4_backend import FRAME_POSE_METHOD_ID
        (self.root / "records").mkdir()
        backend = build_backend(self.root / "records", SESSION, self.root / "out", {
            "placement": "frame_pose", "frame_pose": {"pose_time_offset_ms": 70.0}, "camera_intrinsics": None})
        try:
            self.assertEqual(backend.method_id, FRAME_POSE_METHOD_ID)
            self.assertEqual(backend.frame_pose_config.pose_time_offset_ms, 70.0)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
