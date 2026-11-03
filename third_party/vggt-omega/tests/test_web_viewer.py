"""Small CPU-only check for dashboard render helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import web_viewer
import result_watch_viewer


class WebViewerTest(unittest.TestCase):
    def test_splat_record_and_depth_preview(self):
        scene = trimesh.Scene()
        scene.add_geometry(
            trimesh.PointCloud(
                vertices=[[0, 0, 1], [1, 2, 3]],
                colors=[[10, 20, 30, 255], [40, 50, 60, 255]],
            ),
            transform=trimesh.transformations.translation_matrix([4, 5, 6]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.splat"
            web_viewer.scene_to_splat(scene, path)
            self.assertEqual(path.stat().st_size, 64)
            records = np.fromfile(path, dtype=np.dtype([
                ("position", "<f4", (3,)),
                ("scale", "<f4", (3,)),
                ("rgba", "u1", (4,)),
                ("rotation", "u1", (4,)),
            ]))
            np.testing.assert_allclose(records["position"][0], [4, 5, 7])
            self.assertTrue(np.all(records["scale"] > 0))
            np.testing.assert_array_equal(records["rgba"][:, 3], [220, 220])

        preview, value_range = web_viewer.colorize_depth([[0, 1], [2, 3]])
        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertIsNotNone(value_range)



    def test_scan_result_directories_filters_and_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            root.mkdir()
            older = root / "job-old"
            newer = root / "job-new"
            staging = root / ".drone3d-result-staging"
            missing = root / "job-missing"
            for path in (older, newer, staging, missing):
                path.mkdir()
            for path in (older, newer):
                (path / "predictions.npz").write_bytes(b"npz")
                payload = {
                    "scene_id": "scene",
                    "patch_id": path.name,
                    "profile": {"name": "nx16g_8x256", "image_resolution": 256,
                                "frame_group_size": 8},
                    "actual_frame_count": 8,
                    "stage_timing": {"worker_wall_time_sec": 22.5},
                    "worker_exit": {"cause": "worker_exit", "returncode": 0},
                    "snapshot": {"sha256": "a" * 64},
                }
                import json
                (path / "result.json").write_text(json.dumps(payload))
            (staging / "predictions.npz").write_bytes(b"npz")
            (staging / "result.json").write_text("{}")
            (missing / "result.json").write_text("{}")
            os = __import__("os")
            os.utime(newer / "result.json", ns=(2_000_000_000, 2_000_000_000))
            os.utime(older / "result.json", ns=(1_000_000_000, 1_000_000_000))
            entries = result_watch_viewer.scan_result_directories(root)
            self.assertEqual([Path(e["path"]).name for e in entries], ["job-new", "job-old"])
            self.assertEqual(entries[0]["summary"]["profile_name"], "nx16g_8x256")

    def test_scan_result_directories_rejects_symlink_result_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "results"
            root.mkdir()
            target = Path(directory) / "outside.json"
            target.write_text("{}")
            candidate = root / "job"
            candidate.mkdir()
            (candidate / "predictions.npz").write_bytes(b"npz")
            (candidate / "result.json").symlink_to(target)
            self.assertEqual(result_watch_viewer.scan_result_directories(root), [])


if __name__ == "__main__":
    unittest.main()
