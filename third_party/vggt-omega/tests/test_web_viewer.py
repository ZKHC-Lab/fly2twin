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


if __name__ == "__main__":
    unittest.main()
