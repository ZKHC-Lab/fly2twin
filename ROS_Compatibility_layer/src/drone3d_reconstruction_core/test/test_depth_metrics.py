"""CPU-only tests for first-frame sensor-depth comparison metrics."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from drone3d_reconstruction_core.depth_metrics import (
    _decode_sensor_depth,
    compute_first_frame_depth_metrics,
)


class DepthMetricsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.predictions = self.root / "predictions.npz"

    def _snapshot(self, depth_array, encoding="16UC1", width=None, height=None):
        if width is None:
            height, width = depth_array.shape
        raw = np.asarray(depth_array, dtype=np.uint16).tobytes()
        path = self.root / "depth" / "frame_000000.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return SimpleNamespace(
            canonical_root=self.root,
            save_sensor_depth=True,
            depth_paths=("depth/frame_000000.bin",),
            depth_metadata=(
                {
                    "encoding": encoding,
                    "width": width,
                    "height": height,
                    "step": width * 2,
                    "is_bigendian": False,
                },
            ),
        )

    def test_metrics_recover_known_scale(self):
        true_depth = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        sensor = 2.0 * true_depth + 0.02 * np.arange(6, dtype=np.float32).reshape(2, 3)
        np.savez(self.predictions, depth=true_depth[None, ..., None])
        snapshot = self._snapshot(sensor.astype(np.uint16))
        metrics, reason = compute_first_frame_depth_metrics(snapshot, self.predictions)
        self.assertIsNotNone(metrics)
        self.assertEqual(reason, "")
        self.assertAlmostEqual(metrics["scale"], 2.0, places=1)
        self.assertGreater(metrics["coverage"], 0.9)
        self.assertLess(metrics["abs_rel"], 0.05)
        self.assertLess(metrics["rmse"], 0.2)

    def test_metrics_with_different_sensor_resolution(self):
        # 传感器 4x6 与预测 2x3：必须先把传感器缩放到预测网格再比较。
        true_depth = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        sensor = np.repeat(np.repeat(2.0 * true_depth, 2, axis=0), 2, axis=1)
        np.savez(self.predictions, depth=true_depth[None, ..., None])
        snapshot = self._snapshot(sensor.astype(np.uint16))
        metrics, reason = compute_first_frame_depth_metrics(snapshot, self.predictions)
        self.assertIsNotNone(metrics)
        self.assertEqual(reason, "")
        self.assertAlmostEqual(metrics["scale"], 2.0, places=1)
        self.assertGreater(metrics["coverage"], 0.9)

    def test_truncated_or_unsupported_depth_returns_none(self):
        np.savez(
            self.predictions,
            depth=np.ones((1, 4, 5, 1), dtype=np.float32),
        )
        snapshot = self._snapshot(np.ones((4, 5), dtype=np.uint16))
        meta = dict(snapshot.depth_metadata[0])
        meta["step"] = 3
        snapshot.depth_metadata = (meta,)
        metrics, reason = compute_first_frame_depth_metrics(
            snapshot, self.predictions
        )
        self.assertIsNone(metrics)
        self.assertIn("geometry", reason)
        snapshot = self._snapshot(np.ones((4, 5), dtype=np.uint16))
        meta = dict(snapshot.depth_metadata[0])
        meta["encoding"] = "8UC1"
        snapshot.depth_metadata = (meta,)
        metrics, reason = compute_first_frame_depth_metrics(
            snapshot, self.predictions
        )
        self.assertIsNone(metrics)
        self.assertIn("encoding", reason)

    def test_bigendian_depth_decode_byteswaps_values(self):
        path = self.root / "depth" / "frame_000000.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        # 大端 16UC1 字节 01 00 表示数值 256.0。
        path.write_bytes(b"\x01\x00")
        snapshot = SimpleNamespace(
            canonical_root=self.root,
            save_sensor_depth=True,
            depth_paths=("depth/frame_000000.bin",),
            depth_metadata=(
                {
                    "encoding": "16UC1",
                    "width": 1,
                    "height": 1,
                    "step": 2,
                    "is_bigendian": True,
                },
            ),
        )
        array, reason = _decode_sensor_depth(snapshot)
        self.assertEqual(reason, "")
        self.assertEqual(float(array[0, 0]), 256.0)

    def test_metrics_none_without_sensor_depth(self):
        np.savez(
            self.predictions,
            depth=np.ones((1, 4, 5, 1), dtype=np.float32),
        )
        snapshot = SimpleNamespace(
            canonical_root=self.root,
            save_sensor_depth=False,
            depth_paths=(None,),
            depth_metadata=(None,),
        )
        metrics, reason = compute_first_frame_depth_metrics(
            snapshot, self.predictions
        )
        self.assertIsNone(metrics)
        self.assertIn("sensor depth saving is disabled", reason)


if __name__ == "__main__":
    unittest.main()
