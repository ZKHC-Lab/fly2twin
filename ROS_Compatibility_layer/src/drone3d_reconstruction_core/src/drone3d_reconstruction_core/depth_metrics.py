"""首帧传感器深度对照指标（开发观察值，不作为准确性签署阈值）。

对每个 ``save_sensor_depth=true`` 的成功窗口，读取 snapshot v4 保存的
第一帧对齐传感器深度与 Omega ``predictions.npz`` 的第一帧预测深度，
估计全局尺度（有效像素比值中位数）后计算：

- ``coverage``：两幅深度都有效（>0 且有限）的像素占比；
- ``abs_rel``：|scaled_pred - sensor| / sensor 的均值；
- ``rmse``：sqrt(mean((scaled_pred - sensor)^2))。

该组指标只用于开发对照，不设通过阈值，也不得据此宣称真实尺度精度。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

_SAFE_MAX_DEPTH_BYTES = 256 * 1024 * 1024


def _decode_sensor_depth(snapshot: Any) -> tuple[np.ndarray, str] | tuple[None, str]:
    try:
        if not bool(getattr(snapshot, "save_sensor_depth", False)):
            return None, "sensor depth saving is disabled"
        paths = getattr(snapshot, "depth_paths", ())
        metadata = getattr(snapshot, "depth_metadata", ())
        if not paths or not metadata or not paths[0] or not isinstance(metadata[0], dict):
            return None, "snapshot has no first-frame sensor depth"
        meta = metadata[0]
        relative = str(paths[0])
        path = Path(getattr(snapshot, "canonical_root", Path("."))) / relative
        size = path.lstat().st_size
        if size <= 0 or size > _SAFE_MAX_DEPTH_BYTES:
            return None, "sensor depth file is outside the trusted size bound"
        raw = path.read_bytes()
        encoding = str(meta.get("encoding", ""))
        if encoding == "16UC1":
            dtype = np.dtype("<u2")
            itemsize = 2
        elif encoding == "32FC1":
            dtype = np.dtype("<f4")
            itemsize = 4
        else:
            return None, f"unsupported sensor depth encoding {encoding!r}"
        width = int(meta.get("width", 0))
        height = int(meta.get("height", 0))
        step = int(meta.get("step", width * itemsize))
        if width <= 0 or height <= 0 or step < width * itemsize:
            return None, "sensor depth geometry is invalid"
        if len(raw) < step * height:
            return None, "sensor depth file is truncated"
        data = np.frombuffer(raw[: step * height], dtype=dtype).reshape(
            height, step // itemsize
        )[:, :width]
        if bool(meta.get("is_bigendian", False)):
            # 原始字节按大端解释：frombuffer 已按小端读入，byteswap 元素
            # 字节序后得到正确的数值（例如 bytes 01 00 -> 256.0）。
            data = data.byteswap(inplace=False)
        array = data.astype(np.float64)
        return array, ""
    except BaseException as error:
        return None, f"sensor depth decode failed: {type(error).__name__}: {error}"


def _resize_nearest(source: np.ndarray, height: int, width: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(
            source.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    except BaseException:
        # 无 cv2 的退化路径：按块均值降采样（仅用于开发对照）。
        result = np.zeros((height, width), dtype=np.float64)
        scale_y = source.shape[0] / float(height)
        scale_x = source.shape[1] / float(width)
        for row in range(height):
            y0 = int(math.floor(row * scale_y))
            y1 = max(y0 + 1, int(math.ceil((row + 1) * scale_y)))
            for col in range(width):
                x0 = int(math.floor(col * scale_x))
                x1 = max(x0 + 1, int(math.ceil((col + 1) * scale_x)))
                result[row, col] = float(np.mean(source[y0:y1, x0:x1]))
        return result


def compute_first_frame_depth_metrics(
    snapshot: Any, predictions_path: Path
) -> tuple[dict[str, Any] | None, str]:
    """返回 ``(metrics, reason)``；不可计算时 metrics 为 None。

    reason 用于区分"无传感器深度可对照"（空字符串表示成功）与
    "计算失败"（非空错误描述）。本函数绝不抛异常。
    """
    try:
        sensor, sensor_error = _decode_sensor_depth(snapshot)
        if sensor is None:
            return None, sensor_error
        with np.load(str(predictions_path), allow_pickle=False) as archive:
            if "depth" not in archive:
                return None, "predictions.npz has no depth array"
            depth = archive["depth"]
        if depth.ndim == 4:
            depth = depth[0, ..., 0]
        elif depth.ndim == 3:
            depth = depth[0]
        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim != 2:
            return None, "predictions depth is not a 2D first-frame array"
        # 传感器深度与预测深度分辨率不同：把传感器深度缩放到预测网格后再比较。
        sensor_aligned = _resize_nearest(sensor, depth.shape[0], depth.shape[1])
        valid = (
            (sensor_aligned > 0)
            & np.isfinite(sensor_aligned)
            & (depth > 0)
            & np.isfinite(depth)
        )
        if int(valid.sum()) < 4:
            return None, "fewer than 4 valid depth pixels"
        ratios = sensor_aligned[valid] / depth[valid]
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        if ratios.size < 4:
            return None, "fewer than 4 finite positive depth ratios"
        scale = float(np.median(ratios))
        if not math.isfinite(scale) or scale <= 0:
            return None, "depth scale is not finite/positive"
        scaled = scale * depth
        valid = (
            (sensor_aligned > 0)
            & np.isfinite(sensor_aligned)
            & (scaled > 0)
            & np.isfinite(scaled)
        )
        total = int(sensor_aligned.size)
        if total <= 0 or int(valid.sum()) == 0:
            return None, "no valid pixels after depth scaling"
        sensor_valid = sensor_aligned[valid]
        scaled_valid = scaled[valid]
        abs_rel = float(np.mean(np.abs(scaled_valid - sensor_valid) / sensor_valid))
        rmse = float(np.sqrt(np.mean((scaled_valid - sensor_valid) ** 2)))
        return (
            {
                "frame_index": 0,
                "coverage": float(int(valid.sum()) / total),
                "abs_rel": abs_rel,
                "rmse": rmse,
                "scale": scale,
                "valid_pixels": int(valid.sum()),
                "total_pixels": total,
                "sensor_encoding": str(
                    getattr(snapshot, "depth_metadata", ())[0].get("encoding", "")
                ),
            },
            "",
        )
    except BaseException as error:
        return None, f"depth metrics computation failed: {type(error).__name__}: {error}"
