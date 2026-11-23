"""坐标状态常量与里程计轨迹的共用工具。

- ``COORDINATE_MODEL_LOCAL_UNALIGNED`` / ``COORDINATE_ODOM_LOCAL_APPROX``：
  正式记录与 Web 使用的坐标状态；
- :func:`valid_sim3_alignment`：校验正式记录里的 Sim(3) 对齐载荷；
- :func:`_reference_camera_positions`：由逐帧完整 Odometry 计算参考相机轨迹
  （G4 输入读取使用）；
- :func:`_extract_transform`：把 tf.json 条目转成刚体变换。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

COORDINATE_MODEL_LOCAL_UNALIGNED = "MODEL_LOCAL_UNALIGNED"
COORDINATE_ODOM_LOCAL_APPROX = "ODOM_LOCAL_APPROX"


def _finite_float(value: Any, label: str) -> float:
    if value is None or isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} is not a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} is not a finite number") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{label} is not finite")
    return parsed


def _integral_int(
    value: Any,
    label: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> int:
    """把数值严格转成整数，拒绝 bool/str/非整数（防止 float stamp 被静默截断）。"""
    if value is None or isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{label} is not an integer")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} is not an integer") from error
    if not np.isfinite(parsed) or not float(parsed).is_integer():
        raise ValueError(f"{label} is not an integer")
    result = int(parsed)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _quaternion_to_rotation(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """单位四元数（x,y,z,w）→ 3x3 旋转矩阵。"""
    x, y, z, w = qx, qy, qz, qw
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("quaternion norm is not positive")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _extract_transform(tf_entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """把 tf.json 条目转成 T_parent_child（child 坐标映射到 parent）。"""
    translation = tf_entry["translation"]
    rotation = tf_entry["rotation"]
    t = np.array(
        [
            _finite_float(translation["x"], "tf.translation.x"),
            _finite_float(translation["y"], "tf.translation.y"),
            _finite_float(translation["z"], "tf.translation.z"),
        ],
        dtype=np.float64,
    )
    r = _quaternion_to_rotation(
        _finite_float(rotation["x"], "tf.rotation.x"),
        _finite_float(rotation["y"], "tf.rotation.y"),
        _finite_float(rotation["z"], "tf.rotation.z"),
        _finite_float(rotation["w"], "tf.rotation.w"),
    )
    return r, t


def _reference_camera_positions(
    snapshot: Any,
    body_camera_rotation: np.ndarray,
    body_camera_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, np.ndarray | None]:
    """由逐帧完整 Odometry 计算参考相机轨迹（odom 系）。

    Returns:
        (indices, positions, rotations, reference_frame_id, odom_stamps)
    """
    indices = []
    positions = []
    rotations = []
    stamps = []
    reference_frame_id = None
    pose_valid_flags = getattr(snapshot, "pose_valid", None)
    payloads = snapshot.odometry_payloads or ()
    for index, payload in enumerate(payloads):
        if payload is None:
            continue
        if (
            pose_valid_flags is not None
            and len(pose_valid_flags) == len(payloads)
            and pose_valid_flags[index] is not True
        ):
            continue
        frame_id = str(payload.get("header", {}).get("frame_id", "") or "")
        if not frame_id:
            continue
        if reference_frame_id is None:
            reference_frame_id = frame_id
        elif frame_id != reference_frame_id:
            raise ValueError("odometry reference frame changed inside window")
        position = payload.get("pose", {}).get("position", {})
        try:
            p = np.array(
                [
                    _finite_float(position.get("x"), "odometry.position.x"),
                    _finite_float(position.get("y"), "odometry.position.y"),
                    _finite_float(position.get("z"), "odometry.position.z"),
                ],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"odometry payload at frame {index} has invalid position"
            ) from error
        orientation = payload.get("pose", {}).get("orientation", {})
        try:
            rotation = _quaternion_to_rotation(
                _finite_float(orientation.get("x"), "odometry.orientation.x"),
                _finite_float(orientation.get("y"), "odometry.orientation.y"),
                _finite_float(orientation.get("z"), "odometry.orientation.z"),
                _finite_float(orientation.get("w"), "odometry.orientation.w"),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"odometry payload at frame {index} has invalid orientation"
            ) from error
        # Odometry.pose 是机体原点在 odom 系下的位姿：
        # T_odom_camera = T_odom_body @ T_body_camera，
        # 参考相机位置 = p_odom_body + R_odom_body @ t_body_camera。
        camera_position = p + rotation @ body_camera_translation
        camera_rotation = rotation @ body_camera_rotation
        indices.append(index)
        positions.append(camera_position)
        rotations.append(camera_rotation)
        stamp = payload.get("header", {}).get("stamp", {})
        try:
            stamps.append(
                (
                    _integral_int(
                        stamp.get("sec"), "odometry.stamp.sec"
                    ),
                    _integral_int(
                        stamp.get("nsec"),
                        "odometry.stamp.nsec",
                        minimum=0,
                        maximum=999_999_999,
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"odometry payload at frame {index} has invalid stamp"
            ) from error
    if reference_frame_id is None:
        raise ValueError("no complete odometry payload in snapshot")
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(rotations, dtype=np.float64),
        reference_frame_id,
        tuple(stamps) if stamps else None,
    )


def valid_sim3_alignment(alignment: Any) -> bool:
    """Validate an odometry Sim(3) provenance payload."""
    if not isinstance(alignment, dict):
        return False
    if alignment.get("coordinate_state") != COORDINATE_ODOM_LOCAL_APPROX:
        return False
    scale = alignment.get("scale")
    rotation = alignment.get("rotation")
    translation = alignment.get("translation")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(scale))
        or float(scale) <= 0.0
        or not isinstance(rotation, list)
        or len(rotation) != 3
        or not isinstance(translation, list)
        or len(translation) != 3
    ):
        return False
    try:
        matrix = [[float(value) for value in row] for row in rotation]
        offset = [float(value) for value in translation]
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        any(not isinstance(row, list) or len(row) != 3 for row in rotation)
        or not all(math.isfinite(value) for row in matrix for value in row)
        or not all(math.isfinite(value) for value in offset)
    ):
        return False
    for left in range(3):
        for right in range(3):
            dot = sum(matrix[left][index] * matrix[right][index] for index in range(3))
            if not math.isclose(dot, 1.0 if left == right else 0.0, abs_tol=1e-4):
                return False
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    return math.isclose(determinant, 1.0, abs_tol=1e-4)
