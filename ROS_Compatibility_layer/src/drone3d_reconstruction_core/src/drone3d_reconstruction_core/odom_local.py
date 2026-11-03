"""ODOM_LOCAL_APPROX：窗口级 Omega 相机轨迹与 odom 参考轨迹的 Sim(3) 近似对齐。

本模块只做"可计算性检查 + 质量门禁后的近似对齐"，绝不宣称测量精度：
- 用逐帧相机姿态确定旋转，再由相机位置估计尺度和平移；
- 参考轨迹基线、姿态残差、归一化位置残差均通过门限时标记
  ``ODOM_LOCAL_APPROX``；
- 任一条件不满足时回退 ``MODEL_LOCAL_UNALIGNED`` 并给出明确原因；
- 结果始终携带 ``unreliable=true``（近似、不可靠、未经真值验证）。

输入：
- :class:`~drone3d_reconstruction_core.snapshot_validator.ValidatedSnapshot`
  （v4 snapshot 的逐帧完整 Odometry payload 与 tf.json 条目）
- Omega result 目录中的 ``predictions.npz``（``extrinsic`` 为 N×3×4）

输出：:class:`OdomLocalAlignment`（含 scale/rotation/translation/residual）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

COORDINATE_MODEL_LOCAL_UNALIGNED = "MODEL_LOCAL_UNALIGNED"
COORDINATE_ODOM_LOCAL_APPROX = "ODOM_LOCAL_APPROX"


@dataclass(frozen=True)
class OdomLocalAlignment:
    """一次窗口对齐结果（可计算性门禁之后的只读产物）。"""

    coordinate_state: str
    reference_frame_id: str
    correspondence_count: int
    scale: float | None
    rotation: list[list[float]] | None
    translation: list[float] | None
    residual_rms: float | None
    reason: str
    reference_first_position: tuple[float, float, float] | None = None
    reference_last_position: tuple[float, float, float] | None = None
    reference_first_stamp: tuple[int, int] | None = None
    reference_last_stamp: tuple[int, int] | None = None
    reference_baseline_m: float | None = None
    orientation_residual_deg: float | None = None
    normalized_residual_rms: float | None = None
    reference_samples: tuple[dict, ...] = ()


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


def _invert_transform(rotation: np.ndarray, translation: np.ndarray):
    return rotation.T, -(rotation.T @ translation)


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


def _resolve_body_camera_transform(
    tf_entries: tuple[dict, ...],
    body_frame_id: str | None,
    camera_frame_id: str | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """从 tf.json 解析 T_body_camera（child=camera, parent=body）。"""
    if not body_frame_id or not camera_frame_id:
        return None
    static_entries = [entry for entry in tf_entries if entry.get("static")]

    def _find(entries):
        for entry in entries:
            parent = entry.get("parent_frame_id")
            child = entry.get("child_frame_id")
            if parent == body_frame_id and child == camera_frame_id:
                return _extract_transform(entry)
            if parent == camera_frame_id and child == body_frame_id:
                rotation, translation = _extract_transform(entry)
                return _invert_transform(rotation, translation)
        return None

    # 静态 TF 优先；没有命中时回退到全部 TF（动态外参仍然合法）。
    resolved = _find(static_entries)
    if resolved is not None:
        return resolved
    return _find(tf_entries)


def _camera_centers_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    """由 Omega extrinsic（world→camera，3×4）计算相机光心在世界系的位置。"""
    if extrinsic.ndim == 2:
        extrinsic = extrinsic[None, :, :]
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError("extrinsic must have shape (N,3,4)")
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    # camera = R @ world + t  =>  world = R^T @ (camera - t)；
    # 光心 camera=(0,0,0) 时 world = -R^T @ t
    return -np.einsum("nij,nj->ni", np.transpose(rotation, (0, 2, 1)), translation)


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


def _project_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a 3x3 matrix to the nearest proper rotation."""
    u, _singular, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[2, 2] = -1.0
    return u @ correction @ vt


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _pose_constrained_sim3(
    source_positions: np.ndarray,
    source_world_to_camera: np.ndarray,
    target_positions: np.ndarray,
    target_camera_to_world: np.ndarray,
    *,
    min_reference_baseline_m: float,
) -> tuple[float, np.ndarray, np.ndarray, float, float, float]:
    """Fit model->odom Sim(3), taking rotation from matched camera attitudes."""
    rotations = np.matmul(target_camera_to_world, source_world_to_camera)
    rotation = _project_rotation(np.sum(rotations, axis=0))
    angular_errors = np.asarray(
        [_rotation_angle_deg(rotation.T @ candidate) for candidate in rotations]
    )
    orientation_residual = float(np.sqrt(np.mean(angular_errors * angular_errors)))

    target_centered = target_positions - np.mean(target_positions, axis=0)
    rotated_source = (rotation @ source_positions.T).T
    source_centered = rotated_source - np.mean(rotated_source, axis=0)
    reference_baseline = float(
        np.max(np.linalg.norm(target_positions - target_positions[0], axis=1))
    )
    if reference_baseline < float(min_reference_baseline_m):
        raise ValueError(
            f"reference trajectory baseline {reference_baseline:.6g} m is below "
            f"{float(min_reference_baseline_m):.6g} m"
        )
    denominator = float(np.sum(source_centered * source_centered))
    if denominator <= 1e-12:
        raise ValueError("model trajectory is degenerate")
    scale = float(np.sum(source_centered * target_centered) / denominator)
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("pose-constrained Sim(3) scale is not finite/positive")
    translation = np.mean(target_positions, axis=0) - scale * np.mean(
        rotated_source, axis=0
    )
    transformed = scale * rotated_source + translation
    residual = float(
        np.sqrt(np.mean(np.sum((transformed - target_positions) ** 2, axis=1)))
    )
    if not np.isfinite(residual):
        raise ValueError("pose-constrained Sim(3) residual is not finite")
    return (
        scale,
        rotation,
        translation,
        residual,
        reference_baseline,
        orientation_residual,
    )




def _fallback(reason: str) -> OdomLocalAlignment:
    return OdomLocalAlignment(
        coordinate_state=COORDINATE_MODEL_LOCAL_UNALIGNED,
        reference_frame_id="",
        correspondence_count=0,
        scale=None,
        rotation=None,
        translation=None,
        residual_rms=None,
        reason=reason,
    )


def align_window(
    snapshot: Any,
    predictions_path: Path,
    *,
    min_correspondences: int = 4,
    selected_frame_indices: Sequence[int] | None = None,
    min_reference_baseline_m: float = 0.1,
    max_orientation_residual_deg: float = 15.0,
    max_normalized_residual_rms: float = 0.25,
) -> OdomLocalAlignment:
    """执行一个窗口的可计算性检查与 Sim(3) 近似对齐。

    任何缺失/退化/非有限输入都回退 MODEL_LOCAL_UNALIGNED，绝不抛异常。

    ``selected_frame_indices`` 给出每个模型帧对应的 snapshot 帧序号
    （Omega 输入做均匀抽帧后模型帧序号与 snapshot 帧序号不再相同）。
    长度与模型帧数不一致或未提供时，退回旧行为：模型帧序号 ==
    snapshot 帧序号。
    """
    try:
        body_camera = _resolve_body_camera_transform(
            tuple(snapshot.tf_entries or ()),
            getattr(snapshot, "body_frame_id", None),
            getattr(snapshot, "camera_frame_id", None),
        )
        if body_camera is None:
            return _fallback("missing T_body_camera transform in snapshot tf.json")
        body_camera_rotation, body_camera_translation = body_camera
        (
            indices,
            reference_positions,
            reference_rotations,
            reference_frame_id,
            odom_stamps,
        ) = (
            _reference_camera_positions(
                snapshot, body_camera_rotation, body_camera_translation
            )
        )
        if odom_stamps is not None and len(odom_stamps) > 1:
            previous = None
            for stamp in odom_stamps:
                if previous is not None and stamp < previous:
                    return _fallback("odometry stamp reset inside window")
                previous = stamp
        if not predictions_path.is_file():
            return _fallback(f"missing predictions.npz: {predictions_path}")
        try:
            with np.load(str(predictions_path), allow_pickle=False) as archive:
                if "extrinsic" not in archive:
                    return _fallback("predictions.npz has no extrinsic")
                extrinsic = archive["extrinsic"]
        except Exception as error:
            return _fallback(f"predictions.npz unreadable: {type(error).__name__}")
        model_positions = _camera_centers_from_extrinsic(extrinsic)
        if model_positions.shape[0] < min_correspondences:
            return _fallback(
                "not enough model frames for Sim(3) correspondences"
            )
        if (
            selected_frame_indices is not None
            and len(selected_frame_indices) == model_positions.shape[0]
        ):
            model_indices = selected_frame_indices
        else:
            # 旧调用路径没有选中帧清单，只能假设模型帧序号 == snapshot 帧序号。
            model_indices = range(model_positions.shape[0])
        valid = []
        model_rotations = np.asarray(extrinsic, dtype=np.float64)[:, :3, :3]
        for position, model_rotation, snapshot_index in zip(
            model_positions, model_rotations, model_indices
        ):
            matches = np.where(indices == snapshot_index)[0]
            if matches.size == 1:
                valid.append(
                    (
                        position,
                        model_rotation,
                        reference_positions[matches[0]],
                        reference_rotations[matches[0]],
                    )
                )
        if len(valid) < min_correspondences:
            return _fallback(
                f"only {len(valid)} frame(s) have both model and odom camera poses"
            )
        source = np.asarray([item[0] for item in valid], dtype=np.float64)
        source_rotations = np.asarray([item[1] for item in valid], dtype=np.float64)
        target = np.asarray([item[2] for item in valid], dtype=np.float64)
        target_rotations = np.asarray([item[3] for item in valid], dtype=np.float64)
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            return _fallback("model or odom trajectory contains non-finite values")
        (
            scale,
            rotation,
            translation,
            residual,
            reference_baseline,
            orientation_residual,
        ) = _pose_constrained_sim3(
            source,
            source_rotations,
            target,
            target_rotations,
            min_reference_baseline_m=min_reference_baseline_m,
        )
        normalized_residual = residual / reference_baseline
        if orientation_residual > float(max_orientation_residual_deg):
            return _fallback(
                f"camera orientation residual {orientation_residual:.3f} deg exceeds "
                f"{float(max_orientation_residual_deg):.3f} deg"
            )
        if normalized_residual > float(max_normalized_residual_rms):
            return _fallback(
                f"normalized trajectory residual {normalized_residual:.6g} exceeds "
                f"{float(max_normalized_residual_rms):.6g}"
            )
        reference_positions_all = reference_positions
        return OdomLocalAlignment(
            coordinate_state=COORDINATE_ODOM_LOCAL_APPROX,
            reference_frame_id=reference_frame_id,
            correspondence_count=len(valid),
            scale=scale,
            rotation=rotation.tolist(),
            translation=translation.tolist(),
            residual_rms=residual,
            reason="",
            reference_first_position=tuple(
                float(item) for item in reference_positions_all[0]
            ),
            reference_last_position=tuple(
                float(item) for item in reference_positions_all[-1]
            ),
            reference_first_stamp=(
                tuple(odom_stamps[0]) if odom_stamps else None
            ),
            reference_last_stamp=(
                tuple(odom_stamps[-1]) if odom_stamps else None
            ),
            reference_baseline_m=reference_baseline,
            reference_samples=tuple(
                {"stamp": list(stamp), "position": position.tolist()}
                for stamp, position in zip(odom_stamps, reference_positions_all)
            ),
            orientation_residual_deg=orientation_residual,
            normalized_residual_rms=normalized_residual,
        )
    except Exception as error:
        return _fallback(
            f"alignment failed: {type(error).__name__}: {error}"
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
