"""Publishing one reconstructed window as a formal record (S2 step 3).

``publish_window`` runs the result authenticity gate, re-validates the window's input
snapshot, computes the optional first-frame depth comparison, assembles provenance.json
and hands everything to the atomic formal-record publisher. The session calls it once
per reconstructed window; the record format is unchanged by the move.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Callable
from urllib.parse import unquote_to_bytes

from .action_types import ReconstructActionOutcome
from .continuous_session_config import ContinuousConfig, _valid_sha256
from .continuous_session_types import WindowRecord
from .depth_metrics import compute_first_frame_depth_metrics
from .formal_result import FORMAL_RECORD_SCHEMA, FORMAL_RECORD_SCHEMA_VERSION, FormalResultError
from .snapshot_validator import validate_snapshot

FORMAL_PUBLISHED = "published"
FORMAL_SKIPPED = "skipped"
FORMAL_ERROR = "error"
# 正式化期间检测到 stop_all 的专用原因：消费路径必须按 CANCELLED 处理。
_FORMAL_SKIPPED_STOPPING = "formal record publication skipped: session is stopping"
_MAX_SNAPSHOT_MANIFEST_BYTES = 16 * 1024 * 1024
_NO_SENSOR_DEPTH_REASONS = frozenset(
    (
        "sensor depth saving is disabled",
        "snapshot has no first-frame sensor depth",
    )
)


@dataclass(frozen=True)
class FormalPublication:
    """Outcome of :func:`publish_window`."""

    status: str  # FORMAL_PUBLISHED, FORMAL_SKIPPED or FORMAL_ERROR
    reason: str = ""
    result_uri: str = ""
    provenance_sha256: str = ""


def _skipped() -> FormalPublication:
    return FormalPublication(FORMAL_SKIPPED, _FORMAL_SKIPPED_STOPPING)


def _error(reason: str) -> FormalPublication:
    return FormalPublication(FORMAL_ERROR, reason)


def publish_window(
    record: WindowRecord,
    outcome: ReconstructActionOutcome,
    alignment: dict[str, Any],
    *,
    config: ContinuousConfig,
    records_root: Path,
    reconstruction_mode: str,
    result_gate: Callable[..., Any],
    result_publisher: Callable[..., Any],
    write_journal: Callable[[dict], None],
    session_state: Callable[[], tuple[str, list, bool]],
) -> FormalPublication:
    """执行真实性门禁并把通过门禁的窗口发布为正式记录。

    ``session_state`` 在会话锁内返回 (session_id, config_timeline 的副本, stopping)。
    结果是 PUBLISHED（附正式记录路径和 provenance 摘要）、SKIPPED（会话正在停止，
    窗口按取消处理）或 ERROR（附稳定失败原因）。
    """
    expected_identity = {
        "scene_id": record.scene_id,
        "snapshot_manifest_sha256": record.manifest_sha256,
        "captured_frames": record.captured_frames,
        "camera_role": record.role.role,
        "camera_topic": record.role.capture_config.image_topic,
        "camera_frame_id": record.role.capture_config.camera_frame_id,
    }
    expected_authenticity = {
        "checkpoint_sha256": config.expected_checkpoint_sha256,
        "checkpoint_path": config.expected_checkpoint_path,
        "result_root": config.expected_result_root,
        "algorithm_revision": config.expected_algorithm_revision,
        "ros_revision": config.expected_ros_revision,
    }
    try:
        gate = result_gate(
            outcome.result_manifest_uri,
            outcome.result_manifest_sha256,
            expected_identity=expected_identity,
            expected_authenticity=expected_authenticity,
        )
    except BaseException as error:
        reason = f"result gate raised: {type(error).__name__}: {error}"
        write_journal(
            {
                "event": "window_result_gate_failed",
                "session_id": session_state()[0],
                "window_index": record.window_index,
                "scene_id": record.scene_id,
                "reason": reason,
            }
        )
        return _error(f"formal result gate failed: {reason}")
    if not bool(getattr(gate, "ok", False)):
        reason = str(getattr(gate, "reason", "unknown"))
        write_journal(
            {
                "event": "window_result_gate_failed",
                "session_id": session_state()[0],
                "window_index": record.window_index,
                "scene_id": record.scene_id,
                "reason": reason,
            }
        )
        return _error(f"formal result gate failed: {reason}")
    payload = getattr(gate, "payload", None)
    manifest_bytes = getattr(gate, "manifest_bytes", b"")
    if not isinstance(payload, dict) or not isinstance(manifest_bytes, bytes):
        reason = "formal result gate returned an invalid payload"
        write_journal(
            {
                "event": "window_result_gate_failed",
                "session_id": session_state()[0],
                "window_index": record.window_index,
                "scene_id": record.scene_id,
                "reason": reason,
            }
        )
        return _error(f"formal result gate failed: {reason}")
    if not manifest_bytes:
        reason = "formal result gate returned empty manifest bytes"
        write_journal(
            {
                "event": "window_result_gate_failed",
                "session_id": session_state()[0],
                "window_index": record.window_index,
                "scene_id": record.scene_id,
                "reason": reason,
            }
        )
        return _error(f"formal result gate failed: {reason}")
    if session_state()[2]:
        return _skipped()
    try:
        snapshot = validate_snapshot(
            record.snapshot_uri,
            record.manifest_sha256,
            (record.role.capture_config.snapshot_root,),
            record.role.capture_config.snapshot_limits,
        )
        snapshot_manifest_bytes = _read_snapshot_manifest_bytes(snapshot)
    except BaseException as error:
        return _error(
            "window snapshot validation failed: "
            f"{type(error).__name__}: {error}"
        )
    snapshot_manifest_sha256 = hashlib.sha256(snapshot_manifest_bytes).hexdigest()
    if snapshot_manifest_sha256 != record.manifest_sha256:
        return _error(
            "window snapshot manifest digest mismatch after validation: "
            f"expected {record.manifest_sha256}, got {snapshot_manifest_sha256}"
        )
    depth_metrics, depth_metrics_error = _depth_metrics_window(
        record, outcome
    )
    session_id, timeline, stopping = session_state()
    if stopping:
        return _skipped()
    checkpoint = payload.get("checkpoint") or {}
    stage_timing = payload.get("stage_timing") or {}
    worker_exit = payload.get("worker_exit") or {}
    provenance = {
        "schema": FORMAL_RECORD_SCHEMA,
        "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
        "session_id": session_id,
        "window_index": int(record.window_index),
        "scene_id": record.scene_id,
        "result_manifest_uri": outcome.result_manifest_uri,
        "result_manifest_sha256": outcome.result_manifest_sha256,
        "snapshot_manifest_uri": record.snapshot_uri,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "captured_frames": int(record.captured_frames),
        "new_frame_count": int(
            getattr(record, "new_frame_count", record.captured_frames)
        ),
        "overlap_frame_count": int(
            getattr(record, "overlap_frame_count", 0)
        ),
        "window_frame_ids": list(getattr(record, "window_frame_ids", ())),
        "captured_frame_ids": list(getattr(record, "captured_frame_ids", ())),
        "capture_generation": getattr(record, "capture_generation", 0),
        "model_frame_ids": list(getattr(record, "model_frame_ids", ())),
        "camera_role": record.role.role,
        "camera_topic": record.role.capture_config.image_topic,
        "camera_frame_id": record.role.capture_config.camera_frame_id,
        "camera_name": record.role.capture_config.camera_name,
        "config_digest": record.effective.digest(),
        "effective_config": record.effective.as_dict(),
        "config_timeline": timeline,
        "checkpoint": {
            "path": checkpoint.get("path", ""),
            "sha256": expected_authenticity["checkpoint_sha256"],
        },
        "algorithm_revision": expected_authenticity["algorithm_revision"],
        "ros_revision": expected_authenticity["ros_revision"],
        "cuda_environment": {
            "worker_python": config.worker_python,
            "cuda_visible_devices": config.cuda_visible_devices,
        },
        "worker": {
            "wall_time_sec": stage_timing.get("worker_wall_time_sec"),
            "exit_returncode": worker_exit.get("returncode"),
        },
        "gate": {
            "result_manifest_digest_verified": True,
            "identity_verified": True,
            "authenticity_verified": True,
            "schema_verified": True,
        },
        # pool_* 与 confidence_summary 属于正式记录格式，per_window 下恒为 None；
        # 保留这些键，正式记录的格式和字节都不变。
        "reconstruction_mode": reconstruction_mode,
        "pool_revision": None,
        "pool_epoch": None,
        "pool_size": None,
        "pool_selected_indices": None,
        "confidence_summary": None,
        "alignment": alignment,
        "depth_metrics": depth_metrics,
        "depth_metrics_error": depth_metrics_error,
    }
    try:
        _ensure_formal_session_dir(records_root, session_id)
        final_path, provenance_sha256 = result_publisher(
            records_root,
            session_id,
            record.window_index,
            manifest_bytes,
            provenance,
            snapshot_manifest_bytes,
        )
    except BaseException as error:
        return _error(
            "formal record publication failed: "
            f"{type(error).__name__}: {error}"
        )
    if not _valid_sha256(provenance_sha256):
        return _error("formal record publication returned an invalid provenance sha256")
    return FormalPublication(FORMAL_PUBLISHED, "", str(final_path), provenance_sha256)


def _ensure_formal_session_dir(records_root: Path, session_id: str) -> Path:
    records_root.mkdir(parents=True, exist_ok=True)
    session_dir = records_root / session_id
    try:
        session_dir.mkdir()
    except FileExistsError as error:
        if not session_dir.is_dir() or session_dir.is_symlink():
            raise FormalResultError(
                f"formal session dir exists but is unsafe: {session_dir}"
            ) from error
    return session_dir


def _depth_metrics_window(
    record: WindowRecord, outcome
) -> tuple[dict | None, str | None]:
    """计算首帧传感器深度对照指标。

    返回 ``(depth_metrics, error_reason)``：
    - 计算成功：``(metrics, None)``；
    - 深度开关关闭或快照没有首帧传感器深度：``(None, None)``；
    - 深度开关打开但计算失败：``({"status": "unavailable", ...}, reason)``，
      让正式记录仍落盘 depth_metrics.json 并暴露失败原因。
    """
    if not bool(record.effective.save_sensor_depth):
        return None, None
    try:
        snapshot = validate_snapshot(
            record.snapshot_uri,
            record.manifest_sha256,
            (record.role.capture_config.snapshot_root,),
            record.role.capture_config.snapshot_limits,
        )
    except BaseException as error:
        reason = f"snapshot validation failed for depth metrics: {type(error).__name__}: {error}"
        return {"status": "unavailable", "reason": reason}, reason
    predictions_path = _predictions_path(outcome)
    if predictions_path is None:
        reason = "result manifest uri has no local predictions.npz"
        return {"status": "unavailable", "reason": reason}, reason
    metrics, compute_reason = compute_first_frame_depth_metrics(
        snapshot, predictions_path
    )
    if metrics is not None:
        return metrics, None
    if compute_reason in _NO_SENSOR_DEPTH_REASONS:
        return None, None
    reason = compute_reason or "depth metrics unavailable"
    return {"status": "unavailable", "reason": reason}, reason


def _read_snapshot_manifest_bytes(snapshot) -> bytes:
    """Read the validated snapshot manifest bytes from its canonical root."""
    root = getattr(snapshot, "canonical_root", None)
    if root is None:
        raise FormalResultError(
            "validated snapshot does not expose a canonical root"
        )
    path = Path(root) / "files_sha256.csv"
    try:
        info = os.lstat(str(path))
    except OSError as error:
        raise FormalResultError(
            "snapshot files_sha256.csv cannot be inspected"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FormalResultError(
            "snapshot files_sha256.csv is not a trusted regular file"
        )
    if info.st_size > _MAX_SNAPSHOT_MANIFEST_BYTES:
        raise FormalResultError(
            "snapshot files_sha256.csv exceeds size bound"
        )
    try:
        with open(str(path), "rb") as stream:
            data = stream.read(_MAX_SNAPSHOT_MANIFEST_BYTES + 1)
    except OSError as error:
        raise FormalResultError(
            "snapshot files_sha256.csv cannot be read"
        ) from error
    if len(data) > _MAX_SNAPSHOT_MANIFEST_BYTES:
        raise FormalResultError(
            "snapshot files_sha256.csv exceeds size bound"
        )
    return data


def _predictions_path(outcome) -> Path | None:
    try:
        if not outcome.result_manifest_uri.startswith("file://"):
            return None
        path = Path(
            unquote_to_bytes(
                outcome.result_manifest_uri[len("file://"):]
            ).decode("utf-8", "strict")
        )
        return path.parent / "predictions.npz"
    except BaseException:
        return None
