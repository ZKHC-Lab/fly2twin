"""Bounded storage accounting for continuous reconstruction."""

from __future__ import annotations

import os
from pathlib import Path
import stat


def storage_usage(
    *,
    result_root: str,
    worker_result_root: str = "",
    journal_root: str = "",
    snapshot_roots=(),
    runtime_log_root: str = "",
    ros_log_root: str = "",
) -> dict:
    """Return allocated bytes and explicit paths without following symlinks."""
    snapshots = tuple(sorted({str(value) for value in snapshot_roots if value}))
    result = {
        "storage_capture_roots": ";".join(snapshots),
        "storage_worker_root": str(worker_result_root or ""),
        "storage_formal_root": str(result_root or ""),
        "storage_frame_pool_root": (
            str(Path(result_root).parent / "frame_pool") if result_root else ""
        ),
        "storage_journal_root": str(journal_root or ""),
        "storage_runtime_log_root": str(runtime_log_root or ""),
        "storage_ros_log_root": str(ros_log_root or ""),
        "storage_capture_bytes": 0,
        "storage_worker_bytes": 0,
        "storage_formal_bytes": 0,
        "storage_frame_pool_bytes": 0,
        "storage_journal_bytes": 0,
        "storage_runtime_log_bytes": 0,
        "storage_ros_log_bytes": 0,
        "storage_scan_error": "",
    }
    errors = []

    def measure(label: str, paths) -> None:
        total = 0
        for value in paths:
            if not value:
                continue
            try:
                total += _tree_allocated_bytes(Path(value))
            except OSError as error:
                errors.append(f"{label}:{type(error).__name__}")
        result[f"storage_{label}_bytes"] = total

    measure("capture", snapshots)
    measure("worker", (worker_result_root,))
    measure("formal", (result_root,))
    measure("frame_pool", (result["storage_frame_pool_root"],))
    measure("journal", (journal_root,))
    try:
        result["storage_runtime_log_bytes"] = _direct_log_bytes(
            Path(runtime_log_root)
        ) if runtime_log_root else 0
    except OSError as error:
        errors.append(f"runtime_log:{type(error).__name__}")
    measure("ros_log", (ros_log_root,))
    result["storage_total_bytes"] = sum(
        int(result[key])
        for key in (
            "storage_capture_bytes",
            "storage_worker_bytes",
            "storage_formal_bytes",
            "storage_frame_pool_bytes",
            "storage_journal_bytes",
            "storage_runtime_log_bytes",
            "storage_ros_log_bytes",
        )
    )
    result["storage_scan_error"] = ";".join(errors)
    return result


def _tree_allocated_bytes(path: Path) -> int:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return 0
    total = int(info.st_blocks) * 512
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return total
    with os.scandir(path) as entries:
        for entry in entries:
            total += _tree_allocated_bytes(Path(entry.path))
    return total


def _direct_log_bytes(root: Path) -> int:
    total = 0
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            if not (entry.name.endswith(".log") or ".log." in entry.name):
                continue
            total += int(entry.stat(follow_symlinks=False).st_blocks) * 512
    return total


__all__ = [
    "storage_usage",
]
