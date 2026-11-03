"""Bounded storage accounting and retention for continuous reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time
from urllib.parse import unquote, urlsplit


_SESSION_RE = re.compile(r"s[0-9a-f]{16}\Z")
_WORKER_RE = re.compile(r"cs-(s[0-9a-f]{16})-[wp][0-9]+-.+\Z")
_SNAPSHOT_RE = re.compile(
    r"cs-(s[0-9a-f]{16})-w[0-9]+-[0-9a-f]{16}\Z"
)
_LEGACY_SNAPSHOT_RE = re.compile(
    r"cs-([0-9a-f]{16})-w[0-9]+-[0-9a-f]{16}\Z"
)
_WINDOW_RE = re.compile(r"window-[0-9]{4,}\Z")
_STAGING_RE = re.compile(r"\.drone3d-result-[A-Za-z0-9_]+\Z")
_MAX_PROVENANCE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class StoragePolicy:
    max_sessions: int = 3
    frame_pool_keep_revisions: int = 2
    orphan_grace_sec: float = 86400.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_sessions, bool)
            or not isinstance(self.max_sessions, int)
            or self.max_sessions <= 0
        ):
            raise ValueError("max_sessions must be a positive integer")
        if (
            isinstance(self.frame_pool_keep_revisions, bool)
            or not isinstance(self.frame_pool_keep_revisions, int)
            or self.frame_pool_keep_revisions <= 0
        ):
            raise ValueError("frame_pool_keep_revisions must be a positive integer")
        if (
            isinstance(self.orphan_grace_sec, bool)
            or not isinstance(self.orphan_grace_sec, (int, float))
            or not math.isfinite(float(self.orphan_grace_sec))
            or float(self.orphan_grace_sec) < 0.0
        ):
            raise ValueError("orphan_grace_sec must be nonnegative and finite")


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


def prune_storage(
    *,
    result_root: str,
    worker_result_root: str,
    journal_root: str,
    snapshot_roots=(),
    policy: StoragePolicy = StoragePolicy(),
    protected_session_ids=(),
    now: float | None = None,
) -> dict:
    """Prune whole old sessions, stale staging, and unreferenced workers."""
    report = _report()
    protected = {
        str(value) for value in protected_session_ids if _SESSION_RE.fullmatch(str(value))
    }
    timestamp = time.time() if now is None else float(now)
    try:
        formal_root = _existing_real_root(Path(result_root), "result_root")
        worker_root = _existing_real_root(
            Path(worker_result_root), "worker_result_root"
        )
        journal = _existing_real_root(Path(journal_root), "journal_root")
        snapshots = tuple(
            _existing_real_root(Path(value), "snapshot_root")
            for value in sorted({str(item) for item in snapshot_roots if item})
        )
        pool_root = _optional_real_root(formal_root.parent / "frame_pool")
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
        return report

    sessions = _discover_sessions(formal_root, journal, pool_root, report)
    newest = [
        session_id
        for session_id, _mtime in sorted(
            sessions.items(), key=lambda item: (item[1], item[0]), reverse=True
        )
        if session_id not in protected
    ]
    kept = protected | set(newest[: policy.max_sessions])
    report["kept_sessions"] = sorted(kept)
    for session_id in newest[policy.max_sessions :]:
        _remove_session(
            session_id,
            formal_root,
            worker_root,
            journal,
            pool_root,
            snapshots,
            report,
        )

    for session_id in sorted(kept):
        _prune_frame_pool_revisions(
            formal_root,
            worker_root,
            session_id,
            policy.frame_pool_keep_revisions,
            report,
        )

    references = _worker_references(formal_root, worker_root)
    cutoff = timestamp - float(policy.orphan_grace_sec)
    for entry in _safe_children(worker_root, report):
        session_id = _worker_session(entry.name)
        if session_id in protected or entry in references:
            continue
        if (
            session_id is not None
            and entry.is_dir(follow_symlinks=False)
            and _old_enough(entry, cutoff)
        ):
            _remove_tree(worker_root, Path(entry.path), report)
    for root in (worker_root,) + snapshots:
        for entry in _safe_children(root, report):
            if (
                _STAGING_RE.fullmatch(entry.name)
                and entry.is_dir(follow_symlinks=False)
                and _old_enough(entry, cutoff)
            ):
                _remove_tree(root, Path(entry.path), report)
    return report


def prune_frame_pool_revisions(
    *,
    result_root: str,
    worker_result_root: str,
    session_id: str,
    keep_revisions: int,
) -> dict:
    """Keep only the newest heavy global frame-pool results for one session."""
    report = _report()
    if not _SESSION_RE.fullmatch(str(session_id)):
        report["errors"].append("invalid session_id")
        return report
    if (
        isinstance(keep_revisions, bool)
        or not isinstance(keep_revisions, int)
        or keep_revisions <= 0
    ):
        report["errors"].append("keep_revisions must be a positive integer")
        return report
    try:
        formal_root = _existing_real_root(Path(result_root), "result_root")
        worker_root = _existing_real_root(
            Path(worker_result_root), "worker_result_root"
        )
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
        return report
    _prune_frame_pool_revisions(
        formal_root, worker_root, session_id, keep_revisions, report
    )
    return report


def _report() -> dict:
    return {
        "removed": [],
        "freed_bytes": 0,
        "errors": [],
        "kept_sessions": [],
    }


def _existing_real_root(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{label} must be a real directory")
    if path.resolve(strict=True) != path:
        raise ValueError(f"{label} must be canonical")
    return path


def _optional_real_root(path: Path) -> Path | None:
    if not path.exists():
        return None
    return _existing_real_root(path, "frame_pool_root")


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


def _safe_children(root: Path, report: dict):
    try:
        with os.scandir(root) as entries:
            return list(entries)
    except OSError as error:
        report["errors"].append(f"scan {root}: {type(error).__name__}: {error}")
        return []


def _discover_sessions(
    formal_root: Path,
    journal_root: Path,
    pool_root: Path | None,
    report: dict,
) -> dict[str, float]:
    sessions: dict[str, float] = {}

    def add(session_id: str | None, path: Path) -> None:
        if session_id is None or not _SESSION_RE.fullmatch(session_id):
            return
        try:
            mtime = path.lstat().st_mtime
        except OSError:
            return
        sessions[session_id] = max(sessions.get(session_id, 0.0), mtime)

    sessions_root = formal_root / "sessions"
    if sessions_root.is_dir() and not sessions_root.is_symlink():
        for entry in _safe_children(sessions_root, report):
            if entry.is_dir(follow_symlinks=False):
                add(entry.name, Path(entry.path))
    if pool_root is not None:
        for entry in _safe_children(pool_root, report):
            if entry.is_dir(follow_symlinks=False):
                add(entry.name, Path(entry.path))
    for entry in _safe_children(journal_root, report):
        add(entry.name[:-6] if entry.name.endswith(".jsonl") else None, Path(entry.path))
    return sessions


def _worker_session(name: str) -> str | None:
    match = _WORKER_RE.fullmatch(name)
    return match.group(1) if match is not None else None


def _snapshot_session(name: str) -> str | None:
    match = _SNAPSHOT_RE.fullmatch(name)
    if match is not None:
        return match.group(1)
    match = _LEGACY_SNAPSHOT_RE.fullmatch(name)
    return "s" + match.group(1) if match is not None else None


def _remove_session(
    session_id: str,
    formal_root: Path,
    worker_root: Path,
    journal_root: Path,
    pool_root: Path | None,
    snapshot_roots,
    report: dict,
) -> None:
    sessions_root = formal_root / "sessions"
    if sessions_root.is_dir() and not sessions_root.is_symlink():
        _remove_tree(sessions_root, sessions_root / session_id, report)
    if pool_root is not None:
        _remove_tree(pool_root, pool_root / session_id, report)
    journal = journal_root / f"{session_id}.jsonl"
    _remove_file(journal_root, journal, report)
    for entry in _safe_children(worker_root, report):
        if _worker_session(entry.name) == session_id:
            _remove_tree(worker_root, Path(entry.path), report)
    for root in snapshot_roots:
        for entry in _safe_children(root, report):
            if _snapshot_session(entry.name) == session_id:
                _remove_tree(root, Path(entry.path), report)


def _read_provenance(path: Path) -> dict | None:
    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_size > _MAX_PROVENANCE_BYTES
        ):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _worker_from_provenance(payload: dict, worker_root: Path) -> Path | None:
    uri = payload.get("result_manifest_uri")
    if not isinstance(uri, str):
        return None
    split = urlsplit(uri)
    if split.scheme != "file" or split.netloc or split.query or split.fragment:
        return None
    manifest = Path(unquote(split.path))
    worker = manifest.parent
    if manifest.name != "result.json" or worker.parent != worker_root:
        return None
    if _worker_session(worker.name) is None:
        return None
    return worker


def _worker_references(formal_root: Path, worker_root: Path) -> set[Path]:
    references = set()
    sessions_root = formal_root / "sessions"
    if not sessions_root.is_dir() or sessions_root.is_symlink():
        return references
    for provenance in sessions_root.glob("*/window-*/provenance.json"):
        payload = _read_provenance(provenance)
        worker = (
            _worker_from_provenance(payload, worker_root)
            if payload is not None
            else None
        )
        if worker is not None:
            references.add(worker)
    return references


def _prune_frame_pool_revisions(
    formal_root: Path,
    worker_root: Path,
    session_id: str,
    keep_revisions: int,
    report: dict,
) -> None:
    session_root = formal_root / "sessions" / session_id
    if not session_root.is_dir() or session_root.is_symlink():
        return
    revisions = []
    for entry in _safe_children(session_root, report):
        if (
            not entry.is_dir(follow_symlinks=False)
            or not _WINDOW_RE.fullmatch(entry.name)
        ):
            continue
        window = Path(entry.path)
        payload = _read_provenance(window / "provenance.json")
        if payload is None or payload.get("reconstruction_mode") != "frame_pool":
            continue
        revision = payload.get("pool_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            continue
        worker = _worker_from_provenance(payload, worker_root)
        revisions.append((revision, window, worker))
    revisions.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    for _revision, window, worker in revisions[keep_revisions:]:
        _remove_tree(session_root, window, report)
        if worker is not None and worker not in _worker_references(formal_root, worker_root):
            _remove_tree(worker_root, worker, report)


def _old_enough(entry, cutoff: float) -> bool:
    try:
        return entry.stat(follow_symlinks=False).st_mtime <= cutoff
    except OSError:
        return False


def _remove_tree(root: Path, path: Path, report: dict) -> None:
    if not path.exists():
        return
    try:
        if path.parent != root or path.is_symlink() or not path.is_dir():
            raise ValueError("target is not a real direct child")
        root_info = root.lstat()
        for directory, names, files in os.walk(path, followlinks=False):
            directory_info = os.lstat(directory)
            if stat.S_ISLNK(directory_info.st_mode) or directory_info.st_dev != root_info.st_dev:
                raise ValueError("target crosses a symlink or mount boundary")
            for name in names + files:
                info = os.lstat(os.path.join(directory, name))
                if not stat.S_ISLNK(info.st_mode) and info.st_dev != root_info.st_dev:
                    raise ValueError("target crosses a mount boundary")
        size = _tree_allocated_bytes(path)
        shutil.rmtree(str(path))
    except (OSError, ValueError) as error:
        report["errors"].append(f"remove {path}: {type(error).__name__}: {error}")
        return
    report["removed"].append(str(path))
    report["freed_bytes"] += size


def _remove_file(root: Path, path: Path, report: dict) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        report["errors"].append(f"inspect {path}: {type(error).__name__}: {error}")
        return
    if path.parent != root or not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        report["errors"].append(f"remove {path}: unsafe file")
        return
    try:
        path.unlink()
    except OSError as error:
        report["errors"].append(f"remove {path}: {type(error).__name__}: {error}")
        return
    report["removed"].append(str(path))
    report["freed_bytes"] += int(info.st_blocks) * 512


__all__ = [
    "StoragePolicy",
    "prune_frame_pool_revisions",
    "prune_storage",
    "storage_usage",
]
