"""Bounded storage accounting and retention for continuous reconstruction.

Retention removes whole old sessions, but only sessions whose delivery package was
exported and still verifies (see :func:`export_marker_path`). Unexported sessions are
kept and reported: deleting a flight's source data is not recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import time
from urllib.parse import unquote, urlsplit
from ._common import sha256_file as _file_sha256, fsync_directory as _sync_dir


_SESSION_RE = re.compile(r"s[0-9a-f]{16}\Z")
_WORKER_RE = re.compile(r"cs-(s[0-9a-f]{16})-[wp][0-9]+-.+\Z")
_SNAPSHOT_RE = re.compile(r"cs-(s[0-9a-f]{16})-w[0-9]+-[0-9a-f]{16}\Z")
_STAGING_RE = re.compile(r"\.drone3d-result-[A-Za-z0-9_]+\Z")
_WINDOW_DIR_RE = re.compile(r"window-([0-9]{4,})\Z")
# A formal-record or map-backend directory being deleted is first renamed to this name;
# readers only accept s<16 hex>, so the session disappears from them in one step.
_DELETING_PREFIX = ".deleting-"
_MAX_PROVENANCE_BYTES = 1024 * 1024
_MAX_MARKER_BYTES = 64 * 1024
EXPORT_MARKER_SCHEMA = "drone3d.onboard.export_marker"


@dataclass(frozen=True)
class StoragePolicy:
    max_sessions: int = 3
    orphan_grace_sec: float = 86400.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_sessions, bool)
            or not isinstance(self.max_sessions, int)
            or self.max_sessions <= 0
        ):
            raise ValueError("max_sessions must be a positive integer")
        if (
            isinstance(self.orphan_grace_sec, bool)
            or not isinstance(self.orphan_grace_sec, (int, float))
            or not math.isfinite(float(self.orphan_grace_sec))
            or float(self.orphan_grace_sec) < 0.0
        ):
            raise ValueError("orphan_grace_sec must be nonnegative and finite")


def export_marker_path(result_root, session_id: str) -> Path:
    """``<result_root>/exports/<session_id>.json``: written by drone3d_onboard.export
    after a delivery's manifest.json is published, read only by :func:`prune_storage`."""
    return Path(result_root) / "exports" / f"{session_id}.json"


def formal_window_digests(result_root, session_id: str) -> dict[str, str]:
    """Window index -> SHA-256 of its provenance.json, for the session's live formal records."""
    session_dir = Path(result_root) / "sessions" / session_id
    digests = {}
    for entry in sorted(session_dir.iterdir()) if session_dir.is_dir() else ():
        match = _WINDOW_DIR_RE.fullmatch(entry.name)
        if match is not None and entry.is_dir() and not entry.is_symlink():
            digests[str(int(match.group(1)))] = _file_sha256(entry / "provenance.json")
    return digests


def window_set_digest(windows: dict) -> str:
    """SHA-256 of a window set {index: provenance SHA-256} in one canonical form (sorted by index)."""
    canonical = json.dumps(
        [[int(index), str(digest)] for index, digest in sorted(windows.items(), key=lambda item: int(item[0]))],
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_export_marker(result_root, session_id: str, delivery, windows: dict) -> Path:
    """Record that ``delivery`` (with its published manifest.json) holds the session's
    formal windows ``windows`` (window index -> provenance.json SHA-256, as exported)."""
    delivery = Path(delivery).resolve()
    manifest = delivery / "manifest.json"
    payload = {
        "schema": EXPORT_MARKER_SCHEMA,
        "schema_version": 2,
        "session_id": session_id,
        "delivery": str(delivery),
        "manifest_sha256": _file_sha256(manifest),
        # A fixed-size digest of the window set, so the marker stays small for any session length.
        "window_count": len(windows),
        "windows_sha256": window_set_digest(windows),
        "exported_unix": time.time(),
    }
    path = export_marker_path(result_root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_dir(path.parent)
    return path


def export_verification(result_root, session_id: str) -> str:
    """Why the session may not be deleted yet; an empty string when it may.

    All of these must hold: the marker names this session; its delivery's manifest.json
    is the one recorded, parses, names this session, and every file listed under
    ``hashes`` is still present with that SHA-256; and, while the session's formal records
    are still live, they are exactly the windows the export saw (a window published after
    the export was never delivered).
    """
    path = export_marker_path(result_root, session_id)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return "not exported"
    except OSError as error:
        return f"export marker unreadable: {type(error).__name__}"
    try:
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_MARKER_BYTES:
            return "export marker is not a regular file of bounded size"
        marker = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict) or marker.get("schema") != EXPORT_MARKER_SCHEMA:
            return "export marker has an unknown schema"
        if marker.get("session_id") != session_id:
            return "export marker names another session"
        windows_sha256 = marker.get("windows_sha256")
        if marker.get("schema_version") != 2 or not isinstance(windows_sha256, str) or not marker.get("window_count"):
            return "export marker lists no exported windows"
        delivery = Path(str(marker.get("delivery", "")))
        reason = _delivery_problem(delivery, session_id, marker.get("manifest_sha256"))
        if reason:
            return reason
        if (Path(result_root) / "sessions" / session_id).is_dir():
            if window_set_digest(formal_window_digests(result_root, session_id)) != windows_sha256:
                return "formal windows differ from the exported ones"
    except (OSError, UnicodeError, ValueError) as error:
        return f"export cannot be verified: {type(error).__name__}"
    return ""


def verified_export(result_root, session_id: str) -> bool:
    """True when :func:`export_verification` finds nothing that blocks deletion."""
    return export_verification(result_root, session_id) == ""


def _delivery_problem(delivery: Path, session_id: str, manifest_sha256) -> str:
    manifest = delivery / "manifest.json"
    if not delivery.is_absolute() or manifest.is_symlink() or not manifest.is_file():
        return "delivery manifest.json is missing"
    if _file_sha256(manifest) != manifest_sha256:
        return "delivery manifest.json changed since the export"
    content = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(content, dict) or content.get("session_id") != session_id:
        return "delivery manifest.json names another session"
    hashes = content.get("hashes")
    if not isinstance(hashes, dict) or not hashes:
        return "delivery manifest.json lists no files"
    root = delivery.resolve()
    for relative, digest in hashes.items():
        target = (root / str(relative)).resolve()
        if root not in target.parents or target.is_symlink() or not target.is_file():
            return f"delivery file {relative} is missing"
        if _file_sha256(target) != digest:
            return f"delivery file {relative} changed"
    return ""


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
    remove_staging: bool = False,
    now: float | None = None,
) -> dict:
    """Remove old exported sessions, stale staging and unreferenced old workers.

    The ``policy.max_sessions`` newest sessions are always kept. An older session is
    removed only when it is not protected, its map backend does not hold the writer
    lock, and :func:`export_verification` finds nothing; otherwise it is kept, and reported
    under ``kept_unexported`` (with the reason in ``kept_reasons``) when the export is what
    is missing, incomplete or no longer covers the session.
    """
    report = {"removed": [], "freed_bytes": 0, "errors": [], "kept_sessions": [],
              "kept_unexported": [], "kept_locked": [], "kept_reasons": {}, "kept_staging": 0}
    protected = {str(v) for v in protected_session_ids if _SESSION_RE.fullmatch(str(v))}
    timestamp = time.time() if now is None else float(now)
    try:
        formal_root = _existing_real_root(Path(result_root), "result_root")
        worker_root = _existing_real_root(Path(worker_result_root), "worker_result_root")
        journal = _existing_real_root(Path(journal_root), "journal_root")
        snapshots = tuple(
            _existing_real_root(Path(value), "snapshot_root")
            for value in sorted({str(item) for item in snapshot_roots if item})
        )
        pool_root = _optional_real_root(formal_root.parent / "frame_pool")
    except (OSError, ValueError) as error:
        report["errors"].append(str(error))
        return report
    roots = _SessionRoots(formal_root, worker_root, journal, pool_root, snapshots)
    nested = _nested_roots(roots)
    if nested:
        # Deleting a session's directory under one root could take another root (and other
        # sessions' data, or a delivery) with it. Nothing is deleted until the layout is fixed.
        report["errors"].append("storage roots are nested, cleanup skipped: " + nested)
        return report

    sessions = _discover_sessions(roots, report)
    # Every delivery an export marker points at, read before anything is deleted.
    deliveries = _known_deliveries(formal_root)
    newest = [
        session_id
        for session_id, _mtime in sorted(
            sessions.items(), key=lambda item: (item[1], item[0]), reverse=True
        )
        if session_id not in protected
    ]
    kept = protected | set(newest[: policy.max_sessions])
    for session_id in newest[policy.max_sessions :]:
        reason = (
            export_verification(formal_root, session_id)
            or _delivery_inside_session(formal_root, session_id, roots)
            or _delivery_in_reclaimable_storage(deliveries.get(session_id), roots)
            or _holds_other_delivery(session_id, deliveries, roots)
        )
        if reason:
            report["kept_unexported"].append(session_id)
            report["kept_reasons"][session_id] = reason
            kept.add(session_id)
            continue
        try:
            lock = _map_backend_lock(formal_root / "g4" / session_id)
        except OSError as error:
            report["errors"].append(f"lock {session_id}: {type(error).__name__}: {error}")
            lock = False
        if lock is False:
            report["kept_locked"].append(session_id)
            kept.add(session_id)
            continue
        try:
            if not _remove_session(session_id, roots, report):
                # Aborted: nothing of this session may go this time, including the
                # orphan worker scan below.
                kept.add(session_id)
        finally:
            if lock is not None:
                lock.close()
    report["kept_sessions"] = sorted(kept)
    report["kept_unexported"].sort()

    references = _worker_references(formal_root, worker_root)
    cutoff = timestamp - float(policy.orphan_grace_sec)
    for entry in _safe_children(worker_root, report):
        session_id = _worker_session(entry.name)
        if session_id is None or session_id in kept or Path(entry.path) in references:
            continue
        if entry.is_dir(follow_symlinks=False) and _old_enough(entry, cutoff):
            _remove_tree(worker_root, Path(entry.path), report)
    # A staging directory's name does not say which session it belongs to, and an active
    # capture or reconstruction writes into one. They are removed only when the caller knows
    # no session is running (node start); otherwise they are counted and left alone.
    for root in (worker_root,) + snapshots:
        for entry in _safe_children(root, report):
            if not (_STAGING_RE.fullmatch(entry.name) and entry.is_dir(follow_symlinks=False)):
                continue
            if remove_staging and _old_enough(entry, cutoff):
                _remove_tree(root, Path(entry.path), report)
            else:
                report["kept_staging"] += 1
    return report


def _nested_roots(roots: "_SessionRoots") -> str:
    """Describe the first pair of storage roots where one contains the other ('' if none)."""
    named = [("result_root", roots.formal), ("worker_result_root", roots.worker),
             ("journal_root", roots.journal)]
    if roots.pool is not None:
        named.append(("frame_pool", roots.pool))
    named += [("snapshot_root", root) for root in roots.snapshots]
    for index, (name_a, a) in enumerate(named):
        for name_b, b in named[index + 1:]:
            if a == b or a in b.parents or b in a.parents:
                return f"{name_a} {a} and {name_b} {b}"
    return ""


def _session_targets(session_id: str, roots: "_SessionRoots") -> list[Path]:
    """Every directory or file :func:`_remove_session` would delete for the session."""
    targets = [roots.formal / "sessions" / session_id, roots.formal / "g4" / session_id,
               roots.journal / f"{session_id}.jsonl", export_marker_path(roots.formal, session_id)]
    if roots.pool is not None:
        targets.append(roots.pool / session_id)
    for root in roots.snapshots:
        targets += [root / name for name in _child_names(root) if _snapshot_session(name) == session_id]
    targets += [roots.worker / name for name in _child_names(roots.worker)
                if _worker_session(name) == session_id]
    return targets


def _child_names(root: Path) -> list[str]:
    try:
        return os.listdir(root)
    except OSError:
        return []


def _known_deliveries(result_root: Path) -> dict[str, Path]:
    """Session id -> resolved delivery directory, for every readable export marker."""
    found = {}
    for name in _child_names(Path(result_root) / "exports"):
        session_id = name[:-len(".json")] if name.endswith(".json") else ""
        if not _SESSION_RE.fullmatch(session_id):
            continue
        try:
            path = export_marker_path(result_root, session_id)
            if path.is_symlink() or path.stat().st_size > _MAX_MARKER_BYTES:
                continue
            marker = json.loads(path.read_text(encoding="utf-8"))
            found[session_id] = Path(str(marker["delivery"])).resolve()
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            continue
    return found


def _reclaimable_roots(roots: "_SessionRoots") -> list[Path]:
    """Directories whose contents automatic retention deletes, session by session."""
    found = [roots.formal / "sessions", roots.formal / "g4", roots.worker, roots.journal]
    if roots.pool is not None:
        found.append(roots.pool)
    return found + list(roots.snapshots)


def _delivery_in_reclaimable_storage(delivery: Path | None, roots: "_SessionRoots") -> str:
    """A delivery inside reclaimable storage could go with another session; keep its source."""
    if delivery is None:
        return ""
    for root in _reclaimable_roots(roots):
        root = root.resolve()
        if delivery == root or root in delivery.parents:
            return f"delivery {delivery} lies inside reclaimable storage {root}"
    return ""


def _holds_other_delivery(session_id: str, deliveries: dict, roots: "_SessionRoots") -> str:
    """Reason to keep a session whose data contains another session's delivery ('' if none)."""
    targets = [target.resolve() for target in _session_targets(session_id, roots)]
    for owner, delivery in sorted(deliveries.items()):
        if owner == session_id:
            continue
        for target in targets:
            if delivery == target or target in delivery.parents:
                return f"holds the delivery of session {owner} ({delivery})"
    return ""


def _delivery_inside_session(result_root: Path, session_id: str, roots: "_SessionRoots") -> str:
    """Reason to keep a session whose delivery lies inside its own data ('' if it does not)."""
    marker = json.loads(export_marker_path(result_root, session_id).read_text(encoding="utf-8"))
    delivery = Path(str(marker["delivery"])).resolve()
    for target in _session_targets(session_id, roots):
        resolved = target.resolve()
        if delivery == resolved or resolved in delivery.parents:
            return f"delivery {delivery} lies inside the session's own data {target}"
    return ""


@dataclass(frozen=True)
class _SessionRoots:
    formal: Path
    worker: Path
    journal: Path
    pool: Path | None
    snapshots: tuple


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


def _safe_children(root: Path | None, report: dict):
    if root is None or not root.is_dir() or root.is_symlink():
        return []
    try:
        with os.scandir(root) as entries:
            return list(entries)
    except OSError as error:
        report["errors"].append(f"scan {root}: {type(error).__name__}: {error}")
        return []


def _session_from_name(name: str) -> str | None:
    if name.startswith(_DELETING_PREFIX):
        name = name[len(_DELETING_PREFIX):]
    return name if _SESSION_RE.fullmatch(name) else None


def _discover_sessions(roots: _SessionRoots, report: dict) -> dict[str, float]:
    """Session id -> newest mtime of its formal records, frame pool and journal.

    Sessions seen only through what a removal leaves behind (``.deleting-`` directories,
    map-backend output, export markers) are found too and sort as oldest, so an
    interrupted removal is finished by the next run.
    """
    sessions: dict[str, float] = {}

    def add(session_id: str | None, path: Path | None) -> None:
        if session_id is None:
            return
        mtime = 0.0
        if path is not None:
            try:
                mtime = path.lstat().st_mtime
            except OSError:
                return
        sessions[session_id] = max(sessions.get(session_id, 0.0), mtime)

    for entry in _safe_children(roots.formal / "sessions", report):
        if entry.is_dir(follow_symlinks=False):
            dated = not entry.name.startswith(_DELETING_PREFIX)
            add(_session_from_name(entry.name), Path(entry.path) if dated else None)
    for entry in _safe_children(roots.pool, report):
        if entry.is_dir(follow_symlinks=False):
            add(_session_from_name(entry.name), Path(entry.path))
    for entry in _safe_children(roots.journal, report):
        if entry.name.endswith(".jsonl"):
            add(_session_from_name(entry.name[:-6]), Path(entry.path))
    for entry in _safe_children(roots.formal / "g4", report):
        if entry.is_dir(follow_symlinks=False):
            add(_session_from_name(entry.name), None)
    for entry in _safe_children(roots.formal / "exports", report):
        if entry.name.endswith(".json"):
            add(_session_from_name(entry.name[:-5]), None)
    return sessions


def _map_backend_lock(output_root: Path):
    """Take the map backend's writer lock without blocking.

    Returns None when there is no backend output, False when another process holds the
    lock, otherwise the open lock file (held until the caller closes it).
    """
    lock_path = output_root / ".writer.lock"
    if not lock_path.is_file():
        return None
    stream = open(lock_path, "a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        if error.errno in (errno.EAGAIN, errno.EACCES):
            return False
        raise
    return stream


def _remove_session(session_id: str, roots: _SessionRoots, report: dict) -> bool:
    """Remove all six kinds of session data; the export marker goes last.

    Formal records and the map-backend output are first renamed out of the readers' sight.
    Only when both entry points are hidden is anything deleted, so a power cut never leaves
    a session that looks complete but has lost its snapshots or predictions; if either
    rename fails, the other is undone and nothing of this session is deleted this time.
    Everything left over is found again by the next run while the marker remains.
    Returns False when the removal was aborted before deleting anything, True otherwise.
    """
    parents = (roots.formal / "sessions", roots.formal / "g4")
    hidden = []
    for parent in parents:
        result = _hide(parent, session_id, report)
        if result is False:
            for done in reversed(hidden):
                _unhide(done, session_id, report)
            report["errors"].append(f"{session_id}: not removed, its records could not be hidden")
            return False
        if result is not None:
            hidden.append(result)
    errors_before = len(report["errors"])
    for parent in parents:
        _remove_tree(parent, parent / (_DELETING_PREFIX + session_id), report)
    for root in roots.snapshots:
        for entry in _safe_children(root, report):
            if _snapshot_session(entry.name) == session_id:
                _remove_tree(root, Path(entry.path), report)
    for entry in _safe_children(roots.worker, report):
        if _worker_session(entry.name) == session_id:
            _remove_tree(roots.worker, Path(entry.path), report)
    if roots.pool is not None:
        _remove_tree(roots.pool, roots.pool / session_id, report)
    _remove_file(roots.journal, roots.journal / f"{session_id}.jsonl", report)
    if len(report["errors"]) == errors_before:
        exports = roots.formal / "exports"
        _remove_file(exports, exports / f"{session_id}.json", report)
    return True


def _hide(parent: Path, session_id: str, report: dict):
    """Rename ``parent/<sid>`` to ``parent/.deleting-<sid>``.

    Returns the parent when it renamed, None when there was nothing live to hide (absent,
    or hidden by an interrupted earlier run), False when the rename failed.
    """
    if not parent.is_dir() or parent.is_symlink():
        return None
    live, hidden = parent / session_id, parent / (_DELETING_PREFIX + session_id)
    if not live.exists() and not live.is_symlink():
        return None
    try:
        if live.is_symlink() or not live.is_dir():
            raise ValueError(f"{live} is not a real directory")
        if hidden.exists() or hidden.is_symlink():
            raise ValueError(f"{hidden} already exists")
        os.rename(live, hidden)
        _sync_dir(parent)
    except (OSError, ValueError) as error:
        report["errors"].append(f"hide {live}: {type(error).__name__}: {error}")
        return False
    return parent


def _unhide(parent: Path, session_id: str, report: dict) -> None:
    try:
        os.rename(parent / (_DELETING_PREFIX + session_id), parent / session_id)
        _sync_dir(parent)
    except OSError as error:
        report["errors"].append(f"unhide {parent / session_id}: {type(error).__name__}: {error}")


def _worker_session(name: str) -> str | None:
    match = _WORKER_RE.fullmatch(name)
    return match.group(1) if match is not None else None


def _snapshot_session(name: str) -> str | None:
    match = _SNAPSHOT_RE.fullmatch(name)
    return match.group(1) if match is not None else None


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
        worker = _worker_from_provenance(payload, worker_root) if payload is not None else None
        if worker is not None:
            references.add(worker)
    return references


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
    if path.parent != root or not stat.S_ISREG(info.st_mode):
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
    "export_marker_path",
    "export_verification",
    "formal_window_digests",
    "window_set_digest",
    "prune_storage",
    "storage_usage",
    "verified_export",
    "write_export_marker",
]
