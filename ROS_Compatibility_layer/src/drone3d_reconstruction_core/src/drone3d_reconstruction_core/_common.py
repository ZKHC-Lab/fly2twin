"""Small helpers shared by the onboard modules (internal; no ROS, no NumPy).

Each function replaces copies that behaved identically; variants that differ in a
detail (exception type, accepted inputs, open flags) stay where they are.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import signal
from typing import Any


def sha256_file(path) -> str:
    """Streamed SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path) -> None:
    """Persist a directory's entries (after a create, rename or unlink in it)."""
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def is_relative_to(path: Path, root: Path) -> bool:
    """``Path.is_relative_to`` for Python 3.8."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def reject_duplicate_json_fields(pairs):
    """``json`` object_pairs_hook that refuses a repeated key."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key!r}")
        result[key] = value
    return result


def signal_group(process_group: int, sig: signal.Signals) -> None:
    """Signal a process group; a group that no longer exists is ignored."""
    try:
        os.killpg(process_group, sig)
    except ProcessLookupError:
        pass


def is_finite_number(value: Any) -> bool:
    """True for a finite int or float; bool and every other type are not numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def converts_to_finite(value: Any) -> bool:
    """True when ``float(value)`` succeeds and is finite (bool and numeric strings pass)."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False
