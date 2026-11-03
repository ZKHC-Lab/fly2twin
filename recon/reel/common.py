# SPDX-License-Identifier: Apache-2.0
"""Specification paths and external-program boundary."""
import copy
import json
import shutil
from pathlib import Path


def require_ffmpeg():
    path = shutil.which('ffmpeg')
    if not path:
        raise RuntimeError('ffmpeg is required; install it and make it available on PATH')
    return path


def specification(value):
    """JSON paths resolve relative to their file; dict paths relative to cwd."""
    if isinstance(value, dict) or isinstance(value, list):
        return copy.deepcopy(value), Path.cwd()
    path = Path(value).resolve()
    return json.loads(path.read_text()), path.parent


def resolve(base, path):
    return str((base / Path(path).expanduser()).resolve())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
