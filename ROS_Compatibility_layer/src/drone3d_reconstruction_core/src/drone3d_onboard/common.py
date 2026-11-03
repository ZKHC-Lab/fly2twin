# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import hashlib
import json
from functools import lru_cache


def load_json(path):
    return json.loads(Path(path).read_text())


def load_manifest(path):
    manifest = load_json(path)
    base = Path(path).resolve().parent
    for row in manifest['frames']:
        row['path'] = str((base / Path(row['path']).expanduser()).resolve())
    return manifest


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def image_loader(paths):
    @lru_cache(maxsize=32)
    def image(frame, width, height):
        import cv2
        path = paths[str(frame)]
        rgb = cv2.imread(str(path))
        if rgb is None:
            raise ValueError(f'Cannot decode image: {path}')
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return image
