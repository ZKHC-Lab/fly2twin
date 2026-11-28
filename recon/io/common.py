# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import hashlib
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


def load_json(path):
    return json.loads(Path(path).read_text())


def load_manifest(path):
    manifest = load_json(path)
    base = Path(path).resolve().parent
    for row in manifest['frames']:
        row['path'] = str((base / Path(row['path']).expanduser()).resolve())
    return manifest


def stage_file(root, name):
    """A prepare-stage file of a run: the pose stage rewrites it when its provider drops frames, otherwise prepare's stands."""
    path = Path(root) / 'pose' / name
    return path if path.exists() else Path(root) / 'prepare' / name


def stage_manifest(root):
    return load_json(stage_file(root, 'manifest.json'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def image_loader(paths, cache_frames=32):
    @lru_cache(maxsize=cache_frames)
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


def ahead(items, fn, workers=4, depth=64):
    """Yield fn(item) for every item, in order, with up to `depth` results computed ahead in `workers` threads.

    The values and their order are those of map(fn, items); only the waiting moves off the caller.
    `items` is consumed up to `depth` ahead of what has been yielded. workers=0 calls fn inline.
    """
    if workers < 1:
        yield from map(fn, items)
        return
    with ThreadPoolExecutor(workers) as pool:
        pending = deque()
        try:
            for item in items:
                pending.append(pool.submit(fn, item))
                if len(pending) >= depth:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()
