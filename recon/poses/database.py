# SPDX-License-Identifier: Apache-2.0
import numpy as np

MAX_IMAGE_ID = 2147483647


def points(db):
    result = {}
    for image_id, rows, cols, blob in db.execute('SELECT image_id,rows,cols,data FROM keypoints'):
        xy = np.frombuffer(blob or b'', np.float32).reshape(rows, cols)[:, :2].copy()
        if cols not in (2, 4, 6) or not np.isfinite(xy).all():
            raise ValueError(f'Invalid keypoints for image {image_id}')
        result[image_id] = xy
    return result


def matches(db, table, keypoints):
    result = {}
    for pair_id, rows, cols, blob in db.execute(f'SELECT pair_id,rows,cols,data FROM {table}'):
        a, b = divmod(pair_id, MAX_IMAGE_ID)
        arr = np.frombuffer(blob or b'', np.uint32).reshape(rows, cols)
        if cols != 2 or (rows and ((arr[:, 0] >= len(keypoints[a])).any() or (arr[:, 1] >= len(keypoints[b])).any())):
            raise ValueError(f'Invalid indices in {table}, pair {pair_id}')
        result[pair_id] = arr
    return result


def combine(sift, xfeat, id_map, offsets):
    combined = dict(sift)
    reversed_pairs = 0
    for pair_id, values in xfeat.items():
        x, y = divmod(pair_id, MAX_IMAGE_ID)
        a, b = id_map[x], id_map[y]
        shifted = values + np.array([offsets[a], offsets[b]], np.uint32)
        if a > b:
            a, b = b, a
            shifted = shifted[:, ::-1]
            reversed_pairs += 1
        target = a * MAX_IMAGE_ID + b
        combined[target] = np.concatenate((sift.get(target, np.empty((0, 2), np.uint32)), shifted))
    return combined, reversed_pairs
