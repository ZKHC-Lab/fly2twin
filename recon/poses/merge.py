# SPDX-License-Identifier: Apache-2.0
import numpy as np
import time
import sqlite3
from pathlib import Path
from recon.io.common import dump
from .database import MAX_IMAGE_ID

def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro",uri=True)

def merge(data, sift_path, xfeat_path, destination):
    from . import database as union
    HERE = destination.parent
    SIFT, DB = sift_path, destination
    import pycolmap as pc
    if DB.exists():
        raise FileExistsError(DB)
    start = time.perf_counter()
    with np.load(xfeat_path) as f:
        if not np.array_equal(f['frame_ids'], data['frame_ids']):
            raise ValueError('XFeat frame order differs from cameras.npz')
        pairs, coords, offsets, fundamentals = [f[k] for k in ('pair_indices', 'coords', 'offsets', 'F')]
    # Deduplicate on exact float32 coordinates: refinement is pair-dependent.
    pieces = [[] for _ in data['frame_ids']]
    mappings = [np.empty((b - a, 2), np.uint32) for a, b in zip(offsets[:-1], offsets[1:])]
    for k, (i, j) in enumerate(pairs):
        a, b = offsets[k:k + 2]
        pieces[i].append((k, 0, coords[a:b, :2]))
        pieces[j].append((k, 1, coords[a:b, 2:]))
    keys = []
    for chunks in pieces:
        if not chunks:
            keys.append(np.empty((0, 2), np.float32))
            continue
        unique, inverse = np.unique(np.concatenate([x[2] for x in chunks]), axis=0, return_inverse=True)
        keys.append(unique)
        begin = 0
        for pair, side, xy in chunks:
            mappings[pair][:, side] = inverse[begin:begin + len(xy)]
            begin += len(xy)
    with readonly(SIFT) as src:
        images = src.execute('SELECT image_id,name,camera_id FROM images ORDER BY image_id').fetchall()
        names = {name: i for i, name, _ in images}
        expected = data['image_names']
        if set(names) != set(expected):
            raise ValueError('SIFT database image set differs from manifest')
        image_ids = [names[name] for name in expected]
        cameras = src.execute('SELECT * FROM cameras').fetchall()
        skeys = union.points(src)
        sm, sg = [union.matches(src, table, skeys) for table in ('matches', 'two_view_geometries')]
        geometry = {row[0]: row[1:] for row in src.execute(
            'SELECT pair_id,config,F,E,H,qvec,tvec,camera1,camera2 FROM two_view_geometries')}
    xm = {}
    for (i, j), values, F in zip(pairs, mappings, fundamentals):
        a, b = image_ids[i], image_ids[j]
        if a > b:
            a, b, values, F = b, a, values[:, ::-1], F.T
        pair_id = a * MAX_IMAGE_ID + b
        xm[pair_id] = values
        geometry[pair_id] = (int(pc.TwoViewGeometryConfiguration.UNCALIBRATED),
                             np.asarray(F, np.float64).tobytes(), None, None, None, None, None, None)
    shifts = {i: len(key) for i, key in skeys.items()}
    identity = {i: i for i in image_ids}
    raw, _ = union.combine(sm, xm, identity, shifts)
    verified, _ = union.combine(sg, xm, identity, shifts)
    all_keys = {i: np.concatenate((skeys[i], key)) for i, key in zip(image_ids, keys)}
    with pc.Database.open(DB):
        pass
    with sqlite3.connect(DB) as dst:
        dst.executemany('INSERT INTO cameras VALUES (?,?,?,?,?,?)', cameras)
        dst.executemany('INSERT INTO images VALUES (?,?,?)', images)
        dst.executemany('INSERT INTO keypoints VALUES (?,?,?,?)',
                        ((i, len(key), 2, key.tobytes()) for i, key in all_keys.items()))
        dst.executemany('INSERT INTO matches VALUES (?,?,?,?)',
                        ((p, len(m), 2, m.tobytes()) for p, m in raw.items()))
        dst.executemany('INSERT INTO two_view_geometries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                        ((p, len(m), 2, m.tobytes(), *geometry[p]) for p, m in verified.items()))
    np.savez(HERE / 'mix_keypoint_offsets.npz', frame_ids=data['frame_ids'], image_ids=image_ids,
             xfeat_offsets=[shifts[i] for i in image_ids], xfeat_keypoints=list(map(len, keys)))
    dump(HERE / 'mix_database_metrics.json', dict(images=len(images),
         sift_keypoints=sum(map(len, skeys.values())), xfeat_keypoints=sum(map(len, keys)),
         sift_raw_matches=sum(map(len, sm.values())), sift_verified_matches=sum(map(len, sg.values())),
         xfeat_ransac_inliers=sum(map(len, xm.values())), merged_raw_matches=sum(map(len, raw.values())),
         merged_verified_matches=sum(map(len, verified.values())),
         merged_verified_pairs=sum(len(m) > 0 for m in verified.values()),
         rule='SIFT xy then XFeat xy, shift indices; no cross-source dedup or joint RANSAC; '
              'XFeat F/config where present, original SIFT geometry otherwise; descriptors omitted',
         seconds=time.perf_counter() - start))
