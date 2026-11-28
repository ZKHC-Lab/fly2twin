# SPDX-License-Identifier: Apache-2.0
import numpy as np
from recon.io.common import dump
from .pairs import plan_pairs
from .database import MAX_IMAGE_ID


def stage(name, start):
    """Print one JSON timing row (seconds, process peak RSS in MiB) and return it."""
    import json, resource, time

    row = dict(
        stage=name,
        seconds=time.perf_counter() - start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    print(json.dumps(row), flush=True)
    return row


def number_by_name(db):
    """Renumber images in name order. Parallel extraction writes images, and so assigns ids, in completion order."""
    import sqlite3

    with sqlite3.connect(db) as con:
        if (
            con.execute('SELECT COUNT(*) FROM frame_data WHERE frame_id!=data_id').fetchone()[0]
            or con.execute('SELECT COUNT(*) FROM pose_priors').fetchone()[0]
        ):
            raise ValueError('Expected one frame per image with frame_id == image_id and no pose priors')
        rows = con.execute('SELECT image_id FROM images ORDER BY name').fetchall()
        shift = 1 + max((old for (old,) in rows), default=0)
        moves = [(old, new) for new, (old,) in enumerate(rows, 1)]
        for table, columns in (
            ('images', ('image_id',)),
            ('keypoints', ('image_id',)),
            ('descriptors', ('image_id',)),
            ('frame_data', ('frame_id', 'data_id')),
        ):
            for column in columns:  # through a free id range, so no step collides with an existing id
                con.executemany(
                    f'UPDATE {table} SET {column}=? WHERE {column}=?', ((old + shift, old) for old, _ in moves)
                )
                con.executemany(
                    f'UPDATE {table} SET {column}=? WHERE {column}=?', ((new, old + shift) for old, new in moves)
                )


def reconstruct(data, images, out, settings):
    import time
    import pycolmap as pc

    tick = time.perf_counter()
    from .xfeat import match
    from .merge import merge
    from .ba import build

    data = dict(data, c2w_L=data['c2w'], K_L=data['K'])
    if not np.allclose(data['K'], data['K'][0]):
        raise ValueError('Shared-camera BA requires shared K')
    pairs = plan_pairs(data['c2w_L'])
    np.savez(out / 'candidate_pairs.npz', frame_ids=data['frame_ids'], pair_indices=pairs)
    K = data['K'][0]
    reader = pc.ImageReaderOptions(
        camera_model='PINHOLE', camera_params=','.join(map(str, [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]))
    )
    db = out / 'sift.db'
    pc.extract_features(
        db,
        images,
        image_names=data['image_names'],
        camera_mode=pc.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=pc.FeatureExtractionOptions(num_threads=4, use_gpu=False),
        device=pc.Device.cpu,
    )
    number_by_name(db)
    stage('sift_extraction', tick)
    tick = time.perf_counter()
    # One matcher thread: concurrent FAISS matchers (pycolmap 4.2) lose matches on a few dozen pairs, differently every run.
    matching = pc.FeatureMatchingOptions(num_threads=1, use_gpu=False)
    # Unseeded RANSAC continues each worker's thread-local PRNG, so a pair's result depends on scheduling.
    verification = pc.TwoViewGeometryOptions()
    verification.ransac.num_threads = 1
    verification.ransac.random_seed = 0
    pc.match_sequential(
        db,
        matching_options=matching,
        pairing_options=pc.SequentialPairingOptions(
            overlap=15, quadratic_overlap=False, loop_detection=False, num_threads=4
        ),
        verification_options=verification,
        device=pc.Device.cpu,
    )
    import sqlite3

    with sqlite3.connect(db) as con:
        names = dict(con.execute('SELECT name,image_id FROM images'))
        existing = {
            row[0]
            for table in ('matches', 'two_view_geometries')
            for row in con.execute(f'SELECT pair_id FROM {table}')
        }
    new = []
    for i, j in pairs:
        a, b = data['image_names'][i], data['image_names'][j]
        lo, hi = sorted((names[a], names[b]))
        if lo * MAX_IMAGE_ID + hi not in existing:
            new.append(f'{a} {b}\n')
    pair_file = out / 'pairs.txt'
    pair_file.write_text(''.join(new))
    if new:
        pc.match_image_pairs(
            db,
            matching_options=matching,
            pairing_options=pc.ImportedPairingOptions(match_list_path=pair_file),
            verification_options=verification,
            device=pc.Device.cpu,
        )
    stage('sift_matching', tick)
    tick = time.perf_counter()
    match(data, images, pairs, out / 'xfeat.npz', settings)
    stage('xfeat', tick)
    tick = time.perf_counter()
    merge(data, db, out / 'xfeat.npz', out / 'mixed.db')
    stage('merge', tick)
    tick = time.perf_counter()
    build(
        data,
        out / 'mixed.db',
        images,
        out,
        sigma=settings.get('position_sigma_m', 0.05),
        rotation_sigma_deg=settings.get('rotation_sigma_deg'),
    )
    stage('global_ba', tick)
    dump(
        out / 'report.json',
        dict(
            provider='external',
            refine='ba',
            units=settings.get('units', 'meters'),
            registered=len(data['frame_ids']),
            input_frames=len(data['frame_ids']),
            feature_sources=['SIFT', 'XFeat'],
            candidate_pairs=len(pairs),
        ),
    )
