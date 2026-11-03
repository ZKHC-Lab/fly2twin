# SPDX-License-Identifier: Apache-2.0
"""Windowed inference through the shared VGGT-Omega reconstruction entry point."""
from pathlib import Path
import sys
import numpy as np
from recon.io.common import dump


def plan_windows(ids, first=40, new=30, overlap=10):
    """V1.0 windows: first window, then new frames with preceding context."""
    ids = list(map(int, ids))
    windows = [dict(id='K000', idx=ids[:first], new=ids[:first])]
    start = len(windows[0]['new'])
    while start < len(ids):
        fresh = ids[start:start + new]
        share = max(overlap, new - len(fresh))
        windows.append(dict(id=f'K{len(windows):03d}',
                            idx=ids[max(0, start - share):start] + fresh, new=fresh))
        start += len(fresh)
    return windows


def infer_windows(manifest, settings, out):
    source = Path(settings.get('omega_source', Path(__file__).resolve().parents[2] / 'third_party' / 'vggt-omega'))
    sys.path.insert(0, str(source))
    import reconstruct_omega as omega

    model = omega.load_model(settings['weights'], 'cuda', aggregator_frame_chunk_size=10)
    paths = {row['frame_id']: row['path'] for row in manifest['frames']}
    windows = plan_windows(paths, first=settings.get('first_window', 40),
                           new=settings.get('new_frames', 30), overlap=settings.get('overlap', 10))
    resolution = settings.get('resolution', [512, 288])[0]
    for win in windows:
        images = omega.load_and_preprocess_images([paths[i] for i in win['idx']],
                                                 mode='max_size', image_resolution=resolution)
        prediction = omega.run_inference(model, images, 'cuda')
        folder = out / win['id']
        folder.mkdir()
        np.savez_compressed(folder / 'prediction.npz', **prediction, frame_ids=np.asarray(win['idx']))
        del prediction, images
    dump(out / 'windows.json', dict(windows=windows))
