# SPDX-License-Identifier: Apache-2.0
import json
import numpy as np

def load_windows(folder, ids, width=512, height=288):
    W, H = width, height
    windows = json.loads((folder / 'windows.json').read_text())['windows']
    if sorted(f for win in windows for f in win['new']) != ids.tolist():
        raise ValueError('Every keyframe must have exactly one Omega new-owner window')
    loaded = []
    for win in windows:
        idx, new = np.asarray(win['idx']), np.asarray(win['new'])
        path = folder / win['id'] / 'prediction.npz'
        with np.load(path, allow_pickle=False) as z:
            depth, conf, E = z['depth'], z['depth_conf'], z['extrinsic'].astype(np.float64)
        if (depth.shape != (len(idx), H, W, 1) or conf.shape != (len(idx), H, W)
                or E.shape != (len(idx), 3, 4) or not np.isfinite(depth).all()
                or not np.isfinite(conf).all() or not np.isfinite(E).all()
                or np.any(np.diff(idx) <= 0) or not np.isin(idx, ids).all()
                or not np.isin(new, idx).all()):
            raise ValueError(f'Invalid prediction/window: {path}')
        rotations = E[:, :, :3].transpose(0, 2, 1)
        centers = -np.einsum('nij,nj->ni', rotations, E[:, :, 3])
        loaded.append(dict(id=win['id'], idx=idx, new=new, vc=centers, vR=rotations,
                           depth=depth[..., 0], conf=conf))
    return loaded
