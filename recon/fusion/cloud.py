# SPDX-License-Identifier: Apache-2.0
import numpy as np
from recon_core.fusion import confidence_mask

def cache_depths(windows, scales, ids, folder, width=512, height=288):
    W, H = width, height
    folder.mkdir(parents=True, exist_ok=True)
    depths = np.lib.format.open_memmap(folder / 'depth_m.npy', mode='w+', dtype='float32', shape=(len(ids), H, W))
    good = np.lib.format.open_memmap(folder / 'good.npy', mode='w+', dtype='bool', shape=depths.shape)
    depths[:] = 0
    good[:] = False
    frame_scales = np.full(len(ids), np.nan, np.float64)
    for win, scale in zip(windows, scales):
        for frame in win['new']:
            i, j = int(np.searchsorted(ids, frame)), int(np.searchsorted(win['idx'], frame))
            d, c = win['depth'][j], win['conf'][j]
            depths[i] = d * scale
            good[i] = confidence_mask(d, c)
            frame_scales[i] = scale
    depths.flush()
    good.flush()
    np.savez(folder / 'frames.npz', frame_ids=ids)
    return depths, good, frame_scales
