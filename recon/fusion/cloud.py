# SPDX-License-Identifier: Apache-2.0
import numpy as np
from recon_core.fusion import INIT_CONFIDENCE_PERCENTILE, confidence_mask


def cache_depths(windows, scales, ids, folder, width=512, height=288):
    W, H = width, height
    folder.mkdir(parents=True, exist_ok=True)
    depths = np.lib.format.open_memmap(folder / 'depth_m.npy', mode='w+', dtype='float32', shape=(len(ids), H, W))
    good = np.lib.format.open_memmap(folder / 'good.npy', mode='w+', dtype='bool', shape=depths.shape)
    # Wider mask of the cloud Gaussians start from (recon_core.fusion.gaussian_init_cloud).
    loose = np.lib.format.open_memmap(folder / 'good_init.npy', mode='w+', dtype='bool', shape=depths.shape)
    depths[:] = 0
    good[:] = False
    loose[:] = False
    frame_scales = np.full(len(ids), np.nan, np.float64)
    for win, scale in zip(windows, scales):
        for frame in win['new']:
            i, j = int(np.searchsorted(ids, frame)), int(np.searchsorted(win['idx'], frame))
            d, c = win['depth'][j], win['conf'][j]
            depths[i] = d * scale
            good[i] = confidence_mask(d, c)
            loose[i] = confidence_mask(d, c, INIT_CONFIDENCE_PERCENTILE)
            frame_scales[i] = scale
    depths.flush()
    good.flush()
    loose.flush()
    np.savez(folder / 'frames.npz', frame_ids=ids)
    return depths, good, loose, frame_scales
