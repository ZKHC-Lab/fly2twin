import numpy as np

def normalize(data):
    """Robust scene normalisation: median centre, MAD-guarded radius, P95 scale.

    A handful of mis-registered cameras can sit orders of magnitude away; a plain
    P95 then describes those outliers instead of the room, and dividing by that
    radius compresses the real scene so far that training cannot recover.  Keep
    only cameras inside a robust bound first, then take the P95 of what remains.
    """
    centers = data['c2w'][data['train_indices'], :3, 3]
    if not len(centers):
        raise RuntimeError('No registered training cameras in cameras.npz')
    center = np.median(centers, axis=0)
    distance = np.linalg.norm(centers - center, axis=1)
    median = float(np.median(distance))
    if not np.isfinite(median) or median <= 0:
        raise RuntimeError('Invalid median camera distance')
    spread = float(np.median(np.abs(distance - median))) or 0.5 * median
    limit = max(median + 8.0 * spread, 1.5 * median)
    outlier = data['train_indices'][distance > limit]
    train = data['train_indices'][distance <= limit]
    scale = float(np.percentile(np.linalg.norm(data['c2w'][train, :3, 3] - center, axis=1), 95))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError('Invalid P95 radius for the guarded training centres')

    registered = data['registered_indices']
    spacing = np.linalg.norm(data['c2w'][registered, :3, 3] - center, axis=1)
    far = registered[spacing > 5 * scale]
    keep = registered[spacing <= 5 * scale]
    dropped = np.union1d(outlier, far)
    keep = np.setdiff1d(keep, outlier)
    result = dict(center=center.tolist(), p95_radius=scale, median_camera_distance=median,
                  robust_limit=float(limit),
                  rule='median centre; MAD-guarded bound median+8*MAD (>=1.5*median) for the '
                       'normalisation scale; then drop cameras beyond 5*P95 of the kept set',
                  registered_before=len(registered), registered_after=len(keep),
                  dropped_count=int(len(dropped)), dropped_frames=data['frame_ids'][dropped].tolist(),
                  outlier_frames=data['frame_ids'][outlier].tolist(),
                  far_frames=data['frame_ids'][far].tolist())
    for split in ('train_indices', 'test_indices'):
        data[split] = data[split][np.isin(data[split], keep)]
        result[f'{split}_frames'] = len(data[split])
    if not result['train_indices_frames']:
        raise RuntimeError('Every training camera was rejected by the normalisation guard')
    data['registered_indices'] = keep
    return result
