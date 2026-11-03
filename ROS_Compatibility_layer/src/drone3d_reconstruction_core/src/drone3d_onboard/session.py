"""Read retained formal frame_pose sessions into the common fusion contract."""
from pathlib import Path
import numpy as np
from recon_core.fusion import confidence_mask, colored_cloud


def load_session(records, cache, require_complete=True):
    from drone3d_reconstruction_core.g4.input_adapter import local_path, WindowInputReader
    from drone3d_reconstruction_core.snapshot_validator import validate_snapshot, SnapshotSafetyLimits
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    rows, windows, sources = [], [], []
    seen = set()
    shape = None
    # Validate the source digests and snapshot, never use thumbnails as full RGB.
    for record in records:
        a = record.get('alignment', {})
        if (a.get('placement') != 'frame_pose' or not a.get('quality_passed')
                or (require_complete and a.get('map_stale'))):
            raise ValueError('Delivery/fusion requires a complete, current frame_pose revision')
        inp = WindowInputReader().load(record)
        meta = inp.metadata
        path = local_path(record['snapshot_manifest_uri'])
        snapshot = validate_snapshot(str(path), record['snapshot_manifest_sha256'], [path.parent],
            SnapshotSafetyLimits(512, 10000, 32*1024**2, 1024**3, 2*1024**3, 4096, 4096))
        wh = meta['model_image_size']
        if shape is not None and shape != tuple(wh[::-1]):
            raise ValueError('Mixed model resolutions are not supported')
        shape = tuple(wh[::-1])
        selected, unposed = [], []
        for i in a['merge_frame_indices']:
            fid = meta['frame_ids'][i]
            if fid in seen:
                raise ValueError('Duplicate owned frame ID')
            seen.add(fid)
            if a['frame_poses'][i] is None:
                # No localization pose at this RGB time (e.g. before odometry starts): not placed online either.
                unposed.append(fid)
                continue
            p = a['frame_poses'][i]
            pose = np.eye(4); pose[:3, :3] = p['rotation']; pose[:3, 3] = p['center']
            selected.append((len(rows), i))
            rows.append(dict(frame_id=fid, timestamp_ns=meta['timestamps_ns'][i],
                image_path=str(snapshot.image_paths[i]), c2w=pose, K=np.array(a['intrinsics']),
                camera_info=snapshot.camera_info_payloads[i], window_id=record['window_index']))
        windows.append(dict(window_id=record['window_index'], source=meta, alignment=a,
                            selected=selected, unposed=unposed))
        sources.append(record)
    if not rows:
        raise ValueError('Session contains no owned localized frames')
    depth = np.lib.format.open_memmap(cache/'depth_m.npy', mode='w+', dtype='float32', shape=(len(rows), *shape))
    good = np.lib.format.open_memmap(cache/'good.npy', mode='w+', dtype='bool', shape=depth.shape)
    for win in windows:
        with np.load(win['source']['prediction_uri'], allow_pickle=False) as f:
            d, c = f['depth'], f['depth_conf']
            if d.ndim == 4:
                d = d[..., 0]
            for dst, src in win['selected']:
                depth[dst] = d[src] * win['alignment']['depth_scale']
                good[dst] = confidence_mask(d[src], c[src])
    depth.flush(); good.flush()
    return rows, windows, depth, good


def fuse_session(records, cache, max_points=1000000, require_complete=True):
    rows, windows, depth, good = load_session(records, cache, require_complete)
    ids = np.arange(len(rows))
    xyz, rgb, info = colored_cloud(ids, np.array([r['c2w'] for r in rows]),
        np.array([r['K'] for r in rows]), depth, good,
        {str(i): r['image_path'] for i, r in enumerate(rows)}, max_points)
    info['frames_without_pose'] = sum(len(w['unposed']) for w in windows)
    return xyz, rgb, info, rows, windows
