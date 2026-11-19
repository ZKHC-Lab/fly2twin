"""Read retained formal frame_pose sessions into the common fusion contract."""
import json
import os
from pathlib import Path
import shutil
import numpy as np
from recon_core.fusion import (confidence_mask, colored_cloud, adaptive_cloud, frame_image, frame_points,
                               ranked_partners, symmetric_partners, NEIGHBOR_RADIUS_M, VOXEL_M)


def frame_depth_scale(alignment, k):
    """Metric depth scale of model frame ``k``: the per-frame G4 scale when the revision
    has one (frame_scales), else the window scale."""
    scales = alignment.get('frame_depth_scales')
    if scales is not None and scales[k] is not None:
        return scales[k]
    return alignment['depth_scale']


def _validated_window(record):
    """Formal window input and its validated snapshot (full-size RGB paths, never thumbnails)."""
    from drone3d_reconstruction_core.g4.input_adapter import local_path, WindowInputReader
    from drone3d_reconstruction_core.snapshot_validator import validate_snapshot, SnapshotSafetyLimits
    meta = WindowInputReader().load(record).metadata
    path = local_path(record['snapshot_manifest_uri'])
    snapshot = validate_snapshot(str(path), record['snapshot_manifest_sha256'], [path.parent],
        SnapshotSafetyLimits(512, 10000, 32*1024**2, 1024**3, 2*1024**3, 4096, 4096))
    return meta, snapshot


def _prediction(meta):
    with np.load(meta['prediction_uri'], allow_pickle=False) as f:
        d, c = f['depth'], f['depth_conf']
    return (d[..., 0] if d.ndim == 4 else d), c


def load_session(records, cache, require_complete=True):
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
        meta, snapshot = _validated_window(record)
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
        d, c = _prediction(win['source'])
        for dst, src in win['selected']:
            depth[dst] = d[src] * frame_depth_scale(win['alignment'], src)
            good[dst] = confidence_mask(d[src], c[src])
    depth.flush(); good.flush()
    return rows, windows, depth, good


def fuse_session(records, cache, max_points=1000000, require_complete=True, revisit=False):
    """``revisit``: correct the frame scales with surfaces other windows saw again (recon_core.revisit)
    before fusing; the cached depth is rewritten and every row gets ``revisit_log_scale``."""
    rows, windows, depth, good = load_session(records, cache, require_complete)
    ids = np.arange(len(rows))
    poses, K = np.array([r['c2w'] for r in rows]), np.array([r['K'] for r in rows])
    summary = None
    if revisit:
        from recon_core.revisit import frame_log_scale_corrections
        corrections, summary = frame_log_scale_corrections(poses, K, depth, good, np.array([r['window_id'] for r in rows]))
        for i, row in enumerate(rows):
            depth[i] = depth[i] * np.float32(np.exp(corrections[i]))
            row['revisit_log_scale'] = float(corrections[i])
        depth.flush()
    xyz, rgb, info = colored_cloud(ids, poses, K, depth, good,
        {str(i): r['image_path'] for i, r in enumerate(rows)}, max_points)
    info['frames_without_pose'] = sum(len(w['unposed']) for w in windows)
    if summary is not None:
        info['revisit'] = summary
    return xyz, rgb, info, rows, windows


class _Rows:
    """Row-indexed view, so recon_core.fusion.frame_points reads one frame at a time."""
    def __init__(self, get):
        self._get = get

    def __getitem__(self, i):
        return self._get(i)


_KEY_OFFSET = 1 << 20  # signed voxel index per axis within +-2**20 (21 km at 2 cm)


def _voxel_sums(xyz, rgb):
    """Per 2 cm voxel of a grid anchored at the origin: sorted keys, xyz/rgb sums, counts."""
    idx = np.floor((xyz + VOXEL_M / 2) / VOXEL_M).astype(np.int64) + _KEY_OFFSET
    keys, inverse = np.unique((idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2], return_inverse=True)
    n = len(keys)
    sums = np.stack([np.bincount(inverse, xyz[:, k], n) for k in range(3)]
                    + [np.bincount(inverse, rgb[:, k] / 255.0, n) for k in range(3)], 1)
    return keys, sums, np.bincount(inverse, minlength=n)


class SessionFusion:
    """Live-session cloud that recomputes only the frames whose inputs changed.

    A frame's points depend only on its own and its (at most six) neighbours' depth,
    window scale, pose and intrinsics, so a frame is reused while those are unchanged.
    Window scales are re-solved globally with every new window and drift slightly
    (2026-10-29 office_gs_04: most windows move < 0.1 % per revision, a few > 0.5 %),
    so scales count as unchanged within ``scale_tolerance`` (relative, against the
    value the cached points were computed with).

    The 2 cm voxels are running sums: a recomputed frame subtracts its old points and
    adds its new ones, so a refresh costs the changed frames plus one adaptive
    down-sampling of the 2 cm cloud. The grid is anchored at the world origin, not at
    the cloud minimum as in ``fuse_session``, so voxel boundaries differ from the
    delivery cloud; the delivery package after STOP always uses the exact full fusion.

    Each window is validated and read once per result digest. Its depth, masks and
    stride-2 colours, and each frame's points, are .npy files under ``cache``, read
    when needed, so a long session is neither held in memory nor in open files.
    """

    def __init__(self, cache, scale_tolerance=0.005):
        self.cache = Path(cache)
        self.scale_tolerance = scale_tolerance
        self._windows = {}
        self._cached = {}  # frame id -> (inputs without scales, scales used)
        self._fids, self._poses, self._ranked = [], np.empty((0, 4, 4)), []
        self._keys, self._sums, self._counts = np.empty(0, np.int64), np.empty((0, 6)), np.empty(0, np.int64)
        self.recomputed = 0

    def _frame_sums(self, fid, sign):
        folder = self.cache / 'frames'
        keys, sums, counts = _voxel_sums(np.load(folder / ('%s-xyz.npy' % fid)), np.load(folder / ('%s-rgb.npy' % fid)))
        return keys, sums * sign, counts * sign

    def _accumulate(self, parts):
        """Add (keys, sums, counts) parts into the running 2 cm voxels; drop emptied voxels."""
        if not parts:
            return
        keys, inverse = np.unique(np.concatenate([p[0] for p in parts]), return_inverse=True)
        n = len(keys)
        sums_in, counts_in = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
        sums = np.stack([np.bincount(inverse, sums_in[:, k], n) for k in range(6)], 1)
        counts = np.bincount(inverse, counts_in, n).round().astype(np.int64)
        pos = np.searchsorted(self._keys, keys)
        hit = pos < len(self._keys)
        hit[hit] = self._keys[pos[hit]] == keys[hit]
        self._sums[pos[hit]] += sums[hit]
        self._counts[pos[hit]] += counts[hit]
        new = ~hit
        self._keys = np.insert(self._keys, pos[new], keys[new])
        self._sums = np.insert(self._sums, pos[new], sums[new], axis=0)
        self._counts = np.insert(self._counts, pos[new], counts[new])
        if (self._counts <= 0).any():
            keep = self._counts > 0
            self._keys, self._sums, self._counts = self._keys[keep], self._sums[keep], self._counts[keep]

    def _window(self, record):
        a = record['alignment']
        digest = record['result_manifest_sha256']
        owned = tuple(i for i in a['merge_frame_indices'] if a['frame_poses'][i] is not None)
        key = (record['window_index'], digest, owned)
        win = self._windows.get(key)
        if win is not None:
            return win
        folder = self.cache / 'windows' / ('%06d-%s-%s' % (record['window_index'], digest[:16],
                                                            '-'.join(map(str, owned))[:64]))
        if not (folder / 'meta.json').exists():
            meta, snapshot = _validated_window(record)
            d, c = _prediction(meta)
            H, W = meta['model_image_size'][::-1]
            staging = folder.with_name(folder.name + '.tmp%d' % os.getpid())
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            np.save(staging / 'depth.npy', d[list(owned)])
            np.save(staging / 'good.npy', np.array([confidence_mask(d[i], c[i]) for i in owned]).reshape(-1, H, W))
            np.save(staging / 'image.npy', np.array([frame_image(snapshot.image_paths[i], H, W) for i in owned],
                                                    np.uint8).reshape(-1, (H + 1) // 2, (W + 1) // 2, 3))
            (staging / 'meta.json').write_text(json.dumps(dict(frame_ids=meta['frame_ids'], shape=[H, W])))
            shutil.rmtree(folder, ignore_errors=True)
            os.replace(staging, folder)
        meta = json.loads((folder / 'meta.json').read_text())
        win = dict(digest=digest, folder=folder, frame_ids=meta['frame_ids'], shape=tuple(meta['shape']), owned=owned)
        self._windows[key] = win
        return win

    @staticmethod
    def _read(row, name):
        # Map, copy one frame, drop the map: no file stays open between calls.
        return np.array(np.load(row['win']['folder'] / (name + '.npy'), mmap_mode='r')[row['k']])

    def _neighbors(self, fids, poses):
        from scipy.spatial import cKDTree
        centers, rotations = poses[:, :3, 3], poses[:, :3, :3]
        tree = cKDTree(centers)
        old = len(self._fids)
        if fids[:old] != self._fids:
            old, self._ranked = 0, []
        # A ranking depends only on the poses within 1.5 m, so redo it near new or moved frames.
        moved = [i for i in range(old) if not np.array_equal(poses[i], self._poses[i])]
        changed = moved + list(range(old, len(fids)))
        affected = set(changed)
        for i in changed:
            affected.update(tree.query_ball_point(centers[i], NEIGHBOR_RADIUS_M))
        for i in moved:  # frames that the moved one left
            affected.update(tree.query_ball_point(self._poses[i, :3, 3], NEIGHBOR_RADIUS_M))
        ranked = self._ranked[:old] + [None] * (len(fids) - old)
        for a in affected:
            ranked[a] = ranked_partners(a, centers, rotations, tree)
        self._fids, self._poses, self._ranked = list(fids), poses.copy(), ranked
        return symmetric_partners(ranked)

    def fuse(self, records, max_points=1000000):
        rows, seen, shape, unposed = [], set(), None, 0
        for record in records:
            a = record.get('alignment', {})
            if a.get('placement') != 'frame_pose' or not a.get('quality_passed'):
                raise ValueError('Delivery/fusion requires a complete, current frame_pose revision')
            win = self._window(record)
            if shape is not None and shape != win['shape']:
                raise ValueError('Mixed model resolutions are not supported')
            shape = win['shape']
            for i in a['merge_frame_indices']:
                fid = win['frame_ids'][i]
                if fid in seen:
                    raise ValueError('Duplicate owned frame ID')
                seen.add(fid)
                p = a['frame_poses'][i]
                if p is None:
                    unposed += 1
                    continue
                pose = np.eye(4); pose[:3, :3] = p['rotation']; pose[:3, 3] = p['center']
                rows.append(dict(fid=fid, win=win, k=win['owned'].index(i), c2w=pose,
                                 K=np.array(a['intrinsics']), scale=frame_depth_scale(a, i)))
        if not rows:
            raise ValueError('Session contains no owned localized frames')
        poses, K = np.array([r['c2w'] for r in rows]), np.array([r['K'] for r in rows])
        neighbors = self._neighbors([r['fid'] for r in rows], poses)
        # Same arithmetic as load_session: float32 depth times the Python scale, stored as float32.
        depth = _Rows(lambda i: np.asarray(self._read(rows[i], 'depth') * rows[i]['scale'], np.float32))
        good = _Rows(lambda i: self._read(rows[i], 'good'))

        def inputs(i):
            r = rows[i]
            return (r['fid'], r['win']['digest'], r['c2w'].tobytes(), r['K'].tobytes())

        folder = self.cache / 'frames'
        folder.mkdir(parents=True, exist_ok=True)
        try:
            gone = set(self._cached) - {r['fid'] for r in rows}
            parts = [self._frame_sums(fid, -1) for fid in gone]
            for fid in gone:
                del self._cached[fid]
            for i, r in enumerate(rows):
                signature = (inputs(i), tuple(inputs(j) for j in neighbors[i]))
                scales = np.array([r['scale']] + [rows[j]['scale'] for j in neighbors[i]])
                cached = self._cached.get(r['fid'])
                if (cached is None or cached[0] != signature
                        or np.abs(np.log(scales / cached[1])).max() > self.scale_tolerance):
                    if cached is not None:
                        parts.append(self._frame_sums(r['fid'], -1))
                    xyz, keep = frame_points(i, poses, K, depth, good, neighbors)
                    np.save(folder / ('%s-xyz.npy' % r['fid']), xyz)
                    np.save(folder / ('%s-rgb.npy' % r['fid']), self._read(r, 'image')[keep][:, ::-1])
                    parts.append(self._frame_sums(r['fid'], 1))
                    self._cached[r['fid']] = (signature, scales)
                    self.recomputed += 1
            self._accumulate(parts)
        except BaseException:
            # The running sums no longer match the cached frames: rebuild everything next time.
            self._cached = {}
            self._keys, self._sums, self._counts = np.empty(0, np.int64), np.empty((0, 6)), np.empty(0, np.int64)
            raise
        counts = self._counts[:, None].astype(np.float64)
        xyz, rgb, info = adaptive_cloud(self._sums[:, :3] / counts, self._sums[:, 3:] / counts, max_points)
        info['frames_without_pose'] = unposed
        return xyz, rgb, info
