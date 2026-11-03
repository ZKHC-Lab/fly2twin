# SPDX-License-Identifier: Apache-2.0
"""Clean global LiDAR reference and pooled nearest-neighbour distances.

Ported from T25/lidar_generate.py (header, scan, odometry, voxel keys,
trajectory, exact paired support), T31/cloud_lidar31.py (prepare and
 global_reference), and promo_20261029/lidar_eval/evaluate.py (float32 NN).
Historical sources live under Data/experiments; never imported at runtime.
"""
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

__all__ = ["build_reference", "distances"]
_DT = np.dtype([('t','<u4'),('x','<f4'),('y','<f4'),('z','<f4'),
                ('r','u1'),('tag','u1'),('line','u1')])

def _header(raw):
    stamp = struct.unpack_from('<II', raw, 4)
    n = struct.unpack_from('<I', raw, 12)[0]
    return stamp[0]*10**9+stamp[1], 16+n


def _scan(raw):
    hs, off = _header(raw)
    timebase, n = struct.unpack_from('<QI', raw, off)
    off += 16
    length = struct.unpack_from('<I', raw, off)[0]; off += 4
    if length != n or len(raw) != off+n*_DT.itemsize:
        raise ValueError('Unexpected Livox CustomMsg layout')
    a = np.frombuffer(raw, _DT, count=n, offset=off)
    return hs, timebase, a, np.column_stack([a['x'],a['y'],a['z']])


def _odometry(raw):
    ts, off = _header(raw)
    n = struct.unpack_from('<I', raw, off)[0]; off += 4+n
    return ts, np.array(struct.unpack_from('<7d', raw, off))


def _keys(p, size):
    q = np.floor(p/size).astype(np.int64)+(1<<20)
    if q.size and (q.min()<0 or q.max()>=(1<<21)):
        raise ValueError('Point outside collision-free 21-bit voxel key range')
    return (q[:,0]<<42)|(q[:,1]<<21)|q[:,2]


def _present(a, b):
    j = np.searchsorted(b,a)
    return (j<len(b)) & (b[np.minimum(j,len(b)-1)]==a) if len(b) else np.zeros(len(a),bool)


class _Trajectory:
    def __init__(self, ts, pq):
        if len(ts) < 2: raise ValueError("At least two odometry poses required")
        order = np.argsort(ts,kind='stable'); ts=ts[order]; pq=pq[order]
        unique, at, counts = np.unique(ts,return_index=True,return_counts=True)
        conflicts = sum(not np.all(pq[i:i+n]==pq[i]) for i,n in zip(at,counts) if n>1)
        if conflicts: raise ValueError(f'{conflicts} conflicting repeated odometry timestamps')
        ts=unique; pq=pq[at]; keep=[0]
        # Same exact-pose repeat within 150 ms: match established office preparation.
        for i in range(1,len(ts)):
            j=keep[-1]
            if ts[i]-ts[j]<150_000_000 and np.array_equal(pq[i],pq[j]): continue
            keep.append(i)
        self.stats=dict(messages=len(order),unique_timestamps=len(ts),dedup_poses=len(keep),
                        duplicate_timestamps=int(len(order)-len(ts)),repeated_pose_removed=len(ts)-len(keep),
                        dedup_rule='stable timestamp sort; equal timestamp conflicts rejected; exact same position/quaternion within150ms removed')
        if len(keep) < 2: raise ValueError("At least two distinct odometry poses required")
        self.ts=ts[keep]; self.pq=pq[keep]; self.origin=int(self.ts[0])
        self.sec=(self.ts-self.origin)/1e9
        self.rot=Slerp(self.sec,Rotation.from_quat(self.pq[:,3:]))
        self.stats['interval_ms_percentiles']=np.percentile(np.diff(self.ts)/1e6,[0,50,90,99,100]).tolist()

    def at(self, ns):
        sec=(np.asarray(ns,dtype=np.int64)-self.origin)/1e9
        if np.any((sec<self.sec[0])|(sec>self.sec[-1])): raise ValueError('No odometry extrapolation permitted')
        p=np.column_stack([np.interp(sec,self.sec,self.pq[:,j]) for j in range(3)])
        return p,self.rot(sec)

    def world(self, p, ns, transform):
        translation, rotation = self.at(ns)
        # Keep historical add-then-rotate arithmetic (not p@R.T+t).
        matrix, offset = transform
        local = (p + offset) @ matrix.T
        return rotation.apply(local) + translation


def _static_support(ks,scanid,scan_ns,positions,order=None):
    """Exact paired support in stable key order, without sorted-array copies."""
    if order is None:
        order = np.arange(len(ks))
    starts=[];good=[];witness_a=[];witness_b=[];i=0
    while i<len(ks):
        j=i+1
        while j<len(ks) and ks[order[j]]==ks[order[i]]: j+=1
        ok=False;wa=-1;wb=-1
        for a in range(i,j):
            sa=scanid[order[a]]
            for b in range(a+1,j):
                sb=scanid[order[b]]
                if abs(scan_ns[sb]-scan_ns[sa])<10000000000: continue
                v=positions[sb]-positions[sa]
                if v[0]*v[0]+v[1]*v[1]+v[2]*v[2]>=1.:
                    ok=True;wa=sa;wb=sb;break
            if ok: break
        starts.append(i);good.append(ok);witness_a.append(wa);witness_b.append(wb);i=j
    return np.array(starts),np.array(good),np.array(witness_a),np.array(witness_b)

def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build_reference(bag, out_npz, *, lidar_topic='/livox/lidar',
                    odometry_topic='/Odometry', body_from_lidar=None):
    """Build T25 clean 5 cm reference; T_BL maps LiDAR coordinates into body.

    None retains the historical Ry(+25°)*(p+[-.011,-.02329,.04412]).
    Custom extrinsics are rigid 4x4 matrices, not the inverse body→LiDAR.
    Only the final compressed NPZ and same-stem JSON are written.
    """
    try:
        from rosbags.rosbag1 import Reader
        from numba import njit
    except ImportError as exc:
        raise ImportError("LiDAR reference generation requires: pip install -e '.[lidar]'") from exc
    start = time.perf_counter()
    bag, out_npz = Path(bag), Path(out_npz)
    if out_npz.suffix != '.npz':
        raise ValueError('Reference output must end in .npz')
    if bag.resolve() in (out_npz.resolve(), out_npz.with_suffix('.json').resolve()):
        raise ValueError('Output must not overwrite the input bag')
    rotation = Rotation.from_euler('y', 25, degrees=True)
    matrix = rotation.as_matrix()
    offset = np.array([-.011, -.02329, .04412])
    extrinsic = np.eye(4)
    extrinsic[:3, :3], extrinsic[:3, 3] = matrix, rotation.apply(offset)
    if body_from_lidar is not None:
        extrinsic = np.asarray(body_from_lidar, dtype=np.float64)
        if (extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all()
                or not np.allclose(extrinsic[3], [0, 0, 0, 1])
                or not np.allclose(extrinsic[:3, :3].T @ extrinsic[:3, :3], np.eye(3))
                or not np.isclose(np.linalg.det(extrinsic[:3, :3]), 1)):
            raise ValueError('body_from_lidar must be a rigid 4x4 T_BL')
        matrix = extrinsic[:3, :3]
        offset = np.linalg.solve(matrix, extrinsic[:3, 3])
    transform = matrix, offset
    hist, ts, pq, meta = {}, [], [], []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic in (odometry_topic, lidar_topic)]
        for c, _, raw in reader.messages(connections=conns):
            if c.topic == odometry_topic:
                a, b = _odometry(raw); ts.append(a); pq.append(b)
                continue
            hs, tb, a, p = _scan(raw)
            if not len(p):
                raise ValueError('Empty Livox scan has no midpoint timestamp')
            valid = np.isfinite(p).all(1) & (np.linalg.norm(p, axis=1) >= .5)
            for k in np.unique(_keys(p[valid], .1)):
                hist[int(k)] = hist.get(int(k), 0) + 1
            pt = tb + a['t'].astype(np.int64)
            meta.append((hs, tb, int(pt.min()), int(pt.max()), len(p)))
            if len(meta) % 300 == 0:
                print('lidar metadata scans', len(meta), flush=True)
    if not ts or not meta:
        raise ValueError('Bag lacks odometry or Livox scans on the requested topics')
    traj = _Trajectory(np.array(ts, np.int64), np.array(pq))
    odometry_stats = traj.stats
    # T31 Inputs reconstructed the already-deduplicated trajectory. Preserve this
    # second pass (normally idempotent), rather than silently changing its path.
    traj = _Trajectory(traj.ts, traj.pq)
    fixed = np.array(sorted(k for k, n in hist.items() if n >= np.ceil(.2*len(meta))), np.int64)
    del hist, ts, pq
    meta = np.array(meta, np.int64)
    mid = (meta[:, 2] + meta[:, 3]) // 2
    if np.any(np.diff(mid) < 0):
        raise ValueError('Scan midpoints not ordered')
    pos, rot = traj.at(np.clip(mid, traj.ts[0], traj.ts[-1]))
    origins = pos + rot.apply(np.tile(extrinsic[:3, 3], (len(pos), 1)))
    ks, points, scan_ids = [], [], []
    counts = dict.fromkeys(('raw', 'nonfinite', 'outside_trajectory', 'near', 'fixed', 'rear', 'kept'), 0)
    # Compact scan IDs reduce peak RAM without changing ordering or witnesses.
    sid_dtype = np.uint16 if len(mid) <= 65536 else np.int32
    with Reader(bag) as reader:
        conn = [c for c in reader.connections if c.topic == lidar_topic]
        scans = 0
        for sid, (_, _, raw) in enumerate(reader.messages(connections=conn)):
            _, tb, a, p = _scan(raw); ns = tb + a['t'].astype(np.int64)
            finite = np.isfinite(p).all(1)
            timing = (ns >= traj.ts[0]) & (ns <= traj.ts[-1])
            mask = finite & timing
            counts['raw'] += len(p)
            counts['nonfinite'] += int((~finite).sum())
            counts['outside_trajectory'] += int((finite & ~timing).sum())
            near = mask & (np.linalg.norm(p, axis=1) < .5)
            counts['near'] += int(near.sum()); mask &= ~near
            blocked = np.zeros(len(p), bool)
            blocked[mask] = _present(_keys(p[mask], .1), fixed)
            counts['fixed'] += int(blocked.sum()); mask &= ~blocked
            counts['kept'] += int(mask.sum())
            w = traj.world(p[mask], ns[mask], transform) if mask.any() else np.empty((0, 3))
            k, ix = np.unique(_keys(w, .05), return_index=True)
            ks.append(k); points.append(w[ix].astype(np.float32))
            scan_ids.append(np.full(len(ix), sid, sid_dtype))
            scans += 1
            if sid % 200 == 0:
                print('global lidar scans', sid, flush=True)
    if scans != len(mid):
        raise ValueError('Bag and metadata scan count differ')
    print('global lidar concatenate', flush=True)
    k = np.concatenate(ks); del ks
    p = np.concatenate(points); del points
    s = np.concatenate(scan_ids); del scan_ids
    if not len(k):
        raise ValueError('No eligible reference points')
    print('global lidar stable sort', len(k), flush=True)
    order = np.argsort(k, kind='stable')
    print('global lidar paired support', flush=True)
    support = njit(_static_support)
    starts, good, wa, wb = support(k, s, mid, origins, order)
    ix = order[starts[good]]
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, points=p[ix], voxel_key=k[ix],
                        witness_scan_a=wa[good].astype(np.int32),
                        witness_scan_b=wb[good].astype(np.int32))
    report = dict(clean_points=len(ix), scans=scans, eligible_scan_voxel_records=len(k),
                  filtering=counts, fixed_voxels=len(fixed), odometry=odometry_stats,
                  bag=str(bag), bag_bytes=bag.stat().st_size,
                  reference_sha256=_digest(out_npz), sha256_scope='complete NPZ file bytes',
                  parameters=dict(lidar_topic=lidar_topic, odometry_topic=odometry_topic,
                                  body_from_lidar=extrinsic.tolist(), min_range_m=.5,
                                  sensor_voxel_m=.1, sensor_occupancy_fraction=.2,
                                  reference_voxel_m=.05, revisit_time_s=10., revisit_distance_m=1.,
                                  odometry_repeat_ns=150000000, rear_wedge=False),
                  seconds=time.perf_counter()-start)
    out_npz.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def _load_points(value):
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as z:
                value = z['points']
        elif path.suffix.lower() == '.ply':
            from recon.io.ply import read_ply
            v = read_ply(path)
            value = np.column_stack([v[a] for a in ('x', 'y', 'z')])
        else:
            raise ValueError('Point cloud must be .npz or scalar-vertex .ply')
    p = np.asarray(value)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Expected nonempty finite Nx3 points; no silent filtering')
    return p


def _quant(a):
    return dict(n=int(a.size), p50=float(np.median(a)) if a.size else None,
                p90=float(np.percentile(a, 90)) if a.size else None)


def distances(cloud_points, reference):
    """All fused points → nearest reference; identity, no fit, metres.

    Arrays or NPZ/PLY paths accepted. Distances round to float32 BEFORE pooled
    quantiles, exactly as T31. Thresholds are strict <; world-z bins are [lo,hi).
    """
    p, ref = _load_points(cloud_points), _load_points(reference)
    tree = cKDTree(ref)
    d = np.concatenate([tree.query(p[i:i+100000], workers=4)[0]
                        for i in range(0, len(p), 100000)]).astype(np.float32)
    layers = []
    for label, lo, hi in [('<1', -np.inf, 1), ('1–2', 1, 2), ('2–3', 2, 3),
                          ('3–4', 3, 4), ('>=4', 4, np.inf)]:
        layers.append(dict(world_z_m=label, **_quant(d[(p[:, 2] >= lo) & (p[:, 2] < hi)])))
    return dict(**_quant(d), height_layers=layers, reference_points=len(ref),
                within_distance_fraction={str(t): float(np.mean(d < t)) for t in [.05, .10, .20]},
                protocol='clean global 5 cm 首点地图；所有融合点找最近的参照点，恒等变换，不配准；全部点汇总后取 P50/P90；不是点到平面距离。')
