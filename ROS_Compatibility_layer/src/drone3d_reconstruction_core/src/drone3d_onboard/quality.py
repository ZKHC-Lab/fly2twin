"""Adjacent verified input-match sqrt-Sampson metrics (not held-out truth)."""
from pathlib import Path
import sqlite3
import numpy as np


def adjacent_sampson(database, names, pose_sets, K):
    pairs, xy1, xy2 = [], [], []
    with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro', uri=True) as db:
        ids = dict(db.execute('SELECT name,image_id FROM images'))
        keys = {}
        def keypoints(i):
            if i not in keys:
                n, c, blob = db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (i,)).fetchone()
                keys[i] = np.frombuffer(blob, np.float32).reshape(n,c)[:, :2].astype(float)
            return keys[i]
        for j in range(len(names)-1):
            a, b = ids[names[j]], ids[names[j+1]]
            lo, hi = sorted((a,b))
            row = db.execute('SELECT rows,cols,data FROM two_view_geometries WHERE pair_id=?', (lo*2147483647+hi,)).fetchone()
            if row is None or row[0] < 1:
                continue
            matches = np.frombuffer(row[2], np.uint32).reshape(row[:2])
            if a > b:
                matches = matches[:, ::-1]
            pairs.append((j,j+1)); xy1.append(keypoints(a)[matches[:,0]]); xy2.append(keypoints(b)[matches[:,1]])
    values = {}
    invK = np.linalg.inv(K)
    for name, poses in pose_sets.items():
        medians = np.full(len(pairs), np.nan)
        for n,(a,b) in enumerate(pairs):
            ra,rb=poses[a,:3,:3],poses[b,:3,:3]
            t=rb.T@(poses[a,:3,3]-poses[b,:3,3])
            if not np.isfinite(t).all() or np.linalg.norm(t)<=1e-12:
                continue
            skew=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
            F=invK[b].T@skew@(rb.T@ra)@invK[a]
            x1=np.c_[xy1[n],np.ones(len(xy1[n]))];x2=np.c_[xy2[n],np.ones(len(xy2[n]))]
            u,v=x1@F.T,x2@F
            with np.errstate(divide='ignore',invalid='ignore'):
                error=np.sqrt(np.sum(x2*u,axis=1)**2/(np.sum(u[:,:2]**2,axis=1)+np.sum(v[:,:2]**2,axis=1)))
            if len(error) and np.isfinite(error).all():
                medians[n]=np.median(error)
        values[name]=medians
    valid=np.logical_and.reduce([np.isfinite(v) for v in values.values()])
    return dict(protocol='Adjacent verified mixed matches; per-pair median sqrt-Sampson then pair percentiles; BA inputs, not independent validation',
        candidate_adjacent_pairs=len(names)-1, verified_pairs=len(pairs), common_nondegenerate_pairs=int(valid.sum()),
        excluded_degenerate_pairs=int((~valid).sum()), rows={name:dict(p50_p90_px=np.percentile(v[valid],[50,90]).tolist() if valid.any() else [None,None],
            pair_medians_px=v[valid].tolist()) for name,v in values.items()})
