"""Window or per-frame metric depth scales, solved in the log domain.

Window scales use trajectory priors and depth-ratio constraints:

* trajectory prior: the window's VGGT camera centres, rotated by the mean
  provider/VGGT orientation difference, are fitted to the pose provider's
  centres for scale. Used only when the provider centres span more than
  ``min_extent_m``; weight ``min(extent, extent_weight_cap)``.
* adjacent edge: frames shared by two windows give s_b = s_a * r, where r is
  the median over shared frames of the per-frame median d_a / d_b over pixels
  whose confidence exceeds that frame's median in both windows; weight
  ``edge_weight``.
* cross-window edge (optional): two overlapping frames of different windows,
  frame i's metric depth moved into camera j with the pose provider's poses and
  compared with frame j's depth (``cross_view_log_ratio``); one row per directed
  frame pair with its own weight. This also links revisits that are far apart in
  time, where the adjacent chain alone drifts.

The per-frame solver adds frame-pair rows, frame deviations from the window
scale, and temporal constraints within each window.
A connected component of windows without any trajectory prior has no metric
scale and is reported as unconstrained.
"""

import numpy as np

MIN_EXTENT_M = 0.3
EXTENT_WEIGHT_CAP = 2.0
EDGE_WEIGHT = 3.0
MIN_PIXELS = 500
MIN_SHARED_FRAMES = 3
CROSS_GATE_LOG = 0.25  # |log depth ratio| counted for a cross-window pair
CROSS_MIN_PIXELS = 500  # in model-resolution pixels
CROSS_DEPTH_RANGE_M = (0.3, 4.0)


def rotation_mean(rotations):
    """Chordal L2 mean of a stack of rotation matrices."""
    U, _, Vt = np.linalg.svd(np.sum(rotations, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def trajectory_prior(local_centers, local_rotations, ref_centers, ref_rotations, valid=None):
    """Rotation-first fit of VGGT camera centres to provider centres.

    Returns dict(scale: metres per local unit, rotation: local -> world,
    extent_m: max distance of provider centres from their mean, residual_rms_m,
    frames), or None with fewer than two valid frames. Rotation comes from
    orientations, so a window with little translation still has a frame.
    """
    local_centers, local_rotations = np.asarray(local_centers), np.asarray(local_rotations)
    ref_centers, ref_rotations = np.asarray(ref_centers), np.asarray(ref_rotations)
    if valid is not None:
        valid = np.asarray(valid, bool)
        local_centers, local_rotations = local_centers[valid], local_rotations[valid]
        ref_centers, ref_rotations = ref_centers[valid], ref_rotations[valid]
    if len(local_centers) < 2:
        return None
    R = rotation_mean(np.einsum("nij,nkj->nik", ref_rotations, local_rotations))
    a = local_centers @ R.T
    da, db = a - a.mean(0), ref_centers - ref_centers.mean(0)
    denom = float((da**2).sum())
    extent = float(np.linalg.norm(db, axis=1).max())
    scale = float((da * db).sum() / denom) if denom > 0 else 0.0
    residual = float(np.sqrt(((scale * da - db) ** 2).sum(1).mean())) if scale > 0 else float("inf")
    return {"scale": scale, "rotation": R, "extent_m": extent, "residual_rms_m": residual, "frames": len(local_centers)}


def frame_depth_ratio(depth_a, conf_a, depth_b, conf_b, min_pixels=MIN_PIXELS):
    """Median d_a / d_b of one frame seen by two windows, or None with too few pixels."""
    mask = (conf_a > np.median(conf_a)) & (conf_b > np.median(conf_b)) & (depth_a > 0) & (depth_b > 0)
    if mask.sum() < min_pixels:
        return None
    return np.median(depth_a[mask] / depth_b[mask])


def shared_depth_ratio(frames, min_pixels=MIN_PIXELS):
    """``frames``: iterable of (depth_a, conf_a, depth_b, conf_b) for frames in both windows.

    Returns (r, frames_used) with s_b = s_a * r, or (None, 0).
    """
    ratios = [r for r in (frame_depth_ratio(*f, min_pixels=min_pixels) for f in frames) if r is not None]
    if not ratios:
        return None, 0
    return float(np.median(ratios)), len(ratios)


def cross_view_log_ratio(
    depth_i,
    K_i,
    pose_i,
    depth_j,
    K_j,
    pose_j,
    stride=1,
    gate=CROSS_GATE_LOG,
    min_pixels=CROSS_MIN_PIXELS,
    depth_range=CROSS_DEPTH_RANGE_M,
):
    """Median log(z / d_j) of frame i's metric depth moved into camera j.

    ``depth_*`` are metric depth maps sampled every ``stride`` model pixels
    (``depth[v, u]`` is model pixel (stride*u, stride*v)); ``K_*`` at model
    resolution; ``pose_*`` camera-to-world. A pixel counts when it projects
    inside frame j, j's depth there lies in ``depth_range`` and the ratio is
    within ``gate``. ``min_pixels`` is in model pixels (divided by stride**2).
    > 0 means frame i's depth is larger than frame j's. Returns (value or None, pixels).
    """
    depth_i, depth_j = np.asarray(depth_i, np.float64), np.asarray(depth_j, np.float64)
    h, w = depth_i.shape
    hj, wj = depth_j.shape
    v, u = np.mgrid[0:h, 0:w] * stride
    ray = np.stack([(u - K_i[0, 2]) / K_i[0, 0], (v - K_i[1, 2]) / K_i[1, 1], np.ones((h, w))], -1)
    world = (ray * depth_i[..., None]).reshape(-1, 3) @ pose_i[:3, :3].T + pose_i[:3, 3]
    xyz = (world - pose_j[:3, 3]) @ pose_j[:3, :3]
    z = xyz[:, 2]
    front = z > 0.1
    uu = np.rint((K_j[0, 0] * xyz[front, 0] / z[front] + K_j[0, 2]) / stride).astype(int)
    vv = np.rint((K_j[1, 1] * xyz[front, 1] / z[front] + K_j[1, 2]) / stride).astype(int)
    inside = (uu >= 0) & (uu < wj) & (vv >= 0) & (vv < hj)
    observed = depth_j[vv[inside], uu[inside]]
    ratio = np.log(np.maximum(z[front][inside], 1e-6) / np.maximum(observed, 1e-6))
    ok = (np.abs(ratio) < gate) & (observed > depth_range[0]) & (observed < depth_range[1])
    count = int(ok.sum())
    if count * stride * stride <= min_pixels:
        return None, count
    return float(np.median(ratio[ok])), count


def solve_window_scales(
    window_ids, priors, edges, min_extent_m=MIN_EXTENT_M, extent_weight_cap=EXTENT_WEIGHT_CAP, edge_weight=EDGE_WEIGHT
):
    """Log-scale weighted least squares.

    ``priors``: id -> trajectory_prior result or None. ``edges``: iterable of
    (a, b, r) with s_b = s_a * r, or (a, b, r, weight) to override ``edge_weight``.
    Returns (scales: id -> float for constrained windows, constrained: set of ids,
    diagnostics: id -> dict).
    """
    ids = list(window_ids)
    index = {w: k for k, w in enumerate(ids)}
    rows, rhs, weights = [], [], []
    anchored = set()
    for w in ids:
        p = priors.get(w)
        if p is not None and p["scale"] > 0 and p["extent_m"] > min_extent_m:
            r = np.zeros(len(ids))
            r[index[w]] = 1
            rows.append(r)
            rhs.append(np.log(p["scale"]))
            weights.append(min(p["extent_m"], extent_weight_cap))
            anchored.add(w)
    adjacency = {w: set() for w in ids}
    for edge in edges:
        a, b, rel = edge[:3]
        if a not in index or b not in index or not np.isfinite(rel) or rel <= 0:
            continue
        r = np.zeros(len(ids))
        r[index[a]], r[index[b]] = -1, 1
        rows.append(r)
        rhs.append(np.log(rel))
        weights.append(edge[3] if len(edge) > 3 else edge_weight)
        adjacency[a].add(b)
        adjacency[b].add(a)
    constrained, seen = set(), set()
    for start in ids:
        if start in seen:
            continue
        stack, component = [start], set()
        while stack:
            w = stack.pop()
            if w not in component:
                component.add(w)
                stack.extend(adjacency[w] - component)
        seen |= component
        if component & anchored:
            constrained |= component
    scales = {}
    if constrained:
        cols = [index[w] for w in ids if w in constrained]
        A = np.array(rows)[:, cols] * np.array(weights)[:, None]
        keep = np.any(A != 0, axis=1)
        x = np.linalg.lstsq(A[keep], (np.array(rhs) * np.array(weights))[keep], rcond=None)[0]
        for k, c in enumerate(cols):
            scales[ids[c]] = float(np.exp(x[k]))
    diagnostics = {}
    for w in ids:
        p = priors.get(w)
        diagnostics[w] = {
            "scale": scales.get(w),
            "constrained": w in constrained,
            "trajectory_prior_used": w in anchored,
            "trajectory_scale": None if p is None else p["scale"],
            "trajectory_extent_m": None if p is None else p["extent_m"],
            "trajectory_residual_rms_m": None if p is None else p["residual_rms_m"],
            "adjacent_edges": len(adjacency[w]),
        }
    return scales, constrained, diagnostics


SIGMA_PRIOR = 0.04  # log-scale noise of a trajectory prior (office_gs_04, LiDAR-fitted truth)
SIGMA_EDGE = 0.04  # adjacent shared-frame edge
SIGMA_ROW = 0.08  # one directed frame-pair row (rows of one window pair are correlated)
SIGMA_FRAME = 0.04  # a frame's deviation from its window scale
SIGMA_TEMPORAL = 0.02  # deviation change between consecutive frames of one window


def solve_frame_scales(
    window_ids,
    priors,
    edges,
    frame_window,
    frame_rows,
    temporal_pairs,
    sigma_prior=SIGMA_PRIOR,
    sigma_edge=SIGMA_EDGE,
    sigma_row=SIGMA_ROW,
    sigma_frame=SIGMA_FRAME,
    sigma_temporal=SIGMA_TEMPORAL,
    min_extent_m=MIN_EXTENT_M,
):
    """Joint per-frame depth scales: log s_f = x_w(f) + d_f, sparse weighted least squares.

    ``priors``: id -> trajectory_prior result or None, anchoring x_w = log scale when the
    provider moved more than ``min_extent_m``. ``edges``: (a, b, r) with x_b - x_a = log r.
    ``frame_window``: window id per frame. ``frame_rows``: (i, j, value) on frame indices with
    log s_i - log s_j = value (``cross_view_log_ratio`` at known scales, both windows or one).
    ``temporal_pairs``: (i, j) consecutive frames of one window. Each frame deviation d_f has a
    zero prior. Every constraint is weighted by 1 / its sigma.
    Returns (window scales: id -> float, frame scales: array with NaN for unconstrained
    frames, constrained window ids). Components without an anchor stay unconstrained.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    ids = list(window_ids)
    index = {w: k for k, w in enumerate(ids)}
    frame_window = np.asarray(frame_window)
    frame_w = np.array([index.get(w, -1) for w in frame_window], int)
    n_w, n_f = len(ids), len(frame_window)
    # Window connectivity through edges and cross-window rows; components need an anchor.
    parent = list(range(n_w))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    anchored = [
        w for w in ids if priors.get(w) is not None and priors[w]["scale"] > 0 and priors[w]["extent_m"] > min_extent_m
    ]
    links = [(index[a], index[b]) for a, b, r in edges if a in index and b in index and np.isfinite(r) and r > 0]
    links += [
        (frame_w[i], frame_w[j]) for i, j, v in frame_rows if frame_w[i] >= 0 and frame_w[j] >= 0 and np.isfinite(v)
    ]
    for a, b in links:
        parent[find(a)] = find(b)
    roots = {find(index[w]) for w in anchored}
    constrained = {w for w in ids if find(index[w]) in roots}
    if not constrained:
        return {}, np.full(n_f, np.nan), set()
    cols = {w: k for k, w in enumerate(w for w in ids if w in constrained)}
    frame_col = np.array([n_w + q for q in range(n_f)])  # deviation unknowns after the windows
    rr, cc, vv, rhs = [], [], [], []

    def add(entries, value, sigma):
        row = len(rhs)
        for col, coeff in entries:
            rr.append(row)
            cc.append(col)
            vv.append(coeff / sigma)
        rhs.append(value / sigma)

    for w in anchored:
        if w in cols:
            add([(cols[w], 1.0)], np.log(priors[w]["scale"]), sigma_prior)
    for a, b, r in edges:
        if a in cols and b in cols and np.isfinite(r) and r > 0:
            add([(cols[b], 1.0), (cols[a], -1.0)], np.log(r), sigma_edge)
    in_solve = np.array([frame_w[q] >= 0 and ids[frame_w[q]] in cols for q in range(n_f)])
    for i, j, value in frame_rows:
        if in_solve[i] and in_solve[j] and np.isfinite(value):
            add(
                [
                    (cols[ids[frame_w[i]]], 1.0),
                    (frame_col[i], 1.0),
                    (cols[ids[frame_w[j]]], -1.0),
                    (frame_col[j], -1.0),
                ],
                value,
                sigma_row,
            )
    for q in np.flatnonzero(in_solve):
        add([(frame_col[q], 1.0)], 0.0, sigma_frame)
    for i, j in temporal_pairs:
        if in_solve[i] and in_solve[j]:
            add([(frame_col[i], 1.0), (frame_col[j], -1.0)], 0.0, sigma_temporal)
    A = coo_matrix((vv, (rr, cc)), shape=(len(rhs), n_w + n_f)).tocsr()
    x = lsqr(A, np.asarray(rhs), atol=1e-12, btol=1e-12, iter_lim=50 * (n_w + n_f))[0]
    window_scales = {w: float(np.exp(x[k])) for w, k in cols.items()}
    frame_scales = np.full(n_f, np.nan)
    for q in np.flatnonzero(in_solve):
        frame_scales[q] = np.exp(x[cols[ids[frame_w[q]]]] + x[frame_col[q]])
    return window_scales, frame_scales, constrained


def batch_window_scales(windows, ids, poses):
    """Scales for an ordered list of finished windows (offline and after landing).

    ``windows``: dicts with id, idx (sorted frame ids), vc / vR (VGGT camera
    centres / rotations), depth, conf. ``ids`` sorted frame ids of ``poses``
    (camera-to-world, metric). Consecutive windows sharing at least three frames
    are linked. Raises ValueError unless every window gets a metric scale.
    Returns (scales array, per-window trajectory fits, edges).
    """
    priors, trajectory = {}, []
    for win in windows:
        p = poses[np.searchsorted(ids, win["idx"])]
        prior = trajectory_prior(win["vc"], win["vR"], p[:, :3, 3], p[:, :3, :3])
        priors[win["id"]] = prior
        trajectory.append(
            dict(id=win["id"], scale=prior["scale"] if prior else 0.0, extent_m=prior["extent_m"] if prior else 0.0)
        )
    links, edges = [], []
    for a, b in zip(windows, windows[1:]):
        shared = np.intersect1d(a["idx"], b["idx"])
        if len(shared) < MIN_SHARED_FRAMES:
            continue
        pairs = zip(np.searchsorted(a["idx"], shared), np.searchsorted(b["idx"], shared))
        ratio, _ = shared_depth_ratio((a["depth"][i], a["conf"][i], b["depth"][j], b["conf"][j]) for i, j in pairs)
        if ratio is None:
            raise ValueError("No valid shared depth ratio: %s -> %s" % (a["id"], b["id"]))
        links.append((a["id"], b["id"], ratio))
        edges.append(dict(a=a["id"], b=b["id"], depth_ratio=ratio, shared_frames=len(shared)))
    scales, constrained, _ = solve_window_scales([w["id"] for w in windows], priors, links)
    if len(constrained) < len(windows):
        raise ValueError("Omega windows have no anchored metric scale or a disconnected scale system")
    return np.array([scales[w["id"]] for w in windows]), trajectory, edges
