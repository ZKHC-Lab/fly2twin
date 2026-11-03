"""Window scale: one metric scale per VGGT-Omega window, solved in the log domain.

VGGT depth inside a window is known up to one scale. Two constraint types fix it:

* trajectory prior: the window's VGGT camera centres, rotated by the mean
  provider/VGGT orientation difference, are fitted to the pose provider's
  centres for scale. Used only when the provider centres span more than
  ``min_extent_m``; weight ``min(extent, extent_weight_cap)``.
* adjacent edge: frames shared by two windows give s_b = s_a * r, where r is
  the median over shared frames of the per-frame median d_a / d_b over pixels
  whose confidence exceeds that frame's median in both windows; weight
  ``edge_weight``.

A connected component of windows without any trajectory prior has no metric
scale and is reported as unconstrained.
"""
import numpy as np

MIN_EXTENT_M = 0.3
EXTENT_WEIGHT_CAP = 2.0
EDGE_WEIGHT = 3.0
MIN_PIXELS = 500
MIN_SHARED_FRAMES = 3


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
    denom = float((da ** 2).sum())
    extent = float(np.linalg.norm(db, axis=1).max())
    scale = float((da * db).sum() / denom) if denom > 0 else 0.0
    residual = float(np.sqrt(((scale * da - db) ** 2).sum(1).mean())) if scale > 0 else float("inf")
    return {"scale": scale, "rotation": R, "extent_m": extent, "residual_rms_m": residual,
            "frames": len(local_centers)}


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


def solve_window_scales(window_ids, priors, edges, min_extent_m=MIN_EXTENT_M,
                        extent_weight_cap=EXTENT_WEIGHT_CAP, edge_weight=EDGE_WEIGHT):
    """Log-scale weighted least squares.

    ``priors``: id -> trajectory_prior result or None. ``edges``: iterable of
    (a, b, r) with s_b = s_a * r. Returns (scales: id -> float for constrained
    windows, constrained: set of ids, diagnostics: id -> dict).
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
    for a, b, rel in edges:
        if a not in index or b not in index or not np.isfinite(rel) or rel <= 0:
            continue
        r = np.zeros(len(ids))
        r[index[a]], r[index[b]] = -1, 1
        rows.append(r)
        rhs.append(np.log(rel))
        weights.append(edge_weight)
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
            "scale": scales.get(w), "constrained": w in constrained, "trajectory_prior_used": w in anchored,
            "trajectory_scale": None if p is None else p["scale"],
            "trajectory_extent_m": None if p is None else p["extent_m"],
            "trajectory_residual_rms_m": None if p is None else p["residual_rms_m"],
            "adjacent_edges": len(adjacency[w])}
    return scales, constrained, diagnostics


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
        trajectory.append(dict(id=win["id"], scale=prior["scale"] if prior else 0.0,
                               extent_m=prior["extent_m"] if prior else 0.0))
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
