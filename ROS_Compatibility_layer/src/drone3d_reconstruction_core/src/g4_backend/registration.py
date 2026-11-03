"""E1: same-image pairs, consensus Sim(3), clustered covariance."""
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree
from . import sim3
from .geometry import unproject_pixels
from .types import (RegistrationConfig, RegistrationResult, RelativeEdge, SharedPairs,
                    THRESHOLD_RECOMPUTED)


def point_masks(window, config=None):
    """Per-window P50 confidence and 3x3 relative depth edge masks.

    The edge test uses SciPy morphological filters. On finite depth this is
    identical to the frozen OpenCV ``dilate/erode`` form (measured: 0 differing
    pixels). Where depth contains non-finite holes the two can differ by an
    implementation-order detail of the min/max reduction, so the shared-frame
    count of non-finite depth values is reported in the pair diagnostics instead
    of being silently absorbed.
    """
    config = config or RegistrationConfig()
    threshold = np.percentile(window.confidence, config.confidence_percentile)
    masks = []
    for depth, confidence in zip(window.depth, window.confidence):
        spread = maximum_filter(depth, size=3, mode="nearest") - minimum_filter(depth, size=3, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            edge = spread/np.maximum(depth, 1e-12) > config.edge_relative_threshold
        masks.append((confidence >= threshold) & ~edge & np.isfinite(depth) & (depth > 0))
    return np.asarray(masks)


def build_shared_pairs(older, newer, config=None, exclude_fit_pixels=None, seed=None,
                       threshold=None, threshold_policy=None, exclusion_source=""):
    """Use newer.shared_prefix identical frames; measurement direction is newer→older.

    exclude_fit_pixels maps physical frame IDs to model-pixel coordinates. Only
    fit samples are excluded; accept/report observations stay untouched so that
    holding them out keeps its meaning. Validation neighborhoods remain the
    caller's responsibility. Shared frames must have equal timestamps and, when
    supplied, equal RGB hashes.

    The fit threshold is either taken from ``threshold_policy`` (the value an
    earlier registered run recorded for this ordered edge) or recomputed from
    this run's mask intersection, and ``SharedPairs.threshold_source`` says which
    one was used. A recorded threshold is never silently replaced.
    """
    config = config or RegistrationConfig()
    # Frame identifiers arrive as integers from the ROS records and as strings
    # from JSON tables. Keys are normalized so a mixed pair can never miss
    # silently: a table that matches nothing is reported as zero exclusions.
    exclude_fit_pixels = {} if not exclude_fit_pixels else {
        str(key): value for key, value in exclude_fit_pixels.items()}
    if older.window_id == newer.window_id or newer.shared_prefix <= 0:
        raise ValueError("distinct windows and a nonempty shared prefix are required")
    if older.depth.shape[1:] != newer.depth.shape[1:]:
        raise ValueError("shared-image registration requires identical model image size")
    if bool(older.rgb_hashes) != bool(newer.rgb_hashes):
        raise ValueError("RGB hashes must be supplied for both windows or neither")
    if threshold is not None and (not np.isfinite(threshold) or threshold <= 0):
        raise ValueError("an explicit threshold must be finite and positive")
    lookup = {f: i for i, f in enumerate(older.frame_ids)}
    for k, frame_id in enumerate(newer.frame_ids[:newer.shared_prefix]):
        if frame_id not in lookup:
            raise ValueError("shared-prefix frame absent from older window")
        a = lookup[frame_id]
        if older.timestamps_ns[a] != newer.timestamps_ns[k]:
            raise ValueError("same frame ID has different timestamps")
        if older.rgb_hashes and older.rgb_hashes[a] != newer.rgb_hashes[k]:
            raise ValueError("same frame ID has different RGB hash")
    ma, mb = point_masks(older, config), point_masks(newer, config)
    h, w = newer.depth.shape[1:]
    cell_count = config.grid_rows*config.grid_columns
    # The frozen experiment used seed + newer numeric index, not a global RNG.
    if seed is None:
        if not isinstance(newer.window_id, (int, np.integer)):
            raise ValueError("noninteger window IDs require an explicit deterministic seed")
        seed = config.seed+int(newer.window_id)
    rng = np.random.default_rng(seed)
    arrays = {k: [] for k in ("source", "target", "source_depth", "target_depth", "split", "cluster", "shared_frame", "model_pixels")}
    frame_ids, depth_pool, excluded = [], [], 0
    excluded_frames = 0
    nonfinite_depth = 0
    for k, frame_id in enumerate(newer.frame_ids[:newer.shared_prefix]):
        a = lookup[frame_id]
        valid = ma[a] & mb[k]
        depth_pool.append(older.depth[a][valid])
        nonfinite_depth += int((~np.isfinite(newer.depth[k])).sum()) + int((~np.isfinite(older.depth[a])).sum())
        split = 0 if k % 5 < 3 else (1 if k % 5 == 3 else 2)
        yy, xx = np.nonzero(valid)
        uv = np.c_[xx, yy]
        if split == 0 and str(frame_id) in exclude_fit_pixels and len(uv):
            excluded_uv = np.asarray(exclude_fit_pixels[str(frame_id)], float)
            if excluded_uv.ndim != 2 or excluded_uv.shape[1] != 2 or not np.isfinite(excluded_uv).all():
                raise ValueError("exclusions must contain finite Nx2 model pixels")
            if len(excluded_uv):
                keep = cKDTree(excluded_uv).query(uv)[0] > config.exclusion_model_pixels
                excluded += int((~keep).sum())
                excluded_frames += 1
                uv = uv[keep]
        cells = np.minimum(uv[:, 0]*config.grid_columns//w, config.grid_columns-1) + config.grid_columns*np.minimum(uv[:, 1]*config.grid_rows//h, config.grid_rows-1)
        picked = []
        for cell in range(cell_count):
            pool = np.flatnonzero(cells == cell)
            picked.extend(rng.choice(pool, min(config.points_per_cell, len(pool)), replace=False))
        picked = np.asarray(picked, int)
        uv, cells = uv[picked], cells[picked]
        if not len(uv):
            continue
        target, da = unproject_pixels(older, a, uv)
        source, db = unproject_pixels(newer, k, uv)
        n = len(uv)
        values = {"source": source, "target": target, "source_depth": db, "target_depth": da,
                  "split": np.full(n, split, int), "cluster": k*cell_count+cells,
                  "shared_frame": np.full(n, k, int), "model_pixels": uv}
        for key, value in values.items():
            arrays[key].append(value)
        frame_ids.extend([frame_id]*n)
    out = {}
    for key, chunks in arrays.items():
        shape = (0, 3) if key in ("source", "target") else ((0, 2) if key == "model_pixels" else (0,))
        dtype = int if key in ("split", "cluster", "shared_frame") else float
        out[key] = np.concatenate(chunks) if chunks else np.empty(shape, dtype=dtype)
    pool = np.concatenate(depth_pool) if depth_pool else np.empty(0)
    recomputed = config.threshold_depth_fraction*float(np.median(pool)) if len(pool) else np.nan
    recorded, recorded_source = (None, None) if threshold_policy is None else threshold_policy.lookup(older.window_id, newer.window_id)
    if threshold is not None:
        value, source = float(threshold), "explicit_argument"
    elif recorded is not None:
        value, source = float(recorded), recorded_source
    else:
        value, source = recomputed, THRESHOLD_RECOMPUTED
    if len(out["source"]) and (not np.isfinite(value) or value <= 0):
        raise ValueError("no usable fit threshold for nonempty shared pairs")
    return SharedPairs(older.window_id, newer.window_id, **out, frame_ids=tuple(frame_ids), threshold=value,
                       threshold_source=source, cells_per_frame=cell_count,
                       diagnostics={"excluded_fit_neighborhood_points": excluded,
                                    "excluded_fit_frames": excluded_frames,
                                    "exclusion_source": str(exclusion_source),
                                    "nonfinite_depth_values": nonfinite_depth,
                                    "shared_frames": newer.shared_prefix,
                                    "rgb_hashes_checked": bool(older.rgb_hashes), "seed": int(seed),
                                    "threshold": float(value), "threshold_source": source,
                                    "threshold_recomputed": None if not np.isfinite(recomputed) else float(recomputed),
                                    "threshold_median_source_depth": None if not len(pool) else float(np.median(pool)),
                                    "samples_by_split": {str(s): int(np.sum(out["split"] == s)) for s in range(3)}})


def refine_consensus(x, y, transform, threshold, iterations=5):
    for _ in range(iterations):
        mask = np.linalg.norm(sim3.apply(transform, x)-y, axis=1) < threshold
        if mask.sum() < 3:
            raise ValueError("lost_consensus")
        transform = sim3.fit(x[mask], y[mask])
    return transform


def ransac(source, target, threshold, seed, config=None):
    config = config or RegistrationConfig()
    x, y = np.asarray(source, float), np.asarray(target, float)
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("RANSAC expects finite paired Nx3 points")
    if len(x) < config.min_fit_points:
        raise ValueError("fewer_than_min_fit_points")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    rng, best, score = np.random.default_rng(seed), None, (-1, -np.inf)
    for _ in range(config.ransac_trials):
        ids = rng.choice(len(x), 3, replace=False)
        if not sim3.support(x[ids], config.support_ratio)[0] or not sim3.support(y[ids], config.support_ratio)[0]:
            continue
        try:
            transform = sim3.fit(x[ids], y[ids])
        except (ValueError, np.linalg.LinAlgError):
            # The frozen experiment let a degenerate three-point hypothesis
            # propagate and abort the whole edge; the incremental default keeps
            # trying other hypotheses. Both semantics are selectable so neither
            # is applied under the other's name.
            if config.ransac_degenerate == "abort":
                raise
            continue
        e = np.linalg.norm(sim3.apply(transform, x)-y, axis=1)
        mask = e < threshold
        current = (int(mask.sum()), -float(np.mean(e[mask]**2)) if mask.any() else -np.inf)
        if current > score:
            score, best = current, transform
    if best is None:
        raise ValueError("no_noncollinear_hypothesis")
    transform = refine_consensus(x, y, best, threshold, config.refine_iterations)
    fraction = float(np.mean(np.linalg.norm(sim3.apply(transform, x)-y, axis=1) < threshold))
    if fraction < config.min_consensus:
        raise ValueError("consensus_below_minimum")
    return transform, fraction


def clustered_covariance(x, y, transform, threshold, clusters, length_source, seed, config=None, cells_per_frame=16):
    """Bootstrap right tangent Z^-1 Zb. Translation is divided by source depth."""
    config = config or RegistrationConfig()
    inlier = np.linalg.norm(sim3.apply(transform, x)-y, axis=1) < threshold
    if inlier.sum() < 3:
        return None, {"reason": "few_consensus"}
    a = x[inlier]/length_source
    J = np.concatenate([np.column_stack([np.eye(3), -sim3.skew(v), v]) for v in a], axis=0)
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > sv[0]*config.jacobian_relative_sv))
    if rank < 7:
        return None, {"reason": "measurement_rank", "rank": rank, "singular_values": sv.tolist()}
    rng = np.random.default_rng(seed)
    groups = {int(k): np.flatnonzero(clusters == k) for k in np.unique(clusters)}
    keys = list(groups)
    frames = sorted(set(k//cells_per_frame for k in keys))
    values = []
    for _ in range(config.bootstrap_replicates):
        chosen = []
        for frame in rng.choice(frames, len(frames), replace=True):
            cells = [k for k in keys if k//cells_per_frame == frame]
            chosen.extend(rng.choice(cells, len(cells), replace=True))
        idx = np.concatenate([groups[int(k)] for k in chosen])
        try:
            boot = refine_consensus(x[idx], y[idx], transform, threshold, config.refine_iterations)
            v = sim3.log(sim3.compose(sim3.inverse(transform), boot))
            v[:3] /= length_source
            if np.isfinite(v).all():
                values.append(v)
        except (ValueError, np.linalg.LinAlgError):
            pass
    if len(values) < config.bootstrap_minimum_valid or len(values) < 2:
        return None, {"reason": "bootstrap_valid_below_minimum", "valid": len(values), "rank": rank}
    C = np.cov(np.array(values).T)
    ev, Q = np.linalg.eigh((C+C.T)/2)
    C = (Q*np.maximum(ev, 0))@Q.T + np.eye(7)*config.covariance_sigma_floor**2
    return C, {"valid": len(values), "rejected": config.bootstrap_replicates-len(values), "rank": rank,
               "jacobian_singular_values": sv.tolist(), "raw_eigenvalues": ev.tolist(),
               "regularized_eigenvalues": np.linalg.eigvalsh(C).tolist(),
               "translation_normalizer_source": float(length_source)}


def estimate_adjacent(pairs, seed=None, threshold=None, config=None):
    config = config or RegistrationConfig()
    if seed is None:
        if not isinstance(pairs.j, (int, np.integer)):
            raise ValueError("noninteger window IDs require an explicit deterministic seed")
        seed = config.seed+int(pairs.j)
    threshold = pairs.threshold if threshold is None else threshold
    mask = pairs.split == 0
    x, y = pairs.source[mask], pairs.target[mask]
    diagnostics = {**pairs.diagnostics, "method": "E1_consensus", "fit_points": int(mask.sum()),
                   "threshold": float(threshold) if np.isfinite(threshold) else None,
                   "threshold_source": pairs.threshold_source,
                   "threshold_target_units": float(threshold) if np.isfinite(threshold) else None,
                   "admitted": False, "geometry_accepted": False}
    if not len(x):
        diagnostics["reason"] = "empty_fit_pool"
        return RegistrationResult(None, diagnostics)
    try:
        transform, fraction = ransac(x, y, threshold, seed, config)
        length = float(np.median(pairs.source_depth[mask]))
        C, info = clustered_covariance(x, y, transform, threshold, pairs.cluster[mask], length,
                                      seed+1000, config, pairs.cells_per_frame)
        diagnostics.update(info, consensus=fraction, measurement=transform.to_dict())
        if C is None:
            return RegistrationResult(None, diagnostics)
        diagnostics["admitted"] = True
        return RegistrationResult(RelativeEdge(pairs.i, pairs.j, transform, C, length, len(x)), diagnostics)
    except (ValueError, np.linalg.LinAlgError) as error:
        diagnostics["reason"] = str(error)
        return RegistrationResult(None, diagnostics)
