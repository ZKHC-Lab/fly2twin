"""Revisit alignment: per-frame depth-scale corrections from surfaces that other windows also saw.

The scale solve links views that are close in pose (the fusion partners: within 1.5 m and
30 deg). A surface seen again from somewhere else, by the next window or much later,
constrains nothing, so the scales of two windows that look at the same wall can differ by a
few per cent and the fused cloud shows the wall twice. This step adds that constraint after
the scales are solved:

* members: the points the fusion keeps (the check of ``fusion.frame_points``) on every
  second column of its pixel grid, each with the surface normal of its depth map turned
  towards its camera; grouped by the window that owns the frame.
* matches: a sample of each window's points is matched to the nearest point of the OTHER
  windows (within a gate, normals within about 53 deg of each other, so the two faces of a
  thin wall never match). A match between frames i and j with target normal n gives the
  point-to-plane residual r = n . (p - q); scaling frame i's depth by exp(x_i) moves p along
  its viewing ray, so r changes by n . (p - c_i) x_i.
* solve: one sparse least-squares problem over all frames, sum of Huber-weighted
  (r + a_i x_i - a_j x_j)^2, with every correction pulled to zero (``SIGMA_PRIOR``) and to
  the corrections of its neighbours in time inside the window (``SIGMA_TEMPORAL``).
  Repeated with the gate going 20 -> 12 -> 8 cm; a round changes no correction by more than
  ``MAX_STEP``.

Only depth scales change; poses stay, so a pose error between two passes is absorbed as
scale. A gap between two frames is closed mostly by the one farther from the surface: with
the same prior on every frame, that needs the smaller scale change. Frames that share no
surface with another window keep their scale.
"""
import numpy as np

from .fusion import partners, supported_points

GATES_M = (0.20, 0.12, 0.08)   # match distance per round
NORMAL_DOT = 0.6               # cos between the two normals of a match (both face their cameras)
HUBER_M = 0.03                 # residuals beyond this get weight HUBER_M / |r|
SIGMA_RESIDUAL_M = 0.03        # point-to-plane noise of one match
CORRELATION = 30.0             # matches of one frame pair are not independent: variance factor
SIGMA_PRIOR = 0.03             # log-scale correction of a frame
SIGMA_TEMPORAL = 0.01          # difference between consecutive frames of one window
MAX_STEP = 0.05                # largest change of a correction in one round
MEMBER_STEP = 2                # the pool takes every MEMBER_STEP-th column of the fusion's every-second-pixel grid
SOURCE_POINTS = 30000          # points of a window that look for matches
TARGET_POINTS = 500000         # points of the other windows they are matched against
MIN_POINTS = 2000              # a window (or its surroundings) with fewer points is skipped
MARGIN_M = 0.3                 # the other windows' points are taken from the window's box plus this


def member_points(poses, K, depth, good, step=MEMBER_STEP):
    """Fused points of every frame as vectors from their camera centre, with normals.

    Arguments as ``fusion.colored_cloud``. Returns (rel float32 (M, 3) world-frame vector
    from the camera centre to the point, normal float32 (M, 3) world-frame unit normal
    facing the camera, first int64 (N + 1,): frame f owns rows first[f]:first[f + 1]).
    """
    neighbors = partners(poses)
    H, W = depth.shape[1:]
    v, u = np.mgrid[0:H, 0:W]
    rows, cols = (a.reshape(-1) for a in np.mgrid[0:H:2, 0:W:2 * step])
    rel, normal, count = [], [], np.zeros(len(poses), np.int64)
    for i in range(len(poses)):
        xyz, keep = supported_points(i, poses, K, depth, good, neighbors, rows, cols)
        if not len(xyz):
            continue
        Ki, R = K[i], poses[i, :3, :3]
        cam = np.stack([(u - Ki[0, 2]) / Ki[0, 0], (v - Ki[1, 2]) / Ki[1, 1], np.ones((H, W))], -1) * np.asarray(depth[i])[..., None]
        at = cam[rows[keep], cols[keep]]
        n = np.cross(np.gradient(cam, axis=1)[rows[keep], cols[keep]], np.gradient(cam, axis=0)[rows[keep], cols[keep]])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        n[np.sum(n * at, axis=1) > 0] *= -1
        rel.append((xyz - poses[i, :3, 3]).astype(np.float32))
        normal.append((n @ R.T).astype(np.float32))
        count[i] = len(rel[-1])
    first = np.r_[0, np.cumsum(count)]
    if not rel:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), first
    return np.concatenate(rel), np.concatenate(normal), first


def _thin(index, limit):
    """At most ``limit`` entries of ``index``, evenly spaced (no random numbers)."""
    return index if len(index) <= limit else index[::-(-len(index) // limit)]


def _rows(first, frames):
    """Pool rows of ``frames`` (ascending), concatenated."""
    return np.concatenate([np.arange(first[f], first[f + 1]) for f in frames]) if len(frames) else np.zeros(0, np.int64)


def frame_log_scale_corrections(poses, K, depth, good, frame_window, gates=GATES_M,
                                sigma_prior=SIGMA_PRIOR, sigma_temporal=SIGMA_TEMPORAL):
    """Log-scale correction per frame; multiply frame f's metric depth by exp(result[f]).

    ``poses`` camera-to-world (N, 4, 4); ``K`` intrinsics at the depth resolution (N, 3, 3);
    ``depth`` metric (N, H, W) at the solved scales, may be a read-only mmap; ``good`` the
    per-frame masks (N, H, W); ``frame_window`` (N,) label of the window that owns each frame.
    Frames must be in time order: consecutive rows with the same label are tied together.
    Returns (corrections float64 (N,), info dict). All zeros when fewer than two windows have
    points or nothing matches.
    """
    from scipy.sparse import coo_matrix, identity
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree
    poses = np.asarray(poses, np.float64)
    n = len(poses)
    code = np.unique(np.asarray(frame_window), return_inverse=True)[1].reshape(-1)
    sig = np.zeros(n)
    info = dict(frames=n, windows=int(code.max() + 1) if n else 0, member_points=0, rounds=[],
                correction_abs_median=0.0, correction_abs_max=0.0)
    rel, normal, first = member_points(poses, K, depth, good)
    info["member_points"] = int(len(rel))
    frame = np.repeat(np.arange(n, dtype=np.int32), np.diff(first))
    sources = {}
    for w in range(info["windows"]):
        mine = _rows(first, np.flatnonzero(code == w))
        if len(mine) >= MIN_POINTS:
            sources[w] = _thin(mine, SOURCE_POINTS)
    if len(sources) < 2:
        return sig, info
    centers = poses[:, :3, 3]
    reach = np.array([np.sqrt((rel[first[f]:first[f + 1]].astype(np.float64) ** 2).sum(1).max()) if first[f + 1] > first[f] else 0.0
                      for f in range(n)])
    pairs = np.flatnonzero(code[:-1] == code[1:])
    k = 1.0 / sigma_temporal ** 2
    smooth = coo_matrix((np.r_[np.full(2 * len(pairs), k), np.full(2 * len(pairs), -k)],
                         (np.r_[pairs, pairs + 1, pairs, pairs + 1], np.r_[pairs, pairs + 1, pairs + 1, pairs])), shape=(n, n)).tocsr()
    prior = identity(n, format="csr") / sigma_prior ** 2
    points = np.empty_like(rel)
    for gate in gates:
        for a in range(0, len(rel), 1 << 20):       # world points at the current corrections, in chunks
            f = frame[a:a + (1 << 20)]
            points[a:a + (1 << 20)] = centers[f] + np.exp(sig[f])[:, None] * rel[a:a + (1 << 20)]
        far = reach * np.exp(sig) + MARGIN_M
        I, J, AI, AJ, RR = [], [], [], [], []
        for w, src in sources.items():
            p = points[src].astype(np.float64)
            lo, hi = p.min(0), p.max(0)
            # Frames of other windows that can reach the box; then their points inside it.
            near = np.flatnonzero((code != w) & np.all((centers >= lo - far[:, None]) & (centers <= hi + far[:, None]), axis=1))
            other = _rows(first, near)
            other = other[np.all((points[other] >= lo - MARGIN_M) & (points[other] <= hi + MARGIN_M), axis=1)]
            if len(other) < MIN_POINTS:
                continue
            other = _thin(other, TARGET_POINTS)
            q = points[other].astype(np.float64)
            distance, nearest = cKDTree(q).query(p, distance_upper_bound=gate)
            hit = np.flatnonzero(np.isfinite(distance))
            nearest = nearest[hit]
            nt = normal[other[nearest]].astype(np.float64)
            ok = np.sum(normal[src[hit]] * nt, axis=1) > NORMAL_DOT
            hit, nearest, nt = hit[ok], nearest[ok], nt[ok]
            fi, fj = frame[src[hit]], frame[other[nearest]]
            I.append(fi); J.append(fj)
            RR.append(np.sum(nt * (p[hit] - q[nearest]), axis=1))
            AI.append(np.sum(nt * (p[hit] - centers[fi]), axis=1))
            AJ.append(np.sum(nt * (q[nearest] - centers[fj]), axis=1))
        matches = int(sum(len(r) for r in RR))
        if not matches:
            break
        I, J, AI, AJ, RR = (np.concatenate(a) for a in (I, J, AI, AJ, RR))
        wt = np.minimum(1.0, HUBER_M / np.maximum(np.abs(RR), 1e-9)) / (SIGMA_RESIDUAL_M ** 2 * CORRELATION)
        A = coo_matrix((np.r_[wt * AI * AI, wt * AJ * AJ, -wt * AI * AJ, -wt * AI * AJ],
                        (np.r_[I, J, I, J], np.r_[I, J, J, I])), shape=(n, n)).tocsr() + prior + smooth
        b = np.bincount(J, wt * AJ * RR, minlength=n) - np.bincount(I, wt * AI * RR, minlength=n) - prior @ sig - smooth @ sig
        sig = sig + np.clip(spsolve(A.tocsc(), b), -MAX_STEP, MAX_STEP)
        info["rounds"].append(dict(gate_m=float(gate), matches=matches, residual_median_m=float(np.median(np.abs(RR)))))
    info["correction_abs_median"] = float(np.median(np.abs(sig)))
    info["correction_abs_max"] = float(np.abs(sig).max())
    return sig, info
