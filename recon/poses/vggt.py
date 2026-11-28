# SPDX-License-Identifier: Apache-2.0
"""Image-only poses from VGGT windows: chain by depth, close loops, then feature bundle adjustment.

VGGT-Omega already estimates cameras and depth for every window of frames. This provider uses them:

chain    Window k+1 is placed on window k with a similarity estimated on the frames they share: rotation from
         the shared cameras' orientations, scale from the per-pixel ratio of the two depth maps of the same
         frames, translation from the camera centres. Depth, not camera positions, carries the scale: the
         shared frames span well under a second, so their centres barely move.
loops    For frame pairs with verified feature matches the matched keypoints are lifted to 3D with each
         frame's own VGGT depth and intrinsics. Two windows then see the same points; every window gets one
         similarity correction (window 0 fixed) that makes those points agree, as seen from both cameras
         (direction and log depth), while consecutive windows keep the chain's relative placement. A pair
         counts as about one measurement however many points it has, pairs far from where the chain puts
         them are not used, and pairs that still disagree after the solve are dropped and it is solved again.
refine   Triangulation from the tracked poses and bundle adjustment with loose position priors, three rounds
         (16, 8, 4 px): the rounds of the landing BA, but rotations are free, the prior is a fraction of the
         path, and the shared SIMPLE_RADIAL camera is self-calibrated from VGGT's focal length and no distortion.

No structure-from-motion solve is involved, so there is no registration step that can fail: every frame keeps
a pose, and the scale is one for the whole sequence (arbitrary until `pose.metric` sets it).
Measured 2026-11-25: on five phone walks the global SfM solver this replaced (GLOMAP, removed 2026-11-27) broke
the trajectory into pieces of different scale (speed jumps of 50-2000x) where this chain kept a steady pace.
"""

import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
from recon.io.common import dump

PAIR_ID_BASE = 2147483647  # COLMAP pair_id = image_id1 * base + image_id2
DEPTH_SIGMA = 0.03  # relative depth noise of a lifted point
BEARING_SIGMA = 0.01  # direction noise of a lifted point seen from the other camera (rad)
PAIR_WEIGHT = 10.0  # a matched frame pair counts as this many independent residuals: its points share the depth errors
CHAIN_SIGMA = (0.02, 0.5, 0.02)  # consecutive windows: relative scale, rotation (deg), position / depth
CHAIN_DRIFT = 0.03  # a pair may sit this fraction of the chained path between its frames away from itself
PRIOR_FRACTION = 0.003  # position prior of the BA as a fraction of the path length


def mean_rotation(rotations):
    U, _, Vt = np.linalg.svd(np.sum(rotations, axis=0))
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def similarity(source, target):
    """Least-squares (s, R, t) with target ~ s R source + t (Umeyama)."""
    ms, mt = source.mean(0), target.mean(0)
    a, b = source - ms, target - mt
    U, S, Vt = np.linalg.svd(b.T @ a / len(source))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / max(a.var(0).sum(), 1e-18))
    return s, R, mt - s * R @ ms


def load_windows(folder):
    """Windows of an Omega output folder: idx, new, vR (camera-to-window), vc, K (depth grid), depth, good."""
    folder = Path(folder)
    windows = []
    for win in json.loads((folder / 'windows.json').read_text())['windows']:
        with np.load(folder / win['id'] / 'prediction.npz', allow_pickle=False) as z:
            E = z['extrinsic'].astype(np.float64)
            depth, conf, K = z['depth'][..., 0], z['depth_conf'], z['intrinsic'].astype(np.float64)
        vR = E[:, :, :3].transpose(0, 2, 1)
        windows.append(
            dict(
                idx=[int(i) for i in win['idx']],
                new=[int(i) for i in win['new']],
                vR=vR,
                vc=-np.einsum('nij,nj->ni', vR, E[:, :, 3]),
                K=K,
                depth=depth,
                good=(conf > np.median(conf, axis=(1, 2), keepdims=True)) & (depth > 0),
            )
        )
    return windows


def chain(windows):
    """Per-window similarity (s, A, t) to the frame of window 0: X_chain = s A X_window + t.

    Also returns, per join, how much the shared frames disagree about the scale (range of the log ratios).
    """
    placed, spread = [(1.0, np.eye(3), np.zeros(3))], []
    for k in range(1, len(windows)):
        prev, win = windows[k - 1], windows[k]
        ps, pA, pt = placed[-1]
        where = {i: n for n, i in enumerate(prev['idx'])}
        shared = [(n, where[i]) for n, i in enumerate(win['idx']) if i in where]
        if not shared:
            raise ValueError(f'Window {k} shares no frame with the previous window')
        A = mean_rotation([pA @ prev['vR'][m] @ win['vR'][n].T for n, m in shared])
        ratios = []
        for n, m in shared:
            both = win['good'][n] & prev['good'][m]
            if both.sum() > 100:
                ratios.append(np.median(prev['depth'][m][both] / win['depth'][n][both]))
        if not ratios:
            raise ValueError(f'Window {k} has no confident depth in the frames it shares with the previous window')
        s = ps * float(np.median(ratios))
        spread.append(float(np.ptp(np.log(ratios))) if len(ratios) > 1 else 0.0)
        t = np.mean([ps * pA @ prev['vc'][m] + pt - s * A @ win['vc'][n] for n, m in shared], axis=0)
        placed.append((s, A, t))
    return placed, spread


class Correction:
    """X -> exp(v[0]) Rot(v[1:4]) (X - pivot) + pivot + v[4:7]."""

    def __init__(self, v, pivot):
        from scipy.spatial.transform import Rotation

        self.scale, self.R, self.t, self.pivot = float(np.exp(v[0])), Rotation.from_rotvec(v[1:4]).as_matrix(), v[4:7], pivot

    def point(self, X):
        return self.scale * (X - self.pivot) @ self.R.T + self.pivot + self.t


def frame_poses(windows, placed, corrections=None):
    """c2w (N, 4, 4) and frame ids, each frame from the window in which it is new."""
    ids, poses = [], []
    for k, (win, (s, A, t)) in enumerate(zip(windows, placed)):
        for i in win['new']:
            n = win['idx'].index(i)
            R, c = A @ win['vR'][n], s * A @ win['vc'][n] + t
            if corrections is not None:
                R, c = corrections[k].R @ R, corrections[k].point(c)
            pose = np.eye(4)
            U, _, Vt = np.linalg.svd(R)
            pose[:3, :3] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
            pose[:3, 3] = c
            ids.append(i)
            poses.append(pose)
    order = np.argsort(ids)
    return np.array(ids)[order], np.array(poses)[order]


def lift(win, placement, n, uv):
    """Chain-frame 3D points and depths at depth-grid positions uv (N, 2) of frame slot n; mask of usable ones."""
    H, W = win['depth'].shape[1:]
    col, row = np.rint(uv[:, 0] - 0.5).astype(int), np.rint(uv[:, 1] - 0.5).astype(int)
    ok = (col >= 1) & (col < W - 1) & (row >= 1) & (row < H - 1)
    col, row = np.clip(col, 0, W - 1), np.clip(row, 0, H - 1)
    d = win['depth'][n][row, col].astype(np.float64)
    ok &= win['good'][n][row, col]
    patch = np.stack(
        [win['depth'][n][np.clip(row + a, 0, H - 1), np.clip(col + b, 0, W - 1)] for a in (-1, 0, 1) for b in (-1, 0, 1)]
    )
    ok &= (patch.max(0) - patch.min(0)) / np.maximum(d, 1e-9) < 0.05  # not on a depth edge
    K = win['K'][n]
    cam = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))] * d[:, None]
    s, A, t = placement
    return s * (cam @ win['vR'][n].T + win['vc'][n]) @ A.T + t, s * d, ok


def chain_edges(windows, placed):
    """Per pair of consecutive windows: (k-1, k, centre of the shared cameras, their median depth), chain units."""
    edges = []
    for k in range(1, len(windows)):
        prev, win = windows[k - 1], windows[k]
        s, A, t = placed[k]
        before = set(prev['idx'])
        shared = [n for n, i in enumerate(win['idx']) if i in before]
        centre = np.mean([s * A @ win['vc'][n] + t for n in shared], axis=0)
        depth = s * float(np.median(np.concatenate([win['depth'][n][win['good'][n]] for n in shared])))
        edges.append((k - 1, k, centre, depth))
    return edges


def match_links(windows, placed, database, frame_of_image, scale, min_gap=16, min_matches=30, max_pairs=4000, per_pair=60, seed=0):
    """Links between windows from verified matches: [(window a, window b, points a, points b, camera a, camera b)], counts.

    A pair of frames at least `min_gap` positions apart and owned by different windows is used when
    its lifted points agree after one similarity fit (median residual below 5% of depth, scale within a factor
    3), which rejects frames with bad depth, and when the chain already puts the two point sets within
    CHAIN_DRIFT of the path between the frames (plus 5% of depth), which rejects matches between look-alike
    places. `scale` maps keypoint pixels to the depth grid (x, y factors): the features are detected on the
    frames VGGT saw. Points and camera centres are in chain units.
    """
    owner = {i: k for k, win in enumerate(windows) for i in win['new']}
    ids, poses = frame_poses(windows, placed)
    position = {int(i): n for n, i in enumerate(ids)}
    travelled = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))])
    con = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    image = {image_id: frame_of_image(name) for image_id, name in con.execute('select image_id, name from images')}
    pairs = []
    for pair_id, _ in con.execute('select pair_id, rows from two_view_geometries where rows >= ?', (min_matches,)):
        fa, fb = image.get(pair_id // PAIR_ID_BASE), image.get(pair_id % PAIR_ID_BASE)
        if fa in owner and fb in owner and abs(position[fa] - position[fb]) >= min_gap and owner[fa] != owner[fb]:
            pairs.append((pair_id, fa, fb))
    rng = np.random.default_rng(seed)
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    cache = {}

    def keypoints(image_id):
        if image_id not in cache:
            n, c, blob = con.execute('select rows, cols, data from keypoints where image_id = ?', (image_id,)).fetchone()
            cache[image_id] = np.frombuffer(blob, np.float32).reshape(n, c)[:, :2].astype(np.float64) * scale
        return cache[image_id]

    links, counts = [], dict(candidates=len(pairs), used=0, few_points=0, inconsistent=0, displaced=0, far=0)
    for pair_id, fa, fb in pairs:
        n, blob = con.execute('select rows, data from two_view_geometries where pair_id = ?', (pair_id,)).fetchone()
        m = np.frombuffer(blob, np.uint32).reshape(n, 2)
        ka, kb = owner[fa], owner[fb]
        Pa, da, oa = lift(windows[ka], placed[ka], windows[ka]['idx'].index(fa), keypoints(pair_id // PAIR_ID_BASE)[m[:, 0]])
        Pb, db, ob = lift(windows[kb], placed[kb], windows[kb]['idx'].index(fb), keypoints(pair_id % PAIR_ID_BASE)[m[:, 1]])
        ok = oa & ob
        if ok.sum() < 15:
            counts['few_points'] += 1
            continue
        Pa, Pb, depth = Pa[ok], Pb[ok], 0.5 * (da[ok] + db[ok])
        s, R, t = similarity(Pb, Pa)
        residual = np.linalg.norm(s * Pb @ R.T + t - Pa, axis=1) / depth
        if np.median(residual) > 0.05 or not 1 / 3 < s < 3:
            counts['inconsistent'] += 1
            continue
        allowed = 0.05 * np.median(depth) + CHAIN_DRIFT * abs(travelled[position[fa]] - travelled[position[fb]])
        if np.linalg.norm(np.median(Pa - Pb, axis=0)) > allowed:
            counts['displaced'] += 1
            continue
        keep = np.flatnonzero(residual < 0.15)
        if len(keep) > per_pair:
            keep = rng.choice(keep, per_pair, replace=False)
        links.append((ka, kb, Pa[keep], Pb[keep], poses[position[fa], :3, 3], poses[position[fb], :3, 3]))
        counts['used'] += 1
        counts['far'] += int(abs(ka - kb) >= 3)
    con.close()
    return links, counts


def link_residuals(Xa, Xb, Ca, Cb):
    """Disagreement of two lifted versions of the same points, seen from both cameras: (N, 8) in sigmas.

    Per camera: log of the two depths' ratio and the difference of the two directions. Depth errors are
    multiplicative and act along the viewing ray of the camera that measured them, so this form is unbiased
    where a difference of positions is not (a forward-moving camera would drift in scale by a few percent),
    and it does not change when both sides are scaled together.
    """
    out = []
    for C in (Ca, Cb):
        u, v = Xa - C, Xb - C
        du, dv = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
        out.append(np.log(du / dv)[:, None] / (DEPTH_SIGMA * np.sqrt(2)))
        out.append((u / du[:, None] - v / dv[:, None]) / BEARING_SIGMA)
    return np.concatenate(out, axis=1)


def pose_graph(windows, placed, links, loss_scale=3.0, chain_sigma=CHAIN_SIGMA):
    """One similarity correction per window (window 0 fixed); returns ([Correction], info).

    Consecutive windows keep the chain's relative placement within `chain_sigma`. Every link (a matched frame
    pair) contributes `link_residuals` with a Cauchy loss and a total weight of PAIR_WEIGHT residuals. After
    the first solve, pairs whose median residual is still above `loss_scale` sigmas are dropped and it is
    solved once more. All terms are ratios or directions, so scaling every window together changes nothing
    (a residual in fixed units would let all windows but the first shrink, window after window).
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation

    W = len(windows)
    pivots = [s * A @ win['vc'].mean(0) + t for win, (s, A, t) in zip(windows, placed)]
    edges = chain_edges(windows, placed)
    s_sigma, r_sigma, t_sigma = chain_sigma[0], np.radians(chain_sigma[1]), chain_sigma[2]

    def corrections(x):
        return [Correction(np.zeros(7), pivots[0])] + [Correction(x[7 * (k - 1) : 7 * k], pivots[k]) for k in range(1, W)]

    def pair(cs, link):
        a, b, Pa, Pb, ca, cb = link
        return link_residuals(cs[a].point(Pa), cs[b].point(Pb), cs[a].point(ca), cs[b].point(cb))

    def solve(x, active):
        offsets = np.concatenate([[0], np.cumsum([8 * len(link[2]) for link in active])]).astype(int)
        total = int(offsets[-1]) + 7 * len(edges)

        def residuals(x):
            cs = corrections(x)
            out = np.empty(total)
            for link, o0, o1 in zip(active, offsets[:-1], offsets[1:]):
                r = pair(cs, link).ravel()
                robust = np.sign(r) * loss_scale * np.sqrt(np.log1p((r / loss_scale) ** 2))  # squared: Cauchy
                out[o0:o1] = robust * np.sqrt(PAIR_WEIGHT / len(r))
            for n, (a, b, centre, depth) in enumerate(edges):
                o = offsets[-1] + 7 * n
                out[o] = np.log(cs[a].scale / cs[b].scale) / s_sigma
                out[o + 1 : o + 4] = Rotation.from_matrix(cs[a].R.T @ cs[b].R).as_rotvec() / r_sigma
                unit = t_sigma * np.sqrt(cs[a].scale * cs[b].scale) * depth
                out[o + 4 : o + 7] = (cs[a].point(centre) - cs[b].point(centre)) / unit
            return out

        pattern = lil_matrix((total, 7 * (W - 1)), dtype=np.int8)
        spans = [(link[0], link[1], o0, o1) for link, o0, o1 in zip(active, offsets[:-1], offsets[1:])]
        spans += [(a, b, offsets[-1] + 7 * n, offsets[-1] + 7 * n + 7) for n, (a, b, *_) in enumerate(edges)]
        for a, b, o0, o1 in spans:
            for k in (a, b):
                if k:
                    pattern[o0:o1, 7 * (k - 1) : 7 * k] = 1
        fit = least_squares(residuals, x, jac_sparsity=pattern, x_scale='jac', max_nfev=200)
        return fit.x, dict(status=int(fit.status), evaluations=int(fit.nfev), cost=float(fit.cost))

    def medians(x, active):
        cs = corrections(x)
        return np.array([np.median(np.abs(pair(cs, link))) for link in active])

    x = np.zeros(7 * (W - 1))
    info = dict(pairs=len(links), pairs_dropped=0, solves=[])
    if links and W > 1:
        info['pair_residual_median_before'] = float(np.median(medians(x, links)))
        x, solved = solve(x, links)
        info['solves'].append(solved)
        keep = medians(x, links) <= loss_scale
        if not keep.all() and keep.any():
            links = [link for link, ok in zip(links, keep) if ok]
            x, solved = solve(x, links)
            info['solves'].append(solved)
        info.update(pairs_dropped=int((~keep).sum()), pair_residual_median_after=float(np.median(medians(x, links))))
    cs = corrections(x)
    turn = [float(np.degrees(np.linalg.norm(Rotation.from_matrix(c.R).as_rotvec()))) for c in cs]
    info.update(scale_corrections=[round(c.scale, 4) for c in cs], rotation_corrections_deg=[round(a, 3) for a in turn])
    return cs, info


def track(folder, database, frame_of_image, width, height, rounds=2):
    """(frame ids, c2w after loop closure, c2w of the plain chain, info, VGGT's median focal length in pixels)."""
    windows = load_windows(folder)
    placed, spread = chain(windows)
    H, W = windows[0]['depth'].shape[1:]
    if abs(W / width - H / height) > 0.01 * W / width:
        raise ValueError(f'Depth grid {W}x{H} does not keep the aspect of the {width}x{height} frames')
    _, plain = frame_poses(windows, placed)
    scale = np.array([W / width, H / height])
    links, first = match_links(windows, placed, database, frame_of_image, scale)
    corrections, solved = pose_graph(windows, placed, links)
    if rounds > 1 and first['used']:
        # Second round on the corrected chain: pairs the drifted chain put too far apart can now pass the distance check.
        placed = [(c.scale * s, c.R @ A, c.point(t)) for c, (s, A, t) in zip(corrections, placed)]
        links, counts = match_links(windows, placed, database, frame_of_image, scale)
        corrections, info = pose_graph(windows, placed, links)
        info.update(first_round=dict(match_pairs=first, **{k: solved[k] for k in ('pairs_dropped', 'scale_corrections')}))
    else:
        counts, info = first, solved
    ids, closed = frame_poses(windows, placed, corrections)
    info.update(windows=len(windows), match_pairs=counts, chain_scale_spread_max=max(spread, default=0.0))
    if counts['candidates'] and not counts['used']:
        info['loop_warning'] = 'no matched frame pair passed the checks; the trajectory is the plain chain'
    focal = float(np.median(np.concatenate([w['K'][:, 0, 0] for w in windows])) * width / W)
    return ids, closed, plain, info, focal


def start_camera(camera, focal):
    """Parameters of a COLMAP camera with every focal length set to `focal` and no distortion; the centre stays."""
    params = np.array(camera.params, float)
    params[list(camera.focal_length_idxs())] = focal
    params[list(camera.extra_params_idxs())] = 0
    return params


def refine(c2w, names, database, images, out, sigma, focal, threads=4):
    """Feature BA from the tracked poses with a self-calibrated shared camera, on `threads` threads.

    Returns (c2w, observations per frame, camera, rounds, points xyz, point colours).
    """
    import pycolmap as pc
    from .options import ba_options, known_model, pipeline_options

    data = dict(c2w_L=c2w, image_names=names)
    rec = known_model(data, database)
    for camera in rec.cameras.values():
        camera.params = start_camera(camera, focal)
    lookup = {name: i for i, name in enumerate(names)}
    priors = [
        pc.PosePrior(
            pose_prior_id=im.image_id,
            corr_data_id=im.data_id,
            position=c2w[lookup[im.name], :3, 3],
            position_covariance=np.eye(3) * sigma**2,
            coordinate_system=pc.PosePriorCoordinateSystem.CARTESIAN,
        )
        for im in rec.images.values()
    ]
    options_ba = ba_options()
    options_ba.refine_focal_length = options_ba.refine_extra_params = True
    options_ba.ceres.solver_options.num_threads = threads
    rounds = []
    for iteration, threshold in enumerate((16.0, 8.0, 4.0), 1):
        options = pipeline_options()
        options.num_threads = options.mapper.num_threads = threads
        tri = options.triangulation
        tri.merge_max_reproj_error = tri.complete_max_reproj_error = threshold
        angular = float(np.degrees(np.arctan(threshold / focal)))
        tri.create_max_angle_error = tri.continue_max_angle_error = tri.re_max_angle_error = angular
        options.mapper.filter_max_reproj_error = threshold
        work = out / f'round{iteration}'
        work.mkdir()
        rec = pc.triangulate_points(rec, database, images, work, clear_points=False, options=options, refine_intrinsics=False)
        config = pc.BundleAdjustmentConfig()
        for image_id in rec.reg_image_ids():
            config.add_image(image_id)
        adjuster = pc.create_pose_prior_bundle_adjuster(options_ba, pc.PosePriorBundleAdjustmentOptions(), config, priors, rec)
        summary = adjuster.solve()
        del adjuster
        if not summary.is_solution_usable():
            raise RuntimeError(f'Bundle adjustment returned an unusable solution in round {iteration}')
        filtered = pc.ObservationManager(rec).filter_all_points3D(threshold, 1.5)
        rec.update_point_3d_errors()
        camera = next(iter(rec.cameras.values()))
        rounds.append(
            dict(
                round=iteration,
                threshold_px=threshold,
                points=rec.num_points3D(),
                filtered_observations=int(filtered),
                reprojection_mean_px=rec.compute_mean_reprojection_error(),
                camera_params=np.array(camera.params).tolist(),
            )
        )
        print(json.dumps(rounds[-1]), flush=True)
        shutil.rmtree(work)
    rec.extract_colors_for_all_images(str(images))
    poses = c2w.copy()
    observations = np.zeros(len(poses), np.int64)
    for im in rec.images.values():
        poses[lookup[im.name]][:3] = im.cam_from_world().inverse().matrix()
        observations[lookup[im.name]] = im.num_points3D
    points = np.array([p.xyz for p in rec.points3D.values()], np.float32).reshape(-1, 3)
    colors = (np.array([p.color for p in rec.points3D.values()], float).reshape(-1, 3) / 255).astype(np.float32)
    return poses, observations, next(iter(rec.cameras.values())), rounds, points, colors


def reconstruct(data, manifest, out, settings, depth_settings):
    """Poses of the prepared frames from VGGT windows; return (cameras, report) and write the undistorted manifest.

    Same contract as recon.sfm.reconstruct. `settings` is the `sfm` section (feature and matching options),
    `depth_settings` the `depth` section (weights and window layout). Output is in arbitrary units.
    """
    import cv2
    import pycolmap as pc
    from recon.depth.backends import infer_windows
    from recon.sfm import DEFAULTS, SETTINGS, health, intrinsics, match_features, undistort

    if 'K' in manifest:
        raise ValueError('pose.provider: vggt self-calibrates the camera; calibrated input (manifest K) is not supported yet')
    options = {**SETTINGS, **DEFAULTS['vggt'], **settings}
    cv2.setNumThreads(1)
    ids = data['frame_ids']
    width, height = manifest['resolution']
    windows = out / 'windows'
    windows.mkdir()
    infer_windows(manifest, depth_settings, windows)
    import torch

    torch.cuda.empty_cache()
    database, images, names, loops = match_features(manifest, out, options)
    order = {name: int(frame) for name, frame in zip(names, ids)}
    tracked_ids, tracked, chained, info, focal = track(windows, database, order.get, width, height)
    if not np.array_equal(tracked_ids, ids):
        raise ValueError('Depth windows do not cover the prepared frames exactly once')
    path = float(np.linalg.norm(np.diff(tracked[:, :3, 3], axis=0), axis=1).sum())
    work = out / 'refine.db'
    shutil.copy(database, work)
    c2w, observations, camera, rounds, points, colors = refine(
        tracked, names, work, images, out, PRIOR_FRACTION * path, focal, options['threads']
    )
    work.unlink()
    shutil.rmtree(windows)  # the depth stage infers again on the undistorted frames
    K, distortion = intrinsics(camera)
    turned = np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n', c2w[:, :3, :3], tracked[:, :3, :3]) - 1) / 2, -1, 1)))
    new_K = undistort(manifest, names, images, K, distortion, (width, height), out)
    np.savez(out / 'track.npz', frame_ids=ids, c2w_chain=chained, c2w_loops=tracked, observations=observations)
    report = dict(
        provider='vggt',
        units='arbitrary',
        input_frames=len(ids),
        registered=len(ids),
        registration_rate=1.0,
        dropped_frames=[],
        camera_model=camera.model_name,
        pycolmap_version=pc.__version__,
        **loops,
        track=info,
        vggt_focal_px=focal,
        focal_over_vggt=float(K[0, 0] / focal),
        ba_rotation_change_deg_p50_p90_max=[*np.percentile(turned, [50, 90]).tolist(), float(turned.max())],
        ba_position_change_over_sigma_p50_p90=(np.percentile(np.linalg.norm(c2w[:, :3, 3] - tracked[:, :3, 3], axis=1), [50, 90]) / (PRIOR_FRACTION * path)).tolist(),
        path_length=path,
        position_prior_sigma=PRIOR_FRACTION * path,
        rounds=rounds,
        mean_reprojection_error=rounds[-1]['reprojection_mean_px'],
        points=len(points),
        frames_without_observations=data['frame_ids'][observations == 0].tolist(),
        observations_p10_p50=np.percentile(observations, [10, 50]).tolist(),
        original_K=K.tolist(),
        distortion=distortion.tolist(),
        undistorted_K=new_K.tolist(),
        health=health(c2w, ids),
    )
    cameras = dict(
        frame_ids=ids,
        timestamp_ns=data['timestamp_ns'],
        c2w=c2w,
        K=np.repeat(new_K[None], len(ids), axis=0),
        blur=data['blur'],
        points=points,
        colors=colors,
    )
    dump(out / 'track.json', info)
    return cameras, report
