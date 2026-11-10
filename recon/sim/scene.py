# SPDX-License-Identifier: Apache-2.0
"""Exported scene -> closed flight volume -> start/goal pairs.

prepare(): reads cloud.ply of a recon export (metres, Z up), levels the floor to z = 0 and closes the
ceiling with a flat 2 cm sample grid over the floor footprint, skipping 10 cm cells where the capture
already has points at or above ceiling_z - 0.25 m. Real points come first in the output; the synthetic
ceiling follows (observed_count separates them).

sample_pairs(): 0.1 m grid at flight height; a cell is free when it is over observed floor (within 0.6 m)
and at least path_clearance + resolution/sqrt(2) from every point; endpoints need endpoint_clearance and
the same 4-connected component. Pairs are drawn with a fixed seed until `count` pairs exist, the first
`blocked_min` of them with the straight segment blocked (some point closer than 0.1 m).
"""
from collections import deque
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..io.ply import read_ply

FLOOR_BAND = .18          # |z| below this after levelling counts as floor
CEILING_GRID = .02
CEILING_DILATION = .5     # ceiling footprint = floor cells dilated by this distance
REAL_CEILING_BELOW = .25  # real points this far below ceiling_z suppress synthetic samples
REAL_CEILING_CELL = .1


def load_cloud(export):
    export = Path(export)
    meta = json.loads((export / 'scene.json').read_text())
    if meta.get('units', 'meters') != 'meters':
        raise ValueError('Scene units must be meters')
    v = read_ply(export / meta.get('cloud', 'cloud.ply'))
    points = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
    colors = np.column_stack([v['red'], v['green'], v['blue']]).astype(np.uint8)
    if len(points) < 1000 or not np.isfinite(points).all():
        raise ValueError('Invalid scene point cloud')
    return points, colors


def level_floor(points, seed=20261020):
    """Rigid transform (4x4) that puts the dominant low horizontal plane at z = 0 with normal +Z."""
    rng = np.random.default_rng(seed)
    # Ground is the broad low horizontal surface, not the table-height mode.
    low = points[points[:, 2] <= np.percentile(points[:, 2], 30)]
    sample = low[rng.choice(len(low), min(24000, len(low)), replace=False)].astype(float)
    best, best_mask = 0, None
    for _ in range(700):
        p = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        if normal[2] < 0:
            normal *= -1
        if normal[2] < np.cos(np.deg2rad(25)):
            continue
        mask = np.abs(sample @ normal - p[0] @ normal) < .07
        if mask.sum() > best:
            best, best_mask = int(mask.sum()), mask
    if best_mask is None:
        raise ValueError('No floor plane within 25 degrees of +Z')
    for _ in range(4):
        fit = sample[best_mask]
        center = fit.mean(0)
        _, _, vt = np.linalg.svd(fit - center, full_matrices=False)
        normal = vt[-1] * (1 if vt[-1, 2] > 0 else -1)
        d = -normal @ center
        best_mask = np.abs(sample @ normal + d) < .07
    angle = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
    rotation = np.eye(3)
    if angle > 3:
        cross = np.cross(normal, [0., 0., 1.])
        skew = np.array([[0., -cross[2], cross[1]], [cross[2], 0., -cross[0]], [-cross[1], cross[0], 0.]])
        rotation += skew + skew @ skew / (1 + normal[2])
    reference_xy = np.median(sample[best_mask, :2], axis=0)
    ground_z = float(-(d + normal[:2] @ reference_xy) / normal[2])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[2, 3] = d if angle > 3 else -ground_z
    info = dict(floor_normal_angle_deg=angle, rotated=angle > 3, floor_inlier_fraction=float(best_mask.mean()))
    return transform, info


def ceiling_samples(observed, ceiling_z):
    floor = observed[np.abs(observed[:, 2]) < FLOOR_BAND]
    if len(floor) < 100:
        raise ValueError('Too few floor points after levelling')
    cells = np.floor(floor[:, :2].astype(np.float64) / CEILING_GRID).astype(np.int64)
    padding = int(np.ceil(CEILING_DILATION / CEILING_GRID)) + 1
    low, high = cells.min(0) - padding, cells.max(0) + padding
    occupancy = np.zeros((high - low + 1)[::-1], bool)
    relative = cells - low
    occupancy[relative[:, 1], relative[:, 0]] = True
    distance = ndimage.distance_transform_edt(~occupancy, sampling=CEILING_GRID)
    y, x = np.nonzero(distance <= CEILING_DILATION + 1e-12)
    origin = low * CEILING_GRID
    ceiling = np.column_stack([origin[0] + (x + .5) * CEILING_GRID, origin[1] + (y + .5) * CEILING_GRID,
                               np.full(len(x), ceiling_z)]).astype(np.float32)
    # Keep real ceiling observations; no synthetic sample in a 10 cm cell that has one.
    real = observed[observed[:, 2] >= round(ceiling_z - REAL_CEILING_BELOW, 6)]
    real_cells = np.floor(real[:, :2].astype(float) / REAL_CEILING_CELL).astype(np.int64)
    ceiling_cells = np.floor(ceiling[:, :2].astype(float) / REAL_CEILING_CELL).astype(np.int64)
    origin_cell = np.minimum(real_cells.min(0, initial=0), ceiling_cells.min(0))
    span = int(max(real_cells.max(initial=0), ceiling_cells.max()) - origin_cell.min()) + 1
    key = lambda c: (c[:, 0] - origin_cell[0]) * span + (c[:, 1] - origin_cell[1])
    covered = np.isin(key(ceiling_cells), key(real_cells))
    return ceiling[~covered]


def prepare(export, ceiling_z=None, seed=20261020):
    """Return the scene dict saved by `recon sim prepare` (points in the levelled frame)."""
    points, colors = load_cloud(export)
    transform, info = level_floor(points, seed)
    observed = (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
    if ceiling_z is None:   # top of the reconstruction, rounded down to 5 cm
        ceiling_z = float(np.floor(np.percentile(observed[:, 2], 99.9) / .05) * .05)
    ceiling = ceiling_samples(observed, ceiling_z)
    info.update(ceiling_z=ceiling_z, observed_points=len(observed), synthetic_ceiling_points=len(ceiling))
    return dict(points=np.concatenate([observed, ceiling]),
                colors=np.concatenate([colors, np.full(ceiling.shape, 190, np.uint8)]),
                observed_count=np.int64(len(observed)), ceiling_z=np.float64(ceiling_z),
                world_from_export=transform, export_from_world=np.linalg.inv(transform),
                info=np.array(json.dumps(info)))


def _bfs(free, start, goal):
    height, width = free.shape
    parent = np.full(free.size, -1, np.int64)
    parent[start] = start
    queue, flat = deque([start]), free.ravel()
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        y, x = divmod(cell, width)
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width:
                n = ny * width + nx
                if flat[n] and parent[n] == -1:
                    parent[n] = cell
                    queue.append(n)
    if parent[goal] == -1:
        return []
    route = [goal]
    while route[-1] != start:
        route.append(int(parent[route[-1]]))
    return route[::-1]


def sample_pairs(scene, count=50, seed=20261030, height=1.2, resolution=.1, path_clearance=.6,
                 endpoint_clearance=.9, distance_range=(6., 20.), blocked_min=None, max_proposals=100000):
    blocked_min = count // 2 if blocked_min is None else blocked_min
    observed = scene['points'][:int(scene['observed_count'])]
    points = np.ascontiguousarray(scene['points'], np.float32)
    origin = np.floor(observed[:, :2].min(0) / resolution) * resolution
    stop = np.ceil(observed[:, :2].max(0) / resolution) * resolution
    h, w = (np.rint((stop - origin) / resolution).astype(int) + 1)[::-1]
    xx, yy = np.meshgrid(float(origin[0]) + np.arange(w) * resolution, float(origin[1]) + np.arange(h) * resolution)
    query = np.column_stack([xx.ravel(), yy.ravel(), np.full(h * w, height)])
    tree = cKDTree(points)
    distance = tree.query(query, workers=1)[0].reshape(h, w)
    floor = observed[np.abs(observed[:, 2]) < FLOOR_BAND]
    floor_distance = cKDTree(floor[:, :2]).query(query[:, :2], workers=1)[0].reshape(h, w)
    known = (floor_distance <= .6) & (distance <= 3)
    free = (distance >= path_clearance + resolution / np.sqrt(2)) & known
    candidates = np.flatnonzero(free & (distance >= endpoint_clearance))
    labels = ndimage.label(free)[0].ravel()
    rng = np.random.default_rng(seed)
    pairs, routes, seen, blocked_count, proposals = [], [], set(), 0, 0
    for _ in range(max_proposals):
        if len(pairs) == count or len(candidates) < 2:
            break
        proposals += 1
        a, b = rng.choice(candidates, 2, replace=False)
        key = tuple(sorted((int(a), int(b))))
        if key in seen or labels[a] != labels[b]:
            continue
        start, goal = query[a], query[b]
        length = float(np.linalg.norm(goal - start))
        if not distance_range[0] <= length <= distance_range[1]:
            continue
        line = np.linspace(start, goal, int(np.ceil(length / .025)) + 1)
        line_min = float(tree.query(line, workers=1)[0].min())
        blocked = line_min < .1
        if blocked_count < blocked_min and not blocked:
            continue
        route = _bfs(free, int(a), int(b))
        seen.add(key)
        pairs.append((int(a), int(b), length, blocked, (len(route) - 1) * resolution))
        routes.append(query[route].astype(np.float32))
        blocked_count += int(blocked)
    if len(pairs) < count or blocked_count < blocked_min:
        raise RuntimeError(f'Only {len(pairs)} pairs ({blocked_count} blocked) after {proposals} proposals; '
                           'the free space may be too small for the distance range')
    cells = np.asarray([p[:2] for p in pairs], np.int64)
    starts, goals = query[cells[:, 0]].astype(np.float32), query[cells[:, 1]].astype(np.float32)
    back = scene['export_from_world']
    return dict(starts=starts, goals=goals,
                starts_export=starts @ back[:3, :3].T + back[:3, 3], goals_export=goals @ back[:3, :3].T + back[:3, 3],
                distances=np.asarray([p[2] for p in pairs], np.float32),
                straight_blocked=np.asarray([p[3] for p in pairs], bool),
                shortest_lengths=np.asarray([p[4] for p in pairs], np.float32),
                path_points=np.concatenate(routes),
                path_offsets=np.r_[0, np.cumsum([len(r) for r in routes])].astype(np.int64),
                seed=np.int64(seed), height=np.float64(height), proposals=np.int64(proposals))
