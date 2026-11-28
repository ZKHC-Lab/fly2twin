# SPDX-License-Identifier: Apache-2.0
import numpy as np


def plan_pairs(poses, temporal=15, radius=1.5, angle_deg=45.0, max_neighbors=30):
    centers, axes = poses[:, :3, 3], poses[:, :3, 2]
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
    eligible = (distances < radius) & (axes @ axes.T > np.cos(np.deg2rad(angle_deg)))
    np.fill_diagonal(eligible, False)
    candidates = set()
    for i in range(len(poses)):
        near = np.flatnonzero(eligible[i])
        near = near[np.argsort(distances[i, near], kind='stable')[:max_neighbors]]
        candidates.update(tuple(sorted((i, int(j)))) for j in near)
    degree = np.zeros(len(poses), dtype=np.int64)
    spatial = []
    for i, j in sorted(candidates, key=lambda pair: (distances[pair], pair)):
        if degree[i] < max_neighbors and degree[j] < max_neighbors:
            spatial.append((i, j))
            degree[[i, j]] += 1
    sequential = {(i, j) for i in range(len(poses)) for j in range(i + 1, min(i + temporal + 1, len(poses)))}
    union = np.asarray(sorted(sequential | set(spatial)), dtype=np.int64).reshape(-1, 2)
    return union
