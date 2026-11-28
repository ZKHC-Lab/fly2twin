# SPDX-License-Identifier: Apache-2.0
import numpy as np


def angle(rotation):
    return np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))


def select(times, poses, valid=None, translation_m=0.15, rotation_deg=8.0, max_gap_s=2.0, trim_static=True):
    times = np.asarray(times, np.int64)
    if np.any(np.diff(times) <= 0):
        raise ValueError('Candidate timestamps must strictly increase')
    valid = np.ones(len(times), bool) if valid is None else np.asarray(valid, bool)
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError('No valid candidate frames')
    if trim_static:
        first = ids[0]
        moving = [
            i
            for i in ids
            if np.linalg.norm(poses[i, :3, 3] - poses[first, :3, 3]) >= translation_m
            or angle(poses[first, :3, :3].T @ poses[i, :3, :3]) >= rotation_deg
        ]
        if not moving:
            raise ValueError('No translational or rotational motion')
        ids = ids[ids >= moving[0]]
    selected, previous = [int(ids[0])], int(ids[0])
    gap = int(max_gap_s * 1e9)
    for value in ids[1:]:
        i, a = int(value), selected[-1]
        if times[i] - times[a] > gap:
            if previous == a or times[previous] - times[a] > gap:
                raise ValueError('Candidate gap exceeds strict maximum keyframe interval')
            selected.append(previous)
            a = previous
            if times[i] - times[a] > gap:
                raise ValueError('Candidate gap exceeds strict maximum keyframe interval')
        if (
            np.linalg.norm(poses[i, :3, 3] - poses[a, :3, 3]) >= translation_m
            or angle(poses[a, :3, :3].T @ poses[i, :3, :3]) >= rotation_deg
            or times[i] - times[a] >= gap
        ):
            selected.append(i)
        previous = i
    return np.asarray(selected, np.int64)


def blur_mask(sharpness, radius=10, ratio=0.5):
    sharpness = np.asarray(sharpness)
    local = np.array([np.median(sharpness[max(0, i - radius) : i + radius + 1]) for i in range(len(sharpness))])
    return sharpness < ratio * local, local
