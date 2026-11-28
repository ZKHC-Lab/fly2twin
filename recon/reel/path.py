# SPDX-License-Identifier: Apache-2.0
"""Designed camera paths: sparse keys (position + look-at point, metric world, caller-provided up) -> smooth frames.

Position, yaw (unwrapped) and pitch are natural cubic splines (C2) over a shared parameter that grows
with both travel and view-direction change, so pure pans also advance. Interpolating angles rather
than look-at points keeps the view direction well defined through turns. Time follows that "motion length" with
a constant cruise speed and smootherstep ease-in/out (continuous acceleration). Roll is zero: the
camera right axis stays horizontal. c2w uses OpenCV axes (x right, y down, z forward).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

ANGLE_WEIGHT_M = 1.8  # one radian of view change counts as 1.8 m: pure pans at 0.55 m/s stay under 18 deg/s
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class Shot:
    name: str
    keys: list  # [dict(p=[x,y,z], look=[x,y,z]), ...] metric world
    speed: float = 0.5  # m/s of motion length at cruise
    ease: float = 1.2  # s of ease-in and of ease-out
    fps: int = 30
    note: str = ''
    meta: dict = field(default_factory=dict)


def _dirs(p, look):
    d = np.asarray(look, float) - np.asarray(p, float)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    if not np.isfinite(d).all() or (n < 1e-10).any():
        raise ValueError('Each look target must be finite and distinct from its camera')
    return d / n


def _ramp_distance(x):
    """Integral of smootherstep from 0 to x (x in [0,1])."""
    return x**6 - 3 * x**5 + 2.5 * x**4


def _angles(d, up=UP):
    ex, ey, ez = _basis(up)
    return np.unwrap(np.arctan2(d @ ey, d @ ex)), np.arcsin(np.clip(d @ ez, -1, 1))


def _direction(yaw, pitch, up=UP):
    ex, ey, ez = _basis(up)
    return (
        (np.cos(pitch) * np.cos(yaw))[..., None] * ex
        + (np.cos(pitch) * np.sin(yaw))[..., None] * ey
        + np.sin(pitch)[..., None] * ez
    )


def sample(shot, up=UP):
    """Frame times, positions and look-at points (1 m ahead along the interpolated direction)."""
    if len(shot.keys) < 2 or shot.speed <= 0 or shot.ease < 0 or shot.fps <= 0:
        raise ValueError('Path requires >=2 keys, positive speed/fps and nonnegative ease')
    P = np.array([k['p'] for k in shot.keys], float)
    D = _dirs(P, np.array([k['look'] for k in shot.keys], float))
    yaw, pitch = _angles(D, up)
    step = np.linalg.norm(np.diff(P, axis=0), axis=1) + ANGLE_WEIGHT_M * np.arccos(
        np.clip((D[1:] * D[:-1]).sum(1), -1, 1)
    )
    u = np.r_[0, np.cumsum(np.maximum(step, 1e-3))]
    sp = CubicSpline(u, P, bc_type='natural')
    sy, sq = CubicSpline(u, yaw, bc_type='natural'), CubicSpline(u, pitch, bc_type='natural')
    dense = np.linspace(0, u[-1], 6000)
    dp, dd = sp(dense), _direction(sy(dense), sq(dense), up)
    m = np.r_[
        0,
        np.cumsum(
            np.linalg.norm(np.diff(dp, axis=0), axis=1)
            + ANGLE_WEIGHT_M * np.arccos(np.clip((dd[1:] * dd[:-1]).sum(1), -1, 1))
        ),
    ]
    M, v, te = m[-1], shot.speed, shot.ease
    te = min(te, M / v)  # very short shots: ramps meet in the middle
    T = M / v + te
    t = np.arange(int(round(T * shot.fps)) + 1) / shot.fps
    safe_te = max(te, 1e-12)
    s = np.where(
        t < te,
        v * te * _ramp_distance(np.clip(t / safe_te, 0, 1)),
        np.where(t > T - te, M - v * te * _ramp_distance(np.clip((T - t) / safe_te, 0, 1)), v * te / 2 + v * (t - te)),
    )
    uu = np.interp(np.clip(s, 0, M), m, dense)
    pos = sp(uu)
    return t, pos, pos + _direction(sy(uu), sq(uu), up)


def c2w(pos, look, up=UP):
    f = _dirs(pos, look)
    r = np.cross(f, up)
    if (np.linalg.norm(r, axis=-1) < 1e-8).any():
        raise ValueError('Look direction parallel to world_up has undefined zero-roll orientation')
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    d = np.cross(f, r)
    out = np.tile(np.eye(4), (len(pos), 1, 1))
    out[:, :3, 0], out[:, :3, 1], out[:, :3, 2], out[:, :3, 3] = r, d, f, pos
    return out


def _basis(up):
    up = np.asarray(up, float) / np.linalg.norm(up)
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    return a, np.cross(up, a), up


def check(t, pos, look, fps, up=UP):
    """Motion limits: speed, yaw rate, pitch, linear jerk (finite differences at the frame rate)."""
    f = _dirs(pos, look)
    ex, ey, ez = _basis(up)
    yaw = np.unwrap(np.arctan2(f @ ey, f @ ex))
    pitch = np.degrees(np.arcsin(np.clip(f @ ez, -1, 1)))
    vel = np.gradient(pos, 1 / fps, axis=0)
    acc = np.gradient(vel, 1 / fps, axis=0)
    jerk = np.gradient(acc, 1 / fps, axis=0)
    yaw_rate = np.degrees(np.abs(np.gradient(yaw, 1 / fps)))
    speed = np.linalg.norm(vel, axis=1)
    return dict(
        seconds=float(t[-1]),
        frames=len(t),
        speed_max=float(speed.max()),
        speed_cruise_median=float(np.median(speed[len(speed) // 4 : 3 * len(speed) // 4])),
        yaw_rate_max_deg_s=float(yaw_rate.max()),
        pitch_min_deg=float(pitch.min()),
        pitch_max_deg=float(pitch.max()),
        jerk_p99=float(np.percentile(np.linalg.norm(jerk, axis=1), 99)),
        acc_max=float(np.linalg.norm(acc, axis=1).max()),
    )


def limits_ok(c, speed=(0.0, 0.85), yaw=20.0, pitch=20.0):
    return (
        c['speed_max'] <= speed[1]
        and c['yaw_rate_max_deg_s'] <= yaw
        and -pitch <= c['pitch_min_deg']
        and c['pitch_max_deg'] <= pitch
    )
