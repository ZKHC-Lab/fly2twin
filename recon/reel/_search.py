# SPDX-License-Identifier: Apache-2.0
"""Sampled path coverage scoring."""
import numpy as np
from . import path as P
from . import render as R
from .coverage import stats

def evaluate(sc, cov, shot_spec, K):
    shot = P.Shot('x', shot_spec['keys'], speed=shot_spec.get('speed', 0.5), ease=shot_spec.get('ease', 1.0))
    t, pos, look = P.sample(shot, sc.world_up)
    motion = P.check(t, pos, look, shot.fps, sc.world_up)
    if not P.limits_ok(motion, speed=(0, 0.85), yaw=20.0, pitch=25.0):
        return None
    poses = P.c2w(pos, look, sc.world_up).astype(np.float32)
    tiers = []
    for i in range(0, len(t), 10):
        st = stats(cov.frame(poses[i], K, sc.width, sc.height)[0])
        tiers.append(R.tier(st))
    a, b, c = (tiers.count(x) for x in 'ABC')
    return dict(score=(a + 0.5 * b) / len(tiers), A=a, B=b, C=c, seconds=float(t[-1]))

