# SPDX-License-Identifier: Apache-2.0
"""Metric hero-camera templates; arbitrary world up."""
import numpy as np

def frame_axes(sc, i, test=False):
    m = sc.test_c2w if test else sc.train_c2w
    c = m[i, :3, 3].astype(float)
    up = sc.world_up / np.linalg.norm(sc.world_up)
    f = m[i, :3, 2].astype(float).copy()
    f -= up * (f @ up)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    return c, f, r


def make(sc, s):
    if 'test_frame' in s:   # start exactly at a held-out test camera (for the real/render wipe)
        i = int(np.flatnonzero(sc.test_ids == s['test_frame'])[0])
        c0 = sc.test_c2w[i, :3, 3].astype(float)
        look0 = c0 + sc.test_c2w[i, :3, 2] * s.get('dist', 2.0)
        c, f, r = frame_axes(sc, i, test=True)
        z = sc.world_up / np.linalg.norm(sc.world_up)
        keys = [dict(p=c0.tolist(), look=look0.tolist())]
        for u in (0.35, 0.7, 1.0):
            keys.append(dict(p=(c0 + f * s.get('forward', 0.8) * u + z * s.get('rise', 0.8) * u).tolist(),
                             look=(look0 + f * s.get('forward', 0.8) * u - z * s.get('drop', 0.6) * u).tolist()))
        out = {k: v for k, v in s.items() if k not in ('test_frame', 'type')}
        out.update(keys=keys, speed=s.get('speed', 0.4), ease=s.get('ease', 1.5))
        return out
    c, f, r = frame_axes(sc, s['hero'])
    z = sc.world_up / np.linalg.norm(sc.world_up)
    dist = s.get('dist', 2.0)
    c = c + z * s.get('dz', 0.0) + f * s.get('shift_f', 0.0) + r * s.get('shift_r', 0.0)
    T = c + f * dist + z * s.get('target_dz', -0.15)
    typ = s['type']
    if typ == 'truck':
        L = s.get('length', 2.0)
        keys = [dict(p=(c + r * u).tolist(), look=(T + r * u).tolist()) for u in np.linspace(-L / 2, L / 2, 4) * s.get('direction', 1)]
    elif typ == 'push':
        keys = [dict(p=(c + f * u).tolist(), look=(c + f * (u + 3.0) + z * s.get('target_dz', -0.15)).tolist())
                for u in np.linspace(-s.get('back', 1.0), s.get('forward', 1.5), 4)]
    elif typ == 'arc':
        keys = []
        for a in np.radians(np.linspace(s.get('a0', -30), s.get('a1', 30), 5)):
            d = -(f * np.cos(a) + r * np.sin(a))
            keys.append(dict(p=(T + d * dist + z * ((c - T) @ z)).tolist(), look=T.tolist()))
    elif typ == 'crane':
        keys = [dict(p=(c + f * s.get('forward', 1.0) * u + z * s.get('rise', 1.0) * u).tolist(),
                     look=(T + z * (-s.get('drop', 0.8)) * u).tolist()) for u in np.linspace(0, 1, 4)]
    elif typ == 'pan':
        keys = []
        for a in np.radians(np.linspace(s.get('a0', -30), s.get('a1', 30), 4)):
            d = f * np.cos(a) + r * np.sin(a)
            keys.append(dict(p=c.tolist(), look=(c + d * dist + z * s.get('target_dz', -0.15)).tolist()))
    else:
        raise ValueError(typ)
    out = {k: v for k, v in s.items() if k not in ('hero', 'type')}
    out.update(keys=keys, speed=s.get('speed', 0.5), ease=s.get('ease', 1.0))
    return out

