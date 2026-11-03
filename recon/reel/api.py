# SPDX-License-Identifier: Apache-2.0
"""The six supported reel operations. GPU imports are local to rendering operations."""
from pathlib import Path

from .common import dump, resolve, specification


def shot(spec, out, *, preview=False, only=None):
    """Render a shot specification; return per-shot audits. See docs/REEL.md."""
    import cv2
    import numpy as np
    from recon.io.scene import load_scene
    from . import path as P, render as R
    from .coverage import Coverage, stats
    from .grade import grade
    from .templates import make
    cfg, base = specification(spec)
    sc = load_scene(resolve(base, cfg['scene']))
    cov = Coverage(sc)
    splats = cov.splats
    w, h = (640, 360) if preview else (cfg.get('width', 1920), cfg.get('height', 1080))
    fps = cfg.get('fps', 30)
    out = Path(out)
    results = {}
    points = None
    for entry in cfg['shots']:
        if only and entry['name'] not in only:
            continue
        s = dict(entry)
        if 'template' in s:
            s = dict(make(sc, dict(s['template'], name=s['name'])), **{k: v for k, v in s.items() if k != 'template'})
        name = s['name']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Shot name must be one directory component')
        path = P.Shot(name, s['keys'], speed=s.get('speed', .5), ease=s.get('ease', 1.2), fps=fps)
        t, pos, look = P.sample(path, sc.world_up)
        poses = P.c2w(pos, look, sc.world_up).astype(np.float32)
        motion = P.check(t, pos, look, fps, sc.world_up)
        K, hfov = R.intrinsics(sc, w, h, s.get('hfov_deg'))
        Kc = R.intrinsics(sc, sc.width, sc.height, s.get('hfov_deg'))[0]
        mode = s.get('mode', 'gs')
        if mode not in ('gs', 'points', 'match', 'dissolve'):
            raise ValueError(f'Unknown render mode: {mode}')
        if mode != 'gs':
            pts, rgb = sc.points()
            points = R.point_splats(dict(points=pts, colors=rgb), sc, s.get('point_size', .012))
        bg = s.get('background', cfg.get('background', [.055, .067, .086]))
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        audit, tiles = [], []
        # Optional sparse selection is useful for bounded regression without changing the path timing.
        indices = s.get('frames', list(range(0, len(t), 3 if preview else 1)))
        for i in indices:
            ti = t[i]
            if mode == 'points' or (mode == 'match' and ti < s['cut_s']):
                rgb = R.draw(points, poses[i], K, w, h, degree=0, bg=bg)
            elif mode == 'dissolve' and ti < s['cut_s'] + s.get('fade_s', .6):
                a = np.clip((ti - s['cut_s']) / s.get('fade_s', .6), 0, 1)
                rgb = (1-a)*R.draw(points, poses[i], K, w, h, degree=0, bg=bg) + a*R.draw(splats, poses[i], K, w, h)
            else:
                rgb = R.draw(splats, poses[i], K, w, h)
            img = grade((rgb * 255).round().astype(np.uint8), s.get('grade', cfg.get('grade')))
            cv2.imwrite(str(d / f'{i:05d}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if i % 10 == 0 or (preview and i % 9 == 0) or not audit:
                st = stats(cov.frame(poses[i], Kc, sc.width, sc.height)[0])
                audit.append(dict(frame=i, t=float(ti), tier=R.tier(st), **st))
            if i % fps == 0 or not tiles:
                tiles.append((img, f'{ti:.1f}s {audit[-1]["tier"]}'))
        counts = {k: sum(x['tier'] == k for x in audit) for k in 'ABC'}
        result = dict(shot=s, motion=motion, motion_ok=P.limits_ok(motion), hfov_deg=hfov,
                      tiers=counts, frames=len(t), rendered_frames=len(indices), audit=audit)
        dump(d / 'audit.json', result)
        contact(tiles, out / f'{name}_contact.jpg')
        results[name] = result
    return results


def contact(tiles, out, columns=5):
    import cv2
    import numpy as np
    if not tiles:
        return
    rows = []
    for img, label in tiles:
        im = cv2.cvtColor(cv2.resize(img, (384, 216)), cv2.COLOR_RGB2BGR)
        bar = np.full((24, 384, 3), 255, np.uint8)
        cv2.putText(bar, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        rows.append(np.vstack([im, bar]))
    while len(rows) % columns:
        rows.append(np.full_like(rows[0], 255))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([np.hstack(rows[i:i+columns]) for i in range(0, len(rows), columns)]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def search(scene, templates, out, *, step=6, min_score=.9):
    """Search hero cameras, accepting only sampled paths without C and score >= min_score."""
    import numpy as np
    from recon.io.scene import load_scene
    from . import render as R, path as P
    from .coverage import Coverage
    from .templates import make
    from ._search import evaluate
    if step < 1 or not 0 <= min_score <= 1:
        raise ValueError('step must be positive and min_score in [0,1]')
    specs, _ = specification(templates)
    sc = load_scene(scene)
    cov = Coverage(sc)
    K = R.intrinsics(sc, sc.width, sc.height)[0]
    result, tiles = {}, []
    for spec in specs:
        candidates = []
        for hero in range(0, len(sc.train_c2w), step):
            s = make(sc, dict(spec, hero=hero))
            r = evaluate(sc, cov, s, K)
            if r is not None and r['C'] == 0 and r['score'] >= min_score:
                candidates.append(dict(hero=hero, **r))
        candidates.sort(key=lambda c: -c['score'])
        result[spec['name']] = dict(spec=spec, best=candidates, tried=len(range(0, len(sc.train_c2w), step)))
        for cand in candidates[:6]:
            s = make(sc, dict(spec, hero=cand['hero']))
            t, pos, look = P.sample(P.Shot('x', s['keys'], s['speed'], s['ease']), sc.world_up)
            poses = P.c2w(pos, look, sc.world_up).astype(np.float32)
            for i in (0, len(t)//2, len(t)-1):
                rgb = R.draw(cov.splats, poses[i], R.intrinsics(sc, 384, 216)[0], 384, 216)
                tiles.append(((rgb*255).round().astype(np.uint8), f'{spec["name"]} hero {cand["hero"]} {i}'))
    dump(out, result)
    contact(tiles, Path(out).with_suffix('.jpg'), columns=3)
    return result


def growth(scene, out, *, start_frac=0., grow_s=5.3, dissolve_s=2.8, height=4.5, back=6.5,
           orbit_deg=40., trail_smooth=0., stable=False, point_size=.012, grade=None,
           background=(.055, .067, .086), trail_gain=1., width=1920, fps=30, frames=None):
    """Render first-visible cloud growth. dissolve_s holds the final cloud while the trail fades."""
    from ._growth import run
    return run(scene, out, start_frac=start_frac, grow_s=grow_s, dissolve_s=dissolve_s, height=height,
               back=back, orbit_deg=orbit_deg, trail_smooth=trail_smooth, stable=stable,
               point_size=point_size, grade_params=grade, background=background, trail_gain=trail_gain,
               width=width, fps=fps, frames=frames)


def card(kind, out, **options):
    """Write a 1920x1080 RGBA card; kind-specific keywords are documented in REEL.md."""
    from . import cards
    if kind not in ('headline', 'title', 'label', 'number', 'endcard'):
        raise ValueError(f'Unknown card kind: {kind}')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    getattr(cards, kind)(out=out, **options)
    return str(out)


def cut(storyboard, out):
    """Compose a user-slot storyboard to H.264/AAC and write its actual timeline JSON."""
    from .assemble import encode, timeline
    cfg, base = specification(storyboard)
    edl = timeline(cfg)
    for c in edl['clips']:
        for key in ('src', 'a', 'b'):
            if key in c:
                c[key] = resolve(base, c[key])
        for overlay in c.get('overlays', []):
            overlay['png'] = resolve(base, overlay['png'])
    if edl.get('music'):
        edl['music']['file'] = resolve(base, edl['music']['file'])
    return encode(edl, out)


def beats(music):
    """Return BPM, first beat, beat grid and per-second RMS loudness."""
    from ._beats import analyse
    return analyse(str(music))
