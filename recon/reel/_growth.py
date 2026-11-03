# SPDX-License-Identifier: Apache-2.0
"""First-visible capture-camera indices for cloud growth."""
import numpy as np
import torch
from .coverage import low_K, render_depth, LOW_W, LOW_H

def birth_times(sc, pts_norm):
    K = np.stack([low_K(k, sc.width, sc.height) for k in sc.train_K])
    unit = sc.meta.get('world_from_normalized', {}).get('meters_per_unit', 1.)
    depth, alpha = render_depth(sc.torch_splats(), sc.train_c2w, K, near_plane=.01 * unit)
    depth[alpha < 0.5] = float('inf')
    depth = depth.cuda()
    w2c = torch.linalg.inv(torch.from_numpy(sc.train_c2w.astype(np.float32)).cuda())
    X = torch.from_numpy(pts_norm.astype(np.float32)).cuda()
    birth = torch.full((len(X),), len(K), dtype=torch.int64, device='cuda')
    Kt = torch.from_numpy(K).cuda()
    for i in range(len(K)):
        cam = X @ w2c[i, :3, :3].T + w2c[i, :3, 3]
        z = cam[:, 2]
        u = (Kt[i, 0, 0] * cam[:, 0] / z.clamp_min(1e-6) + Kt[i, 0, 2]).round().long()
        v = (Kt[i, 1, 1] * cam[:, 1] / z.clamp_min(1e-6) + Kt[i, 1, 2]).round().long()
        ok = (z > 0.02 * unit) & (u >= 0) & (u < LOW_W) & (v >= 0) & (v < LOW_H)
        ref = depth[i, v.clamp(0, LOW_H - 1), u.clamp(0, LOW_W - 1)]
        seen = ok & (z <= ref * 1.05) & (birth == len(K))
        birth[seen] = i
    return birth.cpu().numpy()



def run(scene, out, **kwargs):
    from pathlib import Path
    from types import SimpleNamespace
    from scipy.ndimage import gaussian_filter1d
    import cv2
    from recon.io.scene import load_scene
    from . import path as P, render as R
    from .grade import grade
    from .common import dump
    a = SimpleNamespace(**kwargs, out=Path(out), head_weight=.75)
    if not 0 <= a.start_frac <= 1 or a.grow_s <= 0 or a.dissolve_s < 0 or a.point_size <= 0:
        raise ValueError('Require start_frac in [0,1], grow_s/point_size > 0 and dissolve_s >= 0')
    sc = load_scene(scene)
    pts, colors = sc.points()
    cloud = dict(points=pts, colors=colors)
    pts_norm = cloud['points']
    birth = birth_times(sc, pts_norm)
    n_cam = len(sc.train_c2w)
    # Points the strict low-res visibility test never assigns (thin/occluded geometry): born with the
    # nearest training camera centre, so the finished cloud is complete.
    from scipy.spatial import cKDTree
    unborn = birth == n_cam
    if unborn.any():
        birth[unborn] = cKDTree(sc.train_c2w[:, :3, 3]).query(pts_norm[unborn])[1]
    splats = R.point_splats(cloud, sc, size_m=a.point_size)
    base_op = splats['opacities'].clone()
    birth_t = torch.from_numpy(birth).cuda()
    cams = sc.train_c2w[:, :3, 3].astype(float)
    up = sc.world_up / np.linalg.norm(sc.world_up)
    centre = np.median(cams, 0)
    fps = a.fps
    w, h = a.width, round(a.width * 9 / 16)
    K, _ = R.intrinsics(sc, w, h, hfov=60.0)
    total = a.grow_s + a.dissolve_s
    nfr = int(round(total * fps))
    a.out.mkdir(parents=True, exist_ok=True)
    ex, ey, _ = P._basis(up)
    head_s = gaussian_filter1d(cams, 15, axis=0, mode='nearest')
    trail_cams = gaussian_filter1d(cams, a.trail_smooth, axis=0, mode='nearest') if a.trail_smooth > 0 else cams
    bg = np.array(a.background, np.float32)
    timeline = []
    for f in (a.frames if a.frames is not None else range(nfr)):
        t = f / fps
        g = min(1.0, t / a.grow_s)
        g_s = g * g * (3 - 2 * g) * 0.15 + g * 0.85            # slight ease at the start
        ci = int(round((a.start_frac + (1 - a.start_frac) * g_s) * (n_cam - 1)))
        head = head_s[ci]
        target = centre if a.stable else a.head_weight * head + (1 - a.head_weight) * centre
        az = np.radians(-a.orbit_deg / 2 + a.orbit_deg * t / total)
        horiz = np.cos(az) * ex + np.sin(az) * ey
        pos = target - horiz * a.back + up * a.height
        pose = P.c2w(pos[None], target[None], up)[0].astype(np.float32)
        # Points born up to ci; newest ones fade in over ~20 cameras.
        age = (ci - birth_t).float()
        vis = torch.clamp((age + 1) / 20.0, 0, 1)
        splats['opacities'] = torch.where(birth_t <= ci, base_op + torch.log(vis.clamp_min(1e-4)), torch.full_like(base_op, -20.0))
        rgb = R.draw(splats, pose, K, w, h, degree=0, bg=bg.tolist())
        img = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
        img = grade(img, a.grade_params)
        # Light trail: capture path up to ci, projected; additive glow, fades out during the dissolve.
        trail_alpha = 1.0 if t <= a.grow_s else max(0.0, 1 - (t - a.grow_s) / 1.2)
        if trail_alpha > 0 and ci > 1:
            P3 = trail_cams[:ci + 1]
            w2c = np.linalg.inv(pose.astype(np.float64))
            cam = P3 @ w2c[:3, :3].T + w2c[:3, 3]
            ok = cam[:, 2] > 0.05
            uv = (cam[:, :2] / cam[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            layer = np.zeros((h, w, 3), np.float32)
            pts2 = uv[ok].astype(np.int32)
            col = np.array([255, 196, 92], np.float32)  # warm light in RGB
            if len(pts2) > 1:
                cv2.polylines(layer, [pts2.reshape(-1, 1, 2)], False, (col * 0.55).tolist(), max(1, w // 480), cv2.LINE_AA)
                glow = cv2.GaussianBlur(layer, (0, 0), w / 320)
                layer = layer + glow * 1.6
                hx, hy = pts2[-1]
                cv2.circle(layer, (int(hx), int(hy)), max(2, w // 320), (255, 245, 220), -1, cv2.LINE_AA)
                layer = layer + cv2.GaussianBlur(layer, (0, 0), w / 160) * 0.8
            img = np.clip(img.astype(np.float32) + layer * trail_alpha * a.trail_gain, 0, 255).astype(np.uint8)
        cv2.imwrite(str(a.out / f'{f:05d}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        timeline.append(dict(frame=f, t=t, camera_index=ci, points_visible=int((birth <= ci).sum())))
    result = dict(scene=str(scene), points=len(birth), unborn_assigned_nearest=int(unborn.sum()), frames=timeline)
    dump(a.out / 'timeline.json', result)
    return result
