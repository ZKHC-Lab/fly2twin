# SPDX-License-Identifier: Apache-2.0
"""BEST numerical loop: float32 cameras, shuffled epochs, optional camera fitting."""
from dataclasses import asdict
from functools import lru_cache
import json
import time
import random
import resource
import numpy as np
import torch
import torch.nn.functional as F
import gsplat
from .strategy import BaselineStrategy
from fused_ssim import fused_ssim
from .render import initialize, render
from .losses import anisotropy
from .schedule import schedule
from recon.io.common import dump


def memory_stats():
    return dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                peak_cpu_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)

def train(data, out, folder, image, settings):
    seed = int(settings.get("seed", 42))
    plan = schedule(len(data["train_indices"]), settings.get("steps"), **{
        k: settings[k] for k in ('exact_steps', 'scale_densification', 'schedule_rounding', 'sh_interval')
        if k in settings})
    steps = plan["steps"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import cv2
    train_ids, w, h = data['train_indices'], data['width'], data['height']
    depth_lambda = float(settings.get('depth_lambda', .1))
    if depth_lambda:
        with np.load(folder / 'frames.npz', allow_pickle=False) as z:
            if not np.array_equal(z['frame_ids'], data['frame_ids']):
                raise ValueError(f'Prior frame order mismatch: {folder}')
        depths = np.load(folder / 'depth_m.npy', mmap_mode='r', allow_pickle=False)
        good = np.load(folder / 'good.npy', mmap_mode='r', allow_pickle=False)
        native = tuple(settings.get('depth_shape', [288, 512]))
        if depths.shape != (len(data['frame_ids']), *native) or good.shape != depths.shape or good.dtype != np.bool_:
            raise ValueError(f'Expected native {native} depth_m and boolean good cache: {folder}')

    @lru_cache(maxsize=32)
    def prior(index):
        d = cv2.resize(depths[index], (w, h), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(good[index].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        m &= np.isfinite(d) & (d > 0.3) & (d < 8)
        if m.sum() < 2:
            raise ValueError(f'Insufficient prior pixels: frame {data["frame_ids"][index]}')
        return d, m

    splats, optimizers = initialize(data['points'], data['colors'], data['scene_scale'])
    camera = None
    if settings.get('pose_opt', False):
        from .camera import CameraResidual
        camera = CameraResidual(len(train_ids)).cuda()
        camera_optimizer = torch.optim.Adam(camera.parameters(), lr=float(settings.get('pose_lr', 1e-5)))
    strategy = BaselineStrategy(**plan["strategy"], reset_at_zero=settings.get('reset_at_zero', False))
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=data['scene_scale'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers['means'], gamma=0.01 ** (1 / steps))
    config = dict(group=data['group'], camera_group=data['camera_group'], steps=steps, seed=seed,
                  resolution=[w, h], sh_degree=3, sh_degree_interval=plan["sh_interval"], packed=False,
                  pose_opt=camera is not None, appearance='none', background='black', depth_source='Omega' if depth_lambda else 'none',
                  rgb_loss='0.8 L1 + 0.2 (1-fused_ssim padding=valid)', depth_lambda=depth_lambda, depth_min_m=0.3, aniso_lambda=settings.get("aniso_lambda", .1),
                  depth_loss='1-Pearson(1/expected_optical_z_m,1/prior_depth_m), per frame',
                  prior_resize='native scaled depth bilinear; good nearest; finite 0.3<d<8 m',
                  scene_scale=data['scene_scale'], normalization_center=data['center'],
                  meters_per_unit=data['meters_per_unit'], strategy=asdict(strategy),
                  gsplat_version=gsplat.__version__, torch_version=torch.__version__,
                  cpu_image_cache_frames=int(data.get('image_cache_frames',32)), cpu_prior_cache_frames=32)
    config['training_settings'] = dict(settings)
    if not depth_lambda:
        config.update(depth_loss=None, prior_resize=None, cpu_prior_cache_frames=0)
    if data.get('arbitrary_scale', False):
        config['coordinate_units'] = 'arbitrary SfM units'
        config['sfm_units_per_normalized_unit'] = data['meters_per_unit']
    dump(out / 'config.json', config)
    order_rng = np.random.default_rng(seed)
    if settings.get('discard_initial_permutation', False):
        order_rng.permutation(len(train_ids))
    visits, counts = np.zeros(len(train_ids), np.int64), {}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for step in range(steps):
        if step % len(train_ids) == 0:
            order = order_rng.permutation(len(train_ids))
        local = int(order[step % len(train_ids)])
        index = int(train_ids[local])
        target = torch.from_numpy(image(int(data['frame_ids'][index]), w, h)).cuda().float()[None] / 255
        if depth_lambda:
            d, mask = prior(index)
            counts[int(data['frame_ids'][index])] = int(mask.sum())
        pose = data['c2w'][index:index+1]
        if camera is not None:
            pose = camera(pose, [local])
        prediction, _, info = render(splats, pose, data['K'][index:index+1],
                                     w, h, degree=min(step // plan["sh_interval"], 3), depth=bool(depth_lambda))
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        rgb = prediction[..., :3]
        rgb_loss = torch.lerp(F.l1_loss(rgb, target), 1 - fused_ssim(
            rgb.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2), padding='valid'), 0.2)
        depth_loss = rgb_loss.new_zeros(())
        if depth_lambda and mask.sum() >= 2:
            mask = torch.from_numpy(mask).cuda()
            prior_d = torch.from_numpy(d).cuda()
            rendered = prediction[0, ..., 3] * data['meters_per_unit']
            x, y = rendered[mask].clamp_min(1e-6).reciprocal(), prior_d[mask].reciprocal()
            x, y = x - x.mean(), y - y.mean()
            corr = (x*y).sum() / (x.square().sum()*y.square().sum()).sqrt().clamp_min(1e-12)
            depth_loss = 1 - corr
        aniso_loss = anisotropy(splats["scales"])
        loss = rgb_loss + depth_lambda * depth_loss + settings.get("aniso_lambda", .1) * aniso_loss
        if camera is not None:
            loss = loss + float(settings.get('pose_reg', 1e-6)) * camera.values[local].square().sum()
        loss.backward()
        if camera is not None:
            camera_optimizer.step()
            camera_optimizer.zero_grad(set_to_none=True)
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        visits[local] += 1
        if (step + 1) % 250 == 0:
            torch.cuda.synchronize()
            row = dict(step=step+1, loss=float(loss.detach()), rgb_loss=float(rgb_loss.detach()),
                       depth_loss=float(depth_loss.detach()), gaussians=len(splats['means']),
                       seconds=time.perf_counter()-start, **memory_stats())
            dump(out / 'progress.json', row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    stats = dict(completed_steps=steps, train_seconds=time.perf_counter()-start,
                 final_gaussians=len(splats['means']), train_frames=len(train_ids),
                 test_frames=len(data['test_indices']), min_frame_visits=int(visits.min()),
                 max_frame_visits=int(visits.max()), **memory_stats())
    dump(out / 'train_stats.json', stats)
    dump(out / 'prior_pixels.json', counts)
    pose_state = None
    if camera is not None:
        with torch.no_grad():
            optimized = camera(data['c2w'][train_ids], slice(None)).cpu().numpy()
        optimized[:, :3, 3] = optimized[:, :3, 3] * data['meters_per_unit'] + np.asarray(data['center'])
        np.savez(out / 'optimized_poses.npz', frame_ids=data['frame_ids'][train_ids], c2w=optimized)
        pose_state = dict(values=camera.values.detach().cpu(), frame_ids=data['frame_ids'][train_ids])
    torch.save(dict(splats={k: v.detach().cpu() for k, v in splats.items()}, pose=pose_state, affine=None,
                    config=config, stats=stats), out / 'final.pt')
    np.savez(out / 'train_visits.npz', frame_ids=data['frame_ids'][train_ids], visits=visits)
