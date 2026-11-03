#!/usr/bin/env python3
"""Roam video + viewer export for a trained phone 3DGS scene.

The camera path is the capture trajectory itself: registered training cameras
are reduced to keyframes, positions and rotations are smoothed (Gaussian on a
sign-aligned quaternion sequence), and the path is resampled at constant
arc-length speed with Catmull-Rom translations and SLERP rotations.  That is a
smooth fly-through of exactly the volume the reconstruction covers, with no
text and no overlays.

Also exports .splat (32-byte rows) and .ply (INRIA 3DGS fields) in normalized
reconstruction units -- a phone video has no metric scale, so nothing here is
presented as metres.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from recon.io.ply import write_ply
from scipy.spatial.transform import Rotation

from gsplat import rasterization

SH_C0 = 0.28209479177387814


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def gaussian_smooth(values, sigma):
    if sigma <= 0 or len(values) < 3:
        return values.copy()
    radius = max(1, int(round(3 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    padded = np.pad(values, [(radius, radius)] + [(0, 0)] * (values.ndim - 1), mode='reflect')
    return np.stack([np.tensordot(weights, padded[i:i + 2 * radius + 1], axes=(0, 0))
                     for i in range(len(values))])


def slerp(q0, q1, t):
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def catmull_rom(p0, p1, p2, p3, t):
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def frame_texture(path):
    """Edge density of a frame: mean Sobel magnitude on a quarter-size grey copy."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    small = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def median_depths(c2w, K, points, width, height):
    """Per-camera median optical depth of the SfM points that fall inside the frame."""
    out = np.full(len(c2w), np.nan)
    for i in range(len(c2w)):
        points_c = (c2w[i, :3, :3].T @ (points - c2w[i, :3, 3]).T).T
        depth = points_c[:, 2]
        near = depth > 1e-6
        if not near.any():
            continue
        uv = (K @ points_c[near].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        if inside.any():
            out[i] = float(np.median(depth[near][inside]))
    return out


def longest_run(flags):
    """(start, stop) of the longest True run in a boolean array."""
    best, start = (0, -1), None
    for i, ok in enumerate(flags):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(flags) - start > best[1] - best[0]:
        best = (start, len(flags))
    return best


def path_quality(positions, quats, c2w, radius, max_angle_deg):
    """Fraction of output views close to a training camera and looking the same way."""
    train = c2w[:, :3, 3]
    axes = c2w[:, :3, 2]
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    from scipy.spatial.transform import Rotation as R
    out_axes = R.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    out_axes /= np.linalg.norm(out_axes, axis=1, keepdims=True)
    distance = np.linalg.norm(positions[:, None, :] - train[None, :, :], axis=2)
    nearest = distance.argmin(1)
    nearest_distance = distance[np.arange(len(positions)), nearest]
    cosine = np.clip((out_axes * axes[nearest]).sum(1), -1, 1)
    angle = np.degrees(np.arccos(cosine))
    return dict(within_radius_fraction=float((nearest_distance <= radius).mean()),
                within_angle_fraction=float((angle <= max_angle_deg).mean()),
                max_distance=float(nearest_distance.max()),
                p95_distance=float(np.percentile(nearest_distance, 95)),
                max_angle_deg=float(angle.max()),
                p95_angle_deg=float(np.percentile(angle, 95)))


def build_path(c2w, keyframes, sigma, frames):
    """Smooth the capture trajectory, then resample it at constant speed."""
    stride = max(1, int(round(len(c2w) / keyframes)))
    keys = np.arange(0, len(c2w), stride)
    positions = c2w[keys, :3, 3].astype(np.float64)
    # scipy returns xyzw; keep wxyz here so slerp/sign alignment read naturally.
    quats = Rotation.from_matrix(c2w[keys, :3, :3].astype(np.float64)).as_quat()[:, [3, 0, 1, 2]]
    for index in range(1, len(quats)):
        if np.dot(quats[index - 1], quats[index]) < 0:
            quats[index] = -quats[index]
    positions = gaussian_smooth(positions, sigma)
    quats = gaussian_smooth(quats, sigma)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    cumulative = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])
    total = float(cumulative[-1])
    if total < 1e-9:
        raise RuntimeError('Capture trajectory has zero length; cannot build a roam path')
    fraction = np.interp(np.linspace(0, total, frames), cumulative,
                         np.arange(len(positions), dtype=np.float64))
    count = len(positions)
    out_positions = np.empty((frames, 3))
    out_quats = np.empty((frames, 4))
    for i, value in enumerate(fraction):
        base = min(int(np.floor(value)), count - 2) if count > 1 else 0
        t = float(value - base)
        out_positions[i] = catmull_rom(positions[max(0, base - 1)], positions[base],
                                       positions[min(count - 1, base + 1)],
                                       positions[min(count - 1, base + 2)], t)
        out_quats[i] = slerp(quats[base], quats[min(count - 1, base + 1)], t)
    return out_positions, out_quats, dict(keyframes=int(len(keys)), stride=stride, path_length=total)


def export_splat(splats, path, opacity_min, max_count, radius=0.0):
    arrays = {key: value.detach().cpu().numpy() for key, value in splats.items()}
    scale = np.exp(arrays['scales'].astype(np.float64))
    alpha = 1.0 / (1.0 + np.exp(-np.clip(arrays['opacities'].astype(np.float64), -80, 80)))
    quat = arrays['quats'].astype(np.float64)
    norm = np.linalg.norm(quat, axis=1)
    keep = (alpha >= opacity_min) & np.isfinite(norm) & (norm > 1e-12)
    if radius > 0:
        keep &= np.linalg.norm(arrays['means'], axis=1) <= radius
    total_kept = int(keep.sum())
    order = np.flatnonzero(keep)
    order = order[np.argsort(-(scale[order].prod(1) * alpha[order]), kind='stable')]
    if max_count and len(order) > max_count:
        order = order[:max_count]
    dtype = np.dtype([('xyz', '<f4', 3), ('scale', '<f4', 3), ('rgba', 'u1', 4), ('quat', 'u1', 4)])
    rows = np.empty(len(order), dtype=dtype)
    rows['xyz'] = arrays['means'][order].astype(np.float32)
    rows['scale'] = scale[order].astype(np.float32)
    rows['rgba'][:, :3] = np.clip((0.5 + SH_C0 * arrays['sh0'][order, 0]) * 255, 0, 255).astype(np.uint8)
    rows['rgba'][:, 3] = np.clip(alpha[order] * 255, 0, 255).astype(np.uint8)
    rows['quat'] = np.clip(quat[order] / norm[order, None] * 128 + 128, 0, 255).astype(np.uint8)
    path.write_bytes(rows.tobytes())
    return dict(exported=int(len(rows)), bytes=int(rows.nbytes),
                dropped_below_opacity=int(len(alpha) - total_kept),
                truncated=int(total_kept - len(order)))


def export_ply(splats, path, radius=0.0):
    arrays = {key: value.detach().cpu().numpy() for key, value in splats.items()}
    if radius > 0:
        inside = np.linalg.norm(arrays['means'], axis=1) <= radius
        arrays = {key: value[inside] for key, value in arrays.items()}
    n = len(arrays['means'])
    sh = np.concatenate([arrays['sh0'], arrays['shN']], axis=1)       # (n,16,3)
    rest = sh[:, 1:].transpose(0, 2, 1).reshape(n, 45)               # channel-major, INRIA order
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
             + [f'f_rest_{i}' for i in range(45)] + ['opacity']
             + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    values = np.concatenate([arrays['means'].astype(np.float32), np.zeros((n, 3), dtype=np.float32),
                             sh[:, 0].astype(np.float32), rest.astype(np.float32),
                             arrays['opacities'].astype(np.float32)[:, None],
                             arrays['scales'].astype(np.float32),
                             arrays['quats'].astype(np.float32)], axis=1)
    vertex = np.empty(n, dtype=[(name, 'f4') for name in names])
    for column, name in enumerate(names):
        vertex[name] = values[:, column]
    write_ply(path, vertex)
    return dict(gaussians=int(n), bytes=int(path.stat().st_size),
                fields=len(names), convention='INRIA 3DGS: opacity logit, scale log, quat wxyz, '
                                              'f_rest channel-major')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='run directory (gs/final.pt, cameras.npz)')
    parser.add_argument('--checkpoint', type=Path, default=None, help='optional selected/pruned checkpoint')
    parser.add_argument('--path-file', type=Path, default=None, help='reuse saved roam_path.npz exactly')
    parser.add_argument('--frame-start', type=int, default=0, help='optional first source frame for a curated route')
    parser.add_argument('--frame-end', type=int, default=0, help='optional last source frame, inclusive; 0 disables')
    parser.add_argument('--skip-export', action='store_true', help='render a second route without duplicating the same model exports')
    parser.add_argument('--seconds', type=float, default=20.0)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--long-edge', type=int, default=1920, help='1920x1080 for a 16:9 capture')
    parser.add_argument('--keyframes', type=int, default=120)
    parser.add_argument('--smooth-sigma', type=float, default=1.0)
    parser.add_argument('--max-gap', type=int, default=12,
                        help='max index gap between consecutive kept frames inside one path segment')
    parser.add_argument('--max-height-span', type=float, default=0.15,
                        help='max camera-up height span of dense segment, normalized scene units; 0 disables')
    parser.add_argument('--max-step-angle', type=float, default=16.0,
                        help='max angle between consecutive kept cameras inside one path segment')
    parser.add_argument('--crf', type=int, default=17)
    parser.add_argument('--opacity-min', type=float, default=0.02)
    parser.add_argument('--splat-max', type=int, default=0)
    parser.add_argument('--path-mode', choices=['dense', 'all'], default='dense',
                        help='dense: only a well covered, well textured stretch of the capture')
    parser.add_argument('--density-radius', type=float, default=0.12,
                        help='max distance from the nearest training camera, in normalized units')
    parser.add_argument('--max-view-angle', type=float, default=8.0,
                        help='max angle between an output view and the nearest training camera')
    parser.add_argument('--closeup-fraction', type=float, default=0.5,
                        help='drop frames whose median depth is below this fraction of the global '
                             'median depth (the phone pushing into a wall/pillow)')
    parser.add_argument('--texture-percentile', type=float, default=30.0,
                        help='drop the least textured this many percent of frames (blank walls)')
    parser.add_argument('--path-radius', type=float, default=5.0,
                        help='drop cameras farther than this from the scene centre '
                             '(normalized units); 0 disables')
    parser.add_argument('--export-radius', type=float, default=5.0,
                        help='drop Gaussians farther than this from the scene centre; 0 disables')
    parser.add_argument('--skip-video', action='store_true')
    args = parser.parse_args()

    out = args.out.resolve()
    gs = out / 'gs'
    checkpoint = torch.load(args.checkpoint or (gs / 'final.pt'), map_location='cpu', weights_only=False)
    config, tensors = checkpoint['config'], checkpoint['splats']
    center = np.asarray(config['normalization_center'], dtype=np.float64)
    scale = float(config.get('sfm_units_per_normalized_unit', config.get('meters_per_unit')))
    with np.load(out / 'cameras.npz') as handle:
        c2w, K = handle['c2w'].copy(), handle['K'].copy()
        frame_ids = handle['frame_ids']
        usable = np.intersect1d(handle['registered_indices'], handle['train_indices'])
    usable = usable[np.isfinite(c2w[usable]).all(axis=(1, 2))]
    # Use exactly the cameras training accepted: gs/normalization.json lists the
    # frames dropped as mis-registered, and a path through them flies into empty space.
    dropped = set()
    normal = gs / 'normalization.json'
    if normal.is_file():
        dropped = set(json.loads(normal.read_text()).get('dropped_frames', []))
    usable = np.array([i for i in usable if int(frame_ids[i]) not in dropped], dtype=usable.dtype)
    usable = usable[np.argsort(frame_ids[usable])]
    if len(usable) < 4:
        raise RuntimeError(f'Need at least 4 registered training cameras, found {len(usable)}')

    sample = cv2.imread(str(sorted((out / 'undistorted').glob('*.png'))[0]))
    if sample is None:
        raise RuntimeError('No undistorted frames to read the training size from')
    native_height, native_width = sample.shape[:2]
    if native_width >= native_height:
        width = args.long_edge
        height = int(round(args.long_edge * native_height / native_width / 2)) * 2
    else:
        height = args.long_edge
        width = int(round(args.long_edge * native_width / native_height / 2)) * 2
    shrink = width / native_width
    K_out = K[usable[0]].astype(np.float64).copy()
    K_out[0] *= shrink
    K_out[1] *= shrink

    poses = c2w[usable].copy()
    poses[:, :3, 3] = (poses[:, :3, 3] - center) / scale
    selected = np.ones(len(poses), dtype=bool)
    filter_info = dict(mode=args.path_mode, candidates=int(len(poses)))
    selected &= frame_ids[usable] >= args.frame_start
    if args.frame_end:
        selected &= frame_ids[usable] <= args.frame_end
    filter_info['requested_frame_range'] = [args.frame_start, args.frame_end or None]
    if args.path_radius > 0:
        far = np.linalg.norm(poses[:, :3, 3], axis=1) > args.path_radius
        if far.any():
            print(json.dumps(dict(event='path_filter', dropped=int(far.sum()),
                                  radius=args.path_radius)), flush=True)
        selected &= ~far
    if args.path_mode == 'dense':
        # Drop the views this capture got wrong: the phone pushed into a wall/pillow
        # (median depth far below the global median) or pointed at a blank surface.
        with np.load(out / 'cameras.npz') as handle:
            points = handle['points'].astype(np.float64)
            points = (points - center) / scale
        depths = median_depths(poses, K[usable[0]].astype(np.float64), points,
                               native_width, native_height)
        finite = np.isfinite(depths)
        global_median = float(np.median(depths[finite])) if finite.any() else float('nan')
        texture = np.array([frame_texture(out / 'undistorted' / f'{int(frame_ids[i]):05d}.png')
                            for i in usable])
        threshold = float(np.percentile(texture, args.texture_percentile)) if args.texture_percentile else -1
        informative = finite & (depths >= args.closeup_fraction * global_median) & (texture >= threshold)
        # A segment must not skip frames: a gap means the viewpoint jumped, and a path
        # through it cuts corners far away from any training camera.
        kept = np.flatnonzero(informative & selected)
        axes = poses[:, :3, 2] / np.linalg.norm(poses[:, :3, 2], axis=1, keepdims=True)
        groups = []
        if len(kept):
            begin = previous = int(kept[0])
            for index in kept[1:]:
                index = int(index)
                step_angle = float(np.degrees(np.arccos(np.clip(axes[previous] @ axes[index], -1, 1))))
                if index - previous <= args.max_gap and step_angle <= args.max_step_angle:
                    previous = index
                else:
                    groups.append((begin, previous + 1))
                    begin = previous = index
            groups.append((begin, previous + 1))
        up = np.median(-poses[:, :3, 1], axis=0)
        up /= max(np.linalg.norm(up), 1e-12)
        heights = poses[:, :3, 3] @ up
        if args.max_height_span > 0:
            stable_groups = []
            for begin, end in groups:
                left = begin
                best = (begin, begin)
                for right in range(begin, end):
                    while left < right and np.ptp(heights[left:right + 1]) > args.max_height_span:
                        left += 1
                    if right + 1 - left > best[1] - best[0]:
                        best = (left, right + 1)
                stable_groups.append(best)
            groups = stable_groups
        start, stop = max(groups, key=lambda group: group[1] - group[0]) if groups else (0, 0)
        filter_info.update(estimated_up=up.tolist(),
                           height_span=float(np.ptp(heights[start:stop])) if stop > start else None,
                           height_span_limit=args.max_height_span,
                           height_note='median camera-up proxy; no gravity/metric reference available')
        filter_info.update(global_median_depth=global_median,
                           depth_kept=int((depths >= args.closeup_fraction * global_median).sum()),
                           texture_threshold=threshold,
                           texture_kept=int((texture >= threshold).sum()),
                           informative=int(informative.sum()),
                           selected_run=[int(start), int(stop)],
                           run_frames=int(stop - start),
                           run_frame_ids=[int(frame_ids[usable[start]]), int(frame_ids[usable[stop - 1]])])
        filter_info['segments'] = len(groups)
        filter_info['segment_lengths'] = sorted((b - a for a, b in groups), reverse=True)[:5]
        if stop - start < 20:
            raise RuntimeError(f'Only {stop - start} informative consecutive frames; '
                               f'loosen --texture-percentile / --closeup-fraction / --max-gap')
        selected &= informative
        selected[:start] = False
        selected[stop:] = False
        print(json.dumps(dict(event='path_selection', **{k: v for k, v in filter_info.items()
                              if k != 'run_frame_ids'})), flush=True)
    filter_info['used_frame_range'] = [int(frame_ids[usable[selected]][0]), int(frame_ids[usable[selected]][-1])] if selected.any() else []
    poses = poses[selected]
    if len(poses) < 4:
        raise RuntimeError('Fewer than 4 usable cameras after the roam path filter')
    frames = int(round(args.seconds * args.fps))
    positions, quats, path_info = build_path(poses, args.keyframes, args.smooth_sigma, frames)
    if args.path_file:
        with np.load(args.path_file) as frozen:
            positions, quats = frozen['positions'].copy(), frozen['quats'].copy()
            if len(positions) != frames:
                raise RuntimeError('Frozen path frame count differs from seconds * fps')
            if 'K' in frozen:
                np.testing.assert_allclose(K_out, frozen['K'], rtol=1e-6, atol=1e-6)
                K_out = frozen['K'].copy()
        path_info['frozen_from'] = str(args.path_file.resolve())
    quality = path_quality(positions, quats, poses, args.density_radius, args.max_view_angle)
    for _ in range(0 if args.path_file else 3):
        if quality['within_radius_fraction'] >= 0.95 and quality['within_angle_fraction'] >= 0.95:
            break
        args.smooth_sigma = max(0.5, args.smooth_sigma * 0.6)
        positions, quats, path_info = build_path(poses, args.keyframes, args.smooth_sigma, frames)
        quality = path_quality(positions, quats, poses, args.density_radius, args.max_view_angle)
        print(json.dumps(dict(event='path_retry', sigma=args.smooth_sigma, **quality)), flush=True)
    if quality['within_radius_fraction'] < 0.95 or quality['within_angle_fraction'] < 0.95:
        raise RuntimeError(f'Roam path failed training-view coverage after retries: {quality}')
    path_info.update(quality=quality, filters=filter_info, smooth_sigma=args.smooth_sigma)
    print(json.dumps(dict(event='path_quality', **quality)), flush=True)
    np.savez(out / 'roam_path.npz', positions=positions, quats=quats, K=K_out,
             source_frame_ids=frame_ids[usable[selected]], width=width, height=height)

    model = {key: tensors[key].cuda() for key in ('means', 'quats')}
    model['scales'] = tensors['scales'].exp().cuda()
    model['opacities'] = tensors['opacities'].sigmoid().cuda()
    model['colors'] = torch.cat([tensors['sh0'], tensors['shN']], 1).cuda()
    count = len(tensors['means'])
    del checkpoint
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(4)
    started = time.perf_counter()
    render = None
    if not args.skip_video:
        video = out / 'roam_1080p.mp4'
        encoder = subprocess.Popen(
            ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
             '-s', f'{width}x{height}', '-r', str(args.fps), '-i', '-', '-an',
             '-c:v', 'libx264', '-preset', 'medium', '-crf', str(args.crf),
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
            stdin=subprocess.PIPE)
        preview = None
        K_t = torch.from_numpy(K_out[None].astype(np.float32)).cuda()
        try:
            with torch.no_grad():
                for index in range(frames):
                    pose = np.eye(4)
                    pose[:3, :3] = Rotation.from_quat(quats[index][[1, 2, 3, 0]]).as_matrix()
                    pose[:3, 3] = positions[index]
                    pose_t = torch.from_numpy(pose[None].astype(np.float32)).cuda()
                    colors, _, _ = rasterization(
                        means=model['means'], quats=model['quats'], scales=model['scales'],
                        opacities=model['opacities'], colors=model['colors'],
                        viewmats=torch.linalg.inv(pose_t), Ks=K_t, width=width, height=height,
                        sh_degree=3, near_plane=0.01, far_plane=1e10, packed=False,
                        sparse_grad=False, absgrad=False, rasterize_mode='classic',
                        camera_model='pinhole', render_mode='RGB')
                    rgb = (colors[0].clamp(0, 1) * 255).round().byte().cpu().numpy()
                    encoder.stdin.write(rgb.tobytes())
                    if index == frames // 2:
                        preview = rgb.copy()
                    del colors, pose_t
                    if index % 100 == 0:
                        print(json.dumps(dict(event='render', frame=index, of=frames,
                                              seconds=round(time.perf_counter() - started, 1))), flush=True)
        finally:
            encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('ffmpeg encoding failed for the roam video')
        if preview is not None:
            cv2.imwrite(str(out / 'roam_preview.png'), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        probed = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                 'stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames',
                                 '-of', 'json', str(video)], capture_output=True, text=True, check=True)
        render = dict(frames=frames, resolution=[width, height], fps=args.fps,
                      seconds_of_video=round(frames / args.fps, 3),
                      seconds=round(time.perf_counter() - started, 1), text='none',
                      preview='roam_preview.png', bytes=int(video.stat().st_size),
                      probe=json.loads(probed.stdout)['streams'][0])

    if args.skip_export:
        splat = dict(exported=0, bytes=0, skipped=True)
        ply = dict(gaussians=count, bytes=0, skipped=True)
    else:
        splat = export_splat(tensors, out / 'scene.splat', args.opacity_min, args.splat_max,
                             args.export_radius)
        ply = export_ply(tensors, out / 'scene.ply', args.export_radius)
    result = dict(gaussians=count, model_checkpoint=str((args.checkpoint or (gs / 'final.pt')).resolve()), path=path_info,
                  path_quality=path_info.get('quality'),
                  path_definition='registered training cameras -> keyframes -> Gaussian-smoothed positions '
                                  'and sign-aligned quaternions -> constant arc-length resampling with '
                                  'Catmull-Rom translation and SLERP rotation',
                  normalization=dict(center=center.tolist(), sfm_units_per_normalized_unit=scale),
                  coordinates='normalized reconstruction units (centre subtracted, divided by the P95 '
                              'camera radius); a single phone video carries no metric scale',
                  render=render, splat=splat, ply=ply,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
    dump(out / 'render.json', result)
    print(json.dumps(dict(event='complete', gaussians=count, splat=splat['exported'],
                          ply_bytes=ply['bytes'], video_bytes=(render or {}).get('bytes'),
                          seconds=round(time.perf_counter() - started, 1))), flush=True)


if __name__ == '__main__':
    sys.exit(main())
