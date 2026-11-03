#!/usr/bin/env python3
"""SfM for phone frames: pycolmap sequential + keyframe-subset loop closure.

One shared OPENCV/SIMPLE_RADIAL camera for the whole video (a phone recording is
a single intrinsics set), incremental mapping, then the largest model is kept.
Loop closure substitutes for the vocabulary tree (which is not installed): a
sparse subset of keyframes is matched exhaustively through an imported pair
list, restricted to pairs that are far apart in time.

Writes <out>/shared/ (COLMAP model), <out>/undistorted/ (PNG), <out>/cameras.npz
and <out>/sfm.json.  No metrics are invented here: registration rate, the
unregistered list and the camera-centre jump check are all reported.
"""
import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pycolmap as pc

MODELS = {'OPENCV': 4, 'SIMPLE_RADIAL': 1, 'RADIAL': 2}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def loop_pairs(names, overlap, keyframes, max_pairs, min_gap):
    """Pairs among a sparse keyframe subset, only where the time gap is large."""
    total = len(names)
    stride = max(1, total // max(1, keyframes))
    keys = list(range(0, total, stride))
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if b - a < min_gap:
                continue
            pairs.append((a, b))
    pairs.sort(key=lambda pair: pair[1] - pair[0])
    if max_pairs and len(pairs) > max_pairs:
        keep = np.linspace(0, len(pairs) - 1, max_pairs).astype(int)
        pairs = [pairs[i] for i in keep]
    return [(names[a], names[b]) for a, b in pairs], stride


def model_stats(rec, registered, total):
    rec.update_point_3d_errors()
    rows, errors = [], []
    for image_id in sorted(rec.reg_image_ids(), key=lambda i: rec.images[i].name):
        image = rec.images[image_id]
        per_image = []
        for point in image.points2D:
            if point.has_point3D():
                xyz = rec.points3D[point.point3D_id].xyz
                uv = image.camera.img_from_cam(image.cam_from_world() * xyz)
                per_image.append(float(np.linalg.norm(uv - point.xy)))
        errors.extend(per_image)
        rows.append(dict(image_id=int(image_id), name=image.name, observations=len(per_image)))
    observations = [row['observations'] for row in rows]
    return dict(registered_frames=len(rows), register_ratio=len(rows) / max(1, total),
                points=rec.num_points3D(),
                mean_reprojection_error=float(rec.compute_mean_reprojection_error()),
                observation_weighted_mean_error=float(np.mean(errors)) if errors else None,
                observations_per_image=dict(
                    zip(['P0', 'P10', 'P50', 'P90'],
                        np.percentile(observations, [0, 10, 50, 90]).tolist())) if observations else {},
                per_image=rows)


def health(rec):
    """Flag camera-centre jumps: a step >5x the median, or a >60 deg axis turn."""
    images = sorted((rec.images[i] for i in rec.reg_image_ids()),
                    key=lambda image: int(Path(image.name).stem))
    frames = [int(Path(image.name).stem) for image in images]
    c2w = [image.cam_from_world().inverse().matrix() for image in images]
    steps = [float(np.linalg.norm(b[:3, 3] - a[:3, 3])) for a, b in zip(c2w, c2w[1:])]
    median = float(np.median(steps)) if steps else 0.0
    edges, jumps = [], set()
    for i, step in enumerate(steps):
        angle = float(np.degrees(np.arccos(np.clip(c2w[i][:3, 2] @ c2w[i + 1][:3, 2], -1, 1))))
        jump = step > 5 * median or angle > 60
        if jump:
            jumps.update((frames[i], frames[i + 1]))
        edges.append(dict(from_frame=frames[i], to_frame=frames[i + 1], step=step,
                          step_ratio=step / median if median else None,
                          axis_angle_deg=angle, jump=jump))
    return dict(median_step=median, jump_frames=sorted(jumps), jump_count=len(jumps), edges=edges)


def export(out, names, rec, image_dir):
    camera = next(iter(rec.cameras.values()))
    params = np.asarray(camera.params, dtype=np.float64)
    model = camera.model_name
    if model == 'OPENCV':
        fx, fy, cx, cy = params[0:4]
        distortion = params[4:8]
    elif model == 'SIMPLE_RADIAL':      # f, cx, cy, k
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        distortion = np.array([params[3], 0.0, 0.0, 0.0])
    elif model == 'RADIAL':             # f, cx, cy, k1, k2
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        distortion = np.array([params[3], params[4], 0.0, 0.0])
    elif model == 'PINHOLE':            # fx, fy, cx, cy
        fx, fy, cx, cy = params[0:4]
        distortion = np.zeros(4)
    elif model == 'SIMPLE_PINHOLE':     # f, cx, cy
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        distortion = np.zeros(4)
    else:
        raise RuntimeError(f'Unsupported camera model: {model}')
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    size = (int(camera.width), int(camera.height))
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, distortion, size, 0, size)
    mx, my = cv2.initUndistortRectifyMap(K, distortion, None, new_K, size, cv2.CV_32FC1)
    destination = out / 'undistorted'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    lookup = {rec.images[i].name: rec.images[i] for i in rec.reg_image_ids()}
    n = len(names)
    c2w = np.full((n, 4, 4), np.nan, dtype=np.float64)
    registered, sharpness = [], []
    for index, name in enumerate(names):
        image = lookup.get(name)
        if image is None:
            continue
        c2w[index] = np.eye(4)
        c2w[index, :3, :] = image.cam_from_world().inverse().matrix()
        source = cv2.imread(str(image_dir / name))
        if source is None or source.shape[:2] != (size[1], size[0]):
            raise RuntimeError(f'Bad frame {name}: expected {size[0]}x{size[1]}')
        rgb = cv2.remap(source, mx, my, cv2.INTER_LINEAR)
        if not cv2.imwrite(str(destination / f'{Path(name).stem}.png'), rgb):
            raise OSError(f'Cannot write undistorted frame {name}')
        sharpness.append(dict(frame=int(Path(name).stem),
                              value=float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())))
        registered.append(index)

    frame_ids = np.array([int(Path(name).stem) for name in names], dtype=np.int64)
    registered = np.asarray(registered, dtype=np.int64)
    train = np.array([i for i in range(n) if i % 8 != 7], dtype=np.int64)
    test = np.array([i for i in range(n) if i % 8 == 7], dtype=np.int64)
    points = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64).reshape(-1, 3)
    colors = np.array([p.color for p in rec.points3D.values()], dtype=np.float64).reshape(-1, 3) / 255.0
    np.savez(out / 'cameras.npz', frame_ids=frame_ids, c2w=c2w,
             K=np.repeat(new_K[None], n, axis=0).astype(np.float64),
             registered_indices=registered,
             train_indices=train[np.isin(train, registered)],
             test_indices=test[np.isin(test, registered)],
             undistorted_indices=registered,
             points=points.astype(np.float32), colors=colors.astype(np.float32))
    dump(out / 'preparation.json', dict(
        camera_model=model, original_K=K.tolist(), distortion=distortion.tolist(),
        undistorted_K=new_K.tolist(), resolution=list(size), roi=[int(v) for v in roi],
        undistortion='cv2.getOptimalNewCameraMatrix(alpha=0) + cv2.remap INTER_LINEAR, same size, no crop',
        undistorted_frames=len(registered),
        undistorted_sharpness_median=float(np.median([s['value'] for s in sharpness])),
        split='input index % 8 == 7 is a test frame; test frames are used by SfM but never by 3DGS',
        train_frames=int(np.isin(train, registered).sum()), test_frames=int(np.isin(test, registered).sum())))
    return dict(frames=n, registered=len(registered))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='run directory')
    parser.add_argument('--images', type=Path, default=None,
                        help='frames for SfM (default: <out>/frames_sfm if present, else <out>/frames)')
    parser.add_argument('--keep', type=Path, default=None,
                        help='frames that 3DGS will train on (default: <out>/frames)')
    parser.add_argument('--camera-model', default='SIMPLE_RADIAL', choices=sorted(MODELS))
    parser.add_argument('--focal-factor', type=float, default=1.2,
                        help='initial focal = factor * max(width,height); refined by BA')
    parser.add_argument('--overlap', type=int, default=20, help='sequential pairing overlap')
    parser.add_argument('--loop-keyframes', type=int, default=60,
                        help='keyframes in the loop-closure subset (0 disables loop closure)')
    parser.add_argument('--loop-max-pairs', type=int, default=600)
    parser.add_argument('--loop-min-gap', type=int, default=0,
                        help='minimum index distance for a loop pair (0 -> 2*overlap)')
    parser.add_argument('--max-features', type=int, default=8192,
                        help='SIFT features per frame; raise it for texture-poor rooms')
    parser.add_argument('--peak-threshold', type=float, default=0.0,
                        help='SIFT peak threshold; 0 keeps the COLMAP default (0.00667)')
    parser.add_argument('--sfm-mode', choices=['global', 'incremental'], default='global',
                        help='global (GLOMAP) or incremental COLMAP mapping')
    parser.add_argument('--reuse-database', action='store_true',
                        help='skip feature extraction and matching when database.db already exists')
    parser.add_argument('--min-model-size', type=int, default=10)
    parser.add_argument('--init-min-inliers', type=int, default=0,
                        help='incremental mapper init_min_num_inliers override (0 = COLMAP default)')
    parser.add_argument('--abs-pose-min-inliers', type=int, default=0,
                        help='incremental mapper abs_pose_min_num_inliers override (0 = COLMAP default)')
    parser.add_argument('--export-only', action='store_true',
                        help='skip SfM and re-export cameras.npz/undistorted from <out>/shared')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=0, help='max seconds for incremental mapping')
    args = parser.parse_args(argv)

    out = args.out.resolve()
    default_sfm = out / 'frames_sfm'
    image_dir = (args.images or (default_sfm if default_sfm.is_dir() else out / 'frames')).resolve()
    keep_dir = (args.keep or (out / 'frames')).resolve()
    names = sorted((p.name for p in image_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')), key=lambda s: int(Path(s).stem))
    keep_names = sorted((p.name for p in keep_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')), key=lambda s: int(Path(s).stem))
    if len(names) < 8 or len(keep_names) < 4:
        raise RuntimeError(f'Need at least 8 SfM frames and 4 training frames: {image_dir}, {keep_dir}')
    if not set(keep_names) <= set(names):
        raise RuntimeError(f'Training frames must be a subset of the SfM frames: {keep_dir} vs {image_dir}')
    broken = [name for name in names if not (image_dir / name).is_file()]
    if broken:
        raise RuntimeError(f'{len(broken)} unreadable images in {image_dir}, e.g. {broken[:3]}')
    cv2.setNumThreads(1)
    started = time.perf_counter()
    if args.export_only:
        rec = pc.Reconstruction(out / 'shared')
        report = dict(input_frames=len(names), keep_frames=len(keep_names),
                      image_dir=str(image_dir), keep_dir=str(keep_dir),
                      camera_model=next(iter(rec.cameras.values())).model_name,
                      export_only=True, reused_database=True)
        try:
            previous = json.loads((out / 'sfm.json').read_text())
            for key in ('models', 'largest_model', 'mapping', 'loop_pairs', 'max_features',
                        'peak_threshold', 'focal_factor', 'sequential_overlap'):
                if key in previous:
                    report[key] = previous[key]
        except Exception:
            pass
        stats = model_stats(rec, len(names), len(names))
        report.update(stats)
        registered_names = {rec.images[i].name for i in rec.reg_image_ids()}
        kept = [name for name in keep_names if name in registered_names]
        report['kept_registered'] = len(kept)
        report['kept_registered_ratio'] = len(kept) / len(keep_names)
        report['unregistered_frames'] = [int(Path(x).stem) for x in keep_names
                                         if x not in registered_names]
        report['unregistered_count'] = len(report['unregistered_frames'])
        check = health(rec)
        report['health'] = {k: check[k] for k in ('median_step', 'jump_frames', 'jump_count')}
        dump(out / 'health.json', check)
        export(out, keep_names, rec, image_dir)
        dump(out / 'sfm.json', report)
        print(json.dumps(dict(event='complete', export_only=True, keep_frames=len(keep_names),
                              registered=stats['registered_frames'],
                              kept_registered=len(kept),
                              ratio=round(len(kept) / len(keep_names), 3))), flush=True)
        return
    report = dict(input_frames=len(names), keep_frames=len(keep_names),
                  image_dir=str(image_dir), keep_dir=str(keep_dir), camera_model=args.camera_model,
                  focal_factor=args.focal_factor, sequential_overlap=args.overlap,
                  max_features=args.max_features, peak_threshold=args.peak_threshold or 'default',
                  loop_keyframes=args.loop_keyframes, loop_max_pairs=args.loop_max_pairs,
                  pycolmap_version=pc.__version__)
    try:
        database = out / 'database.db'
        reuse = args.reuse_database and database.is_file()
        if not reuse:
            for stale in (database, out / 'database.db-journal'):
                stale.unlink(missing_ok=True)
        models_dir = out / 'models'
        if models_dir.exists():
            shutil.rmtree(models_dir)
        models_dir.mkdir(parents=True)
        report['reused_database'] = bool(reuse)

        extract = time.perf_counter()
        if reuse:
            print(json.dumps(dict(event='reuse_database', path=str(database))), flush=True)
        reader = pc.ImageReaderOptions(camera_model=args.camera_model,
                                       default_focal_length_factor=args.focal_factor)
        sift = pc.SiftExtractionOptions(max_num_features=args.max_features)
        if args.peak_threshold:
            sift.peak_threshold = args.peak_threshold
        if not reuse:
            pc.extract_features(database, image_dir, image_names=names, camera_mode=pc.CameraMode.SINGLE,
                                reader_options=reader,
                                extraction_options=pc.FeatureExtractionOptions(
                                    num_threads=args.threads, use_gpu=False, sift=sift),
                                device=pc.Device.cpu)
        report['extract_seconds'] = time.perf_counter() - extract
        print(json.dumps(dict(event='features', seconds=round(report['extract_seconds'], 1))), flush=True)

        match = time.perf_counter()
        report['loop_pairs'] = 0
        report['loop_warning'] = None
        if not reuse:
            pairing = pc.SequentialPairingOptions(overlap=args.overlap, quadratic_overlap=True,
                                                  loop_detection=False, num_threads=args.threads)
            pc.match_sequential(database,
                                matching_options=pc.FeatureMatchingOptions(num_threads=args.threads, use_gpu=False),
                                pairing_options=pairing, device=pc.Device.cpu)
        report['match_seconds'] = time.perf_counter() - match
        if not reuse and args.loop_keyframes > 0:
            gap = args.loop_min_gap or 2 * args.overlap
            pairs, stride = loop_pairs(names, args.overlap, args.loop_keyframes,
                                       args.loop_max_pairs, gap)
            listing = out / 'loop_pairs.txt'
            listing.write_text(''.join(f'{a} {b}\n' for a, b in pairs))
            try:
                pc.match_image_pairs(
                    database,
                    matching_options=pc.FeatureMatchingOptions(num_threads=args.threads, use_gpu=False),
                    pairing_options=pc.ImportedPairingOptions(match_list_path=str(listing)),
                    device=pc.Device.cpu)
                report['loop_pairs'] = len(pairs)
                report['loop_stride'] = stride
            except Exception as error:  # loop closure is an improvement, not a precondition
                report['loop_warning'] = f'{type(error).__name__}: {error}'
                print(json.dumps(dict(event='loop_closure_failed', detail=report['loop_warning'])), flush=True)
        report['seconds_before_mapping'] = time.perf_counter() - started
        print(json.dumps(dict(event='matched', loop_pairs=report['loop_pairs'],
                              seconds=round(report['match_seconds'], 1))), flush=True)

        mapping = time.perf_counter()
        if args.sfm_mode == 'global':
            options = pc.GlobalPipelineOptions(
                num_threads=args.threads, random_seed=42, multiple_models=True,
                min_model_size=args.min_model_size, image_path=str(image_dir))
            models = pc.global_mapping(database, image_dir, models_dir, options=options)
            report['mapping'] = 'global (GLOMAP)'
        else:
            options = pc.IncrementalPipelineOptions(
                num_threads=args.threads, random_seed=42, multiple_models=True,
                min_model_size=args.min_model_size,
                init_num_trials=200, ba_refine_focal_length=True, ba_refine_principal_point=True,
                ba_refine_extra_params=True, ba_use_gpu=False)
            if args.init_min_inliers:
                options.mapper.init_min_num_inliers = args.init_min_inliers
            if args.abs_pose_min_inliers:
                options.mapper.abs_pose_min_num_inliers = args.abs_pose_min_inliers
            if args.timeout:
                options.max_runtime_seconds = args.timeout
            models = pc.incremental_mapping(database, image_dir, models_dir, options=options)
            report['mapping'] = 'incremental'
        report['mapping_seconds'] = time.perf_counter() - mapping
        counts = {str(k): int(v.num_reg_images()) for k, v in models.items()}
        report['models'] = counts
        if not counts:
            raise RuntimeError('COLMAP produced no model. Check registration warnings; '
                               'try --overlap 30 or more aggressive matching.')
        best = max(counts, key=lambda k: counts[k])
        rec = models[int(best)]
        report['largest_model'] = int(best)
        report['warning'] = None
        if len(counts) > 1:
            report['warning'] = (f'SfM split into {len(counts)} models; kept model {best} with '
                                 f'{counts[best]}/{len(names)} frames')
            print(json.dumps(dict(event='warning', detail=report['warning'])), flush=True)
        (out / 'shared').mkdir(exist_ok=True)
        rec.write(out / 'shared')

        stats = model_stats(rec, len(names), len(names))
        report.update(stats)
        registered_names = {rec.images[i].name for i in rec.reg_image_ids()}
        kept = [name for name in keep_names if name in registered_names]
        report['kept_registered'] = len(kept)
        report['kept_registered_ratio'] = len(kept) / len(keep_names)
        report['unregistered_frames'] = [int(Path(x).stem) for x in keep_names if x not in registered_names]
        report['unregistered_count'] = len(report['unregistered_frames'])
        report['unregistered_sfm_frames'] = [int(Path(x).stem) for x in names if x not in registered_names]
        check = health(rec)
        report['health'] = {k: check[k] for k in ('median_step', 'jump_frames', 'jump_count')}
        dump(out / 'health.json', check)
        export(out, keep_names, rec, image_dir)
        report['seconds'] = time.perf_counter() - started
        dump(out / 'sfm.json', report)
        print(json.dumps(dict(event='complete', registered=stats['registered_frames'],
                              input=len(names), ratio=round(stats['register_ratio'], 3),
                              kept_registered=len(kept),
                              kept_ratio=round(len(kept) / len(keep_names), 3),
                              models=counts, loop_pairs=report['loop_pairs'],
                              jump_frames=check['jump_count'],
                              seconds=round(report['seconds'], 1))), flush=True)
    except Exception as error:
        # Never leave sfm.json behind on failure: it is the stage-complete marker.
        (out / 'sfm_error.txt').write_text(traceback.format_exc())
        dump(out / 'sfm_failed.json', dict(**report, error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
