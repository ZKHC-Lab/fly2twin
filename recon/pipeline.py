# SPDX-License-Identifier: Apache-2.0
"""Content-addressed, independently restartable offline stages."""

from pathlib import Path
import ast
import hashlib
import json
import numpy as np
from .io.common import load_json, load_manifest, stage_file, stage_manifest, dump, digest, image_loader
from .geometry.transforms import resize_K, normalization, robust_normalization

STAGES = ('prepare', 'pose', 'depth', 'fuse', 'train', 'eval', 'export')


def file_inputs(cfg):
    """Enumerate explicit input files, including images and cached predictions."""
    if cfg.get('video'):
        paths = [Path(cfg['video'])]
    elif cfg.get('delivery'):
        from .io.delivery import files

        paths = files(cfg['delivery'])
    else:
        paths = [Path(cfg['manifest'])]
        manifest = load_manifest(cfg['manifest'])
        paths += [Path(row['path']) for row in manifest['frames']]
    for key in ('cameras', 'selection', 'normalization_file', 'body_poses'):
        if cfg.get(key):
            paths.append(Path(cfg[key]))
    depth = cfg.get('depth', {})
    if depth.get('cache'):
        folder = Path(depth['cache'])
        paths.append(folder / 'windows.json')
        paths += [folder / w['id'] / 'prediction.npz' for w in load_json(folder / 'windows.json')['windows']]
    if depth.get('weights'):
        paths.append(Path(depth['weights']))
    pose = cfg.get('pose', {})
    if pose.get('xfeat_weights'):
        paths.append(Path(pose['xfeat_weights']))
    if cfg.get('eval', {}).get('omega_dir'):
        folder = Path(cfg['eval']['omega_dir'])
        paths += [folder / name for name in ('depth.npy', 'conf.npy', 'frame_ids.npy')]
    return sorted(set(paths))


def source_digest(path):
    """Python source by syntax tree, so formatting and comments keep the cache.

    ast.dump changes between Python minor versions, so another interpreter gives another fingerprint.
    """
    return hashlib.sha256(ast.dump(ast.parse(path.read_bytes(), filename=str(path))).encode()).hexdigest()


def fingerprint(cfg):
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for path in file_inputs(cfg):
        h.update(str(path).encode())
        h.update(digest(path).encode())
    # Cache also invalidates when installed numerical code changes.
    for path in sorted(Path(__file__).parent.rglob('*.py')):
        h.update(path.relative_to(Path(__file__).parent).as_posix().encode())
        h.update(source_digest(path).encode())
    import recon_core

    core = Path(recon_core.__file__).parent
    for path in sorted(core.rglob('*.py')):
        relative = path.relative_to(core)
        if relative.parts[0] == 'tests':
            continue
        h.update(('recon_core/' + relative.as_posix()).encode())
        h.update(source_digest(path).encode())
    return h.hexdigest()


def prepare(cfg, out):
    if cfg.get('video'):
        from .video.extract_frames import prepare_video

        manifest = prepare_video(cfg, out)
    elif cfg.get('delivery'):
        from .io.delivery import load

        manifest, source = load(cfg['delivery'])
        dump(
            out / 'delivery.json',
            dict(package=str(Path(cfg['delivery']).resolve()), pose_source=source, frames=len(manifest['frames'])),
        )
    else:
        manifest = load_manifest(cfg['manifest'])
    if cfg.get('exclude_frames'):
        # Explicit, recorded removal (e.g. frames SfM cannot register); never silent.
        drop = set(int(i) for i in cfg['exclude_frames'])
        missing = drop - {int(r['frame_id']) for r in manifest['frames']}
        if missing:
            raise ValueError(f'exclude_frames not in manifest: {sorted(missing)}')
        manifest = dict(manifest, frames=[r for r in manifest['frames'] if int(r['frame_id']) not in drop])
        dump(out / 'excluded_frames.json', dict(frame_ids=sorted(drop), remaining=len(manifest['frames'])))
    rows = manifest['frames']
    ids = np.asarray([r['frame_id'] for r in rows])
    times = np.asarray([r['timestamp_ns'] for r in rows], np.int64)
    if len(ids) < 2 or len(set(ids.tolist())) != len(ids) or np.any(np.diff(ids) <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError('Manifest needs unique increasing integer IDs and increasing timestamp_ns')
    if ids.dtype.kind not in 'iu':
        raise ValueError('Frame IDs must be integers')
    poses = None
    sparse = {}
    if cfg.get('cameras'):
        with np.load(cfg['cameras'], allow_pickle=False) as z:
            np.testing.assert_array_equal(z['frame_ids'], ids)
            poses = z['c2w'].copy()
            K = z['K'].copy()
            sparse = {key: z[key].copy() for key in ('points', 'colors') if key in z}
    else:
        if 'K' not in manifest and (cfg.get('body_poses') or all('c2w' in r for r in rows)):
            raise ValueError('Calibrated pose modes require manifest K')
        # Image-only SfM estimates its shared camera; this placeholder is never used for rendering.
        K = np.broadcast_to(np.asarray(manifest.get('K', np.eye(3)), float), (len(ids), 3, 3)).copy()
        if cfg.get('body_poses'):
            from .geometry.transforms import interpolate, deduplicate

            with np.load(cfg['body_poses'], allow_pickle=False) as z:
                t, p, q = deduplicate(z['timestamp_ns'], z['position'], z['quaternion_xyzw'])
            poses, valid = interpolate(times, t, p, q, cfg['body_from_camera'], cfg.get('time_offset_ms', 0))
            if not valid.all():
                raise ValueError('Images outside body trajectory; trim them explicitly')
        elif all('c2w' in r for r in rows):
            poses = np.array([r['c2w'] for r in rows], float)
    if poses is not None:
        if (
            poses.shape != (len(ids), 4, 4)
            or not np.isfinite(poses).all()
            or not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-7)
        ):
            raise ValueError('Invalid homogeneous c2w')
        R = poses[:, :3, :3]
        if not np.allclose(R.transpose(0, 2, 1) @ R, np.eye(3), atol=1e-5) or not np.allclose(
            np.linalg.det(R), 1, atol=1e-5
        ):
            raise ValueError('c2w rotations must be proper rotations')
    # Blur reference is the candidate sequence, before motion-keyframe thinning.
    from .keyframes.selection import blur_mask
    import cv2

    sharp = []
    for row in rows:
        im = cv2.imread(row['path'], 0)
        if im is None:
            raise ValueError(f'Cannot decode {row["path"]}')
        sharp.append(cv2.Laplacian(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA), cv2.CV_64F).var())
    blur, _ = blur_mask(sharp)
    if cfg.get('keyframes', {}).get('enabled', False):
        if poses is None:
            raise ValueError('Motion keyframes require poses')
        from .keyframes.selection import select

        options = {k: v for k, v in cfg['keyframes'].items() if k != 'enabled'}
        chosen = select(times, poses, **options)
        ids, times, poses, K = ids[chosen], times[chosen], poses[chosen], K[chosen]
        rows = [rows[i] for i in chosen]
        blur = blur[chosen]
    if cfg.get('selection'):
        selection = load_json(cfg['selection'])
        for key in ('train', 'test'):
            ix = np.asarray(selection[key + '_indices'], int)
            if (ix < 0).any() or (ix >= len(ids)).any() or len(set(ix)) != len(ix):
                raise ValueError('Invalid split indices')
            expected = selection.get(key + '_frame_ids')
            if expected is not None:
                np.testing.assert_array_equal(ids[ix], expected)
        # Frozen splits are authoritative; never recalculate historical blur selection.
    else:
        mode = cfg.get('split', {}).get('mode', 'holdout')
        if mode not in ('holdout', 'all_clear'):
            raise ValueError('Unknown split mode')
        test = np.arange(7, len(ids), 8) if mode == 'holdout' else np.array([], dtype=int)
        candidates = np.flatnonzero(~blur) if cfg.get('split', {}).get('exclude_blur', True) else np.arange(len(ids))
        train = np.setdiff1d(candidates, test)
        selection = dict(
            train_indices=train.tolist(),
            test_indices=test.tolist(),
            train_frame_ids=ids[train].tolist(),
            test_frame_ids=ids[test].tolist(),
            interpolation_frame_ids=ids[test].tolist(),
            holdout_frame_ids=[],
        )
    if set(selection['train_indices']) & set(selection['test_indices']):
        raise ValueError('Train/test overlap')
    subsets = {k: selection[k + '_frame_ids'] for k in ('holdout', 'interpolation') if k + '_frame_ids' in selection}
    if subsets and set().union(*map(set, subsets.values())) != set(ids[selection['test_indices']]):
        raise ValueError('Test subset IDs disagree with frozen test indices')
    if not subsets:
        subsets = {'test': ids[selection['test_indices']].tolist()}
    dump(out / 'manifest.json', dict(manifest, frames=rows))
    dump(out / 'selection.json', selection)
    dump(out / 'subsets.json', subsets)
    arrays = dict(frame_ids=ids, timestamp_ns=times, K=K, blur=blur, **sparse)
    if poses is not None:
        arrays['c2w'] = poses
    np.savez(out / 'inputs.npz', **arrays)


def split_without(selection, subsets, prepared, kept):
    """The prepared split and test subsets for the frames `kept` of `prepared` (frame ids, both in prepared order).

    `*_indices` are re-indexed into the kept frames, `*_frame_ids` and the subsets lose the dropped frames.
    """
    position = {int(frame): k for k, frame in enumerate(kept)}
    prepared = [int(frame) for frame in prepared]
    result = {}
    for key, value in selection.items():
        if key.endswith('_indices'):
            result[key] = [position[prepared[i]] for i in value if prepared[i] in position]
        elif key.endswith('_frame_ids'):
            result[key] = [frame for frame in value if int(frame) in position]
        else:
            result[key] = value
    return result, {name: [frame for frame in frames if int(frame) in position] for name, frames in subsets.items()}


def pose(cfg, out, root):
    from .poses.pipeline import estimate

    with np.load(root / 'prepare/inputs.npz', allow_pickle=False) as z:
        data = dict(z)
    manifest = load_json(root / 'prepare/manifest.json')
    settings = cfg.get('pose', {})
    provider = settings.get('provider', 'external' if 'c2w' in data else 'colmap')
    refine = settings.get('refine', 'none')
    if provider == 'external':
        report = dict(
            provider=provider,
            registered=len(data['frame_ids']),
            input_frames=len(data['frame_ids']),
            units=settings.get('units', 'meters'),
            recomputed=False,
        )
    else:
        prepared = data['frame_ids']
        if provider == 'vggt':
            from .poses.vggt import reconstruct

            data, report = reconstruct(data, manifest, out, cfg.get('sfm', {}), cfg.get('depth', {}))
        else:
            from .sfm import reconstruct

            data, report = reconstruct(data, manifest, out, cfg.get('sfm', {}))
        if len(data['frame_ids']) < len(prepared):
            selection, subsets = split_without(
                load_json(root / 'prepare/selection.json'), load_json(root / 'prepare/subsets.json'), prepared, data['frame_ids']
            )
            dump(out / 'selection.json', selection)
            dump(out / 'subsets.json', subsets)
    height, speed = (settings.get('metric', {}).get(key) for key in ('camera_height_m', 'speed_mps'))
    if report['units'] == 'arbitrary' and (height is not None or speed is not None):
        from .poses.metric import camera_height_scale, speed_scale

        if height is not None:
            scale, diagnostic = camera_height_scale(data['c2w'], data['points'], height)
        else:
            scale, diagnostic = speed_scale(data['c2w'], data['timestamp_ns'], speed)
        data['c2w'][:, :3, 3] *= scale
        data['points'] *= scale
        report.update(units='meters', scale_assumption=diagnostic)
    if refine == 'ba':
        refinement = out / 'refine'
        refinement.mkdir()
        if (out / 'manifest.json').exists():
            manifest = load_json(out / 'manifest.json')
        estimate(data, manifest, refinement, dict(settings, units=report['units']))
        with np.load(refinement / 'cameras.npz', allow_pickle=False) as z:
            data = dict(z)
        report['refinement'] = load_json(refinement / 'report.json')
    report.update(provider=provider, refine=refine)
    np.savez(out / 'cameras.npz', **data)
    dump(out / 'report.json', report)


def depth(cfg, out, root):
    settings = cfg.get('depth', {})
    if settings.get('cache'):
        dump(out / 'source.json', dict(folder=str(Path(settings['cache']).resolve()), reused_predictions=True))
        return
    from .depth.backends import infer_windows

    infer_windows(stage_manifest(root), settings, out)
    dump(out / 'source.json', dict(folder=str(out.resolve()), reused_predictions=False))


def depth_grid(cfg, manifest):
    """(width, height) of the depth maps for these frames.

    `depth.resolution` is written long edge first, as for landscape frames; the model keeps the aspect of the
    frames it is given, so for portrait frames (a phone held upright) the grid is the other way round.
    """
    width, height = cfg.get('depth', {}).get('resolution', [512, 288])
    frame_width, frame_height = manifest['resolution']
    return (height, width) if (frame_height > frame_width) != (height > width) else (width, height)


def fuse(cfg, out, root):
    from .depth.cache import load_windows
    from recon_core.scale import batch_window_scales
    from .fusion.cloud import cache_depths
    from recon_core.fusion import colored_cloud, gaussian_init_cloud

    data = load_cameras(root)
    ids, poses, K = data['frame_ids'], data['c2w'], data['K']
    train = data['train_indices']
    train_ids = ids[train]
    folder = Path(load_json(root / 'depth/source.json')['folder'])
    width, height = depth_grid(cfg, data['manifest'])
    windows = load_windows(folder, ids, width, height)
    retained = []
    for win in windows:
        keep = np.isin(win['idx'], train_ids)
        if keep.any():
            retained.append(
                dict(
                    id=win['id'],
                    idx=win['idx'][keep],
                    new=np.intersect1d(win['new'], train_ids),
                    **{k: win[k][keep] for k in ('vc', 'vR', 'depth', 'conf')},
                )
            )
    del windows
    scales, trajectory, edges = batch_window_scales(retained, ids, poses)
    depths, good, loose, frame_scales = cache_depths(retained, scales, ids, out / 'cache', width, height)
    paths = {str(r['frame_id']): r['path'] for r in data['manifest']['frames']}
    sw, sh = data['manifest']['resolution']
    cap = int(cfg.get('fusion', {}).get('max_points', 300000))
    if not 4 <= cap <= 1000000:
        raise ValueError('Initialization cap must be 4..1000000')
    Kd, depth_train, good_train = resize_K(K[train], width, height, sw, sh), depths[train], good[train]
    extra, revisit = {}, None
    if cfg.get('fusion', {}).get('revisit', False):
        # Surfaces seen again by other windows: one more scale factor per frame (recon_core.revisit); the cache keeps the corrected depth.
        from recon_core.revisit import frame_log_scale_corrections

        owner = np.full(len(ids), -1)
        for k, win in enumerate(retained):
            owner[np.searchsorted(ids, win['new'])] = k
        corrections, revisit = frame_log_scale_corrections(poses[train], Kd, depth_train, good_train, owner[train])
        factor = np.exp(corrections)
        depth_train *= factor[:, None, None].astype(np.float32)
        for row, d, f in zip(train, depth_train, factor):
            depths[row] = d
            frame_scales[row] *= f
        depths.flush()
        extra = dict(revisit_log_scale=corrections)
    xyz, rgb, info = colored_cloud(train_ids, poses[train], Kd, depth_train, good_train, paths, max_points=cap)
    if len(xyz) < 4:
        raise ValueError('Insufficient fused points')
    np.savez(out / 'initial.npz', points=xyz.astype(np.float32), colors=rgb.astype(np.float32))
    # A second cloud for the Gaussians to start from: complete rather than accurate (recon_core.fusion docstring).
    start_cap = int(cfg.get('fusion', {}).get('init_max_points', cap))
    if not 4 <= start_cap <= 5000000:
        raise ValueError('fusion.init_max_points must be 4..5000000')
    start, start_rgb, start_info = gaussian_init_cloud(
        train_ids, poses[train], Kd, depth_train, loose[train], paths, max_points=start_cap
    )
    if len(start) < 4:
        raise ValueError('Insufficient points to initialize Gaussians')
    np.savez(out / 'gaussian_init.npz', points=start.astype(np.float32), colors=start_rgb.astype(np.float32))
    np.savez(out / 'scales.npz', frame_ids=train_ids, scales=frame_scales[train], window_scales=scales, **extra)
    dump(
        out / 'report.json',
        dict(
            frame_count=len(train),
            depth_slots=len(ids),
            populated_slots=int(np.any(good, axis=(1, 2)).sum()),
            points_at_2cm=info['points_2cm'],
            points=info['points'],
            voxel_m=info['voxel_m'],
            gaussian_init=start_info,
            window_scales=scales.tolist(),
            trajectory=trajectory,
            edges=edges,
            **({} if revisit is None else dict(revisit=revisit)),
        ),
    )


def load_cameras(root):
    """Pose-stage cameras with the prepared split and the stage manifest, not normalized."""
    with np.load(root / 'pose/cameras.npz', allow_pickle=False) as z:
        data = dict(z)
    selection = load_json(stage_file(root, 'selection.json'))
    data.update(
        manifest=stage_manifest(root),
        train_indices=np.array(selection['train_indices'], int),
        test_indices=np.array(selection['test_indices'], int),
        arbitrary_scale=load_json(root / 'pose/report.json').get('units') == 'arbitrary',
    )
    return data


def load_data(cfg, root, init=False):
    """Cameras in normalized training coordinates and an image source; the protocol picks the normalization.

    drone: mean training centre and maximum radius (or a fixed normalization), K resized to
    train.resolution with pixel centres. phone: the reference phone run's median/P95 normalization,
    which also drops outlier cameras from the split, and K scaled proportionally to train.max_width.
    The phone protocol always loads the fused initial cloud, which its evaluation reads.
    """
    data = load_cameras(root)
    poses, K = data['c2w'], data['K']
    settings = cfg.get('train', {})
    sw, sh = data['manifest']['resolution']
    phone = cfg.get('protocol', 'drone') == 'phone'
    normalized = poses.astype(np.float32).copy()
    if phone:
        data['registered_indices'] = np.arange(len(data['frame_ids']))
        norm = robust_normalization(data)
        center, meters = np.array(norm['center']), norm['p95_radius']
        normalized[:, :3, 3] = (normalized[:, :3, 3] - center) / meters
        locations = normalized[data['registered_indices'], :3, 3]
        scale = float(np.linalg.norm(locations - locations.mean(0), axis=1).max()) * 1.1
        w = min(settings.get('max_width', 1600) or sw, sw)
        h = int(round(sh * w / sw))
        resized = K.copy()
        resized[:, :2, :] *= w / sw
        data.update(normalization=norm, scale=meters, group='B0', camera_group='provider')
    else:
        fixed = load_json(cfg['normalization_file']) if cfg.get('normalization_file') else cfg.get('normalization')
        center, meters, scale = normalization(poses, data['train_indices'], fixed)
        # The reference drone run cast poses to float32 first, then translated in float64.
        source = normalized if cfg.get('normalization_cast_first', False) else poses
        normalized[:, :3, 3] = (source[:, :3, 3] - center) / meters
        w, h = settings.get('resolution', [640, 360])
        resized = resize_K(K, w, h, sw, sh)
        data.update(group='BEST', camera_group='mixed')
    data.update(
        c2w_m=poses,
        K_native=K,
        c2w=normalized,
        K=resized.astype(np.float32),
        center=center.tolist(),
        meters_per_unit=meters,
        scene_scale=scale,
        width=w,
        height=h,
        image_cache_frames=settings.get('image_cache_frames', 32),
    )
    if init or phone:
        # Training starts from the wider cloud; everything else reads the delivered one.
        dense = init and settings.get('init_cloud', 'footprint') == 'footprint'
        with np.load(root / ('fuse/gaussian_init.npz' if dense else 'fuse/initial.npz'), allow_pickle=False) as z:
            data.update(
                points=((z['points'] - center) / meters).astype(np.float32), colors=z['colors'].astype(np.float32)
            )
    image = image_loader(
        {str(r['frame_id']): r['path'] for r in data['manifest']['frames']}, data['image_cache_frames']
    )
    return data, image


def gpu_data(cfg, root, init=False):
    import torch
    import cv2

    torch.set_num_threads(int(cfg.get('threads', 4)))
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for Gaussian training/evaluation')
    data, image = load_data(cfg, root, init)
    data['c2w'], data['K'] = (torch.from_numpy(data[k]).cuda() for k in ('c2w', 'K'))
    return data, image


def train(cfg, out, root):
    from .gaussian.train import train as fit

    data, image = gpu_data(cfg, root, True)
    if cfg.get('protocol', 'drone') == 'phone':
        dump(out / 'normalization.json', data['normalization'])
    settings = dict(cfg.get('train', {}))
    settings.setdefault('depth_shape', list(reversed(depth_grid(cfg, data['manifest']))))
    fit(data, out, root / 'fuse/cache', image, settings)


def evaluate(cfg, out, root):
    import torch
    from .evaluation.evaluate import evaluate as score

    data, image = gpu_data(cfg, root)
    ck = torch.load(root / 'train/final.pt', map_location='cpu', weights_only=False)
    score(cfg, root, out, data, ck, image)


def export(cfg, out, root):
    from .io.export import export_scene

    data, _ = load_data(cfg, root)
    ex = cfg.get('export', {})
    export_scene(
        root,
        out,
        data,
        reference_drone=ex.get('reference_drone', False),
        name=ex.get('name'),
        subtitle=ex.get('subtitle'),
    )


def run(cfg, through='export', only=None):
    from .config import validate

    validate(cfg)
    key = fingerprint(cfg)
    root = Path(cfg['output']) / key
    root.mkdir(parents=True, exist_ok=True)
    dump(Path(cfg['output']) / 'latest.json', dict(root=str(root.resolve()), fingerprint=key))
    functions = dict(
        prepare=lambda c, o, r: prepare(c, o),
        pose=pose,
        depth=depth,
        fuse=fuse,
        train=train,
        eval=evaluate,
        export=export,
    )
    for stage in STAGES:
        if only and stage != only:
            if STAGES.index(stage) < STAGES.index(only) and not (root / stage / 'complete.json').is_file():
                raise RuntimeError(f'Run prerequisite stage {stage} first')
            continue
        out = root / stage
        marker = out / 'complete.json'
        if marker.exists():
            record = load_json(marker)
            if all((out / p).is_file() and digest(out / p) == sha for p, sha in record['outputs'].items()):
                print(f'{stage}: cached', flush=True)
                if stage == through or only:
                    break
                continue
            raise RuntimeError(f'Cached {stage} artifact changed; choose a fresh output directory')
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f'Incomplete stage {stage} at {out}; inspect it and choose a fresh output directory')
        out.mkdir(exist_ok=True)
        print(f'{stage}: running', flush=True)
        functions[stage](cfg, out, root)
        outputs = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
        dump(marker, dict(fingerprint=key, outputs=outputs))
        if stage == through or only:
            break
    return root
