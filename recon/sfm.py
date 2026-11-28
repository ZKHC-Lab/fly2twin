# SPDX-License-Identifier: Apache-2.0
"""Image-only poses with COLMAP's incremental mapper, and the steps the `vggt` provider shares with it.

One shared camera for the whole sequence, sequential matching plus an optional loop-closure
subset (`match_features`), incremental mapping, then the largest model is undistorted (`undistort`:
OpenCV, alpha 0, same size, no crop). Frames the largest model did not register are dropped and
listed in the report when at least `min_registration_rate` of the frames registered; below that
the run is rejected. Output is in arbitrary SfM units.
"""

from pathlib import Path
import numpy as np
from recon.io.common import dump

# Feature and matching defaults per provider; the `sfm` config section overrides either.
DEFAULTS = dict(
    vggt=dict(camera_model='SIMPLE_RADIAL', threads=8, loop_keyframes=60),
    colmap=dict(camera_model='OPENCV', threads=4, loop_keyframes=0),
)
SETTINGS = dict(
    focal_factor=1.2,
    overlap=20,
    loop_max_pairs=600,
    loop_min_gap=0,
    max_features=8192,
    peak_threshold=0.0,
    min_model_size=10,
    min_registration_rate=0.99,
)


def loop_pairs(names, keyframes, max_pairs, min_gap):
    """Pairs among a sparse keyframe subset, only where the time gap is large."""
    total = len(names)
    stride = max(1, total // max(1, keyframes))
    keys = list(range(0, total, stride))
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if b - a < min_gap:
                continue
            pairs.append((a, b))
    pairs.sort(key=lambda pair: pair[1] - pair[0])
    if max_pairs and len(pairs) > max_pairs:
        keep = np.linspace(0, len(pairs) - 1, max_pairs).astype(int)
        pairs = [pairs[i] for i in keep]
    return [(names[a], names[b]) for a, b in pairs], stride


def model_stats(rec):
    rec.update_point_3d_errors()
    observations, errors = [], []
    for image_id in rec.reg_image_ids():
        image = rec.images[image_id]
        count = 0
        for point in image.points2D:
            if point.has_point3D():
                xyz = rec.points3D[point.point3D_id].xyz
                uv = image.camera.img_from_cam(image.cam_from_world() * xyz)
                errors.append(float(np.linalg.norm(uv - point.xy)))
                count += 1
        observations.append(count)
    return dict(
        points=rec.num_points3D(),
        mean_reprojection_error=float(rec.compute_mean_reprojection_error()),
        observation_weighted_mean_error=float(np.mean(errors)) if errors else None,
        observations_per_image=dict(
            zip(['P0', 'P10', 'P50', 'P90'], np.percentile(observations, [0, 10, 50, 90]).tolist())
        )
        if observations
        else {},
    )


def health(c2w, frame_ids):
    """Flag camera-centre jumps: a step >5x the median, or a >60 deg axis turn."""
    steps = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    median = float(np.median(steps)) if len(steps) else 0.0
    jumps = set()
    for i, step in enumerate(steps):
        angle = float(np.degrees(np.arccos(np.clip(c2w[i, :3, 2] @ c2w[i + 1, :3, 2], -1, 1))))
        if step > 5 * median or angle > 60:
            jumps.update((int(frame_ids[i]), int(frame_ids[i + 1])))
    return dict(median_step=median, jump_frames=sorted(jumps), jump_count=len(jumps))


def registered_only(data, manifest, names, kept):
    """The prepared arrays, manifest rows and image names of the frames whose image name is in `kept`."""
    keep = np.array([name in kept for name in names], bool)
    data = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in data.items()}
    manifest = dict(manifest, frames=[row for row, ok in zip(manifest['frames'], keep) if ok])
    return data, manifest, [name for name, ok in zip(names, keep) if ok]


def intrinsics(camera):
    params = np.asarray(camera.params, dtype=np.float64)
    model = camera.model_name
    if model == 'OPENCV':
        fx, fy, cx, cy = params[0:4]
        distortion = params[4:8]
    elif model == 'SIMPLE_RADIAL':  # f, cx, cy, k
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        distortion = np.array([params[3], 0.0, 0.0, 0.0])
    elif model == 'RADIAL':  # f, cx, cy, k1, k2
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        distortion = np.array([params[3], params[4], 0.0, 0.0])
    elif model == 'PINHOLE':  # fx, fy, cx, cy
        fx, fy, cx, cy = params[0:4]
        distortion = np.zeros(4)
    elif model == 'SIMPLE_PINHOLE':  # f, cx, cy
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        distortion = np.zeros(4)
    else:
        raise RuntimeError(f'Unsupported camera model: {model}')
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64), distortion


def match_features(manifest, out, options):
    """SIFT features of the prepared frames, sequential matches and the loop-pair subset.

    Returns (database, image folder, image names in manifest order, loop report entries).
    """
    import pycolmap as pc

    threads = options['threads']
    images = out / 'images'
    images.mkdir()
    names = []
    for i, row in enumerate(manifest['frames']):
        name = f'{i:08d}{Path(row["path"]).suffix}'
        (images / name).symlink_to(Path(row['path']).resolve())
        names.append(name)
    database = out / 'database.db'
    reader = pc.ImageReaderOptions(
        camera_model=options['camera_model'], default_focal_length_factor=options['focal_factor']
    )
    sift = pc.SiftExtractionOptions(max_num_features=options['max_features'])
    if options['peak_threshold']:
        sift.peak_threshold = options['peak_threshold']
    pc.extract_features(
        database,
        images,
        image_names=names,
        camera_mode=pc.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=pc.FeatureExtractionOptions(num_threads=threads, use_gpu=False, sift=sift),
        device=pc.Device.cpu,
    )
    pc.match_sequential(
        database,
        matching_options=pc.FeatureMatchingOptions(num_threads=threads, use_gpu=False),
        pairing_options=pc.SequentialPairingOptions(
            overlap=options['overlap'], quadratic_overlap=True, loop_detection=False, num_threads=threads
        ),
        device=pc.Device.cpu,
    )
    loops = dict(loop_pairs=0, loop_warning=None)
    if options['loop_keyframes'] > 0:
        pairs, stride = loop_pairs(
            names,
            options['loop_keyframes'],
            options['loop_max_pairs'],
            options['loop_min_gap'] or 2 * options['overlap'],
        )
        listing = out / 'loop_pairs.txt'
        listing.write_text(''.join(f'{a} {b}\n' for a, b in pairs))
        try:
            pc.match_image_pairs(
                database,
                matching_options=pc.FeatureMatchingOptions(num_threads=threads, use_gpu=False),
                pairing_options=pc.ImportedPairingOptions(match_list_path=str(listing)),
                device=pc.Device.cpu,
            )
            loops.update(loop_pairs=len(pairs), loop_stride=stride)
        except Exception as error:  # loop closure is an improvement, not a precondition
            loops['loop_warning'] = f'{type(error).__name__}: {error}'
    return database, images, names, loops


def undistort(manifest, names, images, K, distortion, size, out):
    """Write the frames undistorted (OpenCV, alpha 0, same size) to OUT/undistorted and OUT/manifest.json; returns new K."""
    import cv2

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, size, 0, size)
    mx, my = cv2.initUndistortRectifyMap(K, distortion, None, new_K, size, cv2.CV_32FC1)
    undistorted = out / 'undistorted'
    undistorted.mkdir()
    for i, name in enumerate(names):
        source = cv2.imread(str(images / name))
        if source is None or source.shape[:2] != (size[1], size[0]):
            raise RuntimeError(f'Bad frame {name}: expected {size[0]}x{size[1]}')
        target = undistorted / f'{Path(name).stem}.png'
        if not cv2.imwrite(str(target), cv2.remap(source, mx, my, cv2.INTER_LINEAR)):
            raise OSError(target)
        manifest['frames'][i]['path'] = str(target.resolve())
    dump(out / 'manifest.json', manifest)
    return new_K


def reconstruct(data, manifest, out, settings):
    """Run incremental SfM on the prepared frames; return (cameras, report) and write the undistorted manifest."""
    import cv2
    import pycolmap as pc

    options = {**SETTINGS, **DEFAULTS['colmap'], **settings}
    threads = options['threads']
    cv2.setNumThreads(1)
    database, images, names, loops = match_features(manifest, out, options)
    n = len(names)
    report = dict(
        provider='colmap',
        units='arbitrary',
        input_frames=n,
        camera_model=options['camera_model'],
        pycolmap_version=pc.__version__,
        **loops,
    )
    models_dir = out / 'models'
    models_dir.mkdir()
    models = pc.incremental_mapping(
        database,
        images,
        models_dir,
        options=pc.IncrementalPipelineOptions(
            num_threads=threads,
            random_seed=42,
            multiple_models=True,
            min_model_size=options['min_model_size'],
            init_num_trials=200,
            ba_refine_focal_length=True,
            ba_refine_principal_point=True,
            ba_refine_extra_params=True,
            ba_use_gpu=False,
        ),
    )
    counts = {str(k): int(v.num_reg_images()) for k, v in models.items()}
    largest = max(models, key=lambda k: models[k].num_reg_images()) if models else None
    registered = counts.get(str(largest), 0)
    kept = {models[largest].images[i].name for i in models[largest].reg_image_ids()} if models else set()
    report.update(
        registered=registered,
        registration_rate=registered / n,
        models=counts,
        unregistered_frames=[int(f) for f, name in zip(data['frame_ids'], names) if name not in kept],
        disconnected_submodels=max(0, len(models) - 1),
        largest_model=largest,
        recommendation=(
            'When the model fragments (white walls, weak texture), use pose.provider: vggt, which keeps every frame'
            if registered < n
            else None
        ),
    )
    minimum = float(options['min_registration_rate'])
    report.update(min_registration_rate=minimum, dropped_frames=[])
    if largest is None or registered / n < minimum:
        dump(out / 'report.json', report)
        raise RuntimeError(
            f'Partial reconstruction: {registered} of {n} frames registered, below sfm.min_registration_rate '
            f'{minimum:g}; registration and disconnected models are recorded in report.json. '
            'Provide poses, exclude_frames, or lower the rate.'
        )
    if registered < n:
        # Recorded, not silent: the frames are listed here and the pose stage rewrites the split without them.
        data, manifest, names = registered_only(data, manifest, names, kept)
        report.update(dropped_frames=report['unregistered_frames'])
        print(f'pose: dropped {n - registered} unregistered of {n} frames: {report["unregistered_frames"]}', flush=True)
        n = registered
    rec = models[largest]
    report.update(model_stats(rec))
    camera = next(iter(rec.cameras.values()))
    K, distortion = intrinsics(camera)
    size = (int(camera.width), int(camera.height))
    lookup = {im.name: im for im in rec.images.values()}
    c2w = np.tile(np.eye(4), (n, 1, 1))
    for i, name in enumerate(names):
        c2w[i, :3] = lookup[name].cam_from_world().inverse().matrix()
    new_K = undistort(manifest, names, images, K, distortion, size, out)
    report.update(
        original_K=K.tolist(),
        distortion=distortion.tolist(),
        undistorted_K=new_K.tolist(),
        health=health(c2w, data['frame_ids']),
    )
    cameras = dict(
        frame_ids=data['frame_ids'],
        timestamp_ns=data['timestamp_ns'],
        c2w=c2w,
        K=np.repeat(new_K[None], n, axis=0),
        blur=data['blur'],
        points=np.array([p.xyz for p in rec.points3D.values()], np.float32).reshape(-1, 3),
        colors=(np.array([p.color for p in rec.points3D.values()], float).reshape(-1, 3) / 255).astype(np.float32),
    )
    return cameras, report
