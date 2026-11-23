"""Adapters from phone preprocessing/training conventions to recon stages."""
from pathlib import Path
import numpy as np
from recon.io.common import load_json, stage_manifest, dump


def arguments(settings):
    args = []
    for key, value in settings.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args.extend([flag, str(value)])
    return args


def prepare_video(cfg, out):
    import cv2
    from .extract_frames import main
    main(['--video', cfg['video'], '--out', str(out), *arguments(cfg.get('extract', {}))])
    extracted = load_json(out / 'extract.json')
    rows = [dict(frame_id=row['frame'], timestamp_ns=round(row['time_s']*1e9),
                 path=str((out / 'frames' / f'{row["frame"]:05d}.jpg').resolve()))
            for row in extracted['frames']]
    sample = cv2.imread(rows[0]['path'])
    return dict(resolution=[sample.shape[1], sample.shape[0]], frames=rows)


def glomap(data, manifest, out, settings):
    from .sfm import main
    images = out / 'images'
    images.mkdir()
    for row in manifest['frames']:
        source = Path(row['path'])
        (images / f'{row["frame_id"]:05d}{source.suffix}').symlink_to(source.resolve())
    options = dict(settings, sfm_mode='global')
    main(['--out', str(out), '--images', str(images), '--keep', str(images), *arguments(options)])
    with np.load(out / 'cameras.npz', allow_pickle=False) as z:
        cameras = dict(z)
    report = load_json(out / 'sfm.json')
    if len(cameras['registered_indices']) != len(data['frame_ids']):
        raise RuntimeError('GLOMAP did not register every prepared frame; see pose/sfm.json')
    cameras.update(timestamp_ns=data['timestamp_ns'], blur=data['blur'])
    for row in manifest['frames']:
        row['path'] = str((out / 'undistorted' / f'{row["frame_id"]:05d}.png').resolve())
    dump(out / 'manifest.json', manifest)
    report.update(provider='glomap', units='arbitrary', registered=len(cameras['registered_indices']))
    return cameras, report


def load_phone_data(cfg, root):
    from .pipeline import prepare_data, Images
    with np.load(root / 'pose/cameras.npz', allow_pickle=False) as z:
        data = dict(z)
    poses, native_K = data['c2w'], data['K']
    manifest = stage_manifest(root)
    selection = load_json(root / 'prepare/selection.json')
    data.update(train_indices=np.array(selection['train_indices'], int),
                test_indices=np.array(selection['test_indices'], int),
                registered_indices=np.arange(len(data['frame_ids'])))
    with np.load(root / 'fuse/initial.npz', allow_pickle=False) as z:
        data.update(points=z['points'], colors=z['colors'])
    settings = cfg.get('train', {})
    paths = [row['path'] for row in manifest['frames']]
    images = Images(None, data['frame_ids'], settings.get('max_width', 1600),
                    settings.get('image_cache_frames', 32), paths=paths)
    data = prepare_data(data, images)
    data.update(c2w_m=poses, K_native=native_K, manifest=manifest,
                arbitrary_scale=load_json(root / 'pose/report.json')['units'] == 'arbitrary',
                group='B0', camera_group='provider')
    return data, images


def gpu_phone_data(cfg, root):
    import cv2
    import torch
    torch.set_num_threads(int(cfg.get('threads', 4)))
    cv2.setNumThreads(1)
    data, images = load_phone_data(cfg, root)
    data['c2w'], data['K'] = (torch.from_numpy(data[key]).cuda() for key in ('c2w', 'K'))
    return data, images
