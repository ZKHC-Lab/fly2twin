"""Phone camera adapter to the shared Gaussian trainer.

Preprocessing: python -m recon.video.extract_frames / recon.video.sfm.
This entry consumes their cameras.npz and undistorted images, not historical code.
"""
import argparse
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import yaml
from .normalization import normalize
from recon.io.common import dump


class Images:
    """Bounded uint8 cache; same OpenCV resize as the reference stack."""
    def __init__(self, folder, frame_ids, width, cache_frames=32, paths=None):
        if cache_frames < 1:
            raise ValueError('image_cache_frames must be positive')
        self.folder, self.frame_ids = folder, frame_ids
        self.paths = paths if paths is not None else [folder / f'{int(frame):05d}.png' for frame in frame_ids]
        self.get = lru_cache(maxsize=cache_frames)(self._read)
        sample = cv2.imread(str(self.paths[0]))
        if sample is None:
            raise FileNotFoundError('First undistorted frame missing')
        self.native_width = sample.shape[1]
        self.width = min(width, self.native_width) if width else self.native_width
        self.height = int(round(sample.shape[0] * self.width / self.native_width))

    def _read(self, index):
        frame = int(self.frame_ids[index])
        rgb = cv2.imread(str(self.paths[index]))
        if rgb is None:
            raise FileNotFoundError(f'Undistorted frame {frame}')
        if self.width != self.native_width:
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), frame


def load_data(out, width=1600, cache_frames=32):
    with np.load(out / 'cameras.npz', allow_pickle=False) as z:
        data = dict(z)
    images = Images(out / 'undistorted', data['frame_ids'], width, cache_frames)
    return prepare_data(data, images), images


def prepare_data(data, images):
    """Shared phone numeric preparation for historical and staged entry points."""
    norm = normalize(data)
    center, scale = np.array(norm['center']), norm['p95_radius']
    c2w = data['c2w'].astype(np.float32).copy()
    c2w[:, :3, 3] = (c2w[:, :3, 3] - center) / scale
    locations = c2w[data['registered_indices'], :3, 3]
    scene_scale = float(np.linalg.norm(locations-locations.mean(0), axis=1).max()) * 1.1
    K = data['K'].copy()
    K[:, :2, :] *= images.width / images.native_width
    data.update(c2w=c2w, K=K.astype(np.float32),
                points=((data['points']-center)/scale).astype(np.float32),
                colors=data['colors'].astype(np.float32), center=center.tolist(),
                scale=scale, meters_per_unit=scale, scene_scale=scene_scale,
                normalization=norm, group='video', camera_group='sfm', arbitrary_scale=True,
                width=images.width, height=images.height, image_cache_frames=images.get.cache_info().maxsize)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--video', type=Path, help='optionally run extraction and SfM before training')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    settings = cfg['train']
    if args.video:
        if args.eval_only:
            parser.error('--video and --eval-only cannot be combined')
        for module, section, extra in [('extract_frames', 'extract', ['--video', str(args.video)]),
                                        ('sfm', 'sfm', [])]:
            command=[sys.executable, '-m', f'recon.video.{module}', '--out', str(args.out), *extra]
            for key,value in cfg.get(section,{}).items():
                command.extend(['--'+key.replace('_','-'), str(value)])
            subprocess.run(command,check=True)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    data, images = load_data(args.out, settings.get('max_width', 1600), settings.get('image_cache_frames', 32))
    gs = args.out / 'gs'; gs.mkdir(exist_ok=True)
    dump(gs / 'normalization.json', data['normalization'])
    data['c2w'] = torch.from_numpy(data['c2w']).cuda()
    data['K'] = torch.from_numpy(data['K']).cuda()
    lookup = {int(frame): i for i, frame in enumerate(data['frame_ids'])}
    if not args.eval_only:
        from recon.gaussian.train import train
        train(data, gs, None, lambda frame,w,h: images.get(lookup[frame])[0], settings)
    checkpoint = torch.load(gs/'final.pt', map_location='cpu', weights_only=False)
    splats = {k:v.cuda() for k,v in checkpoint['splats'].items()}
    from .evaluation import evaluate
    evaluation = cfg.get('eval', {})
    options = SimpleNamespace(omega_dir=None, depth_conf_percentile=50, floater_ratio=.7,
                              aligned_steps=evaluation.get('alignment_steps', 200),
                              aligned_width=evaluation.get('aligned_width', 960), aligned_colour=True)
    with torch.no_grad():
        evaluate(splats, data, images, checkpoint['stats'], gs, options)


if __name__ == '__main__':
    main()
