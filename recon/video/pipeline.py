"""Phone camera data for the shared Gaussian trainer: bounded image cache and normalization."""
from functools import lru_cache
import cv2
import numpy as np
from .normalization import normalize


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


def prepare_data(data, images):
    """Normalize phone cameras and sparse points; scale K to the cached image width."""
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

