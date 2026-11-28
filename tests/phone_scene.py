"""Synthetic phone video for CPU tests: a textured room filmed by a sideways-moving camera.

Cameras use the optical convention (x right, y down, z forward); the floor lies 1.4 m
below every camera, so the camera-height metric adapter recovers metres. Depth windows
are the true depths, cameras and intrinsics in model units (half a metre), as a VGGT window
would give them; the `vggt` pose provider and the fusion stage both read them.
"""

import json
import subprocess
from pathlib import Path
import numpy as np

FRAMES, WIDTH, HEIGHT = 40, 320, 180
DEPTH_SIZE = (128, 72)
# Planes as (axis, offset): floor y=1.4, ceiling y=-1.2, walls x=+-1.8, back wall z=2.5.
PLANES = [(1, 1.4), (1, -1.2), (0, -1.8), (0, 1.8), (2, 2.5)]
MODEL_UNITS = 0.5


def cameras():
    t = np.linspace(0, 1, FRAMES)
    poses = np.tile(np.eye(4), (FRAMES, 1, 1))
    for i, s in enumerate(t):
        yaw, pitch = np.radians(-15 + 30 * s), np.radians(-28)
        turn = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        tilt = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        poses[i, :3, :3] = turn @ tilt
        poses[i, :3, 3] = [-0.8 + 1.6 * s, 0.05 * np.sin(6 * s), -0.3 + 0.4 * s]
    K = np.array([[260.0, 0, WIDTH / 2], [0, 260.0, HEIGHT / 2], [0, 0, 1]])
    return poses, K


def texture(points, axis):
    """Random cells at three incommensurate scales: distinctive corners for SIFT."""
    uv = np.delete(points, axis, 1)
    value = np.zeros(len(uv))
    for scale, weight in ((3.1, 0.45), (7.3, 0.35), (17.9, 0.2)):
        cells = np.floor(uv * scale + scale).astype(np.int64)
        key = (cells[:, 0] * 73856093) ^ (cells[:, 1] * 19349663) ^ (axis * 83492791 + int(scale * 10))
        value += weight * ((key * 2654435761) % 1009) / 1008.0
    return value


def render(pose, K, width, height):
    """Per-pixel ray casting against the room planes: grey levels and optical depth."""
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    world = rays @ pose[:3, :3].T
    origin = pose[:3, 3]
    best, grey = np.full(len(rays), np.inf), np.zeros(len(rays))
    for axis, offset in PLANES:
        with np.errstate(divide='ignore', invalid='ignore'):
            distance = (offset - origin[axis]) / world[:, axis]
        hit = (distance > 0) & (distance < best)
        points = origin + distance[hit, None] * world[hit]
        best[hit] = distance[hit]
        grey[hit] = texture(points, axis)
    return grey.reshape(height, width), best.reshape(height, width)


def write_video(folder):
    """Encode the frames losslessly-enough with x264; returns the video path."""
    import cv2

    poses, K = cameras()
    frames = Path(folder) / 'rendered'
    frames.mkdir(parents=True)
    for i, pose in enumerate(poses):
        grey, _ = render(pose, K, WIDTH, HEIGHT)
        rgb = np.stack([grey, 0.8 * grey + 0.2, 1 - 0.6 * grey], -1)
        assert cv2.imwrite(str(frames / f'{i:05d}.png'), np.rint(rgb * 255).astype(np.uint8))
    video = Path(folder) / 'phone.mp4'
    subprocess.run(
        [
            'ffmpeg',
            '-v',
            'error',
            '-nostdin',
            '-framerate',
            '10',
            '-i',
            str(frames / '%05d.png'),
            '-c:v',
            'libx264',
            '-preset',
            'ultrafast',
            '-crf',
            '8',
            '-threads',
            '1',
            '-pix_fmt',
            'yuv420p',
            str(video),
        ],
        check=True,
    )
    return video


def write_depth_cache(folder, windows=((0, 24, 0, 24), (16, 40, 24, 40))):
    """Two Omega-like windows; extrinsics and depth in MODEL_UNITS of the true scene."""
    poses, K = cameras()
    width, height = DEPTH_SIZE
    small = K.copy()
    small[0] *= width / WIDTH
    small[1] *= height / HEIGHT
    folder = Path(folder)
    rows = []
    for k, (start, stop, first_new, last_new) in enumerate(windows):
        name = f'K{k:03d}'
        idx = list(range(start, stop))
        depth = np.stack([render(poses[i], small, width, height)[1] for i in idx])
        w2c = np.linalg.inv(poses[idx])[:, :3]
        w2c[:, :, 3] *= MODEL_UNITS
        conf = np.tile(np.linspace(1, 2, width * height, dtype=np.float32).reshape(height, width), (len(idx), 1, 1))
        (folder / name).mkdir(parents=True)
        np.savez(
            folder / name / 'prediction.npz',
            depth=(depth * MODEL_UNITS).astype(np.float32)[..., None],
            depth_conf=conf,
            extrinsic=w2c.astype(np.float32),
            intrinsic=np.repeat(small[None], len(idx), axis=0).astype(np.float32),
            frame_ids=np.asarray(idx),
        )
        rows.append(dict(id=name, idx=idx, new=list(range(first_new, last_new))))
    (folder / 'windows.json').write_text(json.dumps(dict(windows=rows)))
    return folder


def config(folder, video, cache):
    return dict(
        protocol='phone',
        video=str(video),
        output=str(Path(folder) / 'output'),
        extract=dict(
            target_frames=FRAMES,
            min_frames=8,
            sfm_multiple=1,
            train_subset='all',
            long_edge=WIDTH,
            scan_width=160,
            quality=2,
            hdr='off',
        ),
        sfm=dict(camera_model='SIMPLE_RADIAL', overlap=10, loop_keyframes=0, max_features=4096, threads=1),
        pose=dict(provider='vggt', refine='none', metric=dict(camera_height_m=1.4)),
        split=dict(mode='holdout', exclude_blur=False),
        depth=dict(cache=str(cache), resolution=list(DEPTH_SIZE)),
        fusion=dict(max_points=100000),
        train=dict(max_width=WIDTH, image_cache_frames=8),
    )
