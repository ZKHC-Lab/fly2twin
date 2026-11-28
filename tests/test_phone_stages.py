"""CPU coverage of phone stage wiring; no GPU kernels or model downloads."""

import json
import shutil

import numpy as np
import pytest

from recon.io.common import dump
from recon.poses.metric import camera_height_scale


def test_metric_plane_scale():
    rng = np.random.default_rng(1)
    poses = np.tile(np.eye(4), (20, 1, 1))
    poses[:, 0, 3] = np.linspace(-1, 1, 20)
    # Optical -Y points up, floor is at +Y.
    points = rng.uniform(-2, 2, (300, 3))
    points[:, 1] = 2
    scale, diagnostic = camera_height_scale(poses, points, 1.4)
    assert scale == 0.7
    assert diagnostic['measured'] is False
    assert diagnostic['inliers'] == 300
    assert diagnostic['plane_normal'] == [0.0, -1.0, 0.0]


@pytest.mark.parametrize('exclude_blur, expected', [(True, [0, 1, 2, 3, 4, 5, 6]), (False, [0, 1, 2, 3, 4, 5, 6, 8])])
def test_split_blur_switch(tmp_path, monkeypatch, exclude_blur, expected):
    import cv2
    from recon.pipeline import prepare
    import recon.keyframes.selection as selection

    image = tmp_path / 'frame.png'
    cv2.imwrite(str(image), np.zeros((10, 20, 3), np.uint8))
    manifest = tmp_path / 'manifest.json'
    dump(
        manifest,
        dict(
            resolution=[20, 10], frames=[dict(frame_id=i, timestamp_ns=i * 1000000, path=str(image)) for i in range(9)]
        ),
    )
    monkeypatch.setattr(selection, 'blur_mask', lambda _: (np.array([False] * 8 + [True]), None))
    out = tmp_path / 'prepare'
    out.mkdir()
    prepare(dict(manifest=str(manifest), split=dict(exclude_blur=exclude_blur)), out)
    split = json.loads((out / 'selection.json').read_text())
    assert split['train_indices'] == expected
    assert split['test_indices'] == [7]


def test_video_prepare_manifest(tmp_path, monkeypatch):
    import cv2
    import recon.video.extract_frames as extract
    from recon.video.extract_frames import prepare_video

    calls = []

    def fake_extract(video, out, **settings):
        calls.append((video, out, settings))
        (tmp_path / 'frames').mkdir()
        for frame in [3, 17]:
            cv2.imwrite(str(tmp_path / 'frames' / f'{frame:05d}.jpg'), np.zeros((10, 20, 3), np.uint8))
        dump(tmp_path / 'extract.json', dict(frames=[dict(frame=3, time_s=0.125), dict(frame=17, time_s=0.708)]))

    monkeypatch.setattr(extract, 'extract_frames', fake_extract)
    manifest = prepare_video(dict(video='clip.MOV', extract=dict(target_frames=400, sfm_multiple=3)), tmp_path)
    assert calls == [('clip.MOV', tmp_path, dict(target_frames=400, sfm_multiple=3))]
    assert manifest['resolution'] == [20, 10]
    assert [r['frame_id'] for r in manifest['frames']] == [3, 17]
    assert [r['timestamp_ns'] for r in manifest['frames']] == [125000000, 708000000]


@pytest.mark.parametrize('setting', [dict(hdr='yes'), dict(train_subset='half')])
def test_extract_rejects_unknown_choices(tmp_path, setting):
    from recon.video.extract_frames import extract_frames

    with pytest.raises(ValueError, match='must be'):
        extract_frames(tmp_path / 'missing.MOV', tmp_path, **setting)


def test_depth_grid_follows_the_orientation_of_the_frames():
    """A phone held upright gives 1080x1920 frames; the model then returns 288x512 maps, not 512x288."""
    from recon.pipeline import depth_grid

    preset = dict(depth=dict(resolution=[512, 288]))
    assert depth_grid(preset, dict(resolution=[1920, 1080])) == (512, 288)
    assert depth_grid(preset, dict(resolution=[1080, 1920])) == (288, 512)
    assert depth_grid({}, dict(resolution=[1280, 720])) == (512, 288)
    assert depth_grid(dict(depth=dict(resolution=[288, 512])), dict(resolution=[1080, 1920])) == (288, 512)
    assert depth_grid(dict(depth=dict(resolution=[384, 384])), dict(resolution=[1080, 1920])) == (384, 384)


@pytest.mark.parametrize(
    'line,flag',
    [
        ('ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers', '-vsync'),
        ('ffmpeg version 5.0.1 Copyright', '-vsync'),
        ('ffmpeg version 5.1.2-3ubuntu1 Copyright', '-fps_mode'),
        ('ffmpeg version 6.1.1-3ubuntu5 Copyright', '-fps_mode'),
        ('ffmpeg version n7.1 Copyright', '-fps_mode'),
        ('ffmpeg version N-112345-gabcdef0 Copyright', '-fps_mode'),  # a git build: newer than any release that lacks it
    ],
)
def test_frame_passthrough_switch_follows_the_ffmpeg_version(monkeypatch, line, flag):
    import types

    import recon.video.extract_frames as extract

    monkeypatch.setattr(extract, 'run', lambda cmd, **kwargs: types.SimpleNamespace(stdout=line + '\nbuilt with gcc'))
    assert extract.passthrough() == [flag, 'passthrough']


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='needs ffmpeg')
def test_extract_indices_follow_decoder_frames_across_timestamp_gap(tmp_path):
    """60 frames with a 5-frame timestamp gap: a constant-rate scan would see 64 and select frames that do not exist."""
    import subprocess
    from recon.video.extract_frames import extract_frames, passthrough

    video = tmp_path / 'gap.mov'
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc=size=64x48:rate=30', '-frames:v', '60',
         '-vf', "setpts='(N+5*gte(N\\,30))/30/TB'", *passthrough(),
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-video_track_timescale', '600', str(video)],
        check=True,
    )
    out = tmp_path / 'out'
    extract_frames(video, out, target_frames=60, min_frames=1, sfm_multiple=1, long_edge=64, scan_width=32)
    record = json.loads((out / 'extract.json').read_text())
    assert record['decoded_frames'] == 60
    assert sorted(int(p.stem) for p in (out / 'frames_sfm').glob('*.jpg')) == list(range(60))


def test_phone_only_train_eval_input_assembly(tmp_path, monkeypatch):
    import cv2
    import torch
    import recon.pipeline as pipeline
    import recon.gaussian.train as trainer
    import recon.evaluation.evaluate as evaluation

    n = 16
    root = tmp_path / 'run'
    for stage in ('prepare', 'pose', 'depth', 'fuse'):
        (root / stage).mkdir(parents=True)
        dump(root / stage / 'complete.json', dict(outputs={}))
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.linspace(-1, 1, n)
    K = np.tile(np.array([[20.0, 0, 10], [0, 20, 5], [0, 0, 1]]), (n, 1, 1))
    ids = np.arange(n) * 10
    path = tmp_path / 'frame.png'
    cv2.imwrite(str(path), np.full((10, 20, 3), 127, np.uint8))
    manifest = dict(
        resolution=[20, 10],
        frames=[dict(frame_id=int(f), timestamp_ns=i * 1000000, path=str(path)) for i, f in enumerate(ids)],
    )
    dump(root / 'prepare/manifest.json', manifest)
    split = dict(train_indices=[i for i in range(n) if i % 8 != 7], test_indices=[7, 15])
    dump(root / 'prepare/selection.json', split)
    np.savez(
        root / 'pose/cameras.npz',
        frame_ids=ids,
        timestamp_ns=np.arange(n),
        c2w=poses,
        K=K,
        points=np.zeros((2, 3)),
        colors=np.zeros((2, 3)),
    )
    dump(root / 'pose/report.json', dict(units='meters'))
    dump(root / 'depth/source.json', dict(folder=str(tmp_path / 'windows')))
    cloud = np.array([[0.0, 0, 2], [1.0, 0, 2], [0.0, 1, 2], [1.0, 1, 2]], np.float32)
    np.savez(root / 'fuse/initial.npz', points=cloud, colors=np.full_like(cloud, 0.5))
    start = np.concatenate([cloud, cloud + [0, 0, 6]]).astype(np.float32)
    np.savez(root / 'fuse/gaussian_init.npz', points=start, colors=np.full_like(start, 0.25))
    cfg = dict(
        protocol='phone',
        manifest=str(tmp_path / 'manifest.json'),
        output=str(tmp_path),
        pose=dict(provider='external'),
        train=dict(max_width=16, image_cache_frames=2),
        eval=dict(aligned_width=960, alignment_steps=200),
    )
    monkeypatch.setattr(pipeline, 'fingerprint', lambda _: 'run')
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    expected, image = pipeline.load_data(cfg, root)
    started, _ = pipeline.load_data(cfg, root, init=True)
    delivered, _ = pipeline.load_data(dict(cfg, train=dict(cfg['train'], init_cloud='delivered')), root, init=True)
    assert len(started['points']) == len(start) and np.all(started['colors'] == 0.25)
    np.testing.assert_array_equal(delivered['points'], expected['points'])
    np.testing.assert_array_equal(expected['K'], (K * np.array([0.8, 0.8, 1])[None, :, None]).astype(np.float32))
    assert len(expected['points']) == len(cloud)
    assert expected['arbitrary_scale'] is False
    assert image.cache_info().maxsize == 2
    seen = []

    def fit(data, out, folder, image, settings):
        seen.append('train')
        np.testing.assert_array_equal(data['points'], started['points'])
        np.testing.assert_array_equal(data['c2w'].numpy(), expected['c2w'])
        assert data['train_indices'].tolist() == split['train_indices']
        assert folder == root / 'fuse/cache'
        assert image(int(ids[0]), 16, 8).shape == (8, 16, 3)
        assert settings['depth_shape'] == [288, 512]
        torch.save(dict(splats={'means': torch.zeros(1, 3)}, stats={}), out / 'final.pt')

    def score(splats, data, image, stats, out, args):
        seen.append('eval')
        assert args.aligned_width == 960 and args.aligned_steps == 200 and args.aligned_colour
        assert args.floater_ratio == 0.7 and args.depth_conf_percentile == 50
        assert args.depth_windows == str(tmp_path / 'windows')
        assert data['test_indices'].tolist() == [7, 15]
        np.testing.assert_array_equal(data['points'], expected['points'])
        assert data['width'] == 16 and image(int(ids[0]), 16, 8).shape == (8, 16, 3)
        dump(out / 'metrics.json', dict(mock=True))

    monkeypatch.setattr(trainer, 'train', fit)
    monkeypatch.setattr(evaluation, 'phone_protocol', score)
    pipeline.run(cfg, only='train')
    pipeline.run(cfg, only='eval')
    assert seen == ['train', 'eval']


def test_pose_stage_rewrites_the_split_when_sfm_drops_frames(tmp_path, monkeypatch):
    import recon.pipeline as pipeline
    import recon.sfm as sfm

    root = tmp_path
    (root / 'prepare').mkdir()
    out = root / 'pose'
    out.mkdir()
    ids = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    np.savez(root / 'prepare/inputs.npz', frame_ids=ids, timestamp_ns=ids, blur=np.zeros(8, bool), K=np.tile(np.eye(3), (8, 1, 1)))
    rows = [dict(frame_id=int(f), timestamp_ns=int(f), path=f'{f}.jpg') for f in ids]
    dump(root / 'prepare/manifest.json', dict(resolution=[4, 3], frames=rows))
    dump(root / 'prepare/selection.json', dict(train_indices=[0, 1, 2, 3, 4, 5, 6], test_indices=[7], test_frame_ids=[80]))
    dump(root / 'prepare/subsets.json', dict(test=[80]))
    keep = ids != 30

    def reconstruct(data, manifest, destination, settings):
        dump(destination / 'manifest.json', dict(manifest, frames=[r for r, ok in zip(manifest['frames'], keep) if ok]))
        cameras = dict(frame_ids=ids[keep], timestamp_ns=ids[keep], blur=np.zeros(7, bool), K=np.tile(np.eye(3), (7, 1, 1)),
                       c2w=np.tile(np.eye(4), (7, 1, 1)), points=np.ones((4, 3)), colors=np.zeros((4, 3)))
        return cameras, dict(units='arbitrary', registered=7, input_frames=8, dropped_frames=[30])

    monkeypatch.setattr(sfm, 'reconstruct', reconstruct)
    pipeline.pose(dict(pose=dict(provider='colmap')), out, root)
    data = pipeline.load_cameras(root)
    assert data['frame_ids'].tolist() == [10, 20, 40, 50, 60, 70, 80] and len(data['manifest']['frames']) == 7
    assert data['frame_ids'][data['train_indices']].tolist() == [10, 20, 40, 50, 60, 70]
    assert data['frame_ids'][data['test_indices']].tolist() == [80]
    assert json.loads((out / 'subsets.json').read_text()) == dict(test=[80])
    assert json.loads((out / 'report.json').read_text())['dropped_frames'] == [30]


@pytest.mark.parametrize('provider', ['external', 'colmap', 'vggt'])
def test_provider_metric_then_refine(tmp_path, monkeypatch, provider):
    import recon.pipeline as pipeline
    import recon.poses.pipeline as poses
    import recon.poses.metric as metric
    import recon.sfm as sfm
    import recon.poses.vggt as vggt

    root = tmp_path
    (root / 'prepare').mkdir()
    out = root / 'pose'
    out.mkdir()
    cameras = dict(
        frame_ids=np.array([0, 1]),
        timestamp_ns=np.array([0, 1]),
        blur=np.array([False, False]),
        c2w=np.tile(np.eye(4), (2, 1, 1)),
        K=np.tile(np.eye(3), (2, 1, 1)),
        points=np.ones((4, 3)),
        colors=np.zeros((4, 3)),
    )
    cameras['c2w'][:, 0, 3] = [1, 2]
    np.savez(root / 'prepare/inputs.npz', **cameras)
    dump(root / 'prepare/manifest.json', dict(frames=[]))
    monkeypatch.setattr(metric, 'camera_height_scale', lambda *args: (2.0, dict(measured=False)))
    calls = []

    def estimate(data, manifest, destination, settings):
        calls.append(destination.name)
        np.testing.assert_array_equal(data['c2w'][:, 0, 3], [2, 4])
        np.testing.assert_array_equal(data['points'], np.ones((4, 3)) * 2)
        assert settings['units'] == 'meters'
        np.savez(destination / 'cameras.npz', **data)
        dump(destination / 'report.json', dict(refine='ba'))

    monkeypatch.setattr(poses, 'estimate', estimate)

    def reconstruct(name):
        calls.append(name)
        return dict(cameras, c2w=cameras['c2w'].copy(), points=cameras['points'].copy()), dict(units='arbitrary')

    monkeypatch.setattr(sfm, 'reconstruct', lambda data, manifest, destination, settings: reconstruct('colmap'))
    monkeypatch.setattr(vggt, 'reconstruct', lambda data, manifest, destination, settings, depth: reconstruct('vggt'))
    pipeline.pose(
        dict(pose=dict(provider=provider, refine='ba', units='arbitrary', metric=dict(camera_height_m=1.4))), out, root
    )
    assert calls == ([] if provider == 'external' else [provider]) + ['refine']
    report = json.loads((out / 'report.json').read_text())
    assert report['provider'] == provider and report['refine'] == 'ba' and report['units'] == 'meters'


def test_window_prior_uses_owner_and_phone_confidence(tmp_path):
    from recon.evaluation.evaluate import WindowPrior, OmegaPrior

    (tmp_path / 'K000').mkdir()
    depth = np.arange(8, dtype=np.float32).reshape(1, 2, 4) + 1
    confidence = depth.copy()
    np.savez(tmp_path / 'K000/prediction.npz', depth=depth[..., None], depth_conf=confidence)
    dump(tmp_path / 'windows.json', dict(windows=[dict(id='K000', idx=[7], new=[7])]))
    np.save(tmp_path / 'depth.npy', depth)
    np.save(tmp_path / 'conf.npy', confidence)
    np.save(tmp_path / 'frame_ids.npy', [7])
    a = WindowPrior(tmp_path, [7], 8, 4)(0)
    b = OmegaPrior(tmp_path, [7], 8, 4)(0)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize('pose', [dict(mode='cached'), dict(provider='bad'), dict(provider='glomap'), dict(provider='colmap', units='meters')])
def test_pose_configuration_boundary(pose):
    from recon.config import validate

    with pytest.raises(ValueError, match='Invalid reconstruction configuration'):
        validate(dict(manifest='frames.json', output='out', pose=pose))


def test_median_depth_is_none_when_the_frame_sees_no_point(tmp_path):
    """A NaN here used to abort the phone evaluation while writing metrics.json, after the whole training."""
    pytest.importorskip('torch')
    from recon.evaluation.evaluate import median_depth

    K = np.array([[20.0, 0, 16], [0, 20.0, 8], [0, 0, 1]])
    points = np.array([[0.0, 0, 2], [0.1, 0, 3], [-0.1, 0.1, 4]])
    assert median_depth(np.eye(4), K, points, 32, 16) == 3.0
    behind = np.eye(4)
    behind[:3, :3] = np.diag([-1.0, 1, -1])  # looking the other way
    aside = np.eye(4)
    aside[0, 3] = 50.0  # the points are in front but outside the image
    for pose in (behind, aside):
        assert median_depth(pose, K, points, 32, 16) is None
    dump(tmp_path / 'value.json', dict(median_depth=median_depth(behind, K, points, 32, 16)))


def test_speed_metric_scales_poses_and_points(tmp_path, monkeypatch):
    import recon.pipeline as pipeline
    import recon.poses.vggt as vggt

    root = tmp_path
    (root / 'prepare').mkdir()
    out = root / 'pose'
    out.mkdir()
    n = 80
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, 2, 3] = np.arange(n) * 0.01  # 0.01 units per frame at 10 frames per second
    cameras = dict(
        frame_ids=np.arange(n), timestamp_ns=(np.arange(n) * 1e8).astype(np.int64), blur=np.zeros(n, bool), c2w=c2w,
        K=np.tile(np.eye(3), (n, 1, 1)), points=np.ones((4, 3)), colors=np.zeros((4, 3)),
    )
    np.savez(root / 'prepare/inputs.npz', **{k: v for k, v in cameras.items() if k not in ('c2w', 'points', 'colors')})
    dump(root / 'prepare/manifest.json', dict(frames=[]))
    monkeypatch.setattr(vggt, 'reconstruct', lambda data, manifest, destination, settings, depth: (
        dict(cameras, c2w=c2w.copy(), points=cameras['points'].copy()), dict(units='arbitrary')))
    pipeline.pose(dict(pose=dict(provider='vggt', metric=dict(speed_mps=1.0)), depth=dict(weights='w.pt')), out, root)
    with np.load(out / 'cameras.npz') as z:
        np.testing.assert_allclose(z['c2w'][:, 2, 3], np.arange(n) * 0.1)
        np.testing.assert_allclose(z['points'], 10.0)
    report = json.loads((out / 'report.json').read_text())
    assert report['units'] == 'meters' and report['provider'] == 'vggt'
    assert report['scale_assumption']['measured'] is False and report['scale_assumption']['assumed_median_speed_mps'] == 1.0
