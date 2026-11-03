"""CPU coverage of phone stage wiring; no GPU kernels or model downloads."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

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
    assert scale == .7
    assert diagnostic['measured'] is False
    assert diagnostic['inliers'] == 300
    assert diagnostic['plane_normal'] == [0., -1., 0.]


@pytest.mark.parametrize('exclude_blur, expected', [(True, [0, 1, 2, 3, 4, 5, 6]), (False, [0, 1, 2, 3, 4, 5, 6, 8])])
def test_split_blur_switch(tmp_path, monkeypatch, exclude_blur, expected):
    import cv2
    from recon.pipeline import prepare
    import recon.keyframes.selection as selection
    image = tmp_path / 'frame.png'
    cv2.imwrite(str(image), np.zeros((10, 20, 3), np.uint8))
    manifest = tmp_path / 'manifest.json'
    dump(manifest, dict(resolution=[20, 10], frames=[dict(frame_id=i, timestamp_ns=i*1000000, path=str(image)) for i in range(9)]))
    monkeypatch.setattr(selection, 'blur_mask', lambda _: (np.array([False]*8+[True]), None))
    out = tmp_path / 'prepare'
    out.mkdir()
    prepare(dict(manifest=str(manifest), split=dict(exclude_blur=exclude_blur)), out)
    split = json.loads((out / 'selection.json').read_text())
    assert split['train_indices'] == expected
    assert split['test_indices'] == [7]


def test_video_prepare_manifest(tmp_path, monkeypatch):
    import cv2
    import recon.video.extract_frames as extract
    from recon.video.stages import prepare_video
    calls = []
    def fake_main(args):
        calls.append(args)
        (tmp_path / 'frames').mkdir()
        for frame in [3, 17]:
            cv2.imwrite(str(tmp_path / 'frames' / f'{frame:05d}.jpg'), np.zeros((10, 20, 3), np.uint8))
        dump(tmp_path / 'extract.json', dict(frames=[dict(frame=3, time_s=.125), dict(frame=17, time_s=.708)]))
    monkeypatch.setattr(extract, 'main', fake_main)
    manifest = prepare_video(dict(video='clip.MOV', extract=dict(target_frames=400, sfm_multiple=3)), tmp_path)
    assert calls[0][-4:] == ['--target-frames', '400', '--sfm-multiple', '3']
    assert manifest['resolution'] == [20, 10]
    assert [r['frame_id'] for r in manifest['frames']] == [3, 17]
    assert [r['timestamp_ns'] for r in manifest['frames']] == [125000000, 708000000]


def test_glomap_stage_outputs(tmp_path, monkeypatch):
    from recon.video.stages import glomap
    image = tmp_path / 'input.png'
    image.touch()
    out = tmp_path / 'pose'
    out.mkdir()
    calls = []
    def main(args):
        calls.append(args)
        np.savez(out / 'cameras.npz', frame_ids=[3, 7], registered_indices=[0, 1],
                 c2w=np.tile(np.eye(4), (2, 1, 1)), K=np.tile(np.eye(3)*2, (2, 1, 1)),
                 points=np.ones((4, 3)), colors=np.zeros((4, 3)))
        dump(out / 'sfm.json', dict(input_frames=2))
    monkeypatch.setitem(sys.modules, 'recon.video.sfm', SimpleNamespace(main=main))
    data, report = glomap(dict(frame_ids=np.array([3, 7]), timestamp_ns=np.array([10, 20]), blur=np.array([False, False])),
                          dict(resolution=[20, 10], frames=[dict(frame_id=i, path=str(image)) for i in [3, 7]]),
                          out, dict(threads=8, sfm_mode='incremental'))
    assert calls[0][-2:] == ['--sfm-mode', 'global']
    assert report['units'] == 'arbitrary'
    assert data['K'][0, 0, 0] == 2
    assert json.loads((out / 'manifest.json').read_text())['frames'][0]['path'].endswith('undistorted/00003.png')


def test_phone_only_train_eval_input_assembly(tmp_path, monkeypatch):
    import cv2
    import torch
    import recon.pipeline as pipeline
    import recon.gaussian.train as trainer
    import recon.video.evaluation as evaluation
    from recon.video.stages import load_phone_data
    n = 16
    root = tmp_path / 'run'
    for stage in ('prepare', 'pose', 'depth', 'fuse'):
        (root / stage).mkdir(parents=True)
        dump(root / stage / 'complete.json', dict(outputs={}))
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.linspace(-1, 1, n)
    K = np.tile(np.array([[20., 0, 10], [0, 20, 5], [0, 0, 1]]), (n, 1, 1))
    ids = np.arange(n)*10
    path = tmp_path / 'frame.png'
    cv2.imwrite(str(path), np.full((10, 20, 3), 127, np.uint8))
    manifest = dict(resolution=[20, 10], frames=[dict(frame_id=int(f), timestamp_ns=i*1000000, path=str(path)) for i, f in enumerate(ids)])
    dump(root / 'prepare/manifest.json', manifest)
    split = dict(train_indices=[i for i in range(n) if i%8 != 7], test_indices=[7, 15])
    dump(root / 'prepare/selection.json', split)
    np.savez(root / 'pose/cameras.npz', frame_ids=ids, timestamp_ns=np.arange(n), c2w=poses, K=K,
             points=np.zeros((2, 3)), colors=np.zeros((2, 3)))
    dump(root / 'pose/report.json', dict(units='meters'))
    dump(root / 'depth/source.json', dict(folder=str(tmp_path / 'windows')))
    cloud = np.array([[0., 0, 2], [1., 0, 2], [0., 1, 2], [1., 1, 2]], np.float32)
    np.savez(root / 'fuse/initial.npz', points=cloud, colors=np.full_like(cloud, .5))
    cfg = dict(protocol='phone', manifest=str(tmp_path / 'manifest.json'), output=str(tmp_path),
               pose=dict(provider='external'), train=dict(max_width=16, image_cache_frames=2),
               eval=dict(aligned_width=960, alignment_steps=200))
    monkeypatch.setattr(pipeline, 'fingerprint', lambda _: 'run')
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    expected, images = load_phone_data(cfg, root)
    np.testing.assert_array_equal(expected['K'], (K * np.array([.8, .8, 1])[None, :, None]).astype(np.float32))
    assert len(expected['points']) == len(cloud)
    assert expected['arbitrary_scale'] is False
    assert images.get.cache_info().maxsize == 2
    seen = []
    def fit(data, out, folder, image, settings):
        seen.append('train')
        np.testing.assert_array_equal(data['points'], expected['points'])
        np.testing.assert_array_equal(data['c2w'].numpy(), expected['c2w'])
        assert data['train_indices'].tolist() == split['train_indices']
        assert folder == root / 'fuse/cache'
        assert image(int(ids[0]), 16, 8).shape == (8, 16, 3)
        assert settings['depth_shape'] == [288, 512]
        torch.save(dict(splats={'means': torch.zeros(1, 3)}, stats={}), out / 'final.pt')
    def score(splats, data, image_cache, stats, out, args):
        seen.append('eval')
        assert args.aligned_width == 960 and args.aligned_steps == 200 and args.aligned_colour
        assert args.floater_ratio == .7 and args.depth_conf_percentile == 50
        assert args.depth_windows == str(tmp_path / 'windows')
        assert data['test_indices'].tolist() == [7, 15]
        np.testing.assert_array_equal(data['points'], expected['points'])
        assert image_cache.width == 16
        dump(out / 'metrics.json', dict(mock=True))
    monkeypatch.setattr(trainer, 'train', fit)
    monkeypatch.setattr(evaluation, 'evaluate', score)
    pipeline.run(cfg, only='train')
    pipeline.run(cfg, only='eval')
    assert seen == ['train', 'eval']


@pytest.mark.parametrize('provider', ['external', 'glomap', 'colmap'])
def test_provider_metric_then_refine(tmp_path, monkeypatch, provider):
    import recon.pipeline as pipeline
    import recon.poses.pipeline as poses
    import recon.poses.metric as metric
    import recon.video.stages as stages
    root = tmp_path
    (root / 'prepare').mkdir()
    out = root / 'pose'
    out.mkdir()
    cameras = dict(frame_ids=np.array([0, 1]), timestamp_ns=np.array([0, 1]), blur=np.array([False, False]),
                   c2w=np.tile(np.eye(4), (2, 1, 1)), K=np.tile(np.eye(3), (2, 1, 1)),
                   points=np.ones((4, 3)), colors=np.zeros((4, 3)))
    cameras['c2w'][:, 0, 3] = [1, 2]
    np.savez(root / 'prepare/inputs.npz', **cameras)
    dump(root / 'prepare/manifest.json', dict(frames=[]))
    monkeypatch.setattr(metric, 'camera_height_scale', lambda *args: (2., dict(measured=False)))
    calls = []
    def estimate(data, manifest, destination, settings, actual_provider):
        calls.append(actual_provider)
        if destination == out:
            np.savez(destination / 'cameras.npz', **cameras)
            dump(destination / 'report.json', dict(units='arbitrary'))
        else:
            np.testing.assert_array_equal(data['c2w'][:, 0, 3], [2, 4])
            np.testing.assert_array_equal(data['points'], np.ones((4, 3))*2)
            assert settings['units'] == 'meters'
            np.savez(destination / 'cameras.npz', **data)
            dump(destination / 'report.json', dict(refine='ba'))
    monkeypatch.setattr(poses, 'estimate', estimate)
    monkeypatch.setattr(stages, 'glomap', lambda *args: (cameras, dict(units='arbitrary')))
    pipeline.pose(dict(pose=dict(provider=provider, refine='ba', units='arbitrary', metric=dict(camera_height_m=1.4))), out, root)
    assert calls[-1] == 'external'
    report = json.loads((out / 'report.json').read_text())
    assert report['provider'] == provider and report['refine'] == 'ba' and report['units'] == 'meters'


def test_window_prior_uses_owner_and_phone_confidence(tmp_path):
    from recon.video.evaluation import WindowPrior, OmegaPrior
    (tmp_path / 'K000').mkdir()
    depth = np.arange(8, dtype=np.float32).reshape(1, 2, 4)+1
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


@pytest.mark.parametrize('pose', [dict(mode='cached'), dict(provider='bad'), dict(provider='glomap', units='meters')])
def test_pose_configuration_boundary(pose):
    from recon.config import validate
    with pytest.raises(ValueError, match='Invalid reconstruction configuration'):
        validate(dict(manifest='frames.json', output='out', pose=pose))
