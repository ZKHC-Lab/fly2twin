import numpy as np
import pytest
from recon.gaussian.schedule import schedule
from recon.geometry.transforms import robust_normalization


def test_v1_schedule():
    p = schedule(670, exact_steps=True, scale_densification=False, sh_interval=1000)
    assert p == dict(
        steps=40200,
        sh_interval=1000,
        strategy=dict(refine_start_iter=500, refine_stop_iter=15000, reset_every=3000, refine_every=100),
    )


def test_phone_schedule():
    p = schedule(1050, exact_steps=True, schedule_rounding='round')
    assert p == dict(
        steps=63000,
        sh_interval=2100,
        strategy=dict(refine_start_iter=1050, refine_stop_iter=31500, reset_every=6300, refine_every=210),
    )


def test_rgb_only_does_not_open_depth(monkeypatch, tmp_path):
    import recon.gaussian.train as module

    class ReachedInitialization(Exception):
        pass

    def initialized(*args):
        raise ReachedInitialization

    monkeypatch.setattr(module, 'initialize', initialized)
    monkeypatch.setattr(module.torch.cuda, 'manual_seed_all', lambda *a: None)

    def forbidden(*a, **kw):
        raise AssertionError('RGB-only must not open depth')

    monkeypatch.setattr(module.np, 'load', forbidden)
    data = dict(train_indices=[0], width=8, height=8, points=None, colors=None, scene_scale=1)
    with pytest.raises(ReachedInitialization):
        module.train(data, tmp_path, None, None, dict(depth_lambda=0, steps=1))


def test_reset_at_zero(monkeypatch):
    import recon.gaussian.strategy as module

    calls = []
    monkeypatch.setattr(module, 'reset_opa', lambda **kw: calls.append(kw))
    strategy = module.BaselineStrategy(reset_at_zero=True)
    monkeypatch.setattr(strategy, '_update_state', lambda *a, **kw: None)
    strategy.step_post_backward({}, {}, {}, 0, {})
    assert len(calls) == 1
    strategy.reset_at_zero = False
    strategy.step_post_backward({}, {}, {}, 0, {})
    assert len(calls) == 1


def test_phone_normalization_guard():
    poses = np.repeat(np.eye(4)[None], 32, axis=0)
    poses[:, 0, 3] = np.linspace(-2, 2, 32)
    poses[-1, 0, 3] = 1000
    train = np.arange(32)
    data = dict(
        c2w=poses, train_indices=train, test_indices=np.array([], int), registered_indices=train, frame_ids=train
    )
    result = robust_normalization(data)
    assert 31 in result['dropped_frames']
    assert result['p95_radius'] < 3


def test_image_cache_capacity_does_not_change_pixels(tmp_path):
    import cv2
    from recon.io.common import image_loader

    rng = np.random.default_rng(42)
    paths = {}
    for frame in range(4):
        paths[str(frame)] = tmp_path / f'{frame:05d}.png'
        cv2.imwrite(str(paths[str(frame)]), rng.integers(0, 256, (18, 32, 3), dtype=np.uint8))
    small = image_loader(paths, 1)
    full = image_loader(paths, 4)
    for _ in range(2):
        for frame in range(4):
            np.testing.assert_array_equal(small(frame, 16, 9), full(frame, 16, 9))
    assert small.cache_info().hits == 0 and full.cache_info().hits == 4
