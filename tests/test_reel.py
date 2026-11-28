# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for reel geometry, timeline, cards and compositing."""

from types import SimpleNamespace

import numpy as np
import pytest

from recon.reel import path
from recon.reel.assemble import timeline, clip_frame
from recon.reel.grade import grade


def test_path_continuity_and_limits():
    keys = [dict(p=[x, 0, 1], look=[x, 3, 1]) for x in (0, 0.7, 1.3, 2)]
    t, p, look = path.sample(path.Shot('truck', keys, speed=0.5, ease=1, fps=120))
    motion = path.check(t, p, look, 120)
    assert path.limits_ok(motion)
    assert motion['speed_max'] <= 0.501
    assert np.linalg.norm(p[1] - p[0]) < 1e-6
    assert np.linalg.norm(p[-1] - p[-2]) < 1e-6
    vel = np.gradient(p, 1 / 120, axis=0)
    acc = np.gradient(vel, 1 / 120, axis=0)
    assert np.max(np.linalg.norm(np.diff(acc, axis=0), axis=1)) < 0.04
    assert not path.limits_ok(dict(motion, yaw_rate_max_deg_s=21))
    assert not path.limits_ok(dict(motion, speed_max=0.9))
    matrices = path.c2w(p, look)
    np.testing.assert_allclose(np.linalg.det(matrices[:, :3, :3]), 1, atol=1e-7)


def test_zero_ease_and_invalid_paths():
    keys = [dict(p=[0, 0, 0], look=[0, 1, 0]), dict(p=[1, 0, 0], look=[1, 1, 0])]
    assert np.isfinite(path.sample(path.Shot('x', keys, ease=0))[1]).all()
    with pytest.raises(ValueError):
        path.sample(path.Shot('bad', keys, speed=0))
    with pytest.raises(ValueError):
        path.sample(path.Shot('bad', [keys[0]]))
    with pytest.raises(ValueError):
        path.c2w(np.array([[0.0, 0, 0]]), np.array([[0.0, 0, 1]]))


def test_arc_uses_world_up():
    from recon.reel.templates import make

    up = np.array([0.0, -1, 0])
    camera = np.eye(4, dtype=np.float32)[None]
    scene = SimpleNamespace(train_c2w=camera, world_up=up)
    s = make(scene, dict(type='arc', hero=0, target_dz=-0.3))
    p = np.array([k['p'] for k in s['keys']])
    np.testing.assert_allclose(p @ up, 0, atol=1e-7)
    pose = path.c2w(p, np.array([k['look'] for k in s['keys']]), up)
    np.testing.assert_allclose(pose[:, :3, 0] @ up, 0, atol=1e-7)


@pytest.mark.parametrize('views,expected', [(12, 'A'), (6, 'C')])
def test_synthetic_plane_coverage(monkeypatch, views, expected):
    import torch
    from recon.reel import coverage
    from recon.reel.render import tier

    # Identical capture/query cameras facing a plane at z=2: every pixel is observed.
    K = np.array([[100, 0, 79.5], [0, 100, 44.5], [0, 0, 1]], np.float32)
    sc = SimpleNamespace(
        width=160,
        height=90,
        train_K=np.repeat(K[None], views, 0),
        train_c2w=np.repeat(np.eye(4, dtype=np.float32)[None], views, 0),
        torch_splats=lambda device: {},
    )

    def depth(splats, poses, K, *args, **kwargs):
        return torch.full((len(poses), 90, 160), 2.0), torch.ones((len(poses), 90, 160))

    monkeypatch.setattr(coverage, 'render_depth', depth)
    cov = coverage.Coverage(sc, device='cpu')
    counts, _ = cov.frame(np.eye(4, dtype=np.float32), K, 160, 90)
    assert (counts == views).all()
    assert tier(coverage.stats(counts)) == expected
    # A closer opaque plane occludes all queried points.
    cov.depth[:] = 1.0
    counts, _ = cov.frame(np.eye(4, dtype=np.float32), K, 160, 90)
    assert (counts == 0).all()


def test_coverage_tier_boundaries():
    from recon.reel.coverage import stats
    from recon.reel.render import tier

    assert tier(stats(np.array([10] * 80 + [0] * 2 + [3] * 18))) == 'A'
    assert tier(stats(np.array([10] * 60 + [0] * 5 + [3] * 35))) == 'B'
    assert tier(stats(np.array([10] * 60 + [0] * 6 + [3] * 34))) == 'C'
    assert tier(stats(np.full(100, -1))) == 'C'


def test_slot_timeline_no_boundary_drift():
    board = dict(
        slots=[
            dict(name='a', start_s=0, end_s=2),
            dict(name='b', start_s=2, end_s=5, transition='xfade'),
            dict(name='c', start_s=5, end_s=6),
        ],
        sources=dict(
            a=dict(kind='color', rgb=[0, 0, 0]),
            b=[dict(kind='image', src='a.png', dur_s=1), dict(kind='image', src='b.png', dur_s=2)],
            c=dict(kind='color', rgb=[1, 1, 1]),
        ),
        overlays=dict(b=[dict(png='title.png', t0=0.2, t1=0.8)]),
    )
    edl = timeline(board)
    assert [c['timeline_start_s'] for c in edl['clips']] == [0, 1.75, 3, 5]
    assert [c['dur_s'] for c in edl['clips']] == [2, 1.25, 2, 1]
    assert edl['clips'][1]['overlays'][0]['t0'] == 0.45
    assert edl['frames'] == 180 and edl['seconds'] == 6
    board['slots'][1]['start_s'] = 2.1
    with pytest.raises(ValueError):
        timeline(board)


def test_timeline_preserves_source_frame_arithmetic():
    # Decimal slots previously selected one frame earlier when starts were taken
    # directly from the JSON instead of accumulated with the compositor.
    slots = [
        dict(name=str(i), start_s=a, end_s=b)
        for i, (a, b) in enumerate([(0, 4.3), (4.3, 8.5), (8.5, 10.7), (10.7, 18.8), (18.8, 22.0)])
    ]
    sources = {s['name']: dict(kind='color', rgb=[0, 0, 0]) for s in slots}
    clips = timeline(dict(slots=slots, sources=sources))['clips']
    cursor = 0.0
    for c in clips:
        assert c['timeline_start_s'] == cursor
        cursor += c['dur_s']


def test_video_media_offset_not_timeline_offset():
    b = dict(slots=[dict(name='x', start_s=0, end_s=2)], sources=dict(x=dict(kind='video', src='a.mp4', start_s=45)))
    c = timeline(b)['clips'][0]
    assert c['start_s'] == 45 and c['timeline_start_s'] == 0


@pytest.mark.parametrize('kind', ['headline', 'title', 'label', 'number', 'endcard'])
def test_card_size(tmp_path, monkeypatch, kind):
    from PIL import Image, ImageFont
    from recon.reel import card, cards

    # Font discovery is tested independently; geometry must not depend on installed OS fonts.
    monkeypatch.setattr(cards, 'font', lambda *args: ImageFont.load_default())
    options = dict(text='Scene')
    if kind == 'label':
        options = dict(items=[['Scene', 100, 100, 'la']])
    if kind == 'endcard':
        options = dict(name='Scene', tagline='Mapping', url='example.org', footer='Org')
    out = tmp_path / (kind + '.png')
    card(kind, out, **options)
    with Image.open(out) as im:
        assert im.size == (1920, 1080) and im.mode == 'RGBA'


def test_missing_font_and_ffmpeg(monkeypatch):
    import shutil
    from recon.reel.cards import font
    from recon.reel.common import require_ffmpeg

    monkeypatch.setattr(shutil, 'which', lambda _: None)
    with pytest.raises(RuntimeError, match='Noto Sans CJK SC'):
        font('Regular', 30)
    with pytest.raises(RuntimeError, match='ffmpeg'):
        require_ffmpeg()


def test_grade_requires_parameters():
    rgb = np.full((3, 4, 3), 128, np.uint8)
    assert grade(rgb, dict(exposure=1.3)).mean() > grade(rgb, dict(exposure=0.98)).mean()
    with pytest.raises(ValueError):
        grade(rgb, 'scene-name')


def test_color_overlays_and_gamma(tmp_path):
    from PIL import Image

    overlay = tmp_path / 'overlay.png'
    Image.new('RGBA', (16, 8), (255, 0, 0, 255)).save(overlay)
    c = dict(kind='color', rgb=[0, 0, 0], dur_s=1, overlays=[dict(png=str(overlay), t0=0, t1=1, fade=0)])
    im = clip_frame(c, 0.5, (16, 8), 30, {})
    assert (im[:, :, 0] == 255).all() and not im[:, :, 1:].any()
