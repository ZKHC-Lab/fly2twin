"""CPU regression tests for V1.0 window planning and the shared Omega adapter."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pytest

from recon.depth.backends import infer_windows, plan_windows
from recon.depth.cache import load_windows


ROOT = Path(__file__).resolve().parents[1]


def test_t17_windows_exact():
    path = ROOT / 'Data/experiments/office_gs_02_20261019/T17/omega512/kf_T15A8/windows.json'
    if not path.exists():
        pytest.skip('T17 local experiment is not distributed with the repository')
    reference = json.loads(path.read_text())
    assert plan_windows(reference['keyframes']) == reference['windows']


def test_1200_frames():
    windows = plan_windows(range(1200))
    expected = [dict(id='K000', idx=list(range(40)), new=list(range(40)))]
    for start in range(30, 1171, 30):
        expected.append(dict(id=f'K{len(expected):03d}',
                             idx=list(range(start, min(start + 40, 1200))),
                             new=list(range(start + 10, min(start + 40, 1200)))))
    assert windows == expected
    assert windows[-1]['new'] == list(range(1180, 1200))
    assert [i for win in windows for i in win['new']] == list(range(1200))


def test_short_tail_expands_context():
    windows = plan_windows(range(71))
    assert windows[-1] == dict(id='K002', idx=list(range(41, 71)), new=[70])
    assert plan_windows([3, 8]) == [dict(id='K000', idx=[3, 8], new=[3, 8])]


@pytest.mark.parametrize('custom_source', [False, True])
def test_shared_inference_contract(tmp_path, monkeypatch, custom_source):
    calls = []
    model = object()
    result = {}

    def load_model(weights, device, **kwargs):
        calls.append(('model', weights, device, kwargs, sys.path[0]))
        return model

    def preprocess(paths, **kwargs):
        calls.append(('images', paths, kwargs))
        return paths

    def inference(actual_model, images, device):
        assert actual_model is model and device == 'cuda'
        n = len(images)
        result.update(depth=np.ones((n, 2, 4, 1), np.float32),
                      depth_conf=np.ones((n, 2, 4), np.float32),
                      extrinsic=np.tile(np.eye(4, dtype=np.float32)[:3], (n, 1, 1)),
                      intrinsic=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
                      additional_array=np.arange(n))
        return result.copy()

    monkeypatch.setitem(sys.modules, 'reconstruct_omega', SimpleNamespace(
        load_model=load_model, load_and_preprocess_images=preprocess, run_inference=inference))
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    settings = dict(weights='weights.pt', resolution=[4, 2], first_window=4, new_frames=3, overlap=1)
    if custom_source:
        settings['omega_source'] = str(tmp_path / 'omega')
    ids = [10, 20, 30, 40, 50, 60, 70, 80]
    infer_windows(dict(frames=[dict(frame_id=i, path=f'{i}.png') for i in ids]), settings, tmp_path)
    source = settings.get('omega_source', str(ROOT / 'third_party' / 'vggt-omega'))
    assert calls[0] == ('model', 'weights.pt', 'cuda', dict(aggregator_frame_chunk_size=10), source)
    windows = json.loads((tmp_path / 'windows.json').read_text())['windows']
    assert windows == plan_windows(ids, first=4, new=3, overlap=1)
    assert len(calls) == len(windows) + 1
    for win, call in zip(windows, calls[1:]):
        assert call == ('images', [f'{i}.png' for i in win['idx']], dict(mode='max_size', image_resolution=4))
        path = tmp_path / win['id'] / 'prediction.npz'
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == set(result) | {'frame_ids'}
            np.testing.assert_array_equal(z['frame_ids'], win['idx'])
            np.testing.assert_array_equal(z['additional_array'], np.arange(len(win['idx'])))
        with ZipFile(path) as archive:
            assert all(entry.compress_type == ZIP_DEFLATED for entry in archive.infolist())
    assert len(load_windows(tmp_path, np.array(ids), width=4, height=2)) == len(windows)


def test_cli_resolves_omega_source(tmp_path, monkeypatch):
    import recon.cli as cli
    config = tmp_path / 'scene.yaml'
    config.write_text('manifest: frames.json\noutput: out\ndepth:\n  omega_source: omega\n  weights: model.pt\n')
    captured = {}
    monkeypatch.setattr(cli, 'run', lambda cfg, *args: captured.update(cfg))
    monkeypatch.setattr(sys, 'argv', ['recon', 'run', '--config', str(config), '--through', 'depth'])
    cli.main()
    assert captured['depth']['omega_source'] == str(tmp_path / 'omega')
