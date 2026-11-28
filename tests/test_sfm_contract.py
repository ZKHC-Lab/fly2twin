import json
from types import SimpleNamespace
import numpy as np
import pytest


def test_uncalibrated_manifest_can_prepare(tmp_path):
    cv2 = pytest.importorskip('cv2')
    from recon.pipeline import prepare

    path = tmp_path / 'frame.png'
    assert cv2.imwrite(str(path), np.zeros((48, 64, 3), np.uint8))
    rows = [dict(frame_id=i, timestamp_ns=i * 200_000_000, path=str(path)) for i in range(4)]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(resolution=[64, 48], frames=rows)))
    out = tmp_path / 'prepared'
    out.mkdir()
    prepare(dict(manifest=str(manifest)), out)
    with np.load(out / 'inputs.npz', allow_pickle=False) as z:
        assert 'c2w' not in z.files and len(z['frame_ids']) == 4


def fake_colmap(monkeypatch, pc, counts):
    """Mock pycolmap up to mapping; returns the recorded options of every call."""
    calls = {}

    def record(name, result=None):
        def call(*args, **kwargs):
            calls[name] = kwargs
            return result

        monkeypatch.setattr(pc, name, call)

    for name in ('extract_features', 'match_sequential', 'match_image_pairs'):
        record(name)
    models = {
        k: SimpleNamespace(
            num_reg_images=lambda n=n: n,
            reg_image_ids=lambda n=n: range(n),
            images={i: SimpleNamespace(name=f'{i:08d}.png') for i in range(n)},
        )
        for k, n in enumerate(counts)
    }
    record('incremental_mapping', models)
    return calls


def frames(tmp_path, n=4):
    path = tmp_path / 'frame.png'
    path.write_bytes(b'features are mocked in this contract test')
    return dict(
        resolution=[64, 48], frames=[dict(frame_id=i, timestamp_ns=i * 200_000_000, path=str(path)) for i in range(n)]
    )


def test_partial_sfm_reports_registration_and_submodels(tmp_path, monkeypatch):
    pc = pytest.importorskip('pycolmap')
    from recon.sfm import reconstruct

    fake_colmap(monkeypatch, pc, [3, 2])
    out = tmp_path / 'pose'
    out.mkdir()
    with pytest.raises(RuntimeError, match='Partial reconstruction'):
        reconstruct({'frame_ids': np.arange(4)}, frames(tmp_path), out, {})
    report = json.loads((out / 'report.json').read_text())
    assert report['registration_rate'] == 0.75 and report['disconnected_submodels'] == 1
    assert (
        report['models'] == {'0': 3, '1': 2} and report['units'] == 'arbitrary' and report['unregistered_frames'] == [3]
    )
    assert 'pose.provider: vggt' in report['recommendation'] and report['provider'] == 'colmap'


@pytest.mark.parametrize(
    'settings,model,threads,loop',
    [({}, 'OPENCV', 4, False), (dict(camera_model='SIMPLE_RADIAL', threads=2, loop_keyframes=2), 'SIMPLE_RADIAL', 2, True)],
)
def test_colmap_defaults_and_sfm_overrides(tmp_path, monkeypatch, settings, model, threads, loop):
    mapping = 'incremental_mapping'
    pc = pytest.importorskip('pycolmap')
    from recon.sfm import reconstruct

    calls = fake_colmap(monkeypatch, pc, [3])
    out = tmp_path / 'pose'
    out.mkdir()
    with pytest.raises(RuntimeError, match='Partial reconstruction'):
        reconstruct({'frame_ids': np.arange(80)}, frames(tmp_path, 80), out, settings)
    assert set(calls) >= {'extract_features', 'match_sequential', mapping}
    assert ('match_image_pairs' in calls) == loop
    assert calls['extract_features']['reader_options'].camera_model == model
    assert calls['extract_features']['extraction_options'].num_threads == threads
    assert calls[mapping]['options'].num_threads == threads and calls[mapping]['options'].random_seed == 42


def test_registered_only_keeps_matching_rows():
    from recon.sfm import registered_only

    names = [f'{i:08d}.jpg' for i in range(5)]
    data = dict(frame_ids=np.array([10, 20, 30, 40, 50]), blur=np.arange(5) > 2, K=np.zeros((5, 3, 3)), other=np.zeros(3))
    manifest = dict(resolution=[4, 3], frames=[dict(frame_id=int(f), path=f'{f}.jpg') for f in data['frame_ids']])
    kept, rows, left = registered_only(data, manifest, names, {names[0], names[1], names[3], names[4]})
    assert kept['frame_ids'].tolist() == [10, 20, 40, 50] and kept['blur'].tolist() == [False, False, True, True]
    assert kept['K'].shape == (4, 3, 3) and kept['other'].shape == (3,)
    assert [r['frame_id'] for r in rows['frames']] == [10, 20, 40, 50] and rows['resolution'] == [4, 3]
    assert left == [names[0], names[1], names[3], names[4]] and len(manifest['frames']) == 5


def test_split_without_reindexes_the_kept_frames():
    from recon.pipeline import split_without

    prepared = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    selection = dict(
        train_indices=[0, 1, 2, 3, 4, 5, 6],
        test_indices=[7],
        train_frame_ids=[10, 20, 30, 40, 50, 60, 70],
        test_frame_ids=[80],
        interpolation_frame_ids=[80],
        holdout_frame_ids=[],
    )
    kept = np.array([10, 20, 40, 50, 60, 70, 80])
    result, subsets = split_without(selection, dict(interpolation=[80], holdout=[]), prepared, kept)
    assert result['train_indices'] == [0, 1, 2, 3, 4, 5] and result['test_indices'] == [6]
    assert kept[result['train_indices']].tolist() == [10, 20, 40, 50, 60, 70] == result['train_frame_ids']
    assert kept[result['test_indices']].tolist() == [80] and subsets == dict(interpolation=[80], holdout=[])
    result, subsets = split_without(selection, dict(test=[80]), prepared, prepared[:7])
    assert result['test_indices'] == [] and result['test_frame_ids'] == [] and subsets == dict(test=[])


def test_registration_below_the_rate_is_rejected_and_the_rate_is_configurable(tmp_path, monkeypatch):
    pc = pytest.importorskip('pycolmap')
    from recon.config import validate
    from recon.sfm import reconstruct

    fake_colmap(monkeypatch, pc, [98])
    out = tmp_path / 'pose'
    out.mkdir()
    with pytest.raises(RuntimeError, match='98 of 100 frames registered, below sfm.min_registration_rate 0.99'):
        reconstruct({'frame_ids': np.arange(100)}, frames(tmp_path, 100), out, {})
    report = json.loads((out / 'report.json').read_text())
    assert report['unregistered_frames'] == [98, 99] and report['dropped_frames'] == [] and report['min_registration_rate'] == 0.99
    base = dict(manifest='m.json', output='out')
    validate(dict(base, sfm=dict(min_registration_rate=0.9)))
    for bad in (0, 1.5, True, 'high'):
        with pytest.raises(ValueError, match='min_registration_rate'):
            validate(dict(base, sfm=dict(min_registration_rate=bad)))
