import json

import numpy as np
import pytest

from recon.pipeline import prepare


def write_manifest(tmp_path, n=6):
    import cv2
    rows = []
    for i in range(n):
        path = tmp_path / f'{i}.png'
        cv2.imwrite(str(path), np.full((8, 8, 3), 40 * i, np.uint8))
        c2w = np.eye(4); c2w[0, 3] = i * 0.3
        rows.append(dict(frame_id=i * 10, timestamp_ns=i * 10**8, path=str(path), c2w=c2w.tolist()))
    (tmp_path / 'm.json').write_text(json.dumps(dict(K=[[8, 0, 4], [0, 8, 4], [0, 0, 1]], frames=rows)))
    return tmp_path / 'm.json'


def test_exclude_frames_drops_and_records(tmp_path):
    cfg = dict(manifest=str(write_manifest(tmp_path)), output=str(tmp_path), exclude_frames=[20, 40],
               pose=dict(provider='external', units='meters'), split=dict(mode='holdout', exclude_blur=False))
    out = tmp_path / 'prep'; out.mkdir()
    prepare(cfg, out)
    kept = [r['frame_id'] for r in json.loads((out / 'manifest.json').read_text())['frames']]
    assert kept == [0, 10, 30, 50]
    assert json.loads((out / 'excluded_frames.json').read_text())['frame_ids'] == [20, 40]


def test_exclude_unknown_frame_fails(tmp_path):
    cfg = dict(manifest=str(write_manifest(tmp_path)), output=str(tmp_path), exclude_frames=[25],
               pose=dict(provider='external', units='meters'))
    out = tmp_path / 'prep'; out.mkdir()
    with pytest.raises(ValueError, match='exclude_frames'):
        prepare(cfg, out)
