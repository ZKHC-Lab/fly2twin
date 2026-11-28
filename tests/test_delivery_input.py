import json

import numpy as np

from recon.io.delivery import load


def package(tmp_path, ba):
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    lio = np.tile(np.eye(4), (2, 1, 1))
    lio[1, 0, 3] = 1.0
    np.savez(
        tmp_path / 'poses.npz',
        frame_ids=np.arange(2),
        timestamp_ns=np.array([10, 20]),
        c2w=lio,
        K=np.stack([K, K]),
        blur=np.ones(2),
    )
    meta = dict(
        schema_version=1,
        units='meters',
        resolution=[1280, 720],
        hashes={},
        frames=[dict(frame_id=i, path='keyframes/%06d.jpg' % i) for i in range(2)],
    )
    if ba:
        (tmp_path / 'onboard_ba').mkdir()
        refined = lio.copy()
        refined[1, 0, 3] = 1.02
        np.savez(tmp_path / 'onboard_ba/cameras.npz', frame_ids=np.arange(2), c2w=refined)
        meta['onboard_ba'] = dict(poses='onboard_ba/cameras.npz', completed=True)
    (tmp_path / 'manifest.json').write_text(json.dumps(meta))
    return tmp_path


def test_landing_ba_poses_are_preferred(tmp_path):
    manifest, source = load(package(tmp_path, ba=True))
    assert source == 'landing global BA'
    assert manifest['frames'][1]['c2w'][0][3] == 1.02
    assert manifest['K'][0][0] == 900.0 and manifest['resolution'] == [1280, 720]
    assert manifest['frames'][0]['path'].endswith('keyframes/000000.jpg')


def test_lio_poses_without_ba(tmp_path):
    manifest, source = load(package(tmp_path, ba=False))
    assert source == 'onboard LIO frame_pose'
    assert manifest['frames'][1]['c2w'][0][3] == 1.0
