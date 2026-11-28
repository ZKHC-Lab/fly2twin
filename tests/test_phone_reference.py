"""Phone path on a synthetic video, through fusion: recorded content hashes.

The poses come from the `vggt` provider, with the synthetic windows standing in for the model's
output: chain, feature matching on the rendered frames, bundle adjustment with self-calibration,
undistortion, camera-height scale. They are checked against the cameras the video was rendered
from. The hashes then pin every array the phone path produces up to fusion, plus what the train
stage loads. pycolmap, x264 and OpenCV make the digests environment-specific: that part runs only
where they were recorded.
"""

import hashlib
import json
import subprocess
import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
pycolmap = pytest.importorskip('pycolmap')
pytest.importorskip('torch')

RECORDED = dict(pycolmap='4.2.0', opencv='4.11.0', numpy='1.26.4', ffmpeg='6.1.1-3ubuntu5')
EXPECTED = {
    'fuse/cache/depth_m.npy': 'd8ffd36a853e30bea19fbc05783e08d6dd8cff2710dcce32f018125cdd421909',
    'fuse/cache/frames.npz:frame_ids': '34db26d52d64a39a0b665d1b83dbeb4bac579e2d8cbc15974a59f4134beb3ac3',
    'fuse/cache/good.npy': '8a95f646ac3aa738b21470450f8d31ee07bc4bec19d54adc8bcd219f499c6abd',
    'fuse/cache/good_init.npy': 'b609d871f73db0cb311ce53450572f1dbbdb066b03c3606356d0d867cf248a15',
    'fuse/gaussian_init.npz:colors': 'a3ea7de9fce90a3d01cb9c249a9d2b46c18e596f67e70af4aa67266ef896d162',
    'fuse/gaussian_init.npz:points': '9ab91020c1828a2afe8c833bac212a949ad44ebda46e5be7571df62eb2a72981',
    'fuse/initial.npz:colors': '9741363aca6fed188c57e25b78341ea8697534e9f9c7950e9689ad646b46cddd',
    'fuse/initial.npz:points': '3f654eaed20544ec16616d0fdaba3876cc5fc0cf5e62825ed1c238867381e363',
    'fuse/report.json': '1c91dd6b053d9486ce23d21691cd9b0ba86720500357f9117b2533057f7f7b80',
    'fuse/scales.npz:frame_ids': '6bd48b2e256252fc36cff065873fd0d722aaf36164bce455e48f6372a1b3afca',
    'fuse/scales.npz:scales': 'f50712eb8fa59d6186a27fe989916566b7a4b7eff7271a0f99c9c8d1576a5da3',
    'fuse/scales.npz:window_scales': '1366bd2a8f49b82499734a0e5e6742114cf32be34f6f7b3b1eb254e1c7756eaf',
    'load_data[drone]': '5e5e3d679ba28bc935ea2b728441ce4d153658865102c7d0160b0904e8483b4b',
    'load_data[start]': '78a3cfad90fb141b6aa7598688ebd4bd774ed0bece598c94e15a188cf32914fd',
    'load_data[phone]': '53f98f319c6fedcbce666ced4516638c5c715cb5690c84be43b17eb395f60ee8',
    'pose/cameras.npz:K': '46bebd608ae07bbce0c927d7ebb1f9f3f31be0e5a9e160baabf8bc9f69afe3da',
    'pose/cameras.npz:blur': '26b4748cc6e737e2c889716f0fe9a76c0d426effeb36e7e152cc463617dc656a',
    'pose/cameras.npz:c2w': '741736ee8dca53ae19ad9e3283d3e67009c86b15fa4f5b7247f28535e5f3e240',
    'pose/cameras.npz:colors': 'e48d6da555927deb45173ba08986b8707d5b5c8fabbe6b302bd90dd1579daee3',
    'pose/cameras.npz:frame_ids': '34db26d52d64a39a0b665d1b83dbeb4bac579e2d8cbc15974a59f4134beb3ac3',
    'pose/cameras.npz:points': '68367f6d0cdbdbd54bc25a4f6eccc768c6a3e8b1dba5960823a87f06755158f9',
    'pose/cameras.npz:timestamp_ns': '3a57b5b7eeecd8507a2da7712b1bcbd509c6c9e50db64617e5f67ecea57c4504',
    'pose/report.json:scale': '564bb87d6739e118b617b8fc861db07144cfe8e6d2d1d25235c9bc79a9310b73',
    'pose/undistorted': 'f6a47d77c7dd19d20f0d0d3d5d4cadee70e5ac6ce946f415a9e2a6b5a71c3b7d',
    'prepare/frames': '1975b136e5b2c34b44edc5759839c36c5205ece2272e6ef1de90ff145cd92262',
    'prepare/inputs.npz:K': '7ab6bbecb70919a97345c95bd5794464aaf924884e870d9089fc6d7d4fee79bb',
    'prepare/inputs.npz:blur': '26b4748cc6e737e2c889716f0fe9a76c0d426effeb36e7e152cc463617dc656a',
    'prepare/inputs.npz:frame_ids': '34db26d52d64a39a0b665d1b83dbeb4bac579e2d8cbc15974a59f4134beb3ac3',
    'prepare/inputs.npz:timestamp_ns': '3a57b5b7eeecd8507a2da7712b1bcbd509c6c9e50db64617e5f67ecea57c4504',
    'prepare/selection.json': '436f770ac0852ef1b6645da6c4a7aadf54b7f136a4fac92e2567f8c167cfc9b3',
}


def environment():
    try:
        line = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.split('\n')[0]
    except OSError:
        line = ''
    return dict(
        pycolmap=pycolmap.__version__,
        opencv=cv2.__version__,
        numpy=np.__version__,
        ffmpeg=line.split(' ')[2] if line.startswith('ffmpeg version') else None,
    )


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        h.update(str((array.dtype.str, array.shape)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def npz(root, name, keys=None):
    with np.load(root / name, allow_pickle=False) as z:
        return {f'{name}:{key}': digest(z[key]) for key in keys or z.files}


def images(folder, label):
    paths = sorted(folder.iterdir(), key=lambda p: int(p.stem))
    return {label: digest(np.array([int(p.stem) for p in paths]), *(cv2.imread(str(p)) for p in paths))}


def digests(root):
    from recon.pipeline import load_data

    result = {}
    result.update(images(root / 'prepare/frames_sfm', 'prepare/frames'))
    result.update(npz(root, 'prepare/inputs.npz', ('frame_ids', 'timestamp_ns', 'K', 'blur')))
    result['prepare/selection.json'] = digest(np.frombuffer((root / 'prepare/selection.json').read_bytes(), np.uint8))
    result.update(images(root / 'pose/undistorted', 'pose/undistorted'))
    result.update(npz(root, 'pose/cameras.npz', ('frame_ids', 'timestamp_ns', 'c2w', 'K', 'blur', 'points', 'colors')))
    scale = json.loads((root / 'pose/report.json').read_text())['scale_assumption']['sfm_units_to_assumed_meters']
    result['pose/report.json:scale'] = digest(np.float64(scale))
    for name in ('cache/depth_m.npy', 'cache/good.npy', 'cache/good_init.npy'):
        result['fuse/' + name] = digest(np.load(root / 'fuse' / name))
    result.update(npz(root, 'fuse/cache/frames.npz'))
    result.update(npz(root, 'fuse/initial.npz'))
    result.update(npz(root, 'fuse/gaussian_init.npz'))
    result.update(npz(root, 'fuse/scales.npz'))
    report = json.loads((root / 'fuse/report.json').read_text())
    result['fuse/report.json'] = digest(np.frombuffer(json.dumps(report, sort_keys=True).encode(), np.uint8))
    phone, image = load_data(dict(protocol='phone', train=dict(max_width=320, image_cache_frames=8)), root)
    first, last = (int(phone['frame_ids'][i]) for i in (0, -1))
    result['load_data[phone]'] = digest(
        phone['c2w'],
        phone['K'],
        phone['points'],
        phone['colors'],
        np.asarray(phone['center']),
        np.float64(phone['scale']),
        np.float64(phone['scene_scale']),
        phone['train_indices'],
        phone['test_indices'],
        np.array([phone['width'], phone['height']]),
        image(first, phone['width'], phone['height']),
        image(last, phone['width'], phone['height']),
    )
    # The delivered cloud as the start, the path the digest was recorded on; the default start is pinned separately.
    drone, _ = load_data(dict(train=dict(init_cloud='delivered')), root, init=True)
    start, _ = load_data(dict(), root, init=True)
    result['load_data[start]'] = digest(start['points'], start['colors'])
    result['load_data[drone]'] = digest(
        drone['c2w'],
        drone['K'],
        drone['points'],
        drone['colors'],
        np.asarray(drone['center']),
        np.float64(drone['meters_per_unit']),
        np.float64(drone['scene_scale']),
    )
    return result


def test_phone_video_through_fusion(tmp_path, monkeypatch):
    import phone_scene
    import recon.depth.backends as backends
    from recon.pipeline import run
    from recon.poses.vggt import mean_rotation, similarity

    video = phone_scene.write_video(tmp_path)
    cache = phone_scene.write_depth_cache(tmp_path / 'depth')
    monkeypatch.setattr(backends, 'infer_windows', lambda manifest, settings, folder: phone_scene.write_depth_cache(folder))
    root = run(phone_scene.config(tmp_path, video, cache), through='fuse')
    report = json.loads((root / 'pose/report.json').read_text())
    assert report['registered'] == phone_scene.FRAMES and report['units'] == 'meters' and report['provider'] == 'vggt'
    assert report['mean_reprojection_error'] < 1.0 and report['frames_without_observations'] == []
    truth, K = phone_scene.cameras()
    with np.load(root / 'pose/cameras.npz') as z:
        found, found_K = z['c2w'], z['K'][0]
    scale, R, t = similarity(found[:, :3, 3], truth[:, :3, 3])
    error = np.linalg.norm(scale * found[:, :3, 3] @ R.T + t - truth[:, :3, 3], axis=1)
    # Orientations up to one common rotation: the path is nearly a straight line, so the fit of the centres leaves
    # the rotation about that line open. The synthetic windows are exact, so this is what the BA adds: 0.25 deg at
    # most when recorded, about one pixel of these 320-pixel frames.
    offset = mean_rotation(np.einsum('nij,nkj->nik', truth[:, :3, :3], found[:, :3, :3]))
    turn = np.degrees(np.arccos(np.clip((np.einsum('ij,njk,nik->n', offset, found[:, :3, :3], truth[:, :3, :3]) - 1) / 2, -1, 1)))
    print('pose check: scale %.4f, position max %.4f m, rotation max %.3f deg, focal %.2f (true %.0f), k1 %.5f' % (
        scale, error.max(), turn.max(), report['original_K'][0][0], K[0, 0], report['distortion'][0]))
    assert error.max() < 0.02 and turn.max() < 0.4  # the path is 1.7 m long
    assert abs(scale - 1) < 0.1  # metres from the assumed 1.4 m camera height (the floor fit on sparse points is rough)
    assert abs(report['original_K'][0][0] / K[0, 0] - 1) < 0.02 and abs(report['distortion'][0]) < 0.01
    assert abs(found_K[0, 0] / K[0, 0] - 1) < 0.02
    found = environment()
    if found != RECORDED:
        pytest.skip(f'digests recorded with {RECORDED}, found {found}')
    assert digests(root) == EXPECTED
