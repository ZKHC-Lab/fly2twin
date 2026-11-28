"""Small synthetic calibrated prior-BA run against the actual pycolmap API."""

import json
import sqlite3
import numpy as np
import pytest


@pytest.mark.parametrize('rotation_sigma_deg', [None, 1.0])
def test_prior_ba_cpu(tmp_path, rotation_sigma_deg):
    pc = pytest.importorskip('pycolmap')
    cv2 = pytest.importorskip('cv2')
    from recon.poses.ba import build

    n = 4
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * 0.3
    K = np.array([[120.0, 0, 64], [0, 120, 48], [0, 0, 1]])
    xyz = np.array([[x, y, 3 + 0.1 * x] for x in np.linspace(-0.5, 0.5, 6) for y in np.linspace(-0.4, 0.4, 5)])
    images = tmp_path / 'images'
    images.mkdir()
    out = tmp_path / 'out'
    out.mkdir()
    names = [f'{i}.png' for i in range(n)]
    dbpath = out / 'mixed.db'
    with pc.Database.open(dbpath):
        pass
    with sqlite3.connect(dbpath) as db:
        db.execute(
            'INSERT INTO cameras VALUES (?,?,?,?,?,?)',
            (1, 1, 128, 96, np.array([120.0, 120.0, 64.0, 48.0]).tobytes(), 1),
        )
        for i, name in enumerate(names):
            assert cv2.imwrite(str(images / name), np.full((96, 128, 3), 127, np.uint8))
            camera = xyz - poses[i, :3, 3]
            xy = (camera[:, :2] / camera[:, 2:]) * 120 + [64, 48]
            db.execute('INSERT INTO images VALUES (?,?,?)', (i + 1, name, 1))
            db.execute('INSERT INTO keypoints VALUES (?,?,?,?)', (i + 1, len(xy), 2, xy.astype(np.float32).tobytes()))
        for i in range(n):
            for j in range(i + 1, n):
                m = np.repeat(np.arange(len(xyz), dtype=np.uint32)[:, None], 2, axis=1)
                pair = (i + 1) * 2147483647 + j + 1
                F = np.array([[0.0, 0, 0], [0, 0, -1], [0, 1, 0]])
                db.execute('INSERT INTO matches VALUES (?,?,?,?)', (pair, len(m), 2, m.tobytes()))
                db.execute(
                    'INSERT INTO two_view_geometries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (pair, len(m), 2, m.tobytes(), 3, F.tobytes(), None, None, None, None, None, None),
                )
    data = dict(
        frame_ids=np.arange(n),
        timestamp_ns=np.arange(n) * 200_000_000,
        blur=np.zeros(n, bool),
        c2w_L=poses,
        K_L=np.tile(K, (n, 1, 1)),
        image_names=names,
    )
    build(data, dbpath, images, out, rotation_sigma_deg=rotation_sigma_deg)
    with np.load(out / 'cameras.npz', allow_pickle=False) as z:
        assert len(z['registered_indices']) == n
        np.testing.assert_allclose(z['c2w'], poses, atol=1e-4)
    rounds = json.loads((out / 'prior_rounds.json').read_text())
    assert [r['threshold_px'] for r in rounds] == [16.0, 8.0, 4.0]
    assert all(r['prior_ba_solution_usable'] for r in rounds)
    report = json.loads((out / 'reconstruction_prior.json').read_text())
    assert report['rotation_sigma_deg'] == rotation_sigma_deg and ('rotation prior step' in report['mode']) == (rotation_sigma_deg is not None)
    assert all((r['rotation_step_deg_median_p90_max'] is None) == (rotation_sigma_deg is None) for r in rounds)
    options = json.loads((out / 'prior_options_prior.json').read_text())['ba_options']
    assert ('CAUCHY' in options) == (rotation_sigma_deg is not None)


def test_number_by_name(tmp_path):
    pc = pytest.importorskip('pycolmap')
    from recon.poses.prior import number_by_name

    path = tmp_path / 'sift.db'
    with pc.Database.open(path):
        pass
    names = ['c.png', 'a.png', 'b.png']  # completion order of parallel extraction: ids 1, 2, 3
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO rigs VALUES (1,1,0)')
        for i, name in enumerate(names, 1):
            db.execute('INSERT INTO images VALUES (?,?,1)', (i, name))
            db.execute('INSERT INTO keypoints VALUES (?,1,2,?)', (i, np.float32([i, i]).tobytes()))
            db.execute('INSERT INTO descriptors VALUES (?,0,1,1,?)', (i, bytes([i])))
            db.execute('INSERT INTO frames VALUES (?,1)', (i,))
            db.execute('INSERT INTO frame_data VALUES (?,?,1,0)', (i, i))
    number_by_name(path)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT name,image_id FROM images ORDER BY image_id').fetchall() == [
            ('a.png', 1),
            ('b.png', 2),
            ('c.png', 3),
        ]
        old = {name: i for i, name in enumerate(names, 1)}
        for name, new in (('a.png', 1), ('b.png', 2), ('c.png', 3)):
            assert (
                np.frombuffer(
                    db.execute('SELECT data FROM keypoints WHERE image_id=?', (new,)).fetchone()[0], np.float32
                )[0]
                == old[name]
            )
            assert db.execute('SELECT data FROM descriptors WHERE image_id=?', (new,)).fetchone()[0] == bytes(
                [old[name]]
            )
        assert db.execute('SELECT frame_id,data_id FROM frame_data ORDER BY 1').fetchall() == [(1, 1), (2, 2), (3, 3)]


def _rotation_scene(tmp_path):
    """Four cameras around 30 points with exact keypoints, triangulated at the true poses."""
    pc = pytest.importorskip('pycolmap')
    cv2 = pytest.importorskip('cv2')
    from recon.poses.options import known_model, pipeline_options

    n = 4
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * 0.3
    xyz = np.array([[x, y, 3 + 0.3 * x * y] for x in np.linspace(-1, 1, 6) for y in np.linspace(-0.6, 0.6, 5)])
    images = tmp_path / 'images'
    images.mkdir()
    names = [f'{i}.png' for i in range(n)]
    path = tmp_path / 'mixed.db'
    with pc.Database.open(path):
        pass
    with sqlite3.connect(path) as db:
        db.execute(
            'INSERT INTO cameras VALUES (?,?,?,?,?,?)',
            (1, 1, 640, 480, np.array([500.0, 500.0, 320.0, 240.0]).tobytes(), 1),
        )
        for i, name in enumerate(names):
            assert cv2.imwrite(str(images / name), np.full((480, 640, 3), 127, np.uint8))
            camera = xyz - poses[i, :3, 3]
            xy = camera[:, :2] / camera[:, 2:] * 500 + [320, 240]
            db.execute('INSERT INTO images VALUES (?,?,?)', (i + 1, name, 1))
            db.execute('INSERT INTO keypoints VALUES (?,?,?,?)', (i + 1, len(xy), 2, xy.astype(np.float32).tobytes()))
        m = np.repeat(np.arange(len(xyz), dtype=np.uint32)[:, None], 2, axis=1)
        for i in range(n):
            for j in range(i + 1, n):
                pair = (i + 1) * 2147483647 + j + 1
                db.execute('INSERT INTO matches VALUES (?,?,?,?)', (pair, len(m), 2, m.tobytes()))
                db.execute(
                    'INSERT INTO two_view_geometries VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                    (pair, len(m), 2, m.tobytes(), 3, np.eye(3).tobytes(), None, None, None, None, None, None),
                )
    data = dict(c2w_L=poses, image_names=names)
    rec = known_model(data, path)
    out = tmp_path / 'tri'
    rec = pc.triangulate_points(
        rec, path, images, out, clear_points=False, options=pipeline_options(), refine_intrinsics=False
    )
    assert rec.num_points3D() == len(xyz)
    return pc, rec, poses


def _turn(pc, rec, image_id, degrees):
    from scipy.spatial.transform import Rotation

    im = rec.image(image_id)
    pose = im.cam_from_world()
    R0 = pose.rotation.matrix()
    centre = -R0.T @ pose.translation
    R = Rotation.from_rotvec(np.radians(degrees) * np.array([0.0, 0.0, 1.0])).as_matrix() @ R0
    im.frame.set_cam_from_world(im.camera_id, pc.Rigid3d(pc.Rotation3d(R), -R @ centre))


def _angle(a, b):
    return np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1)))


def test_rotation_step_recovers_turned_camera(tmp_path):
    """A camera turned 3 deg about its optical axis comes back; its centre does not move."""
    from recon.poses.ba import refine_rotations

    pc, rec, poses = _rotation_scene(tmp_path)
    truth = {i: rec.image(i).cam_from_world().rotation.matrix() for i in rec.reg_image_ids()}
    centre = rec.image(2).projection_center()
    _turn(pc, rec, 2, 3.0)
    refine_rotations(rec, truth, np.radians(1.0))
    assert _angle(rec.image(2).cam_from_world().rotation.matrix(), truth[2]) < 0.01
    np.testing.assert_allclose(rec.image(2).projection_center(), centre, atol=1e-9)


def test_rotation_step_leaves_correct_input(tmp_path):
    """Exact points and rotations: the step changes nothing, also when the prior itself is 2 deg off."""
    from recon.poses.ba import refine_rotations

    pc, rec, poses = _rotation_scene(tmp_path)
    truth = {i: rec.image(i).cam_from_world().rotation.matrix() for i in rec.reg_image_ids()}
    changes = refine_rotations(rec, truth, np.radians(1.0))
    assert max(changes.values()) < 1e-6
    off = dict(truth)
    from scipy.spatial.transform import Rotation

    off[3] = Rotation.from_rotvec(np.radians([2.0, 0, 0])).as_matrix() @ truth[3]
    refine_rotations(rec, off, np.radians(1.0))
    assert (
        _angle(rec.image(3).cam_from_world().rotation.matrix(), truth[3]) < 0.05
    )  # 30 exact observations outweigh the prior


def test_rotation_prior_is_off_unless_configured():
    from recon.config import validate

    base = dict(manifest='m.json', output='out')
    validate(dict(base, pose=dict(provider='external', refine='ba')))
    validate(dict(base, pose=dict(provider='external', refine='ba', rotation_sigma_deg=1.0)))
    for bad in (0, -1, True, 'one'):
        with pytest.raises(ValueError, match='rotation_sigma_deg'):
            validate(dict(base, pose=dict(provider='external', refine='ba', rotation_sigma_deg=bad)))
