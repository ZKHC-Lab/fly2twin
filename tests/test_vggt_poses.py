# SPDX-License-Identifier: Apache-2.0
"""VGGT pose provider on a synthetic room: windows with their own frame and scale, exact depth."""

import json
import sqlite3

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recon.poses import vggt

H, W, FOCAL = 36, 64, 40.0
K = np.array([[FOCAL, 0, W / 2], [0, FOCAL, H / 2], [0, 0, 1.0]])
ROOM = np.array([[-5.0, 5.0], [-2.0, 2.0], [-4.0, 4.0]])  # x, y, z walls
SCALES = (1.0, 1.7, 0.6, 2.2, 0.9)


def trajectory(n=160):
    """Cameras walking 1.3 turns of a circle of radius 2 m, looking along the direction of travel, 5 cm bob.

    Frame i + 123 stands where frame i stood, so the last window sees what the first one saw.
    """
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i, a in enumerate(np.linspace(0, 2.6 * np.pi, n)):
        poses[i, :3, :3] = Rotation.from_euler('y', -a).as_matrix() @ Rotation.from_euler('x', 0.05 * np.sin(5 * a)).as_matrix()
        poses[i, :3, 3] = [2 * np.cos(a), 0.05 * np.sin(7 * a), 2 * np.sin(a)]
    return poses


FRAMES = [5, 20, 30, 50, 70, 100, 128, 143, 153]  # windows 0 0 0 1 2 3 4 4 4; 128 revisits 5


def depth_map(pose):
    """Optical depth of the room walls seen from `pose`, pixel centres at +0.5."""
    v, u = np.mgrid[0:H, 0:W] + 0.5
    rays = np.stack([(u - K[0, 2]) / FOCAL, (v - K[1, 2]) / FOCAL, np.ones_like(u)], -1) @ pose[:3, :3].T
    with np.errstate(divide='ignore', invalid='ignore'):
        hits = np.stack([(ROOM[axis, side] - pose[axis, 3]) / rays[..., axis] for axis in range(3) for side in range(2)])
    hits[~(hits > 1e-6)] = np.inf
    return hits.min(0).astype(np.float32)


def write_windows(folder, poses, scales=SCALES, first=40, new=30, overlap=10):
    from recon.depth.backends import plan_windows

    rng = np.random.default_rng(0)
    plan = plan_windows(range(len(poses)), first, new, overlap)
    for win, g in zip(plan, scales):
        base = poses[win['idx'][0]]
        E, depth = [], []
        for i in win['idx']:
            R = base[:3, :3].T @ poses[i, :3, :3]
            c = g * base[:3, :3].T @ (poses[i, :3, 3] - base[:3, 3])
            E.append(np.c_[R.T, -R.T @ c])
            depth.append(g * depth_map(poses[i]))
        (folder / win['id']).mkdir(parents=True)
        np.savez(
            folder / win['id'] / 'prediction.npz',
            extrinsic=np.array(E),
            intrinsic=np.tile(K, (len(E), 1, 1)),
            depth=np.array(depth)[..., None],
            depth_conf=rng.random((len(E), H, W)).astype(np.float32),
        )
    (folder / 'windows.json').write_text(json.dumps(dict(windows=plan)))
    return plan


def to_chain(points, poses, g=SCALES[0]):
    """World points in the chain frame (the frame and unit of window 0)."""
    return g * (points - poses[0, :3, 3]) @ poses[0, :3, :3]


def error(found, truth):
    """Largest camera-centre distance after one similarity fit, relative to the path length."""
    s, R, t = vggt.similarity(found[:, :3, 3], truth[:, :3, 3])
    path = np.linalg.norm(np.diff(truth[:, :3, 3], axis=0), axis=1).sum()
    return float(np.linalg.norm(s * found[:, :3, 3] @ R.T + t - truth[:, :3, 3], axis=1).max() / path)


@pytest.fixture
def scene(tmp_path):
    poses = trajectory()
    write_windows(tmp_path, poses)
    windows = vggt.load_windows(tmp_path)
    placed, spread = vggt.chain(windows)
    assert max(spread) < 1e-4  # exact depth: the shared frames agree about the scale
    return poses, windows, placed


def test_chain_recovers_trajectory_and_scale_from_windows_in_their_own_units(scene):
    poses, windows, placed = scene
    np.testing.assert_allclose([s for s, _, _ in placed], [SCALES[0] / g for g in SCALES], rtol=1e-4)
    ids, found = vggt.frame_poses(windows, placed)
    assert ids.tolist() == list(range(len(poses)))
    assert error(found, poses) < 1e-4
    turn = np.einsum('nji,njk->nik', found[:, :3, :3], poses[:, :3, :3])
    assert np.allclose(turn, turn[0], atol=1e-5)  # one fixed rotation between the chain frame and the world


def room_links(poses, rng, a, b, drift=None, noise=0.0, n=40, pairs=6, shift=None):
    """Links between windows a and b: room points seen from a camera of each window.

    `drift[k]` maps chain coordinates to where window k has drifted; `noise` is relative depth noise along each
    camera's own viewing ray; `shift` moves window b's points (a match between two look-alike places).
    """
    first = {0: 0, 1: 40, 2: 70, 3: 100, 4: 130}
    links = []
    for _ in range(pairs):
        world = rng.uniform(ROOM[:, 0], ROOM[:, 1], (n, 3))
        ca, cb = (poses[first[k] + int(rng.integers(0, 25)), :3, 3] for k in (a, b))
        Pa, Pb = (c + (world - c) * np.exp(noise * rng.standard_normal((n, 1))) for c in (ca, cb))
        if shift is not None:
            Pb = Pb + shift
        Pa, Pb, ca, cb = (to_chain(x, poses) for x in (Pa, Pb, ca, cb))
        if drift is not None:
            Pa, ca, Pb, cb = drift[a](Pa), drift[a](ca), drift[b](Pb), drift[b](cb)
        links.append((a, b, Pa, Pb, ca, cb))
    return links


def test_pose_graph_leaves_a_correct_chain_alone(scene):
    poses, windows, placed = scene
    rng = np.random.default_rng(1)
    corrections, info = vggt.pose_graph(windows, placed, room_links(poses, rng, 0, 2) + room_links(poses, rng, 1, 3) + room_links(poses, rng, 0, 4))
    assert info['pair_residual_median_before'] < 1e-9 and info['pairs_dropped'] == 0
    np.testing.assert_allclose([c.scale for c in corrections], 1.0, atol=1e-6)
    _, found = vggt.frame_poses(windows, placed, corrections)
    assert error(found, poses) < 1e-4
    untouched, info = vggt.pose_graph(windows, placed, [])
    assert [c.scale for c in untouched] == [1.0] * 5 and info['pairs'] == 0 and info['solves'] == []


def test_pose_graph_repairs_accumulated_drift(scene):
    poses, windows, placed = scene

    def drift_of(k):  # 3% scale, 1.2 degrees and a few centimetres more with every window, as a chain drifts
        scale, turn = 1.03**k, Rotation.from_euler('xyz', np.array([0.5, 1.0, -0.4]) * k, degrees=True).as_matrix()
        shift = np.array([0.04, -0.01, 0.03]) * k
        return scale, turn, shift

    drift, drifted = [], []
    for k, (s, A, t) in enumerate(placed):
        scale, turn, shift = drift_of(k)
        drift.append(lambda X, scale=scale, turn=turn, shift=shift: scale * X @ turn.T + shift)
        drifted.append((scale * s, turn @ A, scale * turn @ t + shift))
    _, before = vggt.frame_poses(windows, drifted)
    rng = np.random.default_rng(2)
    links = sum((room_links(poses, rng, a, b, drift, pairs=20) for a, b in ((0, 4), (1, 3), (0, 2))), [])
    corrections, info = vggt.pose_graph(windows, drifted, links)
    _, after = vggt.frame_poses(windows, drifted, corrections)
    assert error(before, poses) > 0.01
    assert error(after, poses) < 0.25 * error(before, poses)
    assert info['pair_residual_median_after'] < 0.1 * info['pair_residual_median_before']
    assert all(solve['status'] > 0 for solve in info['solves'])
    assert abs(corrections[4].scale * 1.03**4 - 1) < 0.02


def test_pose_graph_does_not_shrink_windows_that_are_not_tied_to_the_first(scene):
    poses, windows, placed = scene
    rng = np.random.default_rng(3)  # noisy depth, no link to window 0
    links = room_links(poses, rng, 1, 3, noise=0.03, pairs=30) + room_links(poses, rng, 2, 4, noise=0.03, pairs=30)
    corrections, _ = vggt.pose_graph(windows, placed, links)
    np.testing.assert_allclose([c.scale for c in corrections], 1.0, atol=0.01)


def test_noisy_depth_on_a_correct_chain_moves_nothing_much(scene):
    """The input is already right: 3% depth noise along each camera's own ray must not bend or rescale the chain."""
    poses, windows, placed = scene
    for seed in (4, 5):
        rng = np.random.default_rng(seed)
        links = sum((room_links(poses, rng, a, b, noise=0.03, pairs=25) for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3))), [])
        corrections, info = vggt.pose_graph(windows, placed, links)
        np.testing.assert_allclose([c.scale for c in corrections], 1.0, atol=0.01)
        _, found = vggt.frame_poses(windows, placed, corrections)
        assert error(found, poses) < 0.003 and info['pairs_dropped'] <= 2


def test_matches_between_look_alike_places_do_not_drag_the_chain(scene):
    poses, windows, placed = scene
    rng = np.random.default_rng(6)
    true = sum((room_links(poses, rng, a, b, noise=0.02, pairs=12) for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 4))), [])
    false = room_links(poses, rng, 1, 3, noise=0.02, pairs=6, shift=np.array([1.5, 0.0, 0.8]))
    corrections, info = vggt.pose_graph(windows, placed, true + false)
    _, found = vggt.frame_poses(windows, placed, corrections)
    assert info['pairs_dropped'] >= 5
    assert error(found, poses) < 0.005


def write_matches(path, poses, frames, scale, rng, count=200):
    """A minimal COLMAP database: keypoints of room points in `frames` (image pixels = depth grid * scale) and their matches."""
    world = np.concatenate([rng.uniform(ROOM[:, 0], ROOM[:, 1], (count, 3)) * [1, 1, 0] + [0, 0, ROOM[2, side]] for side in (0, 1)]
                           + [rng.uniform(ROOM[:, 0], ROOM[:, 1], (count, 3)) * [0, 1, 1] + [ROOM[0, side], 0, 0] for side in (0, 1)])
    con = sqlite3.connect(path)
    con.execute('create table images (image_id integer primary key, name text)')
    con.execute('create table keypoints (image_id integer primary key, rows integer, cols integer, data blob)')
    con.execute('create table two_view_geometries (pair_id integer primary key, rows integer, data blob)')
    seen = {}
    for image_id, frame in enumerate(frames, 1):
        cam = (world - poses[frame, :3, 3]) @ poses[frame, :3, :3]
        uv = cam[:, :2] / cam[:, 2:3] * FOCAL + [W / 2, H / 2]
        inside = (cam[:, 2] > 0.2) & (uv[:, 0] > 2) & (uv[:, 0] < W - 2) & (uv[:, 1] > 2) & (uv[:, 1] < H - 2)
        seen[image_id] = np.flatnonzero(inside)
        con.execute('insert into images values (?, ?)', (image_id, f'{frame:08d}.jpg'))
        data = (uv[inside] * scale).astype(np.float32)
        con.execute('insert into keypoints values (?, ?, 2, ?)', (image_id, len(data), data.tobytes()))
    for a in seen:
        for b in seen:
            if a < b:
                common, ia, ib = np.intersect1d(seen[a], seen[b], return_indices=True)
                matches = np.c_[ia, ib].astype(np.uint32)
                con.execute('insert into two_view_geometries values (?, ?, ?)', (a * vggt.PAIR_ID_BASE + b, len(matches), matches.tobytes()))
    con.commit()
    con.close()


def test_match_links_lift_the_same_room_points_from_both_frames(scene, tmp_path):
    poses, windows, placed = scene
    database = tmp_path / 'matches.db'
    write_matches(database, poses, FRAMES, 30.0, np.random.default_rng(4))
    links, counts = vggt.match_links(windows, placed, database, lambda name: int(name[:8]), np.array([1 / 30.0, 1 / 30.0]), min_matches=20)
    assert counts['used'] >= 3 and counts['inconsistent'] == 0 and counts['far'] >= 1
    assert (0, 4) in {(a, b) for a, b, *_ in links}  # the revisit
    ids, chained = vggt.frame_poses(windows, placed)
    centres = {tuple(np.round(c, 9)) for c in chained[:, :3, 3]}
    for a, b, Pa, Pb, ca, cb in links:
        depth = np.linalg.norm(Pa - ca, axis=1)
        assert np.median(np.linalg.norm(Pa - Pb, axis=1) / depth) < 0.03  # nearest-pixel depth, exact geometry
        assert tuple(np.round(ca, 9)) in centres and tuple(np.round(cb, 9)) in centres


def test_match_links_reject_matches_between_different_places(scene, tmp_path):
    poses, windows, placed = scene
    database = tmp_path / 'matches.db'
    write_matches(database, poses, [5, 128], 30.0, np.random.default_rng(5))
    con = sqlite3.connect(database)
    n, blob = con.execute('select rows, data from two_view_geometries').fetchone()
    matches = np.frombuffer(blob, np.uint32).reshape(n, 2).copy()
    matches[:, 1] = np.random.default_rng(6).permutation(matches[:, 1])  # the same keypoints, paired wrongly
    con.execute('update two_view_geometries set data = ?', (matches.tobytes(),))
    con.commit()
    con.close()
    links, counts = vggt.match_links(windows, placed, database, lambda name: int(name[:8]), np.array([1 / 30.0, 1 / 30.0]), min_matches=20)
    assert links == [] and counts['inconsistent'] + counts['few_points'] + counts['displaced'] == counts['candidates'] == 1


def test_track_returns_poses_for_every_frame(scene, tmp_path):
    poses, _, _ = scene
    database = tmp_path / 'matches.db'
    write_matches(database, poses, FRAMES, 30.0, np.random.default_rng(7))
    ids, closed, plain, info, focal = vggt.track(tmp_path, database, lambda name: int(name[:8]), W * 30, H * 30)
    assert ids.tolist() == list(range(len(poses))) and closed.shape == plain.shape == (len(poses), 4, 4)
    assert error(closed, poses) < 2e-3 and info['windows'] == 5 and info['match_pairs']['used'] >= 3
    assert focal == pytest.approx(FOCAL * 30)


def test_a_match_to_a_shifted_copy_of_the_place_is_not_used(scene, tmp_path):
    """Frame 128 revisits frame 5. Its keypoints are taken 1.5 m along the wall: the points still fit each other."""
    poses, windows, placed = scene
    database = tmp_path / 'matches.db'
    moved = poses.copy()
    moved[128, :3, 3] += moved[128, :3, :3] @ [1.5, 0, 0]  # what a camera 1.5 m to the side would see
    write_matches(database, moved, [5, 128], 30.0, np.random.default_rng(8))
    links, counts = vggt.match_links(windows, placed, database, lambda name: int(name[:8]), np.array([1 / 30.0, 1 / 30.0]), min_matches=20)
    assert links == [] and counts['candidates'] == 1 and counts['displaced'] + counts['inconsistent'] == 1


def test_start_camera_sets_every_focal_length_and_clears_distortion():
    pc = pytest.importorskip('pycolmap')
    radial = pc.Camera(model='SIMPLE_RADIAL', width=1920, height=1080, params=[2304.0, 960.0, 540.0, 0.3])
    np.testing.assert_allclose(vggt.start_camera(radial, 1500.0), [1500, 960, 540, 0])
    opencv = pc.Camera(model='OPENCV', width=1920, height=1080, params=[2304.0, 2304.0, 960.0, 540.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(vggt.start_camera(opencv, 1500.0), [1500, 1500, 960, 540, 0, 0, 0, 0])


def test_calibrated_input_is_refused(tmp_path):
    with pytest.raises(ValueError, match='calibrated input'):
        vggt.reconstruct(dict(frame_ids=np.arange(3)), dict(resolution=[64, 36], K=np.eye(3).tolist(), frames=[]), tmp_path, {}, {})


def test_speed_scale():
    from recon.poses.metric import speed_scale

    n = 200
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, 0, 3] = np.arange(n) * 0.02  # 0.02 units per frame
    stamps = (np.arange(n) * 0.1e9).astype(np.int64)  # 10 frames per second: 0.2 units per second
    scale, report = speed_scale(c2w, stamps, 1.0)
    assert scale == pytest.approx(5.0) and report['measured'] is False
    assert report['path_length_assumed_m'] == pytest.approx(5.0 * 0.02 * (n - 1))
    with pytest.raises(ValueError, match='moving camera'):
        speed_scale(np.tile(np.eye(4), (n, 1, 1)), stamps, 1.0)


def test_speed_scale_ignores_standing_still_and_jitter():
    from recon.poses.metric import speed_scale

    rng = np.random.default_rng(9)
    n = 600
    step = np.where((np.arange(n) > 150) & (np.arange(n) < 510), 0.0, 0.02)  # stands still for 60% of the time
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, 0, 3] = np.cumsum(step)
    c2w[:, :3, 3] += 0.003 * rng.standard_normal((n, 3))  # hand jitter
    stamps = (np.arange(n) * 0.1e9).astype(np.int64)
    scale, report = speed_scale(c2w, stamps, 1.0)
    assert scale == pytest.approx(5.0, rel=0.03) and report['moving_stretches'] < report['stretches']


def test_up_direction_survives_a_pitched_camera():
    from recon.poses.metric import speed_scale

    n = 240
    c2w = np.tile(np.eye(4), (n, 1, 1))
    for i, a in enumerate(np.linspace(0, 1.5 * np.pi, n)):  # walking a curve, camera pitched 30 degrees down
        c2w[i, :3, :3] = Rotation.from_euler('y', a).as_matrix() @ Rotation.from_euler('x', -np.radians(30)).as_matrix()
        c2w[i, :3, 3] = [3 * np.sin(a), 0, 3 * np.cos(a)]
    _, report = speed_scale(c2w, (np.arange(n) * 0.1e9).astype(np.int64), 1.0)
    np.testing.assert_allclose(report['up_direction'], [0, -1, 0], atol=1e-6)  # world up is -y here (image y points down)


def test_configuration_of_the_provider_and_its_scale():
    from recon.config import validate

    base = dict(video='v.mov', output='out', protocol='phone')
    validate(dict(base, pose=dict(provider='vggt', metric=dict(speed_mps=1.0))))
    validate(dict(base, pose=dict(provider='vggt', metric=dict(camera_height_m=1.4))))
    for metric, message in ((dict(speed_mps=0), 'speed_mps must be positive'), (dict(speed_mps=1.0, camera_height_m=1.4), 'not both')):
        with pytest.raises(ValueError, match=message):
            validate(dict(base, pose=dict(provider='vggt', metric=metric)))
    with pytest.raises(ValueError, match='external, colmap or vggt'):
        validate(dict(base, pose=dict(provider='orb')))
