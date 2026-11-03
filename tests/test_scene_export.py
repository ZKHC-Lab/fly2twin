import json

import numpy as np

from recon.io.export import scene_bounds, scene_up
from recon.io.ply import point_vertices, write_ply
from recon.io.scene import load_scene, read_gaussians


def gaussian_ply(path, n=5, rest=45):
    rng = np.random.default_rng(0)
    cols = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
            *[f'f_rest_{i}' for i in range(rest)], 'opacity', 'scale_0', 'scale_1', 'scale_2',
            'rot_0', 'rot_1', 'rot_2', 'rot_3']
    v = np.empty(n, dtype=[(c, '<f4') for c in cols])
    for c in cols:
        v[c] = rng.normal(size=n)
    write_ply(path, v)
    return v


def test_read_gaussians_channel_major_sh(tmp_path):
    v = gaussian_ply(tmp_path / 'g.ply')
    g = read_gaussians(tmp_path / 'g.ply')
    assert g['shN'].shape == (5, 15, 3) and g['sh0'].shape == (5, 1, 3)
    # export writes rest as channel-major: f_rest_{c*15+k} is coefficient k of channel c
    assert np.isclose(g['shN'][2, 3, 1], v['f_rest_18'][2])
    assert np.allclose(g['quats'][:, 0], v['rot_0'])


def test_load_scene_split_and_intrinsics(tmp_path):
    gaussian_ply(tmp_path / 'gaussians.ply')
    write_ply(tmp_path / 'cloud.ply', point_vertices(np.zeros((3, 3), np.float32), np.ones((3, 3), np.float32)))
    c2w = np.tile(np.eye(4), (3, 1, 1))
    c2w[:, 0, 3] = [0, 1, 2]
    K = np.tile(np.array([[100., 0, 63.5], [0, 100, 47.5], [0, 0, 1]]), (3, 1, 1))
    np.savez(tmp_path / 'cameras.npz', frame_ids=np.array([10, 20, 30]), c2w=c2w, K=K, resolution=[128, 96])
    (tmp_path / 'scene.json').write_text(json.dumps(dict(
        preferred_model=dict(path='gaussians.ply'), cameras='cameras.npz', cloud='cloud.ply',
        capture_corridor=dict(training_frame_ids=[10, 30]), world_up=[0, 0, 2])))
    s = load_scene(tmp_path, width=64)
    assert list(s.train_ids) == [10, 30] and list(s.test_ids) == [20]
    assert (s.width, s.height) == (64, 48)
    assert np.isclose(s.train_K[0][0, 0], 50) and np.isclose(s.train_K[0][0, 2], 31.5)
    assert np.allclose(s.world_up, [0, 0, 1])
    assert np.allclose(s.bounds['min'], [-1, -1, -1]) and np.allclose(s.bounds['max'], [3, 1, 1])
    pts, rgb = s.points()
    assert pts.shape == (3, 3) and np.allclose(rgb, 1)


def test_scene_up_reads_fitted_floor_normal(tmp_path):
    assert scene_up(tmp_path) == [0, 0, 1]
    (tmp_path / 'pose').mkdir()
    (tmp_path / 'pose/report.json').write_text(json.dumps(dict(scale_assumption=dict(plane_normal=[0, -2, 0]))))
    assert np.allclose(scene_up(tmp_path), [0, -1, 0])


def test_scene_bounds_robust():
    pts = np.zeros((1000, 3))
    pts[:, 0] = np.linspace(0, 10, 1000)
    pts[0] = [1e6, 0, 0]
    b = scene_bounds(pts)
    assert b['max'][0] < 11 and b['min'][0] >= 0
