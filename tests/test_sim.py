# SPDX-License-Identifier: Apache-2.0
import json
import textwrap

import numpy as np
import pytest

from recon.io.ply import point_vertices, write_ply
from recon.sim import scene as S


def room(tilt_deg=5., lift=.4):
    """16 x 8 m room, 3 m high, 10 cm point spacing, with a 5 m partition wall; floor tilted and lifted."""
    g = np.arange(0, 16.01, .1)
    h = np.arange(0, 8.01, .1)
    z = np.arange(0, 3.01, .1)
    floor = np.stack(np.meshgrid(g, h, [0.]), -1).reshape(-1, 3)
    walls = [np.stack(np.meshgrid(g, [0., 8.], z), -1).reshape(-1, 3),
             np.stack(np.meshgrid([0., 16.], h, z), -1).reshape(-1, 3),
             np.stack(np.meshgrid([8.], np.arange(0, 5.01, .1), z), -1).reshape(-1, 3)]
    points = np.concatenate([floor, *walls])
    a = np.deg2rad(tilt_deg)
    rotation = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return (points @ rotation.T + [0, 0, lift]).astype(np.float32)


def export(tmp_path, points):
    write_ply(tmp_path / 'cloud.ply', point_vertices(points, np.full(points.shape, .5)))
    (tmp_path / 'scene.json').write_text(json.dumps(dict(units='meters', cloud='cloud.ply')))
    return tmp_path


def test_prepare_levels_floor_and_closes_ceiling(tmp_path):
    scene = S.prepare(export(tmp_path, room()), ceiling_z=2.5)
    n = int(scene['observed_count'])
    observed, ceiling = scene['points'][:n], scene['points'][n:]
    floor = observed[:161 * 81]                                             # room() lists the floor first
    assert np.abs(floor[:, 2]).max() < .005                                 # 5 deg tilt and 0.4 m lift removed
    assert np.all(ceiling[:, 2] == np.float32(2.5)) and len(ceiling) > 0
    # Real points at or above 2.25 m (wall tops) suppress synthetic samples in their 10 cm cells.
    high = np.floor(observed[observed[:, 2] >= 2.25, :2].astype(float) / .1).astype(int)
    cells = np.floor(ceiling[:, :2].astype(float) / .1).astype(int)
    assert not set(map(tuple, high)) & set(map(tuple, cells))
    assert json.loads(str(scene['info']))['rotated']


def test_pairs_respect_clearance_and_blocking(tmp_path):
    scene = S.prepare(export(tmp_path, room(0., 0.)), ceiling_z=2.5)
    pairs = S.sample_pairs(scene, count=6, blocked_min=3, distance_range=(5., 14.))
    from scipy.spatial import cKDTree
    tree = cKDTree(scene['points'])
    ends = np.vstack([pairs['starts'], pairs['goals']])
    assert tree.query(ends)[0].min() >= .9 and np.all(ends[:, 2] == np.float32(1.2))
    assert pairs['straight_blocked'][:3].all()
    assert np.all(pairs['shortest_lengths'] >= pairs['distances'] - .15)
    again = S.sample_pairs(scene, count=6, blocked_min=3, distance_range=(5., 14.))
    assert np.array_equal(pairs['starts'], again['starts'])                  # fixed seed


def test_depth_renderer_sees_wall():
    flight = pytest.importorskip('recon.sim.flight')
    y, z = np.meshgrid(np.arange(-4, 4, .02), np.arange(-3, 3, .02))
    wall = np.column_stack([np.full(y.size, 4.), y.ravel(), z.ravel()]).astype(np.float32)
    renderer = flight.DepthRenderer(wall)
    forward = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], np.float32)   # optical axis along world +X
    depth = renderer.render(np.zeros(3), forward)
    assert depth.shape == (48, 64) and abs(depth[24, 32] - 4.) < .01 and (depth > 0).all()
    assert (renderer.render(np.zeros(3), -forward) == -1).all()               # facing away: no hit


STUB = {
    'model.py': '''
        import torch
        class Model_bigger(torch.nn.Module):
            def __init__(self, dim_obs, dim_action):
                super().__init__()
                self.fc = torch.nn.Linear(dim_obs, dim_action)
    ''',
    'navigation_utils.py': '''
        import torch
        class DronePolicy:
            def __init__(self, model, gravity, near, far):
                self.model = model
            def reset(self):
                pass
            def infer(self, depth, R, v, target, margin, max_speed, thr, h, depth_noise_std=0.):
                return 2 * target / target.norm(dim=1, keepdim=True) - v, None, None, h
        def compute_navigation_metrics_np(distance, collision, reach_radius):
            import numpy as np
            reached = bool((distance <= reach_radius).any())
            return dict(success=reached and not collision.any(), reached_target=reached, collision_free=not collision.any())
    ''',
    'drone_dynamics.py': '''
        import torch
        def update_dg(dg, dt, std):
            return dg
        def simulate_position_step(p, v, a, R, act_curr, act_cmd, dt, **kw):
            v = v + act_cmd * dt
            return p + v * dt, v, act_cmd, act_cmd
        def solve_attitude_from_thrust_and_goal_vec(thrust, goal, R, **kw):
            return R
    ''',
}


def test_evaluate_with_stub_policy(tmp_path):
    flight = pytest.importorskip('recon.sim.flight')
    torch = pytest.importorskip('torch')
    stub = tmp_path / 'diffdrone'
    stub.mkdir()
    for name, source in STUB.items():
        (stub / name).write_text(textwrap.dedent(source))
    scene = S.prepare(export(tmp_path, room(0., 0.)), ceiling_z=2.5)
    pairs = S.sample_pairs(scene, count=2, blocked_min=0, distance_range=(3., 6.))
    import sys
    sys.path.insert(0, str(stub))
    try:
        import model
        torch.save({'model_state_dict': model.Model_bigger(10, 6).state_dict()}, tmp_path / 'policy.pth')
        agent = flight.load_diffdrone(stub, tmp_path / 'policy.pth')
        summary, arrays = flight.evaluate(scene, pairs, agent, steps=150, log=lambda *_: None)
    finally:
        sys.path.remove(str(stub))
        for name in ('model', 'navigation_utils', 'drone_dynamics'):
            sys.modules.pop(name, None)
    assert summary['count'] == 2 and arrays['p'].shape == (2, 150, 3)
    assert all(r['reached'] for r in summary['rows'])                     # straight-line stub reaches open goals
    assert set(summary['counts']) == {'success', 'collision', 'timeout', 'out_of_bounds'}
    with pytest.raises(FileNotFoundError):
        flight.load_diffdrone(tmp_path, tmp_path / 'policy.pth')
