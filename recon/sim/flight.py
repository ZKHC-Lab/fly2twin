# SPDX-License-Identifier: Apache-2.0
"""Closed-loop flight of a DiffDrone-PyTorch3D policy in a prepared scene, on CPU.

The policy network, its pre/post-processing (DronePolicy), the quadrotor dynamics and the navigation
metrics come from a DiffDrone-PyTorch3D checkout (GPL-3.0) that the user installs separately; this module
only imports them at run time. What lives here: the episode state with the training-time randomisation,
the 48x64 depth renderer that splats the scene points through the body-mounted camera, and the episode
judging (collision, out of bounds, timeout).

Coordinates: metres, world Z up; camera rotation columns are right/down/forward; depth is camera Z,
-1 where no point is hit.
"""
import hashlib
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from numba import njit
from scipy.spatial import cKDTree

DT = 1 / 15
# Body radius and camera offset (body frame) of the DiffDrone-PyTorch3D quadrotor, i.e. its
# data/base_model/drone.obj after load_drone_mesh's 500-face decimation, as float32. Depth must come
# from the camera mount the policy was trained with: offset scaled by margin / (radius + 0.05), pitched up 10 deg.
BODY_RADIUS = 0.655479371547699
CAMERA_OFFSET = np.array([0.6343910694122314, 0.0, -0.012925442308187485], np.float32)
CAMERA_PITCH_DEG = 10.
REACH_RADIUS = .5
OUT_OF_BOUNDS_M, OUT_OF_BOUNDS_S = 3., 1.


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_diffdrone(root, checkpoint):
    """Policy, dynamics and metrics from a DiffDrone-PyTorch3D checkout (model.py, navigation_utils.py, drone_dynamics.py)."""
    root = Path(root).resolve()
    missing = [n for n in ('model.py', 'navigation_utils.py', 'drone_dynamics.py') if not (root / n).is_file()]
    if missing:
        raise FileNotFoundError(f'Not a DiffDrone-PyTorch3D checkout ({", ".join(missing)} missing): {root}')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # leave the checkout untouched
    try:
        import model, navigation_utils, drone_dynamics
    finally:
        sys.dont_write_bytecode = write_bytecode
    net = model.Model_bigger(dim_obs=10, dim_action=6).cpu().eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net.load_state_dict(state.get('model_state_dict', state), strict=True)
    policy = navigation_utils.DronePolicy(net, torch.tensor([0., 0., -9.80665]), .3, 24.)
    return SimpleNamespace(policy=policy, dynamics=drone_dynamics,
                           metrics=navigation_utils.compute_navigation_metrics_np, checkpoint=Path(checkpoint))


class State:
    """Single-drone state with the randomisation of the training environment's reset()."""

    def __init__(self, dynamics, start, seed, noise=True):
        torch.manual_seed(seed)
        self.dyn, self.B, self.dt = dynamics, 1, DT
        self.p = torch.as_tensor(start, dtype=torch.float32).reshape(1, 3).clone()
        self.v = torch.zeros_like(self.p)
        self.a = torch.zeros_like(self.p)
        self.act_curr = torch.zeros_like(self.p)
        yaw = (torch.rand(1) * 2 - 1) * torch.pi
        c, s = yaw.cos(), yaw.sin()
        self.R = torch.tensor([[c.item(), -s.item(), 0.], [s.item(), c.item(), 0.], [0., 0., 1.]])[None]
        self.dg = torch.randn(1, 3) * (.2 if noise else 0.)
        self.act_queue = [torch.zeros_like(self.p) for _ in range(2)]
        self.margin = torch.rand(1) * .5 + .3
        self.pitch_ctl_delay = 12. + 1.2 * torch.randn(1, 1)
        self.drag_coef_lin = torch.rand(1, 1) * .15 + .3
        self.gravity = torch.tensor([0., 0., -9.80665]).expand(1, 3)
        self.noise_std, self.wind_std = (.04, .1) if noise else (0., 0.)

    def step(self, act_cmd, target_pos_vector):
        p_old = self.p.clone()
        self.dg = self.dyn.update_dg(self.dg, self.dt, self.noise_std)
        self.act_queue.append(act_cmd)
        command = self.act_queue.pop(0)
        v_wind = torch.randn(self.B, 3) * self.wind_std
        self.p, self.v, self.a, self.act_curr = self.dyn.simulate_position_step(
            p=self.p, v=self.v, a=self.a, R=self.R, act_curr=self.act_curr, act_cmd=command, dt=self.dt,
            enable_airmode=True, enable_induced_drag=False, dg=self.dg, v_wind=v_wind, grad_decay=.4,
            pitch_ctl_delay=self.pitch_ctl_delay, drag_coef_lin=self.drag_coef_lin, drag_coef_quad=0.,
            z_drag_coef=1., rotor_drag_coef=.07, airmode_coef=.5)
        self.R = self.dyn.solve_attitude_from_thrust_and_goal_vec(
            self.act_curr - self.gravity, self.p - p_old if target_pos_vector is None else target_pos_vector,
            self.R, yaw_inertia=5., dt=self.dt, yaw_ctl_delay=12.)


class ClosedLoop:
    def __init__(self, state, policy):
        self.state, self.policy = state, policy
        self.policy.reset()
        self.h = None
        self.act_buffer = [state.act_curr.clone() for _ in range(2)]
        self.max_speed = torch.full((state.B, 1), 2.5)
        self.thr_est_error = torch.ones(state.B, 1)

    @torch.no_grad()
    def advance(self, depth, target):
        s = self.state
        target_raw = torch.as_tensor(target, dtype=torch.float32).reshape(s.B, 3) - s.p
        depth = torch.as_tensor(depth, dtype=torch.float32).reshape(s.B, 48, 64)
        # Training order: the policy sees the previous frame's depth with the post-step odometry.
        s.step(self.act_buffer.pop(0), target_raw)
        act, _, _, self.h = self.policy.infer(depth, s.R, s.v, target_raw, s.margin, self.max_speed,
                                              self.thr_est_error, self.h, depth_noise_std=0.)
        self.act_buffer.append(act)


@njit(cache=True)
def _raster(points, eye, R, width, height, fx, fy, cx, cy, radius, near, far):
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    hit = np.zeros((height, width), dtype=np.bool_)
    for i in range(len(points)):
        dx, dy, dz = points[i, 0] - eye[0], points[i, 1] - eye[1], points[i, 2] - eye[2]
        z = dx * R[0, 2] + dy * R[1, 2] + dz * R[2, 2]
        if z < near or z > far:
            continue
        x = dx * R[0, 0] + dy * R[1, 0] + dz * R[2, 0]
        y = dx * R[0, 1] + dy * R[1, 1] + dz * R[2, 1]
        u, v = fx * x / z + cx, fy * y / z + cy
        rx, ry = fx * radius / z, fy * radius / z
        if u + rx < .5 or u - rx > width - .5 or v + ry < .5 or v - ry > height - .5:
            continue
        x0, x1 = max(0, math.ceil(u - rx - .5)), min(width - 1, math.floor(u + rx - .5))
        y0, y1 = max(0, math.ceil(v - ry - .5)), min(height - 1, math.floor(v + ry - .5))
        for row in range(y0, y1 + 1):
            for col in range(x0, x1 + 1):
                if z < zbuf[row, col]:
                    zbuf[row, col] = z
                    hit[row, col] = True
    for row in range(height):
        for col in range(width):
            if not hit[row, col]:
                zbuf[row, col] = -1.
    return zbuf


class DepthRenderer:
    """Point splats of fixed world radius, z-buffered: the policy's 48x64, 79 deg depth camera."""

    def __init__(self, points, radius=.03, width=64, height=48, hfov=79., near=.3, far=24.):
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.radius, self.width, self.height, self.near, self.far = radius, width, height, near, far
        self.f = width / (2 * np.tan(np.deg2rad(hfov) / 2))

    def render(self, eye, camera_R):
        return _raster(self.points, np.asarray(eye, dtype=np.float32), np.asarray(camera_R, dtype=np.float32),
                       self.width, self.height, self.f, self.f, self.width / 2, self.height / 2,
                       self.radius, self.near, self.far)

    @staticmethod
    def camera_pose(p, R, margin):
        a = np.deg2rad(CAMERA_PITCH_DEG)
        mount = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]], dtype=np.float32)
        optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
        R = np.asarray(R).reshape(3, 3)
        eye = np.asarray(p).reshape(3) + R @ (CAMERA_OFFSET * (float(margin) / (BODY_RADIUS + .05)))
        return eye, R @ mount @ optical

    def render_body(self, p, R, margin):
        return self.render(*self.camera_pose(p, R, margin))


def evaluate(scene, pairs, agent, steps=200, seed=20261020, limit=None, save_depth=False, log=print):
    """Fly every start/goal pair once; return (summary dict, per-step arrays)."""
    torch.set_num_threads(1)
    count = len(pairs['starts']) if limit is None else int(limit)
    starts, goals = pairs['starts'][:count], pairs['goals'][:count]
    points = np.ascontiguousarray(scene['points'], dtype=np.float32)
    n_observed = int(scene['observed_count'])
    renderer = DepthRenderer(points)
    observed_tree = cKDTree(points[:n_observed])
    ceiling_tree = cKDTree(points[n_observed:]) if len(points) > n_observed else None
    episodes, rows, begin = [], [], time.monotonic()
    for i, (start, goal) in enumerate(zip(starts, goals)):
        state = State(agent.dynamics, start, seed + i)
        loop = ClosedLoop(state, agent.policy)
        margin = float(state.margin[0])
        rec = {k: [] for k in ('p', 'R', 'v', 'clearance', 'distance', 'collision', 'ceiling_collision')
               + (('depth',) if save_depth else ())}
        far_since, outside = None, False
        for t in range(steps):
            p, R, v = state.p[0].numpy().copy(), state.R[0].numpy().copy(), state.v[0].numpy().copy()
            observed_d = float(observed_tree.query(p, k=1)[0])
            ceiling_d = float(ceiling_tree.query(p, k=1)[0]) if ceiling_tree else np.inf
            clearance = min(observed_d, ceiling_d)
            far_since = (far_since if far_since is not None else t) if clearance > OUT_OF_BOUNDS_M else None
            outside |= far_since is not None and (t - far_since) * DT >= OUT_OF_BOUNDS_S - 1e-8
            depth = renderer.render_body(p, R, margin)
            rec['p'].append(p), rec['R'].append(R), rec['v'].append(v)
            rec['clearance'].append(clearance)
            rec['distance'].append(np.linalg.norm(goal - p))
            rec['collision'].append(np.sqrt(clearance * clearance + 1e-6) < margin)
            rec['ceiling_collision'].append(np.sqrt(ceiling_d * ceiling_d + 1e-6) < margin)
            if save_depth:
                rec['depth'].append(depth.astype(np.float16))
            loop.advance(depth, goal)
            if not torch.isfinite(state.p).all():
                raise FloatingPointError(f'Non-finite state in episode {i} at step {t}')
        rec = {k: np.asarray(v) for k, v in rec.items()}
        nav = agent.metrics(rec['distance'], rec['collision'], reach_radius=REACH_RADIUS)
        collision = not nav['collision_free']
        success = bool(nav['success'] and not outside)
        result = 'out_of_bounds' if outside else 'collision' if collision else 'success' if success else 'timeout'
        steps_len = np.linalg.norm(np.diff(rec['p'], axis=0), axis=1)
        reached = np.flatnonzero(rec['distance'] <= REACH_RADIUS)
        first = int(reached[0]) if len(reached) else -1
        shortest = float(pairs['shortest_lengths'][i])
        row = dict(index=i, seed=seed + i, start=start.tolist(), goal=goal.tolist(), margin_m=margin,
                   straight_blocked=bool(pairs['straight_blocked'][i]), shortest_length_m=shortest,
                   result=result, success=success, reached=bool(nav['reached_target']), collision=collision,
                   out_of_bounds=bool(outside), ceiling_collision=bool(rec['ceiling_collision'].any()),
                   min_clearance_m=float(rec['clearance'].min()), min_goal_distance_m=float(rec['distance'].min()),
                   first_reach_step=first, avg_speed_mps=float(np.linalg.norm(rec['v'], axis=1).mean()),
                   peak_speed_mps=float(np.linalg.norm(rec['v'], axis=1).max()),
                   path_length_m=float(steps_len.sum()), path_over_shortest=float(steps_len.sum()) / shortest,
                   arrival_path_over_shortest=float(steps_len[:first].sum()) / shortest if first >= 0 else None)
        episodes.append(rec)
        rows.append(row)
        log(f"{i:3d} {result:13s} min clearance {row['min_clearance_m']:.2f} m, path/shortest {row['path_over_shortest']:.2f}")
    good = [r for r in rows if r['success']]
    mean = lambda key: float(np.mean([r[key] for r in good])) if good else None
    summary = dict(count=len(rows), steps=steps, duration_s=steps * DT, seed=seed,
                   counts={k: sum(r['result'] == k for r in rows) for k in ('success', 'collision', 'timeout', 'out_of_bounds')},
                   success_rate=len(good) / len(rows), success_avg_speed_mps=mean('avg_speed_mps'),
                   success_path_over_shortest=mean('path_over_shortest'),
                   success_arrival_path_over_shortest=mean('arrival_path_over_shortest'),
                   ceiling_z=float(scene['ceiling_z']), observed_points=n_observed,
                   synthetic_ceiling_points=len(points) - n_observed, point_radius_m=renderer.radius,
                   checkpoint_sha256=digest(agent.checkpoint), wall_seconds=time.monotonic() - begin, rows=rows)
    arrays = {k: np.stack([e[k] for e in episodes]) for k in episodes[0]}
    arrays.update(starts=starts, goals=goals, margins=np.asarray([r['margin_m'] for r in rows]),
                  results=np.asarray([r['result'] for r in rows]), dt=np.float64(DT))
    return summary, arrays
