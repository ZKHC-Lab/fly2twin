# Policy evaluation in a reconstructed scene (`recon sim`)

2026-11-10

`recon sim` flies a vision-based obstacle-avoidance policy inside a scene written by the `export` stage of `recon run`. It closes the reconstructed space into a flight volume, samples start/goal pairs, renders the policy's depth camera from the point cloud, and runs the closed loop on CPU. The policy, its input/output processing, the quadrotor dynamics and the navigation metrics come from [DiffDrone-PyTorch3D](https://github.com/kirkchinese/Diffable_Drone_Pytorch3D), which is installed separately and imported at run time.

On the office-floor reconstruction, a policy trained only in randomly generated obstacle fields reached all 50 goals without collision (table below).

## Installation

```bash
python -m pip install -e '.[core,sim]'             # numba, torch (CPU is enough)
git clone https://github.com/kirkchinese/Diffable_Drone_Pytorch3D.git
wget https://github.com/kirkchinese/Diffable_Drone_Pytorch3D/releases/download/v0.1.0/exp21_grad_clip_goal_best_ar.pth
sha256sum exp21_grad_clip_goal_best_ar.pth
# 66096218d3f508de615a8478c511fe564746f0a39bf7075a406d53257d841174
```

Only `model.py`, `navigation_utils.py` and `drone_dynamics.py` of the checkout are imported; they need PyTorch only (no PyTorch3D). The checkpoint is loaded with `torch.load(weights_only=False)`, so use one whose SHA-256 you have checked.

## Usage

```bash
recon sim prepare <export_dir> scene.npz --ceiling-z 2.95
recon sim pairs scene.npz pairs.npz                       # 50 pairs, seed 20261030
recon sim eval scene.npz pairs.npz eval/ \
  --diffdrone Diffable_Drone_Pytorch3D --checkpoint exp21_grad_clip_goal_best_ar.pth
```

`eval/summary.json` holds the outcome counts and one row per pair; `eval/episodes.npz` holds per-step position, attitude, velocity, clearance, goal distance and collision flags (`--save-depth` adds every 48×64 policy depth frame, about 1.2 MB per episode).

## Definitions

**Scene (`prepare`).** Input is `cloud.ply` of the export (metres, Z up). The floor is the dominant plane among the lowest 30 % of points: RANSAC over 24,000 samples (700 hypotheses, normal within 25° of +Z, 7 cm inlier band), then four PCA refits. If the floor normal deviates more than 3° from +Z the cloud is rotated; it is always shifted so the floor is z = 0. The capture rarely sees the ceiling, so a flat synthetic ceiling is added at `--ceiling-z`: one point per 2 cm cell over the floor footprint (floor points |z| < 0.18 m, dilated by 0.5 m), except in 10 cm cells that already contain real points at or above `ceiling_z − 0.25 m`. Default `--ceiling-z` is the 99.9th height percentile of the levelled cloud, rounded down to 5 cm. The synthetic points are both visible to the depth camera and collision geometry; `observed_count` separates them from the real points.

**Pairs (`pairs`).** A 0.1 m grid at flight height 1.2 m. A cell is known when it lies within 0.6 m (horizontally) of an observed floor point and within 3 m of any point; it is free when it is known and at least 0.6 + 0.1/√2 m from every point. Start and goal cells need 0.9 m clearance, the same 4-connected free component and a straight-line distance of 6–20 m. Proposals are drawn with a fixed seed; until half of the pairs (`--blocked-min`) have a blocked straight line (some point within 0.1 m of the segment, sampled every 2.5 cm), unblocked proposals are rejected. The shortest-path length is the 4-neighbour BFS length on the free grid.

**Episode (`eval`).** Each pair is flown once for 200 steps at 15 Hz (13.3 s), seeded with `seed + index`. The start state follows the randomisation of the DiffDrone-PyTorch3D training environment: uniform initial yaw; safety margin 0.3–0.8 m; pitch-control delay 12 ± 1.2 steps; linear drag 0.30–0.45; a time-correlated acceleration disturbance (Ornstein–Uhlenbeck, initial σ 0.2 m/s², `noise_std` 0.04) and wind (σ 0.1 m/s); speed limit 2.5 m/s. The policy sees a 48×64 depth image with a 79° horizontal field of view (near 0.3 m, far 24 m), rendered by splatting every scene point as a 3 cm disc through the body-mounted camera of the DiffDrone-PyTorch3D quadrotor (pitched up 10°, offset scaled with the margin). As in training, the policy receives the previous frame's depth together with the updated odometry.

**Outcome.** Clearance is the distance from the drone centre to the nearest scene point (real or synthetic). *Collision*: clearance below the margin at any step. *Out of bounds*: clearance above 3 m for at least 1 s. *Success*: the goal is reached within 0.5 m without collision and without leaving the volume. Otherwise *timeout*. The outcome classes are exclusive in the order out of bounds, collision, success, timeout.

## Reference result

Office floor captured handheld with the S380 sensor suite (27 min, 3,470 keyframes), reconstructed with FAST-LIO poses and post-flight BA; 850,318 real points plus 270,760 synthetic ceiling points at 2.95 m. Policy: `exp21_grad_clip_goal_best_ar.pth`, trained only in randomly generated obstacle fields, not retrained for this scene.

| Metric | Value |
| --- | --- |
| Outcome (50 pairs 6.1–10.8 m apart, 27 with a blocked straight line) | 50 success, 0 collision, 0 timeout, 0 out of bounds |
| Mean speed of successful episodes | 0.757 m/s |
| Flown path / grid shortest path (whole episode) | 1.09 |
| Path to first arrival / grid shortest path | 0.90 |
| Wall time, 50 episodes, one CPU thread | about 50 s |

The ratio to first arrival is below 1 because the grid shortest path is a 4-neighbour (Manhattan) length, longer than a free-space diagonal.

Verified: running `prepare --ceiling-z 2.95`, `pairs` and `eval` with default seeds on this export reproduces the original experiment bit for bit (identical scene points, all 50 start/goal pairs and shortest paths, and every per-step position, attitude, velocity and clearance of all 50 episodes).

## Limitations

- Depth comes from point splats, not from the 3DGS model: thin structures with few points and unobserved regions appear as holes or open space, and glass is invisible to both.
- The synthetic ceiling is an assumption about unobserved space. Choose `--ceiling-z` below the lowest real obstruction you want the policy to respect.
- Pairs are sampled on one horizontal plane (1.2 m); vertical manoeuvres are only exercised by obstacles in the way.
- The dynamics model is the simplified differentiable model of DiffDrone-PyTorch3D, not a full rigid-body simulation. Results show that the policy transfers to the geometry of a real space; they are not a substitute for flight tests.

## License

`recon/sim` is Apache-2.0 like the rest of this repository and contains no DiffDrone-PyTorch3D code; the camera-mount constants in `flight.py` are numbers measured from its drone model. DiffDrone-PyTorch3D is GPL-3.0. If you distribute a work that combines the two, the GPL-3.0 terms apply to that combination.

## Evidence index

Plain-text sources of the reference result (local, not part of the repository):

- Data/experiments/promo_20261029/sim_v7/ — original 50/50 run (office_stats_v7.json, episodes_v7.npz)
- Data/experiments/sim_port_20261110/ — bit-for-bit reproduction with `recon sim`
