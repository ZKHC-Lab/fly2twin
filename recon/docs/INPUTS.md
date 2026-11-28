# Input and output contract

2026-10-29

## Image manifest

JSON, with arbitrary increasing integer frame IDs (filenames need not be numbered), strictly increasing nanosecond timestamps and a shared pinhole calibration. Paths are relative to the manifest, or absolute. All images use the same resolution. Supply undistorted RGB for the external pose provider; the `colmap` and `vggt` providers accept raw distorted images and write undistorted PNGs. Image-only input may omit `K`: the provider estimates a shared camera (`vggt` requires that `K` is omitted). Supplying camera/body poses without a camera archive requires `K`.

```json
{
  "resolution": [1280, 720],
  "K": [[900, 0, 640], [0, 900, 360], [0, 0, 1]],
  "frames": [
    {"frame_id": 101, "timestamp_ns": 1000000000, "path": "rgb/frame-a.png"},
    {"frame_id": 105, "timestamp_ns": 1200000000, "path": "rgb/frame-b.png"}
  ]
}
```

This two-frame fragment illustrates the schema, not a sufficient reconstruction scene. Provide a textured, overlapping sequence with translation and viewpoint diversity. Each row may alternatively include a 4×4 `c2w`. Convention: camera right/down/forward, `T_AB` maps B into A. Poses and translations are in meters unless image-only SfM is used; with `pose.metric.camera_height_m`, SfM output is converted to metres under that height assumption.

Config paths are relative to the YAML file. Supply exactly one of `manifest`, `video` or `delivery`, plus `output`. Optional `exclude_frames` lists frame IDs dropped from the prepared manifest before splitting, written to `prepare/excluded_frames.json`; use it to remove a whole bad segment by hand. Single frames the SfM provider cannot register no longer need it: they are dropped automatically when at least `sfm.min_registration_rate` of the frames registered (see below). `delivery` is an onboard delivery package (`drone3d_onboard.export`): frames are its keyframes, K its shared CameraInfo, and poses the landing-BA cameras when the package records a completed onboard BA, otherwise the onboard LIO poses; use `pose.provider: external`. Optional `cameras` is a trusted NPZ with `frame_ids[N]`, `c2w[N,4,4]`, `K[N,3,3]`; exact IDs and order must match the manifest. To fit metric scale from an arbitrary-scale archive, also include `points[M,3]`; `colors[M,3]` is retained when supplied.

For `video`, prepare calls the phone extractor ([extract_frames.py](../video/extract_frames.py)) using the `extract` mapping. It needs `ffmpeg` and `ffprobe` on the path (4.4 or newer; HDR sources also need the `zscale` filter). It decodes every frame and scores sharpness (Laplacian variance of `scan_width`-wide luma), splits the frame range into `target_frames × sfm_multiple` equal bins and keeps the sharpest frame of each bin. Selection does not look at camera motion. If the video has fewer frames than bins, every frame is kept. One ffmpeg pass writes the selected frames as JPEG, **always** scaled to `long_edge` on the long side (HDR is tone-mapped first when detected); no later stage sees the original resolution. Manifest `frame_id` is the source frame number, `timestamp_ns = round(time_s × 1e9)` from the extraction record, and `path` points to that frame. The phone preset uses 400 × 3 = 1200 bins, `train_subset: all`, minimum 200 × 3, long edge 1920, scan width 640, JPEG quality 2, HDR auto: a 4K video becomes 1200 frames of 1920×1080, and SfM, depth, fusion and training all use these same frames. Motion keyframes (`keyframes.enabled`) need poses and cannot be used on video input.

## Configuration reference

The table lists every accepted key; `recon/config.py` holds the same list (`KEYS`). Any other top-level or section key is an error that names the key and the keys its section allows, so a misspelt `fusion.revisit` stops the run instead of silently running without it.

| Key | Default / meaning |
| --- | --- |
| `manifest`, `video` or `delivery`, `output` | Exactly one input source and the output parent directory |
| `exclude_frames` | Optional frame IDs dropped from the prepared manifest before splitting; see above |
| `protocol` | `drone` (default) or `phone`; train/eval loading and scoring protocol |
| `extract.target_frames` | 300; the number of equal timeline bins is `target_frames × sfm_multiple`, one sharpest frame each |
| `extract.min_frames` | 200; error if fewer than `min_frames × sfm_multiple` bins are selected |
| `extract.sfm_multiple` | 3; multiplies the number of bins. With `train_subset: all` SfM and training use the same frames |
| `extract.train_subset` | `all`: keep every selected frame; `subset`: keep every `sfm_multiple`-th. SfM also sees only the kept frames; the extra frames written to `frames_sfm/` are not read by any stage |
| `extract.long_edge` | 1600; long side of extracted JPEGs, px; applied unconditionally (the phone preset uses 1920) |
| `extract.scan_width` | 640; width of the low-resolution sharpness scan |
| `extract.quality` | 2; ffmpeg JPEG `-q:v` |
| `extract.hdr` | `auto`, `on` or `off`; HDR tone mapping on extraction |
| `sfm.camera_model` | `SIMPLE_RADIAL` for `vggt`, `OPENCV` for `colmap`; the shared COLMAP camera model |
| `sfm.focal_factor` | 1.2; initial focal = factor × max(width, height), refined by BA |
| `sfm.overlap` | 20; sequential matching overlap |
| `sfm.loop_keyframes`, `sfm.loop_max_pairs`, `sfm.loop_min_gap` | 60 for `vggt`, 0 for `colmap`; 600; 0 (= 2 × overlap); loop-closure keyframes (0 disables), pair cap and minimum index distance |
| `sfm.max_features` | 8192; SIFT features per frame. The phone preset uses 2048: on three phone videos (2026-11-27, one run each) held-out PSNR was 34.08 against 33.78, 34.23 against 34.21 and 25.57 against 25.94 dB with 8192, the mean reprojection error rose from 0.59–0.80 to 0.66–1.03 px, and the three BA rounds took 1583 s instead of 5477 s and 1900 s instead of 10724 s (both measured on a busy server). Measured again 2026-11-28 on more videos: weak-texture office videos give the same result at 2048, 4096 and 8192 features (28.48 / 28.90 / 28.59 dB and 31.36 / 31.12 / 31.20 dB; frames without BA observations 88 / 94 / 93 and 60 / 61 / 61, because those frames have few features at any cap). On one feature-rich outdoor video 2048 gave 25.57 and 24.99 dB in two runs against 25.94 dB at 4096 and at 8192, with the self-calibrated focal length at 1640 and 1626 px instead of 1645 px: raise the value for such videos when the extra BA time is acceptable (BA rounds 2778 s at 2048, 4118 s at 4096, 10724 s at 8192, on a busy server) |
| `sfm.peak_threshold` | 0 keeps the COLMAP default |
| `sfm.min_model_size` | 10; smallest model the `colmap` mapper keeps |
| `sfm.min_registration_rate` | 0.99; `colmap` only (`vggt` has no registration step): share of the prepared frames the largest model must register. At or above it the unregistered frames are dropped: `pose/report.json` lists them in `dropped_frames`, and the pose stage rewrites the manifest, the train/test split (`pose/selection.json`) and the test subsets (`pose/subsets.json`) without them. Below it the run is rejected. 1 restores the old rule (every frame must register) |
| `sfm.threads` | 8 for `vggt`, 4 for `colmap`; feature, matching, mapping and (for `vggt`) bundle-adjustment threads. With 1 the `vggt` poses are bit-identical from run to run |
| `cameras` | Optional external camera NPZ; sparse points are needed for camera-height scale fitting |
| `body_poses`, `body_from_camera`, `time_offset_ms` | Optional body trajectory, 4×4 extrinsic and RGB query offset; default offset 0 ms |
| `pose.provider` | `external`, `colmap` or `vggt`; inferred as external with supplied poses, otherwise colmap. `vggt` (image-only, uncalibrated input; the phone preset) chains VGGT windows by depth, closes loops and refines with feature BA; see below. The former `glomap` provider was removed on 2026-11-27 |
| `pose.refine` | `none` (default) or `ba`; optional position-prior BA after provider and metric conversion |
| `pose.units` | `meters` (default) or `arbitrary`, only for external poses |
| `pose.metric.camera_height_m` | Optional assumed median camera height to scale arbitrary provider output |
| `pose.metric.speed_mps` | Optional assumed median moving speed to scale arbitrary provider output: straight distance over stretches of 30 frames, median of the stretches that move (slower than a quarter of the 90th percentile counts as standing still); needs no visible floor. Give this or `camera_height_m`, not both |
| `pose.position_sigma_m` | 0.05; Cartesian position prior standard deviation |
| `pose.rotation_sigma_deg` | unset = off (rotations free in the joint BA, plain squared reprojection loss). A value, e.g. 1.0, turns on the method for weakly observed scenes: rotations stay fixed in the joint BA and move only in a per-image step against a prior of this standard deviation, and reprojection uses a Cauchy loss (`recon/poses/ba.py`). It makes the result stable against match noise where frames have few observations (office_gs_01) and costs about 1 dB of held-out 3DGS PSNR on a well-observed scene (office_gs_04) |
| `pose.xfeat_source` | Optional XFeat checkout, relative to the YAML file; added to the import path before `modules.xfeat` is imported, an error if it has no `modules/xfeat.py`. Without it XFeat must already be importable (`PYTHONPATH`) |
| `pose.xfeat_weights`, `pose.batch_size`, `pose.ransac_threads` | Explicit user-owned checkpoint; matching batch size 8; fundamental-matrix RANSAC on a 4-thread pool while the next batch matches (OpenCV 4.x FM_RANSAC seeds itself per call, so output is bit-identical to one thread; 2026-11-03, 1,500 image pairs on GPU: 134 s → 41 s, all saved arrays identical) |
| `keyframes.enabled` | false; motion keyframe thinning, requires poses (not available for `video` input); only the S380 preset enables it |
| `keyframes.translation_m`, `rotation_deg`, `max_gap_s`, `trim_static` | 0.15, 8, 2, true |
| `selection` | Optional exact split JSON; otherwise every eighth selected frame is test |
| `split.mode`, `split.exclude_blur` | `holdout`, true; phone uses false so all non-test frames train |
| `normalization`, `normalization_file` | Optional fixed three-field normalization (`normalization_center`, `meters_per_unit`, `scene_scale`); file takes precedence |
| `normalization_cast_first` | false; drone protocol casts c2w to float32 before subtracting the normalization centre, as the reference drone run did; reproduces that run's training coordinates exactly |
| `depth.weights`, `depth.cache` | Explicit Omega weights, or a precomputed window folder; cache bypasses inference |
| `depth.omega_source` | Repository `third_party/vggt-omega`, located relative to the installed recon package; explicit paths are relative to the YAML file |
| `depth.resolution` | [512,288]; first component sets inference long edge; both components must match actual model/cache output |
| `depth.first_window`, `depth.new_frames`, `depth.overlap` | 40, 30, 10; window planning below |
| `fusion.max_points` | 300000; permitted range 4..1000000; point cap of the delivered cloud |
| `fusion.init_max_points` | `fusion.max_points`; permitted range 4..5000000; point cap of the cloud the Gaussians start from (`train.init_cloud`) |
| `fusion.revisit` | false (the S380 preset sets true); true corrects every training frame's depth scale with surfaces that other windows also saw ([recon_core](../../recon_core/README.md), `revisit.py`) before fusing. The cached metric depth and `scales.npz` then include the correction, `scales.npz` gains `revisit_log_scale` and `report.json` a `revisit` summary. Absent or false reproduces earlier outputs exactly |
| `train.seed`, `train.resolution` | 42, [640,360] |
| `train.steps` | Optional; default max(30000, nearest-1000(frame_count×60)), or frame_count×60 with `exact_steps` |
| `train.exact_steps` | false; true makes the default step count exactly 60 visits per training frame instead of at least 30000 |
| `train.scale_densification` | true; densification start/stop/reset/refine steps scale with steps/30000 |
| `train.schedule_rounding` | `truncate`; `round` rounds the scaled densification steps instead |
| `train.sh_interval` | steps/30; steps between spherical-harmonic degree increases |
| `train.reset_at_zero` | false; true also resets opacity at step 0, as the phone reference did |
| `train.discard_initial_permutation` | false; true draws and discards one frame permutation before training, as the phone reference did |
| `train.max_width` | 1600; phone protocol only: training image width (never above the source width) |
| `train.image_cache_frames` | 32; decoded training/evaluation frames kept in memory. The phone preset keeps all 1200: measured 2026-11-28, a phone run then holds about 10 GB of host memory while training (9.9–10.5 GB for 1920×1080 frames trained at 1600×900, 12 GB for a portrait video trained at 1080×1920); lower this value on a machine with less memory, at the cost of decoding frames again |
| `train.init_cloud` | `footprint`; which fused cloud the Gaussians start from. `footprint` is `fuse/gaussian_init.npz`: pixels above the 20th confidence percentile of their frame, no depth range, the same multiview check as the delivered cloud, one point per cell the size of the pixel footprint at the point's depth (cells grow together until the cloud fits `fusion.init_max_points`). `delivered` starts from `fuse/initial.npz`, the cloud that is exported (above the confidence median, 0.3–4 m, 2 cm voxels), as before 2026-11-25 |
| `train.loader_workers` | 4; threads that decode the images and resize the depth priors of the next 64 training steps ahead of the loop. The inputs and their order are unchanged; 0 loads inside the loop |
| `train.depth_lambda`, `train.aniso_lambda` | 0.1, 0.1; the reference drone run's values |
| `train.depth_loss` | `log` (default): mean absolute difference of log depth between the render and the fused prior, per frame. `pearson`: 1 − Pearson correlation of inverse depth, the default until 2026-11-27. `pearson` ignores the overall scale of a frame, which let the first frames of a forward walk collapse onto a shrunken copy of the scene (three phone walks, held-out PSNR of those frames 15–19 dB against 22–27 dB with `log`); `log` needs the prior and the poses in one scale, which the fusion stage's window scales provide. Measured 2026-11-26/27, one training each, `pearson` → `log`: drone scans 26.59 → 26.63 and 26.63 → 26.75 dB (a rerun of `pearson` gave 26.53 and 26.39), phone walks 28.19 → 28.85, 21.73 → 23.00 and 24.70 → 25.28 dB, the lounge reference 33.80 → 33.75 dB; floaters did not increase |
| `train.pose_opt` | false; per training camera a right-composed SE(3) residual (translation + continuous 6D rotation), identity at start; test cameras are not optimised |
| `train.pose_lr`, `train.pose_reg` | 1e-5, 1e-6; Adam learning rate (no decay) and weight of the squared 9-value residual of the current camera; translation in normalised scene units |
| `eval.alignment_steps` | 200; use 200 for comparison with the reported protocol |
| `eval.aligned_width` | 960; phone protocol only: image width of the colour+pose fitted evaluation |
| `threads` | 4 for training/evaluation CPU image operations |
| `eval.omega_dir` | Optional phone-only external `depth.npy`, `conf.npy`, `frame_ids.npy` prior; otherwise read raw depth-window new owners |
| `dependency_revision` | Optional free-form label; no code reads it, but the whole configuration is part of the run fingerprint, so changing it starts a fresh output directory. Use it when something outside the hashed inputs changes, e.g. the installed gsplat version |
| `export.reference_drone` | false; true reproduces the reference drone run's export: unit quaternions in `gaussians.ply`, and `preview.splat` colour and opacity truncated rather than rounded |
| `export.name`, `export.subtitle` | Optional scene title and subtitle in `scene.json` |

With `train.pose_opt`, training also writes `optimized_poses.npz` (training `frame_ids` and world `c2w`, metric input stays metric) and stores the residuals with their frame IDs under `pose` in `final.pt`; when off, nothing is created and `pose` stays `None`.

Pose optimisation does not replace landing BA (office_gs_04, 2935 training / 433 interpolation test frames, 30000 steps, BA test cameras; PSNR fixed → after the 200-step test pose fit): BA cameras 23.82 → 24.51 dB; LIO cameras 21.34 → 22.96; LIO + pose_opt at lr 1e-5 / reg 1e-6 21.48 → 24.13, i.e. it recovers about three quarters of the fitted-pose gap but stays 0.38 dB below BA; at lr 1e-4 / reg 1e-4 both LIO (18.75 → 21.79) and BA (18.74 → 21.88) cameras degrade badly. The LIO arms reused the BA-pose fused initial cloud and depth cache, which favours them. The optimised LIO rotations moved 0.17° median and stayed 0.64° from BA (0.72° before), so the residual mostly absorbs inconsistencies rather than recovering the BA cameras.

YAML values under `pose`, `depth`, `train`, etc. are nested mappings. No model checkpoint, GPU lock, toolchain path or local dataset is defined in package source. Published example paths are placeholders and must be replaced; examples do not claim to contain a redistributable dataset.

Blur is evaluated on the candidate sequence **before** motion keyframe thinning, then indexed by selected frames. A frozen split never changes based on this diagnostic blur calculation. Prefer uniformly sampled candidate timestamps so a radius of 10 candidates has a predictable temporal extent.

## Pose provider, metric conversion and refinement

The stage order is **provider → metric scale (when arbitrary) → optional prior BA**. `external` imports the camera archive, manifest c2w or interpolated body poses. `colmap` ([sfm.py](../sfm.py)) runs SIFT on one shared camera, sequential matching, an optional loop-closure subset, then COLMAP's incremental mapping. `vggt` uses the same feature, matching and undistortion steps; the `sfm` keys apply to both, with the provider defaults in the table above. Both export sparse points/colors and undistorted PNGs with the undistorted `new_K`, output arbitrary scale and update `pose/manifest.json`. `colmap` drops unregistered frames only above `sfm.min_registration_rate` and rejects the run below it; `pose/report.json` lists `unregistered_frames`.

`pose.refine: ba` uses the provider output as Cartesian position priors (default σ = 0.05 m), with the existing SIFT/XFeat and prior-BA path, after any metric conversion. Provider-internal BA does not disable this independent refinement slot. The wiring of `colmap` or `vggt` followed by this outer BA is covered with mocked CPU inputs; **real-scene refinement for these combinations is unverified**. For arbitrary-scale providers, supply `pose.metric` before interpreting the prior σ in metres.

`pose.provider: vggt` ([vggt.py](../poses/vggt.py)) needs no structure-from-motion solve. The pose stage runs VGGT-Omega on the prepared (raw) frames with the `depth` section's window layout, and SIFT features with the `sfm` section's matching options (sequential plus loop pairs). Windows are chained through the frames they share: rotation from the shared cameras' orientations, scale from the per-pixel ratio of the two depth maps of the same frames, translation from the camera centres. Verified matches between frames of different windows (at least 16 frames apart) are lifted to 3D with each frame's own VGGT depth and intrinsics. A pair is used when its two point sets fit each other (median residual under 5% of depth) and the chain already puts them within 3% of the path between the two frames plus 5% of depth, which keeps matches between look-alike places out. One similarity correction per window (the first fixed) then makes the points agree as seen from both cameras (direction and log depth, Cauchy loss, each pair weighted as about ten residuals), while consecutive windows keep the chain's relative placement (2% in scale, 0.5°, 2% of depth in position); pairs that still disagree are dropped and it is solved again, and the whole step runs a second time on the corrected chain so that pairs a drifted chain had put too far apart can enter. A feature BA follows, with the landing BA's rounds (16, 8, 4 px), a position prior of 0.3% of the path length and `sfm.threads` threads; the shared `SIMPLE_RADIAL` camera is self-calibrated starting from VGGT's median focal length and zero distortion. Frames are then undistorted as for `colmap`, and the depth stage infers again on the undistorted frames. Every prepared frame keeps a pose (`registration_rate` is 1); frames the BA could not observe keep the tracked pose and are listed in `frames_without_observations`. Frames with only a few observations do take the BA's pose, which is then poorly constrained: measured 2026-11-28, a held-out frame with 5 observations rendered at 8.9 dB in a video whose other held-out frames average 28.6 dB, and about a tenth of the frames of a weak-texture office video have fewer than 10 observations (`pose/track.npz` holds the count per frame). `pose/track.npz` holds the chained and loop-closed poses; `pose/report.json` the calibration and its ratio to VGGT's focal length, the BA rounds, how far the BA turned and moved the cameras, the per-window corrections and the match-pair counts (`track.loop_warning` when no pair passed the checks). Units are arbitrary until `pose.metric` sets them. Calibrated input (a manifest with `K`) is not supported by this provider yet. The pose stage is slow where frames have many features: with 8192 features per frame it took 7–60 minutes on indoor videos and about 4 hours on a 95 s outdoor video (1200 frames, median 4162 BA observations per frame), nearly all of it in the BA; with the phone preset's 2048 features that outdoor video took 56 minutes.

The phone preset sets the scale with `pose.metric.speed_mps: 1.0`: an assumed walking speed, not a measurement (`speed_scale` in [metric.py](../poses/metric.py); `pose/report.json` records `measured: false`). Measured 2026-11-25 on one office walk: halving or doubling the unit changed held-out PSNR by less than 0.25 dB (28.03, 28.19, 28.26 dB). The alternative adapter `camera_height_m` assumes a median camera height instead. It takes median camera centre, trajectory P95 radius `r`, and normalized median `-camera_y` as an up prior. Candidate points lie more than `0.1r` below the centre. Seed 42 samples up to 30000 candidates and performs 3000 three-point RANSAC trials: degenerate cross-product norm <1e-9 is skipped; normals must be within 25° of up, at least 95% of cameras above the plane, median camera height ≥`0.2r`; support is residual <`0.01r`. It chooses maximum support, refits the plane three times with PCA, then returns `camera_height_m / median_camera_height_sfm`. Translation and sparse points are multiplied in their original dtype. `pose/report.json` stores the complete `scale_assumption` diagnostic (plane, prior, thresholds, support, height percentiles, scale and `measured: false`); units become metres based on the assumption. No world-axis rotation is applied. The float32 PCA products depend on the BLAS thread count in the last digits of the scale; no thread setting is forced inside the adapter.

## Body poses, calibration, timestamps

Instead of camera poses, provide `body_poses` NPZ with `timestamp_ns`, `position[N,3]`, `quaternion_xyzw[N,4]`; `body_from_camera` is the 4×4 `T_BC`. Query body poses at `RGB timestamp + time_offset_ms`. Translation uses linear interpolation, rotation uses SLERP. Out-of-range queries are flagged and pipeline preparation rejects them; no silent extrapolation.

Consecutive identical position **and rotation** samples are deduplicated. Pure rotations remain valid samples.

The S380 example uses translation `[0.056558,-0.006717,-0.077751]` m and normalized quaternion xyzw `[-0.494176,0.506142,-0.498990,0.500618]`, with +40 ms query offset. It is a measured calibration for one assembly, not a universal default or flight-validated calibration.

## Splits and normalization

Without a frozen split, every eighth selected frame (`7::8`) is a test frame; `split.exclude_blur: true` excludes blurred frames from training but not BA. The phone preset sets it false: with 1200 frames, training indices are `index % 8 != 7` (1050 frames) and test indices are `index % 8 == 7` (150 frames). Optional `selection` JSON specifies `train_indices`, `test_indices`, `train_frame_ids` and optional `holdout_frame_ids`/`interpolation_frame_ids`. Indices refer to the prepared manifest. Frozen splits are authoritative and never reselected using newly computed blur scores. All test-frame IDs must match the union of evaluation subsets.

For the drone protocol, optional `normalization` mapping or `normalization_file` JSON provides `normalization_center[3]`, `meters_per_unit` and `scene_scale`. New scenes default to mean training-camera center, maximum training radius, and 1.1 times the maximum normalized camera distance from the all-camera mean. Training coordinate is `(world-center)/meters_per_unit`; export inverts it. Reproduction may freeze these three fields rather than recompute them after BA.

## Phone preset training and evaluation

[phone_video.yaml](../../configs/phone_video.yaml) applies the drone method to phone video: 1200 frames at 1920×1080 (see `video` above), poses from the `vggt` provider converted to metres with an assumed walking speed of 1 m/s, 512×288 Omega, 40/30/10 windows, unchanged drone scale estimation and fusion (0.3–4 m depth range, 1.5 m neighbour radius, in the assumed metres) with a 1000000-point cap, no revisit correction, no outer BA, then the shared trainer initialised from the Gaussian-initialization cloud (`train.init_cloud`) with the log-depth loss λ=0.1 (`train.depth_loss`) and anisotropy λ=0.1. Training uses seed 42, exact N×60 steps, scaled densification with round scheduling, initial permutation discarded, and reset at zero. All seven stages run in order; `--only S` requires every earlier stage to be complete. The former RGB-only phone setup (SfM sparse-point initialisation, no depth) has had no entry point since commit e059b7b and cannot be run with the current code; `train.depth_lambda: 0` only removes the depth loss.

`protocol: phone` uses the shared phone normalization: median training centre, radius guarded by `max(median+8*MAD,1.5*median)` (zero MAD replaced by half the median), P95 radius over retained centres; drop guarded outliers and registered cameras beyond five P95 radii. c2w is cast to float32 before normalization; K is scaled proportionally (no pixel-centre offset). Training images are resized to width `train.max_width` (1600; never above the source width), so 1920×1080 frames train at 1600×900. The frame cache is bounded by `train.image_cache_frames` (preset 1200). Initialization comes from `fuse/gaussian_init.npz` (`fuse/initial.npz` with `train.init_cloud: delivered`), never the sparse cloud, and depth loss reads `fuse/cache`. Phone evaluation and export keep reading the delivered cloud `fuse/initial.npz`. Phone train writes its normalization sidecar. Phone export inverts this same normalization; `scene.json` then states `units: meters`, which are the assumed metres (`pose/report.json` records `measured: false`).

Phone eval uses the shared phone evaluator: fixed-pose full-frame 1600-wide RGB scores; 960-wide per-test-frame colour+SE3 alignment (200 Adam steps, learning rate 1e-4); and floater fraction with confidence percentile 50 and threshold 0.7 times linearly aligned prior depth. Default prior reads each `windows.json` new-owner raw depth/confidence, including test frames. Optional `eval.omega_dir` instead reads external depth/conf/frame_ids arrays. Neither source uses the train-only zero-filled fusion test slots.

Reference result (2026-11-27, the 4K lounge reference video run from the video file through all seven stages on a server): all 1200 frames have a pose and were observed by the BA; self-calibrated focal length 1422.2 px and k1 = 0.0138 (VGGT's focal length: 1437 px), mean reprojection error 0.69 px, 186,159 sparse points; fixed-pose 1600-wide test PSNR 34.13 dB, SSIM 0.942, LPIPS 0.095 on 150 test frames (33.82 dB with `sfm.max_features: 8192`). With the removed GLOMAP provider the same video gave 32.71 dB (2026-11-24).

## Depth cache and windows

Depth inference has one path: the `reconstruct_omega` module in `depth.omega_source`, defaulting to repository `third_party/vggt-omega` resolved relative to the recon package. Set this directory explicitly for an installation outside the source checkout. The CLI resolves explicit paths relative to the YAML file. `depth.weights` is an explicit local checkpoint. The adapter calls `load_model(weights, 'cuda', aggregator_frame_chunk_size=10)`, `load_and_preprocess_images(paths, mode='max_size', image_resolution=depth.resolution[0])`, then `run_inference(model, images, 'cuda')`. There is no crop or DA3 alternative. Default long edge is 512; the 16:9 input used here produces 512×288. `depth.resolution` must match the actual output grid for fusion.

`plan_windows(ids, first=40, new=30, overlap=10)` runs in manifest order over all frames, test frames included:

- `K000` contains the first `depth.first_window` frames (default 40), all new.
- Each later window takes the next `depth.new_frames` frames (default 30) and `max(depth.overlap, depth.new_frames - number_of_new_frames)` preceding frames. Default full windows therefore contain 10 shared + 30 new frames.
- The tail expands shared context to reach at least 30 frames when enough preceding frames exist; a sequence shorter than the first window remains one short window. IDs are `K%03d`.
- For 1200 consecutive IDs starting at zero, windows are `[0,40)`, `[30,70)`, …, `[1170,1200)`; the last window owns new IDs 1180–1199. New ownership covers every input ID exactly once.

Optional `depth.cache` imports existing predictions without running a model. Folder schema:

```text
windows.json
K000/prediction.npz
K001/prediction.npz
```

`windows.json` contains `windows`, a list of `{id, idx, new}`. `idx` and `new` contain manifest frame IDs; every frame must have exactly one `new` owner. Each NPZ is written with `numpy.savez_compressed` and retains **all arrays** returned by `run_inference`, plus `frame_ids[N]` matching `idx`. Current returned arrays are `depth[N,H,W,1]`, `depth_conf[N,H,W]`, `extrinsic[N,3,4]` (world-to-camera in model units), and `intrinsic[N,3,3]`. No synthetic `valid` array is added. `load_windows` reads depth, confidence and extrinsics with `allow_pickle=False`; legacy cached window IDs remain readable.

### Downstream use

Before scale estimation, windows are restricted to training frames. Scale anchors use trajectory extent >0.3 m and positive fitted scale; shared-depth edges need at least three shared frames and ≥500 high-confidence pixels per qualifying frame. Disconnected/unanchored scale systems raise errors. Offline solves one scale per window (`batch_window_scales`); per-frame scales are only solved onboard. With `fusion.revisit: true` a per-frame correction is applied afterwards. Empty test depth slots are zero and invalid.

Fusion uses confidence above frame median, 3×3 relative depth span ≤3%, z in (0.3,4) m (fixed in recon_core), stride 2; depth support is `abs(log(observed/projected))<0.03` with at least `min(2,max(seen,1))` confirmations. Partners are the undirected union of six ranked neighbors within 1.5 m and full-rotation angle <30°. Shared recon_core NumPy voxel downsampling begins at 2 cm and is enlarged until the configured cap (default 300k points); NumPy implements PLY I/O independently.

## Image-only limitations

COLMAP sequential overlap 20, quadratic overlap enabled, no loop detection, shared OPENCV camera; intrinsics/distortion refine during BA. Reports include counts for every model, largest-model registration rate and number of additional disconnected models. Registration below `sfm.min_registration_rate` (0.99) is **rejected**; at or above it the unregistered frames are dropped and recorded. The registered frames are undistorted before depth/training. Without an external scale, exported units are arbitrary. Metric depth thresholds are then thresholds in those arbitrary units; tune/validate scale before interpreting physical geometry.

## Archive coordinates and sidecars

Keep the full fingerprint directory when archiving a run: the export stage references the checkpoint in the train stage. The prepare manifest retains source image resolution and paths; the train configuration records normalization. Exported camera arrays are world-coordinate c2w with original-resolution intrinsics. Evaluation `before`/`after` camera snapshots are in normalized training coordinates; use the train configuration's center and meters_per_unit when converting them to world units. The colored cloud and Gaussian PLY are already in world units.

## Display contract (scene schema v2)

New exports prefer `preferred_model.path = gaussians.ply` and record the actual SH degree. `preview_model` identifies the DC-only SPLAT and warns that view-dependent color is lost. Full SH PLY is mandatory; a preview-only result warns and fails completion.

`capture_cameras` contains every frame's K, original `[width,height]`, and horizontal/vertical FOV computed from the calibrated sensor edges. `capture_corridor` records centers, c2w rotations, forward axes and training IDs. Both navigation bounds must hold relative to the same training camera: ≤0.15 meters (or arbitrary exported scene units) and ≤15° full rotation. The normalized distance bound is also recorded. These are quality recommendations for consumers; the repository viewer (`viewer/`) enforces them only in guided mode.

The export directory is the contract for every downstream tool (viewer, shot rendering, evaluation, simulation); they read it through `recon.io.scene.load_scene(export_dir, width=None)`, which returns the Gaussians (metric, log scales, logit opacities, wxyz, SH as `[N,C,3]`), training and held-out capture cameras with intrinsics rescaled to `width`, `world_up`, `bounds` and the fused cloud. Rendering the loaded Gaussians reproduces the normalized training checkpoint to about 100 dB PSNR (office_gs_04, four training views, 640×360).

Top-level `world_up` is the source-world up direction: `[0,0,1]` for drone LIO scenes; for image-only SfM scenes scaled by an assumed camera height it is the fitted floor normal (`pose/report.json` `scale_assumption.plane_normal`), and viewers rotate it onto their vertical. Top-level `bounds` (`min`/`max`, 0.5/99.5 percentiles of the fused cloud per axis) is the reconstruction extent for free navigation. Optional `name`/`subtitle` come from config `export.name`/`export.subtitle`.

World coordinates are right-handed, with wxyz Gaussian quaternions; camera-local axes remain optical right/down/forward. Optional manifest `world_up` defaults to `[0,0,1]`. An explicitly different up axis is rejected on export: align the complete scene before training, or apply a proper joint scene-graph transform and inverse-rotate SH lookup directions. Do not relabel raw SfM coordinates as gravity-aligned, and never change positions without Gaussian orientations and cameras. See [display pitfalls and CPU helpers](PITFALLS.md). Existing schema-v1 exports are not retroactively modified.

## Resource and cache behavior

No resource limits or environment mutation occur at module import. Resource management belongs to the caller. CUDA operations (XFeat, Omega, train/eval) should be run under a shared GPU lock if the GPU is shared. CPU fusion also benefits from limiting BLAS threads. Package fingerprints hash input file contents, not mtimes, plus the syntax trees (`ast.dump`, without line numbers) of the recon and recon_core Python source (excluding recon_core/tests): reformatting or editing comments keeps the cache, a code or docstring change invalidates it. The tree format differs between Python minor versions, so running under another Python version also starts a fresh fingerprint directory. A stage marker hashes all its files; corrupt artifacts are rejected rather than silently used.

The `bag` dependency group only installs `rosbags`; `recon` has no bag importer (bags enter through an onboard delivery package or `ROS_Compatibility_layer/tools/bag`).
