# Input and output contract

2026-10-29

## Image manifest

JSON, with arbitrary increasing integer frame IDs (filenames need not be numbered), strictly increasing nanosecond timestamps and a shared pinhole calibration. Paths are relative to the manifest, or absolute. All images use the same resolution. Supply undistorted RGB for the external pose provider; GLOMAP and COLMAP providers accept raw distorted images and write undistorted PNGs. Image-only input may omit `K`: SfM estimates a shared camera. Supplying camera/body poses without a camera archive requires `K`.

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

This two-frame fragment illustrates the schema, not a sufficient reconstruction scene. Provide a textured, overlapping sequence with translation and viewpoint diversity. Each row may alternatively include a 4×4 `c2w`. Convention: camera right/down/forward, `T_AB` maps B into A. Poses and translations are in meters unless image-only SfM is used.

Config paths are relative to the YAML file. Supply exactly one of `manifest`, `video` or `delivery`, plus `output`. Optional `exclude_frames` lists frame IDs dropped from the prepared manifest before splitting, written to `prepare/excluded_frames.json`; use it for frames the SfM provider cannot register (the providers otherwise reject incomplete registration). `delivery` is an onboard delivery package (`drone3d_onboard.export`): frames are its keyframes, K its shared CameraInfo, and poses the landing-BA cameras when the package records a completed onboard BA, otherwise the onboard LIO poses; use `pose.provider: external`. Optional `cameras` is a trusted NPZ with `frame_ids[N]`, `c2w[N,4,4]`, `K[N,3,3]`; exact IDs and order must match the manifest. To fit metric scale from an arbitrary-scale archive, also include `points[M,3]`; `colors[M,3]` is retained when supplied.

For `video`, prepare calls the shared phone extractor using the `extract` mapping. It scans low-resolution luma, chooses the sharpest source frame in each uniform timeline bin, then extracts selected JPEGs with the configured HDR handling. Manifest `frame_id` is the source frame number, `timestamp_ns = round(time_s × 1e9)` from the extraction record, and `path` points to that frame. The phone preset uses 400 target bins × SfM multiple 3, `train_subset: all`, minimum 200 × 3, long edge 1920, scan width 640, JPEG quality 2, HDR auto: 1200 selected frames on the reference video.

## Configuration reference

| Key | Default / meaning |
| --- | --- |
| `manifest` or `video`, `output` | Exactly one input source and the output parent directory |
| `protocol` | `drone` (default) or `phone`; train/eval loading and scoring protocol |
| `extract`, `sfm` | Shared video extraction and GLOMAP settings, as in the phone preset |
| `cameras` | Optional external camera NPZ; sparse points are needed for camera-height scale fitting |
| `body_poses`, `body_from_camera`, `time_offset_ms` | Optional body trajectory, 4×4 extrinsic and RGB query offset; default offset 0 ms |
| `pose.provider` | `external`, `glomap` or `colmap`; inferred as external with supplied poses, otherwise colmap |
| `pose.refine` | `none` (default) or `ba`; optional position-prior BA after provider and metric conversion |
| `pose.units` | `meters` (default) or `arbitrary`, only for external poses |
| `pose.metric.camera_height_m` | Optional assumed median camera height to scale arbitrary provider output |
| `pose.position_sigma_m` | 0.05; Cartesian position prior standard deviation |
| `pose.xfeat_weights`, `pose.batch_size` | Explicit user-owned checkpoint; matching batch size 8 |
| `keyframes.enabled` | False for already-selected images; enable only on candidate sequences |
| `keyframes.translation_m`, `rotation_deg`, `max_gap_s`, `trim_static` | 0.15, 8, 2, true |
| `selection` | Optional exact split JSON; otherwise every eighth selected frame is test |
| `split.mode`, `split.exclude_blur` | `holdout`, true; phone uses false so all non-test frames train |
| `normalization`, `normalization_file` | Optional fixed three-field normalization; file takes precedence |
| `depth.weights`, `depth.cache` | Explicit Omega weights, or a precomputed window folder; cache bypasses inference |
| `depth.omega_source` | Repository `third_party/vggt-omega`, located relative to the installed recon package; explicit paths are relative to the YAML file |
| `depth.resolution` | [512,288]; first component sets inference long edge; both components must match actual model/cache output |
| `depth.first_window`, `depth.new_frames`, `depth.overlap` | 40, 30, 10; V1.0 window planning below |
| `fusion.max_points` | 300000; permitted range 4..1000000 |
| `train.seed`, `train.resolution` | 42, [640,360] |
| `train.steps` | Optional; default max(30000, nearest-1000(frame_count×60)) |
| `train.depth_lambda`, `train.aniso_lambda` | 0.1, 0.1; BEST settings |
| `eval.alignment_steps` | 200; use 200 for comparison with the reported protocol |
| `threads` | 4 for training/evaluation CPU image operations |
| `eval.omega_dir` | Optional phone-only historical `depth.npy`, `conf.npy`, `frame_ids.npy` prior; otherwise read raw depth-window new owners |
| `dependency_revision` | Optional environment identifier included in the content fingerprint; change when external kernels change |

YAML values under `pose`, `depth`, `train`, etc. are nested mappings. No model checkpoint, GPU lock, toolchain path or local dataset is defined in package source. Published example paths are placeholders and must be replaced; examples do not claim to contain a redistributable dataset.

Blur is evaluated on the candidate sequence **before** motion keyframe thinning, then indexed by selected frames. A frozen split never changes based on this diagnostic blur calculation. Prefer uniformly sampled candidate timestamps so a radius of 10 candidates has a predictable temporal extent.

## Pose provider, metric conversion and refinement

The stage order is **provider → metric scale (when arbitrary) → optional prior BA**. `external` imports the camera archive, manifest c2w or interpolated body poses. `glomap` runs global SfM through the shared phone implementation and `sfm` mapping, exporting sparse points/colors and undistorted PNGs with the undistorted `new_K`. `colmap` retains the incremental shared-OPENCV implementation. Both SfM providers output arbitrary scale, update `pose/manifest.json`, and reject incomplete registration rather than silently dropping split frames.

`pose.refine: ba` uses the provider output as Cartesian position priors (default σ = 0.05 m), with the existing SIFT/XFeat and prior-BA path, after any metric conversion. Provider-internal BA does not disable this independent refinement slot. GLOMAP+outer-BA and COLMAP+outer-BA wiring is covered with mocked CPU inputs; **real-scene refinement for these combinations is unverified**. For arbitrary-scale providers, supply the metric camera height before interpreting the prior σ in metres. Former cached/prior/SfM configuration values are replaced by external+none / external+ba / colmap+none; the old mode key is rejected, not translated.

The B0 metric adapter in [metric.py](../poses/metric.py) assumes a median camera height; it does not measure it. It takes median camera centre, trajectory P95 radius `r`, and normalized median `-camera_y` as an up prior. Candidate points lie more than `0.1r` below the centre. Seed 42 samples up to 30000 candidates and performs 3000 three-point RANSAC trials: degenerate cross-product norm <1e-9 is skipped; normals must be within 25° of up, at least 95% of cameras above the plane, median camera height ≥`0.2r`; support is residual <`0.01r`. It chooses maximum support, refits the plane three times with PCA, then returns `camera_height_m / median_camera_height_sfm`. Translation and sparse points are multiplied in their original dtype. `pose/report.json` stores the complete `scale_assumption` diagnostic (plane, prior, thresholds, support, height percentiles, scale and `measured: false`); units become metres based on the assumption. No world-axis rotation is applied. Exact B0 scale reproduction uses NumPy 1.26.4/OpenBLAS 0.3.23.dev with `OPENBLAS_NUM_THREADS=4`, `OMP_NUM_THREADS=4` supplied by the caller. The unchanged float32 PCA products are thread-sensitive: one thread produced scale 0.6891526456574119 versus reference 0.6891526265461935; 2/4/8/16 threads reproduced every scale diagnostic field exactly. No thread setting is forced inside the adapter.

## Body poses, calibration, timestamps

Instead of camera poses, provide `body_poses` NPZ with `timestamp_ns`, `position[N,3]`, `quaternion_xyzw[N,4]`; `body_from_camera` is the 4×4 `T_BC`. Query body poses at `RGB timestamp + time_offset_ms`. Translation uses linear interpolation, rotation uses SLERP. Out-of-range queries are flagged and pipeline preparation rejects them; no silent extrapolation.

Consecutive identical position **and rotation** samples are deduplicated. Unlike the historical position-only rule, pure rotations remain valid samples.

The S380 example uses translation `[0.056558,-0.006717,-0.077751]` m and normalized quaternion xyzw `[-0.494176,0.506142,-0.498990,0.500618]`, with +40 ms query offset. It is a measured calibration for one assembly, not a universal default or flight-validated calibration.

## Splits and normalization

Without a frozen split, every eighth selected frame (`7::8`) is a test frame; `split.exclude_blur: true` excludes blurred frames from training but not BA. The phone preset sets it false: with 1200 frames, training indices are `index % 8 != 7` (1050 frames) and test indices are `index % 8 == 7` (150 frames). Optional `selection` JSON specifies `train_indices`, `test_indices`, `train_frame_ids` and optional `holdout_frame_ids`/`interpolation_frame_ids`. Indices refer to the prepared manifest. Frozen splits are authoritative and never reselected using newly computed blur scores. All test-frame IDs must match the union of evaluation subsets.

For the drone protocol, optional `normalization` mapping or `normalization_file` JSON provides `normalization_center[3]`, `meters_per_unit` and `scene_scale`. New scenes default to mean training-camera center, maximum training radius, and 1.1 times the maximum normalized camera distance from the all-camera mean. Training coordinate is `(world-center)/meters_per_unit`; export inverts it. Reproduction may freeze these three fields rather than recompute them after BA.

## Phone B0 training and evaluation

[phone_video.yaml](../../configs/phone_video.yaml) is B0, not the historical RGB-only experiment: GLOMAP + assumed camera height 1.4 m, 512×288 Omega, 40/30/10 windows, unchanged drone fusion with a 1000000-point cap, then the shared trainer with inverse-depth Pearson loss λ=0.1 and anisotropy λ=0.1. Training uses seed 42, exact N×60 steps, scaled densification with round scheduling, initial permutation discarded, and reset at zero, matching the formal B0 settings.

`protocol: phone` uses the shared phone normalization: median training centre, radius guarded by `max(median+8*MAD,1.5*median)` (zero MAD replaced by half the median), P95 radius over retained centres; drop guarded outliers and registered cameras beyond five P95 radii. c2w is cast to float32 before normalization; K is scaled proportionally (no pixel-centre offset). The frame cache is bounded by `train.image_cache_frames` (preset 1200); `train.max_width` is 1600. Initialization comes from `fuse/initial.npz`, never the sparse cloud, and depth loss reads `fuse/cache`. Phone train writes its normalization sidecar. Phone export inverts this same normalization.

Phone eval uses the shared phone evaluator: fixed-pose full-frame 1600-wide RGB scores; 960-wide per-test-frame colour+SE3 alignment (200 Adam steps, learning rate 1e-4); and floater fraction with confidence percentile 50 and threshold 0.7 times linearly aligned prior depth. Default prior reads each `windows.json` new-owner raw depth/confidence, including test frames. Optional `eval.omega_dir` instead reads historical depth/conf/frame_ids arrays. The B0 formal prior had an additional intrinsic ray remap and sparse-track scale; the new default is **not claimed numerically equivalent to that historical floater result**. Neither source uses the train-only zero-filled fusion test slots.

The historical B0 formal result was 32.616 dB fixed-pose PSNR at 1600 width and 1848645 Gaussians. Those are reference measurements, not a GPU rerun of these staged changes. `--only train` and `--only eval` input wiring is tested on synthetic CPU data with GPU training/evaluation mocked; staged GPU quality remains to be verified. Historical E2/E3 RGB-only training settings now live inside [the equivalence runner](../../scripts/equivalence.py), not the phone preset.

### B0 staged CPU acceptance (2026-10-29)

With the cached E3 GLOMAP cameras/undistorted images, B0 window predictions converted float16→float32, assumed height 1.4 m and the four-thread BLAS environment above, `recon run --through fuse` reproduced these B0 artifacts bit for bit. Bitwise equality means same dtype/shape/array bytes, with zero tolerance; cloud rows were lexicographically sorted by XYZ before comparing points and associated colours.

| Compared artifact | Size / value | Result |
| --- | --- | --- |
| Metric scale and complete plane diagnostic | 0.6891526265461935 SfM units→assumed m; `measured=false` | Exact |
| c2w / sparse points | 1200 cameras / 466154 points | Bitwise identical |
| Train / test indices | 1050 / 150 | Bitwise identical |
| Window / frame scales | 40 / 1050 | Bitwise identical |
| depth_m / good | 1200×288×512, float32 / bool | Bitwise identical |
| Sorted initialization points / colours | 934231×3 each, float32 | Bitwise identical |
| points_at_2cm / points / voxel_m | 934231 / 934231 / 0.02 m | Exact |

Real-video prepare decoded 2590 frames and selected 1200; extracted source IDs and manifest IDs exactly matched E3, and timestamps exactly matched `round(extract.time_s × 1e9)`. The staged phone loader also exactly matched the formal B0 loader's normalized c2w/K, train/test/registered indices, centre/P95/scene scale and normalization diagnostic, sorted normalized points/colours, output dimensions and one decoded/resized sample image. This acceptance ran no SfM and no GPU training/evaluation. Fresh GLOMAP in the staged adapter, GLOMAP/COLMAP plus outer prior BA, and full staged GPU quality are **not verified on real input**; the provider adapters and train/eval assembly have synthetic CPU tests.

## Depth cache and windows

Depth inference has one path: the `reconstruct_omega` module in `depth.omega_source`, defaulting to repository `third_party/vggt-omega` resolved relative to the recon package. Set this directory explicitly for an installation outside the source checkout. The CLI resolves explicit paths relative to the YAML file. `depth.weights` is an explicit local checkpoint. The adapter calls `load_model(weights, 'cuda', aggregator_frame_chunk_size=10)`, `load_and_preprocess_images(paths, mode='max_size', image_resolution=depth.resolution[0])`, then `run_inference(model, images, 'cuda')`. There is no crop or DA3 alternative. Default long edge is 512; the 16:9 input used here produces 512×288. `depth.resolution` must match the actual output grid for fusion.

`plan_windows(ids, first=40, new=30, overlap=10)` follows V1.0 in manifest order:

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

### Depth equivalence acceptance (2026-10-29)

**Verified on CPU:** the full T17 keyframe sequence produces exactly the reference `{id, idx, new}` list; the independent 1200-frame case and expanded-context tail pass. The depth adapter tests check shared loader arguments, compression, preservation of extra returned arrays, configuration path resolution and `load_windows` compatibility. The depth test file passed 6 tests; the complete CPU suite passed 60 tests.

**Verified on GPU:** rerunning T17 K000/K001 (40 frames each, 512×288) through the new adapter preserves `depth` and `depth_conf` bit for bit. Camera arrays have the small nonzero differences below; whole-prediction bitwise equivalence is **not** established. Measurements used NVIDIA GeForce RTX 5060 Laptop GPU, PyTorch 2.7.1+cu128, CUDA 12.8, frame chunk 10 and `max_size` preprocessing. Shared inference source SHA-256: `6f541f51ff5cc94585419860e507c004492f607fdb77aa30851cb6c843d90da0`.

Errors are computed in float64 against the stored T17 float32 reference: absolute error `abs(new-reference)`, relative error `abs(new-reference)/abs(reference)` with 0/0 defined as 0 and nonzero/0 as infinity. Bitwise equality requires identical dtypes and identical uncompressed array bytes (not identical ZIP files); its tolerance is zero. No nonzero camera-array acceptance tolerance was specified, so these are measured discrepancies, not a claim that camera equivalence passed.

| Window | Array | Maximum absolute error | Maximum relative error | Bitwise identical |
| --- | --- | --- | --- | --- |
| K000 | depth | 0 | 0 | Yes |
| K000 | depth_conf | 0 | 0 | Yes |
| K000 | extrinsic | 1.7940998077392578e-05 | 0.00013124442924460653 | No |
| K000 | intrinsic | 0.0020751953125 | 5.507285815175488e-06 | No |
| K001 | depth | 0 | 0 | Yes |
| K001 | depth_conf | 0 | 0 | Yes |
| K001 | extrinsic | 9.000301361083984e-06 | 0.0007174010129132182 | No |
| K001 | intrinsic | 0.000640869140625 | 1.6872907156908716e-06 | No |

Timing surrounds `run_inference` with CUDA synchronization, excluding model loading, image preprocessing and NPZ compression. Peak allocated memory includes resident model weights; CUDA allocator statistics are reset before each window.

| Window | Inference seconds | Peak allocated MiB | Peak reserved MiB |
| --- | --- | --- | --- |
| K000 | 4.921412346011493 | 4121.87060546875 | 4288 |
| K001 | 4.464301345986314 | 4121.87060546875 | 4328 |

**Not verified:** the cause of the camera-array differences, downstream fused geometry or training impact, full-sequence GPU equivalence, phone B0 rerun, and NX hardware equivalence. This two-window check does not establish those results.

### Downstream use

Before scale estimation, windows are restricted to training frames. Scale anchors use trajectory extent >0.3 m and positive fitted scale; shared-depth edges need at least three shared frames and ≥500 high-confidence pixels per qualifying frame. Disconnected/unanchored scale systems raise errors. No per-frame C1 fit is applied. Empty test depth slots are zero and invalid.

Fusion uses confidence above frame median, 3×3 relative depth span ≤3%, z in (0.3,4) m, stride 2; depth support is `abs(log(observed/projected))<0.03` with at least `min(2,max(seen,1))` confirmations. Partners are the undirected union of six ranked neighbors within 1.5 m and full-rotation angle <30°. Shared recon_core NumPy voxel downsampling begins at 2 cm and is enlarged until the configured cap (default 300k points); NumPy implements PLY I/O independently.

## Image-only limitations

COLMAP sequential overlap 20, quadratic overlap enabled, no loop detection, shared OPENCV camera; intrinsics/distortion refine during BA. Reports include counts for every model, largest-model registration rate and number of additional disconnected models. Default and currently implemented policy is **reject partial registration**. Successful complete models are undistorted before depth/training. Without an external scale, exported units are arbitrary. Metric depth thresholds are then thresholds in those arbitrary units; tune/validate scale before interpreting physical geometry.

## Archive coordinates and sidecars

Keep the full fingerprint directory when archiving a run: the export stage references the checkpoint in the train stage. The prepare manifest retains source image resolution and paths; the train configuration records normalization. Exported camera arrays are world-coordinate c2w with original-resolution intrinsics. Evaluation `before`/`after` camera snapshots are in normalized training coordinates; use the train configuration's center and meters_per_unit when converting them to world units. The colored cloud and Gaussian PLY are already in world units.

## Display contract (scene schema v2)

New exports prefer `preferred_model.path = gaussians.ply` and record the actual SH degree. `preview_model` identifies the DC-only SPLAT and warns that view-dependent color is lost. Full SH PLY is mandatory; a preview-only result warns and fails completion.

`capture_cameras` contains every frame's K, original `[width,height]`, and horizontal/vertical FOV computed from the calibrated sensor edges. `capture_corridor` records centers, c2w rotations, forward axes and training IDs. Both navigation bounds must hold relative to the same training camera: ≤0.15 meters (or arbitrary exported scene units) and ≤15° full rotation. The normalized distance bound is also recorded. These are quality recommendations for consumers; the repository viewer (`viewer/`) enforces them only in guided mode.

The export directory is the contract for every downstream tool (viewer, shot rendering, evaluation, simulation); they read it through `recon.io.scene.load_scene(export_dir, width=None)`, which returns the Gaussians (metric, log scales, logit opacities, wxyz, SH as `[N,C,3]`), training and held-out capture cameras with intrinsics rescaled to `width`, `world_up`, `bounds` and the fused cloud. Rendering the loaded Gaussians reproduces the normalized training checkpoint to about 100 dB PSNR (office_gs_04, four training views, 640×360).

Top-level `world_up` is the source-world up direction: `[0,0,1]` for drone LIO scenes; for image-only SfM scenes scaled by an assumed camera height it is the fitted floor normal (`pose/report.json` `scale_assumption.plane_normal`), and viewers rotate it onto their vertical. Top-level `bounds` (`min`/`max`, 0.5/99.5 percentiles of the fused cloud per axis) is the reconstruction extent for free navigation. Optional `name`/`subtitle` come from config `export.name`/`export.subtitle`.

World coordinates are right-handed, with wxyz Gaussian quaternions; camera-local axes remain optical right/down/forward. Optional manifest `world_up` defaults to `[0,0,1]`. An explicitly different up axis is rejected on export: align the complete scene before training, or apply a proper joint scene-graph transform and inverse-rotate SH lookup directions. Do not relabel raw SfM coordinates as gravity-aligned, and never change positions without Gaussian orientations and cameras. See [display pitfalls and CPU helpers](PITFALLS.md). Existing schema-v1 exports are not retroactively modified.

## Resource and cache behavior

No resource limits or environment mutation occur at module import. Resource management belongs to the caller. CUDA operations (XFeat, Omega, train/eval) should be run under a shared GPU lock if the GPU is shared. CPU fusion also benefits from limiting BLAS threads. Package fingerprints hash input file contents, not mtimes, plus recon and recon_core Python source (excluding recon_core/tests). A stage marker hashes all its files; corrupt artifacts are rejected rather than silently used.

The `bag` dependency group only installs the optional parser dependency; a bag importer is not yet exposed. Exported images/poses are the stable first-release boundary.

## Evidence index

Local source annotations only; all depth acceptance values and criteria are included above:

- Reference windows and predictions: `Data/experiments/office_gs_02_20261019/T17/omega512/kf_T15A8/windows.json` and `K000/prediction.npz`, `K001/prediction.npz` beneath that folder.
- Reference input images and checkpoint: `Data/office_gs_02_frames/inputs/rgb/%06d.png`, `Data/Model/vggt_omega_1b_512.pt`.
- New predictions and reproducible comparison script: `Data/experiments/restructure_20261029/depth_check/` (`compare.py`, `comparison.json`, `gpu.log`, `windows.json`, and the two window folders).
- B0 CPU references: `Data/experiments/vggt_routes_20261028/prepare_B0.py`, `B0_scale.json`, `B0_cameras.npz`, `cloud_B0.npz`, `prior_B0/`, and `B0_formal/` under that experiment directory. Historical floater input is `prior_B/` there.
- E3 cameras/extraction reference: `open_release_repo/outputs/equivalence/e3/`（原离线包仓库的验收输出）. Real video: `Data/Room/Rec709_video_20261027_001-fix.MOV`.
- Staged CPU evidence: `Data/experiments/restructure_20261029/` contains `b0_check.yaml`, `b0_manifest.json`, `compare_b0.py`, `b0_comparison_verified.json`, `b0_metric_threads.json`, `b0_extract.yaml`, `compare_b0_extract.py`, `b0_extract_comparison.json`, `compare_b0_loading.py`, and `b0_loading_comparison.json`. The first single-thread run and its nonzero discrepancies remain in `b0_comparison.json`.
