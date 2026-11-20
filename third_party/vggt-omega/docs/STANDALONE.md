# VGGT-Omega 独立流程(历史/开发用)

> 本文是 S380 时代的独立脚本和开发查看器说明，不定义当前配置。机载主线见 [ROS 兼容层](../../../ROS_Compatibility_layer/README.md)，离线流程见 [recon](../../../recon/README.md)；两者都通过 `reconstruct_omega.py` 做推理，参数由各自的配置给出。

## 环境准备

离线流程按 [recon 安装说明](../../../recon/docs/INSTALL.md)准备 Python 与 CUDA 依赖；机载 ROS 部署按 [ROS 环境说明](../../../ROS_Compatibility_layer/docs/ENVIRONMENT.md)分离 CPU 与模型解释器。
NX aarch64 的 PyTorch wheel 必须匹配目标 JetPack、Python 和 CUDA，不能直接套用本机 x86_64 测试配置。

### 权重

The VGGT-Omega weights are access-gated on HuggingFace. Request access at
[`facebook/VGGT-Omega`](https://huggingface.co/facebook/VGGT-Omega), then
download `vggt_omega_1b_512.pt` into `Data/Model/`.

## 用法

### 1. Prepare images

Place drone images in `colmap/data/images/`:
```bash
cp /path/to/drone/images/* colmap/data/images/
```
Supported: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`.

### 2. Run reconstruction

The single-NX continuous production line uses ROS `continuous_production.sh`;
the offline pipeline uses [recon](../../../recon/README.md). For standalone
single-window model debugging, invoke the model script directly:

```bash
python3 third_party/vggt-omega/reconstruct_omega.py \
  --images_dir colmap/data/images \
  --output_dir /absolute/new-reconstruction-result \
  --checkpoint Data/Model/vggt_omega_1b_512.pt
```

## 流程

The VGGT-Omega path runs the 1B model on Jetson via a bf16 aggregator, fp32
heads, `max_size` image loading, and `expandable_segments`. On the historical
Orin NX 8GB/JetPack 5 machine, the recorded all-sensors-online automatic profile
is 8 frames at resolution 256. The input is uniformly subsampled to that cap.

For a different frame cap, pass `--frame_group_size` directly to
`reconstruct_omega.py`; this does not change the ROS profile.

Open an existing result without rerunning the 1B model. Supply the matching
prepared RGB-D directory explicitly when measured sensor depth is available:

```bash
python3 third_party/vggt-omega/web_viewer.py \
  --output_dir /absolute/reconstruction-result \
  --sensor-dir /absolute/matching-rgbd-input
```

Then visit `http://localhost:8080`. The left panel switches between color or
grayscale GLB point clouds and `.splat` Gaussian-style rendering. The camera
slider on the right shows the matching BAG RGB frame, measured aligned depth
(or Omega-predicted depth), and a per-camera GLB point cloud. The splat mode is
a browser rendering of Omega points with isotropic display kernels, not a
trained 3DGS result; it can be replaced when the pipeline emits learned
covariance, rotation, and opacity.

Run stages manually:

```bash
python3 third_party/vggt-omega/reconstruct_omega.py
python3 third_party/vggt-omega/web_viewer.py --output_dir colmap/data_vggt_omega
```

### Live committed-result watcher (legacy dev viewer, NOT the production Web)

> For the continuous single-NX production line, use the WP8 Web node
> (`continuous_production.sh --web`, bind host/port in the continuous profile)
> instead. It provides session isolation, live RGB/depth, latest-window Omega
> depth, status/control and the accumulated same-map point cloud.

The legacy `third_party/vggt-omega/result_watch_viewer.py` polls a results root every 2 s
and lists every directory that contains a committed `result.json` +
`predictions.npz`. It is a single-result development viewer only: it has no
session isolation, scans every result under one root, and its frame slider is
not bound. Do not use it as evidence for the continuous production Web.

```bash
python3 third_party/vggt-omega/result_watch_viewer.py   --results-root <结果根目录>   --server-port 8080
# 另一台机器的浏览器：http://<运行主机地址>:8080
```

On the local workstation while ROS1 Noetic runs inside Docker (Gradio 5 needs
Python >= 3.10, so use a small viewer container next to the ROS container with
the repository mounted):

```bash
docker run --rm --network host -v "$PWD":/repo -w /repo python:3.11-slim bash -lc   "pip install -q 'numpy<2' pillow matplotlib trimesh gradio &&    python third_party/vggt-omega/result_watch_viewer.py      --results-root /repo/<your-results-root> --server-port 8080"
# browser: http://localhost:8080
```

Run the viewer before starting `rosbag play`; every verified and atomically
committed reconstruction shows up automatically without touching staging data.

### VGGT-Omega key parameters

| Arg | Default | Purpose |
|---|---|---|
| `--images_dir` | `colmap/data/images` | Input images |
| `--output_dir` | `colmap/data_vggt_omega` | Output (predictions.npz, scene.glb) |
| `--checkpoint` | `Data/Model/vggt_omega_1b_512.pt` | Model weights |
| `--image_resolution` | 0 (auto) | Long-side input resolution (8GB: 256; larger tiers: 512) |
| `--load_mode` | `max_size` | `max_size` halves patch tokens vs `balanced` → more frames fit |
| `--frame_group_size` | 0 (auto) | Max frames per pass (0=auto: 8 on the verified 8GB profile). Input is subsampled to this. |
| `--conf_thres` | 50.0 | Confidence percentile for the default GLB |
| `--max_points_k` | 5000 | Max points (thousands) in the default GLB |

> The auto values above are NOT a signed production setting for the
> Ubuntu 20.04 / NX 16 GB target. The continuous production line always uses an
> explicit reconstruct profile (`frame_group_size`, `image_resolution`,
> resource limits); safe values for the physical NX must be frozen by the Step 2
> no-flight bench, not by `0`/auto.

Output is written to `colmap/data_vggt_omega/`:

```
colmap/data_vggt_omega/
├── predictions.npz   # camera/depth/point predictions (reused by the viewer)
├── scene.glb         # default GLB scene (5M points @ conf≥50%)
├── manifest.txt      # images actually fed to the model
└── viewer_cache/     # on-demand color/gray GLB, splat, and camera GLB files
```
