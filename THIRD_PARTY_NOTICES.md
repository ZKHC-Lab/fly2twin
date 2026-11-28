# Third-party dependencies and extraction provenance

2026-11-03

Original code of this repository is released by ZKHC under the Apache-2.0 [license](LICENSE). The repository bundles one third-party model implementation, VGGT-Omega in `third_party/vggt-omega/` under its own license, and the browser libraries in `viewer/vendor/`. It bundles no model weights, datasets, ROS stack or simulator source.

| Component | Integration | License / remaining review |
| --- | --- | --- |
| NumPy / SciPy | Dependencies; array I/O, geometry, nearest neighbors | BSD-family; retain upstream notices when redistributing binaries |
| PyTorch / torchvision | Optional training dependencies | BSD-family; CUDA binary dependency notices also apply |
| gsplat | Optional dependency; rasterizer and strategy primitives; callback ordering adapted in the baseline scheduler | Apache-2.0; copyright retained in the scheduler and full [upstream license](licenses/gsplat-APACHE-2.0.txt) included. Public distribution tested separately from historical checkout |
| fused-ssim | Optional training dependency | Upstream license and exact source revision must be archived before binary redistribution |
| pycolmap / COLMAP | Optional feature/SfM/BA dependency | BSD-3-Clause; Ceres/CUDA/transitive build dependencies have separate licenses |
| OpenCV | Image decoding and RANSAC | Apache-2.0 in modern releases; inspect build-specific notices |
| PyYAML | YAML configuration dependency | MIT |
| Pillow | Image I/O for evaluation and reel rendering | MIT-CMU (HPND) |
| XFeat | User-installed optional implementation, imported `modules.xfeat` | Apache-2.0 code; checkpoint-specific permission/revision still needs release evidence |
| VGGT-Omega | Bundled in `third_party/vggt-omega/`: upstream model code plus this project's inference entry, worker and result-validator scripts; weights are downloaded by the user | FAIR Noncommercial Research License ([`third_party/vggt-omega/LICENSE`](third_party/vggt-omega/LICENSE)), including usage restrictions and restrictions on commercial outputs; the whole directory and the pipeline outputs that use it are not commercially unrestricted |
| torchmetrics / LPIPS / AlexNet | Optional evaluation | torchmetrics Apache-2.0; model/backbone/linear-weight rights must be separately archived |
| rosbags / numba | Optional `lidar` dependency group; LiDAR reference builder reads ROS1 bags | rosbags Apache-2.0; numba BSD-2-Clause |
| three.js, Spark, fflate, spz | Vendored browser libraries in `viewer/vendor/` | MIT; license files kept next to each library |
| DiffDrone-PyTorch3D | Optional run-time import by `recon sim eval` from a checkout the user provides (`model.py`, `navigation_utils.py`, `drone_dynamics.py`) and its released checkpoint; not bundled | GPL-3.0; a distributed work that combines it with this repository is subject to GPL-3.0 |
| playwright-core | Optional headed browser checks (`viewer/tools/package.json`), not needed by the viewer | Apache-2.0 |

`plyfile` is not imported, declared or used. PLY scalar-vertex parsing and writing are implemented directly with NumPy. No GPL simulator or VGGT-X code is included; the differentiable-flight simulator is a separate GPL-3.0 repository.

## Algorithm extraction

Numerical functions originate in the private project's experiment scripts. Those sources are provenance, not runtime dependencies. The new code removes historical imports, global dataset paths, source-string patching, automatic downloads and import-time CUDA/cache environment changes.

The evaluation camera parameterization is a new implementation of the continuous 6D rotation representation (Zhou et al., CVPR 2019) with right-composed local translation, matching the mathematical interface of gsplat's example camera optimizer. No gsplat example source is copied. The Gaussian initialization/render/training algorithms were extracted from the project's own earlier drone reconstruction pipeline; upstream algorithm and dependency attribution is retained here.

## Before each release

- Archive dependency SBOM, model revision/digest/license and binary notices.
- Confirm rights to any example imagery.
- Do not ship private office images, cached model predictions, checkpoints or local configuration.
- Do not describe the Omega default workflow or its outputs as commercially unrestricted.
