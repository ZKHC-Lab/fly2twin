# Omega 结果校验运行时

`result_validator.py` 在 CPU 上校验 VGGT-Omega 的重建结果，不加载 ROS、CUDA 或权重。Omega 原来的同名模块是兼容代理，原有调用入口保持可用。

`predictions.npz` 的六个 float32、C-order 数组为 images、depth、depth_conf、extrinsic、intrinsic、world_points_from_depth。每个数组保留 batch/frame 轴；`extrinsic` 为 world-to-camera。校验器检查数组形状、有限值、相机参数、深度反投影一致性及 manifest 输入顺序，并根据调用参数检查 GLB。

GPU 仲裁由同一 ROS 运行时包安装的 `drone3d_resources` 提供。默认锁文件为 `/tmp/drone3d-gpu-<uid>-0.lock`；跨容器部署需共享同一个锁文件和 UID。
