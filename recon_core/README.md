# recon_core

机载端（ROS，Python 3.8）和离线端（Python 3.10）共用的数值核心，只依赖 NumPy / SciPy；读 RGB 时才导入 OpenCV。
约定：位姿是相机到世界的 4×4 矩阵（OpenCV 光轴：右、下、前），世界系为米制、Z 轴向上，深度是沿光轴的米制 z。

| 模块 | 内容 |
| --- | --- |
| `scale.py` | 窗口尺度：轨迹先验 + 相邻窗口共享帧深度比（可选再加跨窗口重叠帧深度比 `cross_view_log_ratio`），对数域加权最小二乘；`batch_window_scales` 处理一整段已完成的窗口 |
| `fusion.py` | 单帧置信度/深度边缘掩码、近邻视角选取、多视一致性融合、体素降采样 |
| `ply.py` | 二进制 PLY 写出 |

## 参数

| 项 | 值 |
| --- | --- |
| 轨迹先验 | 提供者轨迹范围 > 0.3 m 才使用，权重 min(范围, 2) |
| 相邻边 | 至少 3 个共享帧（批处理）；每帧取两窗置信度都**高于**本帧中位数、深度为正的像素，像素数 ≥ 500；权重 3 |
| 跨窗口行（可选） | 两帧属于不同窗口：帧 i 的米制深度按位姿投到帧 j，与帧 j 深度比较；目标深度在 0.3–4 m、\|log 比\| < 0.25 的像素计入，像素数（换算到模型分辨率）> 500；取中位数。每个有方向的帧对一行，权重由调用方给（机载 G4 用 0.3） |
| 逐帧尺度（可选，`solve_frame_scales`） | log s_f = x_w + d_f；轨迹先验 σ 0.04、相邻边 σ 0.04、帧对行 σ 0.08、帧偏离 σ 0.04、窗内相邻帧偏离变化 σ 0.02，权重 1/σ，scipy lsqr；没有先验锚定的连通块不求解 |
| 单帧掩码 | 置信度高于中位数，深度 > 0，3×3 邻域深度相对极差 ≤ 0.03 |
| 近邻视角 | 1.5 m 内、夹角 < 30°，按 夹角/30 + \|距离 − 0.3 m\| 取前 6 个 |
| 一致性 | \|log 深度比\| < 0.03 计为支持；支持数 ≥ min(2, 可见视角数)；0.3 m < 深度 < 4 m；步长 2 |
| 体素 | 2 cm 起；点数超过上限时按 max(1.1, (点数/上限)^(1/3)) 放大，从 2 cm 点云重新降采样 |

## 来源与等价性

由三份实现合并而来：离线包 `dronerecon` 的 fusion/scale、G4 逐帧位姿放置 `g4_backend/frame_pose.py`、`drone3d_onboard/fusion.py`。合并时统一了一处差异：G4 在线版计算共享帧深度比时取置信度 ≥ 中位数的像素，现与离线版一致取 > 中位数；真实 VGGT 置信度没有重复值，每帧最多相差一个像素。体素降采样用 NumPy 实现，网格原点与 Open3D 相同（min − 体素/2），输出按体素字典序排列；三维体素编号合成一个 int64 键后排序、用 bincount 累加（每轴编号 < 2^21，超出时退回逐行 unique），与逐行 unique + add.at 的旧实现逐位相同（`tests/test_fusion.py`）。

融合按帧拆成 `frame_points`（单帧多视一致性检查后的点）、`frame_image` / `frame_colors`（颜色）和 `cloud_from_frames`（2 cm 体素与自适应放大）；`colored_cloud` 依次调用它们。机载在线点云用这些函数做按帧增量（`drone3d_onboard.session.SessionFusion`）。

已验证（2026-10-29，E1 无人机 V1.0 输入：670 帧、23 个窗口、512×288）：

| 项目 | 结果 |
| --- | --- |
| 窗口尺度、逐帧尺度、轨迹拟合、相邻边 | 与 dronerecon 逐位相同 |
| 深度缓存、单帧掩码 | 逐位相同 |
| 融合点云 | 2 cm 点数 2,732,886、最终 848,586 点、体素 0.03383419681295737 m；按坐标排序后 XYZ / RGB（float32）逐位相同 |
| CPU 融合耗时 | 136 s（4 线程）；2026-11-03 体素降采样改为整数键后 78 s，点数、体素与按坐标排序后的 XYZ / RGB 仍与上表逐位相同 |

## 测试

```bash
python -m pytest recon_core/tests            # 离线环境
python3 -m unittest discover -s recon_core/tests -t .   # ROS 环境
```

ROS 构建通过 `ROS_Compatibility_layer/src/drone3d_reconstruction_core/src/recon_core` 符号链接把本包装进 catkin，测试随 `catkin_make run_tests` 运行。

## 证据索引

- E1 对比脚本与结果：Data/experiments/restructure_20261029/m1_e1_equivalence.py、m1_e1_equivalence.json；2026-11-03 体素提速复验 Data/experiments/onboard_20261103/e1_fuse_equivalence.py 与 jobs/e1-fuse-eq/out.log
- E1 参照输出：open_release_repo/outputs/equivalence/e1/（原离线包仓库的验收输出）
