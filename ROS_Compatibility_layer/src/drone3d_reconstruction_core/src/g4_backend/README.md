# G4 独立核心

这个包提取已冻结的 `legacy_g4_20260916` 方法。
核心依赖 NumPy、SciPy，以及仓库根的 `recon_core`（逐帧位姿放置的尺度求解在那里）。
核心读取数组和数据对象。
ROS 消息、磁盘缓存、模型推理、线程调度和地图发布由调用方负责。

## 方法边界

G4 保持每个窗口的 Omega 深度、内参和相机轨迹。
相邻窗口通过相同 RGB 帧的三维预测建立 E1 Sim(3) 约束。
全局图同时优化窗口 Sim(3) 和定位光心的位置符合程度。
G4 没有定位朝向因子。
G4 原基准没有回环边。
每个窗口使用一个整体 Sim(3)。

`GraphResult.geometry_accepted` 始终为 `False`。
`publishable_numerical` 仅记录原 G4 的数值求解、秩与分量条件。
`component_diagnostics` 额外记录每个分量的变量数、雅可比秩和数值可发布状态。
分量秩使用同一次求解雅可比的相应变量列计算。
这个诊断保留同一次整体优化。
它帮助调用方区分已定位分量与新增待定分量。
局部规范下的分量保留为 `local_only`，没有全局可发布资格。
调用方必须分别决定几何验收、地图发布和待定状态。
定位符合程度是对定位记录的一致性指标。

历史原始算法来自 `Data/experiments/vggt_backend_review_20260916/run_02/scripts/{core,adjacent,graph}.py`。
历史结果定义来自该实验的 `BASELINE.md` 和 `baseline/backend_G4_snapshot.json`。
这些路径仅用于说明来源。
本包的运行和安装均不依赖实验目录。

## 安装和核查

在仓库根目录执行。`recon_core` 没有写进本包的依赖声明，需要它能被导入，仓库根在路径上即可。

```bash
python -m pip install ./ROS_Compatibility_layer/src/drone3d_reconstruction_core/src/g4_backend
python -m unittest discover -s ROS_Compatibility_layer/src/drone3d_reconstruction_core/src/g4_backend/tests -v
```

## 核心模块

| 模块 | 职责 |
|---|---|
| `types.py` | 数据契约、数值验证和版本标识 |
| `sim3.py` | Sim(3) 群运算、指数、对数、拟合和方向支持检查 |
| `geometry.py` | 预测转为首相机规范、深度反投影和像素中心缩放 |
| `registration.py` | 共享帧采样、E1 RANSAC、共识重拟合和分层 bootstrap |
| `localization.py` | 位置插值、杆臂变换、时间块分区和定位采样 |
| `graph.py` | 分量初始化、位置约束和 G4 图求解 |
| `frame_pose.py` | 逐帧外部位姿放置与窗口尺度，机载主线使用；数值在 `recon_core.scale` |
| `window_plan.py` | 长窗分组计划与登记清单校验 |

## 数据契约

`Sim3` 使用列向量约定：`target = scale * rotation @ source + translation`。
批量点以 `N×3` 行存储。
`apply` 内部计算 `scale * points @ rotation.T + translation`。
切空间顺序为 `[translation_xyz, rotation_xyz, logscale]`。
图优化使用右扰动 `S = S_base * Exp(delta)`。

`WindowGeometry` 包含本窗口的帧身份、时间戳、归一化深度、置信度、内参、相机朝向和光心。
`rotations` 表示相机坐标到窗口坐标。
`centers` 表示窗口坐标中的相机光心。
`intrinsics` 使用模型图像的像素中心坐标。
`depth` 表示相机 z 轴深度。
`shared_prefix` 表示新窗口开头的历史共享帧数。
每个 RGB 帧应使用稳定的物理帧身份。
相邻窗口中的相同帧身份必须具有相同时间戳。
提供 `rgb_hashes` 时，两窗必须同时提供并逐帧一致。

`from_prediction` 接收世界到相机的 `N×3×4` 外参。
它将所有几何转到首相机坐标。
它使用首帧中央 60% 区域的深度中位数统一长度单位。
384×288 时，该区域恰好为 `[57:230, 76:307]`。
ROS optical/body 坐标转换和查看器的轴翻转不属于此函数。

`RelativeEdge(i,j,measurement,...)` 中的变换将 j 窗映射到 i 窗。
边残差为 `Log(Z_ij^-1 * S_i^-1 * S_j)`。
残差的平移三维先除以 `length_source`。
协方差对应这个归一化后的右切空间。
协方差白化使用其 Cholesky 下三角矩阵的逆。

`LocalizationAnchor` 的参考位置已经是相机光心。
`select_localization_anchors` 默认把参考轨迹作为相机轨迹。
提供 `body_from_camera` 时，参考轨迹被解释为机体轨迹。
此时必须同时提供机体朝向。
该外参是相机到机体的 4×4 刚体变换。
所有时间戳必须已经属于同一时钟域。
该函数支持仅含 `window_id/frame_ids/timestamps_ns/centers/shared_prefix` 的轻量对象。
调用方无需为定位采样保留所有窗口的稠密深度。

## 最小调用

```python
from g4_backend import build_shared_pairs, estimate_adjacent
from g4_backend import build_initial_states, solve_g4

pairs = build_shared_pairs(older_window, newer_window)
registration = estimate_adjacent(pairs)
if registration.edge is not None:
    edges.append(registration.edge)
initial, components = build_initial_states(edges, window_ids)
result = solve_g4(initial, edges, anchors)
```

配对采样使用新窗口的共享前缀。
默认按帧顺序循环分为 `fit, fit, fit, accept, report`。
每帧划分 4×4 网格。
每格最多采样 32 点。
E1 仅使用 `split == 0` 的拟合点。
原始采样使用两窗 P50 置信度与相对深度边缘过滤的交集。
拟合阶段不读取 accept 和 report。

调用方可以通过 `exclude_fit_pixels` 传入需要保护的验证观测。
该参数使用模型像素坐标。
本包不会自动发现调用方已有的历史验证身份。
调用方必须在拟合前冻结身份与分区，并传入相应保护邻域。

默认 RANSAC 为 1000 次。
默认共识重拟合为 5 次。
默认 bootstrap 为 200 次，最低有效次数为 160。
相邻边按共享帧、再按网格分层重采样。
默认阈值为共同有效目标深度中位数的 3%。
复放历史缓存时，调用方可显式传入原记录的 `threshold`。
数值协方差门槛通过不等于独立几何验收通过。

默认定位策略保持 G4 的 1 Hz RGB 近邻、20 秒时间块和 1 秒边界保护。
时间块使用 seed 20260915 的哈希按 80% 比例进入拟合。
每块定位因子权重为该块因子数平方根的倒数。
精确复放需要提供完整、去重、按时间排序的历史新帧集合。
增量调用可能在当前尾部增加观测时改变近邻选择和时间块权重。
调用方应保留或重建完整因子集合。

## 与后续诊断版本的区别

本包保留 G4 的定位归一化顺序：先乘时间块权重，再施加块 Huber。
reviewer R1 在单次定位残差上先施加 Huber，再乘时间块权重。
这两个顺序产生不同目标函数。
S14 还对四条特定连接施加 0.1 权重。
G4 图求解不混入 R1 顺序、S14 边权或后来的逐帧修正。逐帧位姿放置是独立的 `frame_pose.py`，方法身份不同。

## 工程与运行边界

窗口数量和窗口身份由调用方提供。
核心不识别 F037 或任何实验特例。
核心不会读取文件、创建 ROS 节点、启动进程或退出宿主程序。
时间门限默认关闭。
调用方显式设置求解时间门限时，超时结果带有失败状态和目前最优状态。
调用方应保留上一份已采用地图，不把失败状态直接提交。

图求解继承 G4 的稠密有限差分与 exact trust-region。
计算量随窗口数量增长。
本次提取没有引入新的稀疏增量求解器。
异步调度与设备性能验证属于 ROS 集成层。
本包没有对其他航线的泛化或实时速度作出新的保证。
