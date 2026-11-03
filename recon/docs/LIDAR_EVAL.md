# M4：激光雷达参照地图与点云误差评测

2026-10-30 · 面向离线重建维护者

## 口径

clean global 5 cm 首点地图；所有融合点找最近的参照点，恒等变换，不配准；全部点汇总后取 P50/P90；不是点到平面距离。

最近邻由 SciPy cKDTree 计算，每批 100000 点、workers=4；距离先转换为 float32，再以 NumPy median / percentile（默认 linear）计算全点 P50/P90，单位米。不剔除远点、不逐帧平均、不做 ICP。空输入或非有限输入报错，不静默过滤。

阈值比例是严格 `distance < 0.05 / 0.10 / 0.20`，以全部融合点为分母。世界 z 分层为 `<1`、`[1,2)`、`[2,3)`、`[3,4)`、`>=4` m；空层 n=0、P50/P90=null。世界 z 不是相对地板高度，也不是源观测的光轴深度。

## 安装与调用

```bash
python -m pip install -e '.[lidar]'
recon lidar-reference --bag recording.bag --out reference.npz
recon lidar-distance --cloud initial.npz --reference reference.npz --out result.json
# 外参文件内容为 JSON 4×4 矩阵，话题可替换：
recon lidar-reference --bag recording.bag --out reference.npz \
  --lidar-topic /livox/lidar --odometry-topic /Odometry \
  --body-from-lidar T_BL.json
```

`rosbags` 和 `numba` 只在生成参照时导入；未安装时提示安装 `[lidar]`。距离评测仅需基础 NumPy/SciPy；PLY 使用仓库已有 scalar-vertex reader，不新增 plyfile 依赖。

[模块](../evaluation/lidar.py) 仅公开两个入口：

- `build_reference(bag, out_npz, *, lidar_topic='/livox/lidar', odometry_topic='/Odometry', body_from_lidar=None) -> dict`：读取 ROS1 bag 两遍，返回统计并写压缩 NPZ 与同名 JSON。NPZ 包含 float32 `points`、int64 `voxel_key`、int32 `witness_scan_a/b`。JSON 含点数、扫描数、完整 NPZ 文件 SHA-256、过滤计数、里程计去重统计、参数和耗时。不生成移除点缓存。
- `distances(cloud_points, reference) -> dict`：两参数均可为 Nx3 数组或 NPZ/PLY 路径；NPZ 使用 `points` 数组。返回 `n/p50/p90`、`within_distance_fraction`（米阈值字符串键）、`height_layers` 和参照点数。无坐标变换或尺度拟合。

`body_from_lidar` 按项目 T_AB 约定是 **T_BL，即 LiDAR → body**，不是逆矩阵。None 保留历史 `Ry(+25°) * (p + [-0.011,-0.02329,0.04412])`；平移项是 `Ry(+25°)*offset`，不能直接把 offset 当 T_BL 的平移列。自定义矩阵必须为刚性 4×4。默认仅对应历史装配，不是通用 Livox 标定；其他装配应提供实测外参。

## 参照生成参数与顺序

算法移植自 T25 原语及 T31 prepare/global_reference，默认参数保持原值：

1. 扫描占据统计：全部扫描的有限点、传感器距离 ≥0.5 m，每扫描每 10 cm 传感器体素计一次，无距离上限。出现扫描数 ≥ceil(0.2×全部扫描数) 的体素视为传感器固定体素。
2. 里程计按 header 时间稳定排序；同时间戳不同 position/quaternion 报错。距上个保留位姿 <150 ms 且完整 position/quaternion 完全相同则移除。纯旋转不会因位置未变而被删。
3. 逐点时间为 CustomMsg `timebase + offset_time`（ns），不是 bag 接收时间或扫描 header。依次移除非有限点、里程计覆盖范围外点、距离 <0.5 m 点、固定传感器体素点。不加 rear wedge，不做射线清理。
4. 在逐点时间做线性平移和 quaternion SLERP，先传感器刚性变换再放入原 LIO 世界系；禁止逐点外推。扫描中点是首末点时间的整数均值，仅计算雷达原点时将中点裁到轨迹覆盖范围。
5. 世界 5 cm 体素按 `floor(world/.05)` 编码。每扫描每体素保留最早合格真实点，不取质心；键在 float64 坐标上计算，代表点随后存 float32。float32 舍入可能跨体素边界，不据存储点重新计算键。
6. 同世界体素必须存在一对扫描，**同时**满足中点时间差 ≥10 s、世界雷达原点间距 ≥1 m。不是独立时间/空间极值判据。稳定排序后保留最早合格扫描的代表点，输出按体素键升序；见证扫描对也保留原搜索次序。

资源限制由调用者设置，导入模块不改线程或 CUDA 环境。16221 扫描的内部 ID 使用 uint16 降低内存，更多扫描使用 int32；输出见证仍为 int32，不改变数值或顺序。重访搜索通过稳定排序索引访问原数组，避免物化排序后的键和扫描 ID 副本，搜索次序与原实现相同。与旧脚本相比，省去准备 NPZ 往返、绘图、移除点缓存及硬编码磁盘守卫；保留 T31 对已去重轨迹的第二次构造。传感器刚性变换保留先加 offset 再旋转的浮点运算次序。

## 已验证数字与验收判据

CPU 实测 A/B（2026-10-30）：

| 输入 | 融合点数 | 参照点数 | P50 (m) | P90 (m) | 与原始统计差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1.0 office_gs_02 LIO_new | 3524680 | 947679 | 0.03605738282203674 | 0.13601155728101733 | 0 |
| office_gs_04 initial | 850318 | 1577973 | 0.029156308621168137 | 0.06769340857863428 | 0 |

A 要求绝对误差 <1e-9 m，实测为 0；B 与原始 JSON 完全相等，对应发布的 2.915630862 / 6.769340858 cm（舍入显示）。B 的所有世界 z 分层统计、5/10/20 cm 比例也与原始 JSON 完全相等。

| office04 世界 z (m) | n | P50 (m) | P90 (m) |
| --- | ---: | ---: | ---: |
| <1 | 525614 | 0.02974778041243553 | 0.07052014097571374 |
| [1,2) | 155196 | 0.02828042209148407 | 0.05556322820484638 |
| [2,3) | 110971 | 0.02756856009364128 | 0.0505860261619091 |
| [3,4) | 58537 | 0.030180227011442184 | 0.09160122573375705 |
| >=4 | 0 | null | null |

office04 的 <5/10/20 cm 比例分别为 0.840714885489899 / 0.9528811574022895 / 0.9923558010062118。

历史 office04 参照为 1577973 点、16221 扫描，完整 NPZ 文件 SHA-256 为 `692c7c3f02c376f5f77175998595faddec1166e67a1d031bf8b74627ca07b808`。C 的验收要求从 22784460339-byte 原始 bag 重新生成后，点数一致且数组逐点精确相等（容差 0；同时比较体素键及见证扫描）。**C 已通过**：points、voxel_key、witness_scan_a/b 的 dtype、shape 和全部值均完全相同，完整 NPZ 文件 SHA-256 也与上述历史哈希相同。通过内部任务运行器以 MemoryMax=10G、MemorySwapMax=0、CPUQuota=400%（无 GPU）重新读取 bag，完成耗时 433.25 s，采样峰值 RSS 8873.09 MiB；本次验收输出总量 25 MiB，低于 200 MB 上限。首次运行在 485.48 s、峰值 10214.43 MiB 时 OOM；改为稳定索引访问避免排序数组副本后，在同一限额下重跑成功。失败记录保留，未提高内存限额。

### CPU 测试

在装好 recon 依赖（含 gsplat）的环境里运行 `CUDA_VISIBLE_DEVICES='' python -m pytest -q -p no:cacheprovider tests recon_core/tests`，实测 98 passed（4.16 s）。[M4 合成测试](../../tests/test_lidar_eval.py) 覆盖分位数与严格阈值、世界 z 边界及空层、无效点输入、CustomMsg 字节解析、位姿去重/冲突/禁止外推、联合重访与索引等价、合成 Reader 的完整生成与自定义话题/外参、PLY CLI 和缺少可选依赖报错；不依赖真实 bag。

## 适用范围

需要同一次录制的 Livox CustomMsg 与 LIO nav_msgs/Odometry ROS1 bag，输入融合点必须已经位于该次录制的 LIO 世界系，单位米。不同录制或 FAST-LIO 重启后的参照不可直接混用。世界轴、尺度或外参错误不会在评测时自动修正。

参照本身依赖 LIO，不是独立位姿真值；相同 LIO 位姿放置的点云可能受偏向。重访过滤也排除稀疏重访的真实表面；5 cm 采样有有限底限。office02 和 office04 不是同一控制实验：前者 3524680 点，后者受融合 100 万点上限影响，不能仅凭两行数字宣称算法提升。本模块不提供局部点到平面误差、光轴深度分层或完整度。

## 证据索引

以下仅为本地来源标注；全部口径、数字与判据已在正文内联。

- 原语：`Data/experiments/quality_check_20261021/T25/lidar_generate.py`。
- 流式合格记录路径：`Data/experiments/quality_check_20261021/T31/cloud_lidar31.py`。
- A：`Data/experiments/quality_check_20261021/T31/office/cloud/LIO_new.npz`、同目录 `global.json`；参照 `Data/experiments/quality_check_20261021/T25/lidar_reference_clean.npz`。
- B 与历史参照：`Data/experiments/promo_20261029/lidar_eval/` 中 evaluate.py、reference.json、recon_initial.json、delivery_lio.json、audit.py、REPORT.md 及 lidar_reference_clean.npz；点云 `Data/experiments/promo_20261029/office04/recon_out/92c23db1586b282c7267cb4569f67a4bad4d5ef60e82b6fb928123a39fc493de/fuse/initial.npz`。
- C 原始只读 bag：`office_gs_04.bag`（本机 bag 存储分区，不在仓库内）。
- 本次验收：`Data/experiments/reuse_20261030/m4/`。
