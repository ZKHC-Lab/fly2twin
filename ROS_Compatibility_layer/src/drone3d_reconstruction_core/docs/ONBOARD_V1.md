# 机载融合、交付包与落地全局 BA

2026-10-28 · 维护说明

飞行中使用 LIO（激光惯性里程计）位姿，不做局部 BA。落地后对完整交付包执行全局光束法平差（BA）。机载推理长边 384，40 帧窗口、10 帧共享；NX 实际吞吐和 BA 依赖、内存、散热仍需硬件验收。

## 1. 配置

以下是对现有完整 profile 的覆盖项，不是可以单独启动的完整 profile。通用模板默认值保持不变。

采集 profile：

```yaml
odometry_association: interpolate
pose_time_offset_ms: 40.0
camera_info_topic: /camera/color/camera_info
keyframe_min_translation_m: 0.15
keyframe_min_rotation_deg: 8.0
keyframe_max_interval_sec: 2.0
handoff_buffer_frames: 150
handoff_buffer_max_bytes: 134217728
max_image_width: 1280
max_image_height: 720
```

连续会话 profile：

```yaml
alignment_backend: global_g4
g4_placement: frame_pose
pose_time_offset_ms: 40.0
pose_time_offset_stage: capture
# Omit camera_intrinsics to use snapshot CameraInfo (ROS parameters cannot be null).
initial_sample_fps: 5.0
initial_window_frames: 40
initial_overlap_frames: 10
initial_inference_resolution: 384
reconstruct_frame_group_size: 40
initial_keep_snapshot_on_success: true
initial_queue_mode: lossless_fifo
frame_pose_fusion: true
```

重建 profile 设置 `image_resolution: 384`、`frame_group_size: 40`、`load_mode: max_size`。1280×720 输入对应 384×224 模型深度。保留完整原图快照，不能只留模型缩图。

TF 使用 [新机体—相机外参](../../../docs/CAMERA_EXTRINSIC.md)：平移 `[0.056558, -0.006717, -0.077751]` m，四元数 xyzw `[-0.494176, 0.506142, -0.498990, 0.500618]`。旧 bag 的外参必须覆盖，不能同时发布冲突的 body→camera 变换。

### 时间语义

采集端在去重后的稠密 `/Odometry` 中，针对 RGB 时间 +40 ms 做线性位置 / SLERP 朝向插值；快照关联头仍标记 RGB 时间，`interpolation.query_stamp` 记录实际查询时间，另存偏移值和 `capture` 标记。缺少前后括号时不外推，不以旧姿态冒充偏移后姿态。

后端 profile 与 `run.json` 的 `frame_pose.pose_time_offset_stage=capture` 表示直接使用已偏移定位；实际快照标记与配置不一致时报错。未声明该选项的旧会话继续使用原后端关键帧插值与原回退规则；带采集偏移标记的输入不能再走后端偏移，避免重复 +40 ms。新输出目录不可复用旧代码的缓存。

## 2. 融合

显示与交付包共享 `drone3d_onboard.session` 和仓库根的 [recon_core](../../../../recon_core/README.md)（融合与窗口尺度的唯一实现，离线端共用）。新会话创建时必须声明 `frame_pose_fusion: true`，Web 也需启用该选项；开关保存在会话 backend_profile 中。旧会话没有此标记，即使新 Web 开启融合，仍使用旧显示算法。

融合规则与离线端 recon 一致（同一份 recon_core 实现）：

- 每帧置信度严格大于该帧中位数，3×3 相对深度跨度 ≤0.03；不是旧 Web 的全窗 P50。
- 邻居中心距离 ≤1.5 m、旋转角 <30°；按 `角度/30 + |距离−0.3|` 取前六个，再将邻接关系对称化。
- 投影使用最近像素 `rint`，目标 z>0.1 m。深度对数比绝对值 <0.03 算支持；要求至少 `min(2,max(可见视图数,1))` 个支持。没有邻居不能通过。
- 0.3<深度<4 m，像素 stride=2。该距离范围继承室内已验实现，不代表适用于高空场景。
- 从 2 cm 体素起步，原点为点集最小坐标减半个体素，位置/颜色取体素内均值。超过点数预算时体素乘 `max(1.1,(点数/预算)^(1/3))`，每轮重新采样原 2 cm 点集。
- NumPy 替代 Open3D 的体素累加，输出采用确定的体素字典序；与 Open3D 哈希遍历顺序不同。比较时规范排列，数值容差 1e-12。合成 1518 点初次融合 XYZ/RGB 最大差 0；自适应 195 点最大差分别 4.44e-16 m / 2.22e-16，体素同为 0.17251386221942672 m。

另外，32 帧真实 Omega512 缓存同输入比较：2 cm 点数 162156，自适应预算 20000 后 17102 点，体素 0.06221565156355806 m 完全相同；用最近 XYZ 双射对应后，最大坐标差 7.11e-15 m、颜色差 3.33e-16，均通过 1e-12。此测试验证融合数值逻辑，不替代 384 在线尺度链验收。

新修订会重建融合结果，不能对旧修订点云继续叠加：窗口尺度变化、未来邻居支持变化都可能影响历史帧。只处理 `merge_frame_indices` 拥有的帧，重叠输入不重复累计。

## 3. 导出交付包

加载 catkin 环境后：

```bash
python3 -m drone3d_onboard.export \
  --records-root /absolute/results/sessions --session s0123456789abcdef0123456789abcdef \
  --capture-profile /absolute/capture.yaml --output /absolute/session_delivery
```

会话 STOP 后执行。也可提前运行同一命令并增加 `--wait-stop-service /continuous_session/command`：仅在指定会话进入 STOPPED 后自动导出，会话 ID 改变或超时则失败。普通导出不依赖 ROS master；等待模式才依赖 ROS。

交付目录包括：

- `manifest.json`：版本 1、米制 Z-up、c2w 约定、新外参、偏移位置/取值、全尺寸 CameraInfo（含畸变）、采集参数、窗口清单和 SHA-256。
- `keyframes/NNNNNN.jpg`：CameraInfo 分辨率的原尺寸图像（本机台架为 1280×720），JPEG q95。
- `poses.npz`：`frame_ids`（交付包连续编号）、`timestamp_ns`、`c2w`（Nx4x4）、`K`（Nx3x3）、`blur`。原会话帧 ID 在 manifest 的 `source_frame_id` 中保留。
- `windows/NNNNNN.npz`：每窗原始 Omega `depth` / `confidence`、`depth_scale`、模型分辨率 K、原帧 ID、时间、拥有帧索引。米制深度为原始深度乘窗口尺度，不乘第二次 gauge。
- `cloud.ply`：NumPy 写入的二进制 little-endian XYZ float32、RGB uint8；米制、Z-up，不引入 plyfile。
- `cache/depth_m.npy`、`good.npy`：同帧序的米制深度与过滤前基础掩码，可用于复核融合和未体素化观测评测。

导出校验原正式记录、快照、预测摘要与当前完整地图修订。没有定位位姿的帧（例如早于 FAST-LIO 首个位姿）不导出，按窗记入融合信息的 `frames_without_pose`。整段 CameraInfo 的 K、D、宽高必须一致，个别帧采集时没关联到 CameraInfo 就用这份共享值，帧号记入 manifest 的 `camera_info_shared_for_frames`。外参须全段一致、图像须为 CameraInfo 分辨率、时间偏移须在采集端施加，实际值写入 manifest。目标目录存在则拒绝覆盖；manifest 最后写入，失败的半包不能作为完成交付使用。

## 4. 落地全局 BA

无需 ROS。运行环境需要 NumPy、SciPy、OpenCV、PyTorch、pycolmap 4.2.0、用户安装的 XFeat 源码/权重，以及可导入的离线包 `recon`（在该 Python 环境里对仓库根执行 `pip install -e . --no-deps`）。不会联网下载或静默跳过依赖。

```bash
systemd-run --user --scope -q -p MemoryMax=16G \
  python -m drone3d_onboard.ba /absolute/session_delivery \
  --xfeat-source /absolute/accelerated_features \
  --xfeat-weights /absolute/accelerated_features/weights/xfeat.pt
```

全部 CPU 路径：SIFT、XFeat 和 Ceres；线程预算 4，BLAS 1。此阶段不持 GPU 锁。逐阶段打印耗时和进程高水位 RSS（MiB）；Linux RSS 是全进程历史峰值，不是各阶段独立峰值。

直接调用 `recon.poses.prior.reconstruct`（与离线端精化同一份代码）：时序邻居 15、空间半径 1.5 m/45°、每帧最多 30 空间邻居；SIFT 与 XFeat 匹配集合合并；固定内参、σ=0.05 m 笛卡尔 LIO 位置先验；16→8→4 px 三轮三角化、BA、过滤，三角化角度下限 1.5°、每轮最多 100 次迭代。没有固定相机，沿用全局 G 的 pose-prior adjuster；该接口会先全局对齐，因此不能拿来实现锚定相机。只要求解 usable，不把迭代上限写成已收敛。

结果保存在 `onboard_ba/`：`cameras.npz`、`model_prior/` 稀疏模型、各轮求解记录、合并数据库、`timing_quality.json`、`epipolar.json`。成功后原子更新交付 manifest 的 `onboard_ba` 位姿/模型入口、质量判据及新文件哈希；质量不达标也如实记录，不能仅凭命令退出0宣称验收通过。已有结果拒绝覆盖。BA 所有输入相机保留，有观测比例另算，不能以注册数量冒充视觉约束覆盖率。相邻 Sampson 指标先算每个匹配对内 sqrt-Sampson 中位，再取 pair 间 P50/P90；这些是 BA 输入匹配，不是独立真值。

此参考 BA 接受零畸变共享 PINHOLE 相机；非零畸变必须先做经验证的去畸变，不会忽略畸变继续求解。

## 5. 验收边界

restructure 分支上 catkin 全量 766 项，0 错误、0 失败；7 项跳过是容器里没有 Node（test_continuous_web 的网页脚本测试）。采集 +40 ms 与独立 SciPy 稠密 LIO 插值的位置 1e-9 m / 旋转 1e-9° 一致性测试通过。

R1（2026-10-29，restructure 代码，本机 Noetic 容器 1× 回放 office_gs_02.bag，5 Hz / 40 帧 / 10 共享 / Omega 384）：

| 判据 | 实测 | 结论 |
| --- | --- | --- |
| 窗口全部成功 | 19 / 19（失败 0、丢弃 0） | 通过 |
| 回放中无积压 | 待处理窗口峰值 1，bag 结束后清空 | 未通过 |
| 在线融合点云接口 | 10 次 200、1 次 409（首个全局修订之前）；单次 17–42 s，随窗口数增长 | 可用，未增量化 |
| 交付包 | 580 帧中 6 帧无定位不导出，574 帧；4 帧共享 CameraInfo；1,208 个文件哈希核对通过；导出 79 s | 通过 |
| 落地 BA：有观测帧 ≥97%、相邻 Sampson P50 ≤0.7 px | 98.8%、0.679 px（P90 1.236 px）；相对 LIO 中心 P50 / P90 9.2 / 18.3 cm | 通过 |
| 落地 BA 耗时与内存（本机 CPU 4 线程） | 1,842 s（SIFT 提取 92 s、匹配 95 s、XFeat 1,348 s、全局 BA 299 s），峰值 RSS 3.6 GiB | 已记录 |
| 在线点云对雷达局部平面 P50 ≤3.555 cm、P90 ≤7.775 cm、无参照 ≤25.7% | 3.80 cm、7.72 cm、34.5%（115 帧、140,314 个有参照样本） | P90 通过，P50 与无参照未通过 |

雷达判据来自 L0（512 深度、135 帧），本次是 384 深度、不同关键帧，样本集不同，只能作量级参照。两次回放的关键帧时刻不同（前 18 窗只有 2 窗帧集合相同），窗口尺度差（中位 4%）不能归因于代码改动。NX 的 CPU 比本机慢数倍，XFeat 要放到 GPU 上跑，落地 BA 才可行。

NX aarch64 的 pycolmap/XFeat 安装、完整段 BA 耗时与 RSS、温度以及实际传感器并发尚未验证；本机通过不代表 NX 生产验收通过。

## 证据索引（本机来源标注，不依赖本地文件阅读正文）

- Data/experiments/onboard_v1_20261028/：测试日志、融合等价检查、回放与 BA 原始输出、REPORT.md。
- Data/experiments/restructure_20261029/r1/：restructure 上的 R1 回放、交付包 session_delivery_v2、落地 BA、雷达评测与 REPORT.md。
- Data/experiments/onboard_ba_20261028/REPORT.md：原 L0 雷达局部平面 P50/P90 3.555/7.775 cm，无参照 25.7%；135 帧共同内部像素，0.3<z≤4m、stride4、扫描中点 RGB+40ms±0.5s、10cm 半径、≥8 邻点、PCA 最小特征值/和≤0.05。
