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
initial_overlap_frames: 5
initial_inference_resolution: 384
reconstruct_frame_group_size: 40
initial_keep_snapshot_on_success: true
initial_queue_mode: lossless_fifo
frame_pose_fusion: true
g4_frame_scales: true
```

重建 profile 设置 `image_resolution: 384`、`frame_group_size: 40`、`load_mode: max_size`。1280×720 输入对应 384×224 模型深度。保留完整原图快照，不能只留模型缩图。

共享帧取 G4 允许的最小值 5（2026-11-03 起，之前为 10）：每窗 40 帧中 35 帧是新帧，同样的关键帧需要的窗口少约 14%，NX 上新帧吞吐高约 17%。本机 A/B（office_gs_04 第 60–660 s，1300 个关键帧，交付点云对 office_gs_04 雷达参照地图的全部点最近邻距离，恒等变换）：

| 配置 | 窗口 | P50 | P90 | <5 cm |
| --- | --- | --- | --- | --- |
| 共享 10 | 43 | 3.31 cm | 12.20 cm | 70.1% |
| 共享 5（两次） | 37 | 3.32 / 3.22 cm | 12.57 / 11.87 cm | 70.5 / 72.6% |
| 共享 5，推理 288 | 37 | 4.10 cm | 21.95 cm | 56.4% |

共享 5 与 10 的差异在两次重复的波动之内；全部窗口都有相邻尺度边。288 明显变差，误差集中在少数窗口的整体深度尺度上（3 窗中位误差 17–28 cm），384 下同样的窗口也偏（按雷达拟合的逐窗尺度修正中位 2.7%、P90 9.4%），说明在线窗口尺度估计是当前精度的主要瓶颈。内参与 288×160 缩放一致，不是内参错误。

**跨窗口重叠约束**（`g4_cross_window_weight`，2026-11-04 起回放模板为 0.3；0 = 只用轨迹先验与相邻共享帧）。G4 每缓存一个窗口，就在它的拥有帧和之前窗口的拥有帧之间找融合邻居（`recon_core.fusion.ranked_partners`，1.5 m、30°、每帧前 6 个，对称），用截至该窗口求出的尺度把帧 i 的深度按定位位姿投到帧 j，与帧 j 深度比较（`recon_core.scale.cross_view_log_ratio`，步长 4，目标深度 0.3–4 m，|log 比| < 0.25，> 500 模型像素），每个有方向的帧对一行，存进该窗口缓存的 cross.npz；之后每次修订把所有行（权重 0.3）与轨迹先验（权重 min(范围, 2)）、相邻共享帧边（权重 3）一起求对数最小二乘。行只取决于已提交的输入，与处理顺序和进程重启无关。要求 `pose_time_offset_stage: capture`（行用的就是放置用的位姿）。每窗另存步长 4 的深度，约 1 MB。

验证（office_gs_04；“真值”为逐窗按雷达参照拟合的尺度，交付点云距离同上）：

| 会话 | 对数尺度误差 中位 / P90 | 点云 P50 / P90 | <5 cm |
| --- | --- | --- | --- |
| 10-29 r1_run7 118 窗，共享 10，原求解 | 0.030 / 0.080 | 3.32 / 11.17 cm | 71.5% |
| 同上，同一批窗口重跑 G4 加跨窗口行 | 0.027 / 0.066 | 3.20 / 10.17 cm | 74.3% |
| 11-04 新回放 37 窗，共享 5，在线开启 | — | 3.03 / 8.54 cm | 79.7% |
| 同一会话重跑 G4 不加跨窗口行 | — | 3.07 / 8.85 cm | 78.3% |
| 118 窗按雷达拟合尺度（上限） | 0 | 2.99 / 7.76 cm | 81.1% |

权重 0.3 在另一段 37 窗回放上选定，118 窗是留出验证。本机 G4 后台单线程处理 118 窗 106 → 218 s（每窗约 +0.95 s）。NX（15W）3 min 回放 9 窗：G4 单独重跑 9.5 → 15.0 s（每窗约 +0.6 s，566 行），在线主流程每窗处理中位 42.4 s 不变（G4 在单独进程里异步运行）。同一配置不同次回放的点云 P90 在 8.9–12.6 cm 之间（关键帧和窗口划分不同，窗口尺度误差随之变化），所以只用同一批窗口的配对比较下结论。剩余差距主要在白墙等无纹理区域：VGGT 的深度与它自己的相机运动一起偏，相邻窗口也一起偏，现有约束都看不出来。

**逐帧深度尺度**（`g4_frame_scales`，2026-11-06 起回放模板开启，与 `g4_cross_window_weight` 二选一）。按雷达逐帧拟合，同一窗口内各帧的尺度偏差中位 2.9%（标准差），窗口尺度修不了这部分：最差的窗口与相邻窗口的重叠部分一致，误差在其他窗口看不到的那部分深度上。做法：每帧一个尺度 log s_f = x_w + d_f，用帧对深度比把同窗与跨窗口的帧直接互相约束，`recon_core.scale.solve_frame_scales` 稀疏最小二乘求解。

- 行：与跨窗口重叠约束相同的测量（融合邻居、`cross_view_log_ratio`、步长 4），但同一窗口内的帧对也测；每个有方向的帧对一行。
- 噪声（office_gs_04 118 窗，按雷达逐窗拟合尺度实测的稳健标准差）：轨迹先验 0.040、相邻共享帧边 0.040、窗口对聚合的重叠比 0.039；单行取 0.08（同一对窗口的行相关），帧偏离窗口尺度 0.04，窗内相邻帧的偏离变化 0.02。每条约束权重 1/σ。
- 修订每窗多一个 `frame_depth_scales`（与 `depth_scale` 同一 gauge；共享前缀、无位姿的帧为 null）。地图快照、交付包融合与导出（windows/NNNNNN.npz 的 `frame_depth_scale`）、在线增量点云、Web 旧显示路径都优先用每帧尺度，没有时用窗口尺度。
- 按 VGGT 置信度给像素加权本身不提高测量精度（只用双方置信度高于中位数的像素、或按置信度乘积加权中位数，测量误差 0.029–0.030 vs 中位数 0.028）；有用的是证据的数量：一对窗口之间的帧对行越多，聚合后越准（≥ 64 行误差 0.016，≤ 4 行 0.041），逐行计入的求解自然体现这一点。

验证（office_gs_04，G4 正式代码重跑后按交付包融合路径融合；完整度 = 雷达参照点 10 cm 内有融合点的比例）：

| 会话 | 尺度 | P50 | P90 | <5 cm | 完整度 |
| --- | --- | --- | --- | --- | --- |
| r1_run7 118 窗 | 原求解 | 3.32 cm | 11.17 cm | 71.5% | 48.1% |
| | 跨窗口重叠约束 0.3 | 3.20 cm | 10.17 cm | 74.3% | 48.2% |
| | **逐帧尺度** | 3.06 cm | **8.31 cm** | 79.4% | 48.4% |
| | 逐窗按雷达拟合（窗口上限） | 2.99 cm | 7.76 cm | 81.1% | 48.6% |
| | 逐帧按雷达拟合（帧上限） | 2.76 cm | 5.56 cm | 88.4% | 48.6% |
| 留出段第 960–1560 s，38 窗（参数未调） | 跨窗口重叠约束 0.3 | 3.16 cm | 8.83 cm | 76.5% | 33.3% |
| | **逐帧尺度** | 3.06 cm | **8.27 cm** | 79.1% | 32.6% |
| | 逐窗按雷达拟合（窗口上限） | 2.90 cm | 6.88 cm | 84.3% | 35.1% |

噪声取值在 118 窗上测定，留出段是第二段未用过的录制。同一个求解器只取窗口部分时，窗口尺度误差也下降（留出段中位 3.6% → 2.8%），但点云 P90 不降（8.99 cm），收益来自逐帧一层。本机 G4 后台单线程 118 窗 218 → 258 s；求解本身 0.17 s（3470 帧、2.5 万行），主要开销是测帧对。

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

只处理 `merge_frame_indices` 拥有的帧，重叠输入不重复累计。

**在线点云按帧增量**（`drone3d_onboard.session.SessionFusion`，Web 点云接口使用；交付包导出仍用上面的整段融合 `fuse_session`）。每帧的点只取决于本帧和最多 6 个邻居的深度、窗口尺度、位姿和内参：

- 窗口第一次出现时校验一次快照、读一次预测，把拥有帧的原始深度、掩码和步长 2 的颜色存成 .npy；之后不再重复哈希、重读 PNG。
- 每帧记录本帧和邻居的输入；位姿、内参、邻居集合不变，且尺度相对缓存时变化不超过 0.5%（`scale_tolerance`），就复用该帧的点。每来一个新窗口，全局尺度求解会让几乎所有旧窗口的尺度微调（office_gs_04：大多数窗口每次 < 0.1%，每次有 3–7 个窗口 > 0.5%），因此在线结果不与整段融合逐位相同。`scale_tolerance=0` 且体素网格锚定方式相同时逐位相同，已在 118 窗真实会话的 5 / 20 / 60 / 118 窗处验证。
- 2 cm 体素用累加和维护：重算的帧减去旧点、加上新点；网格锚定在世界原点（交付包的 `fuse_session` 锚定在点云最小值），之后按同样规则自适应放大到点数上限。
- 数据都在会话目录 `fusion_cache/incremental/`（118 窗约 2.6 GB），不常驻内存，也不长期占用文件句柄。

在 10-29 office_gs_04 的 118 个修订上逐个回放（本机 CPU，4 线程）：

| 窗口数 | 增量刷新 | 重算帧数 | 整段融合 | 与整段融合的最近点距离 P50 / P99 |
| --- | --- | --- | --- | --- |
| 20 | 17.0 s | 337 | 35.6 s | 0.66 / 1.94 cm |
| 60 | 12.5 s | 353 | 108 s | 0.85 / 2.62 cm |
| 110 | 5.8 s | 39 | — | — |
| 118 | 7.1 s | 71 | 274 s | 0.86 / 3.10 cm（自适应体素 4.85 cm） |
| 60–118 全部修订 | 中位 7.9 s，最大 18.9 s | 中位 128，最大 428 | — | — |

改动前每次刷新都整段重算，约 82 ms/帧（40 窗 98 s）。早期修订尺度变化大，重算帧多；刷新时间主要由重算帧数决定（约 45 ms/帧），另有约 4 s 的固定开销（2 cm 体素累加与自适应降采样）。NX 上的耗时尚未实测。

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
- `windows/NNNNNN.npz`：每窗原始 Omega `depth` / `confidence`、`depth_scale`、`frame_depth_scale`（每模型帧一个；没有逐帧尺度时等于窗口尺度）、模型分辨率 K、原帧 ID、时间、拥有帧索引。米制深度为原始深度乘该帧尺度，不乘第二次 gauge。
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

全部 CPU 路径：SIFT、XFeat 和 Ceres；线程预算 4，BLAS 1。此阶段不持 GPU 锁。XFeat 匹配后的基础矩阵 RANSAC 在 4 线程池里与下一批匹配并行（2026-11-03 起；结果与单线程逐位相同，R1 的 1,500 对在本机 GPU 上 134 s → 41 s）。3470 帧规模时第一轮三角化内存峰值约 17.5 GB，NX 16 GB 装不下，这一规模的 BA 放到服务器上跑。逐阶段打印耗时和进程高水位 RSS（MiB）；Linux RSS 是全进程历史峰值，不是各阶段独立峰值。

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

联合回放（2026-11-03，修复采集封存、内存保护、队列容量 1000、在线点云增量之后；office_gs_04.bag 第 60–960 s，1×，选帧间隔判据 3.5 s；宿主同时每 10 s 写 512 MiB 并刷盘，回放中每 30 s 请求一次在线点云；不导出交付包）：

| 判据 | 实测 | 结论 |
| --- | --- | --- |
| 窗口全部成功 | 68 / 68（失败 0、丢弃 0），2,050 关键帧 | 通过 |
| 选帧间隔 ≤3.5 s | 最大 2.23 s，>3.5 s 空档 0 | 通过 |
| 回放中无积压 | 待处理窗口峰值 1 | 通过 |
| 在线点云接口 | 17 次 200（首窗完成前 1 次 404）；单次中位 21.8 s、最长 36.9 s（与重建 worker、写盘同时运行）；STOP 后 68 窗 9.3 s | 可用 |

10-29 用旧代码回放同一 bag：定时请求点云时第 21 窗失败（间隔 5.10 s），写盘时在第 47 窗、第 6 窗失败。整段 27 min 回放需约 24 GB 磁盘，本机空间不足，只回放了 15 min。

NX 实机回放（2026-11-03，Orin NX 16 GB，L4T R35.6.1，nvpmodel 15W（这台机器只有 15W / 7W 两档），代码 78911c9；NX 本地 office_gs_04.bag 第 60–660 s，1×，生产栈参数同上，每 30 s 请求一次在线点云；里程计来自 bag，没跑 FAST-LIO、雷达、相机驱动，另有回放专有的 JPEG 解码节点）：

| 项 | NX 实测 |
| --- | --- |
| GPU 推理（40 帧，384） | 每窗 38 s（13 次，最长 39 s），显存峰值 4.35 GB |
| 每窗总处理 | 中位 52.5 s（48.6–79.0 s，12 窗）；GPU 忙碌时间占 66%。其中约 5 s 是每窗都把 4.5 GB 权重重新算一遍 SHA-256（逐目标参数覆盖会重建配置并重跑校验），2026-11-03 已改为同一文件只算一次：3 min 回放每窗中位 48.2 → 42.2 s，GPU 忙碌占比 80% → 89%；剩余约 4.5 s 为资源准入 1.0 s、读图预处理 1.5 s、写结果 0.7 s 和校验提交 |
| 与采集比 | 新窗口每 15.8 s 一个，处理慢约 3.3 倍；10 分钟积压 27 窗 |
| 在线点云刷新 | 21–63 s（中位 30 s，1–9 窗），随窗口数变长 |
| 内存 / 温度 | 已用峰值 10.2 / 15.5 GB，交换区 12 分钟涨到 1.05 GB；结温最高 62 °C |
| 采集 | 选帧判据 3.5 s 时，第 14 窗出现 4.30 s 空档并失败；判据放宽到 10 s 时 38 窗都采集成功，已处理部分最大间隔 2.24 s。采集节点每秒实际收到的图像中位 9 张、p10 2 张（Python 处理 1280×720 原图接近上限） |
| 交付包导出 | 12 窗 343 帧 82.8 s，峰值 RSS 736 MB |

结论：按现参数 NX 跟不上实时，落地后需要排队处理（27 min 飞行估计还要 70–80 min）。测试时的版本 STOP 会删除未重建窗口；2026-11-03 起改为 STOP 只停采集、处理完积压再结束（见 G4_INTEGRATION.md「失败与存储」）。本机验证：office_gs_04 回放 5 min 加写盘压力，STOP 时还剩 5 窗，19/19 窗成功后自动停止；交付包导出前会等 G4 发布含全部窗口的修订。NX 的 worker 环境没有 pycolmap，落地 BA 未在 NX 上测；真实传感器并发（相机驱动、FAST-LIO）尚未验证。本机通过不代表 NX 生产验收通过。

## 证据索引（本机来源标注，不依赖本地文件阅读正文）

- Data/experiments/onboard_v1_20261028/：测试日志、融合等价检查、回放与 BA 原始输出、REPORT.md。
- Data/experiments/restructure_20261029/r1/：restructure 上的 R1 回放、交付包 session_delivery_v2、落地 BA、雷达评测与 REPORT.md。
- Data/experiments/onboard_20261103/：写盘压力对照（stress_old2 / stress_new / stress_new2）、联合回放 joint_900、增量融合逐修订回放 incremental_equivalence.json（逐位核对版 incremental_equivalence_v1.json）与 REPORT.md。
- Data/experiments/overlap_20261103/：共享帧 10/5 与推理 288 的 A/B 回放（eval.json、交付点云）、逐点证据与逐窗尺度分析（evidence_analysis.py、window_scale_check.py）。
- Data/experiments/nx_bench_20261103/：NX 实机回放两次的日志（tegrastats、逐进程 CPU、图像到达时间、journal）、analyze.py 汇总与 REPORT.md。
- Data/experiments/onboard_ba_20261028/REPORT.md：原 L0 雷达局部平面 P50/P90 3.555/7.775 cm，无参照 25.7%；135 帧共同内部像素，0.3<z≤4m、stride4、扫描中点 RGB+40ms±0.5s、10cm 半径、≥8 邻点、PCA 最小特征值/和≤0.05。
