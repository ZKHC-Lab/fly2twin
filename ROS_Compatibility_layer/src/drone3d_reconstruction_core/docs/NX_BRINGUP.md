# NX Step 2 一键启动说明（single-NX 连续重建）

本文档针对 Orin NX 16GB 台架机，下文用 `<nx-user>@<nx-host>` 表示它的登录账号和地址。
传感器部分（第 1–5 步）的启动顺序经过实机验证；第 6 步用的台架配置在 NX 上未运行过（`nx_step2_bringup.sh:7-8`）。**bench-only，绝不启动 mavros/PX4，绝不启动桨叶。**

## 一键启动

```bash
cd <NX 上的仓库>/ROS_Compatibility_layer
source devel/setup.bash
bash src/drone3d_reconstruction_core/scripts/nx_step2_bringup.sh
```

脚本默认的工作空间是它所在的部署目录。`tools/nx_bench/deploy_to_nx.py` 和台架配置里的路径写死为 NX 上的 `~/Drone_3D-01`：配置与 FAST-LIO 启动文件在
`~/Drone_3D-01/ROS_Compatibility_layer/run/nx_step2/`，VGGT-Omega 权重在 `~/Drone_3D-01/Data/Model/vggt_omega_1b_512.pt`。部署到别的目录时要改这些路径。

脚本按已验证顺序启动，并自带幂等检查（已在运行/已在发 topic 就跳过）：

| 顺序 | 组件 | 验证 topic | 依赖的 workspace source |
|---|---|---|---|
| 1 | roscore | `rostopic list` 可用 | `/opt/ros/noetic` |
| 2 | livox MID360 | `/livox/lidar` | `~/ws_livox/devel` |
| 3 | FAST-LIO | `/Odometry` | `~/slam_ws/devel` |
| 4 | Realsense | `/camera/color/image_raw` | 系统包 |
| 5 | body→camera TF | `tf_echo body camera_color_optical_frame` | 系统包 |
| 6 | 生产栈（capture + reconstruct + continuous session + web） | `/reconstruct/reconstruct_segment/status` | 本仓库 `devel` |

第 6 步前会自动执行 `continuous_production_preflight.py`（ROS master + CUDA + checkpoint digest）。
加 `--no-web` 可以不开 Web；路径可用 `--step2-root` / `--workspace` 覆盖。

一键脚本的生产栈按台架配置（`tools/nx_bench/profiles/`）运行：RealSense 默认 640×480@15。取帧、分窗、尺度和放置的取值与 bag 回放配置（`tools/bag/profiles/`）相同，2026-11-25 对齐：按 5 Hz 取样后按运动筛关键帧（0.15 m / 8° / 2 s），开交接缓冲（150 帧），40 帧窗、5 帧共享，推理长边 384；逐帧位姿放置（`g4_placement: frame_pose`，位姿时间偏移 +40 ms 在采集端施加，逐帧深度尺度 `g4_frame_scales: true`）。两套配置只在机器相关的项上不同（路径、解释器、端口、内存门槛、超时、深度话题）；`test_continuous_session_node.py` 逐键比对两套配置，改歪会失败。这份台架配置在 NX 实机上未运行过；NX 上跑过的 40 帧窗是下面 11-03 的那次，当时每窗约 42 s，跟不上采集。各项取值见 [机载融合与交付包](ONBOARD_V1.md) 第 1 节。

## 滑窗运行与验收

滑窗使用 `per_window`，共享帧数由 `initial_overlap_frames` 配置，会话内固定。首窗采集 N 帧，后续复用前窗尾部 O 帧并采集 N−O 帧。窗口由 `global_g4` 放置（见 [G4 接入](G4_INTEGRATION.md)，主线是其中的逐帧位姿放置）。

运行中 `lossless_fifo` 达到 `queue_capacity` 水位后暂停新采集，消费释放容量后恢复；已启动采集和重试保留。暂停期间到达的图像全部丢失，交付包和落地 BA 也就缺了这一段，因此模板默认容量为 1000，实际只受 `min_free_disk_gb` 磁盘红线限制：重建慢于采集时，积压的窗口留在磁盘上排队。2026-11-03 NX 实测（15W，384 / 40 帧窗）：每窗处理中位 52.5 s（GPU 推理 38 s），新窗口每 15.8 s 一个，10 分钟积压 27 窗；去掉每窗重复的权重哈希后每窗约 42 s。落地后按 STOP：采集立即停止，会话进入 `DRAINING`，把排队窗口重建完、等 G4 发布完整地图后自动停下（2026-11-03 起；之前的版本 STOP 会删除所有还没重建的窗口）。处理积压期间 Web 按钮变为“立即停止”，再按一次会丢弃剩余窗口。2026-11-03 本机回放加写盘压力时，重建降到约 40–50 s/窗，容量 8 的队列填满，窗口之间出现 15–48 s 的采集空档。队列只有 `lossless_fifo` 一种，共享帧数和其它会话参数在运行中不能修改。v4 合成输入保留相机参数、odom、TF 和可选传感器深度。

按 5 FPS 时间采样时，每窗新帧数 = 窗口帧数 − 共享帧数：16 帧窗、5 帧共享时名义产窗间隔约 2.2 s，40 帧窗时约 7 s；开启运动关键帧后（台架和 bag 回放配置都开）按实际选帧间隔计。日志 `window_service_timing` 的 `service_wall_time_sec` 覆盖每次消费的推理、校验、发布与正常清理；`service_budget_sec` 是该窗口冻结参数对应的名义产窗间隔。还需检查 FIFO 最老等待时间；满载暂停不代表硬件已达到 5 FPS 连续建图吞吐。每个窗口各自产出正式的局部结果，跨窗口的放置由 G4 在后台完成，不计入这个时间。

## 外部里程计模式（不自起 FAST-LIO）

如果外部系统（其他机器上的 FAST-LIO、VIO、RTK-INS 等）已经发布 `nav_msgs/Odometry`，
可以让算法完全不启动自己的 FAST-LIO：

```bash
bash src/drone3d_reconstruction_core/scripts/nx_step2_bringup.sh \
  --external-odom /Odometry
```

- topic 不是 `/Odometry` 时直接传实际 topic，例如 `--external-odom /vins/odom`。
- 脚本会等待该 topic 在线后继续；**不会修改原始 `capture_profile.front.yaml`**，
  而是生成 `capture_profile.front.external_odom.yaml` 供生产栈使用。
- 外部 Odometry 要求：标准 `nav_msgs/Odometry`，`child_frame_id=body`，
  `frame_id` 稳定非空（如 `map`/`odom`），与相机同一 ROS 时间源，频率建议 ≥10Hz。
- 外部里程计完全独立时，可以连本地 livox 驱动一起关掉：
  `--external-odom TOPIC --no-livox`。

不带 `--external-odom` 时行为不变：脚本自己启动本地 FAST-LIO。

日志全部在 `<step2-root>/`：

- `roscore.log`
- `livox_step2.log`
- `fastlio_step2.log`
- `realsense_pausetest.log`
- `body_camera_tf_YYYYMMDD.log`
- `production.log`

## 录制离线重建数据（3DGS 用）

只起传感器和 TF，不起生产栈，录原始 bag 回本机处理。脚本默认彩色 640×480@15；换彩色分辨率时脚本要求同时加
`--no-production`（`nx_step2_bringup.sh:85-86`）。

```bash
cd <NX 上的仓库>/ROS_Compatibility_layer && source devel/setup.bash
S=src/drone3d_reconstruction_core/scripts
bash $S/nx_step2_bringup.sh --no-production --color 1280x720@30 --rgb-exposure-ms 10 --rgb-gain 96
python3 $S/camera_check.py                  # 先静止看亮度，再按飞行速度转动看清晰度
bash $S/nx_record.sh --name office_gs_02    # Ctrl-C 停止，默认存到 ~/bags/
```

- 曝光或增益不合适时，换参数重跑第一行即可：相机已在同一分辨率运行时脚本只重设曝光。相机已在别的分辨率运行时脚本
  报错退出，需要先停掉相机。`--rgb-exposure-ms` 按 D400 彩色传感器的 100 µs 单位换算，通过 `dynamic_reconfigure`
  写入 `/camera/rgb_camera`（同时关自动曝光）并回读。
- `camera_check.py` 输出分辨率、实测帧率、亮度中位、清晰度（降到不超过 640 宽后的拉普拉斯方差）。参照 testflight3 关键帧：
  亮度中位 123（P10–P90 97–148），清晰度中位 110；BA 中匹配不到特征点的模糊帧中位 31，匹配充分的帧 174。
  手动曝光时还从 `/camera/color/metadata` 输出实际曝光和增益（自动曝光时元数据不报这两项）。噪声也会抬高清晰度：
  台架上亮度相近时，增益 64 为 164，增益 128 为 406，所以清晰度只在同一增益下比较（静止一次，按飞行速度转动一次）。
- `nx_record.sh` 录制：彩色图与 `camera_info`、`/camera/color/metadata`、对齐深度与其 `camera_info`、`/Odometry`、
  `/livox/lidar`、`/livox/imu`、`/tf`、`/tf_static`；`--no-depth`、`--no-lidar` 可去掉对应话题。
  - **压缩：** 彩色默认录 `image_transport` 的 JPEG（`/camera/color/image_raw/compressed`）。脚本开录前把质量设为 95；深度录无损 PNG（`/compressedDepth`，压缩级别 1）。
    `--raw` 恢复录原始图像。bag 本身不压缩。
  - **输入缓冲 3 GB：** 默认 256 MB 在约 100 MB/s 下不到 3 s；office_gs_01 中彩色流丢过一次 1.7 s。
  - 开录前脚本打印剩余空间。
- `nx_capture.sh` 一键录制时加 `--no-depth`：3DGS 用 VGGT-Omega 深度，不用 D435i 深度。
- 读取压缩 bag：`cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)` 得到 BGR 图像，可直接 `cv2.imwrite`。
  原有按 `sensor_msgs/Image` 解码的抽帧脚本要相应修改。

参数依据（离线 3DGS 实验结论）：

| 参数 | 取值 | 依据 |
|---|---|---|
| 彩色分辨率 | 1280×720 | testflight3 为 640×480，3DGS 训练视角只到约 25 dB；同一办公室手机视频（960×540）在相机一致的局部片段上训练视角 33–40 dB。本机 flight-out2 已在该无人机上以 1280×720 录到 29.8 Hz（同时有对齐深度） |
| 帧率 | 30 fps | 自动曝光的最长曝光受帧间隔限制：15 fps 下可到 66 ms，30 fps 下 33 ms |
| 曝光 | 手动 10 ms | 旋转拖影 ≈ 角速度 × 曝光 × 焦距，1280×720 焦距约 906 px。转向不超过 10°/s 时 10 ms 拖影约 1.6 px |
| 增益 | 96 | 台架实测亮度中位（testflight3 为 123）：5 ms 时增益 64 为 42、128 为 71；8 ms、128 为 90；10 ms 时增益 64 为 65、128 为 103；自动曝光为 137。增益 128 时白墙上可见颗粒噪声；10 ms、96 预计约 85。`camera_check.py` 报出的亮度低于 80 时，手动把增益提到 128 |
| 飞行 | 转向不超过约 20°/s | 两次办公室录制合并，按测试帧角速度分档的 3DGS PSNR（固定位姿）：< 5°/s 24.3 dB（11 帧）、5–10 23.5、10–20 23.5、20–40 22.6、≥ 40 21.8 dB。20°/s 以下基本持平 |

台架验证（2026-10-16，realsense2_camera 2.3.2 / librealsense 2.50，相机接 USB 3.0）：`/camera/rgb_camera` 的
`enable_auto_exposure`、`exposure`、`gain` 可写可读回；元数据 `actual_exposure` 与设定值一致，曝光 5 → 10 ms 时亮度
增加 1.45–1.55 倍，符合 gamma 编码下光量翻倍。试录 19.6 s：彩色 29.8 Hz 不掉帧，对齐深度 9.9 Hz，`/Odometry` 100 Hz，
`/livox/imu` 200 Hz，`/livox/lidar` 10 Hz，元数据逐帧都有；1.9 GB，约 5.8 GB/min，NX 剩余 65 GB 约可录 11 min。

压缩录制台架验证（2026-10-22）：
- **画质：** JPEG 95 与原图相比 PSNR 41.8 dB，在 ctrl_wall_01 的 1280×720 帧上离线测得，低于约 39.5 dB 的传感器噪声上限。
- **NX 实录 16 s：** 彩色 456 帧全部可解码；单帧约 380 KB；彩色加激光雷达合计 14.3 MB/s，约 0.86 GB/min，原始录制为 5.8 GB/min。剩余 64 GB 约可录 70 min。
- **CPU：** 相机节点做 JPEG 编码约占一个核心的 68%，整机 CPU 约 30%。
- **帧率：** 当时彩色只有 27.4–28.2 Hz（原始流同样），深度流报 `Depth stream start failure`。同一时刻另有 `drone_bringup drone_d435.launch`（`vision_ws` 的 `rs_bridge`）在另一个 ROS master（11312）上直接打开相机，两个进程争用设备。录制前先停掉它，再重启相机。

飞行验证（office_gs_02，2026-10-19，10 ms、增益 96、刻意放慢转向）：彩色 6991 帧 29.9 Hz 无丢帧；关键帧亮度中位 75（office_gs_01 为 90），
高频残差噪声 1.87（office_gs_01 为 2.83，0–255）；关键帧角速度中位 13.8°/s（office_gs_01 为 21.9°/s）；3DGS 全部测试帧 PSNR
固定 / 对齐 23.5 / 24.2 dB，比 office_gs_01 高 1.3 dB（两次录制航线不同，同时改了里程计去重，不能拆成各项单独收益）。

## 为什么这个顺序不能乱

1. livox 和 FAST-LIO 的包不在 `/opt/ros`：必须分别 source `~/ws_livox/devel` 和 `~/slam_ws/devel`，否则 `roslaunch` 报 `is neither a launch file ...`。
2. FAST-LIO 的 `/Odometry` 是 front 窗口的姿态来源；capture 配置里 `odometry_topic: /Odometry`。
3. `body → camera_color_optical_frame` 的静态外参用 2026-10-23 的重标值（D435i 与局部雷达配准，三次录制合并），推导、验证与影响见 [机体→相机外参](../../../docs/CAMERA_EXTRINSIC.md)。没有它 TF 树断成两棵，每个窗口都会回退到 `MODEL_LOCAL_UNALIGNED`。发布命令（脚本第 5 步）：
   ```bash
   rosrun tf2_ros static_transform_publisher \
     0.056558 -0.006717 -0.077751 -0.494176 0.506142 -0.498990 0.500618 \
     body camera_color_optical_frame
   ```
   原值 `0.329365 0.033964 -0.079647`（四元数 `0.502057 -0.498710 0.500910 -0.498314`）来自 FAST-LIVO2 `Rcl/Pcl` 取逆，并假设 `body==lidar`。它沿相机光轴偏约 27 cm，2026-10-23 之前录的 bag 里存的都是原值，回放时要覆盖。
4. 生产栈启动前必须已有 ROS master；`continuous_production.sh` 自己不会起 roscore，也不会起任何硬件组件。

## 已知问题与恢复

- **开机后 camera 进程在但无 image topic**：脚本检测到该状态会自动重启一次 camera 栈；若仍无 topic，重新插拔 USB 再跑一次。
- **不能重复启动生产栈**：脚本用 `pgrep` 检测 `continuous_production.launch`，已在运行就跳过；重复启动会造成节点名冲突。
- **NX 重启后全套栈不会自动恢复**：重新跑一次一键脚本即可（脚本各步骤都是幂等的）。

## Web UI

- NX 本机：http://127.0.0.1:8080
- 开发机隧道（录屏时用）：
  ```bash
  ssh -L 8080:127.0.0.1:8080 <nx-user>@<nx-host>
  ```
  然后打开 http://127.0.0.1:8080

## 停止 / 重启生产栈

```bash
# 停止生产栈（会等待活跃目标收敛；不会停硬件组件）
pid=$(pgrep -f 'roslaunch drone3d_reconstruction_core continuous_production.launch' | head -1)
kill -INT "$pid"
```

硬件组件（roscore/livox/FAST-LIO/camera/TF）通常不用停；确需全停时再单独处理对应进程。

## warm worker 相关

- `reconstruct_profile.yaml` 里 `warm_worker_enabled: true`，模型跨窗口常驻。
- warm worker 进程日志：`<step2-root>/worker_results/.warm-worker.log`。
- 常驻模型约占 4.3 GB RSS，因此该 profile 的 `min_mem_available_bytes` 已调到 3 GiB；资源闸门逻辑本身不变，每个窗口仍会真实采样判定。
