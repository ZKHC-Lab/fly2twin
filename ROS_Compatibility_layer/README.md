# ROS 兼容层

ROS1 Noetic / catkin 工作空间，无人机机载端的连续重建：相机采集（按时间均匀取样，可选运动关键帧） → 快照封存 → 分窗 VGGT-Omega 推理 → 后台放置与尺度求解 → 增量多视融合点云 → Web 查看。落地后可手动导出交付包；全局 BA 只有手动入口，没有启动文件调用。

当前主线用逐帧位姿放置（`g4_placement: frame_pose`）：每帧按里程计给出的相机位姿放置，VGGT 提供深度，后台求尺度。尺度默认每窗一个（模板 `g4_frame_scales: false`）；NX 台架和 bag 回放配置开了逐帧尺度（`g4_frame_scales: true`）。每窗帧数：NX 台架 16，bag 回放 40，都与上一窗共享 5 帧；推理长边 384。配置见[机载融合、交付包与落地全局 BA](src/drone3d_reconstruction_core/docs/ONBOARD_V1.md)。

逐帧位姿放置是唯一的放置方式；早期的图配准（每窗一个 Sim(3)）和长窗分组已删除，代码可从提交 `f0c0aa3` 恢复。后台完成求解后原子更新历史窗口。

## 目录

- `src/drone3d_reconstruction_msgs/`：Capture、Reconstruct Action 与管线/会话控制服务。
- `src/drone3d_reconstruction_core/`：采集、worker 管理、资源准入、结果校验和会话调度。
- [配置模板](src/drone3d_reconstruction_core/config/)：采集、重建、管线、连续会话 YAML。
- NX 台架工具 `tools/nx_bench/`：台架配置、基准测试、监控与同步脚本（只适用于内部台架，开源导出不包含）。

## 文档

- [模块边界与安装](docs/ARCHITECTURE.md)
- [环境与部署边界](docs/ENVIRONMENT.md)
- [机体→相机外参(2026-10-23 重标与验证)](docs/CAMERA_EXTRINSIC.md)
- 核心包文档:[机载融合与交付包](src/drone3d_reconstruction_core/docs/ONBOARD_V1.md)、[G4 接入](src/drone3d_reconstruction_core/docs/G4_INTEGRATION.md)、[NX 台架启动](src/drone3d_reconstruction_core/docs/NX_BRINGUP.md)
- 其他模块:[drone3d_runtime](src/drone3d_runtime/README.md)

## 构建与测试

在已安装 ROS1 Noetic 的主机或 `ROS1` 容器内执行；以下命令从仓库根目录开始：

```bash
set -eo pipefail
source /opt/ros/noetic/setup.bash
export PYTHONDONTWRITEBYTECODE=1
cd ROS_Compatibility_layer
catkin_make
source devel/setup.bash
catkin_make run_tests -j1
catkin_test_results --verbose
```

## 启动

以下命令在本工作空间根目录执行，需先加载 `devel/setup.bash`。
NX 台架需准备硬件驱动和四份配置，详见 [NX 启动说明](src/drone3d_reconstruction_core/docs/NX_BRINGUP.md)。

```bash
bash src/drone3d_reconstruction_core/scripts/nx_step2_bringup.sh
```

顺序：roscore → Livox → FAST-LIO → RealSense → TF → 生产栈；不启动飞控或桨叶。
一键脚本的生产栈按台架配置运行：RealSense 640×480@15；取帧、分窗、尺度和放置与 bag 回放配置相同（运动关键帧 0.15 m / 8° / 2 s、40 帧窗共享 5 帧、逐帧尺度、逐帧位姿放置，2026-11-25 对齐）；这套配置在 NX 实机上未运行过（`nx_step2_bringup.sh:7-8`）。各项取值见 [机载融合与交付包](src/drone3d_reconstruction_core/docs/ONBOARD_V1.md) 第 1 节。
外部里程计用 `--external-odom /Odometry`，同时禁用 Livox 加 `--no-livox`；关闭 Web 加 `--no-web`。

仅启动生产栈（要求已有可达的 roscore）：

```bash
bash src/drone3d_reconstruction_core/scripts/continuous_production.sh \
  --front-profile /absolute/path/front_capture_profile.yaml \
  --down-profile /absolute/path/down_capture_profile.yaml \
  --reconstruct-profile /absolute/path/reconstruct_profile.yaml \
  --continuous-profile /absolute/path/continuous_profile.yaml --web
```

启动前校验 ROS master、CUDA、checkpoint 路径及 SHA-256；Web 默认地址为 `http://127.0.0.1:8080`。

## 配置与接口

- 按模板填写绝对路径、checkpoint 摘要、资源阈值及快照限制；替换全部占位值。
- ROS1 中超过 `2147483647` 的字节阈值须写成带引号的十进制字符串，如 `"8589934592"`。
- front/down 使用独立采集配置；相机角色、topic、frame_id 在配置中固定。
- `ContinuousSessionCommand`：`SELECT_SOURCE / START_SESSION / PAUSE / RESUME / STOP_ALL / GET_STATUS`；会话参数来自 profile、会话内固定，`SET_PARAMS` 一律返回错误码 5。
- `/diagnostics`：就绪状态、任务阶段、资源占用、队列及错误；CUDA 重建并发上限为 1。采集侧另报辅助数据（里程计、内参、深度、TF）的丢失计数 `auxiliary_errors` / `auxiliary_dropped` / `tf_dropped`，重建侧报常驻推理进程回退成一次性进程的次数 `warm_fallback_count`；这几项不为 0 时要查原因。会话侧的 `missing_pose_frames` 是累计缺位姿帧数，缺位姿超过窗口帧数 20% 的窗口判失败（见[机载融合与交付包](src/drone3d_reconstruction_core/docs/ONBOARD_V1.md)“时间语义”）。地图后端（G4 后台进程）有单独一项 `drone3d/session/map_backend`：进程起不来、求解报错或异常退出时为 ERROR，Web 顶栏同时出现“地图后端出错”。此时采集和逐窗重建不停，窗口照常成功，只是没有被放进地图，落地后可以重算。
- 快照只支持 schema v4（采集只写 v4，读到 v1–v3 报错）；结果经校验后原子提交，取消/超时负责清理 worker 进程组。

## 工具

- `bash tools/nx_bench/bench/run100.sh --help`：参数化基准测试，支持 `--dry-run`（内部台架工具，开源导出不包含）。
- 安装：在上述构建与测试通过后执行 `catkin_make install`，新终端加载 `install/setup.bash`。

## 模式与限制

- 重建模式只有 `per_window`；`alignment_backend` 只接受 `global_g4`（名称沿用，实际做的是逐帧位姿放置，没有窗口间配准）。会话内配置固定。
- `g4_placement: frame_pose`（唯一取值）：逐帧外部定位位姿加 VGGT 深度尺度，见本页开头和[机载融合与交付包](src/drone3d_reconstruction_core/docs/ONBOARD_V1.md)。早期以 D435i 参考面做的验证见 [G4 接入](src/drone3d_reconstruction_core/docs/G4_INTEGRATION.md) 的“逐帧位姿放置”。
- CUDA OOM 进入 `FATAL_OOM` 并停止采集与消费；降低帧数或分辨率后人工重启。
- 快照根、结果根及其父目录必须可信、稳定；NX 候选配置不代表已通过生产验收。
