# ROS 兼容层

ROS1 Noetic / catkin 工作空间，无人机机载端的连续重建：相机采集与运动关键帧 → 快照封存 → 分窗 VGGT-Omega 推理 → 后台放置与尺度求解 → 增量多视融合点云 → Web 查看；落地后导出交付包并做全局 BA。

当前主线用逐帧位姿放置（`g4_placement: frame_pose`）：每帧按位姿来源给出的相机位姿放置，VGGT 提供深度，后台求窗口尺度和逐帧尺度。每窗 40 个关键帧，其中 5 个与上一窗共享，推理长边 384。配置见[机载融合、交付包与落地全局 BA](src/drone3d_reconstruction_core/docs/ONBOARD_V1.md)。

配置模板的默认值仍是早期的 G4 图配准（`g4_placement: graph`）：16 帧窗口，11 个新帧加 5 个共享帧，见 [G4 接入与性能说明](src/drone3d_reconstruction_core/docs/G4_INTEGRATION.md)。后台完成求解后原子更新历史窗口。

## 目录

- `src/drone3d_reconstruction_msgs/`：Capture、Reconstruct Action 与管线/会话控制服务。
- `src/drone3d_reconstruction_core/`：采集、worker 管理、资源准入、结果校验和会话调度。
- [配置模板](src/drone3d_reconstruction_core/config/)：采集、重建、管线、连续会话 YAML。
- NX 台架工具 `tools/nx_bench/`：台架配置、基准测试、监控与同步脚本（只适用于内部台架，开源导出不包含）。

## 文档

- [模块边界与安装](docs/ARCHITECTURE.md)
- [环境与部署边界](docs/ENVIRONMENT.md)
- [G4 算法基线与移植规范](docs/G4_BASELINE_SPEC.md)
- [长窗分组(100/120 与尾窗规则)](docs/LONG_WINDOW_GROUPING.md)
- [3DGS 离线重建(位姿精化、初始点云、评测)](docs/GS_OFFLINE_RECONSTRUCTION.md)
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
一键脚本的生产栈按台架配置运行：640×480、16 帧窗、G4 图配准。机载主线的配置见 [机载融合与交付包](src/drone3d_reconstruction_core/docs/ONBOARD_V1.md) 第 1 节，目前只在 bag 回放中验证过，还没有接入一键脚本。
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

- 按模板填写绝对路径、checkpoint 摘要、设备 SKU、资源阈值及快照限制；替换全部占位值。
- ROS1 中超过 `2147483647` 的字节阈值须写成带引号的十进制字符串，如 `"8589934592"`。
- front/down 使用独立采集配置；相机角色、topic、frame_id 在配置中固定。
- `PipelineCommand`：`SELECT_SOURCE / START_TASK / GET_STATUS / CANCEL_TASK`。
- `ContinuousSessionCommand`：`SELECT_SOURCE / START_SESSION / PAUSE / RESUME / STOP_ALL / GET_STATUS / SET_PARAMS`。
- `/diagnostics`：就绪状态、任务阶段、资源占用、队列及错误；CUDA 重建并发上限为 1。
- 快照支持 schema v1–v4；结果经校验后原子提交，取消/超时负责清理 worker 进程组。

## 工具

- `bash tools/check.sh validator`：真实结果校验与原子提交检查；`bag`、`web` 子命令见 `--help`。
- `bash tools/nx_bench/bench/run100.sh --help`：参数化基准测试，支持 `--dry-run`（内部台架工具，开源导出不包含）。
- 安装：在上述构建与测试通过后执行 `catkin_make install`，新终端加载 `install/setup.bash`。

## 模式与限制

- 会话创建时选择模式，运行中禁止切换；停止后可创建另一种模式的新会话。
- `per_window`：默认G4全局配准。G4会话内固定输入配置。旧 `odom_pose_constrained_sim3` 模式仍支持逐窗里程计对齐和运行时重叠调整。
- `g4_placement: frame_pose`：当前主线，G4 会话改为逐帧外部定位位姿加 VGGT 深度尺度，见本页开头和[机载融合与交付包](src/drone3d_reconstruction_core/docs/ONBOARD_V1.md)。早期以 D435i 参考面做的验证见 [G4 接入](src/drone3d_reconstruction_core/docs/G4_INTEGRATION.md) 的“逐帧位姿放置”。
- `frame_pool`：重叠固定为 0 且不可修改；不进行对齐；Web 仅显示最新一次重建点云。
- `window_frames` 是采集及帧数池重建的目标帧数，可调整；合格帧不足时继续入池等待。
- 采集完成立即入池；每次重建和重试重新按质量、位置、朝向和时间覆盖选帧，允许旧优质帧再次入选。
- 目标帧数受重建配置及快照上限约束；坏图不参与选帧，无朝向帧按质量与时间覆盖选择。
- CUDA OOM 进入 `FATAL_OOM` 并停止采集与消费；降低帧数或分辨率后人工重启。
- 快照根、结果根及其父目录必须可信、稳定；NX 候选配置不代表已通过生产验收。
