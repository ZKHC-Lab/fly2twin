# ROS 模块边界

本目录拥有运行时、ROS 适配和部署入口。模型权重作为显式外部资源配置。

G4 图配准是早期做法，2026-11-23 已删除（可从提交 `f0c0aa3` 恢复）；现行的逐帧位姿放置见[机载融合与交付包](../src/drone3d_reconstruction_core/docs/ONBOARD_V1.md)。

| 模块 | 源码位置（相对本目录） | 职责 |
|---|---|---|
| 消息合同 | src/drone3d_reconstruction_msgs | Action、Service、状态与结果消息 |
| 采集与监督 | src/drone3d_reconstruction_core/src/drone3d_reconstruction_core | 采集、插值、快照、任务生命周期、连续会话、Web |
| G4 数学核心 | src/drone3d_reconstruction_core/src/g4_backend | 逐帧位姿放置（frame_pose.py，数值在 recon_core）、Sim(3) 与窗口几何 |
| 共用核心 | src/drone3d_reconstruction_core/src/recon_core（链接到仓库根的 recon_core） | 窗口尺度与逐帧尺度、回访约束、多视融合，与离线端共用 |
| 机载交付 | src/drone3d_reconstruction_core/src/drone3d_onboard | Web 在线增量融合、交付包导出、落地全局 BA（导出和 BA 都只有手动 `python -m` 入口；bag 回放工具会自动调用导出） |
| G4 输入适配 | 同核心包的 g4/input_adapter.py | 校验输入身份，转换到数学核心的数据结构 |
| G4 地图后端 | g4/map_backend.py | 缓存每窗尺度约束，按逐帧位姿生成完整地图修订 |
| G4 后台进程 | g4/backend_process.py | 会话拥有的进程生命周期 |
| G4 地图快照 | g4/map_snapshot.py | 将同一修订应用于历史正式记录 |
| 公共运行时 | src/drone3d_runtime/src/reconstruction_runtime | Omega 结果校验器 |
| GPU 仲裁 | src/drone3d_runtime/src/drone3d_resources/ | 模型工作进程共用的进程锁。模块不加载 ROS、Torch 或模型 |

G4 的 Python 导入名保持 `g4_backend`。运行时导入名保持 `reconstruction_runtime`。
两个目录都通过 catkin 安装。

## 安装与入口

从 ROS 工作区运行 `catkin_make install`。
新终端先 source `install/setup.bash`。
G4 默认入口为 `rosrun drone3d_reconstruction_core continuous_production.sh`。

ROS CPU 解释器需要 NumPy>=1.20、SciPy>=1.7。
CUDA 解释器与 ROS 解释器分别配置。
监督进程清除传给 CUDA 进程的 ROS PYTHONPATH。
监督进程只通过 `DRONE3D_RUNTIME_PYTHON_ROOT` 传递纯 Python 共享资源模块的安装目录。

Omega 模型代码仍作为外部模型依赖，由 `reconstruct_script`、`warm_worker_script` 和 `validator_script` 显式指定。
它们的 GPU 仲裁调用使用已安装的公共运行时。
权重的路径和 SHA256 保持显式配置。NX 台架配置（`tools/nx_bench/profiles/`）是具体设备示例，不是可直接复制到任意机器的通用配置。
