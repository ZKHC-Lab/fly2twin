# ROS 模块边界

本目录拥有运行时、ROS 适配和部署入口。模型权重作为显式外部资源配置。

G4 的认可基准、数学定义、当前实现差异与逐层验收见 [G4 算法基线与移植规范](G4_BASELINE_SPEC.md)。G4 图配准是早期做法，机载主线见[机载融合与交付包](../src/drone3d_reconstruction_core/docs/ONBOARD_V1.md)。

| 模块 | 源码位置（相对本目录） | 职责 |
|---|---|---|
| 消息合同 | src/drone3d_reconstruction_msgs | Action、Service、状态与结果消息 |
| 采集与监督 | src/drone3d_reconstruction_core/src/drone3d_reconstruction_core | 采集、插值、快照、任务生命周期、连续会话、Web |
| G4 数学核心 | src/drone3d_reconstruction_core/src/g4_backend | Sim(3)、共享点配准、定位抽取、图求解；逐帧位姿放置（frame_pose.py，主线使用，数值在 recon_core） |
| 共用核心 | src/drone3d_reconstruction_core/src/recon_core（链接到仓库根的 recon_core） | 窗口尺度与逐帧尺度、回访约束、多视融合，与离线端共用 |
| 机载交付 | src/drone3d_reconstruction_core/src/drone3d_onboard | 在线增量融合、交付包导出、落地全局 BA |
| G4 输入适配 | 同核心包的 g4/input_adapter.py | 校验输入身份，转换到数学核心的数据结构 |
| G4 地图后端 | g4/map_backend.py | 缓存约束，生成完整地图修订 |
| G4 后台进程 | g4/backend_process.py | 会话拥有的进程生命周期 |
| G4 地图快照 | g4/map_snapshot.py | 将同一修订应用于历史正式记录 |
| 公共运行时 | src/drone3d_runtime/src/reconstruction_runtime | Omega 结果校验器 |
| GPU 仲裁 | src/drone3d_runtime/src/drone3d_resources/ | 模型工作进程共用的进程锁。模块不加载 ROS、Torch 或模型 |

G4 的 Python 导入名保持 `g4_backend`。运行时导入名保持 `reconstruction_runtime`。
两个目录都通过 catkin 正式安装。它们不再位于项目根目录。
同一 GPU 只保留一份锁实现。锁文件规则保持原值。

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
权重的路径和 SHA256 保持显式配置。历史 NX 配置是具体设备示例，不是可直接复制到任意机器的通用配置。

## 内部命名迁移

| 旧名 | 新名 |
|---|---|
| g4/formal_bridge.py | g4/input_adapter.py |
| g4/service.py | g4/map_backend.py |
| g4/runtime.py | g4/backend_process.py |
| g4/map_view.py | g4/map_snapshot.py |
| FormalWindowReader / LoadedWindow | WindowInputReader / WindowInput |
| BackendService / G4Process | G4MapBackend / G4BackendProcess |

内部模块名统一描述职责。ROS 话题、消息、命令、正式产物 schema 和 G4 方法编号保持原值。
旧实验脚本记录历史导入路径。重跑时应在实验副本中更新导入，不修改历史产物。

## 已移除的代码

移除没有调用者的 `_umeyama_sim3`、`_finish_success`、`_terminalize_finished` 包装器和 `_ServiceResult`。
移除未使用的预览构造参数和未使用导入。
移除 `reconstruction_runtime/gpu_lease.py` 的重复锁实现。
保留有入口或回归用途的旧配准模式。

LingBot 备选后端已于 2026-10-29 删除，可从标签 `archive/pre-restructure` 恢复。
