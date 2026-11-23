# ROS 环境与部署边界

ROS CPU 解释器和模型 CUDA 解释器分开安装。
2026-09-23 审计时的本机验证环境为ROS1 Noetic、Python3.8.10、NumPy1.24.4、SciPy1.10.1和OpenCV4.2.0。
模型环境为Python3.10.20、PyTorch2.7.1+cu128和torchvision0.22.1+cu128。
GPU为RTX5060 Laptop。以上CUDA版本不作为NX安装指令。
完整版本证据位于 `Data/experiments/_archive_gpt/ros_deployment_audit_20260923/evidence/`。

## CPU 数值依赖

G4启动前检查NumPy>=1.20和SciPy>=1.7。package.xml同步声明最低版本。
Noetic系统默认软件包可能低于门限。`rosdep install` 不能替代启动检查。
`requirements-ros-cpu.txt` 固定本轮测试的数值依赖版本。

在目标ROS CPU解释器下，将这些wheel装入独立目录。例如：

```bash
python3 -m pip install --only-binary=:all: --target /absolute/ros-cpu-deps \
  -r /absolute/Drone_3D-01/ROS_Compatibility_layer/requirements-ros-cpu.txt
export PYTHONPATH=/absolute/ros-cpu-deps:$PYTHONPATH
```

使用能够提供目标Python版本和架构wheel的环境。
当前测试没有验证新的NX环境安装。目标环境的wheel、CUDA与系统库兼容性需要在部署阶段确认。
本文件提供已验证配置和CPU依赖清单。它不提供跨设备的一键安装保证。

## 安装和模型资源

从ROS工作区运行 `catkin_make install`。运行前 source `install/setup.bash`。
模型脚本、CUDA Python与权重使用重建profile中的绝对路径和权重SHA256。
配置模板中的 `/REPLACE_ME` 必须替换。
`continuous_production.sh` 执行CPU和CUDA启动检查。
CUDA子进程不继承ROS的PYTHONPATH。共享GPU锁模块通过安装目录单独传入。

ROS 部署从本页和 [ROS 配置入口](../README.md)开始；离线流程依赖见 [recon 安装说明](../../recon/docs/INSTALL.md)。
NX aarch64 的 PyTorch wheel 应按目标 JetPack、Python 和 CUDA 版本选择，不能直接套用本机 x86_64 测试配置。

清理实验目录前保留正在使用的权重。当前代码整理没有移动或删除权重。

## GPU 与存储边界

GPU锁用于同一UID、同一文件系统锁路径下的模型进程。
默认路径为 `/tmp/drone3d-gpu-<uid>-0.lock`。
不同UID或隔离的容器 `/tmp` 不共享这把锁。
当前本机测试使用一个生产容器、同一UID和同一锁路径。
部署多个容器时必须另行提供共享仲裁设计。此实现不提供跨容器GPU调度。

G4保留已提交图像、预测和图修订。会话不会自动清理，导出后可删除的目录见[机载融合与交付包](../src/drone3d_reconstruction_core/docs/ONBOARD_V1.md)第 3 节。
原始来源用于身份验证、重新渲染和重新构造约束。
相机与共享点缓存用于图优化。历史修订主要用于复现与诊断。
代码目前没有把这些对象分级回收。
有限航线可按测得增长预留空间。长时间运行需要独立存储方案。
本轮保持数据保留策略，不引入自动删除。
