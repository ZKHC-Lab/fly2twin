# 公共运行时

该包拥有 `reconstruction_runtime` 的 Omega 结果校验器和 `drone3d_resources` 的 GPU 进程锁。
Python 实现不依赖 rospy 或 CUDA，由 catkin 安装，供核心包及 Omega 工作进程使用。
源码边界和入口见 [模块索引](../../docs/ARCHITECTURE.md)。

测试使用 `catkin_make run_tests`。
权重通过显式绝对路径和 SHA256 指定，权重不打包进入源码。
