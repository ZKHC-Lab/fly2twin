#!/usr/bin/env python3
"""生产采集监督节点的 ROS1 可执行入口。

本脚本是 catkin 安装后的可执行入口（install/ 下的同名文件由 setup.py
生成）。实际逻辑在 capture_node.main：初始化 rospy 节点、从私有参数
构造 CaptureNode 并进入 spin 循环。
"""

from drone3d_reconstruction_core.capture_node import main

if __name__ == "__main__":
    raise SystemExit(main())
