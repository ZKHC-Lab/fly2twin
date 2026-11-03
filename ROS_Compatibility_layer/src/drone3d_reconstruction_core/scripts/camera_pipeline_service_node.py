#!/usr/bin/env python3
"""ROS1 上层相机管线服务的 catkin 可执行入口。"""

from drone3d_reconstruction_core.pipeline_service_node import main

if __name__ == "__main__":
    raise SystemExit(main())
