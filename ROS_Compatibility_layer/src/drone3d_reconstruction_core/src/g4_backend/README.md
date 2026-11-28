# G4 独立核心

这个包是连续会话地图后端的纯数值部分：逐帧外部位姿放置（`frame_pose`）和它用到的几何与 Sim(3) 运算。
核心依赖 NumPy、SciPy，以及仓库根的 `recon_core`（窗口尺度和逐帧尺度的数值实现在那里，与离线端共用）。
核心读取数组和数据对象。
ROS 消息、磁盘缓存、模型推理、线程调度和地图发布由调用方负责（`drone3d_reconstruction_core.g4`）。

早期的 G4 图配准（`legacy_g4_20260916`：E1 共享点配准、定位因子、每窗一个 Sim(3) 的整图求解）和长窗分组的窗口清单
在 2026-11-23 删除，代码可从提交 `f0c0aa3` 恢复。

## 安装和核查

在仓库根目录执行。`recon_core` 没有写进本包的依赖声明，需要它能被导入，仓库根在路径上即可。

```bash
python -m pip install ./ROS_Compatibility_layer/src/drone3d_reconstruction_core/src/g4_backend
python -m unittest discover -s ROS_Compatibility_layer/src/drone3d_reconstruction_core/src/g4_backend/tests -v
```

## 核心模块

| 模块 | 职责 |
|---|---|
| `types.py` | `Sim3`、`WindowGeometry` 两个数据契约和数值验证 |
| `sim3.py` | Sim(3) 复合与作用于点 |
| `geometry.py` | 预测转为首相机规范、像素中心约定下的内参缩放 |
| `frame_pose.py` | 逐帧外部位姿放置：配置校验、位姿插值、窗口平移，转发 `recon_core.scale` 的窗口尺度与共享帧深度比；方法身份 `frame_pose_window_scale_20261008`。逐帧尺度由 `g4/map_backend.py` 直接调用 `recon_core.scale.solve_frame_scales` |

## 数据契约

`Sim3` 使用列向量约定：`target = scale * rotation @ source + translation`。
批量点以 `N×3` 行存储。
`apply` 内部计算 `scale * points @ rotation.T + translation`。

`WindowGeometry` 包含本窗口的帧身份、时间戳、归一化深度、置信度、内参、相机朝向和光心。
`rotations` 表示相机坐标到窗口坐标。
`centers` 表示窗口坐标中的相机光心。
`intrinsics` 使用模型图像的像素中心坐标。
`depth` 表示相机 z 轴深度。
`shared_prefix` 表示新窗口开头的历史共享帧数。
每个 RGB 帧应使用稳定的物理帧身份。

`from_prediction` 接收世界到相机的 `N×3×4` 外参。
它将所有几何转到首相机坐标。
它使用首帧中央 60% 区域的深度中位数统一长度单位。
384×288 时，该区域恰好为 `[57:230, 76:307]`。
ROS optical/body 坐标转换和查看器的轴翻转不属于此函数。

## 工程与运行边界

窗口数量和窗口身份由调用方提供。
核心不会读取文件、创建 ROS 节点、启动进程或退出宿主程序。
