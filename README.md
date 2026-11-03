# 飞境 Fly2Twin

中科华创（ZKHC） · 2026

无人机增量重建、可交互孪生场景与可微视觉飞行的开源框架。无人机只需一台 RGB 相机和一个位姿来源（机载激光惯导里程计，或由图像求得的位姿），机载端边飞边分块重建点云；落地后离线精修位姿并训练 3D 高斯泼溅（3DGS）模型；重建好的场景可以在浏览器里漫游，也可以作为可微仿真的训练场景。

![框架](framework.png)

## 模块

| 模块 | 目录 | 内容 |
| --- | --- | --- |
| 无人机增量重建 · 机载 | [ROS_Compatibility_layer/](ROS_Compatibility_layer/README.md) | ROS1 Noetic：相机采集与运动关键帧、40 帧分窗 VGGT-Omega 推理、按位姿逐帧放置、在线多视融合点云、交付包导出、落地全局 BA |
| 无人机增量重建 · 离线 | [recon/](recon/README.md)、[recon_core/](recon_core/README.md) | 位姿来源插槽、图像特征光束法平差（BA）、Omega 深度、窗口尺度、多视一致性融合、3DGS 训练与评测、导出；点云到激光雷达地图的距离评测；按关键帧设计运镜、渲染并剪辑成片 |
| 可交互孪生场景 | [viewer/](viewer/README.md) | 浏览器里加载导出的 3DGS / 点云（V 键切换），键盘自由漫游，或操控一架无人机在场景里飞；一条命令打开多个场景做演示 |
| 可微视觉飞行 | 独立仓库 [DiffDrone-PyTorch3D](https://github.com/kirkchinese/Diffable_Drone_Pytorch3D)（GPL-3.0） | 用重建场景渲染深度图，轻量策略网络直接输出加速度指令，经可微动力学端到端训练 |

机载端与离线端的窗口尺度和多视融合是同一份代码（`recon_core`，NumPy/SciPy，Python 3.8 与 3.10 都可运行）。

## 方法

1. **关键帧与分窗**：相对上一关键帧平移 ≥0.15 m、转角 ≥8° 或间隔达 2 s 时取一帧；首窗 40 帧，之后每窗 30 个新帧加 10 个共享帧。
2. **稠密深度**：每个窗口送入 VGGT-Omega，得到每帧深度（相差一个窗口尺度）和置信度。
3. **位姿来源**：可替换。无人机上用激光惯导里程计（FAST-LIO），按 RGB 时刻加时间偏移插值；没有外部定位时用 GLOMAP 由图像求位姿。离线可再做 SIFT + XFeat 特征的 BA，以来源位姿为位置先验（σ = 5 cm），重投影阈值三轮收紧 16 → 8 → 4 px。
4. **窗口尺度**：每个窗口一个米制尺度，同时满足两类约束：窗口内 VGGT 相机轨迹与位姿来源轨迹的尺度比、相邻窗口共享帧上的深度比；在对数域做加权最小二乘，每来一窗全局重解一次。
5. **多视一致性融合**：每个像素投影到最多 6 个近邻视角，深度对数比 <0.03 计为支持，至少 min(2, 可见视角数) 个支持才保留；2 cm 体素起，点数超上限时自适应放大体素。
6. **3DGS**：用融合点云初始化；损失为 L1 / SSIM 图像项、与 VGGT 逆深度的 Pearson 相关项、高斯长短轴比惩罚。

## 结果

| 场景 | 位姿来源 | 指标 | 数值 |
| --- | --- | --- | --- |
| 整层办公室（S380 的相机与激光雷达，手持采集 27 min，RGB 1280×720） | FAST-LIO，机载融合（本机回放 bag） | 重建点云到激光雷达地图的距离 P50 / P90 | 3.3 / 11.2 cm |
| 同上 | FAST-LIO + 落地 BA | 同上 | 2.9 / 6.8 cm |
| 同上 | FAST-LIO + 落地 BA | 3DGS 测试视角 PSNR（640×360，采集轨迹上留出的 433 帧，评测时按该帧图像微调相机位姿） | 25.7 dB |
| 休息室（手机 4K 视频，抽 1200 帧） | GLOMAP，米制尺度按相机离地 1.4 m 假设 | 3DGS 测试视角 PSNR（1600×900，150 帧未参与训练） | 32.6 dB |
| 机载 Jetson Orin NX 16 GB | FAST-LIO | 每个 40 帧窗口的 Omega 推理（384 宽） | 约 32 s |
| 整层办公室重建场景中的仿真避障 | — | 50 组随机起终点，策略输入为重建点云渲染的深度图，未针对该场景重新训练 | 成功 50 组 |

激光雷达地图由同一次录制的雷达扫描按激光惯导位姿拼成，不是独立真值；距离是每个重建点到地图的最近邻距离，不做配准，口径见[激光雷达评测](recon/docs/LIDAR_EVAL.md)。

离采集路线越远，3DGS 伪影越多，查看器可按 G 开启采集走廊引导。机载在线融合点云每次刷新都重算整段，耗时随窗口数增长（本机回放到 20 个窗口时单次约 78 s），尚未增量化。3470 帧规模的落地 BA 内存峰值超过 17 GB，超出 NX 的 16 GB，这一规模目前要在服务器上跑。

## 快速开始

**离线（Python ≥ 3.10，NVIDIA GPU）**

```bash
python -m pip install -e '.[core,train]'      # 先装好与 CUDA 匹配的 PyTorch
cp configs/phone_video.yaml my_scene.yaml      # 填写 video 与 depth.weights
recon run --config my_scene.yaml               # prepare → pose → depth → fuse → train → eval → export
```

无人机数据（带位姿）用 [drone_with_poses.yaml](configs/drone_with_poses.yaml)；输入格式与全部参数见 [输入契约](recon/docs/INPUTS.md)，展示前请读 [避坑说明](recon/docs/PITFALLS.md)。

**机载（ROS1 Noetic）**：构建、启动与配置见 [ROS 工作空间说明](ROS_Compatibility_layer/README.md)，交付包与落地 BA 见 [机载第一版说明](ROS_Compatibility_layer/src/drone3d_reconstruction_core/docs/ONBOARD_V1.md)。

**查看器**：`python3 -m viewer.tools.demo <导出目录> [<导出目录2> ...]` 在本机起服务并用浏览器打开，默认自由漫游，见 [查看器说明](viewer/README.md)。

**评测与成片**：`recon lidar-reference` / `recon lidar-distance` 计算重建点云到激光雷达地图的距离，需 `pip install -e '.[lidar]'`，见 [激光雷达评测](recon/docs/LIDAR_EVAL.md)；`recon reel` 按关键帧设计运镜、审核像素覆盖度、渲染并剪辑，见 [运镜与剪辑](recon/docs/REEL.md)。

## 模型权重

- VGGT-Omega：在 Hugging Face 申请下载，填入 `depth.weights`（离线）或重建 profile 的 checkpoint（机载）。
- XFeat：BA 需要其源码与权重，按 [安装说明](recon/docs/INSTALL.md) 准备。

本仓库不包含任何权重和采集数据。

## 许可

本项目原创代码采用 [Apache-2.0](LICENSE) 许可。

`third_party/vggt-omega/` 及 VGGT-Omega 的输出受 FAIR 非商业研究许可约束，禁止军事等用途；带 VGGT-Omega 的完整流程只能用于研究。其余第三方组件及其许可见 [第三方声明](THIRD_PARTY_NOTICES.md)。

## 贡献者

本项目由中科华创 ZKHC-Lab 开发并维护，贡献者见 GitHub 仓库页面的 Contributors。

## 引用

```bibtex
@misc{fly2twin2026,
  title  = {Fly2Twin: Fly Once, Get a Trainable Digital Twin},
  author = {ZKHC},
  year   = {2026},
  url    = {https://github.com/ZKHC-Lab/fly2twin}
}
```

使用可微视觉飞行模块时请同时引用：

```bibtex
@misc{diffdrone_pytorch3d,
  title  = {DiffDrone-PyTorch3D: A PyTorch3D-based Differentiable Drone Simulator},
  author = {Xu, Lu},
  year   = {2026},
  url    = {https://github.com/kirkchinese/Diffable_Drone_Pytorch3D}
}

@article{zhang2025diffphysdrone,
  title   = {Learning vision-based agile flight via differentiable physics},
  author  = {Zhang, Yuang and Hu, Yu and Song, Yunlong and Zou, Danping and Lin, Weiyao},
  journal = {Nature Machine Intelligence},
  year    = {2025}
}
```

## 致谢

- 可微视觉飞行模块使用 [DiffDrone-PyTorch3D](https://github.com/kirkchinese/Diffable_Drone_Pytorch3D)（GPL-3.0），它是 DiffPhysDrone（Zhang et al., Nature Machine Intelligence, 2025）的 PyTorch3D 复现与扩展，来自徐路的本科毕业设计，以独立仓库提供，本仓库不包含其源码。
- 稠密深度使用 VGGT-Omega（Meta AI）；3DGS 训练基于 gsplat；位姿求解使用 COLMAP / GLOMAP / pycolmap；特征匹配使用 XFeat；查看器基于 three.js 与 Spark。
