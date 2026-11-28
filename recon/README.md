# recon：离线重建

流程：prepare → pose → depth → fuse → train → eval → export，七个阶段按顺序全部要跑；`--only S` 要求之前各阶段已完成。窗口尺度与多视融合共用 [recon_core](../recon_core/README.md)。离线每个窗口求一个尺度；回访修正默认关，S380 预设打开。

## 安装与运行

在仓库根目录运行，Python ≥3.10；推理、XFeat、训练与评测需要 CUDA，缓存位姿/深度的融合可用 CPU。

```bash
python -m pip install -e '.[core,test]'
# 先安装匹配 CUDA 的 PyTorch，再安装训练依赖：
python -m pip install -e '.[train]'
recon run --config my_scene.yaml --through fuse
```

将预设合入自己的场景配置并填写输入路径：[无人机](../configs/drone_with_poses.yaml)、[手机视频](../configs/phone_video.yaml)、[S380](../configs/s380.yaml)、[纯图像](../configs/image_only.yaml)。手机预设把无人机方法用在手机视频上：时间轴等分 1200 段（400 × 3）、每段取最清晰的一帧，不看运动；抽帧时缩到长边 1920，之后没有原分辨率的图；位姿和训练用同一组帧，训练宽 1600。位姿用 `pose.provider: vggt`（VGGT 各窗口按共享帧的深度串接、特征匹配做回环、特征 BA 并自标定相机），只用图像，每一帧都有位姿；米制尺度来自“走动时每秒 1 米”的假设，不是测量；深度、窗口尺度、融合沿用无人机的设置（0.3–4 m、1.5 m 近邻半径按假设的米计），训练用融合点云初始化、深度损失 0.1，不开回访修正。运动关键帧需要位姿，手机视频用不了。只用图像、用 SfM 稀疏点初始化的旧手机做法已没有入口（e059b7b）。位姿插槽与手机训练评测口径见[输入契约](docs/INPUTS.md)。

## 文档

- [安装与资源隔离](docs/INSTALL.md) · [输入、参数与输出契约](docs/INPUTS.md)
- [镜头设计、覆盖度审核与剪辑](docs/REEL.md)
- [激光雷达参照与点云距离评测](docs/LIDAR_EVAL.md)
- [重建场景中的策略评测（`recon sim`，英文）](docs/SIM.md)
- [展示与复现避坑](docs/PITFALLS.md) · [第三方声明](../THIRD_PARTY_NOTICES.md)

原创代码采用 [Apache-2.0](../LICENSE) 许可；VGGT-Omega 及其输出受上游研究用途许可约束。仓库不含模型权重和采集数据；网页查看器见 [viewer/](../viewer/README.md)。
