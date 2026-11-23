# recon：离线重建

流程：prepare → pose → depth → fuse → train → eval → export。窗口尺度与多视融合共用 [recon_core](../recon_core/README.md)。

## 安装与运行

在仓库根目录运行，Python ≥3.10；推理、XFeat、训练与评测需要 CUDA，缓存位姿/深度的融合可用 CPU。

```bash
python -m pip install -e '.[core,test]'
# 先安装匹配 CUDA 的 PyTorch，再安装训练依赖：
python -m pip install -e '.[train]'
recon run --config my_scene.yaml --through fuse
```

将预设合入自己的场景配置并填写输入路径：[无人机](../configs/drone_with_poses.yaml)、[手机视频](../configs/phone_video.yaml)、[S380](../configs/s380.yaml)、[纯图像](../configs/image_only.yaml)。手机预设把无人机方法用在手机视频上（GLOMAP 位姿、假设相高 1.4 m、Omega 深度与融合），使用同一 `recon run` 分阶段命令；位姿插槽与手机训练评测口径见[输入契约](docs/INPUTS.md)。

## 文档

- [安装与资源隔离](docs/INSTALL.md) · [输入、参数与输出契约](docs/INPUTS.md)
- [等价验收](docs/EQUIVALENCE.md) · [提取来源与历史验收](docs/EXTRACTION_REPORT.md)
- [镜头设计、覆盖度审核与剪辑](docs/REEL.md)
- [激光雷达参照与点云距离评测](docs/LIDAR_EVAL.md)
- [重建场景中的策略评测（`recon sim`，英文）](docs/SIM.md)
- [展示与复现避坑](docs/PITFALLS.md) · [第三方声明](../THIRD_PARTY_NOTICES.md)

原创代码采用 [Apache-2.0](../LICENSE) 许可；VGGT-Omega 及其输出受上游研究用途许可约束。仓库不含模型权重和采集数据；网页查看器见 [viewer/](../viewer/README.md)。
