# 可选融合注意力

默认 `auto` 保持 PyTorch SDPA：有可用融合内核时使用其调度；缺少融合内核时，长序列采用现有 query 分块回退。
`flash_attn_v1` 是显式选择的推理实现，复用 FlashAttention 1.0.9，不改变 checkpoint、完整 K/V、帧窗口、BF16 aggregator 或 FP32 heads。
该实现不支持半精度训练反向；FP32 head 仍调用 PyTorch SDPA。

## 已验证环境与来源

- Jetson Orin（SM 8.7）、CUDA toolkit 11.4.315、Torch 2.1.0a0+41361538.nv23.06、Python 3.8、CXX11 ABI=true。
- FlashAttention v1.0.9：git `6d48e14a6c2f551db96f0badc658a6279a929df3`。
- 上游锁定的 CUTLASS：git `319a389f42b776fae5701afcb943fc03be5b5c25`。
- 上游来源：https://github.com/Dao-AILab/flash-attention/tree/v1.0.9 。许可证随外部源码保留；本仓库不打包其 C++ 源码或编译二进制。

构建命令（使用实际 worker 的 Python；`ninja` 可执行文件需在 PATH 上）：

```bash
export CUDA_HOME=/usr/local/cuda
python -I third_party/vggt-omega/tools/build_flash_attention.py \
  --source /absolute/path/to/flash-attention-v1.0.9 \
  --cutlass /absolute/path/to/cutlass-319a389 \
  --build /absolute/path/to/third_party/vggt-omega/.native/flash_attn_v109
```

脚本只编译 SM 8.7，单个 ninja job / NVCC thread，不安装或替换 torch。推荐在构建进程外施加内存上限。
扩展文件名为 `drone3d_flash_attn_v109.so`；`build_result.json` 记录 torch/CUDA/ABI 和实际源码摘要。
更换 worker Python、torch 或 CUDA 后应重新构建，不能把其他环境的 `.so` 当成兼容包。

## 启用与回退

独立重建：`reconstruct_omega.py --attention_backend flash_attn_v1 ...`。
常驻 worker：`warm_worker.py --attention-backend flash_attn_v1 ...`。
ROS 重建 profile：`attention_backend: flash_attn_v1`，默认 `auto`。该选择写入结果 `profile.attention_backend`；G4 不允许同一会话中混用不同注意力实现。

扩展可作为普通 Python 模块安装；更推荐上述模型目录内的 `.native/flash_attn_v109/` 路径。
ROS worker 会剥离 supervisor 的 PYTHONPATH，本地扩展路径不依赖它。指定了融合实现但库不可用时明确失败，不偷偷退回更慢路径。
恢复 `auto` 后无需卸载扩展；必须新建会话，不能覆盖历史窗口的来源身份。

## 验证边界

NX 原语测试覆盖 BF16 的多 batch、非整数块长度、17960/4096 等序列以及 head_dim 64/128：输出有限，形状/dtype正确，对 math SDPA 相对 L2 ≤1%。
17960-token 原语约 1.72→0.53 s。一次固定40帧、384×288整模型对照的热推理均值 60.57→30.89 s，峰值 allocated 4812→3830 MiB。
这些数据不是所有场景的速度承诺；完整模型不要求也不保证逐位相同。

三个固定40帧窗口的 D435i 同像素近距诊断中，融合版绝对深度误差中位变化约 +4.4% / −1.0% / −4.4%，P90变化均未恶化超过5%，有效像素覆盖一致。
这是以定位轨迹估尺度的单窗深度诊断，不是独立测绘精度或全航线地图验收。分辨率与采集策略变化必须另作对照，不能归入内核本身的收益。

本机证据来源（不入 Git）：Data/experiments/nx_fused_attention_20261014/ 下的 PLAN.md、results/sources.json、
results/quality_all384.json 和 received/*_384_r2/report.json。最初的一份融合结果曾因断电截断，不能使用；质量对照使用带 `_r2` 的完整重跑结果。
