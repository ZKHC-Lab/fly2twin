# 离线重建独立仓库提取报告

2026-10-26 · 面向后续维护者

本文记录 2026-10-26 的来源仓库历史验收，不代表改名后的包已重新完成 GPU 验收。2026-10-29 搬入研发仓库后，尺度和融合调用共享 recon_core；不再依赖 Open3D，根许可证占位不搬入。

## 1. 历史结论

独立包已通过T30复现验收：对齐留出17.051503 dB、插值25.490281 dB，相对目标分别−0.076497 / +0.057281 dB，均在±0.3 dB内。31项CPU测试在源码和安装后的wheel上全部通过；使用允许复用的相机与Omega缓存，完成七阶段流程。原项目文件只读，没有导入历史实验模块，没有复制Omega源码、权重、图像、GPL解析器或仿真代码。窗口尺度、2 cm点数、最终初始化点数和体素尺寸与原T30完全一致；最终高斯数少1.45%，不是逐位复现。

## 2. 提取映射与行为差异

下表来源编号在文末证据索引给出；它们不是运行依赖。行号对应提取时只读源文件。

| 新模块 | 来源及行号 | 修改 / 行为差异 |
| --- | --- | --- |
| [坐标、插值、归一化](../geometry/transforms.py) | S1:15–28；S2:49–51,86–113；S3:90–99 | 普通数组接口、参数化外参和时间偏移、返回越界标记；重复位姿判据从仅位置改为位置和旋转，保留纯旋转。固定归一化由配置提供，新场景重新计算 |
| [关键帧与模糊](../keyframes/selection.py) | S1:30–43,82–97 | 保留 0.15 m / 8° / 严格 2 s、局部 21 帧中位的 0.5×；静止段判断增加旋转，不再误删原地转向；无法满足 2 s 的候选缺口显式拒绝 |
| [输入、PLY、导出](../io/ply.py) | 独立实现；输入契约来自盘点 §3 | NumPy 结构化数组实现 ASCII / 小端 / 大端标量顶点 PLY；写入支持 ASCII / 小端。拒绝网格和 list 属性，不冒充通用网格解析器。无 plyfile |
| [帧对](../poses/pairs.py) | S4:18–36 | 保留时间偏移 1..15、距离 <1.5 m、光轴 <45°、最近 30 再按距离做无向空间度数 ≤30；空间限制不截断时间配对 |
| [特征编排](../poses/prior.py) | S4:51–95；S5:432–458 | 新建 SIFT 数据库；CPU SIFT、补齐序列端点及空间配对；XFeat GPU 由调用方锁控制；不依赖已有数据库 |
| [XFeat](../poses/xfeat.py) | S6:45–121 | 保留 dense/refined XFeat*、top_k4096、F RANSAC 1px/.999/10000、≥30 内点、OpenCV seed0；用户安装代码和权重，无 hub 自动下载 |
| [数据库数值](../poses/database.py)、[合并](../poses/merge.py) | S7:27–62；S6:124–200 | 保留精确 float32 坐标去重、SIFT 后拼 XFeat、索引偏移/反转、F 转置、不做联合 RANSAC、不保存不兼容描述子；按 manifest 文件名映射，不解析文件名数字 |
| [BA](../poses/ba.py)、[选项](../poses/options.py) | S8:39–77；S9:15–91 | 保留位置先验 σ=.05 m、固定内参、16→8→4 px、1.5°过滤、100次迭代；分开记录 termination 和 usable。注册帧缺失时报错，不回填初值 |
| [无位姿 SfM](../poses/pipeline.py) | S10:99–149,164–190 | 共享 OPENCV、顺序 overlap20/quadratic、内参畸变优化、alpha0 去畸变；线程改4；报告每个模型与最大注册率。首版选择“拒绝部分注册”，不静默取最大局部段 |
| [深度接口](../depth/backends.py) | 盘点接口契约；用户安装包公开 API | 抽包时的独立适配器（crop 预处理、首窗 30 帧）已于 2026-10-29 删除，改为调用 third_party/vggt-omega/reconstruct_omega.py 并采用 V1.0 分窗，见 [INPUTS](INPUTS.md) |
| [预测缓存](../depth/cache.py) | S11:26–46 | 网格参数化，默认512×288；严格唯一 new-owner；不使用旧384×224常量 |
| [窗口尺度](../../recon_core/scale.py) | S11:49–98 | 轨迹旋转均值、正尺度锚、相邻窗口共享深度 log-scale 最小二乘保留；没有 C1 逐帧拟合 |
| [融合](../fusion/cloud.py) | S11:101–179；S12:59–99 | 在尺度估计前裁成训练帧窗口；673槽仅训练槽写入，其他明确零/False（原 frame_scales 空槽未初始化）；支持规则、颜色插值、体素网格与均值规则保留，现按体素字典序输出，图像路径来自 manifest |
| [初始化/渲染](../gaussian/render.py) | S2:156–179 | 保留4近邻初始化、SH、Adam、classic rasterization；去掉全局路径与 import-time 环境修改 |
| [训练](../gaussian/train.py)、[损失](../gaussian/losses.py) | S2:182–277；S13:120–152；S14:173–181 | T17循环加T21 BEST各向异性项，改为普通函数；配置 seed/steps/lambda，0.8 L1+0.2 SSIM、逆深度Pearson .1、轴比超10惩罚 .1；无 getsource/exec |
| [时间表](../gaussian/schedule.py) | S15:17–21,25–32 | 按 steps/30000 同比缩放 refine start/stop/every、reset、SH interval；大数据默认约60次/帧，以1000步取整；初始点上限仍300k，不迁入历史百万点实验分支 |
| [策略callback](../gaussian/strategy.py) | S17:167–216；公开gsplat1.5.3对应152–201 | Apache许可改编callback编排；调用公开包数值原语，但显式实现正确的正整数倍reset条件，保留grow/prune/reset顺序，避免公开版本bitwise比较错误 |
| [评测](../evaluation/evaluate.py)、[聚合](../evaluation/metrics.py) | S16:41–145；S3:128–143 | 冻结高斯；固定相机和目标RGB对齐200步分开输出；Adam1e-4、L1+.2SSIM；逐帧PSNR/SSIM/LPIPS算术平均；保留最后迭代，不选最好分数 |
| [相机对齐](../evaluation/camera.py) | 连续6D旋转数学定义；S16:109–127调用约定 | 独立实现右乘局部旋转/平移，匹配历史参数化，不复制gsplat示例文件；CPU identity/梯度测试覆盖 |
| [七阶段与缓存](../pipeline.py)、[CLI](../cli.py) | 新实现 | 文件内容SHA256+配置+包源码指纹，阶段产物校验；失败目录不自动复用；模型推理、外参、图像、缓存全部配置化 |

## 3. 自主拍板

1. 原提取使用 Open3D；迁移后用 recon_core 的 NumPy 体素降采样替代，点序改为体素字典序，按坐标排序后比较 XYZ/RGB。
2. 无位姿分支默认且当前仅实现拒绝部分注册；每个子模型仍写报告，避免把局部重建标作整场景成功。
3. 第一版用 manifest 图像/位姿为稳定边界；`bag` 依赖组保留，但 bag/video 导入命令尚未移植。
4. 归一化可冻结三字段，默认新数据重新算；T30 不用新外参 BA 重新确定 center/scale。
5. 关键帧允许纯旋转触发，不保留原位置去重的已知缺陷。
6. 迁移时许可证尚未确定，不保留占位文件；2026-11-03 定为 Apache-2.0，根目录提供 LICENSE。Omega 适配明确研究用途及输出限制。

## 4. CPU 与安装测试

最终31项CPU测试全部通过。包括真实SQLite合并、内容哈希失效、标准输入到融合的CPU闭环、真实pycolmap四相机三轮BA、0/3000/6000/15000步opacity调度及真实参数替换、候选序列模糊判据、无标定manifest，以及部分注册/子模型报告契约。部分SfM报告测试使用模拟模型返回值，只验证失败报告，不冒充真实图像SfM质量验收。测试使用公开安装的gsplat1.5.3，不使用原工程源码。

源码环境31项通过（4.52 s）；最终wheel在新venv中以`--no-deps`安装后，清除源码pythonpath，再跑相同31项全部通过（4.17 s），并验证`recon --help`与包实际导入位置。该venv共享现有第三方依赖，验证的是打包/入口独立性，不是干净CUDA环境完整安装认证。最终wheel为47,096字节，SHA256=`a4a0ae0eeff75b842aaa0f86be818ddce70c893ad432e0af0b20ccef59a02781`，包含上游callback完整许可与通知。

覆盖：SLERP/时间偏移、外参组合、纯旋转去重、像素中心缩放、固定归一化、运动/旋转/超时关键帧、候选缺口、模糊阈值、ASCII/二进制PLY/截断拒绝、匹配反转与索引偏移、SQLite读取/合并、15帧候选边、窗口尺度/退化拒绝/new-owner、训练专用深度槽、损失梯度、时间表、相机参数化、逐帧算术平均、输入内容/配置缓存失效。

所有测试使用 `CUDA_VISIBLE_DEVICES=''`；测试通过不等于所有新场景 GPU 路径已验收。

## 5. T30 复现验收

冻结输入：T18精确507训练/93留出/71插值，673深度槽仅填507训练帧；seed42、30000步、640×360、新外参的原T30 P-MIX相机、原Omega512预测缓存。使用用户允许的已有相机和模型预测，不重跑标定、BA和Omega网络推理。测试帧参与原BA，不进入训练RGB或融合。

固定归一化：center = `[-7.171480196455258,8.188519718807234,0.9049076656900958]`；meters_per_unit = `8.941428986458085`；scene_scale = `1.1180742025375368`；判据 atol=1e-7，rtol=0。

### 已验证的融合阶段

| 项目 | 新实现 | 原T30 | 差异 |
| --- | ---: | ---: | ---: |
| 训练帧 | 507 | 507 | 0 |
| 深度槽 / 已填槽 | 673 / 507 | 673 / 507 | 0 |
| 2 cm体素点数 | 2,244,232 | 2,244,232 | 0 |
| 初始化点数 | 280,798 | 280,798 | 0 |
| 最终体素(m) | 0.05223113795557858 | 0.05223113795557858 | 0 |
| 窗口尺度最大绝对差 | — | — | 0 |

### GPU结果：通过

| 指标 | 原T30目标 | 新实现 | 差值 |
| --- | ---: | ---: | ---: |
| 固定位姿留出PSNR(dB) | 16.651 | 16.507673 | −0.143327 |
| 固定位姿插值PSNR(dB) | 24.919 | 24.993179 | +0.074179 |
| RGB对齐留出PSNR(dB) | 17.128 | 17.051503 | −0.076497 |
| RGB对齐插值PSNR(dB) | 25.433 | 25.490281 | +0.057281 |
| 最终高斯数 | 1,533,519 | 1,511,338 | −22,181（−1.45%） |

通过判据：RGB对齐留出与插值分别位于目标±0.3 dB内，两个集合均通过。高斯数是诊断参考，不是完全确定性门槛；本次不是逐位重现，未将剩余数量差异强行归因于某一个GPU kernel。

逐帧计算再取算术平均的其余指标：

| 集合 | 帧数 | 口径 | SSIM | LPIPS |
| --- | ---: | --- | ---: | ---: |
| 留出 | 93 | 固定 | 0.552091 | 0.502791 |
| 留出 | 93 | RGB对齐200步 | 0.575864 | 0.496073 |
| 插值 | 71 | 固定 | 0.774207 | 0.328901 |
| 插值 | 71 | RGB对齐200步 | 0.781067 | 0.325785 |

实际训练30,000步，耗时1719.304 s（28.66 min）；每个训练帧访问59–60次。训练过程峰值GPU allocated/reserved为2.258 / 3.592 GiB，CPU RSS为2.097 GiB。验收脚本检查了切分逐元素相同、三项归一化atol=1e-7、673槽仅507槽填充、SH3、两项lambda=.1、200步冻结高斯对齐、真实训练访问次数；并从164行原始评测记录重新聚合，确认与摘要完全一致。

本次数值代码提交为`7a43046`，完整运行指纹为`c9e18f336d80e259923334c895f3b6a87b44051195d7dc9765701ae46bf3779a`。最终checkpoint为356,680,061字节，SHA256=`c4a8546c860cd3866b8b82651334372fa35db30e9bf71c4bc678842987692314`。

### 导出与缓存验收

七个阶段的所有产物均重新计算SHA256，与完成记录一致。NumPy重新读取实际导出的二进制PLY：彩色云280,798点，完整SH3高斯PLY为1,511,338点、62个标量属性；抽查SH通道和opacity与checkpoint逐元素一致，世界坐标反归一化后的XYZ逐元素一致。preview SPLAT为48,362,816字节（每高斯32字节，只有DC颜色，不等同SH3渲染）。相机导出为673帧，其c2w/K与输入精化相机完全相同，单位meters。

运行资源：预处理、融合、扩展编译在GPU锁外；训练和评测用非阻塞共享锁，优先让演示任务使用GPU。使用命名systemd user scope、MemoryMax=16G及独立后台日志；若取消任务，需停止真实scope而不是只杀外层包装。一次训练曾到9000步后主动让出资源，该次没有最终checkpoint，不作为复现指标。基础环境原有gsplat1.5.2，最终新实现使用公开发行的gsplat1.5.3、PyTorch2.7.1+cu128；所有新增依赖/编译缓存留在新仓库忽略目录，不改共享环境。CUDA工具链布局和锁外预编译方法见[安装说明](INSTALL.md)。

重要兼容发现：原实验所用、版本字符串同为1.5.3的本地checkout使用`step % reset_every == 0 and step > 0`。公开1.5.2在第0步reset；公开1.5.3则写成`step % reset_every == 0 & step > 0`，按Python比较优先级永远不会reset。最初仅处理第0步的适配不完整，已撤销该训练的验收资格。[显式调度适配](../gaussian/strategy.py)现在完整保留状态累计→grow→prune→清累计→正整数倍reset的顺序，在refine_stop_iter停止，回归测试覆盖0/3000/6000/15000步和真实opacity/optimizer参数替换。该小段callback编排按gsplat Apache许可改编，已保留版权和完整许可。实际复现不把原工程gsplat路径加入PYTHONPATH，不导入历史模块。

诊断证据：原T30在3000步为670,484个高斯，3250步降到522,367个；未修复公开1.5.3的训练在3000步为667,053个，之后继续增加而没有reset后的下降。在14,500步已达2,538,225个，而原T30在15,000步固定为1,533,519个。输入相机、尺度及初始化点数均已相同，因此先修复callback语义，而不是改动密化阈值或训练参数去凑指标。

实际运行依赖核对：gsplat `1.5.3`、PyTorch `2.7.1+cu128`、Torch CUDA `12.8`、编译架构 `12.0`。公开wheel没有记录上游Git commit，未臆造revision；实际编译扩展为19,376,728字节，SHA256=`53f8eafd246c92305ac0c6857e9a499c43933ff229cc86c49acf0855a816bee5`。导入路径来源标注：`.local/deps/gsplat/__init__.py`，扩展来源标注：`.local/torch_extensions/gsplat_cuda/gsplat_cuda.so`；均在新仓库忽略目录，没有原项目Python源码依赖。

### 冻结输入与数值来源SHA256

| 对象 | SHA256 |
| --- | --- |
| T18切分 | `087b87479583e3e2f516a67615faa0fa97cd1e65fcdc33ea32264becca2fd858` |
| T21归一化配置 | `097878c974e5f350401b14d3900dd11cdd5237da0f95792ca4e7d0e94743083f` |
| T30新外参P-MIX相机 | `98fad16a708c49cc5bcd0febaaf51bed7f818dbe3905e7775ab0950f5950e216` |
| T17训练源 | `c125bcb6e68012459ed10438e7401bb662216ff1b19b2f9d15a723a25af50eda` |
| T21 BEST适配源 | `78c18305c10a6e2b3175e5f58ed40496caebfd1f8c271ca608765a90ae9fca7a` |
| 原融合数值源 | `4b89ee27ef3c49903df56a30c43e85caf2208f873e2428c67a57b7813757789c` |
| T30验收入口源 | `5513dca04e91a4478dd104511dc1524ee6c2927792c611a663e191c9fa9f4e39` |

## 6. 尚未完成与发布边界

- fresh SIFT+XFeat+BA、fresh Omega模型推理、无位姿完整注册/去畸变链路已实现但未完成真实新数据端到端验收。当前质量验收从允许复用的已精化相机和深度缓存开始。
- bag/video导入命令未提供；无ROS、仿真、查看器。
- 无公开授权小场景；依赖锁仅记录本机实际版本，不是干净CUDA镜像全量安装认证。
- XFeat/LPIPS/Omega权重许可、确切上游revision及公司代码权属仍需发布前确认；根许可证未最终授权。
- 图像只支持共享分辨率/标定；PLY只支持标量顶点；无位姿部分注册选择最大模型的继续运行尚未实现。

## 证据索引（只读来源标注，不是依赖）

所有原工程路径均相对原工程根目录，实验数据不随仓库发布：

```text
S1 Data/experiments/office_gs_03_20261022/prepare.py
S2 Data/experiments/office_gs_02_20261019/T17/train.py
S3 Data/experiments/quality_check_20261021/T30/gs30.py
S4 Data/experiments/office_gs_02_20261019/pose/prepare.py
S5 Data/experiments/office_gs_03_20261022/pose_pipeline.py
S6 Data/experiments/ctrl_capture_20261019/T20/mix_pose.py
S7 Data/experiments/接手诊断_20261008/tasks/T13/mix_database.py
S8 Data/experiments/office_gs_02_20261019/pose/common.py
S9 Data/experiments/office_gs_02_20261019/pose/ba.py
S10 Data/experiments/phone_room_20261015/T07/sfm.py
S11 Data/experiments/office_gs_02_20261019/gs/fuse.py
S12 Data/experiments/office_gs_02_20261019/T18/run.py
S13 Data/experiments/quality_check_20261021/T21/run.py
S14 Data/experiments/ctrl_capture_20261019/T20/run.py
S15 Data/experiments/office_gs_03_20261022/long_scaled.py
S16 Data/experiments/office_gs_02_20261019/gs/evaluate.py
S17 gsplat/gsplat/strategy/default.py（callback顺序）；gsplat/LICENSE（上游许可）
规范 ROS_Compatibility_layer/docs/GS_OFFLINE_RECONSTRUCTION.md
标定 ROS_Compatibility_layer/docs/CAMERA_EXTRINSIC.md
盘点 Data/experiments/open_release_20261026/A_inventory.md
目标 Data/experiments/quality_check_20261021/T30_result.md §6
切分 Data/experiments/office_gs_02_20261019/T18/selection.json
归一化 Data/experiments/quality_check_20261021/T21/office/BEST_42/config.json
原融合 Data/experiments/quality_check_20261021/T30/candidates/depth/gs/office/gs/fusion.json
本地输入 configs/local_office02.yaml（忽略，不提交）
本次原始证据 .local/ 下日志、outputs/ 下新产物（忽略，不提交）
```
