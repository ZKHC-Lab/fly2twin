# 长窗分组（100/120 与尾窗规则）

本文描述正式 ROS 代码里的**长窗分组**：新增帧数语义、登记式计划、
逐边阈值来源、非 fit 观测排除池，以及方法身份。本文自含结论与数值，不依赖读者打开任何
`Data/` 下的本地文件；原始材料的位置只在文末“证据索引”里作为来源标注列出。

规范依据见 [G4 算法基线与移植规范](G4_BASELINE_SPEC.md) §7 第 2–3 步。

**一句话结论（当前验证边界）**：

- **已验证**：① 分组调度——含 3774 帧 / 38 窗 / 首窗 100 / 常规 20 共享＋100 新增 / 尾窗
  26 共享＋74 新增的完整 CPU 端到端调度（3.55 s，无模型推理），窗口身份与"每帧只贡献一次"
  逐项核对；② 采样、估计与图求解——与冻结实现同输入对拍（真数据 37 条边：采样逐位一致、
  变换差 4.44e-16、协方差差 0；冻结图回归 10 项全过、差值全 0）；③ 前端**输入**层——RGB
  3774/3774 帧与登记哈希相同、里程计插值后与受保护光心差 0.0、预处理输入张量 38/38 与冻结
  记录相同；④ 配置入口——真实长窗 YAML 经节点加载器贯通，节点与调度测试 19 项在 ROS Noetic
  容器内通过。
- **未验证**：① 与冻结**预测数组**的数值对比——旧数组已删除，无法比对；38 份重跑预测文件的
  哈希与冻结登记全部不同，这只说明文件不是逐字节相同，数值差异不可量化，也没有受控实验说明
  原因；② 前端数值与几何**效果**等价（V4/V5，需要独立评价池）；③ 真实 ROS 传感器/硬件链路
  （V6）；④ NX 部署与吞吐（V8）。

## 1. 两种窗口帧数语义

| | `total_frames`（默认，既有） | `new_frames` |
|---|---|---|
| `initial_window_frames` | 每窗模型输入总帧数（含共享） | 每窗**新增**帧数 |
| 模型输入 | = `window_frames` | = `window_frames + overlap_frames` |
| 首窗 | `window_frames` | `window_frames`（无共享） |
| 采集目标 | `window_frames − overlap` | `window_frames` |
| 上限约束 | `window_frames ≤ frame_group_size` | `window_frames + overlap_frames ≤ frame_group_size` |

`new_frames` 只在 `alignment_backend: global_g4` 上生效；用在别的后端会**报错拒绝**，
不会被静默忽略。上限不足同样被拒绝（`frame_group_size` 不够时不用均匀抽帧凑），
共享帧不足时中止该窗并保留现场，不缩小共享前缀。

```yaml
alignment_backend: global_g4
reconstruct_frame_group_size: 120   # 必须 >= 新增 + 共享
initial_window_frames: 100          # 只数新增帧
initial_overlap_frames: 20
initial_inference_resolution: 384
window_frame_semantics: new_frames
minimum_total_frames: 100           # 尾窗下限
window_plan_path: ""                # 登记清单；留空则按推导规则
fit_exclusion_source: ""            # 保留观测表；留空 = 未启用
threshold_table_path: ""            # 历史阈值表；留空 = 逐边重算
```

候选配置（连续会话、重建、前下视采集四份，**未在 NX 实机验收**）已从主线删除，可从标签 `archive/pre-restructure` 的 `tools/nx_bench/profiles/long_window/` 恢复。

## 2. 分组规则、冻结清单与尾窗

推导规则（`window_plan.py`）：

- 首窗：新增 100，无共享；
- 常规窗：共享 20 ＋ 新增 100 ＝ 模型输入 120；
- 尾窗：剩余新增不足一个常规窗时，共享帧数取 `max(20, minimum_total_frames − 剩余新增)`，
  使模型输入不少于 100 帧。

对冻结航线 3774 帧，规则推出 **38 窗**，逐窗与登记清单完全一致（差异清单为空，
分组摘要相同：`0e95109d2c4a2607ea7c9a7bb5dcf162a5e5f08d3ceb584cb32c6d2bc9556660`）：

| 窗口 | 新增区间 | 共享 | 新增 | 模型输入 |
|---|---|---|---|---|
| F000 | [0, 100) | 0 | 100 | 100 |
| F001–F036 | [100k, 100k+100)，k=1..36 | 20 | 100 | 120 |
| F037（尾窗） | [3700, 3774) | 26 | 74 | 100 |

合计 100 + 36×120 + 100 = 4520 个输入帧实例，覆盖 3774 张唯一 RGB。

`window_plan_path` 给出登记清单时，会话按清单逐窗决定新增与共享帧数，后端再核对一次
（模型输入长度、共享帧数，以及“共享前缀正是上一窗尾部新增帧”的帧身份）；
清单与实到帧不一致会报错并保留现场。

**尾窗只在登记清单里存在**：在线会话无法预知“这是最后一窗”，所以它只推导常规形状
（`window_frames` 新增 ＋ `overlap_frames` 共享）；`plan_from_counts` 的尾窗补足规则用于
离线规划与登记清单的生成，运行时由 `check_plan_shape` 把它变成**校验**——登记清单里
短窗的共享帧数不得少于 `overlap_frames`，模型输入不得低于 `minimum_total_frames`，
常规窗必须与会话声明的 `window_frames`／`overlap_frames` 一致，不一致直接报错。
`minimum_total_frames` 只校验登记清单，不会把在线会话变成长窗分组。路线何时结束无法在线预判，
会话的采集终态检查也不接受部分窗口（“partial windows are not admitted”），该检查未为尾窗放宽。

诊断字段：`effective.window_frame_semantics`、`steady_model_frames`、`window_plan_source`、
`minimum_total_frames`。

## 3. 逐边阈值来源（D03）

`ThresholdPolicy` 把三种来源分开记录，逐边写入窗口元数据与整图修订：

| 情形 | 使用值 | `threshold_source` |
|---|---|---|
| 登记表里有该边 | 登记值 | 表路径（如 `…/seams.json`） |
| 登记表里没有 | `0.03 × 共享帧源深度中位数` | `recomputed_from_mask_intersection` |
| 调用方显式传参 | 传入值 | `explicit_argument` |

表键的拼写差异（`F000|F001` 与整数 `0|1`）按“去前导 `F`/前导零”归一后匹配；含多个分组的表
（冻结 seams 同时有 G0/G1）必须用 `threshold_group` 显式选择，否则报错，不允许最后一条静默生效；
缺记录阈值的边按冻结语义重算，并把跳过条数记为 `skipped_records`。

元数据同时保存 `threshold`、`threshold_source`、`threshold_recomputed`、
`threshold_median_source_depth`，因此“这条边用的是哪来的阈值”可逐边复核；沿用同一公式
不再被当成同一输入。

## 4. 非 fit 观测排除池（D02）

- 只作用于 fit 划分（`k%5<3`），检查（`k%5==3`）与留出（`k%5==4`）样本数不变；
- 半径 3 个**模型像素**（384×288 坐标系）；冻结表存的是 640×480 原始像素，
  由调用方按 `(u+0.5)·0.6−0.5` 换算后传入，核心不做隐式换算；
- 两个来源可分别标记：`fit_exclusion_source` 登记的保留观测表，以及已提交窗口里
  `split=report` 的观测（需要缓存保留样本身份）；
- 在线默认是「没有登记表就不启用」，并在元数据里显式记录 `exclusion_source: none`——
  省略被记成省略；
- 排除池**不取自当前正在估计的边，也不取自未来边**（`_fit_exclusions` 只并入
  `window_index` 更小的已提交窗口，以及调用方显式给出的登记表）。

## 5. 方法身份与产物

| 身份 | 值 | 用途 |
|---|---|---|
| 增量默认 | `legacy_g4_20260916` | 既有 16/5 会话与已发布修订 schema，保持原值 |
| 长窗分组 | `windowed_sim3_graph_20261008` | `run.json`、`revision.json`、状态与图诊断都携带 |

会话把后端 profile 写盘后交给子进程（`g4/backend_profile.json`），子进程不自行推断配置。
`run.json` 增加 `input_policy` 块（阈值策略、排除表、计划摘要）；`revision.json` 增加
`edge_thresholds` 与 `input_policy`；窗口元数据增加 `threshold`、`plan`、`method_id`。
这些是向后兼容的加法，既有 schema 版本号与身份字符串不变。

旧运行目录的兼容规则：没有 `input_policy` 块时，只有「本次请求的策略全为默认值且其余键一致」
才继续复用；任何非默认策略都要求新的输出目录，旧文件不被改写。副作用：`config_digest`
变化会让既有修订按新签名重解一次（是重解，不是损坏）。

## 6. 算法未改动

几何、E1 估计与图求解沿用冻结数学，只把输入策略与调度显式化：

- 采样：每窗置信度 P50 与 3×3 相对深度边缘 0.03 掩码取交；每帧 4×4 格、每格最多 32 点；
  共享前缀按 `fit,fit,fit,accept,report` 循环划分；种子 `20260915 + 新窗编号`，
  bootstrap 再加 1000。
- 估计：1000 次三点 RANSAC、5 次内点重估、正尺度非退化支持、最少 30 个 fit 点、
  共识比例 ≥0.3；200 次“共享帧→网格”两级 bootstrap、至少 160 次有效、7 维可观测性、
  协方差加 `0.001² I` 下限。
- 退化假设：冻结实现里一次退化三点拟合会中止整条边，正式增量默认改为继续换假设；
  `ransac_degenerate` 显式二选一（`skip` / `abort`），长窗分组身份用 `abort` 复现冻结行为，
  两者不会互相冒名。
- 图：`Log(Z⁻¹ S_i⁻¹ S_j)`，平移先除源窗长度再白化；位置残差 `(S·c−c_ref)/0.1m × 块权重`；
  块权重先于 Huber（delta=2.5）；右增量、TRF exact、max_nfev=200、三项容差 1e-8、
  logscale 界 ±12。默认参数见规范 §9.2。

掩码实现有一处**已量化的实现差异**（冻结用 OpenCV `dilate/erode`，正式用 SciPy
`maximum_filter/minimum_filter`）：深度全为有限值时两者 **0 个像素不同**（288×384 随机深度、
阶梯边缘各测 3 次）。深度含 NaN 空洞时两者的差异取决于 OpenCV 的 min/max 归约顺序，
既不是“传播 NaN”也不是“忽略 NaN”：固定脚本（`probe_mask_deviation.py`，见证据索引）在
288×384 上测得“cv2 对 scipy”相差 **61–122 像素/110592**（约 0.06–0.11%），“cv2 对忽略 NaN
的变体”相差 **58–146 像素**——**谁更接近随数据变化**，没有任何 NaN 约定能逐位复现 OpenCV。
因此不追求该情形逐位一致，而是把每个窗口的**非有限深度值个数**记入 pair 诊断
`nonfinite_depth_values`，让“这一窗是否可能受该差异影响”可量化可见；冻结材料是否含 NaN
无法核验（预测数组已删）。
反投影无差异：冻结的 `cv2.remap(INTER_LINEAR)` 在整数像素上与正式的直接取值完全相同（实测差 0.0）。

## 7. 已验证 / 未验证（内联数值）

已执行的检查如下，NX 项单独标明设备；执行完成与数值通过分别记录：

| 检查 | 观测值 |
|---|---|
| 分组推导 ⟷ 冻结登记清单 | 38 窗逐字段相等，差异清单空，摘要相同 |
| 采样 ⟷ 冻结 E1 oracle | 带排除表：fit 点 283/283，划分 173/62/48 两侧相同；不带排除表：286/286，176/62/48 两侧相同；排除 3 点／1 帧两侧相同；反投影与深度差 **0.0**（容差 1e-12） |
| 估计 ⟷ 冻结 E1 oracle | 变换分量最大差 **3.469446951953614e-18**（容差 1e-10）；协方差最大差 **0.0**（容差 1e-10）；共识比例两侧均 0.930635838150289；源窗长度两侧均 7.277507305145264 |
| 阈值来源 | 登记值复用时来源为表路径；未登记边来源为 `recomputed_from_mask_intersection`；三者互不冒名；真实冻结 seams 表在选定 `G0` 后可加载，`F000|F001` 能回答整数 `(0,1)` 查询（0.014139644801616669），缺记录的边落回重算 |
| 排除池 | 仅 fit 受影响：fit 286→283，检查/留出不变；两侧计数一致 |
| 冻结图回归 | 38 节点／35 边／602 因子／3774 去重光心；10 项检查全部通过；代价、同状态代价、光心、旋转、尺度、探针差值全 **0.0**；57.904 s；峰值 RSS 107.03 MiB；14/14 保护文件与输入源码未变 |
| 无 ROS 测试套件 | **453 项通过、0 失败**、164 子测试通过；12 项因本机缺 `rospy/actionlib` 无法收集（计为“未运行”，不计为通过） |
| 前端输入重生成 | 从冻结 bag 重抽 **3774/3774** 帧 RGB，SHA-256 与时间戳与登记清单完全相同；重生成里程计 55477 个位姿，按 RGB 时刻插值后与受保护定位数组的 **3772** 个有效光心逐元素差 **0.0 m / 0.0**（容差 1e-6） |
| 冻结输入复核（40 项） | 13 项 core + 27 项 model 全部按冻结登记哈希一致；冻结的两处模型脚本因仓库整理被改动，改用哈希一致的 git 修订版（6d266d2 / 903fe24）运行，生产模型树未改 |
| 前端重生成（38 窗） | 预处理输入张量 `input_tensor_sha256` **38/38** 与冻结记录相同；38 份预测文件的哈希与冻结登记**全部不同**——这只说明文件不是逐字节相同；旧数组已删除，**数值差异无法量化**，也**没有受控实验**说明原因（驱动/CUDA 运行时变化只是候选因素，未做对照）。显存峰值 5972/6001 MiB 与冻结记录 6001 MiB 相符 |
| ROS 实际推理与同机重生成参考（F000–F029） | 3000 张独立 RGB、3580 帧模型输入实例的身份、像素、时间戳、预处理张量 SHA 全部一致；四数组的形状、float32 类型与有限性均符合。深度和置信度各 395919360 个元素逐位相等；K 最大绝对差 0.02734375 像素，合并 RMS 0.000265576；外参元素最大差 2.57492e-5，合并 RMS 8.20753e-7。严格零容差通过 0/30，相机差异原因未验证，不能宣称前端完全等价。F030–F037 尚未完成此对照 |
| NX 隔离安装与恢复 | NX 台架上的隔离安装含 106 份与源文件逐字节相同的 Python 文件及 24 份 catkin 生成文件；19 项恢复测试通过。未更改正式模型树与基准 |
| NX 完整 F000（128 行 query 分块候选） | 100 帧、384×288；生产 job 耗时 495.447 s，峰值 GPU allocated 5612 MiB，最低系统可用内存 4.376 GiB，swap 最大增长 4.75 MiB；内存守护未触发。输入张量 SHA 与重生成参考相同，四预测数组均有限；严格零容差未通过：深度最大绝对差 1.931443（窗口单位），绝对差 P99 0.045974，相对差 P99 2.35954%；置信度最大差 1.492150，K 最大差 0.618896 px，外参元素最大差 0.000725109。job 时间包含流水线开销，不是纯模型推理时间；不能由单窗推断完整航线吞吐 |
| NX 注意力原语对拍 | Torch 2.1.0a0+41361538.nv23.06、同一 math 后端、固定 N=4096/H=16/D=64/bfloat16/seed=0；整块与 128 行分块各用独立子进程，两路 CPU/GPU 输入 SHA 及 allocated/reserved 基线相同。4194304 个输出元素逐字节相等且有限，最大差/P90/P99/RMS 均为 0；峰值 allocated 1064.125→72.625 MiB，reserved 1086→94 MiB，单次算子耗时 1.1554→1.1639 s。生产 GPU 锁与内存守护生效：合成低可用探针使睡眠子进程以 resource_guard/-15 退出，PID 与进程组均消失；真实两路以 worker_exit/0 退出且 GPU 锁已释放。该输入规模的原语等价不能证明完整 VGGT 等价，也未分离 F000 偏差中的后端与运行库贡献 |
| 真数据实现等价（37 边） | 几何适配器 深度/K/R/c 差全 **0.0**；采样点集/划分/簇逐位一致（不一致 0），fit 点 896–12879；变换分量最大差 **4.44e-16**、协方差最大差 **0.0**；35 条接纳 + 2 条不接纳（F023→F024、F036→F037）与冻结一致 |
| 排除机制（真数据） | 用尚存的冻结 `loop_exclusions.json`：两侧排除点数完全相同 **8532 / 9456 / 11720**；冻结记录的 9920/11286/12385 无法复现，因为另一半来源 `tracks/edges.npz` 已删 |
| 后端子进程 | 用会话实际发出的三类 profile 直接启动 `python -m …g4.backend_process`：默认、分组+空路径、分组+真实清单与阈值表——三者都存活并写出 `run.json`（退出） |
| 惰性导入解耦 | 导入 `drone3d_reconstruction_core.g4.map_backend` 与适配器测试不再需要 `rospy/actionlib`，适配器测试可直接收集运行 |

**未验证**：

| 项 | 阻塞原因 |
|---|---|
| V1 数值/效果等价 | 冻结预测数组已删除，无法与旧数组比对；目前只证明**预处理身份相同**（输入张量 38/38）与**同批预测上的实现等价**（37/37 边）。前端数值与几何效果是否等价仍未验证 |
| V4/V5 点云与几何效果 | 需要独立的点云/几何评价池与真值，未做 |
| V6 ROS 真实链路 | ROS Noetic 环境可用（按 CLAUDE.md 经 `sg docker` 使用 `fishros2/ros:noetic-desktop-full` 镜像；节点与调度测试 19/19 已在容器内通过）。未验证的是**真实传感器/硬件链路**（实际 RGB、定位、TF 到可消费结果），不是"无法在 ROS 环境里验证" |
| V8 NX 完整部署与吞吐 | 隔离安装、恢复测试及一个完整 100 帧窗口已执行；该分块候选未通过 V1 严格数值门槛，未继续剩余 37 窗。完整航线、点云/轨迹与 Web 的同 revision 一致性、效果与近实时能力未验证 |

数值收敛、人工认可、几何过门三件事分别显示；不宣称几何质量改善，也不把图回归通过当成全链路等价。

## 8. 实现位置

| 文件 | 职责 |
|---|---|
| [window_plan.py](../src/drone3d_reconstruction_core/src/g4_backend/window_plan.py) | 分组计划、推导规则、登记清单加载与校验、计划对拍 |
| [types.py](../src/drone3d_reconstruction_core/src/g4_backend/types.py) | `WINDOWED_METHOD_ID`、`ThresholdPolicy`、`SharedPairs.threshold_source` |
| [registration.py](../src/drone3d_reconstruction_core/src/g4_backend/registration.py) | 阈值来源记录、排除池计数与来源标记 |
| [graph.py](../src/drone3d_reconstruction_core/src/g4_backend/graph.py) | 求解结果携带方法身份 |
| [map_backend.py](../src/drone3d_reconstruction_core/src/drone3d_reconstruction_core/g4/map_backend.py) | 显式配置注入、排除池装配、登记计划逐窗核对、修订记录 |
| [backend_process.py](../src/drone3d_reconstruction_core/src/drone3d_reconstruction_core/g4/backend_process.py) | 后台进程按会话给出的 profile 构造后端 |
| [continuous_session.py](../src/drone3d_reconstruction_core/src/drone3d_reconstruction_core/continuous_session.py) | `new_frames` 语义、尾窗规则、计划驱动与诊断 |
| [\_\_init\_\_.py](../src/drone3d_reconstruction_core/src/drone3d_reconstruction_core/__init__.py) | PEP 562 惰性导出，解除 ROS 导入耦合 |

## 9. 证据索引（原始材料不入 Git）

以下为来源标注，不是文档链接；结论与数值已内联在正文：

- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/PLAN.md`：预登记判据与容差（运行前写定）。
- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/report.md`：完整交付记录与逐项差异（D01–D09）。
- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/evidence/frozen_oracle_comparison.json`：采样／估计对拍。
- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/evidence/plan_vs_frozen.json`：分组对拍。
- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/evidence/regenerated_input_identity.json`：RGB 与里程计身份核验。
- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/evidence/current_graph_regression.json`：冻结图回归。
- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/evidence/ros_free_suite.log`：无 ROS 套件输出。
- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/oracle/frozen_e1_oracle.py`：冻结 E1 参考实现（仅离线对拍，生产代码不得导入）。
- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/evidence/regenerate_reference.py`：GPU 主机可执行的重生成与分层对拍流程。
- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/evidence/probe_backend_child.py`、`probe_mask_deviation.py`、`backend_child_probe.json`、`mask_deviation.json`：后端子进程存活探针与掩码差异量化（自含，无需预设 PYTHONPATH）。
- `Data/experiments/_archive_gpt/长窗分组移植核验_20261008/verify_scratch/`：两轮独立复核报告与探针。
- `Data/experiments/testflight3_full_route_20260915/inputs/candidate_windows.json`：冻结 38 窗登记清单（只读）。
- `Data/experiments/vggt_backend_review_20260916/baseline/manifest.json`、`run_02/results/localization.npz`：14 项保护清单与定位对照（只读）。
- `Data/bag/testflight3.bag`：重生成 RGB 与里程计的冻结来源。
- `Data/experiments/_archive_gpt/长窗ROS全链回放_20261008/tools/compare_completed_30.py`：F000–F029 实际 ROS 输入与四数组比较；使用同机 `drone_3d` Python 直接执行，逐窗输出差异及汇总，不依赖旧预测文件。
- `Data/testflight3_longwin/results/F000` 至 `F037`：完整 38 窗重生成参考；原嵌套参考入口的符号链接指向此目录。
- NX 台架隔离根 `<NX 用户目录>/Drone_3D-01-g4-validation/run/g4_validation`：`logs/recovery_tests_20261009_091918.log`、`out/results/sessions/sa5c26bbdeaf682da/window-0000/provenance.json`、`exec/primitive_v2/primitive_summary.json` 及两份 `*_out.bin` 分别提供恢复测试、F000 与原语对拍原始输出。
