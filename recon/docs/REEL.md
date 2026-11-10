# 镜头设计、覆盖度审核与剪辑（reel）

2026-10-30 · 面向导出场景的制片工具。

## 工作流与输入

先写分镜槽位和每段要表达的内容，再搜索合格机位、设计镜头、看覆盖度和联系表，最后渲染与剪辑。画面文字只放标题、小窗名、一行结果数字；帧数、裁切、倍速、指标口径写在制作报告里，不叠在画面上。

场景输入**只有 recon export 导出包目录**，通过 [load_scene](../io/scene.py) 读取 `scene.json`、完整 SH Gaussian PLY、融合点云和相机；不读训练检查点或旧实验布局。位置、注视点、速度、模板距离都在导出世界系中，以米计；竖直方向取 `scene.world_up`，不是固定 Z 轴。输入显示契约见 [INPUTS](INPUTS.md#display-contract-scene-schema-v2)。手机假设相机高度得到的米制尺度不等于实测尺度。

安装 `.[core,train]`；CPU 剪辑和卡片只需 core（OpenCV、Pillow、NumPy、SciPy）。GPU 渲染沿用 torch/gsplat 依赖。`ffmpeg` 是系统外部程序，cut/beats 在执行前检查 PATH；未安装会明确报错。默认字体通过 fontconfig 查找 **Noto Sans CJK SC** 的对应字重和 SC 字面；不静默回退成其它字体。没有该字体时安装 Noto CJK，或显式传字体文件和字面序号。资源队列、内存限制和 CUDA 环境由调用者管理，模块导入不修改环境。

JSON 中的路径相对于该 JSON 文件；Python 传入字典时相对于当前目录。命令行独立的 SCENE、OUT 路径相对于当前目录。

## 六条公开接口

```bash
recon reel shot shots.json frames/ --preview --only shelf aisle
recon reel search export/ templates.json candidates.json --step 30 --min-score 0.9
recon reel growth export/ growth/ --stable --start-frac .55 --grow-s 6 --dissolve-s 2.4 --height 9 --back 11 --trail-smooth 12 --trail-gain .55 --grade '{"exposure":1.3}'
recon reel card title title.png --text '照片级三维场景'
recon reel cut storyboard.json reel.mp4
recon reel beats music.mp3
```

Python 同名函数从 `recon.reel` 导入：

| 函数 | 返回与输出 |
| --- | --- |
| `shot(spec, out, *, preview=False, only=None)` | spec 可为文件路径或字典；返回按镜头名索引的 audit 字典 |
| `search(scene, templates, out, *, step=6, min_score=.9)` | templates 为 JSON 路径或列表；返回各模板候选，写 JSON 和同名 JPG |
| `growth(scene, out, *, start_frac=0, grow_s=5.3, dissolve_s=2.8, height=4.5, back=6.5, orbit_deg=40, trail_smooth=0, stable=False, point_size=.012, grade=None, background=(.055,.067,.086), trail_gain=1, width=1920, fps=30, frames=None)` | 返回并写 `timeline.json`；frames 可选稀疏原时间线帧号 |
| `card(kind, out, **options)` | 写 1920×1080 RGBA PNG，返回输出路径 |
| `cut(storyboard, out)` | storyboard 为路径或字典；返回实际时间线，并写同名 JSON |
| `beats(music)` | 返回 `bpm`、`first_beat_s`、`beat_period_s`、`beats`、`rms_db_per_s`、时长 |

内部 path、coverage、render、grade、templates、cards、assemble 和分析实现可单测，不另设命令行接口。

## 镜头规格

```json
{
  "scene": "export",
  "width": 1920, "height": 1080, "fps": 30,
  "grade": {"exposure": 1.3, "curve": 0.10, "cool": [0.985, 1, 1.03], "saturation": 1.06},
  "background": [0.055, 0.067, 0.086],
  "shots": [
    {"name": "shelf", "template": {"type": "truck", "hero": 12, "length": 2, "dist": 2}, "speed": 0.45, "ease": 1},
    {"name": "aisle", "keys": [
      {"p": [0, 0, 1.4], "look": [0, 3, 1.4]},
      {"p": [0, 1, 1.4], "look": [0, 4, 1.4]}
    ], "speed": 0.5, "ease": 1.2, "mode": "gs"}
  ]
}
```

该例坐标只说明格式，必须换成自己场景中的可见位置。每个镜头给 `keys` 或 `template`。模板 `hero` 是训练相机数组的**零起始序号**，不是 frame ID；`test_frame` 是留出相机的 frame ID。`test_frame` 模式从该相机的位置和视线起步，再按 forward/rise/drop 做连续推进。

| 模板 | 参数（米/度；括号内默认） |
| --- | --- |
| truck | `length`(2)、`dist`(2)、`direction`(1)，相机与注视点平行横移 |
| push | `back`(1)、`forward`(1.5)，注视点在水平前方 3 m |
| arc | `dist`(2)、`a0`(-30)、`a1`(30)，绕注视点弧线；高度沿 world_up 保持 |
| crane | `forward`(1)、`rise`(1)、`drop`(.8)、`dist`(2) |
| pan | `a0`(-30)、`a1`(30)、`dist`(2)，定点转向 |

通用模板偏移：`dz`、`shift_f`、`shift_r` 默认 0；`target_dz` 默认 −.15。上层 speed/ease 优先于模板同名参数。路径用自然三次样条插值位置、展开偏航和俯仰，按“距离 + 1.8 m/rad × 视线转角”重参数化；首尾 smootherstep 速度缓入缓出，横滚相对 world_up 为零。`speed` 默认 .5 m/s（运动长度速度）、`ease` 默认 1.2 s、fps 默认 30。`motion_ok` 检查最大线速度 ≤.85 m/s、偏航 ≤20°/s、俯仰在 ±20°；另记录加速度最大值和 jerk P99。不自动裁剪不合格路径。

`hfov_deg` 缺省保持采集标定内参和主点，提供时改为指定水平视场角的居中针孔。模式：

- `gs`：完整 SH 高斯渲染。
- `points`：融合点云渲染成半径参数 `point_size`（默认 .012 m）的圆高斯。
- `match`：`cut_s` 前为点云，之后硬切高斯。
- `dissolve`：从 `cut_s` 起，用 `fade_s`（默认 .6 s）线性混合点云和高斯。

`grade`、`background` 可逐镜头覆盖。调色不接受场景名：输入 RGB uint8，按 2.2 次幂线性曝光、S 曲线、RGB 通道增益、平均灰度饱和度依次处理；默认 exposure=1、curve=.10、cool=[.985,1,1.03]、saturation=1.06。中性色可显式用 curve=0、cool=[1,1,1]、saturation=1。

输出 `OUT/<name>/%05d.jpg`（JPEG 95）、该目录的 `audit.json` 和 `OUT/<name>_contact.jpg`。预览为 640×360、每 3 帧渲一帧，保留原帧号；正式默认 1920×1080。`frames: [0,30,60]` 可限制渲染帧，仍使用完整路径的时间和运动报告，不把稀疏采样冒充全片。联系表每秒取一帧。

## 覆盖度与搜索

覆盖度在 **160×90** 工作：查询帧 alpha ≥.5 的像素按低分辨率深度回投，再数满足以下条件的训练相机：

1. 位于视锥内、投影到最近像素；
2. 点深度 ≤训练渲染深度的 **1.05** 倍（alpha <.5 的训练像素不遮挡）；
3. 查询/训练视线夹角小于 **30°**；
4. 训练相机到点距离 ≤查询距离的 **1.5** 倍，避免放大低分辨率纹理。

比例以**整幅图**为分母。`hole` 为 alpha 不足像素，`unseen` 为不透明但 0 训练视角的像素；hole 不混入 unseen，但会降低 ge10。分档常数：

| 档 | unseen | 至少 10 视角（ge10） |
| --- | --- | --- |
| A | ≤2% | ≥80% |
| B（未达 A） | ≤5% | ≥60% |
| C | 其它 | 其它 |

历史标定：office_gs_03 T31，366 个留出帧、固定位姿 PSNR；A 门限通过 28 帧，均值/P10 **24.2/22.2 dB**；B 门限通过 82 帧（包含 A），**23.0/20.5 dB**；未过 B 均值 **20.8 dB**。这不是逐帧画质保证；要亲自看联系表上的黑洞、拉丝和漂浮物。

保留迁移参照的深度数值口径：gsplat `RGB+ED` 返回深度再除 alpha。该历史口径在半透明边缘并非严格几何深度，本次未重标定。为使归一化模型与米制导出的裁面一致，低分辨率渲染近裁面为 `.01 × meters_per_unit`，前向可见门槛为 `.02 × meters_per_unit`；比例取导出 `world_from_normalized`，缺失时按 1。模型、相机和路径仍全部在世界系，不读元数据内的 checkpoint 路径。

shot 正式每 10 帧审一次，预览每 9 个原帧审一次；不是逐帧质量保证。search 模板文件为模板对象数组（含 name/type，不带 hero）。每 step 个训练相机尝试一次，以 30 fps 路径每 10 帧审核。运动门槛为 .85 m/s、20°/s、俯仰 ±25°。任何采样 C 档即淘汰，分数 `(A+.5B)/(A+B+C)` 达到 min-score 才保留，降序返回全部候选。联系表为每模板排名前 6 个机位的起点、中点、终点。search 没有合格项时仍写空候选 JSON，不伪造缩略图。

## 生长镜头

按训练相机数组顺序判定融合点首次可见相机。未被低分辨率可见性命中的点按最近相机中心赋出生序号，保证结尾点云完整；这不是原始传感器首次观测时刻。新点在约 20 个相机序号内淡入。`start-frac` 为开场已完成的采集比例，`trail-smooth` 为仅绘制光迹所用的相机序号高斯平滑 σ；不会改变出生判定。

相机默认以 `.75×平滑光迹头+.25×场景中心` 为目标，`--stable` 则固定瞄准训练相机中心的坐标中位数。height 是目标上方距离，back 是水平距离，orbit-deg 是整段绕转角度。默认 point-size=.012 m，光迹增益 trail-gain=1；背景 RGB 浮点通过 `--background R G B` 提供，调色通过 `--grade` JSON 提供。

**`dissolve-s` 保留旧参数名，实际是生长完成后的点云停留尾段**：光迹在尾段开始后的 1.2 s 淡出，不把高空点云溶解成高斯。要点云→高斯溶解，用采集走廊内的 shot mode=dissolve。旧片放弃高空高斯是因为训练视角不支撑该位置；不能靠后处理补回看不到的几何。

## 卡片

kind 为 headline/title/label/number/endcard，全部 1920×1080 RGBA。Python 参数：

- headline、number：`text`；number 可给 `y`（默认 960）。
- title：`text, sub=None, top=False`。
- label：`items=[[text,x,y,anchor],...]`，anchor 为 la/ra/ma。
- endcard：`name, tagline, url, logo=None, footer='开源'`；footer 是最后一行，可换成发布单位全称。
- 通用 `style={font, font_index, background, accent}`；背景/强调色为 RGB 0–255。

CLI 通用 `--font`、`--font-index`、`--background`、`--accent`；标题文字用 `--text`，片尾项目名也用 `--text`，另有 `--tagline`、`--url`、`--logo`、`--footer`。label 可用 `--items` JSON，未给时把 text 放在 (120,100)。

## 分镜格式与剪辑

槽位完全由调用者提供，没有内置片名、SLOTS 或场景名。槽位必须从 0 开始、连续、名称唯一；多段 durations 之和必须填满槽位。交叉淡化向**前**扩展 incoming 首段，不推迟名义槽位起止边界。

```json
{
  "fps": 30, "size": [1920,1080],
  "slots": [
    {"name":"opening","start_s":0,"end_s":3,"transition":"none"},
    {"name":"tour","start_s":3,"end_s":7,"transition":"xfade","xfade_s":0.25}
  ],
  "sources": {
    "opening":{"kind":"image","src":"opening.png","kenburns":[1,1.04],"fit":"contain_blur"},
    "tour":[
      {"kind":"frames","src":"frames/shelf","first":30,"dur_s":2,
       "overlays":[{"png":"title.png","t0":0.2,"t1":1.8,"fade":0.3}]},
      {"kind":"video","src":"capture.mp4","start_s":10,"dur_s":2,"grade":{"exposure":1.1},"gamma":0.9}
    ]
  },
  "overlays": {},
  "music":{"file":"music.mp3","start_s":64,"fade":{"in_s":0.05,"out_s":1.5}}
}
```

该例实际首段 tour 从 2.75 s 开始，持续 2.25 s；第二段在 5 s 开始，末端仍为 7 s。`transition` 为 none/cut/xfade，xfade 默认 .25 s，不能超过上一段长度。实际时间线记录每段 `timeline_start_s`、`dur_s`、`xfade_s` 和名义 slots。内部保持原合成器累计浮点运算顺序，避免 `int(t×fps)` 选源帧时被极小舍入差改成相邻帧。

| kind | 字段 |
| --- | --- |
| frames | `src` 帧目录，`first`(0)、`speed`(1)；数字名 JPG 或 PNG，空缺帧复用之前最近一帧 |
| image | `src`；默认居中填满裁切，`fit: contain_blur` 为完整图片居中+模糊暗化背景 |
| video | `src`、源视频 `start_s`(0)、`speed`(1)、可选 grade |
| wipe | `a` 静态图、`b` 帧目录、`b_first`(0)、`hold_s`、`wipe_s`，`line` 默认白色；静态图逐渐被左到右的帧图揭开，扫完再播放 b |
| color | `rgb: [R,G,B]` |

通用字段 `gamma` 为 8-bit LUT 指数，`kenburns:[from,to]` 为居中缩放倍率，`overlays` 为当前素材局部秒数的叠加列表。叠加记录 `png,t0,t1,fade`，fade 默认 .3 s；按首段交叉淡化长度一起向后补偿，叠加末端不超过素材末端 −.05 s。顶层 overlays 是以槽位名为键的默认叠加；多段中任一段声明自己的 overlays 时，该槽位默认不自动下发。持续跨多段的标签由调用者在每段显式给出，不再特判仿真段、片尾或数字卡。

音频从 start_s 截取，支持 `fade:{in_s,out_s}` 或 `fade_in_s/fade_out_s`（后者优先）。输出 H.264、CRF16、slow、yuv420p；有音乐时 AAC 192k。尺寸必须是正偶数。没有音乐不造空音轨。长度不足的音乐会使 `-shortest` 输出缩短，因此应提供足够长的音源并验收成片时长。beats 使用 22.05 kHz 单声道、STFT 2048/hop512 的正谱通量，70–160 BPM 自相关峰和最强网格相位；不是人工确认的精确节拍标注。

## 验证与限制

CPU 测试覆盖样条首尾连续性和运动上限、非 Z-up 弧线、合成平面可见性/遮挡、A/B/C 边界、槽位和多段时间线、视频源偏移、五类卡片尺寸、依赖缺失报错、调色和叠加。硬件平台为本机，不代表 NX 验收。2026-10-30 全套 CPU 测试 **115 passed**，其中 reel 单测 17 项。

### 迁移回归实测（2026-10-30）

| 回归 | 口径与结果 | 判据 |
| --- | --- | --- |
| R1 shot | office04 同一模型导出；n_shelf/n_aisle 各前 30 帧、1920×1080、JPEG95 解码 RGB；最低/均值分别 64.368/67.841、65.212/68.703 dB | 每帧 >50 dB，通过 |
| R2 growth | office04，stable、start-frac=.55、grow-s=6、dissolve-s=2.4、height=9、back=11、trail-smooth=12、trail-gain=.55；0/60/120/180/240 帧最低/均值 47.410/47.765 dB；192939 点按最近相机赋出生序号 | 五帧 >45 dB，通过 |
| R3 cut | 旧素材转用户槽位格式；1800/1800 帧解码像素相同（MSE=0，PSNR=∞），均为 60.0 s，有音轨 | 同时长、>40 dB、有音轨，通过 |
| R4 arc | 休息室重跑 B0，hero600、±25°、dist1.8、target_dz−.3，11.4 s；预览 115 帧，审核 A/B/C=1/46/0；已亲自检查联系表，沙发/地面朝向正确 | 功能与目检通过，不与旧 B0 逐帧比较 |
| R5 search | office04，step30、min-score.9；shelf_truck/aisle_push 各尝试 98 机位，分别 19/16 候选，最低分 .9118/.9444，所有采样 C=0 | 有合格候选，通过 |

R1/R2 使用曝光1.30、曲线.10、通道增益[.985,1,1.03]、饱和度1.06；R4 曝光.98。R2 首轮未换算近裁面时最低仅32.591 dB，换算后通过；R3 首轮直接使用槽位绝对时间时选源帧相差一帧、总体29.882 dB，保持原浮点累计顺序后像素完全一致。所有渲染以 GPU 排队、12 GiB 进程内存上限执行；未进行 NX 硬件验证。

不做场景外插质量保证。稀疏审核、全帧比例和 SH 渲染都不能替代目检；黑洞和半透明边缘需要人工审查。休息室重跑模型不是旧片训练实例，不作逐帧等价宣称。clip gamma、kenburns 和帧缺失复用的规则为旧工具兼容行为，调用者应提供充分素材长度。

## 证据索引

本地来源标注，不是运行依赖或阅读本文的前提：

- `Data/experiments/reuse_20261030/m2/`：规格、回归脚本、JSON 指标、任务日志和联系表。
- `Data/experiments/promo_20261029/`：参考素材及旧工具，本次没有修改。office04 导出 fingerprint 为 `92c23db1586b282c7267cb4569f67a4bad4d5ef60e82b6fb928123a39fc493de`；休息室用该目录的 `demo/restroom_b0` 重跑导出。
