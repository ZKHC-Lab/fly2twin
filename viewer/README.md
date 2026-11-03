# 网页漫游查看器

2026-10-29 · three.js + Spark，静态部署，无 CDN。

**启动**（仓库根执行）：`python3 -m http.server 8090 --bind 127.0.0.1`，打开 `http://127.0.0.1:8090/viewer/?manifest=/path/to/export/scene.json`。导出目录须在服务根内；需要 scene.json（采集相机、内参、训练走廊）及其 `preferred_model.path` 指向的完整 SH Gaussian PLY。可选 `web_assets` 指向 SPZ 分档；不能仅提供 DC SPLAT。契约见 [INPUTS](../recon/docs/INPUTS.md#display-contract-scene-schema-v2)。

**在本机实测过的启动组合**：本机 Edge 默认使用 Intel 集显，大场景很卡；PRIME 必须配合 `--use-angle=gl-egl` 和 NVIDIA EGL vendor，独立浏览器配置避免转发到已有 Intel 进程。右下角 GPU 标识/其提示中的 renderer 才是实际设备依据，不凭环境变量判断，也不外推帧率。

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
microsoft-edge --user-data-dir="$HOME/.cache/drone-viewer-edge" --use-angle=gl-egl \
  'http://127.0.0.1:8090/viewer/?manifest=/path/to/export/scene.json'
```

**无人机模型**：默认是代码内置的简单四旋翼几何体（无外部素材）。要换成自己的 GLB 模型，在 scene.json 顶层添加以下配置，不需改代码：

```json
"droneModel": {"url":"company-drone.glb", "scale":1, "rotationDeg":[0,90,0], "offset":[0,0,0]}
```

`url` 相对 scene.json；`scale` 为正的等比倍数，默认 1；`rotationDeg` 为 XYZ Euler 角（度），`offset` 为飞行机体局部坐标偏移（场景单位），均默认 `[0,0,0]`。变换顺序为缩放→旋转→偏移，目标方向 +Y 向上、−Z 机头。URL 查询项 `droneUrl`（相对页面）、`droneScale`、`droneRotation=0,90,0`、`droneOffset=0,0,0` 逐项优先覆盖。自定义 GLB 默认不添加旋翼盘；可选 `droneModel.metaUrl`（相对 scene.json）提供 `motor_centers` 与 `rotor_radius_m`，二者均用模型变换前单位。

**多场景与自由漫游**：`?manifest=/a/scene.json,/b/scene.json` 同时载入多个场景，页面上方按钮或 `[` `]` 切换，`&scene=<id>` 指定起始场景；`&guided=0` 关闭采集走廊吸附（G 键切换）。scene.json 可选字段：`id`、`name`、`subtitle`；`bounds`（`min`/`max`，源世界坐标，点云每轴 0.5/99.5 分位范围）；`world_up`（源世界竖直方向，非 Z-up 的场景如图像位姿 SfM 用拟合地面法向，查看器整体旋转到竖直）；`cloud`（点云 PLY，相对 scene.json，`recon export` 默认写 `cloud.ply`；须为 binary little-endian，含 x/y/z 与 red/green/blue），有它时按 V 在 3DGS 与点云之间切换，相机位置不变，点云模式用 58° 视场而不是采集内参。有 bounds 时查看器将其转换到显示坐标后外扩 `?margin=`，默认 1.5 场景单位；没有 bounds 时仍按采集相机外扩 1 m，margin 不影响旧回退。

## 演示

仓库根目录一条命令建站、起服务并用独立 Edge 配置打开；不需要复制大模型，也不需要 NumPy。Python 工具仅用标准库。

```bash
python3 -m viewer.tools.demo /path/to/office/export /path/to/room/export \
  --name 整层办公室 --name 休息室 --start 1 --port 8091 --margin 1.5
# 只打印地址、不打开浏览器
python3 -m viewer.tools.demo /path/to/export --no-launch
# 有头自检（需要桌面 DISPLAY；不会用无头/软件渲染替代）
npm install --prefix viewer/tools
python3 -m viewer.tools.demo /path/to/office/export /path/to/room/export \
  --port 8095 --check /path/to/check-output --no-launch
```

| 参数 | 约定 |
| --- | --- |
| `EXPORT_DIR...` | 一个或多个含 scene.json 的目录，顺序即场景按钮顺序 |
| `--name 名称` | 可重复，按导出顺序补缺失的 name；已有 name 不覆盖，未提供则用目录名 |
| `--start N` | 起始场景序号，从 1 开始，默认 1 |
| `--port` | 默认 8091；同缓存根的本工具服务复用，其他服务占用则报错，不杀进程 |
| `--margin` | 非负有限数，默认 1.5，导出场景单位；只有米制导出才等于米 |
| `--guided` | 开启采集路线吸附；演示默认自由漫游，原始查看器 URL 默认引导 |
| `--no-launch` | 不启动普通浏览器；与 `--check` 同用仍会启动检查用有头浏览器 |
| `--check OUTDIR` | 各场景起点/移动后 PNG（有点云时再加一张点云 PNG）与 check.json；检查失败退出非零 |
| `--browser PATH` | 指定 Edge/Chromium 可执行文件；默认查找 microsoft-edge / microsoft-edge-stable |

站点位于 `$XDG_CACHE_HOME/drone3d-demo/<导出路径哈希>/`（XDG 未设时为 `~/.cache`），viewer、导出目录和覆盖目录里的资产均为符号链接。覆盖 scene.json 保留原字段，补缺失的 name 和 bounds，不修改原包；为多场景补唯一 id。bounds 从 cloud.ply 有限 XYZ 点按每轴 0.5/99.5 线性分位数精确计算，支持 ASCII、大小端 binary PLY 的标量顶点属性，顶点须为第一个非空 element。缺 bounds 且 cloud.ply 不可读时明确失败，不猜测重建范围。已有 bounds 不再计算；world_up、subtitle 和模型相对路径保持不变。

服务只监听 127.0.0.1，所有响应带 `Cache-Control: no-store`，忽略 If-Modified-Since/If-None-Match，防止旧 main.js 缓存造成“场景暂时无法打开”。服务后台保留供再次打开；日志在缓存根 server.log。它是本机可信导出的演示服务，不用于对公网部署。

普通启动每次在缓存根旁 `drone3d-demo-profiles/` 创建新的独立配置，启动前清理该配置 Cache 和 Code Cache，避免转发到用户正在用的集显浏览器。配置及 browser.log 保留供排查，可在对应浏览器退出后自行清理。无 Edge 时只打印 URL；自检没有可用浏览器则失败。Linux 独显环境使用上文三个 PRIME/EGL 变量和 `--use-angle=gl-egl`，不是 GPU 成功保证：以 HUD 提示与 check.json 的实际 GPU 名称为准。

自检用 playwright-core 有头运行每个场景，保持传入的模式：加载成功且无 pageerror/console error、实际 GPU 非软件渲染；自由漫游时 guard 必须为 false。等待开场完成后实际按 W 3 秒、R 2 秒，位移须大于 0.05 场景单位，最终 corridorDistance 须大于 0.3；引导模式检查 guard=true 和移动，明确跳过离开走廊判据。场景有 `cloud` 时再按 V，等点云加载完成。每场景保存两张截图（有点云时三张），全部通过退出 0。画面是否正立仍需目检，工具不会把启动参数当成独显证据。共享 GPU 时应由调用方用自己的 GPU 排队器包装整个检查命令。

## 视角与验收

**初始视角**：scene.json 顶层 `initial_view`（`position`、`target`，世界坐标）；没有时自动选沿采集路线视野最开阔的训练视角（前方 25° 内、1–6 m 处其他相机最多）。自由漫游直接停在该视角，并清除采集相机的横滚以保持竖直；引导模式保留校准姿态与开场采集路线动画。

**验收**：`npm install --prefix viewer/tools`；`node viewer/tools/verify_scene.cjs <服务根> <scene.json绝对路径> <结果.json> 'droneUrl=/other/drone.glb'`，最后一项可省略。服务根下必须有 viewer 和导出目录；工具自带回环 HTTP 服务，有头 Edge（`CHROME` 可换可执行文件），输出实际 GPU、控制台错误、GLB URL/HTTP/网格数与场景/飞行两张 PNG。`CHROME_ARGS='["--use-angle=gl-egl"]'` 可配上述 PRIME 环境验独显；默认不强制 GPU。静态检查：`for f in viewer/src/*.js; do node --check "$f"; done`。

**操作与限制**：WASD/R/F 移动，鼠标拖动转向，Tab 飞行，P 巡航，V 切换 3DGS / 点云（scene.json 有 `cloud` 时），G 切换默认开启的采集走廊引导。需 HTTP/WebGL2；软件渲染默认阻止加载。支持多个导出场景切换，导航只有包围盒、没有表面占据碰撞；离开采集视角会糊，飞行不是飞控仿真。可选压缩：安装 recon、NumPy/SciPy 后执行 `python3 -m viewer.tools.prepare_web --scene /path/to/export/scene.json`，写入同目录 scene-web.json 和三档 SPZ，不覆盖原 manifest；仅支持 SH3 输入，低/中档为有损 SH2，需自行复核画质。
