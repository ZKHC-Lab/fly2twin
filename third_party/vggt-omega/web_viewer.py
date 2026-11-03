# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Browser dashboard for precomputed VGGT-Omega reconstruction output."""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import trimesh
from matplotlib import colormaps
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_util import predictions_to_glb  # noqa: I001


COLOR = "彩色"
GRAY = "灰度"
POINT_CLOUD = "纯点云"
GAUSSIAN = "高斯外观"
PREDICTED_DEPTH = "VGGT-Ω 预测深度"
SENSOR_DEPTH = "RGB-D 实测对齐深度"
DEFAULT_SENSOR_DIR = ""


def _cache_path(output_dir, stem, suffix):
    cache_dir = Path(output_dir) / "viewer_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{stem}{suffix}"


@lru_cache(maxsize=2)
def _load_npz(prediction_path, mtime_ns):
    del mtime_ns
    with np.load(prediction_path) as loaded:
        return {key: np.array(loaded[key]) for key in loaded.files}


def load_predictions(output_dir):
    prediction_path = Path(output_dir) / "predictions.npz"
    if not prediction_path.is_file():
        return None
    return _load_npz(str(prediction_path.resolve()), prediction_path.stat().st_mtime_ns)


@lru_cache(maxsize=8)
def _manifest_names(manifest_path, mtime_ns):
    del mtime_ns
    path = Path(manifest_path)
    names = path.read_text(encoding="utf-8").splitlines()
    return tuple(name for name in names if name and Path(name).name == name)


def frame_names(output_dir, frame_count):
    manifest_path = Path(output_dir) / "manifest.txt"
    if manifest_path.is_file():
        names = _manifest_names(str(manifest_path.resolve()), manifest_path.stat().st_mtime_ns)
        if len(names) == frame_count:
            return names
    return tuple(f"frame_{index:03d}.png" for index in range(frame_count))


def _grayscale_scene(scene):
    for geometry in scene.geometry.values():
        if not isinstance(geometry, trimesh.PointCloud):
            continue
        colors = np.asarray(geometry.colors).copy()
        luminance = np.rint(colors[:, :3] @ np.array([0.299, 0.587, 0.114]))
        colors[:, :3] = luminance[:, None]
        geometry.colors = colors.astype(np.uint8)
    return scene


def _point_cloud_arrays(scene):
    vertices, colors = [], []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        if isinstance(geometry, trimesh.PointCloud):
            vertices.append(trimesh.transform_points(geometry.vertices, transform))
            colors.append(np.asarray(geometry.colors))
    if not vertices:
        raise ValueError("Scene contains no point cloud")
    return np.concatenate(vertices), np.concatenate(colors)


def scene_to_splat(scene, output_path, scale_multiplier=1.0):
    """Write the compact 32-byte Gaussian splat format used by Gradio/Babylon."""
    vertices, colors = _point_cloud_arrays(scene)
    finite = np.isfinite(vertices).all(axis=1)
    vertices, colors = vertices[finite], colors[finite]
    if not len(vertices):
        raise ValueError("Cannot render an empty Gaussian splat")

    lower, upper = np.percentile(vertices, [5, 95], axis=0)
    # ponytail: Omega has no learned covariance; replace this isotropic display
    # scale with trained 3DGS scale/rotation/opacity when those tensors exist.
    base_scale = max(float(np.linalg.norm(upper - lower)) / np.sqrt(len(vertices)), 1e-5)
    scale = np.float32(base_scale * float(scale_multiplier))

    records = np.zeros(
        len(vertices),
        dtype=np.dtype(
            [
                ("position", "<f4", (3,)),
                ("scale", "<f4", (3,)),
                ("rgba", "u1", (4,)),
                ("rotation", "u1", (4,)),
            ]
        ),
    )
    if records.dtype.itemsize != 32:
        raise AssertionError("Unexpected splat record size")
    records["position"] = vertices.astype(np.float32)
    records["scale"] = scale
    records["rgba"] = colors[:, :4].astype(np.uint8)
    records["rgba"][:, 3] = np.minimum(records["rgba"][:, 3], 220)
    records["rotation"] = np.array([255, 128, 128, 128], dtype=np.uint8)

    output_path = Path(output_path)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    records.tofile(temporary)
    os.replace(temporary, output_path)
    return str(output_path)


def _visual_stem(
    render_mode,
    color_mode,
    conf_thres,
    mask_black_bg,
    mask_white_bg,
    show_cam,
    max_points_k,
    splat_scale,
):
    return (
        f"scene_{'splat' if render_mode == GAUSSIAN else 'points'}_"
        f"{'gray' if color_mode == GRAY else 'color'}_conf{float(conf_thres):.1f}_"
        f"black{bool(mask_black_bg)}_white{bool(mask_white_bg)}_"
        f"cam{bool(show_cam)}_max{int(max_points_k)}k_scale{float(splat_scale):.2f}"
    )


def rebuild_visual(
    output_dir,
    render_mode,
    color_mode,
    conf_thres,
    mask_black_bg,
    mask_white_bg,
    show_cam,
    max_points_k,
    splat_scale,
):
    """Build or reuse the selected browser representation without inference."""
    if not output_dir or not Path(output_dir).is_dir():
        return None, "结果目录不存在。"
    predictions = load_predictions(output_dir)
    if predictions is None:
        return None, f"{output_dir} 中没有 predictions.npz。"

    conf_thres = max(3.0, float(conf_thres))
    stem = _visual_stem(
        render_mode,
        color_mode,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        max_points_k,
        splat_scale,
    )
    suffix = ".splat" if render_mode == GAUSSIAN else ".glb"
    visual_path = _cache_path(output_dir, stem, suffix)

    if not visual_path.exists():
        scene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam and render_mode == POINT_CLOUD,
            target_dir=output_dir,
            max_points=int(max_points_k) * 1000,
        )
        if color_mode == GRAY:
            _grayscale_scene(scene)
        if render_mode == GAUSSIAN:
            scene_to_splat(scene, visual_path, splat_scale)
        else:
            scene.export(file_obj=visual_path)

    frame_count = predictions["extrinsic"].shape[0]
    point_count = min(
        int(max_points_k) * 1000,
        int(np.prod(predictions["depth_conf"].shape)),
    )
    note = "Ω 点云转等方性 splat" if render_mode == GAUSSIAN else "GLB 点云"
    camera_note = "；高斯模式不显示相机锥" if render_mode == GAUSSIAN and show_cam else ""
    status = (
        f"**{frame_count} 帧** · {point_count / 1000:.0f}k 点上限 · "
        + f"置信度 ≥ P{conf_thres:.1f} · {note}{camera_note}"
    )
    return str(visual_path), status


def _rgb_array(images, index):
    image = images[index]
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.max(initial=0) <= 1.0:
        image = image * 255
    return np.clip(image, 0, 255).astype(np.uint8)


def _sensor_array(sensor_dir, folder, name):
    if not sensor_dir:
        return None
    path = Path(sensor_dir) / folder / name
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return np.array(image)


def colorize_depth(depth):
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8), None
    low, high = np.percentile(depth[valid], [2, 98])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((depth - low) / (high - low), 0, 1)
    colored = (colormaps["turbo"](normalized)[..., :3] * 255).astype(np.uint8)
    colored[~valid] = (4, 12, 24)
    return colored, (float(low), float(high))


def _frame_glb(
    output_dir,
    predictions,
    index,
    conf_thres,
    mask_black_bg,
    mask_white_bg,
):
    path = _cache_path(
        output_dir,
        f"camera_{index:03d}_conf{float(conf_thres):.1f}_"
        f"black{bool(mask_black_bg)}_white{bool(mask_white_bg)}",
        ".glb",
    )
    if path.exists():
        return str(path)
    frame_count = predictions["extrinsic"].shape[0]
    frame = {
        key: value[index : index + 1]
        if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == frame_count
        else value
        for key, value in predictions.items()
    }
    scene = predictions_to_glb(
        frame,
        conf_thres=max(3.0, float(conf_thres)),
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=True,
        max_points=300_000,
    )
    scene.export(file_obj=path)
    return str(path)


def render_camera_frame(
    output_dir,
    sensor_dir,
    frame_index,
    depth_source,
    conf_thres,
    mask_black_bg,
    mask_white_bg,
):
    predictions = load_predictions(output_dir)
    if predictions is None:
        return None, None, None, "无可用帧。"

    frame_count = predictions["extrinsic"].shape[0]
    index = int(np.clip(round(float(frame_index)), 0, frame_count - 1))
    name = frame_names(output_dir, frame_count)[index]
    sensor_rgb = _sensor_array(sensor_dir, "images", name)
    rgb = sensor_rgb if sensor_rgb is not None else _rgb_array(predictions["images"], index)

    sensor_depth = _sensor_array(sensor_dir, "aligned_depth", name)
    use_sensor = depth_source == SENSOR_DEPTH and sensor_depth is not None
    depth = sensor_depth if use_sensor else predictions["depth"][index, ..., 0]
    depth_image, depth_range = colorize_depth(depth)
    frame_glb = _frame_glb(
        output_dir,
        predictions,
        index,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
    )

    rgb_label = "BAG 原始 RGB" if sensor_rgb is not None else "Ω 模型输入 RGB"
    if use_sensor:
        depth_label = "BAG 16UC1 对齐深度（原始单位）"
    elif depth_source == SENSOR_DEPTH:
        depth_label = "未找到实测深度，已回退到 Ω 预测深度"
    else:
        depth_label = "Ω 预测深度（模型单位）"
    range_text = (
        f"P2–P98: {depth_range[0]:.3g}–{depth_range[1]:.3g}"
        if depth_range
        else "无有效深度"
    )
    metadata = (
        f"**相机 {index + 1:02d}/{frame_count:02d}**  `{name}`  \n"
        f"{rgb_label} · {depth_label} · {range_text}"
    )
    return rgb, depth_image, frame_glb, metadata


def load_dashboard(
    output_dir,
    sensor_dir,
    render_mode,
    color_mode,
    depth_source,
    conf_thres,
    mask_black_bg,
    mask_white_bg,
    show_cam,
    max_points_k,
    splat_scale,
):
    predictions = load_predictions(output_dir)
    if predictions is None:
        return None, "未找到 predictions.npz。", gr.update(maximum=0, value=0), None, None, None, ""
    main, status = rebuild_visual(
        output_dir,
        render_mode,
        color_mode,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        max_points_k,
        splat_scale,
    )
    frame_count = predictions["extrinsic"].shape[0]
    rgb, depth, frame_glb, metadata = render_camera_frame(
        output_dir,
        sensor_dir,
        0,
        depth_source,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
    )
    return (
        main,
        status,
        gr.update(minimum=0, maximum=frame_count - 1, value=0),
        rgb,
        depth,
        frame_glb,
        metadata,
    )


CSS = """
.gradio-container { background: #07111f !important; color: #dce9f7; max-width: 100% !important; }
.contain { max-width: 1880px !important; }
#hero { margin: 0 0 14px; padding: 20px 24px; border: 1px solid #18344b;
  border-radius: 18px; background: radial-gradient(circle at 85% 0%, #0f6d7838, transparent 36%),
  linear-gradient(135deg, #0b1d30 0%, #081522 72%); box-shadow: 0 18px 60px #0007; }
#hero h1 { margin: 0; color: #eefbff !important; font-size: clamp(24px, 2.5vw, 40px);
  letter-spacing: -.04em; }
#hero p { margin: 7px 0 0; color: #8eabc1; }
.omega-badge { color: #73e8dd; font-size: 12px; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; }
.view-card, .control-card { border: 1px solid #173149 !important; border-radius: 16px !important;
  background: linear-gradient(160deg, #0c1b2aee, #091521ee) !important;
  box-shadow: 0 12px 35px #0005; overflow: hidden; }
.view-card .label-wrap { background: #0b1927 !important; border-bottom: 1px solid #173149 !important; }
#main-viewer { min-height: 700px; }
#status, #status * { color: #91c8d2 !important; padding: 2px 8px; }
#frame-meta { min-height: 48px; color: #9eb6c9; }
.primary { box-shadow: 0 8px 24px #0a9c9233; }
@media (max-width: 980px) { #main-viewer { min-height: 520px; } }
"""

# Gradio 5.50 chooses WebGPU whenever the API exists, even when the browser
# cannot obtain an adapter. Babylon's WebGL backend renders both GLB and splat.
WEBGL_HEAD = """
<script>
for (let target = navigator; target; target = Object.getPrototypeOf(target)) {
  const descriptor = Object.getOwnPropertyDescriptor(target, "gpu");
  if (descriptor?.configurable) Reflect.deleteProperty(target, "gpu");
}
</script>
"""


def build_ui(output_dir, sensor_dir):
    theme = gr.themes.Ocean(
        primary_hue="teal",
        secondary_hue="cyan",
        neutral_hue="slate",
    )
    theme.set(
        body_background_fill="#07111f",
        body_text_color="#dce9f7",
        body_text_color_subdued="#8eabc1",
        background_fill_primary="#07111f",
        background_fill_secondary="#0c1b2a",
        block_background_fill="#0c1b2a",
        block_border_color="#173149",
        block_label_background_fill="#0b1927",
        block_label_text_color="#dce9f7",
        input_background_fill="#0f2132",
        input_border_color="#24455f",
        input_placeholder_color="#6f8ba2",
        panel_background_fill="#0c1b2a",
        panel_border_color="#173149",
        accordion_text_color="#dce9f7",
        checkbox_label_background_fill="#102235",
        checkbox_label_background_fill_selected="#43d3c0",
        checkbox_label_text_color="#dce9f7",
        checkbox_label_text_color_selected="#041311",
        button_secondary_background_fill="#102235",
        button_secondary_border_color="#24455f",
        button_secondary_text_color="#dce9f7",
    )
    sensor_available = Path(sensor_dir).is_dir() if sensor_dir else False

    with gr.Blocks(
        theme=theme,
        css=CSS,
        head=WEBGL_HEAD,
        title="VGGT-Ω 三维态势可视化",
    ) as demo:
        gr.HTML(
            """
            <section id="hero">
              <div class="omega-badge">S380 · Spatial Intelligence</div>
              <h1>VGGT-Ω 三维重建态势终端</h1>
              <p>彩色点云、高斯外观与逐相机 RGB-D / GLB 联动预览</p>
            </section>
            """
        )
        output_dir_state = gr.Textbox(value=output_dir, visible=False)
        sensor_dir_state = gr.Textbox(value=sensor_dir or "", visible=False)

        with gr.Row(equal_height=True):
            with gr.Column(scale=7, min_width=620):
                with gr.Row(elem_classes="control-card"):
                    render_mode = gr.Radio(
                        [POINT_CLOUD, GAUSSIAN], value=POINT_CLOUD, label="主视图渲染"
                    )
                    color_mode = gr.Radio([COLOR, GRAY], value=COLOR, label="颜色模式")
                    update_btn = gr.Button("刷新主视图", variant="primary")
                status = gr.Markdown("正在载入重建结果…", elem_id="status")
                main_viewer = gr.Model3D(
                    label="全局三维重建",
                    height=720,
                    display_mode="point_cloud",
                    clear_color=(0.018, 0.045, 0.078, 1.0),
                    zoom_speed=0.25,
                    pan_speed=0.25,
                    interactive=False,
                    elem_id="main-viewer",
                    elem_classes="view-card",
                )

            with gr.Column(scale=4, min_width=390):
                camera_frame = gr.Slider(
                    0, 0, value=0, step=1, precision=0, label="相机视角"
                )
                depth_source = gr.Radio(
                    [PREDICTED_DEPTH, SENSOR_DEPTH],
                    value=SENSOR_DEPTH if sensor_available else PREDICTED_DEPTH,
                    label="深度来源",
                )
                with gr.Row():
                    rgb_view = gr.Image(
                        label="RGB",
                        height=245,
                        show_download_button=True,
                        elem_classes="view-card",
                    )
                    depth_view = gr.Image(
                        label="Depth",
                        height=245,
                        show_download_button=True,
                        elem_classes="view-card",
                    )
                camera_glb = gr.Model3D(
                    label="当前相机·GLB 点云",
                    height=365,
                    display_mode="point_cloud",
                    clear_color=(0.018, 0.045, 0.078, 1.0),
                    zoom_speed=0.35,
                    pan_speed=0.35,
                    interactive=False,
                    elem_classes="view-card",
                )
                frame_metadata = gr.Markdown(elem_id="frame-meta")

        with gr.Accordion("点云过滤与渲染参数", open=False, elem_classes="control-card"):
            with gr.Row():
                conf_thres = gr.Slider(2, 100, value=50, step=0.1, label="置信度百分位")
                max_points_k = gr.Slider(100, 10000, value=1000, step=100, label="最大点数 (K)")
                splat_scale = gr.Slider(0.2, 3.0, value=1.0, step=0.1, label="Splat 尺寸")
            with gr.Row():
                show_cam = gr.Checkbox(label="显示相机锥", value=True)
                mask_black_bg = gr.Checkbox(label="过滤黑色背景", value=False)
                mask_white_bg = gr.Checkbox(label="过滤白色背景", value=False)

        main_inputs = [
            output_dir_state,
            render_mode,
            color_mode,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            max_points_k,
            splat_scale,
        ]
        frame_inputs = [
            output_dir_state,
            sensor_dir_state,
            camera_frame,
            depth_source,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
        ]
        frame_outputs = [rgb_view, depth_view, camera_glb, frame_metadata]

        demo.load(
            load_dashboard,
            inputs=[
                output_dir_state,
                sensor_dir_state,
                render_mode,
                color_mode,
                depth_source,
                conf_thres,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                max_points_k,
                splat_scale,
            ],
            outputs=[
                main_viewer,
                status,
                camera_frame,
                rgb_view,
                depth_view,
                camera_glb,
                frame_metadata,
            ],
        )
        for trigger in (update_btn.click, render_mode.change, color_mode.change):
            trigger(rebuild_visual, inputs=main_inputs, outputs=[main_viewer, status])
        camera_frame.change(render_camera_frame, inputs=frame_inputs, outputs=frame_outputs)
        depth_source.change(render_camera_frame, inputs=frame_inputs, outputs=frame_outputs)

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT-Omega browser dashboard")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="colmap/data_vggt_omega",
        help="Directory containing predictions.npz / scene.glb / manifest.txt",
    )
    parser.add_argument(
        "--sensor-dir",
        default=DEFAULT_SENSOR_DIR,
        help="Optional BAG snapshot containing images/ and aligned_depth/",
    )
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=8080)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_ui(args.output_dir, args.sensor_dir)
    demo.queue(max_size=20).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
