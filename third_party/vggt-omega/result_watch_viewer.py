#!/usr/bin/env python3
"""Live result watcher for VGGT-Omega committed reconstruction directories.

This is the recommended "results appear as they are committed" viewer for both
pre-deployment bag simulations and post-deployment real-data runs:

    python3 vggt-omega/result_watch_viewer.py \
        --results-root /path/to/results \
        --server-port 8080

It polls the results root every few seconds, lists every committed result
directory (one containing ``result.json`` + ``predictions.npz`` and not a
hidden staging directory), and auto-opens the newest patch in the existing
browser dashboard.  Every new Atomic ``renameat2(RENAME_NOREPLACE)`` commit
therefore appears as a new row in the history table and becomes the active
3D view.

Each committed result is currently ``MODEL_LOCAL_UNALIGNED`` and does not
share a common coordinate frame with other patches, so the watcher deliberately
does **not** merge point clouds.  Accumulated/aligned merging becomes valid
only after Phase 4 metric/Sim(3) alignment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr

import web_viewer

MAX_RESULT_SCAN_ENTRIES = 500
_DEFAULT_POLL_SEC = 2.0


class ResultStore:
    """Small scan state for committed result directories."""

    def __init__(self, results_root: Path, sensor_dir: str | None):
        self.results_root = results_root
        self.sensor_dir = sensor_dir
        self.entries: list[dict[str, Any]] = []

    def scan(self) -> dict[str, Any]:
        entries = scan_result_directories(self.results_root)
        self.entries = entries
        return {
            "entries": entries,
            "choices": [entry["path"] for entry in entries],
            "labels": [entry["label"] for entry in entries],
        }


def _read_result_summary(path: Path) -> dict[str, Any]:
    result_json = path / "result.json"
    try:
        if result_json.is_symlink() or not result_json.is_file():
            return {"error": "result.json missing or symlink"}
        with open(result_json, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        profile = payload.get("profile") or {}
        stage_timing = payload.get("stage_timing") or {}
        worker_exit = payload.get("worker_exit") or {}
        snapshot = payload.get("snapshot") or {}
        return {
            "scene_id": payload.get("scene_id"),
            "patch_id": payload.get("patch_id"),
            "profile_name": profile.get("name"),
            "image_resolution": profile.get("image_resolution"),
            "frame_group_size": profile.get("frame_group_size"),
            "actual_frame_count": payload.get("actual_frame_count"),
            "worker_wall_sec": stage_timing.get("worker_wall_time_sec"),
            "verifier_wall_sec": stage_timing.get("verifier_wall_time_sec"),
            "worker_cause": worker_exit.get("cause"),
            "worker_returncode": worker_exit.get("returncode"),
            "snapshot_sha256": (snapshot.get("sha256") or "")[:12],
            "error": None,
        }
    except (OSError, ValueError, TypeError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def scan_result_directories(results_root: Path) -> list[dict[str, Any]]:
    """Return committed result directories, newest first.

    A directory counts only when both ``result.json`` and ``predictions.npz``
    exist as regular files and the directory name is not a hidden staging name.
    The scan is metadata-only and never loads NPZ/GLB payloads into memory.
    """
    root = Path(results_root)
    if not root.is_dir():
        return []
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for current, directories, files in os.walk(root):
        directories[:] = [name for name in directories if not name.startswith(".")]
        current_path = Path(current)
        for name in directories:
            if name.startswith(".drone3d-result") or name.startswith("."):
                continue
            candidate = current_path / name
            if candidate in seen:
                continue
            seen.add(candidate)
            result_json = candidate / "result.json"
            predictions = candidate / "predictions.npz"
            if (
                result_json.is_file()
                and not result_json.is_symlink()
                and predictions.is_file()
                and not predictions.is_symlink()
            ):
                summary = _read_result_summary(candidate)
                found.append(
                    {
                        "path": str(candidate),
                        "label": (
                            f"{candidate.name}  ·  "
                            f"{summary.get('profile_name') or summary.get('error')}"
                        ),
                        "summary": summary,
                        "mtime_ns": result_json.stat().st_mtime_ns,
                        "size_bytes": result_json.stat().st_size
                        + predictions.stat().st_size,
                    }
                )
            if len(found) >= MAX_RESULT_SCAN_ENTRIES:
                found.sort(key=lambda entry: entry["mtime_ns"], reverse=True)
                return found
    found.sort(key=lambda entry: entry["mtime_ns"], reverse=True)
    return found


def _history_rows(entries: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for entry in entries:
        summary = entry["summary"]
        rows.append(
            [
                entry["label"],
                summary.get("image_resolution"),
                summary.get("frame_group_size"),
                summary.get("actual_frame_count"),
                summary.get("worker_wall_sec"),
                summary.get("worker_returncode"),
                summary.get("snapshot_sha256"),
                entry["size_bytes"],
            ]
        )
    return rows


def _history_headers() -> list[str]:
    return [
        "结果目录",
        "分辨率",
        "帧组",
        "实际帧",
        "worker wall (s)",
        "worker exit",
        "snapshot sha",
        "result.json+npz bytes",
    ]


def build_watch_ui(
    results_root: Path,
    sensor_dir: str | None,
    poll_sec: float,
) -> gr.Blocks:
    point_cloud = web_viewer.POINT_CLOUD
    color = web_viewer.COLOR
    predicted_depth = web_viewer.PREDICTED_DEPTH
    store = ResultStore(results_root, sensor_dir)
    initial = store.scan()
    history_rows = _history_rows(initial["entries"])

    def refresh(auto_open: bool, selected: str | None) -> tuple:
        scanned = store.scan()
        entries = scanned["entries"]
        choices = scanned["choices"]
        labels = scanned["labels"]
        newest = choices[0] if choices else None
        target = selected
        if auto_open and newest and newest != selected:
            target = newest
        if target not in choices:
            target = newest
        if target is None:
            return (
                gr.update(choices=[], value=None),
                _history_rows([]),
                "正在等待首个 committed 结果…",
                None,
                None,
                None,
                None,
                None,
                "",
            )
        dashboard = web_viewer.load_dashboard(
            target,
            sensor_dir,
            point_cloud,
            color,
            predicted_depth,
            50.0,
            False,
            False,
            True,
            1000,
            1.0,
        )
        # load_dashboard returns (main, status, slider, rgb, depth, frame_glb, meta)
        rows = _history_rows(entries)
        newest_label = entries[0]["label"] if entries else ""
        status = (
            dashboard[1]
            + f"  \n**已发现 {len(entries)} 个 committed 结果；最新：** {newest_label}"
        )
        return (
            gr.update(
                choices=list(zip(labels, choices)) if choices else [],
                value=target,
            ),
            rows,
            status,
            dashboard[0],
            dashboard[2],
            dashboard[3],
            dashboard[4],
            dashboard[5],
            dashboard[6],
        )

    def select_path(path: str | None) -> tuple:
        if not path:
            return None, None, None, None, None, "", "未选择结果。"
        dashboard = web_viewer.load_dashboard(
            path,
            sensor_dir,
            point_cloud,
            color,
            predicted_depth,
            50.0,
            False,
            False,
            True,
            1000,
            1.0,
        )
        return (
            dashboard[0],
            dashboard[2],
            dashboard[3],
            dashboard[4],
            dashboard[5],
            dashboard[6],
            "已切换结果目录：" + path,
        )

    with gr.Blocks(
        theme=gr.themes.Ocean(
            primary_hue="teal", secondary_hue="cyan", neutral_hue="slate"
        ),
        title="Drone3D 重建结果实时看板",
    ) as demo:
        gr.HTML(
            """
            <section style="padding:18px 22px;border:1px solid #173149;border-radius:18px;
            background:linear-gradient(135deg,#0b1d30,#081522)">
              <div style="color:#73e8dd;letter-spacing:.14em;font-weight:700">S380 · Phase 2/3A</div>
              <h1 style="color:#eefbff;margin:4px 0">重建结果实时看板</h1>
              <p style="color:#8eabc1;margin:0">每完成一次验证并通过原子提交，列表新增一行并自动打开最新彩色点云</p>
            </section>
            """
        )
        with gr.Row():
            auto_open = gr.Checkbox(value=True, label="自动打开最新结果")
            refresh_btn = gr.Button("立即扫描", variant="primary")
        status = gr.Markdown("正在扫描结果根目录…")
        with gr.Row(equal_height=True):
            with gr.Column(scale=3, min_width=420):
                result_selector = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="committed 结果",
                    info="只显示包含 result.json + predictions.npz 的正式目录",
                )
                history = gr.Dataframe(
                    headers=_history_headers(),
                    value=history_rows,
                    interactive=False,
                    wrap=True,
                )
            with gr.Column(scale=7, min_width=620):
                main_viewer = gr.Model3D(
                    label="最新彩色点云（MODEL_LOCAL_UNALIGNED）",
                    height=680,
                    display_mode="point_cloud",
                    clear_color=(0.018, 0.045, 0.078, 1.0),
                    interactive=False,
                )
                camera_frame = gr.Slider(0, 0, value=0, step=1, label="相机视角")
                with gr.Row():
                    rgb_view = gr.Image(label="RGB", height=220)
                    depth_view = gr.Image(label="Depth", height=220)
                camera_glb = gr.Model3D(label="当前相机·GLB", height=360, interactive=False)
                frame_meta = gr.Markdown()

        timer = gr.Timer(value=poll_sec, active=True)
        refresh_inputs = [auto_open, result_selector]
        refresh_outputs = [
            result_selector,
            history,
            status,
            main_viewer,
            camera_frame,
            rgb_view,
            depth_view,
            camera_glb,
            frame_meta,
        ]
        timer.tick(refresh, inputs=refresh_inputs, outputs=refresh_outputs)
        refresh_btn.click(refresh, inputs=refresh_inputs, outputs=refresh_outputs)
        result_selector.change(
            select_path,
            inputs=[result_selector],
            outputs=[
                main_viewer,
                camera_frame,
                rgb_view,
                depth_view,
                camera_glb,
                frame_meta,
                status,
            ],
        )
        if store.entries:
            demo.load(
                refresh,
                inputs=refresh_inputs,
                outputs=refresh_outputs,
            )
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        default=os.environ.get(
            "DRONE3D_RESULTS_ROOT", "results"
        ),
        help="Root directory that receives committed result directories",
    )
    parser.add_argument("--sensor-dir", default=None)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=8080)
    parser.add_argument("--poll-sec", type=float, default=_DEFAULT_POLL_SEC)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)
    if not root.is_dir():
        raise SystemExit(f"results root does not exist: {root}")
    demo = build_watch_ui(root, args.sensor_dir, args.poll_sec)
    demo.queue(max_size=20).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
