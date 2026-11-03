#!/usr/bin/env python3
"""Host-side browser smoke for the WP8 target Web dashboard.

NOT part of the ROS/catkin gate. This script starts the ROS-free
``WebDashboardCore`` with fake data, then launches a headless Chromium from
``~/.cache/ms-playwright`` against the running server and checks that the
dashboard DOM contains the required markers. It also writes a screenshot and
runs ``node --check`` on the embedded JavaScript.

Usage:
  python3 web.py [--screenshot-dir DIR]
  python3 web.py --serve-only
"""

from __future__ import annotations

import argparse
import glob
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = (
    REPO_ROOT
    / "ROS_Compatibility_layer"
    / "src"
    / "drone3d_reconstruction_core"
    / "src"
    / "drone3d_reconstruction_core"
)


def _load_ros_free_web_module():
    """Import continuous_web without executing the package __init__ (which imports rospy)."""
    pkg_name = "drone3d_reconstruction_core"
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(PACKAGE_DIR)]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    return pkg


def _session_id(tag: str) -> str:
    return "s" + hashlib.sha1(tag.encode("utf-8")).hexdigest()[:16]


def _make_fake_formal_window(records_root: Path, worker_root: Path) -> str:
    import numpy as np

    session_id = _session_id("browser-smoke")
    map_id = f"{session_id}-m0"
    worker_dir = worker_root / f"{session_id}-w0"
    worker_dir.mkdir(parents=True)

    height, width = 80, 100
    images = np.zeros((1, 3, height, width), dtype=np.float32)
    images[0, 0, ...] = 0.5
    images[0, 1, ...] = 0.4
    images[0, 2, ...] = 0.3
    yy, xx = np.mgrid[0:height, 0:width]
    world = np.zeros((1, height, width, 3), dtype=np.float32)
    world[0, ..., 0] = (xx - width / 2.0) * 0.02
    world[0, ..., 1] = (yy - height / 2.0) * 0.02
    world[0, ..., 2] = 0.08 * np.sin(xx * 0.12) * np.cos(yy * 0.12)
    conf = np.linspace(1.0, 100.0, height * width, dtype=np.float32).reshape(
        1, height, width
    )
    depth = np.ones((1, height, width, 1), dtype=np.float32)
    np.savez(
        str(worker_dir / "predictions.npz"),
        images=images,
        world_points_from_depth=world,
        depth_conf=conf,
        depth=depth,
        extrinsic=np.zeros((1, 3, 4), dtype=np.float32),
    )
    (worker_dir / "manifest.txt").write_text("browser smoke\n", encoding="utf-8")
    (worker_dir / "run.log").write_text("browser smoke\n", encoding="utf-8")
    artifacts = {}
    for name in ("predictions.npz", "manifest.txt", "run.log"):
        artifact = worker_dir / name
        data = artifact.read_bytes()
        artifacts[name] = {
            "uri": artifact.as_uri(),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest = worker_dir / "result.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "drone3d.reconstruction.result",
                "schema_version": 1,
                "artifacts": artifacts,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    snapshot_manifest_bytes = (
        b"path,size,sha256\nsnapshot.json,2,"
        + hashlib.sha256(b"{}").hexdigest().encode("ascii")
        + b"\n"
    )
    alignment = {
        "coordinate_state": "ODOM_LOCAL_APPROX",
        "map_id": map_id,
        "scale": 1.0,
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation": [0.0, 0.0, 0.0],
        "residual_rms": 0.01,
        "correspondence_count": 8,
        "reference_frame_id": "odom",
        "unreliable": True,
        "reason": "smoke",
    }
    provenance = {
        "schema": "drone3d.continuous.formal_record",
        "schema_version": 2,
        "session_id": session_id,
        "window_index": 0,
        "scene_id": f"cs-{session_id}-w0",
        "result_manifest_uri": manifest.as_uri(),
        "result_manifest_sha256": manifest_sha,
        "snapshot_manifest_sha256": hashlib.sha256(
            snapshot_manifest_bytes
        ).hexdigest(),
        "captured_frames": 1,
        "camera_role": "front",
        "camera_topic": "/front/image_raw",
        "camera_frame_id": "front_optical",
        "camera_name": "front_cam",
        "config_digest": "a" * 64,
        "effective_config": {},
        "config_timeline": [],
        "alignment": alignment,
        "gate": {
            "result_manifest_digest_verified": True,
            "identity_verified": True,
            "authenticity_verified": True,
            "schema_verified": True,
        },
        "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        "depth_metrics": {"status": "unavailable", "reason": "probe failure"},
        "depth_metrics_error": "probe failure",
    }
    formal_dir = records_root / session_id / "window-0000"
    formal_dir.mkdir(parents=True)
    provenance_bytes = json.dumps(provenance, sort_keys=True).encode("utf-8")
    (formal_dir / "provenance.json").write_bytes(provenance_bytes)
    (formal_dir / "provenance.json.sha256").write_text(
        hashlib.sha256(provenance_bytes).hexdigest() + "\n", encoding="ascii"
    )
    (formal_dir / "result.json").write_text(manifest.read_text(), encoding="utf-8")
    (formal_dir / "snapshot_manifest.csv").write_bytes(snapshot_manifest_bytes)
    return session_id


def _find_chromium() -> str:
    candidates = sorted(
        glob.glob(
            os.path.expanduser(
                "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome"
            )
        )
    )
    if not candidates:
        raise FileNotFoundError(
            "no chromium binary under ~/.cache/ms-playwright/chromium-*/"
            "chrome-linux/chrome"
        )
    return candidates[-1]


def _extract_js(html: str) -> str:
    start = html.find("<script>")
    end = html.find("</script>", start)
    if start < 0 or end < 0:
        raise RuntimeError("dashboard HTML has no inline <script> block")
    return html[start + len("<script>") : end]


def _layout_probe(html: str) -> str:
    probe = """<script>
window.addEventListener('load',function(){setTimeout(function(){
  function rect(id){var r=document.getElementById(id).getBoundingClientRect();
    return {left:r.left,right:r.right,top:r.top,bottom:r.bottom};}
  function inside(r,v){return r.left>=v.left-1&&r.right<=v.right+1&&r.top>=v.top-1&&r.bottom<=v.bottom+1;}
  var v=rect('viewer'),b=rect('pc-badge'),c=rect('viewer-controls');
  var overlap=Math.max(0,Math.min(b.right,c.right)-Math.max(b.left,c.left))*
    Math.max(0,Math.min(b.bottom,c.bottom)-Math.max(b.top,c.top));
  document.body.setAttribute('data-omega-layout',JSON.stringify({viewer:v,badge:b,controls:c,
    overlap_pixels:overlap,badge_inside:inside(b,v),controls_inside:inside(c,v)}));
},1200);});
</script>"""
    return html.replace("</body>", probe + "</body>")


def _check_layout(dom: str) -> dict:
    class ProbeParser(HTMLParser):
        measurement = None

        def handle_starttag(self, tag, attrs):
            if tag == "body":
                raw = dict(attrs).get("data-omega-layout")
                if raw:
                    self.measurement = json.loads(raw)

    parser = ProbeParser()
    parser.feed(dom)
    value = parser.measurement
    if value is None:
        raise RuntimeError("browser did not measure the viewer layout")
    if value["overlap_pixels"] > 1 or not value["badge_inside"] or not value["controls_inside"]:
        raise RuntimeError("viewer status/controls overlap or exceed viewer bounds: " + json.dumps(value))
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot-dir", default=None)
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="print the fake dashboard URL and block until Ctrl-C",
    )
    parser.add_argument(
        "--keep-server",
        action="store_true",
        help="deprecated alias for --serve-only",
    )
    args = parser.parse_args()

    _load_ros_free_web_module()
    from drone3d_reconstruction_core.continuous_web import (
        WebDashboardConfig,
        WebDashboardCore,
        WebRoleInfo,
    )
    from drone3d_reconstruction_core import continuous_web as web_module
    original_html = web_module._dashboard_html
    web_module._dashboard_html = lambda: _layout_probe(original_html())

    temp = tempfile.TemporaryDirectory(prefix="wp8-web-smoke-")
    root = Path(temp.name)
    result_root = root / "results"
    records_root = result_root / "sessions"
    records_root.mkdir(parents=True)
    worker_root = root / "worker_results"
    worker_root.mkdir()
    session_id = _make_fake_formal_window(records_root, worker_root)

    config = WebDashboardConfig(
        host="127.0.0.1",
        port=0,
        session_command_service="/continuous_session/command",
        result_root=str(result_root),
        roles={
            "front": WebRoleInfo(
                image_topic="/front/image_raw",
                depth_topic="/front/depth",
                camera_name="front_cam",
                camera_frame_id="front_optical",
            )
        },
        web_preview_fps=5.0,
        max_point_cloud_points=100_000,
        conf_percentile=20.0,
    )
    core = WebDashboardCore(
        config,
        get_status=lambda: {
            "state": "RUNNING",
            "session_id": session_id,
            "role": "front",
            "window_index": 1,
            "captured_frames": 3,
            "capture_target_frames": 4,
            "pending_windows": 0,
            "running_window": 2,
            "worker_wall_time_sec": 12.5,
            "succeeded_windows": 1,
            "failed_windows": 0,
            "dropped_windows": 0,
            "memory_mb": 4096,
            "gpu_name": "Orin-GPU",
            "gpu_util": 55.5,
            "temp_c": 61.2,
            "aux_coverage": "rgb ok / depth ok",
            "effective": {
                "sample_fps": 1.0,
                "window_frames": 4,
                "queue_mode": "lossless_fifo",
                "queue_capacity": 8,
                "max_retries": 0,
                "keep_snapshot_on_success": False,
            },
            "fifo": [],
            "fifo_bytes": 0,
            "last_error": "",
            "min_free_disk_gb": 2.0,
            "disk_free_gb": 180.0,
            "disk_redline": False,
            "map_epoch": 0,
            "last_alignment_state": "ODOM_LOCAL_APPROX",
            "formal_windows": 1,
        },
    )
    core.start()
    url = f"http://127.0.0.1:{core.server_port}/"
    try:
        if args.serve_only or args.keep_server:
            print(json.dumps({"ok": True, "url": url, "mode": "serve-only"}))
            sys.stdout.flush()
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                return 0
        with urllib.request.urlopen(url, timeout=5.0) as response:
            html = response.read().decode("utf-8")
        markers = ["累计点云", "实时 RGB", "实时传感器深度", "Omega 首帧深度", "会话与历史"]
        missing = [marker for marker in markers if marker not in html]
        if missing:
            raise RuntimeError(f"dashboard DOM missing markers: {missing!r}")
        if (
            "function pointCloudSpec()" not in html
            or "if(pcSpec.key!==window.__pcKey)" not in html
        ):
            raise RuntimeError(
                "pointcloud full-fingerprint refresh guard is missing"
            )
        if "setInterval(refreshPointCloud,5000)" in html:
            raise RuntimeError("pointcloud still has an unconditional periodic refresh")
        if "clearPointCloud(); if(pcSpec.hasRecords)refreshPointCloud(pcSpec)" not in html:
            raise RuntimeError("session change does not clear stale point-cloud state")
        if (
            "function fullBounds()" not in html
            or "applyViewBounds(fullBounds())" not in html
            or "*0.05" not in html
            or "Reset/全景" not in html
        ):
            raise RuntimeError("Fit/main-subject and Reset/full-view semantics are missing")
        if "var t=[0,0,0]" not in html or "var t=orbit.target" in html:
            raise RuntimeError("coordinate triad is not anchored at world origin")
        if (
            'id="p-inference-resolution" step="16"' not in html
            or "inference_resolution 必须为 0" not in html
        ):
            raise RuntimeError("inference resolution UI guard is missing")
        if (
            "window.__omegaKey!==spec.key" not in html
            or "r.headers.get(\"X-Session\")!==spec.session" not in html
            or "__omegaTimer" in html
        ):
            raise RuntimeError("Omega depth request key/session guard is missing")
        # Dynamic values that the JS must render from /api/state diagnostics.
        dynamic_markers = [
            "4096 MB",
            "Orin-GPU @ 55.5",
            "61.2 ℃",
            "rgb ok / depth ok",
            "窗口 2",
            "12.5",
            "3 / 4",
            "不可用: probe failure",
        ]

        js = _extract_js(html)
        js_path = root / "dashboard.js"
        js_path.write_text(js, encoding="utf-8")
        node_check = subprocess.run(
            ["node", "--check", str(js_path)],
            capture_output=True,
            text=True,
        )
        if node_check.returncode != 0:
            raise RuntimeError(
                "node --check failed: " + node_check.stderr.strip()
            )

        chromium = _find_chromium()
        layouts = []
        for viewport_width in (1440, 1024):
            dump = subprocess.run(
                [
                    chromium,
                    "--user-data-dir=" + str(root / ("chromium-dom-" + str(viewport_width))),
                    "--headless=new",
                    "--no-sandbox",
                    "--use-gl=angle",
                    "--use-angle=swiftshader",
                    "--enable-unsafe-swiftshader",
                    "--virtual-time-budget=5000",
                    f"--window-size={viewport_width},1200",
                    "--dump-dom",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if dump.returncode != 0:
                raise RuntimeError(
                    "chromium --dump-dom failed: " + dump.stderr.strip()
                )
            dom = dump.stdout
            for marker in markers + dynamic_markers:
                if marker not in dom:
                    raise RuntimeError(
                        f"chromium DOM missing marker {marker!r}; stderr={dump.stderr[:400]}"
                    )
            layouts.append({"viewport_width": viewport_width, **_check_layout(dom)})

        screenshot_dir = Path(
            args.screenshot_dir or (root / "screenshots")
        ).resolve()
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot = screenshot_dir / "wp8_web_dashboard.png"
        shot = subprocess.run(
            [
                chromium,
                "--user-data-dir=" + str(root / "chromium-screenshot"),
                "--headless=new",
                "--no-sandbox",
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--virtual-time-budget=5000",
                f"--screenshot={screenshot}",
                "--window-size=1440,1200",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if shot.returncode != 0:
            raise RuntimeError(
                "chromium --screenshot failed: " + shot.stderr.strip()
            )
        if not screenshot.is_file() or screenshot.stat().st_size == 0:
            raise RuntimeError("chromium screenshot file is missing/empty")
        # 点云像素门禁：fake 数据颜色约 (127,102,76)，截图必须出现足量
        # 近似像素，证明 WebGL 真正把点画上屏幕，而不是只加载了页面。
        try:
            from PIL import Image
        except Exception as error:
            raise RuntimeError("PIL is required for screenshot pixel check") from error
        image = Image.open(screenshot).convert("RGB")
        viewer = layouts[0]["viewer"]  # Same 1440x1200 viewport as the screenshot.
        crop = image.crop((round(viewer["left"]), round(viewer["top"]),
                           round(viewer["right"]), round(viewer["bottom"])))
        pixels = list(crop.getdata())
        point_pixels = 0
        point_x = []
        point_y = []
        for offset, (red, green, blue) in enumerate(pixels):
            if (
                abs(red - 127) < 24
                and abs(green - 102) < 24
                and abs(blue - 76) < 24
            ):
                point_pixels += 1
                point_x.append(offset % crop.width)
                point_y.append(offset // crop.width)
        span_x = max(point_x) - min(point_x) if point_x else 0
        span_y = max(point_y) - min(point_y) if point_y else 0
        if (
            point_pixels < 5000
            or span_x < crop.width * 0.2
            or span_y < crop.height * 0.2
        ):
            raise RuntimeError(
                "WebGL point cloud not visibly framed: "
                f"pixels={point_pixels}, span={span_x}x{span_y}"
            )

        print(
            json.dumps(
                {
                    "ok": True,
                    "url": url,
                    "markers": markers,
                    "dynamic_markers": dynamic_markers,
                    "node_check": "PASS",
                    "chromium_dom": "PASS",
                    "viewer_layouts": layouts,
                    "point_pixels": point_pixels,
                    "point_span": [span_x, span_y],
                    "point_pixel_region": "viewer only",
                    "screenshot": str(screenshot),
                    "screenshot_bytes": screenshot.stat().st_size,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        core.stop(timeout_sec=2.0)
        temp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
