"""CPU-only tests for the ROS-free target Web core (WP8)."""

from __future__ import annotations

import hashlib
import http.client
import json
import struct
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from drone3d_reconstruction_core.continuous_web import (
    COORDINATE_MODEL_LOCAL_UNALIGNED,
    COORDINATE_ODOM_LOCAL_APPROX,
    WebDashboardConfig,
    WebDashboardCore,
    WebDashboardError,
    WebRoleInfo,
)

SESSION_RE = r"s[0-9a-f]{16}"


def _session_id(tag: str) -> str:
    return "s" + hashlib.sha1(tag.encode("utf-8")).hexdigest()[:16]


def _png_magic(data: bytes) -> bool:
    return data[:8] == b"\x89PNG\r\n\x1a\n"


class WebDashboardServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.result_root = self.root / "results"
        self.sessions_root = self.result_root / "sessions"
        self.sessions_root.mkdir(parents=True)
        self.worker_root = self.root / "worker_results"
        self.worker_root.mkdir()
        self.cores = []

    def tearDown(self):
        for core in self.cores:
            try:
                core.stop(timeout_sec=1.0)
            except BaseException:
                pass
        self.cores.clear()

    # ------------------------------------------------------------------
    def _config(self, **overrides):
        defaults = dict(
            host="127.0.0.1",
            port=0,
            session_command_service="/continuous_session/command",
            result_root=str(self.result_root),
            roles={
                "front": WebRoleInfo(
                    image_topic="/front/image_raw",
                    depth_topic="/front/depth",
                    camera_name="front_cam",
                    camera_frame_id="front_optical",
                )
            },
            web_preview_fps=5.0,
            max_session_windows=8,
            max_point_cloud_points=100,
            conf_percentile=20.0,
        )
        defaults.update(overrides)
        return WebDashboardConfig(**defaults)

    def _start_core(self, core):
        self.cores.append(core)
        core.start()
        return core

    def _url(self, core, path):
        return "http://127.0.0.1:%d%s" % (core.server_port, path)

    def _get(self, core, path, expect=200):
        url = self._url(core, path)
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                body = response.read()
                return response.status, dict(response.headers), body
        except urllib.error.HTTPError as error:
            body = error.read()
            if error.code != expect:
                raise AssertionError(
                    "GET %s returned %s, expected %s: %r"
                    % (path, error.code, expect, body[:500])
                )
            return error.code, dict(error.headers), body

    def _post(self, core, payload, expect=200):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._url(core, "/api/command"),
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            body = error.read()
            if error.code != expect:
                raise AssertionError(
                    "POST returned %s, expected %s: %r"
                    % (error.code, expect, body[:500])
                )
            return error.code, dict(error.headers), body

    # ------------------------------------------------------------------
    def _write_formal_window(
        self,
        session_id,
        window_index,
        alignment,
        world,
        images,
        conf,
        depth=None,
        depth_metrics=None,
        depth_metrics_error=None,
    ):
        worker_dir = self.worker_root / f"{session_id}-w{window_index}"
        worker_dir.mkdir(parents=True)
        frames = world.shape[0]
        height, width = world.shape[1], world.shape[2]
        if depth is None:
            depth = np.zeros((frames, height, width, 1), dtype=np.float32)
        extrinsic = np.zeros((frames, 3, 4), dtype=np.float32)
        for frame in range(frames):
            extrinsic[frame, 0, 0] = 1.0
            extrinsic[frame, 1, 1] = 1.0
            extrinsic[frame, 2, 2] = 1.0
        np.savez(
            str(worker_dir / "predictions.npz"),
            images=images,
            world_points_from_depth=world,
            depth_conf=conf,
            depth=depth,
            extrinsic=extrinsic,
        )
        manifest = worker_dir / "result.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "drone3d.reconstruction.result",
                    "schema_version": 1,
                    "scene_id": f"cs-{session_id}-w{window_index}",
                    "actual_frame_count": frames,
                    "coordinate_state": alignment.get(
                        "coordinate_state", COORDINATE_MODEL_LOCAL_UNALIGNED
                    ),
                }
            ),
            encoding="utf-8",
        )
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        provenance = {
            "schema": "drone3d.continuous.formal_record",
            "schema_version": 1,
            "session_id": session_id,
            "window_index": window_index,
            "scene_id": f"cs-{session_id}-w{window_index}",
            "result_manifest_uri": manifest.as_uri(),
            "result_manifest_sha256": manifest_sha,
            "captured_frames": frames,
            "camera_role": "front",
            "camera_topic": "/front/image_raw",
            "camera_frame_id": "front_optical",
            "camera_name": "front_cam",
            "config_digest": "a" * 64,
            "effective_config": {
                "sample_fps": 1.0,
                "window_frames": frames,
                "queue_mode": "lossless_fifo",
                "queue_capacity": 8,
                "max_retries": 0,
                "keep_snapshot_on_success": False,
            },
            "config_timeline": [],
            "alignment": alignment,
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
            "depth_metrics": depth_metrics,
            "depth_metrics_error": depth_metrics_error,
        }
        formal_dir = self.sessions_root / session_id / f"window-{window_index:04d}"
        formal_dir.mkdir(parents=True)
        (formal_dir / "provenance.json").write_text(
            json.dumps(provenance, sort_keys=True), encoding="utf-8"
        )
        (formal_dir / "result.json").write_text(
            manifest.read_text(), encoding="utf-8"
        )
        if alignment.get("coordinate_state") == COORDINATE_ODOM_LOCAL_APPROX:
            (formal_dir / "odom_local_alignment.json").write_text(
                json.dumps(alignment, sort_keys=True), encoding="utf-8"
            )
        return worker_dir

    def _unaligned_alignment(self, reason="not enough odom correspondences"):
        return {
            "coordinate_state": COORDINATE_MODEL_LOCAL_UNALIGNED,
            "map_id": "",
            "scale": None,
            "rotation": None,
            "translation": None,
            "residual_rms": None,
            "correspondence_count": 0,
            "reference_frame_id": "",
            "unreliable": True,
            "reason": reason,
        }

    def _aligned_alignment(self, map_id, scale=1.0, rotation=None, translation=None):
        if rotation is None:
            rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if translation is None:
            translation = [0.0, 0.0, 0.0]
        return {
            "coordinate_state": COORDINATE_ODOM_LOCAL_APPROX,
            "map_id": map_id,
            "scale": scale,
            "rotation": rotation,
            "translation": translation,
            "residual_rms": 0.01,
            "correspondence_count": 12,
            "reference_frame_id": "odom",
            "unreliable": True,
            "reason": "umeyama ok",
        }

    # ------------------------------------------------------------------
    def test_g4_revision_replaces_positions_with_no_new_window_and_keeps_raw_records(self):
        from drone3d_reconstruction_core.g4.backend_process import output_root
        session = _session_id("g4-revisions")
        world = np.array([[[[1., 0., 1.], [2., 0., 1.]]]], dtype=np.float32)
        alignment = {**self._unaligned_alignment(), "method": "global_g4", "pending": True}
        self._write_formal_window(session, 0, alignment, world, np.ones((1,1,2,3))*.5,
                                 np.ones((1,1,2))*100, depth=np.ones((1,1,2,1)))
        recordpath = self.sessions_root/session/"window-0000"/"provenance.json"
        record = json.loads(recordpath.read_text())
        snapshot_sha = hashlib.sha256(b"snapshot manifest test").hexdigest()
        record["snapshot_manifest_sha256"] = snapshot_sha
        (recordpath.parent/"snapshot_manifest.csv").write_bytes(b"snapshot manifest test")
        recordpath.write_text(json.dumps(record))
        original = recordpath.read_bytes()
        (recordpath.parent/"provenance.json.sha256").write_text(hashlib.sha256(original).hexdigest())
        core = WebDashboardCore(self._config())
        with self.assertRaisesRegex(WebDashboardError, "G4 processing"):
            core.build_point_cloud(session, "latest", 100)
        root = output_root(self.sessions_root, session)
        def publish(number, dx):
            directory = root/"revisions"/str(number)
            directory.mkdir(parents=True)
            transform = dict(scale=1., rotation=np.eye(3).tolist(), translation=[dx,0,0])
            revision = dict(session_id=session, revision=number, replacement="complete_map",
                placement="frame_pose", reference_frame_id="odom", windows=[dict(window_id=0,
                included_in_global_preview=True, placement="frame_pose", depth_scale=1.,
                intrinsics=np.eye(3).tolist(), frame_poses=[dict(rotation=np.eye(3).tolist(), center=[dx,0,0])],
                prediction_to_map=transform, global_frame_status="localized", source=dict(
                result_manifest_sha256=record["result_manifest_sha256"], snapshot_manifest_sha256=snapshot_sha,
                shared_prefix=0, frame_ids=[0], gauge=1.))])
            data = json.dumps(revision).encode()
            (directory/"revision.json").write_bytes(data)
            (root/"latest.json").write_text(json.dumps(dict(path=str(directory),sha256=hashlib.sha256(data).hexdigest())))
        publish(1, 0.)
        first = core.build_point_cloud(session, session+"-g4", 100)
        state1 = core.build_state(include_records=False)
        old_pointer = json.loads((root/"latest.json").read_text())
        publish(2, 3.)
        rejected_pointer = json.loads((root/"latest.json").read_text())
        rejected_pointer["display"] = old_pointer
        (root/"latest.json").write_text(json.dumps(rejected_pointer))
        retained = core.build_point_cloud(session, session+"-g4", 100)
        np.testing.assert_array_equal(retained["xyz"], first["xyz"])
        from drone3d_reconstruction_core.g4.map_snapshot import apply_map_snapshot
        from drone3d_reconstruction_core.formal_result import scan_session_windows
        overlay = apply_map_snapshot(list(scan_session_windows(self.sessions_root, session)), self.sessions_root, session)
        self.assertTrue(overlay[0]["alignment"]["map_stale"])
        self.assertEqual(overlay[0]["alignment"]["map_revision"], 1)
        rejected_pointer.pop("display")
        (root/"latest.json").write_text(json.dumps(rejected_pointer))
        second = core.build_point_cloud(session, session+"-g4", 100)
        state2 = core.build_state(include_records=False)
        np.testing.assert_allclose(second["xyz"]-first["xyz"], np.tile([3.,0,0],(len(first["xyz"]),1)))
        self.assertEqual(state1["sessions"][0]["window_count"], state2["sessions"][0]["window_count"])
        self.assertNotEqual(state1["sessions"][0]["map_revision_sha256"], state2["sessions"][0]["map_revision_sha256"])
        self.assertEqual(original, recordpath.read_bytes())
        # A mismatched immutable source must invalidate the entire overlay.
        record["snapshot_manifest_sha256"]="c"*64
        recordpath.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "source binding"):
            from drone3d_reconstruction_core.g4.map_snapshot import apply_map_snapshot
            apply_map_snapshot([record], self.sessions_root, session)

    def test_g4_frame_pose_revision_back_projects_depth_with_each_frame_pose(self):
        from drone3d_reconstruction_core.g4.backend_process import output_root
        session = _session_id("g4-frame-pose")
        # Two frames, one row of two pixels; window-local world points are deliberately wrong.
        world = np.full((2, 1, 2, 3), 99., dtype=np.float32)
        alignment = {**self._unaligned_alignment(), "method": "global_g4", "pending": True}
        self._write_formal_window(session, 0, alignment, world, np.ones((2, 1, 2, 3)) * .5,
                                  np.ones((2, 1, 2)) * 100, depth=np.ones((2, 1, 2, 1)))
        recordpath = self.sessions_root / session / "window-0000" / "provenance.json"
        record = json.loads(recordpath.read_text())
        snapshot_sha = hashlib.sha256(b"snapshot frame pose").hexdigest()
        record["snapshot_manifest_sha256"] = snapshot_sha
        (recordpath.parent / "snapshot_manifest.csv").write_bytes(b"snapshot frame pose")
        recordpath.write_text(json.dumps(record))
        (recordpath.parent / "provenance.json.sha256").write_text(
            hashlib.sha256(recordpath.read_bytes()).hexdigest())
        root = output_root(self.sessions_root, session)

        def publish(second_pose, frame_depth_scales=None):
            directory = root / "revisions" / "1"
            directory.mkdir(parents=True, exist_ok=True)
            window = dict(window_id=0, included_in_global_preview=True, global_frame_status="localized",
                          placement="frame_pose", depth_scale=2.,
                          intrinsics=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                          frame_poses=[{"rotation": np.eye(3).tolist(), "center": [10., 0., 0.]}, second_pose],
                          prediction_to_map=dict(scale=1., rotation=np.eye(3).tolist(), translation=[0., 0., 0.]),
                          source=dict(result_manifest_sha256=record["result_manifest_sha256"],
                                      snapshot_manifest_sha256=snapshot_sha, shared_prefix=0,
                                      frame_ids=[0, 1], gauge=1.))
            if frame_depth_scales is not None:
                window["frame_depth_scales"] = frame_depth_scales
            revision = dict(session_id=session, revision=1, replacement="complete_map",
                            placement="frame_pose", reference_frame_id="odom", windows=[window])
            data = json.dumps(revision).encode()
            (directory / "revision.json").write_bytes(data)
            (root / "latest.json").write_text(json.dumps(dict(path=str(directory),
                                                              sha256=hashlib.sha256(data).hexdigest())))

        # Second frame: rotated 90 deg about z, centred at (0, 5, 0).
        publish({"rotation": [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], "center": [0., 5., 0.]})
        core = WebDashboardCore(self._config())
        cloud = core.build_point_cloud(session, session + "-g4", 100)
        # Pixel (u, v) with K = I and depth 1 * 2 -> camera point (2u, 2v, 2).
        expected = {(10., 0., 2.), (12., 0., 2.), (0., 5., 2.), (0., 7., 2.)}
        self.assertEqual({tuple(np.round(p, 6)) for p in cloud["xyz"].tolist()}, expected)

        # Per-frame scales: frame 0 has none and keeps the window scale 2, frame 1 uses 3.
        publish({"rotation": [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], "center": [0., 5., 0.]}, [None, 3.])
        core = WebDashboardCore(self._config())
        cloud = core.build_point_cloud(session, session + "-g4", 100)
        self.assertEqual({tuple(np.round(p, 6)) for p in cloud["xyz"].tolist()},
                         {(10., 0., 2.), (12., 0., 2.), (0., 5., 3.), (0., 8., 3.)})

        # A frame without a localization pose is dropped, not placed by the window Sim(3).
        publish(None)
        core = WebDashboardCore(self._config())
        cloud = core.build_point_cloud(session, session + "-g4", 100)
        self.assertEqual({tuple(np.round(p, 6)) for p in cloud["xyz"].tolist()}, {(10., 0., 2.), (12., 0., 2.)})

    def test_root_page_and_healthz(self):
        core = self._start_core(WebDashboardCore(self._config()))
        status, _headers, body = self._get(core, "/")
        self.assertEqual(status, 200)
        text = body.decode("utf-8")
        for marker in ("累计点云", "实时 RGB", "实时传感器深度", "Omega 首帧深度", "历史会话"):
            self.assertIn(marker, text)
        self.assertIn('dm.status==="unavailable"', text)
        self.assertIn("depth_metrics_error", text)
        self.assertIn("function omegaSpec()", text)
        self.assertIn('window.__omegaKey!==spec.key', text)
        self.assertIn('r.headers.get("X-Window")!==spec.window', text)
        self.assertNotIn("last_omega_window", text)
        self.assertNotIn("__omegaTimer", text)
        self.assertIn("function pointCloudSpec()", text)
        self.assertIn("function latestRecord(records)", text)
        self.assertIn("selected.map(function(r)", text)
        self.assertIn("window.__pcKey===spec.key", text)
        self.assertIn('r.headers.get("X-Session")!==spec.session', text)
        self.assertIn('var sample="sampling "', text)
        self.assertIn("等效倍率", text)
        self.assertNotIn("· stride ", text)
        self.assertNotIn("setInterval(refreshPointCloud,5000)", text)
        status, _headers, body = self._get(core, "/healthz")
        payload = json.loads(body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["service"], "drone3d-continuous-web")

    def test_state_follow_latest_and_manual_selection(self):
        session_new = _session_id("new")
        session_old = _session_id("old")
        self._write_formal_window(
            session_old,
            0,
            self._unaligned_alignment(),
            np.zeros((1, 1, 2, 3), dtype=np.float32),
            np.zeros((1, 1, 2, 3), dtype=np.float32) + 0.5,
            np.ones((1, 1, 2), dtype=np.float32) * 100.0,
        )
        self._write_formal_window(
            session_new,
            0,
            self._unaligned_alignment(),
            np.zeros((1, 1, 2, 3), dtype=np.float32),
            np.zeros((1, 1, 2, 3), dtype=np.float32) + 0.5,
            np.ones((1, 1, 2), dtype=np.float32) * 100.0,
        )

        def get_status():
            return {"state": "RUNNING", "session_id": session_new}

        core = WebDashboardCore(self._config(), get_status=get_status)
        state = core.build_state()
        self.assertTrue(state["follow_latest"])
        self.assertEqual(state["selected_session"], session_new)

        # Manual selection disables follow_latest.
        result = core.handle_command({"command": "select_session", "session_id": session_old})
        self.assertTrue(result["ok"])
        self.assertFalse(core.follow_latest)
        self.assertEqual(core.selected_session_id, session_old)

        # follow_latest=false blocks automatic reset to the newest session.
        state = core.build_state()
        self.assertEqual(state["selected_session"], session_old)

        # Re-enabling follow_latest snaps back to the current session.
        result = core.handle_command({"command": "follow_latest", "value": True})
        self.assertTrue(result["ok"])
        state = core.build_state()
        self.assertEqual(state["selected_session"], session_new)

    def test_follow_latest_selects_current_session_before_first_formal_window(self):
        current = _session_id("current-empty")
        old = _session_id("old-formal")
        core = WebDashboardCore(
            self._config(),
            get_status=lambda: {"state": "RUNNING", "session_id": current},
            list_sessions=lambda: [
                {"session_id": old, "window_count": 2, "latest_window_index": 1}
            ],
        )

        state = core.build_state()

        self.assertEqual(state["selected_session"], current)
        self.assertEqual(state["formal_records"], [])
        self.assertEqual(state["sessions"][0]["session_id"], current)
        self.assertEqual(state["sessions"][0]["window_count"], 0)

    def test_command_validation_and_dispatch(self):
        sent = []

        def send_command(command):
            sent.append(command)
            return {"ok": True, "error": "", "echo": command}

        core = self._start_core(WebDashboardCore(self._config(), send_command=send_command))
        status, _headers, body = self._post(core, {"command": "pause"})
        self.assertEqual(status, 200)
        payload = json.loads(body.decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(sent[-1], {"command": "pause"})

        # Unknown fields for control commands are rejected.
        status, _headers, body = self._post(
            core, {"command": "pause", "role": "front"}, expect=400
        )
        self.assertIn(b"unknown fields", body)

        # Unknown command rejected.
        self._post(core, {"command": "explode"}, expect=400)

        # set_preview_fps only affects Web polling and never forwards.
        before = len(sent)
        status, _headers, body = self._post(
            core, {"command": "set_preview_fps", "value": 12.5}
        )
        self.assertEqual(status, 200)
        self.assertEqual(len(sent), before)
        self.assertEqual(core.web_preview_fps, 12.5)

        # Strict types.
        self._post(core, {"command": "set_preview_fps", "value": "fast"}, expect=400)
        self._post(core, {"command": "set_preview_fps", "value": -1.0}, expect=400)
        self._post(core, {"command": "follow_latest", "value": "yes"}, expect=400)

    def test_start_session_forwards_only_the_role(self):
        sent = []

        def send_command(command):
            sent.append(command)
            return {"ok": True, "error": ""}

        core = self._start_core(
            WebDashboardCore(self._config(), send_command=send_command)
        )
        self._post(core, {"command": "start_session", "role": "front"}, expect=200)
        self.assertEqual(sent[-1]["command"], "start_session")
        self.assertEqual(sent[-1]["role"], "front")
        self.assertNotIn("reconstruction_mode", sent[-1])
        # 重建模式只有一种，Web 不再接受这个参数；带上它按未知字段拒绝，不会被悄悄忽略。
        for mode in ("per_window", "frame_pool"):
            self._post(
                core,
                {"command": "start_session", "role": "front", "reconstruction_mode": mode},
                expect=400,
            )
        self.assertEqual(len(sent), 1)

    # ------------------------------------------------------------------
    def test_state_formal_records_expose_depth_metrics(self):
        session = _session_id("depthmetrics")
        world = np.zeros((1, 1, 1, 3), dtype=np.float32)
        images = np.zeros((1, 1, 1, 3), dtype=np.float32) + 0.5
        conf = np.ones((1, 1, 1), dtype=np.float32) * 100.0
        depth = np.ones((1, 1, 1, 1), dtype=np.float32)
        self._write_formal_window(
            session,
            0,
            self._unaligned_alignment(),
            world,
            images,
            conf,
            depth=depth,
            depth_metrics={"coverage": 0.8, "abs_rel": 0.12, "rmse": 0.34},
            depth_metrics_error=None,
        )
        core = WebDashboardCore(
            self._config(),
            get_status=lambda: {"state": "RUNNING", "session_id": session},
        )
        state = core.build_state()
        self.assertEqual(len(state["formal_records"]), 1)
        self.assertEqual(
            state["formal_records"][0]["depth_metrics"]["coverage"], 0.8
        )
        self.assertEqual(
            state["formal_records"][0]["depth_metrics"]["abs_rel"], 0.12
        )
        self.assertEqual(
            state["formal_records"][0]["depth_metrics"]["rmse"], 0.34
        )
        self.assertIsNone(state["formal_records"][0]["depth_metrics_error"])

    def test_stream_mjpeg_reads_hot_preview_fps_each_loop(self):
        import drone3d_reconstruction_core.continuous_web as cw

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[..., 2] = 255
        clock = [0.0]
        sleeps = []
        calls = []

        def fake_sleep(seconds):
            clock[0] += seconds
            sleeps.append(seconds)

        core_holder = {}

        def producer(role):
            calls.append(role)
            if len(calls) == 1:
                # 连接建立后热调 web_preview_fps：现有 MJPEG 循环必须生效。
                core_holder["core"].set_preview_fps(1.0)
            if len(calls) <= 3:
                return (float(len(calls)), frame)
            return None

        core = WebDashboardCore(
            self._config(web_preview_fps=10.0),
            read_image=producer,
            now=lambda: clock[0],
        )
        core_holder["core"] = core

        class Wfile:
            def __init__(self):
                self.chunks = []

            def write(self, data):
                self.chunks.append(data)

        class Handler:
            def __init__(self):
                self.wfile = Wfile()

            def send_response(self, *_args, **_kwargs):
                pass

            def send_header(self, *_args, **_kwargs):
                pass

            def end_headers(self):
                pass

        with mock.patch.object(cw.time, "sleep", fake_sleep):
            core._stream_mjpeg(Handler(), "front", "rgb", producer)
        self.assertEqual(len(calls), 4)
        # 初始 fps=10（interval=0.1），热调后 fps=1（interval=1.0）。
        # 若循环只读一次 fps，4 次 producer 调用的总等待约 0.3s；
        # 热调生效时应约 3.0s。
        self.assertGreater(sum(sleeps), 2.5)

    def test_page_is_served_from_the_packaged_static_file(self):
        import drone3d_reconstruction_core.continuous_web_common as common
        static = Path(common.__file__).with_name("static") / "dashboard.html"
        core = self._start_core(WebDashboardCore(self._config()))
        _status, _headers, body = self._get(core, "/")
        self.assertEqual(body, static.read_bytes())

    def test_page_has_view_controls_and_no_session_param_form(self):
        core = self._start_core(WebDashboardCore(self._config()))
        _status, _headers, body = self._get(core, "/")
        html = body.decode("utf-8")
        self.assertNotIn("params-form", html)
        self.assertNotIn("set_params", html)
        self.assertIn('id="preview-form"', html)
        self.assertIn('id="include-rejected"', html)
        self.assertIn('id="map-select"', html)
        self.assertIn("本次新增采集帧 k/N", html)
        self.assertIn("st.capture_target_frames", html)
        self._post(core, {"command": "set_params", "sample_fps": 2.0}, expect=400)

    def test_pointcloud_merge_math_two_aligned_windows(self):
        session = _session_id("pc")
        map_id = f"{session}-m0"
        world0 = np.array(
            [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]],
            dtype=np.float32,
        )
        world1 = world0.copy()
        # conf > percentile(conf, 0) drops only the lowest-confidence point.
        conf = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        images0 = np.zeros((1, 1, 4, 3), dtype=np.float32) + 0.5
        images1 = np.zeros((1, 1, 4, 3), dtype=np.float32) + 0.8
        self._write_formal_window(
            session, 0, self._aligned_alignment(map_id, translation=[0.0, 0.0, 0.0]),
            world0, images0, conf,
        )
        self._write_formal_window(
            session, 1, self._aligned_alignment(map_id, translation=[10.0, 0.0, 0.0]),
            world1, images1, conf,
        )

        core = WebDashboardCore(
            self._config(max_point_cloud_points=100, conf_percentile=0.0)
        )
        result = core.build_point_cloud(session, map_id, 100)
        self.assertTrue(result["aligned"])
        # conf >= percentile(conf, 0) keeps all 4 points per window.
        self.assertEqual(result["original_points"], 8)
        self.assertEqual(result["displayed_points"], 8)
        self.assertEqual(result["windows"], [0, 1])
        xyz = result["xyz"]
        # window0 x = [1,2,3,4]; window1 x = [11,12,13,14]
        self.assertAlmostEqual(float(xyz[:, 0].mean()), 7.5, places=5)
        self.assertTrue(np.allclose(xyz[4:, 0], [11.0, 12.0, 13.0, 14.0]))

        meta = core.point_cloud_meta(result)
        self.assertEqual(meta["coordinate_state"], COORDINATE_ODOM_LOCAL_APPROX)
        self.assertEqual(meta["map_id"], map_id)

    def test_odom_overlap_merge_drops_repeated_frames(self):
        session = _session_id("visual-dedup")
        map_id = f"{session}-v0"
        world = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1)
        world = np.concatenate((world, np.zeros_like(world), np.zeros_like(world)), axis=-1)
        conf = np.ones((4, 1, 1), dtype=np.float32)
        images = np.zeros((4, 1, 1, 3), dtype=np.float32)
        anchor = self._aligned_alignment(map_id)
        anchor["coordinate_state"] = COORDINATE_ODOM_LOCAL_APPROX
        anchor["unreliable"] = False
        anchor["merge_frame_indices"] = [0, 1, 2, 3]
        child = self._aligned_alignment(map_id, translation=[10.0, 0.0, 0.0])
        child["coordinate_state"] = COORDINATE_ODOM_LOCAL_APPROX
        child["unreliable"] = False
        child["merge_frame_indices"] = [2, 3]
        self._write_formal_window(session, 0, anchor, world, images, conf)
        self._write_formal_window(session, 1, child, world, images, conf)

        core = WebDashboardCore(self._config(conf_percentile=0.0))
        result = core.build_point_cloud(session, map_id, 100)

        self.assertEqual(result["coordinate_state"], COORDINATE_ODOM_LOCAL_APPROX)
        self.assertEqual(result["original_points"], 6)
        np.testing.assert_allclose(result["xyz"][-2:, 0], [12.0, 13.0])

    def test_odom_rejected_window_is_hidden_by_default_and_opt_in(self):
        session = _session_id("visual-rejected")
        map_id = f"{session}-v0"
        world = np.array(
            [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]], dtype=np.float32
        )
        conf = np.ones((1, 1, 2), dtype=np.float32)
        images = np.zeros((1, 1, 2, 3), dtype=np.float32)
        anchor = self._aligned_alignment(map_id)
        anchor.update(
            coordinate_state=COORDINATE_ODOM_LOCAL_APPROX,
            quality_passed=True,
            unreliable=False,
            merge_frame_indices=[0],
        )
        rejected = self._aligned_alignment(map_id, translation=[10.0, 0.0, 0.0])
        rejected.update(
            coordinate_state=COORDINATE_ODOM_LOCAL_APPROX,
            quality_passed=False,
            unreliable=True,
            merge_frame_indices=[0],
        )
        self._write_formal_window(session, 0, anchor, world, images, conf)
        self._write_formal_window(session, 1, rejected, world, images, conf)
        core = WebDashboardCore(self._config(conf_percentile=0.0))

        passed = core.build_point_cloud(session, map_id, 100)
        all_candidates = core.build_point_cloud(
            session, map_id, 100, include_rejected=True
        )

        self.assertEqual(passed["windows"], [0])
        self.assertEqual(all_candidates["windows"], [0, 1])
        self.assertFalse(passed["include_rejected"])
        self.assertTrue(all_candidates["include_rejected"])
        np.testing.assert_allclose(all_candidates["xyz"][-2:, 0], [11.0, 12.0])
        self._start_core(core)
        status, headers, _body = self._get(
            core,
            "/api/pointcloud?session=%s&map=%s&include_rejected=1&max_points=100"
            % (session, map_id),
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Include-Rejected"], "1")
        self.assertEqual(headers["X-Windows"], "0,1")
        self._get(
            core,
            "/api/pointcloud?session=%s&map=%s&include_rejected=yes"
            % (session, map_id),
            expect=400,
        )

    def test_pointcloud_constant_conf_renders_all_points(self):
        session = _session_id("const")
        map_id = f"{session}-m0"
        conf = np.full((1, 1, 4), 5.0, dtype=np.float32)
        images = np.zeros((1, 1, 4, 3), dtype=np.float32) + 0.5
        world = np.array(
            [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]],
            dtype=np.float32,
        )
        self._write_formal_window(
            session, 0, self._aligned_alignment(map_id), world, images, conf
        )
        core = WebDashboardCore(
            self._config(max_point_cloud_points=100, conf_percentile=0.0)
        )
        result = core.build_point_cloud(session, map_id, 100)
        self.assertTrue(result["aligned"])
        self.assertEqual(result["original_points"], 4)
        self.assertEqual(result["displayed_points"], 4)

    def test_pointcloud_accumulates_beyond_max_session_windows(self):
        session = _session_id("sl")
        conf = np.array([[[1.0, 2.0]]], dtype=np.float32)
        images = np.zeros((1, 1, 2, 3), dtype=np.float32) + 0.5
        map_id = f"{session}-m0"
        for index in range(10):
            world = np.array(
                [[[[float(index), 0.0, 0.0], [float(index) + 0.5, 0.0, 0.0]]]],
                dtype=np.float32,
            )
            self._write_formal_window(
                session,
                index,
                self._aligned_alignment(map_id),
                world,
                images,
                conf,
            )
        core = WebDashboardCore(
            self._config(max_point_cloud_points=1000, conf_percentile=0.0)
        )
        result = core.build_point_cloud(session, map_id, 1000)
        self.assertTrue(result["aligned"])
        self.assertEqual(len(result["windows"]), 10)
        self.assertEqual(result["original_points"], 20)

    def test_pointcloud_cache_rebuilds_only_when_records_change(self):
        session = _session_id("pc")
        conf = np.array([[[1.0, 2.0]]], dtype=np.float32)
        images = np.zeros((1, 1, 2, 3), dtype=np.float32) + 0.5
        map_id = f"{session}-m0"
        self._write_formal_window(
            session, 0, self._aligned_alignment(map_id),
            np.array([[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]], dtype=np.float32),
            images, conf,
        )
        core = WebDashboardCore(
            self._config(max_point_cloud_points=1000, conf_percentile=0.0)
        )
        first = core.build_point_cloud(session, map_id, 1)
        second = core.build_point_cloud(session, map_id, 1)
        self.assertFalse(first["_cached"])
        self.assertTrue(second["_cached"])
        self.assertTrue(np.array_equal(first["xyz"], second["xyz"]))
        self._write_formal_window(
            session, 1, self._aligned_alignment(map_id),
            np.array([[[[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]], dtype=np.float32),
            images, conf,
        )
        with mock.patch.object(np, "load", wraps=np.load) as load:
            third = core.build_point_cloud(session, map_id, 1)
        self.assertFalse(third["_cached"])
        self.assertEqual(load.call_count, 1)
        self.assertGreater(third["original_points"], first["original_points"])
        self.assertEqual(third["displayed_points"], 1)
        self.assertEqual(third["windows"], [0, 1])
        self.assertEqual(third["sampling"], "uniform_reservoir")
        cold = WebDashboardCore(
            self._config(max_point_cloud_points=1000, conf_percentile=0.0)
        ).build_point_cloud(session, map_id, 1)
        self.assertTrue(np.array_equal(third["xyz"], cold["xyz"]))
        self.assertTrue(np.array_equal(third["rgb"], cold["rgb"]))
        self.assertLessEqual(len(core._pc_cache), 2)

    def test_pointcloud_cache_does_not_retain_full_downsampled_base(self):
        session = _session_id("pc-own")
        map_id = f"{session}-m0"
        points = np.arange(1001, dtype=np.float32).reshape(1, 1, 1001, 1)
        world = np.concatenate(
            [points, np.zeros_like(points), np.zeros_like(points)], axis=-1
        )
        conf = np.arange(1, 1002, dtype=np.float32).reshape(1, 1, 1001)
        images = np.zeros((1, 1, 1001, 3), dtype=np.float32) + 0.5
        self._write_formal_window(
            session, 0, self._aligned_alignment(map_id), world, images, conf
        )
        core = WebDashboardCore(self._config(conf_percentile=0.0))

        result = core.build_point_cloud(session, map_id, 3)

        self.assertLessEqual(result["displayed_points"], 3)
        self.assertIsNone(result["xyz"].base)
        self.assertIsNone(result["rgb"].base)

    def test_pointcloud_concurrent_cold_requests_build_once(self):
        session = _session_id("pc-singleflight")
        map_id = f"{session}-m0"
        world = np.array(
            [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]], dtype=np.float32
        )
        conf = np.array([[[1.0, 2.0]]], dtype=np.float32)
        images = np.zeros((1, 1, 2, 3), dtype=np.float32) + 0.5
        self._write_formal_window(
            session, 0, self._aligned_alignment(map_id), world, images, conf
        )
        core = WebDashboardCore(self._config(conf_percentile=0.0))
        original_load = np.load
        load_calls = []
        barrier = threading.Barrier(3)
        results = []

        def counted_load(*args, **kwargs):
            load_calls.append(args[0])
            time.sleep(0.05)
            return original_load(*args, **kwargs)

        def request():
            barrier.wait()
            results.append(core.build_point_cloud(session, map_id, 100))

        threads = [threading.Thread(target=request) for _ in range(2)]
        with mock.patch.object(np, "load", side_effect=counted_load):
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=2.0)

        self.assertEqual(len(results), 2)
        self.assertEqual(len(load_calls), 1)
        self.assertEqual(sum(bool(result["_cached"]) for result in results), 1)

    def test_pointcloud_unaligned_fallback_does_not_merge(self):
        session = _session_id("un")
        world0 = np.array(
            [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]], dtype=np.float32
        )
        world1 = np.array(
            [[[[9.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]], dtype=np.float32
        )
        conf = np.array([[[1.0, 2.0]]], dtype=np.float32)
        images = np.zeros((1, 1, 2, 3), dtype=np.float32) + 0.5
        self._write_formal_window(session, 0, self._unaligned_alignment(), world0, images, conf)
        self._write_formal_window(session, 1, self._unaligned_alignment("degenerate"), world1, images, conf)

        core = WebDashboardCore(self._config(conf_percentile=0.0))
        result = core.build_point_cloud(session, "latest", 100)
        self.assertFalse(result["aligned"])
        self.assertEqual(result["windows"], [1])
        self.assertEqual(result["original_points"], 2)
        self.assertTrue(np.allclose(result["xyz"][0], [9.0, 0.0, 0.0]))

    def test_pointcloud_max_points_sampling_and_meta(self):
        session = _session_id("stride")
        map_id = f"{session}-m0"
        points = np.arange(10, dtype=np.float32).reshape(1, 1, 10, 1)
        world = np.concatenate([points, np.zeros_like(points), np.zeros_like(points)], axis=-1)
        conf = np.arange(1, 11, dtype=np.float32).reshape(1, 1, 10)
        images = np.zeros((1, 1, 10, 3), dtype=np.float32) + 0.5
        self._write_formal_window(
            session, 0, self._aligned_alignment(map_id), world, images, conf
        )
        core = WebDashboardCore(self._config(max_point_cloud_points=100, conf_percentile=0.0))
        result = core.build_point_cloud(session, map_id, 3)
        # conf >= min keeps every finite-confidence point.
        self.assertEqual(result["original_points"], 10)
        self.assertEqual(result["sampling"], "uniform_reservoir")
        self.assertEqual(result["equivalent_stride"], 4)
        self.assertNotIn("stride", result)
        self.assertEqual(result["displayed_points"], 3)
        meta = core.point_cloud_meta(result)
        self.assertEqual(meta["original_points"], 10)
        self.assertEqual(meta["displayed_points"], 3)
        self.assertEqual(meta["sampling"], "uniform_reservoir")
        self.assertEqual(meta["equivalent_stride"], 4)
        self.assertNotIn("stride", meta)
        self.assertEqual(meta["windows"], [0])

    def test_pointcloud_unknown_session_and_map_errors(self):
        session = _session_id("known")
        map_id = f"{session}-m0"
        world = np.zeros((1, 1, 1, 3), dtype=np.float32)
        conf = np.ones((1, 1, 1), dtype=np.float32) * 100.0
        images = np.zeros((1, 1, 1, 3), dtype=np.float32) + 0.5
        self._write_formal_window(
            session, 0, self._aligned_alignment(map_id), world, images, conf
        )
        core = WebDashboardCore(self._config())
        with self.assertRaises(WebDashboardError):
            core.build_point_cloud(_session_id("missing"), map_id, 100)
        with self.assertRaises(WebDashboardError):
            core.build_point_cloud(session, f"{session}-m99", 100)

    def test_pointcloud_binary_endpoint_and_meta_endpoint(self):
        session = _session_id("bin")
        map_id = f"{session}-m0"
        world = np.array(
            [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
            dtype=np.float32,
        )
        conf = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        images = np.zeros((1, 1, 3, 3), dtype=np.float32) + 0.5
        self._write_formal_window(
            session, 0, self._aligned_alignment(map_id), world, images, conf
        )
        core = self._start_core(WebDashboardCore(self._config(conf_percentile=0.0)))
        path = (
            "/api/pointcloud?session=%s&map=%s&max_points=1"
            % (session, map_id)
        )
        status, headers, body = self._get(core, path)
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("X-Original-Points"), "3")
        self.assertEqual(headers.get("X-Displayed-Points"), "1")
        self.assertEqual(headers.get("X-Aligned"), "1")
        self.assertEqual(headers.get("X-Session"), session)
        self.assertEqual(headers.get("X-Sampling"), "uniform_reservoir")
        self.assertEqual(headers.get("X-Equivalent-Stride"), "3")
        self.assertIsNone(headers.get("X-Stride"))
        count = struct.unpack("<I", body[:4])[0]
        self.assertEqual(count, 1)
        xyz = np.frombuffer(body[4:16], dtype="<f4")
        self.assertTrue(np.isfinite(xyz).all())

        status, _headers, body = self._get(
            core,
            "/api/pointcloud_meta?session=%s&map=%s&max_points=1" % (session, map_id),
        )
        payload = json.loads(body.decode("utf-8"))
        self.assertTrue(payload["aligned"])
        self.assertEqual(payload["original_points"], 3)
        self.assertEqual(payload["displayed_points"], 1)
        self.assertEqual(payload["sampling"], "uniform_reservoir")
        self.assertEqual(payload["equivalent_stride"], 3)
        self.assertNotIn("stride", payload)

    # ------------------------------------------------------------------
    def test_frame_rgb_and_depth_png_magic(self):
        session = _session_id("frame")
        world = np.zeros((1, 2, 2, 3), dtype=np.float32)
        depth = np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=np.float32)
        conf = np.ones((1, 2, 2), dtype=np.float32) * 100.0
        images = np.zeros((1, 2, 2, 3), dtype=np.float32) + 0.5
        self._write_formal_window(
            session, 0, self._unaligned_alignment(), world, images, conf, depth=depth
        )
        core = self._start_core(WebDashboardCore(self._config()))
        status, _headers, body = self._get(
            core, f"/api/window/{session}/0/frames/0?kind=rgb"
        )
        self.assertEqual(status, 200)
        self.assertTrue(_png_magic(body))
        status, _headers, body = self._get(
            core, f"/api/window/{session}/0/frames/0?kind=depth"
        )
        self.assertEqual(status, 200)
        self.assertTrue(_png_magic(body))

    def test_frame_cache_never_bypasses_formal_record_revalidation(self):
        session = _session_id("frame-revalidation")
        world = np.zeros((1, 1, 1, 3), dtype=np.float32)
        conf = np.ones((1, 1, 1), dtype=np.float32) * 100.0
        images = np.zeros((1, 1, 1, 3), dtype=np.float32) + 0.5
        self._write_formal_window(
            session, 0, self._unaligned_alignment(), world, images, conf
        )
        core = self._start_core(WebDashboardCore(self._config()))
        endpoint = f"/api/window/{session}/0/frames/0?kind=rgb"
        status, _headers, body = self._get(core, endpoint)
        self.assertEqual(status, 200)
        self.assertTrue(_png_magic(body))

        provenance = (
            self.sessions_root
            / session
            / "window-0000"
            / "provenance.json"
        )
        provenance.unlink()
        status, _headers, _body = self._get(core, endpoint, expect=404)
        self.assertEqual(status, 404)

    def test_latest_first_depth_endpoint(self):
        session = _session_id("omega")
        depth0 = np.array([[[[1.0]]]], dtype=np.float32)
        depth1 = np.array([[[[9.0]]]], dtype=np.float32)
        world = np.zeros((1, 1, 1, 3), dtype=np.float32)
        conf = np.ones((1, 1, 1), dtype=np.float32) * 100.0
        images = np.zeros((1, 1, 1, 3), dtype=np.float32) + 0.5
        self._write_formal_window(session, 0, self._unaligned_alignment(), world, images, conf, depth=depth0)
        self._write_formal_window(session, 1, self._unaligned_alignment(), world, images, conf, depth=depth1)
        core = self._start_core(WebDashboardCore(self._config()))
        status, headers, body = self._get(
            core, f"/api/latest_first_depth?session={session}"
        )
        self.assertEqual(status, 200)
        self.assertTrue(_png_magic(body))
        self.assertEqual(headers.get("X-Session"), session)
        self.assertEqual(headers.get("X-Window"), "1")
        self.assertEqual(headers.get("X-Frame"), "0")

    # ------------------------------------------------------------------
    def test_stream_rgb_mjpeg_and_depth_unavailable_404(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[..., 2] = 255

        class Producer:
            def __init__(self):
                self.calls = 0

            def __call__(self, role):
                self.calls += 1
                if self.calls <= 2:
                    return (time.time(), frame)
                return None

        producer = Producer()
        core = self._start_core(
            WebDashboardCore(self._config(), read_image=producer, read_depth=lambda role: None)
        )
        conn = http.client.HTTPConnection("127.0.0.1", core.server_port, timeout=5.0)
        conn.request("GET", "/stream/front/rgb")
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(
            response.getheader("Content-Type"),
            "multipart/x-mixed-replace; boundary=frame",
        )
        body = response.read()
        conn.close()
        self.assertIn(b"--frame\r\n", body)
        self.assertIn(b"\xff\xd8", body)  # JPEG SOI
        self.assertGreaterEqual(producer.calls, 2)

        # Depth unavailable must be an explicit 404, never a fake stream.
        conn = http.client.HTTPConnection("127.0.0.1", core.server_port, timeout=5.0)
        conn.request("GET", "/stream/front/depth")
        response = conn.getresponse()
        self.assertEqual(response.status, 404)
        body = response.read()
        conn.close()
        self.assertIn(b"sensor depth unavailable", body)

        # A producer returning (stamp, None) must be treated as "no frame",
        # bounded-wait 404, never an empty 200 multipart stream.
        core.stop(timeout_sec=1.0)

        def no_frame_producer(role):
            return (time.time(), None)

        core2 = self._start_core(
            WebDashboardCore(self._config(), read_image=no_frame_producer)
        )
        conn = http.client.HTTPConnection("127.0.0.1", core2.server_port, timeout=5.0)
        conn.request("GET", "/stream/front/rgb")
        response = conn.getresponse()
        self.assertEqual(response.status, 404)
        body = response.read()
        conn.close()
        self.assertIn(b"image unavailable", body)

    def test_path_traversal_rejected(self):
        core = self._start_core(WebDashboardCore(self._config()))
        # Encoded traversal in the session segment must be rejected before FS access.
        status, _headers, _body = self._get(
            core, "/api/window/..%2F..%2Fetc%2Fpasswd/0/frames/0?kind=rgb", expect=400
        )
        self.assertEqual(status, 400)
        status, _headers, _body = self._get(
            core, "/api/pointcloud?session=..%2F..%2Fetc&map=x&max_points=10", expect=400
        )
        self.assertEqual(status, 400)

    def test_pointcloud_chw_images_layout(self):
        # Real Omega predictions.npz uses images (N,3,H,W); pointcloud and
        # frame endpoints must both normalize that to HWC RGB.
        session = _session_id("chw")
        map_id = f"{session}-m0"
        world = np.array([[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]], dtype=np.float32)
        conf = np.array([[[1.0, 2.0]]], dtype=np.float32)
        images_chw = np.zeros((1, 3, 1, 2), dtype=np.float32)
        images_chw[0, 0, 0, 0] = 0.2
        images_chw[0, 1, 0, 0] = 0.4
        images_chw[0, 2, 0, 0] = 0.6
        depth = np.array([[[[1.0], [2.0]]]], dtype=np.float32)
        self._write_formal_window(
            session,
            0,
            self._aligned_alignment(map_id),
            world,
            images_chw,
            conf,
            depth=depth,
        )
        core = self._start_core(WebDashboardCore(self._config(conf_percentile=0.0)))
        result = core.build_point_cloud(session, map_id, 100)
        # conf >= min keeps both pixels.
        self.assertEqual(result["original_points"], 2)
        self.assertTrue(np.allclose(result["xyz"][0], [1.0, 0.0, 0.0]))
        # Frame endpoint must also accept CHW images.
        status, _headers, body = self._get(
            core, f"/api/window/{session}/0/frames/0?kind=rgb"
        )
        self.assertEqual(status, 200)
        self.assertTrue(_png_magic(body))

    def test_pointcloud_huge_finite_scale_never_emits_nan(self):
        session = _session_id("huge")
        map_id = f"{session}-m0"
        world = np.array(
            [[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]], dtype=np.float32
        )
        conf = np.array([[[1.0, 2.0]]], dtype=np.float32)
        images = np.zeros((1, 1, 2, 3), dtype=np.float32) + 0.5
        self._write_formal_window(
            session,
            0,
            self._aligned_alignment(map_id, scale=1e300),
            world,
            images,
            conf,
        )
        core = self._start_core(WebDashboardCore(self._config(conf_percentile=0.0)))
        status, _headers, body = self._get(
            core,
            f"/api/pointcloud?session={session}&map={map_id}&max_points=10",
            expect=404,
        )
        self.assertEqual(status, 404)
        self.assertNotIn(b"nan", body.lower())
        self.assertNotIn(b"inf", body.lower())

    def test_build_state_does_not_hold_lock_during_get_status(self):
        # get_status re-enters the core (set_preview_fps takes the same lock).
        # If build_state called get_status while holding the lock this would
        # deadlock; the daemon-thread timeout below turns a deadlock into a
        # clean failure instead of hanging the suite.
        result = {}
        core = WebDashboardCore(self._config())

        def get_status():
            core.set_preview_fps(7.5)
            return {"state": "RUNNING", "session_id": _session_id("lock")}

        core = WebDashboardCore(self._config(), get_status=get_status)

        def worker():
            result["ok"] = core.build_state() is not None

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(3.0)
        self.assertFalse(
            thread.is_alive(), "build_state deadlocked while calling get_status"
        )
        self.assertTrue(result.get("ok", False))

    def test_config_preflight_rejects_bad_values(self):
        with self.assertRaises(WebDashboardError):
            self._config(web_preview_fps=0.0)
        with self.assertRaises(WebDashboardError):
            self._config(max_point_cloud_points=0)
        with self.assertRaises(WebDashboardError):
            self._config(conf_percentile=100.1)
        with self.assertRaises(WebDashboardError):
            self._config(max_predictions_bytes=0)
        with self.assertRaises(WebDashboardError):
            self._config(port=70000)
        with self.assertRaises(WebDashboardError):
            self._config(result_root=str(self.root / "missing"))

    def test_healthz_and_state_endpoint_shape(self):
        session = _session_id("shape")
        self._write_formal_window(
            session,
            0,
            self._unaligned_alignment(),
            np.zeros((1, 1, 1, 3), dtype=np.float32),
            np.zeros((1, 1, 1, 3), dtype=np.float32) + 0.5,
            np.ones((1, 1, 1), dtype=np.float32) * 100.0,
        )
        def get_status():
            return {
                "state": "RUNNING",
                "session_id": session,
                "window_index": 7,
                "captured_frames": 43,
                "capture_target_frames": 80,
                "running_window": 2,
                "worker_wall_time_sec": 12.5,
                "service_wall_time_sec": 13.5,
                "service_budget_sec": 16.0,
                "capture_queue_blocked": True,
                "memory_mb": 4096,
                "gpu_name": "Orin-GPU",
                "gpu_util": 55.5,
                "temp_c": 61.2,
                "aux_coverage": "rgb ok / depth ok",
                "capture_disk_bytes": 123456,
                "available_roles": ["front"],
            }

        core = self._start_core(
            WebDashboardCore(self._config(), get_status=get_status)
        )
        status, _headers, body = self._get(core, "/api/state")
        self.assertEqual(status, 200)
        payload = json.loads(body.decode("utf-8"))
        self.assertTrue(payload["follow_latest"])
        self.assertEqual(payload["selected_session"], session)
        self.assertEqual(payload["roles"], ["front"])
        self.assertIn("session_id", payload["status"])
        self.assertEqual(payload["status"]["captured_frames"], 43)
        self.assertEqual(payload["status"]["capture_target_frames"], 80)
        self.assertEqual(payload["status"]["service_wall_time_sec"], 13.5)
        self.assertEqual(payload["status"]["service_budget_sec"], 16.0)
        self.assertTrue(payload["status"]["capture_queue_blocked"])
        self.assertIn("diagnostics", payload)
        # Raw diagnostics must pass through untouched, and compact status must
        # keep the worker object/int for the frontend.
        self.assertEqual(payload["diagnostics"]["memory_mb"], 4096)
        self.assertEqual(payload["diagnostics"]["gpu_name"], "Orin-GPU")
        self.assertEqual(payload["diagnostics"]["temp_c"], 61.2)
        self.assertEqual(payload["diagnostics"]["aux_coverage"], "rgb ok / depth ok")
        self.assertEqual(payload["status"]["running_window"], 2)
        self.assertEqual(payload["status"]["window_index"], 7)
        self.assertEqual(payload["status"]["capture_disk_bytes"], 123456)
        self.assertEqual(payload["status"]["available_roles"], ["front"])
        self.assertTrue(any(record["session_id"] == session for record in payload["formal_records"]))


    def test_polling_keeps_g4_status_from_ros_scalar_diagnostics(self):
        core = self._start_core(WebDashboardCore(self._config(), get_status=lambda: {
            "state": "RUNNING", "g4_state": "processing", "g4_revision": "17",
            "g4_error": "waiting for usable geometry", "alignment_backend": "global_g4",
        }))
        status, _, body = self._get(core, "/api/state?include_records=0")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["diagnostics"]["g4"], {
            "state": "processing", "revision": "17", "error": "waiting for usable geometry",
            "exit_code": None})

    def test_memory_recovery_diagnostics_decode_ros_scalars(self):
        for raw, expected in (("unavailable", None), ("0", 0), ("-9", -9)):
            with self.subTest(exit_code=raw):
                core = WebDashboardCore(self._config(), get_status=lambda: {
                    "state": "MEMORY_PAUSED", "memory_paused": "True",
                    "oom_latched": "False", "retry_window": "30",
                    "attempt_failures": "2", "manual_retries": "1",
                    "g4_state": "error", "g4_exit_code": raw,
                })
                state = core.build_state(include_records=False)["status"]
                self.assertIs(state["memory_paused"], True)
                self.assertIs(state["oom_latched"], False)
                self.assertEqual(state["retry_window"], 30)
                self.assertEqual(state["attempt_failures"], 2)
                self.assertEqual(state["manual_retries"], 1)
                self.assertEqual(state["g4"]["exit_code"], expected)


class G4HintRowDisplayTests(unittest.TestCase):
    """G4 提示行：后台退出/报错时必须显示状态与退出码，而不是“无错误”。

    行文本由页面里的 JS 现场求值，避免只做字符串断言：用页面自身导出的
    esc/fmt/fmtOrNA 与这一行拼成脚本，交给 node 跑出真实显示文本。
    """

    def setUp(self):
        from drone3d_reconstruction_core.continuous_web import _dashboard_html
        self.html = _dashboard_html()

    def js_fragment(self):
        """从页面里取出该行的 JS 片段与它用到的三个格式化函数。"""
        lines = self.html.splitlines()
        row = [line for line in lines if line.strip().startswith("var g4s=")
               or 'row("G4 提示"' in line]
        self.assertEqual(len(row), 2, "G4 提示行不在页面里或已被改写")
        helpers = []
        for name in ("esc", "fmt", "fmtOrNA"):
            match = [line for line in lines if line.startswith("function %s(" % name)]
            self.assertEqual(len(match), 1, "页面里没有唯一的 %s" % name)
            helpers.append(match[0])
        return "\n".join(helpers), "\n".join(row)

    def hint_text(self, status):
        import shutil
        import subprocess
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not available to evaluate the dashboard script")
        helpers, row = self.js_fragment()
        script = "\n".join([
            helpers,
            "var rows=[];function row(k,v){rows.push(v)}",
            "var st=%s;" % json.dumps(status, ensure_ascii=False),
            row,
            "process.stdout.write(JSON.stringify(rows));",
        ])
        done = subprocess.run([node, "-e", script], capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)[0]

    def test_running_backend_shows_no_error(self):
        self.assertEqual(self.hint_text({"g4": {"state": "processing"}}), "无错误")

    def test_backend_without_state_waits_for_startup(self):
        self.assertEqual(self.hint_text({"g4": {}}), "等待后台启动")
        self.assertEqual(self.hint_text({}), "等待后台启动")

    def test_failed_backend_shows_state_and_exit_code(self):
        self.assertEqual(self.hint_text({"g4": {"state": "error", "exit_code": 3}}),
                         "状态 error 退出码 3")
        self.assertEqual(self.hint_text({"g4": {"state": "stopped", "exit_code": 0}}),
                         "状态 stopped 退出码 0")

    def test_error_message_is_shown_once_and_escaped(self):
        self.assertEqual(
            self.hint_text({"g4": {"state": "error", "exit_code": 1,
                                   "error": "boom & <b>stop</b>"}}),
            "boom &amp; &lt;b&gt;stop&lt;/b&gt;")

    def test_row_reads_the_fixed_guard_and_not_the_old_expression(self):
        _, row = self.js_fragment()
        self.assertIn("g4s.exit_code!=null", row)
        # 退出码 0 也是“已退出”，不能用裸真值判断（旧写法 ||g4s.exit_code)）。
        self.assertNotIn("||g4s.exit_code);", row)
        self.assertNotIn("esc(g4s.error", row)


class G4FailureBadgeTests(unittest.TestCase):
    """顶栏的“地图后端出错”徽标：只在后端真的失败时出现，正常停止不算。"""

    def setUp(self):
        from drone3d_reconstruction_core.continuous_web import _dashboard_html
        self.html = _dashboard_html()

    def test_badge_is_in_the_page_and_hidden_by_default(self):
        self.assertIn('<span class="badge unreliable" id="g4-badge" hidden>', self.html)
        self.assertIn('g4Badge.hidden=!g4Failed(g4s)', self.html)

    def test_g4_failed_predicate(self):
        import shutil
        import subprocess
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not available to evaluate the dashboard script")
        predicate = [line for line in self.html.splitlines()
                     if line.startswith("function g4Failed(")]
        self.assertEqual(len(predicate), 1)
        cases = [
            (None, False),
            ({}, False),
            ({"state": "optimizing"}, False),
            ({"state": "ready", "revision": 4}, False),
            ({"state": "stopped", "exit_code": 0}, False),
            ({"state": "error", "error": "G4 background start failed: OSError: x"}, True),
            ({"state": "error", "exit_code": -9}, True),
            ({"state": "error", "exit_code": None}, True),
        ]
        script = "\n".join([
            predicate[0],
            "var cases=%s;" % json.dumps([case for case, _ in cases]),
            "process.stdout.write(JSON.stringify(cases.map(g4Failed)));",
        ])
        done = subprocess.run([node, "-e", script], capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout), [expected for _, expected in cases])


class PauseResumeControlGatingTests(unittest.TestCase):
    """MEMORY_PAUSED 必须在 Web 上给出人工重试按钮；FATAL_OOM 保持禁用。"""

    def setUp(self):
        from drone3d_reconstruction_core.continuous_web import _dashboard_html
        self.html = _dashboard_html()

    def control_state(self, status):
        """用页面里真实的 refreshControls 求值按钮的 disabled / 文案。"""
        import shutil
        import subprocess
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not available to evaluate the dashboard script")
        lines = self.html.splitlines()
        start = next(index for index, line in enumerate(lines)
                     if line.startswith("function refreshControls("))
        end = next(index for index in range(start, len(lines))
                   if lines[index].startswith("function setCommandsDisabled("))
        body = "\n".join(lines[start:end])
        script = "\n".join([
            "function $(id){return elements[id]||(elements[id]={disabled:false,textContent:'',title:'',"
            "selectedOptions:[{disabled:false}],value:'',style:{}})}",
            "var elements={};var commandInFlight=false;",
            "function esc(s){return String(s)}",
            "var document={querySelectorAll:function(){return []}};",
            body,
            "refreshControls(%s);" % json.dumps(status, ensure_ascii=False),
            "process.stdout.write(JSON.stringify({disabled:elements['btn-pause-resume'].disabled,"
            "text:elements['btn-pause-resume'].textContent,title:elements['btn-pause-resume'].title,"
            "stop:elements['btn-stop'].disabled}));",
        ])
        done = subprocess.run([node, "-e", script], capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr)
        return json.loads(done.stdout)

    def test_memory_paused_enables_the_manual_retry_button(self):
        state = self.control_state({"state": "MEMORY_PAUSED"})
        self.assertFalse(state["disabled"])
        self.assertEqual(state["text"], "重试资源失败窗")
        self.assertIn("重试", state["title"])

    def test_running_and_paused_labels_are_unchanged(self):
        self.assertEqual(self.control_state({"state": "PAUSED"})["text"], "恢复")
        self.assertEqual(self.control_state({"state": "RUNNING"})["text"], "暂停")

    def test_oom_pause_never_offers_resume(self):
        for status in ("FATAL_OOM", "IDLE"):
            state = self.control_state({"state": status})
            self.assertTrue(state["disabled"], status)
            self.assertNotEqual(state["text"], "重试资源失败窗")
        # FATAL_OOM 仍可 STOP（保留故障现场）。
        self.assertFalse(self.control_state({"state": "FATAL_OOM"})["stop"])

    def test_click_sends_resume_for_both_paused_states(self):
        lines = self.html.splitlines()
        handler = next(line for line in lines
                       if line.strip().startswith('$("btn-pause-resume").onclick='))
        self.assertIn('s==="MEMORY_PAUSED"', handler)
        self.assertIn('"resume"', handler)


if __name__ == "__main__":
    unittest.main()
