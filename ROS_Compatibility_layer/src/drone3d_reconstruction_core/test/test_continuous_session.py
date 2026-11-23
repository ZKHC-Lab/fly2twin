"""CPU-only tests for the continuous session scheduler core."""

import sys
import copy
import dataclasses
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import unquote, urlsplit

# The shared v4 snapshot fixture lives next to this file; catkin's nose run does not add this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from snapshot_v4 import write_v4_snapshot  # noqa: E402
from drone3d_reconstruction_core.capture import CaptureConfig
from drone3d_reconstruction_core.action_types import (
    ACTION_FAILED,
    ACTION_SUCCEEDED,
    CameraRole,
    CaptureActionOutcome,
    ReconstructActionOutcome,
)
from drone3d_reconstruction_core.snapshot_validator import SnapshotSafetyLimits
from drone3d_reconstruction_core.continuous_session import (
    ContinuousConfig,
    ContinuousConfigError,
    ContinuousSession,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    QUEUE_LOSSLESS,
    SESSION_DISK_PAUSED,
    SESSION_FATAL_OOM,
    SESSION_IDLE,
    SESSION_MEMORY_PAUSED,
    SESSION_PAUSED,
    SESSION_RUNNING,
    SESSION_DRAINING,
    SESSION_STOPPED,
    SESSION_STOPPING,
    WINDOW_CANCELLED,
    WINDOW_CAPTURING,
    WINDOW_FAILED,
    WINDOW_SUCCEEDED,
    WindowRecord,
    JobStatus,
    ReconstructionBackend,
)


def make_capture_config(root: Path, role: str) -> CaptureConfig:
    snapshots = root / f"{role}_snapshots"
    snapshots.mkdir()
    return CaptureConfig(
        snapshot_root=str(snapshots),
        image_topic=f"/{role}/image_raw",
        camera_name=f"{role}_camera",
        camera_role=role,
        camera_frame_id=f"{role}_camera_optical_frame",
        image_encoding="bgr8",
        odometry_topic=None,
        camera_info_topic=None,
        depth_topic=None,
        tf_topic=None,
        tf_static_topic=None,
        save_sensor_depth=False,
        body_frame_id=f"{role}_body",
        tf_relevant_frames=(),
        camera_info_max_delta_ms=100.0,
        depth_max_delta_ms=100.0,
        max_auxiliary_samples=8,
        max_tf_samples=64,
        max_tf_bytes=256 * 1024,
        source_timeout_sec=5.0,
        max_window_gap_sec=30.0,
        min_sample_fps=0.5,
        max_sample_fps=10.0,
        max_goal_frames=8,
        max_goal_duration_sec=30.0,
        min_captured_frames=1,
        max_image_width=640,
        max_image_height=480,
        pose_max_delta_ms=50.0,
        pose_gap_max_ms=200.0,
        max_sample_interval_overshoot_ms=250.0,
        snapshot_limits=SnapshotSafetyLimits(
            max_frame_count=8,
            max_manifest_entries=128,
            max_metadata_bytes=1024 * 1024,
            max_file_bytes=1024 * 1024,
            max_total_bytes=64 * 1024 * 1024,
            max_scene_id_bytes=128,
            max_relative_path_bytes=1024,
        ),
    )


def _fake_result_gate(
    uri, expected_sha256, *, expected_identity, expected_authenticity
):
    """非正式测试门禁：只用于不覆盖工作包 7 的窗口编排测试。"""
    return SimpleNamespace(ok=True, reason="", payload={}, manifest_bytes=b"{}")


def _fake_result_publisher(
    records_root,
    session_id,
    window_index,
    manifest_bytes,
    provenance,
    snapshot_manifest_bytes=b"",
):
    """非正式测试发布器：不落盘，只返回合法摘要。"""
    return Path("/tmp/fake-formal-window"), "a" * 64


def write_minimal_snapshot(snapshot_dir: Path, scene_id: str, frame_count: int, capture_config,
                           missing_pose=()) -> str:
    """Write a valid v4 snapshot as the capture node does and return its manifest SHA-256.

    Stamps increase across windows (window w starts at 100 + 100 w seconds);
    rolling windows rebuild their shared frames from these stamps.
    """
    return write_v4_snapshot(
        snapshot_dir, scene_id, frame_count,
        first_stamp_sec=100 + int(scene_id.rsplit("-w", 1)[1]) * 100,
        camera_name=capture_config.camera_name, image_topic=capture_config.image_topic,
        camera_role=capture_config.camera_role, camera_frame_id=capture_config.camera_frame_id,
        missing_pose=missing_pose,
    )


def write_valid_result_manifest(
    path: Path,
    *,
    scene_id,
    manifest_sha256,
    captured_frames,
    camera_role,
    camera_topic,
    camera_frame_id,
    checkpoint_sha256="b" * 64,
    algorithm_revision="algo-test",
    ros_revision="ros-test",
):
    """写入一份与真实 result.json schema 一致的测试工件并返回其 SHA-256。"""
    payload = {
        "schema": "drone3d.reconstruction.result",
        "schema_version": 1,
        "scene_id": scene_id,
        "snapshot": {"sha256": manifest_sha256, "frame_count": captured_frames},
        "actual_frame_count": captured_frames,
        "camera_role": camera_role,
        "camera_topic": camera_topic,
        "camera_frame_id": camera_frame_id,
        "checkpoint": {
            "path": "/checkpoint/model.pt",
            "sha256": checkpoint_sha256,
        },
        "algorithm_revision": algorithm_revision,
        "ros_revision": ros_revision,
        "coordinate_state": "MODEL_LOCAL_UNALIGNED",
        "worker_exit": {"returncode": 0, "cause": "ok"},
        "stage_timing": {
            "wall_time_sec": 1.5,
            "worker_wall_time_sec": 1.5,
        },
        "artifacts": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, artifact_bytes in (
        ("predictions.npz", b"predictions.npz artifact bytes for continuous tests\n"),
        ("manifest.txt", b"manifest artifact bytes for continuous tests\n"),
        ("run.log", b"run log artifact bytes for continuous tests\n"),
    ):
        artifact_path = path.parent / name
        artifact_path.write_bytes(artifact_bytes)
        payload["artifacts"][name] = {
            "uri": artifact_path.as_uri(),
            "size": len(artifact_bytes),
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        }
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class FakeCaptureClient:
    def __init__(self, role: CameraRole, snapshot_root: Path, fail=False, missing_pose=()):
        self.role = role
        self.snapshot_root = snapshot_root
        self.fail = fail
        # Snapshot frame indices written without a pose (pose_valid false).
        self.missing_pose = tuple(missing_pose)
        self.goal_sent = threading.Event()
        self._ready = threading.Event()
        self.pause_calls = 0
        self.resume_calls = 0
        self.scene_id = ""
        self.sample_fps = 0.0
        self.max_frames = 0
        self.max_duration_sec = 0.0
        self.snapshot_payload = None

    def wait_for_server(self, timeout_sec):
        return True

    def send_goal(
        self,
        scene_id,
        sample_fps,
        max_frames,
        max_duration_sec,
        feedback_cb,
        save_sensor_depth=None,
        set_save_sensor_depth=False,
    ):
        self.scene_id = scene_id
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.max_duration_sec = max_duration_sec
        self.goal_sent.set()
        if feedback_cb is not None:
            feedback_cb(type("Feedback", (), {"frame_count": max_frames})())
        self._ready.set()

    def wait_for_result(self, timeout_sec):
        return self._ready.wait(timeout_sec)

    def cancel_goal(self):
        self._ready.set()

    def outcome(self):
        if self.fail:
            return CaptureActionOutcome(
                action_state=ACTION_FAILED,
                error_code=2,
                message="camera unavailable",
                scene_id=self.scene_id,
                camera_name=self.role.capture_config.camera_name,
                image_topic=self.role.capture_config.image_topic,
                snapshot_uri="",
                manifest_uri="",
                manifest_sha256="",
                frame_count=0,
            )
        snapshot_dir = self.snapshot_root / (
            self.scene_id + "-" + "a" * 16
        )
        manifest_sha256 = write_minimal_snapshot(
            snapshot_dir, self.scene_id, self.max_frames, self.role.capture_config,
            missing_pose=self.missing_pose,
        )
        if self.snapshot_payload is not None:
            (snapshot_dir / "payload.bin").write_bytes(self.snapshot_payload)
        return CaptureActionOutcome(
            action_state=ACTION_SUCCEEDED,
            error_code=0,
            message="ok",
            scene_id=self.scene_id,
            camera_name=self.role.capture_config.camera_name,
            image_topic=self.role.capture_config.image_topic,
            snapshot_uri=snapshot_dir.as_uri(),
            manifest_uri=(snapshot_dir / "files_sha256.csv").as_uri(),
            manifest_sha256=manifest_sha256,
            frame_count=self.max_frames,
        )

    def pause(self):
        self.pause_calls += 1
        return True

    def resume(self):
        self.resume_calls += 1
        return True


class BlockingCaptureClient(FakeCaptureClient):
    def __init__(self, role, snapshot_root):
        super().__init__(role, snapshot_root)
        self._ready = threading.Event()

    def send_goal(
        self,
        scene_id,
        sample_fps,
        max_frames,
        max_duration_sec,
        feedback_cb,
        save_sensor_depth=None,
        set_save_sensor_depth=False,
    ):
        self.scene_id = scene_id
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.max_duration_sec = max_duration_sec
        self.goal_sent.set()

    def release(self):
        self._ready.set()

    def cancel_goal(self):
        self._ready.set()


class CancelSlowCaptureClient(FakeCaptureClient):
    """cancel_goal 后 wait_for_result 持续返回 False，直到显式释放终态。"""

    def __init__(self, role, snapshot_root):
        super().__init__(role, snapshot_root)
        self._ready = threading.Event()
        self.cancel_calls = 0

    def send_goal(
        self,
        scene_id,
        sample_fps,
        max_frames,
        max_duration_sec,
        feedback_cb,
        save_sensor_depth=None,
        set_save_sensor_depth=False,
    ):
        self.scene_id = scene_id
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.max_duration_sec = max_duration_sec
        self.goal_sent.set()

    def wait_for_result(self, timeout_sec):
        return self._ready.wait(timeout_sec)

    def cancel_goal(self):
        self.cancel_calls += 1

    def release_terminal(self):
        self._ready.set()


class WatchdogFreezeCaptureClient(FakeCaptureClient):
    """目标运行中暂停 0.9s：scheduler watchdog 必须冻结，不得取消。"""

    def __init__(self, role, snapshot_root):
        super().__init__(role, snapshot_root)
        self._ready = threading.Event()
        self.cancel_calls = 0

    def send_goal(
        self,
        scene_id,
        sample_fps,
        max_frames,
        max_duration_sec,
        feedback_cb,
        save_sensor_depth=None,
        set_save_sensor_depth=False,
    ):
        self.scene_id = scene_id
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.max_duration_sec = max_duration_sec
        self.goal_sent.set()

    def wait_for_result(self, timeout_sec):
        return self._ready.wait(timeout_sec)

    def cancel_goal(self):
        self.cancel_calls += 1
        self._ready.set()

    def release(self):
        self._ready.set()


class PauseFailCaptureClient(BlockingCaptureClient):
    def pause(self):
        self.pause_calls += 1
        return False


class ResumeFailCaptureClient(BlockingCaptureClient):
    def __init__(self, role, snapshot_root):
        super().__init__(role, snapshot_root)
        self.resume_result = False

    def resume(self):
        self.resume_calls += 1
        return self.resume_result


class SourceStaleCaptureClient(FakeCaptureClient):
    """模拟 source watchdog：发送 SOURCE_STALE 反馈并阻塞 Action 终态。"""

    def __init__(self, role, snapshot_root):
        super().__init__(role, snapshot_root)
        self._ready = threading.Event()
        self.cancel_calls = 0
        self.feedback_cb = None

    def send_goal(
        self,
        scene_id,
        sample_fps,
        max_frames,
        max_duration_sec,
        feedback_cb,
        save_sensor_depth=None,
        set_save_sensor_depth=False,
    ):
        self.scene_id = scene_id
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.max_duration_sec = max_duration_sec
        self.feedback_cb = feedback_cb
        self.goal_sent.set()
        if feedback_cb is not None:
            feedback_cb(
                type("Feedback", (), {"frame_count": 0, "stage": "SOURCE_STALE"})()
            )

    def release(self):
        self._ready.set()

    def cancel_goal(self):
        self.cancel_calls += 1
        self._ready.set()


def _snapshot_frame_count(snapshot_uri) -> int:
    """The worker reconstructs every frame of the submitted snapshot."""
    try:
        frames = Path(unquote(urlsplit(snapshot_uri).path)) / "frames.csv"
        return max(1, len(frames.read_text().splitlines()) - 1)
    except (OSError, TypeError, ValueError):
        return 1


def _patch_g4_process(test):
    """Stand in for the G4 background process: every formal window is already in its map."""
    process = mock.MagicMock()
    process.process.poll.return_value = None
    process.status.return_value = {"state": "ready", "window_count": 1 << 30,
                                   "published_windows": []}
    patcher = mock.patch(
        "drone3d_reconstruction_core.continuous_session.G4BackendProcess",
        return_value=process,
    )
    patcher.start()
    test.addCleanup(patcher.stop)
    return process


class FakeBackend(ReconstructionBackend):
    def __init__(self, block_first=False):
        self.jobs = {}
        self.submitted = []
        self.submit_overrides = []
        self.block_first = block_first
        self.release_first = threading.Event()
        self.frame_count_by_index = None
        self.result_manifest_uri = None
        self.result_manifest_sha256 = None

    def submit(
        self,
        snapshot_uri,
        manifest_sha256,
        window_id,
        inference_resolution=0,
        worker_timeout_sec=0.0,
    ):
        self.submitted.append(window_id)
        self.submit_overrides.append(
            {
                "inference_resolution": inference_resolution,
                "worker_timeout_sec": worker_timeout_sec,
            }
        )
        self.jobs[window_id] = {
            "snapshot_uri": snapshot_uri,
            "manifest_sha256": manifest_sha256,
            "state": JOB_RUNNING if self.block_first else JOB_SUCCEEDED,
        }
        return window_id

    def status(self, job_id):
        job = self.jobs[job_id]
        if self.block_first:
            if job_id == self.submitted[0] and not self.release_first.is_set():
                return JobStatus(job_id, JOB_RUNNING, "blocked")
            if job_id != self.submitted[0] and self.release_first.is_set():
                job["state"] = JOB_SUCCEEDED
        return JobStatus(job_id, job["state"], "")

    def cancel(self, job_id):
        self.jobs[job_id]["state"] = JOB_CANCELLED

    def collect_result(self, job_id):
        job = self.jobs[job_id]
        scene_id = job_id
        frame_count = _snapshot_frame_count(job["snapshot_uri"])
        if self.frame_count_by_index is not None:
            index = min(self.submitted.index(job_id), len(self.frame_count_by_index) - 1)
            frame_count = int(self.frame_count_by_index[index])
        return ReconstructActionOutcome(
            action_state=ACTION_SUCCEEDED if job["state"] == JOB_SUCCEEDED else ACTION_FAILED,
            error_code=0 if job["state"] == JOB_SUCCEEDED else 3,
            message="ok" if job["state"] == JOB_SUCCEEDED else "failed",
            scene_id=scene_id,
            result_manifest_uri=(
                self.result_manifest_uri
                or f"file:///tmp/{scene_id}/result.json"
            ),
            result_manifest_sha256=self.result_manifest_sha256 or ("b" * 64),
            scene_glb_uri="",
            frame_count=frame_count,
            wall_time_sec=0.1,
            camera_role="front",
            camera_topic="/front/image_raw",
            camera_frame_id="front_camera_optical_frame",
            snapshot_manifest_sha256=job["manifest_sha256"],
            result_scene_id=scene_id,
            result_actual_frame_count=frame_count,
        )


class CollectFailOnceBackend(FakeBackend):
    """terminal 后 collect_result 先失败，保留 active job 直到重试收敛。"""

    def __init__(self):
        super().__init__()
        self.release_collect = threading.Event()
        self.collect_calls = 0

    def collect_result(self, job_id):
        self.collect_calls += 1
        if not self.release_collect.is_set():
            raise RuntimeError("injected collect_result failure")
        return super().collect_result(job_id)


class StuckBackend(FakeBackend):
    """cancel 后仍不收敛的 backend：用于 stop_all 线程仍存活竞态测试。"""

    def __init__(self):
        super().__init__(block_first=True)
        self.stuck = True
        self.release = threading.Event()

    def status(self, job_id):
        if self.stuck:
            return JobStatus(job_id, JOB_RUNNING, "stuck")
        return JobStatus(job_id, JOB_CANCELLED, "released")

    def cancel(self, job_id):
        if not self.stuck:
            self.jobs[job_id]["state"] = JOB_CANCELLED


class OomBackend(FakeBackend):
    """第一个 job 返回真实 CUDA OOM 错误文本。"""

    def collect_result(self, job_id):
        job = self.jobs[job_id]
        return ReconstructActionOutcome(
            action_state=ACTION_FAILED,
            error_code=4,
            message="CUDA out of memory",
            scene_id=job_id,
            result_manifest_uri="",
            result_manifest_sha256="",
            scene_glb_uri="",
            frame_count=1,
            wall_time_sec=0.1,
            camera_role="front",
            camera_topic="/front/image_raw",
            camera_frame_id="front_camera_optical_frame",
            snapshot_manifest_sha256=job["manifest_sha256"],
            result_scene_id=job_id,
            result_actual_frame_count=1,
        )


class StatusRaiseOnceBackend(FakeBackend):
    """每个 job 第一次 status 抛异常；清理后可重试成功。"""

    def __init__(self):
        super().__init__()
        self.status_calls = 0
        self.cancel_calls = 0
        self.collect_calls = 0
        self.status_calls_by_job = {}
        self.cancel_calls_by_job = {}
        self.collect_calls_by_job = {}

    def status(self, job_id):
        self.status_calls += 1
        self.status_calls_by_job.setdefault(job_id, 0)
        self.status_calls_by_job[job_id] += 1
        if self.status_calls_by_job[job_id] == 1:
            raise RuntimeError("injected status failure")
        return JobStatus(job_id, self.jobs[job_id]["state"], "")

    def cancel(self, job_id):
        self.cancel_calls += 1
        self.cancel_calls_by_job.setdefault(job_id, 0)
        self.cancel_calls_by_job[job_id] += 1
        self.jobs[job_id]["state"] = JOB_CANCELLED

    def collect_result(self, job_id):
        self.collect_calls += 1
        self.collect_calls_by_job.setdefault(job_id, 0)
        self.collect_calls_by_job[job_id] += 1
        return super().collect_result(job_id)


class AsyncCancelAfterStatusErrorBackend(StatusRaiseOnceBackend):
    """Cancel request stays RUNNING until the simulated Action reaches terminal."""

    def __init__(self):
        super().__init__()
        self.cancel_terminal = threading.Event()

    def status(self, job_id):
        if len(self.submitted) > 1:
            self.status_calls_by_job.setdefault(job_id, 0)
            self.status_calls_by_job[job_id] += 1
            return JobStatus(job_id, self.jobs[job_id]["state"], "")
        self.status_calls += 1
        self.status_calls_by_job.setdefault(job_id, 0)
        self.status_calls_by_job[job_id] += 1
        if self.status_calls_by_job[job_id] == 1:
            raise RuntimeError("injected status failure")
        if not self.cancel_terminal.is_set():
            return JobStatus(job_id, JOB_RUNNING, "cancel pending")
        self.jobs[job_id]["state"] = JOB_CANCELLED
        return JobStatus(job_id, JOB_CANCELLED, "cancelled")

    def cancel(self, job_id):
        self.cancel_calls += 1
        self.cancel_calls_by_job.setdefault(job_id, 0)
        self.cancel_calls_by_job[job_id] += 1

    def collect_result(self, job_id):
        if len(self.submitted) == 1 and not self.cancel_terminal.is_set():
            raise AssertionError("collect_result called before Action terminal")
        return super().collect_result(job_id)


class BlockingSubmitBackend(FakeBackend):
    """Expose STOP_ALL while backend.submit has dispatched but not returned."""

    def __init__(self):
        super().__init__()
        self.submit_entered = threading.Event()
        self.submit_release = threading.Event()
        self.cancel_calls = 0
        self.collect_calls = 0

    def submit(self, *args, **kwargs):
        self.submit_entered.set()
        self.submit_release.wait()
        job_id = super().submit(*args, **kwargs)
        self.jobs[job_id]["state"] = JOB_RUNNING
        return job_id

    def status(self, job_id):
        state = self.jobs[job_id]["state"]
        return JobStatus(
            job_id, state, "cancelled" if state == JOB_CANCELLED else ""
        )

    def cancel(self, job_id):
        self.cancel_calls += 1
        super().cancel(job_id)

    def collect_result(self, job_id):
        self.collect_calls += 1
        return super().collect_result(job_id)


class CleanupFailedOnCancelBackend(FakeBackend):
    """Cancellation reaches FAILED because process-group cleanup was unproven."""

    def __init__(self):
        super().__init__(block_first=True)

    def status(self, job_id):
        state = self.jobs[job_id]["state"]
        message = "cleanup_failed: process group still alive" if state == JOB_FAILED else ""
        return JobStatus(job_id, state, message)

    def cancel(self, job_id):
        self.jobs[job_id]["state"] = JOB_FAILED


class CancelRaisingBackend(FakeBackend):
    """stop_all 期间 cancel 持续抛异常：线程必须保持存活并继续轮询。"""

    def __init__(self):
        super().__init__(block_first=False)
        self.cancel_raises = True
        self.status_terminal = False
        self.cancel_calls = 0

    def status(self, job_id):
        if self.status_terminal:
            return JobStatus(job_id, JOB_CANCELLED, "cancelled")
        return JobStatus(job_id, JOB_RUNNING, "running")

    def cancel(self, job_id):
        self.cancel_calls += 1
        if self.cancel_raises:
            raise RuntimeError("injected cancel failure")
        self.jobs[job_id]["state"] = JOB_CANCELLED


class ContinuousSessionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.service = None
        root = Path(self.temp.name)
        self.journal = root / "journal"
        self.journal.mkdir()
        self.results = root / "results"
        self.results.mkdir()
        front_role = CameraRole(
            role="front",
            capture_action="/front/capture_segment",
            capture_config=make_capture_config(root, "front"),
        )
        down_role = CameraRole(
            role="down",
            capture_action="/down/capture_segment",
            capture_config=make_capture_config(root, "down"),
        )
        self.front_root = Path(front_role.capture_config.snapshot_root)
        roles = tuple(
            replace(role, capture_config=replace(
                role.capture_config, odometry_association="interpolate"))
            for role in (front_role, down_role)
        )
        self.config = ContinuousConfig(
            roles=roles,
            reconstruct_action="/reconstruct/reconstruct_segment",
            journal_root=str(self.journal),
            result_root=str(self.results),
            service_name="/continuous_session/command",
            diagnostics_topic="/diagnostics",
            diagnostics_period_sec=1.0,
            min_free_disk_gb=0.01,
            action_connect_timeout_sec=1.0,
            capture_wait_margin_sec=1.0,
            reconstruct_wait_timeout_sec=5.0,
            stop_timeout_sec=2.0,
            reconstruct_frame_group_size=6,
            initial_sample_fps=5.0,
            initial_window_frames=6,
            initial_overlap_frames=5,
            initial_inference_resolution=384,
            initial_queue_mode=QUEUE_LOSSLESS,
            initial_queue_capacity=8,
            initial_max_retries=0,
            initial_keep_snapshot_on_success=True,
        )
        self.capture_clients = []
        self.backends = []
        # A succeeded window starts the G4 background process.
        _patch_g4_process(self)

    def tearDown(self):
        if getattr(self, "service", None) is not None:
            try:
                self.service.shutdown(2.0)
            except BaseException:
                pass

    def make_session(
        self,
        backend=None,
        capture_factory=None,
        disk_free=None,
        config=None,
        *,
        formal_gate=_fake_result_gate,
        formal_publisher=_fake_result_publisher,
    ):
        service = ContinuousSession(
            config or self.config,
            capture_client_factory=capture_factory or self._capture_factory,
            backend_factory=lambda: backend or self._backend_factory(),
            disk_free_gb=disk_free or (lambda: 100.0),
            result_gate=formal_gate,
            result_publisher=formal_publisher,
        )
        self.service = service
        return service

    def _capture_factory(self, role):
        client = FakeCaptureClient(role, self.front_root)
        self.capture_clients.append(client)
        return client

    def _backend_factory(self):
        backend = FakeBackend()
        self.backends.append(backend)
        return backend

    def wait_until(self, predicate, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        raise AssertionError("condition not reached")

    def assert_only_succeeded_snapshots_remain(self, service):
        """Succeeded windows keep their snapshot (G4 reads it); nothing else remains."""
        kept = {
            Path(unquote(urlsplit(record.snapshot_uri).path))
            for record in service._windows if record.state == WINDOW_SUCCEEDED
        }
        self.assertTrue(kept)
        self.assertEqual(set(self.front_root.iterdir()), kept)

    def journal_events(self):
        last_error = None
        for _attempt in range(20):
            try:
                lines = []
                for path in sorted(self.journal.glob("*.jsonl")):
                    for line in path.read_text().splitlines():
                        if line.strip():
                            lines.append(json.loads(line))
                return lines
            except (json.JSONDecodeError, OSError) as error:
                last_error = error
                time.sleep(0.05)
        raise AssertionError(f"journal never became readable: {last_error}")

    def test_three_windows_are_captured_while_first_is_reconstructing(self):
        backend = FakeBackend(block_first=True)
        service = self.make_session(backend=backend)
        code, _ = service.select_source("front")
        self.assertEqual(code, 0)
        code, _ = service.start_session("")
        self.assertEqual(code, 0)
        self.wait_until(
            lambda: service.get_status().window_index >= 2
            and service.get_status().pending_windows >= 2
        )
        # 消费循环与采集循环是两个线程：pending 计数可见后，还要等
        # 消费线程真正提交第一个 job，避免把线程调度瞬时状态当成失败。
        self.wait_until(lambda: len(backend.submitted) == 1)
        status = service.get_status()
        self.assertEqual(status.running_window, 0)
        self.assertGreaterEqual(status.pending_windows, 2)
        self.assertEqual(len(backend.submitted), 1)  # 单 worker
        backend.release_first.set()
        backend.jobs[backend.submitted[0]]["state"] = JOB_SUCCEEDED
        final = self.wait_until(
            lambda: service.get_status()
            if service.get_status().succeeded_windows >= 3
            else None,
            timeout=8.0,
        )
        self.assertGreaterEqual(final.succeeded_windows, 3)
        self.assertEqual(final.failed_windows, 0)
        service.stop_all(discard_queue=True)
        self.assert_only_succeeded_snapshots_remain(service)

    def test_lossless_capacity_backpressures_capture_and_resumes_without_drops(self):
        backend = FakeBackend(block_first=True)
        third_goal = threading.Event()
        clients = []

        def factory(role):
            client = FakeCaptureClient(role, self.front_root)
            clients.append(client)
            if len(clients) >= 3:
                third_goal.set()
            return client

        from dataclasses import replace
        service = self.make_session(
            backend=backend, capture_factory=factory,
            config=replace(self.config, initial_queue_capacity=1),
        )
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: len(backend.submitted) == 1 and service.get_status().pending_windows == 1)
        self.assertTrue(service.diagnostics_snapshot()["capture_queue_blocked"])
        self.assertFalse(third_goal.wait(0.7), "a full FIFO admitted another capture")
        self.assertEqual(len(clients), 2)
        backend.jobs[backend.submitted[0]]["state"] = JOB_SUCCEEDED
        backend.release_first.set()
        self.assertTrue(third_goal.wait(3.0), "capture did not resume after consumption")
        self.assertEqual(service.get_status().dropped_windows, 0)
        self.wait_until(lambda: service.diagnostics_snapshot()["service_wall_time_sec"] is not None)
        timing = service.diagnostics_snapshot()
        self.assertGreaterEqual(timing["service_wall_time_sec"], 0.0)
        self.assertEqual(timing["service_budget_sec"], 1.0 / self.config.initial_sample_fps)
        service.stop_all(discard_queue=True)
        self.assert_only_succeeded_snapshots_remain(service)

    def test_overlap_windows_capture_only_stride_and_keep_exact_frame_ids(self):
        role = self.config.roles[0]
        clients = []
        blocked = BlockingCaptureClient(role, self.front_root)

        def capture_factory(_role):
            client = blocked if len(clients) == 3 else FakeCaptureClient(role, self.front_root)
            clients.append(client)
            return client

        backend = FakeBackend()
        backend.frame_count_by_index = [6, 6, 6]
        service = self.make_session(
            backend=backend,
            capture_factory=capture_factory,
        )
        self.assertEqual(service.select_source("front")[0], 0)
        self.assertEqual(service.start_session("")[0], 0)
        self.assertTrue(blocked.goal_sent.wait(3.0))
        self.assertEqual(service.get_status().capture_target_frames, 1)
        service.stop_all()

        sent = [client for client in clients if client.goal_sent.is_set()]
        self.assertEqual([client.max_frames for client in sent[:3]], [6, 1, 1])
        windows = service._windows[:3]
        self.assertEqual(windows[0].window_frame_ids, [0, 1, 2, 3, 4, 5])
        self.assertEqual(windows[1].window_frame_ids, [1, 2, 3, 4, 5, 6])
        self.assertEqual(windows[2].window_frame_ids, [2, 3, 4, 5, 6, 7])
        self.assertTrue(all(record.captured_frames == 6 for record in windows))

    def test_g4_assembles_total_16_then_11_new_5_shared(self):
        base = self.large_window_config()
        config = replace(base, alignment_backend="global_g4", initial_window_frames=16,
                         initial_overlap_frames=5, reconstruct_frame_group_size=16,
                         initial_keep_snapshot_on_success=True)
        role = config.roles[0]
        clients = []
        blocked = BlockingCaptureClient(role, self.front_root)
        def factory(_role):
            client = blocked if len(clients) == 3 else FakeCaptureClient(role, self.front_root)
            clients.append(client)
            return client
        backend = FakeBackend(block_first=True)
        service = self.make_session(backend=backend, capture_factory=factory, config=config)
        service.select_source("front")
        service.start_session("")
        self.assertTrue(blocked.goal_sent.wait(5.0))
        self.assertEqual([c.max_frames for c in clients], [16, 11, 11, 11])
        self.assertEqual([w.captured_frames for w in service._windows[:3]], [16]*3)
        for record, first in zip(service._windows[:3], (0, 11, 22)):
            self.assertEqual(record.window_frame_ids, list(range(first, first+16)))
        service.stop_all()

    def test_g4_diagnostics_count_actual_first_and_overlapping_input(self):
        config = replace(self.large_window_config(), alignment_backend="global_g4",
                         initial_window_frames=16, initial_overlap_frames=5,
                         reconstruct_frame_group_size=16,
                         initial_keep_snapshot_on_success=True)
        service = self.make_session(config=config)
        for index, shared in ((0, 0), (1, 5)):
            record = WindowRecord(index, "diagnostic-%d" % index, config.roles[0],
                                  service._runtime, overlap_frame_count=shared)
            record.state = WINDOW_CAPTURING
            service._windows = [record]
            diagnostic = service.diagnostics_snapshot()
            self.assertEqual(diagnostic["new_input_frames"], 16-shared)
            self.assertEqual(diagnostic["shared_input_frames"], shared)
            self.assertEqual(diagnostic["model_input_frames"], 16)

    def test_three_consecutive_capture_failures_pause_until_resume(self):
        config = replace(self.large_window_config(), alignment_backend="global_g4",
                         initial_window_frames=16, initial_overlap_frames=5,
                         reconstruct_frame_group_size=16,
                         initial_keep_snapshot_on_success=True)
        clients = []
        recovered = BlockingCaptureClient(config.roles[0], self.front_root)
        def factory(role):
            client = (FakeCaptureClient(role, self.front_root, fail=True)
                      if len(clients) < 3 else recovered)
            clients.append(client)
            return client
        service = self.make_session(config=config, capture_factory=factory)
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: service.get_status().state == SESSION_PAUSED)
        self.assertEqual(service.get_status().failed_windows, 3)
        self.assertEqual(len(service._windows), 3)
        self.assertEqual(len(clients), 3)
        self.assertEqual(service.resume()[0], 0)
        self.assertTrue(recovered.goal_sent.wait(3.0))
        service.stop_all()

    def _two_windows_then_block(self):
        """第 0 窗立刻采完，第 1 窗等 release()，之后的窗口一直卡在采集里。"""
        role = self.config.roles[0]
        second = BlockingCaptureClient(role, self.front_root)
        clients = []

        def capture_factory(_role):
            clients.append(
                FakeCaptureClient(role, self.front_root) if not clients
                else second if len(clients) == 1
                else BlockingCaptureClient(role, self.front_root)
            )
            return clients[-1]

        return second, capture_factory

    def test_g4_backend_start_failure_is_loud_and_capture_continues(self):
        process = mock.MagicMock()
        process.process.poll.return_value = None
        process.status.return_value = {"state": "ready", "window_count": 1 << 30}
        second, capture_factory = self._two_windows_then_block()
        with mock.patch("drone3d_reconstruction_core.continuous_session.G4BackendProcess",
                        side_effect=[OSError("spawn refused"), process]) as spawn:
            service = self.make_session(capture_factory=capture_factory)
            service.select_source("front")
            self.assertEqual(service.start_session("")[0], 0)
            self.wait_until(lambda: service.get_status().succeeded_windows == 1)
            # 窗口照常成功，会话照常运行；但地图后端没起来必须看得见，不能显示成“等待启动”。
            status = service.get_status()
            self.assertEqual((status.state, status.failed_windows), (SESSION_RUNNING, 0))
            diagnostics = service.diagnostics_snapshot()
            expected = "G4 background start failed: OSError: spawn refused"
            self.assertEqual(diagnostics["g4"], {"state": "error", "error": expected})
            self.assertEqual(diagnostics["last_error"], expected)
            failed = [event for event in self.journal_events()
                      if event.get("event") == "g4_backend_start_failed"]
            self.assertEqual([(event["window_index"], event["error"]) for event in failed],
                             [(0, expected)])
            # 下一个窗口成功时再试一次；起来之后错误消失。
            second.release()
            self.wait_until(lambda: service.get_status().succeeded_windows == 2)
            self.assertEqual(spawn.call_count, 2)
            self.assertEqual(service.diagnostics_snapshot()["g4"]["state"], "ready")

    def test_g4_backend_that_died_is_journaled_and_restarted(self):
        first, replacement = mock.MagicMock(), mock.MagicMock()
        for process in (first, replacement):
            process.process.poll.return_value = None
            process.status.return_value = {"state": "ready", "window_count": 1 << 30}
        second, capture_factory = self._two_windows_then_block()
        with mock.patch("drone3d_reconstruction_core.continuous_session.G4BackendProcess",
                        side_effect=[first, replacement]) as spawn:
            service = self.make_session(capture_factory=capture_factory)
            service.select_source("front")
            self.assertEqual(service.start_session("")[0], 0)
            self.wait_until(lambda: service.get_status().succeeded_windows == 1)
            first.process.poll.return_value = 3
            first.status.return_value = {"state": "error", "exit_code": 3, "error": "killed"}
            self.assertEqual(service.diagnostics_snapshot()["g4"]["state"], "error")
            second.release()
            self.wait_until(lambda: service.get_status().succeeded_windows == 2)
            exited = [event for event in self.journal_events()
                      if event.get("event") == "g4_backend_exited"]
            self.assertEqual(
                [(event["window_index"], event["exit_code"], event["error"]) for event in exited],
                [(1, 3, "killed")],
            )
            first.close.assert_called_once_with()
            self.assertEqual(spawn.call_count, 2)
            self.assertEqual(service.diagnostics_snapshot()["g4"]["state"], "ready")

    def test_g4_background_does_not_block_second_worker_and_stop_closes_it(self):
        from unittest.mock import patch, MagicMock
        base = self.large_window_config()
        config = replace(base, alignment_backend="global_g4", initial_window_frames=16,
                         initial_overlap_frames=5, reconstruct_frame_group_size=16,
                         initial_keep_snapshot_on_success=True)
        role=config.roles[0]; clients=[]; blocked=BlockingCaptureClient(role,self.front_root)
        def factory(_role):
            client=blocked if len(clients)==2 else FakeCaptureClient(role,self.front_root)
            clients.append(client); return client
        process=MagicMock();process.process.poll.return_value=None
        process.status.return_value={"state":"optimizing", "published_windows":[]}
        backend=FakeBackend();backend.frame_count_by_index=[16,16]
        with patch("drone3d_reconstruction_core.continuous_session.G4BackendProcess",return_value=process) as start:
            service=self.make_session(backend=backend,capture_factory=factory,config=config)
            service.select_source("front");service.start_session("")
            self.wait_until(lambda: service.get_status().succeeded_windows==2,timeout=8.)
            self.assertEqual(len(backend.submitted),2)
            self.assertEqual(start.call_count,1)
            self.assertEqual(service.diagnostics_snapshot()["g4"]["state"],"optimizing")
            service.stop_all()
            process.close.assert_called_once()

    def test_g4_frame_pose_session_output_is_pinned(self):
        """主线 G4 frame_pose 会话交给正式发布的内容、诊断键与状态字段（删除旧对齐后端前钉住）。"""
        published = []

        def publisher(*args):
            published.append(copy.deepcopy(args[4]))
            return _fake_result_publisher(*args)

        roles = tuple(replace(role, capture_config=replace(
            role.capture_config, odometry_association="interpolate"))
            for role in self.config.roles)
        config = replace(self.config, roles=roles, alignment_backend="global_g4",
                         g4_placement="frame_pose", reconstruct_frame_group_size=8,
                         initial_window_frames=8, initial_overlap_frames=5,
                         initial_inference_resolution=384,
                         initial_keep_snapshot_on_success=True)
        clients = []

        def factory(role):
            client = (FakeCaptureClient(role, self.front_root) if len(clients) < 2
                      else BlockingCaptureClient(role, self.front_root))
            clients.append(client)
            return client

        backend = FakeBackend()
        backend.frame_count_by_index = [8, 8]
        process = mock.MagicMock()
        process.process.poll.return_value = None
        process.status.return_value = {"state": "optimizing", "published_windows": []}
        with mock.patch("drone3d_reconstruction_core.continuous_session.G4BackendProcess",
                        return_value=process):
            service = self.make_session(backend=backend, capture_factory=factory,
                                        config=config, formal_publisher=publisher)
            service.select_source("front")
            self.assertEqual(service.start_session("")[0], 0)
            self.wait_until(lambda: service.get_status().succeeded_windows == 2, timeout=10.0)
            diagnostics = service.diagnostics_snapshot()
            status = service.get_status()
            service.stop_all()

        self.assertEqual(len(published), 2)
        session_id = status.session_id
        self.assertRegex(session_id, r"^s[0-9a-f]{16}$")
        top_level = [
            "schema", "schema_version", "session_id", "window_index", "scene_id",
            "result_manifest_uri", "result_manifest_sha256", "snapshot_manifest_uri",
            "snapshot_manifest_sha256", "captured_frames", "new_frame_count",
            "overlap_frame_count", "window_frame_ids", "captured_frame_ids",
            "capture_generation", "model_frame_ids", "camera_role", "camera_topic",
            "camera_frame_id", "camera_name", "config_digest", "effective_config",
            "config_timeline", "checkpoint", "algorithm_revision", "ros_revision",
            "cuda_environment", "worker", "gate", "reconstruction_mode", "pool_revision",
            "pool_epoch", "pool_size", "pool_selected_indices", "confidence_summary",
            "alignment", "depth_metrics", "depth_metrics_error",
        ]
        alignment = [
            ("coordinate_state", "MODEL_LOCAL_UNALIGNED"), ("map_id", ""), ("scale", None),
            ("rotation", None), ("translation", None), ("residual_rms", None),
            ("correspondence_count", 0), ("reference_frame_id", ""), ("unreliable", True),
            ("reason", "waiting_for_G4_map_revision"), ("method", "global_g4"),
            ("pending", True), ("quality_passed", False), ("geometry_accepted", False),
        ]
        windows = (
            (0, list(range(0, 8)), 0, 8),
            (1, list(range(3, 11)), 5, 3),
        )
        for provenance, (index, frame_ids, shared, new) in zip(published, windows):
            with self.subTest(window=index):
                self.assertEqual(list(provenance), top_level)
                self.assertEqual(list(provenance["alignment"].items()), alignment)
                self.assertEqual(provenance["session_id"], session_id)
                self.assertEqual(provenance["window_index"], index)
                self.assertEqual(provenance["scene_id"], f"cs-{session_id}-w{index}")
                self.assertEqual(provenance["window_frame_ids"], frame_ids)
                self.assertEqual(provenance["captured_frame_ids"], frame_ids)
                self.assertEqual(provenance["model_frame_ids"], frame_ids)
                self.assertEqual(provenance["captured_frames"], 8)
                self.assertEqual(provenance["overlap_frame_count"], shared)
                self.assertEqual(provenance["new_frame_count"], new)
                self.assertEqual(provenance["reconstruction_mode"], "per_window")
                for key in ("pool_revision", "pool_epoch", "pool_size",
                            "pool_selected_indices", "confidence_summary",
                            "depth_metrics", "depth_metrics_error"):
                    self.assertIsNone(provenance[key])

        # 已删功能专属的键，删除时允许去掉：map_epoch（里程计后端的地图纪元，G4 会话里恒为 0）、
        # alignment_fault（只由分窗清单故障写入）。
        removed_features = {"map_epoch", "alignment_fault"}
        self.assertEqual(set(diagnostics) - removed_features, {
            "alignment_backend", "attempt_failures",
            "capture_action_terminal", "capture_cancel_pending", "capture_queue_blocked",
            "capture_target_frames", "captured_frames", "disk_free_gb", "disk_redline",
            "dropped_windows", "effective", "failed_windows", "fifo", "fifo_bytes",
            "fifo_oldest_wait_sec", "formal_records_root", "formal_windows", "g4",
            "last_alignment_state", "last_error", "last_result_wall_time_sec",
            "manual_retries", "memory_paused", "min_free_disk_gb", "missing_pose_frames",
            "model_input_frames",
            "new_input_frames", "oom_latched", "pending_windows", "reconstruction_mode",
            "retry_window", "role", "running_window", "service_budget_sec",
            "service_wall_time_sec", "session_id", "shared_input_frames", "state",
            "succeeded_windows", "window_index", "worker_disk_blocked",
            "worker_wall_time_sec",
        })
        self.assertEqual(diagnostics["alignment_backend"], "global_g4")
        self.assertEqual(diagnostics["last_alignment_state"], "MODEL_LOCAL_UNALIGNED")
        self.assertEqual(
            [field.name for field in dataclasses.fields(status) if field.name not in removed_features],
            ["state", "session_id", "role", "window_index", "captured_frames",
             "capture_target_frames", "pending_windows", "running_window",
             "succeeded_windows", "failed_windows", "dropped_windows", "effective",
             "last_error", "min_free_disk_gb", "fifo_bytes", "fifo_oldest_wait_sec",
             "disk_free_gb", "last_alignment_state", "formal_windows",
             "reconstruction_mode", "alignment_backend"],
        )

    def large_window_config(self):
        roles = tuple(replace(role, capture_config=replace(
            role.capture_config, max_goal_frames=100, pose_gap_max_ms=500.0,
            odometry_association="interpolate", pose_interpolation_wait_ms=500.0,
            snapshot_limits=replace(role.capture_config.snapshot_limits,
                                    max_frame_count=128, max_manifest_entries=1024),
        )) for role in self.config.roles)
        return replace(self.config, roles=roles,
                       initial_window_frames=100, initial_overlap_frames=20,
                       reconstruct_frame_group_size=120, initial_inference_resolution=384)

    def test_g4_requires_retained_input_and_accepts_total_16(self):
        config = replace(self.large_window_config(), alignment_backend="global_g4",
                         initial_window_frames=16, initial_overlap_frames=5,
                         reconstruct_frame_group_size=16,
                         initial_keep_snapshot_on_success=True)
        self.assertEqual(config.initial_window_frames, 16)
        with self.assertRaisesRegex(ContinuousConfigError, "retained snapshots"):
            replace(config, initial_keep_snapshot_on_success=False)

    # ------------------------------------------------------------------
    # 缺位姿超过 20% 的窗口判失败（MAX_MISSING_POSE_FRACTION）
    # ------------------------------------------------------------------
    def run_missing_pose_windows(self, missing_per_capture):
        """10-frame windows sharing 5 frames; capture k leaves out the poses listed for it.

        Returns the session after every listed capture finished (the next capture blocks).
        """
        config = replace(self.large_window_config(), alignment_backend="global_g4",
                         initial_window_frames=10, initial_overlap_frames=5,
                         reconstruct_frame_group_size=10,
                         initial_keep_snapshot_on_success=True)
        role = config.roles[0]
        clients = []
        blocked = BlockingCaptureClient(role, self.front_root)
        def factory(_role):
            if len(clients) < len(missing_per_capture):
                client = FakeCaptureClient(role, self.front_root,
                                           missing_pose=missing_per_capture[len(clients)])
            else:
                client = blocked
            clients.append(client)
            return client
        backend = FakeBackend()
        service = self.make_session(backend=backend, capture_factory=factory, config=config)
        service.select_source("front")
        service.start_session("")
        self.assertTrue(blocked.goal_sent.wait(5.0))
        self.wait_until(lambda: all(w.done.is_set() for w in service._windows[:len(missing_per_capture)]))
        return service, backend

    def test_window_missing_exactly_20_percent_of_poses_is_reconstructed(self):
        service, backend = self.run_missing_pose_windows([(0, 1)])
        record = service._windows[0]
        self.assertEqual(record.state, WINDOW_SUCCEEDED)
        self.assertEqual(record.missing_pose_frames, 2)
        self.assertEqual(backend.submitted, [record.scene_id])
        self.assertEqual(service.get_status().failed_windows, 0)
        events = self.journal_events()
        pending = [e for e in events if e["event"] == "window_pending" and e["window_index"] == 0]
        succeeded = [e for e in events if e["event"] == "window_succeeded" and e["window_index"] == 0]
        self.assertEqual([e["missing_pose_frames"] for e in pending], [2])
        self.assertEqual([e["missing_pose_frames"] for e in succeeded], [2])
        self.assertEqual(service.diagnostics_snapshot()["missing_pose_frames"], 2)
        service.stop_all()

    def test_window_missing_over_20_percent_fails_without_reconstruction(self):
        # Window 0 misses 3 of 10. Window 1 reuses frames 5–9 of window 0 (all posed)
        # and is reconstructed normally; the session keeps running.
        service, backend = self.run_missing_pose_windows([(0, 1, 2), ()])
        failed, following = service._windows[:2]
        self.assertEqual(failed.state, WINDOW_FAILED)
        self.assertIn("missing poses: 3 of 10 window frames have no pose (limit 20%)", failed.error)
        self.assertEqual(failed.reconstruct_attempts, 0)
        self.assertEqual(following.state, WINDOW_SUCCEEDED)
        self.assertEqual(following.overlap_frame_count, 5)
        self.assertEqual(backend.submitted, [following.scene_id])
        status = service.get_status()
        self.assertEqual(status.state, SESSION_RUNNING)
        self.assertEqual(status.failed_windows, 1)
        self.assertEqual(service.diagnostics_snapshot()["missing_pose_frames"], 3)
        events = [e for e in self.journal_events() if e.get("window_index") == 0]
        failed_events = [e for e in events if e["event"] == "window_failed"]
        self.assertEqual(len(failed_events), 1)
        self.assertEqual((failed_events[0]["missing_pose_frames"], failed_events[0]["window_frames"]), (3, 10))
        self.assertFalse([e for e in events if e["event"] in ("window_pending", "window_retry")])
        service.stop_all()

    def test_shared_frames_without_pose_count_toward_the_next_window(self):
        # Window 0 misses its last two frames (2 of 10: reconstructed); window 1 shares
        # them and misses one new frame: 3 of 10, so window 1 fails.
        service, backend = self.run_missing_pose_windows([(8, 9), (0,)])
        first, second = service._windows[:2]
        self.assertEqual(first.state, WINDOW_SUCCEEDED)
        self.assertEqual(second.state, WINDOW_FAILED)
        self.assertEqual(second.missing_pose_frames, 3)
        self.assertEqual(backend.submitted, [first.scene_id])
        # 会话累计每帧只算一次：窗口 0 的两帧加窗口 1 新采的一帧，共享的两帧不重复计数。
        self.assertEqual(service.diagnostics_snapshot()["missing_pose_frames"], 3)
        service.stop_all()

    def test_scheduler_does_not_timeout_capture_during_source_stale(self):
        role = self.config.roles[0]
        client = SourceStaleCaptureClient(role, self.front_root)
        backend = FakeBackend()
        service = self.make_session(
            backend=backend, capture_factory=lambda _role: client
        )
        code, _ = service.select_source("front")
        self.assertEqual(code, 0)
        code, _ = service.start_session("")
        self.assertEqual(code, 0)
        self.assertTrue(client.goal_sent.wait(3.0))
        # 默认 capture watchdog deadline 约 1.2 s；SOURCE_STALE 期间调度器
        # 不得取消底层 Capture Action。
        time.sleep(2.0)
        self.assertEqual(client.cancel_calls, 0)
        self.assertFalse(client._ready.is_set())
        # 超过原 deadline 后恢复：watchdog 必须重新计时，而不是恢复瞬间误杀。
        client.feedback_cb(
            type("Feedback", (), {"frame_count": 0, "stage": "SOURCE_RECOVERED"})()
        )
        time.sleep(0.5)
        self.assertEqual(client.cancel_calls, 0)
        client.release()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        self.assertEqual(service.get_status().failed_windows, 0)
        service.stop_all()

    def test_reconstruct_retry_uses_own_budget_and_queues_at_tail(self):
        config = replace(self.config, initial_max_retries=1)
        backend = FakeBackend()
        original_submit = backend.submit
        first_w0 = []

        def submit(
            snapshot_uri,
            manifest_sha256,
            window_id,
            inference_resolution=0,
            worker_timeout_sec=0.0,
        ):
            job_id = original_submit(snapshot_uri, manifest_sha256, window_id)
            if window_id.endswith("-w0") and not first_w0:
                first_w0.append(job_id)
                backend.jobs[job_id]["state"] = JOB_RUNNING
            else:
                backend.jobs[job_id]["state"] = JOB_SUCCEEDED
            return job_id

        backend.submit = submit
        service = ContinuousSession(
            config,
            capture_client_factory=self._capture_factory,
            backend_factory=lambda: backend,
            disk_free_gb=lambda: 100.0,
            result_gate=_fake_result_gate,
            result_publisher=_fake_result_publisher,
        )
        self.service = service
        service.select_source("front")
        service.start_session("")
        # w1 已封存排队后，再让 w0 失败：w0 重试只能排在队尾。
        self.wait_until(lambda: service.get_status().pending_windows >= 1)
        self.wait_until(lambda: len(backend.submitted) == 1)
        self.assertTrue(first_w0)
        backend.jobs[first_w0[0]]["state"] = JOB_FAILED
        self.wait_until(lambda: service.get_status().succeeded_windows >= 2)
        service.pause()
        self.wait_until(lambda: service.get_status().pending_windows == 0)
        self.assertEqual(service.get_status().failed_windows, 0)
        # w0 第一次失败后重试排在队尾：至少排在 w1 之后。
        self.assertEqual(backend.submitted[0], first_w0[0])
        self.assertEqual(backend.submitted[1], next(wid for wid in backend.submitted if wid.endswith("-w1")))
        w0_submissions = [
            index for index, wid in enumerate(backend.submitted)
            if wid == first_w0[0]
        ]
        self.assertEqual(len(w0_submissions), 2)
        self.assertGreaterEqual(w0_submissions[1], 2)
        retry_events = [
            event for event in self.journal_events()
            if event.get("event") == "window_retry"
        ]
        self.assertEqual(len(retry_events), 1)
        self.assertEqual(retry_events[0]["attempts"], 1)
        service.stop_all()

    def test_real_oom_enters_fatal_state_and_stops_capture_and_fifo(self):
        backend = OomBackend()
        service = self.make_session(backend=backend)
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: service.get_status().state == SESSION_FATAL_OOM)
        self.assertEqual(service.get_status().failed_windows, 1)
        self.assertIn("FATAL_OOM", service.get_status().last_error)
        snapshot = service.diagnostics_snapshot()
        self.assertEqual(snapshot["state"], SESSION_FATAL_OOM)
        clients_before = len(self.capture_clients)
        submissions_before = len(backend.submitted)
        time.sleep(0.4)
        self.assertEqual(len(self.capture_clients), clients_before)
        self.assertEqual(len(backend.submitted), submissions_before)
        events = [
            event for event in self.journal_events()
            if event.get("event") == "session_fatal_oom"
        ]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["window_index"], 0)
        service.stop_all()
        self.assertEqual(service.get_status().state, SESSION_STOPPED)

    def test_ordinary_failure_does_not_enter_fatal_oom(self):
        backend = FakeBackend()
        original_submit = backend.submit

        def submit(
            snapshot_uri,
            manifest_sha256,
            window_id,
            inference_resolution=0,
            worker_timeout_sec=0.0,
        ):
            job_id = original_submit(snapshot_uri, manifest_sha256, window_id)
            backend.jobs[job_id]["state"] = JOB_FAILED
            return job_id

        backend.submit = submit
        service = self.make_session(backend=backend)
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: service.get_status().failed_windows >= 1)
        self.assertNotEqual(service.get_status().state, SESSION_FATAL_OOM)
        self.assertEqual(
            [
                event for event in self.journal_events()
                if event.get("event") == "session_fatal_oom"
            ],
            [],
        )
        service.stop_all()

    def test_memory_protection_pauses_without_retry_and_preserves_snapshot(self):
        from dataclasses import replace
        for message in ('worker stopped: resource_guard',
                        'resource gate denied launch: LOW_MEM_AVAILABLE',
                        'resource gate denied launch: HIGH_SWAP_GROWTH'):
            with self.subTest(message=message):
                backend = OomBackend()
                collect = backend.collect_result
                backend.collect_result = lambda job_id: replace(collect(job_id), message=message)
                service = self.make_session(backend=backend)
                service.select_source('front')
                service.start_session('')
                try:
                    self.wait_until(lambda: service.get_status().state == 'MEMORY_PAUSED')
                    before = (len(self.capture_clients), len(backend.submitted))
                    time.sleep(.4)
                    self.assertEqual((len(self.capture_clients), len(backend.submitted)), before)
                    self.assertEqual(service.get_status().failed_windows, 1)
                    record = service._windows[0]
                    snapshot = Path(unquote(urlsplit(record.snapshot_uri).path))
                    self.assertTrue(snapshot.is_dir())
                    self.assertEqual(service.stop_all()[0], 0)
                    self.assertTrue(snapshot.is_dir())
                finally:
                    service.stop_all()

    def test_failed_window_does_not_block_next_window(self):
        backend = FakeBackend()
        # 第一个 job 失败，后续成功。
        original_submit = backend.submit

        def submit(
            snapshot_uri,
            manifest_sha256,
            window_id,
            inference_resolution=0,
            worker_timeout_sec=0.0,
        ):
            job_id = original_submit(snapshot_uri, manifest_sha256, window_id)
            if len(backend.submitted) == 1:
                backend.jobs[job_id]["state"] = JOB_FAILED
            else:
                backend.jobs[job_id]["state"] = JOB_SUCCEEDED
            return job_id

        backend.submit = submit
        service = self.make_session(backend=backend)
        service.select_source("front")
        service.start_session("")
        final = self.wait_until(
            lambda: service.get_status()
            if service.get_status().failed_windows >= 1
            and service.get_status().succeeded_windows >= 1
            else None
        )
        self.assertEqual(final.failed_windows, 1)
        self.assertGreaterEqual(final.succeeded_windows, 1)
        failed_scene = self.capture_clients[0].scene_id
        retained = [
            path for path in self.front_root.iterdir()
            if path.name.startswith(failed_scene + "-")
        ]
        self.assertEqual(retained, [])
        service.stop_all(discard_queue=True)
        self.assert_only_succeeded_snapshots_remain(service)

    def test_pause_resume_controls_active_capture(self):
        blocking = None

        def blocking_factory(role):
            nonlocal blocking
            blocking = BlockingCaptureClient(role, self.front_root)
            return blocking

        service = self.make_session(capture_factory=blocking_factory)
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: blocking is not None and blocking.goal_sent.is_set())
        code, _ = service.pause()
        self.assertEqual(code, 0)
        self.assertEqual(service.get_status().state, SESSION_PAUSED)
        self.assertGreater(blocking.pause_calls, 0)
        code, _ = service.resume()
        self.assertEqual(code, 0)
        self.assertEqual(service.get_status().state, SESSION_RUNNING)
        self.assertGreater(blocking.resume_calls, 0)
        blocking.release()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        service.stop_all()

    def test_disk_free_read_failure_is_fail_closed(self):
        service = self.make_session(
            disk_free=lambda: (_ for _ in ()).throw(RuntimeError("statvfs fail"))
        )
        service.select_source("front")
        service.start_session("")
        self.wait_until(
            lambda: service.get_status().state == SESSION_DISK_PAUSED
        )
        snapshot = service.diagnostics_snapshot()
        self.assertTrue(snapshot["disk_redline"])
        self.assertEqual(self.capture_clients, [])
        service.stop_all()

    def test_disk_redline_recovers_automatically(self):
        free_gb = [0.0]
        service = self.make_session(disk_free=lambda: free_gb[0])
        service.select_source("front")
        service.start_session("")
        self.wait_until(
            lambda: service.get_status().state == SESSION_DISK_PAUSED
        )
        self.assertEqual(self.capture_clients, [])

        free_gb[0] = 100.0
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        self.assertEqual(service.get_status().state, SESSION_RUNNING)
        service.stop_all()

    def test_fifo_observability_reports_bytes_and_oldest_wait(self):
        backend = FakeBackend(block_first=True)
        payload = b"x" * 100
        blocked = BlockingCaptureClient(self.config.roles[0], self.front_root)
        calls = [0]

        def capture_factory(role):
            calls[0] += 1
            if calls[0] == 4:
                self.capture_clients.append(blocked)
                return blocked
            client = FakeCaptureClient(role, self.front_root)
            client.snapshot_payload = payload
            self.capture_clients.append(client)
            return client

        service = self.make_session(
            backend=backend, capture_factory=capture_factory
        )
        service.select_source("front")
        service.start_session("")
        self.wait_until(
            lambda: service.get_status().pending_windows == 2
            and len(backend.submitted) == 1
        )
        self.assertTrue(blocked.goal_sent.wait(3.0))
        status = service.get_status()
        with service._lock:
            fifo_records = list(service._fifo)
        self.assertEqual(status.pending_windows, len(fifo_records))
        expected_fifo_bytes = sum(
            service._snapshot_bytes_from_uri(record.snapshot_uri)
            for record in fifo_records
        )
        self.assertEqual(status.fifo_bytes, expected_fifo_bytes)
        self.assertGreaterEqual(status.fifo_bytes, status.pending_windows * len(payload))
        self.assertIsNotNone(status.fifo_oldest_wait_sec)
        self.assertGreaterEqual(status.fifo_oldest_wait_sec, 0.0)
        snapshot = service.diagnostics_snapshot()
        self.assertEqual(snapshot["fifo_bytes"], status.fifo_bytes)
        self.assertAlmostEqual(
            snapshot["fifo_oldest_wait_sec"], status.fifo_oldest_wait_sec, delta=0.5
        )
        backend.release_first.set()
        backend.jobs[backend.submitted[0]]["state"] = JOB_SUCCEEDED
        # 先暂停产生新窗口，再释放阻塞中的 w3：否则 capture 线程会在
        # succeeded_windows>=3 之后抢跑出任意多窗口，FIFO 排空时长随机
        # （慢板 NX 上会超时）。
        code, message = service.pause()
        self.assertEqual(code, 0, message)
        blocked.release()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 3)
        self.wait_until(lambda: service.get_status().fifo_bytes == 0)
        service.stop_all()

    def test_worker_preflight_blocks_consume_until_disk_recovers(self):
        backend = FakeBackend(block_first=True)
        free_gb = [100.0]
        service = self.make_session(backend=backend, disk_free=lambda: free_gb[0])
        service.select_source("front")
        service.start_session("")
        self.wait_until(
            lambda: len(backend.submitted) == 1
            and service.get_status().pending_windows >= 1
        )
        free_gb[0] = 0.0
        backend.release_first.set()
        backend.jobs[backend.submitted[0]]["state"] = JOB_SUCCEEDED
        time.sleep(0.7)
        self.assertEqual(len(backend.submitted), 1)
        # Capture and worker threads both report the same disk shortage. The
        # most recent text can belong to either; assert the worker's own gate.
        self.assertTrue(service.diagnostics_snapshot()["worker_disk_blocked"])
        self.assertGreaterEqual(service.get_status().pending_windows, 1)
        free_gb[0] = 100.0
        self.wait_until(lambda: len(backend.submitted) >= 2)
        self.wait_until(lambda: service.get_status().succeeded_windows >= 2)
        service.stop_all()

    def test_stop_all_removes_pending_snapshots(self):
        backend = FakeBackend(block_first=True)
        service = self.make_session(backend=backend)
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: service.get_status().pending_windows >= 1)
        backend.release_first.set()  # 模拟真实 Action 可被 cancel 终态收敛
        service.stop_all(discard_queue=True)
        self.assertEqual(service.get_status().state, SESSION_STOPPED)
        self.assertFalse(any(self.front_root.iterdir()))

    def test_stop_reconstructs_queued_windows_before_stopping(self):
        backend = FakeBackend(block_first=True)
        service = self.make_session(backend=backend)
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: len(backend.submitted) == 1 and service.get_status().pending_windows >= 2)
        code, message = service.stop_all()
        self.assertEqual(code, 0, message)
        status = service.get_status()
        self.assertEqual(status.state, SESSION_DRAINING)
        queued = status.pending_windows
        self.assertGreaterEqual(queued, 2)
        captured_at_stop = status.window_index
        # Capture has stopped and the queue is kept: START and PAUSE are refused meanwhile.
        self.assertNotEqual(service.start_session("")[0], 0)
        self.assertNotEqual(service.pause()[0], 0)
        backend.jobs[backend.submitted[0]]["state"] = JOB_SUCCEEDED
        backend.release_first.set()
        final = self.wait_until(
            lambda: service.get_status() if service.get_status().state == SESSION_STOPPED else None,
            timeout=10.0,
        )
        self.assertEqual(final.failed_windows, 0)
        self.assertGreaterEqual(final.succeeded_windows, 1 + queued)
        self.assertLessEqual(final.window_index, captured_at_stop + 1)
        events = [json.loads(line)["event"] for line in
                  (Path(self.config.journal_root) / (final.session_id + ".jsonl")).read_text().splitlines()]
        self.assertLess(events.index("session_drain_start"), events.index("session_stop"))

    def test_drain_waits_for_g4_to_publish_every_window_before_stopping(self):
        service = self.make_session(backend=FakeBackend())
        polled = []

        class FakeG4:
            def status(self):
                polled.append(1)
                count = 3 if len(polled) >= 3 else 2  # the last window joins the map late
                return {"state": "ready" if len(polled) >= 3 else "optimizing", "window_count": count}

        stops = []
        service._stop_now = lambda: stops.append(len(polled)) or (0, "stopped")
        service._session_id = "s" + "0" * 16
        service._capture_stopped = True
        service._formal_windows = 3
        service._g4_process = FakeG4()
        service._capture_done.set()
        service._drain_then_stop(service._session_id)
        self.assertEqual(stops, [3])

    def test_second_stop_while_draining_discards_queue(self):
        backend = FakeBackend(block_first=True)
        service = self.make_session(backend=backend)
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: len(backend.submitted) == 1 and service.get_status().pending_windows >= 1)
        self.assertEqual(service.stop_all()[0], 0)
        self.assertEqual(service.get_status().state, SESSION_DRAINING)
        backend.release_first.set()  # 模拟真实 Action 可被 cancel 终态收敛
        self.assertEqual(service.stop_all()[0], 0)
        self.assertEqual(service.get_status().state, SESSION_STOPPED)
        self.assertFalse(any(self.front_root.iterdir()))

    def test_stop_all_reports_stopping_while_worker_thread_still_alive(self):
        backend = StuckBackend()
        service = self.make_session(backend=backend)
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_until(lambda: len(backend.submitted) >= 1)
        code, message = service.stop_all(discard_queue=True)
        self.assertNotEqual(code, 0)
        self.assertIn("still terminating", message)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)
        # STOPPING 期间绝不允许启动新 session 与旧线程共享状态。
        self.assertEqual(service.start_session("")[0], 3)
        backend.stuck = False
        self.wait_until(lambda: service._consume_done.is_set())
        self.assertEqual(service.stop_all(discard_queue=True)[0], 0)
        self.assertEqual(service.get_status().state, SESSION_STOPPED)

    def test_config_rejects_window_frames_above_group(self):
        with self.assertRaisesRegex(ContinuousConfigError, "exceeds reconstruct frame_group_size"):
            replace(self.config, initial_window_frames=9, reconstruct_frame_group_size=8)

    def test_reconstruction_mode_defaults_and_start_override(self):
        service = self.make_session(
            capture_factory=lambda role: BlockingCaptureClient(
                role, self.front_root
            )
        )
        service.select_source("front")
        self.assertEqual(
            service.start_session("", "invalid_mode")[0], 5
        )
        self.assertEqual(service.get_status().state, SESSION_IDLE)
        self.assertEqual(service.start_session("", "frame_pool")[0], 5)
        self.assertEqual(service.get_status().state, SESSION_IDLE)
        self.assertEqual(service.start_session("", "per_window")[0], 0)
        self.assertEqual(
            service.get_status().reconstruction_mode, "per_window"
        )
        service.stop_all()

    def test_config_rejects_unknown_reconstruction_mode(self):
        for mode in ("magic", "frame_pool"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ContinuousConfigError, "initial_reconstruction_mode"
            ):
                replace(self.config, initial_reconstruction_mode=mode)

    def test_selected_frame_indices_parses_result_manifest(self):
        result_dir = self.results / "window-selected"
        result_dir.mkdir()
        manifest = result_dir / "result.json"
        manifest.write_text(
            json.dumps(
                {
                    "selected_frames": [
                        {"index": 0, "selected_position": 0},
                        {"index": 5, "selected_position": 1},
                    ]
                }
            ),
            encoding="utf-8",
        )
        outcome = SimpleNamespace(result_manifest_uri=manifest.as_uri())
        self.assertEqual(
            ContinuousSession._selected_frame_indices(outcome), (0, 5)
        )
        for payload in (
            {"selected_frames": {"index": 0}},
            {"selected_frames": [{"index": True}]},
            {"selected_frames": [{"index": "0"}]},
        ):
            with self.subTest(payload=payload):
                manifest.write_text(json.dumps(payload), encoding="utf-8")
                self.assertIsNone(ContinuousSession._selected_frame_indices(outcome))

    def test_success_publishes_formal_record_with_config_timeline(self):
        config = replace(
            self.config,
            expected_checkpoint_sha256="b" * 64,
            expected_algorithm_revision="algo-test",
            expected_ros_revision="ros-test",
            worker_python="/usr/bin/python3",
            expected_checkpoint_path="/checkpoint/model.pt",
            expected_result_root=str(Path(self.temp.name) / "worker-results"),
        )
        backend = FakeBackend(block_first=True)
        service = self.make_session(
            backend=backend, config=config, formal_gate=None, formal_publisher=None
        )
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_until(lambda: len(backend.submitted) == 1)
        job_id = backend.submitted[0]
        job = backend.jobs[job_id]
        manifest_path = (
            Path(self.temp.name) / "worker-results" / job_id / "result.json"
        )
        manifest_sha = write_valid_result_manifest(
            manifest_path,
            scene_id=job_id,
            manifest_sha256=job["manifest_sha256"],
            captured_frames=self.config.initial_window_frames,
            camera_role="front",
            camera_topic="/front/image_raw",
            camera_frame_id="front_camera_optical_frame",
        )
        backend.result_manifest_uri = manifest_path.as_uri()
        backend.result_manifest_sha256 = manifest_sha
        backend.jobs[job_id]["state"] = JOB_SUCCEEDED
        backend.release_first.set()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        session_id = service.get_status().session_id
        self.assertEqual(service.get_status().formal_windows, 1)
        formal_dir = self.results / "sessions" / session_id / "window-0000"
        self.assertTrue((formal_dir / "result.json").is_file())
        self.assertEqual(
            (formal_dir / "result.json").read_bytes(),
            manifest_path.read_bytes(),
        )
        provenance = json.loads(
            (formal_dir / "provenance.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["schema"], "drone3d.continuous.formal_record")
        self.assertEqual(provenance["session_id"], session_id)
        self.assertEqual(provenance["window_index"], 0)
        self.assertTrue(provenance["effective_config"])
        effective_digest = hashlib.sha256(
            json.dumps(
                provenance["effective_config"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(provenance["config_digest"], effective_digest)
        self.assertEqual(len(provenance["config_timeline"]), 1)
        self.assertEqual(provenance["config_timeline"][0]["event"], "session_start")
        self.assertEqual(
            provenance["config_timeline"][0]["config_digest"],
            provenance["config_digest"],
        )
        self.assertTrue(provenance["gate"]["authenticity_verified"])
        self.assertTrue(provenance["alignment"]["unreliable"])
        window_events = [
            event
            for event in self.journal_events()
            if event.get("event") == "window_succeeded"
        ]
        self.assertGreaterEqual(len(window_events), 1)
        first_success = next(
            event for event in window_events if event.get("window_index") == 0
        )
        self.assertEqual(first_success["config_digest"], provenance["config_digest"])
        self.assertEqual(
            first_success["formal_provenance_sha256"],
            hashlib.sha256(
                (formal_dir / "provenance.json").read_bytes()
            ).hexdigest(),
        )
        service.stop_all()

    def test_stop_all_detected_during_gate_marks_window_cancelled(self):
        backend = FakeBackend(block_first=True)
        service = self.make_session(
            backend=backend, formal_gate=None, formal_publisher=None
        )

        def stop_during_gate(
            uri, expected_sha256, *, expected_identity, expected_authenticity
        ):
            # 模拟门禁执行期间 stop_all 置位：窗口必须按 CANCELLED 收敛，
            # 而不是被计为正式门禁失败。
            with service._lock:
                service._stopping = True
            return SimpleNamespace(ok=True, reason="", payload={}, manifest_bytes=b"{}")

        service._result_gate = stop_during_gate
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_until(lambda: len(backend.submitted) == 1)
        job_id = backend.submitted[0]
        backend.jobs[job_id]["state"] = JOB_SUCCEEDED
        backend.release_first.set()
        self.wait_until(
            lambda: service._windows
            and service._windows[0].state == WINDOW_CANCELLED
        )
        self.assertEqual(service.get_status().failed_windows, 0)
        self.assertEqual(service.get_status().formal_windows, 0)
        service.stop_all()

    def test_formal_gate_failure_keeps_window_failed_and_no_record(self):
        config = replace(
            self.config,
            expected_checkpoint_sha256="c" * 64,  # 与测试工件不符
            expected_algorithm_revision="algo-test",
            expected_ros_revision="ros-test",
            worker_python="/usr/bin/python3",
            expected_checkpoint_path="/checkpoint/model.pt",
            expected_result_root=str(Path(self.temp.name) / "worker-results"),
        )
        backend = FakeBackend(block_first=True)
        service = self.make_session(
            backend=backend, config=config, formal_gate=None, formal_publisher=None
        )
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_until(lambda: len(backend.submitted) == 1)
        job_id = backend.submitted[0]
        job = backend.jobs[job_id]
        manifest_path = (
            Path(self.temp.name) / "worker-results" / job_id / "result.json"
        )
        manifest_sha = write_valid_result_manifest(
            manifest_path,
            scene_id=job_id,
            manifest_sha256=job["manifest_sha256"],
            captured_frames=1,
            camera_role="front",
            camera_topic="/front/image_raw",
            camera_frame_id="front_camera_optical_frame",
        )
        backend.result_manifest_uri = manifest_path.as_uri()
        backend.result_manifest_sha256 = manifest_sha
        backend.jobs[job_id]["state"] = JOB_SUCCEEDED
        backend.release_first.set()
        self.wait_until(lambda: service.get_status().failed_windows >= 1)
        failed_record = service._windows[0]
        failed_snapshot = Path(unquote(urlsplit(failed_record.snapshot_uri).path))
        self.wait_until(lambda: not failed_snapshot.exists())
        session_id = service.get_status().session_id
        self.assertEqual(service.get_status().formal_windows, 0)
        self.assertFalse(
            (self.results / "sessions" / session_id / "window-0000").exists()
        )
        self.assertTrue(
            any(
                event.get("event") == "window_result_gate_failed"
                for event in self.journal_events()
            )
        )
        self.assertTrue(
            any(
                event.get("event") == "window_failed"
                and "formal result gate failed" in str(event.get("reason", ""))
                for event in self.journal_events()
            )
        )
        service.stop_all()

    def test_formal_publication_failure_fails_window_without_success(self):
        config = replace(
            self.config,
            expected_checkpoint_sha256="b" * 64,
            expected_algorithm_revision="algo-test",
            expected_ros_revision="ros-test",
            worker_python="/usr/bin/python3",
            expected_checkpoint_path="/checkpoint/model.pt",
            expected_result_root=str(Path(self.temp.name) / "worker-results"),
        )
        backend = FakeBackend(block_first=True)

        def failing_publisher(
            records_root,
            session_id,
            window_index,
            manifest_bytes,
            provenance,
            snapshot_manifest_bytes=b"",
        ):
            raise OSError("simulated publish failure")

        service = self.make_session(
            backend=backend,
            config=config,
            formal_gate=None,
            formal_publisher=failing_publisher,
        )
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_until(lambda: len(backend.submitted) == 1)
        job_id = backend.submitted[0]
        job = backend.jobs[job_id]
        manifest_path = (
            Path(self.temp.name) / "worker-results" / job_id / "result.json"
        )
        manifest_sha = write_valid_result_manifest(
            manifest_path,
            scene_id=job_id,
            manifest_sha256=job["manifest_sha256"],
            captured_frames=self.config.initial_window_frames,
            camera_role="front",
            camera_topic="/front/image_raw",
            camera_frame_id="front_camera_optical_frame",
        )
        backend.result_manifest_uri = manifest_path.as_uri()
        backend.result_manifest_sha256 = manifest_sha
        backend.jobs[job_id]["state"] = JOB_SUCCEEDED
        backend.release_first.set()
        self.wait_until(lambda: service.get_status().failed_windows >= 1)
        self.assertEqual(service.get_status().formal_windows, 0)
        self.assertTrue(
            any(
                event.get("event") == "window_failed"
                and "formal record publication failed" in str(event.get("reason", ""))
                for event in self.journal_events()
            )
        )
        service.stop_all()

    def test_start_session_initializes_save_sensor_depth_from_selected_role(self):
        service = self.make_session()
        service.select_source("front")
        service.start_session("")
        self.assertEqual(service.get_status().effective.save_sensor_depth, False)
        service.stop_all()

        # 第二个角色配置了 save_sensor_depth=True
        role = self.config.roles[1]
        role = CameraRole(
            role="down",
            capture_action="/down/capture_segment",
            capture_config=replace(
                role.capture_config, save_sensor_depth=True, depth_topic="/down/depth"
            ),
        )
        config = replace(self.config, roles=(self.config.roles[0], role))
        service = self.make_session(config=config)
        service.select_source("down")
        service.start_session("")
        self.assertEqual(service.get_status().effective.save_sensor_depth, True)
        service.stop_all()

    def test_initial_runtime_must_fit_every_capture_profile(self):
        with self.assertRaisesRegex(ContinuousConfigError, "capture profile"):
            replace(self.config, initial_sample_fps=0.1)

    def test_backend_status_exception_cleans_up_and_retries(self):
        backend = StatusRaiseOnceBackend()
        config = replace(self.config, initial_max_retries=1)
        clients = []

        def capture_factory(role):
            # 只允许第一个窗口进入队列：第二个窗口卡在采集里。暂停挡不住已经采完的窗口。
            clients.append(
                FakeCaptureClient(role, self.front_root) if not clients
                else BlockingCaptureClient(role, self.front_root)
            )
            return clients[-1]

        service = self.make_session(
            backend=backend, config=config, capture_factory=capture_factory
        )
        service.select_source("front")
        service.start_session("")
        # 成功后再检查 active slot 必须为空。
        final = self.wait_until(
            lambda: service.get_status()
            if service.get_status().succeeded_windows >= 1
            else None,
            timeout=8.0,
        )
        self.assertEqual(final.failed_windows, 0)
        first_window_id = backend.submitted[0]
        self.assertEqual(backend.status_calls_by_job[first_window_id], 3)
        self.assertEqual(backend.cancel_calls_by_job.get(first_window_id, 0), 1)
        self.assertEqual(backend.collect_calls_by_job.get(first_window_id, 0), 2)
        with service._lock:
            self.assertIsNone(service._active_job_id)
        service.stop_all()

    def test_backend_status_exception_waits_for_async_cancel_terminal(self):
        backend = AsyncCancelAfterStatusErrorBackend()
        config = replace(self.config, stop_timeout_sec=0.1)
        role = config.roles[0]
        first_capture = FakeCaptureClient(role, self.front_root)
        next_capture = BlockingCaptureClient(role, self.front_root)
        capture_calls = [0]

        def capture_factory(_role):
            capture_calls[0] += 1
            return first_capture if capture_calls[0] == 1 else next_capture

        service = self.make_session(
            backend=backend, config=config, capture_factory=capture_factory
        )
        service.start_session("front")
        self.wait_until(lambda: backend.cancel_calls >= 1)

        first_job_id = backend.submitted[0]
        time.sleep(0.3)
        self.assertEqual(backend.collect_calls_by_job.get(first_job_id, 0), 0)
        self.assertEqual(len(backend.submitted), 1)
        with service._lock:
            self.assertEqual(service._active_job_id, first_job_id)

        code, message = service.stop_all(discard_queue=True)
        self.assertEqual(code, 3, message)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)

        backend.cancel_terminal.set()
        self.wait_until(lambda: service._consume_done.is_set(), timeout=5.0)
        self.assertEqual(backend.collect_calls_by_job[first_job_id], 1)
        self.assertEqual(service.get_status().failed_windows, 0)
        with service._lock:
            self.assertEqual(service._windows[0].state, WINDOW_CANCELLED)
            self.assertEqual(service._windows[0].error, "stop_all")
        self.assertEqual(service.stop_all(discard_queue=True)[0], 0)
        self.assertEqual(service.get_status().state, SESSION_STOPPED)
        stop_event = next(
            event
            for event in reversed(self.journal_events())
            if event.get("event") == "session_stop"
        )
        self.assertTrue(stop_event["running_before"])
        self.assertTrue(stop_event["backend_cancel_requested"])
        self.assertTrue(stop_event["backend_terminal_collected"])
        self.assertEqual(stop_event["backend_terminal_state"], JOB_CANCELLED)
        self.assertEqual(stop_event["backend_terminal_message"], "cancelled")
        self.assertEqual(stop_event["terminal_state"], SESSION_STOPPED)
        for field in (
            "running_after",
            "pending_after",
            "worker_active_after",
            "threads_alive_after",
            "snapshot_after",
            "staging_after",
        ):
            self.assertEqual(stop_event[field], 0, field)

    def test_exception_cleanup_helper_does_not_collect_live_job(self):
        backend = AsyncCancelAfterStatusErrorBackend()
        service = self.make_session(backend=backend)
        job_id = backend.submit("file:///tmp/snapshot", "a" * 64, "manual-job")
        with service._lock:
            service._active_job_id = job_id
        cleanup = threading.Thread(
            target=service._cancel_active_job_best_effort, args=(backend,)
        )
        cleanup.start()
        self.wait_until(lambda: backend.cancel_calls >= 1)
        time.sleep(0.3)
        self.assertTrue(cleanup.is_alive())
        self.assertEqual(backend.collect_calls, 0)
        with service._lock:
            self.assertEqual(service._active_job_id, job_id)

        backend.cancel_terminal.set()
        cleanup.join(3.0)
        self.assertFalse(cleanup.is_alive())
        self.assertEqual(backend.collect_calls, 1)
        with service._lock:
            self.assertIsNone(service._active_job_id)

    def test_stop_audit_binds_job_returning_after_stop_started(self):
        backend = BlockingSubmitBackend()
        config = replace(self.config, stop_timeout_sec=0.1)
        role = config.roles[0]
        first_capture = FakeCaptureClient(role, self.front_root)
        next_capture = BlockingCaptureClient(role, self.front_root)
        capture_calls = [0]

        def capture_factory(_role):
            capture_calls[0] += 1
            return first_capture if capture_calls[0] == 1 else next_capture

        service = self.make_session(
            backend=backend, config=config, capture_factory=capture_factory
        )
        service.start_session("front")
        self.assertTrue(backend.submit_entered.wait(3.0))

        code, message = service.stop_all(discard_queue=True)
        self.assertEqual(code, 3, message)
        self.assertIsNone(service._stop_audit["backend_job_id_before"])

        backend.submit_release.set()
        self.wait_until(lambda: service._consume_done.is_set(), timeout=5.0)
        self.assertEqual(backend.cancel_calls, 1)
        self.assertEqual(backend.collect_calls, 1)
        self.assertEqual(service.get_status().failed_windows, 0)
        self.assertEqual(service.stop_all(discard_queue=True)[0], 0)

        stop_event = next(
            event
            for event in reversed(self.journal_events())
            if event.get("event") == "session_stop"
        )
        self.assertEqual(
            stop_event["backend_job_id_before"], backend.submitted[0]
        )
        self.assertTrue(stop_event["backend_cancel_requested"])
        self.assertTrue(stop_event["backend_terminal_collected"])
        self.assertEqual(stop_event["backend_terminal_state"], JOB_CANCELLED)
        self.assertEqual(stop_event["backend_terminal_message"], "cancelled")

    def test_cleanup_failed_terminal_keeps_stop_fail_closed(self):
        backend = CleanupFailedOnCancelBackend()
        role = self.config.roles[0]
        first_capture = FakeCaptureClient(role, self.front_root)
        next_capture = BlockingCaptureClient(role, self.front_root)
        capture_calls = [0]

        def capture_factory(_role):
            capture_calls[0] += 1
            return first_capture if capture_calls[0] == 1 else next_capture

        service = self.make_session(
            backend=backend, capture_factory=capture_factory
        )
        service.start_session("front")
        self.wait_until(lambda: len(backend.submitted) == 1)

        code, message = service.stop_all(discard_queue=True)

        self.assertEqual(code, 3, message)
        self.assertIn("cleanup_failed", message)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)
        self.assertEqual(service.start_session("front")[0], 3)
        stop_event = next(
            event
            for event in reversed(self.journal_events())
            if event.get("event") == "session_stop_pending"
        )
        self.assertEqual(stop_event["backend_terminal_state"], JOB_FAILED)
        self.assertIn("cleanup_failed", stop_event["backend_terminal_message"])
        self.assertTrue(stop_event["backend_terminal_collected"])
        self.assertIn("cleanup_failed", stop_event["reason"])
        self.assertEqual(stop_event["terminal_state"], SESSION_STOPPING)
        self.assertEqual(stop_event["threads_alive_after"], 0)

    def test_cleanup_failed_terminal_immediately_stops_new_submissions(self):
        backend = CleanupFailedOnCancelBackend()
        service = self.make_session(backend=backend)
        service.start_session("front")
        self.wait_until(lambda: len(backend.submitted) == 1)

        backend.jobs[backend.submitted[0]]["state"] = JOB_FAILED
        self.wait_until(
            lambda: service.get_status().state == SESSION_STOPPING,
            timeout=5.0,
        )
        time.sleep(0.7)

        self.assertEqual(len(backend.submitted), 1)
        self.assertEqual(service.start_session("front")[0], 3)
        code, message = service.stop_all()
        self.assertEqual(code, 3, message)
        self.assertIn("cleanup_failed", message)

    def test_stop_all_cancel_raises_keeps_stopping_until_convergence(self):
        backend = CancelRaisingBackend()
        service = self.make_session(backend=backend)
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: len(backend.submitted) >= 1)
        code, message = service.stop_all(discard_queue=True)
        self.assertEqual(code, 3, message)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)
        # 旧 worker/消费线程未收敛时不得启动新 session。
        self.assertEqual(service.start_session("")[0], 3)
        self.assertGreater(backend.cancel_calls, 0)
        # 让 cancel 恢复并让 worker 收敛；消费线程应自行退出。
        backend.cancel_raises = False
        backend.status_terminal = True
        self.wait_until(lambda: service._consume_done.is_set(), timeout=5.0)
        code, message = service.stop_all(discard_queue=True)
        self.assertEqual(code, 0, message)
        self.assertEqual(service.get_status().state, SESSION_STOPPED)


    def test_stop_all_waits_for_capture_terminal_before_stopped(self):
        config = replace(self.config, stop_timeout_sec=0.5)
        client = CancelSlowCaptureClient(config.roles[0], self.front_root)
        service = self.make_session(
            config=config, capture_factory=lambda _role: client
        )
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.assertTrue(client.goal_sent.wait(3.0))
        code, message = service.stop_all()
        self.assertNotEqual(code, 0)
        self.assertIn("capture action terminal", message)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)
        self.assertEqual(service.start_session("")[0], 3)
        with service._lock:
            self.assertIs(service._capture_client, client)
            pending = [
                record
                for record in service._windows
                if record.capture_cancel_pending
                and not record.capture_action_terminal
            ]
            self.assertEqual(len(pending), 1)
        snapshot = service.diagnostics_snapshot()
        self.assertTrue(snapshot["capture_cancel_pending"])
        self.assertFalse(snapshot["capture_action_terminal"])
        client.release_terminal()
        self.assertEqual(service.stop_all()[0], 0)
        self.assertEqual(service.get_status().state, SESSION_STOPPED)
        self.assertFalse(any(self.front_root.iterdir()))

    def test_pause_failure_keeps_running_with_last_error(self):
        client = PauseFailCaptureClient(self.config.roles[0], self.front_root)
        service = self.make_session(capture_factory=lambda _role: client)
        service.select_source("front")
        service.start_session("")
        self.assertTrue(client.goal_sent.wait(3.0))
        code, message = service.pause()
        self.assertEqual(code, 3)
        self.assertIn("pause request failed", message)
        self.assertEqual(service.get_status().state, SESSION_RUNNING)
        self.assertIn("pause request failed", service.get_status().last_error)
        client.release()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        service.stop_all()

    def test_resume_failure_keeps_paused_with_last_error(self):
        client = ResumeFailCaptureClient(self.config.roles[0], self.front_root)
        service = self.make_session(capture_factory=lambda _role: client)
        service.select_source("front")
        service.start_session("")
        self.assertTrue(client.goal_sent.wait(3.0))
        self.assertEqual(service.pause()[0], 0)
        code, message = service.resume()
        self.assertEqual(code, 3)
        self.assertIn("resume request failed", message)
        self.assertEqual(service.get_status().state, SESSION_PAUSED)
        self.assertIn("resume request failed", service.get_status().last_error)
        client.release()
        # 仍处于 PAUSED；resume 成功后窗口应继续完成。
        client.resume_result = True
        self.assertEqual(service.resume()[0], 0)
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        service.stop_all()

    def test_pause_between_client_creation_and_send_goal_suppresses_goal(self):
        client_holder = []
        service_holder = []

        def capture_factory(role):
            client = BlockingCaptureClient(role, self.front_root)
            client_holder.append(client)
            # 确定性注入：pause 恰好落在 client 创建之后、安装/发送 goal 之前。
            code, _message = service_holder[0].pause()
            self.assertEqual(code, 0)
            return client

        service = self.make_session(capture_factory=capture_factory)
        service_holder.append(service)
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: len(client_holder) == 1)
        client = client_holder[0]
        # 此时调度器不得发送 goal；保持 PAUSED。
        time.sleep(0.3)
        self.assertFalse(client.goal_sent.is_set())
        self.assertEqual(service.get_status().state, SESSION_PAUSED)
        # resume 后 goal 才会被发送。
        self.assertEqual(service.resume()[0], 0)
        self.assertTrue(client.goal_sent.wait(3.0))
        client.release()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        service.stop_all()

    def test_scheduler_watchdog_freezes_during_session_paused(self):
        config = replace(
            self.config,
            capture_wait_margin_sec=0.2,
            initial_sample_fps=5.0,
        )
        client = WatchdogFreezeCaptureClient(config.roles[0], self.front_root)
        clients = []

        def capture_factory(role):
            # Window 1 captures one new frame, the deadline the probe below needs.
            clients.append(FakeCaptureClient(role, self.front_root) if not clients else client)
            return clients[-1]

        service = self.make_session(config=config, capture_factory=capture_factory)
        service.select_source("front")
        service.start_session("")
        self.assertTrue(client.goal_sent.wait(3.0))
        self.assertEqual(client.max_frames, 1)
        self.assertEqual(service.pause()[0], 0)
        time.sleep(0.9)
        self.assertEqual(client.cancel_calls, 0)
        self.assertEqual(service.resume()[0], 0)
        client.release()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 2)
        self.assertEqual(client.cancel_calls, 0)
        service.stop_all()

    def test_captured_frames_tracks_active_capturing_window(self):
        client = BlockingCaptureClient(self.config.roles[0], self.front_root)
        def capture_factory(_role):
            if not client.goal_sent.is_set():
                return client
            return BlockingCaptureClient(self.config.roles[0], self.front_root)

        service = self.make_session(capture_factory=capture_factory)
        service.select_source("front")
        service.start_session("")
        self.assertTrue(client.goal_sent.wait(3.0))
        with service._lock:
            active = next(
                record
                for record in service._windows
                if record.state == WINDOW_CAPTURING
            )
            active.captured_frames = 7
            active.capture_goal_frames = 80
        self.assertEqual(service.get_status().captured_frames, 7)
        self.assertEqual(service.get_status().capture_target_frames, 80)
        self.assertEqual(
            service.diagnostics_snapshot()["captured_frames"], 7
        )
        self.assertEqual(
            service.diagnostics_snapshot()["capture_target_frames"], 80
        )
        # Restore the real goal before completion identity checks run.
        with service._lock:
            active.capture_goal_frames = client.max_frames
        client.release()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        self.assertEqual(service.get_status().captured_frames, 0)
        service.stop_all()
        self.assertEqual(service.get_status().capture_target_frames, 0)

    def test_stop_all_retains_active_job_when_collect_fails_then_converges(self):
        backend = CollectFailOnceBackend()
        service = self.make_session(backend=backend)
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: len(backend.submitted) == 1)
        job_id = backend.submitted[0]
        self.wait_until(lambda: backend.collect_calls >= 1)
        self.assertFalse(backend.release_collect.is_set())
        with service._lock:
            self.assertEqual(service._active_job_id, job_id)
        code, message = service.stop_all(discard_queue=True)
        self.assertNotEqual(code, 0)
        self.assertEqual(service.get_status().state, SESSION_STOPPING)
        with service._lock:
            self.assertEqual(service._active_job_id, job_id)
        self.assertEqual(service.start_session("")[0], 3)
        backend.release_collect.set()
        self.assertEqual(service.stop_all(discard_queue=True)[0], 0)
        self.assertEqual(service.get_status().state, SESSION_STOPPED)
        with service._lock:
            self.assertIsNone(service._active_job_id)



class G4SessionFixture(unittest.TestCase):
    """大窗口 G4 会话配置的共享夹具（本身不含用例）。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.journal = root / "journal"
        self.results = root / "results"
        self.journal.mkdir()
        self.results.mkdir()
        self._made = []

    def make_grouped_config(self, capture_overrides=None, **overrides):
        base = make_capture_config(Path(self.temp.name), "front_%d" % len(self._made))
        self._made.append(base)
        capture = replace(
            base,
            odometry_association="interpolate",
            max_goal_frames=128,
            max_goal_duration_sec=300.0,
            snapshot_limits=replace(base.snapshot_limits, max_frame_count=128),
        )
        if capture_overrides:
            limits = capture_overrides.pop("max_frame_count", None)
            capture = replace(capture, **capture_overrides)
            if limits is not None:
                capture = replace(
                    capture, snapshot_limits=replace(capture.snapshot_limits,
                                                     max_frame_count=limits)
                )
        role = CameraRole(
            role="front",
            capture_action="/front/capture_segment",
            capture_config=capture,
        )
        values = dict(
            roles=(role,),
            reconstruct_action="/reconstruct/reconstruct_segment",
            journal_root=str(self.journal),
            result_root=str(self.results),
            service_name="/continuous_session/command",
            diagnostics_topic="/diagnostics",
            diagnostics_period_sec=1.0,
            min_free_disk_gb=0.01,
            action_connect_timeout_sec=1.0,
            capture_wait_margin_sec=1.0,
            reconstruct_wait_timeout_sec=5.0,
            stop_timeout_sec=2.0,
            reconstruct_frame_group_size=120,
            initial_sample_fps=5.0,
            initial_window_frames=100,
            initial_overlap_frames=20,
            initial_queue_mode=QUEUE_LOSSLESS,
            initial_queue_capacity=4,
            initial_max_retries=0,
            initial_keep_snapshot_on_success=True,
            initial_inference_resolution=384,
            alignment_backend="global_g4",
        )
        values.update(overrides)
        return ContinuousConfig(**values)

    def make_session(self, capture_overrides=None, backend=None, publisher=None,
                     capture_factory=None, **overrides):
        return ContinuousSession(
            self.make_grouped_config(capture_overrides, **overrides),
            capture_client_factory=(
                capture_factory
                or (lambda role: FakeCaptureClient(role, Path(self.temp.name)))
            ),
            backend_factory=lambda: backend or FakeBackend(),
            disk_free_gb=lambda: 100.0,
            result_gate=_fake_result_gate,
            result_publisher=publisher or _fake_result_publisher,
        )


class FramePosePlacementConfigTests(G4SessionFixture):
    """g4_placement=frame_pose：会话交给地图后端的 profile。"""

    INTRINSICS = {"width": 640, "height": 480, "fx": 604.2, "fy": 604.0, "cx": 319.0, "cy": 247.5}

    def test_onboard_fusion_and_capture_offset_are_session_bound(self):
        legacy = self.make_session(g4_placement='frame_pose')._backend_profile()
        self.assertNotIn('frame_pose_fusion', legacy)
        session = self.make_session(g4_placement='frame_pose', frame_pose_fusion=True,
                                    pose_time_offset_ms=40., pose_time_offset_stage='capture')
        profile = session._backend_profile()
        self.assertTrue(profile['frame_pose_fusion'])
        self.assertEqual(profile['frame_pose'], dict(pose_time_offset_ms=40.,pose_time_offset_stage='capture'))
        cross = self.make_session(g4_placement='frame_pose', pose_time_offset_ms=40.,
                                  pose_time_offset_stage='capture', g4_cross_window_weight=.3)._backend_profile()
        self.assertEqual(cross['frame_pose']['cross_window_weight'], .3)
        frames = self.make_session(g4_placement='frame_pose', pose_time_offset_ms=40.,
                                   pose_time_offset_stage='capture', g4_frame_scales=True)._backend_profile()
        self.assertTrue(frames['frame_pose']['frame_scales'])
        with self.assertRaises(Exception):  # one scale solve or the other, not both
            self.make_session(g4_placement='frame_pose', pose_time_offset_stage='capture',
                              g4_frame_scales=True, g4_cross_window_weight=.3)
        with self.assertRaises(Exception):  # cross-window rows use the capture-stage poses
            self.make_session(g4_placement='frame_pose', g4_cross_window_weight=.3)

    def test_frame_pose_profile_carries_placement_offset_and_intrinsics(self):
        from g4_backend import FRAME_POSE_METHOD_ID
        session = self.make_session(g4_placement="frame_pose", pose_time_offset_ms=70.0,
                                    camera_intrinsics=dict(self.INTRINSICS))
        profile = session._backend_profile()
        self.assertEqual(profile["placement"], "frame_pose")
        self.assertEqual(profile["method_id"], FRAME_POSE_METHOD_ID)
        self.assertEqual(profile["frame_pose"], {"pose_time_offset_ms": 70.0})
        self.assertEqual(profile["camera_intrinsics"], self.INTRINSICS)
        # The profile must construct the real backend unchanged.
        from drone3d_reconstruction_core.g4.map_backend import build_backend
        root = Path(self.temp.name)
        (root / "records-fp").mkdir()
        backend = build_backend(root / "records-fp", "s0123456789abcdef", root / "out-fp", profile)
        try:
            self.assertEqual(backend.frame_pose_config.pose_time_offset_ms, 70.0)
            self.assertEqual(backend.camera_intrinsics["fx"], 604.2)
        finally:
            backend.close()

    def test_invalid_frame_pose_settings_are_rejected(self):
        for overrides, message in (
            (dict(g4_placement="bundle"), "g4_placement"),
            (dict(g4_placement="graph"), "graph placement was removed"),
            (dict(g4_placement="frame_pose", pose_time_offset_ms=5000.0), "pose_time_offset_ms"),
            (dict(g4_placement="frame_pose", camera_intrinsics={"fx": 600.0}), "camera_intrinsics"),
            (dict(g4_placement="frame_pose", alignment_backend="odom_pose_constrained_sim3",
                  initial_window_frames=16, initial_overlap_frames=5), "global_g4"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ContinuousConfigError, message):
                    self.make_grouped_config(None, **overrides)


class DiagnosticInputFrameDisplayTests(G4SessionFixture):
    """显示口径：每批新增/共享/模型输入按当前窗口的实际共享帧数给出。"""

    def display_numbers(self, session):
        snapshot = session.diagnostics_snapshot()
        return (snapshot["new_input_frames"], snapshot["shared_input_frames"],
                snapshot["model_input_frames"])

    def test_session_without_a_window_shows_the_first_window_shape(self):
        session = self.make_session()
        self.assertEqual(self.display_numbers(session), (100, 0, 100))

    def test_g4_total_frames_semantics_is_unchanged(self):
        session = self.make_session(reconstruct_frame_group_size=16,
                                    initial_window_frames=16,
                                    initial_overlap_frames=5)
        record = WindowRecord(1, "legacy", session._config.roles[0], session._runtime,
                              overlap_frame_count=5)
        record.state = WINDOW_CAPTURING
        session._windows = [record]
        self.assertEqual(self.display_numbers(session), (11, 5, 16))


class ResourceGuardBackend(FakeBackend):
    """按 0 基提交序号拒绝启动的 backend（模拟启动资源门）。

    门禁拒绝是 **终态失败**：worker 从未真正跑起来，消息里带拒绝码
    （LOW_MEM_AVAILABLE / HIGH_SWAP_USED / HIGH_SWAP_GROWTH）。
    """

    def __init__(self, deny_indices=(0,), code="HIGH_SWAP_USED", message=None):
        super().__init__()
        self.deny_indices = tuple(deny_indices)
        self.code = code
        self.message = message
        # 人工重试会用同一个 scene_id 再提交一次，所以拒绝必须按**本次尝试**
        # 记录，而不是按 job id 去 submitted 列表里找（会命中第一次）。
        self.denied_attempts = {}

    def submit(self, *args, **kwargs):
        index = len(self.submitted)
        job_id = super().submit(*args, **kwargs)
        denied = index in self.deny_indices
        self.denied_attempts[job_id] = denied
        if denied:
            self.jobs[job_id]["state"] = JOB_FAILED
        return job_id

    def _denied(self, job_id):
        return bool(self.denied_attempts.get(job_id))

    def collect_result(self, job_id):
        if self._denied(job_id):
            job = self.jobs[job_id]
            text = self.message or f"resource gate denied launch: {self.code}"
            return ReconstructActionOutcome(
                action_state=ACTION_FAILED,
                error_code=3,
                message=text,
                scene_id=job_id,
                result_manifest_uri="",
                result_manifest_sha256="",
                scene_glb_uri="",
                frame_count=1,
                wall_time_sec=0.1,
                camera_role="front",
                camera_topic="/front/image_raw",
                camera_frame_id="front_camera_optical_frame",
                snapshot_manifest_sha256=job["manifest_sha256"],
                result_scene_id=job_id,
                result_actual_frame_count=1,
            )
        return super().collect_result(job_id)


class GuardDeniedAfterDelayBackend(ResourceGuardBackend):
    """第一个被拒 job 先报告 RUNNING 一小段时间，再以启动资源门拒绝结束。

    这段延迟让采集线程有机会把后续窗口排进 FIFO（容量内），用来验证人工
    恢复后的消费顺序：重试窗必须排在所有更晚的窗口之前。
    """

    def __init__(self, deny_indices=(0,), code="HIGH_SWAP_USED", hold_sec=0.6):
        super().__init__(deny_indices=deny_indices, code=code)
        self.hold_sec = hold_sec
        self.hold_until = {}

    def submit(self, *args, **kwargs):
        index = len(self.submitted)
        job_id = super().submit(*args, **kwargs)
        if index in self.deny_indices:
            self.jobs[job_id]["state"] = JOB_RUNNING
            self.hold_until[job_id] = time.monotonic() + self.hold_sec
        return job_id

    def status(self, job_id):
        job = self.jobs[job_id]
        if (
            job_id in self.hold_until
            and job["state"] == JOB_RUNNING
            and time.monotonic() >= self.hold_until[job_id]
        ):
            job["state"] = JOB_FAILED
        return JobStatus(job_id, job["state"], "")


class EventGatedGuardBackend(ResourceGuardBackend):
    """被拒 job 一直报告 RUNNING，直到测试显式放行。

    这样"资源拒绝落地时采集正处于 ACTIVE"是用 Event 精确构造的，不依赖
    任何 sleep 调时序。
    """

    def __init__(self, deny_indices=(0,), code="HIGH_SWAP_USED"):
        super().__init__(deny_indices=deny_indices, code=code)
        self.denial_release = threading.Event()

    def submit(self, *args, **kwargs):
        index = len(self.submitted)
        job_id = super().submit(*args, **kwargs)
        if index in self.deny_indices:
            self.jobs[job_id]["state"] = JOB_RUNNING
        return job_id

    def status(self, job_id):
        job = self.jobs[job_id]
        if (
            job["state"] == JOB_RUNNING
            and self._denied(job_id)
            and self.denial_release.is_set()
        ):
            job["state"] = JOB_FAILED
        return JobStatus(job_id, job["state"], "")


class MemoryPausedManualRecoveryTests(unittest.TestCase):
    """MEMORY_PAUSED 的人工恢复：只重试被资源门拒绝的那个窗口。

    验收要点：仍是同一个启动资源门（不放宽阈值、不自动重试）、
    不重新采集已 sealed 窗口、不遗漏/重复正式结果、失败历史可查、
    CUDA OOM 不可恢复、线程与 FIFO 顺序在恢复后仍然正确。
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.journal = root / "journal"
        self.journal.mkdir()
        self.results = root / "results"
        self.results.mkdir()
        role = CameraRole(
            role="front",
            capture_action="/front/capture_segment",
            capture_config=replace(make_capture_config(root, "front"),
                                   odometry_association="interpolate"),
        )
        self.role = role
        self.front_root = Path(role.capture_config.snapshot_root)
        self.config = ContinuousConfig(
            roles=(role,),
            reconstruct_action="/reconstruct/reconstruct_segment",
            journal_root=str(self.journal),
            result_root=str(self.results),
            service_name="/continuous_session/command",
            diagnostics_topic="/diagnostics",
            diagnostics_period_sec=1.0,
            min_free_disk_gb=0.01,
            action_connect_timeout_sec=1.0,
            capture_wait_margin_sec=1.0,
            reconstruct_wait_timeout_sec=5.0,
            stop_timeout_sec=2.0,
            reconstruct_frame_group_size=6,
            initial_sample_fps=5.0,
            initial_window_frames=6,
            initial_overlap_frames=5,
            initial_inference_resolution=384,
            initial_queue_mode=QUEUE_LOSSLESS,
            initial_queue_capacity=8,
            initial_max_retries=0,
            initial_keep_snapshot_on_success=True,
        )
        _patch_g4_process(self)
        self.services = []
        self.capture_clients = []
        self.formal = []

    def tearDown(self):
        for service in self.services:
            try:
                service.shutdown(2.0)
            except BaseException:
                pass
        self.services.clear()

    def make_session(self, backend, config=None):
        def factory(role):
            client = FakeCaptureClient(role, self.front_root)
            self.capture_clients.append(client)
            return client

        def publisher(records_root, session_id, window_index, *args, **kwargs):
            self.formal.append((session_id, window_index))
            return _fake_result_publisher(
                records_root, session_id, window_index, *args, **kwargs
            )

        service = ContinuousSession(
            config or self.config,
            capture_client_factory=factory,
            backend_factory=lambda: backend,
            disk_free_gb=lambda: 100.0,
            result_gate=_fake_result_gate,
            result_publisher=publisher,
        )
        self.services.append(service)
        return service

    def wait_until(self, predicate, timeout=8.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        raise AssertionError("condition not reached")

    def journal_events(self, session_id):
        path = self.journal / f"{session_id}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def start_and_pause_on_guard(self, backend, config=None):
        """跑到资源门拒绝：返回 (service, 被拒窗口的记录)。"""
        service = self.make_session(backend, config)
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_paused_on_guard(service)
        record = service._memory_failed_window
        self.assertIsNotNone(record)
        return service, record

    def wait_paused_on_guard(self, service):
        """等到资源门暂停真正落定。

        MEMORY_PAUSED 状态先于消费线程放开该窗口可见；在这个空档里 RESUME 按设计
        回答“稍后重试”（代码 3）。测试要发命令，就得先等消费线程放手。
        """
        self.wait_until(lambda: service.get_status().state == SESSION_MEMORY_PAUSED)
        self.wait_until(lambda: service._running_window is None)

    def snapshot_path(self, record):
        return Path(unquote(urlsplit(record.snapshot_uri).path))

    # ------------------------------------------------------------------
    def test_guard_denial_pauses_without_auto_retry_and_keeps_the_input(self):
        backend = ResourceGuardBackend()
        service, record = self.start_and_pause_on_guard(backend)
        status = service.get_status()
        self.assertEqual(status.state, SESSION_MEMORY_PAUSED)
        self.assertEqual(status.failed_windows, 1)
        self.assertEqual(record.state, WINDOW_FAILED)
        self.assertTrue(record.done.is_set())
        self.assertIn("HIGH_SWAP_USED", record.error)
        self.assertTrue(self.snapshot_path(record).is_dir())
        self.assertTrue(service._fatal_oom)
        self.assertFalse(service._oom_latched)
        self.assertTrue(service._memory_guard_paused)
        # 资源门失败不是可自动重试的失败：max_retries=0 也仍然只提交一次。
        self.assertEqual(len(backend.submitted), 1)
        self.assertEqual(service._fifo, [])
        # 采集/消费线程都因硬停退出，单改 state 无法恢复。
        self.assertTrue(service._capture_done.wait(5.0))
        self.assertTrue(service._consume_done.wait(5.0))
        self.assertFalse(service._capture_thread.is_alive())
        self.assertFalse(service._consume_thread.is_alive())
        snapshot = service.diagnostics_snapshot()
        self.assertTrue(snapshot["memory_paused"])
        self.assertFalse(snapshot["oom_latched"])
        self.assertEqual(snapshot["attempt_failures"], 1)
        self.assertEqual(snapshot["manual_retries"], 0)

    def test_manual_resume_retries_only_that_window_through_the_same_gate(self):
        backend = ResourceGuardBackend()
        service, record = self.start_and_pause_on_guard(backend)
        code, message = service.resume()
        self.assertEqual(code, 0, message)
        self.assertIn("window %d" % record.window_index, message)
        self.wait_until(lambda: record.state == WINDOW_SUCCEEDED)
        status = service.get_status()
        # 同一个窗口、第二次提交；门禁本身没有任何放宽（仍是同一 backend 路径）。
        # 恢复后采集继续，后面的窗口几十毫秒内就会排进来，所以只数被重试的这个窗口。
        retried = backend.submitted[0]
        self.assertEqual(backend.submitted[:2], [retried, retried])
        self.assertEqual(backend.submitted.count(retried), 2)
        self.assertEqual(record.state, WINDOW_SUCCEEDED)
        self.assertEqual(record.manual_retries, 1)
        self.assertTrue(any("HIGH_SWAP_USED" in item for item in record.guard_failures))
        self.assertEqual(status.failed_windows, 0)
        self.assertEqual(service._failed, 0)
        self.assertEqual(service._attempt_failures, 1)
        formal_windows = [index for _, index in self.formal]
        self.assertEqual(formal_windows[0], record.window_index)
        self.assertEqual(formal_windows.count(record.window_index), 1)
        # 已 sealed 窗口的采集没有被重做。
        goals = [client.scene_id for client in self.capture_clients
                 if client.goal_sent.is_set()]
        self.assertEqual(goals.count(record.scene_id), 1)
        self.assertTrue(service._capture_thread.is_alive())
        self.assertTrue(service._consume_thread.is_alive())
        events = [row["event"] for row in self.journal_events(status.session_id)]
        self.assertIn("session_memory_paused", events)
        self.assertIn("window_manual_retry", events)
        self.assertIn("window_succeeded", events)

    def test_manual_retry_keeps_fifo_order_with_later_windows_pending(self):
        backend = GuardDeniedAfterDelayBackend()
        service = self.make_session(backend)
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_paused_on_guard(service)
        record = service._memory_failed_window
        pending = [item.window_index for item in service._fifo]
        # 采集已经跑在消费前面：暂停时队列里已有更晚的窗口。
        self.assertGreaterEqual(len(pending), 2)
        self.assertEqual(pending, sorted(pending))
        self.assertTrue(all(index > record.window_index for index in pending))
        expected = [record.window_index] + pending
        code, message = service.resume()
        self.assertEqual(code, 0, message)
        self.wait_until(lambda: len(self.formal) >= len(expected))
        # 重试窗插在队首：正式结果前 len(expected) 条仍是完整、升序、无重复
        # 的窗口序列（采集线程随后还会继续后面的窗口）。
        self.assertEqual([index for _, index in self.formal][:len(expected)],
                         expected)
        self.assertEqual(service._failed, 0)
        self.assertEqual(service.get_status().dropped_windows, 0)
        self.assertEqual(service._attempt_failures, 1)
        service.stop_all()
        self.assertEqual(service._windows[0].manual_retries, 1)

    def test_cuda_oom_is_never_recoverable(self):
        service = self.make_session(OomBackend())
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_until(lambda: service.get_status().state == SESSION_FATAL_OOM)
        self.assertTrue(service._oom_latched)
        self.assertFalse(service._memory_guard_paused)
        self.assertIsNone(service._memory_failed_window)
        code, message = service.resume()
        self.assertEqual(code, 4)
        self.assertIn(SESSION_FATAL_OOM, message)
        self.assertEqual(service.get_status().state, SESSION_FATAL_OOM)
        snapshot = service.diagnostics_snapshot()
        self.assertFalse(snapshot["memory_paused"])
        self.assertTrue(snapshot["oom_latched"])

    def test_message_matching_both_patterns_is_not_treated_as_recoverable(self):
        backend = ResourceGuardBackend(
            message="worker stopped: resource_guard: CUDA out of memory"
        )
        service = self.make_session(backend)
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_until(lambda: service.get_status().state == SESSION_FATAL_OOM)
        self.assertTrue(service._oom_latched)
        self.assertFalse(service._memory_guard_paused)
        self.assertEqual(service.resume()[0], 4)

    def test_resume_refuses_when_the_preserved_snapshot_is_gone(self):
        backend = ResourceGuardBackend()
        service, record = self.start_and_pause_on_guard(backend)
        path = self.snapshot_path(record)
        shutil.rmtree(path)
        code, message = service.resume()
        self.assertEqual(code, 3)
        self.assertIn("snapshot", message)
        self.assertEqual(service.get_status().state, SESSION_MEMORY_PAUSED)
        self.assertIs(service._memory_failed_window, record)
        self.assertEqual(len(backend.submitted), 1)
        self.assertEqual(self.formal, [])

    def test_resume_refuses_while_a_job_is_still_owned(self):
        backend = ResourceGuardBackend()
        service, record = self.start_and_pause_on_guard(backend)
        service._active_job_id = "job-still-owned"
        code, message = service.resume()
        self.assertEqual(code, 3)
        self.assertIn("job-still-owned", message)
        self.assertEqual(service.get_status().state, SESSION_MEMORY_PAUSED)
        service._active_job_id = None
        service._running_window = record
        self.assertEqual(service.resume()[0], 3)
        self.assertEqual(service.get_status().state, SESSION_MEMORY_PAUSED)
        service._running_window = None
        self.assertEqual(service.resume()[0], 0)

    def test_repeated_guard_denial_pauses_again_and_stays_operator_driven(self):
        backend = ResourceGuardBackend(deny_indices=(0, 1))
        service, record = self.start_and_pause_on_guard(backend)
        self.assertEqual(service.resume()[0], 0)
        self.wait_until(lambda: service.get_status().state == SESSION_MEMORY_PAUSED)
        self.assertEqual(record.manual_retries, 1)
        self.assertEqual(len(record.guard_failures), 2)
        self.assertEqual(service._failed, 1)
        self.assertEqual(service._attempt_failures, 2)
        self.assertEqual(service.get_status().failed_windows, 1)
        self.assertEqual(service.resume()[0], 0)
        self.wait_until(lambda: record.state == WINDOW_SUCCEEDED)
        self.assertEqual(service._failed, 0)
        self.assertEqual(service._attempt_failures, 2)

    def test_resume_refuses_when_the_preserved_snapshot_hash_is_wrong(self):
        """目录还在但内容被改过：必须由现有验证器拒绝，而不是只看目录存在。"""
        backend = ResourceGuardBackend()
        service, record = self.start_and_pause_on_guard(backend)
        path = self.snapshot_path(record)
        images = sorted((path / "images").iterdir())
        self.assertTrue(images)
        images[0].write_bytes(images[0].read_bytes() + b"tampered")
        code, message = service.resume()
        self.assertEqual(code, 3)
        self.assertIn("validation", message)
        self.assertEqual(service.get_status().state, SESSION_MEMORY_PAUSED)
        self.assertEqual(len(backend.submitted), 1)

    def test_duplicate_resume_is_refused_while_one_is_in_progress(self):
        backend = ResourceGuardBackend()
        service, record = self.start_and_pause_on_guard(backend)
        real_join = service._join_threads
        entered = threading.Event()
        release = threading.Event()
        calls = {"count": 0}

        def held_join(deadline=None):
            calls["count"] += 1
            if calls["count"] == 1:
                entered.set()
                release.wait(5.0)
            return real_join(deadline)

        service._join_threads = held_join
        results = []
        worker = threading.Thread(target=lambda: results.append(service.resume()))
        worker.start()
        self.assertTrue(entered.wait(5.0))
        self.assertEqual(service.resume()[0], 4)
        release.set()
        worker.join(5.0)
        self.assertEqual(results[0][0], 0)
        self.wait_until(lambda: record.state == WINDOW_SUCCEEDED)
        # 只入队一次、只提交一次重试：不重复正式结果。后面的窗口不在此列。
        self.assertEqual(backend.submitted.count(backend.submitted[0]), 2)
        self.assertEqual(
            [index for _, index in self.formal].count(record.window_index), 1
        )
        self.assertEqual(record.manual_retries, 1)

    def test_stop_all_during_manual_resume_keeps_the_session_stopped(self):
        backend = ResourceGuardBackend()
        service, record = self.start_and_pause_on_guard(backend)
        real_join = service._join_threads
        entered = threading.Event()
        release = threading.Event()
        calls = {"count": 0}

        def held_join(deadline=None):
            calls["count"] += 1
            if calls["count"] == 1:
                entered.set()
                release.wait(5.0)
            return real_join(deadline)

        service._join_threads = held_join
        results = []
        worker = threading.Thread(target=lambda: results.append(service.resume()))
        worker.start()
        self.assertTrue(entered.wait(5.0))
        self.assertEqual(service.stop_all()[0], 0)
        release.set()
        worker.join(5.0)
        self.assertEqual(results[0][0], 4)
        self.assertEqual(service.get_status().state, SESSION_STOPPED)
        for thread in (service._capture_thread, service._consume_thread):
            if thread is not None:
                self.assertFalse(thread.is_alive())
        self.assertEqual(self.formal, [])
        self.assertEqual(service.get_status().succeeded_windows, 0)

    def pause_with_a_cancelled_inflight_capture(self):
        """构造真实场景：资源拒绝落在 k 窗时，采集正在飞 k+1 窗。

        第 2 个 capture client 会一直阻塞在 wait_for_result，直到硬停取消它；
        资源拒绝由 Event 放行，确保拒绝落地时该采集确实处于 ACTIVE。
        """
        backend = EventGatedGuardBackend()
        backend.frame_count_by_index = [self.config.initial_window_frames] * 3
        clients = []
        blocked = BlockingCaptureClient(self.role, self.front_root)

        def factory(role):
            # 索引 0 正常；索引 1 在飞并被硬停取消；恢复后的重新采集用正常 client。
            client = (blocked if len(clients) == 1
                      else FakeCaptureClient(role, self.front_root))
            clients.append(client)
            return client

        service = ContinuousSession(
            self.config,
            capture_client_factory=factory,
            backend_factory=lambda: backend,
            disk_free_gb=lambda: 100.0,
            result_gate=_fake_result_gate,
            result_publisher=lambda *args, **kwargs: (
                self.formal.append((args[1], args[2]))
                or _fake_result_publisher(*args, **kwargs)
            ),
        )
        self.services.append(service)
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        # 先确认在飞采集已 ACTIVE（记录处于 CAPTURING），再放行资源拒绝。
        self.assertTrue(blocked.goal_sent.wait(5.0))
        self.wait_until(lambda: len(service._windows) == 2
                        and service._windows[1].state == WINDOW_CAPTURING)
        backend.denial_release.set()
        self.wait_until(lambda: service.get_status().state == SESSION_MEMORY_PAUSED)
        record = service._memory_failed_window
        self.assertEqual(record.window_index, 0)
        # 硬停 latch 先落下，在飞采集的取消结果稍后才写回记录。
        self.wait_until(lambda: any(
            item is not record and item.state == WINDOW_FAILED
            for item in service._windows
        ))
        cancelled = [item for item in service._windows
                     if item is not record and item.state == WINDOW_FAILED]
        self.assertEqual([item.window_index for item in cancelled], [1])
        self.assertIn("fatal oom", cancelled[0].error)
        # 取消不会推进采集前沿：该索引恢复后还会被重新采集。
        self.assertEqual(service._next_window, 1)
        return service, record, cancelled[0], backend

    def test_cancelled_inflight_capture_is_recaptured_after_manual_retry(self):
        """被硬停取消的在飞采集窗必须在恢复后被重新采集，不能静默跳过。"""
        service, record, cancelled, backend = (
            self.pause_with_a_cancelled_inflight_capture()
        )
        code, message = service.resume()
        self.assertEqual(code, 0, message)
        self.wait_until(lambda: service.get_status().succeeded_windows >= 2)
        # 被取消的计划窗被重新采集并产出正式结果：0 与 1 都不缺、都不重复。
        # 采集线程随后还会继续后面的窗口，所以只要求前沿已越过被取消的索引。
        self.assertEqual([index for _, index in self.formal][:2], [0, 1])
        self.assertGreaterEqual(service._next_window, 2)
        self.assertEqual(service._failed, 0)
        self.assertTrue(all(item is record or item.window_index != 0
                            for item in service._windows))
        self.assertEqual(sum(item.window_index <= 1 for item in service._windows), 3)

    def test_completed_recapture_does_not_block_a_later_resource_retry(self):
        service = self.make_session(ResourceGuardBackend())
        cancelled = WindowRecord(1, "cs-test-w1", self.role, service._runtime,
                                 state=WINDOW_FAILED, error="fatal oom")
        replacement = replace(
            cancelled, state=WINDOW_SUCCEEDED, error="",
            snapshot_uri=(self.front_root / "cs-test-w1-0123456789abcdef").as_uri(),
            manifest_sha256="a" * 64,
        )
        target = WindowRecord(2, "cs-test-w2", self.role, service._runtime,
                              state=WINDOW_FAILED, error="MEMORY_PAUSED: HIGH_SWAP_USED")
        service._windows = [cancelled, replacement, target]
        service._memory_failed_window = target
        service._next_window = 3
        self.assertIsNone(service._unfinished_capture_error_locked())
        # 同索引的未封存记录不能掩盖真实缺窗。
        for changes in ({"manifest_sha256": ""}, {"snapshot_uri": ""},
                        {"state": WINDOW_CAPTURING}, {"capture_action_terminal": False}):
            with self.subTest(invalid_replacement=changes):
                service._windows[1] = replace(replacement, **changes)
                self.assertIn("window 1", service._unfinished_capture_error_locked())

    def test_resume_refuses_while_a_cancelled_capture_has_not_converged(self):
        """在飞采集取消不收敛时，线程仍活着：必须拒绝 RESUME 并保留现场。"""
        backend = EventGatedGuardBackend()
        backend.frame_count_by_index = [self.config.initial_window_frames] * 2
        clients = []
        stuck = CancelSlowCaptureClient(self.role, self.front_root)

        def factory(role):
            client = stuck if len(clients) == 1 else FakeCaptureClient(role, self.front_root)
            clients.append(client)
            return client

        service = ContinuousSession(
            self.config,
            capture_client_factory=factory,
            backend_factory=lambda: backend,
            disk_free_gb=lambda: 100.0,
            result_gate=_fake_result_gate,
            result_publisher=lambda *args, **kwargs: (
                self.formal.append((args[1], args[2]))
                or _fake_result_publisher(*args, **kwargs)
            ),
        )
        self.services.append(service)
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.assertTrue(stuck.goal_sent.wait(5.0))
        self.wait_until(lambda: len(service._windows) == 2
                        and service._windows[1].state == WINDOW_CAPTURING)
        backend.denial_release.set()
        self.wait_until(lambda: service.get_status().state == SESSION_MEMORY_PAUSED)
        # 取消不收敛：采集线程仍卡在在飞动作里。
        self.assertTrue(service._capture_thread.is_alive())
        code, message = service.resume()
        self.assertEqual(code, 3)
        self.assertIn("converging", message)
        self.assertEqual(service.get_status().state, SESSION_MEMORY_PAUSED)
        self.assertEqual(self.formal, [])
        # 现场保留：被拒窗口的输入仍在，且没有第二个消费线程被启动。
        self.assertTrue(self.snapshot_path(service._memory_failed_window).is_dir())
        self.assertEqual(len(backend.submitted), 1)
        stuck.release_terminal()
        self.wait_until(lambda: not service._capture_thread.is_alive())
        code, message = service.resume()
        self.assertEqual(code, 0, message)
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)

    def test_old_resume_never_arms_a_session_started_during_its_join(self):
        """首检→锁外等待期间 STOP 并重建会话：旧 RESUME 不得打到新会话上。"""
        backend = ResourceGuardBackend(deny_indices=(0, 1))
        service, record = self.start_and_pause_on_guard(backend)
        first_session = service.get_status().session_id
        real_join = service._join_threads
        entered = threading.Event()
        release = threading.Event()
        calls = {"count": 0}

        def held_join(deadline=None):
            calls["count"] += 1
            if calls["count"] == 1:
                entered.set()
                release.wait(5.0)
            return real_join(deadline)

        service._join_threads = held_join
        results = []
        worker = threading.Thread(target=lambda: results.append(service.resume()))
        worker.start()
        self.assertTrue(entered.wait(5.0))
        # 锁外等待期间：STOP 完成 → 立刻开新会话 → 新会话同样被资源门拒绝。
        self.assertEqual(service.stop_all()[0], 0)
        self.assertEqual(service.start_session("")[0], 0)
        new_session = service.get_status().session_id
        self.assertNotEqual(new_session, first_session)
        self.wait_until(lambda: service.get_status().state == SESSION_MEMORY_PAUSED)
        # 新会话自己的采集可能在暂停生效前已正常入队；等它的线程退出后再记录队列，
        # 判据是旧事务不改动新会话的队列，而不是队列恰好为空（负载下后者不稳定）。
        self.wait_until(lambda: not service._capture_thread.is_alive()
                        and not service._consume_thread.is_alive())
        new_record = service._memory_failed_window
        self.assertIsNotNone(new_record)
        fifo_before = list(service._fifo)
        submits_before = len(backend.submitted)
        release.set()
        worker.join(5.0)
        # 旧事务必须自认失效：不重试、不启动线程、不动新会话的状态。
        self.assertEqual(results[0][0], 4)
        self.assertEqual(service.get_status().state, SESSION_MEMORY_PAUSED)
        self.assertIs(service._memory_failed_window, new_record)
        self.assertEqual(service._manual_retries, 0)
        self.assertEqual(new_record.manual_retries, 0)
        self.assertEqual(service._fifo, fifo_before)
        self.assertEqual(len(backend.submitted), submits_before)
        self.assertFalse(service._capture_thread.is_alive())
        self.assertFalse(service._consume_thread.is_alive())
        # 新会话自己的 RESUME 仍然可用（旧事务没有清掉新事务的标志）。
        code, message = service.resume()
        self.assertEqual(code, 0, message)
        self.wait_until(lambda: new_record.state == WINDOW_SUCCEEDED)
        self.assertEqual(new_record.manual_retries, 1)

    def test_plain_pause_resume_is_unchanged(self):
        backend = ResourceGuardBackend(deny_indices=())
        service = self.make_session(backend)
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_until(lambda: self.capture_clients and
                        self.capture_clients[0].goal_sent.is_set())
        self.assertEqual(service.pause()[0], 0)
        self.assertEqual(service.get_status().state, SESSION_PAUSED)
        self.assertEqual(service.resume()[0], 0)
        self.assertEqual(service.get_status().state, SESSION_RUNNING)
        self.assertEqual(service.stop_all()[0], 0)
        # MEMORY_PAUSED 但并非资源门暂停（例如被其它路径置位）时拒绝恢复。
        service._state = SESSION_MEMORY_PAUSED
        code, message = service.resume()
        self.assertEqual(code, 4)
        self.assertIn(SESSION_MEMORY_PAUSED, message)
        self.assertEqual(service.get_status().state, SESSION_MEMORY_PAUSED)


if __name__ == "__main__":
    unittest.main()
