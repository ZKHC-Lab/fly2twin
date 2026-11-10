"""CPU-only tests for the continuous session scheduler core."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.parse import unquote, urlsplit

from drone3d_reconstruction_core.capture import CaptureConfig
from drone3d_reconstruction_core.pipeline_service import (
    ACTION_FAILED,
    ACTION_SUCCEEDED,
    CameraRole,
    CaptureActionOutcome,
    ReconstructActionOutcome,
)
from drone3d_reconstruction_core.snapshot_validator import SnapshotSafetyLimits, validate_snapshot
from drone3d_reconstruction_core.continuous_session import (
    COORDINATE_MODEL_LOCAL_UNALIGNED,
    COORDINATE_ODOM_LOCAL_APPROX,
    ContinuousConfig,
    ContinuousConfigError,
    ContinuousSession,
    EffectiveConfig,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    QUEUE_LOSSLESS,
    QUEUE_REALTIME_LATEST,
    SESSION_ALIGNMENT_PAUSED,
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
    WINDOW_PENDING,
    WINDOW_SUCCEEDED,
    WindowRecord,
    JobStatus,
    ReconstructionBackend,
    cleanup_stale_snapshots,
)
from drone3d_reconstruction_core.frame_pool import (
    FramePool,
    FramePoolConfig,
    FramePoolError,
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


def write_minimal_snapshot(snapshot_dir: Path, scene_id: str, frame_count: int, capture_config=None) -> str:
    """Write a valid v1 snapshot and return the files_sha256.csv SHA-256."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    images_dir = snapshot_dir / "images"
    images_dir.mkdir(exist_ok=True)
    snapshot_payload = {
        "schema": "drone3d.reconstruction.snapshot",
        "version": 1,
        "status": "PASS",
        "scene_id": scene_id,
        "frame_count": int(frame_count),
    }
    if capture_config is not None:
        first_stamp = 100 + int(scene_id.rsplit("-w", 1)[1]) * 100
        snapshot_payload.update(
            version=3, camera_name=capture_config.camera_name,
            image_topic=capture_config.image_topic, camera_role=capture_config.camera_role,
            camera_frame_id=capture_config.camera_frame_id,
            first_stamp_sec=first_stamp, first_stamp_nsec=0,
            last_stamp_sec=first_stamp + frame_count - 1, last_stamp_nsec=0,
            has_camera_info=False, has_odometry=True,
        )
    snapshot_bytes = json.dumps(snapshot_payload, sort_keys=True).encode("utf-8")
    (snapshot_dir / "snapshot.json").write_bytes(snapshot_bytes)
    files = [
        (
            "snapshot.json",
            len(snapshot_bytes),
            hashlib.sha256(snapshot_bytes).hexdigest(),
        )
    ]
    frames_lines = ["frame_index,image_path"]
    if capture_config is not None:
        frames_lines[0] += ",stamp_sec,stamp_nsec,pose_valid,pose_delta_ms,odometry_gap_ms"
    for index in range(int(frame_count)):
        relative = f"images/{index}.png"
        import cv2
        import numpy as np
        image = ((np.indices((8, 8)).sum(axis=0) % 2) * 255).astype(np.uint8)
        encoded, png = cv2.imencode(".png", image)
        assert encoded
        image_bytes = png.tobytes()
        image_path = snapshot_dir / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
        frames_lines.append(f"{index},{relative}")
        if capture_config is not None:
            frames_lines[-1] += f",{first_stamp + index},0,true,0,0"
        files.append(
            (relative, len(image_bytes), hashlib.sha256(image_bytes).hexdigest())
        )
    frames_bytes = ("\n".join(frames_lines) + "\n").encode("utf-8")
    (snapshot_dir / "frames.csv").write_bytes(frames_bytes)
    files.append(
        ("frames.csv", len(frames_bytes), hashlib.sha256(frames_bytes).hexdigest())
    )
    files.sort(key=lambda item: item[0])
    manifest_lines = ["path,size,sha256"]
    for path, size, digest in files:
        manifest_lines.append(f"{path},{size},{digest}")
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    (snapshot_dir / "files_sha256.csv").write_bytes(manifest_bytes)
    return hashlib.sha256(manifest_bytes).hexdigest()


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
    def __init__(self, role: CameraRole, snapshot_root: Path, fail=False):
        self.role = role
        self.snapshot_root = snapshot_root
        self.fail = fail
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
            snapshot_dir, self.scene_id, self.max_frames, self.role.capture_config
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
        frame_count = 1
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
        self.config = ContinuousConfig(
            roles=(front_role, down_role),
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
            reconstruct_frame_group_size=8,
            initial_sample_fps=5.0,
            initial_window_frames=1,
            initial_queue_mode=QUEUE_LOSSLESS,
            initial_queue_capacity=8,
            initial_max_retries=0,
            initial_keep_snapshot_on_success=False,
        )
        self.capture_clients = []
        self.backends = []

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

    def test_set_params_rejected_when_session_not_active(self):
        service = self.make_session()
        code, _ = service.set_params(sample_fps=10.0)
        self.assertEqual(code, 4)
        service.select_source("front")
        service.start_session("")
        service.stop_all()
        code, _ = service.set_params(sample_fps=10.0)
        self.assertEqual(code, 4)
        events = [
            event for event in self.journal_events()
            if event.get("event") == "config_changed"
        ]
        self.assertEqual(events, [])

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
        self.assertFalse(any(self.front_root.iterdir()))

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
        self.assertFalse(any(self.front_root.iterdir()))

    def test_hot_tune_freezes_capture_and_pending_configs_and_applies_next_window(self):
        role = self.config.roles[0]
        blocked = BlockingCaptureClient(role, self.front_root)
        clients = [blocked]

        def capture_factory(_role):
            client = FakeCaptureClient(role, self.front_root)
            clients.append(client)
            return client

        blocked_first = [False]

        def mixed_factory(_role):
            if not blocked_first[0]:
                blocked_first[0] = True
                return blocked
            return capture_factory(_role)

        backend = FakeBackend(block_first=True)
        backend.frame_count_by_index = [1, 3, 3, 3]
        service = self.make_session(
            backend=backend, capture_factory=mixed_factory
        )
        code, _ = service.select_source("front")
        self.assertEqual(code, 0)
        code, _ = service.start_session("")
        self.assertEqual(code, 0)
        self.assertTrue(blocked.goal_sent.wait(3.0))
        # 窗口 0 正在采集：此时热调不得改变它的 goal 参数。
        self.assertEqual(blocked.max_frames, 1)
        self.assertEqual(blocked.sample_fps, 5.0)
        code, message = service.set_params(sample_fps=10.0, window_frames=3)
        self.assertEqual(code, 0, message)
        blocked.release()
        # 窗口 0 已封存进 FIFO，配置必须保持旧值；窗口 1 从新配置开始。
        self.wait_until(
            lambda: len(backend.submitted) == 1
            and backend.submitted[0].endswith("-w0")
        )
        self.wait_until(
            lambda: any(
                client.scene_id.endswith("-w1") for client in clients[1:]
            )
        )
        window1 = next(client for client in clients[1:] if client.scene_id.endswith("-w1"))
        self.assertEqual(window1.max_frames, 3)
        self.assertEqual(window1.sample_fps, 10.0)
        self.wait_until(
            lambda: len(
                [
                    event for event in self.journal_events()
                    if event.get("event") == "window_pending"
                    and event.get("window_index") == 1
                ]
            )
            >= 1
        )
        old = EffectiveConfig(
            sample_fps=5.0,
            window_frames=1,
            queue_mode=QUEUE_LOSSLESS,
            queue_capacity=8,
            max_retries=0,
            keep_snapshot_on_success=False,
            save_sensor_depth=False,
            inference_resolution=0,
            worker_timeout_sec=0.0,
        )
        new = EffectiveConfig(
            sample_fps=10.0,
            window_frames=3,
            queue_mode=QUEUE_LOSSLESS,
            queue_capacity=8,
            max_retries=0,
            keep_snapshot_on_success=False,
            save_sensor_depth=False,
            inference_resolution=0,
            worker_timeout_sec=0.0,
        )
        pending = [
            event for event in self.journal_events()
            if event.get("event") == "window_pending"
        ]
        self.assertEqual(pending[0]["window_index"], 0)
        self.assertEqual(pending[0]["config_digest"], old.digest())
        self.assertEqual(pending[1]["window_index"], 1)
        self.assertEqual(pending[1]["config_digest"], new.digest())
        changes = [
            event for event in self.journal_events()
            if event.get("event") == "config_changed"
        ]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["previous_config_digest"], old.digest())
        self.assertEqual(changes[0]["config_digest"], new.digest())
        self.assertEqual(changes[0]["applies_from_window_index"], 1)
        backend.release_first.set()
        backend.jobs[backend.submitted[0]]["state"] = JOB_SUCCEEDED
        self.wait_until(lambda: service.get_status().succeeded_windows >= 2)
        self.assertEqual(service.get_status().failed_windows, 0)
        service.stop_all()

    def test_queue_mode_hot_tune_applies_to_next_enqueued_window(self):
        role = self.config.roles[0]
        clients = []
        blocked2 = BlockingCaptureClient(role, self.front_root)
        calls = [0]

        def capture_factory(_role):
            calls[0] += 1
            if calls[0] == 3:
                clients.append(blocked2)
                return blocked2
            client = FakeCaptureClient(role, self.front_root)
            clients.append(client)
            return client

        backend = FakeBackend(block_first=True)
        backend.frame_count_by_index = [1, 1, 1, 1]
        service = self.make_session(
            backend=backend, capture_factory=capture_factory
        )
        code, _ = service.select_source("front")
        self.assertEqual(code, 0)
        code, _ = service.start_session("")
        self.assertEqual(code, 0)
        self.wait_until(
            lambda: len(backend.submitted) == 1
            and backend.submitted[0].endswith("-w0")
        )
        self.wait_until(
            lambda: len(
                [
                    event for event in self.journal_events()
                    if event.get("event") == "window_pending"
                    and event.get("window_index") == 1
                ]
            )
            >= 1
        )
        self.assertTrue(blocked2.goal_sent.wait(3.0))
        old = EffectiveConfig(
            sample_fps=5.0,
            window_frames=1,
            queue_mode=QUEUE_LOSSLESS,
            queue_capacity=8,
            max_retries=0,
            keep_snapshot_on_success=False,
            save_sensor_depth=False,
            inference_resolution=0,
            worker_timeout_sec=0.0,
        )
        new = EffectiveConfig(
            sample_fps=5.0,
            window_frames=1,
            queue_mode=QUEUE_REALTIME_LATEST,
            queue_capacity=2,
            max_retries=0,
            keep_snapshot_on_success=False,
            save_sensor_depth=False,
            inference_resolution=0,
            worker_timeout_sec=0.0,
        )
        # 窗口 2 已经用旧配置开始采集：热调不得改变它。
        self.assertEqual(blocked2.max_frames, 1)
        code, message = service.set_params(
            queue_mode=QUEUE_REALTIME_LATEST, queue_capacity=2
        )
        self.assertEqual(code, 0, message)
        blocked2.release()
        self.wait_until(lambda: service.get_status().dropped_windows >= 1)
        dropped = [
            event for event in self.journal_events()
            if event.get("event") == "window_dropped"
        ]
        self.assertEqual(dropped[0]["window_index"], 1)
        self.assertEqual(dropped[0]["config_digest"], old.digest())
        self.wait_until(
            lambda: len(
                [
                    event for event in self.journal_events()
                    if event.get("event") == "window_pending"
                    and event.get("window_index") == 3
                ]
            )
            >= 1
        )
        pending_by_index = {
            event["window_index"]: event
            for event in self.journal_events()
            if event.get("event") == "window_pending"
        }
        self.assertEqual(pending_by_index[2]["config_digest"], old.digest())
        self.assertEqual(pending_by_index[3]["config_digest"], new.digest())
        backend.release_first.set()
        backend.jobs[backend.submitted[0]]["state"] = JOB_SUCCEEDED
        self.wait_until(lambda: service.get_status().succeeded_windows >= 2)
        self.assertEqual(service.get_status().failed_windows, 0)
        service.stop_all()

    def test_all_n_frames_and_nonoverlapping_windows_across_hot_tune(self):
        role = self.config.roles[0]
        clients = []
        blocked = BlockingCaptureClient(role, self.front_root)
        calls = [0]

        def capture_factory(_role):
            calls[0] += 1
            if calls[0] == 4:
                clients.append(blocked)
                return blocked
            client = FakeCaptureClient(role, self.front_root)
            clients.append(client)
            return client

        backend = FakeBackend()
        backend.frame_count_by_index = [1, 1, 1, 1, 2, 2, 2, 2]
        service = self.make_session(
            backend=backend, capture_factory=capture_factory
        )
        code, _ = service.select_source("front")
        self.assertEqual(code, 0)
        code, _ = service.start_session("")
        self.assertEqual(code, 0)
        self.wait_until(lambda: service.get_status().succeeded_windows >= 3)
        self.assertTrue(blocked.goal_sent.wait(3.0))
        # 窗口 3 已用旧配置开始采集；热调只影响窗口 4。
        self.assertEqual(blocked.max_frames, 1)
        code, message = service.set_params(window_frames=2)
        self.assertEqual(code, 0, message)
        blocked.release()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 4)
        self.wait_until(
            lambda: len(clients) > 4 and clients[4].scene_id.endswith("-w4")
        )
        window4 = clients[4]
        self.assertEqual(window4.max_frames, 2)
        service.stop_all()
        # stop_all 会立即 latch _stopping：采集线程可能刚创建 client
        # 但尚未 send_goal 就主动压制该窗口。这类 client 从未发起过
        # Capture goal，不属于任何窗口，不能纳入热调断言。
        sent = [client for client in clients if client.goal_sent.is_set()]
        frames = [client.max_frames for client in sent]
        self.assertEqual(frames[:4], [1, 1, 1, 1])
        self.assertTrue(all(item == 2 for item in frames[4:]))
        scenes = [client.scene_id for client in sent]
        self.assertEqual(len(scenes), len(set(scenes)))
        self.assertEqual(
            scenes,
            sorted(scenes, key=lambda text: int(text.rsplit("-w", 1)[1])),
        )
        self.assertTrue(all(client.max_frames >= 1 for client in sent))

    def test_overlap_windows_capture_only_stride_and_keep_exact_frame_ids(self):
        role = self.config.roles[0]
        clients = []
        blocked = BlockingCaptureClient(role, self.front_root)

        def capture_factory(_role):
            client = blocked if len(clients) == 3 else FakeCaptureClient(role, self.front_root)
            clients.append(client)
            return client

        backend = FakeBackend()
        backend.frame_count_by_index = [4, 4, 4]
        service = self.make_session(
            backend=backend,
            capture_factory=capture_factory,
            config=replace(
                self.config,
                initial_window_frames=4,
                initial_overlap_frames=2,
            ),
        )
        self.assertEqual(service.select_source("front")[0], 0)
        self.assertEqual(service.start_session("")[0], 0)
        self.assertTrue(blocked.goal_sent.wait(3.0))
        self.assertEqual(service.get_status().capture_target_frames, 2)
        service.stop_all()

        sent = [client for client in clients if client.goal_sent.is_set()]
        self.assertEqual([client.max_frames for client in sent[:3]], [4, 2, 2])
        windows = service._windows[:3]
        self.assertEqual(windows[0].window_frame_ids, [0, 1, 2, 3])
        self.assertEqual(windows[1].window_frame_ids, [2, 3, 4, 5])
        self.assertEqual(windows[2].window_frame_ids, [4, 5, 6, 7])
        self.assertTrue(all(record.captured_frames == 4 for record in windows))

    def test_g4_assembles_total_16_then_11_new_5_shared(self):
        base = self.shared_a_config()
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
        self.assertNotEqual(service.set_overlap_frames(0)[0], 0)
        service.stop_all()

    def test_g4_diagnostics_count_actual_first_and_overlapping_input(self):
        config = replace(self.shared_a_config(), alignment_backend="global_g4",
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

    # ------------------------------------------------------------------
    # 长窗分组的尾窗、计划耗尽与计划载入失败
    # ------------------------------------------------------------------
    def grouped_plan(self, records):
        path = Path(self.temp.name) / "grouping_plan.json"
        path.write_text(json.dumps(records, ensure_ascii=False))
        return str(path)

    def grouped_config(self, plan_path, **overrides):
        """小尺度长窗分组：窗口 6 新增 + 5 共享，上限 11；尾窗借 6 共享。"""
        roles = tuple(replace(role, capture_config=replace(
            role.capture_config, max_goal_frames=8, max_goal_duration_sec=300.0,
            odometry_association="interpolate",
            snapshot_limits=replace(role.capture_config.snapshot_limits, max_frame_count=11),
        )) for role in self.config.roles)
        values = dict(roles=roles, alignment_backend="global_g4",
                      initial_window_frames=6, initial_overlap_frames=5,
                      reconstruct_frame_group_size=11, initial_inference_resolution=384,
                      initial_keep_snapshot_on_success=True,
                      window_frame_semantics="new_frames", minimum_total_frames=6,
                      window_plan_path=plan_path)
        values.update(overrides)
        return replace(self.config, **values)

    def three_window_plan(self):
        # F000 独立 6；F001 5 共享 + 6 新增；F002 尾窗 6 共享 + 1 新增
        return [
            {"id": "F000", "new_start": 0, "new_end_exclusive": 6,
             "input_start": 0, "input_end_exclusive": 6, "shared_prefix": 0},
            {"id": "F001", "new_start": 6, "new_end_exclusive": 12,
             "input_start": 1, "input_end_exclusive": 12, "shared_prefix": 5},
            {"id": "F002", "new_start": 12, "new_end_exclusive": 13,
             "input_start": 6, "input_end_exclusive": 13, "shared_prefix": 6},
        ]

    def test_grouped_tail_window_assembles_plan_shared_and_new_frames(self):
        config = self.grouped_config(self.grouped_plan(self.three_window_plan()))
        process = mock.MagicMock()
        process.process.poll.return_value = None
        process.status.return_value = {"state": "optimizing"}
        backend = FakeBackend()
        backend.frame_count_by_index = [6, 11, 7]   # 首窗 6；常规 11；尾窗 6 共享＋1 新增
        with mock.patch("drone3d_reconstruction_core.continuous_session.G4BackendProcess",
                        return_value=process):
            service = self.make_session(backend=backend, config=config)
            service.select_source("front")
            self.assertEqual(service.start_session("")[0], 0)
            # captured_frames 在采集后、组装前就会被赋值，因此等组装结果本身。
            self.wait_until(lambda: len(service._windows) == 3
                            and len(service._windows[2].window_frame_ids) > 0, timeout=10.0)
            first, steady, tail = service._windows[:3]
            self.assertEqual(steady.capture_goal_frames, 6)
            self.assertEqual(steady.effective.window_frames, 11)
            self.assertEqual(steady.effective.overlap_frames, 5)
            self.assertEqual(steady.window_frame_ids, list(range(1, 12)))
            # 尾窗：共享必须取计划里的 6，而不是运行时的 5
            self.assertEqual(tail.capture_goal_frames, 1)
            self.assertEqual(tail.effective.overlap_frames, 6)
            self.assertEqual(tail.effective.window_frames, 7)
            self.assertEqual(tail.window_frame_ids, [6, 7, 8, 9, 10, 11, 12])
            self.assertEqual(tail.captured_frames, 7)
            service.stop_all()

    def test_grouped_plan_exhaustion_drains_instead_of_faulting(self):
        config = self.grouped_config(self.grouped_plan(self.three_window_plan()))
        process = mock.MagicMock()
        process.process.poll.return_value = None
        process.status.return_value = {"state": "ready"}
        backend = FakeBackend()
        backend.frame_count_by_index = [6, 11, 7]
        with mock.patch("drone3d_reconstruction_core.continuous_session.G4BackendProcess",
                        return_value=process):
            service = self.make_session(backend=backend, config=config)
            service.select_source("front")
            self.assertEqual(service.start_session("")[0], 0)
            self.wait_until(lambda: service.get_status().succeeded_windows == 3, timeout=15.0)
            self.wait_until(lambda: service.get_status().pending_windows == 0, timeout=10.0)
            status = service.get_status()
            self.assertFalse(service._alignment_fault)
            self.assertNotEqual(status.state, SESSION_ALIGNMENT_PAUSED)
            self.assertNotEqual(status.state, SESSION_PAUSED)
            self.assertEqual(len(service._windows), 3)          # 不创建第 4 个窗口
            self.assertEqual(len(backend.submitted), 3)
            self.assertEqual(status.last_error, "")
            service.stop_all()

    def test_start_session_rejects_missing_plan(self):
        missing = str(Path(self.temp.name) / "does_not_exist.json")
        service = self.make_session(config=self.grouped_config(missing))
        service.select_source("front")
        code, message = service.start_session("")
        self.assertNotEqual(code, 0)
        self.assertIn("plan", message.lower())
        self.assertFalse(service._alignment_fault)
        self.assertIsNone(service._capture_thread)

    def test_start_session_rejects_corrupt_plan(self):
        corrupt = Path(self.temp.name) / "corrupt_plan.json"
        corrupt.write_text("{not valid json")
        service = self.make_session(config=self.grouped_config(str(corrupt)))
        service.select_source("front")
        code, message = service.start_session("")
        self.assertNotEqual(code, 0)
        self.assertIn("plan", message.lower())
        self.assertFalse(service._alignment_fault)

    def test_start_session_rejects_plan_beyond_frame_group_size(self):
        oversized = self.three_window_plan()
        oversized[1] = dict(oversized[1], input_start=-5)   # 触发计划校验失败
        service = self.make_session(config=self.grouped_config(self.grouped_plan(oversized)))
        service.select_source("front")
        code, message = service.start_session("")
        self.assertNotEqual(code, 0)
        self.assertIn("plan", message.lower())

    def test_restart_after_plan_deleted_is_visibly_rejected(self):
        """上一会话读过计划，停止后删除，再次启动必须可见拒绝而不是复用旧缓存。"""
        plan_file = Path(self.temp.name) / "restart_plan.json"
        plan_file.write_text(json.dumps(self.three_window_plan(), ensure_ascii=False))
        service = self.make_session(config=self.grouped_config(str(plan_file)))
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)   # 第一次：读入并缓存计划
        service.stop_all()
        self.assertIsNotNone(service._plan)                 # 缓存里确实有上一份计划
        plan_file.unlink()
        code, message = service.start_session("")
        self.assertNotEqual(code, 0)
        self.assertIn("plan", message.lower())
        self.assertFalse(service._alignment_fault)
        # 被拒绝时不得启动新的采集线程（上一个会话的线程已停止）。
        self.assertFalse(service._capture_thread is not None and service._capture_thread.is_alive())
        service.stop_all()

    def test_restart_after_plan_corrupted_is_visibly_rejected(self):
        plan_file = Path(self.temp.name) / "restart_plan.json"
        plan_file.write_text(json.dumps(self.three_window_plan(), ensure_ascii=False))
        service = self.make_session(config=self.grouped_config(str(plan_file)))
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        service.stop_all()
        plan_file.write_text('{"windows": "not a list"}')
        code, message = service.start_session("")
        self.assertNotEqual(code, 0)
        self.assertIn("plan", message.lower())
        self.assertFalse(service._alignment_fault)
        service.stop_all()

    def full_route_plan_path(self):
        root = Path(os.environ.get("DRONE3D_REPO_ROOT", Path(__file__).resolve().parents[4]))
        return root / "Data/experiments/testflight3_full_route_20260915/inputs/candidate_windows.json"

    @unittest.skipUnless(full_route_plan_path.__qualname__ and
                         (Path(os.environ.get("DRONE3D_REPO_ROOT",
                                              Path(__file__).resolve().parents[4]))
                          / "Data/experiments/testflight3_full_route_20260915/inputs/candidate_windows.json").exists(),
                         "冻结 38 窗登记清单不在本机")
    def test_full_route_3774_frames_38_windows_schedule_end_to_end(self):
        """完整航线 CPU 调度：3774 帧、38 窗、首窗 100、常规 120、尾窗 26 共享＋74 新增。"""
        plan_path = str(self.full_route_plan_path())
        plan = json.loads(Path(plan_path).read_text())
        self.assertEqual(len(plan), 38)
        self.assertEqual(plan[-1]["new_end_exclusive"], 3774)
        roles = tuple(replace(role, capture_config=replace(
            role.capture_config, max_goal_frames=128, max_goal_duration_sec=300.0,
            odometry_association="interpolate",
            snapshot_limits=replace(role.capture_config.snapshot_limits, max_frame_count=128),
        )) for role in self.config.roles)
        config = replace(self.config, roles=roles, alignment_backend="global_g4",
                         initial_window_frames=100, initial_overlap_frames=20,
                         reconstruct_frame_group_size=120, initial_inference_resolution=384,
                         initial_keep_snapshot_on_success=True,
                         window_frame_semantics="new_frames", minimum_total_frames=100,
                         window_plan_path=plan_path,
                         initial_queue_capacity=16)
        process = mock.MagicMock()
        process.process.poll.return_value = None
        process.status.return_value = {"state": "ready"}
        backend = FakeBackend()
        backend.frame_count_by_index = [spec["input_frames"] for spec in plan]
        with mock.patch("drone3d_reconstruction_core.continuous_session.G4BackendProcess",
                        return_value=process):
            service = self.make_session(backend=backend, config=config)
            service.select_source("front")
            self.assertEqual(service.start_session("")[0], 0)
            self.wait_until(lambda: service.get_status().succeeded_windows == 38, timeout=120.0)
            self.wait_until(lambda: service.get_status().pending_windows == 0, timeout=30.0)
            windows = service._windows[:38]
            self.assertEqual(len(service._windows), 38)          # 不创建第 39 个窗口
            self.assertEqual(len(backend.submitted), 38)
            self.assertEqual([w.capture_goal_frames for w in windows], [spec["new_frames"] for spec in plan])
            self.assertEqual([w.captured_frames for w in windows], [spec["input_frames"] for spec in plan])
            self.assertEqual([w.overlap_frame_count for w in windows], [spec["shared_prefix"] for spec in plan])
            self.assertEqual([w.effective.window_frames for w in windows], [spec["input_frames"] for spec in plan])
            self.assertEqual(windows[0].window_frame_ids, list(range(0, 100)))
            self.assertEqual(windows[1].window_frame_ids, list(range(80, 200)))
            self.assertEqual(windows[36].window_frame_ids, list(range(3580, 3700)))
            self.assertEqual(windows[37].window_frame_ids, list(range(3674, 3774)))   # 26 共享＋74 新增
            contributed = []
            for record in windows:
                contributed.extend(record.window_frame_ids[record.overlap_frame_count:])
            self.assertEqual(sorted(contributed), list(range(3774)))   # 3774 帧各贡献一次
            status = service.get_status()
            self.assertFalse(service._alignment_fault)
            self.assertEqual(status.last_error, "")
            self.assertTrue(service.diagnostics_snapshot()["effective"]["grouping_plan_exhausted"])
            service.stop_all()

    def test_three_consecutive_capture_failures_pause_until_resume(self):
        config = replace(self.shared_a_config(), alignment_backend="global_g4",
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
        # One failed attempt is enough for this regression fixture.
        service.set_params(max_retries=0)
        service.start_session("")
        self.wait_until(lambda: service.get_status().state == SESSION_PAUSED)
        self.assertEqual(service.get_status().failed_windows, 3)
        self.assertEqual(len(service._windows), 3)
        self.assertEqual(len(clients), 3)
        self.assertEqual(service.resume()[0], 0)
        self.assertTrue(recovered.goal_sent.wait(3.0))
        service.stop_all()

    def test_g4_background_does_not_block_second_worker_and_stop_closes_it(self):
        from unittest.mock import patch, MagicMock
        base = self.shared_a_config()
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

    def shared_a_config(self):
        roles = tuple(replace(role, capture_config=replace(
            role.capture_config, max_goal_frames=100, pose_gap_max_ms=500.0,
            odometry_association="interpolate", pose_interpolation_wait_ms=500.0,
            snapshot_limits=replace(role.capture_config.snapshot_limits,
                                    max_frame_count=128, max_manifest_entries=1024),
        )) for role in self.config.roles)
        return replace(self.config, roles=roles, alignment_backend="shared_point_sim3_A",
                       initial_window_frames=100, initial_overlap_frames=20,
                       reconstruct_frame_group_size=120, initial_inference_resolution=384)

    def test_shared_a_assembles_100_then_120_without_subsampling(self):
        config = self.shared_a_config()
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
        self.assertEqual([c.max_frames for c in clients], [100] * 4)
        self.assertEqual(service.diagnostics_snapshot()["model_input_frames"], 120)
        self.assertEqual(service.diagnostics_snapshot()["shared_input_frames"], 20)
        self.assertEqual([w.captured_frames for w in service._windows[:3]], [100, 120, 120])
        for record, first, last in zip(service._windows[:3], (0, 80, 180), (100, 200, 300)):
            self.assertEqual(record.window_frame_ids, list(range(first, last)))
        self.assertNotEqual(service.set_params(sample_fps=2.0)[0], 0)
        self.assertNotEqual(service.set_overlap_frames(0)[0], 0)
        service._pause_shared_failure("injected registration failure")
        self.assertEqual(service.get_status().state, "ALIGNMENT_PAUSED")
        self.assertNotEqual(service.resume()[0], 0)
        self.assertNotEqual(service.select_source("down")[0], 0)
        before = len(backend.submitted)
        retained = [Path(unquote(urlsplit(w.snapshot_uri).path)) for w in service._windows[:3]]
        service.stop_all()
        self.assertEqual(len(backend.submitted), before)
        self.assertTrue(all(path.is_dir() for path in retained))
        self.assertEqual(cleanup_stale_snapshots(config), [])
        self.assertTrue(all(path.is_dir() for path in retained))




    def test_g4_requires_retained_input_and_accepts_total_16(self):
        config = replace(self.shared_a_config(), alignment_backend="global_g4",
                         initial_window_frames=16, initial_overlap_frames=5,
                         reconstruct_frame_group_size=16,
                         initial_keep_snapshot_on_success=True)
        self.assertEqual(config.initial_window_frames, 16)
        with self.assertRaisesRegex(ContinuousConfigError, "retained snapshots"):
            replace(config, initial_keep_snapshot_on_success=False)

    def test_shared_a_failed_formal_publication_keeps_previous_reference(self):
        config = self.shared_a_config()
        role = config.roles[0]
        clients = []
        blocked = BlockingCaptureClient(role, self.front_root)
        def factory(_role):
            client = blocked if len(clients) >= 2 else FakeCaptureClient(role, self.front_root)
            clients.append(client)
            return client
        backend = FakeBackend()
        backend.frame_count_by_index = [100, 120]
        calls = []
        def publisher(*args, **kwargs):
            calls.append(args)
            if len(calls) == 2:
                raise RuntimeError("injected publication failure")
            return _fake_result_publisher(*args, **kwargs)
        service = self.make_session(backend=backend, capture_factory=factory, config=config,
                                    formal_publisher=publisher)
        references = []
        def align(record, outcome):
            reference = object()
            references.append(reference)
            record.shared_reference_candidate = reference
            return {"coordinate_state": COORDINATE_ODOM_LOCAL_APPROX,
                    "quality_passed": True, "map_id": service._session_id + "-m0",
                    "scale": 1.0, "rotation": [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                    "translation": [0.,0.,0.], "reference_frame_id": "map"}
        service._align_window = align
        service.select_source("front")
        service.start_session("")
        self.wait_until(lambda: service.get_status().state == "ALIGNMENT_PAUSED")
        self.assertEqual(len(calls), 2)
        self.assertIs(service._shared_reference, references[0])
        self.assertIsNone(service._windows[1].shared_reference_candidate)
        self.assertEqual(len(backend.submitted), 2)

    def test_set_overlap_frames_controls_next_capture_without_registration(self):
        role = self.config.roles[0]
        clients = []

        def capture_factory(_role):
            client = BlockingCaptureClient(role, self.front_root)
            clients.append(client)
            return client

        backend = FakeBackend()
        backend.frame_count_by_index = [4, 4, 4]
        service = self.make_session(
            backend=backend, capture_factory=capture_factory,
            config=replace(self.config, initial_window_frames=4, odom_local_enabled=True),
        )
        self.assertEqual(service.set_overlap_frames(0)[0], 0)
        for invalid in (-1, 4, True, 1.5):
            self.assertEqual(service.set_overlap_frames(invalid)[0], 5)
        self.assertEqual(service.start_session("front")[0], 0)
        self.wait_until(lambda: clients and clients[0].goal_sent.is_set())
        self.assertEqual(service.set_overlap_frames(2)[0], 0)
        self.assertEqual(clients[0].max_frames, 4)
        clients[0].release()
        self.wait_until(lambda: len(clients) >= 2 and clients[1].goal_sent.is_set())
        self.assertEqual(clients[1].max_frames, 2)
        self.assertEqual(service.set_overlap_frames(0)[0], 0)
        clients[1].release()
        self.wait_until(lambda: len(clients) >= 3 and clients[2].goal_sent.is_set())
        self.assertEqual(clients[2].max_frames, 4)
        clients[2].release()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 3)
        self.assertEqual(service.get_status().failed_windows, 0)
        self.assertEqual([w.window_frame_ids for w in service._windows[:3]],
                         [[0, 1, 2, 3], [2, 3, 4, 5], [6, 7, 8, 9]])
        self.assertTrue(all(w.alignment["coordinate_state"] == COORDINATE_MODEL_LOCAL_UNALIGNED
                            for w in service._windows[:3]))
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

    def test_realtime_latest_drops_oldest_waiting_window(self):
        config = ContinuousConfig(
            roles=self.config.roles,
            reconstruct_action=self.config.reconstruct_action,
            journal_root=self.config.journal_root,
            result_root=self.config.result_root,
            service_name=self.config.service_name,
            diagnostics_topic=self.config.diagnostics_topic,
            diagnostics_period_sec=self.config.diagnostics_period_sec,
            min_free_disk_gb=self.config.min_free_disk_gb,
            action_connect_timeout_sec=self.config.action_connect_timeout_sec,
            capture_wait_margin_sec=self.config.capture_wait_margin_sec,
            reconstruct_wait_timeout_sec=self.config.reconstruct_wait_timeout_sec,
            stop_timeout_sec=self.config.stop_timeout_sec,
            reconstruct_frame_group_size=self.config.reconstruct_frame_group_size,
            initial_sample_fps=self.config.initial_sample_fps,
            initial_window_frames=1,
            initial_queue_mode=QUEUE_REALTIME_LATEST,
            initial_queue_capacity=1,
            initial_max_retries=0,
            initial_keep_snapshot_on_success=False,
        )
        backend = FakeBackend(block_first=True)
        w1_blocked = BlockingCaptureClient(self.config.roles[0], self.front_root)
        w2_blocked = BlockingCaptureClient(self.config.roles[0], self.front_root)
        calls = [0]

        def capture_factory(role):
            calls[0] += 1
            if calls[0] == 2:
                return w1_blocked
            if calls[0] == 3:
                return w2_blocked
            client = FakeCaptureClient(role, self.front_root)
            self.capture_clients.append(client)
            return client

        service = ContinuousSession(
            config,
            capture_client_factory=capture_factory,
            backend_factory=lambda: backend,
            disk_free_gb=lambda: 100.0,
            result_gate=_fake_result_gate,
            result_publisher=_fake_result_publisher,
        )
        service.select_source("front")
        service.start_session("")
        # 完全确定性的窗口编排：w0 先进入消费槽；再释放 w1 使其 PENDING；
        # 最后暂停采集循环并释放 w2，只触发一次 realtime_latest 丢弃 w1。
        self.wait_until(
            lambda: len(backend.submitted) == 1
            and backend.submitted[0].endswith("-w0")
        )
        self.assertTrue(w1_blocked.goal_sent.wait(3.0))
        w1_blocked.release()
        self.wait_until(
            lambda: any(
                event.get("event") == "window_pending"
                and event.get("window_index") == 1
                for event in self.journal_events()
            )
        )
        self.assertTrue(w2_blocked.goal_sent.wait(3.0))
        service.pause()
        w2_blocked.release()
        self.wait_until(lambda: service.get_status().dropped_windows == 1)
        dropped = [
            event for event in self.journal_events()
            if event.get("event") == "window_dropped"
        ]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["window_index"], 1)
        dropped_scene = dropped[0]["scene_id"]
        self.assertFalse(
            any(
                path.name.startswith(dropped_scene + "-")
                for path in self.front_root.iterdir()
            )
        )
        self.assertTrue(dropped[0]["config_digest"])
        backend.release_first.set()
        backend.jobs[backend.submitted[0]]["state"] = JOB_SUCCEEDED
        service.stop_all(discard_queue=True)
        self.assertEqual(service.get_status().state, SESSION_STOPPED)
        self.assertFalse(any(self.front_root.iterdir()))

    def test_reconstruct_retry_uses_own_budget_and_queues_at_tail(self):
        config = ContinuousConfig(
            roles=self.config.roles,
            reconstruct_action=self.config.reconstruct_action,
            journal_root=self.config.journal_root,
            result_root=self.config.result_root,
            service_name=self.config.service_name,
            diagnostics_topic=self.config.diagnostics_topic,
            diagnostics_period_sec=self.config.diagnostics_period_sec,
            min_free_disk_gb=self.config.min_free_disk_gb,
            action_connect_timeout_sec=self.config.action_connect_timeout_sec,
            capture_wait_margin_sec=self.config.capture_wait_margin_sec,
            reconstruct_wait_timeout_sec=self.config.reconstruct_wait_timeout_sec,
            stop_timeout_sec=self.config.stop_timeout_sec,
            reconstruct_frame_group_size=self.config.reconstruct_frame_group_size,
            initial_sample_fps=self.config.initial_sample_fps,
            initial_window_frames=1,
            initial_queue_mode=QUEUE_LOSSLESS,
            initial_queue_capacity=8,
            initial_max_retries=1,
            initial_keep_snapshot_on_success=False,
        )
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
        self.assertFalse(any(self.front_root.iterdir()))

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
        with self.assertRaises(Exception):
            ContinuousConfig(
                roles=self.config.roles,
                reconstruct_action=self.config.reconstruct_action,
                journal_root=self.config.journal_root,
                result_root=self.config.result_root,
                service_name=self.config.service_name,
                diagnostics_topic=self.config.diagnostics_topic,
                diagnostics_period_sec=self.config.diagnostics_period_sec,
                min_free_disk_gb=self.config.min_free_disk_gb,
                action_connect_timeout_sec=self.config.action_connect_timeout_sec,
                capture_wait_margin_sec=self.config.capture_wait_margin_sec,
                reconstruct_wait_timeout_sec=self.config.reconstruct_wait_timeout_sec,
                stop_timeout_sec=self.config.stop_timeout_sec,
                reconstruct_frame_group_size=8,
                initial_sample_fps=1.0,
                initial_window_frames=9,
                initial_queue_mode=QUEUE_LOSSLESS,
                initial_queue_capacity=8,
                initial_max_retries=0,
                initial_keep_snapshot_on_success=False,
            )

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
        self.assertEqual(service.start_session("", "frame_pool")[0], 0)
        self.assertEqual(
            service.get_status().reconstruction_mode, "frame_pool"
        )
        service.stop_all()

    def test_incremental_requires_odom_alignment_but_frame_pool_does_not(self):
        service = self.make_session(
            config=replace(self.config, odom_local_enabled=False),
            capture_factory=lambda role: BlockingCaptureClient(role, self.front_root),
        )
        self.assertEqual(service.start_session("front", "per_window")[0], 5)
        self.assertEqual(service.get_status().state, SESSION_IDLE)
        self.assertEqual(service.start_session("front", "frame_pool")[0], 0)
        service.stop_all()

    def test_frame_pool_forces_zero_overlap_and_rejects_changes_or_mode_switch(self):
        service = self.make_session(
            config=replace(self.config, initial_window_frames=4, initial_overlap_frames=2),
            capture_factory=lambda role: BlockingCaptureClient(role, self.front_root),
        )
        self.assertEqual(service.start_session("front", "frame_pool")[0], 0)
        self.assertEqual(service.get_status().effective.overlap_frames, 0)
        for count in (0, 1, 2):
            self.assertEqual(service.set_overlap_frames(count)[0], 5)
            self.assertEqual(service.set_params(overlap_frames=count)[0], 5)
        self.assertEqual(service.start_session("front", "per_window")[0], 3)
        self.assertEqual(service.get_status().reconstruction_mode, "frame_pool")
        self.assertEqual(service.get_status().effective.overlap_frames, 0)
        service.stop_all()
        self.assertEqual(service.start_session("front", "per_window")[0], 0)
        self.assertEqual(service.start_session("front", "frame_pool")[0], 3)
        self.assertEqual(service.get_status().reconstruction_mode, "per_window")
        service.stop_all()

    def test_frame_pool_waits_for_user_target_without_failing_or_subsampling(self):
        clients = []
        def capture_factory(role):
            client = BlockingCaptureClient(role, self.front_root)
            clients.append(client)
            return client
        backend = FakeBackend()
        backend.frame_count_by_index = [3]
        service = self.make_session(
            backend=backend, capture_factory=capture_factory,
            config=replace(self.config, initial_window_frames=3, initial_reconstruction_mode="frame_pool"),
        )
        # First capture contributes only one quality candidate; the next contributes three.
        def score(path):
            return 0.0 if path.stem in ("frame_00000000", "frame_00000001") else 1.0
        with mock.patch("drone3d_reconstruction_core.frame_pool._sharpness_score", new=score):
            self.assertEqual(service.start_session("front")[0], 0)
            self.wait_until(lambda: clients and clients[0].goal_sent.is_set())
            clients[0].release()
            self.wait_until(lambda: service._windows[0].state == "BUFFERED")
            self.assertEqual(backend.submitted, [])
            self.assertEqual(service.get_status().failed_windows, 0)
            self.assertEqual(service.diagnostics_snapshot()["pool_eligible_frames"], 1)
            self.wait_until(lambda: len(clients) > 1 and clients[1].goal_sent.is_set())
            clients[1].release()
            self.wait_until(lambda: service.get_status().succeeded_windows == 1)
            selection = [e for e in self.journal_events() if e.get("event") == "frame_pool_selection"]
            self.assertEqual(len(selection[0]["selected_global_indices"]), 3)
            self.assertEqual(selection[0]["target_frames"], 3)
            self.assertEqual(service._windows[1].alignment["coordinate_state"], COORDINATE_MODEL_LOCAL_UNALIGNED)
            service.stop_all()

    def test_frame_pool_imports_dropped_windows_and_reselects_on_retry(self):
        clients = []
        def capture_factory(role):
            client = BlockingCaptureClient(role, self.front_root)
            clients.append(client)
            return client
        backend = FakeBackend(block_first=True)
        backend.frame_count_by_index = [2, 2, 2]
        service = self.make_session(
            backend=backend, capture_factory=capture_factory,
            config=replace(self.config, initial_window_frames=2, initial_max_retries=1,
                           initial_queue_mode=QUEUE_REALTIME_LATEST, initial_queue_capacity=1),
        )
        self.assertEqual(service.start_session("front", "frame_pool")[0], 0)
        self.wait_until(lambda: clients and clients[0].goal_sent.is_set())
        clients[0].release()
        self.wait_until(lambda: len(backend.submitted) == 1)
        for index in (1, 2):
            self.wait_until(lambda: len(clients) > index and clients[index].goal_sent.is_set())
            clients[index].release()
            self.wait_until(lambda: len(service._frame_pool().entries) >= (index + 1) * 2)
        self.wait_until(lambda: service.get_status().dropped_windows >= 1)
        self.assertEqual(len(service._frame_pool().entries), 6)
        backend.jobs[backend.submitted[0]]["state"] = JOB_FAILED
        backend.release_first.set()
        self.wait_until(lambda: service._windows[0].reconstruct_attempts == 2)
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        selections = [e for e in self.journal_events()
                      if e.get("event") == "frame_pool_selection" and e.get("window_index") == 0]
        self.assertEqual(len(selections), 2)
        self.assertNotEqual(selections[0]["selected_global_indices"], selections[1]["selected_global_indices"])
        self.assertEqual(selections[1]["pool_size"], 6)
        self.assertTrue(all((service._frame_pool().root / e.image_relative_path).exists()
                            for e in service._frame_pool().entries))
        service.stop_all()

    def test_old_pool_epoch_result_does_not_rescore_new_pool_frames(self):
        service = self.make_session()
        service._session_id = "s0123456789abcdef"
        service._map_epoch = 1
        record = SimpleNamespace(pool_epoch=0, pool_revision=0, pool_selected_indices=[0])
        pool = service._frame_pool()
        with mock.patch.object(service, "_predictions_path", return_value=Path("/tmp/old-predictions.npz")), \
             mock.patch.object(pool, "update_from_predictions") as update:
            service._update_pool_from_outcome(record, SimpleNamespace())
        update.assert_not_called()

    def test_frame_pool_mode_submits_synthetic_snapshot(self):
        worker_results = Path(self.temp.name) / "worker-results"
        worker_results.mkdir()
        config = replace(
            self.config,
            initial_reconstruction_mode="frame_pool",
            initial_window_frames=1,
            reconstruct_frame_group_size=1,
            expected_checkpoint_sha256="b" * 64,
            expected_algorithm_revision="algo-test",
            expected_ros_revision="ros-test",
            worker_python="/usr/bin/python3",
            expected_checkpoint_path="/checkpoint/model.pt",
            expected_result_root=str(worker_results),
        )
        backend = FakeBackend(block_first=True)
        service = self.make_session(
            backend=backend,
            config=config,
            formal_gate=None,
            formal_publisher=None,
        )
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_until(lambda: len(backend.submitted) == 1)
        window_id = backend.submitted[0]
        self.assertTrue(window_id.startswith("cs-"))
        self.assertIn("-p0", window_id)
        job = backend.jobs[window_id]
        self.assertIn("cs-", job["snapshot_uri"])
        manifest_path = worker_results / window_id / "result.json"
        manifest_sha = write_valid_result_manifest(
            manifest_path,
            scene_id=window_id,
            manifest_sha256=job["manifest_sha256"],
            captured_frames=1,
            camera_role="front",
            camera_topic="/front/image_raw",
            camera_frame_id="front_camera_optical_frame",
        )
        backend.result_manifest_uri = manifest_path.as_uri()
        backend.result_manifest_sha256 = manifest_sha
        backend.jobs[window_id]["state"] = JOB_SUCCEEDED
        backend.release_first.set()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        session_id = service.get_status().session_id
        formal_dir = self.results / "sessions" / session_id / "window-0000"
        provenance = json.loads(
            (formal_dir / "provenance.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["reconstruction_mode"], "frame_pool")
        self.assertEqual(provenance["pool_revision"], 0)
        self.assertEqual(provenance["pool_epoch"], 0)
        self.assertEqual(provenance["pool_size"], 1)
        self.assertEqual(provenance["pool_selected_indices"], [0])
        self.assertEqual(provenance["confidence_summary"]["selected_count"], 1)
        service.stop_all()

    @mock.patch("drone3d_reconstruction_core.frame_pool._sharpness_score", new=lambda _: 1.0)
    def test_frame_pool_prepare_retry_does_not_import_window_twice(self):
        service = self.make_session()
        service._session_id = "s0123456789abcdef"
        service._reconstruction_mode = "frame_pool"
        role = self.config.roles[0]
        source = self.front_root / "cs-s0123456789abcdef-w0-0123456789abcdef"
        source.mkdir()
        image = source / "frame.png"
        image.write_bytes(b"image")
        snapshot = SimpleNamespace(
            image_paths=(image,),
            frame_stamps=(SimpleNamespace(sec=1, nsec=0),),
            pose_valid=(True,),
            odometry_payloads=(None,),
        )
        record = WindowRecord(
            window_index=0,
            scene_id="cs-s0123456789abcdef-w0",
            role=role,
            effective=service._runtime,
            snapshot_uri=source.as_uri(),
            manifest_sha256="a" * 64,
        )
        pool = service._frame_pool()
        with mock.patch(
            "drone3d_reconstruction_core.continuous_session.validate_snapshot",
            return_value=snapshot,
        ), mock.patch.object(
            pool,
            "build_synthetic_snapshot",
            side_effect=FramePoolError("injected build failure"),
        ):
            with self.assertRaisesRegex(FramePoolError, "injected build failure"):
                service._prepare_pool_reconstruction(record)
        self.assertTrue(record.pool_imported)
        self.assertEqual(record.pool_revision, -1)
        self.assertEqual(len(pool.entries), 1)

        service._prepare_pool_reconstruction(record)
        self.assertEqual(record.pool_revision, 0)
        self.assertEqual(len(pool.entries), 1)

    @mock.patch("drone3d_reconstruction_core.frame_pool._sharpness_score", new=lambda _: 1.0)
    def test_frame_pool_stop_during_prepare_does_not_submit_job(self):
        backend = FakeBackend()
        service = self.make_session(backend=backend)
        service._session_id = "s0123456789abcdef"
        service._reconstruction_mode = "frame_pool"
        role = self.config.roles[0]
        source = self.front_root / "cs-s0123456789abcdef-w0-0123456789abcdef"
        source.mkdir()
        image = source / "frame.png"
        image.write_bytes(b"image")
        snapshot = SimpleNamespace(
            image_paths=(image,),
            frame_stamps=(SimpleNamespace(sec=1, nsec=0),),
            pose_valid=(True,),
            odometry_payloads=(None,),
        )
        record = WindowRecord(
            window_index=0,
            scene_id="cs-s0123456789abcdef-w0",
            role=role,
            effective=service._runtime,
            snapshot_uri=source.as_uri(),
            manifest_sha256="a" * 64,
        )
        pool = service._frame_pool()
        build = pool.build_synthetic_snapshot

        def stop_after_build(*args, **kwargs):
            result = build(*args, **kwargs)
            service._stopping = True
            return result

        with mock.patch(
            "drone3d_reconstruction_core.continuous_session.validate_snapshot",
            return_value=snapshot,
        ), mock.patch.object(
            pool, "build_synthetic_snapshot", side_effect=stop_after_build
        ):
            service._consume_window(record, backend)
        self.assertEqual(backend.submitted, [])
        self.assertEqual(record.state, WINDOW_CANCELLED)

    @mock.patch("drone3d_reconstruction_core.frame_pool._sharpness_score", new=lambda _: 1.0)
    def test_frame_pool_capacity_pauses_new_work(self):
        backend = FakeBackend()
        service = self.make_session(backend=backend)
        service._session_id = "s0123456789abcdef"
        service._reconstruction_mode = "frame_pool"
        service._state = SESSION_RUNNING
        service._pool = FramePool(
            Path(self.temp.name) / "capacity-pool",
            FramePoolConfig(max_frames=1),
        )
        role = self.config.roles[0]
        source = self.front_root / "cs-s0123456789abcdef-w0-0123456789abcdef"
        source.mkdir()
        images = []
        for index in range(2):
            image = source / f"frame-{index}.png"
            image.write_bytes(b"image")
            images.append(image)
        snapshot = SimpleNamespace(
            image_paths=tuple(images),
            frame_stamps=(
                SimpleNamespace(sec=1, nsec=0),
                SimpleNamespace(sec=2, nsec=0),
            ),
            pose_valid=(True, True),
            odometry_payloads=(None, None),
        )
        record = WindowRecord(
            window_index=0,
            scene_id="cs-s0123456789abcdef-w0",
            role=role,
            effective=service._runtime,
            snapshot_uri=source.as_uri(),
            manifest_sha256="a" * 64,
        )
        with mock.patch(
            "drone3d_reconstruction_core.continuous_session.validate_snapshot",
            return_value=snapshot,
        ):
            service._consume_window(record, backend)
        self.assertEqual(service.get_status().state, SESSION_DISK_PAUSED)
        self.assertTrue(service.diagnostics_snapshot()["pool_capacity_blocked"])
        self.assertEqual(backend.submitted, [])

    @mock.patch("drone3d_reconstruction_core.frame_pool._sharpness_score", new=lambda _: 1.0)
    def test_frame_pool_clears_on_odometry_reference_change(self):
        service = self.make_session()
        service._session_id = "s0123456789abcdef"
        service._reconstruction_mode = "frame_pool"
        role = self.config.roles[0]
        records = []
        snapshots = []
        for index, frame_id in enumerate(("odom_a", "odom_b")):
            source = self.front_root / (
                f"cs-s0123456789abcdef-w{index}-{index:016x}"
            )
            source.mkdir()
            image = source / "frame.png"
            image.write_bytes(f"image-{index}".encode("ascii"))
            stamp = {"sec": index + 1, "nsec": 0}
            snapshots.append(
                SimpleNamespace(
                    image_paths=(image,),
                    frame_stamps=(SimpleNamespace(**stamp),),
                    pose_valid=(True,),
                    odometry_payloads=(
                        {
                            "header": {"frame_id": frame_id, "stamp": stamp},
                            "pose": {
                                "position": {"x": float(index), "y": 0.0, "z": 0.0}
                            },
                        },
                    ),
                )
            )
            records.append(
                WindowRecord(
                    window_index=index,
                    scene_id=f"cs-s0123456789abcdef-w{index}",
                    role=role,
                    effective=service._runtime,
                    snapshot_uri=source.as_uri(),
                    manifest_sha256="a" * 64,
                )
            )
        originals = {record.snapshot_uri: snapshot for record, snapshot in zip(records, snapshots)}

        def validate_input(uri, *args, **kwargs):
            if uri in originals:
                return originals[uri]
            return validate_snapshot(uri, *args, **kwargs)

        with mock.patch(
            "drone3d_reconstruction_core.continuous_session.validate_snapshot",
            side_effect=validate_input,
        ):
            service._prepare_pool_reconstruction(records[0])
            service._prepare_pool_reconstruction(records[1])
        pool = service._frame_pool()
        self.assertEqual(service.get_status().map_epoch, 1)
        self.assertEqual([entry.global_index for entry in pool.entries], [1])
        self.assertEqual(records[1].pool_epoch, 1)

    def test_frame_pool_rejects_odometry_jump_within_window(self):
        service = self.make_session()
        payloads = []
        for index, x in enumerate((0.0, 10.0)):
            payloads.append(
                {
                    "header": {
                        "frame_id": "odom",
                        "stamp": {"sec": index + 1, "nsec": 0},
                    },
                    "pose": {"position": {"x": x, "y": 0.0, "z": 0.0}},
                }
            )
        snapshot = SimpleNamespace(
            pose_valid=(True, True),
            odometry_payloads=tuple(payloads),
        )
        with self.assertRaisesRegex(FramePoolError, "jumped within snapshot"):
            service._frame_pool_odom_span(snapshot)

    def test_config_rejects_unknown_reconstruction_mode(self):
        with self.assertRaises(Exception):
            ContinuousConfig(
                roles=self.config.roles,
                reconstruct_action=self.config.reconstruct_action,
                journal_root=self.config.journal_root,
                result_root=self.config.result_root,
                service_name=self.config.service_name,
                diagnostics_topic=self.config.diagnostics_topic,
                diagnostics_period_sec=self.config.diagnostics_period_sec,
                min_free_disk_gb=self.config.min_free_disk_gb,
                action_connect_timeout_sec=self.config.action_connect_timeout_sec,
                capture_wait_margin_sec=self.config.capture_wait_margin_sec,
                reconstruct_wait_timeout_sec=self.config.reconstruct_wait_timeout_sec,
                stop_timeout_sec=self.config.stop_timeout_sec,
                reconstruct_frame_group_size=8,
                initial_sample_fps=1.0,
                initial_window_frames=1,
                initial_queue_mode=QUEUE_LOSSLESS,
                initial_queue_capacity=8,
                initial_max_retries=0,
                initial_keep_snapshot_on_success=False,
                initial_reconstruction_mode="magic",
            )

    def test_cleanup_stale_snapshots_only_removes_scheduler_names(self):
        stale = self.front_root / "cs-s0123456789abcdef-w0-0123456789abcdef"
        legacy = self.front_root / "cs-0123456789abcdef-w1-0123456789abcdef"
        keep = self.front_root / "other-scene-0123456789abcdef"
        stale.mkdir()
        legacy.mkdir()
        keep.mkdir()
        removed = cleanup_stale_snapshots(self.config)
        self.assertIn(str(stale), removed)
        self.assertIn(str(legacy), removed)
        self.assertNotIn(str(keep), removed)
        self.assertTrue(keep.exists())
        self.assertFalse(stale.exists())
        self.assertFalse(legacy.exists())

    def test_align_window_disabled_returns_unaligned(self):
        service = self.make_session(config=replace(self.config, odom_local_enabled=False))
        record = SimpleNamespace(
            snapshot_uri="file:///tmp/snapshot",
            manifest_sha256="a" * 64,
            role=self.config.roles[0],
            effective=service.get_status().effective,
        )
        outcome = SimpleNamespace(
            result_manifest_uri=Path(self.results, "result.json").as_uri()
        )
        payload = service._align_window(record, outcome)
        self.assertEqual(
            payload["coordinate_state"], COORDINATE_MODEL_LOCAL_UNALIGNED
        )
        self.assertIn("disabled", payload["reason"])
        service.stop_all()

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

    def test_align_window_passes_selected_frame_indices_from_result_json(self):
        import drone3d_reconstruction_core.continuous_session as cs_module
        from drone3d_reconstruction_core.odom_local import OdomLocalAlignment

        config = replace(self.config, odom_local_enabled=True)
        service = self.make_session(config=config)
        record = SimpleNamespace(
            snapshot_uri="file:///tmp/snapshot",
            manifest_sha256="a" * 64,
            role=config.roles[0],
            effective=service.get_status().effective,
        )
        result_dir = self.results / "window-passed-indices"
        result_dir.mkdir()
        result_json = result_dir / "result.json"
        result_json.write_text(
            json.dumps(
                {
                    "selected_frames": [
                        {"index": 0},
                        {"index": 3},
                        {"index": 7},
                        {"index": 12},
                    ]
                }
            ),
            encoding="utf-8",
        )
        outcome = SimpleNamespace(
            result_manifest_uri=result_json.as_uri(),
            result_manifest_sha256=hashlib.sha256(b"{}").hexdigest(),
        )
        alignment = OdomLocalAlignment(
            coordinate_state=COORDINATE_ODOM_LOCAL_APPROX,
            reference_frame_id="odom",
            correspondence_count=4,
            scale=1.0,
            rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation=[0.0, 0.0, 0.0],
            residual_rms=0.01,
            reason="",
        )
        for overlap in (0, 2):
            record.effective = replace(record.effective, window_frames=4, overlap_frames=overlap)
            with mock.patch.object(
                cs_module, "validate_snapshot", return_value=SimpleNamespace()
            ), mock.patch.object(
                cs_module, "align_window", return_value=alignment
            ) as align_mock:
                payload = service._align_window(record, outcome)
            align_mock.assert_called_once()
            self.assertEqual(
                align_mock.call_args.kwargs["selected_frame_indices"],
                (0, 3, 7, 12),
            )
            self.assertEqual(payload["coordinate_state"], COORDINATE_ODOM_LOCAL_APPROX)
        service.stop_all()

    def test_align_window_enabled_writes_sidecar_and_map_id(self):
        import drone3d_reconstruction_core.continuous_session as cs_module
        from drone3d_reconstruction_core.odom_local import OdomLocalAlignment

        config = replace(
            self.config,
            odom_local_enabled=True,
            odom_min_correspondences=4,
        )
        service = self.make_session()
        service.stop_all()
        service = ContinuousSession(
            config,
            capture_client_factory=self._capture_factory,
            backend_factory=lambda: FakeBackend(block_first=True),
            disk_free_gb=lambda: 100.0,
            result_gate=_fake_result_gate,
        )
        self.service = service
        service.select_source(config.roles[0].role)
        self.assertEqual(service.start_session("")[0], 0)
        session_id = service.get_status().session_id
        self.assertRegex(session_id, r"^s[0-9a-f]{16}$")
        snapshot_dir = self.front_root / (
            "cs-s0123456789abcdef-w0-" + "a" * 16
        )
        manifest_sha256 = write_minimal_snapshot(
            snapshot_dir, "cs-s0123456789abcdef-w0", 4
        )
        record = SimpleNamespace(
            window_index=0,
            scene_id="cs-s0123456789abcdef-w0",
            snapshot_uri=snapshot_dir.as_uri(),
            manifest_sha256=manifest_sha256,
            role=config.roles[0],
            effective=service.get_status().effective,
            captured_frames=4,
        )
        result_dir = self.results / "window-0"
        result_dir.mkdir()
        outcome = SimpleNamespace(
            result_manifest_uri=(result_dir / "result.json").as_uri(),
            result_manifest_sha256=hashlib.sha256(b"{}").hexdigest(),
        )
        alignment = OdomLocalAlignment(
            coordinate_state=COORDINATE_ODOM_LOCAL_APPROX,
            reference_frame_id="odom",
            correspondence_count=5,
            scale=1.0,
            rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation=[0.0, 0.0, 0.0],
            residual_rms=0.01,
            reason="",
            reference_first_position=(0.0, 0.0, 0.0),
            reference_last_position=(1.0, 1.0, 1.0),
        )
        with mock.patch.object(
            cs_module, "validate_snapshot", return_value=SimpleNamespace()
        ), mock.patch.object(
            cs_module, "align_window", return_value=alignment
        ):
            payload = service._align_window(record, outcome)
        self.assertEqual(
            payload["coordinate_state"], COORDINATE_ODOM_LOCAL_APPROX
        )
        self.assertEqual(payload["correspondence_count"], 5)
        self.assertTrue(payload["unreliable"])
        # 对齐 sidecar 与 map_id 都在门禁通过后的正式记录发布阶段落定。
        self.assertIsNone(service._formalize_window(record, outcome, payload))
        self.assertEqual(payload["map_id"], f"{session_id}-m0")
        formal_dir = self.results / "sessions" / session_id / "window-0000"
        sidecar = formal_dir / "odom_local_alignment.json"
        self.assertTrue(sidecar.is_file())
        self.assertEqual(json.loads(sidecar.read_text())["scale"], 1.0)
        provenance = json.loads(
            (formal_dir / "provenance.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["alignment"]["map_id"], f"{session_id}-m0")
        self.assertFalse((result_dir / "odom_local_alignment.json").exists())
        service.stop_all()

    def test_map_epoch_advances_on_frame_change_and_jump(self):
        from drone3d_reconstruction_core.odom_local import OdomLocalAlignment

        service = self.make_session()

        def alignment(first, last, frame="odom", first_stamp=(10, 0), last_stamp=(10, 5)):
            return OdomLocalAlignment(
                coordinate_state=COORDINATE_ODOM_LOCAL_APPROX,
                reference_frame_id=frame,
                correspondence_count=4,
                scale=1.0,
                rotation=None,
                translation=None,
                residual_rms=0.0,
                reason="",
                reference_first_position=first,
                reference_last_position=last,
                reference_first_stamp=first_stamp,
                reference_last_stamp=last_stamp,
            )

        service._advance_map_epoch_locked(
            alignment(
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                first_stamp=(10, 0), last_stamp=(10, 5),
            )
        )
        self.assertEqual(service.get_status().map_epoch, 0)
        service._advance_map_epoch_locked(
            alignment(
                (1.5, 0.0, 0.0), (2.0, 0.0, 0.0),
                first_stamp=(10, 5), last_stamp=(10, 9),
            )
        )
        self.assertEqual(service.get_status().map_epoch, 0)
        service._advance_map_epoch_locked(
            alignment(
                (20.0, 0.0, 0.0), (21.0, 0.0, 0.0),
                first_stamp=(10, 9), last_stamp=(11, 0),
            )
        )
        self.assertEqual(service.get_status().map_epoch, 1)
        service._advance_map_epoch_locked(
            alignment(
                (21.0, 0.0, 0.0), (22.0, 0.0, 0.0),
                frame="odom_new",
                first_stamp=(11, 0), last_stamp=(11, 5),
            )
        )
        self.assertEqual(service.get_status().map_epoch, 2)
        # 跨窗口 odom stamp 回退也开启新地图。
        service._advance_map_epoch_locked(
            alignment(
                (22.0, 0.0, 0.0),
                (23.0, 0.0, 0.0),
                frame="odom_new",
                first_stamp=(8, 0),
                last_stamp=(9, 0),
            )
        )
        self.assertEqual(service.get_status().map_epoch, 3)
        service.stop_all()

    def test_start_session_resets_map_epoch_and_odom_state(self):
        from drone3d_reconstruction_core.odom_local import OdomLocalAlignment

        service = self.make_session(
            capture_factory=lambda role: BlockingCaptureClient(
                role, self.front_root
            )
        )
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        first_session = service.get_status().session_id
        self.assertNotEqual(first_session, "")

        def alignment(first, last, first_stamp, last_stamp):
            return OdomLocalAlignment(
                coordinate_state=COORDINATE_ODOM_LOCAL_APPROX,
                reference_frame_id="odom",
                correspondence_count=4,
                scale=1.0,
                rotation=None,
                translation=None,
                residual_rms=0.0,
                reason="",
                reference_first_position=first,
                reference_last_position=last,
                reference_first_stamp=first_stamp,
                reference_last_stamp=last_stamp,
            )

        service._advance_map_epoch_locked(
            alignment((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (10, 0), (10, 5))
        )
        self.assertEqual(service.get_status().map_epoch, 0)
        service._advance_map_epoch_locked(
            alignment((20.0, 0.0, 0.0), (21.0, 0.0, 0.0), (11, 0), (12, 0))
        )
        self.assertEqual(service.get_status().map_epoch, 1)
        self.assertEqual(service._last_odom_stamp, (12, 0))
        service.stop_all()
        # 新 session 必须回到全新地图纪元：不能与旧 session 的累计地图拼接。
        self.assertEqual(service.start_session("")[0], 0)
        second_session = service.get_status().session_id
        self.assertNotEqual(second_session, "")
        self.assertNotEqual(second_session, first_session)
        self.assertEqual(service.get_status().map_epoch, 0)
        self.assertIsNone(service._last_odom_stamp)
        self.assertIsNone(service._last_odom_position)
        self.assertIsNone(service._odom_reference_frame_id)
        self.assertEqual(
            service.get_status().last_alignment_state,
            COORDINATE_MODEL_LOCAL_UNALIGNED,
        )
        service.stop_all()

    def test_success_publishes_formal_record_with_config_timeline(self):
        config = replace(
            self.config,
            expected_checkpoint_sha256="b" * 64,
            expected_algorithm_revision="algo-test",
            expected_ros_revision="ros-test",
            worker_python="/usr/bin/python3",
            expected_checkpoint_path="/checkpoint/model.pt",
            expected_result_root=str(Path(self.temp.name) / "worker-results"),
            odom_local_enabled=True,
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
        # 封存 w0 后再热调：时间线随 w0 正式记录一起保存，且只影响下一窗口。
        self.assertEqual(service.set_params(sample_fps=2.0)[0], 0)
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
        self.assertEqual(len(provenance["config_timeline"]), 2)
        self.assertEqual(provenance["config_timeline"][0]["event"], "session_start")
        self.assertEqual(
            provenance["config_timeline"][0]["config_digest"],
            provenance["config_digest"],
        )
        change = provenance["config_timeline"][1]
        self.assertEqual(change["event"], "config_changed")
        self.assertGreaterEqual(change["applies_from_window_index"], 0)
        self.assertNotEqual(
            change["config_digest"], change["previous_config_digest"]
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

    def test_map_epoch_is_not_committed_when_publication_fails(self):
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
            capture_factory=lambda role: BlockingCaptureClient(
                role, self.front_root
            ),
            formal_gate=_fake_result_gate,
            formal_publisher=failing_publisher,
        )
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        snapshot_dir = self.front_root / (
            "cs-s0123456789abcdef-w0-" + "a" * 16
        )
        manifest_sha256 = write_minimal_snapshot(
            snapshot_dir, "cs-s0123456789abcdef-w0", 4
        )
        record = SimpleNamespace(
            window_index=0,
            scene_id="cs-s0123456789abcdef-w0",
            snapshot_uri=snapshot_dir.as_uri(),
            manifest_sha256=manifest_sha256,
            role=self.config.roles[0],
            effective=service.get_status().effective,
            captured_frames=4,
        )
        outcome = SimpleNamespace(
            result_manifest_uri="file:///tmp/result.json",
            result_manifest_sha256=hashlib.sha256(b"{}").hexdigest(),
        )
        alignment = {
            "coordinate_state": COORDINATE_ODOM_LOCAL_APPROX,
            "reference_frame_id": "odom",
            "correspondence_count": 4,
            "scale": 1.0,
            "rotation": None,
            "translation": None,
            "residual_rms": 0.0,
            "reason": "",
            "unreliable": True,
            "map_id": "",
            "reference_first_position": [0.0, 0.0, 0.0],
            "reference_last_position": [1.0, 0.0, 0.0],
            "reference_first_stamp": [10, 0],
            "reference_last_stamp": [10, 5],
        }
        error = service._formalize_window(record, outcome, alignment)
        self.assertIn("formal record publication failed", error)
        self.assertEqual(service.get_status().map_epoch, 0)
        self.assertIsNone(service._last_odom_stamp)
        # 发布成功后重试同一候选纪元：状态只在成功后提交。
        service._result_publisher = _fake_result_publisher
        self.assertIsNone(service._formalize_window(record, outcome, alignment))
        self.assertEqual(service.get_status().map_epoch, 0)
        self.assertEqual(service._last_odom_stamp, [10, 5])
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

    def test_runtime_params_respect_selected_capture_profile(self):
        front = self.config.roles[0]
        front = CameraRole(
            role=front.role,
            capture_action=front.capture_action,
            capture_config=replace(front.capture_config, max_goal_frames=2),
        )
        config = replace(self.config, roles=(front, self.config.roles[1]))
        blocked = BlockingCaptureClient(front, Path(front.capture_config.snapshot_root))
        service = self.make_session(
            config=config, capture_factory=lambda _role: blocked
        )
        self.assertEqual(service.start_session("front")[0], 0)
        self.assertTrue(blocked.goal_sent.wait(3.0))
        for kwargs in (
            {"sample_fps": 0.1},
            {"sample_fps": 11.0},
            {"window_frames": 3},
            {"save_sensor_depth": True},
        ):
            with self.subTest(kwargs=kwargs):
                code, message = service.set_params(**kwargs)
                self.assertEqual(code, 5, message)
                self.assertIn("capture profile", message)
        service.stop_all()

    def test_new_role_does_not_inherit_previous_role_capture_tuning(self):
        down = self.config.roles[1]
        down = CameraRole(
            role=down.role,
            capture_action=down.capture_action,
            capture_config=replace(down.capture_config, max_sample_fps=2.0),
        )
        config = replace(
            self.config,
            roles=(self.config.roles[0], down),
            initial_sample_fps=1.0,
        )
        clients = []

        def capture_factory(role):
            client = BlockingCaptureClient(
                role, Path(role.capture_config.snapshot_root)
            )
            clients.append(client)
            return client

        service = self.make_session(config=config, capture_factory=capture_factory)
        self.assertEqual(service.start_session("front")[0], 0)
        self.wait_until(lambda: clients and clients[0].goal_sent.is_set())
        self.assertEqual(service.set_params(sample_fps=10.0)[0], 0)
        service.stop_all()

        self.assertEqual(service.start_session("down")[0], 0)
        self.wait_until(lambda: len(clients) >= 2 and clients[-1].goal_sent.is_set())
        self.assertEqual(service.get_status().effective.sample_fps, 1.0)
        self.assertEqual(clients[-1].sample_fps, 1.0)
        service.stop_all()

    def test_initial_runtime_must_fit_every_capture_profile(self):
        with self.assertRaisesRegex(ContinuousConfigError, "capture profile"):
            replace(self.config, initial_sample_fps=0.1)

    def test_new_hot_tune_fields_freeze_for_next_window(self):
        role = self.config.roles[0]
        role = CameraRole(
            role=role.role,
            capture_action=role.capture_action,
            capture_config=replace(role.capture_config, depth_topic="/front/depth"),
        )
        config = replace(
            self.config, roles=(role, self.config.roles[1])
        )
        blocked = BlockingCaptureClient(role, self.front_root)
        backend = FakeBackend(block_first=True)
        backend.frame_count_by_index = [1, 1, 1]
        service = self.make_session(
            backend=backend,
            capture_factory=lambda _role: blocked,
            config=config,
        )
        service.select_source("front")
        service.start_session("")
        self.assertTrue(blocked.goal_sent.wait(3.0))
        old = EffectiveConfig(
            sample_fps=5.0,
            window_frames=1,
            queue_mode=QUEUE_LOSSLESS,
            queue_capacity=8,
            max_retries=0,
            keep_snapshot_on_success=False,
            save_sensor_depth=False,
            inference_resolution=0,
            worker_timeout_sec=0.0,
        )
        new = EffectiveConfig(
            sample_fps=5.0,
            window_frames=1,
            queue_mode=QUEUE_LOSSLESS,
            queue_capacity=8,
            max_retries=0,
            keep_snapshot_on_success=False,
            save_sensor_depth=True,
            inference_resolution=256,
            worker_timeout_sec=60.0,
        )
        code, message = service.set_params(
            save_sensor_depth=True,
            inference_resolution=256,
            worker_timeout_sec=60.0,
        )
        self.assertEqual(code, 0, message)
        self.assertEqual(service.get_status().effective.save_sensor_depth, True)
        self.assertEqual(service.get_status().effective.inference_resolution, 256)
        self.assertEqual(service.get_status().effective.worker_timeout_sec, 60.0)
        # 窗口 0 已经以旧配置开始采集；释放后其 config_digest 必须是旧值。
        blocked.release()
        self.wait_until(
            lambda: len(
                [
                    event for event in self.journal_events()
                    if event.get("event") == "window_pending"
                    and event.get("window_index") == 0
                ]
            )
            >= 1
        )
        pending0 = [
            event for event in self.journal_events()
            if event.get("event") == "window_pending"
            and event.get("window_index") == 0
        ][0]
        self.assertEqual(pending0["config_digest"], old.digest())
        # 窗口 1 封存时使用新配置。
        self.wait_until(
            lambda: len(
                [
                    event for event in self.journal_events()
                    if event.get("event") == "window_pending"
                    and event.get("window_index") == 1
                ]
            )
            >= 1
        )
        pending1 = [
            event for event in self.journal_events()
            if event.get("event") == "window_pending"
            and event.get("window_index") == 1
        ][0]
        self.assertEqual(pending1["config_digest"], new.digest())
        self.assertEqual(service.get_status().effective.save_sensor_depth, True)
        service.stop_all()

    def test_backend_receives_runtime_overrides(self):
        role = self.config.roles[0]
        blocked = BlockingCaptureClient(role, self.front_root)
        backend = FakeBackend()
        backend.frame_count_by_index = [1, 1, 1]
        service = self.make_session(
            backend=backend,
            capture_factory=lambda _role: blocked,
        )
        service.select_source("front")
        service.start_session("")
        self.assertTrue(blocked.goal_sent.wait(3.0))
        code, message = service.set_params(
            inference_resolution=384,
            worker_timeout_sec=90.0,
        )
        self.assertEqual(code, 0, message)
        blocked.release()
        self.wait_until(lambda: len(backend.submitted) >= 2)
        # Window 0 froze before the hot-tune; window 1 uses the new values.
        self.assertEqual(
            backend.submit_overrides[0]["inference_resolution"], 0
        )
        self.assertEqual(
            backend.submit_overrides[1]["inference_resolution"], 384
        )
        self.assertEqual(
            backend.submit_overrides[1]["worker_timeout_sec"], 90.0
        )
        service.stop_all()

    def test_set_params_rejects_invalid_new_fields(self):
        service = self.make_session()
        service.select_source("front")
        service.start_session("")
        for kwargs, expected_code in (
            ({"save_sensor_depth": "yes"}, 5),
            ({"inference_resolution": 31}, 5),
            ({"inference_resolution": 4097}, 5),
            ({"worker_timeout_sec": -1.0}, 5),
            ({"overlap_frames": -1}, 5),
            ({"overlap_frames": True}, 5),
            ({"overlap_frames": 1}, 5),
        ):
            with self.subTest(kwargs=kwargs):
                code, _message = service.set_params(**kwargs)
                self.assertEqual(code, expected_code)
        service.stop_all()


    def test_backend_status_exception_cleans_up_and_retries(self):
        backend = StatusRaiseOnceBackend()
        config = replace(self.config, initial_max_retries=1)
        service = self.make_session(backend=backend, config=config)
        service.select_source("front")
        service.start_session("")
        # 只允许第一个窗口进入队列；成功后再检查 active slot 必须为空。
        self.wait_until(lambda: len(backend.submitted) >= 1)
        service.pause()
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
            initial_window_frames=1,
        )
        client = WatchdogFreezeCaptureClient(config.roles[0], self.front_root)
        service = self.make_session(config=config, capture_factory=lambda _role: client)
        service.select_source("front")
        service.start_session("")
        self.assertTrue(client.goal_sent.wait(3.0))
        self.assertEqual(service.pause()[0], 0)
        time.sleep(0.9)
        self.assertEqual(client.cancel_calls, 0)
        self.assertEqual(service.resume()[0], 0)
        client.release()
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        self.assertEqual(client.cancel_calls, 0)
        service.stop_all()

    def test_realtime_latest_retry_overflow_drops_oldest_pending(self):
        config = replace(
            self.config,
            initial_window_frames=1,
            initial_queue_mode=QUEUE_REALTIME_LATEST,
            initial_queue_capacity=1,
            initial_max_retries=1,
        )
        backend = FakeBackend()
        original_submit = backend.submit
        first_w0 = []
        w1_blocked = BlockingCaptureClient(self.config.roles[0], self.front_root)
        w2_blocked = BlockingCaptureClient(self.config.roles[0], self.front_root)
        capture_calls = [0]

        def capture_factory(role):
            capture_calls[0] += 1
            if capture_calls[0] == 2:
                return w1_blocked
            if capture_calls[0] == 3:
                return w2_blocked
            client = FakeCaptureClient(role, self.front_root)
            self.capture_clients.append(client)
            return client

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
            capture_client_factory=capture_factory,
            backend_factory=lambda: backend,
            disk_free_gb=lambda: 100.0,
            result_gate=_fake_result_gate,
            result_publisher=_fake_result_publisher,
        )
        service.select_source("front")
        service.start_session("")
        self.wait_until(
            lambda: len(backend.submitted) == 1
            and backend.submitted[0].endswith("-w0")
        )
        self.assertTrue(first_w0)
        self.assertTrue(w1_blocked.goal_sent.wait(3.0))
        w1_blocked.release()
        self.wait_until(
            lambda: any(
                event.get("event") == "window_pending"
                and event.get("window_index") == 1
                for event in self.journal_events()
            )
        )
        self.assertTrue(w2_blocked.goal_sent.wait(3.0))
        service.pause()
        # w1 已排队，w2 被暂停不会继续入队；此时让 w0 失败，重试必须
        # 因 capacity=1 丢弃最旧的 w1（reason=retry_overflow）。
        backend.jobs[first_w0[0]]["state"] = JOB_FAILED
        self.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        self.wait_until(lambda: service.get_status().dropped_windows >= 1)
        service.pause()
        self.wait_until(lambda: service.get_status().pending_windows == 0)
        with service._lock:
            self.assertLessEqual(len(service._fifo), config.initial_queue_capacity)
        dropped = [
            event for event in self.journal_events()
            if event.get("event") == "window_dropped"
        ]
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["window_index"], 1)
        self.assertEqual(dropped[0]["reason"], "retry_overflow")
        retry_events = [
            event for event in self.journal_events()
            if event.get("event") == "window_retry"
        ]
        self.assertEqual(len(retry_events), 1)
        self.assertEqual(service.get_status().failed_windows, 0)
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



class GroupedSessionFixture(unittest.TestCase):
    """长窗分组用例的共享夹具（本身不含用例）。"""

    # 冻结登记清单是可选的本地材料：缺失时相关用例跳过，而不是失败。
    FROZEN_PLAN = Path(
        os.environ.get("DRONE3D_REPO_ROOT", Path(__file__).resolve().parents[4])
    ) / "Data/experiments/testflight3_full_route_20260915/inputs/candidate_windows.json"

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
            window_frame_semantics="new_frames",
            minimum_total_frames=100,
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


class FramePosePlacementConfigTests(GroupedSessionFixture):
    """g4_placement=frame_pose：只在新放置方式下增加 profile 键，graph 会话逐字不变。"""

    INTRINSICS = {"width": 640, "height": 480, "fx": 604.2, "fy": 604.0, "cx": 319.0, "cy": 247.5}

    def test_graph_profile_is_unchanged(self):
        profile = self.make_session()._backend_profile()
        for key in ("placement", "frame_pose", "camera_intrinsics"):
            self.assertNotIn(key, profile)

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
            self.assertEqual(backend.placement, "frame_pose")
            self.assertEqual(backend.camera_intrinsics["fx"], 604.2)
        finally:
            backend.close()

    def test_invalid_frame_pose_settings_are_rejected(self):
        for overrides, message in (
            (dict(g4_placement="bundle"), "g4_placement"),
            (dict(g4_placement="frame_pose", pose_time_offset_ms=5000.0), "pose_time_offset_ms"),
            (dict(g4_placement="frame_pose", camera_intrinsics={"fx": 600.0}), "camera_intrinsics"),
            (dict(g4_placement="frame_pose", alignment_backend="odom_pose_constrained_sim3",
                  window_frame_semantics="total_frames", initial_window_frames=16,
                  initial_overlap_frames=5), "global_g4"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ContinuousConfigError, message):
                    self.make_grouped_config(None, **overrides)


class LongWindowGroupingTests(GroupedSessionFixture):
    """D01: 长窗分组把 window_frames 解释为新增帧，并按登记清单核对帧数。"""

    def test_steady_window_is_shared_plus_new(self):
        session = self.make_session()
        self.assertTrue(session._grouped_window_semantics())
        self.assertEqual(session._planned_window_shape(0), (100, 0))
        self.assertEqual(session._planned_window_shape(1), (100, 20))
        self.assertEqual(session._grouped_effective(0).window_frames, 100)
        self.assertEqual(session._grouped_effective(1).window_frames, 120)

    @unittest.skipUnless(GroupedSessionFixture.FROZEN_PLAN.exists(),
                         "frozen candidate list is not present")
    def test_registered_plan_drives_every_window_including_the_tail(self):
        session = self.make_session(window_plan_path=str(self.FROZEN_PLAN))
        self.assertEqual(session._planned_window_shape(0), (100, 0))
        self.assertEqual(session._planned_window_shape(1), (100, 20))
        self.assertEqual(session._planned_window_shape(37), (74, 26))
        self.assertEqual(session._grouped_effective(37).window_frames, 100)
        self.assertEqual(len(session._window_plan()), 38)
        with self.assertRaises(FramePoolError):
            session._planned_window_shape(38)

    def test_plan_beyond_frame_group_size_is_rejected_not_subsampled(self):
        oversized = Path(self.temp.name) / "oversized_plan.json"
        oversized.write_text(json.dumps([
            {"id": "F000", "new_start": 0, "new_end_exclusive": 200,
             "input_start": 0, "input_end_exclusive": 200, "shared_prefix": 0},
            {"id": "F001", "new_start": 200, "new_end_exclusive": 300,
             "input_start": 180, "input_end_exclusive": 300, "shared_prefix": 20},
        ]))
        session = self.make_session(window_plan_path=str(oversized))
        with self.assertRaises(FramePoolError):
            session._window_plan()

    def test_new_frames_semantics_needs_headroom_and_the_global_backend(self):
        with self.assertRaises(ContinuousConfigError):
            self.make_grouped_config(reconstruct_frame_group_size=100)
        with self.assertRaises(ContinuousConfigError):
            self.make_grouped_config(alignment_backend="odom_pose_constrained_sim3")

    def test_total_frames_semantics_is_unchanged(self):
        session = self.make_session(window_frame_semantics="total_frames",
                                    reconstruct_frame_group_size=16,
                                    initial_window_frames=16,
                                    initial_overlap_frames=5,
                                    minimum_total_frames=100)
        self.assertFalse(session._grouped_window_semantics())
        # 未启用分组时窗口语义仍是“总帧数”，不追加共享帧。
        self.assertEqual(session._runtime.window_frames, 16)


class DiagnosticInputFrameDisplayTests(GroupedSessionFixture):
    """D01 显示口径：每批新增/共享/模型输入必须按该窗口实际生效的计划给出。

    长窗分组下 window_frames 是模型输入总数，若诊断仍用 effective 反推，
    首窗/稳定窗/尾窗会显示成 100/20/120、80/20/100 这类错误组合。
    非分组后端（含既有 G4 total_frames 语义与 A 模式）必须保持原样。
    """

    def tail_plan(self):
        """三窗登记清单：首窗 100 新增、稳定窗 100+20、尾窗 74+26。"""
        path = Path(self.temp.name) / "display_plan.json"
        path.write_text(json.dumps([
            {"id": "D000", "new_start": 0, "new_end_exclusive": 100,
             "input_start": 0, "input_end_exclusive": 100, "shared_prefix": 0},
            {"id": "D001", "new_start": 100, "new_end_exclusive": 200,
             "input_start": 80, "input_end_exclusive": 200, "shared_prefix": 20},
            {"id": "D002", "new_start": 200, "new_end_exclusive": 274,
             "input_start": 174, "input_end_exclusive": 274, "shared_prefix": 26},
        ]))
        return str(path)

    def display_record(self, session, index):
        """按生产路径构造该窗口的记录（计划值 + 该窗 effective）。"""
        planned_new, planned_shared = session._planned_window_shape(index)
        record = WindowRecord(
            index, "display-%d" % index, session._config.roles[0],
            session._grouped_effective(index),
            planned_new_frames=planned_new,
            planned_shared_frames=planned_shared,
            overlap_frame_count=planned_shared,
        )
        record.state = WINDOW_CAPTURING
        return record

    def display_numbers(self, session, index):
        """把记录挂成交付中的窗口（index=None 表示还没有任何窗口）。"""
        session._windows = [] if index is None else [self.display_record(session, index)]
        snapshot = session.diagnostics_snapshot()
        return (snapshot["new_input_frames"], snapshot["shared_input_frames"],
                snapshot["model_input_frames"])

    def test_first_steady_and_tail_windows_report_planned_numbers(self):
        session = self.make_session(window_plan_path=self.tail_plan())
        self.assertEqual(self.display_numbers(session, 0), (100, 0, 100))
        self.assertEqual(self.display_numbers(session, 1), (100, 20, 120))
        self.assertEqual(self.display_numbers(session, 2), (74, 26, 100))

    def test_derived_grouping_without_a_plan_keeps_first_window_unshared(self):
        session = self.make_session()
        self.assertIsNone(session._window_plan())
        self.assertEqual(self.display_numbers(session, 0), (100, 0, 100))
        self.assertEqual(self.display_numbers(session, 1), (100, 20, 120))

    def test_grouped_session_without_a_window_shows_the_first_window_shape(self):
        session = self.make_session()
        self.assertEqual(self.display_numbers(session, None), (100, 0, 100))

    def test_g4_total_frames_semantics_is_unchanged(self):
        session = self.make_session(window_frame_semantics="total_frames",
                                    reconstruct_frame_group_size=16,
                                    initial_window_frames=16,
                                    initial_overlap_frames=5,
                                    minimum_total_frames=100)
        record = WindowRecord(1, "legacy", session._config.roles[0], session._runtime,
                              overlap_frame_count=5)
        record.state = WINDOW_CAPTURING
        session._windows = [record]
        snapshot = session.diagnostics_snapshot()
        self.assertEqual((snapshot["new_input_frames"], snapshot["shared_input_frames"],
                          snapshot["model_input_frames"]), (11, 5, 16))

    def test_non_grouped_backend_keeps_model_input_at_effective_window_frames(self):
        session = self.make_session(alignment_backend="odom_pose_constrained_sim3",
                                    window_frame_semantics="total_frames")
        self.assertFalse(session._uses_g4())
        self.assertFalse(session._uses_shared_A())
        # 非分组后端不显示“新增”，但模型输入/共享必须沿用 effective 数值。
        self.assertEqual(session._runtime.window_frames, 100)
        self.assertEqual(session._runtime.overlap_frames, 20)
        self.assertEqual(self.display_numbers(session, None), (None, 20, 100))

    def test_shared_a_backend_reports_one_hundred_plus_shared(self):
        session = self.make_session(
            alignment_backend="shared_point_sim3_A",
            window_frame_semantics="total_frames",
            capture_overrides={"pose_interpolation_wait_ms": 500.0,
                               "pose_gap_max_ms": 500.0, "max_frame_count": 128},
        )
        record = WindowRecord(1, "shared-a", session._config.roles[0], session._runtime,
                              overlap_frame_count=20)
        record.state = WINDOW_CAPTURING
        session._windows = [record]
        snapshot = session.diagnostics_snapshot()
        self.assertEqual((snapshot["new_input_frames"], snapshot["shared_input_frames"],
                          snapshot["model_input_frames"]), (100, 20, 120))


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
            capture_config=make_capture_config(root, "front"),
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
            reconstruct_frame_group_size=8,
            initial_sample_fps=5.0,
            initial_window_frames=1,
            initial_queue_mode=QUEUE_LOSSLESS,
            initial_queue_capacity=8,
            initial_max_retries=0,
            initial_keep_snapshot_on_success=False,
        )
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
        self.wait_until(lambda: service.get_status().state == SESSION_MEMORY_PAUSED)
        record = service._memory_failed_window
        self.assertIsNotNone(record)
        return service, record

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
        self.wait_until(lambda: service.get_status().succeeded_windows == 1)
        status = service.get_status()
        # 同一个窗口、第二次提交；门禁本身没有任何放宽（仍是同一 backend 路径）。
        self.assertEqual(len(backend.submitted), 2)
        self.assertEqual(len(set(backend.submitted)), 1)
        self.assertEqual(record.state, WINDOW_SUCCEEDED)
        self.assertEqual(record.manual_retries, 1)
        self.assertTrue(any("HIGH_SWAP_USED" in item for item in record.guard_failures))
        self.assertEqual(status.failed_windows, 0)
        self.assertEqual(service._failed, 0)
        self.assertEqual(service._attempt_failures, 1)
        self.assertEqual([index for _, index in self.formal], [record.window_index])
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
        self.wait_until(lambda: service.get_status().state == SESSION_MEMORY_PAUSED)
        record = service._memory_failed_window
        pending = [item.window_index for item in service._fifo]
        # 采集已经跑在消费前面：暂停时队列里已有更晚的窗口。
        self.assertGreaterEqual(len(pending), 2)
        self.assertEqual(pending, sorted(pending))
        self.assertTrue(all(index > record.window_index for index in pending))
        next_window = service._next_window
        expected = [record.window_index] + pending
        code, message = service.resume()
        self.assertEqual(code, 0, message)
        self.wait_until(lambda: len(self.formal) >= len(expected))
        # 重试窗插在队首：正式结果前 len(expected) 条仍是完整、升序、无重复
        # 的窗口序列（采集线程随后还会继续后面的窗口）。
        self.assertEqual([index for _, index in self.formal][:len(expected)],
                         expected)
        self.assertEqual(service._failed, 0)
        self.assertEqual(service._dropped, 0)
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
        self.wait_until(lambda: service.get_status().succeeded_windows == 1)
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

    def test_realtime_latest_manual_retry_is_refused_without_mutation(self):
        """不能保证保留重试窗的队列模式不得启动人工恢复。"""
        config = replace(self.config, initial_queue_mode=QUEUE_REALTIME_LATEST,
                         initial_queue_capacity=2)
        backend = GuardDeniedAfterDelayBackend()
        service = self.make_session(backend, config)
        service.select_source("front")
        self.assertEqual(service.start_session("")[0], 0)
        self.wait_until(lambda: service.get_status().state == SESSION_MEMORY_PAUSED)
        self.assertEqual(service._join_threads(), [])
        record = service._memory_failed_window
        before = (list(service._fifo), service._fifo_bytes, service._dropped,
                  service._failed, len(backend.submitted))
        code, message = service.resume()
        self.assertEqual(code, 5, message)
        self.assertIn("lossless_fifo", message)
        self.assertEqual(service.get_status().state, SESSION_MEMORY_PAUSED)
        self.assertEqual((list(service._fifo), service._fifo_bytes, service._dropped,
                          service._failed, len(backend.submitted)), before)
        self.assertEqual(record.manual_retries, 0)
        self.assertTrue(record.done.is_set())
        self.assertTrue(self.snapshot_path(record).is_dir())
        self.assertFalse(service._capture_thread.is_alive())
        self.assertFalse(service._consume_thread.is_alive())

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
        self.wait_until(lambda: service.get_status().succeeded_windows == 1)
        # 只入队一次、只提交一次重试：不重复正式结果。
        self.assertEqual(len(backend.submitted), 2)
        self.assertEqual([index for _, index in self.formal], [record.window_index])
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
        backend.frame_count_by_index = [1, 1, 1]
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
        for exhausted in (False, True):
            with self.subTest(plan_exhausted=exhausted):
                service._plan_exhausted = exhausted
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
        backend.frame_count_by_index = [1, 1]
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
        self.wait_until(lambda: service.get_status().succeeded_windows == 1)

    def test_resume_refuses_when_a_cancelled_planned_window_would_be_skipped(self):
        """计划已耗尽且仍有被取消的索引：明确拒绝，不宣布恢复成功。"""
        service, record, cancelled, backend = (
            self.pause_with_a_cancelled_inflight_capture()
        )
        # 人为制造不安全状态：清单已耗尽 => 恢复时不会再开采集线程，
        # 被取消的索引就永远补不回来。
        service._plan_exhausted = True
        code, message = service.resume()
        self.assertEqual(code, 3)
        self.assertIn("cancelled", message)
        self.assertEqual(service.get_status().state, SESSION_MEMORY_PAUSED)
        # 现场保留：被拒窗口的输入、失败历史与提交次数都不变。
        self.assertTrue(self.snapshot_path(record).is_dir())
        self.assertIs(service._memory_failed_window, record)
        self.assertEqual(len(backend.submitted), 1)
        self.assertEqual(self.formal, [])

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
        self.wait_until(lambda: service.get_status().succeeded_windows == 1)
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


class GroupedManualRecoveryTests(unittest.TestCase):
    """登记清单最后一窗被资源门拒绝：人工恢复 + 计划耗尽（只排空、不再采集）。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.journal = root / "journal"
        self.journal.mkdir()
        self.results = root / "results"
        self.results.mkdir()
        base = make_capture_config(root, "front")
        capture = replace(
            base,
            odometry_association="interpolate",
            max_goal_frames=8,
            max_goal_duration_sec=300.0,
            snapshot_limits=replace(base.snapshot_limits, max_frame_count=11),
        )
        role = CameraRole(
            role="front",
            capture_action="/front/capture_segment",
            capture_config=capture,
        )
        self.role = role
        self.front_root = Path(capture.snapshot_root)
        # 两窗清单：首窗 6 新增；尾窗 5 共享 + 6 新增（组上限 11）。
        plan = root / "recovery_plan.json"
        plan.write_text(json.dumps([
            {"id": "R000", "new_start": 0, "new_end_exclusive": 6,
             "input_start": 0, "input_end_exclusive": 6, "shared_prefix": 0},
            {"id": "R001", "new_start": 6, "new_end_exclusive": 12,
             "input_start": 1, "input_end_exclusive": 12, "shared_prefix": 5},
        ]))
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
            reconstruct_frame_group_size=11,
            initial_sample_fps=5.0,
            initial_window_frames=6,
            initial_overlap_frames=5,
            initial_queue_mode=QUEUE_LOSSLESS,
            initial_queue_capacity=4,
            initial_max_retries=0,
            initial_keep_snapshot_on_success=True,
            initial_inference_resolution=384,
            alignment_backend="global_g4",
            window_frame_semantics="new_frames",
            minimum_total_frames=6,
            window_plan_path=str(plan),
        )
        self.clients = []
        self.formal = []
        self.session = None

    def tearDown(self):
        if self.session is not None:
            try:
                self.session.shutdown(2.0)
            except BaseException:
                pass

    def wait_until(self, predicate, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        raise AssertionError("condition not reached")

    def test_manual_retry_of_the_last_planned_window_then_plan_exhaustion(self):
        backend = ResourceGuardBackend(deny_indices=(1,))
        backend.frame_count_by_index = [6, 11]
        process = mock.MagicMock()
        process.process.poll.return_value = None
        process.status.return_value = {"state": "processing", "published_windows": []}

        def factory(role):
            client = FakeCaptureClient(role, self.front_root)
            self.clients.append(client)
            return client

        def publisher(records_root, session_id, window_index, *args, **kwargs):
            self.formal.append(window_index)
            return _fake_result_publisher(records_root, session_id, window_index,
                                          *args, **kwargs)

        with mock.patch(
            "drone3d_reconstruction_core.continuous_session.G4BackendProcess",
            return_value=process,
        ):
            session = ContinuousSession(
                self.config,
                capture_client_factory=factory,
                backend_factory=lambda: backend,
                disk_free_gb=lambda: 100.0,
                result_gate=_fake_result_gate,
                result_publisher=publisher,
            )
            self.session = session
            session.select_source("front")
            self.assertEqual(session.start_session("")[0], 0)
            self.wait_until(
                lambda: session.get_status().state == SESSION_MEMORY_PAUSED
            )
            record = session._memory_failed_window
            self.assertEqual(record.window_index, 1)
            self.assertEqual(record.planned_shared_frames, 5)
            self.assertEqual(record.effective.window_frames, 11)
            self.assertEqual(self.formal, [0])
            code, message = session.resume()
            self.assertEqual(code, 0, message)
            self.wait_until(lambda: session.get_status().succeeded_windows == 2)
            self.wait_until(
                lambda: session.diagnostics_snapshot()["effective"][
                    "grouping_plan_exhausted"
                ]
            )
            self.assertEqual(self.formal, [0, 1])
            self.assertEqual(record.manual_retries, 1)
            self.assertEqual(session._failed, 0)
            self.assertEqual(session._attempt_failures, 1)
            self.assertEqual(session._manual_retries, 1)
            # 计划跑完只排空：不再采集任何新窗口，重试窗仍然被消费。
            self.assertEqual(session._next_window, 2)
            self.assertEqual(len(session._windows), 2)
            goals = [client.scene_id for client in self.clients
                     if client.goal_sent.is_set()]
            self.assertEqual(len(goals), 2)
            self.assertEqual(len(set(goals)), 2)
            session.stop_all()


if __name__ == "__main__":
    unittest.main()
