"""ACC-005 machine-checkable fault matrix for the continuous reconstruction system.

This module is intentionally self-contained: it adapts the fake capture
client / fake backend / session-builder patterns from
``test_continuous_session.py`` but never imports that test module, because
catkin cross-test imports break the gate.  It imports production modules only
and orchestrates the real ``ContinuousSession`` scheduler end-to-end for every
CPU-runnable row of the matrix.

The single machine-readable matrix lives in :data:`ACC_FAULT_MATRIX`; the
runner :func:`run_acc005_fault_matrix` executes every CPU-runnable row and
collects a JSON-serializable evidence report.
"""

from __future__ import annotations

from dataclasses import replace
import sys
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.request
from urllib.parse import unquote, urlsplit

# The shared v4 snapshot fixture lives next to this file; catkin's nose run does not add this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from snapshot_v4 import write_v4_snapshot  # noqa: E402
from drone3d_reconstruction_core.capture import (
    CaptureConfig,
    CaptureGoalParameters,
    CaptureStreamError,
    StampSampler,
)
from drone3d_reconstruction_core.continuous_session import (
    ContinuousConfig,
    ContinuousSession,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    JobStatus,
    QUEUE_LOSSLESS,
    ReconstructionBackend,
    SESSION_DISK_PAUSED,
    SESSION_RUNNING,
    SESSION_STOPPED,
)
from drone3d_reconstruction_core.continuous_web import (
    WebDashboardConfig,
    WebDashboardCore,
    WebRoleInfo,
)
from drone3d_reconstruction_core.formal_result import (
    FORMAL_RECORD_SCHEMA,
    FORMAL_RECORD_SCHEMA_VERSION,
    publish_formal_record,
    scan_session_windows,
    verify_result_manifest,
)
from drone3d_reconstruction_core.action_types import (
    ACTION_CANCELLED,
    ACTION_FAILED,
    ACTION_SUCCEEDED,
    CameraRole,
    CaptureActionOutcome,
    ReconstructActionOutcome,
)
from drone3d_reconstruction_core.snapshot_validator import (
    FrameStamp,
    SnapshotSafetyLimits,
)


# ---------------------------------------------------------------------------
# Machine-readable matrix definition
# ---------------------------------------------------------------------------

ACC_FAULT_MATRIX = {
    "source_stale_recover": {
        "id": "ACC-005",
        "row": "source_stale_recover",
        "title": "source watchdog: short RGB outage freezes the window deadline and recovers into the same window",
        "cpu_runnable": True,
    },
    "source_timeout_abort_window": {
        "id": "ACC-005",
        "row": "source_timeout_abort_window",
        "title": "source watchdog: long gap / timeout aborts the current window and the scheduler continues with the next window",
        "cpu_runnable": True,
    },
    "stamp_zero_duplicate_backward": {
        "id": "ACC-005",
        "row": "stamp_zero_duplicate_backward",
        "title": "StampSampler.classify rejects zero/duplicate stamps and raises on backward stamps with exact counters",
        "cpu_runnable": True,
    },
    "worker_submit_failure": {
        "id": "ACC-005",
        "row": "worker_submit_failure",
        "title": "ReconstructionBackend.submit raising is handled as an ordinary window failure without wedging the FIFO",
        "cpu_runnable": True,
    },
    "worker_nonzero_exit": {
        "id": "ACC-005",
        "row": "worker_nonzero_exit",
        "title": "worker nonzero exit marks the window FAILED and the next window still succeeds",
        "cpu_runnable": True,
    },
    "worker_watchdog_timeout_cancel": {
        "id": "ACC-005",
        "row": "worker_watchdog_timeout_cancel",
        "title": "reconstruction watchdog fires, cancels the hung backend and the FIFO keeps consuming",
        "cpu_runnable": True,
    },
    "disk_redline_pause_resume": {
        "id": "ACC-005",
        "row": "disk_redline_pause_resume",
        "title": "disk redline pauses new capture and automatically resumes after space is freed",
        "cpu_runnable": True,
    },
    "lossless_fifo_order": {
        "id": "ACC-005",
        "row": "lossless_fifo_order",
        "title": "lossless_fifo consumes windows strictly in enqueue order with no silent drops",
        "cpu_runnable": True,
    },
    "failure_then_continue": {
        "id": "ACC-005",
        "row": "failure_then_continue",
        "title": "an ordinary failed window keeps its journal diagnosis, removes its snapshot, and does not block later windows",
        "cpu_runnable": True,
    },
    "formal_gate_rejects_corrupt_result": {
        "id": "ACC-005",
        "row": "formal_gate_rejects_corrupt_result",
        "title": "the real verify_result_manifest gate rejects a validator-corrupted result and the next valid window publishes a formal record",
        "cpu_runnable": True,
    },
    "stop_all_cleanup_no_orphan": {
        "id": "ACC-005",
        "row": "stop_all_cleanup_no_orphan",
        "title": "stop_all cancels the running worker, removes pending snapshots and leaves no staging or formal records",
        "cpu_runnable": True,
    },
    "fake_exclusion": {
        "id": "ACC-005",
        "row": "fake_exclusion",
        "title": "scan_session_windows ignores fake/foreign directories inside result_root/sessions",
        "cpu_runnable": True,
    },
    "web_diagnostics_consistency": {
        "id": "ACC-005",
        "row": "web_diagnostics_consistency",
        "title": "Web /api/state exposes the same diagnostics as get_status and a POST command is forwarded exactly once",
        "cpu_runnable": True,
    },
    "real_cuda_continuous_cancel": {
        "id": "ACC-005",
        "row": "real_cuda_continuous_cancel",
        "title": "continuous-scheduler path cancels a real CUDA worker with terminal state and no orphan/staging",
        "cpu_runnable": False,
    },
}


def _evidence(assertion: str, journal_events=None, **extra) -> dict:
    """Build one JSON-safe evidence dict for a matrix row."""
    excerpt = journal_events or []
    if len(excerpt) > 8:
        excerpt = excerpt[-8:]
    evidence = {
        "assertion": assertion,
        "timestamp": time.time(),
        "journal_event_excerpt": excerpt,
    }
    evidence.update(extra)
    return evidence


# ---------------------------------------------------------------------------
# Shared fakes (adapted from test_continuous_session.py; no cross-test import)
# ---------------------------------------------------------------------------

CHECKPOINT_SHA = "a" * 64
CHECKPOINT_PATH = "/checkpoint/model.pt"
ALGO_REV = "algo-test"
ROS_REV = "ros-test"
COORDINATE_MODEL_LOCAL_UNALIGNED = "MODEL_LOCAL_UNALIGNED"


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


def write_result_manifest(
    path: Path,
    *,
    scene_id: str,
    snapshot_sha: str,
    frame_count: int,
    camera_role: str,
    camera_topic: str,
    camera_frame_id: str,
    checkpoint_path: str = CHECKPOINT_PATH,
    checkpoint_sha: str = CHECKPOINT_SHA,
    algorithm_revision: str = ALGO_REV,
    ros_revision: str = ROS_REV,
) -> str:
    """Write a schema-v1 worker result manifest and return its SHA-256."""
    payload = {
        "schema": "drone3d.reconstruction.result",
        "schema_version": 1,
        "scene_id": scene_id,
        "snapshot": {"sha256": snapshot_sha, "frame_count": frame_count},
        "actual_frame_count": frame_count,
        "camera_role": camera_role,
        "camera_topic": camera_topic,
        "camera_frame_id": camera_frame_id,
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha,
        },
        "algorithm_revision": algorithm_revision,
        "ros_revision": ros_revision,
        "coordinate_state": COORDINATE_MODEL_LOCAL_UNALIGNED,
        "worker_exit": {"returncode": 0, "cause": "ok"},
        "stage_timing": {
            "wall_time_sec": 1.5,
            "worker_wall_time_sec": 1.5,
        },
        "artifacts": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.npz", "manifest.txt", "run.log"):
        artifact_bytes = (name + " artifact bytes for fault matrix tests\n").encode()
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


def write_minimal_snapshot(snapshot_dir: Path, scene_id: str, frame_count: int, capture_config) -> str:
    """Write a valid v4 snapshot as the capture node does and return its manifest SHA-256.

    Stamps increase across windows (window w starts at 100 + 100 w seconds);
    rolling windows rebuild their shared frames from these stamps.
    """
    return write_v4_snapshot(
        snapshot_dir, scene_id, frame_count,
        first_stamp_sec=100 + int(scene_id.rsplit("-w", 1)[1]) * 100,
        camera_name=capture_config.camera_name, image_topic=capture_config.image_topic,
        camera_role=capture_config.camera_role, camera_frame_id=capture_config.camera_frame_id, image_bytes=lambda index: f"fake-image-{index}\n".encode("utf-8"),
    )


class FakeCaptureClient:
    """Deterministic continuous-capture client writing a real snapshot dir."""

    def __init__(self, role: CameraRole, snapshot_root: Path):
        self.role = role
        self.snapshot_root = Path(snapshot_root)
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
            feedback_cb(SimpleNamespace(frame_count=max_frames))
        self._ready.set()

    def wait_for_result(self, timeout_sec):
        return self._ready.wait(timeout_sec)

    def cancel_goal(self):
        self._ready.set()

    def outcome(self):
        snapshot_dir = self.snapshot_root / (self.scene_id + "-" + "a" * 16)
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
    """Capture client that blocks until release()/cancel_goal() is called."""

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


class FailingCaptureClient(FakeCaptureClient):
    """Capture client whose Action terminal state is a source failure."""

    def __init__(self, role, snapshot_root, message):
        super().__init__(role, snapshot_root)
        self.failure_message = message

    def outcome(self):
        return CaptureActionOutcome(
            action_state=ACTION_FAILED,
            error_code=2,
            message=self.failure_message,
            scene_id=self.scene_id,
            camera_name=self.role.capture_config.camera_name,
            image_topic=self.role.capture_config.image_topic,
            snapshot_uri="",
            manifest_uri="",
            manifest_sha256="",
            frame_count=0,
        )


class SourceStaleCaptureClient(FakeCaptureClient):
    """Emits SOURCE_STALE feedback and blocks the Action terminal state."""

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
                SimpleNamespace(frame_count=0, stage="SOURCE_STALE")
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


class FakeBackend(ReconstructionBackend):
    """Deterministic backend whose first job can block or fail."""

    def __init__(
        self,
        role: CameraRole,
        *,
        block_first: bool = False,
        first_job_state: str = JOB_SUCCEEDED,
        failure_code: int = 1,
        failure_message: str = "worker failed",
    ):
        self.role = role
        self.jobs = {}
        self.submitted = []
        self.cancel_calls = []
        self.block_first = block_first
        self.first_job_state = first_job_state
        self.failure_code = failure_code
        self.failure_message = failure_message
        self.release_first = threading.Event()
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
        state = JOB_RUNNING if self.block_first else (
            self.first_job_state if len(self.submitted) == 1 else JOB_SUCCEEDED
        )
        self.jobs[window_id] = {
            "snapshot_uri": snapshot_uri,
            "manifest_sha256": manifest_sha256,
            "state": state,
        }
        return window_id

    def status(self, job_id):
        job = self.jobs[job_id]
        if self.block_first:
            if job_id == self.submitted[0]:
                if job["state"] == JOB_CANCELLED:
                    return JobStatus(job_id, JOB_CANCELLED, "")
                if not self.release_first.is_set():
                    return JobStatus(job_id, JOB_RUNNING, "blocked")
                job["state"] = JOB_SUCCEEDED
            elif self.release_first.is_set():
                job["state"] = JOB_SUCCEEDED
        return JobStatus(job_id, job["state"], "")

    def cancel(self, job_id):
        self.cancel_calls.append(job_id)
        job = self.jobs.get(job_id)
        if job is not None:
            job["state"] = JOB_CANCELLED

    def collect_result(self, job_id):
        job = self.jobs[job_id]
        frame_count = _snapshot_frame_count(job["snapshot_uri"])
        if job["state"] == JOB_SUCCEEDED:
            return ReconstructActionOutcome(
                action_state=ACTION_SUCCEEDED,
                error_code=0,
                message="ok",
                scene_id=job_id,
                result_manifest_uri=(
                    self.result_manifest_uri
                    or f"file:///tmp/{job_id}/result.json"
                ),
                result_manifest_sha256=self.result_manifest_sha256 or ("b" * 64),
                scene_glb_uri="",
                frame_count=frame_count,
                wall_time_sec=0.1,
                camera_role=self.role.role,
                camera_topic=self.role.capture_config.image_topic,
                camera_frame_id=self.role.capture_config.camera_frame_id,
                snapshot_manifest_sha256=job["manifest_sha256"],
                result_scene_id=job_id,
                result_actual_frame_count=frame_count,
            )
        if job["state"] == JOB_CANCELLED:
            return ReconstructActionOutcome(
                action_state=ACTION_CANCELLED,
                error_code=3,
                message="cancelled",
                scene_id=job_id,
                result_manifest_uri="",
                result_manifest_sha256="",
                scene_glb_uri="",
                frame_count=0,
                wall_time_sec=0.1,
                camera_role=self.role.role,
                camera_topic=self.role.capture_config.image_topic,
                camera_frame_id=self.role.capture_config.camera_frame_id,
                snapshot_manifest_sha256=job["manifest_sha256"],
                result_scene_id=job_id,
                result_actual_frame_count=0,
            )
        return ReconstructActionOutcome(
            action_state=ACTION_FAILED,
            error_code=self.failure_code,
            message=self.failure_message,
            scene_id=job_id,
            result_manifest_uri="",
            result_manifest_sha256="",
            scene_glb_uri="",
            frame_count=0,
            wall_time_sec=0.1,
            camera_role=self.role.role,
            camera_topic=self.role.capture_config.image_topic,
            camera_frame_id=self.role.capture_config.camera_frame_id,
            snapshot_manifest_sha256=job["manifest_sha256"],
            result_scene_id=job_id,
            result_actual_frame_count=0,
        )


class SubmitErrorBackend(FakeBackend):
    """Backend whose first submit raises a transport-level error."""

    def __init__(self, role):
        super().__init__(role)
        self._submit_failures = 0

    def submit(
        self,
        snapshot_uri,
        manifest_sha256,
        window_id,
        inference_resolution=0,
        worker_timeout_sec=0.0,
    ):
        if self._submit_failures == 0:
            self._submit_failures += 1
            raise RuntimeError("backend submit exploded")
        return super().submit(snapshot_uri, manifest_sha256, window_id)


class FormalGateBackend(ReconstructionBackend):
    """Backend that lazily writes corrupt/valid worker result manifests."""

    def __init__(self, *, worker_root: Path, role: CameraRole, corrupt_first: bool = True):
        self.worker_root = worker_root
        self.role = role
        self.corrupt_first = corrupt_first
        self.submitted = []
        self.jobs = {}
        self.cancel_calls = []

    def submit(
        self,
        snapshot_uri,
        manifest_sha256,
        window_id,
        inference_resolution=0,
        worker_timeout_sec=0.0,
    ):
        self.submitted.append(window_id)
        self.jobs[window_id] = {
            "snapshot_uri": snapshot_uri,
            "manifest_sha256": manifest_sha256,
            "state": JOB_SUCCEEDED,
        }
        return window_id

    def status(self, job_id):
        return JobStatus(job_id, self.jobs[job_id]["state"], "")

    def cancel(self, job_id):
        self.cancel_calls.append(job_id)
        self.jobs[job_id]["state"] = JOB_CANCELLED

    def collect_result(self, job_id):
        index = self.submitted.index(job_id)
        job = self.jobs[job_id]
        frame_count = _snapshot_frame_count(job["snapshot_uri"])
        corrupt = self.corrupt_first and index == 0
        path = self.worker_root / (
            f"corrupt-{index}" if corrupt else f"valid-{index}"
        ) / "result.json"
        manifest_sha = write_result_manifest(
            path,
            scene_id=job_id,
            snapshot_sha=job["manifest_sha256"],
            frame_count=frame_count,
            camera_role=self.role.role,
            camera_topic=self.role.capture_config.image_topic,
            camera_frame_id=self.role.capture_config.camera_frame_id,
            checkpoint_sha=CHECKPOINT_SHA if not corrupt else ("f" * 64),
        )
        return ReconstructActionOutcome(
            action_state=ACTION_SUCCEEDED,
            error_code=0,
            message="ok",
            scene_id=job_id,
            result_manifest_uri=path.as_uri(),
            result_manifest_sha256=manifest_sha,
            scene_glb_uri="",
            frame_count=frame_count,
            wall_time_sec=0.1,
            camera_role=self.role.role,
            camera_topic=self.role.capture_config.image_topic,
            camera_frame_id=self.role.capture_config.camera_frame_id,
            snapshot_manifest_sha256=job["manifest_sha256"],
            result_scene_id=job_id,
            result_actual_frame_count=frame_count,
        )


def _fake_result_gate(uri, expected_sha256, *, expected_identity, expected_authenticity):
    """Non-formal test gate used by scheduler-orchestration rows only."""
    return SimpleNamespace(ok=True, reason="", payload={}, manifest_bytes=b"{}")


def _fake_result_publisher(
    records_root,
    session_id,
    window_index,
    manifest_bytes,
    provenance,
    snapshot_manifest_bytes=b"",
):
    return Path("/tmp/fake-formal-window"), "a" * 64


# ---------------------------------------------------------------------------
# Session test bed
# ---------------------------------------------------------------------------

class FaultMatrixBed:
    """Tempdir + deterministic waits + ContinuousSession builder."""

    def __init__(
        self,
        *,
        initial_sample_fps: float = 1.0,
        initial_window_frames: int = 6,
        initial_queue_mode: str = QUEUE_LOSSLESS,
        initial_queue_capacity: int = 8,
        initial_max_retries: int = 0,
        reconstruct_wait_timeout_sec: float = 5.0,
        min_free_disk_gb: float = 0.01,
        with_expected_authenticity: bool = False,
    ):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.journal = self.root / "journal"
        self.journal.mkdir()
        self.results = self.root / "results"
        self.results.mkdir()
        self.worker_results = self.root / "worker_results"
        self.worker_results.mkdir()
        front_role = CameraRole(
            role="front",
            capture_action="/front/capture_segment",
            capture_config=replace(make_capture_config(self.root, "front"),
                                   odometry_association="interpolate"),
        )
        down_role = CameraRole(
            role="down",
            capture_action="/down/capture_segment",
            capture_config=replace(make_capture_config(self.root, "down"),
                                   odometry_association="interpolate"),
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
            min_free_disk_gb=min_free_disk_gb,
            action_connect_timeout_sec=1.0,
            capture_wait_margin_sec=1.0,
            reconstruct_wait_timeout_sec=reconstruct_wait_timeout_sec,
            stop_timeout_sec=2.0,
            reconstruct_frame_group_size=initial_window_frames,
            initial_sample_fps=initial_sample_fps,
            initial_window_frames=initial_window_frames,
            initial_overlap_frames=5,
            initial_inference_resolution=384,
            initial_queue_mode=initial_queue_mode,
            initial_queue_capacity=initial_queue_capacity,
            initial_max_retries=initial_max_retries,
            initial_keep_snapshot_on_success=True,
            expected_checkpoint_sha256=CHECKPOINT_SHA if with_expected_authenticity else "",
            expected_checkpoint_path=CHECKPOINT_PATH if with_expected_authenticity else "",
            expected_result_root=str(self.worker_results) if with_expected_authenticity else "",
            expected_algorithm_revision=ALGO_REV if with_expected_authenticity else "",
            expected_ros_revision=ROS_REV if with_expected_authenticity else "",
            worker_python="/usr/bin/python3" if with_expected_authenticity else "",
            cuda_visible_devices="",
        )
        self.services = []
        self.capture_clients = []
        self.backends = []
        self._real_gate_default = with_expected_authenticity
        # A succeeded window starts the G4 background process; stand in for one
        # whose map already holds every formal window.
        g4_process = mock.MagicMock()
        g4_process.process.poll.return_value = None
        g4_process.status.return_value = {"state": "ready", "window_count": 1 << 30,
                                          "published_windows": []}
        self._g4_patch = mock.patch(
            "drone3d_reconstruction_core.continuous_session.G4BackendProcess",
            return_value=g4_process,
        )
        self._g4_patch.start()

    def close(self):
        for service in reversed(self.services):
            try:
                service.shutdown(2.0)
            except BaseException:
                pass
        self.services.clear()
        self._g4_patch.stop()
        self.temp.cleanup()

    def make_session(
        self,
        *,
        capture_factory=None,
        backend_factory=None,
        disk_free=None,
        config=None,
        real_gate=None,
    ):
        if real_gate is None:
            real_gate = self._real_gate_default
        service = ContinuousSession(
            config or self.config,
            capture_client_factory=capture_factory or self._default_capture_factory,
            backend_factory=backend_factory or self._default_backend_factory,
            disk_free_gb=disk_free or (lambda: 100.0),
            result_gate=verify_result_manifest if real_gate else _fake_result_gate,
            result_publisher=publish_formal_record if real_gate else _fake_result_publisher,
        )
        self.services.append(service)
        return service

    def _default_capture_factory(self, role):
        client = FakeCaptureClient(role, Path(role.capture_config.snapshot_root))
        self.capture_clients.append(client)
        return client

    def _default_backend_factory(self):
        backend = FakeBackend(self.config.roles[0])
        self.backends.append(backend)
        return backend

    def wait_until(self, predicate, timeout=8.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.01)
        raise AssertionError(f"condition not reached within {timeout} s")

    def journal_events(self):
        last_error = None
        for _ in range(40):
            try:
                lines = []
                for path in sorted(self.journal.glob("*.jsonl")):
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            lines.append(json.loads(line))
                return lines
            except (json.JSONDecodeError, OSError) as error:
                last_error = error
                time.sleep(0.05)
        raise AssertionError(f"journal never became readable: {last_error}")


# ---------------------------------------------------------------------------
# Matrix scenario functions
# ---------------------------------------------------------------------------

def scenario_source_stale_recover():
    bed = FaultMatrixBed()
    try:
        role = bed.config.roles[0]
        client = SourceStaleCaptureClient(role, bed.front_root)
        backend = FakeBackend(role)
        service = bed.make_session(
            capture_factory=lambda _role: client,
            backend_factory=lambda: backend,
        )
        code, _ = service.select_source("front")
        assert code == 0, "select_source(front) failed"
        code, _ = service.start_session("")
        assert code == 0, "start_session failed"
        assert client.goal_sent.wait(3.0), "capture goal was not sent"
        # Capture watchdog deadline is window_frames/sample_fps + 2*margin = 3 s.
        # While SOURCE_STALE is active the scheduler must not kill the Action.
        time.sleep(2.0)
        assert client.cancel_calls == 0, "scheduler watchdog fired during SOURCE_STALE"
        assert not client._ready.is_set(), "capture Action became terminal during SOURCE_STALE"
        client.feedback_cb(
            SimpleNamespace(frame_count=0, stage="SOURCE_RECOVERED")
        )
        time.sleep(0.5)
        assert client.cancel_calls == 0, (
            "scheduler watchdog fired immediately after SOURCE_RECOVERED"
        )
        client.release()
        bed.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        assert service.get_status().failed_windows == 0, "source recovery must not fail the window"
        events = bed.journal_events()
        relevant = [
            event for event in events
            if event.get("event") in ("window_pending", "window_succeeded")
        ]
        assert relevant, "journal has no pending/succeeded events"
        return _evidence(
            "stale window recovered into the same window with zero cancels",
            [event for event in events if event.get("event") in ("session_start", "window_pending", "window_succeeded")],
            cancel_calls=client.cancel_calls,
            succeeded_windows=service.get_status().succeeded_windows,
        )
    finally:
        bed.close()


def scenario_source_timeout_abort_window():
    bed = FaultMatrixBed()
    try:
        role = bed.config.roles[0]
        calls = [0]

        def capture_factory(_role):
            calls[0] += 1
            if calls[0] == 1:
                return FailingCaptureClient(
                    role,
                    bed.front_root,
                    "CAMERA_UNAVAILABLE: source timeout exceeded max_window_gap_sec; window aborted",
                )
            return FakeCaptureClient(role, bed.front_root)

        service = bed.make_session(
            capture_factory=capture_factory,
            backend_factory=lambda: FakeBackend(role),
        )
        code, _ = service.select_source("front")
        assert code == 0
        code, _ = service.start_session("")
        assert code == 0
        bed.wait_until(
            lambda: service.get_status().failed_windows >= 1
            and service.get_status().succeeded_windows >= 1,
            timeout=10.0,
        )
        events = bed.journal_events()
        capture_failed = [
            event for event in events if event.get("event") == "window_capture_failed"
        ]
        assert capture_failed, "journal has no window_capture_failed event"
        assert capture_failed[0].get("window_index") == 0
        assert "source" in capture_failed[0].get("reason", "").lower()
        assert service.get_status().failed_windows == 1
        assert service.get_status().succeeded_windows >= 1
        return _evidence(
            "source timeout aborted window 0 and the scheduler continued with window 1",
            [event for event in events if event.get("event") in ("window_capture_failed", "window_pending", "window_succeeded")],
            failed_windows=service.get_status().failed_windows,
            succeeded_windows=service.get_status().succeeded_windows,
        )
    finally:
        bed.close()


def scenario_stamp_zero_duplicate_backward():
    goal = CaptureGoalParameters(
        scene_id="cs-s0123456789abcdef-w0",
        sample_fps=1.0,
        max_frames=4,
        max_duration_sec=30.0,
    )
    sampler = StampSampler(goal)
    zero = FrameStamp(0, 0)
    first = FrameStamp(10, 0)
    duplicate = FrameStamp(10, 0)
    half_period = FrameStamp(10, 500_000_000)
    second = FrameStamp(11, 0)
    backward = FrameStamp(9, 0)

    assert sampler.classify(zero) == "zero"
    assert sampler.select(zero) is False
    assert sampler.zero_stamp_count == 1
    assert sampler.classify(first) == "ok"
    assert sampler.select(first) is True, "first valid stamp must be selected"
    assert sampler.classify(duplicate) == "duplicate"
    assert sampler.select(duplicate) is False, "duplicate stamp must be skipped"
    assert sampler.duplicate_stamp_count == 1
    assert sampler.select(half_period) is False, "before the sample period must be skipped"
    assert sampler.select(second) is True, "on-period stamp must be selected"
    assert sampler.last_selected == (11, 0)
    assert sampler.out_of_order_stamp_count == 0
    assert sampler.classify(backward) == "backward"
    try:
        sampler.select(backward)
        raised = False
    except CaptureStreamError:
        raised = True
    assert raised, "backward stamp must raise CaptureStreamError"
    assert sampler.out_of_order_stamp_count == 1
    return _evidence(
        "zero/duplicate counted and skipped; backward counted and raised; "
        "selected stamps are exactly (10,0) and (11,0)",
        zero_stamp_count=sampler.zero_stamp_count,
        duplicate_stamp_count=sampler.duplicate_stamp_count,
        out_of_order_stamp_count=sampler.out_of_order_stamp_count,
        last_selected=list(sampler.last_selected),
    )


def scenario_worker_submit_failure():
    bed = FaultMatrixBed()
    try:
        role = bed.config.roles[0]
        backend = SubmitErrorBackend(role)
        service = bed.make_session(backend_factory=lambda: backend)
        code, _ = service.select_source("front")
        assert code == 0
        code, _ = service.start_session("")
        assert code == 0
        bed.wait_until(
            lambda: service.get_status().failed_windows >= 1
            and service.get_status().succeeded_windows >= 1,
            timeout=10.0,
        )
        assert service.get_status().state != "FATAL_OOM"
        events = bed.journal_events()
        failed = [
            event for event in events if event.get("event") == "window_failed"
        ]
        assert failed, "journal has no window_failed event"
        assert failed[0].get("window_index") == 0
        assert "backend submit failed" in failed[0].get("reason", "")
        return _evidence(
            "submit exception became an ordinary window failure and the FIFO kept moving",
            [event for event in events if event.get("event") in ("window_failed", "window_pending", "window_succeeded")],
            failed_windows=service.get_status().failed_windows,
            succeeded_windows=service.get_status().succeeded_windows,
        )
    finally:
        bed.close()


def scenario_worker_nonzero_exit():
    bed = FaultMatrixBed()
    try:
        role = bed.config.roles[0]
        backend = FakeBackend(
            role,
            first_job_state=JOB_FAILED,
            failure_code=1,
            failure_message="worker exited with returncode 1",
        )
        service = bed.make_session(backend_factory=lambda: backend)
        code, _ = service.select_source("front")
        assert code == 0
        code, _ = service.start_session("")
        assert code == 0
        bed.wait_until(
            lambda: service.get_status().failed_windows >= 1
            and service.get_status().succeeded_windows >= 1,
            timeout=10.0,
        )
        events = bed.journal_events()
        failed = [
            event for event in events if event.get("event") == "window_failed"
        ]
        assert failed, "journal has no window_failed event"
        assert failed[0].get("window_index") == 0
        assert "reconstruct action ended as FAILED" in failed[0].get("reason", "")
        assert service.get_status().succeeded_windows >= 1
        return _evidence(
            "nonzero worker exit marked window 0 FAILED and window 1 still succeeded",
            [event for event in events if event.get("event") in ("window_failed", "window_pending", "window_succeeded")],
            failed_windows=service.get_status().failed_windows,
            succeeded_windows=service.get_status().succeeded_windows,
        )
    finally:
        bed.close()


def scenario_worker_watchdog_timeout_cancel():
    bed = FaultMatrixBed(reconstruct_wait_timeout_sec=0.6)
    try:
        role = bed.config.roles[0]
        backend = FakeBackend(role, block_first=True)
        service = bed.make_session(backend_factory=lambda: backend)
        code, _ = service.select_source("front")
        assert code == 0
        code, _ = service.start_session("")
        assert code == 0
        # Wait for the watchdog to cancel the hung first job, then let the
        # second job (submitted after the cancellation) finish normally.
        bed.wait_until(
            lambda: len(backend.cancel_calls) >= 1 and len(backend.submitted) >= 2,
            timeout=10.0,
        )
        backend.release_first.set()
        bed.wait_until(
            lambda: service.get_status().failed_windows >= 1
            and service.get_status().succeeded_windows >= 1,
            timeout=10.0,
        )
        events = bed.journal_events()
        failed = [
            event for event in events if event.get("event") == "window_failed"
        ]
        assert failed, "journal has no window_failed event"
        assert "reconstruct action ended as CANCELLED" in failed[0].get("reason", "")
        assert len(backend.cancel_calls) == 1, "watchdog must cancel exactly the hung job"
        return _evidence(
            "reconstruction watchdog cancelled the hung worker once and the FIFO continued",
            [event for event in events if event.get("event") in ("window_failed", "window_pending", "window_succeeded")],
            cancel_calls=list(backend.cancel_calls),
            submitted=[wid for wid in backend.submitted],
            failed_windows=service.get_status().failed_windows,
            succeeded_windows=service.get_status().succeeded_windows,
        )
    finally:
        bed.close()


def scenario_disk_redline_pause_resume():
    bed = FaultMatrixBed()
    try:
        free_gb = [0.0]
        service = bed.make_session(disk_free=lambda: free_gb[0])
        code, _ = service.select_source("front")
        assert code == 0
        code, _ = service.start_session("")
        assert code == 0
        bed.wait_until(
            lambda: service.get_status().state == SESSION_DISK_PAUSED
        )
        snapshot = service.diagnostics_snapshot()
        assert snapshot["disk_redline"] is True
        assert bed.capture_clients == [], "no capture may start below the redline"
        free_gb[0] = 100.0
        bed.wait_until(lambda: service.get_status().succeeded_windows >= 1)
        assert service.get_status().state == SESSION_RUNNING
        assert service.diagnostics_snapshot()["disk_redline"] is False
        return _evidence(
            "capture paused below disk redline and auto-resumed after space was freed",
            disk_redline_before=snapshot["disk_redline"],
            disk_redline_after=service.diagnostics_snapshot()["disk_redline"],
            succeeded_windows=service.get_status().succeeded_windows,
        )
    finally:
        bed.close()


def scenario_lossless_fifo_order():
    bed = FaultMatrixBed()
    try:
        role = bed.config.roles[0]
        backend = FakeBackend(role, block_first=True)
        service = bed.make_session(backend_factory=lambda: backend)
        code, _ = service.select_source("front")
        assert code == 0
        code, _ = service.start_session("")
        assert code == 0
        bed.wait_until(
            lambda: len(backend.submitted) == 1
            and backend.submitted[0].endswith("-w0"),
            timeout=10.0,
        )
        bed.wait_until(lambda: service.get_status().pending_windows >= 2, timeout=10.0)
        assert service.get_status().dropped_windows == 0
        assert len(backend.submitted) == 1, "single worker must stay single"
        backend.release_first.set()
        backend.jobs[backend.submitted[0]]["state"] = JOB_SUCCEEDED
        bed.wait_until(lambda: service.get_status().succeeded_windows >= 3, timeout=10.0)
        submitted_indexes = [
            int(wid.rsplit("-w", 1)[1]) for wid in backend.submitted
        ]
        assert submitted_indexes[:3] == [0, 1, 2], (
            "lossless FIFO consumed windows out of order"
        )
        assert submitted_indexes == sorted(submitted_indexes), (
            "lossless FIFO reordered submitted windows"
        )
        assert service.get_status().failed_windows == 0
        assert service.get_status().dropped_windows == 0
        return _evidence(
            "lossless_fifo submitted windows in strict FIFO order with zero drops",
            [event for event in bed.journal_events() if event.get("event") in ("window_pending", "window_succeeded")],
            submitted=[wid for wid in backend.submitted],
            pending_observed=2,
        )
    finally:
        bed.close()


def scenario_failure_then_continue():
    bed = FaultMatrixBed()
    try:
        role = bed.config.roles[0]
        backend = FakeBackend(
            role,
            first_job_state=JOB_FAILED,
            failure_code=2,
            failure_message="ordinary disk I/O failure",
        )
        service = bed.make_session(backend_factory=lambda: backend)
        code, _ = service.select_source("front")
        assert code == 0
        code, _ = service.start_session("")
        assert code == 0
        bed.wait_until(
            lambda: service.get_status().failed_windows >= 1
            and service.get_status().succeeded_windows >= 1,
            timeout=10.0,
        )
        events = bed.journal_events()
        failed = [
            event for event in events if event.get("event") == "window_failed"
        ]
        succeeded = [
            event for event in events if event.get("event") == "window_succeeded"
        ]
        assert failed and succeeded, "journal must contain a failure and a later success"
        failed_scene = failed[0]["scene_id"]
        retained = [
            path for path in bed.front_root.iterdir()
            if path.name.startswith(failed_scene + "-")
        ]
        assert not retained, "terminal failed window snapshot must be removed"
        service.stop_all(discard_queue=True)
        kept = {
            Path(unquote(urlsplit(record.snapshot_uri).path))
            for record in service._windows if record.state == "SUCCEEDED"
        }
        assert kept and set(bed.front_root.iterdir()) == kept, (
            "stop_all must leave only the snapshots G4 keeps for succeeded windows"
        )
        return _evidence(
            "ordinary failure kept its journal diagnosis, removed its snapshot, and did not block later windows",
            [event for event in events if event.get("event") in ("window_failed", "window_pending", "window_succeeded", "session_stop")],
            failed_scene=failed_scene,
            snapshot_removed=True,
            failed_windows=service.get_status().failed_windows,
            succeeded_windows=service.get_status().succeeded_windows,
        )
    finally:
        bed.close()


def scenario_formal_gate_rejects_corrupt_result():
    bed = FaultMatrixBed(with_expected_authenticity=True)
    try:
        role = bed.config.roles[0]
        backend = FormalGateBackend(
            worker_root=bed.worker_results,
            role=role,
            corrupt_first=True,
        )
        calls = [0]

        def capture_factory(_role):
            calls[0] += 1
            if calls[0] >= 3:
                return BlockingCaptureClient(role, bed.front_root)
            return FakeCaptureClient(role, bed.front_root)

        service = bed.make_session(
            capture_factory=capture_factory,
            backend_factory=lambda: backend,
            real_gate=True,
        )
        code, _ = service.select_source("front")
        assert code == 0
        code, _ = service.start_session("")
        assert code == 0
        session_id = service.get_status().session_id
        bed.wait_until(
            lambda: service.get_status().failed_windows >= 1
            and service.get_status().succeeded_windows >= 1,
            timeout=15.0,
        )
        events = bed.journal_events()
        gate_failed = [
            event for event in events if event.get("event") == "window_result_gate_failed"
        ]
        assert gate_failed, "journal has no window_result_gate_failed event"
        assert gate_failed[0].get("window_index") == 0
        assert "checkpoint" in gate_failed[0].get("reason", "")
        records = scan_session_windows(bed.results / "sessions", session_id)
        assert len(records) == 1, f"expected exactly one formal record, got {records}"
        assert records[0].get("window_index") == 1
        sessions_dir = bed.results / "sessions" / session_id
        assert not (sessions_dir / "window-0000").exists(), (
            "corrupt window must never produce a formal record"
        )
        assert (sessions_dir / "window-0001").is_dir()
        assert service.get_status().formal_windows == 1
        return _evidence(
            "real verify_result_manifest gate rejected the corrupt manifest; "
            "the next valid window published exactly one formal record",
            [event for event in events if event.get("event") in ("window_result_gate_failed", "window_failed", "window_succeeded")],
            gate_reason=gate_failed[0].get("reason"),
            formal_records=[record.get("window_index") for record in records],
        )
    finally:
        bed.close()


def scenario_stop_all_cleanup_no_orphan():
    bed = FaultMatrixBed(with_expected_authenticity=True)
    try:
        role = bed.config.roles[0]
        backend = FakeBackend(role, block_first=True)
        service = bed.make_session(
            backend_factory=lambda: backend,
            real_gate=True,
        )
        code, _ = service.select_source("front")
        assert code == 0
        code, _ = service.start_session("")
        assert code == 0
        session_id = service.get_status().session_id
        bed.wait_until(
            lambda: len(backend.submitted) == 1
            and service.get_status().pending_windows >= 1,
            timeout=10.0,
        )
        assert any(bed.front_root.iterdir()), "expected snapshot dirs before stop_all"
        code, _ = service.stop_all(discard_queue=True)
        assert code == 0
        assert service.get_status().state == SESSION_STOPPED
        assert not any(bed.front_root.iterdir()), (
            "stop_all must remove all pending/running snapshots"
        )
        records = scan_session_windows(bed.results / "sessions", session_id)
        assert records == [], "no formal records may exist after stop_all"
        sessions_root = bed.results / "sessions"
        if sessions_root.exists():
            assert not any(sessions_root.rglob("window-*")), (
                "no staging/formal window dirs may remain"
            )
        events = bed.journal_events()
        assert any(event.get("event") == "session_stop" for event in events)
        return _evidence(
            "stop_all cancelled the running worker, removed pending snapshots and left no formal/staging records",
            [event for event in events if event.get("event") in ("window_pending", "session_stop")],
            submitted=[wid for wid in backend.submitted],
            cancel_calls=list(backend.cancel_calls),
        )
    finally:
        bed.close()


def scenario_fake_exclusion():
    bed = FaultMatrixBed()
    try:
        sessions_root = bed.results / "sessions"
        sessions_root.mkdir(parents=True, exist_ok=True)
        # Fake session: valid session-id shape but only foreign/fake content.
        fake_session = "s" + "a" * 16
        fake_dir = sessions_root / fake_session
        (fake_dir / "window-0000").mkdir(parents=True)
        (fake_dir / "window-0000" / "result.json").write_text("{}", encoding="utf-8")
        (fake_dir / "window-0000" / "provenance.json").write_text(
            json.dumps(
                {
                    "schema": "drone3d.continuous.fake_record",
                    "schema_version": 1,
                    "window_index": 0,
                }
            ),
            encoding="utf-8",
        )
        (fake_dir / "window-0001").mkdir()
        (fake_dir / "window-0001" / "result.json").write_text("{}", encoding="utf-8")
        (fake_dir / "not-a-window").mkdir()
        # Foreign session with valid shape but no complete records.
        foreign_session = "s" + "c" * 16
        foreign_dir = sessions_root / foreign_session
        (foreign_dir / "window-0000").mkdir(parents=True)
        (foreign_dir / "window-0000" / "provenance.json").write_text(
            json.dumps(
                {
                    "schema": FORMAL_RECORD_SCHEMA,
                    "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
                    "window_index": 0,
                }
            ),
            encoding="utf-8",
        )
        # A genuine record created through the real publisher must be visible.
        valid_session = "s" + "b" * 16
        (sessions_root / valid_session).mkdir(parents=True, exist_ok=True)
        role = bed.config.roles[0]
        worker_result = bed.worker_results / "fake-exclusion-valid" / "result.json"
        write_result_manifest(
            worker_result,
            scene_id=f"cs-{valid_session}-w0",
            snapshot_sha="a" * 64,
            frame_count=1,
            camera_role=role.role,
            camera_topic=role.capture_config.image_topic,
            camera_frame_id=role.capture_config.camera_frame_id,
        )
        manifest_bytes = worker_result.read_bytes()
        snapshot_manifest_bytes = (
            b"path,size,sha256\nsnapshot.json,2,"
            + hashlib.sha256(b"{}").hexdigest().encode("ascii")
            + b"\n"
        )
        provenance = {
            "schema": FORMAL_RECORD_SCHEMA,
            "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
            "session_id": valid_session,
            "window_index": 0,
            "result_manifest_uri": worker_result.as_uri(),
            "result_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "snapshot_manifest_sha256": hashlib.sha256(
                snapshot_manifest_bytes
            ).hexdigest(),
            "gate": {
                "result_manifest_digest_verified": True,
                "identity_verified": True,
                "authenticity_verified": True,
                "schema_verified": True,
            },
            "worker": {"wall_time_sec": 10.0, "exit_returncode": 0},
        }
        final_path, _ = publish_formal_record(
            sessions_root,
            valid_session,
            0,
            manifest_bytes,
            provenance,
            snapshot_manifest_bytes,
        )
        assert final_path.is_dir()
        assert scan_session_windows(sessions_root, fake_session) == []
        assert scan_session_windows(sessions_root, foreign_session) == []
        records = scan_session_windows(sessions_root, valid_session)
        assert len(records) == 1
        assert records[0].get("window_index") == 0
        return _evidence(
            "scan_session_windows ignored fake/foreign dirs and returned only the genuine record",
            fake_session=fake_session,
            foreign_session=foreign_session,
            valid_session=valid_session,
            scanned_valid_windows=[record.get("window_index") for record in records],
        )
    finally:
        bed.close()


def scenario_web_diagnostics_consistency():
    temp = Path(tempfile.mkdtemp(prefix="acc005-web-"))
    try:
        config = WebDashboardConfig(
            host="127.0.0.1",
            port=0,
            session_command_service="/continuous_session/command",
            result_root=str(temp),
            roles={
                "front": WebRoleInfo(
                    image_topic="/front/image_raw",
                    depth_topic="/front/depth",
                    camera_name="front_cam",
                    camera_frame_id="front_optical",
                )
            },
            web_preview_fps=5.0,
        )
        diagnostics = {
            "state": "RUNNING",
            "session_id": "s0123456789abcdef",
            "role": "front",
            "window_index": 7,
            "captured_frames": 3,
            "pending_windows": 2,
            "running_window": {"window_index": 4, "scene_id": "cs-s0123456789abcdef-w4"},
            "succeeded_windows": 5,
            "failed_windows": 1,
            "dropped_windows": 0,
            "effective": {
                "sample_fps": 1.0,
                "window_frames": 8,
                "queue_mode": "lossless_fifo",
                "queue_capacity": 8,
                "max_retries": 0,
                "keep_snapshot_on_success": False,
            },
            "fifo_bytes": 123,
            "fifo_oldest_wait_sec": 12.5,
            "last_error": "none",
            "min_free_disk_gb": 1.5,
            "disk_free_gb": 42.0,
            "disk_redline": False,
            "map_epoch": 0,
            "last_alignment_state": "MODEL_LOCAL_UNALIGNED",
            "formal_windows": 5,
        }
        sent_commands = []

        def get_status():
            return dict(diagnostics)

        def send_command(command):
            sent_commands.append(dict(command))
            return {"ok": True, "error": "", "response": {"state": "PAUSED"}}

        core = WebDashboardCore(
            config,
            get_status=get_status,
            send_command=send_command,
        )
        core.start()
        try:
            base = f"http://127.0.0.1:{core.server_port}"
            with urllib.request.urlopen(base + "/api/state", timeout=5.0) as response:
                state = json.loads(response.read().decode("utf-8"))
            assert state["diagnostics"] == diagnostics, (
                "/api/state diagnostics differ from injected get_status"
            )
            compact = state["status"]
            for key in (
                "state",
                "window_index",
                "captured_frames",
                "pending_windows",
                "succeeded_windows",
                "failed_windows",
            ):
                assert compact.get(key) == diagnostics.get(key), (
                    f"compact status field {key} mismatch"
                )
            request = urllib.request.Request(
                base + "/api/command",
                data=json.dumps({"command": "pause"}).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5.0) as response:
                command_result = json.loads(response.read().decode("utf-8"))
            assert command_result.get("ok") is True
            assert len(sent_commands) == 1, (
                "POST /api/command must forward exactly once"
            )
            assert sent_commands[0] == {"command": "pause"}
            return _evidence(
                "GET /api/state exposed the injected diagnostics and POST /api/command forwarded exactly once",
                forwarded_commands=sent_commands,
                state_diagnostics=state["diagnostics"],
            )
        finally:
            core.stop(2.0)
    finally:
        shutil.rmtree(str(temp), ignore_errors=True)


SCENARIOS = {
    "source_stale_recover": scenario_source_stale_recover,
    "source_timeout_abort_window": scenario_source_timeout_abort_window,
    "stamp_zero_duplicate_backward": scenario_stamp_zero_duplicate_backward,
    "worker_submit_failure": scenario_worker_submit_failure,
    "worker_nonzero_exit": scenario_worker_nonzero_exit,
    "worker_watchdog_timeout_cancel": scenario_worker_watchdog_timeout_cancel,
    "disk_redline_pause_resume": scenario_disk_redline_pause_resume,
    "lossless_fifo_order": scenario_lossless_fifo_order,
    "failure_then_continue": scenario_failure_then_continue,
    "formal_gate_rejects_corrupt_result": scenario_formal_gate_rejects_corrupt_result,
    "stop_all_cleanup_no_orphan": scenario_stop_all_cleanup_no_orphan,
    "fake_exclusion": scenario_fake_exclusion,
    "web_diagnostics_consistency": scenario_web_diagnostics_consistency,
}


# ---------------------------------------------------------------------------
# Matrix runner
# ---------------------------------------------------------------------------

def _valid_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _valid_nontrivial_sha256(value) -> bool:
    """64-hex digest that is neither all the same character nor all zeros."""
    if not _valid_sha256(value):
        return False
    if len(set(value)) == 1:
        return False
    if value == "0" * 64:
        return False
    return True


def _checked_regular_file(path_value, label: str, maximum_bytes: int) -> Path:
    if not isinstance(path_value, (str, Path)):
        raise ValueError(f"{label} path is missing")
    path = Path(path_value)
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        info = os.lstat(str(path))
    except OSError as error:
        raise ValueError(f"{label} cannot be inspected") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular non-symlink file")
    if info.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds size bound")
    return path


def _sha256_regular_file(path_value, label: str, maximum_bytes: int) -> str:
    path = _checked_regular_file(path_value, label, maximum_bytes)
    digest = hashlib.sha256()
    total = 0
    with open(str(path), "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError(f"{label} exceeds size bound")
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_regular_file(path_value, label: str) -> tuple[Path, dict]:
    path = _checked_regular_file(path_value, label, 16 * 1024 * 1024)
    with open(str(path), "rb") as stream:
        payload = json.loads(stream.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return path, payload


def _positive_finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is malformed")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return value


def _file_uri_path(uri, label: str) -> Path:
    if not isinstance(uri, str) or not uri.startswith("file:///"):
        raise ValueError(f"{label} must be an absolute file URI")
    if "?" in uri or "#" in uri:
        raise ValueError(f"{label} must not contain query or fragment")
    return Path(uri[len("file://") :])


def _validate_real_cuda_evidence(row_evidence) -> tuple[bool, str]:
    """核验可信同机 live harness 所留 cancel bundle 的内部一致性。

    绝不因证据自带 ``status: PASS`` 而放行；本函数不是签名、远程证明或
    抗本机恶意写入的 CUDA attestation，真实运行身份由执行 harness 的审核
    边界提供。
    """
    if not isinstance(row_evidence, dict):
        return False, "real CUDA evidence is missing"
    try:
        formal_count = row_evidence.get("formal_count")
        if type(formal_count) is not int or formal_count < 3:
            return False, "formal_count must be >= 3"
        checkpoint_sha256 = row_evidence.get("checkpoint_sha256")
        if not _valid_nontrivial_sha256(checkpoint_sha256):
            return False, (
                "checkpoint_sha256 must be a 64-hex digest that is "
                "not all the same character and not all zeros"
            )
        checkpoint_path = row_evidence.get("checkpoint_path")
        actual_checkpoint_sha = _sha256_regular_file(
            checkpoint_path, "checkpoint", 8 * 1024 * 1024 * 1024
        )
        if actual_checkpoint_sha != checkpoint_sha256:
            return False, "checkpoint file sha256 does not match evidence"
        revisions = {}
        for field in ("algorithm_revision", "ros_revision"):
            value = row_evidence.get(field)
            if (
                not isinstance(value, str)
                or not value.strip()
                or any(ord(char) < 32 or ord(char) == 127 for char in value)
            ):
                return False, f"{field} is missing or malformed"
            revisions[field] = value

        session_id = row_evidence.get("session_id")
        if not (
            isinstance(session_id, str)
            and len(session_id) == 17
            and session_id.startswith("s")
            and all(char in "0123456789abcdef" for char in session_id[1:])
        ):
            return False, "session_id is malformed"
        records_root_value = row_evidence.get("formal_records_root")
        if not isinstance(records_root_value, str):
            return False, "formal_records_root is missing"
        records_root = Path(records_root_value)
        if (
            not records_root.is_absolute()
            or not records_root.is_dir()
            or records_root.is_symlink()
        ):
            return False, "formal_records_root is not a trusted absolute directory"
        session_dir = records_root / session_id
        if not session_dir.is_dir() or session_dir.is_symlink():
            return False, "formal session directory is missing or untrusted"

        journal_path = row_evidence.get("journal_path")
        journal_sha256 = row_evidence.get("journal_sha256")
        if not _valid_sha256(journal_sha256):
            return False, "journal_sha256 is invalid"
        if (
            _sha256_regular_file(journal_path, "journal", 16 * 1024 * 1024)
            != journal_sha256
        ):
            return False, "journal file sha256 does not match evidence"
        journal_file = _checked_regular_file(
            journal_path, "journal", 16 * 1024 * 1024
        )
        journal_events = []
        with open(str(journal_file), "r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    return False, f"journal line {line_number} is not an object"
                journal_events.append(event)
        if not journal_events or journal_events[-1].get("event") != "session_stop":
            return False, "journal does not end with session_stop"
        stop_event = journal_events[-1]
        if stop_event.get("session_id") != session_id:
            return False, "journal session_stop does not match session_id"
        if stop_event.get("terminal_state") != "STOPPED":
            return False, "journal session_stop has no STOPPED terminal state"
        if stop_event.get("running_before") is not True:
            return False, "journal does not prove cancel occurred while a worker ran"
        backend_job_id = stop_event.get("backend_job_id_before")
        if not isinstance(backend_job_id, str) or not backend_job_id:
            return False, "journal does not bind the cancelled backend job"
        if stop_event.get("backend_cancel_requested") is not True:
            return False, "journal does not prove backend cancel was requested"
        if stop_event.get("backend_terminal_collected") is not True:
            return False, "journal does not prove the cancelled backend reached terminal"
        if stop_event.get("backend_terminal_state") != JOB_CANCELLED:
            return False, "journal backend terminal state is not CANCELLED"
        if not isinstance(stop_event.get("backend_terminal_message"), str) or not stop_event.get(
            "backend_terminal_message"
        ):
            return False, "journal backend terminal message is missing"
        for field in (
            "running_after",
            "pending_after",
            "worker_active_after",
            "threads_alive_after",
            "snapshot_after",
            "staging_after",
        ):
            if type(stop_event.get(field)) is not int or stop_event.get(field) != 0:
                return False, f"journal session_stop {field} must be 0"

        records = row_evidence.get("records")
        if not isinstance(records, list) or len(records) < 3:
            return False, "records must be a list of at least 3 window records"
        if len(records) != formal_count:
            return False, "formal_count must equal the number of retained records"
        seen_windows = set()
        actual_walls = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                return False, f"records[{index}] must be a dict"
            window_index = record.get("window_index")
            if type(window_index) is not int or window_index < 0:
                return False, f"records[{index}] window_index is malformed"
            if window_index in seen_windows:
                return False, f"records[{index}] duplicates a window_index"
            seen_windows.add(window_index)
            if record.get("checkpoint_sha256") != checkpoint_sha256:
                return False, (
                    f"records[{index}] checkpoint_sha256 does not match "
                    "evidence checkpoint_sha256"
                )
            if not _valid_sha256(record.get("result_manifest_sha256")):
                return False, (
                    f"records[{index}] result_manifest_sha256 is not a "
                    "64-hex digest"
                )
            worker_wall_sec = _positive_finite(
                record.get("worker_wall_sec"), f"records[{index}] worker_wall_sec"
            )
            actual_walls.append(worker_wall_sec)

            formal_dir = session_dir / f"window-{window_index:04d}"
            if not formal_dir.is_dir() or formal_dir.is_symlink():
                return False, f"records[{index}] formal directory is missing or untrusted"
            result_path, result_payload = _read_json_regular_file(
                formal_dir / "result.json", f"records[{index}] result.json"
            )
            provenance_path, provenance = _read_json_regular_file(
                formal_dir / "provenance.json", f"records[{index}] provenance.json"
            )
            result_sha = _sha256_regular_file(
                result_path, f"records[{index}] result.json", 16 * 1024 * 1024
            )
            if result_sha != record.get("result_manifest_sha256"):
                return False, f"records[{index}] result.json sha256 mismatch"
            if (
                provenance.get("session_id") != session_id
                or provenance.get("window_index") != window_index
                or provenance.get("result_manifest_sha256") != result_sha
            ):
                return False, f"records[{index}] provenance identity/hash mismatch"
            if (
                provenance.get("schema") != FORMAL_RECORD_SCHEMA
                or provenance.get("schema_version") != FORMAL_RECORD_SCHEMA_VERSION
            ):
                return False, f"records[{index}] is not a current formal record"
            if any(provenance.get(field) != value for field, value in revisions.items()):
                return False, f"records[{index}] provenance revision mismatch"
            checkpoint = provenance.get("checkpoint")
            if not isinstance(checkpoint, dict) or checkpoint != {
                "path": checkpoint_path,
                "sha256": checkpoint_sha256,
            }:
                return False, f"records[{index}] provenance checkpoint mismatch"
            worker = provenance.get("worker")
            if not isinstance(worker, dict) or worker.get("exit_returncode") != 0:
                return False, f"records[{index}] worker result is not successful"
            if worker.get("wall_time_sec") != worker_wall_sec:
                return False, f"records[{index}] worker wall time mismatch"
            gate = provenance.get("gate")
            if not isinstance(gate, dict) or not all(
                gate.get(field) is True
                for field in (
                    "result_manifest_digest_verified",
                    "identity_verified",
                    "authenticity_verified",
                    "schema_verified",
                )
            ):
                return False, f"records[{index}] formal gate is not fully verified"
            worker_manifest_path = _file_uri_path(
                provenance.get("result_manifest_uri"),
                f"records[{index}] result_manifest_uri",
            )
            gate_result = verify_result_manifest(
                provenance.get("result_manifest_uri"),
                result_sha,
                expected_identity={
                    "scene_id": provenance.get("scene_id"),
                    "captured_frames": provenance.get("captured_frames"),
                    "snapshot_manifest_sha256": provenance.get(
                        "snapshot_manifest_sha256"
                    ),
                    "camera_role": provenance.get("camera_role"),
                    "camera_topic": provenance.get("camera_topic"),
                    "camera_frame_id": provenance.get("camera_frame_id"),
                },
                expected_authenticity={
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_sha256": checkpoint_sha256,
                    "result_root": str(worker_manifest_path.parent.parent),
                    "algorithm_revision": revisions["algorithm_revision"],
                    "ros_revision": revisions["ros_revision"],
                },
            )
            if not gate_result.ok:
                return False, f"records[{index}] worker gate failed: {gate_result.reason}"
            if gate_result.payload["stage_timing"]["worker_wall_time_sec"] != worker_wall_sec:
                return False, f"records[{index}] result worker wall time mismatch"

            provenance_sha = _sha256_regular_file(
                provenance_path,
                f"records[{index}] provenance.json",
                16 * 1024 * 1024,
            )
            success_events = [
                event
                for event in journal_events
                if event.get("event") == "window_succeeded"
                and event.get("session_id") == session_id
                and event.get("window_index") == window_index
            ]
            if len(success_events) != 1:
                return False, f"records[{index}] has no unique journal success event"
            success = success_events[0]
            if (
                success.get("result_manifest_sha256") != result_sha
                or success.get("formal_provenance_sha256") != provenance_sha
                or success.get("formal_result_uri") != str(formal_dir)
            ):
                return False, f"records[{index}] journal success evidence mismatch"

        expected_names = {f"window-{window_index:04d}" for window_index in seen_windows}
        if {entry.name for entry in session_dir.iterdir()} != expected_names:
            return False, "formal session contents do not match retained records"
        scanned_windows = {
            record.get("window_index")
            for record in scan_session_windows(records_root, session_id)
        }
        if scanned_windows != seen_windows:
            return False, "formal scanner does not accept exactly the retained records"

        wall_min = _positive_finite(row_evidence.get("worker_wall_min"), "worker_wall_min")
        wall_max = _positive_finite(row_evidence.get("worker_wall_max"), "worker_wall_max")
        if not math.isclose(wall_min, min(actual_walls), rel_tol=1e-12) or not math.isclose(
            wall_max, max(actual_walls), rel_tol=1e-12
        ):
            return False, "worker wall range does not match retained records"
        snapshot_roots = row_evidence.get("snapshot_roots")
        if not isinstance(snapshot_roots, list) or not snapshot_roots:
            return False, "snapshot_roots evidence is missing"
        for root_value in snapshot_roots:
            root = Path(root_value) if isinstance(root_value, str) else Path()
            if not root.is_absolute() or not root.is_dir() or root.is_symlink():
                return False, "snapshot root is missing or untrusted"
            if any(root.iterdir()):
                return False, "snapshot root is not empty after cancel"
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        return False, f"real CUDA evidence fields are malformed: {error}"
    return True, ""


def run_acc005_fault_matrix(real_cuda_evidence: dict | None = None) -> dict:
    """Run every CPU-runnable ACC-005 row and return a JSON-safe report.

    Raises:
        AssertionError: if any CPU-runnable row fails.
    """
    real_cuda_evidence = real_cuda_evidence or {}
    report = {
        "matrix": {},
        "summary": {
            "total_rows": len(ACC_FAULT_MATRIX),
            "passed": 0,
            "failed": 0,
            "pending_real_cuda": 0,
        },
        "real_cuda_evidence": real_cuda_evidence,
        "generated_at": time.time(),
    }
    failures = []
    for row, spec in ACC_FAULT_MATRIX.items():
        if not spec.get("cpu_runnable", False):
            row_evidence = real_cuda_evidence.get(row)
            if not isinstance(row_evidence, dict):
                row_evidence = real_cuda_evidence.get(f"{row}_row", {})
            entry = dict(spec)
            evidence_ok, evidence_error = _validate_real_cuda_evidence(
                row_evidence
            )
            if evidence_ok:
                entry.update(
                    {
                        "status": "PASS",
                        "evidence": dict(row_evidence),
                        "real_cuda_evidence": dict(row_evidence),
                    }
                )
                report["summary"]["passed"] += 1
            elif isinstance(row_evidence, dict) and row_evidence:
                entry.update(
                    {
                        "status": "FAIL",
                        "error": f"real CUDA evidence invalid: {evidence_error}",
                        "real_cuda_evidence": dict(row_evidence),
                    }
                )
                report["summary"]["failed"] += 1
                failures.append(f"{row}: {evidence_error}")
            else:
                entry.update(
                    {
                        "status": "PENDING_REAL_CUDA",
                        "evidence": {
                            "assertion": (
                                "real CUDA continuous-scheduler cancel is intentionally "
                                "not runnable in the CPU matrix"
                            ),
                            "timestamp": time.time(),
                            "journal_event_excerpt": [],
                        },
                        "real_cuda_evidence": row_evidence,
                    }
                )
                report["summary"]["pending_real_cuda"] += 1
            report["matrix"][row] = entry
            continue
        scenario = SCENARIOS[row]
        try:
            evidence = scenario()
        except AssertionError as error:
            entry = dict(spec)
            entry.update({"status": "FAIL", "error": str(error)})
            report["matrix"][row] = entry
            report["summary"]["failed"] += 1
            failures.append(f"{row}: {error}")
        except Exception as error:  # noqa: BLE001 - matrix runner must isolate rows
            entry = dict(spec)
            entry.update({"status": "ERROR", "error": f"{type(error).__name__}: {error}"})
            report["matrix"][row] = entry
            report["summary"]["failed"] += 1
            failures.append(f"{row}: {type(error).__name__}: {error}")
        else:
            entry = dict(spec)
            entry.update({"status": "PASS", "evidence": evidence})
            report["matrix"][row] = entry
            report["summary"]["passed"] += 1
    if failures:
        error = AssertionError(
            "ACC-005 CPU fault matrix failures:\n" + "\n".join(failures)
        )
        error.report = report  # type: ignore[attr-defined]
        raise error
    return report


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

_ENV_REAL_CUDA_EVIDENCE = "DRONE3D_ACC005_REAL_CUDA_EVIDENCE"


def _real_cuda_evidence_from_env() -> dict:
    """从环境变量读取真实 CUDA 故障矩阵证据（默认无）。"""
    path = os.environ.get(_ENV_REAL_CUDA_EVIDENCE)
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


class Acc005FaultMatrixTest(unittest.TestCase):
    _cached_report = None

    def test_acc005_fault_matrix_all_cpu_rows_pass(self):
        real_cuda_evidence = _real_cuda_evidence_from_env()
        report = run_acc005_fault_matrix(real_cuda_evidence=real_cuda_evidence)
        type(self)._cached_report = report
        self.assertEqual(report["summary"]["failed"], 0)
        expected_passed = len(SCENARIOS) + (1 if real_cuda_evidence else 0)
        self.assertEqual(report["summary"]["passed"], expected_passed)
        if real_cuda_evidence:
            self.assertEqual(report["summary"]["pending_real_cuda"], 0)
            self.assertEqual(
                report["matrix"]["real_cuda_continuous_cancel"]["status"],
                "PASS",
            )
        else:
            self.assertEqual(report["summary"]["pending_real_cuda"], 1)
        for row, entry in report["matrix"].items():
            if entry.get("cpu_runnable"):
                self.assertEqual(
                    entry.get("status"),
                    "PASS",
                    f"{row} failed: {entry.get('error', '')}",
                )
        self.assertIn("real_cuda_continuous_cancel", report["matrix"])
        if not real_cuda_evidence:
            self.assertEqual(
                report["matrix"]["real_cuda_continuous_cancel"]["status"],
                "PENDING_REAL_CUDA",
            )

    def test_real_cuda_evidence_status_is_derived_not_trusted(self):
        with self.assertRaises(AssertionError) as context:
            run_acc005_fault_matrix(
                {"real_cuda_continuous_cancel_row": {"status": "PASS"}}
            )
        report = getattr(context.exception, "report", {})
        row = report.get("matrix", {}).get("real_cuda_continuous_cancel", {})
        self.assertEqual(row.get("status"), "FAIL")
        self.assertIn("formal_count must be >= 3", row.get("error", ""))

    def test_real_cuda_evidence_rejects_trivial_sha_and_fake_records(self):
        base = {
            "formal_count": 3,
            "worker_wall_min": 1.0,
            "worker_wall_max": 2.0,
            "cancel": {
                "terminal_state": "STOPPED",
                "running_after": 0,
                "pending_after": 0,
                "staging_after": 0,
                "worker_processes_after": 0,
                "formal_preserved": 1,
            },
        }
        with self.assertRaises(AssertionError) as context:
            run_acc005_fault_matrix(
                {"real_cuda_continuous_cancel_row": dict(base, checkpoint_sha256="0" * 64)}
            )
        report = getattr(context.exception, "report", {})
        row = report.get("matrix", {}).get("real_cuda_continuous_cancel", {})
        self.assertEqual(row.get("status"), "FAIL")
        self.assertIn("checkpoint_sha256", row.get("error", ""))

        with self.assertRaises(AssertionError) as context:
            run_acc005_fault_matrix(
                {"real_cuda_continuous_cancel_row": dict(base, checkpoint_sha256="a" * 64)}
            )
        report = getattr(context.exception, "report", {})
        row = report.get("matrix", {}).get("real_cuda_continuous_cancel", {})
        self.assertEqual(row.get("status"), "FAIL")
        self.assertIn("checkpoint_sha256", row.get("error", ""))

        # 只有通过 checkpoint 但缺少 records 的 evidence 同样不能放行。
        with self.assertRaises(AssertionError) as context:
            run_acc005_fault_matrix(
                {
                    "real_cuda_continuous_cancel_row": dict(
                        base, checkpoint_sha256="ab" * 32
                    )
                }
            )
        report = getattr(context.exception, "report", {})
        row = report.get("matrix", {}).get("real_cuda_continuous_cancel", {})
        self.assertEqual(row.get("status"), "FAIL")
        self.assertIn("checkpoint", row.get("error", ""))

    def test_real_cuda_evidence_accepts_artifact_bound_records_and_journal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoint.pt"
            checkpoint_path.write_bytes(b"real checkpoint fixture\n")
            checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            records_root = root / "results" / "sessions"
            session_id = "s1234567890abcdef"
            session_dir = records_root / session_id
            session_dir.mkdir(parents=True)
            worker_root = root / "worker_results"
            snapshot_roots = [root / "front" / "snapshots", root / "down" / "snapshots"]
            for snapshot_root in snapshot_roots:
                snapshot_root.mkdir(parents=True)
            records = []
            journal_events = []
            for window_index in range(3):
                scene_id = f"cs-{session_id}-w{window_index}"
                worker_manifest = worker_root / f"w{window_index}" / "result.json"
                snapshot_manifest_bytes = (
                    f"fixture snapshot manifest {window_index}\n".encode("utf-8")
                )
                snapshot_sha = hashlib.sha256(snapshot_manifest_bytes).hexdigest()
                result_sha = write_result_manifest(
                    worker_manifest,
                    scene_id=scene_id,
                    snapshot_sha=snapshot_sha,
                    frame_count=2,
                    camera_role="front",
                    camera_topic="/front/image_raw",
                    camera_frame_id="front_optical",
                    checkpoint_path=str(checkpoint_path),
                    checkpoint_sha=checkpoint_sha256,
                )
                provenance = {
                    "schema": FORMAL_RECORD_SCHEMA,
                    "schema_version": FORMAL_RECORD_SCHEMA_VERSION,
                    "session_id": session_id,
                    "window_index": window_index,
                    "scene_id": scene_id,
                    "result_manifest_uri": worker_manifest.as_uri(),
                    "result_manifest_sha256": result_sha,
                    "snapshot_manifest_sha256": snapshot_sha,
                    "captured_frames": 2,
                    "camera_role": "front",
                    "camera_topic": "/front/image_raw",
                    "camera_frame_id": "front_optical",
                    "checkpoint": {
                        "path": str(checkpoint_path),
                        "sha256": checkpoint_sha256,
                    },
                    "algorithm_revision": ALGO_REV,
                    "ros_revision": ROS_REV,
                    "gate": {
                        "result_manifest_digest_verified": True,
                        "identity_verified": True,
                        "authenticity_verified": True,
                        "schema_verified": True,
                    },
                    "worker": {"exit_returncode": 0, "wall_time_sec": 1.5},
                }
                formal_dir, provenance_sha = publish_formal_record(
                    records_root,
                    session_id,
                    window_index,
                    worker_manifest.read_bytes(),
                    provenance,
                    snapshot_manifest_bytes,
                )
                records.append(
                    {
                        "window_index": window_index,
                        "checkpoint_sha256": checkpoint_sha256,
                        "worker_wall_sec": 1.5,
                        "result_manifest_sha256": result_sha,
                    }
                )
                journal_events.append(
                    {
                        "event": "window_succeeded",
                        "session_id": session_id,
                        "window_index": window_index,
                        "formal_result_uri": str(formal_dir),
                        "formal_provenance_sha256": provenance_sha,
                        "result_manifest_sha256": result_sha,
                    }
                )
            journal_events.append(
                {
                    "event": "session_stop",
                    "session_id": session_id,
                    "terminal_state": "STOPPED",
                    "running_before": True,
                    "backend_job_id_before": f"cs-{session_id}-w3",
                    "backend_cancel_requested": True,
                    "backend_terminal_collected": True,
                    "backend_terminal_state": JOB_CANCELLED,
                    "backend_terminal_message": "cancelled",
                    "running_after": 0,
                    "pending_after": 0,
                    "worker_active_after": 0,
                    "threads_alive_after": 0,
                    "snapshot_after": 0,
                    "staging_after": 0,
                }
            )
            journal_path = root / "journal.jsonl"
            journal_path.write_text(
                "".join(json.dumps(event, sort_keys=True) + "\n" for event in journal_events),
                encoding="utf-8",
            )
            evidence = {
                "formal_count": 3,
                "formal_records_root": str(records_root),
                "session_id": session_id,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "algorithm_revision": ALGO_REV,
                "ros_revision": ROS_REV,
                "journal_path": str(journal_path),
                "journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(),
                "snapshot_roots": [str(path) for path in snapshot_roots],
                "worker_wall_min": 1.5,
                "worker_wall_max": 1.5,
                "records": records,
            }
            report = run_acc005_fault_matrix(
                {"real_cuda_continuous_cancel_row": evidence}
            )
            self.assertEqual(
                report["matrix"]["real_cuda_continuous_cancel"]["status"], "PASS"
            )
            (session_dir / "window-0000" / "provenance.json.sha256").unlink()
            ok, reason = _validate_real_cuda_evidence(evidence)
            self.assertFalse(ok)
            self.assertIn("formal scanner", reason)

    def test_fault_matrix_report_json_serializable(self):
        report = type(self)._cached_report
        if report is None:
            report = run_acc005_fault_matrix()
            type(self)._cached_report = report
        serialized = json.dumps(report, sort_keys=True)
        self.assertIsInstance(serialized, str)
        parsed = json.loads(serialized)
        self.assertEqual(
            set(parsed["matrix"].keys()), set(ACC_FAULT_MATRIX.keys())
        )
        for row, entry in parsed["matrix"].items():
            self.assertIn(entry["status"], ("PASS", "PENDING_REAL_CUDA", "FAIL", "ERROR"))


if __name__ == "__main__":
    unittest.main()
