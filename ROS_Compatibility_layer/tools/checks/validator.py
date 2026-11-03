#!/usr/bin/env python3
"""Check the real Omega validator and atomic commit with synthetic NPZ inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading

import numpy as np

from drone3d_reconstruction_core import (
    ALGORITHM_FAILED,
    CompletionError,
    JobFinished,
    ProductionConfig,
    ProductionPlanner,
    ResourceLimits,
    ResourceSample,
    ResourceDecision,
    SnapshotSafetyLimits,
)
from drone3d_reconstruction_core.production import (
    COORDINATE_STATE,
    RESULT_SCHEMA,
    RESULT_SCHEMA_VERSION,
)
from drone3d_reconstruction_core.worker_runner import run_worker


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def passing_resource_enforcer(limits, snapshot_root, result_root):
    sample = ResourceSample(0, 0, 0, 0, 0, 0, 0, limits.swap_sample_window_sec)
    return ResourceDecision(True, None, "manual smoke injected pass", sample, limits)


def make_snapshot(snapshots_root: Path):
    scene = snapshots_root / "real-validator-scene"
    images = scene / "images"
    images.mkdir(parents=True)
    image_relatives = ["images/frame_000.png", "images/frame_001.png"]
    for relative in image_relatives:
        (scene.joinpath(*Path(relative).parts)).write_bytes(
            f"pixel-bytes-{relative}".encode("utf-8")
        )
    snapshot_bytes = json.dumps(
        {
            "schema": "drone3d.reconstruction.snapshot",
            "version": 1,
            "status": "PASS",
            "scene_id": "real-validator-scene",
            "frame_count": 2,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    (scene / "snapshot.json").write_bytes(snapshot_bytes)
    frames_bytes = (
        "frame_index,image_path\n0,images/frame_000.png\n"
        "1,images/frame_001.png\n"
    ).encode("utf-8")
    (scene / "frames.csv").write_bytes(frames_bytes)
    manifest_rows = {
        "snapshot.json": snapshot_bytes,
        "frames.csv": frames_bytes,
    }
    for relative in image_relatives:
        manifest_rows[relative] = (scene.joinpath(*Path(relative).parts)).read_bytes()
    lines = ["path,size,sha256"]
    for relative in sorted(manifest_rows):
        data = manifest_rows[relative]
        lines.append(f"{relative},{len(data)},{sha256_file_bytes(data)}")
    manifest_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    (scene / "files_sha256.csv").write_bytes(manifest_bytes)
    return str(scene), hashlib.sha256(manifest_bytes).hexdigest()


def sha256_file_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_real_npz(staging: Path, basenames) -> None:
    frames = len(basenames)
    height = width = 4
    depth = np.full((frames, height, width, 1), 0.5, dtype=np.float32)
    depth_conf = np.full((frames, height, width), 1.0, dtype=np.float32)
    images = np.full((frames, 3, height, width), 0.25, dtype=np.float32)
    extrinsic = np.zeros((frames, 3, 4), dtype=np.float32)
    extrinsic[:, :3, :3] = np.eye(3, dtype=np.float32)
    extrinsic[:, :3, 3] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    intrinsic = np.zeros((frames, 3, 3), dtype=np.float32)
    intrinsic[:, 0, 0] = 2.0
    intrinsic[:, 1, 1] = 2.0
    intrinsic[:, 0, 2] = width / 2.0
    intrinsic[:, 1, 2] = height / 2.0
    intrinsic[:, 2, 2] = 1.0

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    world_points = np.zeros((frames, height, width, 3), dtype=np.float32)
    for frame in range(frames):
        frame_depth = depth[frame, ..., 0]
        rotation_t = extrinsic[frame, :3, :3].T
        translation = extrinsic[frame, :3, 3]
        fx, fy = intrinsic[frame, 0, 0], intrinsic[frame, 1, 1]
        cx, cy = intrinsic[frame, 0, 2], intrinsic[frame, 1, 2]
        camera_points = np.stack(
            (
                (x - cx) / fx * frame_depth,
                (y - cy) / fy * frame_depth,
                frame_depth,
            ),
            axis=-1,
        )
        world_points[frame] = np.einsum(
            "ij,hwj->hwi",
            rotation_t,
            (camera_points - translation).astype(np.float32),
        ).astype(np.float32)

    np.savez(
        str(staging / "predictions.npz"),
        depth=depth,
        depth_conf=depth_conf,
        images=images,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        world_points_from_depth=world_points,
    )
    (staging / "manifest.txt").write_text(
        "\n".join(basenames) + "\n", encoding="utf-8"
    )


def main() -> int:
    repo_root = Path(
        __import__("os").environ.get(
            "DRONE3D_REPO_ROOT", str(Path(__file__).resolve().parents[3])
        )
    )
    reconstruct_script = repo_root / "third_party" / "vggt-omega" / "reconstruct_omega.py"
    validator_script = repo_root / "third_party" / "vggt-omega" / "result_validator.py"
    assert reconstruct_script.is_file(), reconstruct_script
    assert validator_script.is_file(), validator_script

    with tempfile.TemporaryDirectory(prefix="drone3d-real-smoke-") as temp_text:
        root = Path(temp_text)
        snapshots_root = root / "snapshots"
        snapshots_root.mkdir()
        result_root = root / "results"
        result_root.mkdir()
        checkpoint = root / "checkpoint.pt"
        checkpoint_bytes = b"explicit manual smoke checkpoint"
        checkpoint.write_bytes(checkpoint_bytes)
        snapshot_uri, manifest_sha256 = make_snapshot(snapshots_root)

        config = ProductionConfig(
            supervisor_python=str(Path(sys.executable).resolve()),
            worker_python=str(Path(sys.executable).resolve()),
            reconstruct_script=str(reconstruct_script.resolve()),
            validator_script=str(validator_script.resolve()),
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
            result_root=str(result_root),
            allowed_snapshot_roots=(str(snapshots_root),),
            snapshot_limits=SnapshotSafetyLimits(
                max_frame_count=8,
                max_manifest_entries=64,
                max_metadata_bytes=1024 * 1024,
                max_file_bytes=1024 * 1024,
                max_total_bytes=8 * 1024 * 1024,
                max_scene_id_bytes=128,
                max_relative_path_bytes=1024,
            ),
            resource_limits=ResourceLimits(
                min_mem_available_bytes=1,
                max_swap_used_bytes=sys.maxsize,
                max_swap_growth_bytes=sys.maxsize,
                min_snapshot_free_bytes=1,
                min_result_free_bytes=1,
                swap_sample_window_sec=0.01,
            ),
            image_resolution=4,
            frame_group_size=2,
            max_result_pixels=64,
            max_result_file_bytes=256 * 1024,
            max_result_total_bytes=1024 * 1024,
            max_log_bytes=64 * 1024,
            verifier_timeout_sec=10.0,
            worker_timeout_sec=10.0,
            term_grace_sec=0.5,
            load_mode="max_size",
            require_glb=False,
            show_cameras=True,
            conf_thres=50.0,
            max_points_k=5,
            profile_name="local_docker_test_candidate",
            required_device_sku="docker_test_candidate",
            algorithm_revision="manual-smoke-algorithm-revision",
            ros_revision="manual-smoke-ros-revision",
        )
        planner = ProductionPlanner(
            config,
            token_source=lambda: "realsmoke0001",
            resource_enforcer=passing_resource_enforcer,
        )
        prepared = planner(
            type(
                "Goal",
                (),
                {
                    "snapshot_uri": snapshot_uri,
                    "manifest_sha256": manifest_sha256,
                },
            )()
        )
        print("PLANNER_WORKER_COMMAND:")
        for index, argument in enumerate(prepared.command):
            print(f"  [{index}] {argument}")
        assert "--supervisor_staging" in prepared.command
        assert reconstruct_script.name in prepared.command[6]
        assert validator_script.name not in prepared.command

        # Worker replacement: the real Omega model cannot run in this container
        # (no torch/checkpoint), so the real worker_runner captures a synthetic
        # worker that writes the exact artifact schema the real validator
        # accepts.
        staging = prepared.result_transaction.staging_path
        worker_code = (
            "import sys; from pathlib import Path;"
            "p=Path(sys.argv[1]);"
            "p.joinpath('manifest.txt').write_text('frame_000.png\\nframe_001.png\\n');"
            "print('synthetic worker wrote manifest')"
        )
        run_worker(
            str(staging / "run.log"),
            [
                str(Path(sys.executable).resolve()),
                "-c",
                worker_code,
                str(staging),
            ],
        )
        # First prove the real validator CLI rejects corrupt artifacts before a
        # manifest can exist; only then write the real accepted schema.
        verifier = prepared.verify_completion
        (staging / "predictions.npz").write_bytes(b"corrupt npz payload")
        try:
            verifier(JobFinished("worker_exit", 0, 0.25, 1), threading.Event())
        except CompletionError as error:
            assert error.code == ALGORITHM_FAILED, error.code
            assert "result validator rejected staging" in str(error), error
            print("REAL_VALIDATOR_REJECTED_CORRUPT_NPZ:", error)
        else:
            raise AssertionError("real validator accepted a corrupt NPZ")
        assert not (staging / "result.json").exists()

        write_real_npz(staging, ["frame_000.png", "frame_001.png"])
        verified = verifier(JobFinished("worker_exit", 0, 0.25, 1), threading.Event())
        print("VERIFIED_COMPLETION:", verified)

        result_json = staging / "result.json"
        assert result_json.is_file()
        assert sha256_file(result_json) == verified.result_manifest_sha256
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        assert payload["schema"] == RESULT_SCHEMA
        assert payload["schema_version"] == RESULT_SCHEMA_VERSION
        assert payload["coordinate_state"] == COORDINATE_STATE
        assert payload["checkpoint"]["sha256"] == config.checkpoint_sha256
        assert payload["profile"]["name"] == "local_docker_test_candidate"
        assert payload["profile"]["required_device_sku"] == "docker_test_candidate"

        committed = prepared.result_transaction.commit()
        assert prepared.result_transaction.committed
        assert committed.is_dir()
        final_result = committed / "result.json"
        assert final_result.is_file()
        assert sha256_file(final_result) == verified.result_manifest_sha256
        assert (committed / "predictions.npz").is_file()
        assert not staging.exists()
        print("COMMITTED_RESULT_DIR:", committed)
        print(
            "FINAL_RESULT_JSON_URI:",
            final_result.as_uri(),
        )
        print(
            "FINAL_RESULT_JSON_SHA256:",
            verified.result_manifest_sha256,
        )
        print("REAL_VALIDATOR_COMMIT_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
