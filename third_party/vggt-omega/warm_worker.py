#!/usr/bin/env python3
"""Warm reconstruction worker for the single-NX continuous line.

Loads the VGGT-Omega checkpoint once, then serves reconstruction jobs on
stdin/stdout as newline-delimited JSON. stdout is reserved exclusively for the
job protocol; all human logs are redirected to the configured --log-path and to
each job's <output_dir>/run.log.

Protocol
--------
Request line (one JSON object per job):
  {"job_id": 1, "images_dir": "...", "output_dir": "...",
   "input_images": ["/abs/a.png", ...], "image_resolution": 128,
   "load_mode": "max_size", "conf_thres": 50.0, "max_points_k": 5,
   "no_glb": true, "no_cam": true, "supervisor_staging": true}
Response line (exactly one per job):
  {"job_id": 1, "ok": true, "elapsed_sec": 37.0, "peak_gpu_mb": 5602.0}
  {"job_id": 1, "ok": false, "error": "..."}
Shutdown line: {"action": "shutdown"} or EOF.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import torch  # noqa: E402

from reconstruct_omega import (  # noqa: E402
    _run_reconstruction,
    _supervisor_output_directory,
    load_model,
)

# Must match worker_runner.WORKER_RUN_LOG_MARKER so the existing formal gate
# recognises warm-worker run.log as wrapper-generated.
WORKER_RUN_LOG_MARKER = "drone3d worker wrapper started\n"

# 捕获真实 stdout 管道句柄：后续 sys.stdout 重定向不得污染协议通道。
_PROTOCOL_STDOUT = sys.__stdout__


def _emit(payload: dict) -> None:
    """Write one protocol JSON line to the original stdout."""
    _PROTOCOL_STDOUT.write(json.dumps(payload, sort_keys=True) + "\n")
    _PROTOCOL_STDOUT.flush()


def _namespace_from_job(job: dict, checkpoint: str, checkpoint_sha256: str):
    return argparse.Namespace(
        images_dir=str(job["images_dir"]),
        output_dir=str(job["output_dir"]),
        input_image=[str(path) for path in job["input_images"]],
        image_resolution=int(job.get("image_resolution", 0)),
        frame_group_size=int(job.get("frame_group_size", 0)),
        load_mode=str(job.get("load_mode", "max_size")),
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        conf_thres=float(job.get("conf_thres", 50.0)),
        max_points_k=int(job.get("max_points_k", 5)),
        no_glb=bool(job.get("no_glb", True)),
        no_cam=bool(job.get("no_cam", True)),
        supervisor_staging=True,
    )


def _open_job_log(output_dir: Path):
    """Exclusively create <output_dir>/run.log, mirroring worker_runner safety."""
    parent = str(output_dir)
    path = Path(parent) / "run.log"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags, 0o600)
    return os.fdopen(descriptor, "w", encoding="utf-8", buffering=1)


class _ElapsedLines:
    """Prefix each run.log line with seconds since the job arrived (stage timing on the NX)."""

    def __init__(self, stream, started: float):
        self._stream = stream
        self._started = started
        self._line_start = True

    def write(self, text: str) -> int:
        for part in text.splitlines(keepends=True):
            if self._line_start:
                self._stream.write("[%7.2f s] " % (time.monotonic() - self._started))
            self._stream.write(part)
            self._line_start = part.endswith("\n")
        return len(text)

    def flush(self) -> None:
        self._stream.flush()


def _run_one_job(model, job: dict, checkpoint: str, checkpoint_sha256: str) -> dict:
    job_id = int(job["job_id"])
    started = time.monotonic()
    output_dir = Path(str(job["output_dir"]))
    try:
        if not job.get("supervisor_staging", True):
            raise ValueError("warm worker only accepts supervisor staging outputs")
        # 输出目录必须是监督进程拥有的规范化真实目录
        _supervisor_output_directory(str(output_dir))
        args = _namespace_from_job(job, checkpoint, checkpoint_sha256)
        try:
            run_log = _open_job_log(output_dir)
        except OSError as error:
            raise RuntimeError(f"run.log could not be created: {type(error).__name__}") from error
        with run_log:
            run_log.write(WORKER_RUN_LOG_MARKER)
            run_log.flush()
            try:
                with contextlib.redirect_stdout(_ElapsedLines(run_log, started)):
                    _run_reconstruction(args, str(output_dir), model=model)
                    print("job finished")
            except BaseException:
                traceback.print_exc(file=run_log)
                run_log.flush()
                raise
            run_log.flush()
            os.fsync(run_log.fileno())
        peak_gpu_mb = float(torch.cuda.max_memory_allocated()) / (1024 ** 2)
        return {
            "job_id": job_id,
            "ok": True,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "peak_gpu_mb": round(peak_gpu_mb, 1),
        }
    except BaseException as error:
        return {
            "job_id": job_id,
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _serve(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint_sha256", required=True)
    parser.add_argument("--aggregator-frame-chunk-size", type=int, default=50)
    parser.add_argument("--attention-backend", choices=("auto", "flash_attn_v1"), default="auto")
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args(argv)

    if args.log_path:
        startup_log = open(args.log_path, "a", encoding="utf-8", buffering=1)
        sys.stdout = startup_log
        sys.stderr = startup_log
    else:
        startup_log = None

    model = load_model(
        args.checkpoint,
        "cuda",
        expected_sha256=args.checkpoint_sha256,
        aggregator_frame_chunk_size=args.aggregator_frame_chunk_size,
        attention_backend=args.attention_backend,
    )
    print("warm worker model loaded; waiting for jobs", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError:
            # 协议损坏：回显可辨识的失败后继续，避免单条坏行杀死驻留进程
            _emit({"job_id": -1, "ok": False, "error": "invalid protocol JSON"})
            continue
        if not isinstance(job, dict):
            _emit({"job_id": -1, "ok": False, "error": "protocol JSON must be an object"})
            continue
        if job.get("action") == "shutdown":
            break
        if "job_id" not in job:
            _emit({"job_id": -1, "ok": False, "error": "missing job_id"})
            continue
        response = _run_one_job(model, job, args.checkpoint, args.checkpoint_sha256)
        _emit(response)

    if startup_log is not None:
        startup_log.close()
    return 0


def main(argv=None) -> int:
    resource_root = os.environ.get("DRONE3D_RUNTIME_PYTHON_ROOT")
    if resource_root is None:
        resource_root = str(Path(__file__).resolve().parents[2] / "ROS_Compatibility_layer/src/drone3d_runtime/src")
    if resource_root not in sys.path:
        sys.path.append(resource_root)
    from drone3d_resources import gpu_lease
    with gpu_lease():
        return _serve(argv)


if __name__ == "__main__":
    raise SystemExit(main())
