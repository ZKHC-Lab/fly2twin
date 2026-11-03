"""Warm worker pool: supervisor-side lifecycle for one persistent CUDA worker.

The warm process loads VGGT-Omega once and serves reconstruction jobs over
stdin/stdout JSON lines. This module only owns process/pipes; formal result
validation, resource gates and transaction semantics remain in production.py.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import stat
import subprocess
import threading
import time
from typing import Sequence

from .worker_runner import worker_environment

_WARM_POOL_LOG_HEADER = b"drone3d warm worker pool started\n"


class WarmWorkerUnavailableError(RuntimeError):
    """Warm worker cannot serve a job (spawn/died/timeout/cancelled)."""

    def __init__(self, reason: str, *, returncode: int | None = None) -> None:
        super().__init__(f"warm worker unavailable: {reason}")
        self.reason = reason
        self.returncode = returncode


class WarmWorkerBusyError(RuntimeError):
    """Another job is already in flight on the single warm worker."""


class WarmJobFailedError(RuntimeError):
    """The warm worker served the job and reported ok=false."""


class WarmWorkerPool:
    """Maintain at most one persistent warm worker process."""

    def __init__(
        self,
        warm_command: Sequence[str],
        *,
        pool_log_path: str,
        popen_factory=subprocess.Popen,
        thread_factory=threading.Thread,
        signal_group=None,
        monotonic=time.monotonic,
    ) -> None:
        if not warm_command or any(
            not isinstance(part, str) or not part or "\x00" in part
            for part in warm_command
        ):
            raise ValueError("warm_command must contain nonempty strings without NUL")
        self._command = tuple(warm_command)
        self._pool_log_path = pool_log_path
        self._popen_factory = popen_factory
        self._thread_factory = thread_factory
        self._signal_group = signal_group or _signal_group
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._process_group: int | None = None
        self._stdout_reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._responses: queue.Queue = queue.Queue()
        self._dead = False
        self._in_flight = False

    def is_busy(self) -> bool:
        with self._lock:
            return self._in_flight

    def worker_pid(self):
        with self._lock:
            return self._process.pid if self._process is not None and self._process.poll() is None else None

    def prepare(self) -> None:
        """Spawn the persistent process if absent; raise on spawn failure.

        The process starts asynchronously (model load happens when the first
        job arrives), so this only proves that the executable could be spawned.
        """
        with self._lock:
            self._ensure_process()

    def _ensure_process(self):
        if self._process is not None and self._process.poll() is None:
            return
        self._kill_group()
        self._join_readers()
        # 丢弃已死亡进程遗留在队列里的响应；job_id 单调递增，任何残留
        # 都只可能属于旧进程，绝不能与新进程的响应混用。
        while True:
            try:
                self._responses.get_nowait()
            except queue.Empty:
                break
        try:
            _prepare_pool_log(self._pool_log_path)
        except OSError as error:
            raise WarmWorkerUnavailableError(
                "log_failed", returncode=None
            ) from error
        try:
            process = self._popen_factory(
                list(self._command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=True,
                env=worker_environment(),
            )
        except Exception as error:
            raise WarmWorkerUnavailableError(
                "spawn_failed", returncode=None
            ) from error
        self._process = process
        self._process_group = process.pid
        self._dead = False

        def read_stdout():
            assert self._process is not None
            for raw in self._process.stdout:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and "job_id" in payload:
                    self._responses.put(payload)

        def read_stderr():
            assert self._process is not None
            try:
                with open(self._pool_log_path, "ab") as stream:
                    for raw in self._process.stderr:
                        stream.write(raw)
                        stream.flush()
            except (OSError, ValueError):
                pass

        self._stdout_reader = self._thread_factory(target=read_stdout, daemon=True)
        self._stderr_reader = self._thread_factory(target=read_stderr, daemon=True)
        self._stdout_reader.start()
        self._stderr_reader.start()

    def _join_readers(self):
        if self._stdout_reader is not None and self._stdout_reader.is_alive():
            self._stdout_reader.join(timeout=2.0)
        if self._stderr_reader is not None and self._stderr_reader.is_alive():
            self._stderr_reader.join(timeout=2.0)

    def _kill_group(self):
        process = self._process
        group = self._process_group
        if group is None:
            return
        try:
            self._signal_group(group, signal.SIGTERM)
        except BaseException:
            pass
        deadline = time.monotonic() + 5.0
        while process is not None and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if process is not None and process.poll() is None:
            try:
                self._signal_group(group, signal.SIGKILL)
            except BaseException:
                pass
            try:
                process.wait(timeout=5.0)
            except BaseException:
                pass

    def submit(self, job_payload: dict, *, timeout_sec: float, cancel_event) -> dict:
        """Serve one job on the warm worker and return its response JSON.

        Raises:
            WarmWorkerBusyError: another job is already in flight
            WarmWorkerUnavailableError: process died, timed out or was cancelled
            WarmJobFailedError: worker answered ok=false
        """
        job_id = int(job_payload["job_id"])
        with self._lock:
            if self._in_flight:
                raise WarmWorkerBusyError("warm worker already busy")
            self._in_flight = True
        try:
            self._ensure_process()
            if self._process is None or self._process.poll() is not None:
                self._dead = True
                raise WarmWorkerUnavailableError(
                    "worker_died",
                    returncode=self._process.poll() if self._process is not None else None,
                )
            try:
                self._process.stdin.write(
                    (json.dumps(job_payload, sort_keys=True) + "\n").encode("utf-8")
                )
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                self._dead = True
                raise WarmWorkerUnavailableError(
                    "stdin_failed",
                    returncode=self._process.poll(),
                ) from error
            deadline = self._monotonic() + float(timeout_sec)
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._dead = True
                    self._kill_group()
                    returncode = self._process.poll()
                    raise WarmWorkerUnavailableError(
                        "cancelled", returncode=returncode
                    )
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    self._dead = True
                    self._kill_group()
                    returncode = self._process.poll()
                    raise WarmWorkerUnavailableError(
                        "timeout", returncode=returncode
                    )
                response = self._take_response(job_id, wait_sec=min(0.2, remaining))
                if response is not None:
                    if not response.get("ok"):
                        raise WarmJobFailedError(
                            str(response.get("error", "warm job failed"))
                        )
                    return response
                if self._process.poll() is not None:
                    self._dead = True
                    # 进程死亡后尽力再消费一次响应（可能已进入队列）
                    response = self._take_response(job_id, wait_sec=0.0)
                    if response is not None:
                        if not response.get("ok"):
                            raise WarmJobFailedError(
                                str(response.get("error", "warm job failed"))
                            )
                        return response
                    raise WarmWorkerUnavailableError(
                        "worker_died", returncode=self._process.poll()
                    )
        finally:
            with self._lock:
                self._in_flight = False
            if self._dead:
                self._kill_group()

    def _take_response(self, job_id: int, *, wait_sec: float) -> dict | None:
        """Return the matching queued response; unrelated ones are requeued."""
        deadline = self._monotonic() + max(0.0, wait_sec)
        while True:
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                return None
            try:
                response = self._responses.get(timeout=min(0.05, remaining))
            except queue.Empty:
                return None
            if int(response.get("job_id", -1)) != job_id:
                self._responses.put(response)
                continue
            return response

    def shutdown(self):
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.stdin.write(b'{"action": "shutdown"}\n')
                process.stdin.flush()
                process.stdin.close()
            except BaseException:
                pass
        self._kill_group()
        self._join_readers()
        if process is not None:
            if process.poll() is None:
                raise RuntimeError("resident worker termination was not proven")
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        self._process = None
        self._process_group = None


def _prepare_pool_log(log_path: str) -> None:
    """Reset (or create) the pool log as one trusted regular file."""
    if not isinstance(log_path, str) or not log_path or "\x00" in log_path:
        raise ValueError("pool log path must be a nonempty string without NUL")
    path = os.path.abspath(log_path)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        info = None
    except OSError:
        raise
    if info is not None:
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise OSError("pool log exists but is not a trusted regular file")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, _WARM_POOL_LOG_HEADER)
    finally:
        os.close(descriptor)


def _signal_group(group: int, sig: int) -> None:
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            os.kill(group, sig)
        except OSError:
            pass
