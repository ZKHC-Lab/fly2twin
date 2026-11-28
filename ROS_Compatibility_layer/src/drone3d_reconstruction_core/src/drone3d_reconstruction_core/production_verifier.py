"""Production completion verifier: the post-worker checks and the result.json writer.

Runs the application validator as a separate process (cancellable, bounded), checks
run.log and every artifact, and writes result.json atomically. Split from
``production``, which still re-exports every public name.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable

from .preparation import (
    ALGORITHM_FAILED,
    IO_ERROR,
    CompletionError,
    VerificationCancelled,
)
from .reconstruct_action_server import VerifiedCompletion
from .result_transaction import (
    ResultTransaction,
)
from .snapshot_validator import (
    ValidatedSnapshot,
)
from .worker_runner import WORKER_RUN_LOG_MARKER, worker_environment
from ._common import is_finite_number as _finite_number, signal_group as _signal_group
from .production_config import ProductionConfig


RESULT_SCHEMA = "drone3d.reconstruction.result"
RESULT_SCHEMA_VERSION = 1
COORDINATE_STATE = "MODEL_LOCAL_UNALIGNED"
class ProductionCompletionVerifier:
    """取消感知的 worker 后校验与 result.json 编写器。

    可调用契约为适配器的 verify_completion(finished, cancellation_event)
    -> VerifiedCompletion。只有完全校验且清单已写入的暂存目录才返回给
    适配器；事务提交仍由适配器的事务线性化点负责。
    """

    def __init__(
        self,
        *,
        config: ProductionConfig,
        transaction: ResultTransaction,
        snapshot: ValidatedSnapshot,
        selected_paths: tuple[Path, ...],
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = time.sleep,
        poll_interval_sec: float = 0.02,
    ) -> None:
        """构造校验器。

        Args:
            config: 生产配置（提供校验器脚本、超时与字节上限）
            transaction: 结果事务（暂存目录 = 校验输入/输出）
            snapshot: 已验证快照（写入 result.json 的来源信息）
            selected_paths: 选帧的绝对路径元组（顺序即期望的 manifest 顺序）
            popen_factory: 子进程工厂（测试注入点）
            monotonic: 单调时钟
            sleep: 睡眠函数
            poll_interval_sec: 轮询间隔（秒）
        """
        if not isinstance(config, ProductionConfig):
            raise TypeError("config must be ProductionConfig")
        if not isinstance(transaction, ResultTransaction):
            raise TypeError("transaction must be ResultTransaction")
        if not isinstance(snapshot, ValidatedSnapshot):
            raise TypeError("snapshot must be ValidatedSnapshot")
        if not isinstance(selected_paths, tuple) or not selected_paths:
            raise TypeError("selected_paths must be a nonempty tuple")
        if any(not isinstance(path, Path) for path in selected_paths):
            raise TypeError("selected_paths entries must be Path objects")
        if not _finite_number(poll_interval_sec) or poll_interval_sec <= 0:
            raise TypeError("poll_interval_sec must be positive and finite")
        self._config = config
        self._transaction = transaction
        self._snapshot = snapshot
        self._selected_paths = selected_paths
        self._selected_basenames = tuple(path.name for path in selected_paths)
        self._selected_snapshot_indices = tuple(
            snapshot.image_paths.index(path) for path in selected_paths
        )
        self._popen_factory = popen_factory
        self._monotonic = monotonic
        self._sleep = sleep
        self._poll_interval_sec = float(poll_interval_sec)

    def __call__(
        self, finished: Any, cancellation_event: Any
    ) -> VerifiedCompletion:
        """执行完整校验流水线并返回 VerifiedCompletion。

        步骤：取消检查 → 运行独立校验器子进程（限时/限日志/可取消）→
        几何校验 → 资产校验（run.log/manifest.txt/predictions.npz/scene.glb）→
        原子写 result.json → 总字节上限复查。

        Args:
            finished: worker 的 JobFinished 摘要
            cancellation_event: 取消事件（Action 服务端注入）

        Returns:
            校验通过的 VerifiedCompletion

        Raises:
            CompletionError: 校验/资产/写入任一环节失败（带稳定错误码）
            VerificationCancelled: 校验期间收到取消请求
        """
        if not isinstance(cancellation_event, threading.Event):
            raise TypeError("cancellation_event must be a threading.Event")
        _raise_if_cancelled(cancellation_event)

        verifier_started = self._monotonic()
        summary = self._run_validator(cancellation_event)
        verifier_wall_time_sec = max(0.0, self._monotonic() - verifier_started)
        _raise_if_cancelled(cancellation_event)
        width = _summary_int(summary, "width")
        height = _summary_int(summary, "height")
        _validate_summary_geometry(
            summary, len(self._selected_basenames), self._config.max_result_pixels
        )

        artifact_digests, asset_total_bytes = self._verify_staging_assets()
        _raise_if_cancelled(cancellation_event)

        payload = self._result_payload(
            finished,
            width,
            height,
            artifact_digests,
            verifier_wall_time_sec=verifier_wall_time_sec,
        )
        result_path, result_json_digest = _write_result_json_atomically(
            self._transaction.staging_path,
            payload,
            self._config.max_result_file_bytes,
        )
        _raise_if_cancelled(cancellation_event)
        try:
            result_json_size = result_path.lstat().st_size
        except OSError as error:
            raise CompletionError(
                IO_ERROR, "written result.json could not be inspected"
            ) from error
        if asset_total_bytes + result_json_size > self._config.max_result_total_bytes:
            raise CompletionError(
                ALGORITHM_FAILED,
                "result assets and result.json exceed max_result_total_bytes",
            )

        # 用最终（发布后）路径构造 URI，提交后即指向真实产物
        final_manifest_path = self._transaction.final_path / "result.json"
        final_glb_path = self._transaction.final_path / "scene.glb"
        return VerifiedCompletion(
            scene_id=self._snapshot.scene_id,
            result_manifest_uri=final_manifest_path.as_uri(),
            result_manifest_sha256=result_json_digest,
            scene_glb_uri=(
                final_glb_path.as_uri() if self._config.require_glb else ""
            ),
            frame_count=len(self._selected_basenames),
        )

    def _run_validator(self, cancellation_event: Any) -> dict[str, Any]:
        """以独立子进程运行应用校验器并解析其单行 JSON 输出。

        校验器子进程与 worker 同 python（worker_python），其 stdout 必须是
        恰好一行 JSON；stderr 捕获到暂存目录的 .verifier-stderr.txt（限长）。
        超时/取消/日志超限时按进程组终止并回收。
        """
        config = self._config
        command = [
            config.worker_python,
            config.validator_script,
            "--output_dir",
            str(self._transaction.staging_path),
            "--require_glb" if config.require_glb else "--no_require_glb",
            "--max_frames",
            str(config.frame_group_size),
            "--max_pixels",
            str(config.max_result_pixels),
        ]
        # 每个期望的选帧 basename 以 --expected_image_basename 成对传入
        for basename in self._selected_basenames:
            command.append("--expected_image_basename")
            command.append(basename)

        stderr_path = self._transaction.staging_path / ".verifier-stderr.txt"
        stderr_stream = _create_exclusive_stream(stderr_path)
        success = False
        process = None
        try:
            process = self._spawn_validator(command, stderr_stream)
            started = self._monotonic()
            deadline = started + config.verifier_timeout_sec
            stdout = bytearray()
            while True:
                if process.poll() is not None:
                    break
                if cancellation_event.is_set():
                    self._terminate_and_reap(
                        process,
                        stderr_stream=stderr_stream,
                        maximum=config.max_log_bytes,
                    )
                    raise VerificationCancelled()
                now = self._monotonic()
                if now >= deadline:
                    self._terminate_and_reap(
                        process,
                        stderr_stream=stderr_stream,
                        maximum=config.max_log_bytes,
                    )
                    raise CompletionError(
                        ALGORITHM_FAILED, "result verifier exceeded verifier_timeout_sec"
                    )
                readable, _, _ = select.select(
                    [process.stdout], [], [], self._poll_interval_sec
                )
                if readable:
                    try:
                        chunk = os.read(process.stdout.fileno(), 65536)
                    except OSError:
                        chunk = b""
                    if chunk:
                        stdout.extend(chunk)
                        if len(stdout) > config.max_log_bytes:
                            self._terminate_and_reap(
                                process,
                                stderr_stream=stderr_stream,
                                maximum=config.max_log_bytes,
                            )
                            raise CompletionError(
                                ALGORITHM_FAILED,
                                "verifier stdout exceeded the configured bound",
                            )
                _check_stderr_bound(
                    stderr_stream,
                    config.max_log_bytes,
                    process,
                    self._terminate_and_reap,
                )

            try:
                while True:
                    readable, _, _ = select.select(
                        [process.stdout], [], [], self._poll_interval_sec
                    )
                    if cancellation_event.is_set():
                        self._terminate_and_reap(
                            process,
                            stderr_stream=stderr_stream,
                            maximum=config.max_log_bytes,
                        )
                        raise VerificationCancelled()
                    if self._monotonic() >= deadline:
                        self._terminate_and_reap(
                            process,
                            stderr_stream=stderr_stream,
                            maximum=config.max_log_bytes,
                        )
                        raise CompletionError(
                            ALGORITHM_FAILED,
                            "result verifier exceeded verifier_timeout_sec",
                        )
                    if readable:
                        try:
                            chunk = os.read(process.stdout.fileno(), 65536)
                        except OSError as error:
                            self._terminate_and_reap(
                                process,
                                stderr_stream=stderr_stream,
                                maximum=config.max_log_bytes,
                            )
                            raise CompletionError(
                                IO_ERROR, "verifier stdout could not be read"
                            ) from error
                        if not chunk:
                            break  # EOF：子进程已关闭 stdout
                        stdout.extend(chunk)
                        if len(stdout) > config.max_log_bytes:
                            self._terminate_and_reap(
                                process,
                                stderr_stream=stderr_stream,
                                maximum=config.max_log_bytes,
                            )
                            raise CompletionError(
                                ALGORITHM_FAILED,
                                "verifier stdout exceeded the configured bound",
                            )
                    _check_stderr_bound(
                        stderr_stream,
                        config.max_log_bytes,
                        process,
                        self._terminate_and_reap,
                    )
            except OSError as error:
                self._terminate_and_reap(
                    process,
                    stderr_stream=stderr_stream,
                    maximum=config.max_log_bytes,
                )
                raise CompletionError(
                    IO_ERROR, "verifier stdout could not be read"
                ) from error

            _require_stderr_size(stderr_stream, config.max_log_bytes)
            returncode = process.wait()
            if returncode != 0:
                # 非零退出：容忍 ok=false + error 摘要（应用层拒绝）
                summary = _parse_single_json_line(bytes(stdout), require_ok=False)
                if summary.get("ok") is False and isinstance(summary.get("error"), str):
                    raise CompletionError(
                        ALGORITHM_FAILED,
                        f"result validator rejected staging: {summary['error']}",
                    )
                raise CompletionError(
                    ALGORITHM_FAILED,
                    f"result validator failed with malformed output (exit {returncode})",
                )
            summary = _parse_single_json_line(bytes(stdout), require_ok=True)
            _require_stderr_size(stderr_stream, config.max_log_bytes)
            stderr_stream.close()
            try:
                stderr_path.unlink()
            except OSError as error:
                raise CompletionError(
                    IO_ERROR, "verifier stderr capture could not be removed"
                ) from error
            success = True
            return summary
        finally:
            if process is not None and process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
            if not stderr_stream.closed:
                stderr_stream.close()
            if not success:
                # 失败路径不得在暂存目录留下校验捕获文件。适配器持有权威
                # 清理权，这里的最佳努力 unlink 不会掩盖原始失败。
                try:
                    stderr_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def _spawn_validator(self, command: list[str], stderr_stream) -> subprocess.Popen:
        """spawn 校验器子进程（独立进程组，stdout 管道 + stderr 重定向）。"""
        try:
            return self._popen_factory(
                list(command),
                stdout=subprocess.PIPE,
                stderr=stderr_stream,
                start_new_session=True,
                env=worker_environment(),
            )
        except Exception as error:
            raise CompletionError(
                IO_ERROR, f"result verifier could not be spawned: {type(error).__name__}"
            ) from error

    def _terminate_and_reap(
        self,
        process: subprocess.Popen,
        *,
        stderr_stream: Any = None,
        maximum: int | None = None,
    ) -> None:
        """按进程组终止校验器并回收：SIGTERM → 宽限期 → SIGKILL。"""
        process_group = process.pid
        _signal_group(process_group, signal.SIGTERM)
        deadline = self._monotonic() + self._config.term_grace_sec
        while process.poll() is None and self._monotonic() < deadline:
            if stderr_stream is not None and maximum is not None:
                try:
                    # stderr 已超限则立即升级为 SIGKILL
                    if os.fstat(stderr_stream.fileno()).st_size > maximum:
                        _signal_group(process_group, signal.SIGKILL)
                        break
                except OSError:
                    pass
            self._sleep(self._poll_interval_sec)
        if process.poll() is None:
            _signal_group(process_group, signal.SIGKILL)
        try:
            process.wait()
        except BaseException as error:
            raise CompletionError(
                IO_ERROR, f"result verifier could not be reaped: {type(error).__name__}"
            ) from error

    def _verify_staging_assets(self) -> tuple[dict[str, tuple[str, int, str]], int]:
        """校验暂存目录资产：存在性、类型、大小、前缀与 manifest 顺序。

        Returns:
            (digests, total_bytes)：
            digests 为 {文件名: (文件名, 大小, sha256)}，total_bytes 为资产总字节数
        """
        config = self._config
        staging = self._transaction.staging_path
        named_assets = {
            "run.log": _verify_regular_file(
                staging / "run.log",
                config.max_log_bytes,
                require_nonempty=True,
                expected_prefix=WORKER_RUN_LOG_MARKER,
            ),
            "manifest.txt": _verify_regular_file(
                staging / "manifest.txt",
                config.max_result_file_bytes,
                require_nonempty=True,
            ),
            "predictions.npz": _verify_regular_file(
                staging / "predictions.npz",
                config.max_result_file_bytes,
                require_nonempty=True,
            ),
        }
        if config.require_glb:
            named_assets["scene.glb"] = _verify_regular_file(
                staging / "scene.glb",
                config.max_result_file_bytes,
                require_nonempty=True,
            )
        else:
            # 未要求 GLB 时，scene.glb 必须缺席
            glb_path = staging / "scene.glb"
            try:
                if glb_path.exists() or glb_path.is_symlink():
                    raise CompletionError(
                        ALGORITHM_FAILED,
                        "scene.glb must be absent when GLB output is disabled",
                    )
            except OSError as error:
                raise CompletionError(IO_ERROR, "scene.glb could not be inspected") from error

        manifest_bytes = _read_regular_file(
            staging / "manifest.txt", config.max_result_file_bytes
        )
        try:
            entries = manifest_bytes.decode("utf-8", "strict").splitlines()
        except UnicodeDecodeError as error:
            raise CompletionError(
                ALGORITHM_FAILED, "manifest.txt is not valid UTF-8"
            ) from error
        # manifest 必须与选帧顺序完全一致（每行一个 basename）
        if entries != list(self._selected_basenames):
            raise CompletionError(
                ALGORITHM_FAILED,
                "manifest.txt does not match the selected ordered basenames",
            )

        total_bytes = sum(info.st_size for info in named_assets.values())
        if total_bytes > config.max_result_total_bytes:
            raise CompletionError(
                ALGORITHM_FAILED,
                "result assets exceed max_result_total_bytes",
            )

        digests = {}
        for name, info in named_assets.items():
            path = staging / name
            digests[name] = (name, info.st_size, _hash_verified_file(path, info))
        return digests, total_bytes

    def _result_payload(
        self,
        finished: Any,
        width: int,
        height: int,
        digests: dict[str, tuple[str, int, str]],
        *,
        verifier_wall_time_sec: float,
    ) -> dict[str, Any]:
        """组装 result.json 的完整载荷（schema v1）。"""
        config = self._config
        final_path = self._transaction.final_path
        artifacts = {}
        for name, size, digest in digests.values():
            artifacts[name] = {
                "uri": (final_path / name).as_uri(),
                "size": size,
                "sha256": digest,
            }
        def _stamp_object(stamp):
            return (
                {"sec": stamp.sec, "nsec": stamp.nsec}
                if stamp is not None
                else None
            )

        selected_frames = []
        for selected_position, path in enumerate(self._selected_paths):
            original_index = self._selected_snapshot_indices[selected_position]
            selected_frames.append(
                {
                    "index": original_index,
                    "selected_position": selected_position,
                    "basename": path.name,
                    "path": str(path),
                    "stamp": _stamp_object(
                        self._snapshot.frame_stamps[original_index]
                        if original_index < len(self._snapshot.frame_stamps)
                        else None
                    ),
                    "pose_valid": (
                        self._snapshot.pose_valid[original_index]
                        if original_index < len(self._snapshot.pose_valid)
                        else None
                    ),
                    "pose_delta_ms": (
                        self._snapshot.pose_delta_ms[original_index]
                        if original_index < len(self._snapshot.pose_delta_ms)
                        else None
                    ),
                    "odometry_gap_ms": (
                        self._snapshot.odometry_gap_ms[original_index]
                        if original_index < len(self._snapshot.odometry_gap_ms)
                        else None
                    ),
                }
            )
        model_frame = f"model_local/{int(finished.job_id)}"
        return {
            "schema": RESULT_SCHEMA,
            "schema_version": RESULT_SCHEMA_VERSION,
            "scene_id": self._snapshot.scene_id,
            "job_id": int(finished.job_id),
            "patch_id": self._transaction.final_path.name,
            "snapshot": {
                "uri": self._snapshot.canonical_root.as_uri(),
                "sha256": self._snapshot.manifest_sha256,
                "frame_count": self._snapshot.frame_count,
                "schema_version": self._snapshot.snapshot_version,
                "camera_name": self._snapshot.camera_name,
                "image_topic": self._snapshot.image_topic,
                "camera_role": self._snapshot.camera_role,
                "camera_frame_id": self._snapshot.camera_frame_id,
                "first_stamp": _stamp_object(self._snapshot.first_stamp),
                "last_stamp": _stamp_object(self._snapshot.last_stamp),
                "has_camera_info": self._snapshot.has_camera_info,
                "has_odometry": self._snapshot.has_odometry,
            },
            "selected_frames": selected_frames,
            "frame_stamps_available": True,
            "actual_frame_count": len(selected_frames),
            "image_geometry": {"width": width, "height": height},
            "camera_role": self._snapshot.camera_role,
            "camera_topic": self._snapshot.image_topic,
            "camera_frame_id": self._snapshot.camera_frame_id,
            "calibration_digest": self._snapshot.camera_info_digest,
            "checkpoint": {
                "path": config.checkpoint_path,
                "sha256": config.checkpoint_sha256,
            },
            "algorithm_revision": config.algorithm_revision,
            "ros_revision": config.ros_revision,
            "profile": {
                "backend": config.backend,
                "mode": "independent",
                "name": config.profile_name,
                "image_resolution": config.image_resolution,
                "frame_group_size": config.frame_group_size,
                "aggregator_frame_chunk_size": config.aggregator_frame_chunk_size,
                "attention_backend": config.attention_backend,
                "runtime_memory_guard_enabled": config.runtime_memory_guard_enabled,
                "preserve_failed_worker_staging": config.preserve_failed_worker_staging,
                "load_mode": config.load_mode,
                "require_glb": config.require_glb,
                "show_cameras": config.show_cameras,
                "conf_thres": float(config.conf_thres),
                "max_points_k": config.max_points_k,
                "max_result_pixels": config.max_result_pixels,
                "max_result_file_bytes": config.max_result_file_bytes,
                "max_result_total_bytes": config.max_result_total_bytes,
                "max_log_bytes": config.max_log_bytes,
                "verifier_timeout_sec": float(config.verifier_timeout_sec),
                "worker_timeout_sec": float(config.worker_timeout_sec),
                "term_grace_sec": float(config.term_grace_sec),
                "warm_worker_enabled": bool(config.warm_worker_enabled),
            },
            "resource_limits": {
                "min_mem_available_bytes": config.resource_limits.min_mem_available_bytes,
                "max_swap_used_bytes": config.resource_limits.max_swap_used_bytes,
                "max_swap_growth_bytes": config.resource_limits.max_swap_growth_bytes,
                "min_snapshot_free_bytes": config.resource_limits.min_snapshot_free_bytes,
                "min_result_free_bytes": config.resource_limits.min_result_free_bytes,
                "swap_sample_window_sec": float(
                    config.resource_limits.swap_sample_window_sec
                ),
            },
            "stage_timing": {
                "wall_time_sec": float(finished.elapsed_sec),
                "worker_wall_time_sec": float(finished.elapsed_sec),
                "verifier_wall_time_sec": float(verifier_wall_time_sec),
                "commit_wall_time_sec": None,
            },
            "worker_exit": {
                "cause": str(finished.cause),
                "returncode": int(finished.returncode),
            },
            "coordinate_state": COORDINATE_STATE,
            "model_frame": model_frame,
            "metric_scale_valid": False,
            "map_alignment_valid": False,
            "artifacts": artifacts,
        }


def _raise_if_cancelled(cancellation_event: Any) -> None:
    """取消检查点：事件已置位则立即抛出 VerificationCancelled。"""
    if cancellation_event.is_set():
        raise VerificationCancelled()


def _create_exclusive_stream(path: Path) -> Any:
    """以 O_EXCL|O_NOFOLLOW 独占创建 stderr 捕获文件。"""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except OSError as error:
        raise CompletionError(IO_ERROR, "verifier stderr file could not be created") from error
    return os.fdopen(descriptor, "wb", buffering=0)


def _check_stderr_bound(
    stream: Any,
    maximum: int,
    process: subprocess.Popen,
    terminate: Callable[..., None],
) -> None:
    """运行期检查 stderr 捕获文件是否超限，超限则终止并抛错。"""
    try:
        if os.fstat(stream.fileno()).st_size > maximum:
            terminate(
                process,
                stderr_stream=stream,
                maximum=maximum,
            )
            raise CompletionError(
                ALGORITHM_FAILED, "verifier stderr exceeded the configured bound"
            )
    except CompletionError:
        raise
    except OSError:
        pass


def _require_stderr_size(stream: Any, maximum: int) -> None:
    """进程退出后强制校验 stderr 捕获文件未超限。"""
    try:
        size = os.fstat(stream.fileno()).st_size
    except OSError as error:
        raise CompletionError(
            IO_ERROR, "verifier stderr size could not be inspected"
        ) from error
    if size > maximum:
        raise CompletionError(
            ALGORITHM_FAILED, "verifier stderr exceeded the configured bound"
        )


def _summary_int(summary: dict[str, Any], key: str) -> int:
    """从校验器摘要读取正整数整数字段。"""
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CompletionError(
            ALGORITHM_FAILED, f"verifier summary field {key!r} is not positive"
        )
    return value


def _validate_summary_geometry(
    summary: dict[str, Any], frame_count: int, max_pixels: int
) -> None:
    """校验摘要几何：帧数匹配且像素预算不超限。"""
    if summary.get("frame_count") != frame_count:
        raise CompletionError(
            ALGORITHM_FAILED, "verifier frame count does not match the launch"
        )
    width = _summary_int(summary, "width")
    height = _summary_int(summary, "height")
    if width * height > max_pixels:
        raise CompletionError(
            ALGORITHM_FAILED, "verifier image geometry exceeds max_result_pixels"
        )


def _parse_single_json_line(data: bytes, *, require_ok: bool) -> dict[str, Any]:
    """严格解析校验器 stdout：恰好一行 JSON 对象且含 bool ok 字段。"""
    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise CompletionError(
            ALGORITHM_FAILED, "verifier stdout is not valid UTF-8"
        ) from error
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0].strip():
        raise CompletionError(
            ALGORITHM_FAILED, "verifier stdout must be exactly one JSON line"
        )
    try:
        payload = json.loads(
            lines[0], object_pairs_hook=_reject_duplicate_json_fields
        )
    except CompletionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CompletionError(
            ALGORITHM_FAILED, f"verifier stdout is not strict JSON: {type(error).__name__}"
        ) from error
    if not isinstance(payload, dict):
        raise CompletionError(ALGORITHM_FAILED, "verifier stdout JSON must be an object")
    if type(payload.get("ok")) is not bool:
        raise CompletionError(ALGORITHM_FAILED, "verifier summary must contain a bool ok field")
    if payload["ok"] is not require_ok:
        raise CompletionError(ALGORITHM_FAILED, "verifier summary ok field is unexpected")
    return payload


def _reject_duplicate_json_fields(pairs):
    """JSON 对象钩子：拒绝重复字段（严格 JSON 语义）。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise CompletionError(ALGORITHM_FAILED, "verifier JSON has duplicate fields")
        result[key] = value
    return result


def _open_identity_checked(path: Path, before) -> tuple[int, Any]:
    """打开文件并校验打开前后 inode 身份一致（防替换竞态）。"""
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise CompletionError(IO_ERROR, f"asset cannot be opened: {path.name}") from error
    opened = os.fstat(descriptor)
    if (
        opened.st_dev != before.st_dev
        or opened.st_ino != before.st_ino
        or opened.st_nlink != 1
    ):
        os.close(descriptor)
        raise CompletionError(
            ALGORITHM_FAILED, f"asset identity changed before opening: {path.name}"
        )
    return descriptor, opened


def _assert_unchanged_after_read(path: Path, before, after) -> None:
    """读取结束后校验文件身份与元数据未变（防并发替换）。"""
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_nlink != 1
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
    ):
        raise CompletionError(
            ALGORITHM_FAILED, f"asset changed while reading: {path.name}"
        )


def _verify_regular_file(
    path: Path,
    maximum: int,
    *,
    require_nonempty: bool,
    expected_prefix: bytes | None = None,
):
    """校验资产为可信普通文件：非软链、非硬链接、大小有界、可选前缀。"""
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise CompletionError(ALGORITHM_FAILED, f"required asset is missing: {path.name}") from error
    except OSError as error:
        raise CompletionError(IO_ERROR, f"required asset cannot be inspected: {path.name}") from error
    if stat.S_ISLNK(info.st_mode):
        raise CompletionError(ALGORITHM_FAILED, f"asset must not be a symlink: {path.name}")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CompletionError(ALGORITHM_FAILED, f"asset is not a trusted regular file: {path.name}")
    if info.st_size <= 0 and require_nonempty:
        raise CompletionError(ALGORITHM_FAILED, f"asset is empty: {path.name}")
    if info.st_size > maximum:
        raise CompletionError(ALGORITHM_FAILED, f"asset exceeds configured bound: {path.name}")
    if expected_prefix is not None:
        descriptor, _ = _open_identity_checked(path, info)
        try:
            prefix = os.read(descriptor, len(expected_prefix))
        except OSError as error:
            raise CompletionError(IO_ERROR, f"asset cannot be read: {path.name}") from error
        finally:
            os.close(descriptor)
        if prefix != expected_prefix:
            raise CompletionError(ALGORITHM_FAILED, f"asset has an unexpected prefix: {path.name}")
    return info


def _read_regular_file(path: Path, maximum: int) -> bytes:
    """身份校验后读取整个文件（限长），并确认读取期间未被替换。"""
    info = _verify_regular_file(path, maximum, require_nonempty=True)
    descriptor, _ = _open_identity_checked(path, info)
    try:
        data = b""
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            data += block
            if len(data) > maximum:
                raise CompletionError(
                    ALGORITHM_FAILED, f"asset exceeds configured bound: {path.name}"
                )
        after = os.fstat(descriptor)
    except CompletionError:
        raise
    except OSError as error:
        raise CompletionError(IO_ERROR, f"asset cannot be read: {path.name}") from error
    finally:
        os.close(descriptor)
    _assert_unchanged_after_read(path, info, after)
    return data


def _hash_verified_file(path: Path, before) -> str:
    """身份校验后流式计算 SHA-256，并确认读取期间未被替换。"""
    digest = hashlib.sha256()
    descriptor, _ = _open_identity_checked(path, before)
    try:
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise CompletionError(IO_ERROR, f"asset cannot be hashed: {path.name}") from error
    finally:
        os.close(descriptor)
    _assert_unchanged_after_read(path, before, after)
    return digest.hexdigest()


def _write_result_json_atomically(
    staging_path: Path, payload: dict[str, Any], maximum: int
) -> tuple[Path, str]:
    """原子写 result.json：临时文件 + fsync + replace + 父目录 fsync。

    Returns:
        (result_path, sha256 摘要)
    """
    result_path = staging_path / "result.json"
    try:
        # 暂存目录内不允许预先存在 result.json（防覆盖/防重复发布）
        if result_path.exists() or result_path.is_symlink():
            raise CompletionError(
                ALGORITHM_FAILED, "result.json already exists in staging"
            )
    except OSError as error:
        raise CompletionError(IO_ERROR, "result.json could not be inspected") from error

    descriptor, temporary = tempfile.mkstemp(
        prefix=".result.json-", dir=str(staging_path)
    )
    try:
        data = (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        if len(data) > maximum:
            raise CompletionError(
                ALGORITHM_FAILED, "result.json exceeds max_result_file_bytes"
            )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, str(result_path))
        temporary = None
    except CompletionError:
        raise
    except OSError as error:
        raise CompletionError(IO_ERROR, "result.json could not be written atomically") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass

    _fsync_directory(staging_path)
    try:
        info = result_path.lstat()
    except OSError as error:
        raise CompletionError(IO_ERROR, "written result.json cannot be inspected") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CompletionError(
            ALGORITHM_FAILED, "written result.json is not a trusted regular file"
        )
    return result_path, _hash_verified_file(result_path, info)


def _fsync_directory(path: Path) -> None:
    """fsync 目录条目（确保发布/写入持久化）。"""
    flags = os.O_RDONLY | os.O_DIRECTORY
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise CompletionError(IO_ERROR, "result directory could not be fsynced") from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise CompletionError(IO_ERROR, "result directory could not be fsynced") from error
    finally:
        os.close(descriptor)
