"""Production planner: a ReconstructSegment goal becomes one immutable PreparedLaunch.

Validates the snapshot, admits resources, selects frames deterministically, allocates
the supervisor-owned result transaction and builds the exact worker argv. Split from
``production``, which still re-exports every public name.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import secrets
import time
from typing import Any, Callable

from .preparation import (
    INPUT_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_GOAL,
    IO_ERROR,
    PreparationError,
)
from .reconstruct_action_server import PreparedLaunch
from .resource_gate import (
    ResourceBusyError,
    ResourceDecision,
    ResourceGateError,
    enforce_resources,
)
from .result_transaction import (
    ResultTransaction,
    ResultTransactionError,
    allocate_result_transaction,
)
from .snapshot_validator import (
    SnapshotInvalidError,
    SnapshotMissingError,
    ValidatedSnapshot,
    validate_snapshot,
)
from .production_config import _TOKEN_RE, ProductionConfig, ProductionConfigError
from .production_verifier import ProductionCompletionVerifier


# 快照内部错误码（配置问题）与 IO 错误码（文件系统问题）的映射集
_SNAPSHOT_INTERNAL_CODES = frozenset(("MISSING_ALLOWED_ROOT",))
_SNAPSHOT_IO_CODES = frozenset(("FILESYSTEM_ERROR",))

def _selected_indices(frame_count: int, frame_group_size: int) -> tuple[int, ...]:
    """均匀选择精确有序下标（对齐历史采样器语义）。

    仅在 frame_count > frame_group_size 时调用；等价于
    round(linspace(0, n-1, k)) 后取 unique 的去重升序结果。
    """
    if frame_count <= frame_group_size:
        raise ValueError("selection is only required for oversized inputs")
    if frame_group_size == 1:
        return (0,)
    step = (frame_count - 1) / (frame_group_size - 1)
    selected = set()
    for index in range(frame_group_size):
        selected.add(int(round(index * step)))
    return tuple(sorted(selected))


class ProductionPlanner:
    """从 ReconstructSegment Goal 构造一个不可变 PreparedLaunch。"""

    def __init__(
        self,
        config: ProductionConfig,
        *,
        token_source: Callable[[], str] | None = None,
        resource_enforcer: Callable[..., ResourceDecision] = enforce_resources,
        transaction_factory: Callable[..., ResultTransaction] = allocate_result_transaction,
    ) -> None:
        """构造规划器。

        Args:
            config: 已验证的生产配置
            token_source: 结果目录 token 生成器（测试注入点）
            resource_enforcer: 资源准入执行器（测试注入点）
            transaction_factory: 结果事务工厂（测试注入点）
        """
        if not isinstance(config, ProductionConfig):
            raise TypeError("config must be ProductionConfig")
        self._config = config
        self._token_source = token_source or (lambda: secrets.token_hex(8))
        self._enforce_resources = resource_enforcer
        self._allocate_transaction = transaction_factory
        self._worker_runner_script = _worker_runner_script_path()

    @property
    def config(self) -> ProductionConfig:
        return self._config

    def __call__(
        self,
        goal: Any,
        *,
        image_resolution: int | None = None,
        worker_timeout_sec: float | None = None,
    ) -> PreparedLaunch:
        """目标入口：快照校验 → 资源准入 → 选帧 → 事务 → worker 命令。

        Args:
            goal: ReconstructSegmentGoal 消息
            image_resolution: 每目标推理分辨率覆盖（None = 沿用 profile）
            worker_timeout_sec: 每目标 worker 超时覆盖（None = 沿用 profile）

        Returns:
            绑定好完成校验器的 PreparedLaunch

        Raises:
            PreparationError: 任一准备步骤失败（带稳定错误码）
            ResourceBusyError: 资源准入拒绝（透传，由 Action 层映射为 BUSY）
        """
        config = self._config
        if image_resolution is not None or worker_timeout_sec is not None:
            if image_resolution is not None and (
                isinstance(image_resolution, bool)
                or not isinstance(image_resolution, int)
                or image_resolution <= 0
            ):
                raise PreparationError(
                    INVALID_GOAL, "image_resolution must be a positive integer"
                )
            if worker_timeout_sec is not None and (
                isinstance(worker_timeout_sec, bool)
                or not isinstance(worker_timeout_sec, (int, float))
                or not math.isfinite(float(worker_timeout_sec))
                or float(worker_timeout_sec) <= 0
            ):
                raise PreparationError(
                    INVALID_GOAL,
                    "worker_timeout_sec must be positive and finite",
                )
            try:
                config = replace(
                    self._config,
                    image_resolution=(
                        int(image_resolution)
                        if image_resolution is not None
                        else self._config.image_resolution
                    ),
                    worker_timeout_sec=(
                        float(worker_timeout_sec)
                        if worker_timeout_sec is not None
                        else self._config.worker_timeout_sec
                    ),
                )
            except (ProductionConfigError, TypeError, ValueError) as error:
                raise PreparationError(
                    INVALID_GOAL,
                    f"per-goal runtime override invalid: {error}",
                ) from error
        timings = {}
        step = time.monotonic()
        try:
            snapshot = validate_snapshot(
                goal.snapshot_uri,
                goal.manifest_sha256,
                config.allowed_snapshot_roots,
                config.snapshot_limits,
            )
        except SnapshotMissingError as error:
            if error.code in _SNAPSHOT_INTERNAL_CODES:
                raise PreparationError(
                    INTERNAL_ERROR, f"snapshot admission configuration is broken: {error.code}"
                )
            raise PreparationError(INPUT_NOT_FOUND, f"snapshot input missing: {error.code}")
        except SnapshotInvalidError as error:
            raise PreparationError(_map_snapshot_invalid_code(error.code), str(error))
        except (AttributeError, TypeError):
            raise PreparationError(INVALID_GOAL, "goal is missing snapshot fields")

        timings["validate_snapshot"] = time.monotonic() - step
        step = time.monotonic()
        try:
            self._enforce_resources(
                config.resource_limits,
                str(snapshot.canonical_root),
                config.result_root,
            )
        except ResourceBusyError:
            raise  # 资源忙是合法的可重试拒绝，直接透传给 Action 层
        except ResourceGateError as error:
            raise PreparationError(
                INTERNAL_ERROR, f"resource admission probe failed: {error}"
            )
        except Exception as error:
            raise PreparationError(
                INTERNAL_ERROR,
                f"resource admission probe failed: {type(error).__name__}",
            )

        timings["resource_gate"] = time.monotonic() - step
        step = time.monotonic()
        # 确定性选帧：帧数超限时均匀抽帧到 frame_group_size
        selected = _select_snapshot_frames(snapshot, config.frame_group_size)
        from .backend_spec import validate_independent_request
        try:
            validate_independent_request(config.backend, len(selected), config.image_resolution)
        except ValueError as error:
            raise PreparationError(INVALID_GOAL, str(error)) from error
        final_name = self._final_name(snapshot, selected)
        try:
            transaction = self._allocate_transaction(config.result_root, final_name)
        except ResultTransactionError as error:
            raise PreparationError(
                IO_ERROR, f"result transaction allocation failed: {error.code}"
            )
        except Exception as error:
            raise PreparationError(
                INTERNAL_ERROR,
                f"result transaction allocation failed: {type(error).__name__}",
            )
        try:
            payload = self._job_payload(snapshot, selected, transaction, config)
            command = self._worker_command(payload, config)
            verifier = ProductionCompletionVerifier(
                config=config,
                transaction=transaction,
                snapshot=snapshot,
                selected_paths=selected,
            )
            warm_command = None
            warm_job = None
            if config.warm_worker_enabled:
                warm_command = warm_worker_command(config)
                warm_job = payload
            return PreparedLaunch(
                command=command,
                result_transaction=transaction,
                verify_completion=verifier,
                warm_command=warm_command,
                warm_job=warm_job,
                preserve_uncommitted_staging=config.preserve_failed_worker_staging,
                timings={**timings, "transaction": time.monotonic() - step},
            )
        except BaseException:
            # 构造失败必须回滚已分配的事务暂存目录
            try:
                transaction.cleanup()
            except BaseException as cleanup_error:
                raise PreparationError(
                    IO_ERROR,
                    "prepared launch construction failed and staging cleanup "
                    f"failed: {type(cleanup_error).__name__}",
                ) from cleanup_error
            raise

    def _final_name(
        self,
        snapshot: ValidatedSnapshot,
        selected: tuple[Path, ...],
    ) -> str:
        """生成结果目录名：scene_id-清单摘要16-选帧摘要8-token。"""
        token = self._token_source()
        if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
            raise PreparationError(INTERNAL_ERROR, "result name token is unsafe")
        digest_prefix = snapshot.manifest_sha256[:16]
        selected_prefix = hashlib.sha256(
            "\n".join(path.name for path in selected).encode("utf-8")
        ).hexdigest()[:8]
        return f"{snapshot.scene_id}-{digest_prefix}-{selected_prefix}-{token}"

    def _job_payload(
        self,
        snapshot: ValidatedSnapshot,
        selected: tuple[Path, ...],
        transaction: ResultTransaction,
        config: ProductionConfig,
    ) -> dict[str, Any]:
        """构造重建 job 参数（一次性 argv 与 warm job 共用同一来源）。"""
        return {
            "images_dir": str(snapshot.canonical_root / "images"),
            "output_dir": str(transaction.staging_path),
            "supervisor_staging": True,
            "input_images": [str(path) for path in selected],
            "image_resolution": int(config.image_resolution),
            "load_mode": config.load_mode,
            "frame_group_size": int(config.frame_group_size),
            "aggregator_frame_chunk_size": int(config.aggregator_frame_chunk_size),
            "conf_thres": float(config.conf_thres),
            "max_points_k": int(config.max_points_k),
            "no_glb": not config.require_glb,
            "no_cam": not config.show_cameras,
        }

    def _worker_command(
        self,
        payload: dict[str, Any],
        config: ProductionConfig,
    ) -> tuple[str, ...]:
        """构造精确的无 shell worker argv（经 supervisor_python 启动包装器）。

        结构：supervisor_python worker_runner --log-path <暂存/run.log> --
              worker_python reconstruct_script --images_dir ... --output_dir ...
              随后附加模型/分辨率/选帧等参数（与 warm job 共用 payload）。
        """
        command = [
            config.supervisor_python,
            self._worker_runner_script,
            "--log-path",
            str(Path(payload["output_dir"]) / "run.log"),
            "--",
            config.worker_python,
            config.reconstruct_script,
            "--images_dir",
            str(payload["images_dir"]),
            "--output_dir",
            str(payload["output_dir"]),
            "--supervisor_staging",
            "--checkpoint",
            config.checkpoint_path,
            "--checkpoint_sha256",
            config.checkpoint_sha256,
            "--image_resolution",
            str(payload["image_resolution"]),
            "--load_mode",
            str(payload["load_mode"]),
            "--frame_group_size",
            str(payload["frame_group_size"]),
            "--aggregator_frame_chunk_size",
            str(payload["aggregator_frame_chunk_size"]),
            "--conf_thres",
            repr(float(payload["conf_thres"])),
            "--max_points_k",
            str(payload["max_points_k"]),
        ]
        if config.attention_backend != "auto":
            command.extend(("--attention_backend", config.attention_backend))
        # 每个选中的帧都以 --input_image <path> 成对传入
        for path in payload["input_images"]:
            command.append("--input_image")
            command.append(str(path))
        if payload["no_glb"]:
            command.append("--no_glb")
        if payload["no_cam"]:
            command.append("--no_cam")
        return tuple(command)


def _map_snapshot_invalid_code(code: str) -> str:
    """把快照校验错误码映射到准备阶段稳定错误码。"""
    if code in _SNAPSHOT_INTERNAL_CODES:
        return INTERNAL_ERROR
    if code in _SNAPSHOT_IO_CODES:
        return IO_ERROR
    return INVALID_GOAL


def _select_snapshot_frames(
    snapshot: ValidatedSnapshot, frame_group_size: int
) -> tuple[Path, ...]:
    """确定性选帧：帧数超限时均匀抽帧；校验选帧 basename 无重复。"""
    if snapshot.frame_count > frame_group_size:
        indices = _selected_indices(snapshot.frame_count, frame_group_size)
        selected = tuple(snapshot.image_paths[index] for index in indices)
    else:
        selected = snapshot.image_paths
    basenames = [path.name for path in selected]
    if len(basenames) != len(set(basenames)):
        raise PreparationError(
            INVALID_GOAL,
            "selected snapshot frames contain duplicate basenames",
        )
    return selected


def _worker_runner_script_path() -> str:
    """定位 worker_runner 模块文件路径（作为 argv 中的包装器脚本）。"""
    from . import worker_runner

    path = Path(worker_runner.__file__).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ProductionConfigError("worker_runner script must be a canonical file")
    return str(path)


def warm_worker_pool_log_path(config: ProductionConfig) -> str:
    """驻留 warm worker 的进程级日志路径（result_root 内、创建于首次 spawn）。"""
    if not isinstance(config, ProductionConfig):
        raise TypeError("config must be ProductionConfig")
    if not config.warm_worker_enabled:
        raise ProductionConfigError(
            "warm_worker_pool_log_path requires warm_worker_enabled"
        )
    return str(Path(config.result_root) / ".warm-worker.log")


def warm_worker_command(config: ProductionConfig) -> tuple[str, ...]:
    """构造 warm worker 驻留进程的无 shell argv（worker_python 直启脚本）。"""
    if not isinstance(config, ProductionConfig):
        raise TypeError("config must be ProductionConfig")
    if not config.warm_worker_enabled:
        raise ProductionConfigError(
            "warm_worker_command requires warm_worker_enabled"
        )
    return (
        config.worker_python,
        config.warm_worker_script,
        "--checkpoint",
        config.checkpoint_path,
        "--checkpoint_sha256",
        config.checkpoint_sha256,
        "--aggregator-frame-chunk-size",
        str(config.aggregator_frame_chunk_size),
        "--log-path",
        warm_worker_pool_log_path(config),
    ) + (("--attention-backend", config.attention_backend) if config.attention_backend != "auto" else ())
