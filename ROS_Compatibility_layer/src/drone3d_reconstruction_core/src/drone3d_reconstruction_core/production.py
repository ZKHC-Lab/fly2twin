"""生产规划器与完成校验器模块（不依赖 ROS）。

本模块拥有"已验证快照 → 单个 VGGT-Omega worker 进程"之间的显式生产边界：

  * ProductionConfig 不可变，且在 Action 服务端构造前对每个显式路径、
    摘要与资源参数做 preflight 校验（含 checkpoint 一次性 SHA-256 哈希）；
  * ProductionPlanner 校验目标快照、执行资源准入、确定性选帧、分配
    监督进程拥有的结果事务，并构造精确的无 shell worker argv（只向
    事务暂存目录写入）；
  * ProductionCompletionVerifier 以独立进程运行应用校验器，支持取消、
    单调超时、run.log/资产校验，原子写 result.json，并返回映射到冻结
    的 v1 Action Result 字段的 VerifiedCompletion。

本模块不内嵌任何生产默认值：所有阈值与画像值都来自 ProductionConfig。
模块不导入 torch 与 NumPy，可安全地在 ROS 监督进程内加载。

ROS-independent production planner and completion verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
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
    INPUT_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_GOAL,
    IO_ERROR,
    CompletionError,
    PreparationError,
    VerificationCancelled,
)
from .reconstruct_action_server import PreparedLaunch, VerifiedCompletion
from .resource_gate import (
    ResourceBusyError,
    ResourceDecision,
    ResourceGateError,
    ResourceLimits,
    ResourceSample,
    enforce_resources,
    evaluate_resources,
)
from .result_transaction import (
    ResultTransaction,
    ResultTransactionError,
    allocate_result_transaction,
)
from .snapshot_validator import (
    SnapshotInvalidError,
    SnapshotMissingError,
    SnapshotSafetyLimits,
    SnapshotValidationError,
    ValidatedSnapshot,
    validate_snapshot,
)
from .worker_runner import WORKER_RUN_LOG_MARKER, worker_environment
from .backend_spec import validate_backend

RESULT_SCHEMA = "drone3d.reconstruction.result"
RESULT_SCHEMA_VERSION = 1
COORDINATE_STATE = "MODEL_LOCAL_UNALIGNED"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_LOAD_MODES = ("max_size", "balanced")
# 快照内部错误码（配置问题）与 IO 错误码（文件系统问题）的映射集
_SNAPSHOT_INTERNAL_CODES = frozenset(("MISSING_ALLOWED_ROOT",))
_SNAPSHOT_IO_CODES = frozenset(("FILESYSTEM_ERROR",))


class ProductionConfigError(ValueError):
    """生产配置非法：必须在构造前失败（fail-fast）。"""


@dataclass(frozen=True)
class ProductionConfig:
    """一个监督进程的显式、不可变生产参数。

    __post_init__ 中检查全部路径：必须绝对、规范化、无符号链接遍历且存在；
    checkpoint 以流式方式计算 SHA-256，必须与 checkpoint_sha256 完全一致
    （构造本对象即有一次性哈希成本）。
    """

    supervisor_python: str
    worker_python: str
    reconstruct_script: str
    validator_script: str
    checkpoint_path: str
    checkpoint_sha256: str
    result_root: str
    allowed_snapshot_roots: tuple[str, ...]
    snapshot_limits: SnapshotSafetyLimits
    resource_limits: ResourceLimits
    image_resolution: int
    frame_group_size: int
    max_result_pixels: int
    max_result_file_bytes: int
    max_result_total_bytes: int
    max_log_bytes: int
    verifier_timeout_sec: float
    worker_timeout_sec: float
    term_grace_sec: float
    load_mode: str
    require_glb: bool
    show_cameras: bool
    conf_thres: float
    max_points_k: int
    profile_name: str
    required_device_sku: str
    algorithm_revision: str
    ros_revision: str
    warm_worker_enabled: bool = False
    warm_worker_script: str = ""
    runtime_memory_guard_enabled: bool = False
    preserve_failed_worker_staging: bool = False
    aggregator_frame_chunk_size: int = 50
    attention_backend: str = "auto"
    backend: str = "omega"

    def __post_init__(self) -> None:
        try:
            self.preflight()
        except ProductionConfigError:
            raise
        except Exception as error:
            raise ProductionConfigError(
                f"production preflight failed: {type(error).__name__}: {error}"
            ) from error

    def preflight(self) -> None:
        """对每个显式路径、摘要与参数做恰好一次完整校验。"""
        validate_backend(self)
        if self.attention_backend not in ("auto", "flash_attn_v1"):
            raise ProductionConfigError("attention_backend must be auto or flash_attn_v1")
        if self.backend != "omega" and self.attention_backend != "auto":
            raise ProductionConfigError("attention_backend override is only supported for Omega")
        _canonical_existing_file(self.supervisor_python, "supervisor_python")
        _canonical_existing_file(self.worker_python, "worker_python")
        if not os.access(self.supervisor_python, os.X_OK):
            raise ProductionConfigError("supervisor_python must be executable")
        if not os.access(self.worker_python, os.X_OK):
            raise ProductionConfigError("worker_python must be executable")
        _canonical_existing_file(self.reconstruct_script, "reconstruct_script")
        _canonical_existing_file(self.validator_script, "validator_script")
        _canonical_existing_file(self.checkpoint_path, "checkpoint_path")
        if not isinstance(self.checkpoint_sha256, str) or not _SHA256_RE.fullmatch(
            self.checkpoint_sha256
        ):
            raise ProductionConfigError(
                "checkpoint_sha256 must be a lowercase SHA-256 digest"
            )
        # 模型文件指纹校验：防止意外换模型仍按旧摘要发布
        _verify_checkpoint_digest(self.checkpoint_path, self.checkpoint_sha256)

        _canonical_existing_directory(self.result_root, "result_root")
        if not isinstance(self.allowed_snapshot_roots, tuple) or not self.allowed_snapshot_roots:
            raise ProductionConfigError(
                "allowed_snapshot_roots must be a nonempty tuple of strings"
            )
        for index, root in enumerate(self.allowed_snapshot_roots):
            _canonical_existing_directory(root, f"allowed_snapshot_roots[{index}]")

        if not isinstance(self.snapshot_limits, SnapshotSafetyLimits):
            raise ProductionConfigError("snapshot_limits must be SnapshotSafetyLimits")
        try:
            self.snapshot_limits.validate()
        except SnapshotValidationError as error:
            raise ProductionConfigError(f"snapshot_limits invalid: {error}") from error

        if type(self.runtime_memory_guard_enabled) is not bool:
            raise ProductionConfigError("runtime_memory_guard_enabled must be a bool")
        if type(self.preserve_failed_worker_staging) is not bool:
            raise ProductionConfigError("preserve_failed_worker_staging must be a bool")
        if not isinstance(self.resource_limits, ResourceLimits):
            raise ProductionConfigError("resource_limits must be ResourceLimits")
        _preflight_resource_limits(self.resource_limits)

        for name, value in (
            ("image_resolution", self.image_resolution),
            ("frame_group_size", self.frame_group_size),
            ("max_result_pixels", self.max_result_pixels),
            ("max_result_file_bytes", self.max_result_file_bytes),
            ("max_result_total_bytes", self.max_result_total_bytes),
            ("max_log_bytes", self.max_log_bytes),
            ("max_points_k", self.max_points_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ProductionConfigError(f"{name} must be a positive integer")

        if (
            isinstance(self.aggregator_frame_chunk_size, bool)
            or not isinstance(self.aggregator_frame_chunk_size, int)
            or self.aggregator_frame_chunk_size < 0
        ):
            raise ProductionConfigError(
                "aggregator_frame_chunk_size must be a nonnegative integer"
            )
        if self.snapshot_limits.max_frame_count < self.frame_group_size:
            raise ProductionConfigError(
                "snapshot_limits.max_frame_count must be at least frame_group_size"
            )
        minimum_pixels = 2 * self.image_resolution * self.image_resolution
        if self.max_result_pixels < minimum_pixels:
            raise ProductionConfigError(
                "max_result_pixels must cover the configured resolution: "
                f"at least {minimum_pixels} pixels"
            )
        if self.max_result_file_bytes > self.max_result_total_bytes:
            raise ProductionConfigError(
                "max_result_file_bytes must not exceed max_result_total_bytes"
            )

        for name, value in (
            ("verifier_timeout_sec", self.verifier_timeout_sec),
            ("worker_timeout_sec", self.worker_timeout_sec),
            ("term_grace_sec", self.term_grace_sec),
            ("conf_thres", self.conf_thres),
        ):
            if not _finite_number(value):
                raise ProductionConfigError(f"{name} must be a finite number")
        if self.verifier_timeout_sec <= 0:
            raise ProductionConfigError("verifier_timeout_sec must be positive")
        if self.worker_timeout_sec <= 0:
            raise ProductionConfigError("worker_timeout_sec must be positive")
        if self.term_grace_sec <= 0:
            raise ProductionConfigError("term_grace_sec must be positive")
        # reconstruct_omega 把 GLB 置信度百分位钳制到至少 3：
        # 拒绝低于该值的配置，避免 result.json 记录 worker 实际未使用的参数。
        if self.conf_thres < 3 or self.conf_thres > 100:
            raise ProductionConfigError(
                "conf_thres must be in the range [3, 100] to match worker semantics"
            )

        if self.load_mode not in _LOAD_MODES:
            raise ProductionConfigError(
                f"load_mode must be one of {_LOAD_MODES!r}"
            )
        if type(self.require_glb) is not bool:
            raise ProductionConfigError("require_glb must be a bool")
        if type(self.show_cameras) is not bool:
            raise ProductionConfigError("show_cameras must be a bool")
        if type(self.warm_worker_enabled) is not bool:
            raise ProductionConfigError("warm_worker_enabled must be a bool")
        if not isinstance(self.warm_worker_script, str):
            raise ProductionConfigError("warm_worker_script must be a string")
        if self.warm_worker_enabled:
            _canonical_existing_file(self.warm_worker_script, "warm_worker_script")
        elif self.warm_worker_script:
            raise ProductionConfigError(
                "warm_worker_script must be empty when warm_worker_enabled is false"
            )
        if (
            not isinstance(self.profile_name, str)
            or not _PROFILE_NAME_RE.fullmatch(self.profile_name)
        ):
            raise ProductionConfigError("profile_name must be a safe identifier")
        if (
            not isinstance(self.required_device_sku, str)
            or not _PROFILE_NAME_RE.fullmatch(self.required_device_sku)
        ):
            raise ProductionConfigError(
                "required_device_sku must be a safe nonempty identifier"
            )
        for name in ("algorithm_revision", "ros_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ProductionConfigError(f"{name} must be a nonempty string")


def _finite_number(value: Any) -> bool:
    """校验值为有限数值（拒绝 bool/NaN/无穷）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _reject_symlink_components(path: Path, label: str) -> None:
    """逐路径组件检查符号链接，拒绝任何祖先软链。"""
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            info = current.lstat()
        except OSError as error:
            raise ProductionConfigError(f"{label} cannot be inspected") from error
        if stat.S_ISLNK(info.st_mode):
            raise ProductionConfigError(f"{label} must not traverse symlinks")


def _canonical_existing_file(value: str, label: str) -> str:
    """校验文件路径：绝对、无软链、规范化且存在（普通文件）。"""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProductionConfigError(f"{label} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute():
        raise ProductionConfigError(f"{label} must be absolute")
    _reject_symlink_components(path, label)
    try:
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise ProductionConfigError(f"{label} must exist and resolve") from error
    if canonical != path or path.is_symlink() or not path.is_file():
        raise ProductionConfigError(f"{label} must be a canonical regular file")
    return str(canonical)


def _canonical_existing_directory(value: str, label: str) -> str:
    """校验目录路径：绝对、无软链、规范化且存在（真实目录）。"""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProductionConfigError(f"{label} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute():
        raise ProductionConfigError(f"{label} must be absolute")
    _reject_symlink_components(path, label)
    try:
        canonical = path.resolve(strict=True)
    except OSError as error:
        raise ProductionConfigError(f"{label} must exist and resolve") from error
    if canonical != path or path.is_symlink() or not path.is_dir():
        raise ProductionConfigError(f"{label} must be a canonical real directory")
    return str(canonical)


_VERIFIED_CHECKPOINTS: set = set()


def _verify_checkpoint_digest(checkpoint_path: str, expected: str) -> None:
    """流式计算 checkpoint 的 SHA-256 并与期望值比对。

    每个目标的分辨率/超时覆盖都会 replace() 出新配置并重跑这里；同一文件
    （设备、inode、大小、修改时间不变）只哈希一次。NX 上每次重算 4.5 GB
    约 5 s，还会挤占页缓存。
    """
    try:
        status = os.stat(checkpoint_path)
        identity = (checkpoint_path, status.st_dev, status.st_ino, status.st_size,
                    status.st_mtime_ns, expected)
    except OSError:
        identity = None
    if identity is not None and identity in _VERIFIED_CHECKPOINTS:
        return
    digest = hashlib.sha256()
    try:
        with open(checkpoint_path, "rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise ProductionConfigError("checkpoint cannot be hashed") from error
    actual = digest.hexdigest()
    if actual != expected:
        raise ProductionConfigError(
            f"checkpoint digest mismatch: expected {expected}, got {actual}"
        )
    if identity is not None:
        _VERIFIED_CHECKPOINTS.add(identity)


def _preflight_resource_limits(limits: ResourceLimits) -> None:
    """通过门禁自身的纯判定器校验限额配置（零值采样 + 全零判定）。"""
    if not _finite_number(limits.swap_sample_window_sec) or limits.swap_sample_window_sec <= 0:
        raise ProductionConfigError("resource swap_sample_window_sec must be positive")
    sample = ResourceSample(
        first_mem_available_bytes=0,
        second_mem_available_bytes=0,
        first_swap_used_bytes=0,
        second_swap_used_bytes=0,
        swap_growth_bytes=0,
        snapshot_free_bytes=0,
        result_free_bytes=0,
        observed_window_sec=limits.swap_sample_window_sec,
    )
    try:
        evaluate_resources(sample, limits)
    except ResourceGateError as error:
        raise ProductionConfigError(f"resource_limits invalid: {error}") from error


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
            "frame_stamps_available": bool(
                self._snapshot.snapshot_version >= 2
            ),
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
                "required_device_sku": config.required_device_sku,
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


def _signal_group(process_group: int, sig: signal.Signals) -> None:
    """向进程组发信号；组不存在时静默忽略。"""
    try:
        os.killpg(process_group, sig)
    except ProcessLookupError:
        pass


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


__all__ = [
    "COORDINATE_STATE",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_VERSION",
    "ProductionCompletionVerifier",
    "ProductionConfig",
    "ProductionConfigError",
    "ProductionPlanner",
    "warm_worker_command",
    "warm_worker_pool_log_path",
]
