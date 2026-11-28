"""Production configuration: the explicit, immutable parameters of one supervisor.

``ProductionConfig`` is checked before the action server exists (paths, digests,
resource limits, the checkpoint's SHA-256). Split from ``production``, which still
re-exports every public name.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat

from .resource_gate import (
    ResourceGateError,
    ResourceLimits,
    ResourceSample,
    evaluate_resources,
)
from .snapshot_validator import (
    SnapshotSafetyLimits,
    SnapshotValidationError,
)
from .backend_spec import validate_backend
from ._common import is_finite_number as _finite_number


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_LOAD_MODES = ("max_size", "balanced")

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
        for name in ("algorithm_revision", "ros_revision"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ProductionConfigError(f"{name} must be a nonempty string")


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
