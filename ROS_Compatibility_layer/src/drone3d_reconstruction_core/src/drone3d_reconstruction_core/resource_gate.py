"""资源门禁模块：在启动重建任务前对宿主机资源做准入检查（不依赖 ROS）。

本模块实现"两点采样 + 纯策略判定"的资源准入机制：在 swap 观测窗口内
连续采样两次 /proc/meminfo，再叠加快照/结果文件系统的剩余空间，
得到一个完整的 ResourceSample，交由纯函数 evaluate_resources 判定
是否允许启动重建任务。

主要功能：
  - parse_meminfo：严格解析 /proc/meminfo 得到可用内存与已用 swap 字节数
  - evaluate_resources：对一个完整采样做纯策略判定（内存下限、swap 上限、
    swap 增长上限、快照/结果磁盘下限）
  - enforce_resources：探测真实资源并采样；只有"有效采样被拒绝"时才抛错

设计要点：
  - 所有字节值均为包含式边界（inclusive）；
  - 判定是纯函数（无 IO），便于单测与审计；
  - 拒绝原因通过 ResourceDecision.denial_code 稳定暴露，
    由上层映射为 Action 的 BUSY 拒绝。

ROS-independent host resource admission for reconstruction jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import re
import sys
import time
from typing import Any, Callable


# 拒绝原因码：内存不足 / swap 用量超标 / swap 增长过快 / 快照盘不足 / 结果盘不足
LOW_MEM_AVAILABLE = "LOW_MEM_AVAILABLE"
HIGH_SWAP_USED = "HIGH_SWAP_USED"
HIGH_SWAP_GROWTH = "HIGH_SWAP_GROWTH"
LOW_SNAPSHOT_DISK = "LOW_SNAPSHOT_DISK"
LOW_RESULT_DISK = "LOW_RESULT_DISK"

# meminfo 中必须存在的三个键
_REQUIRED_MEMINFO_KEYS = ("MemAvailable", "SwapTotal", "SwapFree")
# 十进制规范整数的正则（拒绝前导零与科学计数法）
_CANONICAL_INTEGER = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_MAX_BYTES = sys.maxsize


class ResourceGateError(RuntimeError):
    """资源策略或资源探测本身无效（配置错误，而非资源不足）。"""


class ResourceBusyError(RuntimeError):
    """一个有效采样判定为拒绝准入时的异常（可重试的瞬时状态）。

    Attributes:
        decision: 导致拒绝的完整判定（含采样与限额）
    """

    def __init__(self, decision: "ResourceDecision") -> None:
        if not isinstance(decision, ResourceDecision) or decision.allowed:
            raise ResourceGateError("ResourceBusyError requires a denied decision")
        self.decision = decision
        super().__init__(decision.message)


@dataclass(frozen=True)
class ResourceLimits:
    """显式资源下限/上限配置，所有字节值均为包含式边界。

    Attributes:
        min_mem_available_bytes: MemAvailable 下限（字节）
        max_swap_used_bytes: swap 已用量上限（字节）
        max_swap_growth_bytes: swap 观测窗口内增长上限（字节）
        min_snapshot_free_bytes: 快照文件系统剩余空间下限
        min_result_free_bytes: 结果文件系统剩余空间下限
        swap_sample_window_sec: 两次采样之间的观测窗口时长（秒）
    """

    min_mem_available_bytes: int
    max_swap_used_bytes: int
    max_swap_growth_bytes: int
    min_snapshot_free_bytes: int
    min_result_free_bytes: int
    swap_sample_window_sec: float


@dataclass(frozen=True)
class ResourceSample:
    """一次完整的两点采样：内存/swap 观测对 + 文件系统观测。

    Attributes:
        first_mem_available_bytes: 第一次采样时的可用内存
        second_mem_available_bytes: 第二次采样时的可用内存
        first_swap_used_bytes: 第一次采样时的已用 swap
        second_swap_used_bytes: 第二次采样时的已用 swap
        swap_growth_bytes: 窗口内 swap 增长量（= max(0, second-first)）
        snapshot_free_bytes: 快照盘剩余空间
        result_free_bytes: 结果盘剩余空间
        observed_window_sec: 实际观测窗口时长
    """

    first_mem_available_bytes: int
    second_mem_available_bytes: int
    first_swap_used_bytes: int
    second_swap_used_bytes: int
    swap_growth_bytes: int
    snapshot_free_bytes: int
    result_free_bytes: int
    observed_window_sec: float


@dataclass(frozen=True)
class ResourceDecision:
    """对一个完整资源采样做出的纯策略判定。

    Attributes:
        allowed: 是否放行
        denial_code: 拒绝原因码（allowed 时为 None）
        message: 判定说明
        sample: 判定的输入采样
        limits: 判定使用的限额
    """

    allowed: bool
    denial_code: str | None
    message: str
    sample: ResourceSample
    limits: ResourceLimits


def parse_meminfo(meminfo: str) -> tuple[int, int]:
    """严格解析 /proc/meminfo 文本，返回 (可用内存字节, 已用 swap 字节)。

    Args:
        meminfo: /proc/meminfo 的完整文本内容

    Returns:
        (MemAvailable 字节数, SwapTotal-SwapFree 字节数)

    Raises:
        ResourceGateError: 键缺失/重复、值非规范整数或单位非 kB、
            SwapFree 超过 SwapTotal
    """
    if not isinstance(meminfo, str):
        raise ResourceGateError("meminfo probe must return text")
    found: dict[str, int] = {}
    for line in meminfo.splitlines():
        for key in _REQUIRED_MEMINFO_KEYS:
            prefix = key + ":"
            if not line.startswith(prefix):
                continue
            # 同一键只允许出现一次
            if key in found:
                raise ResourceGateError(f"duplicate {key} in meminfo")
            fields = line[len(prefix):].split()
            if len(fields) != 2 or fields[1] != "kB" or not _CANONICAL_INTEGER.fullmatch(fields[0]):
                raise ResourceGateError(f"invalid {key} value in meminfo")
            found[key] = _kilobytes_to_bytes(int(fields[0]), key)
            break
    missing = [key for key in _REQUIRED_MEMINFO_KEYS if key not in found]
    if missing:
        raise ResourceGateError("missing meminfo values: " + ", ".join(missing))
    # swap 已用量 = SwapTotal - SwapFree，且必须非负
    if found["SwapFree"] > found["SwapTotal"]:
        raise ResourceGateError("SwapFree exceeds SwapTotal")
    return found["MemAvailable"], found["SwapTotal"] - found["SwapFree"]


def evaluate_resources(sample: ResourceSample, limits: ResourceLimits) -> ResourceDecision:
    """对 ``sample`` 做稳定的纯准入判定（无任何 IO）。

    Args:
        sample: 完整资源采样
        limits: 准入限额

    Returns:
        第一个被触发的拒绝判定；全部通过时返回放行判定。

    Raises:
        ResourceGateError: 限额/采样非法，或观测窗口短于配置值
    """
    _validate_limits(limits)
    _validate_sample(sample)
    if sample.observed_window_sec < limits.swap_sample_window_sec:
        raise ResourceGateError("swap observation window was shorter than configured")
    # 逐条检查：任一拒绝立即返回，保持判定顺序稳定
    checks = (
        (min(sample.first_mem_available_bytes, sample.second_mem_available_bytes)
         < limits.min_mem_available_bytes,
         LOW_MEM_AVAILABLE, "available memory is below the configured floor"),
        (max(sample.first_swap_used_bytes, sample.second_swap_used_bytes)
         > limits.max_swap_used_bytes,
         HIGH_SWAP_USED, "swap usage exceeds the configured ceiling"),
        (sample.swap_growth_bytes > limits.max_swap_growth_bytes,
         HIGH_SWAP_GROWTH, "swap growth exceeds the configured ceiling"),
        (sample.snapshot_free_bytes < limits.min_snapshot_free_bytes,
         LOW_SNAPSHOT_DISK, "snapshot filesystem free space is below the configured floor"),
        (sample.result_free_bytes < limits.min_result_free_bytes,
         LOW_RESULT_DISK, "result filesystem free space is below the configured floor"),
    )
    for denied, code, message in checks:
        if denied:
            return ResourceDecision(False, code, message, sample, limits)
    return ResourceDecision(True, None, "resource admission passed", sample, limits)


def enforce_resources(
    limits: ResourceLimits,
    snapshot_root: os.PathLike[str] | str,
    result_root: os.PathLike[str] | str,
    *,
    meminfo_reader: Callable[[], str] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], Any] = time.sleep,
    statvfs: Callable[[os.PathLike[str] | str], Any] = os.statvfs,
    stat: Callable[[os.PathLike[str] | str], Any] = os.stat,
) -> ResourceDecision:
    """探测真实资源并在"有效采样判定拒绝"时抛出 ResourceBusyError。

    流程：连续两次读 /proc/meminfo（间隔 limits.swap_sample_window_sec），
    探测快照/结果盘的剩余空间，组装采样后交给 evaluate_resources 判定。

    Args:
        limits: 准入限额
        snapshot_root: 快照根目录（用于 statvfs 探测剩余空间）
        result_root: 结果根目录
        meminfo_reader: meminfo 文本读取器，默认读 /proc/meminfo
        monotonic: 单调时钟，默认 time.monotonic
        sleeper: 观测窗口睡眠函数
        statvfs/stat: 文件系统探测函数（测试注入点）

    Returns:
        放行判定；拒绝时抛出 ResourceBusyError
    """
    _validate_limits(limits)
    reader = meminfo_reader or _read_proc_meminfo
    started = _probe_monotonic(monotonic)
    first_mem, first_swap = _probe_meminfo(reader)
    try:
        sleeper(limits.swap_sample_window_sec)
    except Exception as error:
        raise ResourceGateError(
            f"swap observation sleep failed: {type(error).__name__}"
        ) from error
    second_mem, second_swap = _probe_meminfo(reader)
    finished = _probe_monotonic(monotonic)
    observed = finished - started
    # 时钟回退视为探测失败
    if finished < started:
        raise ResourceGateError("monotonic clock moved backwards")
    if observed < limits.swap_sample_window_sec:
        raise ResourceGateError("swap observation window was shorter than configured")
    snapshot_free, result_free = _probe_filesystems(
        snapshot_root, result_root, statvfs, stat
    )
    sample = ResourceSample(
        first_mem,
        second_mem,
        first_swap,
        second_swap,
        max(0, second_swap - first_swap),
        snapshot_free,
        result_free,
        observed,
    )
    decision = evaluate_resources(sample, limits)
    if not decision.allowed:
        # 有效采样被拒绝 → 抛可重试的忙异常（由 Action 层映射为 BUSY）
        raise ResourceBusyError(decision)
    return decision


def _read_proc_meminfo() -> str:
    """读取 /proc/meminfo 全文；任何失败统一包装为 ResourceGateError。"""
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as stream:
            return stream.read()
    except (OSError, UnicodeError) as error:
        raise ResourceGateError(
            f"meminfo probe failed: {type(error).__name__}"
        ) from error


def _probe_meminfo(reader: Callable[[], str]) -> tuple[int, int]:
    """调用注入的读取器并解析 meminfo；异常统一包装。"""
    try:
        value = reader()
    except ResourceGateError:
        raise
    except Exception as error:
        raise ResourceGateError(
            f"meminfo probe failed: {type(error).__name__}"
        ) from error
    return parse_meminfo(value)


def _probe_monotonic(monotonic: Callable[[], float]) -> float:
    """探测单调时钟并严格校验返回值为有限数。"""
    try:
        value = monotonic()
    except Exception as error:
        raise ResourceGateError(
            f"monotonic probe failed: {type(error).__name__}"
        ) from error
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResourceGateError("monotonic probe must return a number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ResourceGateError(
            f"monotonic probe failed: {type(error).__name__}"
        ) from error
    if not math.isfinite(value):
        raise ResourceGateError("monotonic probe returned a non-finite value")
    return value


def _probe_filesystems(snapshot_root, result_root, statvfs, stat):
    """探测两个根目录的剩余空间；同一设备只 statvfs 一次（按 st_dev 去重）。"""
    roots = (snapshot_root, result_root)
    devices = []
    for root in roots:
        try:
            device = _bounded_bytes(stat(root).st_dev, "st_dev")
            devices.append(device)
        except ResourceGateError:
            raise
        except Exception as error:
            raise ResourceGateError(
                f"filesystem identity probe failed: {type(error).__name__}"
            ) from error
    free_by_device: dict[int, int] = {}
    free_values = []
    for root, device in zip(roots, devices):
        if device not in free_by_device:
            try:
                values = statvfs(root)
                bavail = _nonnegative_integer(values.f_bavail, "f_bavail")
                frsize = _positive_bounded_integer(values.f_frsize, "f_frsize")
                free_by_device[device] = _checked_multiply(bavail, frsize, "filesystem free bytes")
            except ResourceGateError:
                raise
            except Exception as error:
                raise ResourceGateError(
                    f"statvfs probe failed: {type(error).__name__}"
                ) from error
        free_values.append(free_by_device[device])
    return free_values[0], free_values[1]


def _validate_limits(limits: ResourceLimits) -> None:
    """校验限额：字段类型、字节范围与观测窗口时长。"""
    if not isinstance(limits, ResourceLimits):
        raise ResourceGateError("limits must be ResourceLimits")
    for name in (
        "min_mem_available_bytes",
        "max_swap_used_bytes",
        "max_swap_growth_bytes",
        "min_snapshot_free_bytes",
        "min_result_free_bytes",
    ):
        _bounded_bytes(getattr(limits, name), name)
    window = _finite_number(
        limits.swap_sample_window_sec, "swap_sample_window_sec"
    )
    if window <= 0:
        raise ResourceGateError("swap_sample_window_sec must be positive and finite")


def _validate_sample(sample: ResourceSample) -> None:
    """校验采样自洽性：字段非负且 swap_growth 与两点采样一致。"""
    if not isinstance(sample, ResourceSample):
        raise ResourceGateError("sample must be ResourceSample")
    for name in (
        "first_mem_available_bytes", "second_mem_available_bytes",
        "first_swap_used_bytes", "second_swap_used_bytes", "swap_growth_bytes",
        "snapshot_free_bytes", "result_free_bytes",
    ):
        _bounded_bytes(getattr(sample, name), name)
    expected_growth = max(0, sample.second_swap_used_bytes - sample.first_swap_used_bytes)
    if sample.swap_growth_bytes != expected_growth:
        raise ResourceGateError("sample swap growth is inconsistent")
    observed = _finite_number(sample.observed_window_sec, "observed_window_sec")
    if observed < 0:
        raise ResourceGateError("observed_window_sec must be finite and nonnegative")


def _finite_number(value: Any, name: str) -> float:
    """要求值为有限数（拒绝 bool/NaN/无穷）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResourceGateError(f"{name} must be a number")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ResourceGateError(f"{name} must be finite") from error
    if not math.isfinite(converted):
        raise ResourceGateError(f"{name} must be finite")
    return converted


def _positive_bounded_integer(value: Any, name: str) -> int:
    """要求值为正的有界整数。"""
    value = _bounded_bytes(value, name)
    if value == 0:
        raise ResourceGateError(f"{name} must be positive")
    return value


def _kilobytes_to_bytes(value: int, name: str) -> int:
    return _checked_multiply(value, 1024, name)


def _checked_multiply(left: int, right: int, name: str) -> int:
    """带溢出保护的乘法（结果不得超过 sys.maxsize）。"""
    if left and right > _MAX_BYTES // left:
        raise ResourceGateError(f"{name} exceeds supported byte range")
    return left * right


def _bounded_bytes(value: Any, name: str) -> int:
    """要求值为不超过 sys.maxsize 的非负整数。"""
    value = _nonnegative_integer(value, name)
    if value > _MAX_BYTES:
        raise ResourceGateError(f"{name} exceeds supported byte range")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    """要求值为非负整数（拒绝 bool）。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResourceGateError(f"{name} must be a nonnegative integer")
    return value


__all__ = [
    "HIGH_SWAP_GROWTH", "HIGH_SWAP_USED", "LOW_MEM_AVAILABLE",
    "LOW_RESULT_DISK", "LOW_SNAPSHOT_DISK", "ResourceBusyError",
    "ResourceDecision", "ResourceGateError", "ResourceLimits", "ResourceSample",
    "enforce_resources", "evaluate_resources", "parse_meminfo",
]
