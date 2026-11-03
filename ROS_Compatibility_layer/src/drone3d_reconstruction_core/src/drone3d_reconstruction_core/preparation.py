"""生产准备与完成校验失败的类型定义模块。

本模块定义了重建流水线在两个阶段可能抛出的全部类型化失败：
  - PreparationError：目标准备阶段的永久性失败（目标非法、输入缺失、
    IO 错误、内部错误），携带稳定 Action 错误码。
  - CompletionError：结果完成校验阶段的失败（算法失败、IO 错误、
    内部错误），同样携带稳定 Action 错误码。
  - VerificationCancelled：Action 取消事件中断校验子进程时的专用失败。

这些错误码（INVALID_GOAL / INPUT_NOT_FOUND / IO_ERROR / INTERNAL_ERROR /
ALGORITHM_FAILED）由 Action 服务端精确映射到冻结的 ReconstructSegmentResult
错误码，保证 ROS 客户端拿到的错误码稳定、可机器判定。

Typed production preparation and completion failures.
"""

from __future__ import annotations


INVALID_GOAL = "INVALID_GOAL"
INPUT_NOT_FOUND = "INPUT_NOT_FOUND"
IO_ERROR = "IO_ERROR"
INTERNAL_ERROR = "INTERNAL_ERROR"
ALGORITHM_FAILED = "ALGORITHM_FAILED"

# 准备阶段允许的错误码集合（不含 ALGORITHM_FAILED，那是完成阶段专属）
_PREPARATION_CODES = frozenset(
    (INVALID_GOAL, INPUT_NOT_FOUND, IO_ERROR, INTERNAL_ERROR)
)
# 完成校验阶段允许的错误码集合
_COMPLETION_CODES = frozenset((ALGORITHM_FAILED, IO_ERROR, INTERNAL_ERROR))


class PreparationError(RuntimeError):
    """目标准备阶段的永久性失败，携带稳定的 Action 错误码。

    构造时校验错误码必须在 _PREPARATION_CODES 内，且消息必须非空；
    任何非法输入都直接抛 ValueError（编程错误，而非运行期失败）。

    Attributes:
        code: 稳定的 Action 错误码字符串
    """

    def __init__(self, code: str, message: str) -> None:
        if code not in _PREPARATION_CODES:
            raise ValueError("unsupported preparation error code")
        if not isinstance(message, str) or not message:
            raise ValueError("preparation error message must be nonempty")
        self.code = code
        super().__init__(message)


class CompletionError(RuntimeError):
    """完成校验阶段的失败，携带稳定的 Action 错误码。

    Attributes:
        code: 稳定的 Action 错误码字符串
    """

    def __init__(self, code: str, message: str) -> None:
        if code not in _COMPLETION_CODES:
            raise ValueError("unsupported completion error code")
        if not isinstance(message, str) or not message:
            raise ValueError("completion error message must be nonempty")
        self.code = code
        super().__init__(message)


class VerificationCancelled(CompletionError):
    """Action 取消事件中断了校验子进程。

    当客户端在结果校验阶段请求取消时，验证器子进程会被终止并回收，
    本异常用于把这一情形映射为 ALGORITHM_FAILED 错误码（取消语义由
    Action 服务端负责，这里只负责类型化的失败描述）。
    """

    def __init__(self) -> None:
        super().__init__(ALGORITHM_FAILED, "result verification cancelled")


__all__ = [
    "ALGORITHM_FAILED",
    "CompletionError",
    "INPUT_NOT_FOUND",
    "INTERNAL_ERROR",
    "INVALID_GOAL",
    "IO_ERROR",
    "PreparationError",
    "VerificationCancelled",
]
