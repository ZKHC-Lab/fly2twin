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

# The implementation lives in three modules; this one keeps the original import path.
from .production_config import ProductionConfig, ProductionConfigError
from .production_planner import (
    ProductionPlanner,
    _selected_indices,  # noqa: F401  (tests import it from here)
    warm_worker_command,
    warm_worker_pool_log_path,
)
from .production_verifier import (
    COORDINATE_STATE,
    RESULT_SCHEMA,
    RESULT_SCHEMA_VERSION,
    ProductionCompletionVerifier,
)

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
