"""drone3d_reconstruction_core 包入口：统一导出采集/重建流水线的公共 API。

本包是 Drone3D 重建 ROS 兼容层的核心库：它把采集（capture）、
生命周期控制（controller）、目标准备与完成校验（preparation/production）、
资源准入（resource_gate）、结果事务（result_transaction）、快照校验
（snapshot_validator）等 ROS 无关模块组合成一条可被 Action 服务端使用的
完整流水线。

使用方式：
  - 采集侧：CaptureConfig + CaptureSession + allocate_snapshot_transaction
  - 重建侧：ProductionConfig + ProductionPlanner + ProductionCompletionVerifier
  - 公共基建：JobController / ResultTransaction / validate_snapshot /
    enforce_resources 等均可独立使用与测试。

Lifecycle and snapshot admission primitives for Drone3D reconstruction.

Public names are re-exported lazily (PEP 562). Importing this package, or any
ROS-free submodule such as ``g4.map_backend``, must not pull in ``rospy`` or
``actionlib``: the ROS action servers are imported only when one of their names
is actually requested. ``from drone3d_reconstruction_core import X`` and
``from drone3d_reconstruction_core import submodule`` both keep working.
"""

from importlib import import_module

_EXPORTS = {
    "capture": (
        "CAMERA_UNAVAILABLE",
        "CaptureConfig",
        "CaptureConfigError",
        "CaptureError",
        "CaptureGoalParameters",
        "CaptureSession",
        "CaptureStreamError",
        "ImageCodec",
        "OdometryAssociator",
        "SealedSnapshot",
        "SnapshotWriter",
        "StampSampler",
        "allocate_snapshot_transaction",
        "snapshot_final_name",
        "validate_capture_goal",
    ),
    "controller": (
        "BusyError",
        "JobController",
        "JobFinished",
        "WarmWorkerController",
    ),
    "preparation": (
        "ALGORITHM_FAILED",
        "INPUT_NOT_FOUND",
        "INTERNAL_ERROR",
        "INVALID_GOAL",
        "IO_ERROR",
        "CompletionError",
        "PreparationError",
        "VerificationCancelled",
    ),
    "production": (
        "COORDINATE_STATE",
        "RESULT_SCHEMA",
        "RESULT_SCHEMA_VERSION",
        "ProductionCompletionVerifier",
        "ProductionConfig",
        "ProductionConfigError",
        "ProductionPlanner",
    ),
    "reconstruct_action_server": ("VerifiedCompletion",),
    "resource_gate": (
        "HIGH_SWAP_GROWTH",
        "HIGH_SWAP_USED",
        "LOW_MEM_AVAILABLE",
        "LOW_RESULT_DISK",
        "LOW_SNAPSHOT_DISK",
        "ResourceBusyError",
        "ResourceDecision",
        "ResourceGateError",
        "ResourceLimits",
        "ResourceSample",
        "enforce_resources",
        "evaluate_resources",
        "parse_meminfo",
    ),
    "result_transaction": (
        "ResultCollisionError",
        "ResultCommitError",
        "ResultNameError",
        "ResultRootError",
        "ResultSafetyError",
        "ResultStateError",
        "ResultTransaction",
        "ResultTransactionError",
        "allocate_result_transaction",
    ),
    "snapshot_validator": (
        "SNAPSHOT_SCHEMA",
        "SNAPSHOT_VERSION",
        "SUPPORTED_SNAPSHOT_VERSIONS",
        "FrameStamp",
        "SnapshotInvalidError",
        "SnapshotMissingError",
        "SnapshotSafetyLimits",
        "SnapshotValidationError",
        "ValidatedSnapshot",
        "validate_snapshot",
    ),
}

_NAME_TO_MODULE = {
    name: module for module, names in _EXPORTS.items() for name in names
}

__all__ = sorted(_NAME_TO_MODULE)


def __getattr__(name):
    module_name = _NAME_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
