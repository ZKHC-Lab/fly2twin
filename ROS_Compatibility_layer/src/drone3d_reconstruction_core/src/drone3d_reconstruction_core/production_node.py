"""生产级 ROS1 重建监督节点：显式配置 + 设备 SKU 门禁 + /diagnostics。

监督节点既不导入 torch 也不导入应用校验器。它从显式私有参数构造不可变
ProductionConfig，比对配置的实际设备 SKU 与画像要求的 SKU，然后才构造
Action 服务端。任何配置或部署不匹配都会让节点保持存活但 not-ready：
错误通过 /diagnostics 上报，而不是启动一个只会回答 BUSY 并诱导客户端
重试永久部署错误的 Action 服务端。

诊断发布刻意保持最佳努力：Action 收敛绝不依赖诊断，服务端观察者因此
吞掉所有异常，本模块只记录它们的输出。

The supervisor node imports neither torch nor the application validator.  It
constructs an immutable :class:`ProductionConfig` from explicitly supplied
private parameters, compares the configured actual device SKU with the
profile-required SKU, and only then constructs the Action server.  Any config
or deployment mismatch leaves the node alive but not-ready: the error is
reported on ``/diagnostics`` instead of starting an Action server that would
only answer BUSY and invite clients to retry a permanent deployment error.

The diagnostics publisher is deliberately best effort.  Action convergence
must never depend on diagnostics; the server observers therefore swallow all
exceptions and this module only records their output.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Mapping

import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from drone3d_reconstruction_msgs.msg import ReconstructSegmentResult

from .controller import JobFinished, JobController, WarmWorkerController
from .production import (
    ProductionConfig,
    ProductionConfigError,
    ProductionPlanner,
    warm_worker_command,
    warm_worker_pool_log_path,
)
from .reconstruct_action_server import ReconstructActionServer
from .resource_gate import ResourceGateError, ResourceLimits, parse_meminfo
from .snapshot_validator import SnapshotSafetyLimits

NODE_NAME = "reconstruct_production_node"
DEFAULT_DIAGNOSTICS_TOPIC = "/diagnostics"
DEFAULT_DIAGNOSTICS_PERIOD_SEC = 1.0
DEFAULT_ACTION_NAME = "reconstruct_segment"
DECIMAL_BYTES_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z")

# 必填字符串参数（安全相关，全部显式）
_REQUIRED_STR_KEYS = (
    "supervisor_python",
    "worker_python",
    "reconstruct_script",
    "validator_script",
    "checkpoint_path",
    "checkpoint_sha256",
    "result_root",
    "profile_name",
    "algorithm_revision",
    "ros_revision",
    "load_mode",
)
# 必填整型参数
_REQUIRED_INT_KEYS = (
    "image_resolution",
    "frame_group_size",
    "max_result_pixels",
    "max_result_file_bytes",
    "max_result_total_bytes",
    "max_log_bytes",
    "max_points_k",
)
# 必填浮点参数
_REQUIRED_FLOAT_KEYS = (
    "verifier_timeout_sec",
    "worker_timeout_sec",
    "term_grace_sec",
    "conf_thres",
)
# 必填布尔/列表/字典参数
_REQUIRED_BOOL_KEYS = ("require_glb", "show_cameras")
_REQUIRED_LIST_KEYS = ("allowed_snapshot_roots",)
_REQUIRED_DICT_KEYS = ("snapshot_limits", "resource_limits")
# 嵌套限制结构的字段键集（必须完全一致）
_SNAPSHOT_LIMIT_KEYS = frozenset(SnapshotSafetyLimits.__dataclass_fields__)
_RESOURCE_LIMIT_KEYS = frozenset(ResourceLimits.__dataclass_fields__)
_MEMINFO_MAX_BYTES = 1024 * 1024

# 结果错误码 → 人类可读名（诊断展示用）
_ERROR_CODE_NAMES = {
    ReconstructSegmentResult.OK: "OK",
    ReconstructSegmentResult.INVALID_GOAL: "INVALID_GOAL",
    ReconstructSegmentResult.INPUT_NOT_FOUND: "INPUT_NOT_FOUND",
    ReconstructSegmentResult.BUSY: "BUSY",
    ReconstructSegmentResult.ALGORITHM_FAILED: "ALGORITHM_FAILED",
    ReconstructSegmentResult.IO_ERROR: "IO_ERROR",
    ReconstructSegmentResult.CANCELLED: "CANCELLED",
    ReconstructSegmentResult.INTERNAL_ERROR: "INTERNAL_ERROR",
}


class ProductionNodeConfigError(ValueError):
    """节点参数缺失、非法或与部署不兼容。"""


@dataclass(frozen=True)
class ProductionRuntimeParams:
    """监督节点的非安全性运行时接线参数。"""

    action_name: str
    diagnostics_topic: str
    diagnostics_period_sec: float


def load_runtime_params(params: Mapping[str, Any]) -> ProductionRuntimeParams:
    """读取运行时接线参数（仅使用常规接口默认值）。"""
    if not isinstance(params, Mapping):
        raise ProductionNodeConfigError("node parameters must be a mapping")
    action_name = _string_value(params.get("action_name", DEFAULT_ACTION_NAME))
    diagnostics_topic = _string_value(
        params.get("diagnostics_topic", DEFAULT_DIAGNOSTICS_TOPIC)
    )
    diagnostics_period_sec = _finite_number(
        params.get("diagnostics_period_sec", DEFAULT_DIAGNOSTICS_PERIOD_SEC)
    )
    if action_name != action_name.strip() or not action_name:
        raise ProductionNodeConfigError("action_name must be nonempty")
    if (
        not diagnostics_topic.startswith("/")
        or " " in diagnostics_topic
        or "~" in diagnostics_topic
        or "\x00" in diagnostics_topic
    ):
        raise ProductionNodeConfigError(
            "diagnostics_topic must be an absolute ROS topic name"
        )
    if diagnostics_period_sec <= 0:
        raise ProductionNodeConfigError("diagnostics_period_sec must be positive")
    return ProductionRuntimeParams(
        action_name=action_name,
        diagnostics_topic=diagnostics_topic,
        diagnostics_period_sec=float(diagnostics_period_sec),
    )


def load_production_config(params: Mapping[str, Any]) -> ProductionConfig:
    """从显式节点参数构造不可变生产配置。

    每个安全相关值都是必填的。缺失键与错误类型都会在 Action 服务端构造前
    抛出 ProductionNodeConfigError。
    """
    if not isinstance(params, Mapping):
        raise ProductionNodeConfigError("node parameters must be a mapping")

    values: dict[str, Any] = {}
    for key in _REQUIRED_STR_KEYS:
        values[key] = _string_value(_required(params, key))
    for key in _REQUIRED_INT_KEYS:
        values[key] = _integer_value(
            _required(params, key), allow_decimal_string=key.endswith("_bytes")
        )
    for key in _REQUIRED_FLOAT_KEYS:
        values[key] = _finite_number(_required(params, key))
    for key in _REQUIRED_BOOL_KEYS:
        value = _required(params, key)
        if type(value) is not bool:
            raise ProductionNodeConfigError(f"{key} must be a bool")
        values[key] = value

    # warm worker 是可选的向后兼容开关：缺省 false（一次性 worker）
    warm_worker_enabled = params.get("warm_worker_enabled", False)
    if type(warm_worker_enabled) is not bool:
        raise ProductionNodeConfigError("warm_worker_enabled must be a bool")
    warm_worker_script = params.get("warm_worker_script", "")
    if not isinstance(warm_worker_script, str):
        raise ProductionNodeConfigError("warm_worker_script must be a string")
    if warm_worker_enabled and not warm_worker_script:
        raise ProductionNodeConfigError(
            "warm_worker_script is required when warm_worker_enabled is true"
        )
    if not warm_worker_enabled and warm_worker_script:
        raise ProductionNodeConfigError(
            "warm_worker_script must be empty when warm_worker_enabled is false"
        )
    guard_enabled = params.get("runtime_memory_guard_enabled", False)
    if type(guard_enabled) is not bool:
        raise ProductionNodeConfigError("runtime_memory_guard_enabled must be a bool")
    values["runtime_memory_guard_enabled"] = guard_enabled
    preserve_staging = params.get("preserve_failed_worker_staging", False)
    if type(preserve_staging) is not bool:
        raise ProductionNodeConfigError("preserve_failed_worker_staging must be a bool")
    values["preserve_failed_worker_staging"] = preserve_staging
    values["warm_worker_enabled"] = warm_worker_enabled
    values["warm_worker_script"] = warm_worker_script
    values["attention_backend"] = params.get("attention_backend", "auto")
    values["aggregator_frame_chunk_size"] = _integer_value(
        params.get("aggregator_frame_chunk_size", 50)
    )
    values["backend"] = _string_value(params.get("backend", "omega"))

    roots = _required(params, "allowed_snapshot_roots")
    if not isinstance(roots, (list, tuple)) or not roots:
        raise ProductionNodeConfigError(
            "allowed_snapshot_roots must be a nonempty list"
        )
    for root in roots:
        _string_value(root)
    values["allowed_snapshot_roots"] = tuple(roots)

    snapshot_limits = _required(params, "snapshot_limits")
    if not isinstance(snapshot_limits, Mapping):
        raise ProductionNodeConfigError("snapshot_limits must be a mapping")
    # 键集必须与 SnapshotSafetyLimits 字段完全一致
    if set(snapshot_limits) != _SNAPSHOT_LIMIT_KEYS:
        raise ProductionNodeConfigError(
            "snapshot_limits must contain exactly "
            f"{sorted(_SNAPSHOT_LIMIT_KEYS)!r}"
        )
    values["snapshot_limits"] = SnapshotSafetyLimits(
        **{
            key: _integer_value(
                snapshot_limits[key], allow_decimal_string=key.endswith("_bytes")
            )
            for key in snapshot_limits
        }
    )

    resource_limits = _required(params, "resource_limits")
    if not isinstance(resource_limits, Mapping):
        raise ProductionNodeConfigError("resource_limits must be a mapping")
    # 键集必须与 ResourceLimits 字段完全一致
    if set(resource_limits) != _RESOURCE_LIMIT_KEYS:
        raise ProductionNodeConfigError(
            "resource_limits must contain exactly "
            f"{sorted(_RESOURCE_LIMIT_KEYS)!r}"
        )
    parsed_resource_limits = {}
    for key, value in resource_limits.items():
        if key == "swap_sample_window_sec":
            parsed_resource_limits[key] = _finite_number(value)
        else:
            parsed_resource_limits[key] = _integer_value(
                value, allow_decimal_string=key.endswith("_bytes")
            )
    values["resource_limits"] = ResourceLimits(**parsed_resource_limits)

    try:
        return ProductionConfig(**values)
    except TypeError as error:
        raise ProductionNodeConfigError(
            f"production config missing a field: {error}"
        ) from error
    except ProductionConfigError:
        raise


def _controller_factory_for(config: ProductionConfig):
    """按画像选择 Action 服务端的 worker 生命周期控制器工厂。

    warm worker 关闭时保持原有 JobController；开启时返回闭包工厂，
    使 ReconstructActionServer 构造 WarmWorkerController（其内部同时
    持有一次性 JobController 回退路径）。
    """
    if not config.warm_worker_enabled and not config.runtime_memory_guard_enabled:
        return JobController

    def factory(*, worker_timeout_sec, term_grace_sec, on_finished):
        def make(*, on_finished):
            common = dict(worker_timeout_sec=worker_timeout_sec,
                          term_grace_sec=term_grace_sec, on_finished=on_finished)
            if config.warm_worker_enabled:
                return WarmWorkerController(
                    **common, warm_command=warm_worker_command(config),
                    pool_log_path=warm_worker_pool_log_path(config))
            return JobController(**common)
        if config.runtime_memory_guard_enabled:
            from .runtime_memory_guard import MemoryGuardedController
            return MemoryGuardedController(
                make, on_finished=on_finished,
                audit_path=Path(config.result_root) / "runtime_memory_guard.jsonl")
        return make(on_finished=on_finished)

    return factory


class DiagnosticsState:
    """线程安全的遥测状态，由 Action 服务端观察者喂入。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.config: ProductionConfig | None = None
        self.not_ready_error: str | None = None
        self.last_terminal: dict[str, Any] | None = None
        self.last_finished: dict[str, Any] | None = None

    def set_config(self, config: ProductionConfig) -> None:
        """登记有效配置，清除 not-ready 错误。"""
        with self._lock:
            self.config = config
            self.not_ready_error = None

    def set_not_ready(self, error: BaseException) -> None:
        """登记 not-ready 错误（节点保持存活但不可用）。"""
        with self._lock:
            self.not_ready_error = f"{type(error).__name__}: {error}"

    def record_terminal(self, method: str, result: Any, text: str) -> None:
        """最佳努力终态观察者；绝不抛出异常。"""
        try:
            wall_time_sec = float(result.wall_time.to_sec())
        except BaseException:
            wall_time_sec = 0.0
        try:
            error_code = int(getattr(result, "error_code", 0))
        except BaseException:
            error_code = 0
        try:
            payload = {
                "method": str(method),
                "text": str(text),
                "error_code": error_code,
                "error_name": _ERROR_CODE_NAMES.get(error_code, "UNKNOWN"),
                "message": str(getattr(result, "message", "")),
                "wall_time_sec": wall_time_sec,
                "scene_id": str(getattr(result, "scene_id", "")),
                "result_manifest_uri": str(
                    getattr(result, "result_manifest_uri", "")
                ),
                "result_manifest_sha256": str(
                    getattr(result, "result_manifest_sha256", "")
                ),
                "scene_glb_uri": str(getattr(result, "scene_glb_uri", "")),
                "frame_count": int(getattr(result, "frame_count", 0)),
            }
        except BaseException:
            return
        with self._lock:
            self.last_terminal = payload

    def record_finished(self, finished: JobFinished) -> None:
        """最佳努力 worker 完成观察者；绝不抛出异常。"""
        try:
            payload = {
                "job_id": int(finished.job_id),
                "cause": str(finished.cause),
                "returncode": int(finished.returncode),
                "elapsed_sec": float(finished.elapsed_sec),
            }
        except BaseException:
            return
        with self._lock:
            self.last_finished = payload

    def snapshot(self) -> dict[str, Any]:
        """线程安全的诊断快照。"""
        with self._lock:
            return {
                "config": self.config,
                "not_ready_error": self.not_ready_error,
                "last_terminal": dict(self.last_terminal)
                if self.last_terminal is not None
                else None,
                "last_finished": dict(self.last_finished)
                if self.last_finished is not None
                else None,
            }


def sample_board_resources(
    config: ProductionConfig,
    *,
    meminfo_reader: Callable[[], str] | None = None,
    statvfs: Callable[..., os.statvfs_result] = os.statvfs,
) -> dict[str, Any]:
    """仅用于诊断的板载资源采样（MemAvailable/Swap/结果盘/快照盘）。"""
    try:
        reader = meminfo_reader or _read_proc_meminfo
        meminfo = reader()
        mem_available, swap_used = parse_meminfo(meminfo)
        swap_total = _parse_swap_total(meminfo)
    except BaseException as error:
        return {"error": f"meminfo probe failed: {type(error).__name__}"}

    try:
        result_free = _statvfs_free(statvfs(config.result_root))
    except BaseException as error:
        return {"error": f"result disk probe failed: {type(error).__name__}"}
    try:
        # 快照盘取所有允许根目录中的最小剩余空间
        snapshot_free = min(
            _statvfs_free(statvfs(root)) for root in config.allowed_snapshot_roots
        )
    except BaseException as error:
        return {"error": f"snapshot disk probe failed: {type(error).__name__}"}

    return {
        "mem_available_bytes": int(mem_available),
        "swap_total_bytes": int(swap_total),
        "swap_used_bytes": int(swap_used),
        "result_free_bytes": int(result_free),
        "snapshot_free_bytes": int(snapshot_free),
    }


class ProductionNode:
    """运行时监督节点：配置 → 规划器 → Action 服务端 → 诊断。"""

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        param_loader: Callable[[], Mapping[str, Any]] | None = None,
        config_builder: Callable[..., ProductionConfig] = load_production_config,
        runtime_builder: Callable[..., ProductionRuntimeParams] = load_runtime_params,
        planner_factory: Callable[..., ProductionPlanner] = ProductionPlanner,
        server_factory: Callable[..., ReconstructActionServer] = ReconstructActionServer,
        board_sampler: Callable[..., dict[str, Any]] = sample_board_resources,
        publisher_factory: Callable[..., Any] = rospy.Publisher,
        timer_factory: Callable[..., Any] = rospy.Timer,
        auto_start: bool = False,
    ) -> None:
        """构造节点；任何装配失败都进入 not-ready 状态而非崩溃。

        Args:
            params: 节点参数字典（与 param_loader 二选一）
            param_loader: 参数加载函数（如 rospy 私有参数）
            config_builder: 生产配置构造器（测试注入点）
            runtime_builder: 运行时参数构造器（测试注入点）
            planner_factory: 生产规划器工厂（测试注入点）
            server_factory: Action 服务端工厂（测试注入点）
            board_sampler: 板载资源采样器（测试注入点）
            publisher_factory / timer_factory: ROS 发布器/定时器工厂
            auto_start: 构造后立即开始发布诊断
        """
        if param_loader is not None:
            if params is not None:
                raise TypeError("provide params or param_loader, not both")
            params = param_loader()
        if params is None:
            raise TypeError("node parameters are required")
        self._params = dict(params)
        self._state = DiagnosticsState()
        self._runtime = None
        self._config = None
        self._config_builder = config_builder
        self._planner_factory = planner_factory
        self._server_factory = server_factory
        self._board_sampler = board_sampler
        self._publisher_factory = publisher_factory
        self._timer_factory = timer_factory
        self._publisher = None
        self._timer = None
        self._server = None
        self._planner = None

        try:
            self._runtime = runtime_builder(params)
        except Exception as error:
            self._state.set_not_ready(error)

        if self._runtime is not None:
            try:
                config = config_builder(params)
            except Exception as error:
                self._state.set_not_ready(error)
                self._config = None
            else:
                self._config = config
                try:
                    self._state.set_config(config)
                    self._planner = planner_factory(config)
                    self._server = server_factory(
                        action_name=self._runtime.action_name,
                        command_builder=self._planner,
                        worker_timeout_sec=config.worker_timeout_sec,
                        term_grace_sec=config.term_grace_sec,
                        terminal_observer=self._state.record_terminal,
                        finished_observer=self._state.record_finished,
                        controller_factory=_controller_factory_for(config),
                    )
                except Exception as error:
                    self._state.set_not_ready(error)
                    self._planner = None
                    self._server = None
        if auto_start:
            self.start_publishing()

    @property
    def config(self) -> ProductionConfig | None:
        return self._config

    @property
    def server(self) -> ReconstructActionServer | None:
        return self._server

    def is_ready(self) -> bool:
        """服务端是否已成功构造。"""
        return self._server is not None

    def start_publishing(self) -> None:
        """创建诊断发布器与周期定时器。"""
        if self._publisher is not None:
            return
        if self._runtime is None:
            try:
                rospy.logerr(
                    "diagnostics publishing skipped: runtime wiring is invalid"
                )
            except BaseException:
                pass
            return
        self._publisher = self._publisher_factory(
            self._runtime.diagnostics_topic, DiagnosticArray, queue_size=1
        )
        self._timer = self._timer_factory(
            rospy.Duration.from_sec(self._runtime.diagnostics_period_sec),
            self._publish_callback,
        )

    def publish_once(self) -> None:
        """立即发布一次诊断（供测试）。"""
        self._publish_callback(None)

    def _publish_callback(self, _event: Any) -> None:
        """周期发布回调：组装并发布 DiagnosticArray（最佳努力）。"""
        if self._publisher is None:
            return
        try:
            message = assemble_diagnostic_array(
                state=self._state.snapshot(),
                server_snapshot=self._server_snapshot(),
                board_sample=self._board_sampler(self._config)
                if self._config is not None
                else {"error": "production config is not ready"},
            )
            try:
                message.header.stamp = rospy.Time.now()
            except BaseException:
                message.header.stamp = rospy.Time(0)
            self._publisher.publish(message)
        except BaseException as error:
            # 诊断绝不能动摇监督进程
            try:
                rospy.logerr(
                    "diagnostics publisher failed: %s", type(error).__name__
                )
            except BaseException:
                pass

    def _server_snapshot(self) -> dict[str, Any]:
        """服务端诊断快照；未构造时返回 not_ready 占位。"""
        if self._server is None:
            return {
                "running": False,
                "stage": "not_ready",
                "job_id": None,
                "cancel_requested": False,
                "accepted": False,
                "accepted_count": 0,
                "busy_rejection_count": 0,
                "cancel_request_count": 0,
                "terminal_attempt_count": 0,
            }
        return self._server.diagnostics_snapshot()

    def shutdown(self, timeout_sec: float = 3.0) -> None:
        """关闭节点：停止定时器、关闭服务端、注销发布器。"""
        if self._timer is not None:
            try:
                self._timer.shutdown()
            except BaseException:
                pass
            self._timer = None
        if self._server is not None:
            self._server.shutdown(timeout_sec)
        if self._publisher is not None:
            try:
                self._publisher.unregister()
            except BaseException:
                pass
            self._publisher = None


def assemble_diagnostic_array(
    *,
    state: Mapping[str, Any],
    server_snapshot: Mapping[str, Any],
    board_sample: Mapping[str, Any],
) -> DiagnosticArray:
    """纯函数组装一条 /diagnostics DiagnosticArray 消息。"""
    config = state.get("config")
    not_ready_error = state.get("not_ready_error")
    message = DiagnosticArray()
    message.status = [
        _supervisor_status(state, config, not_ready_error),
        _job_status(state, server_snapshot, not_ready_error),
        _resources_status(config, board_sample, not_ready_error),
        _capture_unavailable_status(),
    ]
    return message


def _supervisor_status(
    state: Mapping[str, Any],
    config: ProductionConfig | None,
    not_ready_error: str | None,
) -> DiagnosticStatus:
    """监督进程状态：not-ready 时 ERROR，否则 OK + 画像/版本摘要。"""
    status = DiagnosticStatus()
    status.name = "drone3d/reconstruct/supervisor"
    if not_ready_error is not None or config is None:
        status.level = DiagnosticStatus.ERROR
        status.message = "not ready: " + (not_ready_error or "no production config")
        status.values = [_kv("error", status.message)]
        return status
    status.level = DiagnosticStatus.OK
    status.message = "ready"
    status.values = [
        _kv("profile_name", config.profile_name),
        _kv("checkpoint_sha256", config.checkpoint_sha256),
        _kv("algorithm_revision", config.algorithm_revision),
        _kv("ros_revision", config.ros_revision),
        _kv("worker_timeout_sec", config.worker_timeout_sec),
        _kv("term_grace_sec", config.term_grace_sec),
        _kv("verifier_timeout_sec", config.verifier_timeout_sec),
        _kv("frame_group_size", config.frame_group_size),
        _kv("image_resolution", config.image_resolution),
        _kv("warm_worker_enabled", config.warm_worker_enabled),
    ]
    return status


def _job_status(
    state: Mapping[str, Any],
    server: Mapping[str, Any],
    not_ready_error: str | None,
) -> DiagnosticStatus:
    """作业状态：运行阶段、计数与最近 worker 退出/结果摘要。"""
    status = DiagnosticStatus()
    status.name = "drone3d/reconstruct/job"
    if not_ready_error is not None:
        status.level = DiagnosticStatus.ERROR
        status.message = "Action server not constructed"
    else:
        status.level = DiagnosticStatus.OK
        status.message = (
            "stage " + str(server.get("stage", "unknown"))
            if server.get("running")
            else "idle"
        )
    values = [
        _kv("running", bool(server.get("running", False))),
        _kv("stage", server.get("stage", "unknown")),
        _kv("job_id", server.get("job_id", None)),
        _kv("accepted_count", server.get("accepted_count", 0)),
        _kv("busy_rejection_count", server.get("busy_rejection_count", 0)),
        _kv("warm_fallback_count", server.get("warm_fallback_count", 0)),
        _kv("cancel_request_count", server.get("cancel_request_count", 0)),
        _kv("terminal_attempt_count", server.get("terminal_attempt_count", 0)),
    ]
    last_finished = state.get("last_finished")
    values.append(
        _kv("worker_exit_cause", last_finished.get("cause", "unavailable")
            if last_finished else "unavailable")
    )
    values.append(
        _kv("worker_exit_returncode", last_finished.get("returncode", "unavailable")
            if last_finished else "unavailable")
    )
    last_terminal = state.get("last_terminal")
    values.append(
        _kv("last_result_method", last_terminal.get("method", "unavailable")
            if last_terminal else "unavailable")
    )
    values.append(
        _kv("last_result_error", last_terminal.get("error_name", "unavailable")
            if last_terminal else "unavailable")
    )
    values.append(
        _kv("last_result_message", last_terminal.get("message", "unavailable")
            if last_terminal else "unavailable")
    )
    if last_terminal:
        values.append(_kv("last_result_scene_id", last_terminal["scene_id"]))
        values.append(_kv("last_result_wall_time_sec", last_terminal["wall_time_sec"]))
        values.append(_kv("worker_wall_time_sec", last_terminal["wall_time_sec"]))
    status.values = values
    return status


def _resources_status(
    config: ProductionConfig | None,
    board: Mapping[str, Any],
    not_ready_error: str | None,
) -> DiagnosticStatus:
    """板载资源状态：对照限额给出 OK/WARN，越界项逐条列出。"""
    status = DiagnosticStatus()
    status.name = "drone3d/reconstruct/resources"
    if "error" in board or config is None:
        status.level = DiagnosticStatus.ERROR
        status.message = (
            not_ready_error or str(board.get("error", "resource probe unavailable"))
        )
        status.values = [_kv("error", status.message)]
        return status

    breaches = []
    if board["mem_available_bytes"] < config.resource_limits.min_mem_available_bytes:
        breaches.append("MemAvailable below floor")
    if board["swap_used_bytes"] > config.resource_limits.max_swap_used_bytes:
        breaches.append("Swap used above ceiling")
    if board["result_free_bytes"] < config.resource_limits.min_result_free_bytes:
        breaches.append("result disk below floor")
    if board["snapshot_free_bytes"] < config.resource_limits.min_snapshot_free_bytes:
        breaches.append("snapshot disk below floor")
    status.level = DiagnosticStatus.WARN if breaches else DiagnosticStatus.OK
    status.message = "; ".join(breaches) if breaches else "board resources sampled"
    status.values = [
        _kv("mem_available_bytes", board["mem_available_bytes"]),
        _kv("min_mem_available_bytes", config.resource_limits.min_mem_available_bytes),
        _kv("swap_used_bytes", board["swap_used_bytes"]),
        _kv("swap_total_bytes", board["swap_total_bytes"]),
        _kv("max_swap_used_bytes", config.resource_limits.max_swap_used_bytes),
        _kv("result_free_bytes", board["result_free_bytes"]),
        _kv("min_result_free_bytes", config.resource_limits.min_result_free_bytes),
        _kv("snapshot_free_bytes", board["snapshot_free_bytes"]),
        _kv("min_snapshot_free_bytes", config.resource_limits.min_snapshot_free_bytes),
    ]
    return status


def _capture_unavailable_status() -> DiagnosticStatus:
    """采集生产者状态：Phase 1 重建切片不含采集节点 → STALE。"""
    status = DiagnosticStatus()
    status.name = "drone3d/reconstruct/capture"
    status.level = DiagnosticStatus.STALE
    status.message = (
        "capture producer is not part of the reconstruction node "
        "(Phase 1 reconstruction-only slice)"
    )
    status.values = [
        _kv("input_age", "unavailable"),
        _kv("capture_gap", "unavailable"),
        _kv("camera_topic", "unavailable"),
    ]
    return status


def _kv(key: str, value: Any) -> KeyValue:
    """构造 KeyValue 项（None 值显示为 unavailable）。"""
    item = KeyValue()
    item.key = key
    item.value = "unavailable" if value is None else str(value)
    return item


def _required(params: Mapping[str, Any], key: str) -> Any:
    """读取必填参数，缺失即抛配置错误。"""
    if key not in params:
        raise ProductionNodeConfigError(f"missing required parameter: {key}")
    return params[key]


def _string_value(value: Any) -> str:
    """要求值为非空字符串（拒绝 NUL）。"""
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProductionNodeConfigError(
            f"parameter must be a nonempty string without NUL: {value!r}"
        )
    return value


def _integer_value(value: Any, *, allow_decimal_string: bool = False) -> int:
    """要求值为整数；字节数可用 ROS 兼容的十进制字符串。"""
    if allow_decimal_string and isinstance(value, str):
        if DECIMAL_BYTES_RE.fullmatch(value) is None:
            raise ProductionNodeConfigError(
                f"byte parameter must be a canonical decimal string: {value!r}"
            )
        return int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProductionNodeConfigError(
            f"parameter must be an integer, got {type(value).__name__}"
        )
    return value


def _finite_number(value: Any) -> float:
    """要求值为有限数值（拒绝 bool/NaN/无穷）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionNodeConfigError(
            f"parameter must be a finite number, got {type(value).__name__}"
        )
    try:
        converted = float(value)
        if converted != converted or converted in (float("inf"), float("-inf")):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ProductionNodeConfigError(f"parameter must be finite: {value!r}")
    return converted


def _read_proc_meminfo() -> str:
    """限长读取 /proc/meminfo（上限 1MB）。"""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("/proc/meminfo", flags)
    try:
        data = bytearray()
        while True:
            block = os.read(descriptor, 4096)
            if not block:
                break
            data.extend(block)
            if len(data) > _MEMINFO_MAX_BYTES:
                raise ResourceGateError("/proc/meminfo exceeded bound")
    finally:
        os.close(descriptor)
    return bytes(data).decode("ascii", "strict")


def _parse_swap_total(meminfo: str) -> int:
    """从 meminfo 文本提取 SwapTotal（字节）。"""
    for line in meminfo.splitlines():
        if line.startswith("SwapTotal:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    raise ResourceGateError("SwapTotal unavailable")


def _statvfs_free(info: os.statvfs_result) -> int:
    """从 statvfs 结果计算可用字节数。"""
    return info.f_bavail * info.f_frsize


def _rospy_param_loader() -> Mapping[str, Any]:
    """从 rospy 私有参数命名空间 (~) 读取全部参数。"""
    params = rospy.get_param("~", {})
    if not isinstance(params, Mapping):
        raise ProductionNodeConfigError("rospy private parameters are not a mapping")
    return params


def main(argv=None) -> int:
    """rospy 入口：scripts/reconstruct_production_node.py 的实际执行体。"""
    rospy.init_node(NODE_NAME, argv=argv)
    node = ProductionNode(param_loader=_rospy_param_loader, auto_start=True)
    from std_srvs.srv import Trigger, TriggerResponse
    def release_worker(_request):
        try:
            if node.server is None:raise RuntimeError("supervisor is not ready")
            node.server.release_idle_worker()
            return TriggerResponse(success=True,message="idle model weights released")
        except Exception as error:return TriggerResponse(success=False,message=str(error))
    def gpu_available(_request):
        try:
            if node.server is None:raise RuntimeError("supervisor is not ready")
            from drone3d_resources import require_gpu_available
            require_gpu_available(allowed_owner_pid=node.server.resident_worker_pid())
            return TriggerResponse(success=True,message="GPU available to this backend")
        except Exception as error:return TriggerResponse(success=False,message=str(error))
    _resource_services=[rospy.Service("~release_idle_worker",Trigger,release_worker),
                       rospy.Service("~gpu_available",Trigger,gpu_available)]
    if not node.is_ready():
        rospy.logerr(
            "reconstruction supervisor is not ready; diagnostics will report "
            "the deployment error"
        )
    else:
        rospy.loginfo(
            "reconstruction supervisor ready with profile %s",
            node.config.profile_name,
        )
    rospy.on_shutdown(lambda: node.shutdown(timeout_sec=node.config.worker_timeout_sec
                                            + node.config.term_grace_sec
                                            + 1.0
                                            if node.config is not None else 3.0))
    rospy.spin()
    return 0


__all__ = [
    "DiagnosticsState",
    "NODE_NAME",
    "ProductionNode",
    "ProductionNodeConfigError",
    "ProductionRuntimeParams",
    "assemble_diagnostic_array",
    "load_production_config",
    "load_runtime_params",
    "sample_board_resources",
]
