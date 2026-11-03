"""Worker 运行器模块：无 shell 精确执行一条 worker 命令并捕获其日志。

本模块是重建流水线里隔离度最高的一个环节：重建 worker（VGGT-Omega 推理脚本）
由监督进程以固定 argv 启动，stdout/stderr 合并写入事务暂存目录下的 run.log。

主要安全设计：
  - 不经 shell（shell=False），避免命令注入；
  - run.log 以 O_EXCL + O_NOFOLLOW 在规范化真实目录中独占创建，
    拒绝符号链接替换与既有文件复用；
  - spawn 失败时向日志写入原因并返回固定退出码 127，由上层消费。

Run one worker argv while capturing stdout and stderr in owned staging.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from typing import Callable, Sequence


# spawn 失败时的固定退出码（对应 shell 惯例的 command not found）
SPAWN_FAILED_EXIT = 127
# 写入 run.log 首行的标记，供校验器确认日志确由本包装器产生
WORKER_RUN_LOG_MARKER = b"drone3d worker wrapper started\n"
# 监督解释器可能来自一个 conda env，而真实 worker 来自另一个（CUDA）env。
# 透传下列变量会让 worker 解释器加载错误 env 的扩展模块，必须剥除。
_ENV_KEYS_TO_STRIP = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_PYTHON_EXE",
    "CONDA_PROMPT_MODIFIER",
)


def worker_environment(base=None):
    """返回剥离监督解释器泄漏后的 worker 子进程环境。

    仅复制环境并删除 ``_ENV_KEYS_TO_STRIP``；其余变量（含 LD_LIBRARY_PATH）
    保持不变，调用方可以通过 ``base`` 注入精确环境。
    """
    if base is None:
        base = os.environ
    environment = dict(base)
    for key in _ENV_KEYS_TO_STRIP:
        environment.pop(key, None)
    # Pass only our pure-Python runtime location across interpreter boundaries.
    import drone3d_resources
    environment["DRONE3D_RUNTIME_PYTHON_ROOT"] = str(Path(drone3d_resources.__file__).resolve().parent.parent)
    return environment




def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    """校验 worker 命令：非空、全为不含 NUL 的非空字符串。"""
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError("worker command must be nonempty")
    result = []
    for argument in command:
        if not isinstance(argument, str) or not argument or "\x00" in argument:
            raise ValueError("worker command entries must be nonempty strings without NUL")
        result.append(argument)
    return tuple(result)


def _open_log(log_path: str):
    """以只写+独占创建的方式打开 run.log，拒绝符号链接与相对路径。

    关键点：先以 O_DIRECTORY|O_NOFOLLOW 打开规范化父目录句柄，
    再基于 dir_fd 创建日志文件，避免 TOCTOU 与符号链接替换。
    """
    if not isinstance(log_path, str) or not log_path or "\x00" in log_path:
        raise ValueError("log path must be a nonempty string without NUL")
    path = Path(log_path)
    if not path.is_absolute():
        raise ValueError("log path must be absolute")
    parent = path.parent
    try:
        canonical_parent = parent.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("log parent must be an existing canonical directory") from error
    # 父目录必须是规范化后的真实目录（不允许符号链接路径）
    if parent != canonical_parent or not parent.is_dir() or parent.is_symlink():
        raise ValueError("log parent must be an existing canonical real directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_flags = os.O_RDONLY | os.O_DIRECTORY
    parent_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent, parent_flags)
    try:
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    return os.fdopen(descriptor, "wb", buffering=0)


def run_worker(
    log_path: str,
    command: Sequence[str],
    *,
    popen_factory: Callable = subprocess.Popen,
) -> int:
    """无 shell 精确执行一条 worker 命令并返回其退出码。

    Args:
        log_path: 独占创建的 run.log 绝对路径
        command: worker 命令 argv（非空字符串序列）
        popen_factory: 子进程工厂，测试注入点，默认 subprocess.Popen

    Returns:
        worker 退出码；spawn 失败时返回 SPAWN_FAILED_EXIT (127)
    """
    argv = _validate_command(command)
    stream = _open_log(log_path)
    try:
        stream.write(WORKER_RUN_LOG_MARKER)
        try:
            process = popen_factory(
                list(argv),
                stdout=stream,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=False,
                env=worker_environment(),
            )
        except Exception as error:
            # 把 spawn 失败原因写入日志，方便事后排查
            stream.write(
                "worker spawn failed: {}\n".format(type(error).__name__).encode("ascii")
            )
            return SPAWN_FAILED_EXIT
        return int(process.wait())
    finally:
        stream.close()


def _parse_args(argv=None):
    """解析命令行：--log-path + `--` 之后的 worker 命令。"""
    parser = argparse.ArgumentParser(description="Capture one reconstruction worker log")
    parser.add_argument("--log-path", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a worker command is required after --")
    _validate_command(args.command)
    return args


def main(argv=None) -> int:
    """命令行入口：解析参数后执行 worker 并返回其退出码。"""
    args = _parse_args(argv)
    return run_worker(args.log_path, args.command)


if __name__ == "__main__":
    raise SystemExit(main())
