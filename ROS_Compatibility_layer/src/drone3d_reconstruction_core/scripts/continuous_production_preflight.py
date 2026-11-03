#!/usr/bin/env python3
"""统一 production 启动预检（ROS supervisor 环境执行）。

预检顺序与退出语义：

1. 连接 ``ROS_MASTER_URI`` 指向的既有 ROS master；连接失败打印明确
   报警并非零退出（绝不自行启动第二个 roscore）。
2. 用 reconstruct profile 里的 ``worker_python`` 执行 CUDA/checkpoint
   预检模块；CUDA 不可用、checkpoint 缺失或 digest 不匹配均非零退出。
3. 全部通过则打印 OK 并退出 0，由调用方继续 roslaunch 主流程。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
from pathlib import Path
import subprocess
import sys

import yaml

# 可选：profile 里配置 ``preflight_min_free_vram_mib`` 时传给 worker 预检，
# 不足只 warning，不阻止启动。
_MIN_FREE_KEY = "preflight_min_free_vram_mib"


def _master_pid(master_uri: str) -> int:
    from rosgraph import Master

    master = Master("/continuous_production_preflight", master_uri=master_uri)
    try:
        pid = master.getPid()
    except BaseException as error:
        print(
            "production preflight failed: cannot reach ROS master at "
            f"{master_uri}: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise
    if pid is None:
        print(f"production preflight failed: ROS master {master_uri} returned no pid", file=sys.stderr)
        raise RuntimeError("ros master returned no pid")
    print(f"production preflight: ROS master reachable at {master_uri} (pid {pid})")
    return int(pid)


def _cuda_preflight_module_path() -> Path:
    try:
        spec = importlib.util.find_spec("drone3d_reconstruction_core.cuda_preflight")
    except BaseException:
        spec = None
    if spec is not None and spec.origin:
        return Path(spec.origin)
    here = Path(__file__).resolve()
    # 源码工作区：<pkg>/scripts/.. /src/drone3d_reconstruction_core/
    # install 工作区：<prefix>/lib/drone3d_reconstruction_core/..
    #                 /python3/dist-packages/drone3d_reconstruction_core/
    candidates = (
        here.parent.parent
        / "src"
        / "drone3d_reconstruction_core"
        / "cuda_preflight.py",
        here.parent.parent
        / "python3"
        / "dist-packages"
        / "drone3d_reconstruction_core"
        / "cuda_preflight.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("drone3d_reconstruction_core.cuda_preflight is not importable")


def _run_worker_preflight(profile: dict, module_path: Path) -> None:
    worker_python = profile.get("worker_python")
    if not isinstance(worker_python, str) or not worker_python:
        raise RuntimeError("reconstruct profile worker_python is missing")
    command = [
        worker_python,
        str(module_path),
        "--checkpoint-path",
        str(profile.get("checkpoint_path", "")),
        "--checkpoint-sha256",
        str(profile.get("checkpoint_sha256", "")),
    ]
    min_free = profile.get(_MIN_FREE_KEY)
    if min_free is not None:
        command += ["--min-free-mib", str(min_free)]
    print("production preflight: " + " ".join(command[0:1] + ["..."]))
    from drone3d_reconstruction_core.worker_runner import worker_environment
    completed = subprocess.run(command, env=worker_environment())
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker CUDA/checkpoint preflight exited with {completed.returncode}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruct-profile", required=True)
    parser.add_argument("--continuous-profile")
    args = parser.parse_args(argv)

    profile_path = Path(args.reconstruct_profile)
    try:
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except BaseException as error:
        print(f"production preflight failed: cannot read profile: {error}", file=sys.stderr)
        return 2
    if not isinstance(profile, dict):
        print("production preflight failed: profile must be a mapping", file=sys.stderr)
        return 2

    master_uri = os.environ.get(
        "ROS_MASTER_URI", "http://localhost:11311"
    ).strip()
    try:
        _master_pid(master_uri)
    except BaseException:
        return 2

    try:
        if args.continuous_profile:
            continuous = yaml.safe_load(Path(args.continuous_profile).read_text())
            if not isinstance(continuous, dict):
                raise RuntimeError("continuous profile must be a mapping")
            method = continuous.get("alignment_backend", "global_g4" if profile.get("backend", "omega") == "omega" else "odom_pose_constrained_sim3")
            if method == "global_g4":
                _check_g4_dependencies()
        module_path = _cuda_preflight_module_path()
        _run_worker_preflight(profile, module_path)
    except BaseException as error:
        print(f"production preflight failed: {error}", file=sys.stderr)
        return 2
    print("production preflight OK")
    return 0


def _check_g4_dependencies() -> None:
    """Check CPU backend dependencies in the ROS interpreter, before inference."""
    import numpy
    import scipy
    for module, minimum in ((numpy, (1, 20)), (scipy, (1, 7))):
        match = re.match(r"^(\d+)\.(\d+)", module.__version__)
        if match is None or tuple(map(int, match.groups())) < minimum:
            raise RuntimeError(f"G4 requires {module.__name__}>={minimum[0]}.{minimum[1]} in ROS Python; found {module.__version__}")
    from g4_backend import solve_g4
    if not callable(solve_g4):
        raise RuntimeError("G4 core is unavailable in ROS Python")


if __name__ == "__main__":
    raise SystemExit(main())
