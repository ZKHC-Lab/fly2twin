"""生产启动前的 CUDA/checkpoint 预检（由 worker Python 执行）。

本模块故意不导入 ROS；它由统一 production 启动脚本用 reconstruct
profile 中的 ``worker_python`` 执行。预检语义与单机连续重建需求一致：

- CUDA 不可用：报错并非零退出（禁止 CPU fallback / fake worker）；
- checkpoint 未指定/不可读/SHA-256 不匹配：报错并非零退出；
- 预计可用显存低于 ``--min-free-mib``：仅 warning，允许继续；
  真实 OOM 由连续调度器进入 FATAL_OOM 处理。

直接执行示例：
    python -m drone3d_reconstruction_core.cuda_preflight \
        --checkpoint-path /path/model.pt \
        --checkpoint-sha256 <64 hex> [--min-free-mib 4096]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def preflight(
    *,
    checkpoint_path: str,
    checkpoint_sha256: str,
    min_free_mib: float | None = None,
    _torch=None,
) -> int:
    """执行 CUDA/checkpoint 预检并返回进程退出码（0 表示通过）。

    测试可注入 _torch fake 对象；生产默认导入真实 torch。
    """
    torch = _torch
    if torch is None:
        try:
            import torch  # type: ignore
        except BaseException as error:
            print(f"CUDA preflight failed: torch import failed: {error}", file=sys.stderr)
            return 2

    cuda_available = bool(getattr(torch, "cuda", None) is not None and torch.cuda.is_available())
    if not cuda_available:
        print(
            "CUDA preflight failed: no usable CUDA device; "
            "production refuses to start (CPU fallback is forbidden)",
            file=sys.stderr,
        )
        return 2

    try:
        device_count = int(torch.cuda.device_count())
        device_name = str(torch.cuda.get_device_name(0))
    except BaseException as error:
        print(f"CUDA preflight failed: could not inspect device: {error}", file=sys.stderr)
        return 2
    print(f"CUDA preflight: {device_count} device(s), device 0: {device_name}")

    path = Path(checkpoint_path)
    if not path.is_file():
        print(f"CUDA preflight failed: checkpoint not readable: {path}", file=sys.stderr)
        return 2
    actual = _sha256_file(path)
    if actual != checkpoint_sha256:
        print(
            "CUDA preflight failed: checkpoint SHA-256 mismatch: "
            f"expected {checkpoint_sha256}, got {actual}",
            file=sys.stderr,
        )
        return 2
    print(f"CUDA preflight: checkpoint digest ok ({actual})")

    if min_free_mib is not None:
        try:
            free_mib = float(torch.cuda.mem_get_info()[0]) / (1024.0 * 1024.0)
        except BaseException:
            free_mib = None
        if free_mib is not None:
            print(f"CUDA preflight: free VRAM {free_mib:.1f} MiB")
            if free_mib < float(min_free_mib):
                print(
                    "CUDA preflight warning: expected free VRAM "
                    f"{free_mib:.1f} MiB < required {float(min_free_mib):.1f} MiB; "
                    "starting anyway, real OOM will enter FATAL_OOM",
                    file=sys.stderr,
                )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--min-free-mib", type=float, default=None)
    args = parser.parse_args(argv)
    if not _HEX_DIGEST.fullmatch(args.checkpoint_sha256):
        print("CUDA preflight failed: checkpoint-sha256 must be 64 lowercase hex", file=sys.stderr)
        return 2
    return preflight(
        checkpoint_path=args.checkpoint_path,
        checkpoint_sha256=args.checkpoint_sha256,
        min_free_mib=args.min_free_mib,
    )


if __name__ == "__main__":
    raise SystemExit(main())
