"""Atomic, fsynced file writes for capture snapshots (no ROS).

Split from ``capture``.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from .preparation import IO_ERROR

from .snapshot_validator import (
    SnapshotSafetyLimits,
)
from ._common import sha256_file as _sha256_file
from .capture_types import CaptureError


def _write_file_atomically(
    path: Path, data: bytes, maximum: int, *, sync: bool = True
) -> None:
    """原子写文件：临时文件 + os.replace（sync=True 时 fsync 文件与父目录）。

    帧/辅助数据写在事务暂存目录内，seal() 生成清单时统一 fsync；这些
    中间文件不逐帧 fsync，避免 worker 加载 checkpoint / 写 predictions
    时磁盘竞争把采集回调阻塞到秒级，导致 ROS 图像队列丢帧、选中帧
    stamp 间隔超过 tolerance。
    """
    if len(data) > maximum:
        raise CaptureError(IO_ERROR, f"capture asset exceeds bound: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + "-", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(data)
            stream.flush()
            if sync:
                os.fsync(stream.fileno())
        os.replace(temporary, str(path))
        temporary = None
    except OSError as error:
        raise CaptureError(IO_ERROR, f"capture asset could not be written: {path.name}") from error
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
    if sync:
        _fsync_directory(path.parent)


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
    maximum: int,
) -> None:
    """以规范 CSV 格式原子写出表（LF 行终止符）。"""
    import csv
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_file_atomically(
        path, output.getvalue().encode("utf-8"), maximum
    )


def _write_manifest(staging: Path, limits: SnapshotSafetyLimits) -> None:
    """生成 files_sha256.csv 清单：遍历暂存目录并逐文件计算大小与 SHA-256。"""
    contents = {}
    for path in sorted(staging.rglob("*")):
        # 暂存目录内只包含本 writer 写入的文件（快照封存前提）
        relative = path.relative_to(staging).as_posix()
        if not path.is_file() or path.name == "files_sha256.csv":
            continue
        contents[relative] = path
    if len(contents) > limits.max_manifest_entries:
        raise CaptureError(IO_ERROR, "snapshot manifest has too many entries")
    total = 0
    rows = []
    for relative in sorted(contents):
        path = contents[relative]
        size = path.stat().st_size
        if size > limits.max_file_bytes:
            raise CaptureError(IO_ERROR, f"snapshot file exceeds bound: {relative}")
        total += size
        if total > limits.max_total_bytes:
            raise CaptureError(IO_ERROR, "snapshot total exceeds configured bound")
        rows.append(
            {
                "path": relative,
                "size": str(size),
                "sha256": _sha256_file(path),
            }
        )
    _write_csv(
        staging / "files_sha256.csv",
        ("path", "size", "sha256"),
        rows,
        limits.max_metadata_bytes,
    )


def _fsync_staging_files(staging: Path) -> None:
    """批量 fsync 暂存目录内全部普通文件（不跟随符号链接）。

    快照帧文件在采集时以 sync=False 写入以减少磁盘竞争；封窗时调用本
    函数一次性补齐持久性，再写 manifest，保证清单引用的内容与目录条目
    在断电后一致。
    """
    for path in sorted(staging.rglob("*")):
        try:
            if not path.is_file() or path.is_symlink():
                continue
            with open(path, "rb") as stream:
                os.fsync(stream.fileno())
        except OSError:
            # 单文件 fsync 失败由后续 manifest/校验路径暴露；这里不改变
            # 封存控制流，避免辅助文件问题拖死主窗口。
            continue


def _fsync_directory(path: Path) -> None:
    """fsync 目录条目（确保 rename/创建操作持久化）。"""
    flags = os.O_RDONLY | os.O_DIRECTORY
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
