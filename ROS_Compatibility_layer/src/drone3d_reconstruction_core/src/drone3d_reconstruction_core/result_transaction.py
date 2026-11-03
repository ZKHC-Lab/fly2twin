"""结果目录事务模块：监督进程所有、绝不覆盖的原子发布。

事务在可信结果根目录下分配一个隐藏暂存目录。发布是 Linux
``renameat2(RENAME_NOREPLACE)`` 的单个原子操作——刻意没有任何复制或
覆盖回退路径。这保证了：

  - 最终目录名要么完全不存在，要么恰好一次被完整发布；
  - 并发/恶意写入者无法用符号链接或预创建目录诱导覆盖；
  - 清理只删除身份可证明（dev/ino/mount id）的暂存内容。

The transaction allocates one hidden staging directory beneath a trusted result
root.  Publication is a single Linux ``renameat2(RENAME_NOREPLACE)`` operation;
there is deliberately no copying or overwrite fallback.
"""

from __future__ import annotations

import ctypes
import errno
import os
import re
import stat
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional, Tuple


# renameat2 的 RENAME_NOREPLACE 标志与 AT_FDCWD
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
# 最终目录名的安全字符集
_SAFE_FINAL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class ResultTransactionError(RuntimeError):
    """结果事务失败基类，携带稳定错误码。

    Attributes:
        code: 稳定错误码（如 INVALID_ROOT / FINAL_EXISTS / …）
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ResultRootError(ResultTransactionError):
    """配置的可信根目录缺失或不安全。"""


class ResultNameError(ResultTransactionError):
    """请求的最终目录名不安全。"""


class ResultCollisionError(ResultTransactionError):
    """最终目录名已被占用（RENAME_NOREPLACE 返回 EEXIST）。"""


class ResultStateError(ResultTransactionError):
    """操作对当前事务状态非法（如重复提交/已清理）。"""


class ResultSafetyError(ResultTransactionError):
    """保留的文件系统身份或安全不变量发生变化。"""


class ResultCommitError(ResultTransactionError):
    """原子发布无法执行（系统调用不可用或失败）。"""


def _root_error(code: str, message: str, cause: Optional[BaseException] = None):
    """构造并抛出 ResultRootError（保留 cause 链）。"""
    error = ResultRootError(code, message)
    if cause is not None:
        raise error from cause
    raise error


def _canonical_trusted_root(value: os.PathLike) -> Tuple[Path, os.stat_result]:
    """校验并规范化可信根：绝对路径、逐组件拒软链、真实存在的目录。

    在 resolve 之前逐组件 lstat，确保通过祖先符号链接呈现的"看似规范"
    路径不被放行。返回 (规范化路径, lstat 结果)。
    """
    try:
        root = Path(value)
    except (TypeError, ValueError) as exc:
        _root_error("INVALID_ROOT", "result root is not a valid path", exc)
    if not root.is_absolute():
        _root_error("INVALID_ROOT", "result root must be absolute")

    # 在解析前检查每个命名组件，避免经祖先符号链接的规范外观路径被放行
    current = Path(root.anchor)
    try:
        for component in root.parts[1:]:
            current = current / component
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                _root_error("SYMLINK_ROOT", "result root must not traverse symlinks")
        canonical = root.resolve(strict=True)
    except FileNotFoundError as exc:
        _root_error("MISSING_ROOT", "result root does not exist", exc)
    except ResultRootError:
        raise
    except (OSError, RuntimeError) as exc:
        _root_error("INVALID_ROOT", "result root cannot be resolved safely", exc)

    if canonical != root:
        _root_error("NONCANONICAL_ROOT", "result root must be canonical")
    try:
        info = canonical.lstat()
    except OSError as exc:
        _root_error("INVALID_ROOT", "result root cannot be inspected", exc)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        _root_error("INVALID_ROOT", "result root must be a real directory")
    return canonical, info


def _validate_final_name(final_name: str) -> str:
    """校验最终目录名：单个安全 ASCII 路径组件，最长 255 字节。"""
    if not isinstance(final_name, str) or not _SAFE_FINAL_NAME.fullmatch(final_name):
        raise ResultNameError(
            "UNSAFE_FINAL_NAME",
            "final name must be one safe ASCII path component",
        )
    try:
        encoded = os.fsencode(final_name)
    except (UnicodeError, ValueError) as exc:
        raise ResultNameError("UNSAFE_FINAL_NAME", "final name cannot be encoded") from exc
    if len(encoded) > 255:
        raise ResultNameError("UNSAFE_FINAL_NAME", "final name is too long")
    return final_name


def _load_renameat2():
    """从 libc 加载 renameat2；不可用时返回 None。"""
    if not sys.platform.startswith("linux"):
        return None
    try:
        function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2")
    except (AttributeError, OSError):
        return None
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    return function


_RENAMEAT2 = _load_renameat2()


def _rename_noreplace(
    old_dir_fd: int, old_name: str, new_dir_fd: int, new_name: str
) -> None:
    """执行 renameat2(RENAME_NOREPLACE)；把 errno 映射到类型化异常。"""
    if _RENAMEAT2 is None:
        raise ResultCommitError(
            "RENAMEAT2_UNAVAILABLE",
            "Linux renameat2(RENAME_NOREPLACE) is unavailable",
        )
    ctypes.set_errno(0)
    result = _RENAMEAT2(
        old_dir_fd,
        os.fsencode(old_name),
        new_dir_fd,
        os.fsencode(new_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise ResultCollisionError("FINAL_EXISTS", "final result already exists")
    if error_number in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
        raise ResultCommitError(
            "RENAMEAT2_UNAVAILABLE",
            "renameat2(RENAME_NOREPLACE) is unsupported",
        )
    if error_number == errno.EXDEV:
        raise ResultSafetyError("CROSS_DEVICE", "staging and final are not on one filesystem")
    raise ResultCommitError(
        "COMMIT_FAILED",
        "atomic result publication failed: {}".format(os.strerror(error_number)),
    )


def _lstat_at(directory_fd: int, name: str) -> Optional[os.stat_result]:
    """基于 dirfd 的 lstat（不跟随符号链接）；不存在返回 None。"""
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ResultSafetyError(
            "FILESYSTEM_ERROR", "transaction path cannot be inspected"
        ) from exc


def _same_identity(info: os.stat_result, identity: Tuple[int, int]) -> bool:
    """比较 (st_dev, st_ino) 身份是否一致。"""
    return (info.st_dev, info.st_ino) == identity


def _mount_identity(directory_fd: int) -> int:
    """通过 /proc/self/fdinfo 获取已打开目录的挂载身份（不经过路径）。"""
    if not sys.platform.startswith("linux"):
        raise ResultSafetyError(
            "MOUNT_ID_UNAVAILABLE", "safe cleanup requires Linux mount identity"
        )
    try:
        with open("/proc/self/fdinfo/{}".format(directory_fd), encoding="ascii") as stream:
            for line in stream:
                key, separator, value = line.partition(":")
                if separator and key == "mnt_id":
                    return int(value.strip())
    except (OSError, ValueError) as exc:
        raise ResultSafetyError(
            "MOUNT_ID_UNAVAILABLE", "directory mount identity cannot be inspected"
        ) from exc
    raise ResultSafetyError(
        "MOUNT_ID_UNAVAILABLE", "directory mount identity is not exposed by procfs"
    )


class ResultTransaction:
    """拥有一个隐藏暂存目录并恰好一次原子发布的目录事务。

    关键语义：
      - 构造时校验根目录（无软链/规范化/真实目录）并独占分配暂存目录；
      - commit() 用 renameat2(RENAME_NOREPLACE) 原子发布，绝不覆盖；
      - cleanup() 只删除身份可证明的未提交暂存（含挂载边界检查）；
      - 提交后事务关闭；未提交时支持上下文管理器自动清理。
    """

    def __init__(self, trusted_root: os.PathLike, final_name: str) -> None:
        canonical_root, root_info = _canonical_trusted_root(trusted_root)
        safe_name = _validate_final_name(final_name)
        flags = os.O_RDONLY | os.O_DIRECTORY
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            root_fd = os.open(str(canonical_root), flags)
        except OSError as exc:
            raise ResultRootError("INVALID_ROOT", "result root cannot be opened") from exc
        staging_path = None
        staging_identity = None
        try:
            opened_root = os.fstat(root_fd)
            # 打开后的身份必须与 lstat 时一致（防替换竞态）
            if not _same_identity(opened_root, (root_info.st_dev, root_info.st_ino)):
                raise ResultRootError("ROOT_CHANGED", "result root identity changed")
            if _lstat_at(root_fd, safe_name) is not None:
                raise ResultCollisionError("FINAL_EXISTS", "final result already exists")
            staging_text = tempfile.mkdtemp(
                prefix=".drone3d-result-", dir=str(canonical_root)
            )
            staging_path = Path(staging_text)
            staging_info = staging_path.lstat()
            staging_identity = (staging_info.st_dev, staging_info.st_ino)
            if (
                not stat.S_ISDIR(staging_info.st_mode)
                or stat.S_ISLNK(staging_info.st_mode)
                or staging_info.st_dev != opened_root.st_dev
            ):
                raise ResultSafetyError(
                    "UNSAFE_STAGING", "allocated staging directory is unsafe"
                )
        except BaseException:
            if staging_path is not None and staging_identity is not None:
                try:
                    self._rollback_allocated_staging(
                        root_fd, staging_path.name, staging_identity
                    )
                except BaseException:
                    # 构造回滚是尽力而为且 fail-closed：绝不掩盖分配错误，
                    # 也绝不删除身份未经证明的替换物
                    pass
            os.close(root_fd)
            raise

        self.canonical_root = canonical_root
        self.final_path = canonical_root / safe_name
        self.staging_path = staging_path
        self._final_name = safe_name
        self._staging_name = staging_path.name
        self._root_fd = root_fd
        self._root_identity = (opened_root.st_dev, opened_root.st_ino)
        self._staging_identity = staging_identity
        self._lock = threading.RLock()
        self._commit_attempted = False
        self._committed = False
        self._cleaned = False

    @property
    def committed(self) -> bool:
        """事务是否已成功提交。"""
        with self._lock:
            return self._committed

    def _close_root_fd(self) -> None:
        """关闭根目录描述符（幂等）。"""
        root_fd = self._root_fd
        if root_fd is None:
            return
        os.close(root_fd)
        self._root_fd = None

    def _revalidate_root(self) -> os.stat_result:
        """重新校验根目录：路径规范化 + 身份一致 + 打开描述符一致。"""
        if self._root_fd is None:
            raise ResultStateError("TRANSACTION_CLOSED", "transaction descriptor is closed")
        canonical, info = _canonical_trusted_root(self.canonical_root)
        if canonical != self.canonical_root or not _same_identity(info, self._root_identity):
            raise ResultSafetyError("ROOT_CHANGED", "result root identity changed")
        opened = os.fstat(self._root_fd)
        if not _same_identity(opened, self._root_identity):
            raise ResultSafetyError("ROOT_CHANGED", "opened result root identity changed")
        return opened

    def _revalidate_uncommitted(self, require_final_absent: bool = True) -> os.stat_result:
        """重新校验未提交状态：暂存存在且身份一致、同设备、最终名可选缺席。"""
        root_info = self._revalidate_root()
        staging = _lstat_at(self._root_fd, self._staging_name)
        if staging is None:
            raise ResultSafetyError("STAGING_MISSING", "owned staging directory is missing")
        if (
            not _same_identity(staging, self._staging_identity)
            or not stat.S_ISDIR(staging.st_mode)
            or stat.S_ISLNK(staging.st_mode)
        ):
            raise ResultSafetyError(
                "STAGING_IDENTITY_CHANGED", "owned staging identity changed"
            )
        if staging.st_dev != root_info.st_dev:
            raise ResultSafetyError("CROSS_DEVICE", "staging is on another filesystem")
        final = _lstat_at(self._root_fd, self._final_name)
        if require_final_absent and final is not None:
            raise ResultCollisionError("FINAL_EXISTS", "final result already exists")
        return staging

    def commit(self) -> Path:
        """把暂存目录无替换地发布到最终名（原子线性化点）。

        Returns:
            最终路径（Path）

        Raises:
            ResultStateError: 重复提交/已清理/已尝试过提交
            ResultCollisionError: 最终名已被占用
            ResultCommitError: 发布系统调用失败
        """

        with self._lock:
            if self._committed:
                raise ResultStateError("ALREADY_COMMITTED", "result is already committed")
            if self._cleaned:
                raise ResultStateError("ALREADY_CLEANED", "transaction was cleaned")
            if self._commit_attempted:
                raise ResultStateError(
                    "COMMIT_ALREADY_ATTEMPTED", "commit may only be attempted once"
                )
            self._commit_attempted = True
            self._revalidate_uncommitted()
            _rename_noreplace(
                self._root_fd,
                self._staging_name,
                self._root_fd,
                self._final_name,
            )
            # renameat2 成功后，即使防御性复检失败也绝不能把已发布身份
            # 当作暂存处理
            self._committed = True
            try:
                final = _lstat_at(self._root_fd, self._final_name)
                if final is None or not _same_identity(final, self._staging_identity):
                    raise ResultSafetyError(
                        "COMMIT_IDENTITY_CHANGED", "published result identity is unexpected"
                    )
                try:
                    os.fsync(self._root_fd)
                except OSError as error:
                    # 发布系统调用已成功，这是不可逆提交之上的防御性
                    # 持久性失败
                    raise ResultCommitError(
                        "PUBLISHED_FSYNC_FAILED",
                        "published result directory could not be fsynced",
                    ) from error
                return self.final_path
            finally:
                self._close_root_fd()

    def cleanup(self) -> None:
        """只删除恰好属于本事务、且未提交的暂存目录。"""

        with self._lock:
            if self._committed or self._cleaned:
                return
            self._revalidate_uncommitted(require_final_absent=False)
            root_mount_identity = _mount_identity(self._root_fd)
            flags = os.O_RDONLY | os.O_DIRECTORY
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                staging_fd = os.open(self._staging_name, flags, dir_fd=self._root_fd)
            except OSError as exc:
                raise ResultSafetyError(
                    "STAGING_IDENTITY_CHANGED", "owned staging cannot be opened"
                ) from exc
            try:
                # 打开后的身份必须与登记的暂存身份一致
                if not _same_identity(os.fstat(staging_fd), self._staging_identity):
                    raise ResultSafetyError(
                        "STAGING_IDENTITY_CHANGED", "owned staging identity changed"
                    )
                # 拒绝清理被挂载的暂存目录（防跨挂载边界递归）
                if _mount_identity(staging_fd) != root_mount_identity:
                    raise ResultSafetyError(
                        "MOUNT_BOUNDARY", "cleanup refuses a mounted staging directory"
                    )
                self._remove_directory_contents(staging_fd)
                current = _lstat_at(self._root_fd, self._staging_name)
                if current is None or not _same_identity(current, self._staging_identity):
                    raise ResultSafetyError(
                        "STAGING_IDENTITY_CHANGED", "owned staging identity changed"
                    )
                os.rmdir(self._staging_name, dir_fd=self._root_fd)
            except ResultTransactionError:
                raise
            except OSError as exc:
                raise ResultSafetyError(
                    "CLEANUP_REFUSED", "owned staging could not be safely removed"
                ) from exc
            finally:
                os.close(staging_fd)
            self._cleaned = True
            self._close_root_fd()

    @staticmethod
    def _rollback_allocated_staging(
        root_fd: int, staging_name: str, staging_identity: Tuple[int, int]
    ) -> None:
        """构造失败的尽力回滚：只删除身份已证明的一次分配。"""
        current = _lstat_at(root_fd, staging_name)
        if (
            current is None
            or not _same_identity(current, staging_identity)
            or not stat.S_ISDIR(current.st_mode)
            or stat.S_ISLNK(current.st_mode)
        ):
            return
        flags = os.O_RDONLY | os.O_DIRECTORY
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        staging_fd = os.open(staging_name, flags, dir_fd=root_fd)
        try:
            if not _same_identity(os.fstat(staging_fd), staging_identity):
                return
            # 挂载身份也必须一致，否则拒绝回滚
            if _mount_identity(staging_fd) != _mount_identity(root_fd):
                return
            ResultTransaction._remove_directory_contents(staging_fd)
            current = _lstat_at(root_fd, staging_name)
            if current is not None and _same_identity(current, staging_identity):
                os.rmdir(staging_name, dir_fd=root_fd)
        finally:
            os.close(staging_fd)

    @classmethod
    def _remove_directory_contents(cls, directory_fd: int) -> None:
        """递归删除目录内容；每层都校验身份且拒绝跨越挂载边界。"""
        parent_mount_identity = _mount_identity(directory_fd)
        for name in os.listdir(directory_fd):
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                flags = os.O_RDONLY | os.O_DIRECTORY
                flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
                child_fd = os.open(name, flags, dir_fd=directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    if not _same_identity(opened, (info.st_dev, info.st_ino)):
                        raise ResultSafetyError(
                            "CLEANUP_REFUSED", "staging child identity changed"
                        )
                    if _mount_identity(child_fd) != parent_mount_identity:
                        raise ResultSafetyError(
                            "MOUNT_BOUNDARY",
                            "cleanup refuses to recurse across a mount boundary",
                        )
                    cls._remove_directory_contents(child_fd)
                    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if not _same_identity(current, (info.st_dev, info.st_ino)):
                        raise ResultSafetyError(
                            "CLEANUP_REFUSED", "staging child identity changed"
                        )
                    os.rmdir(name, dir_fd=directory_fd)
                finally:
                    os.close(child_fd)
            else:
                os.unlink(name, dir_fd=directory_fd)

    def __enter__(self) -> "ResultTransaction":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.cleanup()

    def __del__(self) -> None:
        root_fd = getattr(self, "_root_fd", None)
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass
            self._root_fd = None


def allocate_result_transaction(
    trusted_root: os.PathLike, final_name: str
) -> ResultTransaction:
    """在 ``trusted_root`` 下分配一个结果事务。

    Args:
        trusted_root: 可信结果根目录（必须绝对/规范化/真实目录）
        final_name: 最终目录名（单个安全 ASCII 组件）

    Returns:
        已分配暂存目录的 ResultTransaction
    """

    return ResultTransaction(trusted_root, final_name)


__all__ = [
    "ResultCollisionError",
    "ResultCommitError",
    "ResultNameError",
    "ResultRootError",
    "ResultSafetyError",
    "ResultStateError",
    "ResultTransaction",
    "ResultTransactionError",
    "allocate_result_transaction",
]
