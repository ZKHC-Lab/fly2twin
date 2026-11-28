"""正式结果门禁与 session/window 正式记录发布（ROS 无关）。

工作包 7 的真实性门禁只做软件能证明的事：

- 正式结果必须通过现有 Reconstruct Action 的 validator + 原子提交路径产生；
- 调度器在把窗口标记为 ``SUCCEEDED`` 之前，重新读取该 result.json 并核对：
  schema、窗口身份（scene/snapshot/role/topic/frame/实际 N）、checkpoint SHA、
  algorithm/ros revision、worker 返回码与必需本地工件的存在性/大小/摘要；
- 门禁通过后，把 result.json 的原始字节与完整 provenance 通过
  ``ResultTransaction``（renameat2 RENAME_NOREPLACE）原子发布到
  ``result_root/sessions/<session_id>/window-<NNNN>/``；
- 正式 scanner 只认这种记录目录，fake/外来目录不会被列入正式结果。

本模块不导入 rospy，不启动 worker，也不重跑 validator；validator 是
Reconstruct Action 内部的硬门禁，本模块验证的是它的可追溯产物。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import unquote, urlsplit

from .odom_local import (
    COORDINATE_MODEL_LOCAL_UNALIGNED,
    COORDINATE_ODOM_LOCAL_APPROX,
)
from .odom_local import valid_sim3_alignment
from .result_transaction import ResultTransaction
from ._common import reject_duplicate_json_fields as _reject_duplicate_json_fields

FORMAL_RECORD_SCHEMA = "drone3d.continuous.formal_record"
# v2 makes the provenance checksum sidecar mandatory; scanner keeps real v1 records.
FORMAL_RECORD_SCHEMA_VERSION = 2
_LEGACY_FORMAL_RECORD_SCHEMA_VERSION = 1

_WORKER_RESULT_SCHEMA = "drone3d.reconstruction.result"
_WORKER_RESULT_SCHEMA_VERSION = 1
_MAX_RESULT_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_PROVENANCE_BYTES = 16 * 1024 * 1024
_MAX_SNAPSHOT_MANIFEST_BYTES = 16 * 1024 * 1024
_ARTIFACT_DIGEST_REQUIRED = frozenset(("predictions.npz", "manifest.txt", "run.log"))
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SESSION_ID_RE = re.compile(r"s[0-9a-f]{16}\Z")
_WINDOW_DIR_RE = re.compile(r"window-[0-9]{4,}\Z")
_COORDINATE_STATES = frozenset(
    (
        COORDINATE_MODEL_LOCAL_UNALIGNED,
        COORDINATE_ODOM_LOCAL_APPROX,
    )
)


class FormalResultError(ValueError):
    """正式结果门禁或记录发布失败。"""


@dataclass(frozen=True)
class FormalGateResult:
    """result.json 门禁结果；通过时携带严格解析后的载荷与原始字节。"""

    ok: bool
    reason: str
    payload: dict[str, Any]
    manifest_bytes: bytes


def valid_odom_local_alignment(alignment: Any) -> bool:
    """Return whether an ODOM_LOCAL_APPROX payload is a proper finite Sim(3)."""
    return (
        isinstance(alignment, dict)
        and alignment.get("coordinate_state") == COORDINATE_ODOM_LOCAL_APPROX
        and valid_sim3_alignment(alignment)
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _strict_string(value: Any, label: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise FormalResultError(f"{label} is not a bounded nonempty string")
    if any(ord(char) < 32 for char in value):
        raise FormalResultError(f"{label} contains control characters")
    return value


def _strict_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FormalResultError(f"{label} is not an integer >= {minimum}")
    return value


def _strict_nonnegative_int(value: Any, label: str) -> int:
    return _strict_int(value, label, minimum=0)


def _strict_positive_int(value: Any, label: str) -> int:
    return _strict_int(value, label, minimum=1)


def _finite_nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FormalResultError(f"{label} is not a finite number")
    converted = float(value)
    if not _finite(converted) or converted < 0:
        raise FormalResultError(f"{label} is not finite and >= 0")
    return converted


def _finite(value: float) -> bool:
    try:
        return bool(
            value == value and value != float("inf") and value != float("-inf")
        )
    except (TypeError, ValueError):
        return False


def _local_result_path(uri: str) -> Path:
    """只接受本地 file:// URI；返回不跟随符号链接的常规文件路径。"""
    if not isinstance(uri, str) or not uri.startswith("file://"):
        raise FormalResultError("result manifest uri is not a local file uri")
    try:
        split = urlsplit(uri)
    except ValueError as error:
        raise FormalResultError("result manifest uri is not parseable") from error
    if split.scheme != "file" or split.netloc or split.query or split.fragment:
        raise FormalResultError("result manifest uri is not a plain file uri")
    path = Path(unquote(split.path))
    if not path.is_absolute():
        raise FormalResultError("result manifest path is not absolute")
    try:
        info = os.lstat(str(path))
    except OSError as error:
        raise FormalResultError("result manifest file cannot be inspected") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FormalResultError("result manifest is not a trusted regular file")
    if info.st_size <= 0 or info.st_size > _MAX_RESULT_MANIFEST_BYTES:
        raise FormalResultError("result manifest size is outside the trusted bound")
    return path


def _strict_json_object(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(
            data.decode("utf-8", "strict"),
            object_pairs_hook=_reject_duplicate_json_fields,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise FormalResultError(f"result manifest is not strict JSON: {error}") from error
    if not isinstance(payload, dict):
        raise FormalResultError("result manifest must be a JSON object")
    return payload


def _artifact_local_path(uri: str, label: str) -> Path:
    """Return the local path for a file:// URI or absolute path."""
    if uri.startswith("file://"):
        try:
            split = urlsplit(uri)
        except ValueError as error:
            raise FormalResultError(f"artifact {label} uri is not parseable") from error
        if split.scheme != "file" or split.netloc or split.query or split.fragment:
            raise FormalResultError(f"artifact {label} uri is not a plain file uri")
        path = Path(unquote(split.path))
        if not path.is_absolute():
            raise FormalResultError(f"artifact {label} path is not absolute")
        return path
    path = Path(uri)
    if path.is_absolute():
        return path
    raise FormalResultError(
        f"artifact {label} uri is not a local absolute path"
    )


def _sha256_file_bounded(path: Path, label: str, declared_size: int) -> str:
    """Streaming SHA-256 bounded by the declared size; reject concurrent changes."""
    digest = hashlib.sha256()
    total = 0
    try:
        with open(str(path), "rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > declared_size:
                    raise FormalResultError(
                        f"artifact {label} grew beyond its declared size while hashing"
                    )
                digest.update(chunk)
    except FormalResultError:
        raise
    except OSError as error:
        raise FormalResultError(f"artifact {label} cannot be read") from error
    if total != declared_size:
        raise FormalResultError(
            f"artifact {label} size changed while hashing"
        )
    return digest.hexdigest()


def _stat_fingerprint(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


# ponytail: 65k metadata entries cover roughly 16k four-file windows; use a
# persistent digest index if sessions grow beyond that and cache churn matters.
@lru_cache(maxsize=65536)
def _sha256_file_for_metadata(
    path_string: str,
    device: int,
    inode: int,
    size: int,
    mtime_ns: int,
    ctime_ns: int,
    expected_digest: str,
) -> str:
    """Cache successful verification for each file identity/metadata tuple."""
    path = Path(path_string)
    actual = _sha256_file_bounded(path, path.name, size)
    try:
        current = os.lstat(str(path))
    except OSError as error:
        raise FormalResultError(
            f"artifact {path.name!r} cannot be inspected"
        ) from error
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or _stat_fingerprint(current)
        != (device, inode, size, mtime_ns, ctime_ns)
    ):
        raise FormalResultError(f"artifact {path.name!r} changed while hashing")
    # Rapid rewrites can share timestamps on the NX. Do not cache a rejected
    # digest: restoring the file must trigger a fresh verification in that case.
    if actual != expected_digest:
        raise FormalResultError(
            f"artifact {path.name!r} sha256 mismatch: "
            f"declared {expected_digest}, got {actual}"
        )
    return actual


def _verify_artifacts(payload: dict[str, Any], manifest_path: Path) -> None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise FormalResultError("result manifest artifacts are missing or empty")
    missing = sorted(_ARTIFACT_DIGEST_REQUIRED.difference(artifacts))
    if missing:
        raise FormalResultError(
            "result manifest is missing required artifacts: " + ", ".join(missing)
        )
    try:
        canonical_manifest_dir = manifest_path.parent.resolve(strict=True)
    except OSError as error:
        raise FormalResultError(
            "result manifest parent directory cannot be resolved"
        ) from error
    for name, entry in artifacts.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in (".", "..")
        ):
            raise FormalResultError("result manifest has an invalid artifact name")
        if not isinstance(entry, dict):
            raise FormalResultError(f"artifact {name!r} is not an object")
        uri = entry.get("uri")
        if not isinstance(uri, str) or not uri:
            raise FormalResultError(f"artifact {name!r} has no uri")
        size = _strict_nonnegative_int(entry.get("size"), f"artifact {name!r} size")
        if not _valid_sha256(entry.get("sha256")):
            raise FormalResultError(f"artifact {name!r} has no valid sha256")
        local_path = _artifact_local_path(uri, name)
        try:
            info = os.lstat(str(local_path))
        except OSError as error:
            raise FormalResultError(
                f"artifact {name!r} file cannot be inspected"
            ) from error
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise FormalResultError(
                f"artifact {name!r} is not a trusted regular file"
            )
        if info.st_size != size:
            raise FormalResultError(
                f"artifact {name!r} size mismatch: declared {size}, "
                f"actual {info.st_size}"
            )
        try:
            canonical_artifact = local_path.resolve(strict=True)
        except OSError as error:
            raise FormalResultError(
                f"artifact {name!r} path cannot be resolved"
            ) from error
        if canonical_artifact != canonical_manifest_dir / name:
            raise FormalResultError(
                f"artifact {name!r} must sit next to the result manifest"
            )
        if name in _ARTIFACT_DIGEST_REQUIRED:
            actual_digest = _sha256_file_for_metadata(
                str(canonical_artifact),
                *_stat_fingerprint(info),
                entry["sha256"],
            )
            if actual_digest != entry["sha256"]:
                raise FormalResultError(
                    f"artifact {name!r} sha256 mismatch: "
                    f"declared {entry['sha256']}, got {actual_digest}"
                )


def verify_result_manifest(
    uri: str,
    expected_sha256: str,
    *,
    expected_identity: dict[str, Any],
    expected_authenticity: dict[str, Any],
) -> FormalGateResult:
    """重新读取正式 result.json 并执行可证明的真实性/身份门禁。

    任何一步失败都返回 ``ok=False`` 并给出稳定原因；绝不抛异常给调用方。
    """
    try:
        path = _local_result_path(uri)
        try:
            with open(str(path), "rb") as stream:
                data = stream.read(_MAX_RESULT_MANIFEST_BYTES + 1)
        except OSError:
            return FormalGateResult(False, "result manifest cannot be read", {}, b"")
        if len(data) > _MAX_RESULT_MANIFEST_BYTES:
            return FormalGateResult(False, "result manifest exceeds size bound", {}, b"")
        if not _valid_sha256(expected_sha256):
            return FormalGateResult(False, "expected result manifest sha256 is invalid", {}, b"")
        actual = _sha256(data)
        if actual != expected_sha256:
            return FormalGateResult(
                False,
                f"result manifest sha256 mismatch: expected {expected_sha256}, got {actual}",
                {},
                b"",
            )
        payload = _strict_json_object(data)
        if payload.get("schema") != _WORKER_RESULT_SCHEMA:
            return FormalGateResult(
                False,
                f"result schema mismatch: expected {_WORKER_RESULT_SCHEMA!r}",
                {},
                b"",
            )
        if (
            type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != _WORKER_RESULT_SCHEMA_VERSION
        ):
            return FormalGateResult(
                False,
                "result schema_version is not the supported version",
                {},
                b"",
            )
        expected_checkpoint = expected_authenticity.get("checkpoint_sha256", "")
        expected_checkpoint_path = expected_authenticity.get("checkpoint_path", "")
        expected_result_root = expected_authenticity.get("result_root", "")
        expected_algorithm = expected_authenticity.get("algorithm_revision", "")
        expected_ros = expected_authenticity.get("ros_revision", "")
        if not _valid_sha256(expected_checkpoint):
            return FormalGateResult(
                False,
                "continuous session has no expected checkpoint sha256; "
                "production must load expected_reconstruction from the "
                "reconstruct profile",
                {},
                b"",
            )
        if not expected_checkpoint_path or not expected_result_root:
            return FormalGateResult(
                False,
                "continuous session has no expected checkpoint path or result "
                "root; production must load expected_reconstruction from the "
                "reconstruct profile",
                {},
                b"",
            )
        if not expected_algorithm or not expected_ros:
            return FormalGateResult(
                False,
                "continuous session has no expected algorithm/ros revision; "
                "production must load expected_reconstruction from the "
                "reconstruct profile",
                {},
                b"",
            )
        try:
            canonical_result_root = Path(expected_result_root).resolve(strict=True)
            canonical_manifest = path.resolve()
        except OSError:
            return FormalGateResult(
                False,
                "expected result root or manifest path cannot be resolved",
                {},
                b"",
            )
        if os.path.commonpath(
            (str(canonical_result_root), str(canonical_manifest))
        ) != str(canonical_result_root):
            return FormalGateResult(
                False,
                "result manifest is outside the configured worker result root",
                {},
                b"",
            )
        _strict_string(payload.get("scene_id"), "result scene_id")
        if payload["scene_id"] != expected_identity.get("scene_id"):
            return FormalGateResult(
                False,
                "result scene_id does not match this window",
                {},
                b"",
            )
        if _strict_positive_int(payload.get("actual_frame_count"), "actual_frame_count") != expected_identity.get("captured_frames"):
            return FormalGateResult(
                False,
                "result actual_frame_count does not match captured frames",
                {},
                b"",
            )
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            return FormalGateResult(False, "result snapshot block is missing", {}, b"")
        if snapshot.get("sha256") != expected_identity.get("snapshot_manifest_sha256"):
            return FormalGateResult(
                False,
                "result snapshot sha256 does not match this window manifest",
                {},
                b"",
            )
        if _strict_positive_int(snapshot.get("frame_count"), "snapshot frame_count") != expected_identity.get("captured_frames"):
            return FormalGateResult(
                False,
                "result snapshot frame_count does not match captured frames",
                {},
                b"",
            )
        for field in ("camera_role", "camera_topic", "camera_frame_id"):
            actual_value = payload.get(field)
            if not isinstance(actual_value, str) or not actual_value:
                return FormalGateResult(False, f"result {field} is missing", {}, b"")
            if actual_value != expected_identity.get(field):
                return FormalGateResult(
                    False,
                    f"result {field} does not match this window",
                    {},
                    b"",
                )
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, dict):
            return FormalGateResult(False, "result checkpoint block is missing", {}, b"")
        checkpoint_path = _strict_string(checkpoint.get("path"), "checkpoint path")
        if not Path(checkpoint_path).is_absolute():
            return FormalGateResult(
                False, "result checkpoint path is not absolute", {}, b""
            )
        if checkpoint_path != expected_checkpoint_path:
            return FormalGateResult(
                False,
                "result checkpoint path does not match the reconstruct profile",
                {},
                b"",
            )
        if checkpoint.get("sha256") != expected_checkpoint:
            return FormalGateResult(
                False,
                "result checkpoint sha256 does not match the reconstruct profile",
                {},
                b"",
            )
        if _strict_string(payload.get("algorithm_revision"), "algorithm_revision") != expected_algorithm:
            return FormalGateResult(
                False,
                "result algorithm_revision does not match the reconstruct profile",
                {},
                b"",
            )
        if _strict_string(payload.get("ros_revision"), "ros_revision") != expected_ros:
            return FormalGateResult(
                False,
                "result ros_revision does not match the reconstruct profile",
                {},
                b"",
            )
        if payload.get("coordinate_state") not in _COORDINATE_STATES:
            return FormalGateResult(
                False,
                f"result coordinate_state {payload.get('coordinate_state')!r} is unknown",
                {},
                b"",
            )
        worker_exit = payload.get("worker_exit")
        if not isinstance(worker_exit, dict):
            return FormalGateResult(False, "result worker_exit block is missing", {}, b"")
        if _strict_int(worker_exit.get("returncode"), "worker returncode") != 0:
            return FormalGateResult(False, "worker exit returncode is not zero", {}, b"")
        stage_timing = payload.get("stage_timing")
        if not isinstance(stage_timing, dict):
            return FormalGateResult(False, "result stage_timing block is missing", {}, b"")
        if _finite_nonnegative_float(
            stage_timing.get("worker_wall_time_sec"), "worker wall time"
        ) <= 0:
            return FormalGateResult(False, "worker wall time is not positive", {}, b"")
        _verify_artifacts(payload, path)
        return FormalGateResult(True, "", payload, data)
    except FormalResultError as error:
        return FormalGateResult(False, str(error), {}, b"")
    except Exception as error:
        return FormalGateResult(
            False, f"formal result gate failed: {type(error).__name__}: {error}", {}, b""
        )


def _write_publication_warning(final_dir: Path, code: str, message: str) -> None:
    """Write a non-fatal warning sidecar into an already-published record."""
    try:
        payload = {
            "code": code,
            "message": message,
            "fatal": False,
        }
        with open(
            str(final_dir / "publication_warning.json"), "xb"
        ) as stream:
            stream.write(
                (
                    json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
                    + "\n"
                ).encode("utf-8")
            )
    except BaseException:
        # The record is already published; a missing warning sidecar is
        # acceptable and must not turn a published result into a failure.
        pass


def _write_provenance(staging: Path, provenance: dict[str, Any]) -> str:
    data = (
        json.dumps(provenance, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    path = staging / "provenance.json"
    with open(str(path), "xb") as stream:
        stream.write(data)
    provenance_sha256 = _sha256(data)
    with open(str(staging / "provenance.json.sha256"), "xb") as stream:
        stream.write(provenance_sha256.encode("ascii"))
    return provenance_sha256


def publish_formal_record(
    records_root: Path,
    session_id: str,
    window_index: int,
    result_manifest_bytes: bytes,
    provenance: dict[str, Any],
    snapshot_manifest_bytes: bytes = b"",
) -> tuple[Path, str]:
    """把门禁通过后的正式窗口记录原子发布到 sessions/<session>/window-NNNN/。

    返回 (final_dir, provenance_sha256)。任何失败都抛异常，由调度器把该
    窗口按普通失败处理；ResultTransaction 保证绝不会覆盖或产生半成品。
    """
    if not isinstance(records_root, Path) or not records_root.is_absolute():
        raise FormalResultError("formal records root must be an absolute Path")
    if not _SESSION_ID_RE.fullmatch(session_id or ""):
        raise FormalResultError("session_id is not scheduler-generated")
    if (
        isinstance(window_index, bool)
        or not isinstance(window_index, int)
        or window_index < 0
    ):
        raise FormalResultError("window index is not a nonnegative integer")
    final_name = f"window-{window_index:04d}"
    if _WINDOW_DIR_RE.fullmatch(final_name) is None:
        raise FormalResultError("window index is out of range")
    if not isinstance(result_manifest_bytes, bytes) or not result_manifest_bytes:
        raise FormalResultError("result manifest bytes are empty")
    if not isinstance(snapshot_manifest_bytes, bytes):
        raise FormalResultError("snapshot manifest bytes must be bytes")
    if len(snapshot_manifest_bytes) > _MAX_SNAPSHOT_MANIFEST_BYTES:
        raise FormalResultError("snapshot manifest bytes exceed size bound")
    actual_manifest_sha = _sha256(result_manifest_bytes)
    if provenance.get("schema") != FORMAL_RECORD_SCHEMA:
        raise FormalResultError("provenance schema is invalid")
    if (
        type(provenance.get("schema_version")) is not int
        or provenance.get("schema_version") != FORMAL_RECORD_SCHEMA_VERSION
    ):
        raise FormalResultError("provenance schema_version is not current")
    if provenance.get("session_id") != session_id:
        raise FormalResultError("provenance session_id does not match publication")
    if (
        type(provenance.get("window_index")) is not int
        or provenance.get("window_index") != window_index
    ):
        raise FormalResultError("provenance window_index does not match publication")
    if provenance.get("result_manifest_sha256") != actual_manifest_sha:
        raise FormalResultError(
            "provenance result_manifest_sha256 does not match manifest bytes"
        )
    if snapshot_manifest_bytes:
        declared_snapshot_sha = provenance.get("snapshot_manifest_sha256")
        if not _valid_sha256(declared_snapshot_sha):
            raise FormalResultError(
                "provenance snapshot_manifest_sha256 is missing or invalid"
            )
        actual_snapshot_sha = _sha256(snapshot_manifest_bytes)
        if declared_snapshot_sha != actual_snapshot_sha:
            raise FormalResultError(
                "provenance snapshot_manifest_sha256 does not match "
                "snapshot manifest bytes"
            )
    session_dir = records_root / session_id
    transaction = ResultTransaction(session_dir, final_name)
    try:
        with open(str(transaction.staging_path / "result.json"), "xb") as stream:
            stream.write(result_manifest_bytes)
        if snapshot_manifest_bytes:
            with open(
                str(transaction.staging_path / "snapshot_manifest.csv"), "xb"
            ) as stream:
                stream.write(snapshot_manifest_bytes)
        if provenance.get("alignment") is not None:
            alignment = provenance["alignment"]
            if not isinstance(alignment, dict):
                raise FormalResultError("provenance alignment is not an object")
            state = alignment.get("coordinate_state")
            if state == COORDINATE_ODOM_LOCAL_APPROX and not valid_sim3_alignment(alignment):
                raise FormalResultError(
                    "provenance alignment is not a valid Sim(3)"
                )
            with open(
                str(
                    transaction.staging_path
                    / "odom_local_alignment.json"
                ),
                "xb",
            ) as stream:
                stream.write(
                    (
                        json.dumps(
                            alignment, sort_keys=True, indent=2, ensure_ascii=False
                        )
                        + "\n"
                    ).encode("utf-8")
                )
        if provenance.get("depth_metrics") is not None:
            metrics = provenance["depth_metrics"]
            if not isinstance(metrics, dict):
                raise FormalResultError("provenance depth_metrics is not an object")
            with open(
                str(transaction.staging_path / "depth_metrics.json"), "xb"
            ) as stream:
                stream.write(
                    (
                        json.dumps(
                            metrics, sort_keys=True, indent=2, ensure_ascii=False
                        )
                        + "\n"
                    ).encode("utf-8")
                )
        provenance_sha256 = _write_provenance(transaction.staging_path, provenance)
        try:
            transaction.commit()
        except BaseException as error:
            if transaction.committed:
                # renameat2 已经发布；之后的防御性检查/fsync 失败只是
                # 持久性警告，目录本身是完整正式记录，绝不能重试同一
                # 最终名。
                _write_publication_warning(
                    transaction.final_path,
                    getattr(error, "code", type(error).__name__),
                    str(error),
                )
                return transaction.final_path, provenance_sha256
            raise
    except BaseException:
        transaction.cleanup()
        raise
    return transaction.final_path, provenance_sha256


def _read_bounded_regular_file(path: Path, maximum: int, label: str) -> bytes:
    """Read a regular non-symlink file, rejecting symlinks and oversize files."""
    try:
        info = os.lstat(str(path))
    except OSError as error:
        raise FormalResultError(f"{label} cannot be inspected") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FormalResultError(f"{label} is not a trusted regular file")
    if info.st_size > maximum:
        raise FormalResultError(f"{label} exceeds size bound")
    try:
        with open(str(path), "rb") as stream:
            data = stream.read(maximum + 1)
    except OSError as error:
        raise FormalResultError(f"{label} cannot be read") from error
    if len(data) > maximum:
        raise FormalResultError(f"{label} exceeds size bound")
    return data


def scan_session_windows(
    records_root: Path, session_id: str
) -> list[dict[str, Any]]:
    """扫描某一 session 的正式窗口记录（只认原子发布的 window-NNNN）。"""
    if not isinstance(records_root, Path) or not records_root.is_absolute():
        return []
    if not _SESSION_ID_RE.fullmatch(session_id or ""):
        return []
    session_dir = records_root / session_id
    try:
        if not session_dir.is_dir() or session_dir.is_symlink():
            return []
        entries = list(session_dir.iterdir())
    except OSError:
        return []
    records: list[tuple[int, dict[str, Any]]] = []
    for entry in entries:
        if not entry.is_dir() or entry.is_symlink():
            continue
        if _WINDOW_DIR_RE.fullmatch(entry.name) is None:
            continue
        provenance_path = entry / "provenance.json"
        result_path = entry / "result.json"
        if (
            not provenance_path.is_file()
            or provenance_path.is_symlink()
            or not result_path.is_file()
            or result_path.is_symlink()
        ):
            continue
        try:
            provenance_bytes = _read_bounded_regular_file(
                provenance_path, _MAX_PROVENANCE_BYTES, "provenance.json"
            )
            payload = json.loads(
                provenance_bytes.decode("utf-8", "strict"),
                object_pairs_hook=_reject_duplicate_json_fields,
            )
        except FormalResultError:
            continue
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema") != FORMAL_RECORD_SCHEMA:
            continue
        schema_version = payload.get("schema_version")
        if type(schema_version) is not int or schema_version not in (
            _LEGACY_FORMAL_RECORD_SCHEMA_VERSION,
            FORMAL_RECORD_SCHEMA_VERSION,
        ):
            continue
        if payload.get("session_id") != session_id:
            continue
        window_index = payload.get("window_index")
        if (
            isinstance(window_index, bool)
            or not isinstance(window_index, int)
            or window_index < 0
        ):
            continue
        if entry.name != f"window-{window_index:04d}":
            continue
        expected_sha = payload.get("result_manifest_sha256")
        if not _valid_sha256(expected_sha):
            continue
        gate = payload.get("gate")
        if not isinstance(gate, dict):
            continue
        if not all(
            gate.get(key) is True
            for key in (
                "result_manifest_digest_verified",
                "identity_verified",
                "authenticity_verified",
                "schema_verified",
            )
        ):
            continue
        worker = payload.get("worker")
        if not isinstance(worker, dict) or worker.get("exit_returncode") != 0:
            continue
        try:
            result_info = os.lstat(str(result_path))
            if (
                not stat.S_ISREG(result_info.st_mode)
                or result_info.st_size <= 0
                or result_info.st_size > _MAX_RESULT_MANIFEST_BYTES
            ):
                continue
            with open(str(result_path), "rb") as stream:
                result_bytes = stream.read(_MAX_RESULT_MANIFEST_BYTES + 1)
            if (
                len(result_bytes) > _MAX_RESULT_MANIFEST_BYTES
                or _sha256(result_bytes) != expected_sha
            ):
                continue
        except OSError:
            continue
        sidecar_path = entry / "provenance.json.sha256"
        try:
            sidecar_info = os.lstat(str(sidecar_path))
        except FileNotFoundError:
            sidecar_info = None
        except OSError:
            continue
        if sidecar_info is not None:
            if (
                stat.S_ISLNK(sidecar_info.st_mode)
                or not stat.S_ISREG(sidecar_info.st_mode)
            ):
                continue
            try:
                sidecar_bytes = _read_bounded_regular_file(
                    sidecar_path, 4096, "provenance.json.sha256"
                )
                sidecar_text = sidecar_bytes.decode("ascii", "strict").strip()
            except (FormalResultError, UnicodeDecodeError):
                continue
            if sidecar_text != _sha256(provenance_bytes):
                continue
            provenance_integrity = "verified"
        else:
            if (
                schema_version != _LEGACY_FORMAL_RECORD_SCHEMA_VERSION
                or "snapshot_manifest_sha256" in payload
            ):
                continue
            provenance_integrity = "legacy"
        declared_snapshot_sha = payload.get("snapshot_manifest_sha256")
        # current-v2 正式记录必须有 snapshot manifest；legacy v1 保持兼容。
        if schema_version == FORMAL_RECORD_SCHEMA_VERSION:
            if not _valid_sha256(declared_snapshot_sha):
                continue
        if declared_snapshot_sha is not None:
            if not _valid_sha256(declared_snapshot_sha):
                continue
            snapshot_manifest_path = entry / "snapshot_manifest.csv"
            try:
                snapshot_manifest_bytes = _read_bounded_regular_file(
                    snapshot_manifest_path,
                    _MAX_SNAPSHOT_MANIFEST_BYTES,
                    "snapshot_manifest.csv",
                )
            except FormalResultError:
                continue
            if _sha256(snapshot_manifest_bytes) != declared_snapshot_sha:
                continue
        if schema_version == FORMAL_RECORD_SCHEMA_VERSION:
            try:
                worker_result_path = _local_result_path(
                    payload.get("result_manifest_uri")
                )
                worker_result_info = os.lstat(str(worker_result_path))
                if worker_result_info.st_size != len(result_bytes):
                    continue
                worker_result_digest = _sha256_file_for_metadata(
                    str(worker_result_path.resolve(strict=True)),
                    *_stat_fingerprint(worker_result_info),
                    expected_sha,
                )
                if worker_result_digest != expected_sha:
                    continue
                worker_payload = _strict_json_object(result_bytes)
                _verify_artifacts(worker_payload, worker_result_path)
            except (FormalResultError, OSError):
                continue
        record_payload = dict(payload)
        record_payload["provenance_integrity"] = provenance_integrity
        records.append((window_index, record_payload))
    records.sort(key=lambda item: item[0])
    return [payload for _index, payload in records]


__all__ = [
    "FORMAL_RECORD_SCHEMA",
    "FORMAL_RECORD_SCHEMA_VERSION",
    "FormalGateResult",
    "FormalResultError",
    "publish_formal_record",
    "scan_session_windows",
    "valid_sim3_alignment",
    "valid_odom_local_alignment",
    "verify_result_manifest",
]
