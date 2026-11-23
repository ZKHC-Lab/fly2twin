"""不可变快照准入校验模块（仅用 Python 标准库）。

本模块实现快照目录（snapshot root）的严格准入校验：解析 snapshot.json、
frames.csv 与 files_sha256.csv，逐文件核对 SHA-256、大小与路径安全性，
返回不可变的 ValidatedSnapshot 描述，供重建规划器直接使用。

安全设计：
  - 拒绝符号链接（路径组件级与打开后身份双重校验）；
  - 校验器假定每个配置的允许根目录在校验与后续消费期间可信且只追加；
  - 基于路径的接口无法消除校验与后续打开之间的替换竞态（TOCTOU），
    需要对抗性写入者边界的消费者必须保留已校验的文件描述符，或使用
    等价的 dirfd/openat 设计。

The validator assumes each configured allowed root is trusted and append-only while
validation and subsequent consumption run.  It rejects symbolic links, but the
path-based interface cannot eliminate replacement races between validation and a
later open (TOCTOU).  Consumers needing an adversarial-writer boundary must retain
verified file descriptors or use an equivalent dirfd/openat design.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Tuple
from urllib.parse import unquote_to_bytes, urlsplit


SNAPSHOT_SCHEMA = "drone3d.reconstruction.snapshot"
SNAPSHOT_VERSION = 4
SUPPORTED_SNAPSHOT_VERSIONS = (SNAPSHOT_VERSION,)
# v4 帧表列（stamp、姿态关联、完整辅助数据文件与关联时间差）
FRAME_COLUMNS_V4 = (
    "frame_index",
    "image_path",
    "stamp_sec",
    "stamp_nsec",
    "pose_valid",
    "pose_delta_ms",
    "odometry_gap_ms",
    "auxiliary_path",
    "camera_info_delta_ms",
    "odometry_delta_ms",
    "depth_path",
    "depth_delta_ms",
)
# v4 snapshot.json 的字段集合（必须完全一致，不多不少）
_V4_METADATA_FIELDS = frozenset(
    (
        "schema",
        "version",
        "status",
        "scene_id",
        "frame_count",
        "camera_name",
        "image_topic",
        "first_stamp_sec",
        "first_stamp_nsec",
        "last_stamp_sec",
        "last_stamp_nsec",
        "has_camera_info",
        "has_odometry",
        "camera_role",
        "camera_frame_id",
        "camera_info_topic",
        "odometry_topic",
        "depth_topic",
        "tf_topic",
        "tf_static_topic",
        "save_sensor_depth",
        "body_frame_id",
        "has_tf",
        "has_depth",
        "camera_info_digest",
        "auxiliary_summary",
    )
)
# v4 auxiliary_summary 的字段集合（非负整数计数器，必须完全一致）
_AUX_SUMMARY_FIELDS = frozenset(
    (
        "camera_info_samples",
        "odometry_samples",
        "depth_samples",
        "tf_samples",
        "camera_info_matched_frames",
        "odometry_matched_frames",
        "depth_matched_frames",
        "tf_dropped",
        "auxiliary_errors",
    )
)
# v4 auxiliary/frame_*.json 的顶层字段集合（必须完全一致）
_AUX_FRAME_FIELDS = frozenset(
    (
        "frame_index",
        "camera_info",
        "camera_info_delta_ms",
        "odometry",
        "odometry_delta_ms",
        "depth",
    )
)
# v4 depth 条目的字段集合（必须完全一致）
_AUX_DEPTH_FIELDS = frozenset(
    (
        "stamp",
        "delta_ms",
        "frame_id",
        "encoding",
        "width",
        "height",
        "step",
        "is_bigendian",
        "data_file",
    )
)
# v4 tf.json 条目的字段集合（必须完全一致）
_TF_ENTRY_FIELDS = frozenset(
    (
        "stamp",
        "parent_frame_id",
        "child_frame_id",
        "translation",
        "rotation",
        "static",
    )
)
# tf.json 允许的最大条目数（与采集侧 CaptureConfig 的 max_tf_samples 上限一致）
MAX_TF_ENTRIES = 10000
# 单个 CameraInfo / Odometry 全量 payload 的数组长度上限
MAX_CAMERA_INFO_D_COEFFS = 16
_MAX_POSE_COVARIANCE = 36
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SCENE_ID = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9_-])?\Z"
)
_CAMERA_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_CAMERA_FRAME_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_/.-]{0,126}\Z")
_CANONICAL_UINT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
# ROS header stamp 的 sec 上限（uint32）
_STAMP_MAX_SEC = (1 << 32) - 1


class SnapshotValidationError(ValueError):
    """快照准入错误基类，携带稳定错误码。

    Attributes:
        code: 稳定错误码（INVALID_URI / SYMLINK_REJECTED / …）
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SnapshotMissingError(SnapshotValidationError):
    """必需的输入路径不存在。"""


class SnapshotInvalidError(SnapshotValidationError):
    """输入存在但畸形、不安全或校验失败。"""


@dataclass(frozen=True)
class SnapshotSafetyLimits:
    """快照准入前应用的显式资源限制（全部为正整数）。

    Attributes:
        max_frame_count: 帧数上限
        max_manifest_entries: 清单条目上限
        max_metadata_bytes: 元数据文件（snapshot.json/frames.csv/清单）字节上限
        max_file_bytes: 单文件字节上限
        max_total_bytes: 全部文件总字节上限
        max_scene_id_bytes: scene_id 编码字节上限
        max_relative_path_bytes: 相对路径编码字节上限
    """

    max_frame_count: int
    max_manifest_entries: int
    max_metadata_bytes: int
    max_file_bytes: int
    max_total_bytes: int
    max_scene_id_bytes: int
    max_relative_path_bytes: int

    def validate(self) -> None:
        """校验所有字段都是正整数（拒绝 bool/非正数）。"""
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SnapshotInvalidError(
                    "INVALID_LIMIT", "{} must be a positive integer".format(name)
                )


@dataclass(frozen=True)
class FrameStamp:
    """一个 ROS header stamp（uint32 sec / uint32 nsec），用作帧身份。"""

    sec: int
    nsec: int

    def __post_init__(self) -> None:
        _require_valid_stamp_values(self.sec, self.nsec, "frame stamp")


@dataclass(frozen=True)
class ValidatedSnapshot:
    """一次成功校验后的不可变快照描述。

    所有字段都在校验完成后一次性填充；image_paths 为绝对路径元组，
    顺序与 frames.csv 一致，含完整辅助数据（CameraInfo/Odometry/TF/深度）的
    只读回读契约。
    """

    canonical_root: Path
    scene_id: str
    image_paths: Tuple[Path, ...]
    image_basenames: Tuple[str, ...]
    frame_count: int
    manifest_sha256: str
    snapshot_version: int
    frame_stamps: Tuple[FrameStamp | None, ...]
    pose_valid: Tuple[bool | None, ...]
    pose_delta_ms: Tuple[float | None, ...]
    odometry_gap_ms: Tuple[float | None, ...]
    camera_name: str | None
    image_topic: str | None
    camera_role: str | None
    camera_frame_id: str | None
    first_stamp: FrameStamp | None
    last_stamp: FrameStamp | None
    has_camera_info: bool | None
    has_odometry: bool | None
    # ---- v4 辅助数据契约 ----
    camera_info_topic: str | None = None
    odometry_topic: str | None = None
    depth_topic: str | None = None
    tf_topic: str | None = None
    tf_static_topic: str | None = None
    save_sensor_depth: bool | None = None
    body_frame_id: str | None = None
    has_tf: bool | None = None
    has_depth: bool | None = None
    camera_info_digest: str | None = None
    auxiliary_summary: dict | None = None
    auxiliary_paths: Tuple[str | None, ...] = ()
    camera_info_delta_ms: Tuple[float | None, ...] = ()
    odometry_delta_ms: Tuple[float | None, ...] = ()
    depth_paths: Tuple[str | None, ...] = ()
    depth_delta_ms: Tuple[float | None, ...] = ()
    camera_info_payloads: Tuple[dict | None, ...] = ()
    odometry_payloads: Tuple[dict | None, ...] = ()
    depth_metadata: Tuple[dict | None, ...] = ()
    tf_entries: Tuple[dict, ...] = ()


def _invalid(code: str, message: str) -> SnapshotInvalidError:
    """快捷构造 SnapshotInvalidError。"""
    return SnapshotInvalidError(code, message)


def _decode_snapshot_uri(snapshot_uri: str) -> Path:
    """解码并校验快照 URI：仅支持 file:// 或绝对路径，拒绝遍历。"""
    if not isinstance(snapshot_uri, str) or not snapshot_uri:
        raise _invalid("INVALID_URI", "snapshot_uri must be a non-empty string")
    if "\x00" in snapshot_uri:
        raise _invalid("INVALID_URI", "snapshot_uri contains NUL")

    try:
        split = urlsplit(snapshot_uri)
    except ValueError as exc:
        raise _invalid("INVALID_URI", "snapshot_uri is malformed") from exc
    if split.scheme:
        # 只允许无 authority/query/fragment 的 file:// URI
        if split.scheme != "file" or not snapshot_uri.startswith("file://"):
            raise _invalid("INVALID_URI", "only file:// URIs are supported")
        if split.netloc or split.query or split.fragment:
            raise _invalid(
                "INVALID_URI", "file URI authority, query, and fragment must be empty"
            )
        raw_path = split.path
        # 逐字节校验百分号转义的合法性
        index = 0
        while index < len(raw_path):
            if raw_path[index] == "%":
                if index + 2 >= len(raw_path) or not all(
                    char in "0123456789abcdefABCDEF"
                    for char in raw_path[index + 1 : index + 3]
                ):
                    raise _invalid("INVALID_URI", "file URI has invalid percent encoding")
                index += 3
            else:
                index += 1
        try:
            decoded = unquote_to_bytes(raw_path).decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise _invalid("INVALID_URI", "file URI path is not valid UTF-8") from exc
        if "\x00" in decoded:
            raise _invalid("INVALID_URI", "decoded file URI contains NUL")
        path = Path(decoded)
    else:
        path = Path(snapshot_uri)

    if not path.is_absolute():
        raise _invalid("INVALID_URI", "snapshot path must be absolute")
    if ".." in path.parts:
        raise _invalid("PATH_TRAVERSAL", "snapshot path contains traversal")
    return path


def _canonical_allowed_roots(allowed_snapshot_roots: Iterable[os.PathLike]) -> Tuple[Path, ...]:
    """规范化允许根目录：全部绝对、存在、真实目录（拒绝软链）。"""
    try:
        roots = tuple(Path(root) for root in allowed_snapshot_roots)
    except (TypeError, ValueError) as exc:
        raise _invalid("INVALID_ALLOWED_ROOT", "allowed roots are invalid") from exc
    if not roots:
        raise _invalid("INVALID_ALLOWED_ROOT", "at least one allowed root is required")
    canonical = []
    for root in roots:
        if not root.is_absolute():
            raise _invalid("INVALID_ALLOWED_ROOT", "allowed roots must be absolute")
        if not root.exists():
            raise SnapshotMissingError("MISSING_ALLOWED_ROOT", "allowed root is missing")
        if root.is_symlink() or not root.is_dir():
            raise _invalid("INVALID_ALLOWED_ROOT", "allowed root must be a real directory")
        canonical.append(root.resolve(strict=True))
    return tuple(canonical)


def _is_relative_to(path: Path, root: Path) -> bool:
    """判断 path 是否位于 root 之下。"""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _reject_symlinks_from(root: Path, path: Path) -> None:
    """逐组件校验 root→path 之间不存在符号链接。"""
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise SnapshotMissingError("MISSING_INPUT", "required path is missing") from exc
        except OSError as exc:
            raise _invalid("FILESYSTEM_ERROR", "required path cannot be inspected") from exc
        if stat.S_ISLNK(mode):
            raise _invalid("SYMLINK_REJECTED", "symbolic links are not admitted")


def _require_path(root: Path, path: Path, kind: str) -> os.stat_result:
    """要求路径存在且类型正确（文件必须非硬链接）。"""
    _reject_symlinks_from(root, path)
    try:
        info = path.stat()
    except FileNotFoundError as exc:
        raise SnapshotMissingError("MISSING_INPUT", "required path is missing") from exc
    except OSError as exc:
        raise _invalid("FILESYSTEM_ERROR", "required path cannot be inspected") from exc
    if kind == "file":
        if not stat.S_ISREG(info.st_mode):
            raise _invalid("NOT_REGULAR_FILE", "required path is not a regular file")
        if info.st_nlink != 1:
            raise _invalid("HARDLINK_REJECTED", "hard-linked files are not admitted")
    if kind == "directory" and not stat.S_ISDIR(info.st_mode):
        raise _invalid("NOT_DIRECTORY", "required path is not a directory")
    return info


def _read_bounded(root: Path, path: Path, maximum: int) -> bytes:
    """限长读取元数据文件（先验大小后验长度双重检查）。"""
    info = _require_path(root, path, "file")
    if info.st_size > maximum:
        raise _invalid("METADATA_LIMIT", "metadata file exceeds configured limit")
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise SnapshotMissingError("MISSING_INPUT", "required path disappeared") from exc
    except OSError as exc:
        raise _invalid("FILESYSTEM_ERROR", "metadata file cannot be read") from exc
    if len(data) > maximum:
        raise _invalid("METADATA_LIMIT", "metadata file exceeds configured limit")
    return data


def _load_csv(data: bytes, columns: Tuple[str, ...], label: str):
    """严格解析 CSV：UTF-8、列完全一致、无空值（DictReader strict）。"""
    try:
        text = data.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise _invalid("MALFORMED_CSV", "{} is not UTF-8".format(label)) from exc
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != columns:
            raise _invalid("MALFORMED_CSV", "{} has unexpected columns".format(label))
        rows = list(reader)
    except (csv.Error, UnicodeError) as exc:
        raise _invalid("MALFORMED_CSV", "{} is malformed".format(label)) from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise _invalid("MALFORMED_CSV", "{} has malformed rows".format(label))
    return rows


def _safe_relative_path(value: str, limits: SnapshotSafetyLimits) -> PurePosixPath:
    """校验相对路径安全：非空、无 NUL/反斜杠、规范化 POSIX 相对路径。"""
    if not value or "\x00" in value or "\\" in value:
        raise _invalid("UNSAFE_PATH", "manifest path is unsafe")
    try:
        if len(value.encode("utf-8")) > limits.max_relative_path_bytes:
            raise _invalid("PATH_LIMIT", "relative path exceeds configured limit")
    except UnicodeEncodeError as exc:
        raise _invalid("UNSAFE_PATH", "relative path is not valid Unicode") from exc
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(
        part in ("", ".", "..") for part in path.parts
    ):
        raise _invalid("UNSAFE_PATH", "manifest path is not canonical and relative")
    return path


def _hash_file(root: Path, path: Path) -> str:
    """流式计算文件 SHA-256；读前/读后双重身份校验防替换竞态。"""
    before = _require_path(root, path, "file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                opened.st_dev != before.st_dev
                or opened.st_ino != before.st_ino
                or opened.st_nlink != 1
            ):
                raise _invalid("FILE_CHANGED", "file changed before verification")
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(stream.fileno())
    except FileNotFoundError as exc:
        raise SnapshotMissingError("MISSING_INPUT", "required path disappeared") from exc
    except OSError as exc:
        raise _invalid("FILESYSTEM_ERROR", "file cannot be verified") from exc
    # 读后校验：任何身份/大小/时间变化都判定为校验期间被替换
    if (
        after.st_dev != opened.st_dev
        or after.st_ino != opened.st_ino
        or after.st_nlink != 1
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or after.st_ctime_ns != opened.st_ctime_ns
    ):
        raise _invalid("FILE_CHANGED", "file changed during verification")
    return digest.hexdigest()


def _require_valid_stamp_values(sec, nsec, label):
    """校验 ROS header stamp 值域：sec∈[0,2^32-1]，nsec∈[0,999999999]。"""
    if (
        isinstance(sec, bool)
        or not isinstance(sec, int)
        or sec < 0
        or sec > _STAMP_MAX_SEC
        or isinstance(nsec, bool)
        or not isinstance(nsec, int)
        or nsec < 0
        or nsec > 999999999
    ):
        raise _invalid("INVALID_STAMP", f"{label} is not a valid ROS header stamp")


def _frame_stamp_from_json(values, label):
    """从 snapshot.json 字段解析 FrameStamp。"""
    sec = values.get(label + "_sec")
    nsec = values.get(label + "_nsec")
    _require_valid_stamp_values(sec, nsec, label)
    return FrameStamp(sec=int(sec), nsec=int(nsec))


def _optional_milliseconds(value, label):
    """解析可选毫秒字段（-1 → None），拒绝非有限/负数。"""
    if value == "-1":
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise _invalid("MALFORMED_FRAMES", f"{label} is invalid") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise _invalid("MALFORMED_FRAMES", f"{label} is not finite/nonnegative")
    return parsed


@dataclass(frozen=True)
class _SnapshotMeta:
    """snapshot.json 的解析结果。"""

    version: int
    scene_id: str
    frame_count: int
    camera_name: str | None
    image_topic: str | None
    camera_role: str | None
    camera_frame_id: str | None
    first_stamp: FrameStamp | None
    last_stamp: FrameStamp | None
    has_camera_info: bool | None
    has_odometry: bool | None
    camera_info_topic: str | None = None
    odometry_topic: str | None = None
    depth_topic: str | None = None
    tf_topic: str | None = None
    tf_static_topic: str | None = None
    save_sensor_depth: bool | None = None
    body_frame_id: str | None = None
    has_tf: bool | None = None
    has_depth: bool | None = None
    camera_info_digest: str | None = None
    auxiliary_summary: dict | None = None


def _optional_topic(value, label):
    """校验可选 ROS 话题字段：None 或绝对话题名。"""
    if value is None:
        return None
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        raise _invalid("INVALID_TOPIC", f"{label} must be None or an absolute topic")
    return value


def _optional_frame_id(value, label):
    """校验可选 frame_id 字段：None 或安全 frame id。"""
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not _CAMERA_FRAME_ID.fullmatch(value)
    ):
        raise _invalid("INVALID_FRAME", f"{label} must be None or a safe frame id")
    return value


def _aux_summary_value(metadata, name):
    """读取并校验 auxiliary_summary 的非负整数计数器。"""
    value = metadata["auxiliary_summary"][name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _invalid("MALFORMED_SCHEMA", f"auxiliary_summary/{name} must be nonnegative int")
    return int(value)


def _parse_snapshot_metadata(metadata, limits) -> _SnapshotMeta:
    """解析并校验 snapshot.json 元数据（schema/版本/字段集/场景/帧数…）。"""
    version = metadata["version"]
    if isinstance(version, bool) or version != SNAPSHOT_VERSION:
        raise _invalid(
            "UNSUPPORTED_SCHEMA",
            f"only snapshot version {SNAPSHOT_VERSION} is supported, got {version!r}",
        )
    if metadata["schema"] != SNAPSHOT_SCHEMA:
        raise _invalid("UNSUPPORTED_SCHEMA", "snapshot schema identity is unsupported")
    if set(metadata) != _V4_METADATA_FIELDS:
        raise _invalid("MALFORMED_SCHEMA", "snapshot.json has unexpected fields")
    if metadata["status"] != "PASS":
        raise _invalid("INVALID_STATUS", "snapshot status must be PASS")
    scene_id = metadata["scene_id"]
    if (
        not isinstance(scene_id, str)
        or scene_id in (".", "..")
        or not _SCENE_ID.fullmatch(scene_id)
        or len(scene_id.encode("ascii")) > limits.max_scene_id_bytes
    ):
        raise _invalid("INVALID_SCENE_ID", "scene_id is unsafe")
    frame_count = metadata["frame_count"]
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or frame_count > limits.max_frame_count
    ):
        raise _invalid("FRAME_COUNT_LIMIT", "frame_count is not positive and bounded")

    camera_name = metadata["camera_name"]
    if (
        not isinstance(camera_name, str)
        or not _CAMERA_NAME.fullmatch(camera_name)
        or len(camera_name.encode("ascii")) > 128
    ):
        raise _invalid("INVALID_CAMERA_NAME", "camera_name is unsafe")
    image_topic = metadata["image_topic"]
    if (
        not isinstance(image_topic, str)
        or not image_topic.startswith("/")
        or "\x00" in image_topic
    ):
        raise _invalid("INVALID_IMAGE_TOPIC", "image_topic must be an absolute topic")
    first_stamp = _frame_stamp_from_json(metadata, "first_stamp")
    last_stamp = _frame_stamp_from_json(metadata, "last_stamp")
    if (first_stamp.sec, first_stamp.nsec) > (last_stamp.sec, last_stamp.nsec):
        raise _invalid("INVALID_STAMP", "first_stamp exceeds last_stamp")
    for name in ("has_camera_info", "has_odometry"):
        if type(metadata[name]) is not bool:
            raise _invalid("MALFORMED_SCHEMA", f"{name} must be a bool")

    camera_role = metadata["camera_role"]
    if (
        not isinstance(camera_role, str)
        or not _CAMERA_NAME.fullmatch(camera_role)
        or len(camera_role.encode("ascii")) > 64
    ):
        raise _invalid("INVALID_CAMERA_ROLE", "camera_role is unsafe")
    camera_frame_id = metadata["camera_frame_id"]
    if (
        not isinstance(camera_frame_id, str)
        or not _CAMERA_FRAME_ID.fullmatch(camera_frame_id)
    ):
        raise _invalid("INVALID_CAMERA_FRAME", "camera_frame_id is unsafe")

    # ---- 完整辅助数据契约 ----
    for name in ("save_sensor_depth", "has_tf", "has_depth"):
        if type(metadata[name]) is not bool:
            raise _invalid("MALFORMED_SCHEMA", f"{name} must be a bool")
    camera_info_digest = metadata["camera_info_digest"]
    if camera_info_digest is not None and (
        not isinstance(camera_info_digest, str)
        or not _HEX_DIGEST.fullmatch(camera_info_digest)
    ):
        raise _invalid("INVALID_DIGEST", "camera_info_digest must be None or sha256")
    if metadata["has_camera_info"] and camera_info_digest is None:
        raise _invalid(
            "MALFORMED_SCHEMA", "has_camera_info requires camera_info_digest"
        )
    if not metadata["has_camera_info"] and camera_info_digest is not None:
        raise _invalid(
            "MALFORMED_SCHEMA", "camera_info_digest requires has_camera_info"
        )
    summary = metadata["auxiliary_summary"]
    if not isinstance(summary, dict):
        raise _invalid("MALFORMED_SCHEMA", "auxiliary_summary must be an object")
    if set(summary) != _AUX_SUMMARY_FIELDS:
        raise _invalid("MALFORMED_SCHEMA", "auxiliary_summary has unexpected fields")
    aux = {name: _aux_summary_value(metadata, name) for name in sorted(_AUX_SUMMARY_FIELDS)}
    if (
        aux["camera_info_matched_frames"] > aux["camera_info_samples"]
        or aux["odometry_matched_frames"] > aux["odometry_samples"]
        or aux["depth_matched_frames"] > aux["depth_samples"]
    ):
        raise _invalid(
            "MALFORMED_SCHEMA", "auxiliary matched counts exceed received samples"
        )
    if aux["camera_info_matched_frames"] > frame_count:
        raise _invalid(
            "MALFORMED_SCHEMA", "camera_info_matched_frames exceeds frame_count"
        )
    if aux["odometry_matched_frames"] > frame_count:
        raise _invalid(
            "MALFORMED_SCHEMA", "odometry_matched_frames exceeds frame_count"
        )
    if aux["depth_matched_frames"] > frame_count:
        raise _invalid(
            "MALFORMED_SCHEMA", "depth_matched_frames exceeds frame_count"
        )
    if metadata["has_camera_info"] != bool(aux["camera_info_matched_frames"]):
        raise _invalid(
            "MALFORMED_SCHEMA",
            "has_camera_info must equal (camera_info_matched_frames > 0)",
        )
    if metadata["has_odometry"] != bool(aux["odometry_matched_frames"]):
        raise _invalid(
            "MALFORMED_SCHEMA",
            "has_odometry must equal (odometry_matched_frames > 0)",
        )
    if metadata["has_depth"] != bool(aux["depth_matched_frames"]):
        raise _invalid(
            "MALFORMED_SCHEMA", "has_depth must equal (depth_matched_frames > 0)"
        )
    if metadata["has_tf"] != bool(aux["tf_samples"]):
        raise _invalid(
            "MALFORMED_SCHEMA", "has_tf must equal (tf_samples > 0)"
        )
    return _SnapshotMeta(
        version=version,
        scene_id=scene_id,
        frame_count=frame_count,
        camera_name=camera_name,
        image_topic=image_topic,
        camera_role=camera_role,
        camera_frame_id=camera_frame_id,
        first_stamp=first_stamp,
        last_stamp=last_stamp,
        has_camera_info=metadata["has_camera_info"],
        has_odometry=metadata["has_odometry"],
        camera_info_topic=_optional_topic(metadata["camera_info_topic"], "camera_info_topic"),
        odometry_topic=_optional_topic(metadata["odometry_topic"], "odometry_topic"),
        depth_topic=_optional_topic(metadata["depth_topic"], "depth_topic"),
        tf_topic=_optional_topic(metadata["tf_topic"], "tf_topic"),
        tf_static_topic=_optional_topic(metadata["tf_static_topic"], "tf_static_topic"),
        save_sensor_depth=metadata["save_sensor_depth"],
        body_frame_id=_optional_frame_id(metadata["body_frame_id"], "body_frame_id"),
        has_tf=metadata["has_tf"],
        has_depth=metadata["has_depth"],
        camera_info_digest=camera_info_digest,
        auxiliary_summary=aux,
    )


@dataclass(frozen=True)
class _FrameRows:
    """frames.csv 的解析结果。"""

    image_relatives: Tuple[str, ...]
    frame_stamps: Tuple[FrameStamp | None, ...]
    pose_valid: Tuple[bool | None, ...]
    pose_delta_ms: Tuple[float | None, ...]
    odometry_gap_ms: Tuple[float | None, ...]
    auxiliary_paths: Tuple[str | None, ...] = ()
    camera_info_delta_ms: Tuple[float | None, ...] = ()
    odometry_delta_ms: Tuple[float | None, ...] = ()
    depth_paths: Tuple[str | None, ...] = ()
    depth_delta_ms: Tuple[float | None, ...] = ()


def _optional_relative_path(value, limits, label, prefix):
    """解析可选相对路径字段：空字符串 → None，否则必须位于指定前缀下。"""
    if value == "":
        return None
    if not isinstance(value, str):
        raise _invalid("UNSAFE_PATH", f"{label} must be empty or a path")
    relative = _safe_relative_path(value, limits)
    if len(relative.parts) < 2 or relative.parts[0] != prefix:
        raise _invalid("UNSAFE_PATH", f"{label} must be beneath {prefix}/")
    return relative.as_posix()


def _parse_frame_rows(frames_bytes, limits, frame_count) -> _FrameRows:
    """解析并校验 frames.csv：行数/顺序/stamp 递增/路径唯一且位于 images/ 下。"""
    frame_rows = _load_csv(frames_bytes, FRAME_COLUMNS_V4, "frames.csv")
    if len(frame_rows) != frame_count:
        raise _invalid("FRAME_COUNT_MISMATCH", "frames row count does not match")
    image_relatives = []
    seen_images = set()
    frame_stamps = []
    pose_valid = []
    pose_delta_ms = []
    odometry_gap_ms = []
    auxiliary_paths = []
    camera_info_delta_ms = []
    odometry_delta_ms = []
    depth_paths = []
    depth_delta_ms = []
    previous_stamp = None
    for expected_index, row in enumerate(frame_rows):
        if row["frame_index"] != str(expected_index):
            raise _invalid("FRAME_ORDER", "frame indices must be contiguous from zero")
        relative = _safe_relative_path(row["image_path"], limits)
        if len(relative.parts) < 2 or relative.parts[0] != "images":
            raise _invalid("UNSAFE_IMAGE_PATH", "image path must be beneath images/")
        relative_text = relative.as_posix()
        if relative_text in seen_images:
            raise _invalid("DUPLICATE_IMAGE", "image paths must be unique")
        seen_images.add(relative_text)
        image_relatives.append(relative_text)
        # stamp 必须是规范无符号整数且严格递增
        if (
            not _CANONICAL_UINT.fullmatch(row["stamp_sec"])
            or not _CANONICAL_UINT.fullmatch(row["stamp_nsec"])
        ):
            raise _invalid("INVALID_STAMP", "frame stamp fields are not canonical")
        stamp = FrameStamp(int(row["stamp_sec"]), int(row["stamp_nsec"]))
        if stamp.sec > _STAMP_MAX_SEC or stamp.nsec > 999999999:
            raise _invalid("INVALID_STAMP", "frame stamp exceeds ROS range")
        current = (stamp.sec, stamp.nsec)
        if previous_stamp is not None and current <= previous_stamp:
            raise _invalid("STAMP_ORDER", "frame stamps must be strictly increasing")
        previous_stamp = current
        frame_stamps.append(stamp)
        if row["pose_valid"] not in ("true", "false"):
            raise _invalid("MALFORMED_FRAMES", "pose_valid must be true or false")
        pose_valid.append(row["pose_valid"] == "true")
        delta = _optional_milliseconds(row["pose_delta_ms"], "pose_delta_ms")
        gap = _optional_milliseconds(row["odometry_gap_ms"], "odometry_gap_ms")
        if pose_valid[-1] and (delta is None or gap is None):
            raise _invalid(
                "MALFORMED_FRAMES",
                "valid pose rows must carry pose_delta_ms and odometry_gap_ms",
            )
        pose_delta_ms.append(delta)
        odometry_gap_ms.append(gap)
        aux_path = _optional_relative_path(
            row["auxiliary_path"], limits, "auxiliary_path", "auxiliary"
        )
        ci_delta = _optional_milliseconds(
            row["camera_info_delta_ms"], "camera_info_delta_ms"
        )
        odom_delta = _optional_milliseconds(
            row["odometry_delta_ms"], "odometry_delta_ms"
        )
        depth_path = _optional_relative_path(
            row["depth_path"], limits, "depth_path", "depth"
        )
        depth_delta = _optional_milliseconds(
            row["depth_delta_ms"], "depth_delta_ms"
        )
        if ci_delta is not None and aux_path is None:
            raise _invalid(
                "MALFORMED_FRAMES",
                "camera_info_delta_ms requires auxiliary_path",
            )
        if odom_delta is not None and aux_path is None:
            raise _invalid(
                "MALFORMED_FRAMES",
                "odometry_delta_ms requires auxiliary_path",
            )
        if (depth_delta is None) != (depth_path is None):
            raise _invalid(
                "MALFORMED_FRAMES",
                "depth_delta_ms must be present iff depth_path is",
            )
        if depth_path is not None and aux_path is None:
            raise _invalid(
                "MALFORMED_FRAMES", "depth_path requires auxiliary_path"
            )
        if pose_valid[-1] and odom_delta is None:
            raise _invalid(
                "MALFORMED_FRAMES",
                "valid pose rows must carry odometry_delta_ms",
            )
        auxiliary_paths.append(aux_path)
        camera_info_delta_ms.append(ci_delta)
        odometry_delta_ms.append(odom_delta)
        depth_paths.append(depth_path)
        depth_delta_ms.append(depth_delta)
    return _FrameRows(
        image_relatives=tuple(image_relatives),
        frame_stamps=tuple(frame_stamps),
        pose_valid=tuple(pose_valid),
        pose_delta_ms=tuple(pose_delta_ms),
        odometry_gap_ms=tuple(odometry_gap_ms),
        auxiliary_paths=tuple(auxiliary_paths),
        camera_info_delta_ms=tuple(camera_info_delta_ms),
        odometry_delta_ms=tuple(odometry_delta_ms),
        depth_paths=tuple(depth_paths),
        depth_delta_ms=tuple(depth_delta_ms),
    )


def _json_object(value, label):
    """要求值为 JSON 对象（拒绝重复字段由调用方钩子保证）。"""
    if not isinstance(value, dict):
        raise _invalid("MALFORMED_AUX", f"{label} must be an object")
    return value


def _finite_float(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid("MALFORMED_AUX", f"{label} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _invalid("MALFORMED_AUX", f"{label} must be finite")
    return parsed


def _finite_float_list(value, label, length=None, maximum=None):
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise _invalid("MALFORMED_AUX", f"{label} must be a numeric array")
    if length is not None and len(value) != length:
        raise _invalid("MALFORMED_AUX", f"{label} must have length {length}")
    if maximum is not None and len(value) > maximum:
        raise _invalid("MALFORMED_AUX", f"{label} exceeds {maximum} items")
    return [_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _invalid("MALFORMED_AUX", f"{label} must be a nonnegative integer")
    return int(value)


def _stamp_object(value, label):
    obj = _json_object(value, label)
    if set(obj) != {"sec", "nsec"}:
        raise _invalid("MALFORMED_AUX", f"{label} must contain sec and nsec only")
    _require_valid_stamp_values(obj.get("sec"), obj.get("nsec"), label)
    return {"sec": int(obj["sec"]), "nsec": int(obj["nsec"])}


def _aux_header(value, label):
    obj = _json_object(value, label)
    if "stamp" not in obj or "frame_id" not in obj:
        raise _invalid("MALFORMED_AUX", f"{label} must contain stamp and frame_id")
    stamp = _stamp_object(obj["stamp"], f"{label}.stamp")
    frame_id = obj["frame_id"]
    if (
        not isinstance(frame_id, str)
        or not frame_id
        or not _CAMERA_FRAME_ID.fullmatch(frame_id)
    ):
        raise _invalid("MALFORMED_AUX", f"{label}.frame_id is unsafe")
    return stamp, frame_id


def _validate_camera_info_payload(value, label="camera_info"):
    """校验完整 CameraInfo payload：K/D/R/P、frame、stamp、尺寸均需完整可回读。"""
    obj = _json_object(value, label)
    if "header" not in obj:
        raise _invalid("MALFORMED_AUX", f"{label} must contain header")
    stamp, frame_id = _aux_header(obj["header"], f"{label}.header")
    height = _nonnegative_int(obj.get("height"), f"{label}.height")
    width = _nonnegative_int(obj.get("width"), f"{label}.width")
    distortion_model = obj.get("distortion_model")
    if not isinstance(distortion_model, str) or "\x00" in distortion_model:
        raise _invalid("MALFORMED_AUX", f"{label}.distortion_model is invalid")
    d = _finite_float_list(
        obj.get("D"), f"{label}.D", maximum=MAX_CAMERA_INFO_D_COEFFS
    )
    for name in ("K", "R"):
        _finite_float_list(obj.get(name), f"{label}.{name}", length=9)
    # 标准 sensor_msgs/CameraInfo.P 是 float64[12]；K/R 仍为 3x3。
    _finite_float_list(obj.get("P"), f"{label}.P", length=12)
    if "binning_x" in obj:
        _nonnegative_int(obj.get("binning_x"), f"{label}.binning_x")
    if "binning_y" in obj:
        _nonnegative_int(obj.get("binning_y"), f"{label}.binning_y")
    roi = obj.get("roi")
    if roi is not None:
        roi_obj = _json_object(roi, f"{label}.roi")
        for name in ("x_offset", "y_offset", "width", "height"):
            _nonnegative_int(roi_obj.get(name), f"{label}.roi.{name}")
        if "do_rectify" in roi_obj and type(roi_obj["do_rectify"]) is not bool:
            raise _invalid("MALFORMED_AUX", f"{label}.roi.do_rectify must be bool")
    return stamp, frame_id, dict(
        height=height,
        width=width,
        distortion_model=distortion_model,
        d=tuple(d),
    )


def _validate_pose(value, label):
    obj = _json_object(value, label)
    for group in ("position", "orientation"):
        part = _json_object(obj.get(group), f"{label}.{group}")
        if group == "position":
            keys = ("x", "y", "z")
        else:
            keys = ("x", "y", "z", "w")
        for key in keys:
            _finite_float(part.get(key), f"{label}.{group}.{key}")
    _finite_float_list(
        obj.get("covariance"), f"{label}.covariance", length=_MAX_POSE_COVARIANCE
    )
    return obj


def _validate_twist(value, label):
    """校验 TwistWithCovariance：linear/angular + covariance。"""
    obj = _json_object(value, label)
    for group, keys in (("linear", ("x", "y", "z")), ("angular", ("x", "y", "z"))):
        part = _json_object(obj.get(group), f"{label}.{group}")
        for key in keys:
            _finite_float(part.get(key), f"{label}.{group}.{key}")
    _finite_float_list(
        obj.get("covariance"), f"{label}.covariance", length=_MAX_POSE_COVARIANCE
    )
    return obj


def _validate_odometry_payload(value, label="odometry"):
    """校验完整 Odometry payload：header、child_frame_id、pose/twist。"""
    obj = _json_object(value, label)
    if "header" not in obj or "child_frame_id" not in obj:
        raise _invalid(
            "MALFORMED_AUX", f"{label} must contain header and child_frame_id"
        )
    stamp, frame_id = _aux_header(obj["header"], f"{label}.header")
    child_frame_id = obj["child_frame_id"]
    if (
        not isinstance(child_frame_id, str)
        or not child_frame_id
        or not _CAMERA_FRAME_ID.fullmatch(child_frame_id)
    ):
        raise _invalid("MALFORMED_AUX", f"{label}.child_frame_id is unsafe")
    _validate_pose(obj.get("pose"), f"{label}.pose")
    _validate_twist(obj.get("twist"), f"{label}.twist")
    return stamp, frame_id, child_frame_id


def _validate_depth_meta(value, label="depth"):
    """校验 depth 条目元数据；原始字节文件由 manifest 覆盖校验。"""
    obj = _json_object(value, label)
    if set(obj) != _AUX_DEPTH_FIELDS:
        raise _invalid("MALFORMED_AUX", f"{label} has unexpected fields")
    _stamp_object(obj.get("stamp"), f"{label}.stamp")
    delta = obj.get("delta_ms")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool):
        raise _invalid("MALFORMED_AUX", f"{label}.delta_ms must be a number")
    delta = float(delta)
    if not math.isfinite(delta) or delta < 0:
        raise _invalid("MALFORMED_AUX", f"{label}.delta_ms is invalid")
    frame_id = obj.get("frame_id")
    if (
        not isinstance(frame_id, str)
        or not frame_id
        or not _CAMERA_FRAME_ID.fullmatch(frame_id)
    ):
        raise _invalid("MALFORMED_AUX", f"{label}.frame_id is unsafe")
    encoding = obj.get("encoding")
    if (
        not isinstance(encoding, str)
        or not encoding
        or "\x00" in encoding
        or len(encoding.encode("ascii")) > 32
    ):
        raise _invalid("MALFORMED_AUX", f"{label}.encoding is unsafe")
    width = _nonnegative_int(obj.get("width"), f"{label}.width")
    height = _nonnegative_int(obj.get("height"), f"{label}.height")
    step = _nonnegative_int(obj.get("step"), f"{label}.step")
    is_bigendian = obj.get("is_bigendian")
    if is_bigendian not in (0, 1) or isinstance(is_bigendian, bool):
        raise _invalid("MALFORMED_AUX", f"{label}.is_bigendian must be 0 or 1")
    data_file = obj.get("data_file")
    if not isinstance(data_file, str) or not data_file:
        raise _invalid("MALFORMED_AUX", f"{label}.data_file must be a path")
    return dict(
        delta_ms=delta,
        frame_id=frame_id,
        encoding=encoding,
        width=width,
        height=height,
        step=step,
        is_bigendian=int(is_bigendian),
        data_file=data_file,
    )


def _validate_tf_entry(value, index):
    label = f"tf.json[{index}]"
    obj = _json_object(value, label)
    if set(obj) != _TF_ENTRY_FIELDS:
        raise _invalid("MALFORMED_AUX", f"{label} has unexpected fields")
    _stamp_object(obj.get("stamp"), f"{label}.stamp")
    for name in ("parent_frame_id", "child_frame_id"):
        frame = obj.get(name)
        if (
            not isinstance(frame, str)
            or not frame
            or not _CAMERA_FRAME_ID.fullmatch(frame)
        ):
            raise _invalid("MALFORMED_AUX", f"{label}.{name} is unsafe")
    translation = _json_object(obj.get("translation"), f"{label}.translation")
    if set(translation) != {"x", "y", "z"}:
        raise _invalid("MALFORMED_AUX", f"{label}.translation has unexpected fields")
    for key in ("x", "y", "z"):
        _finite_float(translation.get(key), f"{label}.translation.{key}")
    rotation = _json_object(obj.get("rotation"), f"{label}.rotation")
    if set(rotation) != {"x", "y", "z", "w"}:
        raise _invalid("MALFORMED_AUX", f"{label}.rotation has unexpected fields")
    for key in ("x", "y", "z", "w"):
        _finite_float(rotation.get(key), f"{label}.rotation.{key}")
    if type(obj.get("static")) is not bool:
        raise _invalid("MALFORMED_AUX", f"{label}.static must be bool")
    return obj


def _parse_aux_frame_json(root, path, relative, limits, expected_index, rows):
    """解析并校验 v4 auxiliary/frame_*.json，返回逐帧辅助 payload。"""
    bytes_data = _read_bounded(root, path, limits.max_metadata_bytes)
    try:
        payload = json.loads(
            bytes_data.decode("utf-8", "strict"),
            object_pairs_hook=lambda pairs: _unique_json_object(pairs, "auxiliary frame json"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid("MALFORMED_AUX", "auxiliary frame json is malformed") from exc
    obj = _json_object(payload, relative)
    if set(obj) != _AUX_FRAME_FIELDS:
        raise _invalid("MALFORMED_AUX", f"{relative} has unexpected fields")
    if obj["frame_index"] != expected_index or isinstance(obj["frame_index"], bool):
        raise _invalid("FRAME_ORDER", f"{relative} frame_index does not match row")
    ci = obj["camera_info"]
    ci_delta = obj["camera_info_delta_ms"]
    odom = obj["odometry"]
    odom_delta = obj["odometry_delta_ms"]
    depth = obj["depth"]
    if ci is None:
        if ci_delta is not None:
            raise _invalid("MALFORMED_AUX", "camera_info_delta_ms without camera_info")
    else:
        _validate_camera_info_payload(ci, f"{relative}.camera_info")
        if not isinstance(ci_delta, (int, float)) or isinstance(ci_delta, bool):
            raise _invalid("MALFORMED_AUX", "camera_info_delta_ms must be a number")
        ci_delta = float(ci_delta)
        if not math.isfinite(ci_delta) or ci_delta < 0:
            raise _invalid("MALFORMED_AUX", "camera_info_delta_ms is invalid")
    if odom is None:
        if odom_delta is not None:
            raise _invalid("MALFORMED_AUX", "odometry_delta_ms without odometry")
    else:
        _validate_odometry_payload(odom, f"{relative}.odometry")
        if not isinstance(odom_delta, (int, float)) or isinstance(odom_delta, bool):
            raise _invalid("MALFORMED_AUX", "odometry_delta_ms must be a number")
        odom_delta = float(odom_delta)
        if not math.isfinite(odom_delta) or odom_delta < 0:
            raise _invalid("MALFORMED_AUX", "odometry_delta_ms is invalid")
    depth_meta = None
    if depth is not None:
        depth_meta = _validate_depth_meta(depth, f"{relative}.depth")
        if depth_meta["data_file"] != rows.depth_paths[expected_index]:
            raise _invalid(
                "MALFORMED_AUX", f"{relative}.depth.data_file does not match frames.csv"
            )
    # frames.csv 声明存在而 JSON 缺失、或反之，均为契约违反
    if rows.auxiliary_paths[expected_index] is None:
        raise _invalid("MANIFEST_COVERAGE", "frames.csv references missing auxiliary json")
    if ci_delta != rows.camera_info_delta_ms[expected_index]:
        raise _invalid("MALFORMED_AUX", "camera_info_delta_ms mismatch with frames.csv")
    if odom_delta != rows.odometry_delta_ms[expected_index]:
        raise _invalid("MALFORMED_AUX", "odometry_delta_ms mismatch with frames.csv")
    if depth_meta is None:
        if rows.depth_paths[expected_index] is not None:
            raise _invalid("MALFORMED_AUX", "frames.csv depth_path has no depth metadata")
    elif depth_meta["delta_ms"] != rows.depth_delta_ms[expected_index]:
        raise _invalid("MALFORMED_AUX", "depth_delta_ms mismatch with frames.csv")
    return ci, ci_delta, odom, odom_delta, depth_meta


def _unique_json_object(pairs, label):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _invalid("MALFORMED_AUX", f"{label} has duplicate fields")
        result[key] = value
    return result


def _parse_tf_file(root, path, limits):
    """解析并校验 v4 tf.json，返回不可变的 TF 条目元组。"""
    bytes_data = _read_bounded(root, path, limits.max_file_bytes)
    try:
        payload = json.loads(
            bytes_data.decode("utf-8", "strict"),
            object_pairs_hook=lambda pairs: _unique_json_object(pairs, "tf.json"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid("MALFORMED_AUX", "tf.json is malformed") from exc
    if not isinstance(payload, list):
        raise _invalid("MALFORMED_AUX", "tf.json must be an array")
    if len(payload) > MAX_TF_ENTRIES:
        raise _invalid("MALFORMED_AUX", "tf.json has too many entries")
    entries = []
    for index, entry in enumerate(payload):
        entries.append(_validate_tf_entry(entry, index))
    return tuple(entries)


def validate_snapshot(
    snapshot_uri: str,
    manifest_sha256: str,
    allowed_snapshot_roots: Iterable[os.PathLike],
    limits: SnapshotSafetyLimits,
) -> ValidatedSnapshot:
    """校验并返回不可变的规范化快照描述。

    校验流程：
      1. URI 解码与允许根归属判定（拒绝越界/软链/遍历）；
      2. 限长读取 snapshot.json / frames.csv / files_sha256.csv；
      3. 校验清单摘要与 manifest_sha256 一致；
      4. 解析元数据（只支持 v4）与帧表（stamp 递增、路径唯一）；
      5. 校验清单覆盖恰为 {snapshot.json, frames.csv} ∪ 图像路径；
      6. 逐文件核对大小与 SHA-256（读前/读后双重身份检查）。

    采集只写 v4（逐帧辅助数据文件与关联时间差）；更早的 v1–v3 快照会被拒绝。

    Args:
        snapshot_uri: 快照根目录的 file:// URI 或绝对路径
        manifest_sha256: files_sha256.csv 的期望 SHA-256（小写十六进制）
        allowed_snapshot_roots: 允许的快照根目录集合（可迭代）
        limits: 准入资源限制

    Returns:
        校验通过的 ValidatedSnapshot（不可变）

    Raises:
        SnapshotMissingError: 必需路径不存在
        SnapshotInvalidError: 输入畸形/不安全/校验失败
    """

    if not isinstance(limits, SnapshotSafetyLimits):
        raise _invalid("INVALID_LIMIT", "limits must be SnapshotSafetyLimits")
    limits.validate()
    if not isinstance(manifest_sha256, str) or not _HEX_DIGEST.fullmatch(
        manifest_sha256
    ):
        raise _invalid("INVALID_DIGEST", "manifest_sha256 must be lowercase SHA-256")

    requested = _decode_snapshot_uri(snapshot_uri)
    roots = _canonical_allowed_roots(allowed_snapshot_roots)
    try:
        canonical_root = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise SnapshotMissingError("MISSING_SNAPSHOT", "snapshot root is missing") from exc
    except OSError as exc:
        raise _invalid("FILESYSTEM_ERROR", "snapshot path cannot be resolved") from exc
    # 快照必须位于某个允许根之下
    allowed_root = next(
        (root for root in roots if _is_relative_to(canonical_root, root)), None
    )
    if allowed_root is None:
        raise _invalid("OUTSIDE_ALLOWED_ROOT", "snapshot is outside allowed roots")
    # 请求路径本身（解码后）也必须位于允许根内，拒绝经软链的别名路径
    if not _is_relative_to(requested, allowed_root):
        raise _invalid("SYMLINK_REJECTED", "snapshot path uses an alias outside its root")
    _require_path(allowed_root, requested, "directory")

    snapshot_path = canonical_root / "snapshot.json"
    frames_path = canonical_root / "frames.csv"
    manifest_path = canonical_root / "files_sha256.csv"
    images_dir = canonical_root / "images"
    _require_path(canonical_root, images_dir, "directory")
    snapshot_bytes = _read_bounded(
        canonical_root, snapshot_path, limits.max_metadata_bytes
    )
    frames_bytes = _read_bounded(canonical_root, frames_path, limits.max_metadata_bytes)
    manifest_bytes = _read_bounded(
        canonical_root, manifest_path, limits.max_metadata_bytes
    )
    # 清单摘要必须与客户端提供的期望值一致（防中间人替换清单）
    verified_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if verified_digest != manifest_sha256:
        raise _invalid("MANIFEST_DIGEST_MISMATCH", "manifest digest does not match")

    def unique_object(pairs):
        """JSON 对象钩子：拒绝重复字段。"""
        result = {}
        for key, value in pairs:
            if key in result:
                raise _invalid("MALFORMED_SCHEMA", "snapshot.json has duplicate fields")
            result[key] = value
        return result

    try:
        metadata = json.loads(
            snapshot_bytes.decode("utf-8", "strict"), object_pairs_hook=unique_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid("MALFORMED_SCHEMA", "snapshot.json is malformed") from exc
    if not isinstance(metadata, dict):
        raise _invalid("MALFORMED_SCHEMA", "snapshot.json must be an object")
    meta = _parse_snapshot_metadata(metadata, limits)
    rows = _parse_frame_rows(frames_bytes, limits, meta.frame_count)
    # 帧表端点 stamp 必须与 snapshot.json 的 first/last_stamp 一致
    first_frame = rows.frame_stamps[0]
    last_frame = rows.frame_stamps[-1]
    if (
        (first_frame.sec, first_frame.nsec)
        != (meta.first_stamp.sec, meta.first_stamp.nsec)
        or (last_frame.sec, last_frame.nsec)
        != (meta.last_stamp.sec, meta.last_stamp.nsec)
    ):
        raise _invalid(
            "STAMP_MISMATCH",
            "snapshot.json stamps do not match frames.csv endpoints",
        )

    manifest_rows = _load_csv(
        manifest_bytes, ("path", "size", "sha256"), "files_sha256.csv"
    )
    if len(manifest_rows) > limits.max_manifest_entries:
        raise _invalid("MANIFEST_ENTRY_LIMIT", "manifest has too many entries")
    manifest = {}
    previous = None
    declared_total = 0
    for row in manifest_rows:
        relative = _safe_relative_path(row["path"], limits).as_posix()
        # 清单不得自引用（防止用自包含摘要欺骗）
        if relative == "files_sha256.csv":
            raise _invalid("MANIFEST_SELF_REFERENCE", "manifest must not list itself")
        if relative in manifest:
            raise _invalid("DUPLICATE_MANIFEST_PATH", "manifest paths must be unique")
        if previous is not None and relative <= previous:
            raise _invalid("MANIFEST_ORDER", "manifest paths must be sorted")
        previous = relative
        try:
            size = int(row["size"], 10)
        except ValueError as exc:
            raise _invalid("MALFORMED_MANIFEST", "manifest size is invalid") from exc
        if str(size) != row["size"] or size < 0 or size > limits.max_file_bytes:
            raise _invalid("FILE_SIZE_LIMIT", "manifest file size is not canonical/bounded")
        digest = row["sha256"]
        if not _HEX_DIGEST.fullmatch(digest):
            raise _invalid("MALFORMED_MANIFEST", "manifest digest is invalid")
        declared_total += size
        if declared_total > limits.max_total_bytes:
            raise _invalid("TOTAL_SIZE_LIMIT", "manifest total exceeds configured limit")
        manifest[relative] = (size, digest)

    # 清单覆盖必须恰好包含：固定元数据 + 图像 + 辅助数据文件
    expected_paths = {"snapshot.json", "frames.csv"}.union(rows.image_relatives)
    expected_paths.update(
        path for path in rows.auxiliary_paths if path is not None
    )
    expected_paths.update(path for path in rows.depth_paths if path is not None)
    if meta.has_tf:
        expected_paths.add("tf.json")
    if set(manifest) != expected_paths:
        raise _invalid("MANIFEST_COVERAGE", "manifest coverage is not exact")

    canonical_images = []
    for relative in sorted(expected_paths):
        path = canonical_root.joinpath(*PurePosixPath(relative).parts)
        info = _require_path(canonical_root, path, "file")
        expected_size, expected_digest = manifest[relative]
        if info.st_size != expected_size:
            raise _invalid("SIZE_MISMATCH", "file size does not match manifest")
        # 逐文件哈希校验，与清单声明逐一比对
        if _hash_file(canonical_root, path) != expected_digest:
            raise _invalid("HASH_MISMATCH", "file digest does not match manifest")
    for relative in rows.image_relatives:
        canonical_images.append(canonical_root.joinpath(*PurePosixPath(relative).parts))

    # ---- 解析并回读辅助 payload（读前/读后双重身份已由 manifest 保证）----
    camera_info_payloads = []
    odometry_payloads = []
    depth_metadata = []
    tf_entries = ()
    for index, aux_path in enumerate(rows.auxiliary_paths):
        if aux_path is None:
            camera_info_payloads.append(None)
            odometry_payloads.append(None)
            depth_metadata.append(None)
            continue
        ci, _ci_delta, odom, _odom_delta, depth_meta = _parse_aux_frame_json(
            canonical_root,
            canonical_root.joinpath(*PurePosixPath(aux_path).parts),
            aux_path,
            limits,
            index,
            rows,
        )
        camera_info_payloads.append(ci)
        odometry_payloads.append(odom)
        depth_metadata.append(depth_meta)
    if meta.camera_info_digest is not None:
        digest = hashlib.sha256()
        for ci in camera_info_payloads:
            if ci is None:
                continue
            digest.update(
                json.dumps(
                    ci,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
        if digest.hexdigest() != meta.camera_info_digest:
            raise _invalid(
                "CAMERA_INFO_DIGEST_MISMATCH",
                "camera_info_digest does not match saved payloads",
            )
    if meta.has_tf:
        tf_path = canonical_root / "tf.json"
        tf_entries = _parse_tf_file(canonical_root, tf_path, limits)

    return ValidatedSnapshot(
        canonical_root=canonical_root,
        scene_id=meta.scene_id,
        image_paths=tuple(canonical_images),
        image_basenames=tuple(path.name for path in canonical_images),
        frame_count=meta.frame_count,
        manifest_sha256=verified_digest,
        snapshot_version=meta.version,
        frame_stamps=rows.frame_stamps,
        pose_valid=rows.pose_valid,
        pose_delta_ms=rows.pose_delta_ms,
        odometry_gap_ms=rows.odometry_gap_ms,
        camera_name=meta.camera_name,
        image_topic=meta.image_topic,
        camera_role=meta.camera_role,
        camera_frame_id=meta.camera_frame_id,
        first_stamp=meta.first_stamp,
        last_stamp=meta.last_stamp,
        has_camera_info=meta.has_camera_info,
        has_odometry=meta.has_odometry,
        camera_info_topic=meta.camera_info_topic,
        odometry_topic=meta.odometry_topic,
        depth_topic=meta.depth_topic,
        tf_topic=meta.tf_topic,
        tf_static_topic=meta.tf_static_topic,
        save_sensor_depth=meta.save_sensor_depth,
        body_frame_id=meta.body_frame_id,
        has_tf=meta.has_tf,
        has_depth=meta.has_depth,
        camera_info_digest=meta.camera_info_digest,
        auxiliary_summary=meta.auxiliary_summary,
        auxiliary_paths=rows.auxiliary_paths,
        camera_info_delta_ms=rows.camera_info_delta_ms,
        odometry_delta_ms=rows.odometry_delta_ms,
        depth_paths=rows.depth_paths,
        depth_delta_ms=rows.depth_delta_ms,
        camera_info_payloads=tuple(camera_info_payloads),
        odometry_payloads=tuple(odometry_payloads),
        depth_metadata=tuple(depth_metadata),
        tf_entries=tf_entries,
    )


__all__ = [
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
]
