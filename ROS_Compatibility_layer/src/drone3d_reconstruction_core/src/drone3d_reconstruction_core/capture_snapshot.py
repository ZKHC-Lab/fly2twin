"""Writing one capture snapshot: the staging writer, the sealed result and its final name.

Split from ``capture``, which still re-exports the public names.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import secrets
from typing import TYPE_CHECKING, Any


from .preparation import INTERNAL_ERROR, INVALID_GOAL

from .result_transaction import (
    ResultTransaction,
    allocate_result_transaction,
)
from .snapshot_validator import (
    FRAME_COLUMNS_V4,
    SNAPSHOT_SCHEMA,
    SNAPSHOT_VERSION,
    FrameStamp,
)
from ._common import sha256_file as _sha256_file
from .capture_files import (
    _fsync_directory,
    _fsync_staging_files,
    _write_csv,
    _write_file_atomically,
    _write_manifest,
)
from .capture_messages import AuxiliaryFrameData
from .capture_types import _SCENE_ID, _TOKEN, CAMERA_UNAVAILABLE, CapturedFrame, CaptureError

if TYPE_CHECKING:
    from .capture import CaptureConfig, CaptureGoalParameters


@dataclass(frozen=True)
class SealedSnapshot:
    """封存完成的快照摘要（供 Action 结果字段填充）。"""

    scene_id: str
    snapshot_uri: str
    manifest_uri: str
    manifest_sha256: str
    frame_count: int
    first_stamp: FrameStamp
    last_stamp: FrameStamp
    camera_role: str
    camera_frame_id: str
    has_camera_info: bool
    has_odometry: bool
    has_depth: bool = False
    has_tf: bool = False
    camera_info_digest: str | None = None
    auxiliary_summary: dict | None = None


class SnapshotWriter:
    """在事务暂存目录内写帧并封存不可变快照。

    帧 PNG 直接写入暂存目录；seal() 时生成 frames.csv、snapshot.json 与
    files_sha256.csv 清单，全部满足 snapshot_limits 的字节/条目上限。

    v4 契约在每帧 PNG 之外落盘完整辅助数据：每帧一个 auxiliary/frame_*.json
    （完整 CameraInfo + Odometry payload 与关联时间差）、可选的
    depth/frame_*.bin 原始传感器深度、以及窗口级 tf.json。辅助数据遵循
    “尽量保存，但不阻塞 RGB”：超预算/超条目时逐项丢弃并计数，绝不让
    辅助源造成整窗失败。
    """

    def __init__(
        self,
        config: CaptureConfig,
        goal: CaptureGoalParameters,
        transaction: ResultTransaction,
    ) -> None:
        self._config = config
        self._goal = goal
        self._transaction = transaction
        self._frames: list[CapturedFrame] = []
        self._auxiliary: list[AuxiliaryFrameData] = []
        self._aux_file_count = 0
        self._tf_entries: list[dict] = []
        self._tf_bytes = 0
        self.tf_dropped = 0
        self.auxiliary_dropped = 0
        self.camera_info_samples = 0
        self.odometry_samples = 0
        self.depth_samples = 0
        self.auxiliary_errors = 0
        self.camera_info_matched_frames = 0
        self.odometry_matched_frames = 0
        self.depth_matched_frames = 0
        self._bytes_written = 0
        self._metadata_reserve = (
            3 * config.snapshot_limits.max_metadata_bytes + config.max_tf_bytes
        )

    @property
    def staging_path(self) -> Path:
        """事务暂存目录（worker/帧数据的唯一写入目标）。"""
        return self._transaction.staging_path

    @property
    def frame_count(self) -> int:
        """已写入的帧数。"""
        return len(self._frames)

    @property
    def has_camera_info(self) -> bool:
        """是否存在任何一帧保存了完整 CameraInfo payload。"""
        return self.camera_info_matched_frames > 0

    @property
    def has_odometry(self) -> bool:
        """是否存在任何一帧保存了完整 Odometry payload。"""
        return self.odometry_matched_frames > 0

    @property
    def has_depth(self) -> bool:
        return self.depth_matched_frames > 0

    @property
    def has_tf(self) -> bool:
        return bool(self._tf_entries)

    def _can_reserve(self, size: int, entries: int) -> bool:
        """判断辅助数据在总字节与 manifest 条目预算内是否仍可落盘。

        RGB 图像不属于本预算：图像超限是真正的采集失败。这里只保证辅助
        数据永远不会把整窗推到 total/entry 上限之外。
        """
        limits = self._config.snapshot_limits
        if self._bytes_written + size > limits.max_total_bytes - self._metadata_reserve:
            return False
        used_entries = (
            len(self._frames)
            + 3
            + (1 if self._tf_entries else 0)
            + self._aux_file_count
            + entries
        )
        return used_entries < limits.max_manifest_entries

    def _drop_auxiliary(self) -> None:
        self.auxiliary_dropped += 1

    def _canonical_json(self, value: Any) -> bytes:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    def add_transform(self, entry: dict) -> bool:
        """记录一条与相机/机体相关的 TF 变换。

        超过 max_tf_samples 或 max_tf_bytes 时拒绝并计数，不抛异常。
        """
        entry_bytes = len(self._canonical_json(entry))
        if entry_bytes > self._config.max_tf_bytes:
            self.tf_dropped += 1
            return False
        if len(self._tf_entries) >= self._config.max_tf_samples:
            self.tf_dropped += 1
            return False
        if self._tf_bytes + entry_bytes > self._config.max_tf_bytes:
            self.tf_dropped += 1
            return False
        self._tf_entries.append(entry)
        self._tf_bytes += entry_bytes
        return True

    def write_frame(
        self,
        stamp: FrameStamp,
        image_bytes: bytes,
        *,
        pose_valid: bool,
        pose_delta_ms: float | None,
        odometry_gap_ms: float | None,
        auxiliary: AuxiliaryFrameData | None = None,
    ) -> CapturedFrame:
        """原子写入一帧 PNG、辅助 JSON/深度并登记帧元数据。"""
        index = len(self._frames)
        relative = f"images/frame_{index:06d}.png"
        path = self.staging_path.joinpath(*Path(relative).parts)
        _write_file_atomically(
            path,
            image_bytes,
            self._config.snapshot_limits.max_file_bytes,
            sync=False,
        )
        self._bytes_written += len(image_bytes)
        frame = CapturedFrame(
            index=index,
            stamp=stamp,
            image_relative_path=relative,
            pose_valid=pose_valid,
            pose_delta_ms=pose_delta_ms,
            odometry_gap_ms=odometry_gap_ms,
        )
        self._frames.append(frame)
        auxiliary = auxiliary or AuxiliaryFrameData()
        self._store_auxiliary(index, auxiliary)
        self._auxiliary.append(auxiliary)
        return frame

    def _store_auxiliary(self, index: int, auxiliary: AuxiliaryFrameData) -> None:
        """按预算写入一帧的辅助 JSON 与原始深度文件（超预算逐项丢弃）。

        写入顺序保证一致性：先写 auxiliary JSON（其中声明深度文件），再写
        depth bin；任何一步失败都会把这一帧的辅助数据整体降级为不落盘，
        staging 中绝不残留未被 frames.csv 引用的孤儿文件。
        """
        depth_relative = f"depth/frame_{index:06d}.bin"
        depth_fits = False
        if auxiliary.depth is not None:
            size = len(auxiliary.depth.data)
            depth_fits = (
                size <= self._config.snapshot_limits.max_file_bytes
                and self._can_reserve(size, 1)
            )

        def build_payload(with_depth: bool) -> dict:
            return {
                "frame_index": index,
                "camera_info": (
                    auxiliary.camera_info.payload
                    if auxiliary.camera_info is not None
                    else None
                ),
                "camera_info_delta_ms": auxiliary.camera_info_delta_ms,
                "odometry": (
                    auxiliary.odometry.payload
                    if auxiliary.odometry is not None
                    else None
                ),
                "odometry_delta_ms": auxiliary.odometry_delta_ms,
                "depth": (
                    {
                        "stamp": {
                            "sec": auxiliary.depth.stamp.sec,
                            "nsec": auxiliary.depth.stamp.nsec,
                        },
                        "delta_ms": auxiliary.depth_delta_ms,
                        "frame_id": auxiliary.depth.frame_id,
                        "encoding": auxiliary.depth.encoding,
                        "width": auxiliary.depth.width,
                        "height": auxiliary.depth.height,
                        "step": auxiliary.depth.step,
                        "is_bigendian": auxiliary.depth.is_bigendian,
                        "data_file": depth_relative,
                    }
                    if with_depth and auxiliary.depth is not None
                    else None
                ),
            }

        if (
            auxiliary.camera_info is None
            and auxiliary.odometry is None
            and not depth_fits
        ):
            return

        relative = f"auxiliary/frame_{index:06d}.json"
        path = self.staging_path.joinpath(*Path(relative).parts)
        json_bytes = self._canonical_json(build_payload(depth_fits))
        if not (
            len(json_bytes) <= self._config.snapshot_limits.max_file_bytes
            and self._can_reserve(len(json_bytes), 1)
        ):
            self._drop_auxiliary()
            return
        try:
            _write_file_atomically(
                path,
                json_bytes,
                self._config.snapshot_limits.max_file_bytes,
                sync=False,
            )
            self._bytes_written += len(json_bytes)
            self._aux_file_count += 1
        except BaseException:
            self._drop_auxiliary()
            return

        # auxiliary JSON 已成功落盘：CameraInfo/Odometry 匹配帧计数在此定格。
        ci_matched = auxiliary.camera_info is not None
        odom_matched = auxiliary.odometry is not None
        if ci_matched:
            self.camera_info_matched_frames += 1
        if odom_matched:
            self.odometry_matched_frames += 1

        if depth_fits:
            depth_path_obj = self.staging_path.joinpath(*Path(depth_relative).parts)
            try:
                _write_file_atomically(
                    depth_path_obj,
                    auxiliary.depth.data,
                    self._config.snapshot_limits.max_file_bytes,
                    sync=False,
                )
                self._bytes_written += len(auxiliary.depth.data)
                self._aux_file_count += 1
                self.depth_matched_frames += 1
            except BaseException:
                # 深度写失败：把已写 JSON 改写为无深度版本，保持两者一致。
                self._drop_auxiliary()
                try:
                    json_without_depth = self._canonical_json(build_payload(False))
                    _write_file_atomically(
                        path,
                        json_without_depth,
                        self._config.snapshot_limits.max_file_bytes,
                        sync=False,
                    )
                    self._bytes_written += len(json_without_depth) - len(json_bytes)
                except BaseException:
                    # 连改写都失败：本帧辅助整体放弃，删除已写文件。
                    if ci_matched:
                        self.camera_info_matched_frames -= 1
                    if odom_matched:
                        self.odometry_matched_frames -= 1
                    self._remove_aux_files(path, depth_path_obj, len(json_bytes))

    def _remove_aux_files(self, json_path: Path, depth_path: Path, json_bytes: int) -> None:
        """辅助写盘完全失败时清理本帧已写文件（尽力而为，不抛异常）。"""
        for path in (json_path, depth_path):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass
        self._bytes_written = max(0, self._bytes_written - json_bytes)
        self._aux_file_count = max(0, self._aux_file_count - 1)

    def _aux_path_for(self, index: int) -> str | None:
        """返回已成功落盘的 auxiliary JSON 相对路径（不存在返回 None）。"""
        relative = f"auxiliary/frame_{index:06d}.json"
        path = self.staging_path.joinpath(*Path(relative).parts)
        if path.is_file():
            return relative
        return None

    def _camera_info_digest(self) -> str | None:
        """由全部已保存 CameraInfo payload 计算确定性 SHA-256。

        只有保存成功（auxiliary JSON 落盘）的 payload 参与摘要，顺序为帧序。
        """
        if self.camera_info_matched_frames == 0:
            return None
        digest = hashlib.sha256()
        counted = 0
        for index, auxiliary in enumerate(self._auxiliary):
            if auxiliary.camera_info is None or self._aux_path_for(index) is None:
                continue
            payload = json.dumps(
                auxiliary.camera_info.payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(payload)
            digest.update(b"\n")
            counted += 1
        if counted != self.camera_info_matched_frames:
            return None
        return digest.hexdigest()

    def seal(self) -> SealedSnapshot:
        """封存快照：写 frames.csv / snapshot.json / tf.json / 清单并返回摘要。

        Raises:
            CaptureError: 帧数不足本次 Goal 的有效下限时抛 CAMERA_UNAVAILABLE
        """
        config = self._config
        minimum_frames = min(config.min_captured_frames, self._goal.max_frames)
        if len(self._frames) < minimum_frames:
            raise CaptureError(
                CAMERA_UNAVAILABLE,
                "capture ended before effective min_captured_frames was reached "
                f"({len(self._frames)} < {minimum_frames})",
            )
        first = self._frames[0].stamp
        last = self._frames[-1].stamp
        camera_info_digest = self._camera_info_digest()
        auxiliary_summary = {
            "camera_info_samples": self.camera_info_samples,
            "odometry_samples": self.odometry_samples,
            "depth_samples": self.depth_samples,
            "tf_samples": len(self._tf_entries),
            "camera_info_matched_frames": self.camera_info_matched_frames,
            "odometry_matched_frames": self.odometry_matched_frames,
            "depth_matched_frames": self.depth_matched_frames,
            "tf_dropped": self.tf_dropped,
            "auxiliary_errors": self.auxiliary_errors,
        }
        snapshot_metadata = {
            "schema": SNAPSHOT_SCHEMA,
            "version": SNAPSHOT_VERSION,
            "status": "PASS",
            "scene_id": self._goal.scene_id,
            "frame_count": len(self._frames),
            "camera_name": config.camera_name,
            "image_topic": config.image_topic,
            "camera_role": config.camera_role,
            "camera_frame_id": config.camera_frame_id,
            "first_stamp_sec": first.sec,
            "first_stamp_nsec": first.nsec,
            "last_stamp_sec": last.sec,
            "last_stamp_nsec": last.nsec,
            "has_camera_info": self.has_camera_info,
            "has_odometry": self.has_odometry,
            # v4 辅助数据契约
            "camera_info_topic": config.camera_info_topic,
            "odometry_topic": config.odometry_topic,
            "depth_topic": config.depth_topic,
            "tf_topic": config.tf_topic,
            "tf_static_topic": config.tf_static_topic,
            "save_sensor_depth": config.save_sensor_depth,
            "body_frame_id": config.body_frame_id,
            "has_tf": self.has_tf,
            "has_depth": self.has_depth,
            "camera_info_digest": camera_info_digest,
            "auxiliary_summary": auxiliary_summary,
        }
        frame_rows = []
        for index, frame in enumerate(self._frames):
            auxiliary = (
                self._auxiliary[index] if index < len(self._auxiliary) else None
            )
            aux_path = self._aux_path_for(index)
            depth_path = self._depth_path_for(index)
            row = {
                "frame_index": str(frame.index),
                "image_path": frame.image_relative_path,
                "stamp_sec": str(frame.stamp.sec),
                "stamp_nsec": str(frame.stamp.nsec),
                "pose_valid": "true" if frame.pose_valid else "false",
                "pose_delta_ms": _format_optional_ms(frame.pose_delta_ms),
                "odometry_gap_ms": _format_optional_ms(frame.odometry_gap_ms),
            }
            if aux_path is not None:
                row.update(
                    {
                        "auxiliary_path": aux_path,
                        "camera_info_delta_ms": _format_optional_ms(
                            auxiliary.camera_info_delta_ms
                        ),
                        "odometry_delta_ms": _format_optional_ms(
                            auxiliary.odometry_delta_ms
                        ),
                        "depth_path": depth_path or "",
                        "depth_delta_ms": (
                            _format_optional_ms(auxiliary.depth_delta_ms)
                            if depth_path is not None
                            else "-1"
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "auxiliary_path": "",
                        "camera_info_delta_ms": "-1",
                        "odometry_delta_ms": "-1",
                        "depth_path": "",
                        "depth_delta_ms": "-1",
                    }
                )
            frame_rows.append(row)
        _write_csv(
            self.staging_path / "frames.csv",
            FRAME_COLUMNS_V4,
            frame_rows,
            config.snapshot_limits.max_metadata_bytes,
        )
        if self._tf_entries:
            tf_bytes = self._canonical_json(self._tf_entries)
            try:
                _write_file_atomically(
                    self.staging_path / "tf.json",
                    tf_bytes,
                    config.snapshot_limits.max_file_bytes,
                )
                self._bytes_written += len(tf_bytes)
            except BaseException:
                # TF 是辅助数据：写不下只能丢弃，不能让整窗失败。
                self.tf_dropped += len(self._tf_entries)
                self._tf_entries = []
                auxiliary_summary["tf_samples"] = 0
                auxiliary_summary["tf_dropped"] = self.tf_dropped
                snapshot_metadata["has_tf"] = False
        snapshot_bytes = (
            json.dumps(
                snapshot_metadata,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        _write_file_atomically(
            self.staging_path / "snapshot.json",
            snapshot_bytes,
            config.snapshot_limits.max_metadata_bytes,
        )
        # 帧/辅助文件写盘时 sync=False（避免逐帧 fsync 阻塞采集）；这里在
        # 封窗前一次性补 fsync，保证清单引用到的每个文件内容也已持久化。
        _fsync_staging_files(self.staging_path)
        _write_manifest(self.staging_path, config.snapshot_limits)
        _fsync_directory(self.staging_path)
        manifest_path = self.staging_path / "files_sha256.csv"
        manifest_sha256 = _sha256_file(manifest_path)
        snapshot_uri = self._transaction.final_path.as_uri()
        manifest_uri = (
            self._transaction.final_path / "files_sha256.csv"
        ).as_uri()
        return SealedSnapshot(
            scene_id=self._goal.scene_id,
            snapshot_uri=snapshot_uri,
            manifest_uri=manifest_uri,
            manifest_sha256=manifest_sha256,
            frame_count=len(self._frames),
            first_stamp=first,
            last_stamp=last,
            camera_role=config.camera_role,
            camera_frame_id=config.camera_frame_id,
            has_camera_info=self.has_camera_info,
            has_odometry=self.has_odometry,
            has_depth=self.has_depth,
            has_tf=bool(self._tf_entries),
            camera_info_digest=camera_info_digest,
            auxiliary_summary=auxiliary_summary,
        )

    def _depth_path_for(self, index: int) -> str | None:
        """返回已成功落盘的深度文件相对路径（不存在返回 None）。"""
        auxiliary = self._auxiliary[index] if index < len(self._auxiliary) else None
        if auxiliary is None or auxiliary.depth is None:
            return None
        relative = f"depth/frame_{index:06d}.bin"
        path = self.staging_path.joinpath(*Path(relative).parts)
        if path.is_file():
            return relative
        return None


def snapshot_final_name(
    scene_id: str, *, token_source=None
) -> str:
    """生成快照最终目录名：{scene_id}-{16 位十六进制 token}。"""
    if not _SCENE_ID.fullmatch(scene_id):
        raise CaptureError(INVALID_GOAL, "scene_id is unsafe")
    token = (token_source or secrets_token)()
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        raise CaptureError(INTERNAL_ERROR, "snapshot name token is unsafe")
    return f"{scene_id}-{token}"


def allocate_snapshot_transaction(
    config: CaptureConfig,
    scene_id: str,
    *,
    token_source=None,
) -> ResultTransaction:
    """分配监督进程拥有的快照暂存/发布事务。

    Args:
        config: 采集配置（提供 snapshot_root）
        scene_id: 场景标识符（用于生成最终目录名）
        token_source: 随机 token 生成器（测试注入点）

    Returns:
        已分配的结果事务（暂存目录位于 snapshot_root 下）
    """
    return allocate_result_transaction(
        config.snapshot_root,
        snapshot_final_name(scene_id, token_source=token_source),
    )


def _format_optional_ms(value: float | None) -> str:
    """把可选毫秒值格式化为 CSV 字段（None → -1）。"""
    return "-1" if value is None else repr(float(value))


_token_sequence = [0]


def secrets_token() -> str:
    """生成 16 位十六进制随机 token（快照目录名后缀）。"""
    return secrets.token_hex(8)
