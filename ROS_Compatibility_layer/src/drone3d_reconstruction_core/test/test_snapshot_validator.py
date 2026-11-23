#!/usr/bin/env python3

import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from drone3d_reconstruction_core.snapshot_validator import (
    SNAPSHOT_SCHEMA,
    SnapshotInvalidError,
    SnapshotMissingError,
    SnapshotSafetyLimits,
    ValidatedSnapshot,
    validate_snapshot,
)


LIMITS = SnapshotSafetyLimits(
    max_frame_count=8,
    max_manifest_entries=16,
    max_metadata_bytes=8192,
    max_file_bytes=4096,
    max_total_bytes=16384,
    max_scene_id_bytes=128,
    max_relative_path_bytes=1024,
)


FRAME_COLUMNS = (
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


def _row(index, image_path, stamp_sec=None, **changes):
    """One v4 frames.csv row without a pose or auxiliary file."""
    row = {
        "frame_index": str(index),
        "image_path": image_path,
        "stamp_sec": str(10 + index if stamp_sec is None else stamp_sec),
        "stamp_nsec": "0",
        "pose_valid": "false",
        "pose_delta_ms": "-1",
        "odometry_gap_ms": "-1",
        "auxiliary_path": "",
        "camera_info_delta_ms": "-1",
        "odometry_delta_ms": "-1",
        "depth_path": "",
        "depth_delta_ms": "-1",
    }
    row.update(changes)
    return row


def _minimal_metadata():
    """v4 snapshot.json for two frames without any auxiliary data."""
    return {
        "schema": SNAPSHOT_SCHEMA,
        "version": 4,
        "status": "PASS",
        "scene_id": "scene-01",
        "frame_count": 2,
        "camera_name": "camera4",
        "image_topic": "/camera4/image_raw",
        "first_stamp_sec": 10,
        "first_stamp_nsec": 0,
        "last_stamp_sec": 11,
        "last_stamp_nsec": 0,
        "has_camera_info": False,
        "has_odometry": False,
        "camera_role": "front",
        "camera_frame_id": "front_camera_optical_frame",
        "camera_info_topic": None,
        "odometry_topic": None,
        "depth_topic": None,
        "tf_topic": None,
        "tf_static_topic": None,
        "save_sensor_depth": False,
        "body_frame_id": None,
        "has_tf": False,
        "has_depth": False,
        "camera_info_digest": None,
        "auxiliary_summary": {
            "camera_info_samples": 0,
            "odometry_samples": 0,
            "depth_samples": 0,
            "tf_samples": 0,
            "camera_info_matched_frames": 0,
            "odometry_matched_frames": 0,
            "depth_matched_frames": 0,
            "tf_dropped": 0,
            "auxiliary_errors": 0,
        },
    }


def _csv_bytes(columns, rows):
    import io

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


class SnapshotFixture:
    def __init__(self, base):
        self.allowed = Path(base) / "trusted"
        self.root = self.allowed / "snapshot"
        (self.root / "images").mkdir(parents=True)
        self.images = {
            "images/frame_000.jpg": b"first-image",
            "images/frame_001.jpg": b"second-image",
        }
        self.metadata = _minimal_metadata()
        self.frame_columns = FRAME_COLUMNS
        self.frame_rows = [
            _row(0, "images/frame_000.jpg"),
            _row(1, "images/frame_001.jpg"),
        ]
        self.extra_manifest = {}
        self.write()

    def write(self, sort_manifest=True):
        for relative, data in self.images.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        snapshot = json.dumps(
            self.metadata, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        frames = _csv_bytes(self.frame_columns, self.frame_rows)
        (self.root / "snapshot.json").write_bytes(snapshot)
        (self.root / "frames.csv").write_bytes(frames)
        contents = {"snapshot.json": snapshot, "frames.csv": frames}
        contents.update(self.images)
        contents.update(self.extra_manifest)
        items = sorted(contents.items()) if sort_manifest else list(contents.items())
        rows = [
            {
                "path": path,
                "size": str(len(data)),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for path, data in items
        ]
        manifest = _csv_bytes(("path", "size", "sha256"), rows)
        (self.root / "files_sha256.csv").write_bytes(manifest)
        self.digest = hashlib.sha256(manifest).hexdigest()

    def validate(self, uri=None, digest="fixture", limits=LIMITS, roots=None):
        return validate_snapshot(
            str(self.root) if uri is None else uri,
            self.digest if digest == "fixture" else digest,
            [self.allowed] if roots is None else roots,
            limits,
        )


class SnapshotValidatorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = SnapshotFixture(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def assertInvalid(self, code, operation=None):
        if operation is None:
            operation = self.fixture.validate
        with self.assertRaises(SnapshotInvalidError) as caught:
            operation()
        self.assertEqual(code, caught.exception.code)

    def test_valid_absolute_path_returns_frozen_ordered_value(self):
        result = self.fixture.validate()
        self.assertIsInstance(result, ValidatedSnapshot)
        self.assertEqual(self.fixture.root.resolve(), result.canonical_root)
        self.assertEqual("scene-01", result.scene_id)
        self.assertEqual(2, result.frame_count)
        self.assertEqual(
            ("frame_000.jpg", "frame_001.jpg"), result.image_basenames
        )
        self.assertEqual(
            tuple(
                (self.fixture.root / row["image_path"]).resolve()
                for row in self.fixture.frame_rows
            ),
            result.image_paths,
        )
        self.assertEqual(self.fixture.digest, result.manifest_sha256)
        with self.assertRaises(Exception):
            result.frame_count = 3

    def test_valid_file_uri_with_percent_encoding(self):
        renamed = self.fixture.allowed / "snapshot space"
        self.fixture.root.rename(renamed)
        self.fixture.root = renamed
        result = self.fixture.validate(uri=self.fixture.root.as_uri())
        self.assertEqual(renamed.resolve(), result.canonical_root)

    def test_rejects_uri_forms_and_bad_encoding(self):
        cases = (
            "relative/snapshot",
            "https://example/snapshot",
            "file://localhost/tmp/snapshot",
            "file:///tmp/snapshot?x=1",
            "file:///tmp/snapshot#x",
            "file:///tmp/%ZZ",
            "file:///tmp/%00snapshot",
            "file://[::1/snapshot",
            "file://[v.invalid]/snapshot",
            "file://[broken/path",
        )
        for uri in cases:
            self.assertInvalid("INVALID_URI", lambda uri=uri: self.fixture.validate(uri=uri))

    def test_rejects_traversal_before_canonicalization(self):
        uri = str(self.fixture.allowed / "other" / ".." / "snapshot")
        self.assertInvalid("PATH_TRAVERSAL", lambda: self.fixture.validate(uri=uri))

    def test_rejects_snapshot_outside_allowed_root(self):
        outside = Path(self.temp.name) / "outside"
        self.fixture.root.rename(outside)
        self.fixture.root = outside
        self.assertInvalid("OUTSIDE_ALLOWED_ROOT")

    def test_missing_snapshot_and_required_file_are_typed_separately(self):
        with self.assertRaises(SnapshotMissingError) as caught:
            self.fixture.validate(uri=str(self.fixture.allowed / "missing"))
        self.assertEqual("MISSING_SNAPSHOT", caught.exception.code)
        (self.fixture.root / "frames.csv").unlink()
        with self.assertRaises(SnapshotMissingError) as caught:
            self.fixture.validate()
        self.assertEqual("MISSING_INPUT", caught.exception.code)

    def test_rejects_snapshot_file_and_missing_images_directory(self):
        image_dir = self.fixture.root / "images"
        for path in image_dir.iterdir():
            path.unlink()
        image_dir.rmdir()
        with self.assertRaises(SnapshotMissingError) as caught:
            self.fixture.validate()
        self.assertEqual("MISSING_INPUT", caught.exception.code)

    def test_rejects_symlink_at_snapshot_or_member(self):
        alias = self.fixture.allowed / "alias"
        alias.symlink_to(self.fixture.root, target_is_directory=True)
        self.assertInvalid("SYMLINK_REJECTED", lambda: self.fixture.validate(uri=str(alias)))

        outside_alias = Path(self.temp.name) / "trusted-alias"
        outside_alias.symlink_to(self.fixture.allowed, target_is_directory=True)
        self.assertInvalid(
            "SYMLINK_REJECTED",
            lambda: self.fixture.validate(uri=str(outside_alias / "snapshot")),
        )

        target = self.fixture.root / "real.jpg"
        target.write_bytes(self.fixture.images["images/frame_000.jpg"])
        image = self.fixture.root / "images/frame_000.jpg"
        image.unlink()
        image.symlink_to(target)
        self.assertInvalid("SYMLINK_REJECTED")

    def test_rejects_hard_link_and_filesystem_path_errors(self):
        external = Path(self.temp.name) / "external.jpg"
        external.write_bytes(self.fixture.images["images/frame_000.jpg"])
        image = self.fixture.root / "images/frame_000.jpg"
        image.unlink()
        os.link(external, image)
        self.assertInvalid("HARDLINK_REJECTED")

        too_long = "/tmp/" + "x" * 5000
        self.assertInvalid(
            "FILESYSTEM_ERROR",
            lambda: self.fixture.validate(uri=too_long),
        )

    def test_rejects_manifest_digest_and_content_mutations(self):
        self.assertInvalid(
            "MANIFEST_DIGEST_MISMATCH",
            lambda: self.fixture.validate(digest="0" * 64),
        )
        image = self.fixture.root / "images/frame_000.jpg"
        image.write_bytes(b"first-imagf")
        self.assertInvalid("HASH_MISMATCH")
        self.fixture.write()
        image.write_bytes(b"different-size")
        self.assertInvalid("SIZE_MISMATCH")

    def test_rejects_schema_identity_status_scene_and_count(self):
        cases = (
            ("schema", "other", "UNSUPPORTED_SCHEMA"),
            ("version", 1, "UNSUPPORTED_SCHEMA"),
            ("version", 2, "UNSUPPORTED_SCHEMA"),
            ("version", 3, "UNSUPPORTED_SCHEMA"),
            ("version", 5, "UNSUPPORTED_SCHEMA"),
            ("version", True, "UNSUPPORTED_SCHEMA"),
            ("version", 4.5, "UNSUPPORTED_SCHEMA"),
            ("status", "FAIL", "INVALID_STATUS"),
            ("scene_id", "../bad", "INVALID_SCENE_ID"),
            ("scene_id", "trailing.", "INVALID_SCENE_ID"),
            ("frame_count", 0, "FRAME_COUNT_LIMIT"),
            ("frame_count", True, "FRAME_COUNT_LIMIT"),
            ("frame_count", 9, "FRAME_COUNT_LIMIT"),
        )
        for field, value, code in cases:
            fixture = SnapshotFixture(self.temp.name + "-" + field + str(value))
            fixture.metadata[field] = value
            fixture.write()
            self.assertInvalid(code, fixture.validate)

    def test_rejects_malformed_json_and_extra_schema_fields(self):
        path = self.fixture.root / "snapshot.json"
        path.write_bytes(b"{")
        self.assertInvalid("MALFORMED_SCHEMA")
        self.fixture.metadata["extra"] = 1
        self.fixture.write()
        self.assertInvalid("MALFORMED_SCHEMA")

    def test_rejects_frames_columns_order_duplicates_and_unsafe_paths(self):
        bad_rows = (
            ([_row(1, "images/frame_000.jpg", stamp_sec=10),
              _row(0, "images/frame_001.jpg", stamp_sec=11)], "FRAME_ORDER"),
            ([_row(0, "images/frame_000.jpg"),
              _row(1, "images/frame_000.jpg")], "DUPLICATE_IMAGE"),
            ([_row(0, "../frame.jpg"),
              _row(1, "images/frame_001.jpg")], "UNSAFE_PATH"),
            ([_row(0, "other/frame.jpg"),
              _row(1, "images/frame_001.jpg")], "UNSAFE_IMAGE_PATH"),
        )
        for index, (rows, code) in enumerate(bad_rows):
            fixture = SnapshotFixture(self.temp.name + "-frames" + str(index))
            fixture.frame_rows = rows
            fixture.write()
            self.assertInvalid(code, fixture.validate)

        (self.fixture.root / "frames.csv").write_bytes(b"image_path,frame_index\n")
        self.assertInvalid("MALFORMED_CSV")
        self.fixture.write()
        self.fixture.metadata["frame_count"] = 1
        self.fixture.write()
        self.assertInvalid("FRAME_COUNT_MISMATCH")

    def test_rejects_manifest_columns_duplicate_self_unsorted_and_coverage(self):
        path = self.fixture.root / "files_sha256.csv"
        path.write_bytes(b"sha256,size,path\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertInvalid("MALFORMED_CSV", lambda: self.fixture.validate(digest=digest))

        data = b"x"
        duplicate_rows = [
            {"path": "frames.csv", "size": "1", "sha256": hashlib.sha256(data).hexdigest()},
            {"path": "frames.csv", "size": "1", "sha256": hashlib.sha256(data).hexdigest()},
        ]
        manifest = _csv_bytes(("path", "size", "sha256"), duplicate_rows)
        path.write_bytes(manifest)
        self.assertInvalid(
            "DUPLICATE_MANIFEST_PATH",
            lambda: self.fixture.validate(digest=hashlib.sha256(manifest).hexdigest()),
        )

        self.fixture.write()
        with (self.fixture.root / "files_sha256.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        rows.reverse()
        manifest = _csv_bytes(("path", "size", "sha256"), rows)
        path.write_bytes(manifest)
        self.assertInvalid(
            "MANIFEST_ORDER",
            lambda: self.fixture.validate(digest=hashlib.sha256(manifest).hexdigest()),
        )

        self.fixture.extra_manifest["unlisted.bin"] = b"x"
        self.fixture.write()
        self.assertInvalid("MANIFEST_COVERAGE")

    def test_rejects_manifest_self_reference_and_unsafe_entry(self):
        for entry, code in (("files_sha256.csv", "MANIFEST_SELF_REFERENCE"),
                            ("../escape", "UNSAFE_PATH")):
            row = [{"path": entry, "size": "0", "sha256": hashlib.sha256(b"").hexdigest()}]
            manifest = _csv_bytes(("path", "size", "sha256"), row)
            path = self.fixture.root / "files_sha256.csv"
            path.write_bytes(manifest)
            self.assertInvalid(
                code,
                lambda manifest=manifest: self.fixture.validate(
                    digest=hashlib.sha256(manifest).hexdigest()
                ),
            )

    def test_enforces_metadata_manifest_file_total_and_path_limits(self):
        cases = (
            (SnapshotSafetyLimits(8, 16, 10, 4096, 16384, 128, 1024), "METADATA_LIMIT"),
            (SnapshotSafetyLimits(8, 2, 8192, 4096, 16384, 128, 1024), "MANIFEST_ENTRY_LIMIT"),
            (SnapshotSafetyLimits(8, 16, 8192, 5, 16384, 128, 1024), "FILE_SIZE_LIMIT"),
            (SnapshotSafetyLimits(8, 16, 8192, 4096, 10, 128, 1024), "TOTAL_SIZE_LIMIT"),
            (SnapshotSafetyLimits(8, 16, 8192, 4096, 16384, 3, 1024), "INVALID_SCENE_ID"),
            (SnapshotSafetyLimits(8, 16, 8192, 4096, 16384, 128, 5), "PATH_LIMIT"),
        )
        for limits, code in cases:
            self.assertInvalid(code, lambda limits=limits: self.fixture.validate(limits=limits))

    def test_rejects_invalid_limit_and_digest_types(self):
        bad = SnapshotSafetyLimits(0, 1, 1, 1, 1, 1, 1)
        self.assertInvalid("INVALID_LIMIT", lambda: self.fixture.validate(limits=bad))
        for digest in ("A" * 64, "abc", None):
            self.assertInvalid(
                "INVALID_DIGEST",
                lambda digest=digest: self.fixture.validate(digest=digest),
            )


    def _v2_rows(self):
        """Stamp and pose columns of the two v4 fixture frames."""
        return [
            {
                "frame_index": "0",
                "image_path": "images/frame_000.jpg",
                "stamp_sec": "10",
                "stamp_nsec": "0",
                "pose_valid": "true",
                "pose_delta_ms": "12.5",
                "odometry_gap_ms": "0.0",
            },
            {
                "frame_index": "1",
                "image_path": "images/frame_001.jpg",
                "stamp_sec": "12",
                "stamp_nsec": "500000000",
                "pose_valid": "false",
                "pose_delta_ms": "-1",
                "odometry_gap_ms": "240.0",
            },
        ]

    # ---- v4 完整辅助数据契约 ----

    _V4_COLUMNS = FRAME_COLUMNS

    def _v4_camera_info(self):
        return {
            "header": {
                "seq": 1,
                "stamp": {"sec": 10, "nsec": 0},
                "frame_id": "front_camera_optical_frame",
            },
            "height": 3,
            "width": 4,
            "distortion_model": "plumb_bob",
            "D": [0.1, 0.2],
            "K": [1.0] * 9,
            "R": [0.0] * 9,
            "P": [2.0] * 12,
            "binning_x": 1,
            "binning_y": 1,
            "roi": {
                "x_offset": 0,
                "y_offset": 0,
                "width": 4,
                "height": 3,
                "do_rectify": False,
            },
        }

    def _v4_odometry(self):
        return {
            "header": {
                "seq": 1,
                "stamp": {"sec": 10, "nsec": 0},
                "frame_id": "odom",
            },
            "child_frame_id": "body",
            "pose": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "covariance": [0.0] * 36,
            },
            "twist": {
                "linear": {"x": 1.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.5},
                "covariance": [0.0] * 36,
            },
        }

    def _v4_aux_frame(self):
        ci = self._v4_camera_info()
        digest = hashlib.sha256(
            json.dumps(
                ci, ensure_ascii=False, allow_nan=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        ).hexdigest()
        return {
            "frame_index": 0,
            "camera_info": ci,
            "camera_info_delta_ms": 0.0,
            "odometry": self._v4_odometry(),
            "odometry_delta_ms": 1.0,
            "depth": {
                "stamp": {"sec": 10, "nsec": 0},
                "delta_ms": 0.0,
                "frame_id": "depth_optical_frame",
                "encoding": "16UC1",
                "width": 2,
                "height": 1,
                "step": 2,
                "is_bigendian": 0,
                "data_file": "depth/frame_000000.bin",
            },
        }, digest

    def _v4_metadata(self, aux_frame, camera_info_digest):
        return {
            "schema": SNAPSHOT_SCHEMA,
            "version": 4,
            "status": "PASS",
            "scene_id": "scene-v4",
            "frame_count": 2,
            "camera_name": "camera4",
            "image_topic": "/camera4/image_raw",
            "first_stamp_sec": 10,
            "first_stamp_nsec": 0,
            "last_stamp_sec": 12,
            "last_stamp_nsec": 500000000,
            "has_camera_info": True,
            "has_odometry": True,
            "camera_role": "front",
            "camera_frame_id": "front_camera_optical_frame",
            "camera_info_topic": "/camera_info",
            "odometry_topic": "/Odometry",
            "depth_topic": "/depth",
            "tf_topic": "/tf",
            "tf_static_topic": "/tf_static",
            "save_sensor_depth": True,
            "body_frame_id": "body",
            "has_tf": True,
            "has_depth": True,
            "camera_info_digest": camera_info_digest,
            "auxiliary_summary": {
                "camera_info_samples": 1,
                "odometry_samples": 1,
                "depth_samples": 1,
                "tf_samples": 1,
                "camera_info_matched_frames": 1,
                "odometry_matched_frames": 1,
                "depth_matched_frames": 1,
                "tf_dropped": 0,
                "auxiliary_errors": 0,
            },
        }

    def _v4_rows(self):
        rows = self._v2_rows()
        rows[0].update(
            {
                "auxiliary_path": "auxiliary/frame_000000.json",
                "camera_info_delta_ms": "0.0",
                "odometry_delta_ms": "1.0",
                "depth_path": "depth/frame_000000.bin",
                "depth_delta_ms": "0.0",
            }
        )
        rows[1].update(
            {
                "auxiliary_path": "",
                "camera_info_delta_ms": "-1",
                "odometry_delta_ms": "-1",
                "depth_path": "",
                "depth_delta_ms": "-1",
            }
        )
        return rows

    def _write_v4_fixture(self, mutate=None):
        aux_frame, digest = self._v4_aux_frame()
        if mutate is not None:
            aux_frame = mutate(aux_frame)
        extra = {
            "auxiliary/frame_000000.json": json.dumps(
                aux_frame, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            "depth/frame_000000.bin": b"\x01\x02",
            "tf.json": json.dumps(
                [
                    {
                        "stamp": {"sec": 10, "nsec": 0},
                        "parent_frame_id": "odom",
                        "child_frame_id": "body",
                        "translation": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        "static": False,
                    }
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        }
        self._write_extra_files(extra)
        self.fixture.extra_manifest = extra
        self.fixture.metadata = self._v4_metadata(aux_frame, digest)
        self.fixture.frame_columns = self._V4_COLUMNS
        self.fixture.frame_rows = self._v4_rows()
        self.fixture.write()

    def _write_extra_files(self, extra):
        for relative, data in extra.items():
            path = self.fixture.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def test_v4_snapshot_returns_full_auxiliary_readback(self):
        self._write_v4_fixture()
        result = self.fixture.validate()
        self.assertEqual(result.snapshot_version, 4)
        self.assertTrue(result.has_camera_info)
        self.assertTrue(result.has_odometry)
        self.assertTrue(result.has_depth)
        self.assertTrue(result.has_tf)
        self.assertEqual(result.auxiliary_paths[0], "auxiliary/frame_000000.json")
        self.assertIsNone(result.auxiliary_paths[1])
        self.assertEqual(result.depth_paths[0], "depth/frame_000000.bin")
        self.assertIsNone(result.depth_paths[1])
        ci = result.camera_info_payloads[0]
        self.assertEqual(ci["K"], [1.0] * 9)
        self.assertEqual(ci["P"], [2.0] * 12)
        self.assertEqual(ci["D"], [0.1, 0.2])
        self.assertEqual(ci["header"]["frame_id"], "front_camera_optical_frame")
        odom = result.odometry_payloads[0]
        self.assertEqual(odom["child_frame_id"], "body")
        self.assertEqual(len(odom["pose"]["covariance"]), 36)
        self.assertEqual(odom["twist"]["linear"]["x"], 1.0)
        self.assertEqual(odom["twist"]["angular"]["z"], 0.5)
        self.assertEqual(len(odom["twist"]["covariance"]), 36)
        depth = result.depth_metadata[0]
        self.assertEqual(depth["encoding"], "16UC1")
        self.assertEqual(result.tf_entries[0]["parent_frame_id"], "odom")
        self.assertIsNotNone(result.camera_info_digest)
        self.assertEqual(result.auxiliary_summary["depth_matched_frames"], 1)

    def test_v4_rejects_camera_info_digest_mismatch(self):
        aux_frame, digest = self._v4_aux_frame()
        extra = {
            "auxiliary/frame_000000.json": json.dumps(
                aux_frame, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            "depth/frame_000000.bin": b"\x01\x02",
            "tf.json": json.dumps(
                [
                    {
                        "stamp": {"sec": 10, "nsec": 0},
                        "parent_frame_id": "odom",
                        "child_frame_id": "body",
                        "translation": {"x": 1.0, "y": 0.0, "z": 0.0},
                        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        "static": False,
                    }
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        }
        self._write_extra_files(extra)
        self.fixture.extra_manifest = extra
        self.fixture.metadata = self._v4_metadata(aux_frame, "0" * 64)
        self.fixture.frame_columns = self._V4_COLUMNS
        self.fixture.frame_rows = self._v4_rows()
        self.fixture.write()
        self.assertInvalid("CAMERA_INFO_DIGEST_MISMATCH")

    def test_v4_rejects_camera_info_with_bad_k_matrix(self):
        def mutate(aux_frame):
            aux_frame["camera_info"]["K"] = [1.0] * 8
            return aux_frame

        self._write_v4_fixture(mutate)
        self.assertInvalid("MALFORMED_AUX")

    def test_v4_requires_exact_pose_and_twist_covariance(self):
        for group, covariance in (
            ("pose", None),
            ("pose", [0.0] * 35),
            ("twist", None),
            ("twist", [0.0] * 35),
        ):
            with self.subTest(group=group, covariance=covariance):
                def mutate(aux_frame):
                    if covariance is None:
                        aux_frame["odometry"][group].pop("covariance")
                    else:
                        aux_frame["odometry"][group]["covariance"] = covariance
                    return aux_frame

                self._write_v4_fixture(mutate)
                self.assertInvalid("MALFORMED_AUX")

    def test_v4_rejects_depth_data_file_mismatch(self):
        def mutate(aux_frame):
            aux_frame["depth"]["data_file"] = "depth/other.bin"
            return aux_frame

        self._write_v4_fixture(mutate)
        self.assertInvalid("MALFORMED_AUX")

    def test_v4_requires_tf_file_when_has_tf(self):
        aux_frame, digest = self._v4_aux_frame()
        self.fixture.metadata = self._v4_metadata(aux_frame, digest)
        self.fixture.frame_columns = self._V4_COLUMNS
        self.fixture.frame_rows = self._v4_rows()
        extra = {
            "auxiliary/frame_000000.json": json.dumps(
                aux_frame, sort_keys=True, separators=(",", ":")
            ).encode("utf-8"),
            "depth/frame_000000.bin": b"\x01\x02",
        }
        self._write_extra_files(extra)
        self.fixture.extra_manifest = extra
        self.fixture.write()
        self.assertInvalid("MANIFEST_COVERAGE")

    def test_v4_snapshot_returns_stamp_camera_role_and_pose_metadata(self):
        self._write_v4_fixture()
        result = self.fixture.validate()
        self.assertEqual(result.camera_name, "camera4")
        self.assertEqual(result.image_topic, "/camera4/image_raw")
        self.assertEqual(result.camera_role, "front")
        self.assertEqual(result.camera_frame_id, "front_camera_optical_frame")
        self.assertEqual(result.frame_count, 2)
        self.assertEqual(
            [(stamp.sec, stamp.nsec) for stamp in result.frame_stamps],
            [(10, 0), (12, 500000000)],
        )
        self.assertEqual(result.pose_valid, (True, False))
        self.assertEqual(result.pose_delta_ms, (12.5, None))
        self.assertEqual(result.odometry_gap_ms, (0.0, 240.0))
        self.assertTrue(result.has_odometry)

    def test_older_snapshot_versions_are_rejected(self):
        """v1–v3 (no auxiliary columns) are refused with an explicit message."""
        old_columns = FRAME_COLUMNS[:7]
        for version in (1, 2, 3):
            with self.subTest(version=version):
                fixture = SnapshotFixture(self.temp.name + "-old" + str(version))
                metadata = {key: fixture.metadata[key] for key in
                            ("schema", "status", "scene_id", "frame_count")}
                metadata["version"] = version
                fixture.metadata = metadata
                fixture.frame_columns = ("frame_index", "image_path") if version == 1 else old_columns
                fixture.frame_rows = [{key: row[key] for key in fixture.frame_columns}
                                      for row in fixture.frame_rows]
                fixture.write()
                with self.assertRaisesRegex(SnapshotInvalidError,
                                            "only snapshot version 4 is supported") as caught:
                    fixture.validate()
                self.assertEqual("UNSUPPORTED_SCHEMA", caught.exception.code)

    def test_invalid_role_and_frame_are_rejected(self):
        for field, value, code in (
            ("camera_role", "", "INVALID_CAMERA_ROLE"),
            ("camera_role", "bad role", "INVALID_CAMERA_ROLE"),
            ("camera_frame_id", "", "INVALID_CAMERA_FRAME"),
            ("camera_frame_id", "/leading", "INVALID_CAMERA_FRAME"),
        ):
            with self.subTest(field=field, value=value):
                fixture = SnapshotFixture(self.temp.name + "-" + field + "-" + str(value))
                fixture.metadata[field] = value
                fixture.write()
                self.assertInvalid(code, fixture.validate)

    def test_rejects_non_monotonic_frame_stamps(self):
        self._write_v4_fixture()
        rows = self._v4_rows()
        rows[1]["stamp_sec"] = "10"
        rows[1]["stamp_nsec"] = "0"
        self.fixture.frame_rows = rows
        self.fixture.write()
        self.assertInvalid("STAMP_ORDER")

    def test_rejects_endpoint_stamp_mismatch(self):
        self._write_v4_fixture()
        self.fixture.metadata["last_stamp_sec"] = 13
        self.fixture.write()
        self.assertInvalid("STAMP_MISMATCH")

    def test_rejects_nul_image_topic(self):
        self._write_v4_fixture()
        self.fixture.metadata["image_topic"] = "/camera\u0000escape"
        self.fixture.write()
        self.assertInvalid("INVALID_IMAGE_TOPIC")

    def test_rejects_valid_pose_with_unavailable_delta(self):
        self._write_v4_fixture()
        rows = self._v4_rows()
        rows[0]["pose_delta_ms"] = "-1"
        self.fixture.frame_rows = rows
        self.fixture.write()
        self.assertInvalid("MALFORMED_FRAMES")

if __name__ == "__main__":
    unittest.main()
