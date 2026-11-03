"""CPU-only tests for the frozen VGGT-Omega result artifact contract."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from result_validator import (
    ResultValidationError,
    ValidatedResult,
    validate_result_directory,
)
import result_validator
import reconstruct_omega
from vggt_omega.models.aggregator import Aggregator


class ResultValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.write_valid_result()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def arrays(self) -> dict[str, np.ndarray]:
        frames, height, width = 2, 3, 4
        extrinsic = np.zeros((frames, 3, 4), dtype=np.float32)
        extrinsic[:, :3, :3] = np.eye(3, dtype=np.float32)
        intrinsic = np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1))
        intrinsic[:, 0, 0] = 20.0
        intrinsic[:, 1, 1] = 21.0
        intrinsic[:, 0, 2] = width / 2.0
        intrinsic[:, 1, 2] = height / 2.0
        depth = np.ones((frames, height, width, 1), dtype=np.float32)
        world_points = reconstruct_omega.unproject_depth_map_to_point_map(
            depth, extrinsic, intrinsic
        )
        return {
            "depth": depth,
            "depth_conf": np.full((frames, height, width), 1.5, dtype=np.float32),
            "images": np.zeros((frames, 3, height, width), dtype=np.float32),
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
            "world_points_from_depth": world_points,
        }

    def write_valid_result(self) -> None:
        (self.root / "manifest.txt").write_text("frame_000.png\nframe_001.png\n")
        np.savez(self.root / "predictions.npz", **self.arrays())
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.box())
        scene.export(self.root / "scene.glb")

    def rewrite_npz(self, arrays: dict[str, np.ndarray]) -> None:
        np.savez(self.root / "predictions.npz", **arrays)

    def assert_invalid(self, fragment: str, *, require_glb: bool = True) -> None:
        with self.assertRaisesRegex(ResultValidationError, fragment):
            validate_result_directory(self.root, require_glb=require_glb)

    def test_valid_result(self) -> None:
        result = validate_result_directory(self.root, require_glb=True)
        self.assertEqual(result.frame_count, 2)
        self.assertEqual((result.height, result.width), (3, 4))
        self.assertTrue(result.glb_validated)

    def test_no_glb_mode_skips_glb_requirement(self) -> None:
        (self.root / "scene.glb").unlink()
        result = validate_result_directory(self.root, require_glb=False)
        self.assertFalse(result.glb_validated)

    def test_empty_manifest_is_rejected(self) -> None:
        (self.root / "manifest.txt").write_text("")
        self.assert_invalid("at least one")

    def test_unsafe_manifest_entry_is_rejected(self) -> None:
        (self.root / "manifest.txt").write_text("../frame.png\n")
        self.assert_invalid("unsafe")

    def test_missing_key_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays.pop("depth_conf")
        self.rewrite_npz(arrays)
        self.assert_invalid("missing required keys.*depth_conf")

    def test_object_dtype_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["images"] = np.array([object()], dtype=object)
        self.rewrite_npz(arrays)
        self.assert_invalid("object dtype")

    def test_frame_count_mismatch_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["depth"] = arrays["depth"][:1]
        self.rewrite_npz(arrays)
        self.assert_invalid("depth.*expected")

    def test_spatial_shape_mismatch_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["depth_conf"] = arrays["depth_conf"][:, :, :-1]
        self.rewrite_npz(arrays)
        self.assert_invalid("depth_conf.*expected")

    def test_non_finite_dense_value_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["world_points_from_depth"][0, 0, 0, 0] = np.nan
        self.rewrite_npz(arrays)
        self.assert_invalid("world_points_from_depth.*non-finite")

    def test_nonpositive_focal_length_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["intrinsic"][0, 0, 0] = 0.0
        self.rewrite_npz(arrays)
        self.assert_invalid("positive focal")

    def test_non_orthonormal_rotation_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["extrinsic"][0, 0, 0] = 2.0
        self.rewrite_npz(arrays)
        self.assert_invalid("orthonormal")

    def test_reflected_rotation_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["extrinsic"][0, 0, 0] = -1.0
        self.rewrite_npz(arrays)
        self.assert_invalid("determinants")

    def test_no_positive_depth_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["depth"].fill(0.0)
        self.rewrite_npz(arrays)
        self.assert_invalid("no positive")

    def test_negative_depth_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["depth"][0, 0, 0, 0] = -1.0
        self.rewrite_npz(arrays)
        self.assert_invalid("depth.*below")

    def test_confidence_below_one_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["depth_conf"][0, 0, 0] = 0.5
        self.rewrite_npz(arrays)
        self.assert_invalid("depth_conf.*below")

    def test_image_outside_unit_range_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["images"][0, 0, 0, 0] = 42.0
        self.rewrite_npz(arrays)
        self.assert_invalid("images.*above")

    def test_float64_array_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["depth_conf"] = arrays["depth_conf"].astype(np.float64)
        self.rewrite_npz(arrays)
        self.assert_invalid("depth_conf.*float32")

    def test_impossible_intrinsic_layout_is_rejected(self) -> None:
        arrays = self.arrays()
        arrays["intrinsic"][0, 0, 1] = 7.0
        self.rewrite_npz(arrays)
        self.assert_invalid("camera layout")

    def test_world_points_must_match_depth_and_cameras(self) -> None:
        arrays = self.arrays()
        arrays["world_points_from_depth"].fill(123.0)
        self.rewrite_npz(arrays)
        self.assert_invalid("inconsistent")

    def test_world_point_check_matches_producer_arithmetic(self) -> None:
        arrays = self.arrays()
        arrays["extrinsic"][0, 0, 3] = -20000.0
        arrays["intrinsic"][0, 0, 0] = 0.01
        arrays["intrinsic"][0, 1, 1] = 0.01
        arrays["depth"][0].fill(100.0)
        arrays["world_points_from_depth"] = (
            reconstruct_omega.unproject_depth_map_to_point_map(
                arrays["depth"], arrays["extrinsic"], arrays["intrinsic"]
            )
        )
        self.rewrite_npz(arrays)
        validate_result_directory(self.root, require_glb=True)

    def test_compressed_npz_members_are_rejected_before_loading(self) -> None:
        np.savez_compressed(self.root / "predictions.npz", **self.arrays())
        self.assert_invalid("without compression")

    def test_manifest_frame_limit_is_checked_before_array_loading(self) -> None:
        (self.root / "manifest.txt").write_text(
            "".join(f"frame_{index:03}.png\n" for index in range(257)),
            encoding="utf-8",
        )
        self.assert_invalid("exceeds validation limit")

    def test_pixel_limit_is_checked_before_array_loading(self) -> None:
        with self.assertRaisesRegex(ResultValidationError, "exceeds validation limit"):
            validate_result_directory(
                self.root,
                require_glb=True,
                max_pixels=1,
            )

    def test_malformed_npz_uses_validation_error(self) -> None:
        (self.root / "predictions.npz").write_bytes(b"PK\\x03\\x04")
        self.assert_invalid("could not be loaded safely")

    def test_malformed_npz_member_uses_validation_error(self) -> None:
        with zipfile.ZipFile(self.root / "predictions.npz", "w") as archive:
            for key in (
                "depth",
                "depth_conf",
                "images",
                "extrinsic",
                "intrinsic",
                "world_points_from_depth",
            ):
                archive.writestr(f"{key}.npy", b"notnpy")
        self.assert_invalid("invalid NPY header")

    def test_symlinked_artifact_is_rejected(self) -> None:
        external = self.root / "external.npz"
        (self.root / "predictions.npz").replace(external)
        (self.root / "predictions.npz").symlink_to(external)
        self.assert_invalid("regular file")

    def test_expected_manifest_names_and_order_are_checked(self) -> None:
        with self.assertRaisesRegex(ResultValidationError, "expected input"):
            validate_result_directory(
                self.root,
                require_glb=True,
                expected_image_basenames=["other.png", "frame_001.png"],
            )

    def test_no_glb_mode_rejects_stale_scene(self) -> None:
        self.assert_invalid("must be absent", require_glb=False)

    def test_malformed_glb_is_rejected(self) -> None:
        (self.root / "scene.glb").write_bytes(b"not a GLB")
        self.assert_invalid("could not be parsed")

    def test_empty_glb_is_rejected(self) -> None:
        (self.root / "scene.glb").write_bytes(b"")
        self.assert_invalid("nonempty")

    def test_explicit_input_images_preserve_caller_order(self) -> None:
        ordered = [
            str(self.root / "frame_001.PNG"),
            str(self.root / "frame_000.jpg"),
        ]
        self.assertEqual(
            reconstruct_omega._select_input_images("unused", ordered),
            ordered,
        )

    def test_explicit_input_image_rejections_are_deterministic(self) -> None:
        duplicate = str(self.root / "frame.png")
        cases = (
            ([], "at least one"),
            ([""], "nonempty"),
            (["relative.png"], "absolute"),
            ([str(self.root / "frame.gif")], "Unsupported"),
            ([duplicate, duplicate], "Duplicate --input_image path"),
            (
                [
                    str(self.root / "first" / "same.png"),
                    str(self.root / "second" / "same.png"),
                ],
                "Duplicate --input_image basename",
            ),
        )
        for paths, message in cases:
            with self.subTest(paths=paths):
                with self.assertRaisesRegex(ValueError, message):
                    reconstruct_omega._validate_ordered_input_images(paths)

    def test_images_dir_discovery_remains_sorted_when_explicit_list_absent(self) -> None:
        images_dir = self.root / "legacy-images"
        images_dir.mkdir()
        for name in ("z.png", "ignored.txt", "a.JPG"):
            (images_dir / name).write_bytes(b"placeholder")
        self.assertEqual(
            reconstruct_omega._select_input_images(str(images_dir), None),
            [str(images_dir / "a.JPG"), str(images_dir / "z.png")],
        )

    def test_aggregator_frame_chunking_preserves_independent_frame_blocks(self) -> None:
        class AddOne(torch.nn.Module):
            def forward(self, value, rope):
                return value + 1

        aggregator = object.__new__(Aggregator)
        torch.nn.Module.__init__(aggregator)
        aggregator.frame_blocks = torch.nn.ModuleList([AddOne()])
        tokens = torch.arange(60, dtype=torch.float32).view(5, 3, 4)

        aggregator.frame_chunk_size = None
        reference, _ = aggregator._run_frame_block(tokens, 1, 5, 3, 4, 0, None)
        aggregator.frame_chunk_size = 2
        with torch.inference_mode():
            chunked, shaped = aggregator._run_frame_block(
                tokens, 1, 5, 3, 4, 0, None
            )

        self.assertTrue(torch.equal(chunked, reference))
        self.assertEqual(tuple(shaped.shape), (1, 5, 3, 4))
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            aggregator._run_frame_block(tokens, 1, 5, 3, 4, 0, None)

    def test_supervisor_output_must_be_existing_absolute_canonical_directory(self) -> None:
        real_directory = self.root / "real-staging"
        real_directory.mkdir()
        alias = self.root / "staging-alias"
        alias.symlink_to(real_directory, target_is_directory=True)

        self.assertEqual(
            reconstruct_omega._supervisor_output_directory(real_directory),
            real_directory,
        )
        for invalid in (
            Path("relative-staging"),
            self.root / "missing-staging",
            alias,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "supervisor_staging"):
                    reconstruct_omega._supervisor_output_directory(invalid)

    def test_supervisor_mode_never_owns_directory_lifecycle_on_success(self) -> None:
        args = reconstruct_omega.parse_args(
            [
                "--supervisor_staging",
                "--output_dir",
                str(self.root),
            ]
        )
        validation = ValidatedResult(2, 3, 4, True)
        with (
            mock.patch.object(
                reconstruct_omega,
                "_run_reconstruction",
                return_value=validation,
            ) as run_reconstruction,
            mock.patch.object(reconstruct_omega, "_create_output_transaction") as allocate,
            mock.patch.object(reconstruct_omega, "_rename_noreplace") as rename,
            mock.patch.object(reconstruct_omega.shutil, "rmtree") as rmtree,
        ):
            self.assertEqual(reconstruct_omega._run_main(args), validation)

        run_reconstruction.assert_called_once_with(args, self.root)
        allocate.assert_not_called()
        rename.assert_not_called()
        rmtree.assert_not_called()
        self.assertTrue(self.root.is_dir())

    def test_supervisor_mode_never_owns_directory_lifecycle_on_failure(self) -> None:
        args = reconstruct_omega.parse_args(
            [
                "--supervisor_staging",
                "--output_dir",
                str(self.root),
            ]
        )
        marker = self.root / "supervisor-owned"
        marker.write_text("keep", encoding="utf-8")
        with (
            mock.patch.object(
                reconstruct_omega,
                "_run_reconstruction",
                side_effect=RuntimeError("worker failed"),
            ),
            mock.patch.object(reconstruct_omega, "_create_output_transaction") as allocate,
            mock.patch.object(reconstruct_omega, "_rename_noreplace") as rename,
            mock.patch.object(reconstruct_omega.shutil, "rmtree") as rmtree,
        ):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                reconstruct_omega._run_main(args)

        allocate.assert_not_called()
        rename.assert_not_called()
        rmtree.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_validator_cli_emits_one_strict_success_summary(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = result_validator.main(
                [
                    "--output_dir",
                    str(self.root),
                    "--require_glb",
                    "--expected_image_basename",
                    "frame_000.png",
                    "--expected_image_basename",
                    "frame_001.png",
                    "--max_frames",
                    "8",
                    "--max_pixels",
                    "64",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(stdout.getvalue().splitlines()), 1)
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {
                "frame_count": 2,
                "glb_validated": True,
                "height": 3,
                "ok": True,
                "width": 4,
            },
        )

    def test_validator_cli_emits_one_strict_failure_summary(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = result_validator.main(
                [
                    "--output_dir",
                    str(self.root),
                    "--require_glb",
                    "--expected_image_basename",
                    "wrong.png",
                    "--max_frames",
                    "8",
                    "--max_pixels",
                    "64",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertEqual(len(stdout.getvalue().splitlines()), 1)
        summary = json.loads(stdout.getvalue())
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["error_type"], "ResultValidationError")
        self.assertIn("expected input", summary["error"])

    def test_cli_refuses_to_overwrite_existing_generation(self) -> None:
        manifest_before = (self.root / "manifest.txt").read_bytes()
        predictions_before = (self.root / "predictions.npz").read_bytes()
        with self.assertRaisesRegex(FileExistsError, "fresh output directory"):
            reconstruct_omega._create_output_transaction(self.root)
        self.assertEqual((self.root / "manifest.txt").read_bytes(), manifest_before)
        self.assertEqual((self.root / "predictions.npz").read_bytes(), predictions_before)

    def test_atomic_publish_does_not_replace_late_destination(self) -> None:
        source = self.root / "staging"
        destination = self.root / "late-destination"
        source.mkdir()
        (source / "artifact").write_text("new", encoding="utf-8")
        destination.mkdir()
        (destination / "owner").write_text("other", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            reconstruct_omega._rename_noreplace(source, destination)

        self.assertTrue((source / "artifact").is_file())
        self.assertEqual(
            (destination / "owner").read_text(encoding="utf-8"),
            "other",
        )

    def test_cli_does_not_report_success_for_invalid_output(self) -> None:
        images_dir = self.root / "images"
        images_dir.mkdir()
        for name in ("frame_000.png", "frame_001.png"):
            (images_dir / name).write_bytes(b"placeholder")
        output_dir = self.root / "cli-output"
        invalid_predictions = self.arrays()
        invalid_predictions["depth"].fill(0.0)

        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "reconstruct_omega.py",
                    "--images_dir",
                    str(images_dir),
                    "--output_dir",
                    str(output_dir),
                    "--checkpoint",
                    "unused.pt",
                    "--no_glb",
                ],
            ),
            mock.patch.object(reconstruct_omega, "detect_config", return_value=(256, 8, 0.0)),
            mock.patch.object(reconstruct_omega, "load_and_preprocess_images", return_value=np.zeros((2, 3, 3, 4))),
            mock.patch.object(reconstruct_omega, "load_model", return_value=object()),
            mock.patch.object(reconstruct_omega, "run_inference", return_value=invalid_predictions),
            mock.patch.object(reconstruct_omega.torch.cuda, "reset_peak_memory_stats"),
            mock.patch.object(reconstruct_omega.torch.cuda, "max_memory_allocated", return_value=0),
            mock.patch.object(reconstruct_omega.torch.cuda, "empty_cache"),
            mock.patch("sys.stdout") as stdout,
        ):
            with self.assertRaisesRegex(ResultValidationError, "no positive"):
                reconstruct_omega.main()

        output = "".join(call.args[0] for call in stdout.write.call_args_list)
        self.assertNotIn("Done.", output)
        self.assertFalse(output_dir.exists())
        self.assertEqual(list(self.root.glob(".cli-output.partial-*")), [])

    def test_cli_atomically_publishes_valid_no_glb_result(self) -> None:
        images_dir = self.root / "success-images"
        images_dir.mkdir()
        for name in ("frame_000.png", "frame_001.png"):
            (images_dir / name).write_bytes(b"placeholder")
        output_dir = self.root / "success-output"

        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "reconstruct_omega.py",
                    "--images_dir",
                    str(images_dir),
                    "--output_dir",
                    str(output_dir),
                    "--checkpoint",
                    "unused.pt",
                    "--no_glb",
                ],
            ),
            mock.patch.object(reconstruct_omega, "detect_config", return_value=(256, 8, 0.0)),
            mock.patch.object(reconstruct_omega, "load_and_preprocess_images", return_value=np.zeros((2, 3, 3, 4))),
            mock.patch.object(reconstruct_omega, "load_model", return_value=object()),
            mock.patch.object(reconstruct_omega, "run_inference", return_value=self.arrays()),
            mock.patch.object(reconstruct_omega.torch.cuda, "reset_peak_memory_stats"),
            mock.patch.object(reconstruct_omega.torch.cuda, "max_memory_allocated", return_value=0),
            mock.patch.object(reconstruct_omega.torch.cuda, "empty_cache"),
        ):
            reconstruct_omega.main()

        self.assertTrue((output_dir / "manifest.txt").is_file())
        self.assertTrue((output_dir / "predictions.npz").is_file())
        self.assertFalse((output_dir / "scene.glb").exists())
        self.assertEqual(list(self.root.glob(".success-output.partial-*")), [])

    def test_checkpoint_digest_verification_accepts_exact_and_rejects_mismatch(self) -> None:
        import hashlib

        checkpoint = self.root / "checkpoint.pt"
        checkpoint.write_bytes(b"explicit checkpoint bytes")
        expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        reconstruct_omega._verify_checkpoint_digest(str(checkpoint), expected)
        with self.assertRaisesRegex(RuntimeError, "checkpoint digest mismatch"):
            reconstruct_omega._verify_checkpoint_digest(str(checkpoint), "a" * 64)
        for malformed in ("", "A" * 64, "abc", 123):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    reconstruct_omega._verify_checkpoint_digest(
                        str(checkpoint), malformed
                    )
        reconstruct_omega._verify_checkpoint_digest(str(checkpoint), None)
        link = self.root / "checkpoint-link.pt"
        link.symlink_to(checkpoint)
        with self.assertRaises(OSError):
            reconstruct_omega._verify_checkpoint_digest(str(link), expected)

    def test_camera_trunk_depth_requires_contiguous_blocks(self) -> None:
        state = {
            "camera_head.trunk.0.attn.weight": object(),
            "camera_head.trunk.1.attn.weight": object(),
            "camera_head.trunk_norm.weight": object(),
        }
        self.assertEqual(reconstruct_omega._camera_trunk_depth(state), 2)
        state.pop("camera_head.trunk.1.attn.weight")
        state["camera_head.trunk.2.attn.weight"] = object()
        with self.assertRaisesRegex(RuntimeError, "noncontiguous"):
            reconstruct_omega._camera_trunk_depth(state)

    def test_legacy_compact_checkpoint_is_rejected(self) -> None:
        for dtype_key in (
            "aggregator_storage_dtype",
            "aggregator_buffer_storage_dtype",
        ):
            with self.subTest(dtype_key=dtype_key), self.assertRaisesRegex(
                RuntimeError, "Aggregator buffers"
            ):
                reconstruct_omega._validate_deployment_metadata(
                    {
                        "camera_head_trunk_depth": 2,
                        dtype_key: "bfloat16",
                    },
                    2,
                )


if __name__ == "__main__":
    unittest.main()
