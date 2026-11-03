"""Validate VGGT-Omega reconstruction artifacts before publishing success.

This CPU-side validator is intentionally independent of CUDA, checkpoints, and
ROS.  It validates the arrays emitted by ``reconstruct_omega.run_inference``
and the files written by its CLI, so a later supervisor can reject corrupt
worker output with a single, field-specific exception type.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Final, Sequence
import zipfile

import numpy as np


REQUIRED_KEYS: Final = (
    "depth",
    "depth_conf",
    "images",
    "extrinsic",
    "intrinsic",
    "world_points_from_depth",
)
REQUIRED_KEY_SET: Final = frozenset(REQUIRED_KEYS)
ROTATION_ORTHOGONALITY_ATOL: Final = 1e-3
ROTATION_DETERMINANT_ATOL: Final = 1e-3
CAMERA_LAYOUT_ATOL: Final = 1e-5
WORLD_POINTS_RTOL: Final = 1e-4
WORLD_POINTS_ATOL: Final = 1e-4
FINITE_CHUNK_ELEMENTS: Final = 1_000_000
DEFAULT_MAX_FRAMES: Final = 256
DEFAULT_MAX_PIXELS: Final = 2 * 512 * 512


class ResultValidationError(ValueError):
    """Raised when an Omega result directory violates the artifact contract."""


@dataclass(frozen=True)
class ValidatedResult:
    """Small summary suitable for a future supervisor result manifest."""

    frame_count: int
    height: int
    width: int
    glb_validated: bool


def _fail(message: str) -> None:
    raise ResultValidationError(message)


def _read_manifest(path: Path) -> list[str]:
    if path.is_symlink() or not path.is_file():
        _fail(f"manifest.txt must be a regular file in the result directory: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _fail(f"manifest.txt could not be read: {exc}")

    entries = text.splitlines()
    if not entries:
        _fail("manifest.txt must contain at least one image basename")

    for entry in entries:
        candidate = Path(entry)
        if not entry or candidate.name != entry or entry in {".", ".."}:
            _fail(f"manifest.txt contains an unsafe image basename: {entry!r}")
    if len(entries) != len(set(entries)):
        _fail("manifest.txt contains duplicate image basenames")
    return entries


def _load_array(arrays: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    try:
        value = arrays[key]
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        if "Object arrays cannot be loaded" in str(exc):
            _fail(f"predictions.npz[{key!r}] must not use object dtype")
        _fail(f"predictions.npz[{key!r}] could not be loaded: {exc}")
    except MemoryError as exc:
        _fail(f"predictions.npz[{key!r}] is too large to load: {exc}")

    if not isinstance(value, np.ndarray):
        _fail(f"predictions.npz[{key!r}] is not a NumPy array")
    if value.dtype.hasobject:
        _fail(f"predictions.npz[{key!r}] must not use object dtype")
    if value.dtype != np.dtype(np.float32):
        _fail(
            f"predictions.npz[{key!r}] must use float32, got {value.dtype}"
        )
    if value.size == 0:
        _fail(f"predictions.npz[{key!r}] must not be empty")
    return value


def _flat_chunks(value: np.ndarray):
    flat = np.ravel(value, order="K")
    for start in range(0, flat.size, FINITE_CHUNK_ELEMENTS):
        yield flat[start : start + FINITE_CHUNK_ELEMENTS]


def _require_finite(value: np.ndarray, key: str) -> None:
    if any(not np.isfinite(chunk).all() for chunk in _flat_chunks(value)):
        _fail(f"predictions.npz[{key!r}] contains non-finite values")


def _require_range(
    value: np.ndarray,
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    for chunk in _flat_chunks(value):
        if minimum is not None and np.any(chunk < minimum):
            _fail(f"predictions.npz[{key!r}] contains values below {minimum}")
        if maximum is not None and np.any(chunk > maximum):
            _fail(f"predictions.npz[{key!r}] contains values above {maximum}")


def _require_shape(value: np.ndarray, key: str, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        _fail(
            f"predictions.npz[{key!r}] has shape {value.shape}; expected {expected}"
        )


def _validate_world_points(
    depth: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    actual: np.ndarray,
) -> None:
    """Recompute one frame at a time to bound memory without Python row loops."""
    _, height, width, _ = depth.shape
    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    for frame in range(depth.shape[0]):
        frame_depth = depth[frame, ..., 0]
        rotation_t = extrinsic[frame, :3, :3].T
        translation = extrinsic[frame, :3, 3]
        fx, fy = intrinsic[frame, 0, 0], intrinsic[frame, 1, 1]
        cx, cy = intrinsic[frame, 0, 2], intrinsic[frame, 1, 2]
        camera_points = np.stack(
            (
                (x - cx) / fx * frame_depth,
                (y - cy) / fy * frame_depth,
                frame_depth,
            ),
            axis=-1,
        )
        expected = np.einsum(
            "ij,hwj->hwi",
            rotation_t,
            (camera_points - translation).astype(np.float32),
        ).astype(np.float32)
        if not np.allclose(
            actual[frame],
            expected,
            rtol=WORLD_POINTS_RTOL,
            atol=WORLD_POINTS_ATOL,
        ):
            _fail(
                "predictions.npz['world_points_from_depth'] is inconsistent "
                "with depth, extrinsic, and intrinsic"
            )


def _read_npy_member_header(member, key: str):
    try:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
        else:
            _fail(
                f"predictions.npz[{key!r}] uses unsupported NPY version {version}"
            )
    except (OSError, ValueError, EOFError) as exc:
        _fail(f"predictions.npz[{key!r}] has an invalid NPY header: {exc}")
    return tuple(shape), bool(fortran_order), np.dtype(dtype), member.tell()


def _preflight_npz(
    path: Path,
    frame_count: int,
    *,
    max_frames: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Validate stored NPY headers before NumPy allocates dense arrays."""
    if frame_count > max_frames:
        _fail(
            f"manifest frame count {frame_count} exceeds validation limit {max_frames}"
        )
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        _fail(f"predictions.npz could not be loaded safely: {exc}")

    with archive:
        names = [info.filename for info in archive.infolist()]
        headers = {}
        for key in REQUIRED_KEYS:
            member_name = f"{key}.npy"
            count = names.count(member_name)
            if count == 0:
                _fail(f"predictions.npz is missing required keys: [{key!r}]")
            if count != 1:
                _fail(
                    f"predictions.npz must contain exactly one member {member_name!r}"
                )
            info = archive.getinfo(member_name)
            if info.compress_type != zipfile.ZIP_STORED:
                _fail(
                    f"predictions.npz[{key!r}] must be stored without compression"
                )
            with archive.open(info) as member:
                shape, fortran_order, dtype, header_bytes = _read_npy_member_header(
                    member, key
                )
            if fortran_order:
                _fail(f"predictions.npz[{key!r}] must use C array order")
            if dtype.hasobject:
                _fail(f"predictions.npz[{key!r}] must not use object dtype")
            if dtype != np.dtype(np.float32):
                _fail(
                    f"predictions.npz[{key!r}] must use float32, got {dtype}"
                )
            element_count = math.prod(shape)
            expected_size = header_bytes + element_count * dtype.itemsize
            if expected_size != info.file_size:
                _fail(
                    f"predictions.npz[{key!r}] member size does not match its header"
                )
            headers[key] = shape

    depth_shape = headers["depth"]
    if len(depth_shape) != 4 or depth_shape[-1] != 1:
        _fail(
            "predictions.npz['depth'] must have shape [N, H, W, 1]; "
            f"got {depth_shape}"
        )
    _, height, width, _ = depth_shape
    if height <= 0 or width <= 0 or height * width > max_pixels:
        _fail(
            f"predictions.npz image geometry {height}x{width} exceeds "
            f"validation limit {max_pixels} pixels"
        )
    expected_shapes = {
        "depth": (frame_count, height, width, 1),
        "depth_conf": (frame_count, height, width),
        "images": (frame_count, 3, height, width),
        "extrinsic": (frame_count, 3, 4),
        "intrinsic": (frame_count, 3, 3),
        "world_points_from_depth": (frame_count, height, width, 3),
    }
    for key, expected in expected_shapes.items():
        if headers[key] != expected:
            _fail(
                f"predictions.npz[{key!r}] has shape {headers[key]}; expected {expected}"
            )
    return height, width


def _validate_npz(
    path: Path,
    frame_count: int,
    *,
    max_frames: int,
    max_pixels: int,
) -> tuple[int, int]:
    if path.is_symlink() or not path.is_file():
        _fail(f"predictions.npz must be a regular file in the result directory: {path}")

    _preflight_npz(
        path,
        frame_count,
        max_frames=max_frames,
        max_pixels=max_pixels,
    )
    try:
        arrays = np.load(path, allow_pickle=False)
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        _fail(f"predictions.npz could not be loaded safely: {exc}")
    except MemoryError as exc:
        _fail(f"predictions.npz is too large to load safely: {exc}")
    if not isinstance(arrays, np.lib.npyio.NpzFile):
        _fail("predictions.npz must be an NPZ archive")

    with arrays:
        keys = set(arrays.files)
        missing = sorted(REQUIRED_KEY_SET - keys)
        if missing:
            _fail(f"predictions.npz is missing required keys: {missing}")

        depth = _load_array(arrays, "depth")
        if depth.ndim != 4 or depth.shape[-1] != 1:
            _fail(
                "predictions.npz['depth'] must have shape [N, H, W, 1]; "
                f"got {depth.shape}"
            )
        _, height, width, _ = depth.shape
        if height <= 0 or width <= 0:
            _fail("predictions.npz['depth'] must have positive H and W")
        _require_shape(depth, "depth", (frame_count, height, width, 1))
        _require_finite(depth, "depth")
        _require_range(depth, "depth", minimum=0.0)
        if not any(np.any(chunk > 0) for chunk in _flat_chunks(depth)):
            _fail("predictions.npz['depth'] contains no positive finite depth values")

        depth_conf = _load_array(arrays, "depth_conf")
        _require_shape(depth_conf, "depth_conf", (frame_count, height, width))
        _require_finite(depth_conf, "depth_conf")
        _require_range(depth_conf, "depth_conf", minimum=1.0)
        del depth_conf

        images = _load_array(arrays, "images")
        _require_shape(images, "images", (frame_count, 3, height, width))
        _require_finite(images, "images")
        _require_range(images, "images", minimum=0.0, maximum=1.0)
        del images

        world_points = _load_array(arrays, "world_points_from_depth")
        _require_shape(
            world_points,
            "world_points_from_depth",
            (frame_count, height, width, 3),
        )
        _require_finite(world_points, "world_points_from_depth")

        extrinsic = _load_array(arrays, "extrinsic")
        _require_shape(extrinsic, "extrinsic", (frame_count, 3, 4))
        _require_finite(extrinsic, "extrinsic")
        rotation = extrinsic[:, :3, :3]
        gram = np.matmul(np.transpose(rotation, (0, 2, 1)), rotation)
        identity = np.broadcast_to(np.eye(3, dtype=rotation.dtype), gram.shape)
        if not np.allclose(gram, identity, rtol=0.0, atol=ROTATION_ORTHOGONALITY_ATOL):
            _fail(
                "predictions.npz['extrinsic'] rotation blocks are not approximately "
                "orthonormal"
            )
        determinant = np.linalg.det(rotation.astype(np.float64))
        if not np.allclose(
            determinant, 1.0, rtol=0.0, atol=ROTATION_DETERMINANT_ATOL
        ):
            _fail(
                "predictions.npz['extrinsic'] rotation determinants must be approximately +1"
            )
        del rotation, gram

        intrinsic = _load_array(arrays, "intrinsic")
        _require_shape(intrinsic, "intrinsic", (frame_count, 3, 3))
        _require_finite(intrinsic, "intrinsic")
        if not np.all(intrinsic[:, 0, 0] > 0) or not np.all(intrinsic[:, 1, 1] > 0):
            _fail("predictions.npz['intrinsic'] must contain positive focal lengths")
        expected_layout = np.zeros_like(intrinsic)
        expected_layout[:, 0, 0] = intrinsic[:, 0, 0]
        expected_layout[:, 1, 1] = intrinsic[:, 1, 1]
        expected_layout[:, 0, 2] = width / 2.0
        expected_layout[:, 1, 2] = height / 2.0
        expected_layout[:, 2, 2] = 1.0
        if not np.allclose(
            intrinsic, expected_layout, rtol=0.0, atol=CAMERA_LAYOUT_ATOL
        ):
            _fail(
                "predictions.npz['intrinsic'] does not match the producer camera layout"
            )

        _validate_world_points(depth, extrinsic, intrinsic, world_points)
        del depth, extrinsic, intrinsic, world_points

    return height, width


def _validate_glb(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        _fail(f"scene.glb must be a nonempty regular file in the result directory: {path}")

    try:
        import trimesh
    except ImportError as exc:
        _fail(f"scene.glb validation requires trimesh: {exc}")

    try:
        loaded = trimesh.load(path, force="scene", process=False)
    except (OSError, ValueError, TypeError, IndexError) as exc:
        _fail(f"scene.glb could not be parsed: {exc}")

    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        _fail("scene.glb contains no geometry")
    bounds = np.asarray(loaded.bounds)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        _fail("scene.glb bounds must be finite [min, max] XYZ values")
    if np.any(bounds[0] > bounds[1]):
        _fail("scene.glb bounds are inverted")


def validate_result_directory(
    output_dir: str | Path,
    *,
    require_glb: bool,
    expected_image_basenames: Sequence[str] | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> ValidatedResult:
    """Validate artifacts emitted by one VGGT-Omega CLI invocation.

    Additional NPZ keys are allowed for forward-compatible metadata, but the
    six keys currently produced by ``run_inference`` remain mandatory.
    """
    root = Path(output_dir)
    if max_frames <= 0 or max_pixels <= 0:
        raise ValueError("max_frames and max_pixels must be positive")
    if root.is_symlink() or not root.is_dir():
        _fail(f"result directory must be a real directory: {root}")

    manifest_entries = _read_manifest(root / "manifest.txt")
    if (
        expected_image_basenames is not None
        and manifest_entries != list(expected_image_basenames)
    ):
        _fail("manifest.txt does not match the expected input image basenames and order")
    frame_count = len(manifest_entries)
    height, width = _validate_npz(
        root / "predictions.npz",
        frame_count,
        max_frames=max_frames,
        max_pixels=max_pixels,
    )
    glb_path = root / "scene.glb"
    if require_glb:
        _validate_glb(glb_path)
    elif glb_path.exists() or glb_path.is_symlink():
        _fail("scene.glb must be absent when GLB output is disabled")

    return ValidatedResult(
        frame_count=frame_count,
        height=height,
        width=width,
        glb_validated=require_glb,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate one VGGT-Omega reconstruction result directory"
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Existing reconstruction result directory",
    )
    glb_group = parser.add_mutually_exclusive_group(required=True)
    glb_group.add_argument(
        "--require_glb",
        "--require-glb",
        dest="require_glb",
        action="store_true",
        help="Require and validate scene.glb",
    )
    glb_group.add_argument(
        "--no_require_glb",
        "--no-require-glb",
        dest="require_glb",
        action="store_false",
        help="Require scene.glb to be absent",
    )
    parser.add_argument(
        "--expected_image_basename",
        "--expected-image-basename",
        dest="expected_image_basenames",
        action="append",
        required=True,
        help="Expected manifest basename; repeat in exact input order",
    )
    parser.add_argument(
        "--max_frames",
        "--max-frames",
        dest="max_frames",
        type=_positive_int,
        required=True,
    )
    parser.add_argument(
        "--max_pixels",
        "--max-pixels",
        dest="max_pixels",
        type=_positive_int,
        required=True,
    )
    return parser.parse_args(argv)


def _emit_json_summary(summary) -> None:
    print(json.dumps(summary, allow_nan=False, separators=(",", ":"), sort_keys=True))


def main(argv=None) -> int:
    """Validate from the command line and emit exactly one JSON summary line."""
    args = _parse_cli_args(argv)
    try:
        result = validate_result_directory(
            args.output_dir,
            require_glb=args.require_glb,
            expected_image_basenames=args.expected_image_basenames,
            max_frames=args.max_frames,
            max_pixels=args.max_pixels,
        )
    except ResultValidationError as exc:
        _emit_json_summary(
            {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "ok": False,
            }
        )
        return 1

    _emit_json_summary(
        {
            "frame_count": result.frame_count,
            "glb_validated": result.glb_validated,
            "height": result.height,
            "ok": True,
            "width": result.width,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
