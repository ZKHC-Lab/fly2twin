# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VGGT-Omega reconstruction entry point (Drone_3D S380).

Runs the 1B model on an 8 GB Jetson: bf16 aggregator + fp32 heads, max_size
image load, expandable_segments, and frame_group_size subsampling. Output goes
to <output_dir>/predictions.npz + scene.glb + manifest.txt.
"""

import argparse
import ctypes
import hashlib
import errno
import gc
import glob
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

# Set before any CUDA alloc; avoids a fragmentation OOM-cliff on Jetson (~32 frames).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn

# Allow `python vggt-omega/reconstruct_omega.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from result_validator import validate_result_directory


def detect_config(num_images):
    """Pick resolution + frame cap from VRAM.

    The 8 GB tier is deliberately conservative because CUDA and CPU share the
    same physical RAM on Jetson.  It was verified while two cameras, aligned
    depth, Livox and FAST-LIO were running; 16x256 and 8x512 were OOM-killed.

    8 GB  → 8 @ 256 (verified twice with the static sensor stack online)
    16 GB → 160 (measured on Orin 16GB: 9.6GB peak, no swap, ~3GB headroom.
                 180 still completes but spills ~320MB to swap, so 160 is the
                 robust no-swap default; pass --frame_group_size 180 to push.)
    24 GB+ → 256 (estimate; ~20GB available → linear-model ceiling ~300.
                  Untested on this device - override if it OOMs.)
    """
    if torch.cuda.is_available():
        total_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    else:
        total_mem_mb = 0.0

    if total_mem_mb > 0 and total_mem_mb < 12000:
        resolution, max_frames = 256, 8
    elif total_mem_mb < 24000:
        resolution, max_frames = 512, 160
    else:
        resolution, max_frames = 512, 256

    return resolution, max_frames, total_mem_mb


def amp_dtype():
    """bf16 on Ampere+ (cc>=8), else fp16 — matches VGGT-X."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float16


def _assign_state_dict(model, state_dict, aggregator_dtype):
    """Materialize a meta-device model without a second fp32 weight copy.

    NVIDIA's JetPack 5 torch predates ``load_state_dict(assign=True)``.  The
    equivalent private-module assignment below is kept small and guarded by
    exact key/shape checks so a changed checkpoint fails before inference.
    """
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint keys do not match model: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    for key, tensor in state_dict.items():
        if tuple(tensor.shape) != tuple(expected[key].shape):
            raise RuntimeError(
                f"Checkpoint shape mismatch for {key}: "
                f"{tuple(tensor.shape)} != {tuple(expected[key].shape)}"
            )

    parameter_names = {name for name, _ in model.named_parameters()}
    cast_count = 0
    for key in parameter_names:
        tensor = state_dict[key]
        if key.startswith("aggregator.") and tensor.is_floating_point():
            state_dict[key] = tensor.to(aggregator_dtype)
            cast_count += 1

    for key, tensor in state_dict.items():
        parts = key.split(".")
        owner = model
        for part in parts[:-1]:
            owner = owner._modules[part]
        leaf = parts[-1]
        if leaf in owner._parameters:
            old = owner._parameters[leaf]
            owner._parameters[leaf] = nn.Parameter(
                tensor, requires_grad=old.requires_grad
            )
        elif leaf in owner._buffers:
            owner._buffers[leaf] = tensor
        else:
            raise RuntimeError(f"Checkpoint target is not a parameter/buffer: {key}")

    if any(parameter.device.type == "meta" for parameter in model.parameters()):
        raise RuntimeError("Model still contains meta parameters after checkpoint load")
    return cast_count


def _camera_trunk_depth(state_dict):
    """Return the contiguous CameraHead trunk depth encoded by a state dict."""
    prefix = "camera_head.trunk."
    indices = {
        int(key[len(prefix):].split(".", 1)[0])
        for key in state_dict
        if key.startswith(prefix) and key[len(prefix):].split(".", 1)[0].isdigit()
    }
    if not indices or indices != set(range(max(indices) + 1)):
        raise RuntimeError("Checkpoint CameraHead trunk blocks are missing or noncontiguous")
    return len(indices)


def _validate_deployment_metadata(metadata, camera_trunk_depth):
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        raise TypeError("deployment metadata must be a dictionary")
    if metadata.get("aggregator_storage_dtype") == "bfloat16" or metadata.get(
        "aggregator_buffer_storage_dtype"
    ) == "bfloat16":
        raise RuntimeError(
            "Rejected legacy compact checkpoint: Aggregator buffers were cast to BF16"
        )
    if metadata.get("camera_head_trunk_depth") != camera_trunk_depth:
        raise RuntimeError("deployment metadata disagrees with checkpoint structure")


def _validate_checkpoint_digest(expected_sha256):
    if expected_sha256 is None:
        return None
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("--checkpoint_sha256 must be a lowercase SHA-256 digest")
    return expected_sha256


def _open_checkpoint(checkpoint_path):
    """Open one regular checkpoint fd without following its final component."""
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(checkpoint_path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("checkpoint must be a regular file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _hash_open_checkpoint(stream, expected_sha256):
    """Hash ``stream`` and rewind the same fd that torch.load will consume."""
    expected_sha256 = _validate_checkpoint_digest(expected_sha256)
    if expected_sha256 is None:
        stream.seek(0)
        return None
    digest = hashlib.sha256()
    while True:
        block = stream.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            "checkpoint digest mismatch: expected "
            f"{expected_sha256}, got {actual}"
        )
    stream.seek(0)
    return actual


def _verify_checkpoint_digest(checkpoint_path, expected_sha256):
    """Compatibility helper; production load hashes and loads one shared fd."""
    with _open_checkpoint(checkpoint_path) as stream:
        return _hash_open_checkpoint(stream, expected_sha256)


def load_model(
    checkpoint_path,
    device,
    expected_sha256=None,
    aggregator_frame_chunk_size=50,
    attention_backend="auto",
):
    """Load one weight copy; bf16 aggregator parameters and fp32 heads."""
    if attention_backend not in ("auto", "flash_attn_v1"):
        raise ValueError("attention_backend must be auto or flash_attn_v1")
    if attention_backend == "flash_attn_v1":
        from vggt_omega.models.layers.flash_attention import extension
        extension()  # Fail before loading the checkpoint if the optional library is unavailable.
    if (
        not isinstance(aggregator_frame_chunk_size, int)
        or isinstance(aggregator_frame_chunk_size, bool)
        or aggregator_frame_chunk_size < 0
    ):
        raise ValueError("aggregator frame chunk size must be a nonnegative integer")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run VGGT-Omega.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path} (single-copy CPU) ...")
    with _open_checkpoint(checkpoint_path) as checkpoint_stream:
        checkpoint_before = os.fstat(checkpoint_stream.fileno())
        _hash_open_checkpoint(checkpoint_stream, expected_sha256)
        checkpoint_payload = torch.load(
            checkpoint_stream, map_location="cpu", weights_only=False
        )
        checkpoint_after = os.fstat(checkpoint_stream.fileno())
        before_fingerprint = (
            checkpoint_before.st_dev,
            checkpoint_before.st_ino,
            checkpoint_before.st_size,
            checkpoint_before.st_mtime_ns,
            checkpoint_before.st_ctime_ns,
        )
        after_fingerprint = (
            checkpoint_after.st_dev,
            checkpoint_after.st_ino,
            checkpoint_after.st_size,
            checkpoint_after.st_mtime_ns,
            checkpoint_after.st_ctime_ns,
        )
        if after_fingerprint != before_fingerprint:
            del checkpoint_payload
            raise RuntimeError("checkpoint changed while it was hashed and loaded")
    if (
        isinstance(checkpoint_payload, dict)
        and isinstance(checkpoint_payload.get("model"), dict)
    ):
        state_dict = checkpoint_payload["model"]
        deployment_metadata = checkpoint_payload.get("deployment")
    else:
        state_dict = checkpoint_payload
        deployment_metadata = None
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint must contain a model state dict")

    with torch.device("meta"):
        model = VGGTOmega().eval()
        camera_trunk_depth = _camera_trunk_depth(state_dict)
        if camera_trunk_depth == 2:
            model.camera_head.trunk = nn.ModuleList(list(model.camera_head.trunk[:2]))
        elif camera_trunk_depth != 4:
            raise RuntimeError(
                f"Unsupported CameraHead trunk depth: {camera_trunk_depth}"
            )
    _validate_deployment_metadata(deployment_metadata, camera_trunk_depth)
    model.aggregator.frame_chunk_size = aggregator_frame_chunk_size or None
    from vggt_omega.models.layers.attention import SelfAttention
    for module in model.modules():
        if isinstance(module, SelfAttention):
            module.attention_backend = attention_backend
    model.attention_backend = attention_backend

    target_dtype = amp_dtype()
    n_cast = _assign_state_dict(model, state_dict, target_dtype)
    del state_dict
    del checkpoint_payload
    gc.collect()

    # Parameters are already in their final dtype, so this creates only the
    # GPU copy. Buffers such as _resnet_mean/_std remain fp32.
    model = model.to(device).eval()
    # Return the released CPU checkpoint/model pages to Jetson's unified RAM.
    ctypes.CDLL(None).malloc_trim(0)
    print(
        f"Model loaded; aggregator cast to {target_dtype} ({n_cast} param tensors). "
        f"Heads kept fp32; CameraHead blocks={camera_trunk_depth}; "
        f"Aggregator frame chunk={model.aggregator.frame_chunk_size or 'disabled'}; "
        f"attention backend={attention_backend}."
    )
    return model


def unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic):
    """Depth -> world point map (mirrors demo_gradio.run_model)."""
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    # Broadcast float32 coordinate vectors instead of materializing two int64
    # N*H*W grids and promoting the entire calculation to float64.
    x = np.arange(width, dtype=np.float32)[None, None, :]
    y = np.arange(height, dtype=np.float32)[None, :, None]

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3].astype(np.float32, copy=False)
    translation = extrinsic[:, :3, 3].astype(np.float32, copy=False)
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    ).astype(np.float32, copy=False)


def run_inference(model, images, device):
    """Run GPU inference and copy only required outputs back to CPU."""
    num_frames = images.shape[0]
    print(f"Running VGGT-Omega on {num_frames} frames ...")
    with torch.inference_mode():
        predictions = model(images.to(device))

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )

    keep_keys = ("depth", "depth_conf")
    predictions_np = {}
    for key in keep_keys:
        value = predictions[key].detach().float().cpu().numpy()
        if value.shape[0] == 1:
            value = value[0]
        predictions_np[key] = value
    predictions_np["extrinsic"] = extrinsic.detach().float().cpu().numpy()
    if predictions_np["extrinsic"].shape[0] == 1:
        predictions_np["extrinsic"] = predictions_np["extrinsic"][0]
    predictions_np["intrinsic"] = intrinsic.detach().float().cpu().numpy()
    if predictions_np["intrinsic"].shape[0] == 1:
        predictions_np["intrinsic"] = predictions_np["intrinsic"][0]

    del predictions, extrinsic, intrinsic
    return predictions_np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="VGGT-Omega reconstruction (Drone_3D S380)")
    parser.add_argument("--images_dir", type=str, default="colmap/data/images",
                        help="Directory of input images (used when --input_image is absent)")
    parser.add_argument(
        "--input_image",
        action="append",
        default=None,
        help="Absolute input image path; repeat to provide the exact ordered input list",
    )
    parser.add_argument("--output_dir", type=str, default="colmap/data_vggt_omega",
                        help="Fresh standalone output path, or existing supervisor staging path")
    parser.add_argument(
        "--supervisor_staging",
        action="store_true",
        help="Write directly into an existing canonical directory owned by the supervisor",
    )
    parser.add_argument("--checkpoint", type=str,
                        default="Data/Model/vggt_omega_1b_512.pt",
                        help="VGGT-Omega checkpoint path")
    parser.add_argument("--checkpoint_sha256", type=str, default=None,
                        help="Optional exact supervisor-owned checkpoint digest; "
                             "verified immediately before model load")
    parser.add_argument("--image_resolution", type=int, default=0,
                        help="Input image resolution (0=auto-detect from VRAM)")
    parser.add_argument("--load_mode", type=str, default="max_size",
                        choices=["max_size", "balanced"],
                        help="'max_size' (default) resizes the longest side to the "
                             "selected resolution and uses fewer patch tokens. "
                             "'balanced' keeps token count near resolution**2.")
    parser.add_argument("--frame_group_size", type=int, default=0,
                        help="Max frames per forward pass (0=auto). Directory input is "
                             "uniformly subsampled; an oversized explicit list is rejected.")
    parser.add_argument(
        "--aggregator_frame_chunk_size",
        type=int,
        default=50,
        help="Frames per independent Aggregator frame block (0 disables chunking)",
    )
    parser.add_argument("--attention_backend", choices=("auto", "flash_attn_v1"), default="auto",
                        help="Optional fused inference implementation; does not change checkpoint or head precision")
    parser.add_argument("--conf_thres", type=float, default=50.0,
                        help="Confidence percentile threshold for the default GLB (0-100)")
    parser.add_argument("--max_points_k", type=int, default=5000,
                        help="Max points (thousands) in the default GLB")
    parser.add_argument("--no_cam", action="store_true",
                        help="Do not show camera frustums in the default GLB")
    parser.add_argument("--no_glb", action="store_true",
                        help="Skip GLB export (only save predictions.npz)")
    return parser.parse_args(argv)


def _validate_ordered_input_images(input_images):
    """Return a validated explicit image list without changing its order."""
    if not input_images:
        raise ValueError("--input_image requires at least one nonempty path")

    ordered_paths = []
    seen_paths = set()
    seen_basenames = set()
    for raw_path in input_images:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("--input_image paths must be nonempty strings")
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError(f"--input_image path must be absolute: {raw_path!r}")
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported --input_image extension: {raw_path!r}")

        normalized = os.path.normcase(os.path.normpath(raw_path))
        if normalized in seen_paths:
            raise ValueError(f"Duplicate --input_image path: {raw_path!r}")
        basename = path.name
        if "\n" in basename or "\r" in basename:
            raise ValueError("Input image basenames must not contain newline characters")
        if basename in seen_basenames:
            raise ValueError(f"Duplicate --input_image basename: {basename!r}")

        seen_paths.add(normalized)
        seen_basenames.add(basename)
        ordered_paths.append(raw_path)
    return ordered_paths


def _select_input_images(images_dir, input_images):
    """Select explicit inputs in caller order, or retain legacy directory discovery."""
    if input_images is not None:
        return _validate_ordered_input_images(input_images)
    return sorted(
        path
        for path in glob.glob(os.path.join(images_dir, "*"))
        if os.path.splitext(path)[1].lower() in IMAGE_EXTS
    )


def _create_output_transaction(output_dir):
    """Create a unique sibling staging directory for one immutable generation."""
    final_dir = Path(output_dir).expanduser().resolve(strict=False)
    if os.path.lexists(final_dir):
        raise FileExistsError(
            f"Output path already exists: {final_dir}; use a fresh output directory"
        )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_dir.name}.partial-",
            dir=final_dir.parent,
        )
    )
    return final_dir, staging_dir


def _run_reconstruction(args, output_dir, model=None):
    """Write and validate one reconstruction generation in ``output_dir``.

    ``model`` 为空时按 one-shot 语义自行加载 checkpoint；warm worker 传入
    已加载模型以跨窗口复用（本函数只持有局部引用，不会释放共享模型）。
    """
    device = "cuda"

    image_paths = _select_input_images(args.images_dir, args.input_image)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")
    if args.input_image is not None:
        print(f"Using {len(image_paths)} explicitly ordered input images")
    else:
        print(f"Found {len(image_paths)} images in {args.images_dir}")

    resolution, max_frames, total_mem_mb = detect_config(len(image_paths))
    if args.image_resolution > 0:
        resolution = args.image_resolution
    if args.frame_group_size > 0:
        max_frames = args.frame_group_size
    print(f"GPU total memory: {total_mem_mb:.0f} MB | resolution={resolution} | "
          f"frame_group_size={max_frames}")

    # Legacy directory discovery uniformly subsamples to frame_group_size. An
    # explicit list is a supervisor boundary and must never be silently changed.
    if len(image_paths) > max_frames:
        if args.input_image is not None:
            raise ValueError(
                f"Explicit --input_image list has {len(image_paths)} frames, "
                f"exceeding frame_group_size={max_frames}; exact input lists "
                "are not subsampled"
            )
        selected = np.linspace(0, len(image_paths) - 1, max_frames).round().astype(int)
        selected = np.unique(selected)
        image_paths = [image_paths[i] for i in selected]
        print(f"Input exceeds frame_group_size; uniformly sampled {len(image_paths)} frames")
    else:
        print(f"Using all {len(image_paths)} frames for inference")

    image_basenames = [os.path.basename(path) for path in image_paths]
    if any("\n" in name or "\r" in name for name in image_basenames):
        raise ValueError("Input image basenames must not contain newline characters")
    manifest_path = Path(output_dir) / "manifest.txt"
    manifest_path.write_text("\n".join(image_basenames) + "\n", encoding="utf-8")

    # Preprocess on CPU to keep GPU free for the model.
    images = load_and_preprocess_images(
        image_paths, mode=args.load_mode, image_resolution=resolution
    )
    print(f"Preprocessed images shape: {tuple(images.shape)} (mode={args.load_mode})")

    torch.cuda.reset_peak_memory_stats()

    if model is None:
        model = load_model(
            args.checkpoint,
            device,
            expected_sha256=args.checkpoint_sha256,
            aggregator_frame_chunk_size=args.aggregator_frame_chunk_size,
            attention_backend=getattr(args, "attention_backend", "auto"),
        )
    predictions = run_inference(model, images, device)

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    # The model is the largest live object. For one-shot mode release it before
    # CPU unprojection, NPZ serialization, or optional GLB construction. Warm
    # workers keep the shared model alive; deleting the local reference is still
    # harmless.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ``predictions['images']`` is exactly the GPU copy of this CPU tensor.
    # Reuse the original to avoid a needless device-to-host transfer.
    if isinstance(images, torch.Tensor):
        predictions["images"] = images.detach().float().cpu().numpy()
    else:
        predictions["images"] = np.asarray(images, dtype=np.float32)

    if "world_points_from_depth" not in predictions:
        predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
            predictions["depth"],
            predictions["extrinsic"],
            predictions["intrinsic"],
        )
    del images
    gc.collect()

    print(
        "VGGT-Omega GPU stages completed; "
        f"peak GPU memory: {peak_mem_mb:.0f} MB; artifact validation pending"
    )

    pred_path = Path(output_dir) / "predictions.npz"
    np.savez(pred_path, **predictions)
    print(f"Saved predictions to {pred_path}")

    if not args.no_glb:
        # Heavy visualization dependencies are unnecessary in the production
        # no-GLB profile and remain out of its import/RSS path.
        from visual_util import predictions_to_glb

        print("Building default GLB scene ...")
        scene = predictions_to_glb(
            predictions,
            conf_thres=max(3.0, float(args.conf_thres)),
            show_cam=not args.no_cam,
            target_dir=str(output_dir),
            max_points=int(args.max_points_k) * 1000,
        )
        glb_path = Path(output_dir) / "scene.glb"
        scene.export(file_obj=glb_path)
        print(f"GLB exported to {glb_path}")
        del scene

    # Validation reloads the artifacts from disk. Release large CPU objects first.
    del predictions
    gc.collect()
    return validate_result_directory(
        output_dir,
        require_glb=not args.no_glb,
        expected_image_basenames=image_basenames,
        max_frames=max_frames,
        max_pixels=2 * resolution * resolution,
    )


def _rename_noreplace(source, destination):
    """Atomically publish a directory without replacing an existing path."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("Atomic no-replace publication requires Linux renameat2")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _supervisor_output_directory(output_dir):
    """Require an existing absolute, canonical, nonsymlink supervisor directory."""
    requested = Path(output_dir)
    if not requested.is_absolute():
        raise ValueError(
            "--supervisor_staging requires an absolute canonical --output_dir"
        )
    try:
        canonical = requested.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError, OSError, RuntimeError) as exc:
        raise ValueError(
            "--supervisor_staging requires an existing real --output_dir: "
            f"{requested}"
        ) from exc
    if requested != canonical or requested.is_symlink() or not requested.is_dir():
        raise ValueError(
            "--supervisor_staging requires an existing canonical real directory: "
            f"{requested}"
        )
    return requested


def _run_main(args):
    """Run one worker invocation while keeping supervisor ownership boundaries explicit."""
    if args.supervisor_staging:
        output_dir = _supervisor_output_directory(args.output_dir)
        validation = _run_reconstruction(args, output_dir)
        print(
            "Validated output in supervisor staging: "
            f"{validation.frame_count} frames at "
            f"{validation.width}x{validation.height} -> {output_dir}"
        )
        return validation

    final_dir, staging_dir = _create_output_transaction(args.output_dir)
    try:
        validation = _run_reconstruction(args, staging_dir)
        _rename_noreplace(staging_dir, final_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(
        "Validated and published output: "
        f"{validation.frame_count} frames at "
        f"{validation.width}x{validation.height} -> {final_dir}"
    )
    return validation


def main(argv=None):
    args = parse_args(argv)
    resource_root = os.environ.get("DRONE3D_RUNTIME_PYTHON_ROOT")
    if resource_root is None:
        resource_root = str(Path(__file__).resolve().parents[2] / "ROS_Compatibility_layer/src/drone3d_runtime/src")
    if resource_root not in sys.path:
        sys.path.append(resource_root)
    from drone3d_resources import gpu_lease
    with gpu_lease():
        _run_main(args)
    print("Done.")


if __name__ == "__main__":
    main()
