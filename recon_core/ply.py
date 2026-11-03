"""Binary little-endian PLY point cloud writer (XYZ float32, RGB uint8)."""
import numpy as np


def write_ply(path, xyz, rgb, comment="meters Z-up"):
    """``rgb`` in [0, 1]."""
    xyz, rgb = np.asarray(xyz), np.asarray(rgb)
    if xyz.shape != rgb.shape or xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("PLY requires finite Nx3 XYZ and matching RGB")
    data = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for i, name in enumerate(("x", "y", "z")):
        data[name] = xyz[:, i]
    for i, name in enumerate(("red", "green", "blue")):
        data[name] = np.clip(np.rint(rgb[:, i] * 255), 0, 255).astype("u1")
    header = "ply\nformat binary_little_endian 1.0\ncomment %s\nelement vertex %d\n" % (comment, len(xyz))
    header += "".join("property float %s\n" % k for k in ("x", "y", "z"))
    header += "".join("property uchar %s\n" % k for k in ("red", "green", "blue")) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)
