"""Validated, ROS-independent contracts for the per-frame placement backend."""
from dataclasses import dataclass, field
from typing import Hashable, Sequence

import numpy as np


def array(value, shape=None, name="array", finite=True, preserve_float32=False):
    dtype = np.float32 if preserve_float32 and np.asarray(value).dtype == np.float32 else np.float64
    out = np.asarray(value, dtype=dtype).copy()
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {out.shape}")
    if finite and not np.isfinite(out).all():
        raise ValueError(f"{name}: non-finite values")
    out.setflags(write=False)
    return out


def rotation(value, name="rotation"):
    out = array(value, (3, 3), name)
    if not np.allclose(out.T @ out, np.eye(3), atol=2e-5) or not np.isclose(np.linalg.det(out), 1, atol=2e-5):
        raise ValueError(f"{name}: expected a proper rotation")
    return out


@dataclass(frozen=True)
class Sim3:
    """Column-vector transform: target = scale * rotation @ source + translation."""
    scale: float = 1.0
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Sim3.scale must be finite and positive")
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "rotation", rotation(self.rotation))
        object.__setattr__(self, "translation", array(self.translation, (3,), "translation"))

    def to_dict(self):
        return {"scale": self.scale, "rotation": self.rotation.tolist(), "translation": self.translation.tolist()}

    @classmethod
    def from_dict(cls, value):
        return cls(value["scale"], value["rotation"], value["translation"])


@dataclass(frozen=True)
class WindowGeometry:
    """Immutable model output in one local gauge. K uses model pixel centers."""
    window_id: Hashable
    frame_ids: Sequence[Hashable]
    timestamps_ns: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    intrinsics: np.ndarray
    rotations: np.ndarray
    centers: np.ndarray
    shared_prefix: int = 0
    rgb_hashes: Sequence[str] = ()
    gauge: float = 1.0

    def __post_init__(self):
        ids = tuple(self.frame_ids)
        n = len(ids)
        if not n or len(set(ids)) != n:
            raise ValueError("frame_ids must be nonempty and unique within a window")
        hash(self.window_id)
        ts = np.asarray(self.timestamps_ns)
        if ts.shape != (n,) or not np.issubdtype(ts.dtype, np.integer):
            raise ValueError("timestamps_ns must be an integer vector, one per frame")
        ts = ts.astype(np.int64, copy=True)
        if np.any(np.diff(ts) <= 0):
            raise ValueError("window timestamps must increase strictly")
        ts.setflags(write=False)
        depth = array(self.depth, name="depth", finite=False, preserve_float32=True)
        if depth.ndim != 3 or depth.shape[0] != n or min(depth.shape[1:]) < 2:
            raise ValueError("depth must have shape (frames, height, width)")
        conf = array(self.confidence, depth.shape, "confidence", preserve_float32=True)
        K = array(self.intrinsics, (n, 3, 3), "intrinsics")
        if np.any(K[:, 0, 0] <= 0) or np.any(K[:, 1, 1] <= 0) or not np.allclose(K[:, 2], [0, 0, 1]):
            raise ValueError("intrinsics must have positive focal lengths and pinhole last row")
        if np.any(np.abs(np.linalg.det(K)) < 1e-12):
            raise ValueError("singular intrinsics")
        R = array(self.rotations, (n, 3, 3), "rotations")
        for r in R:
            rotation(r)
        c = array(self.centers, (n, 3), "centers")
        if not isinstance(self.shared_prefix, (int, np.integer)) or not 0 <= self.shared_prefix < n:
            raise ValueError("shared_prefix must leave at least one new frame")
        hashes = tuple(self.rgb_hashes)
        if hashes and (len(hashes) != n or any(not h for h in hashes)):
            raise ValueError("rgb_hashes must be empty or one nonempty hash per frame")
        if not np.isfinite(self.gauge) or self.gauge <= 0:
            raise ValueError("gauge must be finite and positive")
        for key, val in (("frame_ids", ids), ("timestamps_ns", ts), ("depth", depth), ("confidence", conf), ("intrinsics", K), ("rotations", R), ("centers", c), ("rgb_hashes", hashes)):
            object.__setattr__(self, key, val)
