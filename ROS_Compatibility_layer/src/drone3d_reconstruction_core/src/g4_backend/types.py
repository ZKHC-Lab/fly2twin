"""Validated, ROS-independent contracts for the frozen graph backend."""
from dataclasses import dataclass, field
from typing import Hashable, Mapping, Optional, Sequence

import numpy as np

# Incremental 16-frame identity. The method number of already published
# revisions is part of their artifact schema and keeps its original value.
METHOD_ID = "legacy_g4_20260916"

# Identity of the ported long-window grouping (first 100, then shared 20 + new
# 100). It is a separate method identity: equal class names alone never make two
# groupings the same method, and every run/revision records which one produced it.
WINDOWED_METHOD_ID = "windowed_sim3_graph_20261008"

# Threshold provenance labels. A recomputed threshold and a threshold reused
# from an earlier registered table are different inputs and stay distinguishable.
THRESHOLD_RECOMPUTED = "recomputed_from_mask_intersection"
THRESHOLD_RECORDED = "recorded_table"


def edge_key(i, j) -> str:
    """Stable key for one ordered edge, usable in JSON tables."""
    return f"{i}|{j}"


def canonical_identifier(value) -> str:
    """Collapse the identifier spellings one recorded edge may use.

    A registered table names windows the way the experiment did (``F000``) while
    the ROS records carry integer window indices (``0``). Both must select the
    same record: dropping a leading ``F``/``f`` and leading zeros is the only
    normalization applied, and it never merges two different windows.
    """
    text = str(value).strip()
    core = text[1:] if len(text) > 1 and text[0] in ("F", "f") and text[1:].isdigit() else text
    return str(int(core)) if core.isdigit() else text.lower()


@dataclass(frozen=True)
class ThresholdPolicy:
    """Per-edge distance threshold provenance.

    ``recorded`` maps an ordered edge key to a threshold in the source window's
    normalized length units. Lookup is spelling-insensitive (see
    ``canonical_identifier``) so a registered table keyed by ``F000|F001``
    answers an integer ``(0, 1)`` query. A recorded value is reused when present;
    otherwise the threshold is recomputed from this run's mask intersection and
    labelled as recomputed. The policy never invents a value and never hides
    which of the two happened.
    """

    recorded: Mapping = field(default_factory=dict)
    source: str = THRESHOLD_RECORDED
    skipped_records: int = 0

    def __post_init__(self):
        values, index = {}, {}
        for key, value in dict(self.recorded).items():
            if isinstance(key, (tuple, list)):
                if len(key) != 2:
                    raise ValueError("threshold edge keys must have two components")
                parts = (key[0], key[1])
            else:
                parts = str(key).split("|")
                if len(parts) != 2:
                    raise ValueError(f"threshold edge key {key!r} must be 'i|j'")
            number = float(value)
            if not np.isfinite(number) or number <= 0:
                raise ValueError(f"recorded threshold for {key} must be finite and positive")
            canonical = "|".join(canonical_identifier(part) for part in parts)
            if canonical in index and index[canonical] != number:
                # 同一物理边的两种拼写给了不同阈值：不猜、不 last-wins。
                raise ValueError(
                    "conflicting recorded thresholds for the same edge "
                    f"{canonical}: {index[canonical]} vs {number}"
                )
            values[edge_key(*parts)] = number
            index[canonical] = number
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("threshold policy source must be a nonempty string")
        if isinstance(self.skipped_records, bool) or not isinstance(self.skipped_records, int) or self.skipped_records < 0:
            raise ValueError("skipped_records must be a nonnegative integer")
        object.__setattr__(self, "recorded", values)
        object.__setattr__(self, "_index", index)

    def __len__(self):
        return len(self.recorded)

    @property
    def canonical(self):
        """按规范边键索引的有效策略；同拼写别名已合并，冲突已在构造时拒绝。"""
        return dict(self._index)

    def lookup(self, i, j):
        """Return (value, source) for one edge; value is None when absent."""
        value = self._index.get("|".join(canonical_identifier(p) for p in (i, j)))
        return (None, THRESHOLD_RECOMPUTED) if value is None else (value, self.source)

    @classmethod
    def from_records(cls, records, *, i_field="i", j_field="j", value_field="threshold",
                     source=THRESHOLD_RECORDED):
        """Build a policy from records or from a plain edge-key mapping.

        A record without a usable value is skipped, not treated as zero: the
        frozen experiment recomputed exactly the edges whose recorded threshold
        was missing.
        """
        if records is None:
            return cls({}, source)
        if isinstance(records, Mapping):
            return cls(dict(records), source)
        values, skipped = {}, 0
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("threshold records must be objects")
            value = record.get(value_field)
            if value is None:
                skipped += 1
                continue
            key = edge_key(record[i_field], record[j_field])
            if key in values and float(values[key]) != float(value):
                # 组 dict 时先吞掉冲突会让"哪个值生效"取决于记录顺序。
                raise ValueError(f"conflicting recorded thresholds for edge {key}")
            values[key] = value
        return cls(values, source, skipped)


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


@dataclass(frozen=True)
class RelativeEdge:
    """measurement maps window j to i; covariance is normalized right tangent."""
    i: Hashable
    j: Hashable
    measurement: Sim3
    covariance: np.ndarray
    length_source: float
    point_count: int = 0

    def __post_init__(self):
        if self.i == self.j:
            raise ValueError("self edges are invalid")
        if not isinstance(self.measurement, Sim3):
            raise ValueError("measurement must be Sim3")
        C = array(self.covariance, (7, 7), "covariance")
        if not np.allclose(C, C.T, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        np.linalg.cholesky(C)
        if not np.isfinite(self.length_source) or self.length_source <= 0:
            raise ValueError("length_source must be finite and positive")
        object.__setattr__(self, "covariance", C)


@dataclass(frozen=True)
class LocalizationAnchor:
    window_id: Hashable
    local_center: np.ndarray
    reference_center: np.ndarray
    weight: float = 1.0
    frame_id: Optional[Hashable] = None
    block: int = 0

    def __post_init__(self):
        object.__setattr__(self, "local_center", array(self.local_center, (3,), "local_center"))
        object.__setattr__(self, "reference_center", array(self.reference_center, (3,), "reference_center"))
        if not np.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("anchor weight must be finite and positive")


@dataclass(frozen=True)
class RegistrationConfig:
    seed: int = 20260915
    confidence_percentile: float = 50.0
    edge_relative_threshold: float = .03
    grid_rows: int = 4
    grid_columns: int = 4
    points_per_cell: int = 32
    exclusion_model_pixels: float = 3.0
    ransac_trials: int = 1000
    refine_iterations: int = 5
    min_fit_points: int = 30
    min_consensus: float = .3
    support_ratio: float = 1e-3
    threshold_depth_fraction: float = .03
    # "skip" keeps trying hypotheses after a degenerate three-point fit (the
    # incremental default); "abort" reproduces the frozen experiment, which let
    # the failure end the edge.
    ransac_degenerate: str = "skip"
    bootstrap_replicates: int = 200
    bootstrap_minimum_valid: int = 160
    jacobian_relative_sv: float = 1e-8
    covariance_sigma_floor: float = .001

    def __post_init__(self):
        if not 0 <= self.confidence_percentile <= 100:
            raise ValueError("confidence_percentile outside [0,100]")
        for key in ("grid_rows", "grid_columns", "points_per_cell", "ransac_trials", "refine_iterations", "bootstrap_replicates", "bootstrap_minimum_valid"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive integer")
        if self.min_fit_points < 3 or not 0 < self.min_consensus <= 1 or self.bootstrap_minimum_valid > self.bootstrap_replicates:
            raise ValueError("invalid registration support limits")
        for key in ("edge_relative_threshold", "support_ratio", "threshold_depth_fraction", "jacobian_relative_sv", "covariance_sigma_floor"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not np.isfinite(self.exclusion_model_pixels) or self.exclusion_model_pixels < 0:
            raise ValueError("exclusion_model_pixels must be nonnegative")
        if self.ransac_degenerate not in ("skip", "abort"):
            raise ValueError("ransac_degenerate must be 'skip' or 'abort'")


@dataclass(frozen=True)
class GraphConfig:
    position_sigma_m: float = .1
    huber: float = 2.5
    initialization_min_reference_rms_m: float = .01
    support_ratio: float = 1e-3
    max_nfev: int = 200
    tolerance: float = 1e-8
    logscale_increment_bound: float = 12.0
    jacobian_relative_sv: float = 1e-8
    time_limit_seconds: Optional[float] = None

    def __post_init__(self):
        for key in ("position_sigma_m", "huber", "initialization_min_reference_rms_m", "support_ratio", "tolerance", "logscale_increment_bound", "jacobian_relative_sv"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not isinstance(self.max_nfev, int) or self.max_nfev < 1:
            raise ValueError("max_nfev must be positive")
        if self.time_limit_seconds is not None and (not np.isfinite(self.time_limit_seconds) or self.time_limit_seconds <= 0):
            raise ValueError("time_limit_seconds must be positive or None")


@dataclass(frozen=True)
class LocalizationConfig:
    seed: int = 20260915
    stride_seconds: float = 1.0
    block_seconds: float = 20.0
    fit_fraction: float = .8
    boundary_guard_seconds: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.stride_seconds) or self.stride_seconds <= 0 or not np.isfinite(self.block_seconds) or self.block_seconds <= 0:
            raise ValueError("localization time intervals must be positive")
        if not 0 <= self.fit_fraction <= 1 or not 0 <= self.boundary_guard_seconds < self.block_seconds / 2:
            raise ValueError("invalid localization partition configuration")


@dataclass
class SharedPairs:
    i: Hashable
    j: Hashable
    source: np.ndarray
    target: np.ndarray
    source_depth: np.ndarray
    target_depth: np.ndarray
    split: np.ndarray
    cluster: np.ndarray
    shared_frame: np.ndarray
    model_pixels: np.ndarray
    frame_ids: Sequence[Hashable]
    threshold: float
    threshold_source: str = THRESHOLD_RECOMPUTED
    cells_per_frame: int = 16
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.source)
        for key in ("source", "target"):
            setattr(self, key, array(getattr(self, key), (n, 3), key))
        for key in ("source_depth", "target_depth"):
            z = array(getattr(self, key), (n,), key, preserve_float32=True)
            if np.any(z <= 0):
                raise ValueError(f"{key} must be positive")
            setattr(self, key, z)
        self.model_pixels = array(self.model_pixels, (n, 2), "model_pixels")
        for key in ("split", "cluster", "shared_frame"):
            z = np.asarray(getattr(self, key))
            if z.shape != (n,) or not np.issubdtype(z.dtype, np.integer):
                raise ValueError(f"{key} must be an integer vector")
            setattr(self, key, z.copy())
        if not np.isin(self.split, [0, 1, 2]).all() or len(self.frame_ids) != n:
            raise ValueError("invalid split or frame identifiers")
        if not isinstance(self.threshold_source, str) or not self.threshold_source:
            raise ValueError("threshold_source must be a nonempty string")
        if n and (not np.isfinite(self.threshold) or self.threshold <= 0):
            raise ValueError("threshold must be positive for nonempty pairs")


@dataclass
class RegistrationResult:
    edge: Optional[RelativeEdge]
    diagnostics: dict


@dataclass
class GraphResult:
    states: Mapping[Hashable, Sim3]
    diagnostics: dict
    method_id: str = METHOD_ID
    geometry_accepted: bool = False


@dataclass
class LocalizationSelection:
    anchors: Sequence[LocalizationAnchor]
    frame_ids: Sequence[Hashable]
    owner: Sequence[Hashable]
    reference_centers: np.ndarray
    valid: np.ndarray
    fit: np.ndarray
    heldout: np.ndarray
    blocks: np.ndarray
    selected: np.ndarray
    block_fit: Mapping[int, bool]
