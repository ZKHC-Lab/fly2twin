"""Localization interpolation and the frozen G4 temporal sampling policy."""
import hashlib
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .types import LocalizationAnchor, LocalizationConfig, LocalizationSelection, array


def select_localization_anchors(windows, reference_timestamps_ns, reference_centers,
                                reference_rotations=None, body_from_camera=None, config=None):
    """Build camera-center factors, excluding duplicate shared-prefix frames.

    References describe camera centers unless body_from_camera is supplied.
    body_from_camera is a rigid 4x4 camera-to-body transform. Body rotations
    are then required. All clocks must already share the same timestamp domain.
    G4 uses 1 Hz nearest RGB samples, 20 s hash-partitioned blocks and a 1 s
    boundary guard. Holdout is a diagnostic split, not a geometry truth source.
    """
    config = config or LocalizationConfig()
    windows = list(windows)
    if not windows or len({w.window_id for w in windows}) != len(windows):
        raise ValueError("windows must be nonempty with unique IDs")
    records = []
    for window in windows:
        for k in range(window.shared_prefix, len(window.frame_ids)):
            records.append((int(window.timestamps_ns[k]), window.frame_ids[k], window.window_id, window.centers[k]))
    records.sort(key=lambda x: x[0])
    wanted = np.array([r[0] for r in records], dtype=np.int64)
    ids = tuple(r[1] for r in records)
    if len(set(ids)) != len(ids) or np.any(np.diff(wanted) <= 0):
        raise ValueError("new frames must be globally unique in identity and timestamp")
    ts = np.asarray(reference_timestamps_ns)
    if ts.ndim != 1 or len(ts) < 2 or not np.issubdtype(ts.dtype, np.integer) or np.any(np.diff(ts) <= 0):
        raise ValueError("reference timestamps must be strictly increasing integer nanoseconds")
    ts = ts.astype(np.int64)
    ref = array(reference_centers, (len(ts), 3), "reference_centers")
    valid = (wanted >= ts[0]) & (wanted <= ts[-1])
    t, tq = (ts-ts[0])/1e9, (wanted[valid]-ts[0])/1e9
    positions = np.column_stack([np.interp(tq, t, ref[:, k]) for k in range(3)])
    if body_from_camera is not None:
        ext = array(body_from_camera, (4, 4), "body_from_camera")
        if not np.allclose(ext[3], [0, 0, 0, 1]):
            raise ValueError("body_from_camera must be homogeneous rigid transform")
        from .types import rotation
        rotation(ext[:3, :3])
        if reference_rotations is None:
            raise ValueError("body rotations are required for a camera extrinsic")
        rotations = array(reference_rotations, (len(ts), 3, 3), "reference_rotations")
        for R in rotations:
            rotation(R)
        if len(tq):
            rb = Slerp(t, Rotation.from_matrix(np.array(rotations, copy=True)))(tq).as_matrix()
            positions += np.einsum("nij,j->ni", rb, ext[:3, 3])
    centers = np.full((len(records), 3), np.nan)
    centers[valid] = positions
    elapsed = (wanted-wanted[0])/1e9
    blocks = (elapsed//config.block_seconds).astype(int)
    block_fit = {int(b): int(hashlib.sha256(f"{config.seed}|localization|{b}".encode()).hexdigest()[:16], 16)/2**64 < config.fit_fraction for b in np.unique(blocks)}
    fitblock = np.array([block_fit[int(b)] for b in blocks])
    phase = elapsed % config.block_seconds
    boundary = (phase >= config.boundary_guard_seconds) & (phase <= config.block_seconds-config.boundary_guard_seconds)
    chosen = [int(np.argmin(np.abs(elapsed-second))) for second in np.arange(0, np.floor(elapsed[-1])+config.stride_seconds, config.stride_seconds)]
    selected = np.zeros(len(records), bool)
    selected[np.unique(chosen)] = True
    fit = valid & fitblock & boundary & selected
    heldout = valid & ~fitblock
    counts = Counter(blocks[fit])
    anchors = [LocalizationAnchor(records[k][2], records[k][3], centers[k], 1/np.sqrt(counts[blocks[k]]), ids[k], int(blocks[k])) for k in np.flatnonzero(fit)]
    return LocalizationSelection(anchors, ids, tuple(r[2] for r in records), centers, valid, fit, heldout, blocks, selected, block_fit)
