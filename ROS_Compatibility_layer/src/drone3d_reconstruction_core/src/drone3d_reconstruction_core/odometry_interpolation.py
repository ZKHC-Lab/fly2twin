"""Interpolate body position and orientation at an RGB header timestamp."""

from __future__ import annotations

import copy

import numpy as np


def stamp_ns(stamp):
    return int(stamp["sec"]) * 1_000_000_000 + int(stamp["nsec"])


def slerp_quaternion(first, second, fraction):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    for value in (first, second):
        if not np.isfinite(value).all() or np.linalg.norm(value) <= 0:
            raise ValueError("invalid quaternion for interpolation")
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(first @ second)
    if dot < 0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, 0.0, 1.0))
    # The normalized limit avoids dividing by sin(theta) near zero rotation.
    if dot > 1.0 - 1e-12:
        result = first + fraction * (second - first)
    else:
        theta = np.arccos(dot)
        result = (
            np.sin((1.0 - fraction) * theta) * first
            + np.sin(fraction * theta) * second
        ) / np.sin(theta)
    return result / np.linalg.norm(result)


def interpolate_payload(before, after, query_stamp, max_gap_ms):
    """Return derived odometry. Covariance and twist retain the earlier sample."""
    if (
        before["header"]["frame_id"] != after["header"]["frame_id"]
        or before["child_frame_id"] != after["child_frame_id"]
    ):
        raise ValueError("odometry coordinate frames changed across bracket")
    first_ns = stamp_ns(before["header"]["stamp"])
    last_ns = stamp_ns(after["header"]["stamp"])
    query_ns = stamp_ns(query_stamp)
    if not first_ns <= query_ns <= last_ns:
        raise ValueError("odometry extrapolation is forbidden")
    gap_ms = (last_ns - first_ns) / 1e6
    if gap_ms > max_gap_ms:
        raise ValueError("odometry interpolation bracket is too wide")
    fraction = (query_ns - first_ns) / (last_ns - first_ns) if last_ns > first_ns else 0.0
    result = copy.deepcopy(before)
    result["header"]["stamp"] = dict(query_stamp)
    for axis in "xyz":
        a = float(before["pose"]["position"][axis])
        b = float(after["pose"]["position"][axis])
        result["pose"]["position"][axis] = a + fraction * (b - a)
    quaternion = slerp_quaternion(
        [before["pose"]["orientation"][key] for key in "xyzw"],
        [after["pose"]["orientation"][key] for key in "xyzw"],
        fraction,
    )
    result["pose"]["orientation"] = dict(zip("xyzw", quaternion.tolist()))
    result["interpolation"] = {
        "method": "linear_position_slerp_orientation",
        "before_stamp": dict(before["header"]["stamp"]),
        "after_stamp": dict(after["header"]["stamp"]),
        "bracket_gap_ms": gap_ms,
        "fraction": fraction,
        "other_fields_source": "before",
    }
    return result
