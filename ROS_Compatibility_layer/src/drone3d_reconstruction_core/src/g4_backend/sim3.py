"""Sim(3) composition and application for window placements."""
import numpy as np
from .types import Sim3


def _algebra_result(scale, rotation, translation):
    """Algebra of already validated rotations avoids repeated SVD/shape checks."""
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(translation).all():
        raise ValueError("non-finite Sim3 algebra result")
    result = object.__new__(Sim3)
    rotation.setflags(write=False)
    translation.setflags(write=False)
    object.__setattr__(result, "scale", float(scale))
    object.__setattr__(result, "rotation", rotation)
    object.__setattr__(result, "translation", translation)
    return result


def apply(transform, points):
    return transform.scale * (np.asarray(points) @ transform.rotation.T) + transform.translation


def compose(a, b):
    return _algebra_result(a.scale*b.scale, a.rotation@b.rotation, a.scale*(a.rotation@b.translation)+a.translation)
