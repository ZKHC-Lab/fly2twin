"""G4 Sim(3) algebra. Tangent order is translation, rotation, log scale."""
import numpy as np
from scipy.spatial.transform import Rotation
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


def identity():
    return Sim3()


def apply(transform, points):
    return transform.scale * (np.asarray(points) @ transform.rotation.T) + transform.translation


def compose(a, b):
    return _algebra_result(a.scale*b.scale, a.rotation@b.rotation, a.scale*(a.rotation@b.translation)+a.translation)


def inverse(t):
    return _algebra_result(1/t.scale, t.rotation.T, -t.rotation.T@t.translation/t.scale)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.]])


def v_matrix(w, sigma):
    a = skew(w) + sigma*np.eye(3)
    v, term = np.eye(3), np.eye(3)
    for k in range(1, 48):
        term = term@a/(k+1)
        v += term
        if np.max(np.abs(term)) < 1e-16:
            break
    return v


def exp(v):
    v = np.asarray(v, dtype=float)
    if v.shape != (7,) or not np.isfinite(v).all():
        raise ValueError("Sim3 exponential expects seven finite values")
    return _algebra_result(np.exp(v[6]), Rotation.from_rotvec(v[3:6]).as_matrix(), v_matrix(v[3:6], v[6])@v[:3])


def log(t):
    # SciPy 1.10 (ROS Noetic/Python 3.8) requires a writable input buffer.
    w, sigma = Rotation.from_matrix(np.array(t.rotation, copy=True)).as_rotvec(), np.log(t.scale)
    return np.r_[np.linalg.solve(v_matrix(w, sigma), t.translation), w, sigma]


def edge_error(z, a, b):
    return log(compose(inverse(z), compose(inverse(a), b)))


def fit(source, target, weights=None):
    x, y = np.asarray(source, float), np.asarray(target, float)
    if x.ndim != 2 or x.shape != y.shape or x.shape[1] != 3 or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("fit expects at least three finite paired 3D points")
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(x),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid fit weights")
    w = w/w.sum()
    mx, my = w@x, w@y
    a, b = x-mx, y-my
    u, d, vt = np.linalg.svd((b*w[:, None]).T@a)
    fix = np.ones(3)
    fix[-1] = np.linalg.det(u@vt)
    R = (u*fix)@vt
    var = np.sum(w*np.sum(a*a, axis=1))
    if var <= 0:
        raise ValueError("zero source spread")
    scale = d@fix/var
    return Sim3(scale, R, my-scale*R@mx)


def support(points, ratio=1e-3):
    x = np.asarray(points, float)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 3 or not np.isfinite(x).all():
        return False, np.zeros(3)
    sv = np.linalg.svd(x-x.mean(0), compute_uv=False)
    return bool(sv[0] > 0 and sv[1]/sv[0] >= ratio), sv
