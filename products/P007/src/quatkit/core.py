"""Core vectorized quaternion operations on raw numpy arrays.

Convention (used everywhere in quatkit — stated prominently on purpose)
-----------------------------------------------------------------------
* **Scalar-first storage**: a quaternion is the array ``q = [w, x, y, z]``
  with the scalar (real) part ``w`` FIRST and the vector part ``(x, y, z)`` last.
* **Hamilton product**: right-handed algebra with ``i j = k``. The product
  ``quat_multiply(q2, q1)`` composes rotations "apply ``q1`` first, then ``q2``".
* **Active rotation**: a unit quaternion rotates a vector by
  ``v' = q ⊗ [0, v] ⊗ q*`` (conjugation), equivalently ``v' = R(q) v`` with the
  rotation matrix from :func:`quatkit.conversions.quat_to_dcm`.

All functions broadcast over leading axes: quaternion arguments have shape
``(..., 4)`` and vectors ``(..., 3)``. Quaternion components are dimensionless;
angles are in radians throughout.

References
----------
- Markley, F. L. and Crassidis, J. L., *Fundamentals of Spacecraft Attitude
  Determination and Control*, Springer, 2014 — quaternion algebra and
  attitude representations (Chapters 2-3).
- Shuster, M. D., "A Survey of Attitude Representations", *Journal of the
  Astronautical Sciences*, Vol. 41, No. 4, 1993, pp. 439-517.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_conjugate",
    "quat_exp",
    "quat_identity",
    "quat_inverse",
    "quat_log",
    "quat_multiply",
    "quat_norm",
    "quat_normalize",
    "quat_rotate",
    "quat_slerp",
]

_EPS = 1e-12


def _asq(q: np.ndarray) -> np.ndarray:
    """Validate and convert input to a float array of shape (..., 4)."""
    q = np.asarray(q, dtype=float)
    if q.shape[-1:] != (4,):
        raise ValueError(f"quaternion array must have shape (..., 4), got {q.shape}")
    return q


def _asv(v: np.ndarray) -> np.ndarray:
    """Validate and convert input to a float array of shape (..., 3)."""
    v = np.asarray(v, dtype=float)
    if v.shape[-1:] != (3,):
        raise ValueError(f"vector array must have shape (..., 3), got {v.shape}")
    return v


def quat_identity() -> np.ndarray:
    """Return the identity quaternion ``[1, 0, 0, 0]`` (scalar-first, no rotation)."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_norm(q: np.ndarray) -> np.ndarray:
    """Euclidean norm of quaternion(s), shape ``(...,)``. Dimensionless."""
    return np.linalg.norm(_asq(q), axis=-1)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return q / |q|. Raises ValueError if any norm is zero or non-finite.

    Parameters
    ----------
    q : (..., 4) array, scalar-first [w, x, y, z].
    """
    q = _asq(q)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.all(np.isfinite(n)) or np.any(n < _EPS):
        raise ValueError("cannot normalize quaternion with zero or non-finite norm")
    return q / n


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2 (scalar-first, i j = k).

    Composition rule: ``quat_multiply(q2, q1)`` is the rotation "q1 then q2".

    Source: Markley & Crassidis 2014, Eq. (2.82) (adapted to scalar-first ordering).
    """
    q1, q2 = _asq(q1), _asq(q2)
    w1, v1 = q1[..., :1], q1[..., 1:]
    w2, v2 = q2[..., :1], q2[..., 1:]
    w = w1 * w2 - np.sum(v1 * v2, axis=-1, keepdims=True)
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
    return np.concatenate([w, v], axis=-1)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate q* = [w, -x, -y, -z]. Inverse rotation for unit q."""
    q = _asq(q)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse q⁻¹ = q* / |q|². Equals the conjugate for unit quaternions.

    Raises ValueError for (near-)zero norm.
    """
    q = _asq(q)
    n2 = np.sum(q * q, axis=-1, keepdims=True)
    if np.any(n2 < _EPS**2):
        raise ValueError("cannot invert quaternion with (near-)zero norm")
    return quat_conjugate(q) / n2


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by unit quaternion(s) q: v' = q ⊗ [0, v] ⊗ q* (active).

    Uses the expanded form v' = v + 2 w (qv × v) + 2 qv × (qv × v), which is
    algebraically identical to conjugation for unit q and cheaper to evaluate.

    Parameters
    ----------
    q : (..., 4) unit quaternion(s), scalar-first. Callers are responsible for
        unit norm (the `Quaternion` class enforces it); a non-unit q scales the
        result by |q|².
    v : (..., 3) vector(s), any units (units are preserved).
    """
    q, v = _asq(q), _asv(v)
    w, qv = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_exp(rotvec: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector (radians) -> unit quaternion.

    ``q = [cos(θ/2), sin(θ/2) * a]`` where ``θ = |rotvec|`` and ``a = rotvec/θ``.
    Small angles use the series limit sin(θ/2)/θ → 1/2 via ``np.sinc`` (exact
    at θ = 0, no division by zero).

    Source: Markley & Crassidis 2014, Sec. 2.9.3 (quaternion kinematics /
    rotation-vector parameterization). Valid for any finite rotation vector.
    """
    rotvec = _asv(rotvec)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    w = np.cos(0.5 * angle)
    # sin(angle/2)/angle = 0.5*sinc(angle/(2*pi)) with numpy's normalized sinc.
    v = 0.5 * np.sinc(angle / (2.0 * np.pi)) * rotvec
    return np.concatenate([w, v], axis=-1)


def quat_log(q: np.ndarray) -> np.ndarray:
    """Logarithmic map: unit quaternion -> rotation vector (radians).

    Returns the minimal rotation vector, i.e. the sign ambiguity q ≡ -q is
    resolved so the returned angle lies in [0, π]. Inverse of :func:`quat_exp`.
    """
    q = _asq(q)
    # Resolve double cover: enforce non-negative scalar part.
    q = q * np.where(q[..., :1] < 0.0, -1.0, 1.0)
    w, v = q[..., :1], q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(n, w)
    safe_n = np.where(n < _EPS, 1.0, n)
    factor = np.where(n < _EPS, 2.0 / np.maximum(w, _EPS), angle / safe_n)
    return factor * v


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions q0 (t=0) and q1 (t=1).

    Follows the shortest great-circle arc on the unit 3-sphere (the sign of q1
    is flipped if the dot product is negative), giving constant angular
    velocity in t. Falls back to normalized linear interpolation when the
    quaternions are nearly parallel (dot > 1 - 1e-10).

    Source: Shoemake, K., "Animating Rotation with Quaternion Curves",
    ACM SIGGRAPH Computer Graphics, Vol. 19, No. 3, 1985.

    Parameters
    ----------
    q0, q1 : (4,) unit quaternions, scalar-first.
    t : scalar or (n,) array of interpolation parameters (dimensionless;
        values outside [0, 1] extrapolate along the same geodesic).
    """
    q0, q1 = quat_normalize(q0), quat_normalize(q1)
    t = np.asarray(t, dtype=float)[..., np.newaxis]
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 1.0 - 1e-10:
        out = (1.0 - t) * q0 + t * q1
        return out / np.linalg.norm(out, axis=-1, keepdims=True)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / s) * q0 + (np.sin(t * theta) / s) * q1
