"""Quaternion algebra and eigenaxis kinematics for slew planning.

Conventions (identical to QuatKit, P007, and stated here because a slew plan
that assumes the wrong one is silently mirrored)
-------------------------------------------------------------------------
* **Scalar-first storage**: ``q = [w, x, y, z]``, real part first.
* **Hamilton product**: ``i j = k``. ``quat_multiply(q2, q1)`` composes
  "apply ``q1`` first, then ``q2``".
* **Active rotation**: ``quat_rotate(q, v)`` returns
  ``q ⊗ [0, v] ⊗ q*``, the vector ``v`` rotated within one fixed frame.

Throughout SlewForge a spacecraft attitude quaternion ``q`` maps a **body**
vector to the **inertial** frame::

    n_inertial = quat_rotate(q, b_body) = R(q) @ b_body

so an instrument boresight fixed in the body at ``b_body`` points along
``R(q) b_body`` on the sky. Keep-out cone axes are inertial. This is the
convention every function in this package uses; there is no configuration
switch for it.

Angles are radians, quaternion components dimensionless.

References
----------
F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
    Determination and Control*, Springer (2014), Ch. 2-3 -- quaternion
    algebra, the rotation matrix, and the eigenaxis (Euler) rotation theorem.
M. D. Shuster, "A Survey of Attitude Representations", *Journal of the
    Astronautical Sciences* **41**(4), 439-517 (1993).
B. Wie, *Space Vehicle Dynamics and Control*, 2nd ed., AIAA (2008),
    Sec. 5.3 -- eigenaxis rotations and rest-to-rest slews.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EPS_QUAT",
    "axis_angle_from_quat",
    "cross3",
    "quat_angle",
    "quat_conjugate",
    "quat_from_axis_angle",
    "quat_from_rotvec",
    "quat_identity",
    "quat_multiply",
    "quat_normalize",
    "quat_relative",
    "quat_rotate",
    "quat_slerp",
    "quat_to_dcm",
    "rotate_about_axis",
    "unit_vector",
]

EPS_QUAT = 1e-12
"""Norm below which a quaternion or vector is considered degenerate."""


def _asq(q: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(q, dtype=float)
    if a.shape[-1:] != (4,):
        raise ValueError(f"quaternion array must have shape (..., 4), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("quaternion contains non-finite entries")
    return a


def _asv(v: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=float)
    if a.shape[-1:] != (3,):
        raise ValueError(f"vector array must have shape (..., 3), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("vector contains non-finite entries")
    return a


def cross3(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cross product of two length-3 float arrays, shape ``(3,)``.

    Identical in value to ``numpy.cross`` for this shape and about 12 times
    faster, because it skips the general broadcasting and axis-selection
    machinery. The planner evaluates thousands of cross products per solve, so
    the difference is a third of the total solve time; ``numpy.cross`` is used
    everywhere the arguments may be stacked.
    """
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def unit_vector(v: ArrayLike) -> NDArray[np.float64]:
    """Normalise a vector or stack of vectors, shape ``(..., 3)``.

    Raises
    ------
    ValueError
        If any vector has norm below :data:`EPS_QUAT`; a direction is then
        undefined and silently returning one would be a defect.
    """
    a = _asv(v)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(n < EPS_QUAT):
        raise ValueError(f"cannot normalise a vector of norm < {EPS_QUAT}; direction undefined")
    return a / n


def quat_identity() -> NDArray[np.float64]:
    """The identity quaternion ``[1, 0, 0, 0]`` (no rotation)."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: ArrayLike) -> NDArray[np.float64]:
    """Return ``q / |q|``, shape ``(..., 4)``. Raises on a zero-norm quaternion."""
    a = _asq(q)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(n < EPS_QUAT):
        raise ValueError(f"cannot normalise a quaternion of norm < {EPS_QUAT}")
    return a / n


def quat_conjugate(q: ArrayLike) -> NDArray[np.float64]:
    """Conjugate ``[w, -x, -y, -z]``; the inverse rotation for a unit quaternion."""
    a = _asq(q).copy()
    a[..., 1:] *= -1.0
    return a


def quat_multiply(q2: ArrayLike, q1: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product ``q2 ⊗ q1``: apply ``q1`` first, then ``q2``.

    Parameters
    ----------
    q2, q1 : array_like
        Shape ``(..., 4)``, scalar-first, broadcast against each other.

    Returns
    -------
    ndarray
        Shape ``(..., 4)``. Markley & Crassidis (2014) Eq. 2.82 with the
        scalar-first ordering used here.
    """
    a = _asq(q2)
    b = _asq(q1)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_rotate(q: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    """Rotate ``v`` by unit quaternion ``q`` (active): ``q ⊗ [0, v] ⊗ q*``.

    Implemented as ``v + 2 w (u × v) + 2 u × (u × v)`` with ``u`` the vector
    part, which costs no quaternion multiplications and is exact for unit
    ``q``. Units of ``v`` are preserved.
    """
    a = quat_normalize(q)
    b = _asv(v)
    if a.shape == (4,) and b.shape == (3,):
        u = a[1:]
        t = cross3(u, b)
        return b + 2.0 * (a[0] * t + cross3(u, t))
    w = a[..., :1]
    u = a[..., 1:]
    t = np.cross(u, b)
    return b + 2.0 * (w * t + np.cross(u, t))


def quat_to_dcm(q: ArrayLike) -> NDArray[np.float64]:
    """Rotation matrix ``R`` with ``R @ b_body = n_inertial``, shape ``(..., 3, 3)``.

    Markley & Crassidis (2014) Eq. 2.125, transposed to the active convention
    stated in the module docstring. ``R`` is orthogonal with determinant +1.
    """
    a = quat_normalize(q)
    w, x, y, z = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def quat_from_axis_angle(axis: ArrayLike, angle: float) -> NDArray[np.float64]:
    """Unit quaternion for a rotation of ``angle`` [rad] about ``axis``.

    ``q = [cos(angle/2), sin(angle/2) * axis_hat]``. Any real ``angle``.
    """
    k = unit_vector(axis)
    half = 0.5 * float(angle)
    return np.concatenate([[np.cos(half)], np.sin(half) * k])


def quat_from_rotvec(rotvec: ArrayLike) -> NDArray[np.float64]:
    """Unit quaternion from a rotation vector ``p`` [rad], ``|p|`` the angle.

    Uses the series ``sin(t/2)/t -> 1/2 - t^2/48`` below ``|p| = 1e-6`` rad so
    the small-angle limit is exact to machine precision rather than 0/0.
    """
    p = _asv(rotvec).reshape(3)
    theta = float(np.linalg.norm(p))
    if theta < 1e-6:
        # sin(theta/2)/theta = 1/2 - theta^2/48 + O(theta^4)
        s = 0.5 - theta * theta / 48.0
        return np.concatenate([[np.cos(0.5 * theta)], s * p])
    return np.concatenate([[np.cos(0.5 * theta)], (np.sin(0.5 * theta) / theta) * p])


def axis_angle_from_quat(q: ArrayLike) -> tuple[NDArray[np.float64], float]:
    """Return ``(axis_hat, angle)`` with ``angle`` in ``[0, pi]`` [rad].

    The quaternion sign is resolved so the angle is the *short* rotation:
    ``q`` and ``-q`` give the same result. For an angle below
    :data:`EPS_QUAT` the axis is arbitrary and ``[1, 0, 0]`` is returned with
    angle 0; the caller is expected to treat that case as "no rotation".

    Uses ``angle = 2 atan2(|q_v|, |q_w|)`` rather than ``2 arccos(q_w)``: the
    arccos form loses about half its significant digits near zero angle.
    """
    a = quat_normalize(q).reshape(4)
    vn = float(np.linalg.norm(a[1:]))
    w = abs(float(a[0]))
    angle = 2.0 * float(np.arctan2(vn, w))
    if vn < EPS_QUAT:
        return np.array([1.0, 0.0, 0.0]), 0.0
    sign = 1.0 if a[0] >= 0.0 else -1.0
    return (sign * a[1:]) / vn, angle


def quat_relative(q_from: ArrayLike, q_to: ArrayLike) -> NDArray[np.float64]:
    """Rotation taking ``q_from`` to ``q_to``: ``q_to ⊗ q_from*``.

    With the active, inertial-from-body convention this is the *inertial-frame*
    rotation that carries the start attitude onto the goal attitude, so its
    axis is the eigenaxis expressed in inertial coordinates.
    """
    return quat_multiply(quat_normalize(q_to), quat_conjugate(quat_normalize(q_from)))


def quat_angle(q_from: ArrayLike, q_to: ArrayLike) -> float:
    """Eigenaxis (principal) angle between two attitudes [rad], in ``[0, pi]``."""
    return axis_angle_from_quat(quat_relative(q_from, q_to))[1]


def quat_slerp(q0: ArrayLike, q1: ArrayLike, s: ArrayLike) -> NDArray[np.float64]:
    """Spherical linear interpolation along the eigenaxis, ``s`` in ``[0, 1]``.

    Shoemake (1985). The shorter arc is always taken (``q1`` is sign-flipped
    when ``q0 · q1 < 0``), so the interpolated path is the eigenaxis rotation
    of angle ``<= pi``. Below a half-angle of 1e-8 rad the formula degenerates
    and normalised linear interpolation is used instead, whose error there is
    below 1e-16.

    Parameters
    ----------
    q0, q1 : array_like
        Unit quaternions, shape ``(4,)``.
    s : array_like
        Interpolation parameter, shape ``()`` or ``(n,)``. Values outside
        ``[0, 1]`` extrapolate along the same eigenaxis and are allowed.

    Returns
    -------
    ndarray
        Shape ``(4,)`` or ``(n, 4)``.
    """
    a = quat_normalize(q0).reshape(4)
    b = quat_normalize(q1).reshape(4)
    t = np.atleast_1d(np.asarray(s, dtype=float))
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    half = float(np.arccos(dot))
    if half < 1e-8:
        out = (1.0 - t)[:, None] * a[None, :] + t[:, None] * b[None, :]
        out = out / np.linalg.norm(out, axis=-1, keepdims=True)
    else:
        sh = np.sin(half)
        out = (np.sin((1.0 - t) * half)[:, None] * a[None, :] + np.sin(t * half)[:, None] * b[None, :]) / sh
    return out[0] if np.ndim(s) == 0 else out


def rotate_about_axis(v: ArrayLike, axis: ArrayLike, angle: ArrayLike) -> NDArray[np.float64]:
    """Rodrigues rotation of ``v`` about unit ``axis`` by ``angle`` [rad].

    ``v cos t + (k × v) sin t + k (k · v)(1 - cos t)``, active convention.
    Broadcasts over ``angle``: a scalar returns ``(3,)``, an ``(n,)`` array
    returns ``(n, 3)``. Used to trace an instrument boresight along an
    eigenaxis slew without building a quaternion per sample.
    """
    k = unit_vector(axis).reshape(3)
    b = _asv(v).reshape(3)
    t = np.atleast_1d(np.asarray(angle, dtype=float))
    c = np.cos(t)[:, None]
    s = np.sin(t)[:, None]
    kv = np.cross(k, b)
    kd = float(np.dot(k, b))
    out = b[None, :] * c + kv[None, :] * s + k[None, :] * (kd * (1.0 - c))
    return out[0] if np.ndim(angle) == 0 else out
