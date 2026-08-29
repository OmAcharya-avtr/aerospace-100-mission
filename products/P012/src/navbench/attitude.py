"""Quaternion algebra and rigid-body attitude kinematics/dynamics.

CONVENTIONS (these govern every function in this module and in ``mekf.py``)

* Quaternions are **scalar-first**: ``q = [q0, q1, q2, q3]`` with ``q0`` the
  scalar part.  Unit norm is required for every rotation-carrying quaternion.
* **Hamilton** product (``i j = k``).  ``quat_multiply(a, b)`` returns
  ``a ⊗ b``.
* ``q`` is the **body-to-inertial** attitude quaternion.  ``dcm_from_quat(q)``
  returns ``R`` with ``v_inertial = R @ v_body`` and ``v_body = R.T @ v_i``.
  ``R`` is therefore the *active* rotation matrix and matches
  ``scipy.spatial.transform.Rotation.from_quat([q1, q2, q3, q0]).as_matrix()``
  (scipy stores the scalar last).
* Angular rate ``omega`` is expressed in the **body** frame, rad/s.
* Attitude kinematics: ``q̇ = ½ q ⊗ [0, ω_body]``
  (Markley & Crassidis 2014, "Fundamentals of Spacecraft Attitude
  Determination and Control", Springer, Eq. (2.88) in the equivalent
  scalar-last form; Shuster 1993, "A Survey of Attitude Representations",
  J. Astronautical Sciences 41(4), §2).
* Rigid-body dynamics (Euler's equations, body frame, principal or full
  inertia tensor):  ``J ω̇ = τ − ω × (J ω)``
  (Wertz 1978, "Spacecraft Attitude Determination and Control", Eq. (16-3);
  Markley & Crassidis 2014 Eq. (3.81)).  Units: ``J`` kg·m², ``τ`` N·m,
  ``ω`` rad/s.  Validity: rigid body, no flexible modes, no internal momentum
  storage.

RELATED PRIOR ART.  Product P007 (QuatKit) in this portfolio is a dedicated
quaternion toolbox with the same scalar-first Hamilton convention.  NavBench
does **not** import it — every product in this portfolio is self-contained —
and the implementation here is independent and independently validated
(``validation/v4_mekf_quaternion.py``).  P007 is cited as related work, not
reused.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "skew",
    "quat_identity",
    "quat_multiply",
    "quat_conjugate",
    "quat_normalize",
    "quat_norm",
    "dcm_from_quat",
    "quat_from_dcm",
    "quat_from_axis_angle",
    "axis_angle_from_quat",
    "quat_from_small_angle",
    "small_angle_from_quat",
    "quat_rotate",
    "quat_derivative",
    "quat_propagate",
    "quat_angle_between",
    "quat_canonical",
    "euler_zyx_from_quat",
    "quat_from_euler_zyx",
    "euler_moment_derivative",
]

_EPS = np.finfo(float).eps


def _as_quat(q: ArrayLike, name: str = "q") -> NDArray[np.float64]:
    arr = np.asarray(q, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr!r}")
    return arr


def _as_vec3(v: ArrayLike, name: str = "v") -> NDArray[np.float64]:
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr!r}")
    return arr


def skew(v: ArrayLike) -> NDArray[np.float64]:
    """Skew-symmetric cross-product matrix ``[v×]`` with ``[v×] u = v × u``.

    Parameters
    ----------
    v : array_like, shape (3,)
        Any 3-vector (units are carried through unchanged).

    Returns
    -------
    ndarray, shape (3, 3)
        ``[[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]]``.
    """
    a = _as_vec3(v, "v")
    return np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )


def quat_identity() -> NDArray[np.float64]:
    """Identity (zero-rotation) quaternion ``[1, 0, 0, 0]`` (dimensionless)."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_multiply(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product ``a ⊗ b`` (scalar-first, dimensionless).

    Composition semantics under the body-to-inertial convention: if ``b``
    rotates body→intermediate and ``a`` rotates intermediate→inertial, then
    ``a ⊗ b`` rotates body→inertial.
    """
    p = _as_quat(a, "a")
    r = _as_quat(b, "b")
    w0, x0, y0, z0 = p
    w1, x1, y1, z1 = r
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def quat_conjugate(q: ArrayLike) -> NDArray[np.float64]:
    """Quaternion conjugate ``[q0, -qv]``; the inverse rotation for unit ``q``."""
    a = _as_quat(q)
    return np.array([a[0], -a[1], -a[2], -a[3]])


def quat_norm(q: ArrayLike) -> float:
    """Euclidean norm of a quaternion (dimensionless)."""
    return float(np.linalg.norm(_as_quat(q)))


def quat_normalize(q: ArrayLike) -> NDArray[np.float64]:
    """Return ``q / |q|``.

    Raises
    ------
    ValueError
        If ``|q| < 1e-12``, which cannot represent a rotation.
    """
    a = _as_quat(q)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError(f"cannot normalize a quaternion of norm {n:.3e} (needs >= 1e-12)")
    return a / n


def quat_canonical(q: ArrayLike) -> NDArray[np.float64]:
    """Return the representative of ``±q`` with non-negative scalar part.

    ``q`` and ``-q`` describe the same rotation (the unit quaternions double
    cover SO(3)); this picks a unique representative so that quaternion
    differences are meaningful.
    """
    a = _as_quat(q)
    return -a if a[0] < 0.0 else a.copy()


def dcm_from_quat(q: ArrayLike) -> NDArray[np.float64]:
    """Rotation matrix ``R`` with ``v_inertial = R @ v_body``.

    ``R = (q0² − |qv|²) I + 2 qv qvᵀ + 2 q0 [qv×]``
    (Shuster 1993 §2, active form; Markley & Crassidis 2014 Eq. (2.125)
    with the scalar-last ordering).  Dimensionless, orthogonal, det = +1 for
    a unit quaternion.
    """
    a = quat_normalize(q)
    q0, qv = a[0], a[1:]
    return (q0 * q0 - float(qv @ qv)) * np.eye(3) + 2.0 * np.outer(qv, qv) + 2.0 * q0 * skew(qv)


def quat_from_dcm(dcm: ArrayLike) -> NDArray[np.float64]:
    """Unit quaternion from a rotation matrix (Shepperd's method).

    Shepperd, S. W. (1978), "Quaternion from Rotation Matrix",
    J. Guidance and Control 1(3), 223-224.  Picks the branch with the largest
    pivot to avoid cancellation; returns the canonical (``q0 >= 0``) sign.

    Raises
    ------
    ValueError
        If ``dcm`` is not 3×3 or is not orthogonal to 1e-6.
    """
    m = np.asarray(dcm, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"dcm must have shape (3, 3), got {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("dcm must be finite")
    orth = float(np.max(np.abs(m @ m.T - np.eye(3))))
    if orth > 1e-6:
        raise ValueError(f"dcm is not orthogonal: max|R Rᵀ − I| = {orth:.3e} (tolerance 1e-6)")
    if float(np.linalg.det(m)) < 0.0:
        raise ValueError("dcm has determinant < 0 (reflection, not a rotation)")

    tr = float(np.trace(m))
    pivots = np.array([tr, m[0, 0], m[1, 1], m[2, 2]])
    k = int(np.argmax(pivots))
    if k == 0:
        s = np.sqrt(1.0 + tr) * 2.0
        q = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    elif k == 1:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    elif k == 2:
        s = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2]) * 2.0
        q = np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2]) * 2.0
        q = np.array(
            [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        )
    return quat_canonical(quat_normalize(q))


def quat_from_axis_angle(axis: ArrayLike, angle_rad: float) -> NDArray[np.float64]:
    """Quaternion for a rotation of ``angle_rad`` [rad] about ``axis``.

    ``q = [cos(θ/2), sin(θ/2) û]``.  ``axis`` need not be normalised but must
    have norm ≥ 1e-12.
    """
    a = _as_vec3(axis, "axis")
    theta = float(angle_rad)
    if not np.isfinite(theta):
        raise ValueError(f"angle_rad must be finite, got {theta!r}")
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError(f"axis norm {n:.3e} is too small to define a rotation axis")
    u = a / n
    half = 0.5 * theta
    return np.concatenate(([np.cos(half)], np.sin(half) * u))


def axis_angle_from_quat(q: ArrayLike) -> tuple[NDArray[np.float64], float]:
    """Inverse of :func:`quat_from_axis_angle`.

    Returns ``(axis, angle_rad)`` with ``angle_rad ∈ [0, π]`` and a unit axis
    (``[1, 0, 0]`` by convention only when the vector part is *exactly* zero).

    The direction ``qv/|qv|`` stays accurate to round-off for arbitrarily small
    ``|qv|``, so the degenerate branch is taken only at underflow.  An earlier
    version of this function used a ``1e-12`` cut-off and silently returned the
    wrong axis for rotations below ~2e-12 rad; that defect was caught by
    ``validation/v4_mekf_quaternion.py`` PART A and the threshold is now the
    smallest normal double.
    """
    a = quat_canonical(quat_normalize(q))
    sin_half = float(np.linalg.norm(a[1:]))
    angle = 2.0 * float(np.arctan2(sin_half, a[0]))
    if sin_half < np.finfo(float).tiny:
        return np.array([1.0, 0.0, 0.0]), angle
    return a[1:] / sin_half, angle


def quat_from_small_angle(a: ArrayLike) -> NDArray[np.float64]:
    """Error quaternion from a small rotation vector ``a`` [rad].

    Uses the exact rotation-vector map ``δq = [cos(|a|/2), sin(|a|/2) a/|a|]``
    rather than the first-order ``[1, a/2]``, so the result is a unit
    quaternion for any ``|a|``.  For ``|a| → 0`` the series expansion
    ``sin(|a|/2)/|a| → 1/2`` is used to avoid 0/0.

    This is the map used for the MEKF multiplicative reset (Lefferts, Markley
    & Shuster 1982, "Kalman Filtering for Spacecraft Attitude Estimation",
    J. Guidance 5(5), 417-429, §III).
    """
    v = _as_vec3(a, "a")
    theta = float(np.linalg.norm(v))
    if theta < 1e-8:
        # sin(θ/2)/θ = 1/2 − θ²/48 + O(θ⁴); cos(θ/2) = 1 − θ²/8 + O(θ⁴)
        scale = 0.5 - theta * theta / 48.0
        q = np.concatenate(([1.0 - theta * theta / 8.0], scale * v))
        return quat_normalize(q)
    return np.concatenate(([np.cos(0.5 * theta)], np.sin(0.5 * theta) * v / theta))


def small_angle_from_quat(q: ArrayLike) -> NDArray[np.float64]:
    """Rotation vector [rad] of ``q``; exact inverse of :func:`quat_from_small_angle`."""
    axis, angle = axis_angle_from_quat(q)
    return axis * angle


def quat_rotate(q: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    """Rotate a body-frame vector into the inertial frame: ``R(q) v``.

    Units of ``v`` are carried through unchanged.
    """
    return dcm_from_quat(q) @ _as_vec3(v, "v")


def quat_derivative(q: ArrayLike, omega_body: ArrayLike) -> NDArray[np.float64]:
    """``q̇ = ½ q ⊗ [0, ω_body]`` [1/s].

    Markley & Crassidis 2014 Eq. (3.20) (scalar-last equivalent); Shuster 1993
    §2.  ``omega_body`` in rad/s.  Validity: any rate; the quaternion norm is
    conserved analytically but not by a finite-difference integrator, hence
    :func:`quat_propagate` renormalises.
    """
    a = _as_quat(q)
    w = _as_vec3(omega_body, "omega_body")
    return 0.5 * quat_multiply(a, np.concatenate(([0.0], w)))


def quat_propagate(q: ArrayLike, omega_body: ArrayLike, dt: float) -> NDArray[np.float64]:
    """Exact closed-form propagation for a body rate held constant over ``dt``.

    ``q(t+dt) = q(t) ⊗ [cos(|ω| dt/2), sin(|ω| dt/2) ω/|ω|]``
    (Markley & Crassidis 2014 Eq. (3.21); exact for constant ω, first-order
    accurate for a time-varying rate sampled at ``dt``).

    Parameters
    ----------
    q : array_like, shape (4,)
        Unit quaternion at time t.
    omega_body : array_like, shape (3,)
        Body angular rate [rad/s], assumed constant across the interval.
    dt : float
        Step [s]; must be finite.
    """
    a = quat_normalize(q)
    w = _as_vec3(omega_body, "omega_body")
    step = float(dt)
    if not np.isfinite(step):
        raise ValueError(f"dt must be finite, got {step!r}")
    return quat_normalize(quat_multiply(a, quat_from_small_angle(w * step)))


def quat_angle_between(q1: ArrayLike, q2: ArrayLike) -> float:
    """Rotation angle [rad] separating two attitudes, in ``[0, π]``.

    Computed from the error quaternion ``q1* ⊗ q2`` so that the ``±q``
    ambiguity is removed.
    """
    dq = quat_multiply(quat_conjugate(quat_normalize(q1)), quat_normalize(q2))
    _, angle = axis_angle_from_quat(dq)
    return angle


def quat_from_euler_zyx(yaw: float, pitch: float, roll: float) -> NDArray[np.float64]:
    """Aerospace 3-2-1 (yaw-pitch-roll, intrinsic) Euler angles [rad] → quaternion.

    The returned quaternion is body-to-inertial under this module's
    convention, i.e. ``R = Rz(yaw) Ry(pitch) Rx(roll)``.
    """
    for name, val in (("yaw", yaw), ("pitch", pitch), ("roll", roll)):
        if not np.isfinite(float(val)):
            raise ValueError(f"{name} must be finite, got {val!r}")
    qz = quat_from_axis_angle([0.0, 0.0, 1.0], float(yaw))
    qy = quat_from_axis_angle([0.0, 1.0, 0.0], float(pitch))
    qx = quat_from_axis_angle([1.0, 0.0, 0.0], float(roll))
    return quat_normalize(quat_multiply(quat_multiply(qz, qy), qx))


def euler_zyx_from_quat(q: ArrayLike) -> tuple[float, float, float]:
    """Quaternion → aerospace 3-2-1 Euler angles ``(yaw, pitch, roll)`` [rad].

    ``pitch`` is clipped into ``[-1, 1]`` before ``arcsin`` to tolerate
    round-off.  Near ``|pitch| = π/2`` the yaw/roll split is ill-conditioned
    (gimbal lock); the caller is responsible for avoiding that regime.
    """
    r = dcm_from_quat(q)
    pitch = float(np.arcsin(-np.clip(r[2, 0], -1.0, 1.0)))
    yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    roll = float(np.arctan2(r[2, 1], r[2, 2]))
    return yaw, pitch, roll


def euler_moment_derivative(
    omega_body: ArrayLike, inertia: ArrayLike, torque_body: ArrayLike
) -> NDArray[np.float64]:
    """Euler's rigid-body equation solved for ``ω̇`` [rad/s²].

    ``ω̇ = J⁻¹ (τ − ω × (J ω))``

    Wertz 1978 Eq. (16-3); Markley & Crassidis 2014 Eq. (3.81).

    Parameters
    ----------
    omega_body : array_like, shape (3,)
        Body angular rate [rad/s].
    inertia : array_like, shape (3, 3)
        Inertia tensor about the centre of mass, body axes [kg·m²].  Must be
        symmetric positive definite.
    torque_body : array_like, shape (3,)
        External torque in body axes [N·m].

    Validity: rigid body, constant inertia, no reaction wheels or fuel slosh.
    """
    w = _as_vec3(omega_body, "omega_body")
    tau = _as_vec3(torque_body, "torque_body")
    j = np.asarray(inertia, dtype=float)
    if j.shape != (3, 3):
        raise ValueError(f"inertia must have shape (3, 3), got {j.shape}")
    if not np.all(np.isfinite(j)):
        raise ValueError("inertia must be finite")
    if float(np.max(np.abs(j - j.T))) > 1e-9 * max(1.0, float(np.max(np.abs(j)))):
        raise ValueError("inertia must be symmetric")
    eig = np.linalg.eigvalsh(j)
    if float(eig.min()) <= 0.0:
        raise ValueError(f"inertia must be positive definite; min eigenvalue {eig.min():.3e}")
    return np.linalg.solve(j, tau - np.cross(w, j @ w))
