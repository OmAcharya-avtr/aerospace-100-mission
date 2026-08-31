"""Rigid-body attitude kinematics and dynamics.

Conventions
-----------
Quaternion ``q = (q0, q1, q2, q3)`` is **scalar first** and represents the
rotation that takes a vector from the inertial frame to the body frame:
``v_body = dcm(q) @ v_inertial``.  Angular velocity ``omega`` is the body rate
of the body frame with respect to the inertial frame, expressed in body axes,
in rad/s.

Equations
---------
Attitude matrix (Markley & Crassidis 2014, eq. 2.125):

    A(q) = (q0^2 - |qv|^2) I + 2 qv qv^T - 2 q0 [qv x]

Quaternion kinematics (Markley & Crassidis 2014, eq. 3.20, scalar-first form):

    q0_dot = -0.5 qv . omega
    qv_dot =  0.5 (q0 omega + qv x omega)

Euler's rigid-body equation (Wertz 1978, eq. 16-3; Markley & Crassidis 2014,
eq. 3.81):

    J omega_dot = L - omega x (J omega)

with ``J`` the body-frame inertia tensor [kg m^2] and ``L`` the external
torque [N m].  Valid for a rigid body with constant inertia; no flexible
modes, no internal momentum storage, no fuel slosh.

References
----------
Markley, F. L. and Crassidis, J. L., "Fundamentals of Spacecraft Attitude
    Determination and Control", Springer, 2014.
Wertz, J. R. (ed.), "Spacecraft Attitude Determination and Control", D. Reidel,
    1978.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def skew(v: ArrayLike) -> NDArray[np.float64]:
    """Return the 3x3 skew-symmetric matrix ``[v x]`` such that ``[v x] w = v x w``.

    Parameters
    ----------
    v : array_like, shape (3,)
        Any 3-vector (units are carried through unchanged).

    Returns
    -------
    ndarray, shape (3, 3)
    """
    a = np.asarray(v, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"skew expects a 3-vector, got shape {a.shape}")
    return np.array(
        [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=float
    )


def quat_normalize(q: ArrayLike) -> NDArray[np.float64]:
    """Normalise a scalar-first quaternion to unit norm.

    Raises
    ------
    ValueError
        If the quaternion has (numerically) zero norm.
    """
    a = np.asarray(q, dtype=float)
    if a.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {a.shape}")
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("cannot normalise a zero-norm or non-finite quaternion")
    return a / n


def quat_to_dcm(q: ArrayLike) -> NDArray[np.float64]:
    """Attitude matrix ``A(q)`` mapping inertial vectors to body vectors.

    Parameters
    ----------
    q : array_like, shape (4,)
        Scalar-first quaternion (normalised internally).

    Returns
    -------
    ndarray, shape (3, 3)
        Orthonormal, det = +1, dimensionless.
    """
    a = quat_normalize(q)
    q0, qv = a[0], a[1:]
    return (
        (q0 * q0 - float(qv @ qv)) * np.eye(3)
        + 2.0 * np.outer(qv, qv)
        - 2.0 * q0 * skew(qv)
    )


def dcm_to_quat(dcm: ArrayLike) -> NDArray[np.float64]:
    """Scalar-first quaternion from an attitude matrix (Shepperd's method).

    The sign is fixed so that ``q0 >= 0``.  ``q`` and ``-q`` describe the same
    rotation, so this is a convention, not information.
    """
    a = np.asarray(dcm, dtype=float)
    if a.shape != (3, 3):
        raise ValueError(f"dcm must have shape (3, 3), got {a.shape}")
    tr = float(np.trace(a))
    candidates = np.array([tr, a[0, 0], a[1, 1], a[2, 2]])
    idx = int(np.argmax(candidates))
    if idx == 0:
        s = np.sqrt(max(1.0 + tr, 1e-30)) * 2.0
        q = np.array(
            [0.25 * s, (a[1, 2] - a[2, 1]) / s, (a[2, 0] - a[0, 2]) / s,
             (a[0, 1] - a[1, 0]) / s]
        )
    elif idx == 1:
        s = np.sqrt(max(1.0 + a[0, 0] - a[1, 1] - a[2, 2], 1e-30)) * 2.0
        q = np.array(
            [(a[1, 2] - a[2, 1]) / s, 0.25 * s, (a[1, 0] + a[0, 1]) / s,
             (a[2, 0] + a[0, 2]) / s]
        )
    elif idx == 2:
        s = np.sqrt(max(1.0 - a[0, 0] + a[1, 1] - a[2, 2], 1e-30)) * 2.0
        q = np.array(
            [(a[2, 0] - a[0, 2]) / s, (a[1, 0] + a[0, 1]) / s, 0.25 * s,
             (a[2, 1] + a[1, 2]) / s]
        )
    else:
        s = np.sqrt(max(1.0 - a[0, 0] - a[1, 1] + a[2, 2], 1e-30)) * 2.0
        q = np.array(
            [(a[0, 1] - a[1, 0]) / s, (a[2, 0] + a[0, 2]) / s,
             (a[2, 1] + a[1, 2]) / s, 0.25 * s]
        )
    if q[0] < 0.0:
        q = -q
    return quat_normalize(q)


def quat_multiply(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Hamilton product of two scalar-first quaternions."""
    p = np.asarray(a, dtype=float)
    r = np.asarray(b, dtype=float)
    if p.shape != (4,) or r.shape != (4,):
        raise ValueError("quat_multiply expects two shape-(4,) quaternions")
    p0, pv = p[0], p[1:]
    r0, rv = r[0], r[1:]
    return np.concatenate(([p0 * r0 - pv @ rv], p0 * rv + r0 * pv + np.cross(pv, rv)))


def quat_kinematics(q: ArrayLike, omega_body: ArrayLike) -> NDArray[np.float64]:
    """Quaternion derivative ``q_dot`` [1/s] for body rate ``omega_body`` [rad/s]."""
    a = np.asarray(q, dtype=float)
    w = np.asarray(omega_body, dtype=float)
    if a.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {a.shape}")
    if w.shape != (3,):
        raise ValueError(f"omega_body must have shape (3,), got {w.shape}")
    q0, qv = a[0], a[1:]
    return np.concatenate(([-0.5 * float(qv @ w)], 0.5 * (q0 * w + np.cross(qv, w))))


def rigid_body_derivative(
    omega_body: ArrayLike,
    inertia: ArrayLike,
    torque_body: ArrayLike,
    inertia_inv: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Angular acceleration [rad/s^2] from Euler's equation.

    ``omega_dot = J^-1 (L - omega x (J omega))``

    Parameters
    ----------
    omega_body : array_like, shape (3,)
        Body rate [rad/s].
    inertia : array_like, shape (3, 3)
        Inertia tensor [kg m^2], symmetric positive definite.
    torque_body : array_like, shape (3,)
        External torque in body axes [N m].
    inertia_inv : array_like, optional
        Precomputed inverse of ``inertia`` (avoids a solve in inner loops).
    """
    w = np.asarray(omega_body, dtype=float)
    j = np.asarray(inertia, dtype=float)
    lt = np.asarray(torque_body, dtype=float)
    if w.shape != (3,) or lt.shape != (3,):
        raise ValueError("omega_body and torque_body must have shape (3,)")
    if j.shape != (3, 3):
        raise ValueError(f"inertia must have shape (3, 3), got {j.shape}")
    rhs = lt - np.cross(w, j @ w)
    if inertia_inv is None:
        return np.linalg.solve(j, rhs)
    return np.asarray(inertia_inv, dtype=float) @ rhs


def kinetic_energy(omega_body: ArrayLike, inertia: ArrayLike) -> float:
    """Rotational kinetic energy ``0.5 omega^T J omega`` [J]."""
    w = np.asarray(omega_body, dtype=float)
    j = np.asarray(inertia, dtype=float)
    return 0.5 * float(w @ (j @ w))


def angular_momentum(omega_body: ArrayLike, inertia: ArrayLike) -> NDArray[np.float64]:
    """Body-frame angular momentum ``J omega`` [N m s]."""
    return np.asarray(inertia, dtype=float) @ np.asarray(omega_body, dtype=float)
