"""Attitude kinematics: quaternion derivative and RK4 propagation.

Convention: scalar-first ``[w, x, y, z]``, Hamilton product, active rotation.
The quaternion q maps body-frame vectors to reference (inertial) frame
vectors: ``v_ref = q ⊗ [0, v_body] ⊗ q*``.

Kinematic equation (Markley & Crassidis 2014, Eq. 3.21, scalar-first form)::

    q̇ = ½ q ⊗ [0, ω]

with ω the angular velocity of the body frame relative to the reference
frame, **expressed in body coordinates**, in rad/s.

Renormalization strategy
------------------------
RK4 does not preserve the unit-norm constraint exactly: each step drifts the
norm by O(dt⁵) truncation terms. ``propagate`` applies brute-force
renormalization (q ← q/|q|) after every accepted step, which is the standard
practical remedy and keeps |q| = 1 to machine precision without biasing the
attitude (normalization is a projection along the constraint direction only).
The alternative — integrating with the exact power-series/exponential map per
step — is used in :func:`closed_form_constant_omega` for the constant-ω case
and serves as the validation reference.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .core import _asq, _asv, quat_exp, quat_multiply, quat_normalize

__all__ = [
    "closed_form_constant_omega",
    "propagate",
    "quat_derivative",
    "rk4_step",
]


def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Quaternion time derivative q̇ = ½ q ⊗ [0, ω].

    Parameters
    ----------
    q : (..., 4) unit quaternion(s), scalar-first [w, x, y, z], body-to-reference.
    omega : (..., 3) angular velocity in body frame [rad/s].

    Returns
    -------
    (..., 4) array, dq/dt in 1/s.

    Source: Markley & Crassidis 2014, Eq. (3.21) (scalar-first arrangement).
    """
    q, omega = _asq(q), _asv(omega)
    omega_quat = np.concatenate(
        [np.zeros(omega.shape[:-1] + (1,)), omega], axis=-1
    )
    return 0.5 * quat_multiply(q, omega_quat)


def rk4_step(
    q: np.ndarray,
    t: float,
    dt: float,
    omega_fn: Callable[[float], np.ndarray],
) -> np.ndarray:
    """One classical Runge-Kutta 4 step of q̇ = ½ q ⊗ [0, ω(t)].

    Local truncation error O(dt⁵); the result is NOT renormalized here
    (see module docstring — ``propagate`` renormalizes after each step).

    Parameters
    ----------
    q : (4,) quaternion at time t.
    t : current time [s].
    dt : step size [s].
    omega_fn : callable t [s] -> (3,) body angular velocity [rad/s].
    """
    k1 = quat_derivative(q, omega_fn(t))
    k2 = quat_derivative(q + 0.5 * dt * k1, omega_fn(t + 0.5 * dt))
    k3 = quat_derivative(q + 0.5 * dt * k2, omega_fn(t + 0.5 * dt))
    k4 = quat_derivative(q + dt * k3, omega_fn(t + dt))
    return q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate(
    q0: np.ndarray,
    omega_fn: Callable[[float], np.ndarray],
    times: np.ndarray,
    renormalize: bool = True,
) -> np.ndarray:
    """Propagate attitude through the time grid ``times`` with RK4.

    One RK4 step is taken between each pair of consecutive grid points, so the
    grid spacing IS the integrator step size — choose it small enough for the
    fastest ω dynamics (rule of thumb: dt · |ω| ≲ 0.1 rad for ~1e-9 accuracy).

    Parameters
    ----------
    q0 : (4,) initial unit quaternion (scalar-first). Raises ValueError if not
        within 1e-6 of unit norm (it is then normalized before use).
    omega_fn : callable t [s] -> (3,) body angular velocity [rad/s].
    times : (n,) strictly increasing sample times [s]; q0 corresponds to times[0].
    renormalize : renormalize after every step (default True; see module
        docstring for the strategy rationale).

    Returns
    -------
    (n, 4) array of unit quaternions at ``times``.
    """
    q0 = _asq(q0)
    if q0.shape != (4,):
        raise ValueError(f"q0 must have shape (4,), got {q0.shape}")
    if abs(np.linalg.norm(q0) - 1.0) > 1e-6:
        raise ValueError(
            f"q0 must be a unit quaternion (|q0| = {np.linalg.norm(q0):.6f}); "
            "normalize it first (policy: rotations require unit quaternions)"
        )
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("times must be a 1-D array with at least one element")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    out = np.empty((times.size, 4))
    q = quat_normalize(q0)
    out[0] = q
    for i in range(times.size - 1):
        dt = times[i + 1] - times[i]
        q = rk4_step(q, times[i], dt, omega_fn)
        if renormalize:
            q = quat_normalize(q)
        out[i + 1] = q
    return out


def closed_form_constant_omega(
    q0: np.ndarray, omega: np.ndarray, t: float | np.ndarray
) -> np.ndarray:
    """Exact attitude solution for constant body angular velocity.

    For constant ω the kinematic equation integrates exactly to

        q(t) = q0 ⊗ exp_q(ω t),

    where exp_q is the rotation-vector exponential map (rotation by |ω| t
    about ω̂, Markley & Crassidis 2014, Eq. 3.25). Used as the analytic
    reference for RK4 validation.

    Parameters
    ----------
    q0 : (4,) initial unit quaternion.
    omega : (3,) constant body angular velocity [rad/s].
    t : scalar or (n,) times [s].

    Returns
    -------
    (4,) or (n, 4) unit quaternion(s).
    """
    q0 = quat_normalize(q0)
    omega = _asv(omega)
    t = np.asarray(t, dtype=float)
    rotvec = t[..., np.newaxis] * omega
    return quat_multiply(q0, quat_exp(rotvec))
