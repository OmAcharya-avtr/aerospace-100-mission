"""Rigid-body attitude dynamics with reaction-wheel actuation.

Equations of motion (Wie 2008 Eq. 6.72; Markley & Crassidis 2014 Eq. 3.147)
--------------------------------------------------------------------------
With ``J`` the spacecraft inertia about its centre of mass including the
wheels' own inertia [kg m^2], ``omega`` the body rate [rad/s], ``A`` the
wheel distribution matrix (3 x m, dimensionless), ``h`` the wheel momenta
[N*m*s] and ``tau_ext`` an external torque [N*m],

    J omega_dot = tau_ext - A h_dot - omega x (J omega + A h)
    q_dot       = 1/2 q ⊗ [0, omega_body]

Storage is scalar-first, the product is Hamilton, and ``q`` maps body to
inertial (see :mod:`slewforge.attitude`). With those conventions the
kinematics above is ``q_dot = 1/2 quat_multiply(q, [0, omega])``: the rate is
expressed in the body frame, so the increment multiplies on the *right*.

Total inertial angular momentum ``L = R(q) (J omega + A h)`` is constant when
``tau_ext = 0``, whatever the wheels do. That is the conservation law
`validation/validate_momentum_conservation.py` measures.

Exact eigenaxis torque
----------------------
An eigenaxis slew has ``omega(t) = psi_dot(t) e_b`` with ``e_b`` the eigenaxis
expressed in the body frame, constant. Substituting,

    tau_required(t) = psi_ddot J e_b + psi_dot^2 (e_b x J e_b)

The second term is the gyroscopic term the scalar sizing model in
:mod:`slewforge.profiles` drops. It vanishes if and only if ``e_b`` is a
principal axis, and otherwise grows with the *square* of the rate, so it is
worst exactly at the middle of a fast slew.

Integrator: classical fourth-order Runge-Kutta with the quaternion
renormalised after every step. Fixed step; no adaptive control, no symplectic
structure. Order is verified in `validation/validate_momentum_conservation.py`
by step halving.

Units: torque N*m, momentum N*m*s, inertia kg m^2, rate rad/s, time s.

References
----------
B. Wie, *Space Vehicle Dynamics and Control*, 2nd ed., AIAA (2008), Ch. 6-7.
F. L. Markley and J. L. Crassidis, *Fundamentals of Spacecraft Attitude
    Determination and Control*, Springer (2014), Sec. 3.7.
P. C. Hughes, *Spacecraft Attitude Dynamics*, Wiley (1986), Ch. 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .attitude import cross3, quat_normalize, quat_rotate
from .profiles import SlewProfile
from .wheels import WheelArray

__all__ = [
    "RigidBody",
    "SimulationResult",
    "eigenaxis_torque",
    "inertial_momentum",
    "propagate",
    "simulate_profile",
]


def _as_inertia(j: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(j, dtype=float)
    if a.shape == (3,):
        a = np.diag(a)
    if a.shape != (3, 3):
        raise ValueError(f"inertia must have shape (3, 3) or (3,), got {np.shape(j)}")
    if not np.all(np.isfinite(a)):
        raise ValueError("inertia contains non-finite entries")
    if not np.allclose(a, a.T, rtol=0.0, atol=1e-12 * max(1.0, float(np.max(np.abs(a))))):
        raise ValueError("inertia matrix must be symmetric")
    w = np.linalg.eigvalsh(a)
    if np.any(w <= 0.0):
        raise ValueError(f"inertia must be positive definite; eigenvalues {w}")
    # Triangle inequality on the principal moments: a physical rigid body
    # satisfies J1 + J2 >= J3 for every permutation (Hughes 1986, Sec. 2.4).
    j1, j2, j3 = float(w[0]), float(w[1]), float(w[2])
    if j1 + j2 < j3 * (1.0 - 1e-12):
        raise ValueError(
            f"principal moments {j1, j2, j3} violate the triangle inequality; "
            "no rigid mass distribution has this inertia"
        )
    return a


@dataclass(frozen=True)
class RigidBody:
    """A rigid spacecraft with an optional wheel array.

    Parameters
    ----------
    inertia : array_like
        ``(3, 3)`` symmetric positive-definite inertia tensor [kg m^2] about
        the centre of mass, in body axes, *including* the wheels. A ``(3,)``
        array is read as principal moments. The triangle inequality on the
        principal moments is enforced.
    wheels : WheelArray or None
        Momentum-exchange actuators. ``None`` means torques are applied
        directly to the body, which is the model used for the torque-free
        conservation checks.
    name : str
        Label.
    """

    inertia: NDArray[np.float64]
    wheels: WheelArray | None = None
    name: str = "spacecraft"

    def __post_init__(self) -> None:
        object.__setattr__(self, "inertia", _as_inertia(self.inertia))
        if self.wheels is not None and not isinstance(self.wheels, WheelArray):
            raise TypeError(f"wheels must be a WheelArray or None, got {type(self.wheels).__name__}")

    @property
    def principal_moments(self) -> NDArray[np.float64]:
        """Eigenvalues of the inertia tensor [kg m^2], ascending."""
        return np.linalg.eigvalsh(self.inertia)

    def effective_inertia(self, axis: ArrayLike) -> float:
        """``e^T J e`` [kg m^2] -- the scalar inertia about ``axis``.

        This is the quantity the single-axis sizing model uses. It equals the
        true moment of inertia about the axis; it is *not* enough to predict
        the torque, because ``J e`` need not be parallel to ``e``.
        """
        e = np.asarray(axis, dtype=float).reshape(3)
        n = float(np.linalg.norm(e))
        if n < 1e-12:
            raise ValueError("axis has zero length")
        e = e / n
        return float(e @ self.inertia @ e)

    def is_principal_axis(self, axis: ArrayLike, tol: float = 1e-9) -> bool:
        """``True`` if ``J e`` is parallel to ``e`` to within ``tol`` [rad]."""
        e = np.asarray(axis, dtype=float).reshape(3)
        e = e / float(np.linalg.norm(e))
        je = self.inertia @ e
        return bool(np.linalg.norm(np.cross(e, je)) <= tol * float(np.linalg.norm(je)))


def eigenaxis_torque(
    inertia: ArrayLike, axis_body: ArrayLike, rate: ArrayLike, accel: ArrayLike
) -> NDArray[np.float64]:
    """Exact body torque an eigenaxis slew requires [N*m].

    ``tau = psi_ddot J e + psi_dot^2 (e x J e)``, from Euler's equation with
    ``omega = psi_dot e``, ``e`` constant in the body frame.

    Parameters
    ----------
    inertia : array_like
        ``(3, 3)`` or ``(3,)`` [kg m^2].
    axis_body : array_like
        Unit eigenaxis in body coordinates, shape ``(3,)``.
    rate, accel : array_like
        ``psi_dot`` [rad/s] and ``psi_ddot`` [rad/s^2], scalar or shape ``(n,)``.

    Returns
    -------
    ndarray
        ``(3,)`` for scalar inputs, ``(n, 3)`` otherwise.
    """
    j = _as_inertia(inertia)
    e = np.asarray(axis_body, dtype=float).reshape(3)
    e = e / float(np.linalg.norm(e))
    je = j @ e
    gyro = np.cross(e, je)
    r = np.atleast_1d(np.asarray(rate, dtype=float))
    a = np.atleast_1d(np.asarray(accel, dtype=float))
    r, a = np.broadcast_arrays(r, a)
    out = a[:, None] * je[None, :] + (r**2)[:, None] * gyro[None, :]
    return out[0] if np.ndim(rate) == 0 and np.ndim(accel) == 0 else out


def inertial_momentum(
    body: RigidBody, quat: ArrayLike, omega: ArrayLike, wheel_momentum: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Total inertial angular momentum ``R(q)(J omega + A h)`` [N*m*s], shape ``(3,)``."""
    w = np.asarray(omega, dtype=float).reshape(3)
    hb = body.inertia @ w
    if wheel_momentum is not None:
        if body.wheels is None:
            raise ValueError("wheel_momentum given but the body has no wheel array")
        h = np.asarray(wheel_momentum, dtype=float).reshape(body.wheels.n_wheels)
        hb = hb + body.wheels.distribution @ h
    return quat_rotate(quat, hb)


def _qmul_raw(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hamilton product of two ``(4,)`` quaternions, no validation.

    Identical to :func:`slewforge.attitude.quat_multiply` for this shape;
    ``tests/test_dynamics.py::test_raw_quaternion_product_matches_public``
    asserts that. The integrator calls it four times per step, so the
    validation overhead of the public function was the single largest cost in
    a long propagation.
    """
    return np.array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )


def _derivative(
    inertia: NDArray[np.float64],
    inertia_inv: NDArray[np.float64],
    distribution: NDArray[np.float64] | None,
    quat: NDArray[np.float64],
    omega: NDArray[np.float64],
    wheel_h: NDArray[np.float64] | None,
    wheel_hdot: NDArray[np.float64] | None,
    tau_ext: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(q_dot, omega_dot)`` from Euler's equation and the kinematics."""
    hb = inertia @ omega
    if distribution is not None and wheel_h is not None:
        hb = hb + distribution @ wheel_h
    torque = tau_ext - cross3(omega, hb)
    if distribution is not None and wheel_hdot is not None:
        torque = torque - distribution @ wheel_hdot
    omega_dot = inertia_inv @ torque
    q_dot = 0.5 * _qmul_raw(quat, np.array([0.0, omega[0], omega[1], omega[2]]))
    return q_dot, omega_dot


@dataclass(frozen=True)
class SimulationResult:
    """Output of :func:`propagate` and :func:`simulate_profile`.

    Attributes
    ----------
    time : ndarray
        ``(n,)`` sample times [s].
    quat : ndarray
        ``(n, 4)`` attitude, scalar-first, body to inertial.
    omega : ndarray
        ``(n, 3)`` body rate [rad/s].
    wheel_momentum : ndarray
        ``(n, m)`` per-wheel momentum [N*m*s]; empty ``(n, 0)`` without wheels.
    body_torque : ndarray
        ``(n, 3)`` commanded body torque [N*m].
    momentum : ndarray
        ``(n, 3)`` total inertial angular momentum [N*m*s].
    saturated_torque, saturated_momentum : ndarray
        ``(n,)`` booleans; ``True`` where a per-wheel limit was exceeded by
        the minimum-norm allocation. The simulation does **not** clip: it
        records the violation and keeps integrating the commanded torque, so
        the reported trajectory is the one the command asked for and the flag
        says it was not deliverable.
    """

    time: NDArray[np.float64]
    quat: NDArray[np.float64]
    omega: NDArray[np.float64]
    wheel_momentum: NDArray[np.float64]
    body_torque: NDArray[np.float64]
    momentum: NDArray[np.float64]
    saturated_torque: NDArray[np.bool_]
    saturated_momentum: NDArray[np.bool_]

    @property
    def any_saturation(self) -> bool:
        """``True`` if any wheel limit was exceeded at any sample."""
        return bool(np.any(self.saturated_torque) or np.any(self.saturated_momentum))

    def momentum_drift(self) -> float:
        """Largest deviation of ``|L|`` from its initial value [N*m*s]."""
        n = np.linalg.norm(self.momentum, axis=1)
        return float(np.max(np.abs(n - n[0])))

    def momentum_direction_drift(self) -> float:
        """Largest change in the direction of ``L`` [rad].

        Computed as ``atan2(|u x u0|, u . u0)``, not ``arccos(u . u0)``: the
        arccos form has a floor of about ``sqrt(2 eps) = 2.1e-8`` rad for
        nearly parallel unit vectors and would report that floor as a physical
        drift. Returns 0.0 when the initial momentum is numerically zero,
        where the direction is undefined.
        """
        n0 = self.momentum[0]
        mag = float(np.linalg.norm(n0))
        if mag < 1e-14:
            return 0.0
        u0 = n0 / mag
        u = self.momentum / np.linalg.norm(self.momentum, axis=1, keepdims=True)
        return float(np.max(np.arctan2(np.linalg.norm(np.cross(u, u0), axis=1), u @ u0)))


def propagate(
    body: RigidBody,
    quat0: ArrayLike,
    omega0: ArrayLike,
    duration: float,
    dt: float,
    torque_fn=None,
    wheel_momentum0: ArrayLike | None = None,
) -> SimulationResult:
    """Integrate the attitude dynamics with RK4 and a fixed step.

    Parameters
    ----------
    body : RigidBody
    quat0 : array_like
        Initial attitude ``(4,)``, scalar-first.
    omega0 : array_like
        Initial body rate ``(3,)`` [rad/s].
    duration : float
        Total time [s], ``> 0``.
    dt : float
        Step [s], ``> 0``. The last step is shortened to land exactly on
        ``duration``.
    torque_fn : callable or None
        ``f(t) -> (3,)`` commanded **body** torque [N*m]. With a wheel array
        the torque is realised by the wheels, so ``h_dot = -A^+ tau`` and the
        external torque is zero; without one it is applied as an external
        torque. ``None`` means torque-free.
    wheel_momentum0 : array_like or None
        Initial per-wheel momentum ``(m,)`` [N*m*s]; zeros by default.

    Returns
    -------
    SimulationResult
    """
    if duration <= 0.0:
        raise ValueError(f"duration must be > 0 s, got {duration}")
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0 s, got {dt}")
    if torque_fn is not None and not callable(torque_fn):
        raise TypeError("torque_fn must be callable or None")

    q = quat_normalize(quat0).reshape(4)
    w = np.asarray(omega0, dtype=float).reshape(3)
    nw = body.wheels.n_wheels if body.wheels is not None else 0
    if wheel_momentum0 is None:
        h = np.zeros(nw)
    else:
        if body.wheels is None:
            raise ValueError("wheel_momentum0 given but the body has no wheel array")
        h = np.asarray(wheel_momentum0, dtype=float).reshape(nw).copy()

    n_steps = max(1, int(math.ceil(duration / dt - 1e-12)))
    times = [0.0]
    qs, ws, hs, taus = [q.copy()], [w.copy()], [h.copy()], []

    def torque_at(t: float) -> NDArray[np.float64]:
        if torque_fn is None:
            return np.zeros(3)
        return np.asarray(torque_fn(t), dtype=float).reshape(3)

    taus.append(torque_at(0.0))

    inertia_inv = np.linalg.inv(body.inertia)
    dist = body.wheels.distribution if body.wheels is not None else None
    pinv = body.wheels.pseudo_inverse() if body.wheels is not None else None
    zeros3 = np.zeros(3)
    zeros0 = np.zeros(0)

    def rhs(tt: float, qq, ww, hh):
        tau = torque_at(tt)
        if dist is not None:
            hdot = -pinv @ tau
            der = _derivative(body.inertia, inertia_inv, dist, qq, ww, hh, hdot, zeros3)
            return (der[0], der[1], hdot)
        der = _derivative(body.inertia, inertia_inv, None, qq, ww, None, None, tau)
        return (der[0], der[1], zeros0)

    t = 0.0
    for _ in range(n_steps):
        step = min(dt, duration - t)
        if step <= 0.0:
            break
        k1 = rhs(t, q, w, h)
        k2 = rhs(t + 0.5 * step, q + 0.5 * step * k1[0], w + 0.5 * step * k1[1],
                 h + 0.5 * step * k1[2])
        k3 = rhs(t + 0.5 * step, q + 0.5 * step * k2[0], w + 0.5 * step * k2[1],
                 h + 0.5 * step * k2[2])
        k4 = rhs(t + step, q + step * k3[0], w + step * k3[1], h + step * k3[2])
        q = q + (step / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w = w + (step / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        h = h + (step / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        q = q / math.sqrt(float(q @ q))
        t += step
        times.append(t)
        qs.append(q.copy())
        ws.append(w.copy())
        hs.append(h.copy())
        taus.append(torque_at(t))

    time = np.asarray(times)
    quat = np.asarray(qs)
    omega = np.asarray(ws)
    wheel_h = np.asarray(hs) if nw else np.zeros((len(times), 0))
    torque = np.asarray(taus)
    hb = omega @ body.inertia.T
    if nw:
        hb = hb + wheel_h @ body.wheels.distribution.T
    mom = quat_rotate(quat, hb)
    if body.wheels is not None:
        hdots = -torque @ body.wheels.pseudo_inverse().T
        sat_t = np.max(np.abs(hdots), axis=1) > body.wheels.max_torque
        sat_h = np.max(np.abs(wheel_h), axis=1) > body.wheels.max_momentum
    else:
        sat_t = np.zeros(len(time), dtype=bool)
        sat_h = np.zeros(len(time), dtype=bool)
    return SimulationResult(time, quat, omega, wheel_h, torque, mom, sat_t, sat_h)


def simulate_profile(
    body: RigidBody,
    quat0: ArrayLike,
    axis_inertial: ArrayLike,
    profile: SlewProfile,
    dt: float,
    exact_torque: bool = True,
) -> SimulationResult:
    """Integrate one eigenaxis segment driven by its own feed-forward torque.

    Parameters
    ----------
    body : RigidBody
    quat0 : array_like
        Attitude at the start of the segment ``(4,)``.
    axis_inertial : array_like
        Eigenaxis in **inertial** coordinates ``(3,)``; converted to the body
        frame internally, where it is constant for an eigenaxis slew.
    profile : SlewProfile
        Timing law from :mod:`slewforge.profiles`.
    dt : float
        Integration step [s].
    exact_torque : bool
        ``True`` commands ``psi_ddot J e + psi_dot^2 (e x J e)``, the torque an
        eigenaxis slew actually needs. ``False`` commands the scalar-model
        torque ``J_e psi_ddot e`` -- the rule of thumb -- so the resulting
        pointing error can be measured. See
        `validation/validate_eigenaxis_time.py`.

    Returns
    -------
    SimulationResult
        Open loop: there is no feedback, so the terminal error is the sum of
        the model error and the integration error. The profile is integrated
        phase by phase, splitting at :attr:`~slewforge.profiles.SlewProfile.
        switch_times`, because a fixed-step RK4 that steps across the
        bang-bang torque discontinuity falls to first order.
    """
    q0 = quat_normalize(quat0).reshape(4)
    e_i = np.asarray(axis_inertial, dtype=float).reshape(3)
    e_i = e_i / float(np.linalg.norm(e_i))
    # body-frame eigenaxis: R(q0)^T e_i
    e_b = quat_rotate(np.concatenate([[q0[0]], -q0[1:]]), e_i)
    je = body.inertia @ e_b
    j_eff = float(e_b @ je)

    if profile.duration <= 0.0:
        raise ValueError("cannot simulate a zero-duration profile")

    edges = (0.0, *profile.switch_times, profile.duration)
    results: list[SimulationResult] = []
    q, w = q0, 0.0 * e_b
    h = np.zeros(body.wheels.n_wheels) if body.wheels is not None else None
    for a, b in zip(edges[:-1], edges[1:], strict=False):
        span = b - a
        if span <= 0.0:
            continue
        # Clip evaluation times a nanosecond inside the phase so a sample
        # landing exactly on a switch instant takes this phase's acceleration
        # rather than the neighbouring phase's. The clip changes the rate by a
        # relative 1e-9 and fixes a first-order error at the discontinuity.
        pad = 1e-9 * span
        lo, hi = a + pad, b - pad

        def torque_fn(t: float, a=a, lo=lo, hi=hi) -> NDArray[np.float64]:
            tt = min(max(a + t, lo), hi)
            r = float(profile.rate_at(tt))
            acc = float(profile.accel_at(tt))
            if exact_torque:
                return acc * je + (r * r) * cross3(e_b, je)
            return j_eff * acc * e_b

        res = propagate(body, q, w, span, min(dt, span), torque_fn=torque_fn,
                        wheel_momentum0=h)
        results.append(res)
        q, w = res.quat[-1], res.omega[-1]
        h = res.wheel_momentum[-1] if body.wheels is not None else None

    time = np.concatenate([results[0].time] + [r.time[1:] + e for r, e in
                                               zip(results[1:], edges[1:-1], strict=False)])
    return SimulationResult(
        time=time,
        quat=np.concatenate([results[0].quat] + [r.quat[1:] for r in results[1:]]),
        omega=np.concatenate([results[0].omega] + [r.omega[1:] for r in results[1:]]),
        wheel_momentum=np.concatenate(
            [results[0].wheel_momentum] + [r.wheel_momentum[1:] for r in results[1:]]
        ),
        body_torque=np.concatenate(
            [results[0].body_torque] + [r.body_torque[1:] for r in results[1:]]
        ),
        momentum=np.concatenate([results[0].momentum] + [r.momentum[1:] for r in results[1:]]),
        saturated_torque=np.concatenate(
            [results[0].saturated_torque] + [r.saturated_torque[1:] for r in results[1:]]
        ),
        saturated_momentum=np.concatenate(
            [results[0].saturated_momentum] + [r.saturated_momentum[1:] for r in results[1:]]
        ),
    )
