"""Truth trajectory generation: rigid-body attitude plus a position track.

Two independent truth generators are provided and can be combined into a
single :class:`Trajectory`:

1. **Attitude** — Euler's rigid-body equation driven by a user torque
   function, integrated together with the quaternion kinematics by classical
   RK4 with per-step renormalisation.
2. **Position** — either a two-body Keplerian orbit in an Earth-centred
   inertial frame, or an airborne constant-velocity / coordinated-turn track
   in a local Cartesian frame.

Everything here is deterministic given the initial condition; all randomness
lives in :mod:`navbench.sensors`.

REFERENCES
* Wertz, J. R. (1978), *Spacecraft Attitude Determination and Control*,
  Reidel — Eq. (16-3) Euler's equations, §16.2 torque-free motion.
* Markley, F. L. & Crassidis, J. L. (2014), *Fundamentals of Spacecraft
  Attitude Determination and Control*, Springer — Ch. 3 kinematics and
  dynamics.
* Vallado, D. A. (2013), *Fundamentals of Astrodynamics and Applications*,
  4th ed., Microcosm — §1.3 the two-body problem, §2.2 Kepler propagation.
* Petit, G. & Luzum, B. (eds.) (2010), *IERS Conventions (2010)*, IERS
  Technical Note 36, Table 1.1 — GM_Earth = 3.986004418e14 m³/s².

VALIDITY.  The orbit model is pure two-body: no J2, no drag, no third body,
no solar radiation pressure.  It is adequate as a *navigation-filter test
signal* over the minutes-to-hours spans used here, and is **not** an
ephemeris-grade propagator.  The airborne model is a flat-Earth kinematic
track with no wind, no gravity model and no Earth rotation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .attitude import euler_moment_derivative, quat_derivative, quat_normalize

__all__ = [
    "MU_EARTH",
    "R_EARTH",
    "AttitudeTruth",
    "OrbitTruth",
    "AirborneTruth",
    "Trajectory",
    "attitude_trajectory",
    "orbit_trajectory",
    "airborne_trajectory",
    "circular_orbit_state",
]

#: Earth gravitational parameter [m³/s²] — IERS Conventions (2010), Table 1.1.
MU_EARTH = 3.986004418e14
#: Earth equatorial radius [m] — WGS-84 / IERS Conventions (2010).
R_EARTH = 6378137.0

TorqueFn = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


def _check_dt_n(dt: float, n_steps: int) -> tuple[float, int]:
    step = float(dt)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f"dt must be finite and > 0 s, got {dt!r}")
    n = int(n_steps)
    if n <= 0:
        raise ValueError(f"n_steps must be >= 1, got {n_steps!r}")
    return step, n


@dataclass(frozen=True)
class AttitudeTruth:
    """Sampled rigid-body attitude history.

    Attributes
    ----------
    t : ndarray, shape (N+1,)
        Sample times [s], ``t[0] = 0``.
    quat : ndarray, shape (N+1, 4)
        Body-to-inertial unit quaternions, scalar first.
    omega : ndarray, shape (N+1, 3)
        Body angular rate [rad/s].
    torque : ndarray, shape (N+1, 3)
        Applied body torque at each sample [N·m].
    inertia : ndarray, shape (3, 3)
        Inertia tensor used [kg·m²].
    """

    t: NDArray[np.float64]
    quat: NDArray[np.float64]
    omega: NDArray[np.float64]
    torque: NDArray[np.float64]
    inertia: NDArray[np.float64]

    @property
    def n_steps(self) -> int:
        """Number of propagation steps (one fewer than the number of samples)."""
        return int(self.t.size - 1)

    def angular_momentum(self) -> NDArray[np.float64]:
        """Inertial-frame angular momentum ``R(q) J ω`` [kg·m²/s], shape (N+1, 3).

        For a torque-free rigid body this is conserved exactly; the residual
        drift is a direct integrator-accuracy diagnostic (validation v4).
        """
        from .attitude import dcm_from_quat

        return np.array(
            [
                dcm_from_quat(q) @ (self.inertia @ w)
                for q, w in zip(self.quat, self.omega, strict=True)
            ]
        )

    def kinetic_energy(self) -> NDArray[np.float64]:
        """Rotational kinetic energy ``½ ωᵀ J ω`` [J], shape (N+1,).

        Conserved exactly for torque-free motion (Wertz 1978 §16.2).
        """
        return 0.5 * np.einsum("ki,ij,kj->k", self.omega, self.inertia, self.omega)

    def interval_rate(self) -> NDArray[np.float64]:
        """Effective constant body rate over each interval [rad/s], shape (N, 3).

        ``ω̄_k = rotvec(q_k* ⊗ q_{k+1}) / Δt`` — the unique constant rate that,
        held over the interval, reproduces the attitude change exactly.

        **Use this, not** ``omega[:-1]`` **or** ``omega[1:]``, **as the true
        rate driving a gyro model.**  A rate-integrating gyro reports the
        integrated angle increment over the sampling interval divided by the
        interval, not an instantaneous rate (Farrenkopf 1978; Markley &
        Crassidis 2014 §4.7.2), and a filter that propagates with
        ``q̂ ⊗ δq(ω̂ Δt)`` assumes exactly this quantity.  Feeding it an
        endpoint sample instead injects a deterministic O(ω̇ Δt²/2) attitude
        error that is *not* in the filter's noise model and shows up
        immediately as a NEES/NIS consistency failure — measured at
        3.2e-5 rad/step for the default scenario in
        ``validation/v4_mekf_quaternion.py``.
        """
        from .attitude import quat_conjugate, quat_multiply, small_angle_from_quat

        dt = float(self.t[1] - self.t[0])
        return (
            np.array(
                [
                    small_angle_from_quat(
                        quat_multiply(quat_conjugate(self.quat[k]), self.quat[k + 1])
                    )
                    for k in range(self.n_steps)
                ]
            )
            / dt
        )


def attitude_trajectory(
    *,
    inertia: ArrayLike,
    quat0: ArrayLike,
    omega0: ArrayLike,
    dt: float,
    n_steps: int,
    torque_fn: TorqueFn | None = None,
) -> AttitudeTruth:
    """Integrate rigid-body attitude with RK4.

    The coupled state is ``(q, ω)`` with
    ``q̇ = ½ q ⊗ [0, ω]`` and ``ω̇ = J⁻¹(τ(t, q, ω) − ω × Jω)``.
    Classical 4th-order Runge-Kutta; the quaternion is renormalised after each
    completed step (never inside a stage), which is the standard treatment —
    the normalisation error removed is O(dt⁵) per step and does not degrade
    the RK4 order.

    Parameters
    ----------
    inertia : array_like, shape (3, 3)
        Body inertia tensor [kg·m²], symmetric positive definite.
    quat0 : array_like, shape (4,)
        Initial body-to-inertial quaternion (normalised on entry).
    omega0 : array_like, shape (3,)
        Initial body rate [rad/s].
    dt : float
        Integration/sample step [s], > 0.
    n_steps : int
        Number of steps; the returned arrays have ``n_steps + 1`` samples.
    torque_fn : callable, optional
        ``torque_fn(t, q, omega) -> (3,)`` external body torque [N·m].
        Default: torque-free motion.

    Returns
    -------
    AttitudeTruth
    """
    step, n = _check_dt_n(dt, n_steps)
    j = np.asarray(inertia, dtype=float)
    q = quat_normalize(quat0)
    w = np.asarray(omega0, dtype=float).reshape(3).copy()
    if not np.all(np.isfinite(w)):
        raise ValueError("omega0 must be finite")
    tau_fn: TorqueFn = torque_fn if torque_fn is not None else (lambda _t, _q, _w: np.zeros(3))

    def deriv(t: float, qq: NDArray[np.float64], ww: NDArray[np.float64]):
        tau = np.asarray(tau_fn(t, qq, ww), dtype=float).reshape(3)
        return quat_derivative(qq, ww), euler_moment_derivative(ww, j, tau)

    ts = np.zeros(n + 1)
    qs = np.zeros((n + 1, 4))
    ws = np.zeros((n + 1, 3))
    taus = np.zeros((n + 1, 3))
    qs[0], ws[0] = q, w
    taus[0] = np.asarray(tau_fn(0.0, q, w), dtype=float).reshape(3)

    for k in range(n):
        t = k * step
        k1q, k1w = deriv(t, q, w)
        k2q, k2w = deriv(t + 0.5 * step, q + 0.5 * step * k1q, w + 0.5 * step * k1w)
        k3q, k3w = deriv(t + 0.5 * step, q + 0.5 * step * k2q, w + 0.5 * step * k2w)
        k4q, k4w = deriv(t + step, q + step * k3q, w + step * k3w)
        q = quat_normalize(q + (step / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q))
        w = w + (step / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
        ts[k + 1] = t + step
        qs[k + 1], ws[k + 1] = q, w
        taus[k + 1] = np.asarray(tau_fn(ts[k + 1], q, w), dtype=float).reshape(3)

    return AttitudeTruth(t=ts, quat=qs, omega=ws, torque=taus, inertia=j)


@dataclass(frozen=True)
class OrbitTruth:
    """Sampled two-body orbital position/velocity history in an inertial frame."""

    t: NDArray[np.float64]
    position: NDArray[np.float64]  # (N+1, 3) [m]
    velocity: NDArray[np.float64]  # (N+1, 3) [m/s]
    mu: float  # [m³/s²]

    def specific_energy(self) -> NDArray[np.float64]:
        """Specific orbital energy ``v²/2 − μ/r`` [J/kg], conserved in two-body motion."""
        r = np.linalg.norm(self.position, axis=1)
        v = np.linalg.norm(self.velocity, axis=1)
        return 0.5 * v * v - self.mu / r

    def angular_momentum(self) -> NDArray[np.float64]:
        """Specific angular momentum ``r × v`` [m²/s], shape (N+1, 3); conserved."""
        return np.cross(self.position, self.velocity)


def circular_orbit_state(
    altitude_m: float, inclination_rad: float = 0.0, mu: float = MU_EARTH
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Position/velocity of a circular orbit at the ascending node.

    ``r = R_E + h``; ``v = sqrt(μ/r)`` (Vallado 2013 Eq. (1-32)).  The state is
    placed on the +X axis with velocity in the X-Y plane rotated by the
    inclination about X.

    Returns
    -------
    (position [m], velocity [m/s])
    """
    h = float(altitude_m)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(f"altitude_m must be finite and > 0, got {altitude_m!r}")
    inc = float(inclination_rad)
    if not np.isfinite(inc):
        raise ValueError(f"inclination_rad must be finite, got {inclination_rad!r}")
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError(f"mu must be finite and > 0, got {mu!r}")
    r = R_EARTH + h
    v = np.sqrt(mu / r)
    return (
        np.array([r, 0.0, 0.0]),
        np.array([0.0, v * np.cos(inc), v * np.sin(inc)]),
    )


def orbit_trajectory(
    *,
    position0: ArrayLike,
    velocity0: ArrayLike,
    dt: float,
    n_steps: int,
    mu: float = MU_EARTH,
) -> OrbitTruth:
    """Propagate the two-body problem ``r̈ = −μ r/|r|³`` with RK4.

    Vallado 2013 §1.3 Eq. (1-14).  Units: metres, seconds.  Validity: point-mass
    central body only; no J2, drag, third-body or SRP perturbations.  The
    integrator is fixed-step RK4, so ``dt`` must be small relative to the orbit
    period (``dt <= T/500`` keeps the specific-energy drift below ~1e-9
    relative over one revolution — measured in validation v5).
    """
    step, n = _check_dt_n(dt, n_steps)
    if not np.isfinite(mu) or mu <= 0.0:
        raise ValueError(f"mu must be finite and > 0, got {mu!r}")
    r = np.asarray(position0, dtype=float).reshape(3).copy()
    v = np.asarray(velocity0, dtype=float).reshape(3).copy()
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
        raise ValueError("position0 and velocity0 must be finite")
    if float(np.linalg.norm(r)) < 1.0:
        raise ValueError("position0 is at (or within 1 m of) the central body singularity")

    def acc(rr: NDArray[np.float64]) -> NDArray[np.float64]:
        return -mu * rr / float(np.linalg.norm(rr)) ** 3

    ts = np.zeros(n + 1)
    rs = np.zeros((n + 1, 3))
    vs = np.zeros((n + 1, 3))
    rs[0], vs[0] = r, v
    for k in range(n):
        k1r, k1v = v, acc(r)
        k2r, k2v = v + 0.5 * step * k1v, acc(r + 0.5 * step * k1r)
        k3r, k3v = v + 0.5 * step * k2v, acc(r + 0.5 * step * k2r)
        k4r, k4v = v + step * k3v, acc(r + step * k3r)
        r = r + (step / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        v = v + (step / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        ts[k + 1] = (k + 1) * step
        rs[k + 1], vs[k + 1] = r, v
    return OrbitTruth(t=ts, position=rs, velocity=vs, mu=float(mu))


@dataclass(frozen=True)
class AirborneTruth:
    """Sampled flat-Earth airborne track (position, velocity, specific force)."""

    t: NDArray[np.float64]
    position: NDArray[np.float64]  # (N+1, 3) [m], local ENU-like Cartesian
    velocity: NDArray[np.float64]  # (N+1, 3) [m/s]
    acceleration: NDArray[np.float64]  # (N+1, 3) [m/s²]


def airborne_trajectory(
    *,
    position0: ArrayLike,
    velocity0: ArrayLike,
    dt: float,
    n_steps: int,
    turn_rate_rad_s: float = 0.0,
    climb_rate_m_s2: float = 0.0,
) -> AirborneTruth:
    """Constant-velocity or coordinated-turn airborne track (flat Earth, no wind).

    The horizontal velocity rotates about the vertical (+Z) axis at
    ``turn_rate_rad_s`` — the standard coordinated-turn kinematic model
    (Bar-Shalom, Li & Kirubarajan 2001, *Estimation with Applications to
    Tracking and Navigation*, §11.7):

    ``a = [-Ω v_y, +Ω v_x, climb_rate]``

    Integrated analytically per step for the horizontal plane (exact for
    constant Ω) and by closed form in Z.

    Units: m, m/s, m/s², rad/s.  Validity: flat non-rotating Earth, constant
    turn rate over each step, no aerodynamic model.
    """
    step, n = _check_dt_n(dt, n_steps)
    omega = float(turn_rate_rad_s)
    climb = float(climb_rate_m_s2)
    if not np.isfinite(omega):
        raise ValueError(f"turn_rate_rad_s must be finite, got {turn_rate_rad_s!r}")
    if not np.isfinite(climb):
        raise ValueError(f"climb_rate_m_s2 must be finite, got {climb_rate_m_s2!r}")
    p = np.asarray(position0, dtype=float).reshape(3).copy()
    v = np.asarray(velocity0, dtype=float).reshape(3).copy()
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(v))):
        raise ValueError("position0 and velocity0 must be finite")

    ts = np.zeros(n + 1)
    ps = np.zeros((n + 1, 3))
    vs = np.zeros((n + 1, 3))
    accs = np.zeros((n + 1, 3))

    def accel(vv: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([-omega * vv[1], omega * vv[0], climb])

    ps[0], vs[0], accs[0] = p, v, accel(v)
    for k in range(n):
        if abs(omega) < 1e-12:
            dp_xy = v[:2] * step
            v_xy = v[:2].copy()
        else:
            s, c = np.sin(omega * step), np.cos(omega * step)
            dp_xy = np.array(
                [
                    (v[0] * s - v[1] * (1.0 - c)) / omega,
                    (v[0] * (1.0 - c) + v[1] * s) / omega,
                ]
            )
            v_xy = np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])
        p = np.array(
            [p[0] + dp_xy[0], p[1] + dp_xy[1], p[2] + v[2] * step + 0.5 * climb * step * step]
        )
        v = np.array([v_xy[0], v_xy[1], v[2] + climb * step])
        ts[k + 1] = (k + 1) * step
        ps[k + 1], vs[k + 1], accs[k + 1] = p, v, accel(v)
    return AirborneTruth(t=ts, position=ps, velocity=vs, acceleration=accs)


@dataclass(frozen=True)
class Trajectory:
    """A combined attitude + position truth on a common time grid.

    Raises
    ------
    ValueError
        If the two components are not on the same time grid.
    """

    attitude: AttitudeTruth
    position: NDArray[np.float64]  # (N+1, 3) [m]
    velocity: NDArray[np.float64]  # (N+1, 3) [m/s]
    acceleration: NDArray[np.float64]  # (N+1, 3) [m/s²]

    def __post_init__(self) -> None:
        n = self.attitude.t.size
        for name, arr in (
            ("position", self.position),
            ("velocity", self.velocity),
            ("acceleration", self.acceleration),
        ):
            if arr.shape != (n, 3):
                raise ValueError(
                    f"{name} must have shape ({n}, 3) to match the attitude grid, "
                    f"got {arr.shape}"
                )

    @property
    def t(self) -> NDArray[np.float64]:
        """Common sample times [s]."""
        return self.attitude.t

    @property
    def dt(self) -> float:
        """Sample step [s]."""
        return float(self.attitude.t[1] - self.attitude.t[0])
