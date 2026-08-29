r"""Truth trajectory generation: rigid-body attitude plus a position track.

Two independent truth generators are provided; the bench runs estimators
against them.

1. Attitude — Euler's rigid-body equation with applied torque
--------------------------------------------------------------
.. math::
    J \dot\omega = \tau - \omega \times (J \omega), \qquad
    \dot q = \tfrac12 q \otimes [0, \omega]

``J`` inertia tensor [kg m²], ``ω`` body rate [rad/s], ``τ`` external torque in
body axes [N m], ``q`` attitude (body → inertial, see :mod:`navbench.quaternion`).

Source: Markley, F. L. and Crassidis, J. L. (2014), *Fundamentals of Spacecraft
Attitude Determination and Control*, Springer, Eqs. (3.81) and (3.21);
Hughes, P. C. (1986), *Spacecraft Attitude Dynamics*, Wiley, Ch. 4.

Assumptions: rigid body, no flexible modes, no momentum wheels, torque supplied
by the caller as a function of time. Validity: any body rate; the integrator is
fixed-step RK4, so the step must resolve the fastest rate — the module enforces
``dt · ‖ω‖ < 0.5 rad`` at generation time and raises otherwise.

Energy/momentum invariants (torque-free case) used as validation checks:
``‖J ω‖`` expressed in inertial axes is conserved, and the rotational kinetic
energy ``T = ½ ωᵀ J ω`` is conserved.

2. Position — restricted two-body orbit
---------------------------------------
.. math::
    \ddot{\mathbf r} = -\mu \frac{\mathbf r}{\|\mathbf r\|^{3}}

with ``μ = 3.986004418e14 m³/s²`` (WGS-84 / EGM-96 value, NIMA TR8350.2, 3rd ed.,
1997). Assumptions: point-mass central body, no J2, no drag, no third body.
Validity: any bound Keplerian orbit; RK4 at the default 1 s step keeps the
specific-energy drift below 1e-9 relative over a single orbit (measured in
``validation/v6_failure_and_perf.py``).

An alternative **airborne** track (constant-speed coordinated turn in the local
horizontal plane, constant climb rate) is provided for lower-dynamics cases.
Source for the coordinated-turn kinematics: Bar-Shalom, Y., Rong Li, X. and
Kirubarajan, T. (2001), *Estimation with Applications to Tracking and
Navigation*, Wiley, §11.7.

All arrays are returned with time along axis 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .quaternion import quat_multiply, quat_normalize, skew

__all__ = [
    "MU_EARTH",
    "AttitudeTruth",
    "PositionTruth",
    "generate_attitude",
    "generate_orbit",
    "generate_coordinated_turn",
    "angular_momentum_inertial",
    "kinetic_energy",
]

#: Earth gravitational parameter [m³/s²]. NIMA TR8350.2 3rd ed. (1997), WGS-84.
MU_EARTH: float = 3.986004418e14


@dataclass(frozen=True)
class AttitudeTruth:
    """Sampled rigid-body attitude truth.

    Attributes
    ----------
    t : ndarray, shape (K,)
        Sample times [s].
    q : ndarray, shape (K, 4)
        Attitude quaternion, scalar-first, body → inertial.
    omega : ndarray, shape (K, 3)
        Body angular rate [rad/s].
    torque : ndarray, shape (K, 3)
        Applied torque in body axes [N m] evaluated at ``t``.
    inertia : ndarray, shape (3, 3)
        Inertia tensor [kg m²].
    """

    t: NDArray[np.float64]
    q: NDArray[np.float64]
    omega: NDArray[np.float64]
    torque: NDArray[np.float64]
    inertia: NDArray[np.float64]

    @property
    def dt(self) -> float:
        """Sample interval [s]."""
        return float(self.t[1] - self.t[0])


@dataclass(frozen=True)
class PositionTruth:
    """Sampled position/velocity truth.

    Attributes
    ----------
    t : ndarray, shape (K,)
        Sample times [s].
    pos : ndarray, shape (K, 3)
        Position [m] in the inertial frame.
    vel : ndarray, shape (K, 3)
        Velocity [m/s] in the inertial frame.
    acc : ndarray, shape (K, 3)
        Non-gravitational plus gravitational specific force bookkeeping is left
        to :mod:`navbench.sensors`; this is the total inertial acceleration
        [m/s²].
    """

    t: NDArray[np.float64]
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]
    acc: NDArray[np.float64]

    @property
    def dt(self) -> float:
        """Sample interval [s]."""
        return float(self.t[1] - self.t[0])


def _validate_inertia(inertia: ArrayLike) -> NDArray[np.float64]:
    j = np.asarray(inertia, dtype=float)
    if j.shape == (3,):
        j = np.diag(j)
    if j.shape != (3, 3):
        raise ValueError(f"inertia must be 3x3 or a length-3 diagonal, got shape {np.shape(inertia)}")
    if not np.allclose(j, j.T, atol=1e-12):
        raise ValueError("inertia tensor must be symmetric")
    eig = np.linalg.eigvalsh(j)
    if float(eig.min()) <= 0.0:
        raise ValueError(f"inertia tensor must be positive definite; eigenvalues {eig}")
    # Triangle inequality on principal moments (a physical rigid body must satisfy it).
    a, b, c = np.sort(eig)
    if a + b < c - 1e-12 * c:
        raise ValueError(
            f"principal moments {eig} violate the triangle inequality "
            "(no physical rigid body has this inertia tensor)"
        )
    return j


def angular_momentum_inertial(truth: AttitudeTruth) -> NDArray[np.float64]:
    """Angular momentum ``R(q) J ω`` in inertial axes [kg m²/s], shape (K, 3)."""
    from .quaternion import quat_to_dcm

    hb = truth.omega @ truth.inertia.T
    return np.array([quat_to_dcm(truth.q[k]) @ hb[k] for k in range(len(truth.t))])


def kinetic_energy(truth: AttitudeTruth) -> NDArray[np.float64]:
    """Rotational kinetic energy ``½ ωᵀ J ω`` [J], shape (K,)."""
    return 0.5 * np.einsum("ki,ij,kj->k", truth.omega, truth.inertia, truth.omega)


def _attitude_derivative(
    q: NDArray[np.float64],
    w: NDArray[np.float64],
    inertia: NDArray[np.float64],
    inertia_inv: NDArray[np.float64],
    tau: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    qdot = 0.5 * quat_multiply(q, np.concatenate(([0.0], w)))
    wdot = inertia_inv @ (tau - skew(w) @ (inertia @ w))
    return qdot, wdot


def generate_attitude(
    *,
    inertia: ArrayLike,
    q0: ArrayLike,
    omega0: ArrayLike,
    dt: float,
    n_steps: int,
    torque: Callable[[float], ArrayLike] | ArrayLike | None = None,
) -> AttitudeTruth:
    r"""Propagate rigid-body attitude with RK4.

    Parameters
    ----------
    inertia : array_like, shape (3, 3) or (3,)
        Inertia tensor [kg m²] (a length-3 input is read as the diagonal).
    q0 : array_like, shape (4,)
        Initial attitude quaternion, scalar-first.
    omega0 : array_like, shape (3,)
        Initial body rate [rad/s].
    dt : float
        Step [s]; must be positive.
    n_steps : int
        Number of steps; ``n_steps + 1`` samples are returned.
    torque : callable or array_like or None
        External torque in body axes [N m]. A callable is evaluated at the RK4
        stage times; an array is used as a constant; ``None`` means torque-free.

    Returns
    -------
    AttitudeTruth

    Raises
    ------
    ValueError
        On invalid inertia, non-positive ``dt``/``n_steps``, or if the rate is
        too fast for the step (``dt·‖ω‖ >= 0.5`` rad at any sample), which would
        make the fixed-step RK4 attitude error dominate the sensor errors.
    """
    j = _validate_inertia(inertia)
    jinv = np.linalg.inv(j)
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    q = quat_normalize(q0)
    w = np.asarray(omega0, dtype=float).reshape(-1)
    if w.shape != (3,):
        raise ValueError(f"omega0 must be a 3-vector, got shape {np.shape(omega0)}")

    if torque is None:
        def tau_fn(_t: float) -> NDArray[np.float64]:
            return np.zeros(3)
    elif callable(torque):
        def tau_fn(t: float) -> NDArray[np.float64]:
            return np.asarray(torque(t), dtype=float).reshape(3)
    else:
        tau_const = np.asarray(torque, dtype=float).reshape(3)

        def tau_fn(_t: float) -> NDArray[np.float64]:
            return tau_const

    ts = np.arange(n_steps + 1, dtype=float) * dt
    qs = np.zeros((n_steps + 1, 4))
    ws = np.zeros((n_steps + 1, 3))
    taus = np.zeros((n_steps + 1, 3))
    qs[0], ws[0], taus[0] = q, w, tau_fn(0.0)

    for k in range(n_steps):
        t = ts[k]
        k1q, k1w = _attitude_derivative(q, w, j, jinv, tau_fn(t))
        k2q, k2w = _attitude_derivative(q + 0.5 * dt * k1q, w + 0.5 * dt * k1w, j, jinv,
                                        tau_fn(t + 0.5 * dt))
        k3q, k3w = _attitude_derivative(q + 0.5 * dt * k2q, w + 0.5 * dt * k2w, j, jinv,
                                        tau_fn(t + 0.5 * dt))
        k4q, k4w = _attitude_derivative(q + dt * k3q, w + dt * k3w, j, jinv, tau_fn(t + dt))
        q = quat_normalize(q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q))
        w = w + (dt / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w)
        qs[k + 1], ws[k + 1], taus[k + 1] = q, w, tau_fn(ts[k + 1])

    wmax = float(np.max(np.linalg.norm(ws, axis=1)))
    if dt * wmax >= 0.5:
        raise ValueError(
            f"dt·max‖ω‖ = {dt * wmax:.3f} rad >= 0.5: the RK4 step does not resolve the "
            "attitude motion. Reduce dt or the torque."
        )
    return AttitudeTruth(t=ts, q=qs, omega=ws, torque=taus, inertia=j)


def _two_body_derivative(state: NDArray[np.float64], mu: float) -> NDArray[np.float64]:
    r = state[:3]
    rn = float(np.linalg.norm(r))
    return np.concatenate((state[3:], -mu * r / (rn ** 3)))


def generate_orbit(
    *,
    r0: ArrayLike,
    v0: ArrayLike,
    dt: float,
    n_steps: int,
    mu: float = MU_EARTH,
) -> PositionTruth:
    r"""Propagate a restricted two-body orbit with RK4.

    Parameters
    ----------
    r0, v0 : array_like, shape (3,)
        Initial inertial position [m] and velocity [m/s].
    dt : float
        Step [s].
    n_steps : int
        Number of steps.
    mu : float
        Gravitational parameter [m³/s²]; defaults to :data:`MU_EARTH`.

    Raises
    ------
    ValueError
        For non-positive ``dt``, ``mu``, or a starting radius below 6.0e6 m
        (inside the Earth — almost always a unit error).
    """
    r = np.asarray(r0, dtype=float).reshape(-1)
    v = np.asarray(v0, dtype=float).reshape(-1)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("r0 and v0 must both be 3-vectors")
    if dt <= 0.0 or n_steps < 1:
        raise ValueError(f"need dt > 0 and n_steps >= 1, got dt={dt}, n_steps={n_steps}")
    if mu <= 0.0:
        raise ValueError(f"mu must be > 0, got {mu}")
    if float(np.linalg.norm(r)) < 6.0e6:
        raise ValueError(
            f"initial radius {np.linalg.norm(r):.3e} m is below 6.0e6 m — check units (metres "
            "expected)"
        )

    ts = np.arange(n_steps + 1, dtype=float) * dt
    xs = np.zeros((n_steps + 1, 6))
    xs[0] = np.concatenate((r, v))
    for k in range(n_steps):
        s = xs[k]
        k1 = _two_body_derivative(s, mu)
        k2 = _two_body_derivative(s + 0.5 * dt * k1, mu)
        k3 = _two_body_derivative(s + 0.5 * dt * k2, mu)
        k4 = _two_body_derivative(s + dt * k3, mu)
        xs[k + 1] = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    pos, vel = xs[:, :3], xs[:, 3:]
    rn = np.linalg.norm(pos, axis=1, keepdims=True)
    acc = -mu * pos / rn ** 3
    return PositionTruth(t=ts, pos=pos, vel=vel, acc=acc)


def generate_coordinated_turn(
    *,
    speed: float,
    turn_rate: float,
    climb_rate: float = 0.0,
    heading0: float = 0.0,
    p0: ArrayLike = (0.0, 0.0, 0.0),
    dt: float = 1.0,
    n_steps: int = 200,
) -> PositionTruth:
    r"""Constant-speed coordinated turn with constant climb rate (closed form).

    .. math::
        \psi(t) &= \psi_0 + \Omega t \\
        \dot x &= V \cos\psi,\quad \dot y = V \sin\psi,\quad \dot z = w

    Units: ``speed`` m/s, ``turn_rate`` rad/s, ``climb_rate`` m/s, ``heading0``
    rad. Source: Bar-Shalom, Li & Kirubarajan (2001) §11.7 (coordinated turn
    model). Assumption: flat non-rotating Earth over the track length; valid for
    airborne tracks of a few tens of km.

    The straight-line limit ``Ω → 0`` is handled analytically (no 0/0).
    """
    if speed < 0.0:
        raise ValueError(f"speed must be >= 0, got {speed}")
    if dt <= 0.0 or n_steps < 1:
        raise ValueError(f"need dt > 0 and n_steps >= 1, got dt={dt}, n_steps={n_steps}")
    p = np.asarray(p0, dtype=float).reshape(-1)
    if p.shape != (3,):
        raise ValueError("p0 must be a 3-vector")
    ts = np.arange(n_steps + 1, dtype=float) * dt
    psi = heading0 + turn_rate * ts
    if abs(turn_rate) < 1e-12:
        x = p[0] + speed * np.cos(heading0) * ts
        y = p[1] + speed * np.sin(heading0) * ts
    else:
        x = p[0] + speed / turn_rate * (np.sin(psi) - np.sin(heading0))
        y = p[1] - speed / turn_rate * (np.cos(psi) - np.cos(heading0))
    z = p[2] + climb_rate * ts
    pos = np.column_stack((x, y, z))
    vel = np.column_stack((speed * np.cos(psi), speed * np.sin(psi), np.full_like(ts, climb_rate)))
    acc = np.column_stack(
        (-speed * turn_rate * np.sin(psi), speed * turn_rate * np.cos(psi), np.zeros_like(ts))
    )
    return PositionTruth(t=ts, pos=pos, vel=vel, acc=acc)


@dataclass(frozen=True)
class TorqueProfile:
    """A small library of reproducible torque profiles [N m].

    ``constant`` plus a sinusoid on each axis:
    ``τ(t) = c + a ⊙ sin(2π f t + φ)``.
    """

    constant: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    amplitude: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    frequency: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    phase: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))

    def __call__(self, t: float) -> NDArray[np.float64]:
        """Torque [N m] at time ``t`` [s]."""
        return np.asarray(self.constant, dtype=float) + np.asarray(
            self.amplitude, dtype=float
        ) * np.sin(2.0 * np.pi * np.asarray(self.frequency, dtype=float) * t
                   + np.asarray(self.phase, dtype=float))
