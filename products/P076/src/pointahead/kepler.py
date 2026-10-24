"""Two-body propagation by universal variables, and the trajectory protocol.

The point-ahead reference in :mod:`pointahead.angle` needs the remote
terminal's position at the retarded and advanced epochs, not just at the
nominal epoch. That needs a trajectory, and the trajectory model is a declared
input, not an implementation detail: a linear extrapolation and a two-body
propagation give different answers for the light-time geometry, and the
difference is one of the budget lines.

Two trajectory models are provided.

``LinearTrajectory``
    Constant velocity. Chosen deliberately, not as a toy: for a straight-line
    trajectory the light-time equation is a quadratic with a closed-form root,
    so the iterated solver in :mod:`pointahead.lighttime` can be checked
    against an exact answer rather than against itself.

``KeplerTrajectory``
    Two-body Keplerian motion about a central mass, propagated with universal
    variables. Universal variables are used rather than classical elements
    because the classical set is singular at zero eccentricity and zero
    inclination, which is exactly the near-circular, low-inclination regime a
    crosslink scenario sits in.

Universal-variable formulation and the Stumpff functions ``c2``, ``c3`` follow
Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
Microcosm Press 2013, Algorithm 8 (KEPLER) and section 2.2. Structure only: no
numerical value in this repository is taken from that book. The propagator is
checked in ``validation/validate_kepler.py`` against ``scipy.integrate.solve_ivp``
with DOP853 at tight tolerance, which is an independent integration of the same
equation of motion, and against the specific energy and angular momentum, which
are conserved exactly in the two-body problem.

Validity range: elliptical and hyperbolic orbits with a non-zero angular
momentum and a positive gravitational parameter. Parabolic orbits
(``|1/a| < 1e-12 1/m``) are refused rather than approximated. No oblateness,
no drag, no third bodies, no solar radiation pressure. Over the light time of
an inter-satellite link -- tens of milliseconds -- the difference between
two-body and a full force model is far below the angles this package resolves;
over an ephemeris staleness of minutes to hours it is not, and that is recorded
as a limitation rather than modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .geometry import as_vector

MU_EARTH = 3.986004418e14
"""Earth gravitational parameter, m^3/s^2 (WGS-84 / EGM-96 value)."""

R_EARTH = 6378137.0
"""WGS-84 semi-major axis of the Earth ellipsoid, m. Used only for sanity
checks on scenario altitudes, never in the dynamics."""

_PARABOLIC_TOLERANCE = 1.0e-12


def stumpff_c2(psi: float) -> float:
    """Stumpff function ``c2(psi) = (1 - cos sqrt(psi)) / psi``.

    Series expansion is used for ``|psi| < 1e-6`` because the closed form is a
    difference of two nearly equal numbers there and loses most of its
    significant digits; the series is ``1/2 - psi/24 + psi^2/720``.
    """
    value = float(psi)
    if value > 1.0e-6:
        root = np.sqrt(value)
        return float((1.0 - np.cos(root)) / value)
    if value < -1.0e-6:
        root = np.sqrt(-value)
        return float((1.0 - np.cosh(root)) / value)
    return float(0.5 - value / 24.0 + value * value / 720.0)


def stumpff_c3(psi: float) -> float:
    """Stumpff function ``c3(psi) = (sqrt(psi) - sin sqrt(psi)) / psi^1.5``.

    Series ``1/6 - psi/120 + psi^2/5040`` near zero, for the same reason as
    :func:`stumpff_c2`.
    """
    value = float(psi)
    if value > 1.0e-6:
        root = np.sqrt(value)
        return float((root - np.sin(root)) / (root * root * root))
    if value < -1.0e-6:
        root = np.sqrt(-value)
        return float((np.sinh(root) - root) / (root * root * root))
    return float(1.0 / 6.0 - value / 120.0 + value * value / 5040.0)


@dataclass(frozen=True)
class KeplerStep:
    """Result of one universal-variable propagation.

    Attributes
    ----------
    position_m, velocity_mps
        The propagated state.
    iterations
        Newton iterations used on the universal anomaly.
    revolutions_removed
        Whole orbital periods subtracted before iterating, for an elliptical
        orbit. Zero for a hyperbolic one, which is not periodic.
    chi
        Converged universal anomaly, in ``sqrt(m)``.
    fg_residual
        ``|f * gdot - fdot * g - 1|``. Exactly zero in exact arithmetic for any
        two-body propagation, so it is a pure arithmetic health check and is
        reported rather than asserted bit-exactly.
    """

    position_m: np.ndarray
    velocity_mps: np.ndarray
    iterations: int
    chi: float
    fg_residual: float
    revolutions_removed: float = 0.0


def propagate_two_body(
    position_m: object,
    velocity_mps: object,
    dt_s: float,
    mu: float = MU_EARTH,
    tol: float = 1.0e-10,
    max_iter: int = 60,
) -> KeplerStep:
    """Propagate a state by ``dt_s`` seconds under two-body gravity.

    Parameters
    ----------
    position_m, velocity_mps
        State in a central-body-centred inertial frame, metres and m/s.
    dt_s
        Signed propagation interval in seconds. Negative is supported and is
        used by the receive-side light-time solution.
    mu
        Gravitational parameter in m^3/s^2.
    tol
        Convergence tolerance on the universal anomaly, in ``sqrt(m)``.
    max_iter
        Newton iteration cap.

    Raises
    ------
    ValueError
        On a non-positive ``mu``, a zero position, a near-parabolic orbit, or a
        failure to converge within ``max_iter``.
    """
    r0 = as_vector(position_m, "position_m")
    v0 = as_vector(velocity_mps, "velocity_mps")
    dt = float(dt_s)
    if not np.isfinite(dt):
        raise ValueError(f"dt_s must be finite, got {dt_s!r}")
    if not (np.isfinite(mu) and mu > 0.0):
        raise ValueError(f"mu must be a positive finite gravitational parameter, got {mu!r}")
    r0_magnitude = float(np.linalg.norm(r0))
    if r0_magnitude == 0.0:
        raise ValueError("position_m must be non-zero for a two-body propagation")
    if dt == 0.0:
        return KeplerStep(r0.copy(), v0.copy(), 0, 0.0, 0.0)

    sqrt_mu = float(np.sqrt(mu))
    alpha = -float(np.dot(v0, v0)) / mu + 2.0 / r0_magnitude
    if abs(alpha) < _PARABOLIC_TOLERANCE:
        raise ValueError(
            "near-parabolic orbit (|1/a| < 1e-12 1/m) is outside the validity range of "
            "propagate_two_body; the universal-variable initial guess is not defined there"
        )

    # Reduce an elliptical propagation to the half-period nearest zero before
    # iterating. The state of a two-body elliptical orbit is exactly periodic, so
    # subtracting a whole number of periods changes nothing physically, and it
    # keeps the universal anomaly small. Without it the Newton iteration on chi
    # has to cross many revolutions from a linear initial guess, and it does not
    # always get there: Hypothesis falsified the energy-conservation property at
    # radius 4.0e5 m, 0.6 of circular speed and dt = 3354.9375 s, which is 88
    # revolutions of a 37.9 s orbit, with "did not converge in 60 iterations".
    # That counterexample is pinned in tests/test_regressions.py. There is no
    # equivalent reduction for a hyperbolic orbit, which is not periodic.
    revolutions = 0.0
    period = float("inf")
    if alpha > 0.0:
        period = 2.0 * float(np.pi) / (sqrt_mu * alpha ** 1.5)
        revolutions = float(np.round(dt / period))
        dt = dt - revolutions * period
        if dt == 0.0:
            return KeplerStep(r0.copy(), v0.copy(), 0, 0.0, 0.0)

    rdotv_over_sqrt_mu = float(np.dot(r0, v0)) / sqrt_mu

    if alpha > 0.0:
        chi = sqrt_mu * dt * alpha
    else:
        semi_major = 1.0 / alpha
        chi = float(
            np.sign(dt)
            * np.sqrt(-semi_major)
            * np.log(
                max(
                    (-2.0 * mu * alpha * dt)
                    / (
                        float(np.dot(r0, v0))
                        + np.sign(dt) * np.sqrt(-mu * semi_major) * (1.0 - r0_magnitude * alpha)
                    ),
                    1.0e-12,
                )
            )
        )

    iterations = 0
    r_magnitude = r0_magnitude
    c2 = 0.5
    c3 = 1.0 / 6.0
    psi = 0.0
    for step_index in range(1, max_iter + 1):
        iterations = step_index
        psi = chi * chi * alpha
        c2 = stumpff_c2(psi)
        c3 = stumpff_c3(psi)
        r_magnitude = (
            chi * chi * c2
            + rdotv_over_sqrt_mu * chi * (1.0 - psi * c3)
            + r0_magnitude * (1.0 - psi * c2)
        )
        if r_magnitude == 0.0:  # pragma: no cover - would need a collision trajectory
            raise ValueError("universal-variable iteration reached a zero radius")
        residual = (
            sqrt_mu * dt
            - chi * chi * chi * c3
            - rdotv_over_sqrt_mu * chi * chi * c2
            - r0_magnitude * chi * (1.0 - psi * c3)
        )
        delta = residual / r_magnitude
        chi += delta
        if abs(delta) < tol:
            break
    else:
        raise ValueError(
            f"universal-variable Kepler iteration did not converge in {max_iter} iterations "
            f"for a reduced interval of {dt!r} s (orbital period {period!r} s, "
            f"{revolutions!r} whole revolutions removed); last step {delta!r}"
        )

    psi = chi * chi * alpha
    c2 = stumpff_c2(psi)
    c3 = stumpff_c3(psi)
    f = 1.0 - chi * chi * c2 / r0_magnitude
    g = dt - chi * chi * chi * c3 / sqrt_mu
    g_dot = 1.0 - chi * chi * c2 / r_magnitude
    f_dot = sqrt_mu * chi * (psi * c3 - 1.0) / (r_magnitude * r0_magnitude)
    r1 = f * r0 + g * v0
    v1 = f_dot * r0 + g_dot * v0
    return KeplerStep(
        position_m=r1,
        velocity_mps=v1,
        iterations=iterations,
        chi=float(chi),
        fg_residual=abs(f * g_dot - f_dot * g - 1.0),
        revolutions_removed=revolutions,
    )


class Trajectory(Protocol):
    """A terminal's motion, queryable at an absolute epoch in seconds."""

    def position(self, t_s: float) -> np.ndarray:
        """Position in metres at absolute epoch ``t_s``."""

    def velocity(self, t_s: float) -> np.ndarray:
        """Velocity in m/s at absolute epoch ``t_s``."""


@dataclass(frozen=True)
class LinearTrajectory:
    """Constant-velocity motion from a state at ``epoch_s``.

    Used for the exact closed-form light-time check and as the honest model for
    a terminal whose ephemeris is a state vector with no dynamics attached.
    """

    position_m: np.ndarray
    velocity_mps: np.ndarray
    epoch_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", as_vector(self.position_m, "position_m"))
        object.__setattr__(self, "velocity_mps", as_vector(self.velocity_mps, "velocity_mps"))
        if not np.isfinite(float(self.epoch_s)):
            raise ValueError(f"epoch_s must be finite, got {self.epoch_s!r}")

    def position(self, t_s: float) -> np.ndarray:
        return self.position_m + self.velocity_mps * (float(t_s) - float(self.epoch_s))

    def velocity(self, t_s: float) -> np.ndarray:
        del t_s
        return self.velocity_mps.copy()

    def acceleration(self, t_s: float) -> np.ndarray:
        """Zero, by construction. Present so both trajectories share a surface."""
        del t_s
        return np.zeros(3)


@dataclass(frozen=True)
class KeplerTrajectory:
    """Two-body motion about a central mass from a state at ``epoch_s``."""

    position_m: np.ndarray
    velocity_mps: np.ndarray
    epoch_s: float = 0.0
    mu: float = MU_EARTH

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", as_vector(self.position_m, "position_m"))
        object.__setattr__(self, "velocity_mps", as_vector(self.velocity_mps, "velocity_mps"))
        if not np.isfinite(float(self.epoch_s)):
            raise ValueError(f"epoch_s must be finite, got {self.epoch_s!r}")
        if not (np.isfinite(float(self.mu)) and float(self.mu) > 0.0):
            raise ValueError(f"mu must be positive and finite, got {self.mu!r}")

    def step(self, t_s: float) -> KeplerStep:
        """Full propagation record to absolute epoch ``t_s``."""
        return propagate_two_body(
            self.position_m, self.velocity_mps, float(t_s) - float(self.epoch_s), self.mu
        )

    def position(self, t_s: float) -> np.ndarray:
        return self.step(t_s).position_m

    def velocity(self, t_s: float) -> np.ndarray:
        return self.step(t_s).velocity_mps

    def acceleration(self, t_s: float) -> np.ndarray:
        """Two-body gravitational acceleration at ``t_s``, m/s^2."""
        r = self.position(t_s)
        magnitude = float(np.linalg.norm(r))
        return -self.mu * r / (magnitude**3)
