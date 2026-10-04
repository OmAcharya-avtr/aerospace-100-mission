"""Closed-form circular-orbit overhead pass: the known-answer reference.

Geometry
--------
A satellite in a circular orbit of geocentric radius ``r_s`` passes directly
over an observer at geocentric radius ``r_o``, with the observer **fixed in
the inertial frame** and lying in the orbital plane.  Put the observer on the
+x axis and the satellite at the zenith at t = 0:

    r_obs(t) = (r_o, 0, 0)
    r_sat(t) = r_s (cos n t, sin n t, 0)
    v_sat(t) = r_s n (-sin n t, cos n t, 0)
    a_sat(t) = -r_s n^2 (cos n t, sin n t, 0)

with mean motion ``n = sqrt(mu / r_s^3)`` (Vallado, *Fundamentals of
Astrodynamics and Applications*, 4th ed., two-body circular orbit).  Then,
exactly:

    rho(t)      = sqrt(r_s^2 + r_o^2 - 2 r_s r_o cos n t)                 [m]
    rho_dot(t)  = r_s r_o n sin(n t) / rho(t)                           [m/s]
    rho_ddot(t) = r_s r_o n^2 cos(n t) / rho(t) - rho_dot(t)^2 / rho(t)  [m/s^2]

The ``rho_dot`` expression follows from the dot product
``rho . v_sat / rho`` with the cross terms cancelling; ``rho_ddot`` is its
exact time derivative.  Both are derived in ``validation/VALIDATION.md``.

Sign convention: ``rho_dot`` is **positive when receding**, matching
:mod:`dopplerkit.geometry`.  For t > 0 (after the zenith) ``sin n t > 0`` so
``rho_dot > 0``: receding.  For t < 0, approaching.  At t = 0, exactly zero --
which is why closest approach and the Doppler zero crossing coincide here by
construction, and why this model is the reference for that check.

Known answers used as tests
---------------------------
* ``rho(0) = r_s - r_o`` exactly (zenith).
* ``rho_dot(0) = 0`` exactly (closest approach).
* ``rho_ddot(0) = r_s r_o n^2 / (r_s - r_o)``, the maximum magnitude of the
  range acceleration over the pass, giving the maximum Doppler-rate
  magnitude ``f_c r_s r_o n^2 / (c (r_s - r_o))``.
* Geometric horizon at ``cos n t = r_o / r_s``, i.e. zero elevation.

Assumptions and validity
------------------------
* Two-body circular orbit; no J2, drag, or third-body perturbation.
* **Non-rotating Earth**: the observer is fixed in the inertial frame.  A real
  station moves at up to 465 m/s inertially, which is 3.4 kHz at 2.2 GHz -- an
  order of magnitude larger than every relativistic term in this package.
  This model is therefore a *numerical reference*, not a pass predictor; use
  :mod:`dopplerkit.frames` with ``sgp4`` for a real station.
* Coplanar, exactly-overhead geometry.  Off-zenith passes are not represented.
* Spherical Earth of radius ``r_o`` for the horizon calculation.
* No light time, no refraction, no relativity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import C_M_S, MU_EARTH_M3_S2, WGS84_A_M
from .doppler import one_way_doppler_hz
from .geometry import State


@dataclass(frozen=True)
class CircularOverheadPass:
    """Closed-form circular overhead pass, SI units.

    Parameters
    ----------
    orbit_radius_m
        Geocentric radius of the circular orbit [m], must exceed
        ``observer_radius_m``.
    observer_radius_m
        Geocentric radius of the inertially fixed observer [m], default the
        WGS-84 equatorial radius.
    mu_m3_s2
        Gravitational parameter [m^3/s^2], default WGS-84 Earth GM.
    """

    orbit_radius_m: float
    observer_radius_m: float = WGS84_A_M
    mu_m3_s2: float = MU_EARTH_M3_S2

    def __post_init__(self) -> None:
        if not self.orbit_radius_m > 0.0:
            raise ValueError(f"orbit_radius_m must be > 0, got {self.orbit_radius_m}")
        if not self.observer_radius_m > 0.0:
            raise ValueError(f"observer_radius_m must be > 0, got {self.observer_radius_m}")
        if self.orbit_radius_m <= self.observer_radius_m:
            raise ValueError(
                f"orbit_radius_m ({self.orbit_radius_m}) must exceed observer_radius_m "
                f"({self.observer_radius_m}); a satellite below the observer has no pass"
            )
        if not self.mu_m3_s2 > 0.0:
            raise ValueError(f"mu_m3_s2 must be > 0, got {self.mu_m3_s2}")

    @property
    def mean_motion_rad_s(self) -> float:
        """Circular mean motion n = sqrt(mu / r_s^3) [rad/s]."""
        return math.sqrt(self.mu_m3_s2 / self.orbit_radius_m**3)

    @property
    def orbital_period_s(self) -> float:
        """Orbital period 2 pi / n [s]."""
        return 2.0 * math.pi / self.mean_motion_rad_s

    @property
    def orbital_speed_mps(self) -> float:
        """Inertial speed r_s n [m/s]."""
        return self.orbit_radius_m * self.mean_motion_rad_s

    def horizon_time_s(self) -> float:
        """Time after the zenith at which elevation reaches 0 deg [s], > 0.

        Zero elevation occurs when the line of sight is perpendicular to the
        observer's radius vector, i.e. ``cos n t = r_o / r_s`` (spherical
        Earth, no refraction).  The pass runs from ``-horizon_time_s()`` to
        ``+horizon_time_s()``.
        """
        return math.acos(self.observer_radius_m / self.orbit_radius_m) / self.mean_motion_rad_s

    def range_m(self, t_s: float | np.ndarray) -> float | np.ndarray:
        """Slant range [m] at time ``t_s`` [s] from the zenith instant.

        ``sqrt(r_s^2 + r_o^2 - 2 r_s r_o cos n t)``, exact for this geometry.
        """
        theta = self.mean_motion_rad_s * np.asarray(t_s, dtype=float)
        rng = np.sqrt(
            self.orbit_radius_m**2
            + self.observer_radius_m**2
            - 2.0 * self.orbit_radius_m * self.observer_radius_m * np.cos(theta)
        )
        return float(rng) if np.isscalar(t_s) or np.ndim(t_s) == 0 else rng

    def range_rate_mps(self, t_s: float | np.ndarray) -> float | np.ndarray:
        """Range-rate [m/s] at time ``t_s`` [s], **positive when receding**.

        ``r_s r_o n sin(n t) / rho(t)``.  Zero at t = 0 by construction, which
        is closest approach.
        """
        n = self.mean_motion_rad_s
        theta = n * np.asarray(t_s, dtype=float)
        amp = self.orbit_radius_m * self.observer_radius_m * n
        out = amp * np.sin(theta) / np.asarray(self.range_m(t_s), dtype=float)
        return float(out) if np.isscalar(t_s) or np.ndim(t_s) == 0 else out

    def range_acceleration_mps2(self, t_s: float | np.ndarray) -> float | np.ndarray:
        """Range-acceleration [m/s^2] at time ``t_s`` [s].

        ``r_s r_o n^2 cos(n t) / rho - rho_dot^2 / rho``, the exact derivative
        of :meth:`range_rate_mps`.  Positive means the recession rate is
        increasing.  At t = 0 it takes its maximum value
        ``r_s r_o n^2 / (r_s - r_o)``.
        """
        n = self.mean_motion_rad_s
        theta = n * np.asarray(t_s, dtype=float)
        rng = np.asarray(self.range_m(t_s), dtype=float)
        rate = np.asarray(self.range_rate_mps(t_s), dtype=float)
        amp = self.orbit_radius_m * self.observer_radius_m * n**2
        out = amp * np.cos(theta) / rng - rate**2 / rng
        return float(out) if np.isscalar(t_s) or np.ndim(t_s) == 0 else out

    def one_way_doppler_hz(
        self, t_s: float | np.ndarray, carrier_hz: float
    ) -> float | np.ndarray:
        """Closed-form one-way Doppler shift [Hz]: ``-f_c rho_dot(t) / c``.

        Positive (up-shift) before the zenith, zero at the zenith, negative
        after.  Uses :func:`dopplerkit.doppler.one_way_doppler_hz` for the
        scalar case so there is exactly one place where the sign lives.
        """
        if np.isscalar(t_s) or np.ndim(t_s) == 0:
            return one_way_doppler_hz(self.range_rate_mps(float(t_s)), carrier_hz)
        rate = np.asarray(self.range_rate_mps(t_s), dtype=float)
        return -float(carrier_hz) * rate / C_M_S

    def max_doppler_rate_hz_per_s(self, carrier_hz: float) -> float:
        """Doppler rate at closest approach [Hz/s], the most negative value of the pass.

        ``-f_c r_s r_o n^2 / (c (r_s - r_o))``.  Negative, because the Doppler
        shift falls through zero at the zenith.
        """
        n = self.mean_motion_rad_s
        rho_ddot = (
            self.orbit_radius_m
            * self.observer_radius_m
            * n**2
            / (self.orbit_radius_m - self.observer_radius_m)
        )
        return -float(carrier_hz) * rho_ddot / C_M_S

    def observer_state(self, t_s: float = 0.0) -> State:
        """Observer :class:`~dopplerkit.geometry.State` (inertially fixed) at ``t_s``.

        Position ``(r_o, 0, 0)`` [m], zero velocity and acceleration -- the
        non-rotating-Earth assumption made explicit.  ``t_s`` is accepted and
        ignored so that the signature matches :meth:`satellite_state`.
        """
        del t_s
        return State(
            position_m=np.array([self.observer_radius_m, 0.0, 0.0]),
            velocity_mps=np.zeros(3),
            acceleration_mps2=np.zeros(3),
            label="observer (inertially fixed)",
        )

    def satellite_state(self, t_s: float) -> State:
        """Satellite :class:`~dopplerkit.geometry.State` at ``t_s`` [s].

        Position, velocity and two-body acceleration in the same inertial
        frame as :meth:`observer_state`, so that the generic vector code in
        :mod:`dopplerkit.geometry` can be cross-checked against the closed
        forms on this page.
        """
        n = self.mean_motion_rad_s
        theta = n * float(t_s)
        c_t, s_t = math.cos(theta), math.sin(theta)
        r_s = self.orbit_radius_m
        return State(
            position_m=np.array([r_s * c_t, r_s * s_t, 0.0]),
            velocity_mps=np.array([-r_s * n * s_t, r_s * n * c_t, 0.0]),
            acceleration_mps2=np.array([-r_s * n**2 * c_t, -r_s * n**2 * s_t, 0.0]),
            label="satellite (circular, two-body)",
        )


__all__ = ["CircularOverheadPass"]
