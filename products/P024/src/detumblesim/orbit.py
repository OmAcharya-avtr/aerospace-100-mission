"""Circular Keplerian orbit used to drive the magnetic-field history.

Only the geometry that the field model needs is required, so the orbit is a
*circular two-body* orbit with no perturbations: no J2, no drag, no third
body, no eccentricity.  Over the few-orbit detumble horizons modelled here the
resulting position error is far smaller than the dipole field-model error
(``magfield``), which is the dominant modelling error in this package.

Equations (Vallado 2013, sec. 2.2 and 2.5)
-----------------------------------------
Mean motion for a circular orbit of radius ``a``:

    n = sqrt(mu / a^3)                                              [rad/s]
    T = 2 pi / n                                                    [s]

Position in the Earth-centred inertial frame from argument of latitude
``u = u0 + n t``, inclination ``i`` and right ascension of the ascending node
``Omega``:

    r/a = ( cos u cos O - sin u cos i sin O,
            cos u sin O + sin u cos i cos O,
            sin u sin i )

References
----------
Vallado, D. A., "Fundamentals of Astrodynamics and Applications", 4th ed.,
    Microcosm Press, 2013.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .constants import MU_EARTH, R_EARTH_M


@dataclass(frozen=True)
class CircularOrbit:
    """A circular Keplerian orbit.

    Parameters
    ----------
    altitude_km : float
        Altitude above the WGS 84 equatorial radius [km].  Must be > 0.
    inclination_deg : float
        Inclination [deg], in ``[0, 180]``.
    raan_deg : float
        Right ascension of the ascending node [deg].
    arg_lat0_deg : float
        Argument of latitude at ``t = 0`` [deg].
    gmst0_rad : float
        Earth rotation angle at ``t = 0`` [rad]; sets the phase of the tilted
        dipole relative to the orbit plane.
    """

    altitude_km: float = 500.0
    inclination_deg: float = 97.4
    raan_deg: float = 0.0
    arg_lat0_deg: float = 0.0
    gmst0_rad: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.altitude_km) or self.altitude_km <= 0.0:
            raise ValueError(f"altitude_km must be positive, got {self.altitude_km}")
        if not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError(
                f"inclination_deg must lie in [0, 180], got {self.inclination_deg}"
            )

    @property
    def radius_m(self) -> float:
        """Orbit radius [m]."""
        return R_EARTH_M + self.altitude_km * 1000.0

    @property
    def mean_motion_rad_s(self) -> float:
        """Mean motion ``n = sqrt(mu / a^3)`` [rad/s]."""
        return float(np.sqrt(MU_EARTH / self.radius_m**3))

    @property
    def period_s(self) -> float:
        """Orbital period [s]."""
        return 2.0 * np.pi / self.mean_motion_rad_s

    def arg_lat_rad(self, t_s: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Argument of latitude [rad] at time(s) ``t_s`` [s]."""
        return np.radians(self.arg_lat0_deg) + self.mean_motion_rad_s * np.asarray(
            t_s, dtype=float
        )

    def position_eci(self, t_s: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Inertial position [m], shape ``(3,)`` for scalar ``t_s`` else ``(N, 3)``."""
        u = self.arg_lat_rad(t_s)
        inc = np.radians(self.inclination_deg)
        raan = np.radians(self.raan_deg)
        cu, su = np.cos(u), np.sin(u)
        ci, si = np.cos(inc), np.sin(inc)
        co, so = np.cos(raan), np.sin(raan)
        x = cu * co - su * ci * so
        y = cu * so + su * ci * co
        z = su * si
        out = self.radius_m * np.stack(np.broadcast_arrays(x, y, z), axis=-1)
        return out

    def velocity_eci(self, t_s: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Inertial velocity [m/s] (analytic derivative of ``position_eci``)."""
        u = self.arg_lat_rad(t_s)
        inc = np.radians(self.inclination_deg)
        raan = np.radians(self.raan_deg)
        n = self.mean_motion_rad_s
        cu, su = np.cos(u), np.sin(u)
        ci, si = np.cos(inc), np.sin(inc)
        co, so = np.cos(raan), np.sin(raan)
        x = -su * co - cu * ci * so
        y = -su * so + cu * ci * co
        z = cu * si
        return self.radius_m * n * np.stack(np.broadcast_arrays(x, y, z), axis=-1)

    def orbit_normal_eci(self) -> NDArray[np.float64]:
        """Unit orbit-normal vector in ECI (constant for a Keplerian orbit)."""
        inc = np.radians(self.inclination_deg)
        raan = np.radians(self.raan_deg)
        return np.array(
            [np.sin(inc) * np.sin(raan), -np.sin(inc) * np.cos(raan), np.cos(inc)],
            dtype=float,
        )
