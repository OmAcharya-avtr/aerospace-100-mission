"""Physical and geodetic constants, SI units throughout.

Every value is a defining or published constant, quoted to the precision its
source defines.  No value in this module is fitted, tuned or estimated.

Sources
-------
``C_M_S``
    Speed of light in vacuum, 299 792 458 m/s *exactly*.  This is a defining
    constant of the SI (BIPM, *The International System of Units (SI)*, 9th
    ed., 2019), not a measurement, so it carries no uncertainty.
``MU_EARTH_M3_S2``
    Earth gravitational parameter GM = 3.986 004 418e14 m^3/s^2, the WGS-84
    value (NIMA TR8350.2, 3rd ed., 2000).  Used only by the analytic circular
    orbit model in :mod:`dopplerkit.analytic`; the ``sgp4`` path uses the
    WGS-72 constants built into that package instead, which is why the two
    paths are never compared at better than the ~1e-6 relative level.
``WGS84_A_M``, ``WGS84_F``
    WGS-84 ellipsoid semi-major axis [m] and flattening [-] (NIMA TR8350.2).
``OMEGA_EARTH_RAD_S``
    Earth rotation rate 7.292 115e-5 rad/s, the WGS-84 / IERS nominal mean
    angular velocity (NIMA TR8350.2, 3rd ed., 2000).  This is the value that
    matters most in this package after ``C_M_S``: a ground station at the
    equator moves at 465 m/s in the inertial frame, which is 3.4 kHz of
    carrier Doppler at 2.2 GHz.  Omitting station velocity is the second most
    common way to get a Doppler prediction wrong, after the sign.

Validity
--------
The constants are exact-as-published; the *models* that use them carry the
validity ranges, stated in each module.
"""

from __future__ import annotations

C_M_S: float = 299792458.0
"""Speed of light in vacuum [m/s], exact by SI definition."""

MU_EARTH_M3_S2: float = 3.986004418e14
"""Earth gravitational parameter GM [m^3/s^2], WGS-84 (NIMA TR8350.2)."""

WGS84_A_M: float = 6378137.0
"""WGS-84 ellipsoid semi-major axis [m]."""

WGS84_F: float = 1.0 / 298.257223563
"""WGS-84 ellipsoid flattening [-]."""

WGS84_E2: float = WGS84_F * (2.0 - WGS84_F)
"""WGS-84 first eccentricity squared [-], derived from the flattening."""

OMEGA_EARTH_RAD_S: float = 7.292115e-5
"""Earth nominal mean rotation rate [rad/s], WGS-84 / IERS."""

__all__ = [
    "C_M_S",
    "MU_EARTH_M3_S2",
    "OMEGA_EARTH_RAD_S",
    "WGS84_A_M",
    "WGS84_E2",
    "WGS84_F",
]
