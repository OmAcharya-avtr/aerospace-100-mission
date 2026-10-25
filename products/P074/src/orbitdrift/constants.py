"""Physical constants, each with the source it was taken from.

Two gravity models appear in this package and the distinction matters for
every number it reports.

``WGS72`` is the model SGP4 itself uses (Hoots and Roehrich 1980, Spacetrack
Report #3; Vallado, Crawford, Hujsak and Kelso 2006, AIAA 2006-6753). The
independent numerical propagator in :mod:`orbitdrift.forces` is deliberately
run on the **same** constants, so that the divergence it measures against SGP4
is attributable to the structure of the two dynamical models -- mean-element
analytic theory against osculating numerical integration -- and not to a
difference in the value of ``mu`` or ``J2``. A divergence caused by a constant
mismatch would be an artefact of this package; a divergence caused by model
structure is the signal.

``WGS84`` is provided for geodetic altitude only, where the reference
ellipsoid is what the user means by "altitude".

Every value in :data:`WGS72` is asserted equal to ``sgp4.earth_gravity.wgs72``
by ``tests/test_constants.py``, so these literals cannot drift away from the
propagator they are meant to match.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GravityModel:
    """Zonal gravity model. Lengths in km, time in s, so ``mu`` is km^3/s^2."""

    name: str
    mu_km3_s2: float
    radius_km: float
    j2: float
    j3: float
    j4: float
    flattening: float | None = None

    @property
    def zonal_coefficients(self) -> tuple[tuple[int, float], ...]:
        """``((degree, J_n), ...)`` for the zonal terms of degree 2 to 4."""
        return ((2, self.j2), (3, self.j3), (4, self.j4))


WGS72 = GravityModel(
    name="WGS72",
    mu_km3_s2=398600.8,
    radius_km=6378.135,
    j2=0.001082616,
    j3=-2.53881e-06,
    j4=-1.65597e-06,
)
"""The gravity model SGP4 uses. Source: Spacetrack Report #3; AIAA 2006-6753."""

WGS84 = GravityModel(
    name="WGS84",
    mu_km3_s2=398600.5,
    radius_km=6378.137,
    j2=0.00108262998905,
    j3=-2.53215306e-06,
    j4=-1.61098761e-06,
    flattening=1.0 / 298.257223563,
)
"""WGS84, used only for geodetic altitude. Source: NIMA TR8350.2, 3rd ed., 1997."""

OMEGA_EARTH_RAD_S = 7.292115e-05
"""Earth rotation rate, rad/s. Source: NIMA TR8350.2, 3rd ed., 1997, Table 3.1."""

ASTRONOMICAL_UNIT_KM = 149597870.7
"""Astronomical unit, km. Source: IAU 2012 Resolution B2."""

MU_SUN_KM3_S2 = 1.32712440018e11
"""Heliocentric gravitational constant, km^3/s^2. Source: IAU 2009 system of constants."""

MU_MOON_KM3_S2 = 4902.800066
"""Selenocentric gravitational constant, km^3/s^2. Source: DE430, Folkner et al. 2014."""

SECONDS_PER_DAY = 86400.0

BSTAR_REFERENCE_DENSITY_KG_M2_ER = 0.157
"""Reference density in the SGP4 B* definition, kg/(m^2 * Earth radius).

``B* = (1/2) * C_D * (A/m) * rho_0`` with ``rho_0 = 0.1570 kg/(m^2 ER)``
(Vallado 2013, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
section 9.4, after Spacetrack Report #3). Inverting it to recover a drag area
to mass ratio for an independent propagator is an **assumption**, not a
measurement: the real ``C_D``, area and mass are not recoverable from a single
scalar, and the reference density is nominal. The assumption is declared in
README.md under Limitations and its effect is bounded by
``validation/validate_bstar_sensitivity.py``.
"""
