"""Physical constants, each with source, units and validity.

Nothing here is fitted. Where the standard references disagree the disagreement is
recorded in the docstring rather than averaged away, because the disagreement is what a
reader needs in order to judge whether two independent codes should agree.
"""

from __future__ import annotations

__all__ = [
    "MU_EARTH",
    "R_EARTH_EQUATORIAL",
    "R_EARTH_MEAN",
    "OMEGA_EARTH",
    "SPEED_OF_LIGHT",
    "ASTRONOMICAL_UNIT",
    "SOLAR_IRRADIANCE_1AU",
    "SRP_PRESSURE_1AU",
    "EARTH_REDUCED_DIPOLE",
    "STANDARD_GRAVITY",
    "DEFAULT_DRAG_COEFFICIENT",
]

MU_EARTH: float = 3.986004418e14
"""Earth gravitational parameter GM [m^3 s^-2]. WGS-84 defining constant, reproduced in
Vallado, *Fundamentals of Astrodynamics and Applications*. Point-mass field only; no J2
anywhere in this package. Validity: Earth orbits."""

R_EARTH_EQUATORIAL: float = 6378137.0
"""WGS-84 equatorial radius [m]. Used for altitude bookkeeping and for the cylindrical
umbra test."""

R_EARTH_MEAN: float = 6371200.0
"""IUGG mean Earth radius [m], the reference radius of the IGRF and of the centred-dipole
reduction. Used only to convert between a surface field and a reduced dipole moment."""

OMEGA_EARTH: float = 7.292115e-5
"""Earth rotation rate [rad s^-1], WGS-84 nominal mean angular velocity. Used for the
co-rotating-atmosphere correction to the aerodynamic relative wind."""

SPEED_OF_LIGHT: float = 299792458.0
"""Speed of light in vacuum [m s^-1]; exact by the 1983 SI definition."""

ASTRONOMICAL_UNIT: float = 1.495978707e11
"""Astronomical unit [m]; exact by IAU 2012 Resolution B2."""

SOLAR_IRRADIANCE_1AU: float = 1361.0
"""Total solar irradiance at 1 AU [W m^-2], solar-cycle mean (modern TIM-era consensus).

Wertz, *Spacecraft Attitude Determination and Control*, and Larson & Wertz, *Space
Mission Analysis and Design*, predate that revision and use 1367 W m^-2, which is 0.441 %
higher and propagates linearly into solar-radiation-pressure torque. Everything in this
package that quotes an SRP number uses 1361 W m^-2 and says so."""

SRP_PRESSURE_1AU: float = SOLAR_IRRADIANCE_1AU / SPEED_OF_LIGHT
"""Solar radiation pressure at 1 AU on a perfectly absorbing surface [N m^-2],
P = Phi / c = 4.53938e-06 N m^-2. The (1 + q) reflectance factor and the incidence
cosine are applied by the torque model, not folded in here."""

EARTH_REDUCED_DIPOLE: float = 7.96e15
"""Earth reduced magnetic dipole moment B0 * Re^3 [T m^3].

The value used for disturbance and magnetorquer sizing in Wertz, *Spacecraft Attitude
Determination and Control*, and Larson & Wertz, *Space Mission Analysis and Design*. With
:data:`R_EARTH_MEAN` it corresponds to an equatorial surface field of 3.078e-05 T. An
IGRF-epoch centred-dipole reduction gives roughly 3.0e-05 T, about 2.5 % lower; that
spread is reported, not reconciled. A centred non-tilted dipole is a sizing model:
pointwise errors against IGRF reach 20-30 % in magnitude and tens of degrees in
direction, which matters here because magnetic desaturation depends on the field
*direction*. See README Limitations."""

STANDARD_GRAVITY: float = 9.80665
"""Standard gravity g0 [m s^-2], the defining constant in the rocket-equation form of
specific impulse, I_sp * g0 = effective exhaust velocity. BIPM/ISO 80000 definition."""

DEFAULT_DRAG_COEFFICIENT: float = 2.2
"""Free-molecular drag coefficient, dimensionless. The conventional sizing value for
compact LEO bodies (Vallado; Larson & Wertz). Physically 2.0 to 2.6 depending on shape
and surface accommodation, i.e. about +/-20 % uncertainty propagating linearly into
aerodynamic torque. Validity: free-molecular flow, above roughly 150 km."""
