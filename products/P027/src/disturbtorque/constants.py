"""Physical and model constants, each with its source, units and the value's provenance.

No constant in this module is fitted or tuned. Where two standard references disagree
(the Earth's reduced magnetic dipole moment, the total solar irradiance) both values are
given and the disagreement is documented rather than averaged away.
"""

from __future__ import annotations

# --- Gravitational and geometric --------------------------------------------------

MU_EARTH: float = 3.986004418e14
"""Earth gravitational parameter GM [m^3 s^-2].

Source: WGS-84 / EGM-96 defining constant, as reproduced in Vallado, *Fundamentals of
Astrodynamics and Applications*. Units: m^3 s^-2. Assumption: point-mass central field;
J2 and higher harmonics are not represented anywhere in this package.
Validity: Earth orbits.
"""

R_EARTH_EQUATORIAL: float = 6378137.0
"""Earth equatorial radius [m]. Source: WGS-84 defining constant (semi-major axis a).

Used for altitude bookkeeping and for the cylindrical eclipse test. Units: m.
"""

R_EARTH_MEAN: float = 6371200.0
"""Earth mean (IUGG) radius [m], the reference radius used by IGRF and by the centred
dipole reduction. Units: m. Used only to convert between a surface field strength and a
reduced dipole moment; never for orbit geometry.
"""

OMEGA_EARTH: float = 7.292115e-5
"""Earth rotation rate [rad s^-1]. Source: WGS-84 defining constant (nominal mean
angular velocity). Units: rad s^-1. Used for the co-rotating-atmosphere correction to
the aerodynamic relative velocity.
"""

# --- Solar ------------------------------------------------------------------------

SPEED_OF_LIGHT: float = 299792458.0
"""Speed of light in vacuum [m s^-1]. Exact by SI definition (1983 CGPM)."""

ASTRONOMICAL_UNIT: float = 1.495978707e11
"""Astronomical unit [m]. Exact by IAU 2012 Resolution B2 definition."""

SOLAR_IRRADIANCE_1AU: float = 1361.0
"""Total solar irradiance at 1 AU [W m^-2], solar-cycle mean.

Source: the modern TSI consensus value of about 1361 W m^-2 (SORCE/TIM-era
measurements). Units: W m^-2. Note that Larson & Wertz, *Space Mission Analysis and
Design*, and Wertz, *Spacecraft Attitude Determination and Control*, both predate that
revision and use 1367 W m^-2; see :data:`SOLAR_IRRADIANCE_1AU_SMAD`. The two differ by
0.44 %, which propagates linearly into solar-radiation-pressure torque.
Validity: 1 AU, whole-spectrum; no spectral or solar-cycle dependence is modelled.
"""

SOLAR_IRRADIANCE_1AU_SMAD: float = 1367.0
"""Total solar irradiance at 1 AU [W m^-2] as used in the older textbook literature
(Wertz 1978; Larson & Wertz, *Space Mission Analysis and Design*). Provided so a user
can reproduce textbook worked examples exactly. Units: W m^-2.
"""

SRP_PRESSURE_1AU: float = SOLAR_IRRADIANCE_1AU / SPEED_OF_LIGHT
"""Solar radiation pressure at 1 AU for a perfectly absorbing surface [N m^-2].

Derived here as P = Phi / c = 1361 / 299792458 = 4.5398e-06 N m^-2. Units: N m^-2.
Assumption: normal incidence, black surface; the (1 + q) reflectance factor and the
incidence cosine are applied by the torque model, not folded in here.
"""

# --- Geomagnetic ------------------------------------------------------------------

EARTH_DIPOLE_MOMENT: float = 7.96e15
"""Earth reduced magnetic dipole moment B0 * Re^3 [T m^3].

Source: the value used in the disturbance-torque estimate of Larson & Wertz, *Space
Mission Analysis and Design*, and Wertz, *Spacecraft Attitude Determination and
Control*. Units: tesla metre^3 (equivalently Wb m). With R_EARTH_MEAN this corresponds
to an equatorial surface field of 7.96e15 / 6371200^3 = 3.078e-05 T = 30 780 nT.

An IGRF-epoch centred-dipole reduction gives an equatorial surface field near
3.0e-05 T, i.e. a moment near 7.76e15 T m^3 — about 2.5 % lower. That discrepancy is
reported, not reconciled; see validation/VALIDATION.md section 6.
Validity: centred, non-tilted dipole; error of order 20-30 % against IGRF at a given
point, which is why this is a sizing model and not a navigation model.
"""

EARTH_SURFACE_FIELD_EQUATORIAL: float = EARTH_DIPOLE_MOMENT / R_EARTH_MEAN**3
"""Equatorial magnetic field at the mean Earth radius [T], derived from
:data:`EARTH_DIPOLE_MOMENT`. Units: T.
"""

# --- Aerodynamic ------------------------------------------------------------------

DEFAULT_DRAG_COEFFICIENT: float = 2.2
"""Free-molecular drag coefficient, dimensionless, commonly used for compact
spacecraft bodies in LEO (Vallado; Larson & Wertz). Physically Cd lies between about
2.0 and 2.6 depending on shape, surface accommodation and the ratio of molecular to
orbital speed; 2.2 is the conventional sizing choice and carries roughly +/-20 %
uncertainty that propagates linearly into aerodynamic torque.
Validity: free-molecular flow, i.e. above roughly 150 km.
"""
