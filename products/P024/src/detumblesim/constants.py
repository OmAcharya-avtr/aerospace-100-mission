"""Physical and geodetic constants used across DetumbleSim.

All values are quoted from published standards; none are fitted or tuned here.

References
----------
WGS 84 : National Geospatial-Intelligence Agency, "Department of Defense World
    Geodetic System 1984", NGA.STND.0036_1.0.0_WGS84, 2014 (equatorial radius,
    gravitational parameter, Earth rotation rate).
IGRF-14 : Alken, P. et al. / IAGA Working Group V-MOD, "International
    Geomagnetic Reference Field, 14th generation", coefficient file
    ``igrf14coeffs.txt`` (IAGA / NOAA NCEI / WDC Kyoto), main-field epoch
    2025.0.  Only the degree-1 (dipole) terms are used here.
Vallado, D. A., "Fundamentals of Astrodynamics and Applications", 4th ed.,
    Microcosm Press, 2013 - for the two-body circular-orbit relations.
"""

from __future__ import annotations

import math

#: WGS 84 Earth equatorial radius [m].
R_EARTH_M: float = 6378137.0

#: WGS 84 Earth gravitational parameter GM [m^3 s^-2].
MU_EARTH: float = 3.986004418e14

#: WGS 84 Earth rotation rate [rad s^-1].
OMEGA_EARTH_RAD_S: float = 7.292115e-5

#: IGRF-14 main-field degree-1 Gauss coefficient g(1,0) at epoch 2025.0 [nT].
IGRF14_2025_G10_NT: float = -29350.0

#: IGRF-14 main-field degree-1 Gauss coefficient g(1,1) at epoch 2025.0 [nT].
IGRF14_2025_G11_NT: float = -1410.3

#: IGRF-14 main-field degree-1 Gauss coefficient h(1,1) at epoch 2025.0 [nT].
IGRF14_2025_H11_NT: float = 4545.5

#: Reference epoch of the degree-1 coefficients above (decimal year).
IGRF14_EPOCH_YEAR: float = 2025.0

#: Tesla per nanotesla.
NT_TO_T: float = 1.0e-9

#: Equatorial surface field strength of the centred dipole,
#: B0 = sqrt(g10^2 + g11^2 + h11^2) [nT].  Derived, not quoted.
B0_NT: float = math.sqrt(
    IGRF14_2025_G10_NT**2 + IGRF14_2025_G11_NT**2 + IGRF14_2025_H11_NT**2
)
