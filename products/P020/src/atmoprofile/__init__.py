"""AtmoProfile - deterministic atmospheric turbulence integral toolkit.

From any Cn^2(h) profile compute the Fried parameter r0, the isoplanatic angle
theta0, the Greenwood frequency f_G, the plane- and spherical-wave Rytov
variance on a slant path, and the weak-regime scintillation index.  Every
function's docstring states the weighting integral it evaluates, its units, its
assumptions and its validity range, and every zenith dependence is applied as
an explicit, stated power of sec(zeta).

There is no randomness and no fitting anywhere in this package: given a profile
and a wavelength the answer is a deterministic quadrature.

Version 0.1.0.  MIT licence.  (C) 2026 OPTIMA Organisation.
"""

from __future__ import annotations

from .constants import (
    C_FRIED,
    C_GREENWOOD,
    C_ISOPLANATIC,
    C_RYTOV,
    C_THETA0_OVER_R0,
    EXPONENT_SEC_ZENITH,
    EXPONENT_WAVELENGTH,
)
from .integrals import (
    INTEGRATION_METHODS,
    ConvergenceRecord,
    effective_turbulence_height,
    grid_convergence,
    turbulence_moment,
    weighted_integral,
    wind_weighted_moment,
)
from .metrics import (
    WEAK_FLUCTUATION_LIMIT,
    TurbulenceSummary,
    coherence_length_to_seeing,
    fried_parameter,
    greenwood_frequency,
    isoplanatic_angle,
    rytov_variance,
    scintillation_index,
    summarize,
)
from .profiles import (
    STANDARD_PROFILES,
    Cn2Profile,
    constant_profile,
    hufnagel_valley,
    hv57,
    slc_day,
    slc_night,
    standard_profile,
    tabulated_profile,
)
from .wind import (
    WindProfile,
    bufton_wind,
    constant_wind,
    rms_upper_wind,
    tabulated_wind,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # constants
    "C_FRIED",
    "C_GREENWOOD",
    "C_ISOPLANATIC",
    "C_RYTOV",
    "C_THETA0_OVER_R0",
    "EXPONENT_SEC_ZENITH",
    "EXPONENT_WAVELENGTH",
    # profiles
    "Cn2Profile",
    "hufnagel_valley",
    "hv57",
    "slc_day",
    "slc_night",
    "constant_profile",
    "tabulated_profile",
    "standard_profile",
    "STANDARD_PROFILES",
    # wind
    "WindProfile",
    "bufton_wind",
    "constant_wind",
    "tabulated_wind",
    "rms_upper_wind",
    # integrals
    "INTEGRATION_METHODS",
    "weighted_integral",
    "turbulence_moment",
    "wind_weighted_moment",
    "effective_turbulence_height",
    "ConvergenceRecord",
    "grid_convergence",
    # metrics
    "WEAK_FLUCTUATION_LIMIT",
    "fried_parameter",
    "isoplanatic_angle",
    "greenwood_frequency",
    "rytov_variance",
    "scintillation_index",
    "coherence_length_to_seeing",
    "TurbulenceSummary",
    "summarize",
]
