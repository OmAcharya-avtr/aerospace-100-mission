"""Shared physical constants, unit conventions and validity ranges.

Units used throughout this package, unless a function documents otherwise:

* ``cn2`` / ``cn2_path`` : refractive-index structure parameter, m^-2/3.
  TurbScope models a **path-averaged** Cn^2 that is assumed constant along
  the sensing path (the standard assumption for a scintillometer or a DIMM
  looking along one line of sight) -- it is not a vertical profile. See
  ``README.md`` "Engineering theory" for why that simplification is made and
  what it costs.
* ``path_length_m`` : sensing path length, m (transmitter-receiver distance
  for a scintillometer; effective along-path length for a DIMM sightline).
* ``wavelength_m`` : optical wavelength, m.
* Angles in degrees at public APIs, radians internally.

References
----------
Tatarski, V. I. (1961), *Wave Propagation in a Turbulent Medium*, McGraw-Hill
    -- the original weak-fluctuation (Rytov) theory of scintillation.
Andrews, L. C. and Phillips, R. L. (2005), *Laser Beam Propagation through
    Random Media*, 2nd ed., SPIE Press -- standard modern reference for the
    Rytov variance, wave-type coefficients and the qualitative behaviour of
    scintillation from weak to strong (saturated) fluctuations (Ch. 1, 5, 9).
Fried, D. L. (1965), "Statistics of a geometric representation of wavefront
    distortion", *J. Opt. Soc. Am.* 55(11), 1427-1435; Fried, D. L. (1966),
    "Optical resolution through a randomly inhomogeneous medium for very long
    and very short exposures", *J. Opt. Soc. Am.* 56(10), 1372-1379 -- the
    Fried parameter r0.
"""

from __future__ import annotations

__all__ = [
    "FRIED_CONSTANT",
    "OPTICAL_WAVELENGTH_RANGE_M",
    "PLANE_WAVE_RYTOV_COEFF",
    "SPHERICAL_WAVE_RYTOV_COEFF",
    "WAVE_TYPES",
    "WEAK_REGIME_MAX_SIGMA_R2",
]

PLANE_WAVE_RYTOV_COEFF: float = 1.23
"""Rytov-variance coefficient for an unbounded plane wave (dimensionless).

sigma_R^2 = 1.23 * Cn2 * k^(7/6) * L^(11/6).  Source: Tatarski (1961);
reproduced as the standard plane-wave Rytov variance in Andrews & Phillips
(2005) Ch. 1 and Ch. 5, and used in this form for line-of-sight scintillometer
theory by Wang, T., Ochs, G. R. and Lawrence, R. S. (1978), "Wind
measurements by the temporal cross-correlation of the optical scintillations",
*Appl. Opt.* 20(23), 4073-4081, and the earlier scintillometer literature they
cite. Valid for k*L >> 1 (far-field, geometric-optics-scale path) and for
weak fluctuations (see :data:`WEAK_REGIME_MAX_SIGMA_R2`).
"""

SPHERICAL_WAVE_RYTOV_COEFF: float = 0.50
"""Rytov-variance coefficient for an unbounded spherical (point-source) wave.

sigma_R^2 = 0.50 * Cn2 * k^(7/6) * L^(11/6).  Source: Tatarski (1961);
Andrews & Phillips (2005) Ch. 5. The spherical-wave case applies to small
transmitter apertures dominated by beam divergence, e.g. classic point-source
displaced-beam scintillometers (Wang, Ochs & Lawrence 1978).
"""

WAVE_TYPES: tuple[str, ...] = ("plane", "spherical")
"""Wave-front geometries supported by :mod:`turbscope.scintillometer`."""

WEAK_REGIME_MAX_SIGMA_R2: float = 0.3
"""Conservative upper bound of Rytov variance for the weak-fluctuation
(log-normal, Rytov) approximation to be considered valid.

Andrews & Phillips (2005) note the log-normal approximation begins to break
down as sigma_R^2 approaches 1; this package uses the more conservative 0.3
threshold (also common in the scintillometer literature) as the boundary
flagged by :func:`turbscope.scintillometer.is_weak_regime`. Between 0.3 and
about 1 the weak-theory closed-form inversion is *increasingly biased but
still single-valued*; genuine multi-valuedness only appears once the full
saturation curve (:func:`turbscope.scintillometer.scintillation_index_full`)
develops its local maximum, near sigma_R^2 ~ 1-3 (see
``validation/VALIDATION.md`` for the measured onset).
"""

OPTICAL_WAVELENGTH_RANGE_M: tuple[float, float] = (1.0e-7, 1.0e-4)
"""Accepted optical/near-IR wavelength range, m (0.1-100 um), matching the
Kolmogorov-turbulence optical scintillation and seeing theory used here."""

FRIED_CONSTANT: float = 0.423
"""Constant in the Fried-parameter integral r0 = [0.423 k^2 sec(zeta)
integral(Cn2 dh)]^(-3/5). Source: Fried (1965, 1966); reproduced in Andrews &
Phillips (2005) Eq. (12.35) and standard adaptive-optics texts (e.g. Hardy
1998, *Adaptive Optics for Astronomical Telescopes*, Eq. 3.4)."""
