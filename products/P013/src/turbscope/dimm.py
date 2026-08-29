r"""Forward model for differential image motion (the DIMM channel).

A differential image motion monitor images the same source through two
sub-apertures of diameter ``D`` whose centres are separated by a baseline ``d``,
and measures the variance of the *difference* of the two image centroids.  The
difference cancels telescope shake and tracking error, which is what makes the
method absolute.

Sarazin & Roddier (1990), *A&A* 227, 294-300, "The ESO differential image motion
monitor", give for a Kolmogorov atmosphere

.. math::
   \sigma_\ell^2 = 2\lambda^2 r_0^{-5/3}\bigl[0.179 D^{-1/3} - 0.0968 d^{-1/3}\bigr]
   \qquad\text{(longitudinal, along the baseline)}

   \sigma_t^2 = 2\lambda^2 r_0^{-5/3}\bigl[0.179 D^{-1/3} - 0.145 d^{-1/3}\bigr]
   \qquad\text{(transverse)}

with ``sigma^2`` the differential angle-of-arrival variance in rad^2, ``lambda``
and ``D``, ``d``, ``r_0`` in metres.  Validity as published: ``d >= 2D``,
Kolmogorov spectrum, ``D`` well inside the inertial subrange, zero exposure time,
infinite signal-to-noise.  Tokovinin (2002), *PASP* 114, 1156-1166, gives
higher-order corrections (finite exposure, aperture-shape, propagation) that are
**not** implemented here; see the README limitations.

Path-averaged ``Cn2`` from ``r_0``
---------------------------------
Fried (1966), *JOSA* 56(10), 1372-1379:

``r_0 = [0.423 k^2 int_0^L Cn2(z) W_co(z/L) dz]^(-3/5)``

with the coherence kernel ``W_co`` of :mod:`turbscope.geometry` -- ``u^(5/3)`` for
a spherical wave from a beacon at ``z = 0``, ``1`` for a plane wave.  Note that
this kernel is *different* from the scintillation kernel, so a DIMM and a
scintillometer on the same path measure different weighted averages of the same
``Cn2(z)``.  Quantifying that mismatch is one of the things this package is for.

A useful algebraic consequence: because ``r_0^(-5/3)`` is proportional to
``sigma^2/lambda^2`` and ``k^2 = 4 pi^2/lambda^2``, the ``Cn2`` recovered from a
DIMM is **independent of wavelength** in Kolmogorov theory.  That identity is
checked with Hypothesis in the test suite.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from ._validate import check_path_samples, check_positive, check_wavelength
from .geometry import PathGeometry, coherence_weight, weight_normalisation

__all__ = [
    "FRIED_COEFFICIENT",
    "cn2_average_from_fried",
    "dimm_coefficient",
    "dimm_variance",
    "fried_from_average",
    "fried_parameter",
    "r0_from_dimm_variance",
    "seeing_fwhm_rad",
]

#: Fried (1966) coefficient in ``r_0 = [0.423 k^2 int Cn2 dz]^(-3/5)``.
FRIED_COEFFICIENT: float = 0.423

#: Sarazin & Roddier (1990) DIMM coefficients (dimensionless).
_K_APERTURE = 0.179
_K_BASELINE = {"longitudinal": 0.0968, "transverse": 0.145}


def _check_dimm_geometry(subaperture_m: float, baseline_m: float) -> tuple[float, float]:
    d_sub = check_positive("subaperture_m", subaperture_m)
    d_base = check_positive("baseline_m", baseline_m)
    if d_base < 2.0 * d_sub:
        raise ValueError(
            "the Sarazin & Roddier (1990) DIMM formulae are stated for baseline >= "
            f"2 x subaperture; got baseline_m={d_base:g} m and subaperture_m={d_sub:g} m"
        )
    return d_sub, d_base


def dimm_coefficient(subaperture_m: float, baseline_m: float, component: str) -> float:
    """Sarazin & Roddier (1990) bracket ``K``, units m^(-1/3).

    ``K = 0.179 D^(-1/3) - c d^(-1/3)`` with ``c = 0.0968`` (longitudinal) or
    ``0.145`` (transverse).  ``sigma^2 = 2 lambda^2 r_0^(-5/3) K``.
    """
    if component not in _K_BASELINE:
        raise ValueError(
            f"component must be 'longitudinal' or 'transverse', got {component!r}"
        )
    d_sub, d_base = _check_dimm_geometry(subaperture_m, baseline_m)
    return _K_APERTURE * d_sub ** (-1.0 / 3.0) - _K_BASELINE[component] * d_base ** (-1.0 / 3.0)


def dimm_variance(
    r0_m: float,
    wavelength_m: float,
    subaperture_m: float,
    baseline_m: float,
    component: str = "longitudinal",
) -> float:
    """Differential image-motion variance, rad^2.

    ``sigma^2 = 2 lambda^2 r_0^(-5/3) K(D, d, component)``  (Sarazin & Roddier 1990).

    Parameters
    ----------
    r0_m
        Fried parameter at ``wavelength_m`` for the relevant wave geometry, metres.
    wavelength_m, subaperture_m, baseline_m
        Metres.  ``baseline_m >= 2 * subaperture_m`` is enforced.
    component
        ``"longitudinal"`` (displacement along the baseline) or ``"transverse"``.
    """
    r0 = check_positive("r0_m", r0_m)
    lam = check_wavelength(wavelength_m)
    k_coef = dimm_coefficient(subaperture_m, baseline_m, component)
    return 2.0 * lam * lam * r0 ** (-5.0 / 3.0) * k_coef


def r0_from_dimm_variance(
    variance_rad2: float,
    wavelength_m: float,
    subaperture_m: float,
    baseline_m: float,
    component: str = "longitudinal",
) -> float:
    """Fried parameter, metres, from a measured differential-motion variance.

    Closed-form inverse of :func:`dimm_variance`:
    ``r_0 = [2 lambda^2 K / sigma^2]^(3/5)``.
    """
    var = check_positive("variance_rad2", variance_rad2)
    lam = check_wavelength(wavelength_m)
    k_coef = dimm_coefficient(subaperture_m, baseline_m, component)
    return float((2.0 * lam * lam * k_coef / var) ** (3.0 / 5.0))


def fried_parameter(z_m: np.ndarray, cn2: np.ndarray, path: PathGeometry) -> float:
    """Fried coherence length ``r_0`` in metres from a sampled ``Cn2(z)``.

    ``r_0 = [0.423 k^2 int_0^L Cn2(z) W_co(z/L) dz]^(-3/5)`` (Fried 1966), with the
    spherical- or plane-wave coherence kernel selected by ``path.geometry``.
    Simpson quadrature on the supplied grid.
    """
    z, c = check_path_samples(z_m, cn2)
    if not isinstance(path, PathGeometry):
        raise TypeError(f"path must be a PathGeometry, got {type(path).__name__}")
    if abs(z[-1] - path.length_m) > 1e-6 * path.length_m:
        raise ValueError(
            f"path samples must span the geometry length: z[-1]={z[-1]:g} m but "
            f"path.length_m={path.length_m:g} m"
        )
    w = coherence_weight(z / path.length_m, path.geometry)
    integral = float(integrate.simpson(w * c, x=z))
    if integral <= 0.0:
        raise ValueError("Cn2 integrates to zero along the path; r_0 is undefined")
    return float((FRIED_COEFFICIENT * path.k**2 * integral) ** (-3.0 / 5.0))


def fried_from_average(cn2_average: float, path: PathGeometry) -> float:
    """``r_0`` in metres from the coherence-kernel weighted path average of ``Cn2``.

    ``r_0 = [0.423 k^2 N_co L <Cn2>_co]^(-3/5)`` with ``N_co = 3/8`` (spherical) or
    ``1`` (plane).
    """
    c = check_positive("cn2_average", cn2_average)
    norm = weight_normalisation("coherence", path.geometry)
    return float((FRIED_COEFFICIENT * path.k**2 * norm * path.length_m * c) ** (-3.0 / 5.0))


def cn2_average_from_fried(r0_m: float, path: PathGeometry) -> float:
    """Coherence-kernel weighted path average of ``Cn2``, m^(-2/3), from ``r_0``."""
    r0 = check_positive("r0_m", r0_m)
    norm = weight_normalisation("coherence", path.geometry)
    return float(r0 ** (-5.0 / 3.0) / (FRIED_COEFFICIENT * path.k**2 * norm * path.length_m))


def seeing_fwhm_rad(r0_m: float, wavelength_m: float) -> float:
    """Long-exposure seeing FWHM in radians, ``0.98 lambda / r_0``.

    Source: Dierickx (1992) / Roddier (1981), *Prog. Opt.* 19, 281-376.  Included
    only so that DIMM output can be reported in the arcsecond units observers use.
    """
    r0 = check_positive("r0_m", r0_m)
    lam = check_wavelength(wavelength_m)
    return 0.98 * lam / r0
