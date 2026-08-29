r"""Differential Image Motion Monitor (DIMM) forward model and inversion.

A DIMM splits a telescope aperture into two circular subapertures of diameter
``D`` separated (centre-to-centre) by ``d``, and measures the variance of the
*differential* centroid motion between the two resulting star images, along
(longitudinal) and across (transverse) the baseline joining the subapertures.
Differencing the two centroids removes fast tip-tilt and vibration common to
both images so the residual variance reflects atmospheric turbulence along
the line of sight, parameterised through the Fried parameter r0.

References
----------
Sarazin, M. and Roddier, F. (1990), "The ESO differential image motion
    monitor", *Astron. Astrophys.* 227, 294-300 -- the DIMM technique and the
    long-baseline (diffraction-neglected) variance formulas used here.
Tokovinin, A. (2002), "From differential image motion to seeing", *Publ.
    Astron. Soc. Pac.* 114, 1156-1166 -- review, generalisation and the
    diffraction-corrected formula (not implemented here; see Limitations).
Fried, D. L. (1965, 1966) -- the Fried parameter r0 (see
    :mod:`turbscope.constants`).

Honesty note on validity
-------------------------
The coefficients below (0.358, 0.541, 0.798) are the standard **geometric
-optics, long-baseline approximation** (``d/D`` not close to 1), which
neglects diffraction at the subaperture edges. Tokovinin (2002) gives a more
accurate formula involving numerical integrals of the aperture autocorrelation
that is not reproduced here; using the simpler formula is a documented
simplification of this product (``README.md`` Limitations), not a claim that
it matches the exact diffraction-corrected result. The implementation is
validated here against its own closed form and against known scaling
identities (``r0^(-5/3)`` scaling, positivity of the bracket for ``d > D``),
not against an independent published numerical table.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import FRIED_CONSTANT, OPTICAL_WAVELENGTH_RANGE_M

__all__ = [
    "DIMM_LONG_SEP_COEFF",
    "DIMM_PREFACTOR",
    "DIMM_TRANS_SEP_COEFF",
    "differential_variance",
    "fried_parameter_from_cn2_path",
    "cn2_path_from_fried_parameter",
    "invert_cn2_from_variance",
]

DIMM_PREFACTOR: float = 0.358
"""Leading coefficient of the DIMM differential-variance formula
(dimensionless). Source: Sarazin & Roddier (1990); reproduced in Tokovinin
(2002) Eq. 6-7 and widely in DIMM instrument documentation."""

DIMM_LONG_SEP_COEFF: float = 0.541
"""Separation-dependence coefficient for the longitudinal (along-baseline)
differential-motion component. Source: Sarazin & Roddier (1990)."""

DIMM_TRANS_SEP_COEFF: float = 0.798
"""Separation-dependence coefficient for the transverse (across-baseline)
differential-motion component. Source: Sarazin & Roddier (1990)."""

_COMPONENTS = {"longitudinal": DIMM_LONG_SEP_COEFF, "transverse": DIMM_TRANS_SEP_COEFF}


def _validate_positive(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0 (got {value!r}).")
    return v


def _validate_wavelength(wavelength_m: float) -> float:
    lam = float(wavelength_m)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"wavelength_m must be finite and > 0 (got {wavelength_m!r}).")
    lo, hi = OPTICAL_WAVELENGTH_RANGE_M
    if not (lo <= lam <= hi):
        raise ValueError(f"wavelength_m = {lam:g} is outside the {lo:g}-{hi:g} m range.")
    return lam


def _validate_geometry(aperture_diam_m: float, separation_m: float) -> tuple[float, float]:
    d_ap = _validate_positive(aperture_diam_m, "aperture_diam_m")
    sep = _validate_positive(separation_m, "separation_m")
    if sep <= d_ap:
        raise ValueError(
            f"separation_m ({sep:g}) must be > aperture_diam_m ({d_ap:g}): DIMM subapertures "
            "must not overlap, and the long-baseline approximation used here requires d/D "
            "clearly above 1."
        )
    return d_ap, sep


def _validate_component(component: str) -> float:
    if component not in _COMPONENTS:
        raise ValueError(f"component must be one of {tuple(_COMPONENTS)} (got {component!r}).")
    return _COMPONENTS[component]


def _validate_zenith(zenith_deg: float) -> float:
    z = float(zenith_deg)
    if not np.isfinite(z) or not (0.0 <= z < 90.0):
        raise ValueError(f"zenith_deg must satisfy 0 <= zeta < 90 (got {zenith_deg!r}).")
    return z


def fried_parameter_from_cn2_path(
    cn2_path: ArrayLike, path_length_m: float, wavelength_m: float, zenith_deg: float = 0.0
) -> NDArray[np.float64]:
    r"""Fried parameter from a path-averaged Cn2, treating the path as a
    homogeneous slab: ``integral(Cn2 dh) = Cn2_path * path_length_m``.

    .. math::

        r_0 = \left[0.423\, k^2 \sec\zeta\; C_n^2\, L \right]^{-3/5}

    Parameters
    ----------
    cn2_path : array_like
        Path-averaged Cn2, m^-2/3 (>= 0).
    path_length_m : float
        Path length, m (> 0).
    wavelength_m : float
        Wavelength, m.
    zenith_deg : float
        Angle from zenith, degrees, ``0 <= zeta < 90``. The plane-parallel
        ``sec(zeta)`` correction is standard and accurate to ~1% below ~60
        deg (see ``turbscope.dimm`` tests); it is not used to correct for
        Earth curvature or a genuinely horizontal path (``zeta ~= 90``,
        excluded).

    Returns
    -------
    ndarray
        r0, m.

    Raises
    ------
    ValueError
        On invalid Cn2 (negative/non-finite), path length, wavelength or
        zenith angle, or if Cn2 is exactly 0 everywhere (r0 undefined,
        infinite).
    """
    c = np.asarray(cn2_path, dtype=float)
    if not np.all(np.isfinite(c)):
        raise ValueError("cn2_path must be finite.")
    if np.any(c < 0.0):
        raise ValueError("cn2_path must be >= 0 m^-2/3.")
    if np.any(c == 0.0):
        raise ValueError("cn2_path must be > 0 (r0 is undefined/infinite for Cn2 = 0).")
    length = _validate_positive(path_length_m, "path_length_m")
    lam = _validate_wavelength(wavelength_m)
    zeta = _validate_zenith(zenith_deg)
    k = 2.0 * np.pi / lam
    sec_z = 1.0 / np.cos(np.radians(zeta))
    integral = c * length
    return np.asarray((FRIED_CONSTANT * k**2 * sec_z * integral) ** (-3.0 / 5.0), dtype=float)


def cn2_path_from_fried_parameter(
    r0_m: float, path_length_m: float, wavelength_m: float, zenith_deg: float = 0.0
) -> float:
    """Inverse of :func:`fried_parameter_from_cn2_path`: Cn2_path from r0.

    .. math:: C_n^2 = \\frac{r_0^{-5/3}}{0.423\\, k^2 \\sec\\zeta\\, L}

    Parameters, validity and raised errors mirror
    :func:`fried_parameter_from_cn2_path`.
    """
    r0 = _validate_positive(r0_m, "r0_m")
    length = _validate_positive(path_length_m, "path_length_m")
    lam = _validate_wavelength(wavelength_m)
    zeta = _validate_zenith(zenith_deg)
    k = 2.0 * np.pi / lam
    sec_z = 1.0 / np.cos(np.radians(zeta))
    return float(r0 ** (-5.0 / 3.0) / (FRIED_CONSTANT * k**2 * sec_z * length))


def differential_variance(
    cn2_path: ArrayLike,
    path_length_m: float,
    wavelength_m: float,
    aperture_diam_m: float,
    separation_m: float,
    component: str = "longitudinal",
    zenith_deg: float = 0.0,
) -> NDArray[np.float64]:
    r"""Forward DIMM differential-motion variance.

    .. math::

        \sigma^2_{l,t} = 0.358 \left(\frac{\lambda}{D}\right)^2
            \left(\frac{D}{r_0}\right)^{5/3}
            \left[1 - c_{l,t} \left(\frac{d}{D}\right)^{-1/3}\right]

    with :math:`c_l` = 0.541 (longitudinal), :math:`c_t` = 0.798 (transverse).

    Parameters
    ----------
    cn2_path : array_like
        Path-averaged Cn2, m^-2/3 (> 0).
    path_length_m : float
        Path length, m (> 0).
    wavelength_m : float
        Wavelength, m.
    aperture_diam_m : float
        Subaperture diameter D, m (> 0).
    separation_m : float
        Centre-to-centre subaperture separation d, m; must be > D (see module
        docstring "Honesty note on validity").
    component : {"longitudinal", "transverse"}
        Which differential-motion component.
    zenith_deg : float
        Angle from zenith, degrees.

    Returns
    -------
    ndarray
        Differential-motion variance, rad^2, same shape as ``cn2_path``.

    Raises
    ------
    ValueError
        On invalid Cn2, geometry, wavelength, zenith angle or component name.
    """
    r0 = fried_parameter_from_cn2_path(cn2_path, path_length_m, wavelength_m, zenith_deg)
    d_ap, sep = _validate_geometry(aperture_diam_m, separation_m)
    c_sep = _validate_component(component)
    bracket = 1.0 - c_sep * (sep / d_ap) ** (-1.0 / 3.0)
    var = DIMM_PREFACTOR * (wavelength_m / d_ap) ** 2 * (d_ap / r0) ** (5.0 / 3.0) * bracket
    return np.asarray(var, dtype=float)


def invert_cn2_from_variance(
    variance_rad2: float,
    path_length_m: float,
    wavelength_m: float,
    aperture_diam_m: float,
    separation_m: float,
    component: str = "longitudinal",
    zenith_deg: float = 0.0,
) -> float:
    """Closed-form single-sensor DIMM inversion: differential variance -> Cn2_path.

    Monotone in ``variance_rad2`` (no saturation ambiguity of the kind
    :mod:`turbscope.scintillometer` has); this inversion is exact given the
    forward model in :func:`differential_variance`. It is *not* claimed to be
    free of real DIMM limitations -- very poor seeing (``D`` comparable to or
    larger than r0) pushes real instruments outside the geometric-optics
    regime this formula assumes; that regime is not separately modelled here
    (see ``README.md`` Limitations).

    Parameters
    ----------
    variance_rad2 : float
        Measured differential-motion variance, rad^2 (> 0).
    path_length_m, wavelength_m, aperture_diam_m, separation_m, component,
    zenith_deg
        As in :func:`differential_variance`.

    Returns
    -------
    float
        Cn2_path, m^-2/3.

    Raises
    ------
    ValueError
        On a non-positive/non-finite variance or invalid geometry, or if the
        implied bracket ``[1 - c(d/D)^(-1/3)]`` is not positive (unphysical
        geometry for this approximation).
    """
    var = _validate_positive(variance_rad2, "variance_rad2")
    d_ap, sep = _validate_geometry(aperture_diam_m, separation_m)
    lam = _validate_wavelength(wavelength_m)
    c_sep = _validate_component(component)
    bracket = 1.0 - c_sep * (sep / d_ap) ** (-1.0 / 3.0)
    if bracket <= 0.0:
        raise ValueError(
            f"Geometry gives a non-positive bracket [1 - c*(d/D)^-1/3] = {bracket:g}; "
            "increase separation_m relative to aperture_diam_m."
        )
    r0 = d_ap / (var / (DIMM_PREFACTOR * (lam / d_ap) ** 2 * bracket)) ** (3.0 / 5.0)
    return cn2_path_from_fried_parameter(r0, path_length_m, lam, zenith_deg)
