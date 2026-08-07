"""Integrated seeing quantities derived from a Cn^2(h) profile.

Each function states its weighting integral, units, assumptions and validity
range.  All integrals are evaluated with the composite trapezoid rule over the
supplied altitude grid; grid resolution is the user's responsibility and the
sensitivity to it is quantified in ``validation/VALIDATION.md`` §2.

Conventions
-----------
* ``h_m``   : altitude above the observer, metres, strictly increasing.
* ``cn2``   : Cn^2 in m^-2/3 on the same grid.
* ``wavelength_m`` : optical wavelength in metres.
* ``zenith_angle_deg`` : angle from zenith, degrees, 0 <= zeta < 90.  The plane-
  parallel (flat-Earth) approximation sec(zeta) is used, which is standard and
  is accurate to ~1 % below about 60 deg and degrades rapidly beyond 70 deg.

References
----------
Fried, D. L. (1966), "Optical resolution through a randomly inhomogeneous
    medium for very long and very short exposures", *J. Opt. Soc. Am.* 56(10),
    1372-1379.  (r0)
Fried, D. L. (1982), "Anisoplanatism in adaptive optics", *J. Opt. Soc. Am.*
    72(1), 52-61.  (theta0)
Greenwood, D. P. (1977), "Bandwidth specification for adaptive optics systems",
    *J. Opt. Soc. Am.* 67(3), 390-393.  (f_G)
Andrews, L. C. and Phillips, R. L. (2005), *Laser Beam Propagation through
    Random Media*, 2nd ed., SPIE Press, Ch. 12 - the forms coded below
    (Eqs. 12.35, 12.39, 12.41 in that chapter's numbering).
Roddier, F. (1981), "The effects of atmospheric turbulence in optical
    astronomy", *Progress in Optics* 19, 281-376.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "fried_parameter",
    "greenwood_frequency",
    "isoplanatic_angle",
    "seeing_fwhm_arcsec",
    "turbulence_moment",
]


def _check_profile(h_m: ArrayLike, cn2: ArrayLike) -> tuple[NDArray, NDArray]:
    """Validate an (altitude, Cn^2) pair; return float arrays."""
    h = np.asarray(h_m, dtype=float)
    c = np.asarray(cn2, dtype=float)
    if h.ndim != 1 or c.ndim != 1:
        raise ValueError("h_m and cn2 must both be 1-D arrays.")
    if h.shape != c.shape:
        raise ValueError(f"h_m and cn2 must have the same shape (got {h.shape} and {c.shape}).")
    if h.size < 2:
        raise ValueError("A profile needs at least 2 altitude samples to integrate.")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(c)):
        raise ValueError("h_m and cn2 must be finite.")
    if np.any(np.diff(h) <= 0.0):
        raise ValueError("h_m must be strictly increasing.")
    if np.any(h < 0.0):
        raise ValueError("h_m must be >= 0 m.")
    if np.any(c < 0.0):
        raise ValueError("cn2 must be >= 0 m^-2/3 (Cn^2 is a variance-like quantity).")
    return h, c


def _sec_zenith(zenith_angle_deg: float) -> float:
    z = float(zenith_angle_deg)
    if not np.isfinite(z) or z < 0.0 or z >= 90.0:
        raise ValueError(
            f"zenith_angle_deg must satisfy 0 <= zeta < 90 (got {zenith_angle_deg!r})."
        )
    return float(1.0 / np.cos(np.radians(z)))


def _check_wavelength(wavelength_m: float) -> float:
    lam = float(wavelength_m)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError(f"wavelength_m must be finite and > 0 (got {wavelength_m!r}).")
    if not (1e-7 <= lam <= 1e-4):
        raise ValueError(
            f"wavelength_m = {lam:g} is outside the 0.1-100 um optical/IR range these "
            "Kolmogorov scalings are used for."
        )
    return lam


def turbulence_moment(h_m: ArrayLike, cn2: ArrayLike, order: float = 0.0) -> float:
    r"""Turbulence moment :math:`\mu_m = \int C_n^2(h)\, h^m\,\mathrm{d}h`.

    Parameters
    ----------
    h_m, cn2 : array_like
        Profile grid (m) and Cn^2 (m^-2/3), same shape, ``h_m`` increasing.
    order : float
        Moment order ``m``.  ``0`` gives the integrated turbulence
        (units m^1/3); ``5/3`` is the isoplanatic weighting.

    Returns
    -------
    float
        Moment value, units m^(1/3 + order).

    Notes
    -----
    Composite trapezoid rule.  ``h**order`` with ``h = 0`` and ``order > 0`` is
    0, which is correct; ``order < 0`` is rejected because the integrand
    diverges at the ground.
    """
    h, c = _check_profile(h_m, cn2)
    m = float(order)
    if not np.isfinite(m):
        raise ValueError("order must be finite.")
    if m < 0.0:
        raise ValueError("order must be >= 0; negative moments diverge at h = 0.")
    weight = np.ones_like(h) if m == 0.0 else h**m
    return float(np.trapezoid(c * weight, h))


def fried_parameter(
    h_m: ArrayLike,
    cn2: ArrayLike,
    wavelength_m: float = 500e-9,
    zenith_angle_deg: float = 0.0,
) -> float:
    r"""Fried coherence length r0.

    .. math::

        r_0 = \left[0.423\,k^2 \sec\zeta \int C_n^2(h)\,\mathrm{d}h\right]^{-3/5},
        \qquad k = 2\pi/\lambda

    Parameters
    ----------
    h_m, cn2 : array_like
        Profile (m, m^-2/3).
    wavelength_m : float
        Wavelength, metres (0.1-100 um enforced).
    zenith_angle_deg : float
        Zenith angle in degrees, 0 <= zeta < 90 (plane-parallel sec law).

    Returns
    -------
    float
        r0 in metres.

    Notes
    -----
    Source: Fried (1966); this plane-wave form is Andrews & Phillips (2005)
    Eq. (12.35).  Assumptions: Kolmogorov spectrum with inner scale far below
    and outer scale far above r0, weak-fluctuation (Rytov) regime, plane-wave
    (astronomical / distant-source) geometry.  For a spherical wave the constant
    0.423 is replaced by 0.423 x 3/8; that case is out of scope here.
    Wavelength scaling is r0 ~ lambda^(6/5); zenith scaling r0 ~ cos(zeta)^(3/5).

    Raises
    ------
    ValueError
        On invalid profile, wavelength, zenith angle, or an all-zero profile
        (r0 is then infinite and not representable).
    """
    lam = _check_wavelength(wavelength_m)
    sec_z = _sec_zenith(zenith_angle_deg)
    mu0 = turbulence_moment(h_m, cn2, 0.0)
    if mu0 <= 0.0:
        raise ValueError("Integrated Cn^2 is zero; r0 is undefined (infinite) for no turbulence.")
    k = 2.0 * np.pi / lam
    return float((0.423 * k**2 * sec_z * mu0) ** (-3.0 / 5.0))


def isoplanatic_angle(
    h_m: ArrayLike,
    cn2: ArrayLike,
    wavelength_m: float = 500e-9,
    zenith_angle_deg: float = 0.0,
) -> float:
    r"""Isoplanatic angle theta0.

    .. math::

        \theta_0 = \left[2.914\,k^2 \sec^{8/3}\!\zeta
                   \int C_n^2(h)\,h^{5/3}\,\mathrm{d}h\right]^{-3/5}

    Parameters
    ----------
    h_m, cn2 : array_like
        Profile (m, m^-2/3); ``h_m`` measured from the observer along the
        vertical, so the ``h^(5/3)`` weight makes this quantity dominated by
        high-altitude layers.
    wavelength_m : float
        Wavelength, metres.
    zenith_angle_deg : float
        Zenith angle, degrees.

    Returns
    -------
    float
        theta0 in radians.

    Notes
    -----
    Source: Fried (1982); form as Andrews & Phillips (2005) Eq. (12.39).  Same
    Kolmogorov / weak-fluctuation assumptions as :func:`fried_parameter`.
    theta0 is the angular separation at which the mean-square wavefront
    difference between two paths reaches 1 rad^2.  An equivalent and widely used
    identity is theta0 = 2.914^(-3/5) x 0.423^(3/5) x r0 / h_eff = 0.314 r0/h_eff
    with h_eff the (5/3)-weighted effective turbulence height.
    """
    lam = _check_wavelength(wavelength_m)
    sec_z = _sec_zenith(zenith_angle_deg)
    mu53 = turbulence_moment(h_m, cn2, 5.0 / 3.0)
    if mu53 <= 0.0:
        raise ValueError(
            "The 5/3 turbulence moment is zero; theta0 is undefined (infinite). "
            "A ground-only profile (all turbulence at h = 0) has no anisoplanatism."
        )
    k = 2.0 * np.pi / lam
    return float((2.914 * k**2 * sec_z ** (8.0 / 3.0) * mu53) ** (-3.0 / 5.0))


def greenwood_frequency(
    h_m: ArrayLike,
    cn2: ArrayLike,
    wind_m_s: ArrayLike,
    wavelength_m: float = 500e-9,
    zenith_angle_deg: float = 0.0,
) -> float:
    r"""Greenwood frequency f_G (adaptive-optics closed-loop bandwidth scale).

    .. math::

        f_G = 2.31\,\lambda^{-6/5}
              \left[\sec\zeta \int C_n^2(h)\,V^{5/3}(h)\,\mathrm{d}h\right]^{3/5}

    Parameters
    ----------
    h_m, cn2 : array_like
        Profile (m, m^-2/3).
    wind_m_s : array_like
        Transverse wind speed on the same grid, m/s (>= 0).  See
        :func:`cncast.baselines.bufton_wind`.
    wavelength_m : float
        Wavelength, metres.
    zenith_angle_deg : float
        Zenith angle, degrees.

    Returns
    -------
    float
        f_G in Hz.

    Notes
    -----
    Source: Greenwood (1977); the ``2.31 lambda^(-6/5)`` prefactor is the form
    in Andrews & Phillips (2005) Eq. (12.41) and is algebraically identical to
    the equivalent ``[0.102 k^2 ...]^(3/5)`` form, since
    (0.102 (2 pi)^2)^(3/5) = 2.307.  Assumptions: frozen-flow (Taylor) transport
    of a Kolmogorov phase screen, single-axis wind.  A first-order AO servo whose
    -3 dB bandwidth equals f_G leaves ~1 rad^2 of residual temporal phase
    variance, so practical designs target several times f_G.
    """
    lam = _check_wavelength(wavelength_m)
    sec_z = _sec_zenith(zenith_angle_deg)
    h, c = _check_profile(h_m, cn2)
    v = np.asarray(wind_m_s, dtype=float)
    if v.shape != h.shape:
        raise ValueError(f"wind_m_s must match h_m in shape (got {v.shape} vs {h.shape}).")
    if not np.all(np.isfinite(v)) or np.any(v < 0.0):
        raise ValueError("wind_m_s must be finite and >= 0 m/s.")
    integral = float(np.trapezoid(c * v ** (5.0 / 3.0), h))
    if integral <= 0.0:
        raise ValueError("Integral of Cn^2 V^(5/3) is zero; f_G is undefined (zero).")
    return float(2.31 * lam ** (-6.0 / 5.0) * (sec_z * integral) ** (3.0 / 5.0))


def seeing_fwhm_arcsec(r0_m: float, wavelength_m: float = 500e-9) -> float:
    r"""Long-exposure seeing disc FWHM from r0.

    .. math::  \mathrm{FWHM} = 0.98\,\lambda / r_0  \ \ [\mathrm{rad}]

    Parameters
    ----------
    r0_m : float
        Fried parameter, metres (> 0).
    wavelength_m : float
        Wavelength, metres.

    Returns
    -------
    float
        FWHM in arcseconds.

    Notes
    -----
    Source: Dierickx (1992) / Roddier (1981); the 0.98 factor is the standard
    Kolmogorov long-exposure result for an infinite outer scale.  A finite outer
    scale reduces the FWHM by several per cent to tens of per cent, so this is
    an upper bound on real seeing.
    """
    r0 = float(r0_m)
    if not np.isfinite(r0) or r0 <= 0.0:
        raise ValueError(f"r0_m must be finite and > 0 (got {r0_m!r}).")
    lam = _check_wavelength(wavelength_m)
    return float(np.degrees(0.98 * lam / r0) * 3600.0)
