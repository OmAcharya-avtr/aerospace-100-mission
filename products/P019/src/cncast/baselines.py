"""Published vertical Cn^2 profile models (the analytic baselines).

These are implemented FIRST and are the reference against which the learned
model in :mod:`cncast.model` is benchmarked (mission rule: baseline before ML).

All profiles return the refractive-index structure parameter Cn^2 in units of
m^(-2/3) as a function of altitude ``h`` in metres.  Two altitude conventions
appear in the literature; they are stated per model below and NOT silently
mixed:

* Hufnagel-Valley: ``h`` is height above ground level (AGL) in metres, with the
  ground-layer term anchored at h = 0.
* SLC day/night: ``h`` is height above the site in metres.  The models were fit
  at the AMOS observatory, Mt Haleakala, Maui (site elevation ~3.05 km MSL), so
  "h = 0" means the observatory floor, not sea level.

.. warning::
   Every model here is a **climatological** model: it describes an average or
   a representative condition for a site/season, not a forecast for a
   particular night.  Instantaneous Cn^2 at a real site routinely departs from
   these curves by an order of magnitude at a given altitude.  Nothing in this
   module predicts turbulence; it parameterises typical turbulence.

References
----------
Hufnagel, R. E. (1974), "Variations of atmospheric turbulence", in *Digest of
    Technical Papers, Topical Meeting on Optical Propagation through
    Turbulence*, Optical Society of America, Boulder CO, paper WA1.
Valley, G. C. (1980), "Isoplanatic degradation of tilt correction and short-term
    imaging systems", *Applied Optics* 19(4), 574-577.
Andrews, L. C. and Phillips, R. L. (2005), *Laser Beam Propagation through
    Random Media*, 2nd ed., SPIE Press, Ch. 12 (Section 12.2: "Atmospheric
    turbulence profile models"), where the H-V, HV 5/7, SLC-Day and SLC-Night
    parameterisations and the Bufton wind model are all tabulated.
Beland, R. R. (1993), "Propagation through atmospheric optical turbulence", in
    *The Infrared and Electro-Optical Systems Handbook*, Vol. 2, SPIE Press,
    Ch. 2 - source of the SLC day/night piecewise fits.
Bufton, J. L. (1973), "Comparison of vertical profile turbulence structure with
    stellar observations", *Applied Optics* 12(8), 1785-1793 - wind profile.
Greenwood, D. P. (1977), "Bandwidth specification for adaptive optics systems",
    *J. Opt. Soc. Am.* 67(3), 390-393.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "HV57_A0",
    "HV57_RMS_WIND",
    "HV_VALID_RANGE_M",
    "SLC_DAY_VALID_RANGE_M",
    "SLC_NIGHT_VALID_RANGE_M",
    "bufton_wind",
    "hufnagel_valley",
    "hv57",
    "rms_high_altitude_wind",
    "slc_day",
    "slc_night",
]

# --- stated validity ranges (metres) -------------------------------------
HV_VALID_RANGE_M: tuple[float, float] = (0.0, 20_000.0)
"""Altitude range over which the Hufnagel-Valley form is normally quoted (AGL).

Andrews & Phillips (2005) §12.2.2 present H-V as a 0-20 km model; above the
tropopause the exponential terms have decayed and the model contributes
negligibly, but it was never fitted there.
"""

SLC_DAY_VALID_RANGE_M: tuple[float, float] = (0.0, 20_500.0)
"""SLC-Day is defined piecewise on 0-20.5 km above the AMOS site; zero above."""

SLC_NIGHT_VALID_RANGE_M: tuple[float, float] = (0.0, 20_000.0)
"""SLC-Night is defined piecewise on 0-20 km above the AMOS site; zero above."""

HV57_A0: float = 1.7e-14
"""Ground-level Cn^2 (m^-2/3) of the HV 5/7 parameterisation."""

HV57_RMS_WIND: float = 21.0
"""High-altitude rms wind (m/s) of the HV 5/7 parameterisation."""


def _as_altitude(h_m: ArrayLike, name: str = "h_m") -> NDArray[np.float64]:
    """Validate and coerce an altitude argument to a float array in metres."""
    h = np.asarray(h_m, dtype=float)
    if h.size == 0:
        raise ValueError(f"{name} must contain at least one altitude.")
    if not np.all(np.isfinite(h)):
        raise ValueError(f"{name} must be finite (got NaN or inf).")
    if np.any(h < 0.0):
        raise ValueError(f"{name} must be >= 0 m (altitude above the site).")
    return h


def hufnagel_valley(
    h_m: ArrayLike,
    rms_wind_m_s: float = HV57_RMS_WIND,
    a0: float = HV57_A0,
) -> NDArray[np.float64]:
    r"""Hufnagel-Valley Cn^2 profile.

    .. math::

        C_n^2(h) = 0.00594\,(v/27)^2 (10^{-5} h)^{10} e^{-h/1000}
                 + 2.7\times10^{-16} e^{-h/1500}
                 + A\, e^{-h/100}

    Parameters
    ----------
    h_m : array_like
        Height above ground level, metres.  Validity: 0-20 000 m
        (``HV_VALID_RANGE_M``).  Values outside are still evaluated (the form is
        analytic everywhere) but are extrapolation.
    rms_wind_m_s : float
        Pseudowind ``v``: the rms wind speed over 5-20 km, m/s.  Controls the
        tropopause bump.  HV 5/7 uses 21 m/s.
    a0 : float
        Ground-level (h = 0) Cn^2 in m^-2/3, the ``A`` coefficient.  HV 5/7 uses
        1.7e-14 m^-2/3.

    Returns
    -------
    ndarray
        Cn^2 in m^-2/3, same shape as ``h_m``.

    Notes
    -----
    Source: Hufnagel (1974) with the Valley (1980) two-parameter extension; the
    form above is Eq. (12.30) of Andrews & Phillips (2005).  The numeric
    coefficients carry units that make each term m^-2/3 with ``h`` in metres and
    ``v`` in m/s.

    Assumptions: horizontally homogeneous atmosphere, mid-latitude continental
    climatology, clear air.  The model is an *average*: it has no boundary-layer
    diurnal cycle, no terrain, and no inversion layers.

    Raises
    ------
    ValueError
        If ``h_m`` is negative/non-finite, or if ``rms_wind_m_s`` or ``a0`` are
        non-finite or negative.
    """
    h = _as_altitude(h_m)
    v = float(rms_wind_m_s)
    a = float(a0)
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"rms_wind_m_s must be finite and >= 0 m/s (got {rms_wind_m_s!r}).")
    if not np.isfinite(a) or a < 0.0:
        raise ValueError(f"a0 must be finite and >= 0 m^-2/3 (got {a0!r}).")

    high = 0.00594 * (v / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
    tropo = 2.7e-16 * np.exp(-h / 1500.0)
    ground = a * np.exp(-h / 100.0)
    return np.asarray(high + tropo + ground, dtype=float)


def hv57(h_m: ArrayLike) -> NDArray[np.float64]:
    """Hufnagel-Valley 5/7: the standard reference profile.

    ``hufnagel_valley(h, rms_wind_m_s=21.0, a0=1.7e-14)``.

    The name records its defining property at lambda = 0.5 um, vertical path:
    Fried parameter r0 ~= 5 cm and isoplanatic angle theta0 ~= 7 urad.  Those
    two numbers are recomputed from this implementation in
    ``validation/validate_baselines.py`` rather than asserted.

    Parameters
    ----------
    h_m : array_like
        Height above ground level, metres (valid 0-20 000 m).

    Returns
    -------
    ndarray
        Cn^2 in m^-2/3.
    """
    return hufnagel_valley(h_m, HV57_RMS_WIND, HV57_A0)


def slc_day(h_m: ArrayLike) -> NDArray[np.float64]:
    """SLC-Day piecewise Cn^2 profile (AMOS/Mt Haleakala, daytime).

    Piecewise fit (Beland 1993; tabulated in Andrews & Phillips 2005 §12.2.1),
    with ``h`` in metres above the site:

    ===================  ==================================
    Altitude band (m)    Cn^2 (m^-2/3)
    ===================  ==================================
    0 <= h < 18.5        1.70e-14
    18.5 <= h < 240      3.13e-13 / h**1.05
    240 <= h < 880       1.30e-15
    880 <= h < 7220      8.87e-07 / h**3
    7220 <= h < 20500    2.00e-16 / h**0.5
    h >= 20500           0
    ===================  ==================================

    Parameters
    ----------
    h_m : array_like
        Height above the site, metres.  Validity ``SLC_DAY_VALID_RANGE_M``.

    Returns
    -------
    ndarray
        Cn^2 in m^-2/3.  Exactly 0 above 20 500 m, by definition of the fit.

    Notes
    -----
    The published fit is NOT continuous: at h = 18.5 m and h = 240 m the two
    adjacent branches disagree by ~16 % and ~31 % respectively.  That is a
    property of the published model, not of this implementation, and is
    quantified in ``validation/VALIDATION.md`` §1.3.  Daytime solar heating of
    the surface is what makes the near-ground values ~2x the night model.
    """
    h = _as_altitude(h_m)
    # Clamp the power-law base: every power-law branch starts at h >= 18.5 m, so
    # clamping to 1 m changes no returned value and avoids underflow warnings.
    hs = np.maximum(h, 1.0)
    out = np.zeros_like(h, dtype=float)
    out = np.where(h < 18.5, 1.7e-14, out)
    out = np.where((h >= 18.5) & (h < 240.0), 3.13e-13 / hs**1.05, out)
    out = np.where((h >= 240.0) & (h < 880.0), 1.3e-15, out)
    out = np.where((h >= 880.0) & (h < 7220.0), 8.87e-7 / hs**3, out)
    out = np.where((h >= 7220.0) & (h < 20500.0), 2.0e-16 / np.sqrt(hs), out)
    return np.asarray(out, dtype=float)


def slc_night(h_m: ArrayLike) -> NDArray[np.float64]:
    """SLC-Night piecewise Cn^2 profile (AMOS/Mt Haleakala, night-time).

    Piecewise fit (Beland 1993; tabulated in Andrews & Phillips 2005 §12.2.1),
    with ``h`` in metres above the site:

    ===================  ==================================
    Altitude band (m)    Cn^2 (m^-2/3)
    ===================  ==================================
    0 <= h < 18.5        8.40e-15
    18.5 <= h < 110      2.87e-12 / h**2
    110 <= h < 1500      2.50e-16
    1500 <= h < 7200     8.87e-07 / h**3
    7200 <= h < 20000    2.00e-16 / h**0.5
    h >= 20000           0
    ===================  ==================================

    Parameters
    ----------
    h_m : array_like
        Height above the site, metres.  Validity ``SLC_NIGHT_VALID_RANGE_M``.

    Returns
    -------
    ndarray
        Cn^2 in m^-2/3.  Exactly 0 above 20 000 m, by definition of the fit.

    Notes
    -----
    All four internal branch boundaries agree to better than 6 % (see
    ``validation/VALIDATION.md`` §1.3), so this fit is effectively continuous.
    """
    h = _as_altitude(h_m)
    hs = np.maximum(h, 1.0)
    out = np.zeros_like(h, dtype=float)
    out = np.where(h < 18.5, 8.4e-15, out)
    out = np.where((h >= 18.5) & (h < 110.0), 2.87e-12 / hs**2, out)
    out = np.where((h >= 110.0) & (h < 1500.0), 2.5e-16, out)
    out = np.where((h >= 1500.0) & (h < 7200.0), 8.87e-7 / hs**3, out)
    out = np.where((h >= 7200.0) & (h < 20000.0), 2.0e-16 / np.sqrt(hs), out)
    return np.asarray(out, dtype=float)


def bufton_wind(h_m: ArrayLike, ground_wind_m_s: float = 5.0) -> NDArray[np.float64]:
    r"""Bufton wind-speed profile.

    .. math::  V(h) = w_g + 30 \exp\!\left[-\left(\frac{h-9400}{4800}\right)^2\right]

    Parameters
    ----------
    h_m : array_like
        Height above ground level, metres.
    ground_wind_m_s : float
        Surface wind speed ``w_g``, m/s (>= 0).

    Returns
    -------
    ndarray
        Wind speed in m/s.

    Notes
    -----
    Source: Bufton (1973); form as given by Andrews & Phillips (2005) Eq.
    (12.32).  The Gaussian bump is the mid-latitude jet stream, peaking at
    9.4 km with a 4.8 km scale and 30 m/s amplitude.  It is a climatological
    average - real jet-stream cores move several km in altitude and exceed
    60 m/s.  Used here only to compute the Greenwood frequency.
    """
    h = _as_altitude(h_m)
    wg = float(ground_wind_m_s)
    if not np.isfinite(wg) or wg < 0.0:
        raise ValueError(f"ground_wind_m_s must be finite and >= 0 (got {ground_wind_m_s!r}).")
    return np.asarray(wg + 30.0 * np.exp(-(((h - 9400.0) / 4800.0) ** 2)), dtype=float)


def rms_high_altitude_wind(ground_wind_m_s: float = 5.0, n_points: int = 20001) -> float:
    r"""Pseudowind ``v`` of the H-V model from a Bufton wind profile.

    .. math::  v = \left[\frac{1}{15\times10^3}\int_{5\,\mathrm{km}}^{20\,\mathrm{km}}
                V^2(h)\,\mathrm{d}h\right]^{1/2}

    Parameters
    ----------
    ground_wind_m_s : float
        Surface wind speed, m/s.
    n_points : int
        Trapezoid samples over 5-20 km (>= 3).

    Returns
    -------
    float
        rms wind speed in m/s, the ``rms_wind_m_s`` argument of
        :func:`hufnagel_valley`.

    Notes
    -----
    Definition from Andrews & Phillips (2005) Eq. (12.31).  This is how the
    ground wind speed enters the H-V high-altitude term.
    """
    if int(n_points) < 3:
        raise ValueError("n_points must be >= 3.")
    h = np.linspace(5_000.0, 20_000.0, int(n_points))
    v = bufton_wind(h, ground_wind_m_s)
    return float(np.sqrt(np.trapezoid(v**2, h) / 15_000.0))
