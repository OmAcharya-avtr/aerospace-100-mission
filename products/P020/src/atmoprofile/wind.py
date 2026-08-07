"""Wind-speed profiles v(h), required by the Greenwood frequency.

The Greenwood frequency is the only quantity in this package that needs
something other than Cn^2: it needs the transverse wind speed along the path,
because the temporal behaviour of the phase screen comes from Taylor's frozen
flow hypothesis (the turbulence pattern is advected past the aperture without
evolving).

References (work-level; no page or equation numbers quoted):

* J. L. Bufton, "Comparison of vertical profile turbulence structure with
  stellar observations", Applied Optics 12(8), 1785-1793, 1973 - the
  ground-plus-tropopause wind model used here.
* D. P. Greenwood, "Bandwidth specification for adaptive optics systems",
  J. Opt. Soc. Am. 67(3), 390-393, 1977 - the frequency itself.
* L. C. Andrews and R. L. Phillips, "Laser Beam Propagation through Random
  Media", 2nd ed., SPIE Press, 2005 - the Bufton model in the form used here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from ._validate import check_positive, check_profile_samples

__all__ = [
    "WindProfile",
    "bufton_wind",
    "constant_wind",
    "tabulated_wind",
    "rms_upper_wind",
]

_TOL = 1e-6


@dataclass(frozen=True)
class WindProfile:
    """Wind speed transverse to the line of sight, v(h), in m/s.

    Attributes
    ----------
    name, reference, validity:
        Provenance metadata, as for :class:`atmoprofile.profiles.Cn2Profile`.
    func:
        Vectorised callable mapping altitude (m) to wind speed (m/s).
    h_min, h_max:
        Altitude support in metres.
    """

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    h_min: float
    h_max: float
    reference: str
    validity: str

    def __call__(self, h_m: float | np.ndarray) -> np.ndarray:
        """Return wind speed in m/s at altitude(s) ``h_m`` in metres."""
        h = np.asarray(h_m, dtype=float)
        if not np.all(np.isfinite(h)):
            raise ValueError("altitude must be finite (no NaN/inf)")
        if np.any(h < self.h_min - _TOL) or np.any(h > self.h_max + _TOL):
            raise ValueError(
                f"altitude outside the validity range [{self.h_min:g}, {self.h_max:g}] m "
                f"of wind profile {self.name!r}"
            )
        v = np.asarray(self.func(np.clip(h, self.h_min, self.h_max)), dtype=float)
        if np.any(v < 0.0):
            raise ValueError(f"wind profile {self.name!r} returned a negative speed")
        return v


def bufton_wind(
    v_ground_ms: float = 5.0,
    *,
    v_peak_ms: float = 30.0,
    h_peak_m: float = 9400.0,
    h_scale_m: float = 4800.0,
    h_max_m: float = 20_000.0,
) -> WindProfile:
    r"""Bufton wind model.

    .. math::

        v(h) = v_g + 30\,\exp\!\left[-\left(\frac{h - 9400}{4800}\right)^2\right]
        \quad \mathrm{[m/s]}

    Parameters
    ----------
    v_ground_ms:
        Ground wind speed v_g, m/s (default 5 m/s, the customary value).
    v_peak_ms, h_peak_m, h_scale_m:
        Amplitude (m/s), centre altitude (m) and Gaussian half-width (m) of the
        tropopause jet.  Defaults 30 m/s at 9.4 km with a 4.8 km scale are the
        Bufton values.
    h_max_m:
        Support limit, m.

    Assumptions and validity
    ------------------------
    * Climatological mid-latitude average; a smooth model, not a sounding.  Any
      real day departs from it substantially, and the Greenwood frequency
      depends on v^(5/3), so wind errors propagate with exponent 1 in f_G
      (f_G ~ [int Cn^2 v^(5/3)]^(3/5)).
    * The speed returned is treated as the component *transverse to the line of
      sight* (see :func:`atmoprofile.metrics.greenwood_frequency` for what that
      assumption costs at non-zero zenith angle).
    * Support 0-20 km.
    * The literature associates the Bufton wind with the Hufnagel-Valley
      pseudowind v = 21 m/s.  Evaluating :func:`rms_upper_wind` on this model
      with v_g = 5 m/s over 5-20 km gives **22.96 m/s**, 9.3 % higher.  The
      convention behind the quoted 21 m/s (band, ground-term treatment, or
      slew-rate term) could not be established during this build, so the
      discrepancy is reported rather than tuned away; see
      ``validation/VALIDATION.md``.
    * Bufton's model is sometimes written with an additional slew term
      ``omega_s * h`` for a ground station tracking a moving target.  That term
      is not included here; supply a :func:`tabulated_wind` if it is needed.
    """
    vg = check_positive("v_ground_ms", v_ground_ms, allow_zero=True)
    vp = check_positive("v_peak_ms", v_peak_ms, allow_zero=True)
    hp = check_positive("h_peak_m", h_peak_m, allow_zero=True)
    hs = check_positive("h_scale_m", h_scale_m)
    top = check_positive("h_max_m", h_max_m)

    def _f(h: np.ndarray) -> np.ndarray:
        return vg + vp * np.exp(-(((h - hp) / hs) ** 2))

    return WindProfile(
        name=f"Bufton(v_g={vg:g} m/s)",
        func=_f,
        h_min=0.0,
        h_max=top,
        reference=(
            "Bufton, Applied Optics 12(8), 1973; in the form given by Andrews & "
            "Phillips, 'Laser Beam Propagation through Random Media', 2nd ed., SPIE 2005"
        ),
        validity=(
            "climatological mid-latitude clear-air average, 0-20 km; smooth model with a "
            "single tropopause jet, no shear layers, no seasonal or site information"
        ),
    )


def constant_wind(v_ms: float, *, h_max_m: float = 20_000.0) -> WindProfile:
    """Uniform wind speed at all altitudes (analytic reference case)."""
    v = check_positive("v_ms", v_ms, allow_zero=True)
    top = check_positive("h_max_m", h_max_m)

    def _f(h: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(h, dtype=float), v)

    return WindProfile(
        name=f"constant {v:g} m/s",
        func=_f,
        h_min=0.0,
        h_max=top,
        reference="analytic test profile (no physical source)",
        validity="closed-form reference case only; a real wind profile is never uniform",
    )


def tabulated_wind(
    heights_m: np.ndarray | list[float],
    speeds_ms: np.ndarray | list[float],
    *,
    name: str = "tabulated wind",
    reference: str = "user-supplied samples",
    validity: str = "user-supplied; linear interpolation in speed, no extrapolation",
) -> WindProfile:
    """Wind profile linearly interpolated from user-supplied samples.

    Raises
    ------
    ValueError
        If heights are not strictly increasing or negative, if lengths differ,
        or if any speed is negative.
    """
    h, v = check_profile_samples(
        heights_m, speeds_ms, value_name="speeds_ms", allow_zero_values=True
    )

    def _f(hq: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(hq, dtype=float), h, v)

    return WindProfile(
        name=name,
        func=_f,
        h_min=float(h[0]),
        h_max=float(h[-1]),
        reference=reference,
        validity=validity,
    )


def rms_upper_wind(
    wind: WindProfile, h_lo_m: float = 5_000.0, h_hi_m: float = 20_000.0
) -> float:
    r"""Root-mean-square wind speed over an altitude band, m/s.

    .. math::

        v_{rms} = \left[\frac{1}{h_{hi}-h_{lo}}
                  \int_{h_{lo}}^{h_{hi}} v^2(h)\,dh\right]^{1/2}

    This is the quantity that parameterises the Hufnagel-Valley model (its
    ``v`` argument), conventionally evaluated over 5-20 km.

    Returns
    -------
    float
        rms wind speed in m/s.
    """
    lo = check_positive("h_lo_m", h_lo_m, allow_zero=True)
    hi = check_positive("h_hi_m", h_hi_m)
    if hi <= lo:
        raise ValueError(f"h_hi_m ({hi!r}) must exceed h_lo_m ({lo!r})")
    if lo < wind.h_min - _TOL or hi > wind.h_max + _TOL:
        raise ValueError(
            f"band [{lo:g}, {hi:g}] m outside the support [{wind.h_min:g}, "
            f"{wind.h_max:g}] m of wind profile {wind.name!r}"
        )
    val, _ = quad(lambda h: float(wind(h)) ** 2, lo, hi, limit=200)
    return float(np.sqrt(val / (hi - lo)))
