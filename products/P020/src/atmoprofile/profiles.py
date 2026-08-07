"""Refractive-index structure-constant profiles Cn^2(h).

A profile is a callable object returning Cn^2 in m^(-2/3) for an altitude in
metres, together with the metadata every honest turbulence calculation needs:
the source of the model, its stated validity range in altitude, and its
assumptions.

Models provided
---------------
============== ================================================================
``hufnagel_valley`` Hufnagel-Valley with free upper-wind and ground parameters
``hv57``            Hufnagel-Valley 5/7 (v_rms = 21 m/s, A = 1.7e-14)
``slc_day``         SLC (AMOS) daytime piecewise model
``slc_night``       SLC (AMOS) night-time piecewise model
``constant_profile`` homogeneous slab, for closed-form validation
``tabulated_profile`` log-linear interpolation of measured samples
============== ================================================================

References (work-level; no page or equation numbers are quoted - see the
"Citation policy" section of the README):

* L. C. Andrews and R. L. Phillips, "Laser Beam Propagation through Random
  Media", 2nd ed., SPIE Press, 2005 - Hufnagel-Valley and SLC models.
* R. R. Beland, "Propagation through Atmospheric Optical Turbulence", in
  "The Infrared and Electro-Optical Systems Handbook", Vol. 2, SPIE/ERIM,
  1993 - tabulation of the SLC Day/Night models.
* J. W. Hardy, "Adaptive Optics for Astronomical Telescopes", Oxford
  University Press, 1998 - Hufnagel-Valley form and typical parameters.
* R. E. Hufnagel, "Variations of atmospheric turbulence", Digest of Topical
  Meeting on Optical Propagation through Turbulence, OSA, 1974 - the
  upper-atmosphere term.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._validate import check_positive, check_profile_samples

__all__ = [
    "Cn2Profile",
    "hufnagel_valley",
    "hv57",
    "slc_day",
    "slc_night",
    "constant_profile",
    "tabulated_profile",
    "STANDARD_PROFILES",
    "standard_profile",
]

_TOL = 1e-6  # metres of slack allowed at the support endpoints


def _thin(values: np.ndarray, max_count: int) -> tuple[float, ...]:
    """Return at most ``max_count`` evenly spaced entries of ``values``."""
    arr = np.asarray(values, dtype=float)
    if arr.size > max_count:
        idx = np.linspace(0, arr.size - 1, max_count).round().astype(int)
        arr = arr[np.unique(idx)]
    return tuple(float(x) for x in arr)


@dataclass(frozen=True)
class Cn2Profile:
    """A refractive-index structure-constant profile Cn^2(h).

    Attributes
    ----------
    name:
        Short identifier, e.g. ``"HV5/7"``.
    func:
        Vectorised callable mapping altitude array (m) to Cn^2 (m^(-2/3)).
    h_min, h_max:
        Altitude support of the model in metres above the model's ground
        datum.  Evaluating or integrating outside it raises ``ValueError``:
        Cn^2 models are fits over a stated range and do not extrapolate.
    reference:
        Source of the model.
    validity:
        Free-text statement of the conditions under which the model applies.
    breakpoints:
        Interior altitudes (m) at which the model is non-smooth.  Passed to the
        adaptive quadrature so that piecewise models integrate accurately.
    """

    name: str
    func: Callable[[np.ndarray], np.ndarray]
    h_min: float
    h_max: float
    reference: str
    validity: str
    breakpoints: tuple[float, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    def __call__(self, h_m: float | np.ndarray) -> np.ndarray:
        """Return Cn^2 in m^(-2/3) at altitude(s) ``h_m`` in metres."""
        h = np.asarray(h_m, dtype=float)
        if not np.all(np.isfinite(h)):
            raise ValueError("altitude must be finite (no NaN/inf)")
        if np.any(h < self.h_min - _TOL) or np.any(h > self.h_max + _TOL):
            raise ValueError(
                f"altitude outside the validity range [{self.h_min:g}, {self.h_max:g}] m "
                f"of profile {self.name!r}: requested "
                f"[{float(np.min(h)):g}, {float(np.max(h)):g}] m"
            )
        out = np.asarray(self.func(np.clip(h, self.h_min, self.h_max)), dtype=float)
        return out

    def describe(self) -> str:
        """One-paragraph human-readable description including provenance."""
        return (
            f"{self.name}: Cn^2(h) in m^(-2/3), valid for h in "
            f"[{self.h_min:g}, {self.h_max:g}] m.\n"
            f"  Source   : {self.reference}\n"
            f"  Validity : {self.validity}"
        )


# ---------------------------------------------------------------------------
# Hufnagel-Valley
# ---------------------------------------------------------------------------

_HV_REFERENCE = (
    "Hufnagel (1974) upper-atmosphere term with the Valley ground term, in the "
    "form given by Andrews & Phillips, 'Laser Beam Propagation through Random "
    "Media', 2nd ed., SPIE 2005, and Hardy, 'Adaptive Optics for Astronomical "
    "Telescopes', OUP 1998"
)


def hufnagel_valley(
    v_rms_ms: float = 21.0,
    ground_a: float = 1.7e-14,
    *,
    h_max_m: float = 20_000.0,
    name: str | None = None,
) -> Cn2Profile:
    r"""Hufnagel-Valley Cn^2 profile.

    .. math::

        C_n^2(h) = 0.00594\,(v/27)^2\,(10^{-5} h)^{10} e^{-h/1000}
                   + 2.7\times10^{-16} e^{-h/1500}
                   + A\,e^{-h/100}

    Parameters
    ----------
    v_rms_ms:
        Pseudo-wind (rms wind speed over 5-20 km), m/s.  Controls the strength
        of the tropopause bump.  21 m/s is the Hufnagel-Valley 5/7 value.  Note
        that evaluating the rms of the Bufton wind model (v_g = 5 m/s) over
        5-20 km gives 22.96 m/s, not 21 m/s; the discrepancy is documented in
        ``validation/VALIDATION.md`` rather than hidden.
    ground_a:
        Ground-level (boundary-layer) coefficient A, m^(-2/3).  1.7e-14 is the
        HV 5/7 value.
    h_max_m:
        Upper altitude limit of the model support, m (default 20 km).

    Returns
    -------
    Cn2Profile
        Callable profile in m^(-2/3), altitude in metres above ground.

    Units
    -----
    ``h`` metres; ``v_rms_ms`` m/s; ``A`` and the returned Cn^2 m^(-2/3).

    Assumptions and validity
    ------------------------
    * Horizontally homogeneous, time-averaged climatological model; it is not
      a measurement and carries no site or season information beyond the two
      free parameters.
    * Altitude measured above the *ground* (sea-level site assumed).
    * Standard support 0-20 km; above the stratosphere the model decays to a
      negligible level but was not fitted there.
    * The three terms represent, in order: the tropopause maximum near 10 km,
      the free-troposphere background, and the surface boundary layer.
    """
    v = check_positive("v_rms_ms", v_rms_ms)
    a = check_positive("ground_a", ground_a, allow_zero=True)
    top = check_positive("h_max_m", h_max_m)

    def _f(h: np.ndarray) -> np.ndarray:
        return (
            0.00594 * (v / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
            + 2.7e-16 * np.exp(-h / 1500.0)
            + a * np.exp(-h / 100.0)
        )

    label = name or f"HV(v={v:g} m/s, A={a:.3g})"
    return Cn2Profile(
        name=label,
        func=_f,
        h_min=0.0,
        h_max=top,
        reference=_HV_REFERENCE,
        validity=(
            "climatological clear-air model, 0-20 km above a sea-level ground site; "
            "horizontally homogeneous; parameters v (5-20 km rms wind) and A "
            "(surface strength) are the only site/season handles"
        ),
        breakpoints=(),
        meta={"v_rms_ms": v, "ground_a": a},
    )


def hv57(*, h_max_m: float = 20_000.0) -> Cn2Profile:
    """Hufnagel-Valley 5/7: ``v_rms = 21 m/s``, ``A = 1.7e-14 m^(-2/3)``.

    The model is *named* for the values it produces at 0.5 um wavelength for a
    vertical path (zenith angle 0) from the ground to 20 km: r0 = 5 cm and
    theta0 = 7 urad.  That definitional property is used as a validation
    target in ``validation/VALIDATION.md`` (source: Andrews & Phillips 2005;
    Hardy 1998).

    Returns
    -------
    Cn2Profile
        Cn^2 in m^(-2/3) as a function of altitude in metres.
    """
    return hufnagel_valley(21.0, 1.7e-14, h_max_m=h_max_m, name="HV5/7")


# ---------------------------------------------------------------------------
# SLC Day / Night (AMOS, Maui)
# ---------------------------------------------------------------------------

_SLC_REFERENCE = (
    "SLC (Submarine Laser Communication) Day and Night models measured at the "
    "AMOS observatory, Maui; tabulated by Beland, 'Propagation through "
    "Atmospheric Optical Turbulence', in The Infrared and Electro-Optical "
    "Systems Handbook Vol. 2, SPIE/ERIM 1993, and reproduced by Andrews & "
    "Phillips, 'Laser Beam Propagation through Random Media', 2nd ed., SPIE 2005"
)

_SLC_DAY_BREAKS = (18.5, 110.0, 1500.0, 7200.0)
_SLC_NIGHT_BREAKS = (18.5, 110.0, 850.0, 7000.0)


def _slc_day_values(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    hs = np.clip(h, 1e-3, None)  # avoid 0^-1.05 at the ground; first branch is constant
    b1, b2, b3, b4 = _SLC_DAY_BREAKS
    out = np.where(hs < b1, 1.7e-14, 0.0)
    out = np.where((hs >= b1) & (hs < b2), 3.13e-13 / hs**1.05, out)
    out = np.where((hs >= b2) & (hs < b3), 1.3e-15, out)
    out = np.where((hs >= b3) & (hs < b4), 8.87e-7 / hs**3.0, out)
    out = np.where(hs >= b4, 2.0e-16 / hs**0.5, out)
    return out


def _slc_night_values(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    hs = np.clip(h, 1e-3, None)
    b1, b2, b3, b4 = _SLC_NIGHT_BREAKS
    out = np.where(hs < b1, 8.4e-15, 0.0)
    out = np.where((hs >= b1) & (hs < b2), 2.87e-12 / hs**2.0, out)
    out = np.where((hs >= b2) & (hs < b3), 2.5e-16, out)
    out = np.where((hs >= b3) & (hs < b4), 8.87e-7 / hs**3.0, out)
    out = np.where(hs >= b4, 2.0e-16 / hs**0.5, out)
    return out


def slc_day(*, h_max_m: float = 20_000.0) -> Cn2Profile:
    """SLC-Day piecewise Cn^2 model (AMOS, Maui), altitude in metres.

    Piecewise form (h in m, Cn^2 in m^(-2/3)):

    ==================  ===========================
    altitude band       Cn^2
    ==================  ===========================
    h < 18.5 m          1.7e-14
    18.5 <= h < 110 m   3.13e-13 * h^(-1.05)
    110 <= h < 1500 m   1.3e-15
    1500 <= h < 7200 m  8.87e-7 * h^(-3)
    7200 <= h <= 20 km  2.0e-16 * h^(-0.5)
    ==================  ===========================

    Assumptions and validity
    ------------------------
    * Daytime, clear sky, AMOS site (Maui); the model is a fit to measurements
      there and is not a global average.  Its strong near-surface term is a
      daytime-convection feature and would be pessimistic at night or at a
      different site.
    * Support 0-20 km above the site.  The profile is *discontinuous* at the
      band edges by construction; those altitudes are supplied to the
      quadrature as breakpoints.
    """
    top = check_positive("h_max_m", h_max_m)
    return Cn2Profile(
        name="SLC-Day",
        func=_slc_day_values,
        h_min=0.0,
        h_max=top,
        reference=_SLC_REFERENCE,
        validity=(
            "daytime clear-sky fit at the AMOS site (Maui), 0-20 km; piecewise and "
            "discontinuous at the band edges; site-specific, not a global average"
        ),
        breakpoints=tuple(b for b in _SLC_DAY_BREAKS if b < top),
    )


def slc_night(*, h_max_m: float = 20_000.0) -> Cn2Profile:
    """SLC-Night piecewise Cn^2 model (AMOS, Maui), altitude in metres.

    Piecewise form (h in m, Cn^2 in m^(-2/3)):

    ==================  ===========================
    altitude band       Cn^2
    ==================  ===========================
    h < 18.5 m          8.4e-15
    18.5 <= h < 110 m   2.87e-12 * h^(-2)
    110 <= h < 850 m    2.5e-16
    850 <= h < 7000 m   8.87e-7 * h^(-3)
    7000 <= h <= 20 km  2.0e-16 * h^(-0.5)
    ==================  ===========================

    Assumptions and validity
    ------------------------
    * Night-time, clear sky, AMOS site (Maui).  The weak, rapidly decaying
      surface layer is the night-time signature; do not use it for daytime
      links.
    * Support 0-20 km above the site; discontinuous at the band edges, which
      are supplied to the quadrature as breakpoints.
    """
    top = check_positive("h_max_m", h_max_m)
    return Cn2Profile(
        name="SLC-Night",
        func=_slc_night_values,
        h_min=0.0,
        h_max=top,
        reference=_SLC_REFERENCE,
        validity=(
            "night-time clear-sky fit at the AMOS site (Maui), 0-20 km; piecewise and "
            "discontinuous at the band edges; site-specific, not a global average"
        ),
        breakpoints=tuple(b for b in _SLC_NIGHT_BREAKS if b < top),
    )


# ---------------------------------------------------------------------------
# Analytic / measured helpers
# ---------------------------------------------------------------------------


def constant_profile(cn2: float, h_min_m: float = 0.0, h_max_m: float = 1000.0) -> Cn2Profile:
    """Homogeneous slab with Cn^2 constant between two altitudes.

    Not a physical atmosphere: this exists so that every integral in the
    package has a closed form that can be hand-checked (see
    ``validation/VALIDATION.md``).

    Parameters
    ----------
    cn2:
        Structure constant, m^(-2/3), constant across the slab.
    h_min_m, h_max_m:
        Slab bottom and top, metres.
    """
    c = check_positive("cn2", cn2)
    lo = float(h_min_m)
    hi = float(h_max_m)
    if lo < 0.0:
        raise ValueError(f"h_min_m must be >= 0 m, got {lo!r}")
    if hi <= lo:
        raise ValueError(f"h_max_m ({hi!r}) must exceed h_min_m ({lo!r})")

    def _f(h: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(h, dtype=float), c)

    return Cn2Profile(
        name=f"constant Cn^2 = {c:.3g}",
        func=_f,
        h_min=lo,
        h_max=hi,
        reference="analytic test profile (no physical source)",
        validity="closed-form reference case only; a real atmosphere is never homogeneous",
        meta={"cn2": c},
    )


def tabulated_profile(
    heights_m: np.ndarray | list[float],
    cn2_values: np.ndarray | list[float],
    *,
    name: str = "tabulated",
    reference: str = "user-supplied samples",
    validity: str = "user-supplied; interpolation is log-linear in Cn^2, no extrapolation",
) -> Cn2Profile:
    """Profile interpolated from measured or user-supplied samples.

    Interpolation is linear in ``log(Cn^2)`` versus altitude, which is the
    usual choice because Cn^2 spans many decades and must stay positive.

    Raises
    ------
    ValueError
        If the heights are not strictly increasing, if any height is negative,
        if any Cn^2 sample is non-positive, or if fewer than two samples are
        given.
    """
    h, c = check_profile_samples(heights_m, cn2_values, value_name="cn2_values")
    log_c = np.log(c)

    def _f(hq: np.ndarray) -> np.ndarray:
        hq = np.asarray(hq, dtype=float)
        return np.exp(np.interp(hq, h, log_c))

    return Cn2Profile(
        name=name,
        func=_f,
        h_min=float(h[0]),
        h_max=float(h[-1]),
        reference=reference,
        validity=validity,
        # The interpolant is continuous but kinked at every knot; declaring the
        # knots lets the adaptive quadrature integrate each smooth segment on
        # its own.  Capped at 1000 panels to bound the cost of dense tables.
        breakpoints=_thin(h[1:-1], 1000),
    )


#: Registry of the standard models, used by the CLI and the examples.
STANDARD_PROFILES: dict[str, Callable[[], Cn2Profile]] = {
    "hv57": hv57,
    "slc_day": slc_day,
    "slc_night": slc_night,
}


def standard_profile(key: str) -> Cn2Profile:
    """Return a standard profile by key (``hv57``, ``slc_day``, ``slc_night``)."""
    try:
        factory = STANDARD_PROFILES[key.lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(
            f"unknown profile {key!r}; available: {sorted(STANDARD_PROFILES)}"
        ) from exc
    return factory()
