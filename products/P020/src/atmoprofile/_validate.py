"""Input validation helpers.

All public entry points route their arguments through this module so that
invalid input raises ``ValueError``/``TypeError`` with an actionable message
instead of silently producing a meaningless number.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

__all__ = [
    "PLANE_PARALLEL_WARN_DEG",
    "WAVELENGTH_MIN_M",
    "WAVELENGTH_MAX_M",
    "check_wavelength",
    "check_zenith",
    "check_altitude_range",
    "check_positive",
    "check_profile_samples",
    "check_choice",
]

#: Beyond this zenith angle the flat-Earth sec(zeta) airmass model degrades
#: (Earth curvature and refraction are ignored here).  Andrews & Phillips
#: (2005) and Hardy (1998) both restrict the sec(zeta) slant-path forms to
#: moderate zenith angles; 60 deg is the customary cut, so exceeding it warns.
PLANE_PARALLEL_WARN_DEG: float = 60.0

#: Accepted optical/IR wavelength band.  The Kolmogorov refractive-index
#: structure constant Cn^2 and the coefficients in :mod:`atmoprofile.constants`
#: are visible/IR results; they do not transfer to radio wavelengths without
#: the wet-term dispersion corrections, which this package does not implement.
WAVELENGTH_MIN_M: float = 100e-9
WAVELENGTH_MAX_M: float = 20e-6


def check_positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    """Return ``value`` as a finite positive float or raise ``ValueError``."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    if out < 0.0 or (out == 0.0 and not allow_zero):
        bound = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {bound}, got {out!r}")
    return out


def check_choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    """Return ``value`` if it is one of ``allowed``; otherwise raise."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, one of {allowed}, got {value!r}")
    out = value.lower()
    if out not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return out


def check_wavelength(wavelength_m: float) -> float:
    """Validate an optical wavelength in metres.

    Raises ``ValueError`` outside 100 nm - 20 um, the band over which the
    visible/IR Kolmogorov coefficients used here are meaningful.
    """
    lam = check_positive("wavelength_m", wavelength_m)
    if not (WAVELENGTH_MIN_M <= lam <= WAVELENGTH_MAX_M):
        raise ValueError(
            f"wavelength_m = {lam:g} m is outside the supported optical/IR band "
            f"[{WAVELENGTH_MIN_M:g}, {WAVELENGTH_MAX_M:g}] m; the Kolmogorov "
            "coefficients in this package are visible/IR results and do not "
            "apply at radio wavelengths."
        )
    return lam


def check_zenith(zenith_rad: float) -> float:
    """Validate a zenith angle in radians and return it.

    Valid range is ``0 <= zenith_rad < pi/2``.  ``sec(zeta)`` diverges at
    90 deg, so the horizon and beyond are rejected rather than returning
    ``inf``.  Angles above :data:`PLANE_PARALLEL_WARN_DEG` emit a
    ``UserWarning`` because the flat-Earth airmass model is used.
    """
    try:
        zen = float(zenith_rad)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"zenith_rad must be a real number, got {zenith_rad!r}") from exc
    if not math.isfinite(zen):
        raise ValueError(f"zenith_rad must be finite, got {zen!r}")
    if zen < 0.0:
        raise ValueError(f"zenith_rad must be >= 0, got {zen!r} rad")
    if zen >= math.pi / 2.0:
        raise ValueError(
            f"zenith_rad must be < pi/2 (90 deg), got {zen!r} rad "
            f"({math.degrees(zen):.3f} deg); sec(zeta) diverges at the horizon and "
            "the plane-parallel atmosphere model has no meaning below it."
        )
    if math.degrees(zen) > PLANE_PARALLEL_WARN_DEG:
        warnings.warn(
            f"zenith angle {math.degrees(zen):.1f} deg exceeds "
            f"{PLANE_PARALLEL_WARN_DEG:.0f} deg: the flat-Earth sec(zeta) airmass "
            "used here neglects Earth curvature and refraction, so the result is "
            "an extrapolation.",
            UserWarning,
            stacklevel=3,
        )
    return zen


def check_altitude_range(
    h_ground: float,
    h_top: float,
    *,
    profile_h_min: float,
    profile_h_max: float,
    profile_name: str,
) -> tuple[float, float]:
    """Validate the integration limits against the profile's support.

    Altitudes are metres above mean sea level (or above the profile's own
    reference datum, which the profile documents).  Negative altitudes are
    rejected: none of the profile models in this package is defined below its
    stated ground level.
    """
    lo = float(h_ground)
    hi = float(h_top)
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError(f"altitude limits must be finite, got ({lo!r}, {hi!r})")
    if lo < 0.0:
        raise ValueError(f"h_ground must be >= 0 m, got {lo!r} m (negative altitude)")
    if hi <= lo:
        raise ValueError(f"h_top ({hi!r} m) must be strictly greater than h_ground ({lo!r} m)")
    if lo < profile_h_min - 1e-9 or hi > profile_h_max + 1e-9:
        raise ValueError(
            f"integration range [{lo:g}, {hi:g}] m lies outside the validity range "
            f"[{profile_h_min:g}, {profile_h_max:g}] m of profile {profile_name!r}; "
            "extrapolating a Cn^2 model beyond its published range is not supported."
        )
    return lo, hi


def check_profile_samples(
    heights_m: np.ndarray | list[float],
    values: np.ndarray | list[float],
    *,
    value_name: str,
    allow_zero_values: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate tabulated (height, value) samples for an interpolated profile.

    Requires: equal lengths, at least two points, finite entries, strictly
    increasing heights, non-negative heights, and positive values (log-space
    interpolation is used, so zeros are rejected unless explicitly allowed).
    """
    h = np.asarray(heights_m, dtype=float)
    v = np.asarray(values, dtype=float)
    if h.ndim != 1 or v.ndim != 1:
        raise ValueError(f"heights_m and {value_name} must be 1-D arrays")
    if h.size != v.size:
        raise ValueError(
            f"heights_m and {value_name} must have the same length, got {h.size} and {v.size}"
        )
    if h.size < 2:
        raise ValueError(f"need at least 2 samples to interpolate, got {h.size}")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(v)):
        raise ValueError(f"heights_m and {value_name} must be finite (no NaN/inf)")
    if np.any(h < 0.0):
        raise ValueError("heights_m must be >= 0 m (negative altitude is not supported)")
    if np.any(np.diff(h) <= 0.0):
        bad = int(np.argmin(np.diff(h)))
        raise ValueError(
            "heights_m must be strictly increasing (monotonic); "
            f"h[{bad}] = {h[bad]:g} m is followed by h[{bad + 1}] = {h[bad + 1]:g} m"
        )
    if allow_zero_values:
        if np.any(v < 0.0):
            raise ValueError(f"{value_name} must be >= 0")
    elif np.any(v <= 0.0):
        raise ValueError(
            f"{value_name} must be > 0 (log-space interpolation is used); "
            f"minimum given was {v.min():g}"
        )
    return h, v
