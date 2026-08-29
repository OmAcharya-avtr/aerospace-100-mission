"""Input validation helpers.

Every public entry point in :mod:`turbscope` funnels its arguments through these
checks so that invalid physical input raises a ``ValueError``/``TypeError`` with
an actionable message instead of silently producing a nonsense number.
"""

from __future__ import annotations

import numpy as np

# Optical band this package claims validity for.  The Kolmogorov refractive-index
# spectrum with the standard visible/near-IR value of the refractivity
# coefficient is used throughout (Andrews & Phillips 2005, ch. 2); it is not
# valid in the far infrared or at radio wavelengths where humidity dominates the
# refractive-index fluctuations.
WAVELENGTH_MIN_M = 3.0e-7
WAVELENGTH_MAX_M = 3.0e-6

GEOMETRIES = ("spherical", "plane")


def check_positive(name: str, value: float, *, allow_zero: bool = False) -> float:
    """Return ``value`` as a float, raising if it is not (strictly) positive."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    if allow_zero:
        if out < 0.0:
            raise ValueError(f"{name} must be >= 0, got {out!r}")
    elif out <= 0.0:
        raise ValueError(f"{name} must be > 0, got {out!r}")
    return out


def check_wavelength(wavelength_m: float) -> float:
    """Validate an optical wavelength in metres.

    Raises if outside ``[3e-7, 3e-6]`` m, the band over which the Kolmogorov
    optical-turbulence relations implemented here are stated to hold.
    """
    out = check_positive("wavelength_m", wavelength_m)
    if not (WAVELENGTH_MIN_M <= out <= WAVELENGTH_MAX_M):
        raise ValueError(
            f"wavelength_m must be within [{WAVELENGTH_MIN_M:g}, {WAVELENGTH_MAX_M:g}] m "
            f"(300 nm - 3 um), got {out:g} m"
        )
    return out


def check_geometry(geometry: str) -> str:
    """Validate the wave geometry label."""
    if not isinstance(geometry, str):
        raise TypeError(f"geometry must be a string, got {type(geometry).__name__}")
    key = geometry.strip().lower()
    if key not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    return key


def check_path_samples(z_m, cn2) -> tuple[np.ndarray, np.ndarray]:
    """Validate a sampled Cn^2 path.

    Parameters
    ----------
    z_m
        Distance from the transmitter along the path, metres, strictly increasing,
        at least 3 samples (Simpson's rule needs an interval to work with).
    cn2
        Refractive-index structure parameter at those distances, m^(-2/3),
        non-negative and finite.
    """
    z = np.asarray(z_m, dtype=float)
    c = np.asarray(cn2, dtype=float)
    if z.ndim != 1 or c.ndim != 1:
        raise ValueError(f"z_m and cn2 must be 1-D, got shapes {z.shape} and {c.shape}")
    if z.size != c.size:
        raise ValueError(f"z_m and cn2 must have equal length, got {z.size} and {c.size}")
    if z.size < 3:
        raise ValueError(f"at least 3 path samples are required, got {z.size}")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(c)):
        raise ValueError("z_m and cn2 must be finite everywhere")
    if not np.all(np.diff(z) > 0.0):
        raise ValueError("z_m must be strictly increasing")
    if z[0] < 0.0:
        raise ValueError(f"z_m must start at or after the transmitter (z >= 0), got {z[0]!r}")
    if np.any(c < 0.0):
        raise ValueError("cn2 must be non-negative everywhere")
    return z, c


def check_probability(name: str, value: float) -> float:
    """Validate a value in the open interval (0, 1)."""
    out = float(value)
    if not (0.0 < out < 1.0):
        raise ValueError(f"{name} must lie strictly inside (0, 1), got {out!r}")
    return out


def check_count(name: str, value: int, *, minimum: int = 1) -> int:
    """Validate a sample/frame count."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}")
    return out
