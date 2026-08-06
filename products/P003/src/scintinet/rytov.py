"""Analytic weak-fluctuation scintillation theory (Rytov approximation).

References
----------
- L. C. Andrews and R. L. Phillips, *Laser Beam Propagation through Random
  Media*, 2nd ed., SPIE Press, 2005 (Rytov variance, weak-fluctuation
  scintillation index; Kolmogorov spectrum results).
- L. C. Andrews, "Aperture-averaging factor for optical scintillations of
  plane and spherical waves in the atmosphere," J. Opt. Soc. Am. A 9(4),
  597-600, 1992 (aperture-averaging approximation).

All functions assume:
- Kolmogorov refractive-index spectrum (no inner/outer-scale corrections),
- horizontally homogeneous turbulence (constant Cn^2 along the path),
- weak fluctuations (Rytov variance sigma_R^2 < ~1; best below ~0.5).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rytov_variance",
    "aperture_averaging_factor",
    "scintillation_index_weak",
]

_WAVE_COEFF = {
    # Plane wave:      sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6)
    # Spherical wave:  beta_0^2  = 0.50 Cn^2 k^(7/6) L^(11/6)
    # Andrews & Phillips 2005, standard Kolmogorov-spectrum results.
    "plane": 1.23,
    "spherical": 0.50,
}


def _validate_scalar(name: str, value: float, *, positive: bool = True) -> float:
    """Coerce to float and validate sign; raise ValueError with actionable message."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    if positive and v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def rytov_variance(
    cn2: float,
    wavelength: float,
    path_length: float,
    wave: str = "plane",
) -> float:
    """Rytov variance for a plane or spherical wave in Kolmogorov turbulence.

    Plane wave:      sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6)
    Spherical wave:  beta_0^2  = 0.50 Cn^2 k^(7/6) L^(11/6)

    Source: Andrews & Phillips, *Laser Beam Propagation through Random
    Media*, SPIE Press, 2005 (standard Kolmogorov results). Assumes constant
    Cn^2 along the path and a Kolmogorov spectrum with no inner/outer scale.
    The Rytov variance equals the scintillation index only in the weak
    fluctuation regime (sigma_R^2 < ~1).

    Parameters
    ----------
    cn2 : float
        Refractive-index structure parameter Cn^2 [m^(-2/3)]. Must be >= 0.
    wavelength : float
        Optical wavelength [m]. Must be > 0.
    path_length : float
        Propagation path length L [m]. Must be > 0.
    wave : str
        "plane" or "spherical".

    Returns
    -------
    float
        Rytov variance [dimensionless].
    """
    try:
        c = float(cn2)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"cn2 must be a real number, got {cn2!r}") from exc
    if not np.isfinite(c) or c < 0.0:
        raise ValueError(f"cn2 must be finite and >= 0 (m^(-2/3)), got {c}")
    lam = _validate_scalar("wavelength", wavelength)
    ell = _validate_scalar("path_length", path_length)
    if wave not in _WAVE_COEFF:
        raise ValueError(f"wave must be one of {sorted(_WAVE_COEFF)}, got {wave!r}")
    k = 2.0 * np.pi / lam  # optical wavenumber [rad/m]
    return _WAVE_COEFF[wave] * c * k ** (7.0 / 6.0) * ell ** (11.0 / 6.0)


def aperture_averaging_factor(
    wavelength: float,
    path_length: float,
    aperture_diameter: float,
) -> float:
    """Aperture-averaging factor A for a circular aperture, plane wave.

    A = [1 + 1.062 * (k D^2 / (4 L))]^(-7/6)

    Source: Andrews, J. Opt. Soc. Am. A 9(4), 597 (1992); also Andrews &
    Phillips 2005. Assumptions: plane-wave illumination, Kolmogorov
    spectrum, weak fluctuations, inner scale << sqrt(L/k) << outer scale.
    A -> 1 for a point aperture (D << Fresnel zone sqrt(L/k)) and decreases
    monotonically with D. The aperture-averaged scintillation index is
    A * sigma_I^2(point).

    Parameters
    ----------
    wavelength : float
        Optical wavelength [m].
    path_length : float
        Path length L [m].
    aperture_diameter : float
        Receiver aperture diameter D [m]. Must be > 0.

    Returns
    -------
    float
        Aperture-averaging factor in (0, 1] [dimensionless].
    """
    lam = _validate_scalar("wavelength", wavelength)
    ell = _validate_scalar("path_length", path_length)
    dia = _validate_scalar("aperture_diameter", aperture_diameter)
    k = 2.0 * np.pi / lam
    d2 = k * dia**2 / (4.0 * ell)  # (D / (2 * Fresnel zone))^2, dimensionless
    return float((1.0 + 1.062 * d2) ** (-7.0 / 6.0))


def scintillation_index_weak(
    cn2: float,
    wavelength: float,
    path_length: float,
    aperture_diameter: float | None = None,
    wave: str = "plane",
) -> float:
    """Weak-fluctuation scintillation index sigma_I^2, optionally aperture-averaged.

    In the weak-fluctuation regime the point scintillation index equals the
    Rytov variance: sigma_I^2 = sigma_R^2 (Andrews & Phillips 2005). With a
    circular receiving aperture of diameter D the index is reduced by the
    Andrews (1992) aperture-averaging factor: sigma_I^2(D) = A(D) * sigma_R^2.

    Validity: sigma_R^2 < ~1 (weak fluctuations); aperture averaging factor
    is a plane-wave approximation, so ``aperture_diameter`` is only accepted
    for ``wave="plane"``.

    Parameters
    ----------
    cn2 : float
        Refractive-index structure parameter [m^(-2/3)].
    wavelength : float
        Wavelength [m].
    path_length : float
        Path length [m].
    aperture_diameter : float or None
        Receiver aperture diameter [m]; None means point receiver.
    wave : str
        "plane" or "spherical".

    Returns
    -------
    float
        Scintillation index sigma_I^2 [dimensionless].
    """
    sig2 = rytov_variance(cn2, wavelength, path_length, wave=wave)
    if aperture_diameter is None:
        return float(sig2)
    if wave != "plane":
        raise ValueError(
            "aperture averaging is implemented with the plane-wave Andrews (1992) "
            "approximation only; use wave='plane' or aperture_diameter=None"
        )
    return float(sig2 * aperture_averaging_factor(wavelength, path_length, aperture_diameter))
