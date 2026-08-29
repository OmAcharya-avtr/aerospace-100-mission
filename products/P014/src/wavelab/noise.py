"""Photon-noise slope model and subaperture dropout.

WaveLab is a reconstruction package, not a sensor simulator (that is the
scope of product P018 ShackSim, cited here as related work rather than
duplicated). Rather than simulating individual detector spots, slope noise is
modeled directly at the level a reconstructor consumes: additive, zero-mean,
i.i.d. Gaussian noise on every slope component with a variance that follows
the standard shot-noise-limited centroid scaling,

    sigma_slope(N)^2 = sigma_ref^2 * (N_ref / N)                              (1)

i.e. ``sigma_slope propto 1 / sqrt(N)`` for photon count ``N``. This is the
photon-noise term of the classical centroid-variance result (the full
derivation, including the read-noise/background term this module omits, is
in Hardy, J. W. (1998), *Adaptive Optics for Astronomical Telescopes*, Oxford
University Press, ch. 5; also Thomas, S. et al. (2006), "Comparison of
centroid computation algorithms in a Shack-Hartmann sensor", *MNRAS* **371**,
323, and reproduced with a full pixel-level derivation in ShackSim's
``cog_noise_sigma``). This module intentionally omits the read-noise and
background terms (which scale with window area, not just photon count) and
the spot-shape dependence: it isolates the ``1/sqrt(N)`` shot-noise floor so
the flux-dependence validation (`validation/validate_photon_noise.py`) tests
one clean, analytically checkable law rather than several conflated effects.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["slope_sigma", "add_slope_noise", "apply_dropout"]


def slope_sigma(photon_flux: float, sigma_ref: float = 1.0, flux_ref: float = 100.0) -> float:
    """Predicted 1-sigma slope noise at a given photon flux, Eq. (1).

    Parameters
    ----------
    photon_flux: photons per subaperture ``N`` [-], ``> 0``.
    sigma_ref: slope sigma at `flux_ref` photons [same unit as one slope
        component], ``> 0``.
    flux_ref: reference photon count [-], ``> 0``.

    Returns
    -------
    Predicted slope standard deviation at `photon_flux`, same unit as
    `sigma_ref`.
    """
    n = float(photon_flux)
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError(f"photon_flux must be finite and > 0, got {photon_flux!r}")
    s_ref = float(sigma_ref)
    if not np.isfinite(s_ref) or s_ref <= 0.0:
        raise ValueError(f"sigma_ref must be finite and > 0, got {sigma_ref!r}")
    n_ref = float(flux_ref)
    if not np.isfinite(n_ref) or n_ref <= 0.0:
        raise ValueError(f"flux_ref must be finite and > 0, got {flux_ref!r}")
    return s_ref * np.sqrt(n_ref / n)


def add_slope_noise(
    slopes: NDArray[np.float64],
    photon_flux: float,
    rng: np.random.Generator,
    sigma_ref: float = 1.0,
    flux_ref: float = 100.0,
) -> NDArray[np.float64]:
    """Add i.i.d. Gaussian shot-noise to a slope vector, per `slope_sigma`.

    Parameters
    ----------
    slopes: ``(...,)`` noise-free slopes, any shape.
    photon_flux, sigma_ref, flux_ref: see `slope_sigma`.
    rng: `numpy.random.Generator`, e.g. `numpy.random.default_rng(seed)`, so
        callers control reproducibility explicitly.

    Returns
    -------
    Noisy slopes, same shape as `slopes`.
    """
    s = np.asarray(slopes, dtype=np.float64)
    if not np.all(np.isfinite(s)):
        raise ValueError("slopes contain non-finite values")
    sigma = slope_sigma(photon_flux, sigma_ref=sigma_ref, flux_ref=flux_ref)
    return s + rng.normal(0.0, sigma, size=s.shape)


def apply_dropout(
    n_sub: int, dropout_rate: float, rng: np.random.Generator
) -> NDArray[np.bool_]:
    """Draw a random subaperture-active mask.

    Parameters
    ----------
    n_sub: number of subapertures, ``>= 1``.
    dropout_rate: probability that a given subaperture is dropped [-], in
        ``[0, 1)``. ``1.0`` is rejected because it drops every subaperture,
        leaving no data for any reconstructor to use.
    rng: `numpy.random.Generator`.

    Returns
    -------
    ``(n_sub,)`` bool array, True = active (kept). At least one subaperture
    is always kept (if the draw would drop all of them, one is restored at
    random) so downstream reconstructors always receive a solvable, if
    poorly conditioned, problem.
    """
    if isinstance(n_sub, bool) or not isinstance(n_sub, (int, np.integer)):
        raise TypeError(f"n_sub must be an integer, got {n_sub!r}")
    if n_sub < 1:
        raise ValueError(f"n_sub must be >= 1, got {n_sub}")
    rate = float(dropout_rate)
    if not (0.0 <= rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate!r}")
    active = rng.random(int(n_sub)) >= rate
    if not np.any(active):
        active[rng.integers(0, n_sub)] = True
    return active
