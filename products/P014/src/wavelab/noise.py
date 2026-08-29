"""Photon-noise model for Shack-Hartmann slope measurements, and subaperture dropout.

Measurement noise on ``u``
--------------------------
``u = h * dphi/dx`` is the wavefront phase difference across one subaperture
[rad], the quantity the geometry matrices of :mod:`wavelab.geometry` relate to
the phase. Its photon-noise standard deviation follows from three exact steps
plus one stated approximation.

1. **Centroid variance.** A centre-of-gravity estimate formed from ``N``
   independent photon arrivals drawn from the spot intensity distribution is a
   sample mean, so its variance is ``sigma_spot**2 / N`` per axis, with
   ``sigma_spot`` the spot's intensity standard deviation on the detector [m].
   This is elementary and exact **given** a finite ``sigma_spot``.

2. **Spot size.** For a square lenslet of clear aperture ``d`` [m] and focal
   length ``f`` [m] the diffraction pattern has FWHM ``1.0287938 lambda f / d``
   [m] (M. Born and E. Wolf, *Principles of Optics*, 7th ed., CUP 1999,
   Sec. 8.5.2: the Airy intensity ``(2 J1(v)/v)**2`` halves at
   ``v = 1.616339``).

   *Approximation.* The true Airy pattern has an ``r**-3`` intensity envelope
   whose second moment **diverges**, so ``sigma_spot`` is not defined for it.
   As is standard, a Gaussian of equal FWHM is substituted,
   ``sigma_spot = FWHM / (2 sqrt(2 ln 2)) = FWHM / 2.3548200``. Any real
   centroider truncates or thresholds the window, which is what makes its
   variance finite in practice and always makes it **worse** than this figure
   (S. Thomas et al., "Comparison of centroid computation algorithms in a
   Shack-Hartmann sensor", *MNRAS* **371**, 323-336, 2006).

3. **Displacement to phase.** A spot displacement ``Delta`` [m] corresponds to
   a ray angle ``Delta / f`` [rad], equal in the small-angle limit to the
   wavefront gradient ``dW/dx`` [m/m]; the phase gradient is
   ``dphi/dx = (2 pi / lambda) dW/dx`` [rad/m] (J. W. Hardy, *Adaptive Optics
   for Astronomical Telescopes*, OUP 1998, ch. 5).

Putting them together with ``h = d`` (pitch equals lenslet aperture), the
lenslet focal length, the wavelength and the subaperture size **all cancel**:

    sigma_u = 2 pi * (1.0287938 / 2.3548200) / sqrt(N) = 2.744707 / sqrt(N)   [rad]

So the phase-difference measurement error across one subaperture depends only
on the photon count. This is a **photon-noise floor**: it excludes detector
read noise, sky/background, spot truncation, thresholding bias, centroid gain
error, and non-diffraction-limited spots, every one of which increases the
error. Treat every number derived from it as an optimistic bound.

Validity range: ``N >= 1`` photoelectron and a spot well inside its subaperture
field. At very low ``N`` the centroid distribution is not Gaussian and a
Gaussian error model is only indicative.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AIRY_FWHM_COEFF",
    "GAUSSIAN_FWHM_TO_SIGMA",
    "SLOPE_NOISE_COEFF",
    "photon_slope_noise",
    "add_slope_noise",
    "random_dropout_mask",
]

#: Airy FWHM in units of ``lambda f / d`` (Born & Wolf 1999, Sec. 8.5.2). [-]
AIRY_FWHM_COEFF: float = 1.0287938

#: ``2 sqrt(2 ln 2)``, converting a Gaussian FWHM to its standard deviation. [-]
GAUSSIAN_FWHM_TO_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))

#: ``sigma_u * sqrt(N)`` for a diffraction-limited spot [rad]. See module docstring.
SLOPE_NOISE_COEFF: float = 2.0 * np.pi * AIRY_FWHM_COEFF / GAUSSIAN_FWHM_TO_SIGMA


def photon_slope_noise(n_photons: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Photon-limited standard deviation of ``u`` [rad] for ``n_photons`` per subaperture.

    ``sigma_u = SLOPE_NOISE_COEFF / sqrt(N)``. See the module docstring for the
    derivation, the Gaussian-equivalent-spot approximation, and the list of
    effects it excludes.

    Parameters
    ----------
    n_photons : float or ndarray
        Detected photons (photoelectrons) per subaperture per measurement [-],
        > 0.

    Returns
    -------
    ndarray
        Standard deviation of the phase difference across one subaperture [rad].
    """
    n = np.asarray(n_photons, dtype=float)
    if np.any(~np.isfinite(n)) or np.any(n <= 0.0):
        raise ValueError("n_photons must be finite and > 0")
    return SLOPE_NOISE_COEFF / np.sqrt(n)


def add_slope_noise(
    u: NDArray[np.float64],
    n_photons: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Add zero-mean Gaussian photon noise to scaled slopes ``u`` [rad].

    Parameters
    ----------
    u : ndarray
        Noise-free scaled slopes [rad]; any shape.
    n_photons : float
        Photons per subaperture [-], > 0.
    rng : numpy.random.Generator
        Source of randomness; pass a seeded generator for reproducibility.

    Notes
    -----
    A Gaussian is used because the centroid is a sum of many photon positions;
    at ``N`` of order ten this is a rough approximation (module docstring).
    """
    arr = np.asarray(u, dtype=float)
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
    sigma = float(photon_slope_noise(n_photons))
    return arr + sigma * rng.standard_normal(arr.shape)


def random_dropout_mask(
    n_sub: int, rate: float, rng: np.random.Generator, n_samples: int = 1
) -> NDArray[np.bool_]:
    """Independent Bernoulli subaperture-availability masks.

    Parameters
    ----------
    n_sub : int
        Number of illuminated subapertures [-], >= 1.
    rate : float
        Probability that a subaperture is **lost** [-], in ``[0, 1)``.
    rng : numpy.random.Generator
        Source of randomness.
    n_samples : int
        Number of independent masks [-], >= 1.

    Returns
    -------
    ndarray of bool, shape (n_samples, n_sub)
        ``True`` where the subaperture is available.

    Notes
    -----
    Independent Bernoulli loss models random per-subaperture failures --
    vignetted, dead or saturated lenslets, or a flux threshold not met. It does
    **not** model spatially correlated loss such as a central obscuration, a
    spider shadow, or a contiguous detector defect, which are harder for any
    reconstructor because they can disconnect a region of the phase grid.
    """
    if isinstance(n_sub, bool) or not isinstance(n_sub, (int, np.integer)) or n_sub < 1:
        raise ValueError(f"n_sub must be an integer >= 1, got {n_sub!r}")
    r = float(rate)
    if not np.isfinite(r) or not (0.0 <= r < 1.0):
        raise ValueError(f"rate must be in [0, 1), got {rate!r}")
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise TypeError(f"n_samples must be an integer, got {n_samples!r}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
    return rng.random((int(n_samples), int(n_sub))) >= r
