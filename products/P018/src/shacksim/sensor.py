"""Per-subaperture spot formation and the detector noise chain.

The optical model is deliberately simple and stated in full:

1. Each lenslet forms one spot. Its size is set by diffraction through the
   lenslet aperture, ``FWHM = 1.0288 lambda f / d`` (Born & Wolf 1999,
   sec. 8.5.2) — see `shacksim.geometry.LensletArray.spot_fwhm`.
2. The Airy core is approximated by a Gaussian of equal FWHM. Optionally the
   Gaussian is made elliptical (axis-aligned) to represent spot elongation.
3. The local wavefront gradient displaces the spot by ``dx = f * g / p``
   pixels — derivation in `LensletArray.slope_to_displacement`.
4. The detector adds a uniform background, Poisson shot noise on
   (spot + background), and additive Gaussian read noise.

References
----------
Born, M. & Wolf, E. (1999), *Principles of Optics*, 7th ed., CUP, sec. 8.5.2.
Hardy, J. W. (1998), *Adaptive Optics for Astronomical Telescopes*, OUP, ch. 5.
Thomas, S., Fusco, T., Tokovinin, A., Nicolle, M., Michau, V. & Rousset, G.
(2006), "Comparison of centroid computation algorithms in a Shack-Hartmann
sensor", *MNRAS* **371**, 323 — noise chain and centroid noise propagation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from .geometry import LensletArray

__all__ = [
    "subaperture_spot",
    "simulate_frame",
    "extract_subapertures",
    "generate_subaperture_dataset",
]


def _as_rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an int, a numpy Generator or None, got {type(seed).__name__}")
    return np.random.default_rng(int(seed))


def _check_nonneg(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0, got {v!r}")
    return v


def _erf_profile(n_pix: int, centre: float, sigma: float) -> NDArray[np.float64]:
    """Exact 1-D pixel integral of a unit Gaussian over ``n_pix`` unit pixels.

    ``P_i = F(u_i + 1/2) - F(u_i - 1/2)`` with
    ``F(u) = 0.5 (1 + erf((u - centre) / (sigma sqrt(2))))``.
    Coordinates are pixels from the block centre ``(n_pix - 1)/2``.
    """
    u = np.arange(n_pix, dtype=float) - (n_pix - 1) / 2.0
    a = (u + 0.5 - centre) / (sigma * np.sqrt(2.0))
    b = (u - 0.5 - centre) / (sigma * np.sqrt(2.0))
    return 0.5 * (erf(a) - erf(b))


def subaperture_spot(
    array: LensletArray,
    dx_px: float,
    dy_px: float,
    photons: float = 1000.0,
    sigma_x_px: float | None = None,
    sigma_y_px: float | None = None,
) -> NDArray[np.float64]:
    """Noise-free spot on one subaperture's pixel block.

    Parameters
    ----------
    array: lenslet geometry.
    dx_px, dy_px: spot displacement from the block centre [pixels].
    photons: total signal in the (untruncated) spot [photoelectrons], >= 0.
    sigma_x_px, sigma_y_px: Gaussian RMS widths [pixels]. Default: the
        diffraction-limited ``array.spot_sigma_px`` on both axes.

    Returns
    -------
    ``(pixels_per_sub, pixels_per_sub)`` array in photoelectrons.

    Notes
    -----
    Flux that falls outside the block is *lost*, so the returned sum is
    slightly below `photons` for large displacements. Only axis-aligned
    elongation is representable — a Gaussian rotated to an arbitrary angle is
    not separable in x and y and is not implemented (README Limitations).
    """
    n = array.pixels_per_sub
    sx = array.spot_sigma_px if sigma_x_px is None else float(sigma_x_px)
    sy = array.spot_sigma_px if sigma_y_px is None else float(sigma_y_px)
    if not np.isfinite(sx) or sx <= 0 or not np.isfinite(sy) or sy <= 0:
        raise ValueError(f"spot sigmas must be finite and > 0, got ({sx!r}, {sy!r})")
    photons = _check_nonneg("photons", photons)
    if not np.isfinite(dx_px) or not np.isfinite(dy_px):
        raise ValueError(f"displacement must be finite, got ({dx_px!r}, {dy_px!r})")
    px = _erf_profile(n, float(dx_px), sx)
    py = _erf_profile(n, float(dy_px), sy)
    return photons * np.outer(py, px)


def simulate_frame(
    array: LensletArray,
    slopes: NDArray[np.float64],
    photons: float | NDArray[np.float64] = 1000.0,
    background: float = 0.0,
    read_noise: float = 0.0,
    elongation: float = 1.0,
    elongation_axis: str = "x",
    shot_noise: bool = True,
    seed: int | np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Simulate a full Shack-Hartmann detector frame.

    Parameters
    ----------
    array: lenslet geometry.
    slopes: ``(n_valid, 2)`` wavefront gradients ``(g_x, g_y)`` [rad], ordered
        row-major over the illuminated subapertures (`LensletArray.valid_mask`).
    photons: total signal per subaperture [photoelectrons]. Scalar, or an
        ``(n_valid,)`` array for a non-uniform pupil illumination. >= 0.
    background: uniform background [photoelectrons per pixel], >= 0. Applied
        over the whole frame, including unilluminated subapertures.
    read_noise: Gaussian read noise [photoelectrons RMS per pixel], >= 0.
    elongation: ratio of the elongated to the diffraction-limited spot width
        [-], >= 1. ``1.0`` = round diffraction-limited spot.
    elongation_axis: ``"x"`` or ``"y"`` — which axis is stretched.
    shot_noise: apply Poisson noise to (spot + background).
    seed: int, ``numpy.random.Generator`` or None. A fixed int gives bitwise
        reproducible output.

    Returns
    -------
    ``(image_size, image_size)`` frame in photoelectrons. Values may be
    negative because additive read noise is not clipped.

    Notes
    -----
    Noise chain, in order: signal + background -> Poisson -> + N(0, read^2).
    This is the standard photon-counting detector model (Thomas et al. 2006).
    Shot noise is independent per pixel; there is no charge diffusion, no
    inter-subaperture crosstalk and no dark current.
    """
    slopes = np.asarray(slopes, dtype=float)
    n_valid = array.n_valid
    if slopes.shape != (n_valid, 2):
        raise ValueError(
            f"slopes must have shape ({n_valid}, 2) — one (gx, gy) per illuminated "
            f"subaperture — got {slopes.shape}"
        )
    if not np.all(np.isfinite(slopes)):
        raise ValueError("slopes must all be finite")
    elong = float(elongation)
    if not np.isfinite(elong) or elong < 1.0:
        raise ValueError(f"elongation must be >= 1, got {elongation!r}")
    if elongation_axis not in ("x", "y"):
        raise ValueError(f"elongation_axis must be 'x' or 'y', got {elongation_axis!r}")
    background = _check_nonneg("background", background)
    read_noise = _check_nonneg("read_noise", read_noise)

    flux = np.asarray(photons, dtype=float)
    if flux.ndim == 0:
        flux = np.full(n_valid, float(flux))
    elif flux.shape != (n_valid,):
        raise ValueError(f"photons must be scalar or shape ({n_valid},), got {flux.shape}")
    if np.any(flux < 0) or not np.all(np.isfinite(flux)):
        raise ValueError("photons must be finite and >= 0")

    sigma = array.spot_sigma_px
    sx = sigma * (elong if elongation_axis == "x" else 1.0)
    sy = sigma * (elong if elongation_axis == "y" else 1.0)

    disp = array.slope_to_displacement(slopes)
    image = np.zeros((array.image_size, array.image_size), dtype=float)
    mask = array.valid_mask()
    idx = 0
    for row in range(array.n_lenslets):
        for col in range(array.n_lenslets):
            if not mask[row, col]:
                continue
            rs, cs = array.subaperture_slice(row, col)
            image[rs, cs] = subaperture_spot(
                array, disp[idx, 0], disp[idx, 1], flux[idx], sx, sy
            )
            idx += 1

    image += background
    rng = _as_rng(seed)
    if shot_noise:
        image = rng.poisson(np.clip(image, 0.0, None)).astype(float)
    if read_noise > 0.0:
        image = image + rng.normal(0.0, read_noise, size=image.shape)
    return image


def extract_subapertures(
    image: NDArray[np.float64], array: LensletArray
) -> NDArray[np.float64]:
    """Cut a full frame into the stack of illuminated subaperture stamps.

    Returns ``(n_valid, pixels_per_sub, pixels_per_sub)`` in the same row-major
    order as `LensletArray.valid_centres`.
    """
    image = np.asarray(image, dtype=float)
    n = array.image_size
    if image.shape != (n, n):
        raise ValueError(f"image must have shape ({n}, {n}), got {image.shape}")
    mask = array.valid_mask()
    p = array.pixels_per_sub
    out = np.empty((array.n_valid, p, p), dtype=float)
    idx = 0
    for row in range(array.n_lenslets):
        for col in range(array.n_lenslets):
            if not mask[row, col]:
                continue
            rs, cs = array.subaperture_slice(row, col)
            out[idx] = image[rs, cs]
            idx += 1
    return out


def generate_subaperture_dataset(
    array: LensletArray,
    n_samples: int,
    photons: float | tuple[float, float] = 1000.0,
    background: float = 0.0,
    read_noise: float = 0.0,
    elongation: float | tuple[float, float] = 1.0,
    elongation_axis: str = "x",
    slope_fraction: float = 0.6,
    shot_noise: bool = True,
    seed: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Draw independent single-subaperture stamps with exact slope labels.

    This is the training/test data generator for the learned estimator. Each
    sample is one subaperture stamp; the label is the exact slope that
    generated it, so there is no label noise.

    Parameters
    ----------
    array: lenslet geometry (sets stamp size and spot width).
    n_samples: number of stamps, >= 1.
    photons: signal per stamp [photoelectrons]. Scalar for a fixed level, or
        ``(lo, hi)`` to draw log-uniformly in ``[lo, hi]``.
    background: background [photoelectrons/pixel].
    read_noise: read noise [photoelectrons RMS].
    elongation: spot elongation ratio [-], scalar or ``(lo, hi)`` drawn
        uniformly.
    elongation_axis: ``"x"`` or ``"y"``.
    slope_fraction: slopes are drawn uniformly in
        ``+/- slope_fraction * array.max_slope`` [-], in (0, 1].
    shot_noise: apply Poisson noise.
    seed: int / Generator / None.

    Returns
    -------
    ``(stamps, slopes)`` with shapes ``(n_samples, p, p)`` [photoelectrons] and
    ``(n_samples, 2)`` [rad].
    """
    if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, (bool, np.bool_)):
        raise TypeError(f"n_samples must be an int, got {type(n_samples).__name__}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    frac = float(slope_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"slope_fraction must be in (0, 1], got {slope_fraction!r}")
    if elongation_axis not in ("x", "y"):
        raise ValueError(f"elongation_axis must be 'x' or 'y', got {elongation_axis!r}")
    background = _check_nonneg("background", background)
    read_noise = _check_nonneg("read_noise", read_noise)
    rng = _as_rng(seed)

    limit = frac * array.max_slope
    slopes = rng.uniform(-limit, limit, size=(n_samples, 2))

    if np.ndim(photons) == 0:
        flux = np.full(n_samples, _check_nonneg("photons", photons))
    else:
        lo, hi = (float(v) for v in photons)
        if not (0.0 < lo <= hi):
            raise ValueError(f"photons range must satisfy 0 < lo <= hi, got {photons!r}")
        flux = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n_samples))

    if np.ndim(elongation) == 0:
        elong = np.full(n_samples, float(elongation))
    else:
        lo_e, hi_e = (float(v) for v in elongation)
        if not (1.0 <= lo_e <= hi_e):
            raise ValueError(f"elongation range must satisfy 1 <= lo <= hi, got {elongation!r}")
        elong = rng.uniform(lo_e, hi_e, size=n_samples)
    if np.any(elong < 1.0):
        raise ValueError("elongation must be >= 1")

    p = array.pixels_per_sub
    disp = array.slope_to_displacement(slopes)
    stamps = np.empty((n_samples, p, p), dtype=float)
    sigma = array.spot_sigma_px
    for i in range(n_samples):
        sx = sigma * (elong[i] if elongation_axis == "x" else 1.0)
        sy = sigma * (elong[i] if elongation_axis == "y" else 1.0)
        stamps[i] = subaperture_spot(array, disp[i, 0], disp[i, 1], flux[i], sx, sy)

    stamps += background
    if shot_noise:
        stamps = rng.poisson(np.clip(stamps, 0.0, None)).astype(float)
    if read_noise > 0.0:
        stamps = stamps + rng.normal(0.0, read_noise, size=stamps.shape)
    return stamps, slopes
