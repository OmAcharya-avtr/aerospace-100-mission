"""Synthetic optical-spot image generator for centroid-estimation studies.

Physical model
--------------
A single 2-D Gaussian spot (a common approximation of a diffraction-limited
point-spread function near its core; see Hardy 1998, "Adaptive Optics for
Astronomical Telescopes", Oxford Univ. Press, ch. 5) is imaged onto an N x N
pixel grid:

    I(x, y) = S / (2 pi sigma^2) * exp(-((x - x0)^2 + (y - y0)^2) / (2 sigma^2))

with
    S      total signal [photoelectrons], S >= 0
    sigma  spot RMS width [pixels], sigma > 0
    x0,y0  true spot centre [pixels], measured from the geometric centre of
           the array, x along columns (+x = increasing column index),
           y along rows (+y = increasing row index).

Pixelation: when ``pixelated=True`` each pixel integrates the Gaussian over
its unit square using the error function,

    P(i, j) = S * [F(xc+1/2) - F(xc-1/2)] * [F(yc+1/2) - F(yc-1/2)],
    F(u)    = 0.5 * (1 + erf((u - x0) / (sigma * sqrt(2)))),

which is the exact integral of the Gaussian over the pixel (separable in x
and y).  With ``pixelated=False`` the Gaussian is point-sampled at pixel
centres (an approximation valid for sigma >> 1 px).

Noise model (idealized sensor):
    * shot noise: Poisson on (spot + background) photoelectrons,
    * read noise: additive zero-mean Gaussian, std ``read_noise`` [e-],
    * uniform background ``background`` [e-/pixel].

Assumptions / validity range: single unresolved spot, uniform pixel response
(no PRNU/DSNU), no dead pixels, no optical aberrations beyond the Gaussian
approximation, no stray light, linear detector, no saturation.  See
DATASET_CARD.md for the full list of unmodelled effects.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf

__all__ = ["generate_spots", "spot_image", "snr_estimate"]


def _validate_scalar(name: str, value: float, minimum: float, strict: bool = False) -> float:
    """Validate a scalar parameter; return it as float or raise ValueError."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if strict and value <= minimum:
        raise ValueError(f"{name} must be > {minimum}, got {value}")
    if not strict and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def spot_image(
    x0: float,
    y0: float,
    grid_size: int = 16,
    sigma: float = 1.5,
    signal: float = 1000.0,
    pixelated: bool = True,
) -> np.ndarray:
    """Noise-free Gaussian spot image [photoelectrons].

    Parameters
    ----------
    x0, y0 : float
        True spot centre [pixels] from the array centre ``(grid_size - 1) / 2``;
        x along columns, y along rows.
    grid_size : int
        Side length N of the square image, N >= 4.
    sigma : float
        Spot RMS width [pixels], > 0.
    signal : float
        Total signal [photoelectrons], >= 0 (fully captured only for a spot
        well inside the array; flux falling off-array is truncated).
    pixelated : bool
        If True, integrate the Gaussian over each pixel (erf model);
        otherwise point-sample at pixel centres.

    Returns
    -------
    numpy.ndarray, shape (N, N), float64
        Expected photoelectron count per pixel.
    """
    if not isinstance(grid_size, (int, np.integer)):
        raise TypeError(f"grid_size must be an int, got {type(grid_size).__name__}")
    if grid_size < 4:
        raise ValueError(f"grid_size must be >= 4, got {grid_size}")
    sigma = _validate_scalar("sigma", sigma, 0.0, strict=True)
    signal = _validate_scalar("signal", signal, 0.0)
    x0 = _validate_scalar("x0", x0, -np.inf)
    y0 = _validate_scalar("y0", y0, -np.inf)

    centre = (grid_size - 1) / 2.0
    coords = np.arange(grid_size, dtype=float) - centre  # pixel centres [px]
    if pixelated:
        # Exact integral of the normalized Gaussian over each unit pixel.
        edges = np.arange(grid_size + 1, dtype=float) - 0.5 - centre
        fx = 0.5 * (1.0 + erf((edges - x0) / (sigma * np.sqrt(2.0))))
        fy = 0.5 * (1.0 + erf((edges - y0) / (sigma * np.sqrt(2.0))))
        px = np.diff(fx)
        py = np.diff(fy)
        img = signal * np.outer(py, px)  # rows = y, cols = x
    else:
        gx = np.exp(-((coords - x0) ** 2) / (2.0 * sigma**2))
        gy = np.exp(-((coords - y0) ** 2) / (2.0 * sigma**2))
        img = signal / (2.0 * np.pi * sigma**2) * np.outer(gy, gx)
    return img


def generate_spots(
    n_spots: int = 100,
    grid_size: int = 16,
    sigma: float = 1.5,
    signal: float = 1000.0,
    background: float = 0.5,
    read_noise: float = 2.0,
    shot_noise: bool = True,
    pixelated: bool = True,
    offset_range: float = 2.0,
    offsets: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a batch of synthetic spot images with known true centroids.

    Parameters
    ----------
    n_spots : int
        Number of images, >= 1 (ignored if ``offsets`` is given).
    grid_size : int
        Image side length N [pixels], >= 4.
    sigma : float
        Spot RMS width [pixels], > 0.
    signal : float
        Total spot signal [photoelectrons], >= 0.
    background : float
        Uniform background level [photoelectrons/pixel], >= 0.
    read_noise : float
        Gaussian read-noise standard deviation [photoelectrons], >= 0.
        0 disables read noise.
    shot_noise : bool
        Apply Poisson shot noise to spot + background. False disables it.
    pixelated : bool
        Integrate the Gaussian over pixels (erf model) instead of
        point-sampling; see :func:`spot_image`.
    offset_range : float
        True centroids drawn uniformly in [-offset_range, +offset_range]
        pixels in x and y (from the array centre).
    offsets : ndarray shape (M, 2), optional
        Explicit true centroids [pixels]; overrides ``n_spots``/``offset_range``.
    seed : int, optional
        Seed for :func:`numpy.random.default_rng`; fixed seed gives bitwise
        reproducible output.

    Returns
    -------
    images : numpy.ndarray, shape (M, N, N), float64
        Simulated frames [photoelectrons] (may contain negative values from
        read noise).
    truths : numpy.ndarray, shape (M, 2), float64
        True (x, y) centroids [pixels] from the array centre.
    """
    background = _validate_scalar("background", background, 0.0)
    read_noise = _validate_scalar("read_noise", read_noise, 0.0)
    offset_range = _validate_scalar("offset_range", offset_range, 0.0)

    rng = np.random.default_rng(seed)
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] != 2:
            raise ValueError(f"offsets must have shape (M, 2), got {offsets.shape}")
        if not np.all(np.isfinite(offsets)):
            raise ValueError("offsets must be finite")
    else:
        if not isinstance(n_spots, (int, np.integer)) or n_spots < 1:
            raise ValueError(f"n_spots must be a positive int, got {n_spots!r}")
        offsets = rng.uniform(-offset_range, offset_range, size=(n_spots, 2))

    images = np.empty((offsets.shape[0], grid_size, grid_size), dtype=float)
    for k, (x0, y0) in enumerate(offsets):
        clean = spot_image(x0, y0, grid_size, sigma, signal, pixelated) + background
        frame = rng.poisson(clean).astype(float) if shot_noise else clean.copy()
        if read_noise > 0.0:
            frame += rng.normal(0.0, read_noise, size=frame.shape)
        images[k] = frame
    return images, offsets


def snr_estimate(
    signal: float,
    background: float,
    read_noise: float,
    grid_size: int = 16,
) -> float:
    """Detection signal-to-noise ratio of a spot summed over the full window.

    SNR = S / sqrt(S + Npix * (B + R^2))   [dimensionless]

    Standard CCD aperture-photometry SNR expression (shot noise on signal and
    background plus read-noise variance summed over Npix = grid_size^2
    pixels); see e.g. Howell 2006, "Handbook of CCD Astronomy", 2nd ed.,
    Cambridge Univ. Press, eq. for the CCD signal-to-noise ratio.  Assumes the
    whole spot is inside the window and no dark current.

    Parameters
    ----------
    signal : float
        Total spot signal S [photoelectrons], >= 0.
    background : float
        Background B [photoelectrons/pixel], >= 0.
    read_noise : float
        Read noise R [photoelectrons RMS], >= 0.
    grid_size : int
        Window side length [pixels].

    Returns
    -------
    float
        Dimensionless SNR (0 if signal is 0).
    """
    signal = _validate_scalar("signal", signal, 0.0)
    background = _validate_scalar("background", background, 0.0)
    read_noise = _validate_scalar("read_noise", read_noise, 0.0)
    npix = float(grid_size) ** 2
    var = signal + npix * (background + read_noise**2)
    if var <= 0.0:
        return float("inf") if signal > 0 else 0.0
    return float(signal / np.sqrt(var))
