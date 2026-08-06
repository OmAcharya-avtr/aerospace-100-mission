"""Classical centroid estimators: centre of gravity and quad-cell.

References (real, standard literature)
--------------------------------------
* K. A. Winick, "Cramer-Rao lower bounds on the performance of
  charge-coupled-device optical position estimators", J. Opt. Soc. Am. A 3,
  1809-1815 (1986) -- fundamental accuracy limits of CCD centroiding.
* S. Thomas, T. Fusco, A. Tokovinin, M. Nicolle, V. Michau, G. Rousset,
  "Comparison of centroid computation algorithms in a Shack-Hartmann
  sensor", Mon. Not. R. Astron. Soc. 371, 323-336 (2006) -- CoG,
  thresholded CoG and related estimators under photon/read noise.
* G. A. Tyler, D. L. Fried, "Image-position error associated with a quadrant
  detector", J. Opt. Soc. Am. 72, 804-808 (1982) -- quad-cell position
  estimation and its error.
* J. W. Hardy, "Adaptive Optics for Astronomical Telescopes", Oxford Univ.
  Press (1998) -- quad-cell response and linear-range discussion.

Coordinate convention: positions are in pixels measured from the geometric
centre of the array, (N-1)/2; x along columns (+x = increasing column
index), y along rows (+y = increasing row index).
"""

from __future__ import annotations

import numpy as np

__all__ = ["cog_centroid", "quadcell_centroid"]


def _check_image(img: np.ndarray) -> np.ndarray:
    """Validate a 2-D image array; return it as float64 or raise."""
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2-D, got {arr.ndim}-D array of shape {arr.shape}")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"image must be at least 2x2, got shape {arr.shape}")
    arr = arr.astype(float, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError("image contains NaN or Inf values")
    return arr


def cog_centroid(img: np.ndarray, threshold: float | None = None) -> tuple[float, float]:
    """Intensity-weighted centroid (centre of gravity).

    x_hat = sum_i w_i * x_i / sum_i w_i   [pixels],  same for y,

    with weights w_i = max(I_i - threshold, 0) (negative pixel values, e.g.
    from read noise, are clipped to zero).  The plain CoG is unbiased for a
    symmetric noise-free spot fully inside the window but degrades quickly
    with background/read noise; thresholding trades that noise sensitivity
    for a small bias (Thomas et al. 2006, MNRAS 371, 323).

    Parameters
    ----------
    img : ndarray, shape (H, W)
        Image [photoelectrons or any linear intensity unit].
    threshold : float, optional
        Threshold subtracted before weighting [same unit as ``img``].
        ``None`` (default) applies no threshold (only negative clipping).

    Returns
    -------
    (x, y) : tuple of float
        Centroid [pixels] from the array centre ((W-1)/2, (H-1)/2).

    Raises
    ------
    ValueError
        If the image is not 2-D/finite or the thresholded flux is <= 0.
    """
    arr = _check_image(img)
    if threshold is not None:
        thr = float(threshold)
        if not np.isfinite(thr):
            raise ValueError(f"threshold must be finite, got {threshold!r}")
        arr = arr - thr
    w = np.clip(arr, 0.0, None)
    total = w.sum()
    if total <= 0.0:
        raise ValueError(
            "total (thresholded) intensity is <= 0; centroid undefined. "
            "Lower the threshold or check the input image."
        )
    ny, nx = w.shape
    xs = np.arange(nx, dtype=float) - (nx - 1) / 2.0
    ys = np.arange(ny, dtype=float) - (ny - 1) / 2.0
    x = float((w.sum(axis=0) * xs).sum() / total)
    y = float((w.sum(axis=1) * ys).sum() / total)
    return x, y


def quadcell_centroid(img: np.ndarray, scale: float = 1.0) -> tuple[float, float]:
    """Quad-cell (quadrant detector) position estimate.

    The image is summed into four quadrants (split at the array centre;
    dimensions must be even) and

        x_hat = scale * (I_right - I_left) / I_total
        y_hat = scale * (I_bottom - I_top) / I_total      [pixels if scale
                                                           is in pixels]

    For a Gaussian spot of RMS width sigma displaced by d in x, the ideal
    response is (I_R - I_L)/I_tot = erf(d / (sigma * sqrt(2))), which is
    linear only for |d| << sigma with small-signal slope sqrt(2/pi)/sigma;
    the estimate saturates at +/-scale for large offsets (Tyler & Fried
    1982, JOSA 72, 804; Hardy 1998 ch. 5).  Choosing
    ``scale = sigma * sqrt(pi/2)`` calibrates the small-offset slope to
    unity for that spot size.

    Parameters
    ----------
    img : ndarray, shape (H, W), H and W even
        Image [linear intensity units]; negative pixels are clipped to 0.
    scale : float
        Output scaling [pixels]; default 1.0 returns the raw normalized
        imbalance (dimensionless, in [-1, 1]).

    Returns
    -------
    (x, y) : tuple of float
        Estimated position; units of ``scale``.

    Raises
    ------
    ValueError
        If the image is not 2-D/finite, has odd dimensions, or total
        intensity is <= 0.
    """
    arr = _check_image(img)
    scale = float(scale)
    if not np.isfinite(scale):
        raise ValueError(f"scale must be finite, got {scale!r}")
    ny, nx = arr.shape
    if ny % 2 or nx % 2:
        raise ValueError(f"quad-cell requires even image dimensions, got {arr.shape}")
    w = np.clip(arr, 0.0, None)
    total = w.sum()
    if total <= 0.0:
        raise ValueError("total intensity is <= 0; quad-cell estimate undefined.")
    hy, hx = ny // 2, nx // 2
    left = w[:, :hx].sum()
    right = w[:, hx:].sum()
    top = w[:hy, :].sum()
    bottom = w[hy:, :].sum()
    x = float(scale * (right - left) / total)
    y = float(scale * (bottom - top) / total)
    return x, y
