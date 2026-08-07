"""Classical slope-extraction baselines and their noise-propagation model.

Implemented **before** the learned estimator (`shacksim.ml`), which is
benchmarked against `cog_slopes` on identical held-out data.

Two estimators, both standard:

* **Thresholded centre of gravity (CoG).** The first moment of the
  subaperture intensity after subtracting a threshold and clipping negatives.
  Cheap, unbiased for a symmetric spot in a noise-free, background-free
  window, and the default in essentially every Shack-Hartmann system
  (Hardy 1998, ch. 5; Thomas et al. 2006, MNRAS 371, 323).
* **Correlation.** Cross-correlate the subaperture with a reference spot
  template and locate the correlation peak to subpixel accuracy by 3-point
  parabolic interpolation. Standard for extended/elongated spots and
  scene-based sensing (Poyneer 2003, *Appl. Opt.* **42**, 5807; Thomas et al.
  2006 compares it against the CoG family).

References
----------
Hardy, J. W. (1998), *Adaptive Optics for Astronomical Telescopes*, Oxford
University Press, ch. 5 (Shack-Hartmann sensor, centroid noise).
Thomas, S. et al. (2006), "Comparison of centroid computation algorithms in a
Shack-Hartmann sensor", *MNRAS* **371**, 323.
Poyneer, L. A. (2003), "Scene-based Shack-Hartmann wavefront sensing: analysis
and simulation", *Applied Optics* **42**, 5807.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate

from .geometry import LensletArray
from .sensor import extract_subapertures, subaperture_spot

__all__ = [
    "cog_displacement",
    "cog_slopes",
    "reference_template",
    "correlation_displacement",
    "correlation_slopes",
    "cog_noise_sigma",
]


def _as_stamps(stamps: NDArray[np.float64]) -> NDArray[np.float64]:
    s = np.asarray(stamps, dtype=float)
    if s.ndim == 2:
        s = s[None, ...]
    if s.ndim != 3 or s.shape[1] != s.shape[2]:
        raise ValueError(
            f"stamps must have shape (n, p, p) or (p, p) with a square stamp, got {s.shape}"
        )
    if s.shape[1] < 3:
        raise ValueError(f"stamp side must be >= 3 pixels, got {s.shape[1]}")
    if not np.all(np.isfinite(s)):
        raise ValueError("stamps contain non-finite values (NaN or inf)")
    return s


def _pixel_axis(n: int) -> NDArray[np.float64]:
    """Pixel coordinates from the stamp centre ``(n - 1) / 2`` [pixels]."""
    return np.arange(n, dtype=float) - (n - 1) / 2.0


def cog_displacement(
    stamps: NDArray[np.float64],
    threshold: float = 0.0,
    clip_negative: bool = True,
) -> NDArray[np.float64]:
    """Thresholded centre-of-gravity spot displacement [pixels].

    .. math::  \\hat{x} = \\frac{\\sum_i w_i x_i}{\\sum_i w_i},
               \\qquad w_i = \\max(I_i - t,\\, 0)

    with ``x_i`` measured from the stamp centre ``(p - 1) / 2``
    (Thomas et al. 2006).

    Parameters
    ----------
    stamps: ``(n, p, p)`` or ``(p, p)`` subaperture intensities
        [photoelectrons].
    threshold: ``t`` [photoelectrons], subtracted from every pixel before
        weighting. >= 0. A common choice is ``background + k * read_noise``.
    clip_negative: clip the weights at zero. ``True`` (default) is the usual
        practical estimator; ``False`` keeps the estimator linear in the pixel
        values, which is the form for which the noise propagation of
        `cog_noise_sigma` is derived.

    Returns
    -------
    ``(n, 2)`` array of ``(dx, dy)`` displacements [pixels].

    Notes
    -----
    Where the total weight is zero (an empty or fully sub-threshold stamp) the
    estimate returned is ``(0, 0)`` — the a-priori centre — because the
    estimator has no information. This is a defined fallback, not a
    measurement; it biases such subapertures toward zero slope.

    *Assumptions:* symmetric spot fully inside the stamp, background removed
    by `threshold`. *Validity:* unbiased noise-free (validation §1); variance
    grows with window area times per-pixel noise variance (validation §2); a
    non-zero background that is **not** removed biases the estimate toward the
    stamp centre (validation §3).
    """
    s = _as_stamps(stamps)
    t = float(threshold)
    if not np.isfinite(t) or t < 0.0:
        raise ValueError(f"threshold must be finite and >= 0, got {threshold!r}")
    w = s - t
    if clip_negative:
        w = np.clip(w, 0.0, None)
    n = s.shape[1]
    axis = _pixel_axis(n)
    total = w.sum(axis=(1, 2))
    sx = (w.sum(axis=1) * axis).sum(axis=1)
    sy = (w.sum(axis=2) * axis).sum(axis=1)
    out = np.zeros((s.shape[0], 2), dtype=float)
    ok = total != 0.0
    out[ok, 0] = sx[ok] / total[ok]
    out[ok, 1] = sy[ok] / total[ok]
    return out


def cog_slopes(
    image: NDArray[np.float64],
    array: LensletArray,
    threshold: float = 0.0,
    clip_negative: bool = True,
) -> NDArray[np.float64]:
    """Wavefront slopes [rad] from a full frame by thresholded centre of gravity.

    Cuts the frame into illuminated subaperture stamps, centroids each one and
    converts pixels to slope with ``g = dx * p / f``
    (`LensletArray.displacement_to_slope`).

    Returns ``(n_valid, 2)`` slopes [rad] in `LensletArray.valid_centres`
    order.
    """
    stamps = extract_subapertures(image, array)
    disp = cog_displacement(stamps, threshold=threshold, clip_negative=clip_negative)
    return array.displacement_to_slope(disp)


def reference_template(
    array: LensletArray, elongation: float = 1.0, elongation_axis: str = "x"
) -> NDArray[np.float64]:
    """Noise-free, centred, unit-flux spot used as the correlation reference.

    Returns ``(pixels_per_sub, pixels_per_sub)``. For a real sensor this
    template would be measured on a calibration source; here it is generated
    from the same optical model, so the correlation estimator is given a
    *perfectly matched* template. That is optimistic — template mismatch is a
    real and unmodelled error source (README Limitations).
    """
    if elongation_axis not in ("x", "y"):
        raise ValueError(f"elongation_axis must be 'x' or 'y', got {elongation_axis!r}")
    e = float(elongation)
    if not np.isfinite(e) or e < 1.0:
        raise ValueError(f"elongation must be >= 1, got {elongation!r}")
    sigma = array.spot_sigma_px
    sx = sigma * (e if elongation_axis == "x" else 1.0)
    sy = sigma * (e if elongation_axis == "y" else 1.0)
    return subaperture_spot(array, 0.0, 0.0, 1.0, sx, sy)


def _parabolic_peak(c_minus: float, c_zero: float, c_plus: float) -> float:
    """Subpixel offset of a parabola through three samples [pixels].

    ``delta = (c_- - c_+) / (2 (c_- - 2 c_0 + c_+))``, the standard 3-point
    parabolic peak interpolator (Poyneer 2003). Returns 0 when the
    denominator vanishes or the sampled peak is not a maximum.
    """
    denom = c_minus - 2.0 * c_zero + c_plus
    if denom == 0.0 or denom >= 0.0:
        return 0.0
    delta = (c_minus - c_plus) / (2.0 * denom)
    if not np.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -1.0, 1.0))


def correlation_displacement(
    stamps: NDArray[np.float64],
    template: NDArray[np.float64],
    subtract_mean: bool = True,
) -> NDArray[np.float64]:
    """Correlation-based spot displacement [pixels].

    For each stamp, compute the full 2-D cross-correlation with `template`,
    find the integer peak, then refine it to subpixel accuracy with a 3-point
    parabolic fit along each axis independently (Poyneer 2003).

    Parameters
    ----------
    stamps: ``(n, p, p)`` or ``(p, p)`` subaperture intensities.
    template: ``(p, p)`` reference spot, e.g. from `reference_template`.
    subtract_mean: remove the mean of the stamp and of the template before
        correlating. This makes the estimate **invariant to a uniform
        background level** (measured: identical to 1e-15 px for B from 5 to
        500 e-/px, whereas the un-subtracted variant drifts with B). It is not
        uniformly more accurate: with a positive matched template the
        un-subtracted pedestal broadens the correlation peak, which happens to
        suit the parabolic interpolator, so in noise-free tests the
        un-subtracted variant can have the smaller sub-pixel bias. Default
        ``True`` because background invariance is the more useful property in
        a real sensor.

    Returns
    -------
    ``(n, 2)`` ``(dx, dy)`` displacements [pixels].

    Notes
    -----
    *Assumptions:* the template matches the true spot shape; the shift is
    within +/- (p - 1) pixels; the correlation peak is well sampled.
    *Validity:* the parabolic interpolator is exact only for a parabolic peak,
    so it carries a small periodic "S-curve" bias with sub-pixel shift — a
    known and documented property of this interpolator, quantified in
    validation §1.
    """
    s = _as_stamps(stamps)
    tpl = np.asarray(template, dtype=float)
    if tpl.shape != s.shape[1:]:
        raise ValueError(
            f"template shape {tpl.shape} must match the stamp shape {s.shape[1:]}"
        )
    if not np.all(np.isfinite(tpl)):
        raise ValueError("template contains non-finite values")
    if subtract_mean:
        tpl = tpl - tpl.mean()
    if not np.any(tpl):
        raise ValueError("template is identically zero after mean subtraction")

    n_pix = s.shape[1]
    out = np.zeros((s.shape[0], 2), dtype=float)
    for i, stamp in enumerate(s):
        a = stamp - stamp.mean() if subtract_mean else stamp
        corr = correlate(a, tpl, mode="full")
        peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
        row, col = int(peak[0]), int(peak[1])
        dx = float(col - (n_pix - 1))
        dy = float(row - (n_pix - 1))
        if 0 < col < corr.shape[1] - 1:
            dx += _parabolic_peak(corr[row, col - 1], corr[row, col], corr[row, col + 1])
        if 0 < row < corr.shape[0] - 1:
            dy += _parabolic_peak(corr[row - 1, col], corr[row, col], corr[row + 1, col])
        out[i] = (dx, dy)
    return out


def correlation_slopes(
    image: NDArray[np.float64],
    array: LensletArray,
    template: NDArray[np.float64] | None = None,
    subtract_mean: bool = True,
) -> NDArray[np.float64]:
    """Wavefront slopes [rad] from a full frame by correlation.

    `template` defaults to the diffraction-limited `reference_template`.
    Returns ``(n_valid, 2)`` slopes [rad].
    """
    stamps = extract_subapertures(image, array)
    tpl = reference_template(array) if template is None else template
    disp = correlation_displacement(stamps, tpl, subtract_mean=subtract_mean)
    return array.displacement_to_slope(disp)


def cog_noise_sigma(
    array: LensletArray,
    photons: float,
    background: float = 0.0,
    read_noise: float = 0.0,
    elongation: float = 1.0,
    elongation_axis: str = "x",
    axis: str = "x",
    displacement_px: float | NDArray[np.float64] = 0.0,
) -> float | NDArray[np.float64]:
    """Standard noise-propagation prediction for the CoG slope error [rad].

    Derivation (linear, un-thresholded CoG; the standard first-order result)
    ---------------------------------------------------------------------
    Let pixel ``i`` collect ``n_i = N p_i + e_i`` photoelectrons, where ``N``
    is the total signal, ``p_i`` the normalized noise-free spot profile
    (``sum p_i = 1``) and ``e_i`` zero-mean noise with

        Var(e_i) = N p_i          (spot shot noise, Poisson)
                 + B              (background shot noise, Poisson)
                 + R^2            (read noise).

    Linearizing ``xhat = sum n_i x_i / sum n_i`` about the noiseless value and
    holding the denominator at ``N`` gives

        xhat - x_true ~= (1/N) sum_i e_i (x_i - xbar)

    and therefore

        Var(xhat) = M2 / N + (B + R^2) / N^2 * sum_i (x_i - xbar)^2      (1)

    where ``M2 = sum_i p_i (x_i - xbar)^2`` is the second central moment of the
    spot profile [px^2], ``xbar`` is the spot position, and the second sum runs
    over **every pixel of the window**. The first term is the photon-noise
    floor
    ``sigma_x = sqrt(M2)/sqrt(N)``, the classical ``sigma_spot / sqrt(N)``
    result (Hardy 1998, ch. 5; Thomas et al. 2006, eq. for the CoG variance;
    the photon-limited form also appears in Winick 1986, *JOSA A* **3**, 1809).
    The second term is the read/background term: it grows as the *window area*
    because every pixel contributes noise with a lever arm ``x_i``, which is
    exactly why thresholding or windowing helps at low flux.

    For a ``p x p`` window whose pixel coordinates are centred on the block
    (so ``sum_i x_i = 0``) and a spot displaced by ``d`` pixels along x,

        sum_i (x_i - d)^2 = p^2 (p^2 - 1) / 12 + p^2 d^2                 (1a)

    — the read-noise lever arm grows when the spot sits off centre, because the
    noisy pixels furthest from the spot get the largest weight. Ignoring the
    ``p^2 d^2`` term under-predicts the error by 16 % for the default geometry
    with slopes drawn over +/- 0.6 of the field, which is measurable, so it is
    kept.

    Slope units follow from ``g = x * p_pix / f``:

        sigma_g = sqrt(Var(xhat)) * pixel_angle                          (2)

    In the pure-photon limit and for a diffraction-limited spot
    (``sqrt(M2) ~ 0.437 lambda f / (d p_pix)`` px) equation (2) reduces to
    ``sigma_g ~ 0.437 (lambda / d) / sqrt(N)``, i.e. the noise-equivalent
    angle scales as the diffraction angle divided by the square root of the
    photon count — the standard Shack-Hartmann scaling (Hardy 1998, ch. 5).

    Here ``M2`` is evaluated **numerically** from the actual pixel-integrated,
    window-truncated spot profile rather than from the Gaussian ``sigma``, so
    pixelation (Sheppard's correction, +1/12 px^2) and edge truncation are
    included exactly.

    Parameters
    ----------
    array: lenslet geometry.
    photons: total signal per subaperture ``N`` [photoelectrons], > 0.
    background: ``B`` [photoelectrons/pixel], >= 0.
    read_noise: ``R`` [photoelectrons RMS], >= 0.
    elongation, elongation_axis: spot shape used to compute ``M2``.
    axis: ``"x"`` or ``"y"`` — which slope component to predict (they differ
        when the spot is elongated).
    displacement_px: spot displacement ``d`` along `axis` [pixels]. Scalar or
        array; enters only through the lever term (1a). Default 0 = centred.

    Returns
    -------
    Predicted 1-sigma slope error [rad] — a float for a scalar
    `displacement_px`, otherwise an array of the same shape.

    Validity
    --------
    First-order (small-noise) linearization of a ratio estimator; assumes no
    thresholding and no negative clipping, a centred spot, and independent
    per-pixel noise. It under-predicts the error once the per-frame SNR is so
    low that the denominator fluctuates by an appreciable fraction, and it
    does not describe the thresholded estimator at all (thresholding removes
    part of the read-noise term at the price of a bias).
    """
    n_photons = float(photons)
    if not np.isfinite(n_photons) or n_photons <= 0.0:
        raise ValueError(f"photons must be finite and > 0, got {photons!r}")
    b = float(background)
    r = float(read_noise)
    if not np.isfinite(b) or b < 0 or not np.isfinite(r) or r < 0:
        raise ValueError(f"background and read_noise must be finite and >= 0, got ({b!r}, {r!r})")
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    profile = reference_template(array, elongation=elongation, elongation_axis=elongation_axis)
    profile = profile / profile.sum()
    n_pix = array.pixels_per_sub
    coord = _pixel_axis(n_pix)
    marginal = profile.sum(axis=0) if axis == "x" else profile.sum(axis=1)
    mean = float((marginal * coord).sum())
    m2 = float((marginal * (coord - mean) ** 2).sum())

    d = np.asarray(displacement_px, dtype=float)
    if not np.all(np.isfinite(d)):
        raise ValueError("displacement_px must be finite")
    lever = n_pix**2 * (n_pix**2 - 1) / 12.0 + n_pix**2 * d**2
    var_px = m2 / n_photons + (b + r**2) / n_photons**2 * lever
    sigma = np.sqrt(var_px) * array.pixel_angle
    return float(sigma) if d.ndim == 0 else sigma
