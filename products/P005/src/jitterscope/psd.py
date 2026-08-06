"""Welch PSD estimation and band-limited RMS jitter integration.

Theory
------
The one-sided power spectral density (PSD) ``S(f)`` of a zero-mean
stationary process ``x(t)`` satisfies (Parseval / Wiener-Khinchin,
see Bendat & Piersol 2010, "Random Data: Analysis and Measurement
Procedures", 4th ed., ch. 5)::

    var(x) = sigma^2 = integral_0^inf S(f) df

so band-limited RMS jitter in a band ``[f1, f2]`` is::

    sigma_band = sqrt( integral_f1^f2 S(f) df )

Units: if ``x`` is a pointing angle in rad, ``S(f)`` is rad^2/Hz and
``sigma_band`` is rad RMS.

The PSD is estimated with Welch's method (Welch 1967, IEEE Trans.
Audio Electroacoust. 15(2):70-73): the record is split into
overlapping segments, each windowed and periodogram-averaged, trading
frequency resolution for variance reduction. Assumptions: ``x`` is
(wide-sense) stationary over the record and adequately sampled
(no aliasing); validity degrades for strongly nonstationary data.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _sig

__all__ = ["psd", "band_rms", "cumulative_rms"]


def _validate_signal(x: np.ndarray) -> np.ndarray:
    """Validate a 1-D real telemetry array. NaN/Inf policy: raise.

    NaN handling policy (documented, tested): non-finite samples are
    rejected with ``ValueError``. Gap-filling or interpolation is left
    to the caller because silently imputing telemetry can mask real
    dropouts; see README 'Limitations'.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if x.size < 8:
        raise ValueError(f"x must have at least 8 samples, got {x.size}")
    if not np.all(np.isfinite(x)):
        n_bad = int(np.sum(~np.isfinite(x)))
        raise ValueError(
            f"x contains {n_bad} non-finite samples (NaN/Inf). "
            "Clean or interpolate gaps before PSD estimation; "
            "jitterscope does not impute telemetry."
        )
    return x


def psd(
    x: np.ndarray,
    fs: float,
    **welch_kw: object,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD estimate of a jitter/vibration record.

    Parameters
    ----------
    x : array_like, shape (n,)
        Telemetry samples (e.g. pointing angle [rad], acceleration
        [m/s^2]). Must be finite; NaN raises ``ValueError``.
    fs : float
        Sample rate [Hz], > 0.
    **welch_kw
        Passed through to :func:`scipy.signal.welch`. The exposed
        knobs and their defaults here:

        - ``nperseg`` (default ``min(len(x), 1024)``): segment length;
          frequency resolution is ``fs / nperseg`` [Hz].
        - ``window`` (default ``"hann"``): taper controlling spectral
          leakage (Hann: -31.5 dB first sidelobe, Harris 1978,
          Proc. IEEE 66(1)).
        - ``noverlap`` (default ``nperseg // 2``): 50 % overlap is the
          usual variance/efficiency compromise for Hann (Welch 1967).
        - ``detrend`` (default ``"constant"``): mean removal per segment.
        - ``average`` (default ``"mean"``; ``"median"`` is robust to
          sporadic transients).

    Returns
    -------
    f : ndarray
        Frequency bins [Hz], from 0 to ``fs/2``.
    Pxx : ndarray
        One-sided PSD [x-units^2 / Hz] (``scaling="density"``), so that
        ``np.trapezoid(Pxx, f) ~= var(x)`` (Parseval).

    Raises
    ------
    ValueError
        If ``fs <= 0``, ``x`` is not 1-D, too short, or non-finite.
    """
    x = _validate_signal(x)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs must be a positive finite sample rate in Hz, got {fs}")
    welch_kw.setdefault("nperseg", min(x.size, 1024))
    welch_kw.setdefault("window", "hann")
    welch_kw.setdefault("detrend", "constant")
    f, pxx = _sig.welch(x, fs=fs, scaling="density", **welch_kw)  # type: ignore[arg-type]
    return f, pxx


def band_rms(
    psd_result: tuple[np.ndarray, np.ndarray],
    bands: list[tuple[float, float]],
) -> np.ndarray:
    """Band-limited RMS from a one-sided PSD: sigma = sqrt(int_f1^f2 S df).

    Parameters
    ----------
    psd_result : (f, Pxx)
        Frequency bins [Hz] and one-sided PSD [u^2/Hz] as returned by
        :func:`psd`.
    bands : list of (f_lo, f_hi)
        Band edges [Hz], each with ``0 <= f_lo < f_hi``.

    Returns
    -------
    ndarray, shape (len(bands),)
        RMS value in each band [same units as the underlying signal].
        Integration uses the trapezoidal rule on the PSD grid restricted
        to the band (band edges are included by linear interpolation of
        the PSD), so accuracy is limited by the Welch frequency
        resolution ``fs/nperseg``.

    Raises
    ------
    ValueError
        On malformed bands or PSD arrays.
    """
    f, pxx = psd_result
    f = np.asarray(f, dtype=float)
    pxx = np.asarray(pxx, dtype=float)
    if f.shape != pxx.shape or f.ndim != 1:
        raise ValueError("psd_result must be (f, Pxx) 1-D arrays of equal length")
    if np.any(pxx < 0):
        raise ValueError("PSD values must be non-negative")
    out = np.empty(len(bands), dtype=float)
    for i, (lo, hi) in enumerate(bands):
        if not (0.0 <= lo < hi):
            raise ValueError(f"band {i}: need 0 <= f_lo < f_hi, got ({lo}, {hi})")
        lo_c = max(lo, float(f[0]))
        hi_c = min(hi, float(f[-1]))
        if hi_c <= lo_c:
            out[i] = 0.0
            continue
        mask = (f > lo_c) & (f < hi_c)
        fk = np.concatenate(([lo_c], f[mask], [hi_c]))
        pk = np.concatenate(
            ([np.interp(lo_c, f, pxx)], pxx[mask], [np.interp(hi_c, f, pxx)])
        )
        out[i] = float(np.sqrt(np.trapezoid(pk, fk)))
    return out


def cumulative_rms(
    psd_result: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative RMS curve sigma_c(f) = sqrt(int_0^f S(nu) dnu).

    Standard jitter-budget visualization: the curve's plateaus identify
    which frequency bands contribute most of the total RMS (e.g.
    reaction-wheel tones appear as steps).

    Parameters
    ----------
    psd_result : (f, Pxx)
        As returned by :func:`psd`.

    Returns
    -------
    f : ndarray
        Frequency bins [Hz] (same grid as input).
    sigma_c : ndarray
        Cumulative RMS up to each frequency [signal units]; the last
        element approximates the total RMS (Parseval).
    """
    f, pxx = psd_result
    f = np.asarray(f, dtype=float)
    pxx = np.asarray(pxx, dtype=float)
    if f.shape != pxx.shape or f.ndim != 1:
        raise ValueError("psd_result must be (f, Pxx) 1-D arrays of equal length")
    var_c = np.concatenate(([0.0], np.cumsum(0.5 * (pxx[1:] + pxx[:-1]) * np.diff(f))))
    return f, np.sqrt(var_c)
