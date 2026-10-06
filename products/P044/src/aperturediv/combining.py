"""Diversity combining over a fading optical channel.

Convention, stated once and used everywhere
-------------------------------------------
Each aperture ``k`` sees a normalised irradiance ``I_k`` with ``E[I_k] = 1``
(:mod:`aperturediv.channel`). The instantaneous electrical SNR of branch
``k`` is taken to be **linear in irradiance**,

``gamma_k = gamma_bar * I_k``

so that ``E[gamma_k] = gamma_bar``. Equivalently the amplitude gain is
``h_k = sqrt(I_k)`` and ``gamma_k = gamma_bar h_k^2``. This is the standard
flat-fading convention with a lognormal (or gamma-gamma) *power* gain, and
it is the convention used by the BPSK BER of :mod:`aperturediv.ber` and by
the X2 cross-check script. A thermal-noise-limited intensity-modulated
receiver instead has ``gamma proportional to I^2``; that case is **not**
used here, and the difference is a factor of two in every dB figure, so it
is the first thing to check if a cross-comparison disagrees.

Combining schemes, with weights on the amplitude
------------------------------------------------
For unit-norm weights ``w`` applied to the branch amplitudes ``h``, the
post-combining SNR is ``gamma_out = gamma_bar (w . h)^2`` (:func:`weighted_snr`).

Maximal ratio (MRC)
    ``w ∝ h``, giving ``gamma_out = gamma_bar sum_k I_k``.
    By Cauchy-Schwarz, ``(w . h)^2 <= ||h||^2`` for every unit-norm ``w``,
    with equality only for ``w ∝ h``. **MRC with the true channel state is
    therefore optimal by construction and no other combiner, learned or
    otherwise, can exceed it.** This is Brennan's result (D. G. Brennan,
    "Linear Diversity Combining Techniques", *Proceedings of the IRE* 47(6),
    1959), and in this package it is additionally enforced as a
    property-based test rather than assumed.

Equal gain (EGC)
    ``w = 1/sqrt(L)``, giving ``gamma_out = gamma_bar (sum_k h_k)^2 / L``.
    Needs no channel-state estimate at all, so it is immune to
    channel-estimation error. That immunity is the whole point of
    :mod:`aperturediv.learned`.

Selection (SC)
    ``gamma_out = gamma_bar max_k I_k``. One branch, no co-phasing.

Ordering, stated correctly
    ``gamma_MRC >= gamma_EGC`` and ``gamma_MRC >= gamma_SC`` hold pointwise,
    for every realisation, by Cauchy-Schwarz. ``gamma_EGC >= gamma_SC`` is
    **false** pointwise: with ``L = 2`` and ``I = (1, 0)``, EGC gives 0.5 and
    SC gives 1. EGC beats SC on average but loses on the realisations where
    one branch is in a deep fade, which is exactly the regime diversity is
    bought for. The fraction of realisations on which SC beats EGC is
    measured in ``validation/validate_combining.py`` instead of being
    asserted away, and only the two true inequalities are tested.

Power normalisation across ``L``
--------------------------------
Comparing ``L = 1`` with ``L = 4`` requires a stated convention for how much
mean power each branch gets. :func:`branch_mean_snr` offers both:

``"fixed_total"``   ``gamma_bar_branch = gamma_bar_total / L``. The total
                    collecting area is held fixed, so this is the honest
                    "one big aperture against several small ones"
                    comparison and it contains no array gain.
``"fixed_branch"``  ``gamma_bar_branch = gamma_bar_total``. Each aperture is
                    full size, so the array also collects ``L`` times the
                    power. Diversity order is unchanged; the curve shifts
                    left by ``10 log10 L`` dB.

Diversity order
---------------
:func:`diversity_order` measures the negative log-log slope of outage
probability against mean SNR over a stated window, by least squares. It is
a **finite-window measurement**, not an asymptotic limit, and the window is
reported with the number. With correlated branches the asymptotic slope and
the slope visible over a 20 dB design window are different things, and the
difference is reported rather than smoothed over.

Nothing here is flight-qualified or certified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "COMBINERS",
    "NORMALISATIONS",
    "DiversityOrderResult",
    "branch_mean_snr",
    "combined_gain",
    "diversity_order",
    "egc_gain",
    "mrc_gain",
    "outage_probability",
    "sc_gain",
    "weighted_snr",
]

COMBINERS: tuple[str, ...] = ("mrc", "egc", "sc")
NORMALISATIONS: tuple[str, ...] = ("fixed_total", "fixed_branch")
Combiner = Literal["mrc", "egc", "sc"]
Normalisation = Literal["fixed_total", "fixed_branch"]


def _as_irradiance(irradiance: np.ndarray) -> np.ndarray:
    arr = np.asarray(irradiance, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("irradiance must have shape (n_samples, n_apertures)")
    if arr.shape[1] < 1:
        raise ValueError("irradiance must have at least one aperture column")
    if np.any(~np.isfinite(arr)):
        raise ValueError("irradiance must be finite")
    if np.any(arr < 0.0):
        raise ValueError("irradiance must be >= 0")
    return arr


def mrc_gain(irradiance: np.ndarray) -> np.ndarray:
    """Maximal-ratio combining gain ``sum_k I_k`` (dimensionless).

    Multiply by the branch mean SNR to get the post-combining SNR.
    """
    return _as_irradiance(irradiance).sum(axis=1)


def egc_gain(irradiance: np.ndarray) -> np.ndarray:
    """Equal-gain combining gain ``(sum_k sqrt(I_k))^2 / L`` (dimensionless)."""
    arr = _as_irradiance(irradiance)
    return np.sqrt(arr).sum(axis=1) ** 2 / arr.shape[1]


def sc_gain(irradiance: np.ndarray) -> np.ndarray:
    """Selection combining gain ``max_k I_k`` (dimensionless)."""
    return _as_irradiance(irradiance).max(axis=1)


def combined_gain(irradiance: np.ndarray, scheme: Combiner) -> np.ndarray:
    """Dimensionless combining gain for one of ``COMBINERS``."""
    if scheme == "mrc":
        return mrc_gain(irradiance)
    if scheme == "egc":
        return egc_gain(irradiance)
    if scheme == "sc":
        return sc_gain(irradiance)
    raise ValueError(f"scheme must be one of {COMBINERS}, got {scheme!r}")


def weighted_snr(
    weights: np.ndarray, amplitudes: np.ndarray, mean_snr_linear: float
) -> np.ndarray:
    """Post-combining SNR for arbitrary linear weights.

    ``gamma_out = mean_snr * (w . h)^2 / ||w||^2``, so the result depends on
    the *direction* of ``w`` only. ``h`` is the branch amplitude gain
    ``sqrt(I)``.

    Parameters
    ----------
    weights
        Shape ``(..., L)``. Non-zero norm required.
    amplitudes
        Shape ``(..., L)``, broadcastable against ``weights``.
    mean_snr_linear
        Branch mean SNR ``gamma_bar`` (linear, dimensionless).

    Returns
    -------
    numpy.ndarray
        Shape ``(...,)`` post-combining SNR, linear.
    """
    w = np.asarray(weights, dtype=float)
    h = np.asarray(amplitudes, dtype=float)
    if w.shape[-1] != h.shape[-1]:
        raise ValueError("weights and amplitudes must agree on the last axis")
    g = float(mean_snr_linear)
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError(f"mean_snr_linear must be finite and > 0, got {g!r}")
    norm2 = np.sum(w * w, axis=-1)
    if np.any(norm2 <= 0.0):
        raise ValueError("weights must have non-zero norm on the last axis")
    inner = np.sum(w * h, axis=-1)
    return g * inner * inner / norm2


def branch_mean_snr(
    total_mean_snr_linear: float,
    n_apertures: int,
    normalisation: Normalisation = "fixed_total",
) -> float:
    """Per-branch mean SNR under the stated power normalisation."""
    g = float(total_mean_snr_linear)
    n = int(n_apertures)
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError(f"total_mean_snr_linear must be finite and > 0, got {g!r}")
    if n < 1:
        raise ValueError(f"n_apertures must be >= 1, got {n!r}")
    if normalisation == "fixed_total":
        return g / n
    if normalisation == "fixed_branch":
        return g
    raise ValueError(f"normalisation must be one of {NORMALISATIONS}, got {normalisation!r}")


def outage_probability(
    gain_samples: np.ndarray,
    mean_snr_db: np.ndarray | float,
    threshold_snr_db: float,
) -> np.ndarray:
    """Outage probability from a set of combining-gain samples.

    Outage is ``gamma_out < gamma_th``, i.e. ``gain < gamma_th / gamma_bar``.
    Because the combining gain does not depend on ``gamma_bar``, one sample
    set of gains evaluates every mean SNR consistently — the same sample
    path for every point on the curve, which is what makes the log-log slope
    in :func:`diversity_order` meaningful.

    Parameters
    ----------
    gain_samples
        Shape ``(n,)`` dimensionless combining gains.
    mean_snr_db
        Branch mean SNR ``gamma_bar`` in dB, scalar or array.
    threshold_snr_db
        Outage threshold ``gamma_th`` in dB.

    Returns
    -------
    numpy.ndarray
        Empirical outage probability, same shape as ``mean_snr_db``. The
        resolution floor is ``1/n``; a returned 0 means "below ``1/n``", not
        "zero", and :func:`diversity_order` excludes such points.
    """
    g = np.asarray(gain_samples, dtype=float)
    if g.ndim != 1 or g.size == 0:
        raise ValueError("gain_samples must be a non-empty 1-D array")
    snr_db = np.asarray(mean_snr_db, dtype=float)
    thr_lin = 10.0 ** (float(threshold_snr_db) / 10.0)
    snr_lin = 10.0 ** (snr_db / 10.0)
    needed = thr_lin / snr_lin
    gs = np.sort(g)
    counts = np.searchsorted(gs, np.atleast_1d(needed), side="left")
    out = counts / g.size
    return out.reshape(snr_db.shape) if snr_db.shape else out[0]


@dataclass(frozen=True)
class DiversityOrderResult:
    """Outcome of a finite-window diversity-order fit.

    Attributes
    ----------
    order
        Negative least-squares slope of ``log10 P_out`` against
        ``log10 gamma_bar``, dimensionless.
    n_points
        Number of ``(gamma_bar, P_out)`` points inside the window.
    window
        ``(p_low, p_high)`` outage bounds that defined the window.
    snr_db_range
        ``(min, max)`` branch mean SNR in dB of the points used.
    residual_rms
        RMS residual of the fit in decades of ``P_out``; a large value
        means the curve is not a straight line over the window and the
        single number is not a good summary.
    """

    order: float
    n_points: int
    window: tuple[float, float]
    snr_db_range: tuple[float, float]
    residual_rms: float


def diversity_order(
    mean_snr_db: np.ndarray,
    outage: np.ndarray,
    *,
    window: tuple[float, float] = (1e-4, 1e-2),
    min_points: int = 4,
) -> DiversityOrderResult:
    """Measure diversity order as a log-log slope over a stated window.

    ``d = -d log10 P_out / d log10 gamma_bar``, fitted by least squares over
    the points whose outage lies inside ``window``. Points with
    ``P_out == 0`` (below the Monte Carlo resolution) are dropped.

    Raises
    ------
    ValueError
        If fewer than ``min_points`` usable points fall in the window. That
        is a real failure — it means the sample size or the SNR grid was too
        small to measure the slope — and it is reported rather than filled
        in with a wider window.
    """
    snr_db = np.asarray(mean_snr_db, dtype=float).ravel()
    p = np.asarray(outage, dtype=float).ravel()
    if snr_db.shape != p.shape:
        raise ValueError("mean_snr_db and outage must have the same shape")
    lo, hi = float(window[0]), float(window[1])
    if not 0.0 < lo < hi <= 1.0:
        raise ValueError("window must satisfy 0 < low < high <= 1")
    keep = (p > 0.0) & (p >= lo) & (p <= hi)
    if int(keep.sum()) < int(min_points):
        raise ValueError(
            f"only {int(keep.sum())} usable points in outage window {window}; "
            f"need >= {min_points}. Widen the SNR grid or raise the sample count."
        )
    x = snr_db[keep] / 10.0  # log10 of linear mean SNR
    y = np.log10(p[keep])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    return DiversityOrderResult(
        order=float(-slope),
        n_points=int(keep.sum()),
        window=(lo, hi),
        snr_db_range=(float(snr_db[keep].min()), float(snr_db[keep].max())),
        residual_rms=float(np.sqrt(np.mean(resid**2))),
    )
