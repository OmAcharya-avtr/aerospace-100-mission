"""Poisson counting statistics for a photon-counting receiver.

A photon-counting receiver does not measure a voltage; it counts avalanche
events in a time slot. The channel is therefore not additive Gaussian noise
but a Poisson point process whose rate is the sum of three contributions, all
expressed in **detected counts per slot**:

===========================  =====================================
``n_s``                      signal counts per signalled slot
``n_b``                      background (sky, stray light) counts
``n_d``                      detector dark counts per slot
===========================  =====================================

For an optical field of ``P`` watts at wavelength ``lambda`` over a slot of
``T_s`` seconds reaching a detector of quantum efficiency ``eta``, the mean
detected count is

    n = eta * P * T_s * lambda / (h * c)                                  (1)

which is the standard semiclassical photon-counting result (Gagliardi & Karp,
*Optical Communications*, 2nd ed., Wiley 1995, chapter 3: the photodetection
counting process is Poisson with rate proportional to the incident optical
power). Equation (1) assumes a constant rate over the slot, no gain
fluctuation, and no detector saturation. Gain fluctuation is handled in
:mod:`photoncount.webb`; saturation and dead time in
:mod:`photoncount.deadtime`.

All means in this module are dimensionless counts per slot. ``None`` of these
functions models dead time: they are the ideal-counter limit, and the ideal
counter is exactly what a real detector is not. Use them as the reference
against which :mod:`photoncount.deadtime` measures loss.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "PLANCK_H",
    "SPEED_OF_LIGHT",
    "counting_snr",
    "exact_count_interval",
    "false_alarm_probability",
    "fano_factor",
    "mean_counts_from_power",
    "missed_detection_probability",
    "photons_per_second",
    "slot_mean",
    "threshold_detection_probability",
]

#: Planck constant, J s (SI defining constant, exact since the 2019 SI revision).
PLANCK_H = 6.626_070_15e-34

#: Speed of light in vacuum, m/s (SI defining constant, exact).
SPEED_OF_LIGHT = 299_792_458.0


def _check_nonneg(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0, got {v!r}")
    return v


def _check_positive(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


def photons_per_second(optical_power_w: float, wavelength_m: float) -> float:
    """Photon arrival rate for an optical power, photons/s.

    ``rate = P * lambda / (h * c)``. Units: W, m -> 1/s. Assumes a
    monochromatic field; for a broadband source this is the rate at the centre
    wavelength and is accurate to the fractional bandwidth.
    """
    p = _check_nonneg("optical_power_w", optical_power_w)
    lam = _check_positive("wavelength_m", wavelength_m)
    return p * lam / (PLANCK_H * SPEED_OF_LIGHT)


def mean_counts_from_power(
    optical_power_w: float,
    wavelength_m: float,
    slot_seconds: float,
    quantum_efficiency: float = 1.0,
) -> float:
    """Mean detected counts in one slot, equation (1). Dimensionless.

    ``quantum_efficiency`` is the end-to-end detection efficiency (coupling
    times quantum efficiency times any blocking filter transmission), in
    ``(0, 1]``.
    """
    eta = _check_positive("quantum_efficiency", quantum_efficiency)
    if eta > 1.0:
        raise ValueError(f"quantum_efficiency must be <= 1, got {eta!r}")
    t_s = _check_positive("slot_seconds", slot_seconds)
    return eta * photons_per_second(optical_power_w, wavelength_m) * t_s


def slot_mean(
    signal_counts: float = 0.0,
    background_counts: float = 0.0,
    dark_counts: float = 0.0,
) -> float:
    """Total mean counts in a slot: ``n_s + n_b + n_d``. Dimensionless.

    The three contributions are independent Poisson processes, so their sum is
    Poisson with the summed rate. That additivity is the only reason a single
    number suffices here.
    """
    return (
        _check_nonneg("signal_counts", signal_counts)
        + _check_nonneg("background_counts", background_counts)
        + _check_nonneg("dark_counts", dark_counts)
    )


def fano_factor(mean: float, variance: float) -> float:
    """``variance / mean``. Exactly 1 for an ideal Poisson counter.

    Departures are diagnostic: dead time pushes it below 1 (counts are
    anti-bunched by the detector's own refractory period), afterpulsing and
    gain fluctuation push it above 1.
    """
    m = _check_positive("mean", mean)
    return _check_nonneg("variance", variance) / m


def threshold_detection_probability(mean: float, threshold: int) -> float:
    """``P(K >= threshold)`` for ``K ~ Poisson(mean)``. Dimensionless.

    ``threshold`` is the smallest count declared a detection. ``threshold=1``
    gives the familiar ``1 - exp(-mean)``.
    """
    m = _check_nonneg("mean", mean)
    t = int(threshold)
    if t < 0:
        raise ValueError(f"threshold must be >= 0, got {t!r}")
    if t == 0:
        return 1.0
    return float(stats.poisson.sf(t - 1, m))


def false_alarm_probability(background_mean: float, threshold: int) -> float:
    """``P(K >= threshold)`` with only background present. Dimensionless."""
    return threshold_detection_probability(background_mean, threshold)


def missed_detection_probability(mean: float, threshold: int) -> float:
    """``P(K < threshold)`` for ``K ~ Poisson(mean)``. Dimensionless."""
    return 1.0 - threshold_detection_probability(mean, threshold)


def counting_snr(signal_counts: float, background_counts: float = 0.0) -> float:
    """Counting SNR ``n_s / sqrt(n_s + n_b)``, dimensionless (amplitude ratio).

    This is the shot-noise-limited SNR of a slot measurement: the mean excess
    over background divided by the standard deviation of the total count. It is
    reported because link budgets are written in these terms, not because a
    photon-counting receiver should be designed by SNR --- at ``n_s`` of order
    one the Gaussian intuition behind an SNR is wrong, and the symbol-error
    expressions in :mod:`photoncount.ppm` are what matter.
    """
    n_s = _check_positive("signal_counts", signal_counts)
    n_b = _check_nonneg("background_counts", background_counts)
    return n_s / np.sqrt(n_s + n_b)


def exact_count_interval(count: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Garwood) two-sided interval for a Poisson mean from one count.

    Returns ``(lower, upper)`` in counts. Uses the chi-square relation

        lower = chi2.ppf(alpha/2, 2k) / 2,  upper = chi2.ppf(1-alpha/2, 2k+2) / 2

    with ``alpha = 1 - confidence`` and ``lower = 0`` when ``k = 0``. The
    interval is conservative: its actual coverage is at least ``confidence``
    and is not equal to it, because the count is discrete.
    """
    k = int(count)
    if k < 0:
        raise ValueError(f"count must be >= 0, got {k!r}")
    c = float(confidence)
    if not 0.0 < c < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {c!r}")
    alpha = 1.0 - c
    lower = 0.0 if k == 0 else float(stats.chi2.ppf(alpha / 2.0, 2 * k)) / 2.0
    upper = float(stats.chi2.ppf(1.0 - alpha / 2.0, 2 * k + 2)) / 2.0
    return lower, upper
