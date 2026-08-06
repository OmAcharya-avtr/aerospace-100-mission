"""Result dataclasses returned by the analytic and Monte Carlo engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["AnalyticResult", "MCResult"]


@dataclass(frozen=True)
class AnalyticResult:
    """Analytic BER over an SNR sweep.

    Attributes
    ----------
    mod : str
        Modulation: "ook", "bpsk" or "ppm".
    channel : str
        "awgn" or "lognormal".
    snr_db : ndarray
        Electrical SNR per bit, Eb/N0 in dB.
    ber : ndarray
        Bit error ratio (dimensionless, in [0, 1]), same shape as snr_db.
    method : str
        Which expression produced `ber` (e.g. "exact" vs "union-bound" for PPM).
    reference : str
        Literature source of the expression.
    params : dict
        Extra parameters (M, sigma_i2, threshold, ...).
    """

    mod: str
    channel: str
    snr_db: np.ndarray
    ber: np.ndarray
    method: str
    reference: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo BER estimate over an SNR sweep.

    Attributes
    ----------
    mod, channel, snr_db : as in AnalyticResult.
    ber : ndarray
        Estimated BER = n_errors / n_bits per SNR point.
    n_bits : ndarray of int
        Bits actually simulated per point (may be < requested if the
        max_seconds budget was hit; check `budget_exhausted`).
    n_errors : ndarray of int
        Bit errors counted per point.
    ci_low, ci_high : ndarray
        Wilson score confidence interval on the true BER at `ci_level`.
    ci_level : float
        Two-sided confidence level (default 0.95).
    ci_method : str
        Interval type, always "wilson" in v0.1.0.
    seed : int
        Seed of the numpy default_rng PCG64 generator (reproducible).
    elapsed_s : float
        Wall-clock seconds spent simulating (all points).
    budget_exhausted : bool
        True if a max_seconds budget cut the run short.
    params : dict
        Extra parameters (M, sigma_i2, threshold, ...).
    """

    mod: str
    channel: str
    snr_db: np.ndarray
    ber: np.ndarray
    n_bits: np.ndarray
    n_errors: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    ci_level: float
    ci_method: str
    seed: int
    elapsed_s: float
    budget_exhausted: bool = False
    params: dict[str, Any] = field(default_factory=dict)
