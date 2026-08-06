"""Vectorised Monte Carlo BER engine.

Simulation models match the analytic module exactly (same SNR convention:
gamma = Eb/N0 electrical, noise sigma_n = 1):

* BPSK:  y = I * sqrt(2 gamma) * (2b - 1) + n,        decide sign(y).
* OOK:   y = I * 2 sqrt(gamma) * b + n,               decide y > tau;
         tau = I * sqrt(gamma) (adaptive/optimal, CSI) or tau = t * 2
         sqrt(gamma) (fixed fraction t of the mean-irradiance ON amplitude).
* M-PPM: M matched-filter branches y_j = n_j + a * delta(j, sent) with
         a = I * sqrt(2 k gamma), k = log2 M (orthogonal signalling,
         Proakis & Salehi 2008 Sec. 4.4); decide argmax. Bit errors counted
         as the Hamming distance between sent and decided symbol labels.

Fading: i.i.d. per-symbol mean-normalised lognormal irradiance (ergodic
average; see channels.sample_lognormal_irradiance).

Statistics
----------
The BER estimate is k_err / n_bits with a WILSON SCORE confidence interval
(Wilson 1927), chosen over the normal (Wald) approximation because BER
estimates are proportions near zero where Wald intervals undercover.
Sample-size rule of thumb: the relative CI half-width is ~ z/sqrt(k_err),
so ~100 observed errors give ~±20% relative accuracy at 95% confidence;
size n_bits >= ~100 / BER_expected (helper: ``n_bits_for_target``).
For M-PPM, bit errors within one symbol are correlated, so the interval is
built on the SYMBOL error count (independent trials) and scaled to BER by
the observed wrong-bits-per-symbol-error ratio. The remaining narrowing from
neglecting the ratio's own variance is small (~8% of the half-width for
M=16; empirically verified coverage ~93-95%, see VALIDATION.md).

Runtime: batched generation (default 2^21 values per batch) keeps memory
< ~150 MB; an optional ``max_seconds`` budget stops the run early and sets
``budget_exhausted`` in the result rather than overrunning.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from ._math import wilson_interval
from .analytic import _validate_common, _validate_m, _validate_threshold
from .channels import sample_lognormal_irradiance, validate_channel
from .results import MCResult

__all__ = ["mc_ber", "n_bits_for_target"]

_BATCH_VALUES = 2**21  # random values per batch (memory / speed tradeoff)


def n_bits_for_target(ber_expected: float, min_errors: int = 100) -> int:
    """Bits needed so the expected error count reaches ``min_errors``.

    n = ceil(min_errors / ber_expected). With ~100 errors the 95% Wilson CI
    has ~±20% relative half-width (half-width ~ 1.96/sqrt(k_err)).
    """
    if not 0.0 < ber_expected <= 1.0:
        raise ValueError(f"ber_expected must be in (0, 1], got {ber_expected}")
    if min_errors < 1:
        raise ValueError(f"min_errors must be >= 1, got {min_errors}")
    return int(math.ceil(min_errors / ber_expected))


def _sim_bpsk(
    rng: np.random.Generator, nb: int, gamma: float, sigma_i2: float | None
) -> tuple[int, int]:
    bits = rng.integers(0, 2, size=nb, dtype=np.int8)
    noise = rng.standard_normal(nb)
    amp = math.sqrt(2.0 * gamma)
    if sigma_i2 is not None:
        amp = amp * sample_lognormal_irradiance(rng, nb, sigma_i2)
    y = amp * (2.0 * bits - 1.0) + noise
    errors = int(np.count_nonzero((y > 0.0) != (bits == 1)))
    return errors, nb


def _sim_ook(
    rng: np.random.Generator,
    nb: int,
    gamma: float,
    sigma_i2: float | None,
    threshold: Any,
) -> tuple[int, int]:
    bits = rng.integers(0, 2, size=nb, dtype=np.int8)
    noise = rng.standard_normal(nb)
    a_on = 2.0 * math.sqrt(gamma)  # ON amplitude for sigma_n = 1, Eb = A^2/2
    if sigma_i2 is not None:
        irr = sample_lognormal_irradiance(rng, nb, sigma_i2)
        signal = a_on * irr * bits
        tau = 0.5 * a_on * irr if threshold == "optimal" else threshold * a_on
    else:
        signal = a_on * bits.astype(float)
        tau = 0.5 * a_on if threshold == "optimal" else threshold * a_on
    y = signal + noise
    errors = int(np.count_nonzero((y > tau) != (bits == 1)))
    return errors, nb


def _sim_ppm(
    rng: np.random.Generator,
    n_symbols: int,
    gamma: float,
    m: int,
    sigma_i2: float | None,
    popcount: np.ndarray,
) -> tuple[int, int, int]:
    k = int(math.log2(m))
    a = math.sqrt(2.0 * k * gamma)  # correct-branch mean, sigma_n = 1
    sent = rng.integers(0, m, size=n_symbols)
    y = rng.standard_normal((n_symbols, m))
    if sigma_i2 is not None:
        amp = a * sample_lognormal_irradiance(rng, n_symbols, sigma_i2)
    else:
        amp = np.full(n_symbols, a)
    y[np.arange(n_symbols), sent] += amp
    dec = np.argmax(y, axis=1)
    xor = np.bitwise_xor(sent, dec)
    bit_errors = int(popcount[xor].sum())
    sym_errors = int(np.count_nonzero(xor))
    return bit_errors, sym_errors, n_symbols * k


def mc_ber(
    mod: str,
    snr_db: float | np.ndarray,
    n: int = 1_000_000,
    seed: int = 0,
    *,
    channel: str = "awgn",
    sigma_i2: float | None = None,
    M: int = 4,
    threshold: str | float = "optimal",
    ci_level: float = 0.95,
    max_seconds: float | None = None,
) -> MCResult:
    """Monte Carlo BER estimate for a modulation / channel pair.

    Parameters
    ----------
    mod : {"ook", "bpsk", "ppm"}
        Modulation format (see module docstring for the exact signal models).
    snr_db : float or 1-D array
        Electrical SNR per bit Eb/N0 in dB. Negative dB allowed; NaN/inf
        raise ValueError.
    n : int
        Requested number of bits per SNR point (>= 1). For PPM this is
        rounded up to a whole number of symbols. Rule of thumb: choose
        n >= ~100 / BER_expected so >= ~100 errors are observed
        (see ``n_bits_for_target``).
    seed : int
        Seed for numpy's default_rng (PCG64); identical inputs => identical
        results.
    channel, sigma_i2, M, threshold :
        As in :func:`berbench.analytic.analytic_ber`.
    ci_level : float
        Two-sided Wilson confidence level in (0, 1), default 0.95.
    max_seconds : float, optional
        Wall-clock budget for the whole sweep. When exceeded the run stops
        early (result flags ``budget_exhausted=True`` and ``n_bits`` records
        what was actually simulated). Use to enforce hard runtime budgets.

    Returns
    -------
    MCResult
        Arrays per SNR point: ber, n_bits, n_errors, ci_low, ci_high.
    """
    snr = _validate_common(mod, snr_db)
    validate_channel(channel, sigma_i2)
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError(f"n must be a positive integer number of bits, got {n!r}")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError(f"max_seconds must be > 0, got {max_seconds}")

    params: dict[str, Any] = {}
    popcount = None
    if mod == "ppm":
        m = _validate_m(M)
        params["M"] = m
        popcount = np.array([bin(i).count("1") for i in range(m)], dtype=np.int64)
    if mod == "ook":
        threshold = _validate_threshold(threshold)
        params["threshold"] = threshold
    s2 = float(sigma_i2) if channel == "lognormal" else None
    if s2 is not None:
        params["sigma_i2"] = s2

    rng = np.random.default_rng(int(seed))
    gammas = np.power(10.0, snr / 10.0)

    n_pts = snr.size
    bits_done = np.zeros(n_pts, dtype=np.int64)
    errs_done = np.zeros(n_pts, dtype=np.int64)
    sym_errs_done = np.zeros(n_pts, dtype=np.int64)  # PPM only
    syms_done = np.zeros(n_pts, dtype=np.int64)  # PPM only
    t0 = time.perf_counter()
    budget_hit = False

    for i, gamma in enumerate(gammas):
        while bits_done[i] < n:
            if max_seconds is not None and time.perf_counter() - t0 > max_seconds:
                budget_hit = True
                break
            remaining = int(n - bits_done[i])
            if mod == "ppm":
                k = int(math.log2(params["M"]))
                max_sym = max(1, _BATCH_VALUES // params["M"])
                n_sym = min(max_sym, (remaining + k - 1) // k)
                e, se, b = _sim_ppm(rng, n_sym, float(gamma), params["M"], s2, popcount)
                sym_errs_done[i] += se
                syms_done[i] += n_sym
            elif mod == "bpsk":
                nb = min(_BATCH_VALUES, remaining)
                e, b = _sim_bpsk(rng, nb, float(gamma), s2)
            else:
                nb = min(_BATCH_VALUES, remaining)
                e, b = _sim_ook(rng, nb, float(gamma), s2, threshold)
            errs_done[i] += e
            bits_done[i] += b
        if budget_hit:
            break

    elapsed = time.perf_counter() - t0
    ber = np.where(bits_done > 0, errs_done / np.maximum(bits_done, 1), np.nan)
    ci_lo = np.empty(n_pts)
    ci_hi = np.empty(n_pts)
    for i in range(n_pts):
        if bits_done[i] <= 0:  # point never reached before budget ran out
            ci_lo[i], ci_hi[i] = 0.0, 1.0
        elif mod == "ppm":
            # Bits within one symbol are NOT independent: build the interval
            # from the symbol-error count (independent trials), then scale by
            # the observed wrong-bits-per-symbol-error ratio. See module
            # docstring for the residual (small) narrowing this leaves.
            k = int(math.log2(params["M"]))
            lo_s, hi_s = wilson_interval(
                int(sym_errs_done[i]), int(syms_done[i]), ci_level
            )
            if sym_errs_done[i] > 0:
                ratio = errs_done[i] / (k * sym_errs_done[i])  # in (0, 1]
            else:  # no errors observed: use the theoretical E[wrong bits]/k
                ratio = params["M"] / (2.0 * (params["M"] - 1.0))
            ci_lo[i], ci_hi[i] = ratio * lo_s, min(1.0, ratio * hi_s)
        else:
            ci_lo[i], ci_hi[i] = wilson_interval(int(errs_done[i]), int(bits_done[i]), ci_level)

    return MCResult(
        mod=mod, channel=channel, snr_db=snr, ber=ber,
        n_bits=bits_done, n_errors=errs_done, ci_low=ci_lo, ci_high=ci_hi,
        ci_level=ci_level, ci_method="wilson", seed=int(seed),
        elapsed_s=elapsed, budget_exhausted=budget_hit, params=params,
    )
