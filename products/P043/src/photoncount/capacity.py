"""Rates available on the PPM Poisson channel, for sanity-checking a link.

This module exists to answer one question: before anyone chooses a code, is the
rate being asked for even available at the photon flux in the link budget? It
ships **only bounds that are computed here and verified here**. Where a
published result could not be checked in this environment it is named as
context and no number is taken from it.

**What is shipped.**

1. *Background-free PPM erasure capacity, exact.* With ``n_0 = 0`` the channel
   is an ``M``-ary erasure channel with erasure probability ``exp(-n_s)``, so

       C_sym = (1 - exp(-n_s)) log2(M)   bits per symbol,                 (C1)
       C_slot = C_sym / M                bits per slot.

   (C1) is the capacity of the erasure channel and needs no optimisation: the
   uniform input achieves it.

2. *Hard-decision capacity of the symmetric M-ary channel, exact.* With
   background present, the hard decision ``argmax_i k_i`` induces an ``M``-ary
   channel that is symmetric by slot relabelling: every wrong symbol is equally
   likely. Its capacity is

       C_sym = log2(M) + (1 - P_e) log2(1 - P_e) + P_e log2(P_e / (M - 1))  (C2)

   with ``P_e`` from :func:`photoncount.ppm.symbol_error_probability`. (C2) is
   a **lower** bound on what the channel can carry, because hard-deciding
   discards the counts.

3. *Soft-decision achievable rate, by Monte Carlo.* Keeping the count vector,

       I(X; Y) = log2(M) + E[ log2 P(X | Y) ]                              (C3)

   with the exact posterior of :func:`photoncount.ppm.symbol_log_posterior`.
   (C3) is estimated by averaging over simulated symbols, so it carries a
   standard error, and it is an achievable rate **under PPM signalling** --- a
   lower bound on the capacity of the underlying Poisson channel, not that
   capacity. By the data-processing inequality (C3) must be at least (C2); the
   validation checks that it is, which is a real test of both.

4. *Photons per bit and its PPM limit.* Dividing the signal photons by the rate,

       photons/bit = n_s / C_sym.                                          (C4)

   For the background-free case, substituting (C1),

       photons/bit = n_s / ((1 - exp(-n_s)) log2 M)  ->  1 / log2(M)       (C5)

   as ``n_s -> 0``, so the best any ``M``-ary PPM scheme can do on a
   background-free Poisson channel is ``1/log2(M)`` photons per bit,
   equivalently ``log2(M)`` bits per photon. (C5) is derived here by taking the
   limit of (C1) and is checked numerically in the validation. Because
   ``log2(M)`` grows without bound in ``M``, bits per photon is **unbounded**
   within this family --- the familiar statement that the background-free
   Poisson channel has no finite capacity-per-photon limit, obtained here from
   expressions this repository can verify rather than quoted.

**What is not shipped.** No bound from Lapidoth--Moser or Wyner for the
peak-and-average-constrained Poisson channel, and no capacity-per-unit-cost
result, is implemented or cited with a number: those were not verified in this
environment, and the mission's rule is that an unverifiable citation is worse
than an omission. (C1)-(C5) are enough to tell a link designer whether a
requested rate is reachable, which is the stated purpose.

Units: ``C_sym`` bits/symbol, ``C_slot`` bits/slot, photons/bit dimensionless
(signal counts per bit, i.e. *detected* photons per bit --- divide by the
detection efficiency for photons at the aperture).
"""

from __future__ import annotations

import numpy as np

from .ppm import (
    PPMConfig,
    bits_per_symbol,
    sample_counts,
    symbol_error_probability,
    symbol_log_posterior,
)

__all__ = [
    "erasure_channel_capacity",
    "hard_decision_capacity",
    "minimum_photons_per_bit",
    "photons_per_bit",
    "soft_decision_achievable_rate",
]


def erasure_channel_capacity(config: PPMConfig) -> dict[str, float]:
    """(C1) exact erasure-channel capacity. Requires ``n_0 == 0``.

    Returns ``bits_per_symbol``, ``bits_per_slot``, ``erasure_probability``.
    """
    if not config.is_background_free:
        raise ValueError(
            "erasure_channel_capacity requires n_b + n_d == 0; with background "
            "present use hard_decision_capacity or soft_decision_achievable_rate"
        )
    p_er = float(np.exp(-config.signal_counts))
    c_sym = (1.0 - p_er) * bits_per_symbol(config.order)
    return {
        "bits_per_symbol": c_sym,
        "bits_per_slot": c_sym / config.order,
        "erasure_probability": p_er,
    }


def hard_decision_capacity(config: PPMConfig) -> dict[str, float]:
    """(C2) exact capacity of the hard-decision symmetric ``M``-ary channel.

    Returns ``bits_per_symbol``, ``bits_per_slot``, ``symbol_error_probability``.
    At ``P_e = 0`` the entropy terms are taken as 0 by continuity.
    """
    m = config.order
    p_e = symbol_error_probability(config)["error"]
    log2m = bits_per_symbol(m)
    term_correct = 0.0 if p_e >= 1.0 else (1.0 - p_e) * np.log2(1.0 - p_e)
    term_wrong = 0.0 if p_e <= 0.0 else p_e * np.log2(p_e / (m - 1))
    c_sym = float(log2m + term_correct + term_wrong)
    c_sym = max(c_sym, 0.0)
    return {
        "bits_per_symbol": c_sym,
        "bits_per_slot": c_sym / m,
        "symbol_error_probability": p_e,
    }


def soft_decision_achievable_rate(
    config: PPMConfig,
    n_symbols: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """(C3) Monte Carlo soft-decision achievable rate under PPM signalling.

    Returns ``bits_per_symbol``, ``standard_error``, ``bits_per_slot``,
    ``n_symbols``. Requires background: on a background-free channel the
    posterior is degenerate and :func:`erasure_channel_capacity` is exact, so
    this function refuses rather than returning an infinity-scaled metric.
    """
    if config.is_background_free:
        raise ValueError(
            "soft_decision_achievable_rate requires n_b + n_d > 0; the "
            "background-free channel is exactly an erasure channel, use "
            "erasure_channel_capacity"
        )
    n = int(n_symbols)
    if n < 2:
        raise ValueError(f"n_symbols must be >= 2, got {n!r}")
    sent = rng.integers(0, config.order, size=n)
    counts = sample_counts(config, sent, rng)
    log_post = symbol_log_posterior(counts, config)
    chosen = log_post[np.arange(n), sent] / np.log(2.0)
    log2m = bits_per_symbol(config.order)
    per_symbol = log2m + chosen
    mean = float(per_symbol.mean())
    se = float(per_symbol.std(ddof=1) / np.sqrt(n))
    return {
        "bits_per_symbol": mean,
        "standard_error": se,
        "bits_per_slot": mean / config.order,
        "n_symbols": float(n),
    }


def photons_per_bit(config: PPMConfig, bits_per_symbol_rate: float) -> float:
    """(C4) ``n_s / C_sym``, detected signal photons per bit (-).

    ``bits_per_symbol_rate`` must be ``> 0``; pass the output of (C1), (C2) or
    (C3) depending on which receiver is being costed.
    """
    r = float(bits_per_symbol_rate)
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError(f"bits_per_symbol_rate must be finite and > 0, got {r!r}")
    return config.signal_counts / r


def minimum_photons_per_bit(order: int) -> float:
    """(C5) ``1 / log2(M)``, the ``n_s -> 0`` limit for background-free PPM (-).

    Unreachable at any finite ``n_s``: (C4) with (C1) is strictly above it and
    approaches it only as the pulsed slot becomes almost always empty, which
    means almost every symbol is erased. The cost of approaching the limit is
    the erasure rate, and that is the trade a link designer is actually making.
    """
    return 1.0 / bits_per_symbol(order)
