"""Goodput, outage, and mis-selection split into its two distinct kinds.

Why one mis-selection number is not enough
------------------------------------------
A policy that picks the wrong MODCOD has made one of two opposite mistakes, and
they cost different things:

* **too aggressive** --- chose a mode the channel could not carry. The slot is
  lost entirely. Cost: the full ``eta[best]`` that a correct choice would have
  delivered, plus whatever the retransmission costs upstream.
* **too conservative** --- chose a mode below what the channel could carry. The
  slot succeeds. Cost: only the difference ``eta[best] - eta[chosen]``.

Reporting a single "selection error rate" averages a total loss against a
partial one and hides the only trade the designer controls. Every result in this
package reports the two separately, with their separate costs.

A third category is **unavoidable**: slots where the true SNR does not support
even the lowest MODCOD (``best == -1``). Those are outages no selection rule
could have prevented, and attributing them to the policy would flatter whichever
policy happened to be most conservative. They are counted and reported on their
own.

Definitions, all over the accounted slots
-----------------------------------------
With ``best[n]`` the highest MODCOD index the true SNR supports (``-1`` if
none), ``chosen[n]`` the policy's index, and ``eta`` the spectral efficiencies:

    success[n]          = chosen[n] <= best[n]
    goodput             = mean( eta[chosen] * success )            bit/symbol
    outage_fraction     = mean( not success )
    aggressive_fraction = mean( chosen > best )            == outage_fraction
    unavoidable_fraction= mean( best < 0 )
    avoidable_outage    = mean( (chosen > best) & (best >= 0) )
    conservative_fraction = mean( chosen < best )
    exact_fraction      = mean( chosen == best )
    wasted_bits         = mean( (eta[best] - eta[chosen]) * (chosen < best) )
    lost_bits           = mean( eta[best] * (chosen > best) * (best >= 0) )
    clairvoyant_goodput = mean( eta[best] * (best >= 0) )
    efficiency          = goodput / clairvoyant_goodput

``aggressive_fraction`` and ``outage_fraction`` are identical whenever the
threshold ladder is monotone, which is checked, and both are reported because
they answer different questions and a non-monotone ladder would separate them.
``switch_rate`` counts index changes per slot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .modcod import ModcodTable


@dataclass(frozen=True)
class Accounting:
    """Summary of one policy run. Rates are fractions of accounted slots.

    Attributes
    ----------
    n_slots
        Number of slots included in the accounting (after warm-up).
    goodput_bit_per_symbol
        Mean delivered spectral efficiency.
    clairvoyant_goodput_bit_per_symbol
        Mean deliverable spectral efficiency under perfect knowledge.
    efficiency
        ``goodput / clairvoyant_goodput``, dimensionless, in ``[0, 1]``.
    outage_fraction, aggressive_fraction, avoidable_outage_fraction,
    unavoidable_outage_fraction, conservative_fraction, exact_fraction
        Fractions of accounted slots, dimensionless.
    wasted_bit_per_symbol
        Mean spectral efficiency forgone by conservative choices.
    lost_bit_per_symbol
        Mean spectral efficiency lost to avoidable outage.
    switch_rate_per_slot
        MODCOD index changes per slot.
    mean_index, mean_best_index
        Mean chosen and mean best-available index.
    """

    n_slots: int
    goodput_bit_per_symbol: float
    clairvoyant_goodput_bit_per_symbol: float
    efficiency: float
    outage_fraction: float
    aggressive_fraction: float
    avoidable_outage_fraction: float
    unavoidable_outage_fraction: float
    conservative_fraction: float
    exact_fraction: float
    wasted_bit_per_symbol: float
    lost_bit_per_symbol: float
    switch_rate_per_slot: float
    mean_index: float
    mean_best_index: float

    def as_dict(self) -> dict[str, float]:
        """Flat dict of every field, for CSV or JSON output."""
        return asdict(self)


def account(
    table: ModcodTable,
    chosen: np.ndarray,
    true_snr_db: np.ndarray,
    *,
    warmup: int = 0,
) -> Accounting:
    """Score a selection sequence against the true channel.

    Parameters
    ----------
    table
        The MODCOD table whose thresholds define success.
    chosen
        Selected MODCOD index per slot, each in ``[0, n_modes-1]``.
    true_snr_db
        True SNR per symbol per slot, dB, same length as ``chosen``.
    warmup
        Number of leading slots excluded from the accounting, >= 0. The feedback
        delay makes the first ``d`` slots unrepresentative, so callers normally
        pass at least ``d``.

    Returns
    -------
    Accounting
    """
    k = np.asarray(chosen, dtype=int)
    snr = np.asarray(true_snr_db, dtype=float)
    if k.shape != snr.shape:
        raise ValueError(f"chosen shape {k.shape} != true_snr_db shape {snr.shape}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    if warmup >= k.size:
        raise ValueError(f"warmup ({warmup}) must be < n_slots ({k.size})")
    if k.size and (k.min() < 0 or k.max() >= table.n_modes):
        raise ValueError(
            f"chosen indices must lie in [0, {table.n_modes - 1}], "
            f"got [{k.min()}, {k.max()}]"
        )

    eta = table.spectral_efficiencies
    best_full = np.atleast_1d(table.best_supported(snr))
    switches = float(np.count_nonzero(np.diff(k[warmup:]))) / max(1, k.size - warmup - 1)

    sl = slice(warmup, None)
    kk, bb = k[sl], best_full[sl]
    n = int(kk.size)
    eta_k = eta[kk]
    eta_best = np.where(bb >= 0, eta[np.maximum(bb, 0)], 0.0)

    success = kk <= bb
    aggressive = kk > bb
    conservative = kk < bb
    unavoidable = bb < 0
    avoidable_outage = aggressive & ~unavoidable

    goodput = float(np.mean(eta_k * success))
    clair = float(np.mean(eta_best))
    return Accounting(
        n_slots=n,
        goodput_bit_per_symbol=goodput,
        clairvoyant_goodput_bit_per_symbol=clair,
        efficiency=float(goodput / clair) if clair > 0 else float("nan"),
        outage_fraction=float(np.mean(~success)),
        aggressive_fraction=float(np.mean(aggressive)),
        avoidable_outage_fraction=float(np.mean(avoidable_outage)),
        unavoidable_outage_fraction=float(np.mean(unavoidable)),
        conservative_fraction=float(np.mean(conservative)),
        exact_fraction=float(np.mean(kk == bb)),
        wasted_bit_per_symbol=float(np.mean((eta_best - eta_k) * conservative)),
        lost_bit_per_symbol=float(np.mean(eta_best * avoidable_outage)),
        switch_rate_per_slot=switches,
        mean_index=float(np.mean(kk)),
        mean_best_index=float(np.mean(bb)),
    )
