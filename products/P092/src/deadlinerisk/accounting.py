"""Deadline accounting: reconstructing releases, misses and the miss indicator.

The arithmetic is the measurement model in :mod:`deadlinerisk.trace`, applied:

``releases = sum(1 + dropped_before)`` and
``misses = sum(missed + dropped_before)``.

The per-release **miss indicator** is the sequence those two totals come from,
and the order within it matters for everything that measures serial dependence.
``dropped_before[i]`` counts the releases between executed job ``i`` and the
next executed job, so a row expands to its own miss flag followed by
``dropped_before[i]`` ones:

``[missed[i]] + [1] * dropped_before[i]``

The indicator therefore has length ``releases`` and sums to ``misses`` by
construction, which :func:`miss_indicator` asserts rather than assumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trace import JobTrace


@dataclass(frozen=True)
class DeadlineAccount:
    """Reconstructed totals for one run.

    Attributes
    ----------
    executed
        Rows in the trace: releases that actually ran.
    releases
        ``sum(1 + dropped_before)``.
    dropped
        ``sum(dropped_before)``: releases that never ran.
    misses
        ``sum(missed + dropped_before)``.
    miss_rate
        ``misses / releases``, dimensionless. The product's ground truth.
    executed_miss_rate
        ``sum(missed) / executed``. Reported beside the above because it is
        what a tool that never counted the dropped releases would report, and
        the difference between them is not cosmetic.
    """

    run: str
    executed: int
    releases: int
    dropped: int
    misses: int
    miss_rate: float
    executed_miss_rate: float


def account(jobs: JobTrace) -> DeadlineAccount:
    """Reconstruct the release and miss totals for one run.

    Raises
    ------
    ValueError
        If the trace has no rows, since a miss rate over zero releases is not a
        number and silently returning one would be worse than refusing.
    """
    if len(jobs) == 0:
        raise ValueError(
            f"run {jobs.run!r} has no executed job rows; a miss rate over zero "
            "releases is undefined"
        )
    executed = len(jobs)
    dropped = int(jobs.dropped_before.sum())
    releases = executed + dropped
    misses = int(jobs.missed.sum()) + dropped
    return DeadlineAccount(
        run=jobs.run,
        executed=executed,
        releases=releases,
        dropped=dropped,
        misses=misses,
        miss_rate=misses / releases,
        executed_miss_rate=int(jobs.missed.sum()) / executed,
    )


def miss_indicator(jobs: JobTrace) -> np.ndarray:
    """Per-release 0/1 miss indicator, in release order.

    Returns
    -------
    numpy.ndarray
        ``int8`` array of length ``releases``. Guaranteed, in exact arithmetic
        and here also in integer arithmetic, to sum to ``misses``.

    Raises
    ------
    ValueError
        If the expansion does not reproduce the totals from :func:`account`.
        That cannot happen for a well-formed trace; the check exists because the
        two quantities are computed two different ways and a disagreement would
        mean the trace is not the shape this module documents.
    """
    totals = account(jobs)
    repeats = 1 + jobs.dropped_before
    indicator = np.empty(int(repeats.sum()), dtype=np.int8)
    position = 0
    for flag, drop in zip(jobs.missed.tolist(), jobs.dropped_before.tolist(), strict=True):
        indicator[position] = flag
        position += 1
        if drop:
            indicator[position : position + drop] = 1
            position += drop
    if indicator.size != totals.releases or int(indicator.sum()) != totals.misses:
        raise ValueError(
            f"run {jobs.run!r}: expanded indicator has {indicator.size} entries summing "
            f"to {int(indicator.sum())}, but the totals are {totals.releases} releases "
            f"and {totals.misses} misses"
        )
    return indicator


def autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of ``series`` at lags ``0..max_lag`` inclusive.

    Uses the biased (divide-by-``n``) estimator, which is the one the block
    bootstrap literature's block-length rules are stated against.

    Returns
    -------
    numpy.ndarray
        Length ``max_lag + 1``, with ``rho[0] == 1`` when the series has
        non-zero variance. A constant series has no autocorrelation defined; a
        zero vector is returned with ``rho[0] = 0`` so that a caller's
        block-length rule degenerates to a block length of one rather than to a
        division by zero.

    Raises
    ------
    ValueError
        On ``max_lag < 0`` or ``max_lag >= len(series)``.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"series must be one-dimensional, got shape {values.shape}")
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")
    if max_lag >= values.size:
        raise ValueError(
            f"max_lag ({max_lag}) must be < len(series) ({values.size})"
        )
    centred = values - values.mean()
    denominator = float(centred @ centred)
    if denominator <= 0.0:
        return np.zeros(max_lag + 1, dtype=np.float64)
    out = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        if lag == 0:
            out[lag] = 1.0
        else:
            out[lag] = float(centred[lag:] @ centred[:-lag]) / denominator
    return out
