"""Moving-block bootstrap, and the headline understatement factor built on it.

Why not an i.i.d. bootstrap. Consecutive deadlines on a shared multicore SoC do
not have independent outcomes: they share the contention state that caused the
miss. Resampling single releases with replacement destroys that dependence and
produces an interval whose width corresponds to a sample larger than the one
actually collected. The moving-block bootstrap of Kuensch 1989, "The Jackknife
and the Bootstrap for General Stationary Observations", Annals of Statistics
17(3):1217-1241, resamples overlapping blocks of consecutive observations and so
preserves dependence up to the block length. Lahiri 2003, "Resampling Methods
for Dependent Data", Springer, chapter 2, is the standard treatment.

Block length is not a free parameter here: it is selected from the **measured**
autocorrelation of the per-release miss indicator by the rule in
:data:`BLOCK_RULE` and the selection is reported with the interval.

Units: every rate and every factor in this module is dimensionless.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .accounting import account, autocorrelation, miss_indicator
from .baselines import mean_plus_margin
from .trace import JobTrace

#: Confidence level of every interval this module reports.
CONFIDENCE = 0.95

#: The declared block-length rule, fixed in code so a report prints the rule.
BLOCK_RULE = (
    "Let rho(k) be the sample autocorrelation of the per-release miss indicator and "
    "let the white-noise band be 2/sqrt(n). Let L be the smallest lag >= 1 with "
    "rho(L) below that band, or the largest lag examined if there is none. The "
    "integrated autocorrelation time is tau = 1 + 2*sum(rho(1..L-1)), and the block "
    "length is ceil(tau), clipped below at 1 and above at n//20 so that at least "
    "twenty blocks span the series. Lags are examined up to min(n-1, 500)."
)


@dataclass(frozen=True)
class BlockChoice:
    """The block length, and the measured autocorrelation it was chosen from.

    Attributes
    ----------
    n
        Series length, releases.
    band
        ``2/sqrt(n)``, the approximate white-noise band the cut-off lag uses.
    lag_cut
        ``L``: the smallest lag at which the autocorrelation falls below
        ``band``.
    tau_integrated
        ``1 + 2*sum(rho(1..L-1))``, dimensionless.
    block_length
        The block length actually used, releases.
    rho_head
        The first few autocorrelation values, for the report.
    capped
        True when the ``n//20`` cap, rather than ``tau``, set the length.
    """

    n: int
    band: float
    lag_cut: int
    tau_integrated: float
    block_length: int
    rho_head: tuple[float, ...]
    capped: bool
    rule: str = BLOCK_RULE


def block_length_from_autocorrelation(indicator: np.ndarray) -> BlockChoice:
    """Apply :data:`BLOCK_RULE` to a 0/1 miss indicator.

    Raises
    ------
    ValueError
        On a series shorter than 2 entries.
    """
    series = np.asarray(indicator, dtype=np.float64)
    if series.ndim != 1:
        raise ValueError(f"indicator must be one-dimensional, got shape {series.shape}")
    n = series.size
    if n < 2:
        raise ValueError(f"indicator must have at least 2 entries, got {n}")
    max_lag = int(min(n - 1, 500))
    rho = autocorrelation(series, max_lag)
    band = 2.0 / np.sqrt(n)
    below = np.nonzero(rho[1:] < band)[0]
    lag_cut = int(below[0]) + 1 if below.size else max_lag
    tau = 1.0 + 2.0 * float(rho[1:lag_cut].sum()) if lag_cut > 1 else 1.0
    cap = max(1, n // 20)
    raw = max(1, int(np.ceil(tau)))
    block = min(raw, cap)
    return BlockChoice(
        n=int(n),
        band=float(band),
        lag_cut=lag_cut,
        tau_integrated=float(tau),
        block_length=int(block),
        rho_head=tuple(float(value) for value in rho[: min(6, rho.size)]),
        capped=bool(raw > cap),
    )


def moving_block_index(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """One moving-block resample of the index set ``0..n-1``.

    ``ceil(n/block_length)`` blocks are drawn uniformly from the ``n -
    block_length + 1`` possible overlapping start positions, concatenated, and
    truncated to length ``n``.

    Raises
    ------
    ValueError
        On ``n < 1`` or ``block_length`` outside ``1..n``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 1 <= block_length <= n:
        raise ValueError(f"block_length must be in 1..{n}, got {block_length}")
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n - block_length + 1, size=n_blocks)
    offsets = np.arange(block_length)
    return (starts[:, None] + offsets[None, :]).ravel()[:n]


def moving_block_bootstrap(
    statistic: Callable[[np.ndarray], float],
    n: int,
    block_length: int,
    resamples: int,
    seed: int,
) -> np.ndarray:
    """Moving-block bootstrap replicates of ``statistic`` over index resamples.

    ``statistic`` receives an index array of length ``n`` and returns a float.

    Raises
    ------
    ValueError
        On ``resamples < 1``.
    """
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    rng = np.random.default_rng(seed)
    out = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        out[draw] = statistic(moving_block_index(n, block_length, rng))
    return out


def percentile_interval(
    replicates: np.ndarray, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """Percentile interval of bootstrap ``replicates`` at ``confidence``.

    Uses ``numpy.percentile`` with its default linear interpolation. Non-finite
    replicates are excluded and the caller is expected to report how many there
    were; an interval over an empty finite set is ``(nan, nan)``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    finite = np.asarray(replicates, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    alpha = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(finite, [alpha, 100.0 - alpha])
    return (float(low), float(high))


@dataclass(frozen=True)
class RateInterval:
    """A counted miss rate with its moving-block bootstrap interval."""

    rate: float
    low: float
    high: float
    confidence: float
    block: BlockChoice
    resamples: int
    seed: int


def counted_rate_interval(
    jobs: JobTrace, resamples: int = 2000, seed: int = 920092
) -> RateInterval:
    """Counted miss rate over all releases, with a moving-block interval.

    The resampled series is the per-release miss indicator from
    :func:`deadlinerisk.accounting.miss_indicator`, so dropped releases are
    resampled as the misses they are.
    """
    indicator = miss_indicator(jobs).astype(np.float64)
    block = block_length_from_autocorrelation(indicator)
    replicates = moving_block_bootstrap(
        lambda index: float(indicator[index].mean()),
        indicator.size,
        block.block_length,
        resamples,
        seed,
    )
    low, high = percentile_interval(replicates)
    return RateInterval(
        rate=float(indicator.mean()),
        low=low,
        high=high,
        confidence=CONFIDENCE,
        block=block,
        resamples=resamples,
        seed=seed,
    )


@dataclass(frozen=True)
class UnderstatementFactor:
    """The product's headline: how far the mean-plus-margin rule is wrong.

    Attributes
    ----------
    counted_rate
        Counted miss rate over all releases, dimensionless.
    margin_probability
        The mean-plus-margin rule's ``P(resp > D)``, dimensionless.
    factor
        ``counted_rate / margin_probability``. Greater than 1 means the rule
        UNDERSTATES the measured miss rate by that factor; less than 1 means it
        overstates it; an interval containing 1 means the rule is
        indistinguishable from adequate on this data.
    low, high
        Percentile interval of the factor from the moving-block bootstrap.
    block_rows
        Block length in units of trace ROWS. The declared block rule is stated
        over releases; this resample has to move whole executed rows, because a
        resample must carry the response times the rule is computed from as well
        as the miss and drop counts. The row block length is the release block
        length scaled by ``executed/releases`` and rounded, never below 1.
    n_degenerate
        Replicates in which the rule's probability underflowed to zero, so the
        factor is not finite. These are excluded from the interval and reported.
    """

    counted_rate: float
    margin_probability: float
    factor: float
    low: float
    high: float
    confidence: float
    block: BlockChoice
    block_rows: int
    resamples: int
    seed: int
    n_degenerate: int
    interval_contains_one: bool


def understatement_factor(
    jobs: JobTrace, resamples: int = 2000, seed: int = 920092
) -> UnderstatementFactor:
    """The factor by which mean-plus-margin understates the counted miss rate.

    Both quantities are recomputed inside every resample, so the interval
    carries the sampling variability of the rule's own mean and standard
    deviation as well as that of the counted rate.

    Raises
    ------
    ValueError
        If the trace has fewer than two executed rows, or if the rule's
        probability on the full sample is zero, in which case the factor is not
        a finite number and the caller must report the two rates instead.
    """
    totals = account(jobs)
    indicator = miss_indicator(jobs).astype(np.float64)
    block = block_length_from_autocorrelation(indicator)
    rows = len(jobs)
    if rows < 2:
        raise ValueError(f"run {jobs.run!r} has {rows} executed row(s), need at least 2")
    block_rows = max(1, min(rows, int(round(block.block_length * rows / totals.releases))))

    response = jobs.response_s
    missed = jobs.missed.astype(np.int64)
    dropped = jobs.dropped_before.astype(np.int64)
    deadline = jobs.deadline_s

    point = mean_plus_margin(response, deadline)
    if point.probability <= 0.0:
        raise ValueError(
            f"run {jobs.run!r}: the mean-plus-margin probability underflowed to 0.0 at "
            f"k = {point.k_to_deadline:.6f}, so the understatement FACTOR is not a finite "
            "number. Report the counted rate and the rule's probability separately."
        )

    def statistic(index: np.ndarray) -> float:
        releases = int(rows + dropped[index].sum())
        misses = int(missed[index].sum() + dropped[index].sum())
        counted = misses / releases
        sample = response[index]
        sd = float(sample.std(ddof=1))
        if sd <= 0.0:
            return float("nan")
        k = (deadline - float(sample.mean())) / sd
        probability = float(stats.norm.sf(k))
        if probability <= 0.0:
            return float("inf")
        return counted / probability

    replicates = moving_block_bootstrap(statistic, rows, block_rows, resamples, seed)
    n_degenerate = int(np.count_nonzero(~np.isfinite(replicates)))
    low, high = percentile_interval(replicates)
    factor = totals.miss_rate / point.probability
    return UnderstatementFactor(
        counted_rate=totals.miss_rate,
        margin_probability=point.probability,
        factor=float(factor),
        low=low,
        high=high,
        confidence=CONFIDENCE,
        block=block,
        block_rows=block_rows,
        resamples=resamples,
        seed=seed,
        n_degenerate=n_degenerate,
        interval_contains_one=bool(low <= 1.0 <= high),
    )
