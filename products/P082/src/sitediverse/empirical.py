"""Baseline 3: the direct empirical joint availability, with bootstrap intervals.

This is the reference every model in this package is scored against, and it is
also the thing whose uncertainty decides whether any of those scores differ. It
makes no assumption at all beyond stationarity of the climatology over the
record: it counts epochs.

**(20) The empirical joint unavailability.** For a station set ``A``,

    P_hat = (1/T) sum_t prod_{i in A} c_it

the fraction of epochs at which every station is cloudy. Unbiased for the
epoch-averaged joint probability, and that is all it is: it says nothing about
any epoch outside the record.

**(21) The moving-block bootstrap.** The epochs are serially dependent -- the
latent field of equation (5) has an integral time scale of
``(1 + phi) / (1 - phi)`` epochs -- so resampling epochs independently
understates the variance of equation (20) by about that factor. The moving-block
bootstrap of Kunsch 1989 (*Annals of Statistics* 17, 1217) resamples
overlapping blocks of ``L`` consecutive epochs, which preserves dependence out
to lag ``L``. The original stationary-block idea is Efron & Tibshirani 1993
chapter 8 for the iid case and Kunsch for the dependent case; the percentile
interval is Efron & Tibshirani 1993 chapter 13.

A block bootstrap is consistent as ``L -> inf`` and ``L / T -> 0``. The declared
block length here is ``BLOCK_EPOCHS = 64``, which is 8 days and 10.4 integral
time scales, with ``L / T = 0.0027`` on the reference record. The sensitivity of
the interval width to ``L`` is MEASURED across ``L`` in
``validation/validate_empirical.py`` and reported, because a block bootstrap
whose answer depends on its block length is reporting the block length.

The iid bootstrap is also computed and reported BESIDE the block bootstrap, for
one reason: a reader who sees only the block interval cannot tell how much of it
is the dependence. The two differ by a factor the validation output states, and
the iid interval is the one a careless analysis would quote.

**(22) The interval on a difference.** The quantity that decides this product's
headline comparison is not an availability but a DIFFERENCE of two models'
errors against equation (20). Resampling the record changes equation (20) and
therefore changes both errors together, so the interval on the difference must
be computed from the paired bootstrap replicates and not from the two models'
intervals separately -- the errors are strongly positively correlated and
treating them as independent would widen the interval by up to sqrt(2) and make
a real difference look insignificant. ``bootstrap_difference`` below takes the
two replicate series and reports the percentile interval of their difference.

What is resampled, and what is not. The bootstrap here resamples THE RECORD and
holds the fitted models fixed. It therefore measures the uncertainty in a
comparison that comes from the reference being a finite record, which is what
"the bootstrap interval on the empirical reference" means and is the figure the
specification asks for. It does NOT measure the uncertainty in the models'
fitted parameters, which would require refitting inside every replicate. That
limitation is stated in README.md and in ``validation/VALIDATION.md`` rather
than left for a reader to discover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Declared moving-block length in epochs. 64 epochs is 8 days at the
#: three-hour cadence, or 10.4 integral time scales at the declared
#: persistence.
BLOCK_EPOCHS: int = 64

#: Declared number of bootstrap replicates. 400, not 1000: the whole validation
#: suite is executed four times over by the working-directory probe and the
#: percentile interval of a 95 per cent level is stable to about one per cent of
#: its width at 400 replicates, which is far inside the width itself.
REPLICATES: int = 400

#: Declared two-sided confidence level of every interval in this module.
LEVEL: float = 0.95


def joint_rate(cloud: np.ndarray, sites: tuple[int, ...] | list[int]) -> float:
    """Equation (20): the fraction of epochs at which every station is cloudy."""
    indicator = np.asarray(cloud)
    if indicator.ndim != 2:
        raise ValueError(f"cloud must be two-dimensional, got {indicator.ndim}D")
    index = [int(site) for site in sites]
    if len(index) < 1:
        raise ValueError("at least one station index is required")
    for site in index:
        if not 0 <= site < indicator.shape[1]:
            raise ValueError(
                f"station index {site} is outside [0, {indicator.shape[1]})"
            )
    return float(np.mean(np.all(indicator[:, index], axis=1)))


def pair_joint_matrix(cloud: np.ndarray) -> np.ndarray:
    """``(S, S)`` matrix of observed joint cloud rates. Diagonal is the marginal."""
    indicator = np.asarray(cloud, dtype=bool)
    if indicator.ndim != 2:
        raise ValueError(f"cloud must be two-dimensional, got {indicator.ndim}D")
    values = indicator.astype(np.float64)
    return (values.T @ values) / float(indicator.shape[0])


def pair_clear_matrix(cloud: np.ndarray) -> np.ndarray:
    """``(S, S)`` matrix of observed joint CLEAR rates, for equation (18)."""
    return pair_joint_matrix(~np.asarray(cloud, dtype=bool))


def block_indices(
    epochs: int, block: int, replicates: int, seed: int
) -> np.ndarray:
    """Moving-block resampling indices, shape ``(replicates, epochs)``.

    The last block is truncated so that every replicate has exactly ``epochs``
    entries, which keeps equation (20)'s denominator fixed across replicates.
    """
    if epochs < 2:
        raise ValueError(f"epochs must be at least 2, got {epochs!r}")
    if not 1 <= block <= epochs:
        raise ValueError(
            f"block must lie in [1, {epochs}] for this record, got {block!r}"
        )
    if replicates < 2:
        raise ValueError(f"replicates must be at least 2, got {replicates!r}")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(epochs / block))
    starts = rng.integers(0, epochs - block + 1, size=(replicates, blocks))
    offsets = np.arange(block)
    drawn = (starts[:, :, None] + offsets).reshape(replicates, blocks * block)
    return drawn[:, :epochs]


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile interval and the replicates behind it.

    Attributes
    ----------
    estimate:
        The statistic on the original record.
    lower, upper:
        Percentile interval at ``level``.
    replicate_values:
        The replicate statistics. Carried so that an interval on a DIFFERENCE
        can be formed from paired replicates, which equation (22) requires.
    block:
        Block length used; ``1`` means the iid bootstrap.
    level:
        Two-sided confidence level.
    """

    estimate: float
    lower: float
    upper: float
    replicate_values: np.ndarray
    block: int
    level: float

    @property
    def half_width(self) -> float:
        """``(upper - lower) / 2``. The figure a margin is compared against."""
        return 0.5 * (self.upper - self.lower)

    @property
    def standard_error(self) -> float:
        """Standard deviation of the replicates."""
        return float(np.std(self.replicate_values, ddof=1))

    @property
    def relative_half_width(self) -> float:
        """``half_width / estimate``, or ``inf`` at an estimate of zero."""
        return (
            self.half_width / self.estimate if self.estimate > 0.0 else float("inf")
        )

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the interval, inclusive."""
        return self.lower <= float(value) <= self.upper


def _percentile_interval(
    estimate: float, replicates: np.ndarray, block: int, level: float
) -> BootstrapInterval:
    tail = 0.5 * (1.0 - level)
    low, high = np.quantile(replicates, [tail, 1.0 - tail])
    return BootstrapInterval(
        estimate=float(estimate),
        lower=float(low),
        upper=float(high),
        replicate_values=np.asarray(replicates, dtype=float),
        block=int(block),
        level=float(level),
    )


def bootstrap_joint_rate(
    cloud: np.ndarray,
    sites: tuple[int, ...] | list[int],
    block: int = BLOCK_EPOCHS,
    replicates: int = REPLICATES,
    level: float = LEVEL,
    seed: int = 909001,
) -> BootstrapInterval:
    """Equations (20) and (21): the joint rate with a moving-block interval."""
    indicator = np.asarray(cloud, dtype=bool)
    index = [int(site) for site in sites]
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level!r}")
    all_cloudy = np.all(indicator[:, index], axis=1)
    draws = block_indices(indicator.shape[0], block, replicates, seed)
    values = all_cloudy[draws].mean(axis=1)
    return _percentile_interval(float(all_cloudy.mean()), values, block, level)


def bootstrap_pair_rates(
    cloud: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    block: int = BLOCK_EPOCHS,
    replicates: int = REPLICATES,
    level: float = LEVEL,
    seed: int = 909002,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap every pair's joint rate on ONE shared set of resampling draws.

    The draws are shared deliberately. Two pairs' joint rates on the same record
    are dependent, and equation (22)'s interval on a difference of errors
    aggregated over pairs is only correct if the replicates are paired across
    pairs as well as across models. Using an independent draw per pair would
    destroy that and understate the interval.

    Returns
    -------
    tuple
        ``(estimate, replicate_matrix, draws)`` with shapes ``(P,)``,
        ``(replicates, P)`` and ``(replicates, epochs)``.
    """
    indicator = np.asarray(cloud, dtype=bool)
    if not pairs:
        raise ValueError("at least one pair is required")
    both = np.column_stack(
        [indicator[:, i] & indicator[:, j] for i, j in pairs]
    )
    draws = block_indices(indicator.shape[0], block, replicates, seed)
    replicate_matrix = both[draws].mean(axis=1)
    return both.mean(axis=0), replicate_matrix, draws


def bootstrap_difference(
    left_replicates: np.ndarray,
    right_replicates: np.ndarray,
    estimate: float,
    level: float = LEVEL,
    block: int = BLOCK_EPOCHS,
) -> BootstrapInterval:
    """Equation (22): the paired percentile interval of a difference.

    Parameters
    ----------
    left_replicates, right_replicates:
        ``(replicates,)`` statistics from the SAME resampling draws.
    estimate:
        The difference on the original record.
    """
    left = np.asarray(left_replicates, dtype=float)
    right = np.asarray(right_replicates, dtype=float)
    if left.shape != right.shape:
        raise ValueError(
            "paired replicate series must have the same shape, got "
            f"{left.shape} and {right.shape}; an interval on a difference "
            "requires the same resampling draws for both"
        )
    return _percentile_interval(estimate, left - right, block, level)
