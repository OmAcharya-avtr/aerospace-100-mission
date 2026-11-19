"""Rank-correlation statistics for how far two rankings of the same tests disagree.

The headline output of this product is not a ranking. It is **how far the
cost ranking and the flake-rate ranking disagree**, with an interval on that
disagreement and the identity of the tests that move most. A dashboard that
ranks by flake rate and a cost model that ranks by expected wasted time agree
on which tests matter exactly when this statistic is 1; everything below 1 is
the part of the problem the dashboard cannot see.

KENDALL TAU-B
-------------

For ``n`` items with two score vectors ``x`` and ``y``, consider all
``n0 = n(n-1)/2`` unordered pairs. A pair is *concordant* when the two vectors
order it the same way, *discordant* when they order it oppositely, and *tied*
when either vector gives the pair equal scores. With ``C`` concordant and ``D``
discordant pairs,

    tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))                   (1)

where ``n1 = sum_g t_g (t_g - 1) / 2`` over the tie groups ``g`` of ``x`` and
``n2`` is the same over the tie groups of ``y``.

*Sources.* The statistic without the tie correction is Kendall, M. G. (1938),
"A new measure of rank correlation", *Biometrika* 30(1/2), 81-93. The tie
correction -- the ``n1`` and ``n2`` terms that make the denominator the
geometric mean of the two untied pair counts -- is Kendall, M. G. (1945), "The
treatment of ties in ranking problems", *Biometrika* 33(3), 239-251. The
textbook treatment of both is Kendall, M. G. & Gibbons, J. D. (1990), *Rank
Correlation Methods*, 5th edition, Edward Arnold.

**Tie handling, stated explicitly.** A pair tied in ``x``, in ``y`` or in both
counts as neither concordant nor discordant, so it enters neither ``C`` nor
``D``. It is removed from the *denominator* too, separately for each vector,
which is what distinguishes tau-b from tau-a (which divides by ``n0``
regardless) and from tau-c (which uses a rectangular-table correction). tau-b
attains 1 only when the two orderings agree on every pair that *either* vector
breaks; so a vector with many ties can reach 1 against a vector with none. This
matters here: the flake-rate vector is heavily tied, because every test with no
observed failure has rate exactly 0. The product reports ``n1``, ``n2`` and the
tie groups alongside the statistic for that reason, and reports Spearman rho as
well, which handles ties differently.

Why ``tau_b`` and not ``tau_a``: with a tied vector, tau-a cannot reach 1 and
its ceiling depends on how many ties there are, so a tau-a of 0.6 cannot be
compared across two histories with different numbers of never-failing tests.

SPEARMAN RHO
------------

Spearman, C. (1904), "The proof and measurement of association between two
things", *American Journal of Psychology* 15(1), 72-101. Computed here as the
Pearson product-moment correlation of the **midranks** of ``x`` and ``y``: tied
values receive the arithmetic mean of the ranks they span. With no ties this
reduces to the familiar ``1 - 6 * sum(d**2) / (n**3 - n)``; with ties that
shortcut is wrong and the midrank-Pearson form is the standard correction
(Kendall & Gibbons 1990). The product computes the midrank form always, never
the shortcut.

AN INTERVAL ON THE DISAGREEMENT
-------------------------------

A rank correlation over 72 tests is a statistic with sampling error, and a
product whose headline is a disagreement measure has to say how well that
measure is pinned down. The interval here is a percentile bootstrap over
**tests**: tests are resampled with replacement and the statistic recomputed.
The unit is declared on the returned :class:`~flakerank.resample.BootstrapInterval`
and asserted before it is returned.

Resampling tests, and not sessions, is the right unit for *this* statistic
because the statistic is a property of the set of tests being ranked: it asks
how much the two rankings disagree over the tests in this suite. The
*ground-truth cost* of each test has its own interval from resampling
**sessions**, which is a different question about a different unit; both are
reported, both declare their unit, and the two must not be confused.

The bootstrap of a rank correlation is known to be awkward: a resample that
draws the same test twice creates an artificial tie, which inflates ``n1`` and
``n2`` and biases tau-b upward. The product does not correct for that and says
so here and in ``README.md`` under Limitations. The interval is therefore
better read as a width than as a calibrated coverage statement.

An independent implementation check: ``scipy.stats.kendalltau(variant="b")``
and ``scipy.stats.spearmanr`` are a separately written implementation of both
statistics, and ``tests/test_disagreement.py`` and
``validation/validate_disagreement.py`` compare against them. This is a
cross-check against an independent implementation and is *not* the
independently-derived-bound check, which lives in :mod:`flakerank.bounds`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .resample import BootstrapInterval, bootstrap_statistic_interval

__all__ = [
    "RankDisagreement",
    "TieStructure",
    "kendall_tau_b",
    "midranks",
    "rank_disagreement",
    "spearman_rho",
]


def midranks(values: np.ndarray) -> np.ndarray:
    """Midranks of ``values``: tied values get the mean of the ranks they span.

    Args:
        values: shape ``(n,)``, any real dtype.

    Returns:
        Shape ``(n,)`` float array of midranks, smallest value ranked 1.

    Raises:
        ValueError: for an empty input or one containing NaN. NaN has no rank,
            and silently sorting it to one end would make the statistic depend
            on an implementation detail of the sort.

    Examples:
        >>> midranks(np.asarray([1.0, 1.0, 2.0, 3.0]))
        array([1.5, 1.5, 3. , 4. ])
    """
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {sample.shape}")
    if sample.size == 0:
        raise ValueError("values must be non-empty")
    if np.isnan(sample).any():
        raise ValueError(
            "values must not contain NaN: NaN has no rank, and sorting it to one end "
            "would make the statistic depend on an implementation detail of the sort"
        )
    order = np.argsort(sample, kind="stable")
    sorted_values = sample[order]
    ranks = np.empty(sample.size, dtype=np.float64)
    position = 0
    while position < sample.size:
        end = position + 1
        while end < sample.size and sorted_values[end] == sorted_values[position]:
            end += 1
        # Ranks are 1-based, so the group spans positions position+1 .. end.
        ranks[order[position:end]] = 0.5 * (position + 1 + end)
        position = end
    return ranks


@dataclass(frozen=True)
class TieStructure:
    """Pair counts that make tau-b's denominator, for the reader's inspection.

    Attributes:
        n_items: number of items ranked.
        n_pairs: ``n(n-1)/2``, the total unordered pairs (``n0``).
        tied_x_pairs: ``n1``, pairs tied in the first vector.
        tied_y_pairs: ``n2``, pairs tied in the second vector.
        tied_both_pairs: pairs tied in both, counted in ``n1`` and ``n2`` alike.
        concordant: pairs ordered the same way by both vectors.
        discordant: pairs ordered oppositely.
        tie_groups_x: sizes of the tie groups of the first vector, above 1.
        tie_groups_y: sizes of the tie groups of the second vector, above 1.
    """

    n_items: int
    n_pairs: int
    tied_x_pairs: int
    tied_y_pairs: int
    tied_both_pairs: int
    concordant: int
    discordant: int
    tie_groups_x: tuple[int, ...] = field(default=())
    tie_groups_y: tuple[int, ...] = field(default=())


def _tie_group_sizes(values: np.ndarray) -> tuple[int, ...]:
    """Sizes above 1 of the tie groups of ``values``, descending."""
    _unique, counts = np.unique(np.asarray(values, dtype=np.float64), return_counts=True)
    return tuple(sorted((int(count) for count in counts if count > 1), reverse=True))


def _pair_counts(x: np.ndarray, y: np.ndarray) -> TieStructure:
    """Concordant, discordant and tied pair counts over all unordered pairs.

    Counted directly over the ``n(n-1)/2`` pairs with a vectorised sign
    comparison. ``O(n**2)`` in time and memory, which for the few hundred tests
    a CI suite has is a few megabytes and a few milliseconds, and which is
    chosen over the ``O(n log n)`` merge-sort algorithm because the definition
    is then visible in the code: a reader can check the implementation against
    equation (1) without reconstructing an inversion count.
    """
    count = x.size
    sign_x = np.sign(x[:, None] - x[None, :])
    sign_y = np.sign(y[:, None] - y[None, :])
    upper = np.triu(np.ones((count, count), dtype=bool), k=1)
    product = (sign_x * sign_y)[upper]
    tied_x = (sign_x == 0)[upper]
    tied_y = (sign_y == 0)[upper]
    return TieStructure(
        n_items=count,
        n_pairs=count * (count - 1) // 2,
        tied_x_pairs=int(np.count_nonzero(tied_x)),
        tied_y_pairs=int(np.count_nonzero(tied_y)),
        tied_both_pairs=int(np.count_nonzero(tied_x & tied_y)),
        concordant=int(np.count_nonzero(product > 0)),
        discordant=int(np.count_nonzero(product < 0)),
        tie_groups_x=_tie_group_sizes(x),
        tie_groups_y=_tie_group_sizes(y),
    )


def _validate_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce and validate two score vectors of equal length."""
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError(
            f"both score vectors must be one-dimensional, got shapes "
            f"{first.shape} and {second.shape}"
        )
    if first.size != second.size:
        raise ValueError(
            f"score vectors must have equal length: got {first.size} and {second.size}. "
            "A rank correlation compares two orderings of the SAME items, so a length "
            "mismatch means the two rankings were built over different test sets."
        )
    if first.size < 2:
        raise ValueError(
            f"a rank correlation needs at least 2 items, got {first.size}: with one item "
            "there are no pairs to be concordant or discordant"
        )
    if np.isnan(first).any() or np.isnan(second).any():
        raise ValueError("score vectors must not contain NaN; see midranks for why")
    return first, second


def kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b, equation (1), with the tie correction of Kendall (1945).

    Args:
        x: first score vector, shape ``(n,)``. Higher is ranked first; the
            statistic is invariant to that choice because it reverses sign
            consistently.
        y: second score vector, shape ``(n,)``.

    Returns:
        tau-b in ``[-1, 1]``, dimensionless. Returns ``0.0`` when either vector
        is entirely tied, where the denominator of (1) is zero: with no pair
        broken by one of the vectors there is no information about agreement,
        and 0 is the value that says so. That convention matches
        ``scipy.stats.kendalltau``, which returns NaN; the difference is stated
        here and asserted in ``tests/test_disagreement.py``, which is the only
        place the two implementations are allowed to differ.

    Raises:
        ValueError: for unequal lengths, fewer than two items, or NaN.

    Examples:
        >>> round(kendall_tau_b(np.asarray([1.0, 2, 3, 4]), np.asarray([1.0, 2, 4, 3])), 10)
        0.6666666667
        >>> kendall_tau_b(np.asarray([1.0, 1, 2, 3]), np.asarray([1.0, 2, 2, 3]))
        0.8
    """
    first, second = _validate_pair(x, y)
    counts = _pair_counts(first, second)
    untied_x = counts.n_pairs - counts.tied_x_pairs
    untied_y = counts.n_pairs - counts.tied_y_pairs
    if untied_x == 0 or untied_y == 0:
        return 0.0
    return (counts.concordant - counts.discordant) / float(
        np.sqrt(float(untied_x) * float(untied_y))
    )


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho as the Pearson correlation of midranks.

    Args:
        x: first score vector, shape ``(n,)``.
        y: second score vector, shape ``(n,)``.

    Returns:
        rho in ``[-1, 1]``, dimensionless. Returns ``0.0`` when either midrank
        vector has zero variance, where the Pearson correlation is undefined.

    Raises:
        ValueError: for unequal lengths, fewer than two items, or NaN.

    Examples:
        >>> round(spearman_rho(np.asarray([1.0, 1, 2, 3]), np.asarray([1.0, 2, 2, 3])), 10)
        0.8333333333
    """
    first, second = _validate_pair(x, y)
    rank_x = midranks(first)
    rank_y = midranks(second)
    dx = rank_x - rank_x.mean()
    dy = rank_y - rank_y.mean()
    denominator = float(np.sqrt(float(dx @ dx) * float(dy @ dy)))
    if denominator == 0.0:
        return 0.0
    return float(dx @ dy) / denominator


@dataclass(frozen=True)
class Mover:
    """A test whose rank position differs most between the two rankings.

    Attributes:
        name: the test's identifier.
        rank_x: 1-based position in the first ranking (1 is worst/most costly).
        rank_y: 1-based position in the second ranking.
        displacement: ``rank_x - rank_y``. Negative means the test is ranked
            HIGHER (nearer position 1) by the first ranking than by the second.
        score_x: the first ranking's score for the test.
        score_y: the second ranking's score.
    """

    name: str
    rank_x: int
    rank_y: int
    displacement: int
    score_x: float
    score_y: float


@dataclass(frozen=True)
class RankDisagreement:
    """How far two rankings of the same tests disagree, with intervals.

    Attributes:
        label_x: name of the first ranking.
        label_y: name of the second ranking.
        n_items: number of tests ranked.
        tau_b: Kendall tau-b, equation (1).
        tau_b_interval: bootstrap interval on ``tau_b``, resampling tests.
        rho: Spearman rho from midranks.
        rho_interval: bootstrap interval on ``rho``, resampling tests.
        ties: the pair counts behind tau-b's denominator.
        movers: tests with the largest absolute rank displacement, largest
            first, then by name for a deterministic order among equal
            displacements.
        max_displacement: the largest absolute displacement observed.
        mean_absolute_displacement: mean over tests of ``abs(rank_x - rank_y)``.
        top_k: size of the prefix used for ``top_k_overlap``.
        top_k_overlap: number of tests present in the first ``top_k`` positions
            of BOTH rankings. The operational question "would a team working
            the top ten of the dashboard fix the right tests" is this number.
    """

    label_x: str
    label_y: str
    n_items: int
    tau_b: float
    tau_b_interval: BootstrapInterval
    rho: float
    rho_interval: BootstrapInterval
    ties: TieStructure
    movers: tuple[Mover, ...]
    max_displacement: int
    mean_absolute_displacement: float
    top_k: int
    top_k_overlap: int


def _descending_positions(scores: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """1-based positions under a descending sort, ties broken by name.

    Ties are broken by name rather than left to the sort so that the reported
    positions are a deterministic function of the inputs. A tie broken by index
    would make the mover list depend on the order tests happen to appear in the
    history.
    """
    order = sorted(range(scores.size), key=lambda index: (-scores[index], names[index]))
    positions = np.empty(scores.size, dtype=np.int64)
    for position, index in enumerate(order, start=1):
        positions[index] = position
    return positions


def rank_disagreement(
    names: tuple[str, ...],
    scores_x: np.ndarray,
    scores_y: np.ndarray,
    *,
    label_x: str,
    label_y: str,
    top_movers: int = 10,
    top_k: int = 10,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 97044,
) -> RankDisagreement:
    """Measure the disagreement between two score vectors over the same tests.

    Args:
        names: test identifiers, shape ``(n,)``.
        scores_x: first ranking's scores; higher is ranked nearer position 1.
        scores_y: second ranking's scores.
        label_x: human-readable name of the first ranking.
        label_y: human-readable name of the second ranking.
        top_movers: how many movers to report.
        top_k: prefix length for the top-k overlap.
        resamples: bootstrap resamples for both intervals.
        confidence: nominal coverage of both intervals.
        seed: seed for ``numpy.random.default_rng``. The two intervals use
            ``seed`` and ``seed + 1`` so that tau-b's and rho's resamples are
            not the same draws, which would make their widths correlated for a
            reason unrelated to the data.

    Returns:
        A :class:`RankDisagreement`. Both intervals declare
        ``resampling_unit == "test"`` and the declaration is asserted before
        the result is returned.

    Raises:
        ValueError: for a names/scores length mismatch, fewer than two tests,
            NaN scores, or a non-positive ``top_k``.
    """
    first, second = _validate_pair(scores_x, scores_y)
    if len(names) != first.size:
        raise ValueError(
            f"names has {len(names)} entries but the score vectors have {first.size}"
        )
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_movers < 0:
        raise ValueError(f"top_movers must be >= 0, got {top_movers}")

    stacked = np.column_stack([first, second])

    def tau_statistic(resampled: np.ndarray) -> float:
        return kendall_tau_b(resampled[:, 0], resampled[:, 1])

    def rho_statistic(resampled: np.ndarray) -> float:
        return spearman_rho(resampled[:, 0], resampled[:, 1])

    tau_interval = bootstrap_statistic_interval(
        stacked,
        tau_statistic,
        resampling_unit="test",
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        value_units="dimensionless",
    ).require_resampling_unit("test")
    rho_interval = bootstrap_statistic_interval(
        stacked,
        rho_statistic,
        resampling_unit="test",
        resamples=resamples,
        confidence=confidence,
        seed=seed + 1,
        value_units="dimensionless",
    ).require_resampling_unit("test")

    positions_x = _descending_positions(first, names)
    positions_y = _descending_positions(second, names)
    displacement = positions_x - positions_y
    order = sorted(
        range(first.size), key=lambda index: (-abs(int(displacement[index])), names[index])
    )
    movers = tuple(
        Mover(
            name=names[index],
            rank_x=int(positions_x[index]),
            rank_y=int(positions_y[index]),
            displacement=int(displacement[index]),
            score_x=float(first[index]),
            score_y=float(second[index]),
        )
        for index in order[:top_movers]
    )
    effective_k = min(top_k, first.size)
    set_x = {names[index] for index in range(first.size) if positions_x[index] <= effective_k}
    set_y = {names[index] for index in range(first.size) if positions_y[index] <= effective_k}

    return RankDisagreement(
        label_x=label_x,
        label_y=label_y,
        n_items=int(first.size),
        tau_b=kendall_tau_b(first, second),
        tau_b_interval=tau_interval,
        rho=spearman_rho(first, second),
        rho_interval=rho_interval,
        ties=_pair_counts(first, second),
        movers=movers,
        max_displacement=int(np.abs(displacement).max()),
        mean_absolute_displacement=float(np.abs(displacement).mean()),
        top_k=effective_k,
        top_k_overlap=len(set_x & set_y),
    )
