"""Test the exchangeability that the bootstrap and conformal bounds rest on.

Why this module exists
----------------------
:mod:`transfergap.resample` is tighter than :mod:`transfergap.bounds` and the
whole of that difference is paid for by one assumption. The specification this
package was built to requires the assumption to be STATED AND TESTED, not
assumed, so it is tested here against the alternative that actually occurs in
flight test: a campaign flown to a card, in order of increasing Mach number or
increasing dynamic pressure, so that the discrepancy drifts with the index.

What is and is not being tested
-------------------------------
Exchangeability is not testable against every alternative -- no finite sample
can rule out all departures from permutation invariance. What is testable is a
**named** alternative, and this module names two:

**(16) The trend statistic.** Spearman's rank correlation between the
observation index ``i`` and the discrepancy ``d_i``, computed as the Pearson
correlation of the ranks. Under exchangeability every ordering of the ranks is
equally likely, so the statistic has a known permutation law. Sensitive to a
monotone drift. Dimensionless, in ``[-1, 1]``.

**(17) The half-split statistic.** ``mean(second half) - mean(first half)``, in
the unit of the measurement. Sensitive to a step change -- a software load
mid-campaign, a different airframe -- which a monotone rank statistic can miss.

Both are referred to a **permutation null** rather than to an asymptotic one.
Under the exchangeability hypothesis the labels are exchangeable by
definition, so the randomisation distribution obtained by permuting them is
exact in finite samples; this is the classical randomisation argument of
Fisher, and the modern statement is Lehmann, E. L. and Romano, J. P. (2005),
*Testing Statistical Hypotheses*, 3rd edition, Springer, chapter 15.

**(18) The p-value.** With ``B`` random permutations and ``T`` the observed
statistic,

    phat = ( 1 + #{ b : T_b >= T } ) / ( 1 + B ).                         (18)

The ``+1`` in both places is not a smoothing fudge: it makes ``phat`` a valid
p-value, in the sense that ``P(phat <= alpha) <= alpha`` under the null, for
every ``B``. Phipson, B. and Smyth, G. K. (2010), "Permutation p-values should
never be zero: calculating exact p-values when permutations are randomly
drawn", *Statistical Applications in Genetics and Molecular Biology* 9(1),
article 39. The naive ratio without the ``+1`` can return zero, which claims
more than any finite number of permutations can support.

Validity range
--------------
The test has power against drift and against a step; it has essentially none
against a change in higher moments with the mean and the trend unchanged, and
it says so. A non-rejection is NOT evidence of exchangeability, and
:attr:`ExchangeabilityTest.verdict` says ``"not rejected"`` rather than
``"exchangeable"`` for that reason.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .discrepancy import validate_confidence

__all__ = [
    "ExchangeabilityTest",
    "STATISTICS",
    "half_split_statistic",
    "null_distribution",
    "permutation_test",
    "trend_statistic",
]


def trend_statistic(values: np.ndarray) -> float:
    """Spearman rank correlation between index and value, equation (16).

    Dimensionless, in ``[-1, 1]``. Returns ``0.0`` when every value is tied,
    where the rank vector is constant and the correlation is undefined; a
    constant sample carries no evidence of drift, so zero is the honest value
    and not a silent failure.

    **Ties take AVERAGE ranks**, via ``scipy.stats.rankdata``, and that is not
    a detail. A defect found during this build: the first implementation
    ranked by ``argsort(argsort(x))``, which breaks ties by position. On a
    constant sample that returns the index itself as the rank vector, so the
    correlation with the index is exactly 1.0 and the statistic reports a
    perfect monotone drift in a sample that has no variation at all. The
    constant sample is pinned with literal inputs in
    ``tests/test_regressions.py``. Average ranks give a constant rank vector
    there, a zero denominator, and the documented 0.0.
    """
    from scipy import stats

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 3:
        raise ValueError(
            f"the trend statistic needs at least 3 one-dimensional values; got shape "
            f"{array.shape}"
        )
    order = np.asarray(stats.rankdata(array), dtype=np.float64)
    index = np.arange(array.size, dtype=np.float64)
    order -= order.mean()
    index -= index.mean()
    denominator = float(np.sqrt(np.sum(order**2) * np.sum(index**2)))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(order * index) / denominator)


def half_split_statistic(values: np.ndarray) -> float:
    """``mean(second half) - mean(first half)``, equation (17).

    In the unit of ``values``. With an odd count the middle observation is
    dropped, so the two halves always have the same size and the statistic is
    not biased by an unequal split.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 4:
        raise ValueError(
            f"the half-split statistic needs at least 4 one-dimensional values; got "
            f"shape {array.shape}"
        )
    half = array.size // 2
    return float(np.mean(array[-half:]) - np.mean(array[:half]))


#: The named alternatives this module tests against, by key.
STATISTICS: dict[str, Callable[[np.ndarray], float]] = {
    "trend": trend_statistic,
    "half_split": half_split_statistic,
}


@dataclass(frozen=True)
class ExchangeabilityTest:
    """The outcome of one permutation test.

    Attributes
    ----------
    statistic:
        Which named alternative was tested, ``"trend"`` or ``"half_split"``.
    observed:
        The statistic on the data as ordered.
    p_value:
        Equation (18), two-sided on the absolute statistic.
    permutations:
        ``B``, the number of random permutations drawn.
    alpha:
        The declared significance level.
    rejected:
        ``p_value <= alpha``.
    null_mean, null_sd:
        Mean and standard deviation of the permutation null, for context.
    n:
        Sample size.
    """

    statistic: str
    observed: float
    p_value: float
    permutations: int
    alpha: float
    rejected: bool
    null_mean: float
    null_sd: float
    n: int

    @property
    def verdict(self) -> str:
        """A sentence that does not overclaim in either direction.

        A rejection is evidence against exchangeability. A non-rejection is
        **not** evidence for it: the test has power against drift and against
        a step change and little against anything else.
        """
        if self.rejected:
            return (
                f"exchangeability REJECTED against the {self.statistic} alternative at "
                f"alpha = {self.alpha:g} (p = {self.p_value:.4f}); the conformal and "
                f"bootstrap bounds have no guarantee on this sample"
            )
        return (
            f"exchangeability not rejected against the {self.statistic} alternative at "
            f"alpha = {self.alpha:g} (p = {self.p_value:.4f}); this is not evidence that "
            f"the sample IS exchangeable, only that this alternative was not detected"
        )


def _permutations(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """``(count, n)`` independent uniform permutations of ``range(n)``.

    Built by ``argsort`` of a uniform matrix rather than by ``count`` calls to
    ``Generator.permutation``, which is the same distribution and one C-level
    call instead of thousands of Python-level ones. The distinction is not
    cosmetic: the size and power measurement in
    ``validation/validate_exchangeability_output.txt`` runs several thousand
    permutation tests and does not finish in the declared compute budget with
    a Python loop.
    """
    return np.argsort(rng.random((count, n)), axis=1)


def null_distribution(
    array: np.ndarray, statistic: str, count: int, rng: np.random.Generator
) -> np.ndarray:
    """The permutation null of ``statistic``, computed for all permutations at once.

    Both named statistics are computed in closed form over a ``(count, n)``
    permutation matrix. The results are the same as applying the statistic to
    each permuted sample in turn, and the vectorised form is what makes the
    power study affordable.
    """
    n = array.size
    order = _permutations(n, count, rng)
    if statistic == "trend":
        from scipy import stats as scipy_stats

        ranks = np.asarray(scipy_stats.rankdata(array), dtype=np.float64)
        ranks = ranks - ranks.mean()
        index = np.arange(n, dtype=np.float64)
        index = index - index.mean()
        denominator = float(np.sqrt(np.sum(ranks**2) * np.sum(index**2)))
        if denominator == 0.0:
            return np.zeros(count, dtype=np.float64)
        return (ranks[order] @ index) / denominator
    if statistic == "half_split":
        half = n // 2
        permuted = array[order]
        return permuted[:, -half:].mean(axis=1) - permuted[:, :half].mean(axis=1)
    raise ValueError(f"no vectorised null is defined for {statistic!r}")  # pragma: no cover


def permutation_test(
    values: np.ndarray,
    statistic: str = "trend",
    permutations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> ExchangeabilityTest:
    """Test exchangeability of ``values`` in their given order, equations (16)-(18).

    Parameters
    ----------
    values:
        ``(n,)`` finite observations **in the order they were taken**. The
        order is the whole point: permuting them before calling this destroys
        the evidence.
    statistic:
        A key of :data:`STATISTICS`.
    permutations:
        ``B`` in equation (18). At least 99, so that the smallest attainable
        p-value ``1/(B+1)`` is below 0.01.
    alpha:
        Declared significance level, strictly inside ``(0, 1)``.
    seed:
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    ExchangeabilityTest
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"values must be one-dimensional; got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contains a non-finite value (nan or inf)")
    if statistic not in STATISTICS:
        raise ValueError(
            f"statistic must be one of {sorted(STATISTICS)}; got {statistic!r}"
        )
    count = int(permutations)
    if count < 99:
        raise ValueError(
            f"permutations must be at least 99 so that the smallest attainable p-value "
            f"1/(B+1) is below 0.01; got {permutations}"
        )
    level = validate_confidence(alpha, "alpha")
    function = STATISTICS[statistic]
    observed = abs(function(array))
    rng = np.random.default_rng(int(seed))
    null = np.abs(null_distribution(array, statistic, count, rng))
    extreme = int(np.count_nonzero(null >= observed))
    p_value = (1.0 + extreme) / (1.0 + count)
    return ExchangeabilityTest(
        statistic=statistic,
        observed=float(function(array)),
        p_value=float(p_value),
        permutations=count,
        alpha=level,
        rejected=bool(p_value <= level),
        null_mean=float(np.mean(null)),
        null_sd=float(np.std(null, ddof=1)),
        n=int(array.size),
    )
