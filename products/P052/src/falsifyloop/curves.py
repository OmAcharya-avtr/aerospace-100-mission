r"""The headline deliverable: sample-efficiency curves with bootstrap bands.

A **sample-efficiency curve** is the probability that a strategy has found a
violation by simulation ``n``, as a function of ``n``. It is estimated from
``R`` independent seeded runs of the same strategy on the same instance at the
same budget, as the fraction of runs whose first violation came at or before
``n``.

Right censoring, stated plainly
-------------------------------
A run that reaches its budget without a violation contributes ``None``. It is
**not** a run that found a violation late and it is **not** evidence that the
instance has no violation; it is a run that found nothing. The curve is
therefore a right-censored empirical distribution function and it is only ever
evaluated out to the budget. Any statement about behaviour past the budget would
be extrapolation and this module does not make one -- which is also why the
summary statistic is a **median** reported as ``> budget`` when fewer than half
the runs found anything, rather than a mean, which censoring makes undefined.

The bootstrap, and what its resampling unit is
----------------------------------------------
Per instance: the resampling unit is **one run**. ``n_boot`` resamples of the
``R`` runs are drawn with replacement, the curve recomputed from each, and the
band is the pointwise percentile interval (Efron & Tibshirani 1993, ch. 13). The
band is therefore pointwise, not simultaneous: it is a confidence statement
about the curve at a single ``n``, and reading the whole band as a 95 % envelope
for the entire curve overstates it. That is stated here because it is the single
most common way a plot like this is over-read.

Across instances: the aggregate curve is the **unweighted mean of the
per-instance curves**, so every instance counts once regardless of how many runs
it contributed, and the bootstrap is **stratified** -- runs are resampled within
each instance and the instance set is held fixed. The alternative, resampling
instances too, would make the band describe uncertainty about a population of
instances this suite is not a random sample from.

A monotonicity note
-------------------
Each per-instance curve is non-decreasing in ``n`` by construction. The
pointwise percentile band is also non-decreasing because every bootstrap curve
is. The aggregate curve is a mean of non-decreasing curves and so is
non-decreasing as well; nothing here enforces monotonicity after the fact, so if
a plotted band is not monotone that is a defect and not a smoothing artefact.

References
----------
Efron, B. and Tibshirani, R. J. (1993), *An Introduction to the Bootstrap*,
Chapman & Hall. The percentile bootstrap interval, chapter 13.

Clopper, C. J. and Pearson, E. S. (1934), "The use of confidence limits
illustrated in the case of the binomial", Biometrika 26(4), 404-413. The exact
binomial interval used for the per-instance difficulty estimate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import beta

#: Default bootstrap resample count. 2000 is enough for a 95 % percentile
#: interval to be stable to about one part in a hundred of its width.
DEFAULT_BOOTSTRAP = 2000


def _check_first_violations(first_violations: Sequence[int | None], budget: int) -> np.ndarray:
    """Validate and convert to a float array with ``inf`` for censored runs."""
    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")
    if len(first_violations) == 0:
        raise ValueError("need at least one run to form a curve")
    out = np.empty(len(first_violations), dtype=float)
    for i, value in enumerate(first_violations):
        if value is None:
            out[i] = math.inf
            continue
        index = int(value)
        if index < 1 or index > budget:
            raise ValueError(
                f"first-violation index {value!r} at position {i} is outside "
                f"1..budget ({budget}); a run cannot find a violation it did not simulate"
            )
        out[i] = float(index)
    return out


def efficiency_curve(first_violations: Sequence[int | None], budget: int) -> np.ndarray:
    """Empirical ``P(found by n)`` for ``n = 1..budget``, shape ``(budget,)``.

    Parameters
    ----------
    first_violations:
        One entry per run: the 1-based simulation index of its first violation,
        or ``None`` for a run that found nothing within the budget.
    budget:
        Simulation budget every run was given.
    """
    values = _check_first_violations(first_violations, budget)
    n = np.arange(1, int(budget) + 1)
    return (values[:, None] <= n[None, :]).mean(axis=0)


def bootstrap_band(
    first_violations: Sequence[int | None],
    budget: int,
    n_boot: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile bootstrap band for :func:`efficiency_curve`.

    Parameters
    ----------
    first_violations, budget:
        As for :func:`efficiency_curve`.
    n_boot:
        Bootstrap resamples, at least 100.
    alpha:
        Two-sided miss rate, in ``(0, 0.5)``. ``0.05`` gives a 95 % band.
    seed:
        Seed for the resampling, so a committed band is reproducible.

    Returns
    -------
    tuple of numpy.ndarray
        ``(lower, upper)``, each shape ``(budget,)``.
    """
    values = _check_first_violations(first_violations, budget)
    if n_boot < 100:
        raise ValueError(f"n_boot must be at least 100, got {n_boot}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must lie in (0, 0.5), got {alpha}")
    rng = np.random.default_rng(seed)
    n = np.arange(1, int(budget) + 1)
    indicators = values[:, None] <= n[None, :]
    picks = rng.integers(0, values.size, size=(int(n_boot), values.size))
    draws = indicators[picks].mean(axis=1)
    lower = np.quantile(draws, alpha / 2.0, axis=0)
    upper = np.quantile(draws, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


def aggregate_curve(
    per_instance: Mapping[str, Sequence[int | None]], budget: int
) -> np.ndarray:
    """Unweighted mean of the per-instance curves, shape ``(budget,)``.

    Every instance contributes one curve and therefore equal weight, so an
    instance that happened to be run more times does not dominate.
    """
    if not per_instance:
        raise ValueError("need at least one instance to aggregate")
    curves = [efficiency_curve(runs, budget) for runs in per_instance.values()]
    return np.mean(np.stack(curves), axis=0)


def bootstrap_aggregate_band(
    per_instance: Mapping[str, Sequence[int | None]],
    budget: int,
    n_boot: int = DEFAULT_BOOTSTRAP,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified percentile bootstrap band for :func:`aggregate_curve`.

    Runs are resampled with replacement **within** each instance; the instance
    set is held fixed. See the module docstring for why.
    """
    if not per_instance:
        raise ValueError("need at least one instance to aggregate")
    if n_boot < 100:
        raise ValueError(f"n_boot must be at least 100, got {n_boot}")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"alpha must lie in (0, 0.5), got {alpha}")
    rng = np.random.default_rng(seed)
    n = np.arange(1, int(budget) + 1)
    stacks = []
    for runs in per_instance.values():
        values = _check_first_violations(runs, budget)
        indicators = values[:, None] <= n[None, :]
        picks = rng.integers(0, values.size, size=(int(n_boot), values.size))
        stacks.append(indicators[picks].mean(axis=1))
    draws = np.mean(np.stack(stacks), axis=0)
    return (
        np.quantile(draws, alpha / 2.0, axis=0),
        np.quantile(draws, 1.0 - alpha / 2.0, axis=0),
    )


def median_first_violation(
    first_violations: Sequence[int | None], budget: int
) -> float | None:
    """Median simulations to the first violation, or ``None`` if censored.

    Returns ``None`` when fewer than half the runs found a violation, which is
    the only honest answer: the median lies beyond the budget and the data do
    not say where.
    """
    values = _check_first_violations(first_violations, budget)
    found = np.sort(values[np.isfinite(values)])
    if found.size * 2 < values.size:
        return None
    return float(np.median(values)) if np.isfinite(np.median(values)) else None


def success_rate(first_violations: Sequence[int | None], budget: int) -> float:
    """Fraction of runs that found a violation within the budget."""
    values = _check_first_violations(first_violations, budget)
    return float(np.mean(np.isfinite(values)))


def area_under_curve(first_violations: Sequence[int | None], budget: int) -> float:
    """Mean of the efficiency curve over ``n = 1..budget``, in ``[0, 1]``.

    A single scalar summary of a whole curve: the average probability of having
    found a violation over the budget. It is **not** an AUROC and carries no
    classification meaning; it is used only to order strategies compactly in the
    report table, with the curve itself as the actual deliverable.
    """
    return float(np.mean(efficiency_curve(first_violations, budget)))


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval for a violation probability.

    Parameters
    ----------
    successes:
        Number of violating draws, ``0 <= successes <= trials``.
    trials:
        Number of draws.
    alpha:
        Two-sided miss rate, in ``(0, 1)``.

    Returns
    -------
    tuple of float
        ``(lower, upper)``. Exact rather than normal-approximate because the
        difficulty of the hardest instances in this suite is of order ``1e-3``,
        where a Wald interval can reach below zero and is simply wrong.
    """
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must lie in 0..{trials}, got {successes}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return lower, upper
