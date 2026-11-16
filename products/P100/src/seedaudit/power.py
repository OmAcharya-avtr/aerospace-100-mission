"""Closed-form detection power for intermittent non-determinism, and its inverses.

This module is the core of the product. Everything else in `seedaudit` samples;
this is what turns a sample into a statement about what the sample could have
missed.

Model
-----
An intermittency manifests on an execution with rate ``p`` in ``(0, 1)``,
independently across executions, and is sampled with ``n`` executions. Two
sampling designs give two different detection probabilities, and conflating
them is the mistake this module exists to prevent.

**Reference-comparison design.** Each run is compared against a committed
reference value. Detection needs at least one anomalous run, so

.. math:: P_{ref}(p, n) = 1 - (1-p)^n

**Pairwise-difference design.** The runs are compared only against each other.
Detection needs at least one anomalous run *and* at least one normal run,
because ``n`` identically anomalous runs agree with each other, so

.. math:: P_{pair}(p, n) = 1 - p^n - (1-p)^n

``P_pair <= P_ref`` always, with equality only as ``p**n -> 0``. The difference
is exactly ``p**n``. Run-twice-and-diff is the pairwise design; calling it the
reference design overstates its power, by a negligible amount at low rates and
by a third of the answer at ``p = 0.5, n = 2`` (0.5 against 0.75).

Both forms are elementary results for independent Bernoulli trials; see any
standard treatment, e.g. Feller, *An Introduction to Probability Theory and Its
Applications*, Vol. I, 3rd ed. (1968), Ch. IV.

Inversion
---------
For a required detection probability ``1 - alpha`` and a repeat budget ``n``,
the smallest rate the design catches with that probability is

.. math:: p_{min}(n, \\alpha) = 1 - \\alpha^{1/n}

and if ``n`` repeats *agree* — a clean audit — the one-sided upper confidence
bound on ``p`` at confidence ``1 - alpha`` is the Clopper-Pearson bound for
zero successes in ``n`` trials, which has the same closed form

.. math:: p_{max}(n, \\alpha) = 1 - \\alpha^{1/n}

The coincidence is not an accident: the smallest rate the design would have
caught with probability ``1 - alpha`` is the largest rate consistent with
having caught nothing. For large ``n`` this is the rule of three,
``p_max ~ 3/n`` at ``alpha = 0.05`` (Hanley and Lippman-Hand, *JAMA* 249(13),
1983, 1743-1745).

Reporting ``p_max`` is the whole point. A reproducibility checker that runs
twice, finds nothing and reports "reproducible" has claimed an implied zero it
has not earned: at ``n = 2`` the honest statement is that the rate could be as
high as 0.776.

Assumptions, which the bound is worthless without
-------------------------------------------------
1. **Independence across repeats is the strong one.** Repeats inside one
   process share RNG state, import caches, a fixed ``PYTHONHASHSEED`` and a
   warm filesystem cache; they are positively correlated, so the true power is
   **lower** than the formula and ``p_max`` is **optimistic**. The mitigation
   is to make each repeat a fresh subprocess in a fresh pristine copy with a
   fresh hash seed, which :mod:`seedaudit.dynamic` does. The residual
   correlation is not zero and this module does not estimate it.
2. **Stationarity.** ``p`` is constant over the repeats.
3. **The anomaly is detectable when it manifests.** An anomaly below the
   printed precision of the output is invisible to a text comparison at any
   ``n``.
4. **Seed draws are not repeat draws.** See :func:`seed_axis_detection`.

Units: every rate and probability here is a dimensionless probability in
``[0, 1]``; every repeat count is a dimensionless positive integer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Default one-sided significance level for every bound in this module.
DEFAULT_ALPHA = 0.05

#: Default confidence level for bootstrap intervals.
DEFAULT_CONFIDENCE = 0.95

#: The rate this mission actually shipped: one intermittent failure in 44.
#: Named here so that no two reported numbers rest on different arithmetic.
MISSION_RATE = 1.0 / 44.0


def _check_rate(p: float, *, name: str = "p") -> float:
    """Return ``p`` as a float, raising if it is not a probability."""
    value = float(p)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {p!r}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return value


def _check_repeats(n: int, *, name: str = "n") -> int:
    """Return ``n`` as an int, raising if it is not a positive repeat count."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(n).__name__}")
    value = int(n)
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _check_alpha(alpha: float) -> float:
    """Return ``alpha`` as a float, raising if it is not in the open unit interval."""
    value = float(alpha)
    if not math.isfinite(value):
        raise ValueError(f"alpha must be finite, got {alpha!r}")
    if not 0.0 < value < 1.0:
        raise ValueError(f"alpha must lie in (0, 1) exclusive, got {value!r}")
    return value


def detection_probability_reference(p: float, n: int) -> float:
    """Probability of detecting rate ``p`` in ``n`` runs against a reference.

    ``1 - (1-p)**n``, evaluated as ``-expm1(n * log1p(-p))`` so that the
    cancellation at small ``p`` does not eat the answer. Dimensionless.

    Args:
        p: per-run manifestation rate, dimensionless, in ``[0, 1]``.
        n: number of runs, dimensionless, at least 1.

    Returns:
        Detection probability, dimensionless, in ``[0, 1]``.
    """
    rate = _check_rate(p)
    runs = _check_repeats(n)
    if rate == 0.0:
        return 0.0
    if rate == 1.0:
        return 1.0
    return float(-math.expm1(runs * math.log1p(-rate)))


def detection_probability_pairwise(p: float, n: int) -> float:
    """Probability of detecting rate ``p`` in ``n`` runs compared only to each other.

    ``1 - p**n - (1-p)**n``. The expression is symmetric under
    ``p -> 1 - p``, which is used for numerical stability: the small of the two
    is raised to the power ``n`` directly and the large one goes through
    ``expm1``/``log1p``.

    At ``n = 1`` this is identically 0: one run has nothing to disagree with.

    Args:
        p: per-run manifestation rate, dimensionless, in ``[0, 1]``.
        n: number of runs, dimensionless, at least 1.

    Returns:
        Detection probability, dimensionless, in ``[0, 1]``.
    """
    rate = _check_rate(p)
    runs = _check_repeats(n)
    if runs == 1 or rate in (0.0, 1.0):
        return 0.0
    small = min(rate, 1.0 - rate)
    value = -math.expm1(runs * math.log1p(-small)) - small**runs
    return float(min(max(value, 0.0), 1.0))


def smallest_detectable_rate(n: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Smallest rate detected with probability ``1 - alpha`` by ``n`` runs.

    ``1 - alpha**(1/n)``, from inverting :func:`detection_probability_reference`.
    Evaluated through ``expm1``/``log`` for the same cancellation reason.

    Args:
        n: repeat budget, dimensionless, at least 1.
        alpha: one-sided significance level, dimensionless, in ``(0, 1)``.

    Returns:
        Rate, dimensionless, in ``(0, 1)``.
    """
    runs = _check_repeats(n)
    level = _check_alpha(alpha)
    return float(-math.expm1(math.log(level) / runs))


def escaped_rate_upper_bound(n: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Upper confidence bound on ``p`` after ``n`` runs that all agreed.

    The Clopper-Pearson one-sided upper bound for zero successes in ``n``
    Bernoulli trials, ``1 - alpha**(1/n)``. This is the number a clean audit
    reports **instead of zero**.

    It is numerically the same expression as :func:`smallest_detectable_rate`
    and is a separate function because it is a different claim: that one is
    what the design can catch, this one is what a clean result permits. A test
    asserts they agree to within rounding, so the identity is checked rather
    than assumed.

    Args:
        n: number of agreeing runs, dimensionless, at least 1.
        alpha: one-sided significance level, dimensionless, in ``(0, 1)``.

    Returns:
        Upper bound on the rate, dimensionless, in ``(0, 1)``.
    """
    return smallest_detectable_rate(n, alpha)


def repeats_for_detection(p: float, power: float = 1.0 - DEFAULT_ALPHA) -> int:
    """Smallest ``n`` with :func:`detection_probability_reference` at least ``power``.

    ``ceil(log(1 - power) / log(1 - p))``. The ceiling is taken on the exact
    real solution and then verified by evaluating the closed form at ``n`` and
    ``n - 1``, because a ceiling of a floating-point logarithm ratio can land
    one short.

    Args:
        p: per-run manifestation rate, dimensionless, in ``(0, 1)``.
        power: required detection probability, dimensionless, in ``(0, 1)``.

    Returns:
        Repeat count, dimensionless, at least 1.
    """
    rate = _check_rate(p)
    if rate <= 0.0:
        raise ValueError("a rate of 0 is never detected; no finite n exists")
    target = float(power)
    if not 0.0 < target < 1.0:
        raise ValueError(f"power must lie in (0, 1) exclusive, got {target!r}")
    if rate == 1.0:
        return 1
    exact = math.log1p(-target) / math.log1p(-rate)
    runs = max(1, int(math.ceil(exact)))
    while runs > 1 and detection_probability_reference(rate, runs - 1) >= target:
        runs -= 1
    while detection_probability_reference(rate, runs) < target:
        runs += 1
    return runs


def repeats_for_escaped_rate_bound(p_max: float, alpha: float = DEFAULT_ALPHA) -> int:
    """Smallest ``n`` whose clean-audit bound is at or below ``p_max``.

    Args:
        p_max: the escaped-rate bound to achieve, dimensionless, in ``(0, 1)``.
        alpha: one-sided significance level, dimensionless, in ``(0, 1)``.

    Returns:
        Repeat count, dimensionless, at least 1.
    """
    bound = _check_rate(p_max, name="p_max")
    if bound <= 0.0:
        raise ValueError("no finite number of agreeing runs bounds the rate at 0")
    level = _check_alpha(alpha)
    if bound >= 1.0:
        return 1
    runs = max(1, int(math.ceil(math.log(level) / math.log1p(-bound))))
    while runs > 1 and escaped_rate_upper_bound(runs - 1, level) <= bound:
        runs -= 1
    while escaped_rate_upper_bound(runs, level) > bound:
        runs += 1
    return runs


# ---------------------------------------------------------------------------
# Independently derived bounds.
#
# The mission's build guide requires at least one check whose bound comes from
# mathematics the implementation does not use, because a product whose every
# test compares an expression against expressions sharing its own cancellation
# cannot catch that cancellation. The three functions below are derived from
# inclusion-exclusion and from the exponential inequality, neither of which is
# the product form `(1-p)**n` that `detection_probability_reference` evaluates.
# ---------------------------------------------------------------------------


def union_bound_reference(p: float, n: int) -> float:
    """Bonferroni upper bound ``min(1, n*p)`` on the reference detection probability.

    First-order inclusion-exclusion: ``P(union A_i) <= sum P(A_i) = n*p``.
    Derived from the union, not from ``(1-p)**n``.

    Returns:
        Upper bound, dimensionless, in ``[0, 1]``.
    """
    rate = _check_rate(p)
    runs = _check_repeats(n)
    return float(min(1.0, runs * rate))


def bonferroni_bracket_reference(p: float, n: int) -> tuple[float, float]:
    """Two-term inclusion-exclusion bracket on the reference detection probability.

    ``S1 - S2 <= P(union A_i) <= S1`` with ``S1 = n*p`` and
    ``S2 = C(n,2) * p**2`` for independent equiprobable events. Both ends come
    from truncating the inclusion-exclusion series, which is different
    mathematics from the complement product.

    Returns:
        ``(lower, upper)``, both dimensionless, in ``[0, 1]``.
    """
    rate = _check_rate(p)
    runs = _check_repeats(n)
    s1 = runs * rate
    s2 = 0.5 * runs * (runs - 1) * rate * rate
    return float(max(0.0, s1 - s2)), float(min(1.0, s1))


def exponential_bracket_reference(p: float, n: int) -> tuple[float, float]:
    """Bracket from the exponential inequality on ``log(1-p)``.

    From ``-p/(1-p) <= log(1-p) <= -p`` for ``p`` in ``[0, 1)``:

    ``1 - exp(-n*p) <= 1 - (1-p)**n <= 1 - exp(-n*p/(1-p))``

    Derived from a bound on the logarithm, not from the product itself.

    Returns:
        ``(lower, upper)``, both dimensionless, in ``[0, 1]``.
    """
    rate = _check_rate(p)
    runs = _check_repeats(n)
    if rate >= 1.0:
        return 1.0, 1.0
    lower = -math.expm1(-runs * rate)
    upper = -math.expm1(-runs * rate / (1.0 - rate))
    return float(min(max(lower, 0.0), 1.0)), float(min(max(upper, 0.0), 1.0))


def seed_axis_detection(population: int, sensitive: int, draws: int) -> float:
    """Probability a seed sweep of ``draws`` distinct seeds hits a sensitive seed.

    The seed axis is **not** the repeat axis. A quantity that is a
    deterministic function of the seed yields the same value on every repeat at
    a fixed seed, so repeats carry zero information about seed-dependence, and
    the Bernoulli formulae above do not apply.

    Seeds are drawn **without replacement** from a finite declared set, so the
    count of sensitive seeds drawn is hypergeometric and the detection
    probability is one minus the probability of drawing none:

    ``1 - C(population - sensitive, draws) / C(population, draws)``

    Args:
        population: size of the declared seed set, dimensionless, at least 1.
        sensitive: how many of those seeds reveal the dependence, dimensionless.
        draws: how many distinct seeds are drawn, dimensionless.

    Returns:
        Detection probability, dimensionless, in ``[0, 1]``.
    """
    pop = _check_repeats(population, name="population")
    if isinstance(sensitive, bool) or not isinstance(sensitive, (int, np.integer)):
        raise TypeError(f"sensitive must be an int, got {type(sensitive).__name__}")
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)):
        raise TypeError(f"draws must be an int, got {type(draws).__name__}")
    sens = int(sensitive)
    take = int(draws)
    if not 0 <= sens <= pop:
        raise ValueError(f"sensitive must lie in [0, {pop}], got {sens}")
    if not 0 <= take <= pop:
        raise ValueError(f"draws must lie in [0, {pop}], got {take}")
    if sens == 0 or take == 0:
        return 0.0
    if take > pop - sens:
        return 1.0
    miss = math.comb(pop - sens, take) / math.comb(pop, take)
    return float(min(max(1.0 - miss, 0.0), 1.0))


@dataclass(frozen=True)
class PowerRow:
    """One row of the detection-power table.

    Attributes:
        repeats: number of runs, dimensionless.
        rate: the per-run manifestation rate the row is evaluated at,
            dimensionless.
        reference: :func:`detection_probability_reference` at that pair.
        pairwise: :func:`detection_probability_pairwise` at that pair.
        smallest_detectable: :func:`smallest_detectable_rate` at ``repeats``.
        escaped_upper_bound: :func:`escaped_rate_upper_bound` at ``repeats``.
    """

    repeats: int
    rate: float
    reference: float
    pairwise: float
    smallest_detectable: float
    escaped_upper_bound: float


def power_table(
    repeats: tuple[int, ...] | list[int],
    rate: float = MISSION_RATE,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[PowerRow, ...]:
    """Build the detection-power table for a list of repeat budgets.

    Args:
        repeats: repeat budgets, dimensionless.
        rate: the per-run manifestation rate, dimensionless. Defaults to
            :data:`MISSION_RATE`, the rate this mission actually shipped.
        alpha: one-sided significance level, dimensionless.

    Returns:
        One :class:`PowerRow` per entry of ``repeats``, in the order given.
    """
    if not repeats:
        raise ValueError("repeats must be a non-empty sequence")
    return tuple(
        PowerRow(
            repeats=_check_repeats(n),
            rate=_check_rate(rate),
            reference=detection_probability_reference(rate, n),
            pairwise=detection_probability_pairwise(rate, n),
            smallest_detectable=smallest_detectable_rate(n, alpha),
            escaped_upper_bound=escaped_rate_upper_bound(n, alpha),
        )
        for n in repeats
    )


def monte_carlo_detection(
    p: float,
    n: int,
    trials: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    """Monte Carlo estimate of both detection probabilities at ``(p, n)``.

    Exists so that the closed forms are checked against a simulation rather
    than against the algebra that produced them.

    Args:
        p: per-run manifestation rate, dimensionless.
        n: runs per simulated audit, dimensionless.
        trials: number of simulated audits, dimensionless, at least 1.
        rng: numpy generator; the caller owns the seed.

    Returns:
        ``(reference_estimate, reference_standard_error, pairwise_estimate,
        pairwise_standard_error)``, all dimensionless.
    """
    rate = _check_rate(p)
    runs = _check_repeats(n)
    count = _check_repeats(trials, name="trials")
    draws = rng.random((count, runs)) < rate
    hits = draws.sum(axis=1)
    reference = hits > 0
    pairwise = (hits > 0) & (hits < runs)
    ref_mean = float(reference.mean())
    pair_mean = float(pairwise.mean())
    ref_se = float(math.sqrt(max(ref_mean * (1.0 - ref_mean), 0.0) / count))
    pair_se = float(math.sqrt(max(pair_mean * (1.0 - pair_mean), 0.0) / count))
    return ref_mean, ref_se, pair_mean, pair_se


def bootstrap_interval(
    indicators: np.ndarray,
    rng: np.random.Generator,
    resamples: int = 2000,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for the mean of a 0/1 indicator array.

    The resampling unit is the **audit**, which is the independent unit of this
    experiment: each entry of ``indicators`` records whether one audit built
    from its own disjoint block of runs detected the injected anomaly.

    Args:
        indicators: 1-D array of 0/1, one entry per audit, dimensionless.
        rng: numpy generator; the caller owns the seed.
        resamples: bootstrap resamples, dimensionless, at least 1.
        confidence: interval coverage, dimensionless, in ``(0, 1)``.

    Returns:
        ``(point_estimate, lower, upper)``, all dimensionless. The point
        estimate is the plain mean, never a bootstrap mean.
    """
    values = np.asarray(indicators, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("indicators must be non-empty")
    if np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("indicators must contain only 0 and 1")
    draws = _check_repeats(resamples, name="resamples")
    level = float(confidence)
    if not 0.0 < level < 1.0:
        raise ValueError(f"confidence must lie in (0, 1) exclusive, got {level!r}")
    point = float(values.mean())
    idx = rng.integers(0, values.size, size=(draws, values.size))
    means = values[idx].mean(axis=1)
    tail = 0.5 * (1.0 - level)
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return point, float(lower), float(upper)


# ---------------------------------------------------------------------------
# The two-arrangement classifier's own error rate.
#
# This section exists because a measurement forced it. Running the classifier
# over the injected corpus at n = 2 classified an intrinsically
# non-deterministic script at rate 0.5 as a SELF-POLLUTING CHECK: its two
# pristine runs happened to agree and its two reused-tree runs happened to
# differ, which is the self-pollution signature. The classifier is not wrong
# about what it saw; it is a statement about what two runs per arrangement can
# distinguish, and the honest response is to quantify it rather than to raise
# the repeat count until the example stops embarrassing the tool.
# ---------------------------------------------------------------------------


def agreement_probability(p: float, n: int) -> float:
    """Probability that ``n`` independent runs at rate ``p`` all agree.

    ``p**n + (1-p)**n``: either every run manifests or none does. This is the
    complement of :func:`detection_probability_pairwise`.

    Args:
        p: per-run manifestation rate, dimensionless.
        n: runs in the arrangement, dimensionless.

    Returns:
        Probability, dimensionless, in ``[0, 1]``.
    """
    rate = _check_rate(p)
    runs = _check_repeats(n)
    return float(min(max(1.0 - detection_probability_pairwise(rate, runs), 0.0), 1.0))


def self_pollution_misclassification(p: float, n: int, trials: int = 1) -> float:
    """Probability a merely non-deterministic script is called self-polluting.

    The two-arrangement classifier reports ``SELF_POLLUTING`` when the pristine
    pair agrees and the reused tree differs. For a script with intrinsic rate
    ``p`` and **no self-pollution at all**, both arrangements are independent
    blocks of ``n`` Bernoulli runs, so that outcome has probability

    .. math:: q(1-q), \\quad q = p^n + (1-p)^n

    and the classifier is wrong. Repeating the whole two-arrangement
    experiment ``trials`` times and requiring the same pattern every time
    raises the bar to ``(q(1-q))**trials``.

    Args:
        p: intrinsic per-run manifestation rate, dimensionless.
        n: runs per arrangement, dimensionless.
        trials: independent repetitions of the whole experiment,
            dimensionless, at least 1.

    Returns:
        Misclassification probability, dimensionless, in ``[0, 1]``.
    """
    rate = _check_rate(p)
    runs = _check_repeats(n)
    repetitions = _check_repeats(trials, name="trials")
    q = agreement_probability(rate, runs)
    return float((q * (1.0 - q)) ** repetitions)


def worst_case_misclassification(n: int, trials: int = 1) -> float:
    """Worst-case :func:`self_pollution_misclassification` over all rates.

    ``q(1-q)`` is maximised at ``q = 0.5`` with value 0.25, and
    ``q = p**n + (1-p)**n`` attains 0.5 for every ``n >= 2`` -- at ``n = 2``
    only at ``p = 0.5``, and for larger ``n`` at a rate nearer the ends. So
    **the worst case is 0.25 for every repeat budget from 2 upwards, and
    raising the repeat count does not lower it.** It moves the rate at which
    the classifier is worst, and nothing else.

    The mitigation is not more runs per arrangement but more **independent
    repetitions of the whole two-arrangement experiment**, which multiplies:
    ``0.25**trials``. At ``n = 1`` the classifier cannot fire at all and the
    worst case is 0.

    Args:
        n: runs per arrangement, dimensionless.
        trials: independent repetitions of the whole experiment, dimensionless.

    Returns:
        Worst-case misclassification probability, dimensionless.
    """
    runs = _check_repeats(n)
    repetitions = _check_repeats(trials, name="trials")
    if runs == 1:
        return 0.0
    return float(0.25**repetitions)


def worst_case_misclassification_rate(n: int) -> float:
    """The rate at which :func:`self_pollution_misclassification` is worst.

    Solves ``p**n + (1-p)**n = 0.5`` on ``(0, 0.5]`` by bisection, which is
    monotone decreasing there, so the bracket is unambiguous.

    Args:
        n: runs per arrangement, dimensionless, at least 2.

    Returns:
        The rate, dimensionless, in ``(0, 0.5]``.
    """
    runs = _check_repeats(n)
    if runs < 2:
        raise ValueError(f"the classifier cannot fire below 2 runs, got {runs}")
    if runs == 2:
        return 0.5
    low, high = 0.0, 0.5
    for _ in range(200):
        middle = 0.5 * (low + high)
        if agreement_probability(middle, runs) > 0.5:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))
