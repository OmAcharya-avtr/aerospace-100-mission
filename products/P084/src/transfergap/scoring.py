"""Score a bound on ACHIEVED COVERAGE, and report width beside it but never instead.

The scoring rule
----------------
A bound is scored on whether it **covers** at the confidence it declares, not
on how tight it is. A tighter interval that under-covers is worse than a loose
one that holds: the loose one is conservative and the engineer reading it knows
it, while the tight one is wrong and reads as a measurement. Width is reported
in its own column, always, because a bound that covers by being enormous is
also useless -- but width never enters the verdict.

Two coverage definitions, kept apart
------------------------------------
**(24) Mean-target coverage.** For a bound on ``mu = E[d]``, the coverage is
the fraction of repetitions in which ``upper >= mu_true``. The declared target
is the stated confidence ``1 - delta``.

**(25) Prediction-target marginal coverage.** For a limit covering a future
observation, the per-repetition coverage is ``P(d_new <= upper)`` estimated on
a large fixed pool of operating conditions drawn from the same envelope, and
the reported marginal coverage is its average over repetitions. The declared
target is the proportion ``p``.

**(26) Prediction-target tolerance rate.** The fraction of repetitions whose
per-repetition coverage is at least ``p``. This is the stronger,
training-conditional statement, and it is the one a tolerance interval makes
and a conformal interval does not. Reporting (25) alone makes a marginally
valid method look like a tolerance interval; both are reported.

Monte Carlo error
-----------------
Every rate here is a binomial proportion over ``repetitions`` draws, so its
standard error is ``sqrt(r (1 - r) / repetitions)`` and that is reported with
it. A coverage of 0.93 against a declared 0.95 at 300 repetitions is within
1.5 standard errors and is not evidence of under-coverage; the same number at
30000 repetitions is. No verdict in this package is stated without the
standard error beside it.

**The held-out pool contributes its own error, and leaving it out was a defect
found during this build.** Equation (25) estimates each repetition's coverage
on a FIXED pool of ``m`` operating conditions, so the reported coverage
inherits the pool's own sampling error as well as the repetition error, and
the two are independent:

    se = sqrt( s^2 / repetitions  +  p (1 - p) / m ).                     (27)

At ``m = 4000`` and ``p = 0.95`` the second term alone is 0.0034, which is
larger than the first term at 300 repetitions. The first version of this
module reported only the first term and consequently declared split conformal
-- which is exactly marginally valid by construction -- to be measurably
under-covering at ``n = 60``, on a measured 0.9454 against 0.95 with a stated
error of 0.0019. With the pool term the stated error is 0.0039 and the same
measurement is within one and a half standard errors, which is what the theory
says it should be. The pool term is in equation (27) because the comparison
being made needs it, not because it makes a row pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .bounds import Bound
from .discrepancy import GapSample

__all__ = [
    "CoverageScore",
    "score_mean_bounds",
    "score_prediction_bounds",
]

#: A method that turns a field sample into an upper confidence bound on the mean.
MeanMethod = Callable[[GapSample], Bound]

#: A method that turns a field sample into an upper limit at each test condition.
#: A marginal method returns a constant array; a conditional one does not.
PredictionMethod = Callable[[GapSample, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CoverageScore:
    """One method's measured coverage and width at one sample size.

    Attributes
    ----------
    method:
        Method name.
    target:
        ``"mean"`` or ``"quantile"``.
    n:
        Field sample size.
    repetitions:
        Number of independent field samples drawn.
    declared:
        The coverage the method claims: ``1 - delta`` for a mean bound, ``p``
        for a prediction limit.
    coverage:
        Equation (24) or (25).
    coverage_se:
        Binomial standard error of :attr:`coverage` over :attr:`repetitions`.
        For equation (25) this is the standard error of the MEAN of a bounded
        quantity, computed from its sample spread rather than assumed binomial.
    tolerance_rate:
        Equation (26); ``None`` for a mean target.
    tolerance_rate_se:
        Binomial standard error of :attr:`tolerance_rate`.
    mean_upper:
        Mean of the upper limit over repetitions, in ``unit``. For a
        conditional method this is the mean over test conditions as well.
    mean_width:
        Mean of ``upper - estimate`` for a mean bound, or of the limit itself
        for a prediction limit, in ``unit``. Stated per row in the reporting
        code so it is never ambiguous.
    failures:
        Repetitions in which the method could not produce a limit at all --
        for example Wilks' limit below the ``n`` of equation (7). Counted, not
        hidden, and excluded from the rates.
    unit:
        Unit of :attr:`mean_upper` and :attr:`mean_width`.
    distribution_free:
        Whether the method carries a distribution-free finite-sample
        guarantee.
    """

    method: str
    target: str
    n: int
    repetitions: int
    declared: float
    coverage: float
    coverage_se: float
    tolerance_rate: float | None
    tolerance_rate_se: float | None
    mean_upper: float
    mean_width: float
    failures: int
    unit: str
    distribution_free: bool

    @property
    def covers(self) -> bool:
        """Whether the measured coverage reaches the declared level.

        The comparison carries a two-standard-error allowance for Monte Carlo
        error, because a strict comparison of a rate against a threshold is a
        boolean produced by a threshold comparison on a random quantity, and
        asserting such a boolean is one of the defect classes this mission has
        shipped. A verdict of ``True`` means "not measurably below", not
        "proved to hold".
        """
        return bool(self.coverage >= self.declared - 2.0 * self.coverage_se)


def _binomial_se(rate: float, count: int) -> float:
    if count <= 0:
        return float("nan")
    return float(np.sqrt(max(rate * (1.0 - rate), 0.0) / count))


def score_mean_bounds(
    methods: dict[str, MeanMethod],
    draw: Callable[[int, int], GapSample],
    true_mean: float,
    n: int,
    repetitions: int,
    seed: int,
) -> list[CoverageScore]:
    """Measure equation (24) for each method at one sample size.

    Parameters
    ----------
    methods:
        Name to callable taking a :class:`~transfergap.discrepancy.GapSample`
        and returning a :class:`~transfergap.bounds.Bound` with
        ``target="mean"``.
    draw:
        ``draw(n, seed) -> GapSample``, the field-sample generator.
    true_mean:
        The true ``E[d]``, in the sample's unit. Every coverage here is
        anchored on this and never on another method's output.
    n:
        Field sample size.
    repetitions:
        Number of independent field samples.
    seed:
        Base seed; repetition ``r`` uses ``seed + r``.
    """
    if repetitions < 1:
        raise ValueError(f"repetitions must be at least 1; got {repetitions}")
    covered = {name: 0 for name in methods}
    uppers: dict[str, list[float]] = {name: [] for name in methods}
    slacks: dict[str, list[float]] = {name: [] for name in methods}
    failures = {name: 0 for name in methods}
    free = {name: True for name in methods}
    unit = ""
    for repetition in range(repetitions):
        sample = draw(n, seed + repetition)
        unit = sample.unit
        for name, method in methods.items():
            try:
                bound = method(sample)
            except (ValueError, RuntimeError):
                failures[name] += 1
                continue
            free[name] = bound.distribution_free
            covered[name] += int(bound.upper >= true_mean)
            uppers[name].append(bound.upper)
            slacks[name].append(bound.slack)
    scores = []
    for name in methods:
        valid = repetitions - failures[name]
        rate = covered[name] / valid if valid else float("nan")
        declared = 0.0
        sample = draw(n, seed)
        bound = None
        try:
            bound = methods[name](sample)
            declared = bound.confidence
        except (ValueError, RuntimeError):
            declared = float("nan")
        scores.append(
            CoverageScore(
                method=name,
                target="mean",
                n=int(n),
                repetitions=int(repetitions),
                declared=float(declared),
                coverage=float(rate),
                coverage_se=_binomial_se(rate, valid),
                tolerance_rate=None,
                tolerance_rate_se=None,
                mean_upper=float(np.mean(uppers[name])) if uppers[name] else float("nan"),
                mean_width=float(np.mean(slacks[name])) if slacks[name] else float("nan"),
                failures=int(failures[name]),
                unit=unit,
                distribution_free=bool(free[name]),
            )
        )
    return scores


def score_prediction_bounds(
    methods: dict[str, PredictionMethod],
    draw: Callable[[int, int], GapSample],
    test_conditions: np.ndarray,
    test_gaps: np.ndarray,
    n: int,
    repetitions: int,
    seed: int,
    proportion: float,
    distribution_free: dict[str, bool] | None = None,
) -> list[CoverageScore]:
    """Measure equations (25) and (26) for each method at one sample size.

    Parameters
    ----------
    methods:
        Name to callable ``(field_sample, test_conditions) -> upper limits``,
        one limit per test condition.
    draw:
        ``draw(n, seed) -> GapSample``.
    test_conditions, test_gaps:
        A large fixed pool of operating conditions and their realised
        discrepancies, drawn once from the same envelope. Fixing the pool
        removes the Monte Carlo noise of the inner coverage estimate from the
        comparison between methods, which is the comparison being made.
    proportion:
        The declared ``p``.
    distribution_free:
        Per-method flag for the reported column; defaults to ``False``, the
        honest default for anything not explicitly claimed.
    """
    if repetitions < 1:
        raise ValueError(f"repetitions must be at least 1; got {repetitions}")
    pool_gaps = np.asarray(test_gaps, dtype=np.float64)
    conditions = np.asarray(test_conditions, dtype=np.float64)
    if pool_gaps.ndim != 1 or conditions.ndim != 2 or pool_gaps.size != conditions.shape[0]:
        raise ValueError(
            f"test_conditions {conditions.shape} and test_gaps {pool_gaps.shape} do not "
            f"describe the same rows"
        )
    flags = distribution_free or {}
    per_rep: dict[str, list[float]] = {name: [] for name in methods}
    widths: dict[str, list[float]] = {name: [] for name in methods}
    failures = {name: 0 for name in methods}
    unit = ""
    for repetition in range(repetitions):
        sample = draw(n, seed + repetition)
        unit = sample.unit
        for name, method in methods.items():
            try:
                band = np.asarray(method(sample, conditions), dtype=np.float64)
            except (ValueError, RuntimeError):
                failures[name] += 1
                continue
            if band.shape != pool_gaps.shape:
                band = np.broadcast_to(band, pool_gaps.shape)
            per_rep[name].append(float(np.mean(pool_gaps <= band)))
            widths[name].append(float(np.mean(band)))
    scores = []
    for name in methods:
        values = np.asarray(per_rep[name], dtype=np.float64)
        valid = values.size
        if valid == 0:
            scores.append(
                CoverageScore(
                    method=name, target="quantile", n=int(n), repetitions=int(repetitions),
                    declared=float(proportion), coverage=float("nan"),
                    coverage_se=float("nan"), tolerance_rate=float("nan"),
                    tolerance_rate_se=float("nan"), mean_upper=float("nan"),
                    mean_width=float("nan"), failures=int(failures[name]), unit=unit,
                    distribution_free=bool(flags.get(name, False)),
                )
            )
            continue
        coverage = float(np.mean(values))
        spread = float(np.std(values, ddof=1)) if valid > 1 else 0.0
        # Equation (27): the repetition term and the fixed-pool term, added in
        # quadrature because they are independent sources of error.
        pool_variance = float(proportion) * (1.0 - float(proportion)) / pool_gaps.size
        standard_error = (
            float(np.sqrt(spread**2 / valid + pool_variance)) if valid > 1 else float("nan")
        )
        tolerance = float(np.mean(values >= proportion))
        scores.append(
            CoverageScore(
                method=name,
                target="quantile",
                n=int(n),
                repetitions=int(repetitions),
                declared=float(proportion),
                coverage=coverage,
                coverage_se=standard_error,
                tolerance_rate=tolerance,
                tolerance_rate_se=_binomial_se(tolerance, valid),
                mean_upper=float(np.mean(widths[name])),
                mean_width=float(np.mean(widths[name])),
                failures=int(failures[name]),
                unit=unit,
                distribution_free=bool(flags.get(name, False)),
            )
        )
    return scores
