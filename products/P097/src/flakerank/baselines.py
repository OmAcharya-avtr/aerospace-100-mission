"""The three baselines, implemented and validated before the learned component.

The mission's build guide requires the classical baselines first, because an
under-strength baseline inflates an apparent learned-model win: one earlier
product of this mission measured a learned margin about 50 per cent too large
for exactly that reason, and another found its specified baseline was the
weakest of three candidates. So all three here are implemented at full
strength, and baseline 1 -- the one the specification names as *the thing being
refuted* -- gets the same care as the other two.

BASELINE 1 -- RANK BY FLAKE RATE
--------------------------------

``score = failed_attempts / attempts``, the per-attempt flake rate, measured
over the window. This is what a CI flaky-test dashboard reports, and it is
implemented deliberately and at full strength because it is the thing being
refuted. Full strength means: the rate is a pooled maximum-likelihood estimate
over every recorded attempt including retries, which is the best point estimate
of ``p`` under the model's own independence assumption, and ties are broken by
name rather than by history order so the ranking is reproducible.

What is wrong with it is not the estimator; it is the *question*. The rate
carries no runtime, so a 1-in-44 failure in a 600-second test scores 0.0227 and
a 1-in-2 failure in a 0.1-second test scores 0.5, and the dashboard puts the
second 22 times higher. By expected wasted time the first costs
``600 / 43 = 13.95`` seconds per invocation and the second costs ``0.1``, so
the true ordering is the other way round by a factor of 140. That is a factor
of about 3000 between the two rankings' verdicts on one pair, and it is not an
edge case: a slow integration test that fails rarely is the common shape of an
expensive flake.

BASELINE 2 -- THE ANALYTIC EXPECTED RE-RUN COST
-----------------------------------------------

``score = S * t_fail * E[F_m(p)]`` with ``S`` the number of recorded
invocations, ``t_fail`` the measured mean duration of a failing attempt,
``p = failed_attempts / attempts``, ``m`` the retry budget, and ``E[F_m]`` the
closed-form expected failed-attempt count of :mod:`flakerank.geometric`
equation (5). The derivation, the assumptions and the validity limits are in
that module's docstring and are not repeated here.

Two choices inside it are worth stating because they are where the model can go
wrong:

* ``t_fail`` is the mean duration of the **failing** attempts, not of all
  attempts, because a failing attempt that exits early costs less than a
  passing one. For a test with no observed failure the mean over all attempts
  is substituted (see
  :attr:`~flakerank.history.TestSummary.cost_runtime_seconds`); such a test has
  ``p = 0`` so its analytic cost is zero either way.
* ``S`` is the number of distinct sessions in which the test was invoked, not
  the number of attempts. ``E[F_m]`` is already per invocation.

BASELINE 3 -- ACCUMULATED GROUND TRUTH
--------------------------------------

``score = sum of the recorded durations of the test's FAILED attempts`` over
the window, which is the wasted time that actually happened. No model, no
distributional assumption: an accumulation over a recorded history.

This is the ground truth the other two are scored against, and it carries a
percentile bootstrap interval whose **resampling unit is the session** -- the
unit CI repeats and the unit over which the history is (taken to be)
exchangeable. The unit is declared on every returned interval and asserted
before the result is returned, because resampling attempts instead would give
an interval too narrow by the within-session correlation and nothing about the
number would say so.

Being ground truth does not make it the right thing to *rank* by in practice,
and the product says so rather than pretending otherwise: accumulated wasted
time is backward-looking, it is zero for any test that has not failed yet
however slow and however fragile, and over a short window it is dominated by
sampling noise, which is precisely what the interval exposes. Baseline 2 exists
because it extrapolates; baseline 3 exists because it is true.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometric import expected_failed_attempts
from .history import RunHistory, TestSummary
from .resample import BootstrapInterval, bootstrap_mean_interval, bootstrap_statistic_interval

__all__ = [
    "GroundTruthCost",
    "Ranking",
    "accumulated_wasted_time",
    "analytic_expected_cost",
    "flake_rate_ranking",
]

#: The resampling unit of every ground-truth interval in this module.
GROUND_TRUTH_RESAMPLING_UNIT = "session"


@dataclass(frozen=True)
class Ranking:
    """Tests scored and ordered by one criterion.

    Attributes:
        label: short name of the criterion, e.g. ``"flake rate"``.
        units: units of ``scores``, e.g. ``"second"`` or ``"dimensionless"``.
        names: test identifiers, in the order the scores are given.
        scores: one score per test; higher is ranked nearer position 1.
        description: one sentence on what the score means.
    """

    label: str
    units: str
    names: tuple[str, ...]
    scores: np.ndarray
    description: str = ""

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=np.float64)
        if scores.ndim != 1:
            raise ValueError(f"scores must be one-dimensional, got shape {scores.shape}")
        if scores.size != len(self.names):
            raise ValueError(
                f"ranking {self.label!r} has {len(self.names)} names and {scores.size} scores"
            )
        if np.isnan(scores).any():
            raise ValueError(
                f"ranking {self.label!r} contains NaN. A NaN score cannot be ordered, and "
                "sorting it to one end would make the ranking depend on an implementation "
                "detail of the sort."
            )
        object.__setattr__(self, "scores", scores)

    @property
    def order(self) -> tuple[int, ...]:
        """Indices in descending score order, ties broken by name."""
        return tuple(
            sorted(range(self.scores.size), key=lambda index: (-self.scores[index],
                                                               self.names[index]))
        )

    def top(self, count: int) -> tuple[tuple[str, float], ...]:
        """The ``count`` highest-scoring tests as ``(name, score)`` pairs."""
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        return tuple((self.names[index], float(self.scores[index])) for index in self.order[:count])


def _summary_arrays(summaries: tuple[TestSummary, ...]) -> tuple[str, ...]:
    """Names of the summarised tests, with an empty-input guard."""
    if not summaries:
        raise ValueError(
            "no test has a recorded attempt in this window; a ranking over an empty "
            "test set is not a ranking"
        )
    return tuple(summary.name for summary in summaries)


def flake_rate_ranking(summaries: tuple[TestSummary, ...]) -> Ranking:
    """Baseline 1: rank by the measured per-attempt flake rate.

    Args:
        summaries: per-test measurements from
            :meth:`flakerank.history.RunHistory.summaries`.

    Returns:
        A :class:`Ranking` whose scores are ``failed_attempts / attempts``,
        dimensionless.

    Raises:
        ValueError: for an empty summary tuple.
    """
    names = _summary_arrays(summaries)
    scores = np.asarray(
        [summary.failure_probability for summary in summaries], dtype=np.float64
    )
    return Ranking(
        label="flake rate",
        units="dimensionless",
        names=names,
        scores=scores,
        description=(
            "pooled per-attempt failure probability over the window; what a CI "
            "flaky-test dashboard reports, and the thing this product refutes"
        ),
    )


def analytic_expected_cost(
    summaries: tuple[TestSummary, ...],
    retry_budget: int | None,
) -> Ranking:
    """Baseline 2: rank by the analytic expected wasted time.

    ``S * t_fail * E[F_m(p)]`` per the module docstring, in seconds over the
    window.

    Args:
        summaries: per-test measurements.
        retry_budget: maximum attempts per invocation, or ``None`` for an
            unbounded retry policy. With ``None`` a test whose measured ``p``
            is exactly 1 would diverge; such a test is scored with the
            unbounded form only if ``p < 1``, and a measured ``p == 1`` raises
            from :func:`flakerank.geometric.expected_failed_attempts` with a
            message naming the retry budget as the way out. That is deliberate:
            a test that failed every recorded attempt is not an intermittent
            failure and should not be ranked as one.

    Returns:
        A :class:`Ranking` in seconds.

    Raises:
        ValueError: for an empty summary tuple, or when a measured ``p`` of 1
            meets an unbounded retry policy.
    """
    names = _summary_arrays(summaries)
    scores = np.empty(len(summaries), dtype=np.float64)
    for index, summary in enumerate(summaries):
        expected = expected_failed_attempts(summary.failure_probability, retry_budget)
        scores[index] = summary.sessions * summary.cost_runtime_seconds * expected
    return Ranking(
        label="analytic expected re-run cost",
        units="second",
        names=names,
        scores=scores,
        description=(
            "S * t_fail * E[F_m(p)] with the geometric expected failed-attempt count "
            "in closed form; see flakerank.geometric equation (5)"
        ),
    )


@dataclass(frozen=True)
class GroundTruthCost:
    """Baseline 3: accumulated wasted time with session-bootstrap intervals.

    Attributes:
        ranking: the per-test accumulated wasted seconds.
        total_seconds: wasted seconds summed over all tests in the window.
        total_interval: bootstrap interval on ``total_seconds``, resampling
            sessions.
        per_test_intervals: one interval per test, in ``ranking.names`` order,
            each from the same session resample.
        resampling_unit: the literal ``"session"``.
        n_sessions: sessions in the window.
        failed_attempts: failed attempts in the window.
    """

    ranking: Ranking
    total_seconds: float
    total_interval: BootstrapInterval
    per_test_intervals: tuple[BootstrapInterval, ...]
    resampling_unit: str
    n_sessions: int
    failed_attempts: int

    def require_resampling_unit(self, expected: str) -> GroundTruthCost:
        """Assert the declared resampling unit, on this object and every interval.

        Args:
            expected: the unit the caller believes was resampled.

        Returns:
            ``self``.

        Raises:
            ValueError: on any disagreement, from this object or from any
                interval it carries.
        """
        if self.resampling_unit != expected:
            raise ValueError(
                f"ground truth was accumulated by resampling {self.resampling_unit!r}, "
                f"not {expected!r}"
            )
        self.total_interval.require_resampling_unit(expected)
        for interval in self.per_test_intervals:
            interval.require_resampling_unit(expected)
        return self


def accumulated_wasted_time(
    history: RunHistory,
    *,
    minimum_attempts: int = 1,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 97044,
) -> GroundTruthCost:
    """Baseline 3: accumulate the recorded wasted time, with session bootstrap.

    Args:
        history: the recorded history or a window of it.
        minimum_attempts: tests with fewer attempts in the window are omitted.
        resamples: bootstrap resamples.
        confidence: nominal coverage of every reported interval.
        seed: seed for ``numpy.random.default_rng``. The total uses ``seed``
            and the per-test intervals use ``seed + 1``, so a reader comparing
            the two is not comparing the same draws.

    Returns:
        A :class:`GroundTruthCost` whose resampling unit is ``"session"``,
        asserted before return.

    Raises:
        ValueError: for an empty window.
    """
    summaries = history.summaries(minimum_attempts=minimum_attempts)
    names = _summary_arrays(summaries)
    scores = np.asarray([summary.wasted_seconds for summary in summaries], dtype=np.float64)
    ranking = Ranking(
        label="accumulated wasted time (ground truth)",
        units="second",
        names=names,
        scores=scores,
        description=(
            "summed recorded duration of the test's failed attempts over the window; "
            "no model and no distributional assumption"
        ),
    )

    per_session = history.session_wasted_seconds()
    total_interval = bootstrap_mean_interval(
        per_session,
        resampling_unit=GROUND_TRUTH_RESAMPLING_UNIT,
        total=True,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        value_units="second",
    ).require_resampling_unit(GROUND_TRUTH_RESAMPLING_UNIT)

    # Per-test intervals come from resampling the SAME unit -- whole sessions --
    # so that every test's interval is built from one consistent resample of the
    # history rather than from a per-test resample that would ignore the fact
    # that tests share sessions.
    table = history.session_test_wasted_seconds()
    name_to_index = {name: index for index, name in enumerate(history.test_names)}
    selected = np.asarray([name_to_index[name] for name in names], dtype=np.int64)
    per_test = table[:, selected]
    n_sessions = history.n_sessions

    per_test_intervals: list[BootstrapInterval] = []
    for column in range(per_test.shape[1]):
        values = per_test[:, column]

        def statistic(resampled: np.ndarray) -> float:
            return float(resampled.mean()) * n_sessions

        per_test_intervals.append(
            bootstrap_statistic_interval(
                values,
                statistic,
                resampling_unit=GROUND_TRUTH_RESAMPLING_UNIT,
                resamples=resamples,
                confidence=confidence,
                seed=seed + 1,
                value_units="second",
            ).require_resampling_unit(GROUND_TRUTH_RESAMPLING_UNIT)
        )

    return GroundTruthCost(
        ranking=ranking,
        total_seconds=float(per_session.sum()),
        total_interval=total_interval,
        per_test_intervals=tuple(per_test_intervals),
        resampling_unit=GROUND_TRUTH_RESAMPLING_UNIT,
        n_sessions=history.n_sessions,
        failed_attempts=int(np.count_nonzero(history.attempt_failed)),
    ).require_resampling_unit(GROUND_TRUTH_RESAMPLING_UNIT)
