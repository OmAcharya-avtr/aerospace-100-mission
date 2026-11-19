"""The three baselines, on the hand-checkable fixture and on the real history.

Baseline 1 is the thing being refuted and is tested as carefully as the other
two, because an under-strength baseline inflates an apparent learned win and
this mission has published a figure about 50 per cent too large for exactly
that reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from flakerank.baselines import (
    Ranking,
    accumulated_wasted_time,
    analytic_expected_cost,
    flake_rate_ranking,
)
from flakerank.geometric import expected_failed_attempts
from flakerank.history import RunHistory

# ---------------------------------------------------------------------------
# Baseline 1 -- the flake-rate ranking
# ---------------------------------------------------------------------------


def test_flake_rate_scores_on_the_tiny_history(tiny: RunHistory) -> None:
    """Test 0 has p = 1/3 and test 1 has p = 0.

    Hand arithmetic: test 0 had 1 failure in 3 attempts; test 1 had 0 in 2.
    """
    ranking = flake_rate_ranking(tiny.summaries())
    assert ranking.units == "dimensionless"
    assert ranking.scores[0] == pytest.approx(1.0 / 3.0, rel=1e-15, abs=0.0)
    assert ranking.scores[1] == 0.0
    assert ranking.order[0] == 0


def test_flake_rate_carries_no_runtime(tiny: RunHistory) -> None:
    """Doubling every duration leaves the flake-rate ranking unchanged.

    This is the defect, stated as a test: the rate is blind to runtime, so a
    history in which every test is twice as slow has an identical rate ranking
    and twice the wasted time. The equality is exact in exact arithmetic
    because the rate is a ratio of integer counts, which doubling a duration
    does not touch, and is asserted to tolerance rather than bit-exactly
    because the scores are floats.
    """
    doubled = RunHistory(
        retry_budget=tiny.retry_budget,
        test_names=tiny.test_names,
        test_suites=tiny.test_suites,
        true_failure_probability=tiny.true_failure_probability,
        true_mean_ms=tiny.true_mean_ms,
        session_days=tiny.session_days,
        attempt_session=tiny.attempt_session,
        attempt_test=tiny.attempt_test,
        attempt_number=tiny.attempt_number,
        attempt_failed=tiny.attempt_failed,
        attempt_duration_ms=tiny.attempt_duration_ms * 2,
    )
    original = flake_rate_ranking(tiny.summaries()).scores
    scaled = flake_rate_ranking(doubled.summaries()).scores
    np.testing.assert_allclose(original, scaled, rtol=1e-15, atol=0.0)
    truth_original = accumulated_wasted_time(tiny, resamples=10).total_seconds
    truth_scaled = accumulated_wasted_time(doubled, resamples=10).total_seconds
    assert truth_scaled == pytest.approx(2.0 * truth_original, rel=1e-12, abs=0.0)


def test_flake_rate_rejects_an_empty_window() -> None:
    """A ranking over an empty test set is not a ranking."""
    with pytest.raises(ValueError, match="no test has a recorded attempt"):
        flake_rate_ranking(())


# ---------------------------------------------------------------------------
# Baseline 2 -- the analytic expected re-run cost
# ---------------------------------------------------------------------------


def test_analytic_cost_on_the_tiny_history(tiny: RunHistory) -> None:
    """Test 0: S = 2, t_fail = 2.0 s, p = 1/3, m = 3.

    Hand arithmetic for E[F_3] at p = 1/3 via the recursion
    E[F_m] = p(1 + E[F_{m-1}]):
        E[F_1] = 1/3                        = 0.3333333333333333
        E[F_2] = (1/3)(1 + 1/3)   = 4/9     = 0.4444444444444444
        E[F_3] = (1/3)(1 + 4/9)   = 13/27   = 0.48148148148148145
    So the analytic cost is 2 * 2.0 * 13/27 = 52/27 = 1.925925925925926 s.
    """
    ranking = analytic_expected_cost(tiny.summaries(), tiny.retry_budget)
    assert ranking.units == "second"
    assert ranking.scores[0] == pytest.approx(52.0 / 27.0, rel=1e-13, abs=0.0)
    assert ranking.scores[1] == 0.0


def test_analytic_cost_unbounded_on_the_tiny_history(tiny: RunHistory) -> None:
    """With no retry budget: 2 * 2.0 * (1/3)/(2/3) = 2 * 2.0 * 0.5 = 2.0 s."""
    ranking = analytic_expected_cost(tiny.summaries(), None)
    assert ranking.scores[0] == pytest.approx(2.0, rel=1e-14, abs=0.0)


def test_analytic_cost_uses_the_failing_mean_not_the_overall_mean(tiny: RunHistory) -> None:
    """The failing mean is 2.0 s and the overall mean is 4/3 s; the cost uses 2.0.

    Using the overall mean would give 2 * (4/3) * (13/27) = 1.283950617...,
    which is 2/3 of the correct value. The two differ here by construction,
    which is why the fixture has a failing attempt twice as long as a passing
    one.
    """
    summary = tiny.summaries()[0]
    expected = expected_failed_attempts(summary.failure_probability, tiny.retry_budget)
    correct = summary.sessions * summary.mean_failing_seconds * expected
    wrong = summary.sessions * summary.mean_attempt_seconds * expected
    ranking = analytic_expected_cost(tiny.summaries(), tiny.retry_budget)
    assert ranking.scores[0] == pytest.approx(correct, rel=1e-13, abs=0.0)
    assert ranking.scores[0] != pytest.approx(wrong, rel=1e-6, abs=0.0)
    assert wrong / correct == pytest.approx(2.0 / 3.0, rel=1e-13, abs=0.0)


def test_analytic_cost_at_measured_p_one_with_unbounded_policy_raises() -> None:
    """A test that failed every attempt is not an intermittent failure.

    The analytic baseline refuses to score it with an unbounded retry policy,
    with the message from ``flakerank.geometric`` naming the retry budget as
    the way out, rather than returning ``inf`` and sorting it above every real
    flake.
    """
    from flakerank.history import history_from_mapping

    from _support import tiny_history

    history = history_from_mapping(
        tiny_history(
            retry_budget=2,
            attempts=[(0, 0, 1, 1, 100), (0, 0, 2, 1, 100), (0, 1, 1, 0, 50)],
        )
    )
    with pytest.raises(ValueError, match="diverges at failure_probability = 1"):
        analytic_expected_cost(history.summaries(), None)
    bounded = analytic_expected_cost(history.summaries(), 2)
    # p = 1, m = 2, S = 1, t_fail = 0.1 s: E[F_2] = 2, so cost = 0.2 s.
    assert bounded.scores[0] == pytest.approx(0.2, rel=1e-13, abs=0.0)


# ---------------------------------------------------------------------------
# Baseline 3 -- accumulated ground truth
# ---------------------------------------------------------------------------


def test_ground_truth_on_the_tiny_history(tiny: RunHistory) -> None:
    """Total wasted time is 2.0 s, all of it test 0 in session 0."""
    truth = accumulated_wasted_time(tiny, resamples=50, seed=11)
    assert truth.total_seconds == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert truth.ranking.scores[0] == pytest.approx(2.0, rel=0.0, abs=1e-12)
    assert truth.ranking.scores[1] == 0.0
    assert truth.failed_attempts == 1
    assert truth.n_sessions == 2


def test_ground_truth_declares_the_session_resampling_unit(tiny: RunHistory) -> None:
    """The unit is on the object and on every interval it carries."""
    truth = accumulated_wasted_time(tiny, resamples=50, seed=11)
    assert truth.resampling_unit == "session"
    truth.require_resampling_unit("session")
    assert truth.total_interval.resampling_unit == "session"
    for interval in truth.per_test_intervals:
        assert interval.resampling_unit == "session"
    with pytest.raises(ValueError, match="resampling"):
        truth.require_resampling_unit("attempt")


def test_ground_truth_interval_brackets_its_own_point_estimate(history: RunHistory) -> None:
    """The percentile interval of a mean contains the observed mean.

    Not guaranteed in general for a bootstrap interval of a skewed statistic,
    but for the total of a non-negative per-session sample with 330 sessions
    and 2000 resamples it is, and failing it would mean the resampler is wrong.
    """
    truth = accumulated_wasted_time(history, resamples=400, seed=5)
    assert truth.total_interval.contains(truth.total_seconds)
    assert truth.total_interval.low <= truth.total_seconds <= truth.total_interval.high


def test_ground_truth_total_equals_the_sum_of_per_test_scores(history: RunHistory) -> None:
    """Every failed attempt belongs to exactly one test.

    Compared to a tolerance that follows the CONDITIONING of the two
    summations: 72 per-test sums against one 330-session sum of the same
    1417 terms, so the admissible difference is of order
    ``n * eps * total``, not an absolute constant.
    """
    truth = accumulated_wasted_time(history, resamples=20, seed=5)
    per_test_total = float(truth.ranking.scores.sum())
    tolerance = 4.0 * (2.0**-53) * history.n_attempts * truth.total_seconds
    assert abs(per_test_total - truth.total_seconds) <= tolerance


def test_ground_truth_zero_failure_test_gets_a_degenerate_interval(
    history: RunHistory,
) -> None:
    """A test with no observed failure gets [0, 0], which is honest and misleading.

    Honest, because every resample of the recorded sessions gives zero.
    Misleading, because the test's true rate is only bounded above, not zero.
    The limitation is recorded in README.md; this test pins the behaviour so it
    cannot change silently.
    """
    truth = accumulated_wasted_time(history, resamples=100, seed=5)
    zeros = [
        index for index, score in enumerate(truth.ranking.scores) if score == 0.0
    ]
    assert zeros, "the committed history is expected to contain never-failing tests"
    for index in zeros:
        interval = truth.per_test_intervals[index]
        assert interval.low == 0.0
        assert interval.high == 0.0


def test_ground_truth_on_the_real_history(history: RunHistory) -> None:
    """253493.035 s wasted over 330 sessions from 1417 failed attempts.

    The total is the generator transcript's 253493035 ms.
    """
    truth = accumulated_wasted_time(history, resamples=50, seed=5)
    assert truth.total_seconds == pytest.approx(253493.035, rel=0.0, abs=1e-6)
    assert truth.failed_attempts == 1417
    assert truth.n_sessions == 330


# ---------------------------------------------------------------------------
# The Ranking container
# ---------------------------------------------------------------------------


def test_ranking_rejects_nan_scores() -> None:
    """A NaN score cannot be ordered."""
    with pytest.raises(ValueError, match="contains NaN"):
        Ranking(label="x", units="second", names=("a",), scores=np.asarray([float("nan")]))


def test_ranking_rejects_a_length_mismatch() -> None:
    """Names and scores must be parallel."""
    with pytest.raises(ValueError, match="names and"):
        Ranking(label="x", units="second", names=("a", "b"), scores=np.asarray([1.0]))


def test_ranking_order_breaks_ties_by_name() -> None:
    """Equal scores are ordered by name so the ranking is reproducible."""
    ranking = Ranking(
        label="x",
        units="second",
        names=("zz", "aa", "mm"),
        scores=np.asarray([1.0, 1.0, 1.0]),
    )
    assert [ranking.names[index] for index in ranking.order] == ["aa", "mm", "zz"]


def test_ranking_top_validates_count() -> None:
    """A top-0 list is not a list."""
    ranking = Ranking(label="x", units="second", names=("a",), scores=np.asarray([1.0]))
    with pytest.raises(ValueError, match="count must be >= 1"):
        ranking.top(0)
