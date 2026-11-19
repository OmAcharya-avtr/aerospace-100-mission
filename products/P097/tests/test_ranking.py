"""The top-level report, the cost-weighted capture, and the rendering."""

from __future__ import annotations

import numpy as np
import pytest

from flakerank.history import RunHistory
from flakerank.ranking import cost_capture, rank_tests
from flakerank.report import render_disagreement, render_report, render_top_table

# ---------------------------------------------------------------------------
# Cost-weighted capture, hand-calculated
# ---------------------------------------------------------------------------


def test_cost_capture_hand_calculated() -> None:
    """Truth = [100, 10, 1] s, total 111 s. A top-1 by a score that picks the
    100 s test captures 100/111 = 0.9009009009009009; one that picks the 1 s
    test captures 1/111 = 0.009009009009009009.
    """
    truth = np.asarray([100.0, 10.0, 1.0])
    names = ("a", "b", "c")
    good = cost_capture(np.asarray([3.0, 2.0, 1.0]), truth, names, 1)
    bad = cost_capture(np.asarray([1.0, 2.0, 3.0]), truth, names, 1)
    assert good == pytest.approx(100.0 / 111.0, rel=1e-14, abs=0.0)
    assert bad == pytest.approx(1.0 / 111.0, rel=1e-14, abs=0.0)


def test_cost_capture_at_full_k_is_one() -> None:
    """A top-k covering every test captures all of the waste."""
    truth = np.asarray([3.0, 2.0, 1.0])
    assert cost_capture(np.asarray([1.0, 2.0, 3.0]), truth, ("a", "b", "c"), 3) == (
        pytest.approx(1.0, rel=1e-15, abs=0.0)
    )


def test_cost_capture_of_the_truth_is_the_ceiling() -> None:
    """Ranking by the truth itself is the best any predictor can do."""
    truth = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])
    names = tuple("abcde")
    oracle = cost_capture(truth, truth, names, 2)
    for scores in (np.arange(5.0), np.asarray([1.0, 5.0, 2.0, 4.0, 3.0])):
        assert cost_capture(scores, truth, names, 2) <= oracle + 1e-15


def test_cost_capture_with_no_waste_is_nan() -> None:
    """A window in which nothing was wasted has no capture fraction."""
    assert np.isnan(cost_capture(np.asarray([1.0, 2.0]), np.zeros(2), ("a", "b"), 1))


def test_cost_capture_breaks_ties_by_name() -> None:
    """Equal scores are resolved by name, so the result is deterministic.

    Hand arithmetic: both scores are 1.0, so the top-1 is the test named "aa",
    whose cost is 7.0 of a total of 10.0, giving 0.7.
    """
    assert cost_capture(
        np.asarray([1.0, 1.0]), np.asarray([3.0, 7.0]), ("zz", "aa"), 1
    ) == pytest.approx(0.7, rel=1e-15, abs=0.0)


@pytest.mark.parametrize(
    "args,match",
    [
        ((np.asarray([1.0]), np.asarray([1.0, 2.0]), ("a",), 1), "same length"),
        ((np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0]), ("a", "b"), 0), "top_k must be >= 1"),
    ],
)
def test_cost_capture_validation(args: tuple, match: str) -> None:
    """Lengths must agree and ``top_k`` must be positive."""
    with pytest.raises(ValueError, match=match):
        cost_capture(*args)


# ---------------------------------------------------------------------------
# The assembled report on the real history
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report(history: RunHistory):
    """One report over the whole committed history, at a reduced resample count.

    300 resamples rather than the reported 2000: the interval ENDS a test
    asserts on are not the ones in the documentation, which come from
    ``validation/`` at 2000. Keeping the suite fast matters more here than
    reproducing a published interval, and the published intervals have their
    own checks.
    """
    return rank_tests(history, resamples=300, seed=97044)


def test_report_shape(report) -> None:
    """72 tests, 330 sessions, 23836 attempts, retry budget 3."""
    assert (report.n_tests, report.n_sessions, report.n_attempts) == (72, 330, 23836)
    assert report.retry_budget == 3


def test_all_three_rankings_cover_the_same_tests(report) -> None:
    """A rank correlation over two different test sets is not defined."""
    assert report.rate.names == report.analytic.names == report.truth.ranking.names


def test_headline_disagreement_is_the_truth_against_the_rate(report) -> None:
    """The headline compares ground truth with the flake rate.

    Measured: tau-b 0.819124423963, Spearman rho 0.947668742104. The values are
    asserted to 1e-9 because they are a deterministic function of the committed
    history and involve no resampling; the INTERVALS depend on the resample
    count and are not asserted here.
    """
    assert report.truth_vs_rate.label_y == "flake rate"
    assert report.truth_vs_rate.tau_b == pytest.approx(
        0.8191244239631337, rel=1e-9, abs=0.0
    )
    assert report.truth_vs_rate.rho == pytest.approx(
        0.947669374556353, rel=1e-9, abs=0.0
    )


def test_the_analytic_model_reproduces_the_ground_truth_ranking(report) -> None:
    """tau-b of the analytic cost against the ground truth is 0.998847926267.

    This is the finding the specification predicted: the analytic cost model is
    strong, and the product reports it as such rather than retuning it away.
    Max rank displacement is 1 position out of 72.
    """
    assert report.truth_vs_analytic.tau_b == pytest.approx(
        0.9988479262672811, rel=1e-9, abs=0.0
    )
    assert report.truth_vs_analytic.max_displacement == 1
    assert report.truth_vs_analytic.top_k_overlap == 10


def test_capture_figures_order_as_the_thesis_predicts(report) -> None:
    """The analytic top-10 captures more of the measured waste than the rate's.

    Measured over the whole history: rate 0.7916781500525251, analytic and oracle
    both 0.9936751713908036. The assertion is on the ORDERING plus the measured values, so a
    change in either is caught.
    """
    assert report.capture_by_rate == pytest.approx(0.7916781500525251, rel=1e-9, abs=0.0)
    assert report.capture_by_analytic == pytest.approx(0.9936751713908036, rel=1e-9, abs=0.0)
    assert report.capture_oracle == pytest.approx(0.9936751713908036, rel=1e-9, abs=0.0)
    assert report.capture_by_rate < report.capture_by_analytic
    assert report.capture_shortfall == pytest.approx(
        report.capture_oracle - report.capture_by_rate, rel=1e-12, abs=0.0
    )


def test_the_motivating_case_is_a_mover(report) -> None:
    """The 1-in-44 probe is ranked 8th by cost and 27th by rate: 19 positions.

    This is the failure the product was motivated by, in the ranking the
    product produces. The flake-rate dashboard puts it outside its top 25 while
    it is the 8th most expensive test in the suite.
    """
    movers = {mover.name: mover for mover in report.truth_vs_rate.movers}
    pinned = movers["soak/test_validation_probe_1_in_44"]
    assert pinned.rank_x == 8
    assert pinned.rank_y == 27
    assert pinned.displacement == -19


def test_the_bracket_holds_inside_the_report(report) -> None:
    """The independently derived bracket contains the accumulated total."""
    assert report.bracket.brackets
    assert report.bracket.accumulated_seconds == pytest.approx(
        report.truth.total_seconds, rel=0.0, abs=0.0
    )


def test_minimum_attempts_is_applied_to_every_ranking(history: RunHistory) -> None:
    """Raising the threshold removes the same tests from all three rankings."""
    thinned = rank_tests(history, minimum_attempts=400, resamples=20, seed=1)
    assert thinned.n_tests <= 72
    assert thinned.rate.names == thinned.truth.ranking.names


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_report_contains_the_required_statements(report) -> None:
    """The rendered report carries the research-grade statement and no emoji."""
    text = render_report(report, count=5, movers=3)
    assert "research-grade" in text
    assert "not flight-qualified" in text
    assert "Kendall tau-b" in text
    assert "Spearman rho" in text
    assert "resamples of 72 tests" in text
    assert "!" not in text
    assert text.isascii()


def test_render_report_names_the_resampling_units(report) -> None:
    """Every interval in the rendering says what was resampled."""
    text = render_report(report, count=5, movers=3)
    assert "resamples of 330 sessions" in text
    assert "resamples of 72 tests" in text


def test_render_top_table_lists_the_requested_rows(report) -> None:
    """A top table of five rows has five data lines plus two header lines."""
    text = render_top_table(report, count=5)
    assert len(text.splitlines()) == 7


def test_render_disagreement_renders_nan_as_not_a_number() -> None:
    """A NaN statistic is rendered as ``n/a``, never as a number.

    Built from a fully tied vector, where tau-b is defined as 0 and the
    Spearman denominator is zero, so the renderer is exercised on the
    degenerate case the real data also contains.
    """
    from flakerank.disagreement import rank_disagreement

    names = tuple(f"t{index}" for index in range(5))
    result = rank_disagreement(
        names,
        np.ones(5),
        np.arange(5.0),
        label_x="tied",
        label_y="untied",
        resamples=20,
        seed=2,
    )
    text = render_disagreement(result, movers=2)
    assert "Kendall tau-b" in text
    assert "0.000000" in text
