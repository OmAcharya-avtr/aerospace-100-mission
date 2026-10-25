"""The two benchmark tasks, their ordering, and the verdict they produce.

Both run on the 48-object subset fixture: the full-table numbers are produced
once by ``validation/validate_predictors.py`` and no figure measured here
appears in any document.
"""

from __future__ import annotations

import numpy as np
import pytest

from orbitdrift import run_benchmark
from orbitdrift.benchmark import (
    EXTRA_CURVE_FEATURES,
    calibrated_rows,
    not_under_covering_rows,
    run_extrapolation_benchmark,
)
from orbitdrift.errors import InsufficientDataError


@pytest.fixture(scope="module")
def cold(small_table):
    return run_benchmark(small_table, nominal_coverage=0.90)


@pytest.fixture(scope="module")
def extrapolation(small_table):
    return run_extrapolation_benchmark(small_table, nominal_coverage=0.90)


def test_cold_benchmark_puts_every_analytic_row_before_every_learned_row(cold) -> None:
    """The ordering is the experimental protocol, enforced in code.

    The analytic baselines are fitted and measured before the learned model is
    constructed. A learned model whose hyperparameters were chosen after
    seeing the baseline's test score would be a different and much weaker
    experiment.
    """
    kinds = [row.kind for row in cold.rows]
    assert kinds.count("analytic") == 11
    assert kinds.count("learned") == 2
    assert kinds.index("learned") == kinds.count("analytic")


def test_cold_benchmark_split_is_by_object_and_disjoint(cold) -> None:
    assert cold.n_train_objects + cold.n_test_objects == 48
    assert cold.n_train_rows + cold.n_test_rows == 48 * 28
    assert cold.n_test_objects > 5


def test_every_row_has_finite_metrics_and_a_coverage_interval(cold) -> None:
    for row in cold.rows:
        assert 0.0 <= row.metrics.coverage <= 1.0
        low, high = row.metrics.coverage_ci
        assert low <= row.metrics.coverage <= high
        assert np.isfinite(row.metrics.interval_score)
        assert np.isfinite(row.median_absolute_log_error)
        assert row.metrics.nominal == 0.90


def test_rule_of_thumb_is_present_and_badly_calibrated(cold) -> None:
    """The thing the product is positioned against must be measured, not asserted.

    A fixed kilometre-per-day rule with a factor-of-three band is what an
    operator uses in the absence of anything better, and its measured coverage
    is the honest statement of what that costs. It is in the table so the
    comparison is against practice and not against a straw man nobody uses.
    """
    row = cold.row("rule_of_thumb")
    assert row.extra["rate_km_per_day"] == 1.0
    assert row.metrics.coverage < 0.75


def test_coverage_tables_are_present_for_every_predictor_and_sum_to_the_test_rows(
    cold,
) -> None:
    for row in cold.rows:
        for table in (cold.coverage_by_altitude[row.name], cold.coverage_by_age[row.name]):
            assert sum(cell.n for cell in table) == cold.n_test_rows
            assert [cell.stratum for cell in table] == sorted(
                cell.stratum for cell in table
            )


def test_calibrated_and_safe_subsets_are_subsets(cold) -> None:
    kept = calibrated_rows(cold.rows, cold.nominal_coverage)
    safe = not_under_covering_rows(cold.rows, cold.nominal_coverage)
    names = {row.name for row in cold.rows}
    assert {row.name for row in kept} <= names
    assert {row.name for row in safe} <= names
    for row in safe:
        assert row.metrics.coverage >= cold.nominal_coverage
    for row in kept:
        low, high = row.metrics.coverage_ci
        assert low <= cold.nominal_coverage <= high
    assert cold.calibrated_rows() == kept
    assert cold.not_under_covering_rows() == safe


def test_verdict_names_the_task_and_mentions_both_families(cold) -> None:
    text = cold.verdict_line()
    assert "cold-prediction task" in text
    assert "point prediction" in text
    assert "interval score" in text


def test_row_lookup_raises_on_an_unknown_name(cold) -> None:
    with pytest.raises(KeyError, match="no predictor named"):
        cold.row("not_a_predictor")


def test_extrapolation_benchmark_keeps_the_same_ordering(extrapolation) -> None:
    kinds = [row.kind for row in extrapolation.rows]
    assert kinds.count("analytic") == 8
    assert kinds.count("learned") == 4
    assert kinds.index("learned") == kinds.count("analytic")


def test_extrapolation_measures_only_rows_beyond_the_fitting_horizon(
    extrapolation, small_table
) -> None:
    ages = small_table.column("epoch_age_days")
    beyond = int(np.count_nonzero(ages > extrapolation.fit_horizon_days))
    inside = int(np.count_nonzero(ages <= extrapolation.fit_horizon_days))
    assert extrapolation.n_target_rows == beyond
    assert extrapolation.n_fit_rows == inside
    assert extrapolation.n_test_target_rows < beyond
    assert extrapolation.n_train_objects + extrapolation.n_test_objects == 48


def test_extrapolation_gives_the_learned_model_the_baseline_s_own_fit_as_features(
    extrapolation,
) -> None:
    """The learned model is not handicapped by being denied the curve.

    Both families see the object's own inside-horizon fit: the analytic one
    *is* that fit extrapolated, and the learned one gets its amplitude,
    exponent and residual spread as three extra columns. The pair of learned
    rows with and without those columns measures what the curve is worth,
    separately from what the learning is worth.
    """
    assert EXTRA_CURVE_FEATURES == (
        "own_log_amplitude_km",
        "own_exponent",
        "own_log_residual_std",
    )
    with_curve = extrapolation.row("learned_with_curve_features")
    without = extrapolation.row("learned_features_only")
    assert with_curve.median_absolute_log_error < without.median_absolute_log_error


def test_extrapolation_form_counts_cover_every_object_with_a_fit(extrapolation) -> None:
    assert sum(extrapolation.form_counts.values()) == 48
    assert set(extrapolation.form_counts).issubset(
        {"linear", "quadratic", "power", "drag_exponential"}
    )


def test_extrapolation_nominal_intervals_under_cover_badly(extrapolation) -> None:
    """A fit's own in-sample residual spread says nothing about extrapolation.

    This is the single most important measured fact in the product: the
    interval you get by trusting a fit's own residuals, which is what a
    practitioner would naturally do, covers far less than it claims once you
    leave the range the fit was made on. The conformal variants exist because
    of it.
    """
    nominal = extrapolation.row("own_curve_power")
    conformal = extrapolation.row("own_curve_power_conformal")
    assert nominal.metrics.coverage < 0.60
    assert conformal.metrics.coverage > nominal.metrics.coverage + 0.25


def test_extrapolation_verdict_mentions_the_safe_subset(extrapolation) -> None:
    text = extrapolation.verdict_line()
    assert "extrapolation task" in text
    assert "under-cover" in text


def test_extrapolation_refuses_a_horizon_that_leaves_no_rows(small_table) -> None:
    with pytest.raises(InsufficientDataError, match="rows beyond it"):
        run_extrapolation_benchmark(small_table, fit_horizon_days=100.0)


def test_best_by_interval_score_raises_for_an_unknown_kind(cold) -> None:
    with pytest.raises(KeyError, match="no rows of kind"):
        cold.best_by_interval_score("imaginary")
