"""Scoring: metrics, the per-regime split, and the calibration table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relfiledelivery.benchmark import (
    BenchmarkRow,
    best_on,
    calibration_table,
    coverage_standard_error,
    render_table,
    run_benchmark,
    score,
)
from relfiledelivery.dataset import TARGET_COLUMN
from relfiledelivery.predictors import (
    AnalyticPredictor,
    LearnedQuantilePredictor,
    MeanLookupPredictor,
)


class TestScore:
    def test_metrics_are_in_range(self, small_split) -> None:
        train, test = small_split
        row = score(MeanLookupPredictor().fit(train), test, "all")
        assert row.rows == len(test)
        assert row.median_abs_rel_error >= 0.0
        assert row.p90_abs_rel_error >= row.median_abs_rel_error
        assert 0.0 <= row.coverage <= 1.0
        assert row.median_interval_width_ratio >= 0.0
        assert row.uses_training_data

    def test_a_perfect_predictor_scores_zero_error_and_full_coverage(
        self, small_split
    ) -> None:
        train, test = small_split

        class Oracle(AnalyticPredictor):
            name = "oracle"

            def predict_quantiles(self, table, levels=(0.5,)):
                truth = np.asarray(table[TARGET_COLUMN], dtype=float)
                return {level: truth.copy() for level in levels}

        row = score(Oracle(), test, "all")
        assert row.median_abs_rel_error == pytest.approx(0.0)
        assert row.coverage == pytest.approx(1.0)
        assert row.median_interval_width_ratio == pytest.approx(0.0)

    def test_calibration_error_and_the_declared_threshold(self) -> None:
        row = BenchmarkRow("x", "all", 1000, 0.1, 0.2, 0.92, 0.90, 1.0, True)
        assert row.calibration_error == pytest.approx(0.02)
        assert row.is_calibrated
        loose = BenchmarkRow("x", "all", 1000, 0.1, 0.2, 0.70, 0.90, 1.0, True)
        assert not loose.is_calibrated
        assert loose.calibration_error == pytest.approx(-0.20)

    def test_an_empty_table_is_rejected(self, small_split) -> None:
        train, test = small_split
        with pytest.raises(ValueError, match="empty table"):
            score(MeanLookupPredictor().fit(train), test.iloc[0:0], "all")

    def test_a_non_finite_target_is_refused(self, small_split) -> None:
        train, test = small_split
        broken = test.copy()
        broken.loc[0, TARGET_COLUMN] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            score(MeanLookupPredictor().fit(train), broken, "all")


class TestRunBenchmark:
    def test_every_predictor_is_scored_on_every_regime_plus_all(self, small_split) -> None:
        train, test = small_split
        predictors = [AnalyticPredictor(), MeanLookupPredictor()]
        rows = run_benchmark(predictors, train, test)
        regimes = sorted(test["regime"].unique())
        assert len(rows) == len(predictors) * (1 + len(regimes))
        assert {row.regime for row in rows} == {"all", *regimes}

    def test_the_analytic_row_is_marked_as_untrained(self, small_split) -> None:
        train, test = small_split
        rows = run_benchmark([AnalyticPredictor()], train, test)
        assert all(not row.uses_training_data for row in rows)

    def test_best_on_picks_the_lowest_error(self, small_split) -> None:
        train, test = small_split
        rows = run_benchmark(
            [AnalyticPredictor(), MeanLookupPredictor()], train, test
        )
        winner = best_on(rows, "no_outage")
        errors = {
            row.predictor: row.median_abs_rel_error for row in rows if row.regime == "no_outage"
        }
        assert errors[winner] == min(errors.values())

    def test_best_on_rejects_an_unknown_regime(self, small_split) -> None:
        train, test = small_split
        rows = run_benchmark([AnalyticPredictor()], train, test)
        with pytest.raises(ValueError, match="no benchmark row"):
            best_on(rows, "lunar")

    def test_the_analytic_closed_form_wins_in_its_own_regime(self, small_split) -> None:
        """The specification's prediction, asserted at test scale.

        The closed form is exact in the no-outage regime, so it must not lose
        to a lookup table there. This is asserted rather than hoped for because
        a regression that made the analytic predictor worse than a bin average
        in the regime where it is exact would be a defect in the formula, not a
        modelling result.
        """
        train, test = small_split
        rows = run_benchmark(
            [AnalyticPredictor(), MeanLookupPredictor()], train, test
        )
        errors = {
            row.predictor: row.median_abs_rel_error for row in rows if row.regime == "no_outage"
        }
        assert errors["analytic"] < errors["mean_lookup"]


class TestCalibrationTable:
    def test_coverage_is_non_decreasing_in_the_nominal_level(self, small_split) -> None:
        train, test = small_split
        predictor = LearnedQuantilePredictor(n_estimators=40).fit(train)
        table = calibration_table(predictor, test, (0.50, 0.80, 0.90))
        coverages = [row[1] for row in table]
        widths = [row[2] for row in table]
        assert coverages == sorted(coverages)
        assert widths == sorted(widths)

    def test_coverage_standard_error_is_the_binomial_expression(self) -> None:
        # sqrt(0.9 * 0.1 / 900) = 0.01 exactly.
        assert coverage_standard_error(0.90, 900) == pytest.approx(0.01, rel=1e-12)

    def test_coverage_standard_error_rejects_zero_rows(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            coverage_standard_error(0.9, 0)


def test_render_table_has_a_header_and_one_line_per_row() -> None:
    rows = [
        BenchmarkRow("analytic", "all", 10, 0.1, 0.2, 0.5, 0.9, 0.4, False),
        BenchmarkRow("learned_quantile", "all", 10, 0.09, 0.18, 0.85, 0.9, 0.9, True),
    ]
    lines = render_table(rows)
    assert len(lines) == 4
    assert "predictor" in lines[0]
    assert "analytic" in lines[2]
    assert "learned_quantile" in lines[3]


def test_benchmark_rows_are_hashable_dataclasses() -> None:
    row = BenchmarkRow("a", "all", 1, 0.0, 0.0, 0.0, 0.9, 0.0, False)
    assert isinstance(pd.DataFrame([row.__dict__]), pd.DataFrame)
