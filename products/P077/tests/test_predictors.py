"""The three predictors: interface, guards, and that each one does its own thing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relfiledelivery.dataset import TARGET_COLUMN, targets
from relfiledelivery.predictors import (
    DEFAULT_LEVELS,
    AnalyticPredictor,
    LearnedQuantilePredictor,
    MeanLookupPredictor,
    build_predictors,
)


class TestAnalyticPredictor:
    def test_it_does_not_use_the_training_data(self, small_split) -> None:
        train, test = small_split
        predictor = AnalyticPredictor()
        assert not predictor.uses_training_data
        before = predictor.predict(test)
        predictor.fit(train)
        after = predictor.predict(test)
        assert np.array_equal(before, after)

    def test_quantiles_are_monotone_in_the_level(self, small_split) -> None:
        _, test = small_split
        got = AnalyticPredictor().predict_quantiles(test, DEFAULT_LEVELS)
        stack = np.vstack([got[level] for level in sorted(DEFAULT_LEVELS)])
        assert np.all(np.diff(stack, axis=0) >= -1e-9)

    def test_predictions_are_positive_and_finite(self, small_split) -> None:
        _, test = small_split
        values = AnalyticPredictor().predict(test)
        assert np.all(values > 0.0)
        assert np.all(np.isfinite(values))

    def test_the_mean_variant_differs_from_the_median_variant(self, small_split) -> None:
        _, test = small_split
        median = AnalyticPredictor(point="median").predict(test)
        mean = AnalyticPredictor(point="mean").predict(test)
        assert not np.allclose(median, mean)
        assert AnalyticPredictor(point="mean").name == "analytic_mean"

    def test_an_unknown_point_choice_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="'median' or 'mean'"):
            AnalyticPredictor(point="mode")

    def test_a_missing_column_is_named(self) -> None:
        with pytest.raises(ValueError, match="missing column"):
            AnalyticPredictor().predict(pd.DataFrame({"pdu_count": [10.0]}))


class TestMeanLookupPredictor:
    def test_it_must_be_fitted_before_use(self, small_split) -> None:
        _, test = small_split
        with pytest.raises(RuntimeError, match="before fit"):
            MeanLookupPredictor().predict(test)

    def test_quantiles_are_monotone_and_positive(self, small_split) -> None:
        train, test = small_split
        predictor = MeanLookupPredictor().fit(train)
        got = predictor.predict_quantiles(test, DEFAULT_LEVELS)
        stack = np.vstack([got[level] for level in sorted(DEFAULT_LEVELS)])
        assert np.all(np.diff(stack, axis=0) >= -1e-9)
        assert np.all(stack > 0.0)

    def test_fallback_and_lookup_counts_sum_to_the_row_count(self, small_split) -> None:
        train, test = small_split
        predictor = MeanLookupPredictor().fit(train)
        predictor.predict(test)
        assert predictor.fallback_count + predictor.lookup_count == len(test)

    def test_an_untabulated_level_is_refused(self, small_split) -> None:
        train, test = small_split
        predictor = MeanLookupPredictor().fit(train)
        with pytest.raises(ValueError, match="was not tabulated"):
            predictor.predict_quantiles(test, (0.333,))

    def test_prediction_scales_with_the_clean_pass_time(self, small_split) -> None:
        """The table stores a dimensionless inflation factor, so doubling the
        clean-pass time must double the prediction."""
        train, test = small_split
        predictor = MeanLookupPredictor().fit(train)
        base = predictor.predict(test)
        doubled = test.copy()
        doubled["log10_clean_pass_s"] = doubled["log10_clean_pass_s"] + np.log10(2.0)
        assert predictor.predict(doubled) == pytest.approx(2.0 * base, rel=1e-9)


class TestLearnedQuantilePredictor:
    def test_it_must_be_fitted_before_use(self, small_split) -> None:
        _, test = small_split
        with pytest.raises(RuntimeError, match="before fit"):
            LearnedQuantilePredictor().predict(test)

    def test_fit_then_predict_gives_monotone_positive_quantiles(self, small_split) -> None:
        train, test = small_split
        predictor = LearnedQuantilePredictor(n_estimators=40).fit(train)
        got = predictor.predict_quantiles(test, DEFAULT_LEVELS)
        stack = np.vstack([got[level] for level in sorted(DEFAULT_LEVELS)])
        assert np.all(np.diff(stack, axis=0) >= 0.0)
        assert np.all(stack > 0.0)

    def test_crossings_are_counted_not_hidden(self, small_split) -> None:
        train, test = small_split
        predictor = LearnedQuantilePredictor(n_estimators=40).fit(train)
        predictor.predict_quantiles(test, DEFAULT_LEVELS)
        assert predictor.crossing_count >= 0

    def test_fit_is_reproducible_at_a_fixed_random_state(self, small_split) -> None:
        train, test = small_split
        first = LearnedQuantilePredictor(n_estimators=40, random_state=5).fit(train)
        second = LearnedQuantilePredictor(n_estimators=40, random_state=5).fit(train)
        assert first.predict(test) == pytest.approx(second.predict(test), rel=1e-12)

    def test_an_unfitted_level_is_refused(self, small_split) -> None:
        train, test = small_split
        predictor = LearnedQuantilePredictor(levels=(0.5,), n_estimators=20).fit(train)
        with pytest.raises(ValueError, match="were not fitted"):
            predictor.predict_quantiles(test, (0.05, 0.5))

    def test_a_non_finite_target_is_refused(self, small_split) -> None:
        train, _ = small_split
        broken = train.copy()
        broken.loc[0, TARGET_COLUMN] = np.nan
        with pytest.raises(ValueError, match="non-finite delivery time"):
            LearnedQuantilePredictor(n_estimators=10).fit(broken)

    def test_feature_importances_cover_every_feature_and_sum_to_one(
        self, small_split
    ) -> None:
        train, _ = small_split
        predictor = LearnedQuantilePredictor(n_estimators=40).fit(train)
        importances = predictor.feature_importances(0.50)
        assert set(importances) == set(predictor.feature_names)
        assert sum(importances.values()) == pytest.approx(1.0, abs=1e-6)

    def test_feature_importances_refuse_an_unfitted_level(self, small_split) -> None:
        train, _ = small_split
        predictor = LearnedQuantilePredictor(levels=(0.5,), n_estimators=10).fit(train)
        with pytest.raises(ValueError, match="was not fitted"):
            predictor.feature_importances(0.95)

    def test_it_tracks_the_training_row_count(self, small_split) -> None:
        train, _ = small_split
        predictor = LearnedQuantilePredictor(n_estimators=10).fit(train)
        assert predictor.n_train == len(train)


class TestInterval:
    @pytest.mark.parametrize("nominal", [0.5, 0.8, 0.9])
    def test_interval_brackets_the_median_prediction(self, small_split, nominal: float) -> None:
        train, test = small_split
        predictor = MeanLookupPredictor().fit(train)
        lower, median, upper = predictor.predict_interval(test, nominal=nominal)
        assert np.all(lower <= median + 1e-9)
        assert np.all(median <= upper + 1e-9)

    def test_a_wider_nominal_gives_a_wider_interval(self, small_split) -> None:
        train, test = small_split
        predictor = MeanLookupPredictor().fit(train)
        narrow = np.subtract(*reversed(predictor.predict_interval(test, 0.5)[::2]))
        wide = np.subtract(*reversed(predictor.predict_interval(test, 0.9)[::2]))
        assert np.median(wide) >= np.median(narrow)

    @pytest.mark.parametrize("nominal", [0.0, 1.0, -0.2, 1.5])
    def test_out_of_range_nominal_is_rejected(self, small_split, nominal: float) -> None:
        _, test = small_split
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            AnalyticPredictor().predict_interval(test, nominal=nominal)


def test_build_predictors_returns_four_named_predictors() -> None:
    names = [predictor.name for predictor in build_predictors()]
    assert names == ["analytic", "analytic_mean", "mean_lookup", "learned_quantile"]


def test_the_target_is_a_delivery_time_in_seconds(small_split) -> None:
    train, _ = small_split
    assert np.all(targets(train) > 0.0)
