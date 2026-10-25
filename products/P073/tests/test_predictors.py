"""Conflict predictors, their metrics and their calibration.

The metric functions are checked against arithmetic done by hand, written out in
the test comments. The predictors themselves are fitted on a small seeded dataset
with an instance-level split, exactly as the validation script does, and are
checked for interface behaviour and for beating a coin flip -- not for a specific
score, which belongs in ``validation/validate_predictors_output.txt``.
"""

from __future__ import annotations

import numpy as np
import pytest

from contactsched.dataset import build_dataset, split_by_instance
from contactsched.features import FEATURE_NAMES
from contactsched.predictors import (
    LearnedConflictPredictor,
    RatioLogisticBaseline,
    RatioRankingBaseline,
    average_precision,
    best_f1_threshold,
    brier_score,
    expected_calibration_error,
    reliability_table,
    roc_auc,
    threshold_metrics,
)


@pytest.fixture(scope="module")
def split():
    """Train and test matrices from disjoint seed ranges."""
    frame = build_dataset(range(900, 936), horizon_s=21600.0)
    train, test = split_by_instance(frame, range(900, 924), range(924, 936))
    x_train = train[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y_train = train["unsatisfiable"].to_numpy(dtype=int)
    x_test = test[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y_test = test["unsatisfiable"].to_numpy(dtype=int)
    x_train = np.where(np.isfinite(x_train), x_train, 1.0e6)
    x_test = np.where(np.isfinite(x_test), x_test, 1.0e6)
    return x_train, y_train, x_test, y_test


class TestMetricKnownAnswers:
    def test_roc_auc_on_the_textbook_example(self):
        # Labels [0, 0, 1, 1] with scores [0.1, 0.4, 0.35, 0.8]: of the four
        # positive-negative pairs, three are correctly ordered and one is not,
        # so the area is 0.75.
        assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(
            0.75, rel=0.0, abs=1e-15
        )

    def test_roc_auc_is_one_for_a_perfect_ranking(self):
        assert roc_auc([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0]) == pytest.approx(
            1.0, rel=0.0, abs=0.0
        )

    def test_roc_auc_is_half_for_a_constant_score(self):
        assert roc_auc([0, 1, 0, 1], [0.5] * 4) == pytest.approx(0.5, rel=0.0, abs=0.0)

    def test_roc_auc_is_nan_with_one_class(self):
        assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.5, 0.9]))

    def test_average_precision_is_one_for_a_perfect_ranking(self):
        assert average_precision([0, 0, 1, 1], [0.0, 0.1, 0.9, 1.0]) == pytest.approx(
            1.0, rel=0.0, abs=0.0
        )

    def test_average_precision_is_nan_with_one_class(self):
        assert np.isnan(average_precision([0, 0], [0.1, 0.2]))

    def test_brier_score_by_hand(self):
        # ((0.75 - 1)**2 + (0.25 - 0)**2) / 2 = (0.0625 + 0.0625) / 2 = 0.0625.
        assert brier_score([1, 0], [0.75, 0.25]) == pytest.approx(0.0625, rel=0.0, abs=1e-15)

    def test_brier_score_is_zero_for_a_perfect_forecast(self):
        assert brier_score([1, 0, 1], [1.0, 0.0, 1.0]) == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_brier_score_is_one_for_a_perfectly_wrong_forecast(self):
        assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_brier_score_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            brier_score([1, 0], [0.5])

    def test_threshold_metrics_by_hand(self):
        # Labels [1, 1, 0, 0], probabilities [0.9, 0.4, 0.6, 0.1] at 0.5 predict
        # [1, 0, 1, 0]: one true positive, one false positive, one false
        # negative, one true negative. Precision = recall = F1 = accuracy = 0.5.
        result = threshold_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)
        assert result["true_positives"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert result["false_positives"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert result["false_negatives"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert result["true_negatives"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert result["precision"] == pytest.approx(0.5, rel=0.0, abs=0.0)
        assert result["recall"] == pytest.approx(0.5, rel=0.0, abs=0.0)
        assert result["f1"] == pytest.approx(0.5, rel=0.0, abs=0.0)
        assert result["accuracy"] == pytest.approx(0.5, rel=0.0, abs=0.0)

    def test_threshold_metrics_with_no_positive_prediction(self):
        result = threshold_metrics([1, 0], [0.1, 0.2], 0.5)
        assert result["precision"] == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert result["recall"] == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert result["f1"] == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_threshold_is_inclusive(self):
        result = threshold_metrics([1], [0.5], 0.5)
        assert result["true_positives"] == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_reliability_table_by_hand(self):
        # Probabilities 0.1, 0.1, 0.9, 0.9 with labels 1, 0, 1, 1 and ten bins:
        # bin [0.1, 0.2) holds two rows predicting 0.1 and observing 0.5, bin
        # [0.9, 1.0] holds two predicting 0.9 and observing 1.0.
        table = reliability_table([1, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], bins=10)
        assert len(table) == 2
        assert table[0] == (
            pytest.approx(0.1, rel=0.0, abs=1e-15),
            pytest.approx(0.1, rel=0.0, abs=1e-15),
            pytest.approx(0.5, rel=0.0, abs=0.0),
            2,
        )
        assert table[1][2] == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_expected_calibration_error_by_hand(self):
        # (|0.1 - 0.5| * 2 + |0.9 - 1.0| * 2) / 4 = (0.8 + 0.2) / 4 = 0.25.
        value = expected_calibration_error([1, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], bins=10)
        assert value == pytest.approx(0.25, rel=0.0, abs=1e-15)

    def test_expected_calibration_error_is_zero_when_calibrated(self):
        # Four rows at 0.5 with two positives: predicted 0.5, observed 0.5.
        value = expected_calibration_error([1, 0, 1, 0], [0.5] * 4, bins=10)
        assert value == pytest.approx(0.0, rel=0.0, abs=1e-15)

    def test_expected_calibration_error_is_nan_on_empty_input(self):
        assert np.isnan(expected_calibration_error([], [], bins=10))

    def test_best_f1_threshold_picks_a_separating_cut(self):
        # Scores 0.1, 0.2 for negatives and 0.8, 0.9 for positives: any cut in
        # (0.2, 0.8] gives F1 = 1, and the returned threshold must be one of them.
        threshold = best_f1_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert 0.2 < threshold <= 0.8
        assert threshold_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold)[
            "f1"
        ] == pytest.approx(1.0, rel=0.0, abs=0.0)


class TestRatioRankingBaseline:
    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="unknown feature"):
            RatioRankingBaseline(feature="not-a-feature")

    def test_name_includes_the_feature(self):
        assert "ratio_max" in RatioRankingBaseline(feature="ratio_max").name

    def test_fit_is_a_no_op_returning_self(self, split):
        x_train, y_train, _x, _y = split
        model = RatioRankingBaseline()
        assert model.fit(x_train, y_train) is model

    def test_score_is_the_raw_feature_column(self, split):
        x_train, _y, _xt, _yt = split
        model = RatioRankingBaseline(feature="demand_over_usable")
        column = FEATURE_NAMES.index("demand_over_usable")
        assert np.allclose(model.score(x_train), x_train[:, column], rtol=0.0, atol=0.0)

    def test_probability_is_monotone_in_the_score(self, split):
        x_train, _y, _xt, _yt = split
        model = RatioRankingBaseline()
        order_score = np.argsort(model.score(x_train), kind="stable")
        order_proba = np.argsort(model.predict_proba(x_train), kind="stable")
        assert np.array_equal(order_score, order_proba)

    def test_probability_is_inside_the_unit_interval(self, split):
        x_train, _y, _xt, _yt = split
        probabilities = RatioRankingBaseline().predict_proba(x_train)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    def test_the_default_feature_is_the_measured_strongest_ratio(self):
        assert RatioRankingBaseline().feature == "demand_over_usable"

    def test_beats_a_coin_flip_on_held_out_rows(self, split):
        x_train, y_train, x_test, y_test = split
        model = RatioRankingBaseline().fit(x_train, y_train)
        assert roc_auc(y_test, model.score(x_test)) > 0.6

    def test_the_per_bin_ratios_are_measurably_weaker(self, split):
        # The specification names a per-station-window demand/supply ratio. It is
        # weaker than the per-request ratio on this generator, and on this split
        # it is below a coin flip. That is why it is not the default baseline,
        # and the inequality is asserted rather than left in prose.
        x_train, y_train, x_test, y_test = split
        strong = RatioRankingBaseline("demand_over_usable").fit(x_train, y_train)
        for weak_feature in ("ratio_max", "ratio_weighted"):
            weak = RatioRankingBaseline(weak_feature).fit(x_train, y_train)
            assert roc_auc(y_test, weak.score(x_test)) < roc_auc(
                y_test, strong.score(x_test)
            )


class TestRatioLogisticBaseline:
    def test_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="unknown feature"):
            RatioLogisticBaseline(feature="not-a-feature")

    def test_predicting_before_fitting_raises(self, split):
        _x, _y, x_test, _yt = split
        with pytest.raises(RuntimeError, match="has not been fitted"):
            RatioLogisticBaseline().predict_proba(x_test)

    def test_probability_is_inside_the_unit_interval(self, split):
        x_train, y_train, x_test, _y = split
        model = RatioLogisticBaseline().fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        assert np.all((probabilities > 0.0) & (probabilities < 1.0))

    def test_is_deterministic(self, split):
        x_train, y_train, x_test, _y = split
        first = RatioLogisticBaseline(random_state=0).fit(x_train, y_train).predict_proba(x_test)
        second = RatioLogisticBaseline(random_state=0).fit(x_train, y_train).predict_proba(x_test)
        assert np.allclose(first, second, rtol=0.0, atol=0.0)

    def test_beats_a_coin_flip_on_held_out_rows(self, split):
        x_train, y_train, x_test, y_test = split
        model = RatioLogisticBaseline().fit(x_train, y_train)
        assert roc_auc(y_test, model.score(x_test)) > 0.6

    def test_a_monotone_one_feature_fit_preserves_the_ranking(self, split):
        # A logistic regression on one feature is monotone in that feature, so
        # the logistic and ranking forms must agree exactly on rank statistics.
        x_train, y_train, x_test, y_test = split
        ranking = RatioRankingBaseline().fit(x_train, y_train)
        logistic = RatioLogisticBaseline().fit(x_train, y_train)
        assert roc_auc(y_test, logistic.score(x_test)) == pytest.approx(
            roc_auc(y_test, ranking.score(x_test)), rel=0.0, abs=1e-12
        )

    def test_scores_better_than_an_uninformative_forecast(self, split):
        x_train, y_train, x_test, y_test = split
        model = RatioLogisticBaseline().fit(x_train, y_train)
        base_rate = float(np.mean(y_train))
        constant = brier_score(y_test, np.full(len(y_test), base_rate))
        assert brier_score(y_test, model.predict_proba(x_test)) < constant


class TestLearnedConflictPredictor:
    def test_predicting_before_fitting_raises(self, split):
        _x, _y, x_test, _yt = split
        with pytest.raises(RuntimeError, match="has not been fitted"):
            LearnedConflictPredictor().predict_proba(x_test)

    def test_probability_is_inside_the_unit_interval(self, split):
        x_train, y_train, x_test, _y = split
        model = LearnedConflictPredictor().fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    def test_output_length_matches_the_input(self, split):
        x_train, y_train, x_test, _y = split
        model = LearnedConflictPredictor().fit(x_train, y_train)
        assert model.predict_proba(x_test).shape == (len(x_test),)

    def test_is_deterministic_for_a_fixed_seed(self, split):
        x_train, y_train, x_test, _y = split
        first = LearnedConflictPredictor(random_state=3).fit(x_train, y_train)
        second = LearnedConflictPredictor(random_state=3).fit(x_train, y_train)
        assert np.allclose(
            first.predict_proba(x_test), second.predict_proba(x_test), rtol=0.0, atol=0.0
        )

    def test_beats_a_coin_flip_on_held_out_rows(self, split):
        x_train, y_train, x_test, y_test = split
        model = LearnedConflictPredictor().fit(x_train, y_train)
        assert roc_auc(y_test, model.score(x_test)) > 0.7

    def test_beats_the_strongest_single_ratio_baseline(self, split):
        # The claim the product makes about its learned component, asserted on a
        # small split so a regression is caught by the suite and not only by the
        # validation script.
        x_train, y_train, x_test, y_test = split
        learned = LearnedConflictPredictor().fit(x_train, y_train)
        baseline = RatioLogisticBaseline().fit(x_train, y_train)
        assert roc_auc(y_test, learned.score(x_test)) > roc_auc(
            y_test, baseline.score(x_test)
        )

    def test_name_is_stable(self):
        assert LearnedConflictPredictor().name == "learned-hgb"

    def test_score_and_predict_proba_agree(self, split):
        x_train, y_train, x_test, _y = split
        model = LearnedConflictPredictor().fit(x_train, y_train)
        assert np.allclose(
            model.score(x_test), model.predict_proba(x_test), rtol=0.0, atol=0.0
        )
