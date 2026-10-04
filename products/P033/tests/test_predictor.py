"""Learned predictor: fitting, uncertainty output, metrics and the comparison."""

from __future__ import annotations

import numpy as np
import pytest

from edgeinfer.features import FEATURE_NAMES
from edgeinfer.predictor import (
    LatencyPredictor,
    compare_predictors,
    evaluate_predictions,
    split_indices,
)


def _synthetic(n: int = 160, seed: int = 0, noise: float = 0.05):
    """A target that is a smooth function of two features plus noise.

    The target is ``10**(-5 + 0.4 * f0 + 0.1 * f1)`` so that it is positive,
    spans orders of magnitude, and is learnable: a predictor that cannot fit
    this cannot fit anything, so a failure here is a bug in the wrapper rather
    than a fact about the model population.
    """
    rng = np.random.default_rng(seed)
    features = rng.uniform(0.0, 6.0, size=(n, len(FEATURE_NAMES)))
    log_target = -5.0 + 0.4 * features[:, 0] + 0.1 * features[:, 1]
    log_target = log_target + rng.normal(0.0, noise, size=n)
    return features, np.power(10.0, log_target)


class TestFitting:
    def test_fitting_is_deterministic_in_the_seed(self) -> None:
        features, target = _synthetic()
        a = LatencyPredictor(n_estimators=40, seed=3).fit(features, target)
        b = LatencyPredictor(n_estimators=40, seed=3).fit(features, target)
        assert np.allclose(a.predict(features), b.predict(features))

    def test_a_different_seed_changes_the_fit(self) -> None:
        features, target = _synthetic()
        a = LatencyPredictor(n_estimators=40, seed=3).fit(features, target)
        b = LatencyPredictor(n_estimators=40, seed=4).fit(features, target)
        assert not np.allclose(a.predict(features), b.predict(features))

    def test_a_learnable_target_is_learned(self) -> None:
        features, target = _synthetic(noise=0.02)
        train, test = split_indices(len(target), 0.3, seed=1)
        model = LatencyPredictor(n_estimators=80, seed=1).fit(
            features[train], target[train]
        )
        metrics = evaluate_predictions(model.predict(features[test]), target[test], "rf")
        assert metrics.median_abs_rel_error < 0.25
        assert metrics.spearman > 0.9

    def test_predicting_before_fitting_is_an_error(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            LatencyPredictor().predict(np.zeros((1, len(FEATURE_NAMES))))

    def test_a_non_positive_target_is_rejected(self) -> None:
        features, target = _synthetic(n=10)
        target[0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            LatencyPredictor(n_estimators=4).fit(features, target)

    def test_a_wrong_feature_width_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            LatencyPredictor(n_estimators=4).fit(np.zeros((5, 3)), np.ones(5))

    def test_mismatched_row_counts_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            LatencyPredictor(n_estimators=4).fit(
                np.zeros((5, len(FEATURE_NAMES))), np.ones(4)
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_estimators": 1}, "n_estimators"),
            ({"min_samples_leaf": 0}, "min_samples_leaf"),
            ({"max_depth": 0}, "max_depth"),
        ],
    )
    def test_invalid_hyperparameters_are_rejected(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            LatencyPredictor(**kwargs)


class TestUncertaintyOutput:
    def test_every_prediction_carries_an_uncertainty(self) -> None:
        features, target = _synthetic()
        model = LatencyPredictor(n_estimators=40, seed=2).fit(features, target)
        results = model.predict_with_uncertainty(features[:7])
        assert len(results) == 7
        for result in results:
            assert result.value > 0
            assert result.uncertainty >= 0
            assert result.n_estimators == 40

    def test_the_point_value_agrees_with_predict(self) -> None:
        features, target = _synthetic()
        model = LatencyPredictor(n_estimators=40, seed=2).fit(features, target)
        point = model.predict(features[:5])
        with_unc = [r.value for r in model.predict_with_uncertainty(features[:5])]
        assert np.allclose(point, with_unc, rtol=1e-9)

    def test_uncertainty_is_larger_away_from_the_training_support(self) -> None:
        """The ensemble disagrees more where it has seen nothing. This is the
        property that makes the output useful at all; it is not a claim that
        the interval is calibrated."""
        features, target = _synthetic(noise=0.02)
        model = LatencyPredictor(n_estimators=80, seed=2).fit(features, target)
        inside = model.predict_with_uncertainty(features[:40])
        outside_features = features[:40].copy()
        outside_features[:, 0] = 50.0  # far outside the 0-6 training range
        outside = model.predict_with_uncertainty(outside_features)
        median_inside = float(np.median([r.relative_uncertainty for r in inside]))
        median_outside = float(np.median([r.relative_uncertainty for r in outside]))
        assert median_outside >= median_inside

    def test_relative_uncertainty_is_uncertainty_over_value(self) -> None:
        features, target = _synthetic(n=40)
        model = LatencyPredictor(n_estimators=20, seed=2).fit(features, target)
        result = model.predict_with_uncertainty(features[:1])[0]
        assert result.relative_uncertainty == pytest.approx(
            result.uncertainty / result.value
        )

    def test_the_propagation_follows_the_gum_sensitivity_coefficient(self) -> None:
        """u(y) = y ln(10) u(log10 y)."""
        features, target = _synthetic(n=60)
        model = LatencyPredictor(n_estimators=20, seed=2).fit(features, target)
        result = model.predict_with_uncertainty(features[:1])[0]
        assert result.uncertainty == pytest.approx(
            result.value * np.log(10.0) * result.log10_uncertainty
        )

    def test_feature_importances_cover_every_feature(self) -> None:
        features, target = _synthetic(n=60)
        model = LatencyPredictor(n_estimators=20, seed=2).fit(features, target)
        importances = model.feature_importances
        assert set(importances) == set(FEATURE_NAMES)
        assert sum(importances.values()) == pytest.approx(1.0)

    def test_importances_before_fitting_are_an_error(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            _ = LatencyPredictor().feature_importances


class TestSplit:
    def test_split_is_deterministic_and_disjoint(self) -> None:
        train, test = split_indices(100, 0.3, seed=5)
        train2, test2 = split_indices(100, 0.3, seed=5)
        assert np.array_equal(train, train2)
        assert np.array_equal(test, test2)
        assert set(train).isdisjoint(set(test))
        assert len(train) + len(test) == 100
        assert len(test) == 30

    def test_a_different_seed_changes_the_split(self) -> None:
        _, a = split_indices(100, 0.3, seed=5)
        _, b = split_indices(100, 0.3, seed=6)
        assert not np.array_equal(a, b)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"n": 3}, "at least 4"), ({"test_fraction": 0.0}, "test_fraction"),
         ({"test_fraction": 1.0}, "test_fraction")],
    )
    def test_invalid_split_arguments_are_rejected(self, kwargs: dict, match: str) -> None:
        base = {"n": 100, "test_fraction": 0.3}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            split_indices(**base)


class TestMetrics:
    def test_perfect_predictions_give_zero_error(self) -> None:
        truth = np.array([1.0, 2.0, 4.0, 8.0])
        metrics = evaluate_predictions(truth, truth, "perfect")
        assert metrics.median_abs_rel_error == 0.0
        assert metrics.p90_abs_rel_error == 0.0
        assert metrics.max_abs_rel_error == 0.0
        assert metrics.log10_rmse == pytest.approx(0.0)
        assert metrics.spearman == pytest.approx(1.0)

    def test_known_answer_for_a_constant_relative_error(self) -> None:
        truth = np.array([1.0, 2.0, 4.0, 8.0])
        metrics = evaluate_predictions(truth * 1.1, truth, "ten-percent-high")
        assert metrics.median_abs_rel_error == pytest.approx(0.1)
        assert metrics.log10_rmse == pytest.approx(np.log10(1.1))

    def test_the_error_tail_is_reported_separately_from_the_median(self) -> None:
        truth = np.logspace(-5, -3, 100)
        predicted = truth.copy()
        predicted[:95] *= 1.01
        predicted[95:] *= 3.0
        metrics = evaluate_predictions(predicted, truth, "tail")
        assert metrics.median_abs_rel_error == pytest.approx(0.01, abs=1e-6)
        assert metrics.p90_abs_rel_error == pytest.approx(0.01, abs=1e-6)
        assert metrics.max_abs_rel_error == pytest.approx(2.0)

    def test_non_positive_values_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            evaluate_predictions(np.array([1.0, 2.0]), np.array([0.0, 2.0]), "x")

    def test_mismatched_shapes_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_predictions(np.ones(3), np.ones(4), "x")

    def test_a_single_point_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            evaluate_predictions(np.ones(1), np.ones(1), "x")

    def test_summary_lines_carry_units(self) -> None:
        truth = np.array([1.0, 2.0, 4.0, 8.0])
        text = "\n".join(evaluate_predictions(truth, truth, "p").summary_lines())
        assert "%" in text
        assert "Spearman" in text


class TestComparison:
    def test_a_clearly_better_predictor_wins(self) -> None:
        truth = np.logspace(-5, -3, 80)
        good = truth * 1.01
        bad = truth * 2.0
        _m_a, _m_l, verdict = compare_predictors(bad, good, truth)
        assert "learned random forest wins" in verdict

    def test_a_clearly_better_analytic_model_wins_and_is_said_to(self) -> None:
        truth = np.logspace(-5, -3, 80)
        _m_a, _m_l, verdict = compare_predictors(truth * 1.01, truth * 2.0, truth)
        assert "analytic roofline (calibrated) wins" in verdict

    def test_an_indistinguishable_pair_reports_no_measurable_advantage(self) -> None:
        rng = np.random.default_rng(0)
        truth = np.logspace(-5, -3, 200)
        a = truth * np.exp(rng.normal(0.0, 0.2, size=200))
        b = truth * np.exp(rng.normal(0.0, 0.2, size=200))
        _m_a, _m_l, verdict = compare_predictors(a, b, truth)
        assert "no measurable advantage" in verdict

    def test_the_verdict_quotes_the_bootstrap_settings(self) -> None:
        truth = np.logspace(-5, -3, 60)
        _a, _b, verdict = compare_predictors(truth * 1.5, truth * 1.02, truth)
        assert "bootstrap" in verdict

    def test_both_metric_sets_are_returned(self) -> None:
        truth = np.logspace(-5, -3, 60)
        m_a, m_l, _verdict = compare_predictors(truth * 1.5, truth * 1.02, truth)
        assert m_a.n_test == m_l.n_test == 60
        assert m_a.label != m_l.label
