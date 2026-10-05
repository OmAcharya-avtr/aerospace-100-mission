"""The held-out comparison harness."""

from __future__ import annotations

import numpy as np
import pytest

from latencynet.compare import (
    REFERENCE_MODEL,
    compare_models,
    default_models,
    format_comparison,
)
from latencynet.learned import BoostingHyperparameters, LearnedTailPredictor
from latencynet.linear import LinearTailPredictor
from latencynet.predictors import AnalyticTailPredictor


def _small_models():
    """Cheap stand-ins so the test suite does not fit 300-tree ensembles."""
    return (
        AnalyticTailPredictor(assume_independent=True),
        LinearTailPredictor(),
        LearnedTailPredictor(
            BoostingHyperparameters(n_estimators=40, max_depth=2), native_interval_level=0.9
        ),
        AnalyticTailPredictor(assume_independent=False),
    )


def test_default_models_order_and_names():
    models = default_models()
    assert [m.name for m in models] == [
        "analytic_sum_indep",
        "linear_ols",
        "learned_gbt",
        "analytic_sum_cov",
    ]
    assert models[0].name == REFERENCE_MODEL
    assert models[0].uses_dependence_features is False
    assert models[3].uses_dependence_features is True


def test_comparison_structure(tiny_dataset):
    result = compare_models(tiny_dataset, 0.99, level=0.9, models=_small_models())
    assert result.regime == "correlated"
    assert result.p == 0.99
    assert result.n_train == 32
    assert result.n_test == 14
    assert len(result.scores) == 4
    assert result.winner in [s.name for s in result.scores]
    assert result.coverage_quantisation == pytest.approx(1.0 / 15.0, rel=1e-14)
    assert result.mean_reference_relative_se > 0.0
    baselines = [s.name for s in result.scores if s.is_baseline]
    assert baselines == ["analytic_sum_indep", "linear_ols"]


def test_reference_model_has_no_paired_p_value(tiny_dataset):
    result = compare_models(tiny_dataset, 0.99, models=_small_models())
    ref = result.score(REFERENCE_MODEL)
    assert np.isnan(ref.paired_p_value)
    assert ref.paired_mean_difference == 0.0
    for s in result.scores:
        if s.name != REFERENCE_MODEL:
            assert np.isfinite(s.paired_p_value)


def test_every_model_gets_both_interval_kinds(tiny_dataset):
    result = compare_models(tiny_dataset, 0.99, models=_small_models())
    for s in result.scores:
        assert s.native_coverage is not None
        assert 0.0 <= s.native_coverage.measured <= 1.0
        assert 0.0 <= s.conformal_coverage.measured <= 1.0
        assert s.conformal_coverage.nominal == 0.9
        assert s.log_errors.shape == (14,)


def test_independence_assumption_biases_the_analytic_model_low_when_correlated(tiny_dataset):
    # The whole point of the correlated regime: dropping the covariance terms
    # understates the total variance, so the predicted tail is too small and
    # the signed log error is negative. This is the unsafe direction for a
    # deadline and is the finding the README reports.
    result = compare_models(tiny_dataset, 0.99, models=_small_models())
    assert result.score("analytic_sum_indep").accuracy.bias_log < 0.0


def test_covariance_variant_is_closer_than_the_independence_variant(tiny_dataset):
    result = compare_models(tiny_dataset, 0.99, models=_small_models())
    indep = result.score("analytic_sum_indep").accuracy.mean_abs_log_error
    with_cov = result.score("analytic_sum_cov").accuracy.mean_abs_log_error
    assert with_cov < indep


def test_score_lookup_raises_for_an_unknown_name(tiny_dataset):
    result = compare_models(tiny_dataset, 0.99, models=_small_models())
    with pytest.raises(KeyError, match="no model named"):
        result.score("nonexistent")


def test_format_comparison_is_printable(tiny_dataset):
    text = format_comparison(compare_models(tiny_dataset, 0.99, models=_small_models()))
    assert "winner by mean absolute log error" in text
    assert "analytic_sum_indep" in text
    assert "linear_ols" in text
    assert "learned_gbt" in text
    assert "reference-target mean relative MC standard error" in text


def test_validation(tiny_dataset):
    with pytest.raises(ValueError, match="p must lie"):
        compare_models(tiny_dataset, 1.0)
    with pytest.raises(ValueError, match="level must lie"):
        compare_models(tiny_dataset, 0.99, level=0.0)
    with pytest.raises(ValueError, match="models must be non-empty"):
        compare_models(tiny_dataset, 0.99, models=())
