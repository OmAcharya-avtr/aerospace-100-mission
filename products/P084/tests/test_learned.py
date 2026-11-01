"""Unit, input-validation and edge-case tests for the learned discrepancy model.

The model's measured coverage against the baselines is in
``tests/test_integration.py`` and, over many draws, in
``validation/validate_learned_model_output.txt``. What is tested here is that
the model does what its API says, including the parts of what it says that are
unflattering.
"""

from __future__ import annotations

import numpy as np
import pytest

from transfergap.learned import (
    ConformalisedDiscrepancyModel,
    DiscrepancyModel,
    gap_sample_split,
    marginal_bound_from_model,
)


def test_model_is_not_fitted_before_fit(field_sample):
    model = DiscrepancyModel(seed=0)
    assert model.fitted is False
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(field_sample.features)


def test_model_fits_and_predicts_one_value_per_row(field_sample, pool):
    model = DiscrepancyModel(seed=1).fit(field_sample.features, field_sample.gaps)
    assert model.fitted is True
    assert model.n_train == 60
    assert model.predict(pool.features).shape == (pool.size,)
    assert model.predict_upper(pool.features).shape == (pool.size,)


def test_predict_with_uncertainty_returns_a_band_not_a_point(field_sample, pool):
    model = DiscrepancyModel(upper_level=0.95, lower_level=0.05, seed=1)
    model.fit(field_sample.features, field_sample.gaps)
    prediction = model.predict_with_uncertainty(pool.features)
    assert prediction.point.shape == (pool.size,)
    assert prediction.lower.shape == (pool.size,)
    assert prediction.upper.shape == (pool.size,)
    assert np.allclose(prediction.width, prediction.upper - prediction.lower)
    assert prediction.lower_level == 0.05
    assert prediction.upper_level == 0.95


def test_quantile_crossing_is_reported_and_not_sorted_away(small_field, pool):
    """Three independently fitted quantile ensembles can cross on thin data.

    The crossing rate is a reported diagnostic. This test asserts only that
    the field exists and is a rate in [0, 1]; the measured value at each
    sample size is in validation/validate_learned_model_output.txt, because
    whether it is zero depends on the draw and asserting a particular value
    would be asserting a seed.
    """
    model = DiscrepancyModel(seed=2).fit(small_field.features, small_field.gaps)
    prediction = model.predict_with_uncertainty(pool.features)
    assert 0.0 <= prediction.crossing_rate <= 1.0
    assert prediction.crossing.shape == (pool.size,)


def test_model_is_reproducible_from_its_seed(field_sample, pool):
    first = DiscrepancyModel(seed=5).fit(field_sample.features, field_sample.gaps)
    second = DiscrepancyModel(seed=5).fit(field_sample.features, field_sample.gaps)
    assert np.array_equal(first.predict_upper(pool.features), second.predict_upper(pool.features))


def test_model_rejects_fewer_than_five_training_rows():
    with pytest.raises(ValueError, match="at least 5 training rows"):
        DiscrepancyModel(seed=0).fit(np.zeros((4, 4)), np.zeros(4))


def test_model_rejects_mismatched_rows():
    with pytest.raises(ValueError, match="one entry per feature row"):
        DiscrepancyModel(seed=0).fit(np.zeros((10, 4)), np.zeros(9))


def test_model_rejects_a_nan_in_the_training_data():
    features = np.zeros((10, 4))
    features[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        DiscrepancyModel(seed=0).fit(features, np.arange(10.0))


def test_model_rejects_a_wrong_feature_count_at_predict_time(field_sample):
    model = DiscrepancyModel(seed=0).fit(field_sample.features, field_sample.gaps)
    with pytest.raises(ValueError, match="shape"):
        model.predict(np.zeros((5, 2)))


def test_model_rejects_crossed_quantile_levels():
    with pytest.raises(ValueError, match="below upper_level"):
        DiscrepancyModel(upper_level=0.1, lower_level=0.9)


def test_conformalised_model_band_is_above_the_raw_band_when_the_offset_is_positive(
    field_sample, pool
):
    calibrated = ConformalisedDiscrepancyModel(confidence=0.95, seed=3)
    calibrated.fit(field_sample.features, field_sample.gaps)
    band = calibrated.predict_upper(pool.features)
    raw = calibrated.model.predict_upper(pool.features)
    assert np.allclose(band - raw, calibrated.offset)
    assert calibrated.n_fit + calibrated.n_calibration == field_sample.size


def test_conformalised_model_refuses_when_the_split_cannot_be_made():
    """n = 10 at calibration_fraction 0.5 leaves 5 to fit and 5 to calibrate.

    Split conformal at alpha = 0.05 needs 19 calibration scores by equation
    (12), so the offset cannot be computed and the error names that.
    """
    features = np.random.default_rng(1).uniform(size=(10, 4))
    gaps = np.random.default_rng(2).uniform(0.0, 5.0, size=10)
    with pytest.raises(ValueError, match="needs n >= 19"):
        ConformalisedDiscrepancyModel(confidence=0.95, seed=0).fit(features, gaps)


def test_conformalised_model_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least 5 rows"):
        ConformalisedDiscrepancyModel(seed=0).fit(np.zeros((6, 4)), np.zeros(6))


def test_conformalised_model_is_not_usable_before_fit(pool):
    with pytest.raises(RuntimeError, match="not fitted"):
        ConformalisedDiscrepancyModel(seed=0).predict_upper(pool.features)


def test_marginal_bound_from_model_is_not_marked_distribution_free(field_sample, pool):
    model = DiscrepancyModel(seed=4).fit(field_sample.features, field_sample.gaps)
    bound = marginal_bound_from_model(model, pool.features, 0.95, 0.95, "kN.m")
    assert bound.distribution_free is False
    assert bound.target == "quantile"
    assert bound.n == 60
    assert bound.unit == "kN.m"


def test_gap_sample_split_partitions_without_overlap(field_sample):
    fit, calibration = gap_sample_split(field_sample, 0.5, seed=1)
    assert fit.size + calibration.size == field_sample.size
    combined = np.concatenate([fit.gaps, calibration.gaps])
    assert np.allclose(np.sort(combined), np.sort(field_sample.gaps))


def test_gap_sample_split_rejects_a_degenerate_split(field_sample):
    with pytest.raises(ValueError, match="leaves"):
        gap_sample_split(field_sample, 0.999, seed=1)
