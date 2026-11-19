"""The learned model: confidence output, degenerate folds and input validation."""

from __future__ import annotations

import numpy as np
import pytest

from evidencestale.features import feature_matrix
from evidencestale.learned import MAX_DEPTH, N_ESTIMATORS, StalenessRiskModel


def _toy() -> tuple[np.ndarray, list[int]]:
    """A separable toy problem: label 1 when the first column exceeds 5."""
    rng = np.random.default_rng(980098)
    features = rng.uniform(0.0, 10.0, size=(60, 3))
    labels = [int(value > 5.0) for value in features[:, 0]]
    return features, labels


def test_hyperparameters_are_the_published_values() -> None:
    model = StalenessRiskModel()
    assert (model.n_estimators, model.max_depth) == (N_ESTIMATORS, MAX_DEPTH)
    assert (N_ESTIMATORS, MAX_DEPTH) == (200, 6)


def test_probability_spread_and_confidence_are_in_range() -> None:
    features, labels = _toy()
    model = StalenessRiskModel().fit(features, labels)
    prediction = model.predict_with_confidence(features)
    assert prediction.probability.shape == (60,)
    assert np.all((prediction.probability >= 0.0) & (prediction.probability <= 1.0))
    # The standard deviation of 0/1 votes cannot exceed 0.5.
    assert np.all((prediction.spread >= 0.0) & (prediction.spread <= 0.5))
    assert np.all((prediction.confidence >= 0.5) & (prediction.confidence <= 1.0))
    assert np.allclose(
        prediction.confidence,
        np.maximum(prediction.probability, 1.0 - prediction.probability),
    )


def test_confidence_is_not_a_point_estimate() -> None:
    """The model exposes three numbers per item, not one."""
    features, labels = _toy()
    prediction = StalenessRiskModel().fit(features, labels).predict_with_confidence(
        features
    )
    assert prediction.spread.max() >= 0.0
    assert set(("probability", "spread", "confidence")) <= set(
        prediction.__dataclass_fields__
    )


def test_decide_thresholds_the_probability() -> None:
    features, labels = _toy()
    prediction = StalenessRiskModel().fit(features, labels).predict_with_confidence(
        features
    )
    decisions = prediction.decide(0.5)
    assert set(np.unique(decisions)) <= {0, 1}
    assert np.array_equal(decisions, (prediction.probability >= 0.5).astype(np.int64))


def test_decide_rejects_a_threshold_outside_the_open_unit_interval() -> None:
    features, labels = _toy()
    prediction = StalenessRiskModel().fit(features, labels).predict_with_confidence(
        features
    )
    with pytest.raises(ValueError, match="threshold must lie"):
        prediction.decide(0.0)


def test_single_class_training_set_is_recorded_not_rejected() -> None:
    features = np.zeros((5, 3), dtype=np.float64)
    model = StalenessRiskModel().fit(features, [1, 1, 1, 1, 1])
    assert model.degenerate is True
    prediction = model.predict_with_confidence(features)
    assert np.all(prediction.probability == 1.0)
    assert np.all(prediction.spread == 0.0)
    assert model.feature_importances().size == 0


def test_predicting_before_fitting_is_an_error() -> None:
    with pytest.raises(ValueError, match="before fit"):
        StalenessRiskModel().predict_with_confidence(np.zeros((1, 3)))


def test_misaligned_features_and_labels_rejected() -> None:
    with pytest.raises(ValueError, match="aligned item by item"):
        StalenessRiskModel().fit(np.zeros((3, 2)), [1, 0])


def test_one_dimensional_features_rejected() -> None:
    with pytest.raises(ValueError, match="2-D array"):
        StalenessRiskModel().fit(np.zeros(5), [1, 0, 1, 0, 1])


def test_empty_training_set_rejected() -> None:
    with pytest.raises(ValueError, match="at least one training item"):
        StalenessRiskModel().fit(np.zeros((0, 3)), [])


def test_bad_hyperparameters_rejected() -> None:
    with pytest.raises(ValueError, match="n_estimators must be at least 1"):
        StalenessRiskModel(n_estimators=0)
    with pytest.raises(ValueError, match="max_depth must be at least 1"):
        StalenessRiskModel(max_depth=0)


def test_the_same_seed_gives_the_same_predictions() -> None:
    features, labels = _toy()
    first = StalenessRiskModel().fit(features, labels).predict_with_confidence(features)
    second = StalenessRiskModel().fit(features, labels).predict_with_confidence(features)
    assert np.array_equal(first.probability, second.probability)
    assert np.array_equal(first.spread, second.spread)


def test_importances_sum_to_one_and_cover_every_column(layered_corpus) -> None:
    features = feature_matrix(
        layered_corpus.evidence,
        layered_corpus.history,
        layered_corpus.true_graph,
        layered_corpus.evaluation_time,
    )
    model = StalenessRiskModel().fit(features, layered_corpus.labels())
    importances = model.feature_importances()
    assert importances.size == features.shape[1]
    assert float(importances.sum()) == pytest.approx(1.0, rel=1e-9, abs=0.0)
