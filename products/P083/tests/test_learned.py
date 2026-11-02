"""The learned component: its spec, its uncertainty output and its refusals."""

from __future__ import annotations

import numpy as np
import pytest

from hilverdict.features import FEATURE_NAMES
from hilverdict.learned import (
    DEFAULT_SPEC,
    ClassifierSpec,
    MissClassifier,
)


def _data(count: int = 200, seed: int = 0):
    generator = np.random.default_rng(seed)
    features = generator.random((count, len(FEATURE_NAMES)))
    labels = features[:, 0] + 0.3 * features[:, 3] > 1.0
    return features, labels


def test_default_spec_values_are_declared():
    assert DEFAULT_SPEC.n_estimators == 150
    assert DEFAULT_SPEC.learning_rate == pytest.approx(0.08, rel=1e-15, abs=0.0)
    assert DEFAULT_SPEC.max_depth == 3
    assert DEFAULT_SPEC.min_samples_leaf == 4
    assert DEFAULT_SPEC.subsample == pytest.approx(0.9, rel=1e-15, abs=0.0)
    assert DEFAULT_SPEC.positive_weight == pytest.approx(8.0, rel=1e-15, abs=0.0)
    assert DEFAULT_SPEC.seed == 20260301
    assert DEFAULT_SPEC.abstention_band == pytest.approx(0.05, rel=1e-15, abs=0.0)


def test_spec_rejects_zero_estimators():
    with pytest.raises(ValueError, match="n_estimators must be at least 1"):
        ClassifierSpec(n_estimators=0)


@pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
def test_spec_rejects_an_out_of_range_learning_rate(rate):
    with pytest.raises(ValueError, match="learning_rate must lie"):
        ClassifierSpec(learning_rate=rate)


def test_spec_rejects_zero_depth():
    with pytest.raises(ValueError, match="max_depth must be at least 1"):
        ClassifierSpec(max_depth=0)


def test_spec_rejects_a_non_positive_positive_weight():
    with pytest.raises(ValueError, match="positive_weight must be positive"):
        ClassifierSpec(positive_weight=0.0)


@pytest.mark.parametrize("band", [-0.1, 0.5, 0.9])
def test_spec_rejects_an_out_of_range_abstention_band(band):
    with pytest.raises(ValueError, match="abstention_band must lie"):
        ClassifierSpec(abstention_band=band)


def test_an_unfitted_classifier_reports_itself_unfitted():
    assert MissClassifier().fitted is False


def test_predict_before_fit_raises():
    features, _labels = _data(10)
    with pytest.raises(ValueError, match="has not been fitted"):
        MissClassifier().predict_proba(features)


def test_feature_importance_before_fit_raises():
    with pytest.raises(ValueError, match="has not been fitted"):
        _ = MissClassifier().feature_importance


def test_fit_returns_self_and_records_the_sample_size():
    features, labels = _data()
    classifier = MissClassifier()
    assert classifier.fit(features, labels) is classifier
    assert classifier.fitted is True
    assert classifier.trained_on == features.shape[0]


def test_fit_rejects_a_single_class_label_set():
    features, _labels = _data()
    with pytest.raises(ValueError, match="both consequential and"):
        MissClassifier().fit(features, np.zeros(features.shape[0], dtype=bool))


def test_fit_rejects_an_all_positive_label_set():
    features, _labels = _data()
    with pytest.raises(ValueError, match="both consequential and"):
        MissClassifier().fit(features, np.ones(features.shape[0], dtype=bool))


def test_fit_rejects_the_wrong_feature_width():
    generator = np.random.default_rng(1)
    features = generator.random((50, 3))
    labels = features[:, 0] > 0.5
    with pytest.raises(ValueError, match="features must have shape"):
        MissClassifier().fit(features, labels)


def test_fit_rejects_a_label_count_mismatch():
    features, labels = _data(50)
    with pytest.raises(ValueError, match="feature rows and"):
        MissClassifier().fit(features, labels[:-1])


def test_predict_proba_is_a_probability():
    features, labels = _data()
    classifier = MissClassifier().fit(features, labels)
    probability = classifier.predict_proba(features)
    assert probability.shape == (features.shape[0],)
    assert float(np.min(probability)) >= 0.0
    assert float(np.max(probability)) <= 1.0


def test_predict_proba_rejects_the_wrong_feature_width():
    features, labels = _data()
    classifier = MissClassifier().fit(features, labels)
    with pytest.raises(ValueError, match="features must have shape"):
        classifier.predict_proba(features[:, :3])


def test_prediction_fields_are_consistent():
    features, labels = _data()
    classifier = MissClassifier().fit(features, labels).set_threshold(0.4)
    prediction = classifier.predict_with_uncertainty(features)
    assert prediction.threshold == pytest.approx(0.4, rel=0.0, abs=0.0)
    assert prediction.margin == pytest.approx(
        prediction.probability - 0.4, rel=0.0, abs=0.0
    )
    assert np.array_equal(prediction.flagged, prediction.probability >= 0.4)


def test_referred_is_flagged_or_abstained():
    features, labels = _data()
    classifier = MissClassifier().fit(features, labels).set_threshold(0.5)
    prediction = classifier.predict_with_uncertainty(features)
    assert np.array_equal(
        prediction.referred, prediction.flagged | prediction.abstained
    )


def test_abstention_band_covers_the_threshold_neighbourhood():
    features, labels = _data()
    spec = ClassifierSpec(abstention_band=0.4)
    classifier = MissClassifier(spec=spec).fit(features, labels).set_threshold(0.5)
    prediction = classifier.predict_with_uncertainty(features)
    assert np.array_equal(prediction.abstained, np.abs(prediction.margin) < 0.4)


def test_a_zero_abstention_band_abstains_on_nothing():
    features, labels = _data()
    spec = ClassifierSpec(abstention_band=0.0)
    classifier = MissClassifier(spec=spec).fit(features, labels)
    prediction = classifier.predict_with_uncertainty(features)
    assert not bool(np.any(prediction.abstained))


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_set_threshold_rejects_an_out_of_range_value(value):
    with pytest.raises(ValueError, match="threshold must lie"):
        MissClassifier().set_threshold(value)


def test_set_threshold_returns_self():
    classifier = MissClassifier()
    assert classifier.set_threshold(0.3) is classifier


def test_feature_importance_sums_to_one():
    features, labels = _data()
    classifier = MissClassifier().fit(features, labels)
    assert sum(classifier.feature_importance.values()) == pytest.approx(
        1.0, rel=1e-9, abs=0.0
    )


def test_feature_importance_names_every_feature():
    features, labels = _data()
    classifier = MissClassifier().fit(features, labels)
    assert set(classifier.feature_importance) == set(FEATURE_NAMES)


def test_the_informative_feature_dominates_the_importance():
    features, labels = _data(400, seed=7)
    classifier = MissClassifier().fit(features, labels)
    importance = classifier.feature_importance
    ranked = sorted(importance, key=lambda key: -importance[key])
    assert ranked[0] == FEATURE_NAMES[0]


def test_reliability_bins_are_consistent():
    features, labels = _data(400, seed=3)
    classifier = MissClassifier().fit(features, labels)
    reliability = classifier.reliability(features, labels, bins=4)
    assert reliability.edges.shape == (5,)
    assert reliability.predicted.shape == reliability.observed.shape
    assert int(np.sum(reliability.counts)) == features.shape[0]


def test_reliability_brier_is_in_range():
    features, labels = _data(300, seed=4)
    classifier = MissClassifier().fit(features, labels)
    reliability = classifier.reliability(features, labels)
    assert 0.0 <= reliability.brier <= 1.0


def test_reliability_rejects_too_few_bins():
    features, labels = _data(100)
    classifier = MissClassifier().fit(features, labels)
    with pytest.raises(ValueError, match="bins must be at least 2"):
        classifier.reliability(features, labels, bins=1)


def test_reliability_rejects_a_label_count_mismatch():
    features, labels = _data(100)
    classifier = MissClassifier().fit(features, labels)
    with pytest.raises(ValueError, match="predictions and"):
        classifier.reliability(features, labels[:-1])


def test_worst_gap_is_nan_on_an_empty_reliability():
    from hilverdict.learned import Reliability

    empty = Reliability(
        edges=np.linspace(0.0, 1.0, 3),
        predicted=np.array([]),
        observed=np.array([]),
        counts=np.array([], dtype=int),
        brier=0.0,
    )
    assert np.isnan(empty.worst_gap)


def test_the_same_seed_gives_the_same_model():
    features, labels = _data(300, seed=5)
    first = MissClassifier().fit(features, labels).predict_proba(features)
    second = MissClassifier().fit(features, labels).predict_proba(features)
    assert first == pytest.approx(second, rel=0.0, abs=0.0)


def test_a_different_seed_gives_a_different_model():
    features, labels = _data(300, seed=5)
    first = MissClassifier().fit(features, labels).predict_proba(features)
    other = MissClassifier(spec=ClassifierSpec(seed=99)).fit(
        features, labels
    ).predict_proba(features)
    assert float(np.max(np.abs(first - other))) > 0.0


def test_shuffling_the_labels_destroys_the_fit():
    features, labels = _data(400, seed=11)
    honest = MissClassifier().fit(features, labels)
    generator = np.random.default_rng(0)
    shuffled = labels.copy()
    generator.shuffle(shuffled)
    broken = MissClassifier().fit(features, shuffled)
    rate = float(np.mean(labels))
    honest_brier = honest.reliability(features, labels).brier
    broken_brier = broken.reliability(features, labels).brier
    assert honest_brier < rate * (1.0 - rate)
    assert broken_brier > honest_brier
