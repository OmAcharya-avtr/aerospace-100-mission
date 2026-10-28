"""The learned classifier, the majority guess and both uncertainty outputs."""

from __future__ import annotations

import numpy as np
import pytest

from telemgap import features
from telemgap.learned import (
    ABSTAIN,
    CLASS_NAMES,
    DEFAULT_ABSTAIN_THRESHOLD,
    LearnedClassifier,
    MajorityClassifier,
    dispersion_error_auc,
    expected_calibration_error,
)


def test_class_names_match_the_dataset_classes() -> None:
    from telemgap.dataset import VALUE_CLASSES

    assert CLASS_NAMES == VALUE_CLASSES
    assert ABSTAIN == -1
    assert 0.0 < DEFAULT_ABSTAIN_THRESHOLD < 1.0


def test_fit_rejects_bad_hyperparameters() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        LearnedClassifier(n_estimators=1)
    with pytest.raises(ValueError, match="clean_ratio"):
        LearnedClassifier(clean_ratio=0.0)


def test_fit_validates_shapes(generated_train) -> None:
    matrix, _ = features.build(generated_train.record)
    labels = features.truth_labels(generated_train.value_truth)
    model = LearnedClassifier(n_estimators=5)
    with pytest.raises(ValueError, match="columns matching"):
        model.fit(matrix[:, :3], labels)
    with pytest.raises(ValueError, match="rows but labels"):
        model.fit(matrix, labels[:-1])


def test_fit_rejects_single_class() -> None:
    matrix = np.zeros((50, len(features.FEATURE_NAMES)))
    labels = np.zeros(50, dtype=np.int64)
    with pytest.raises(ValueError, match="fewer than two classes"):
        LearnedClassifier(n_estimators=5).fit(matrix, labels)


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit must be called"):
        LearnedClassifier().predict(np.zeros((2, len(features.FEATURE_NAMES))))
    with pytest.raises(RuntimeError, match="fit must be called"):
        _ = LearnedClassifier().feature_importances


def test_training_subsample_is_seeded(generated_train) -> None:
    matrix, _ = features.build(generated_train.record)
    labels = features.truth_labels(generated_train.value_truth)
    first = LearnedClassifier(n_estimators=5, seed=7).fit(matrix, labels)
    second = LearnedClassifier(n_estimators=5, seed=7).fit(matrix, labels)
    assert first.training_counts == second.training_counts
    anomalies = int((labels != 0).sum())
    # Every anomaly is kept; the clean class is cut to clean_ratio times that.
    assert first.training_counts["corrupt"] + first.training_counts["misparsed"] == anomalies
    assert first.training_counts["clean"] <= int(round(10.0 * anomalies))


def test_predict_shapes_and_ranges(trained_model, generated_test) -> None:
    matrix, _ = features.build(generated_test.record)
    prediction = trained_model.predict(matrix)
    assert prediction.labels.shape == (matrix.shape[0],)
    assert prediction.proba.shape == (matrix.shape[0], len(CLASS_NAMES))
    assert np.all(prediction.probability >= 1.0 / len(CLASS_NAMES) - 1e-12)
    assert np.all(prediction.probability <= 1.0 + 1e-12)
    assert np.all(prediction.dispersion >= 0.0)
    assert prediction.abstention_rate == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert set(np.unique(prediction.labels)).issubset({0, 1, 2})


def test_probabilities_sum_to_one(trained_model, generated_test) -> None:
    matrix, _ = features.build(generated_test.record)
    prediction = trained_model.predict(matrix[:500])
    assert np.allclose(prediction.proba.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)


def test_abstention_raises_the_undecided_count(trained_model, generated_test) -> None:
    matrix, _ = features.build(generated_test.record)
    low = trained_model.predict(matrix, abstain_threshold=0.5)
    high = trained_model.predict(matrix, abstain_threshold=0.95)
    assert high.abstention_rate >= low.abstention_rate
    assert np.all(high.labels[high.abstained] == ABSTAIN)
    assert high.abstain_threshold == pytest.approx(0.95, rel=0.0, abs=0.0)


def test_predict_validates_arguments(trained_model) -> None:
    with pytest.raises(ValueError, match="columns"):
        trained_model.predict(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="abstain_threshold"):
        trained_model.predict(
            np.zeros((3, len(features.FEATURE_NAMES))), abstain_threshold=1.5
        )


def test_dispersion_is_zero_for_a_unanimous_forest(trained_model, generated_test) -> None:
    # Where every tree agrees on a probability of 1, the per-member standard
    # deviation must be zero. Checked as a tolerance, not as an equality,
    # because it is the square root of a Welford accumulation.
    matrix, _ = features.build(generated_test.record)
    prediction = trained_model.predict(matrix)
    unanimous = prediction.probability >= 1.0 - 1e-12
    assert unanimous.any()
    assert float(prediction.dispersion[unanimous].max()) == pytest.approx(
        0.0, rel=0.0, abs=1e-9
    )


def test_feature_importances_cover_every_feature(trained_model) -> None:
    importances = trained_model.feature_importances
    assert set(importances) == set(features.FEATURE_NAMES)
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-9, abs=0.0)


def test_majority_classifier_known_answer() -> None:
    labels = np.array([0, 0, 0, 1, 2], dtype=np.int64)
    model = MajorityClassifier().fit(labels)
    assert model.label == 0
    assert model.frequency == pytest.approx(0.6, rel=1e-12, abs=0.0)
    prediction = model.predict(np.zeros((4, 1)))
    assert list(prediction.labels) == [0, 0, 0, 0]
    assert np.all(prediction.probability == pytest.approx(0.6, rel=1e-12, abs=0.0))
    assert np.all(prediction.dispersion == 0.0)
    # Above its own frequency the majority guess abstains everywhere.
    assert np.all(model.predict(np.zeros((4, 1)), abstain_threshold=0.9).labels == ABSTAIN)


def test_majority_classifier_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="at least one label"):
        MajorityClassifier().fit(np.empty(0, dtype=np.int64))


def test_expected_calibration_error_known_answers() -> None:
    # Perfectly calibrated: confidence 1.0 and always right -> 0.
    probability = np.ones(100)
    correct = np.ones(100, dtype=bool)
    ece, centres, confidence, accuracy = expected_calibration_error(probability, correct)
    assert ece == pytest.approx(0.0, rel=0.0, abs=1e-15)
    assert centres.size == 10
    assert np.isnan(confidence[0])
    assert accuracy[-1] == pytest.approx(1.0, rel=0.0, abs=0.0)
    # Maximally overconfident: confidence 1.0 and always wrong -> 1.
    worst, _, _, _ = expected_calibration_error(np.ones(100), np.zeros(100, dtype=bool))
    assert worst == pytest.approx(1.0, rel=0.0, abs=1e-15)
    # Half at 1.0 right, half at 1.0 wrong -> gap of 0.5 in one bin.
    mixed, _, _, _ = expected_calibration_error(
        np.ones(100), np.concatenate((np.ones(50, bool), np.zeros(50, bool)))
    )
    assert mixed == pytest.approx(0.5, rel=1e-12, abs=0.0)


def test_expected_calibration_error_validates_arguments() -> None:
    with pytest.raises(ValueError, match="must match"):
        expected_calibration_error(np.ones(5), np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error(np.ones(5), np.ones(5, dtype=bool), bins=1)


def test_dispersion_error_auc_known_answers() -> None:
    # Perfect separation: every wrong row has a higher dispersion -> 1.0.
    dispersion = np.array([0.0, 0.1, 0.2, 0.9, 1.0])
    wrong = np.array([False, False, False, True, True])
    assert dispersion_error_auc(dispersion, wrong) == pytest.approx(1.0, rel=0.0, abs=1e-15)
    # Reversed -> 0.0.
    assert dispersion_error_auc(dispersion, ~wrong) == pytest.approx(0.0, rel=0.0, abs=1e-15)
    # All ties -> 0.5 exactly, by the mid-rank convention.
    assert dispersion_error_auc(np.ones(6), np.array([True] * 3 + [False] * 3)) == pytest.approx(
        0.5, rel=0.0, abs=1e-15
    )
    # One class empty -> nan, which is the honest answer, not 0.5.
    assert np.isnan(dispersion_error_auc(dispersion, np.zeros(5, dtype=bool)))
    with pytest.raises(ValueError, match="must match"):
        dispersion_error_auc(np.ones(5), np.ones(4, dtype=bool))


def test_model_beats_nothing_claim_is_measured(trained_model, generated_test) -> None:
    # The model must at least classify the clean class better than chance; a
    # stronger claim than this is made only by the benchmark, with numbers.
    matrix, _ = features.build(generated_test.record)
    labels = features.truth_labels(generated_test.value_truth)
    prediction = trained_model.predict(matrix)
    accuracy = float((prediction.labels == labels).mean())
    assert accuracy > 0.9
