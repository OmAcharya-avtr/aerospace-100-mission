"""The learned classifier and the majority-class baseline it must beat."""

from __future__ import annotations

import numpy as np
import pytest

from tlecheck.consequence import BENIGN, SILENTLY_WRONG
from tlecheck.learned import (
    CLASS_ORDER,
    CONFIGURATIONS,
    RANDOM_STATE,
    LearnedClassifier,
    MajorityClassifier,
)


def test_class_order_is_fixed():
    assert CLASS_ORDER == (BENIGN, SILENTLY_WRONG, "will_fail_loudly")


def test_random_state_is_fixed():
    assert RANDOM_STATE == 20261029


def test_three_declared_configurations():
    assert set(CONFIGURATIONS) == {
        "elements_and_history",
        "elements_only",
        "with_rule_score",
    }


def test_fit_and_predict_shapes(prepared_train, prepared_test):
    model = LearnedClassifier().fit(
        prepared_train.matrices["elements_and_history"], prepared_train.measured_classes
    )
    prediction = model.predict(prepared_test.matrices["elements_and_history"])
    rows = len(prepared_test.records)
    assert prediction.labels.shape == (rows,)
    assert prediction.probabilities.shape == (rows, 3)
    assert prediction.silent_probability.shape == (rows,)
    assert prediction.dispersion.shape == (rows,)
    assert prediction.margin.shape == (rows,)


def test_probabilities_sum_to_one(prepared_train, prepared_test):
    model = LearnedClassifier().fit(
        prepared_train.matrices["elements_and_history"], prepared_train.measured_classes
    )
    prediction = model.predict(prepared_test.matrices["elements_and_history"])
    assert np.allclose(prediction.probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-9)


def test_silent_probability_is_the_silent_column(prepared_train, prepared_test):
    model = LearnedClassifier().fit(
        prepared_train.matrices["elements_and_history"], prepared_train.measured_classes
    )
    prediction = model.predict(prepared_test.matrices["elements_and_history"])
    column = CLASS_ORDER.index(SILENTLY_WRONG)
    assert np.allclose(
        prediction.silent_probability,
        prediction.probabilities[:, column],
        rtol=0.0,
        atol=0.0,
    )


def test_uncertainty_outputs_are_in_range(prepared_train, prepared_test):
    model = LearnedClassifier().fit(
        prepared_train.matrices["elements_and_history"], prepared_train.measured_classes
    )
    prediction = model.predict(prepared_test.matrices["elements_and_history"])
    assert np.all(prediction.dispersion >= 0.0)
    assert np.all(prediction.dispersion <= 0.5)
    assert np.all(prediction.margin >= 0.0)
    assert np.all(prediction.margin <= 1.0)


def test_margin_is_the_top_two_gap(prepared_train, prepared_test):
    model = LearnedClassifier().fit(
        prepared_train.matrices["elements_and_history"], prepared_train.measured_classes
    )
    prediction = model.predict(prepared_test.matrices["elements_and_history"])
    ordered = np.sort(prediction.probabilities, axis=1)
    assert np.allclose(
        prediction.margin, ordered[:, -1] - ordered[:, -2], rtol=0.0, atol=0.0
    )


def test_labels_are_the_argmax(prepared_train, prepared_test):
    model = LearnedClassifier().fit(
        prepared_train.matrices["elements_and_history"], prepared_train.measured_classes
    )
    prediction = model.predict(prepared_test.matrices["elements_and_history"])
    expected = [CLASS_ORDER[index] for index in np.argmax(prediction.probabilities, axis=1)]
    assert list(prediction.labels) == expected


def test_fit_is_deterministic(prepared_train, prepared_test):
    first = LearnedClassifier().fit(
        prepared_train.matrices["elements_and_history"], prepared_train.measured_classes
    )
    second = LearnedClassifier().fit(
        prepared_train.matrices["elements_and_history"], prepared_train.measured_classes
    )
    a = first.predict(prepared_test.matrices["elements_and_history"])
    b = second.predict(prepared_test.matrices["elements_and_history"])
    assert np.allclose(a.probabilities, b.probabilities, rtol=0.0, atol=0.0)


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        LearnedClassifier().predict(np.zeros((2, 3)))


def test_predict_rejects_the_wrong_column_count(prepared_train):
    model = LearnedClassifier().fit(
        prepared_train.matrices["elements_and_history"], prepared_train.measured_classes
    )
    with pytest.raises(ValueError, match="expected"):
        model.predict(np.zeros((2, 3)))


def test_fit_rejects_a_one_dimensional_matrix():
    with pytest.raises(ValueError, match="two-dimensional"):
        LearnedClassifier().fit(np.zeros(5), np.array([BENIGN] * 5))


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="labels supplied"):
        LearnedClassifier().fit(np.zeros((5, 2)), np.array([BENIGN] * 4))


def test_fit_rejects_unknown_labels():
    with pytest.raises(ValueError, match="unknown classes"):
        LearnedClassifier().fit(np.zeros((4, 2)), np.array(["maybe"] * 4))


@pytest.mark.parametrize(("trees", "depth"), [(0, 5), (-1, 5), (10, 0)])
def test_constructor_validates_architecture(trees, depth):
    with pytest.raises(ValueError):
        LearnedClassifier(trees=trees, max_depth=depth)


def test_feature_configuration_is_recorded():
    model = LearnedClassifier(feature_configuration="elements_only")
    assert model.feature_configuration == "elements_only"


def test_majority_predicts_the_majority(prepared_train):
    majority = MajorityClassifier().fit(prepared_train.measured_classes)
    prediction = majority.predict(7)
    assert set(prediction.labels) == {majority.majority}
    assert prediction.labels.shape == (7,)


def test_majority_prevalences_sum_to_one(prepared_train):
    majority = MajorityClassifier().fit(prepared_train.measured_classes)
    assert sum(majority.prevalence.values()) == pytest.approx(1.0, rel=0.0, abs=1.0e-12)


def test_majority_probability_is_constant(prepared_train):
    majority = MajorityClassifier().fit(prepared_train.measured_classes)
    prediction = majority.predict(5)
    assert np.allclose(
        prediction.silent_probability,
        prediction.silent_probability[0],
        rtol=0.0,
        atol=0.0,
    )


def test_majority_rejects_empty_labels():
    with pytest.raises(ValueError, match="non-empty"):
        MajorityClassifier().fit(np.array([]))


def test_majority_predict_before_fit_raises():
    with pytest.raises(ValueError, match="not fitted"):
        MajorityClassifier().predict(3)


def test_majority_rejects_negative_rows(prepared_train):
    majority = MajorityClassifier().fit(prepared_train.measured_classes)
    with pytest.raises(ValueError, match="non-negative"):
        majority.predict(-1)


def test_majority_zero_rows_is_empty(prepared_train):
    majority = MajorityClassifier().fit(prepared_train.measured_classes)
    assert majority.predict(0).labels.shape == (0,)


def test_members_are_available_for_the_dispersion_output(prepared_train):
    model = LearnedClassifier(trees=12).fit(
        prepared_train.matrices["elements_only"], prepared_train.measured_classes
    )
    assert len(model.forest.estimators_) == 12
