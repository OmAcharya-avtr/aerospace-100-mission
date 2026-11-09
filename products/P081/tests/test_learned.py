"""The learned classifier: its label sets, its probabilities, and its refusals."""

from __future__ import annotations

import numpy as np
import pytest

from twindiverge.learned import (
    ABSTAIN_THRESHOLD,
    VARIANTS,
    Prediction,
    ResidualClassifier,
)
from twindiverge.plant import PARAMETER_NAMES
from twindiverge.synthetic import UNIDENTIFIABLE

RNG = np.random.default_rng(314159)


def _separable(cases: int = 60, classes: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """A declared separable toy problem, so these tests measure the wrapper and
    not the physics. Class ``k`` has its ``k``-th feature raised by 10."""
    labels = np.array([index % classes for index in range(cases)])
    features = RNG.normal(0.0, 1.0, size=(cases, 16))
    for row, label in enumerate(labels):
        features[row, label % 16] += 10.0
    return features, labels


def test_the_parameter_variant_has_five_labels_and_the_refusal_variant_six():
    assert ResidualClassifier("parameter").labels == tuple(range(5))
    assert ResidualClassifier("with_refusal").labels == (0, 1, 2, 3, 4, UNIDENTIFIABLE)
    assert VARIANTS == ("parameter", "with_refusal")


def test_an_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        ResidualClassifier("magic")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stages": 0}, "stages"),
        ({"max_depth": 0}, "max_depth"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": 1.5}, "learning_rate"),
    ],
)
def test_hyperparameter_validation(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        ResidualClassifier("parameter", **kwargs)


def test_probabilities_sum_to_one_and_are_in_label_order():
    features, labels = _separable()
    model = ResidualClassifier("parameter", stages=20).fit(features, labels)
    probabilities = model.predict_probabilities(features)
    assert probabilities.shape == (features.shape[0], 5)
    assert probabilities.sum(axis=1) == pytest.approx(
        np.ones(features.shape[0]), rel=1e-12, abs=0.0
    )


def test_a_label_absent_from_training_gets_a_zero_column():
    features, labels = _separable()
    keep = labels != 4
    model = ResidualClassifier("parameter", stages=20).fit(features[keep], labels[keep])
    probabilities = model.predict_probabilities(features)
    assert float(np.abs(probabilities[:, 4]).max()) == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert probabilities.sum(axis=1) == pytest.approx(
        np.ones(features.shape[0]), rel=1e-12, abs=0.0
    )


def test_predictions_carry_the_whole_probability_vector():
    features, labels = _separable()
    model = ResidualClassifier("parameter", stages=20).fit(features, labels)
    predictions = model.predict(features)
    assert len(predictions) == features.shape[0]
    for prediction, truth in zip(predictions, labels, strict=True):
        assert isinstance(prediction, Prediction)
        assert prediction.probabilities.shape == (5,)
        assert prediction.label == truth
        assert prediction.confidence == pytest.approx(
            float(prediction.probabilities.max()), rel=0.0, abs=0.0
        )
        assert prediction.label_name == PARAMETER_NAMES[truth]
        assert 0.0 <= prediction.margin <= 1.0


def test_there_is_no_bare_label_method():
    """The class deliberately exposes no point prediction. A caller has to take
    the label out of an object that also carries the confidence."""
    assert not hasattr(ResidualClassifier, "predict_labels")
    assert not hasattr(ResidualClassifier, "predict_label")


def test_the_abstention_flag_follows_the_confidence():
    """A thresholded boolean: the CONFIDENCE is asserted and the flag is only
    required to agree with it away from the threshold."""
    for value in (0.1, 0.3, 0.49, 0.51, 0.9):
        prediction = Prediction((0, 1), np.array([value, 1.0 - value]), 0, value)
        if abs(value - ABSTAIN_THRESHOLD) > 1e-9:
            assert prediction.abstains == (value < ABSTAIN_THRESHOLD)


def test_the_refusal_variant_names_its_sixth_class():
    prediction = Prediction(
        (0, 1, 2, 3, 4, UNIDENTIFIABLE), np.eye(6)[5], UNIDENTIFIABLE, 1.0
    )
    assert prediction.label_name == "UNIDENTIFIABLE"


def test_predict_one_takes_a_single_row():
    features, labels = _separable()
    model = ResidualClassifier("parameter", stages=20).fit(features, labels)
    assert model.predict_one(features[0]).label == labels[0]
    with pytest.raises(ValueError, match="single feature row"):
        model.predict_one(features)


def test_an_unfitted_model_refuses_to_predict():
    model = ResidualClassifier("parameter")
    assert not model.fitted
    with pytest.raises(ValueError, match="not been fitted"):
        model.predict_probabilities(np.zeros((1, 16)))
    with pytest.raises(ValueError, match="not been fitted"):
        _ = model.feature_importances


def test_fit_validates_its_labels():
    features, labels = _separable()
    model = ResidualClassifier("parameter", stages=10)
    with pytest.raises(ValueError, match="labels"):
        model.fit(features, labels[:-1])
    with pytest.raises(ValueError, match="cannot be trained on"):
        model.fit(features, np.full(features.shape[0], UNIDENTIFIABLE))
    with pytest.raises(ValueError, match="two distinct labels"):
        model.fit(features, np.zeros(features.shape[0], dtype=int))


def test_feature_importances_sum_to_one():
    features, labels = _separable()
    model = ResidualClassifier("parameter", stages=20).fit(features, labels)
    importances = model.feature_importances
    assert importances.shape == (16,)
    assert float(importances.sum()) == pytest.approx(1.0, rel=1e-9, abs=0.0)


def test_fitting_is_reproducible_at_the_declared_random_state():
    features, labels = _separable()
    first = ResidualClassifier("parameter", stages=20).fit(features, labels)
    second = ResidualClassifier("parameter", stages=20).fit(features, labels)
    assert first.predict_probabilities(features) == pytest.approx(
        second.predict_probabilities(features), rel=0.0, abs=0.0
    )
