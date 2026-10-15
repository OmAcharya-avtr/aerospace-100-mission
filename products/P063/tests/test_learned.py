"""The learned classifier and the ceiling the declaration puts on it."""

from __future__ import annotations

import numpy as np
import pytest

from safemodetree import (
    NOMINAL_LABEL,
    LearnedFaultClassifier,
    accuracy_ceiling,
    make_observations,
    split_observations,
)


@pytest.fixture(scope="module")
def fitted(fmea):
    observations = make_observations(fmea, n_per_fault=150, noise_scale=2.0, seed=41)
    split = split_observations(observations, seed=42)
    model = LearnedFaultClassifier(calibration="isotonic", random_state=0)
    model.fit(split.train.features, split.train.labels)
    return model, split


def test_classes_cover_nominal_and_every_fault(fitted, fmea):
    model, _split = fitted
    assert set(model.classes) == {NOMINAL_LABEL, *fmea.fault_ids}
    assert model.is_fitted


def test_posterior_rows_sum_to_one(fitted):
    model, split = fitted
    proba = model.predict_proba(split.test.features)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_confidence_is_the_top_label_probability(fitted):
    model, split = fitted
    proba = model.predict_proba(split.test.features)
    assert np.allclose(model.confidence(split.test.features), proba.max(axis=1))


def test_predict_with_confidence_matches_the_separate_calls(fitted):
    model, split = fitted
    labels, confidence = model.predict_with_confidence(split.test.features)
    assert np.array_equal(labels, model.predict(split.test.features))
    assert np.allclose(confidence, model.confidence(split.test.features))


def test_fault_probability_is_one_minus_the_nominal_posterior(fitted):
    model, split = fitted
    proba = model.predict_proba(split.test.features)
    column = model.classes.index(NOMINAL_LABEL)
    assert np.allclose(
        model.fault_probability(split.test.features), 1.0 - proba[:, column], atol=1e-12
    )


def test_confidence_inside_an_ambiguity_class_cannot_exceed_one_half(fitted):
    model, split = fitted
    labels = np.asarray(split.test.labels).astype(str)
    pair = np.isin(labels, ["st_blinding", "st_latchup"])
    proba = model.predict_proba(split.test.features[pair])
    blinding = model.classes.index("st_blinding")
    latchup = model.classes.index("st_latchup")
    # The two classes have identical observation distributions by construction,
    # so neither posterior can systematically exceed the other. The mean of the
    # larger of the two is bounded above by 0.5 plus sampling noise; 0.60 is a
    # loose bound that still fails if the model separates them.
    assert float(np.mean(np.maximum(proba[:, blinding], proba[:, latchup]))) < 0.60


def test_accuracy_ceiling_hand_calculation(fmea):
    observations = make_observations(fmea, n_per_fault=100, nominal_fraction=0.5, seed=1)
    # Hand arithmetic: 800 nominal and 100 of each of 8 faults, 1600 rows.
    # Four faults sit in ambiguity classes of size 2 and are capped at 1/2;
    # the other four and the nominal class are uncapped.
    #   (800 + 4*100 + 4*100/2) / 1600 = (800 + 400 + 200) / 1600 = 0.875
    assert accuracy_ceiling(fmea, observations) == pytest.approx(0.875, abs=1e-12)


def test_accuracy_ceiling_is_one_without_ambiguity(minimal):
    observations = make_observations(minimal, n_per_fault=20, seed=2)
    assert accuracy_ceiling(minimal, observations) == pytest.approx(1.0, abs=1e-12)


def test_uncalibrated_and_sigmoid_variants_both_fit(fmea):
    observations = make_observations(fmea, n_per_fault=80, noise_scale=2.0, seed=43)
    split = split_observations(observations, seed=44)
    for method in ("none", "sigmoid"):
        model = LearnedFaultClassifier(calibration=method, cv=3).fit(
            split.train.features, split.train.labels
        )
        proba = model.predict_proba(split.test.features[:20])
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_fit_rejects_mismatched_lengths():
    model = LearnedFaultClassifier(calibration="none")
    with pytest.raises(ValueError, match="different lengths"):
        model.fit(np.zeros((3, 2)), np.array(["a", "b"]))
    with pytest.raises(ValueError, match="must be 2-D"):
        model.fit(np.zeros(3), np.array(["a", "b", "c"]))


def test_unfitted_classes_are_empty():
    assert LearnedFaultClassifier().classes == ()
    assert not LearnedFaultClassifier().is_fitted


def test_accuracy_ceiling_rejects_an_empty_set(fmea):
    observations = make_observations(fmea, n_per_fault=5, seed=1)
    empty = observations.__class__(
        features=observations.features[:0],
        labels=observations.labels[:0],
        channel_ids=observations.channel_ids,
        class_names=observations.class_names,
        noise_scale=1.0,
    )
    with pytest.raises(ValueError, match="no observations"):
        accuracy_ceiling(fmea, empty)
