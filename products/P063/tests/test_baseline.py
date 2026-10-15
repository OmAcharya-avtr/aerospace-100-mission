"""The declared threshold-logic baseline."""

from __future__ import annotations

import numpy as np
import pytest

from safemodetree import (
    NOMINAL_LABEL,
    ThresholdLogicClassifier,
    make_observations,
    symptom_word,
)


def test_nominal_vector_gives_the_all_zero_word_and_nominal_label(fmea):
    classifier = ThresholdLogicClassifier(fmea)
    nominal = np.array(fmea.observability.nominal_vector())
    assert symptom_word(fmea, nominal) == (0,) * 7
    label, confidence = classifier.classify_word(symptom_word(fmea, nominal))
    assert label == NOMINAL_LABEL
    assert confidence == pytest.approx(1.0)


def test_unique_signature_gives_confidence_one(fmea):
    classifier = ThresholdLogicClassifier(fmea)
    word = fmea.observability.signature("bat_cell_short")
    assert classifier.classify_word(word) == ("bat_cell_short", 1.0)


def test_ambiguity_class_gives_confidence_one_over_two(fmea):
    classifier = ThresholdLogicClassifier(fmea)
    word = fmea.observability.signature("st_blinding")
    label, confidence = classifier.classify_word(word)
    assert label in ("st_blinding", "st_latchup")
    assert confidence == pytest.approx(0.5)


def test_hamming_resolution_known_answer(fmea):
    # The word with only attitude_residual asserted is the declared signature
    # of no fault. Its Hamming-1 neighbours among the declared signatures are
    # rw_motor_open (attitude + torque error), st_blinding and st_latchup
    # (attitude + tracker valid). Three faults tie at distance 1, so the
    # declared confidence is 1 / (3 * (1 + 1)) = 1/6.
    classifier = ThresholdLogicClassifier(fmea)
    index = fmea.observability.channel_ids.index("attitude_residual")
    word = tuple(1 if position == index else 0 for position in range(7))
    label, confidence = classifier.classify_word(word)
    assert label in ("rw_motor_open", "st_blinding", "st_latchup")
    assert confidence == pytest.approx(1.0 / 6.0, abs=1e-15)


def test_classifier_classes_list_nominal_first(fmea):
    classifier = ThresholdLogicClassifier(fmea)
    assert classifier.classes[0] == NOMINAL_LABEL
    assert set(classifier.classes[1:]) == set(fmea.fault_ids)


def test_predict_and_confidence_agree_with_predict_with_confidence(fmea):
    classifier = ThresholdLogicClassifier(fmea)
    observations = make_observations(fmea, n_per_fault=30, noise_scale=2.0, seed=17)
    labels, confidence = classifier.predict_with_confidence(observations.features)
    assert np.array_equal(labels, classifier.predict(observations.features))
    assert np.allclose(confidence, classifier.confidence(observations.features))


def test_margin_sd_only_turns_symptom_bits_off(fmea):
    observations = make_observations(fmea, n_per_fault=40, noise_scale=2.0, seed=21)
    tight = np.array(
        [symptom_word(fmea, row, 0.0) for row in observations.features], dtype=int
    )
    loose = np.array(
        [symptom_word(fmea, row, 1.5) for row in observations.features], dtype=int
    )
    assert np.all(loose <= tight)


def test_raising_margin_lowers_the_false_alarm_rate(fmea):
    observations = make_observations(fmea, n_per_fault=200, noise_scale=2.5, seed=23)
    nominal = observations.features[observations.labels == NOMINAL_LABEL]
    strict = ThresholdLogicClassifier(fmea, margin_sd=0.0).predict(nominal)
    lenient = ThresholdLogicClassifier(fmea, margin_sd=2.0).predict(nominal)
    assert np.mean(lenient != NOMINAL_LABEL) <= np.mean(strict != NOMINAL_LABEL)


def test_symptom_word_rejects_wrong_width(fmea):
    with pytest.raises(ValueError, match="expected 7"):
        symptom_word(fmea, np.zeros(3))


def test_margin_sd_type_is_checked(fmea):
    with pytest.raises(TypeError, match="real number"):
        ThresholdLogicClassifier(fmea, margin_sd="wide")  # type: ignore[arg-type]


def test_at_low_noise_the_baseline_identifies_unambiguous_faults(fmea):
    classifier = ThresholdLogicClassifier(fmea)
    observations = make_observations(fmea, n_per_fault=200, noise_scale=1.0, seed=29)
    unambiguous = np.isin(
        observations.labels, ["rw_bearing_drag", "rw_motor_open", "rw_tach_stale",
                              "bat_cell_short"]
    )
    predicted = classifier.predict(observations.features[unambiguous])
    # With noise_scale 1.0 every declared threshold is at least 3.6 sigma from
    # both the nominal and the faulted mean, so the per-channel error rate is
    # below 2e-4 and 0.98 is a loose floor.
    assert np.mean(predicted == observations.labels[unambiguous]) > 0.98
