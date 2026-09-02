"""Learned classifier: contract, determinism, confidence and validation.

The tests here use a tiny synthetic feature set rather than simulated runs, so
they check the model's *contract* -- shapes, class ordering, probability
normalisation, reproducibility -- in a second.  Its measured performance
against the classical baselines lives in ``validation/`` and in the
integration test, which is where it belongs.
"""

from __future__ import annotations

import numpy as np
import pytest

from fdiscope.classifier import FaultClassifier
from fdiscope.faults import FAULT_CLASSES, FaultType, class_index
from fdiscope.features import N_FEATURES, feature_names


def toy_data(n_per_class: int = 30, seed: int = 0):
    """Separable clusters, one per class, in feature space."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for i in range(len(FAULT_CLASSES)):
        centre = np.zeros(N_FEATURES)
        centre[i % N_FEATURES] = 5.0 * (1 + i)
        rows.append(centre + 0.1 * rng.standard_normal((n_per_class, N_FEATURES)))
        labels += [i] * n_per_class
    return np.vstack(rows), np.asarray(labels)


class TestConstruction:
    @pytest.mark.parametrize(
        "kwargs", [{"n_estimators": 0}, {"max_depth": 0}, {"min_samples_leaf": 0}]
    )
    def test_rejects_bad_parameters(self, kwargs):
        with pytest.raises(ValueError):
            FaultClassifier(**kwargs)

    def test_accepts_unbounded_depth(self):
        assert FaultClassifier(max_depth=None) is not None

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            FaultClassifier().predict_proba(np.zeros((1, N_FEATURES)))

    def test_feature_importances_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            FaultClassifier().feature_importances()


class TestFitValidation:
    def test_rejects_wrong_feature_width(self):
        clf = FaultClassifier(n_estimators=5)
        with pytest.raises(ValueError, match=f"must be \\(n, {N_FEATURES}\\)"):
            clf.fit(np.zeros((10, 3)), np.zeros(10, dtype=int))

    def test_rejects_length_mismatch(self):
        clf = FaultClassifier(n_estimators=5)
        with pytest.raises(ValueError, match="feature rows and"):
            clf.fit(np.zeros((10, N_FEATURES)), np.zeros(5, dtype=int))

    def test_rejects_non_finite_features(self):
        x = np.zeros((10, N_FEATURES))
        x[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            FaultClassifier(n_estimators=5).fit(x, np.zeros(10, dtype=int))

    def test_rejects_out_of_range_labels(self):
        with pytest.raises(ValueError, match=r"\[0, 8\)"):
            FaultClassifier(n_estimators=5).fit(np.zeros((4, N_FEATURES)), [0, 1, 2, 99])

    def test_rejects_a_single_row(self):
        with pytest.raises(ValueError, match="at least 2 training rows"):
            FaultClassifier(n_estimators=5).fit(np.zeros((1, N_FEATURES)), [0])


@pytest.fixture(scope="module")
def fitted():
    x, y = toy_data()
    return FaultClassifier(n_estimators=40, random_state=0).fit(x, y), x, y


class TestPredictions:

    def test_probabilities_have_eight_columns_and_sum_to_one(self, fitted):
        clf, x, _ = fitted
        proba = clf.predict_proba(x[:20])
        assert proba.shape == (20, len(FAULT_CLASSES))
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_missing_classes_get_exactly_zero(self):
        # Train on two classes only; the other six columns must be zero and the
        # array must still have eight columns in canonical order.
        x, y = toy_data(n_per_class=10)
        mask = y < 2
        clf = FaultClassifier(n_estimators=20, random_state=0).fit(x[mask], y[mask])
        proba = clf.predict_proba(x[:5])
        assert proba.shape[1] == len(FAULT_CLASSES)
        assert np.allclose(proba[:, 2:], 0.0)

    def test_separable_clusters_are_classified_correctly(self, fitted):
        clf, x, y = fitted
        pred = clf.predict_with_confidence(x)
        assert np.mean([class_index(c) == t for c, t in zip(pred.classes, y, strict=True)]) > 0.95

    def test_confidence_is_the_winning_probability(self, fitted):
        clf, x, _ = fitted
        pred = clf.predict_with_confidence(x[:30])
        assert np.allclose(pred.confidence, pred.proba.max(axis=1))
        assert np.all(pred.confidence > 0.0) and np.all(pred.confidence <= 1.0)

    def test_detection_score_is_one_minus_p_none(self, fitted):
        clf, x, _ = fitted
        proba = clf.predict_proba(x[:30])
        none_col = FAULT_CLASSES.index(FaultType.NONE)
        assert np.allclose(clf.detection_score(x[:30]), 1.0 - proba[:, none_col])

    def test_detection_score_is_low_on_healthy_rows(self, fitted):
        clf, x, y = fitted
        healthy = x[y == class_index(FaultType.NONE)]
        assert float(np.mean(clf.detection_score(healthy))) < 0.2

    def test_prediction_rejects_wrong_width(self, fitted):
        clf, _, _ = fitted
        with pytest.raises(ValueError, match=f"must be \\(n, {N_FEATURES}\\)"):
            clf.predict_proba(np.zeros((3, 5)))


class TestReproducibility:
    def test_same_seed_gives_identical_probabilities(self):
        x, y = toy_data(seed=1)
        a = FaultClassifier(n_estimators=30, random_state=7).fit(x, y).predict_proba(x[:50])
        b = FaultClassifier(n_estimators=30, random_state=7).fit(x, y).predict_proba(x[:50])
        assert np.array_equal(a, b)

    def test_different_seeds_can_differ(self):
        x, y = toy_data(n_per_class=8, seed=2)
        a = FaultClassifier(n_estimators=5, random_state=0).fit(x, y).predict_proba(x)
        b = FaultClassifier(n_estimators=5, random_state=99).fit(x, y).predict_proba(x)
        assert not np.array_equal(a, b)

    def test_fit_returns_self(self):
        x, y = toy_data(n_per_class=5)
        clf = FaultClassifier(n_estimators=5)
        assert clf.fit(x, y) is clf


class TestFeatureImportances:
    def test_keys_are_the_feature_names_and_sum_to_one(self):
        x, y = toy_data(n_per_class=20)
        clf = FaultClassifier(n_estimators=30, random_state=0).fit(x, y)
        importances = clf.feature_importances()
        assert tuple(importances) == feature_names()
        assert np.isclose(sum(importances.values()), 1.0)
        assert all(v >= 0.0 for v in importances.values())
