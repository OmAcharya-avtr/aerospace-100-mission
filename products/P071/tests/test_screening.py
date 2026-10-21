"""The learned screen, its metric, its uncertainty output and the comparison."""

from __future__ import annotations

import numpy as np
import pytest

from conjprob.dataset import generate_population
from conjprob.screening import (
    ScreeningClassifier,
    brier_score,
    compare_screens,
    recall_at_fpr,
    reliability_table,
)


def test_recall_at_fpr_hand_calculated():
    # Scores 1..10 with the top four positive. Negatives are 1..6. The
    # false-positive rate at threshold 6 is 1/6 = 0.1667, at threshold 5 it is
    # 2/6 = 0.3333, so 6 is the smallest threshold meeting a 0.2 target.
    # Flagged: 6, 7, 8, 9, 10. All four positives are flagged, so recall is 1
    # and precision is 4/5.
    scores = np.arange(1.0, 11.0)
    labels = np.array([False] * 6 + [True] * 4)
    point = recall_at_fpr(labels, scores, 0.2)
    assert point.threshold == pytest.approx(6.0, rel=0.0)
    assert point.fpr_achieved == pytest.approx(1.0 / 6.0, rel=1e-12)
    assert point.recall == pytest.approx(1.0, rel=0.0)
    assert point.precision == pytest.approx(4.0 / 5.0, rel=1e-12)
    assert point.n_positive == 4
    assert point.n_negative == 6


def test_recall_at_fpr_achieved_rate_never_exceeds_the_target():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=4000)
    labels = rng.random(4000) < 0.3
    for target in (0.01, 0.05, 0.2):
        point = recall_at_fpr(labels, scores, target)
        assert point.fpr_achieved <= target + 1e-12


def test_recall_at_fpr_is_monotone_in_the_target():
    rng = np.random.default_rng(1)
    labels = rng.random(4000) < 0.3
    scores = labels * 1.0 + rng.normal(scale=1.0, size=4000)
    recalls = [recall_at_fpr(labels, scores, t).recall for t in (0.01, 0.05, 0.2, 0.5)]
    assert all(b >= a for a, b in zip(recalls, recalls[1:], strict=False))


@pytest.mark.parametrize(
    "labels, scores, target, message",
    [
        ([True, False], [1.0], 0.1, "same length"),
        ([True, True], [1.0, 2.0], 0.1, "both classes"),
        ([True, False], [1.0, 2.0], 0.0, "fpr_target"),
        ([True, False], [1.0, 2.0], 1.0, "fpr_target"),
    ],
)
def test_recall_at_fpr_rejects_bad_input(labels, scores, target, message):
    with pytest.raises(ValueError, match=message):
        recall_at_fpr(labels, scores, target)


def test_brier_score_hand_calculated():
    # Predictions 0.9, 0.2 against labels 1, 0: mean of 0.01 and 0.04 = 0.025.
    assert brier_score([1, 0], [0.9, 0.2]) == pytest.approx(0.025, rel=1e-13)


def test_brier_score_of_the_base_rate_matches_the_variance():
    labels = np.array([1] * 30 + [0] * 70)
    rate = labels.mean()
    assert brier_score(labels, np.full(100, rate)) == pytest.approx(
        rate * (1.0 - rate), rel=1e-13
    )


def test_brier_score_rejects_bad_input():
    with pytest.raises(ValueError, match="same length"):
        brier_score([1, 0], [0.5])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        brier_score([1, 0], [1.5, 0.5])


def test_reliability_table_bins_and_counts():
    probabilities = np.array([0.05, 0.15, 0.95, 0.96])
    labels = np.array([0, 0, 1, 1])
    rows = reliability_table(labels, probabilities, n_bins=10)
    assert sum(row[3] for row in rows) == 4
    assert rows[-1][2] == pytest.approx(1.0, rel=0.0)
    assert rows[0][2] == pytest.approx(0.0, abs=0.0)


def test_reliability_table_rejects_too_few_bins():
    with pytest.raises(ValueError, match="n_bins"):
        reliability_table([1, 0], [0.5, 0.5], n_bins=1)


@pytest.fixture(scope="module")
def trained():
    train = generate_population(6000, seed=0)
    test = generate_population(3000, seed=1000)
    model = ScreeningClassifier(n_estimators=60, random_state=0).fit(
        train.features, train.labels
    )
    return model, train, test


def test_classifier_requires_fit_before_prediction():
    model = ScreeningClassifier(n_estimators=5)
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.predict_proba(np.zeros((1, 10)))


def test_classifier_rejects_bad_training_input():
    model = ScreeningClassifier(n_estimators=5)
    with pytest.raises(ValueError, match=r"shape \(n, 10\)"):
        model.fit(np.zeros((4, 3)), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(np.zeros((4, 10)), [0, 1])
    with pytest.raises(ValueError, match="both classes"):
        model.fit(np.zeros((4, 10)), [1, 1, 1, 1])


@pytest.mark.parametrize(
    "kwargs, message",
    [({"n_estimators": 0}, "n_estimators"), ({"min_samples_leaf": 0}, "min_samples_leaf")],
)
def test_classifier_rejects_bad_hyperparameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ScreeningClassifier(**kwargs)


def test_probabilities_are_in_range_and_training_is_deterministic(trained):
    model, train, test = trained
    first = model.predict_proba(test.features)
    assert np.all((first >= 0.0) & (first <= 1.0))
    again = ScreeningClassifier(n_estimators=60, random_state=0).fit(
        train.features, train.labels
    )
    # Two identical fits agree to within rounding, not bit for bit: the forest
    # is averaged across two worker processes and the summation order is not
    # fixed. The measured worst disagreement here is of order 1e-16.
    np.testing.assert_allclose(again.predict_proba(test.features), first, rtol=1e-12, atol=1e-15)


def test_uncertainty_output_is_present_and_non_negative(trained):
    model, _, test = trained
    mean, dispersion = model.predict_with_uncertainty(test.features[:500])
    assert mean.shape == dispersion.shape == (500,)
    assert np.all(dispersion >= 0.0)
    np.testing.assert_allclose(mean, model.predict_proba(test.features[:500]), rtol=1e-12)


def test_uncertainty_is_larger_where_the_model_is_wrong(trained):
    model, _, test = trained
    mean, dispersion = model.predict_with_uncertainty(test.features)
    error = np.abs(mean - test.labels.astype(float))
    wrong = error > 0.5
    assert wrong.sum() > 0
    assert dispersion[wrong].mean() > dispersion[~wrong].mean()


def test_feature_importances_sum_to_one(trained):
    model, _, _ = trained
    importances = model.feature_importances()
    assert set(importances) == set(model.feature_names)
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-9)


def test_compare_screens_returns_every_screen_at_every_operating_point(trained):
    model, _, test = trained
    comparison = compare_screens(
        test.labels, model.score_samples(test.features), test.features, (0.01, 0.05)
    )
    assert set(comparison) == {"miss_distance", "mahalanobis", "small_body", "learned"}
    for points in comparison.values():
        assert [point.fpr_target for point in points] == [0.01, 0.05]


def test_learned_screen_beats_the_two_specified_baselines_on_the_nominal_regime(trained):
    # Measured, not assumed. The product specification predicted that a
    # Mahalanobis threshold would match the learned model; on this generator it
    # does not, because the Mahalanobis distance is blind to R / sigma_eff,
    # which varies over more than four decades here. The measurement is in
    # validation/validate_screening_output.txt and the finding is reported in
    # README.md and MODEL_CARD.md as a prediction that did not hold.
    model, _, test = trained
    comparison = compare_screens(
        test.labels, model.score_samples(test.features), test.features, (0.01,)
    )
    learned = comparison["learned"][0].recall
    assert learned > comparison["mahalanobis"][0].recall
    assert learned > comparison["miss_distance"][0].recall


def test_closed_form_screen_is_close_to_the_learned_model(trained):
    # The one-line small-hard-body closed form, using exactly the features the
    # classifier is given, comes within a few points of it. This is the honest
    # comparison and it is a test so that a future change which quietly loses
    # it fails here.
    model, _, test = trained
    comparison = compare_screens(
        test.labels, model.score_samples(test.features), test.features, (0.01,)
    )
    learned = comparison["learned"][0].recall
    closed_form = comparison["small_body"][0].recall
    assert closed_form > 0.9
    assert closed_form > learned - 0.1
