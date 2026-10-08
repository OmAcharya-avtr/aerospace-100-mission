"""Tests for the learned drift classifier and its confidence output."""

from __future__ import annotations

import numpy as np
import pytest

from twininvalidate import (
    FEATURE_NAMES,
    N_FEATURES,
    AssetChange,
    DriftClassifier,
    StreamSpec,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
    simulate_residuals,
    window_features,
)
from twininvalidate.datasets import LabelledWindows, build_labelled_set


def test_classifier_fits_and_reports_its_sizes(classifier):
    assert classifier.is_fitted
    assert classifier.n_train_windows > 0
    assert classifier.n_calibration_windows > 0
    assert classifier.window == 50


def test_confidence_is_a_probability(classifier, in_control):
    f = window_features(in_control[:5, :300], classifier.window).reshape(-1, N_FEATURES)
    p = classifier.confidence(f)
    assert p.shape == (f.shape[0],)
    assert p.min() >= 0.0
    assert p.max() <= 1.0


def test_confidence_separates_the_classes(classifier, in_control, stepped):
    w = classifier.window
    p_ic = classifier.confidence(
        window_features(in_control[:30, :400], w).reshape(-1, N_FEATURES)
    )
    p_oc = classifier.confidence(window_features(stepped[:30, :400], w).reshape(-1, N_FEATURES))
    # The declared step is a 0.40-sigma mean shift, so a 50-sample window has
    # about 2.9 sigma of evidence: separation is expected, certainty is not.
    assert p_oc.mean() > p_ic.mean() + 0.2


def test_statistic_path_is_zero_until_the_window_fills(classifier, in_control):
    stat = classifier.statistic(in_control[:4, :300])
    assert stat.shape == (4, 300)
    assert np.all(stat[:, : classifier.window - 1] == 0.0)
    assert np.any(stat[:, classifier.window - 1 :] > 0.0)


def test_statistic_path_accepts_a_single_stream(classifier, in_control):
    stat = classifier.statistic(in_control[0])
    assert stat.shape == (1, in_control.shape[1])


def test_feature_importance_sums_to_one(classifier):
    imp = classifier.feature_importance()
    assert set(imp) == set(FEATURE_NAMES)
    assert sum(imp.values()) == pytest.approx(1.0, abs=1e-9)
    # The window mean is the obvious feature for the parameter step, which is
    # one of the three scenarios in the training set, so it should carry real
    # weight. This is a description of the fitted forest, not a physical claim.
    assert imp["mean"] > 0.1


def test_unfitted_classifier_refuses_to_predict():
    clf = DriftClassifier()
    assert not clf.is_fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.confidence(np.zeros((2, N_FEATURES)))
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.statistic(np.zeros((1, 200)))
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.feature_importance()


def test_fit_validates_its_feature_matrices():
    clf = DriftClassifier()
    x = np.zeros((10, N_FEATURES))
    y = np.array([0] * 5 + [1] * 5)
    with pytest.raises(ValueError, match="x_train must have shape"):
        clf.fit(np.zeros((10, 3)), y, x, y)
    with pytest.raises(ValueError, match="x_train is empty"):
        clf.fit(np.zeros((0, N_FEATURES)), np.zeros(0), x, y)
    with pytest.raises(ValueError, match="must be finite"):
        clf.fit(np.full((10, N_FEATURES), np.nan), y, x, y)


def test_fit_validates_its_labels():
    clf = DriftClassifier()
    x = np.zeros((10, N_FEATURES))
    y = np.array([0] * 5 + [1] * 5)
    with pytest.raises(ValueError, match="y_train must have shape"):
        clf.fit(x, np.zeros(3), x, y)
    with pytest.raises(ValueError, match="only 0 and 1"):
        clf.fit(x, np.full(10, 2), x, y)
    with pytest.raises(ValueError, match="y_train must contain both classes"):
        clf.fit(x, np.zeros(10, dtype=int), x, y)
    with pytest.raises(ValueError, match="y_calibration must contain both classes"):
        clf.fit(x, y, x, np.ones(10, dtype=int))


def test_brier_score_known_answers():
    # Perfect forecast: 0. Always 0.5 on a balanced pair: 0.25.
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert brier_score(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == pytest.approx(0.25)
    # Confident and wrong: (0.9-0)^2 = 0.81 and (0.1-1)^2 = 0.81, mean 0.81.
    assert brier_score(np.array([0.9, 0.1]), np.array([0.0, 1.0])) == pytest.approx(0.81)


def test_brier_score_validates_its_inputs():
    with pytest.raises(ValueError, match="shapes must match"):
        brier_score(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="empty forecast"):
        brier_score(np.zeros(0), np.zeros(0))


def test_reliability_diagram_known_answer():
    # Four forecasts at 0.05 and four at 0.95; the first bin holds the low
    # ones with observed frequency 0, the last the high ones with 1.
    p = np.array([0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    bins = reliability_diagram(p, y, n_bins=10)
    assert len(bins) == 2
    assert bins[0].count == 4
    assert bins[0].mean_confidence == pytest.approx(0.05)
    assert bins[0].observed_frequency == pytest.approx(0.0)
    assert bins[1].observed_frequency == pytest.approx(1.0)
    # ECE = (4/8)|0.05 - 0| + (4/8)|0.95 - 1| = 0.025 + 0.025 = 0.05
    assert expected_calibration_error(bins) == pytest.approx(0.05)


def test_reliability_diagram_omits_empty_bins():
    bins = reliability_diagram(np.full(10, 0.42), np.ones(10, dtype=int), n_bins=10)
    assert len(bins) == 1
    assert bins[0].count == 10


def test_reliability_diagram_includes_the_upper_edge():
    bins = reliability_diagram(np.array([1.0]), np.array([1]), n_bins=4)
    assert len(bins) == 1
    assert bins[0].count == 1


def test_reliability_helpers_validate_their_inputs():
    with pytest.raises(ValueError, match="shapes must match"):
        reliability_diagram(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="n_bins must be at least 2"):
        reliability_diagram(np.zeros(3), np.zeros(3), n_bins=1)
    with pytest.raises(ValueError, match="no populated bins"):
        expected_calibration_error([])


def test_labelled_set_is_balanced_and_split_by_run():
    data = build_labelled_set(
        9001, 9002, n_runs_in_control=12, n_runs_per_scenario=4, n_samples=200, stride=20
    )
    assert isinstance(data, LabelledWindows)
    assert data.y.mean() == pytest.approx(0.5)
    assert len(data) == data.x.shape[0]
    # Every run index appears with exactly one label, which is what makes a
    # by-run split a clean split.
    for run in np.unique(data.run):
        assert len(set(data.y[data.run == run].tolist())) == 1


def test_labelled_set_validates_its_inputs():
    with pytest.raises(ValueError, match="different seeds"):
        build_labelled_set(1, 1)
    with pytest.raises(ValueError, match="stride must be at least 1"):
        build_labelled_set(1, 2, stride=0)


def test_labelled_windows_validates_its_shapes():
    with pytest.raises(ValueError, match="x must have shape"):
        LabelledWindows(x=np.zeros((3, 2)), y=np.zeros(3), run=np.zeros(3))
    with pytest.raises(ValueError, match="must agree on the number of rows"):
        LabelledWindows(x=np.zeros((3, N_FEATURES)), y=np.zeros(2), run=np.zeros(3))


def test_confidence_is_higher_on_a_larger_change(classifier):
    w = classifier.window
    small = simulate_residuals(
        StreamSpec(
            change=AssetChange("parameter_step", 0, -0.004), n_runs=20, n_samples=400, seed=7001
        )
    )
    large = simulate_residuals(
        StreamSpec(
            change=AssetChange("parameter_step", 0, -0.03), n_runs=20, n_samples=400, seed=7002
        )
    )
    p_small = classifier.confidence(window_features(small, w).reshape(-1, N_FEATURES)).mean()
    p_large = classifier.confidence(window_features(large, w).reshape(-1, N_FEATURES)).mean()
    assert p_large > p_small
