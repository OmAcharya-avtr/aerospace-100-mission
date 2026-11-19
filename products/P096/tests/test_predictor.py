"""The learned component: uncertainty output, validation, and the scoring harness."""

from __future__ import annotations

import numpy as np
import pytest

from handoverloss.corpus import FEATURE_COLUMNS, station_pair_split, time_split
from handoverloss.predictor import (
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_N_ESTIMATORS,
    HYPERPARAMETER_CANDIDATES,
    HandoverLossPredictor,
    baseline_predictions,
    score_baselines,
    score_learned,
)
from handoverloss.sequence import MECHANISMS


@pytest.fixture(scope="module")
def fitted(request):
    from handoverloss.corpus import load_corpus

    loaded = load_corpus()
    split = station_pair_split(loaded)
    model = HandoverLossPredictor(n_estimators=60).fit(
        loaded.features[split.train], loaded.targets[split.train]
    )
    return loaded, split, model


def test_selected_hyperparameters_are_in_the_candidate_grid():
    assert (
        DEFAULT_MAX_DEPTH,
        DEFAULT_MIN_SAMPLES_LEAF,
        DEFAULT_N_ESTIMATORS,
    ) in HYPERPARAMETER_CANDIDATES


def test_predict_returns_an_uncertainty_output_not_a_point_estimate(fitted):
    corpus, split, model = fitted
    prediction = model.predict(corpus.features[split.test])
    count = split.test.size
    assert prediction.loss_bits.shape == (count, 3)
    assert prediction.low_bits.shape == (count, 3)
    assert prediction.high_bits.shape == (count, 3)
    assert prediction.log_sd.shape == (count, 3)
    # The uncertainty output is non-negative and not identically zero.
    assert np.all(prediction.log_sd >= 0.0)
    assert float(prediction.log_sd.max()) > 0.0
    assert np.all(prediction.low_bits <= prediction.loss_bits + 1e-6)
    assert np.all(prediction.high_bits >= prediction.loss_bits - 1e-6)


def test_predictions_are_never_negative(fitted):
    corpus, split, model = fitted
    prediction = model.predict(corpus.features[split.test])
    assert np.all(prediction.loss_bits >= 0.0)
    assert np.all(prediction.low_bits >= 0.0)


def test_dominance_confidence_is_a_vote_share(fitted):
    corpus, split, model = fitted
    prediction = model.predict(corpus.features[split.test])
    assert np.all(prediction.dominance_confidence >= 0.0)
    assert np.all(prediction.dominance_confidence <= 1.0)
    # The winning option cannot have less than a third of the votes when it is
    # the argmax of the across-tree mean of three options in only the loosest
    # sense, so the floor asserted is the trivial one; what matters is that the
    # confidence is not a constant.
    assert float(prediction.dominance_confidence.std()) > 0.0


def test_dominant_names_match_the_indices(fitted):
    corpus, split, model = fitted
    prediction = model.predict(corpus.features[split.test])
    names = prediction.dominant_names()
    assert len(names) == split.test.size
    for row, name in enumerate(names):
        assert name == MECHANISMS[int(prediction.dominant_index[row])]


def test_a_wider_band_covers_at_least_as_much(fitted):
    corpus, split, model = fitted
    narrow = model.predict(corpus.features[split.test], confidence=0.50)
    wide = model.predict(corpus.features[split.test], confidence=0.99)
    assert np.all(wide.low_bits <= narrow.low_bits + 1e-6)
    assert np.all(wide.high_bits >= narrow.high_bits - 1e-6)


def test_forest_cannot_predict_outside_its_training_target_range(fitted):
    # A property of the estimator, not of its API: every leaf holds a mean of
    # training targets, so a forest average lies inside the training range.
    corpus, split, model = fitted
    prediction = model.predict(corpus.features[split.test])
    ceiling = float(corpus.targets[split.train].max())
    assert float(prediction.loss_bits.max()) <= ceiling * (1.0 + 1e-9)


def test_refit_with_the_same_seed_is_reproducible(fitted):
    corpus, split, _ = fitted
    first = HandoverLossPredictor(n_estimators=40).fit(
        corpus.features[split.train], corpus.targets[split.train]
    )
    second = HandoverLossPredictor(n_estimators=40).fit(
        corpus.features[split.train], corpus.targets[split.train]
    )
    a = first.predict(corpus.features[split.test])
    b = second.predict(corpus.features[split.test])
    assert np.array_equal(a.loss_bits, b.loss_bits)


def test_feature_importances_cover_every_feature(fitted):
    _, _, model = fitted
    importances = model.feature_importances()
    assert set(importances) == set(FEATURE_COLUMNS)
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-9, abs=0.0)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        HandoverLossPredictor().predict(np.zeros((3, len(FEATURE_COLUMNS))))


def test_feature_importances_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        HandoverLossPredictor().feature_importances()


def test_fitted_flag(fitted):
    corpus, split, model = fitted
    assert model.fitted
    assert not HandoverLossPredictor().fitted


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (dict(n_estimators=5), "n_estimators"),
        (dict(max_depth=0), "max_depth"),
        (dict(min_samples_leaf=0), "min_samples_leaf"),
    ],
)
def test_invalid_hyperparameters_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HandoverLossPredictor(**kwargs)


def test_fit_rejects_a_wrong_feature_width():
    with pytest.raises(ValueError, match=r"shape \(n, 18\)"):
        HandoverLossPredictor().fit(np.zeros((30, 3)), np.zeros((30, 3)))


def test_fit_rejects_a_wrong_target_width():
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        HandoverLossPredictor().fit(
            np.zeros((30, len(FEATURE_COLUMNS))), np.zeros((30, 2))
        )


def test_fit_rejects_mismatched_rows():
    with pytest.raises(ValueError, match="rows"):
        HandoverLossPredictor().fit(
            np.zeros((30, len(FEATURE_COLUMNS))), np.zeros((29, 3))
        )


def test_fit_rejects_too_few_rows():
    with pytest.raises(ValueError, match="at least 20 rows"):
        HandoverLossPredictor().fit(
            np.zeros((10, len(FEATURE_COLUMNS))), np.zeros((10, 3))
        )


def test_fit_rejects_negative_targets():
    with pytest.raises(ValueError, match="non-negative"):
        HandoverLossPredictor().fit(
            np.zeros((30, len(FEATURE_COLUMNS))), -np.ones((30, 3))
        )


def test_fit_rejects_non_finite_features():
    features = np.zeros((30, len(FEATURE_COLUMNS)))
    features[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        HandoverLossPredictor().fit(features, np.zeros((30, 3)))


def test_predict_rejects_a_wrong_confidence(fitted):
    _, _, model = fitted
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        model.predict(np.zeros((3, len(FEATURE_COLUMNS))), confidence=1.0)


def test_predict_rejects_a_wrong_feature_width(fitted):
    _, _, model = fitted
    with pytest.raises(ValueError, match=r"shape \(n, 18\)"):
        model.predict(np.zeros((3, 4)))


def test_baseline_one_puts_everything_in_the_acquisition_column(corpus):
    split = time_split(corpus)
    predictions = baseline_predictions(corpus, split.test)
    acquisition_only = predictions["acquisition_only"]
    assert np.all(acquisition_only[:, 1] == 0.0)
    assert np.all(acquisition_only[:, 2] == 0.0)
    # Its dominance prediction is therefore always "acquisition", which is the
    # assumption being refuted stated as a predictor.
    assert set(np.argmax(acquisition_only, axis=1).tolist()) <= {0}


def test_baselines_see_no_realised_duration(corpus):
    # Both analytic baselines are evaluated from pre-handover state alone, so
    # the comparison with the learned component is on the same information.
    split = time_split(corpus)
    predictions = baseline_predictions(corpus, split.test)
    realised = np.asarray(corpus.table["realised_acquisition_s"], dtype=float)[split.test]
    # Perturbing the realised durations cannot change a baseline prediction,
    # because the baselines never read that column. Checked by recomputing after
    # overwriting it in a copy of the table.
    altered = dict(corpus.table)
    altered["realised_acquisition_s"] = realised * 3.0
    from handoverloss.corpus import Corpus

    copy = Corpus(
        features=corpus.features,
        targets=corpus.targets,
        pair_id=corpus.pair_id,
        epoch=corpus.epoch,
        table=altered,
        sha256=corpus.sha256,
    )
    again = baseline_predictions(copy, split.test)
    assert np.array_equal(predictions["fluid"], again["fluid"])
    assert np.array_equal(
        predictions["acquisition_only"], again["acquisition_only"]
    )


def test_score_baselines_returns_both_in_order(corpus):
    split = time_split(corpus)
    scores = score_baselines(corpus, split, resamples=100)
    assert [s.predictor for s in scores] == [
        "baseline1_acquisition_only",
        "baseline2_analytic_fluid",
    ]
    for score in scores:
        assert score.split == "time"
        assert score.n_test == split.test.size
        assert 0.0 <= score.dominance_accuracy <= 1.0
        assert score.dominance_accuracy_low <= score.dominance_accuracy_high
        assert score.band_coverage != score.band_coverage  # nan: no band


def test_baseline_at_an_upper_quantile_is_labelled(corpus):
    split = time_split(corpus)
    scores = score_baselines(
        corpus, split, resamples=100, acquisition_quantile=0.95
    )
    assert all(s.predictor.endswith("@q0.95") for s in scores)


def test_score_learned_reports_band_coverage(corpus):
    split = time_split(corpus)
    score, prediction, model = score_learned(
        corpus, split, predictor=HandoverLossPredictor(n_estimators=60), resamples=100
    )
    assert score.predictor == "learned_random_forest"
    assert 0.0 <= score.band_coverage <= 1.0
    assert model.fitted
    assert prediction.loss_bits.shape == (split.test.size, 3)
