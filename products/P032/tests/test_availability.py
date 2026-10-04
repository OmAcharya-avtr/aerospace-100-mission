"""Link-availability predictors and the comparison harness."""

from __future__ import annotations

import numpy as np
import pytest

from constellink.availability import (
    ClimatologyBaseline,
    LinkAvailabilityModel,
    LogisticBaseline,
    PredictorScores,
    grouped_split,
    row_split,
    score_predictor,
)
from constellink.synthdata import N_FEATURES


def test_climatology_predicts_the_stratum_base_rate():
    x = np.zeros((6, N_FEATURES))
    y = np.array([1, 1, 0, 0, 0, 1])
    stratum = np.array([0, 0, 0, 1, 1, 1])
    model = ClimatologyBaseline().fit(x, y, stratum)
    p = model.predict_proba(x, stratum)
    assert np.allclose(p[:3], 2.0 / 3.0)
    assert np.allclose(p[3:], 1.0 / 3.0)


def test_climatology_falls_back_on_an_unseen_stratum():
    x = np.zeros((4, N_FEATURES))
    y = np.array([1, 1, 0, 1])
    model = ClimatologyBaseline().fit(x, y, np.array([0, 0, 1, 1]))
    p = model.predict_proba(x, np.array([0, 9, 9, 1]))
    assert model.n_fallback_last_call == 2
    assert p[1] == pytest.approx(0.75)      # global base rate
    assert p[0] == pytest.approx(1.0)


def test_climatology_requires_fit_first():
    with pytest.raises(RuntimeError, match="fit"):
        ClimatologyBaseline().predict_proba(np.zeros((1, N_FEATURES)),
                                            np.zeros(1))


def test_climatology_uncertainty_is_zero():
    x = np.zeros((3, N_FEATURES))
    model = ClimatologyBaseline().fit(x, np.array([1, 0, 1]), np.zeros(3))
    assert np.all(model.predict_std(x) == 0.0)


def test_climatology_validates_stratum_length():
    x = np.zeros((3, N_FEATURES))
    with pytest.raises(ValueError, match="one entry per row"):
        ClimatologyBaseline().fit(x, np.array([1, 0, 1]), np.zeros(2))


def test_logistic_baseline_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    x = np.zeros((200, N_FEATURES))
    x[:, 2] = rng.normal(0.0, 1.0, 200)
    y = (x[:, 2] > 0.0).astype(int)
    model = LogisticBaseline().fit(x, y)
    p = model.predict_proba(x)
    assert np.mean((p >= 0.5).astype(int) == y) > 0.95
    assert np.all((p >= 0.0) & (p <= 1.0))


def test_logistic_baseline_requires_fit_first():
    with pytest.raises(RuntimeError, match="fit"):
        LogisticBaseline().predict_proba(np.zeros((1, N_FEATURES)))


@pytest.mark.parametrize("kwargs", [{"c": 0.0}, {"max_iter": 0}])
def test_logistic_baseline_validation(kwargs):
    with pytest.raises(ValueError):
        LogisticBaseline(**kwargs)


def test_logistic_uncertainty_is_zero():
    x = np.zeros((4, N_FEATURES))
    x[:, 2] = [-1.0, -0.5, 0.5, 1.0]
    model = LogisticBaseline().fit(x, np.array([0, 0, 1, 1]))
    assert np.all(model.predict_std(x) == 0.0)


@pytest.mark.parametrize("kwargs", [{"n_members": 1}, {"max_iter": 0},
                                    {"max_depth": 0}, {"learning_rate": 0.0},
                                    {"method": "platt"},
                                    {"calibration_folds": 1}])
def test_learned_model_validation(kwargs):
    with pytest.raises(ValueError):
        LinkAvailabilityModel(**kwargs)


def test_learned_model_requires_fit_first():
    with pytest.raises(RuntimeError, match="fit"):
        LinkAvailabilityModel().predict_proba(np.zeros((1, N_FEATURES)))


def test_learned_model_rejects_single_class_labels():
    x = np.zeros((40, N_FEATURES))
    with pytest.raises(ValueError, match="single-class"):
        LinkAvailabilityModel(n_members=2, max_iter=5).fit(x, np.zeros(40,
                                                                       dtype=int))


def test_learned_model_outputs_probabilities_and_spread(small_dataset):
    d = small_dataset
    train, test = grouped_split(d, test_fraction=0.3, seed=0)
    model = LinkAvailabilityModel(n_members=3, seed=1, max_iter=40).fit(
        d.x[train], d.y[train])
    p, s = model.predict_with_uncertainty(d.x[test])
    assert p.shape == (test.size,)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(s >= 0.0)
    assert s.max() > 0.0        # a bagged ensemble must disagree somewhere
    assert np.allclose(p, model.predict_proba(d.x[test]))
    assert np.allclose(s, model.predict_std(d.x[test]))


def test_learned_model_is_deterministic_for_a_seed(small_dataset):
    d = small_dataset
    train, test = grouped_split(d, test_fraction=0.3, seed=0)
    a = LinkAvailabilityModel(n_members=2, seed=5, max_iter=30).fit(
        d.x[train], d.y[train]).predict_proba(d.x[test])
    b = LinkAvailabilityModel(n_members=2, seed=5, max_iter=30).fit(
        d.x[train], d.y[train]).predict_proba(d.x[test])
    assert np.array_equal(a, b)


def test_feature_shape_is_enforced():
    model = LogisticBaseline()
    with pytest.raises(ValueError, match="shape"):
        model.fit(np.zeros((4, 3)), np.array([0, 1, 0, 1]))


def test_non_finite_features_are_rejected():
    x = np.zeros((4, N_FEATURES))
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        LogisticBaseline().fit(x, np.array([0, 1, 0, 1]))


def test_score_predictor_fields():
    p = np.array([0.9, 0.9, 0.1, 0.1])
    y = np.array([1, 1, 0, 0])
    s = score_predictor("x", p, y, n_bins=10)
    assert s.n == 4
    assert s.brier == pytest.approx(0.01)
    assert s.accuracy_at_half == pytest.approx(1.0)
    assert s.base_rate == pytest.approx(0.5)
    assert s.mean_forecast == pytest.approx(0.5)
    assert "x" in s.format_row()
    assert "Brier" in PredictorScores.header()


def test_grouped_split_keeps_links_whole(small_dataset):
    d = small_dataset
    train, test = grouped_split(d, test_fraction=0.3, seed=0)
    assert train.size + test.size == len(d)
    assert set(d.group[train]).isdisjoint(set(d.group[test]))


def test_grouped_split_validation(small_dataset):
    with pytest.raises(ValueError, match="test_fraction"):
        grouped_split(small_dataset, test_fraction=0.0)
    with pytest.raises(ValueError, match="test_fraction"):
        grouped_split(small_dataset, test_fraction=1.0)


def test_row_split_partitions_every_row(small_dataset):
    train, test = row_split(small_dataset, test_fraction=0.25, seed=0)
    assert train.size + test.size == len(small_dataset)
    assert set(train.tolist()).isdisjoint(set(test.tolist()))


def test_row_split_validation(small_dataset):
    with pytest.raises(ValueError):
        row_split(small_dataset, test_fraction=1.5)


def test_row_split_leaks_groups_which_is_why_it_is_not_used(small_dataset):
    # The documented reason grouped_split exists: a row-wise split puts the
    # same link on both sides.
    train, test = row_split(small_dataset, test_fraction=0.3, seed=0)
    shared = set(small_dataset.group[train]) & set(small_dataset.group[test])
    assert shared, "expected row-wise splitting to share links across the split"
