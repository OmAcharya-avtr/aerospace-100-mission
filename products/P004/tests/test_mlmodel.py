"""Synthetic-data and availability-model tests (baseline + ML)."""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from passplanner import ClimatologyBaselineModel, PassSuccessModel, generate_dataset
from passplanner.synthdata import FEATURE_NAMES


def test_dataset_is_reproducible_from_seed():
    a = generate_dataset(500, seed=7)
    b = generate_dataset(500, seed=7)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.p_true, b.p_true)


def test_different_seeds_give_different_data():
    a = generate_dataset(500, seed=1)
    b = generate_dataset(500, seed=2)
    assert not np.array_equal(a.x, b.x)


def test_dataset_shapes_and_ranges():
    ds = generate_dataset(800, seed=0)
    assert ds.x.shape == (800, len(FEATURE_NAMES))
    assert ds.y.shape == (800,)
    assert set(np.unique(ds.y)) <= {0, 1}
    assert np.all((ds.p_true > 0.0) & (ds.p_true < 1.0))
    assert np.all(np.isfinite(ds.x))
    # Column ranges: prior in [0,1], RH in [5,100], cloud fraction in [0,1].
    assert np.all((ds.x[:, 0] >= 0.0) & (ds.x[:, 0] <= 1.0))
    assert np.all((ds.x[:, 1] >= 5.0) & (ds.x[:, 1] <= 100.0))
    assert np.all((ds.x[:, 2] >= 0.0) & (ds.x[:, 2] <= 1.0))


def test_labels_track_true_probability():
    # Sanity: empirical success rate must be close to mean(p_true) for large n.
    ds = generate_dataset(6000, seed=3)
    assert ds.y.mean() == pytest.approx(ds.p_true.mean(), abs=0.02)


def test_dataset_rejects_bad_size():
    with pytest.raises(ValueError, match="n_samples"):
        generate_dataset(0, seed=0)


def test_baseline_predicts_the_prior_column():
    ds = generate_dataset(200, seed=4)
    base = ClimatologyBaselineModel().fit(ds.x, ds.y)
    assert np.allclose(base.predict_proba(ds.x), ds.x[:, 0])


def test_model_is_reproducible_and_beats_baseline():
    train = generate_dataset(4000, seed=11)
    test = generate_dataset(2000, seed=12)
    m1 = PassSuccessModel(n_members=3, seed=5).fit(train.x, train.y)
    m2 = PassSuccessModel(n_members=3, seed=5).fit(train.x, train.y)
    p1 = m1.predict_proba(test.x)
    p2 = m2.predict_proba(test.x)
    assert np.array_equal(p1, p2)  # seeded reproducibility, bit-identical

    base = ClimatologyBaselineModel().fit(train.x, train.y)
    brier_model = brier_score_loss(test.y, p1)
    brier_base = brier_score_loss(test.y, base.predict_proba(test.x))
    # The ML model must beat the climatology baseline on held-out Brier score.
    assert brier_model < brier_base


def test_uncertainty_output_shape_and_sign():
    train = generate_dataset(1500, seed=21)
    model = PassSuccessModel(n_members=3, seed=1).fit(train.x, train.y)
    p, sigma = model.predict_with_uncertainty(train.x[:200])
    assert p.shape == sigma.shape == (200,)
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(sigma >= 0.0)
    assert sigma.max() > 0.0  # ensemble members genuinely disagree somewhere


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        PassSuccessModel().predict_proba(np.zeros((2, len(FEATURE_NAMES))))


@pytest.mark.parametrize("bad_x", [
    np.zeros((3, 2)),
    np.zeros((3,)),
    np.full((3, len(FEATURE_NAMES)), np.nan),
])
def test_model_rejects_bad_features(bad_x):
    model = PassSuccessModel(n_members=2, seed=0)
    with pytest.raises(ValueError):
        model.fit(bad_x, np.zeros(3, dtype=int))


def test_model_rejects_bad_labels():
    ds = generate_dataset(50, seed=0)
    model = PassSuccessModel(n_members=2, seed=0)
    with pytest.raises(ValueError, match="binary"):
        model.fit(ds.x, np.full(50, 2))
    with pytest.raises(ValueError, match="shape"):
        model.fit(ds.x, np.zeros(49, dtype=int))


def test_model_rejects_too_few_members():
    with pytest.raises(ValueError, match="n_members"):
        PassSuccessModel(n_members=1)
