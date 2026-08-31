"""Learned allocator: shapes, validation, determinism, and the honest claims.

These tests do **not** assert that the learned allocator is accurate or that
it respects the actuator bounds, because it is neither; the measured numbers
are in ``validation/ml_vs_qp_output.txt`` and ``MODEL_CARD.md``. What is
asserted here is that the interface behaves, that the model is reproducible
from its seed, and that it beats a trivial constant predictor -- the only
accuracy claim the package makes for it.
"""

import numpy as np
import pytest

from alloclab.dataset import generate_dataset, reference_thruster_cluster
from alloclab.ml import LearnedAllocator


@pytest.fixture(scope="module")
def cluster():
    return reference_thruster_cluster(1.0, 0.5)


@pytest.fixture(scope="module")
def train(cluster):
    return generate_dataset(cluster, 400, seed=101)


@pytest.fixture(scope="module")
def model(cluster, train):
    return LearnedAllocator(
        cluster, n_estimators=3, hidden_layer_sizes=(32, 16), max_iter=120, random_state=0
    ).fit(train.torques, train.health, train.commands)


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------


def test_unfitted_model_reports_it(cluster):
    m = LearnedAllocator(cluster, n_estimators=2)
    assert not m.fitted
    assert m.n_effectors == 8
    with pytest.raises(RuntimeError, match="called before fit"):
        m.predict(np.zeros((1, 3)), np.ones((1, 8)))


def test_fit_returns_self_and_sets_fitted(model):
    assert model.fitted


def test_predict_shapes(model, cluster):
    tau = np.array([[0.1, 0.0, 0.0], [0.0, -0.05, 0.02]])
    health = np.ones((2, 8))
    out = model.predict(tau, health)
    assert out.commands.shape == (2, 8)
    assert out.std.shape == (2, 8)
    assert out.confidence.shape == (2,)
    assert not out.clipped


def test_confidence_is_in_the_unit_interval(model):
    ds = generate_dataset(reference_thruster_cluster(1.0, 0.5), 60, seed=555)
    out = model.predict(ds.torques, ds.health)
    assert np.all(out.confidence > 0.0)
    assert np.all(out.confidence <= 1.0)


def test_ensemble_spread_is_non_negative(model):
    out = model.predict(np.array([[0.1, 0.1, 0.1]]), np.ones((1, 8)))
    assert np.all(out.std >= 0.0)


def test_clip_removes_bound_violations(model, cluster):
    ds = generate_dataset(cluster, 200, seed=777)
    raw = model.predict(ds.torques, ds.health, clip=False)
    clipped = model.predict(ds.torques, ds.health, clip=True)
    assert clipped.clipped
    assert np.max(cluster.bound_violation(clipped.commands)) <= 1e-12
    # Clipping is not free: it changes the achieved torque.
    assert not np.allclose(raw.commands, clipped.commands)


def test_training_is_deterministic_in_the_random_state(cluster, train):
    kwargs = dict(n_estimators=2, hidden_layer_sizes=(16,), max_iter=60, random_state=3)
    a = LearnedAllocator(cluster, **kwargs).fit(train.torques, train.health, train.commands)
    b = LearnedAllocator(cluster, **kwargs).fit(train.torques, train.health, train.commands)
    tau = np.array([[0.2, -0.1, 0.05]])
    assert np.allclose(
        a.predict(tau, np.ones((1, 8))).commands, b.predict(tau, np.ones((1, 8))).commands
    )


def test_features_include_the_health_mask(model):
    x = model.features(np.array([[0.1, 0.2, 0.3]]), np.array([[1.0] * 7 + [0.0]]))
    assert x.shape == (1, 11)
    assert x[0, -1] == 0.0


# --------------------------------------------------------------------------
# The only accuracy claim: better than a constant predictor
# --------------------------------------------------------------------------


def test_model_beats_the_training_mean_baseline(cluster, model, train):
    test = generate_dataset(cluster, 200, seed=909)
    pred = model.predict(test.torques, test.health).commands
    trivial = np.broadcast_to(train.commands.mean(axis=0), pred.shape)
    err_model = np.mean(np.linalg.norm(pred - test.commands, axis=1))
    err_trivial = np.mean(np.linalg.norm(trivial - test.commands, axis=1))
    assert err_model < err_trivial


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_ensemble_size_must_allow_a_spread(cluster):
    with pytest.raises(ValueError, match="n_estimators must be >= 2"):
        LearnedAllocator(cluster, n_estimators=1)


def test_max_iter_must_be_positive(cluster):
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        LearnedAllocator(cluster, max_iter=0)


def test_feature_shape_validation(model):
    with pytest.raises(ValueError, match="torques must have 3 columns"):
        model.features(np.zeros((2, 2)), np.ones((2, 8)))
    with pytest.raises(ValueError, match="health must have 8 columns"):
        model.features(np.zeros((2, 3)), np.ones((2, 4)))
    with pytest.raises(ValueError, match="same number of rows"):
        model.features(np.zeros((2, 3)), np.ones((3, 8)))


def test_fit_requires_enough_samples(cluster):
    m = LearnedAllocator(cluster, n_estimators=2)
    with pytest.raises(ValueError, match="at least 10 training samples"):
        m.fit(np.zeros((5, 3)), np.ones((5, 8)), np.zeros((5, 8)))


def test_fit_shape_mismatch_is_rejected(cluster):
    m = LearnedAllocator(cluster, n_estimators=2)
    with pytest.raises(ValueError, match="must match health shape"):
        m.fit(np.zeros((20, 3)), np.ones((20, 8)), np.zeros((20, 4)))


def test_bad_torque_scale_is_rejected(cluster, train):
    m = LearnedAllocator(cluster, n_estimators=2, max_iter=20)
    with pytest.raises(ValueError, match="torque_scale must be > 0"):
        m.fit(train.torques, train.health, train.commands, torque_scale=0.0)
