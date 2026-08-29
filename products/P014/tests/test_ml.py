"""Unit, edge-case, integration and regression tests for wavelab.ml.ZernikeSlopeEnsemble."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.dataset import build_modal_geometry, generate_batch
from wavelab.ml import ZernikeSlopeEnsemble
from wavelab.modal import ModalReconstructor


def test_rejects_bad_n_sub():
    with pytest.raises(ValueError):
        ZernikeSlopeEnsemble(0, 5)


def test_rejects_bad_n_modes():
    with pytest.raises(ValueError):
        ZernikeSlopeEnsemble(10, 0)


def test_rejects_too_few_estimators():
    with pytest.raises(ValueError):
        ZernikeSlopeEnsemble(10, 5, n_estimators=1)


def test_rejects_non_integer_estimators():
    with pytest.raises(TypeError):
        ZernikeSlopeEnsemble(10, 5, n_estimators=3.5)


def test_predict_before_fit_raises():
    model = ZernikeSlopeEnsemble(10, 5)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros(20), np.ones(10, dtype=bool))


def test_features_rejects_wrong_slope_width():
    model = ZernikeSlopeEnsemble(10, 5)
    with pytest.raises(ValueError):
        model.features(np.zeros((3, 19)), np.ones((3, 10), dtype=bool))


def test_features_rejects_wrong_active_width():
    model = ZernikeSlopeEnsemble(10, 5)
    with pytest.raises(ValueError):
        model.features(np.zeros((3, 20)), np.ones((3, 9), dtype=bool))


def test_features_rejects_row_count_mismatch():
    model = ZernikeSlopeEnsemble(10, 5)
    with pytest.raises(ValueError):
        model.features(np.zeros((3, 20)), np.ones((2, 10), dtype=bool))


def test_features_rejects_non_finite():
    model = ZernikeSlopeEnsemble(10, 5)
    s = np.zeros((1, 20))
    s[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.features(s, np.ones((1, 10), dtype=bool))


def test_features_shape_and_mask_concatenation():
    model = ZernikeSlopeEnsemble(4, 3)
    s = np.arange(8, dtype=float)
    a = np.array([True, False, True, True])
    feats = model.features(s, a)
    assert feats.shape == (1, 12)
    np.testing.assert_allclose(feats[0, 8:], [1.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(feats[0, :8], s)


def test_fit_rejects_too_few_samples():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    model = ZernikeSlopeEnsemble(geo.n_sub, geo.n_modes)
    batch = generate_batch(geo, 5, photon_flux=1000.0, dropout_rate=0.0, seed=0)
    with pytest.raises(ValueError):
        model.fit(batch.slopes, batch.active, batch.coeffs)


def test_fit_rejects_coeff_shape_mismatch():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    model = ZernikeSlopeEnsemble(geo.n_sub, geo.n_modes)
    batch = generate_batch(geo, 20, photon_flux=1000.0, dropout_rate=0.0, seed=0)
    with pytest.raises(ValueError):
        model.fit(batch.slopes, batch.active, batch.coeffs[:, :-1])


def test_fit_predict_round_trip_shapes():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    model = ZernikeSlopeEnsemble(geo.n_sub, geo.n_modes, n_estimators=2, max_iter=50)
    train = generate_batch(geo, 40, photon_flux=1000.0, dropout_rate=0.1, seed=0)
    model.fit(train.slopes, train.active, train.coeffs)
    assert model.fitted_
    pred = model.predict(train.slopes, train.active)
    assert pred.shape == train.coeffs.shape
    pred2, std = model.predict(train.slopes, train.active, return_std=True)
    np.testing.assert_allclose(pred, pred2)
    assert std.shape == train.coeffs.shape
    assert np.all(std >= 0.0)


def test_predict_single_sample_1d_input():
    geo = build_modal_geometry(list(range(2, 8)), 6)
    model = ZernikeSlopeEnsemble(geo.n_sub, geo.n_modes, n_estimators=2, max_iter=50)
    train = generate_batch(geo, 40, photon_flux=1000.0, dropout_rate=0.1, seed=0)
    model.fit(train.slopes, train.active, train.coeffs)
    pred = model.predict(train.slopes[0], train.active[0])
    assert pred.shape == (1, geo.n_modes)


# --------------------------------------------------------------------- integration / regression


def test_ensemble_learns_better_than_untrained_zero_predictor():
    # Sanity integration check: a fitted model should beat the trivial
    # always-predict-zero baseline on its own training distribution.
    geo = build_modal_geometry(list(range(2, 10)), 8)
    model = ZernikeSlopeEnsemble(geo.n_sub, geo.n_modes, n_estimators=3, max_iter=200)
    train = generate_batch(geo, 200, photon_flux=1000.0, dropout_rate=0.1, seed=10)
    model.fit(train.slopes, train.active, train.coeffs)
    test = generate_batch(geo, 80, photon_flux=1000.0, dropout_rate=0.1, seed=11)
    pred = model.predict(test.slopes, test.active)
    ml_rms = np.sqrt(np.mean((pred - test.coeffs) ** 2))
    zero_rms = np.sqrt(np.mean(test.coeffs**2))
    assert ml_rms < zero_rms


def test_pinned_regression_seeded_ensemble_error_at_reference_point():
    # Benchmark/regression test with a pinned seed: an untracked change to the
    # ensemble architecture, feature encoding or training procedure should
    # move this number; a generous band (not a tight bound) catches gross
    # regressions without being a tolerance-hiding rubber stamp.
    geo = build_modal_geometry(list(range(2, 10)), 8)
    model = ZernikeSlopeEnsemble(
        geo.n_sub, geo.n_modes, n_estimators=3, hidden_layer_sizes=(48, 24),
        max_iter=200, random_state=123,
    )
    train = generate_batch(geo, 250, photon_flux=1000.0, dropout_rate=0.15, seed=42)
    model.fit(train.slopes, train.active, train.coeffs)
    test = generate_batch(geo, 100, photon_flux=1000.0, dropout_rate=0.15, seed=4242)
    pred = model.predict(test.slopes, test.active)
    rms = float(np.sqrt(np.mean((pred - test.coeffs) ** 2)))
    assert 0.0 < rms < 0.20, f"pinned-seed ensemble RMS error out of expected band: {rms}"


def test_baseline_beats_ml_at_very_high_flux_no_dropout():
    # Documented, expected honest result (README/MODEL_CARD): at high flux
    # with no dropout, the regularized least-squares baseline -- which has
    # an essentially exact analytic forward model -- outperforms the learned
    # ensemble, which must approximate that mapping from finite training data.
    geo = build_modal_geometry(list(range(2, 10)), 8)
    baseline = ModalReconstructor(
        list(range(2, 10)), geo.sub_x, geo.sub_y, method="tikhonov", reg=1e-4
    )
    model = ZernikeSlopeEnsemble(geo.n_sub, geo.n_modes, n_estimators=3, max_iter=200)
    train = generate_batch(geo, 250, photon_flux=2000.0, dropout_rate=0.0, seed=1)
    model.fit(train.slopes, train.active, train.coeffs)

    test = generate_batch(geo, 100, photon_flux=200000.0, dropout_rate=0.0, seed=2)
    base_pred = np.array(
        [baseline.reconstruct(test.slopes[i], active=test.active[i]) for i in range(len(test))]
    )
    ml_pred = model.predict(test.slopes, test.active)
    base_rms = np.sqrt(np.mean((base_pred - test.coeffs) ** 2))
    ml_rms = np.sqrt(np.mean((ml_pred - test.coeffs) ** 2))
    assert base_rms < ml_rms
