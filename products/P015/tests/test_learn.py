"""Tests for linkswitch.learn: OutagePredictor and training-set construction."""

import numpy as np
import pytest

from linkswitch.learn import OutagePredictor, build_training_set, train_outage_predictor
from linkswitch.optical import OpticalParams
from linkswitch.scenario import ScenarioConfig, generate_telemetry


class TestOutagePredictorBasics:
    def test_predict_before_fit_raises(self):
        model = OutagePredictor()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.zeros((1, 5)))

    def test_is_fitted_flag(self):
        model = OutagePredictor()
        assert not model.is_fitted
        model.fit(np.array([[0.0] * 5, [1.0] * 5]), np.array([0, 1]))
        assert model.is_fitted

    def test_mismatched_rows_rejected(self):
        model = OutagePredictor()
        with pytest.raises(ValueError):
            model.fit(np.zeros((3, 5)), np.zeros(2))

    def test_single_row_rejected(self):
        model = OutagePredictor()
        with pytest.raises(ValueError):
            model.fit(np.zeros((1, 5)), np.zeros(1))

    def test_invalid_n_estimators_rejected(self):
        with pytest.raises(ValueError):
            OutagePredictor(n_estimators=0)

    def test_invalid_max_depth_rejected(self):
        with pytest.raises(ValueError):
            OutagePredictor(max_depth=0)

    def test_single_class_degenerate_fallback(self):
        # All-zero labels: the model should still fit (constant predictor)
        # rather than raising, since this is a real scenario (e.g. a short
        # horizon on a fade-free episode), documented in the code.
        model = OutagePredictor().fit(np.zeros((5, 5)), np.zeros(5))
        assert model.is_fitted
        proba = model.predict_proba(np.random.default_rng(0).normal(size=(10, 5)))
        assert np.all(proba == 0.0)

    def test_single_class_all_ones_fallback(self):
        model = OutagePredictor().fit(np.zeros((5, 5)), np.ones(5))
        proba = model.predict_proba(np.zeros((3, 5)))
        assert np.all(proba == 1.0)

    def test_predict_proba_in_unit_interval(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 5))
        y = (x[:, 0] > 0).astype(int)
        model = OutagePredictor(n_estimators=10, random_state=0).fit(x, y)
        proba = model.predict_proba(x)
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    def test_learns_separable_signal(self):
        # A clearly separable synthetic signal: the model should recover it
        # with reasonably high accuracy (sanity check the pipeline works,
        # not a tight numeric bound).
        rng = np.random.default_rng(1)
        x = rng.normal(size=(400, 5))
        y = (x[:, 0] + x[:, 1] > 0).astype(int)
        model = OutagePredictor(n_estimators=20, random_state=0).fit(x, y)
        pred = (model.predict_proba(x) >= 0.5).astype(int)
        accuracy = np.mean(pred == y)
        assert accuracy > 0.8

    def test_predict_proba_wrong_ndim_rejected(self):
        model = OutagePredictor().fit(np.zeros((5, 5)), np.array([0, 1, 0, 1, 0]))
        with pytest.raises(ValueError):
            model.predict_proba(np.zeros(5))

    def test_seeded_reproducibility(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(100, 5))
        y = (x[:, 0] > 0).astype(int)
        m1 = OutagePredictor(random_state=7).fit(x, y)
        m2 = OutagePredictor(random_state=7).fit(x, y)
        np.testing.assert_array_equal(m1.predict_proba(x), m2.predict_proba(x))


class TestBuildTrainingSet:
    def test_shapes_match(self):
        cfg = ScenarioConfig()
        tels = [generate_telemetry(cfg, 200, seed=i) for i in range(3)]
        x, y = build_training_set(tels, tau_phys=cfg.optical.tau_phys, horizon=5, window=4)
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == 5

    def test_drops_last_horizon_rows_per_episode(self):
        cfg = ScenarioConfig()
        tels = [generate_telemetry(cfg, 50, seed=0)]
        horizon = 10
        x, _ = build_training_set(tels, tau_phys=cfg.optical.tau_phys, horizon=horizon, window=4)
        assert x.shape[0] == 50 - horizon

    def test_all_episodes_too_short_raises(self):
        cfg = ScenarioConfig()
        tels = [generate_telemetry(cfg, 5, seed=0)]
        with pytest.raises(ValueError):
            build_training_set(tels, tau_phys=cfg.optical.tau_phys, horizon=10, window=2)

    def test_empty_telemetry_list_rejected(self):
        with pytest.raises(ValueError):
            train_outage_predictor([], tau_phys=1.0, horizon=5, window=4)


class TestTrainOutagePredictor:
    def test_end_to_end_produces_fitted_model(self):
        opt = OpticalParams(sigma_i2=0.4, coherence_steps=3.0, margin_db=3.0)
        cfg = ScenarioConfig(optical=opt)
        tels = [generate_telemetry(cfg, 500, seed=i) for i in range(10)]
        model = train_outage_predictor(tels, tau_phys=opt.tau_phys, horizon=5, window=6)
        assert model.is_fitted

    def test_invalid_tau_phys_rejected(self):
        cfg = ScenarioConfig()
        tels = [generate_telemetry(cfg, 200, seed=0)]
        with pytest.raises(ValueError):
            train_outage_predictor(tels, tau_phys=-1.0, horizon=5, window=4)
