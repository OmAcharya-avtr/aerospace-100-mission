"""Feature map and the learned null-motion policy."""

import numpy as np
import pytest

from cmgsteer.arrays import pyramid_array
from cmgsteer.dataset import generate_policy_dataset
from cmgsteer.ml import LearnedNullMotion, feature_names, policy_features

D = np.array([0.30, -0.50, 0.80, 0.20])
TAU = np.array([0.10, -0.05, 0.20])


def _tiny_dataset():
    return generate_policy_dataset(
        pyramid_array(), 40, seed=13, horizon=10, n_candidates=5, stride=19, n_manoeuvres=4
    )


class TestFeatures:
    def test_length_matches_the_names(self):
        array = pyramid_array()
        feats = policy_features(array, D, TAU)
        assert feats.shape == (len(feature_names(4)),)

    def test_all_features_are_finite_and_bounded(self):
        array = pyramid_array()
        rng = np.random.default_rng(31)
        for _ in range(200):
            d = rng.uniform(-np.pi, np.pi, 4)
            tau = rng.normal(size=3) * 0.5
            feats = policy_features(array, d, tau)
            assert np.all(np.isfinite(feats))
            assert np.max(np.abs(feats)) < 20.0

    def test_sin_cos_encoding_is_wrap_free(self):
        array = pyramid_array()
        a = policy_features(array, D, TAU)
        b = policy_features(array, D + 2.0 * np.pi, TAU)
        assert np.allclose(a, b, atol=1e-9)

    def test_zero_torque_gives_a_zero_direction(self):
        array = pyramid_array()
        feats = policy_features(array, D, np.zeros(3))
        names = feature_names(4)
        for label in ("tau_x_hat", "tau_y_hat", "tau_z_hat", "tau_mag_norm"):
            assert feats[names.index(label)] == 0.0

    def test_null_gradient_cosine_is_in_the_unit_interval(self):
        array = pyramid_array()
        names = feature_names(4)
        rng = np.random.default_rng(32)
        for _ in range(50):
            d = rng.uniform(-np.pi, np.pi, 4)
            value = policy_features(array, d, TAU)[names.index("null_dot_grad_hat")]
            assert -1e-9 <= value <= 1.0 + 1e-9

    def test_measure_feature_matches_the_definition(self):
        array = pyramid_array()
        names = feature_names(4)
        feats = policy_features(array, np.zeros(4), TAU)
        assert feats[names.index("measure_norm")] == pytest.approx(1.152, abs=1e-12)

    def test_singular_configuration_features_are_finite(self):
        array = pyramid_array()
        feats = policy_features(array, np.full(4, np.pi / 2), TAU)
        assert np.all(np.isfinite(feats))

    def test_bad_torque_raises(self):
        with pytest.raises(ValueError, match=r"torque must have shape \(3,\)"):
            policy_features(pyramid_array(), D, [1.0, 2.0])


class TestLearnedPolicy:
    @pytest.fixture(scope="class")
    @classmethod
    def trained(cls):
        data = _tiny_dataset()
        policy = LearnedNullMotion(n_estimators=3, hidden_layer_sizes=(16,), max_iter=60)
        policy.fit(data.features, data.coefficients)
        return policy, data

    def test_fitted_flag(self, trained):
        policy, _ = trained
        assert policy.fitted
        assert not LearnedNullMotion().fitted

    def test_predictions_are_in_range(self, trained):
        policy, data = trained
        coeff, std = policy.predict(data.features)
        assert coeff.shape == (data.n_samples,)
        assert np.all(coeff >= -1.0) and np.all(coeff <= 1.0)
        assert np.all(std >= 0.0)

    def test_confidence_is_in_the_unit_interval(self, trained):
        policy, data = trained
        _, std = policy.predict(data.features)
        conf = policy.confidence(std)
        assert np.all(conf > 0.0) and np.all(conf <= 1.0)

    def test_confidence_falls_as_spread_grows(self, trained):
        policy, _ = trained
        conf = policy.confidence([0.0, 0.05, 0.5])
        assert conf[0] == pytest.approx(1.0)
        assert conf[0] > conf[1] > conf[2]

    def test_act_returns_null_space_rates(self, trained):
        policy, _ = trained
        array = pyramid_array()
        action = policy.act(array, D, TAU)
        assert np.max(np.abs(array.jacobian(D) @ action.rates)) < 1e-12
        assert -1.0 <= action.coefficient <= 1.0
        assert 0.0 < action.confidence <= 1.0

    def test_rates_respect_the_maximum_null_rate(self, trained):
        policy, _ = trained
        array = pyramid_array()
        rates = policy.rates(array, D, TAU)
        assert np.max(np.abs(rates)) <= policy.max_null_rate + 1e-12

    def test_zero_rates_when_the_null_space_is_degenerate(self, trained):
        policy, _ = trained
        array = pyramid_array()
        assert np.max(np.abs(policy.rates(array, np.full(4, np.pi / 2), TAU))) == 0.0

    def test_confidence_floor_falls_back_to_the_gradient_direction(self):
        data = _tiny_dataset()
        policy = LearnedNullMotion(
            n_estimators=3, hidden_layer_sizes=(16,), max_iter=60, confidence_floor=1.1
        )
        policy.fit(data.features, data.coefficients)
        action = policy.act(pyramid_array(), D, TAU)
        assert action.coefficient == 1.0

    def test_training_is_deterministic(self):
        data = _tiny_dataset()
        a = LearnedNullMotion(n_estimators=2, hidden_layer_sizes=(16,), max_iter=50)
        b = LearnedNullMotion(n_estimators=2, hidden_layer_sizes=(16,), max_iter=50)
        a.fit(data.features, data.coefficients)
        b.fit(data.features, data.coefficients)
        pa, sa = a.predict(data.features)
        pb, sb = b.predict(data.features)
        assert np.allclose(pa, pb)
        assert np.allclose(sa, sb)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="called before fit"):
            LearnedNullMotion().predict(np.zeros((1, 20)))

    def test_feature_count_mismatch_raises(self, trained):
        policy, _ = trained
        with pytest.raises(ValueError, match="expected 20 features"):
            policy.predict(np.zeros((1, 5)))

    def test_fit_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="but coefficients has"):
            LearnedNullMotion().fit(np.zeros((10, 20)), np.zeros(9))

    def test_fit_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="need at least 5 samples"):
            LearnedNullMotion(n_estimators=5).fit(np.zeros((3, 20)), np.zeros(3))

    def test_fit_rejects_a_bad_null_rate(self):
        with pytest.raises(ValueError, match="max_null_rate must be positive"):
            LearnedNullMotion(max_null_rate=0.0, n_estimators=2).fit(
                np.zeros((10, 20)), np.zeros(10)
            )
