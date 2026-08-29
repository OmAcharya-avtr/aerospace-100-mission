"""Unit tests for navbench.adaptive — features, Mehra IAE, and the learned tuner."""

from __future__ import annotations

import numpy as np
import pytest

from navbench import (
    FEATURE_NAMES,
    N_FEATURES,
    KalmanFilter,
    LearnedAdaptiveQ,
    MehraAdaptiveQ,
    constant_velocity_cwna,
    generate_adaptive_dataset,
    innovation_features,
    run_adaptive_kf,
    simulate_linear_system,
)


@pytest.fixture(scope="module")
def small_dataset():
    """A small deterministic dataset, shared across tests to keep the suite fast."""
    return generate_adaptive_dataset(n_runs=25, n_steps=300, seed=4242)


@pytest.fixture(scope="module")
def fitted_model(small_dataset):
    x, y, _ = small_dataset
    return LearnedAdaptiveQ(n_members=3, n_estimators=40, random_state=1).fit(x, y)


class TestInnovationFeatures:
    def test_length_and_names(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((30, 1))
        s = np.repeat(np.eye(1)[None], 30, axis=0)
        f = innovation_features(v, s)
        assert f.shape == (N_FEATURES,)
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_all_finite(self):
        rng = np.random.default_rng(1)
        v = rng.standard_normal((40, 2))
        s = np.repeat(np.eye(2)[None], 40, axis=0)
        assert np.all(np.isfinite(innovation_features(v, s)))

    def test_consistent_innovations_give_zero_log_nis(self):
        rng = np.random.default_rng(2)
        v = rng.standard_normal((4000, 1))
        s = np.repeat(np.eye(1)[None], 4000, axis=0)
        f = innovation_features(v, s)
        assert f[0] == pytest.approx(0.0, abs=0.03)
        assert f[1] == pytest.approx(0.0, abs=0.03)

    def test_inflated_innovations_raise_log_nis(self):
        rng = np.random.default_rng(3)
        v = 10.0 * rng.standard_normal((400, 1))
        s = np.repeat(np.eye(1)[None], 400, axis=0)
        assert innovation_features(v, s)[0] == pytest.approx(2.0, abs=0.15)

    def test_lag1_autocorrelation_detects_correlation(self):
        rng = np.random.default_rng(4)
        x = np.cumsum(rng.standard_normal(300))[:, None]
        s = np.repeat(np.eye(1)[None] * np.var(x), 300, axis=0)
        assert innovation_features(x, s)[2] > 0.7

    def test_lag1_near_zero_for_white_input(self):
        rng = np.random.default_rng(5)
        v = rng.standard_normal((2000, 1))
        s = np.repeat(np.eye(1)[None], 2000, axis=0)
        assert abs(innovation_features(v, s)[2]) < 0.1

    def test_fraction_gt_2_matches_gaussian_tail(self):
        rng = np.random.default_rng(6)
        v = rng.standard_normal((20000, 1))
        s = np.repeat(np.eye(1)[None], 20000, axis=0)
        assert innovation_features(v, s)[5] == pytest.approx(0.0455, abs=0.01)

    def test_scale_free_in_measurement_units(self):
        """Multiplying nu and S consistently must not change the features."""
        rng = np.random.default_rng(7)
        v = rng.standard_normal((100, 1))
        s = np.repeat(np.eye(1)[None], 100, axis=0)
        a = innovation_features(v, s)
        b = innovation_features(1000.0 * v, 1e6 * s)
        assert np.allclose(a, b, atol=1e-12)

    def test_nan_rows_dropped(self):
        rng = np.random.default_rng(8)
        v = rng.standard_normal((50, 1))
        v[10] = np.nan
        s = np.repeat(np.eye(1)[None], 50, axis=0)
        assert np.all(np.isfinite(innovation_features(v, s)))

    def test_single_channel_duplicates_variance_feature(self):
        rng = np.random.default_rng(9)
        v = rng.standard_normal((50, 1))
        s = np.repeat(np.eye(1)[None], 50, axis=0)
        f = innovation_features(v, s)
        assert f[3] == pytest.approx(f[4])

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            innovation_features(np.zeros((2, 1)), np.repeat(np.eye(1)[None], 2, axis=0))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            innovation_features(np.zeros((10, 2)), np.repeat(np.eye(1)[None], 10, axis=0))

    def test_indefinite_covariance_raises(self):
        s = np.repeat((-np.eye(1))[None], 10, axis=0)
        with pytest.raises(ValueError, match="positive definite"):
            innovation_features(np.ones((10, 1)), s)


class TestMehraAdaptiveQ:
    def test_dim(self):
        assert MehraAdaptiveQ(np.eye(3)).dim == 3

    def test_estimate_q_shape_and_symmetry(self, rng):
        m = MehraAdaptiveQ(np.eye(2))
        v = rng.standard_normal((30, 1))
        k = np.array([[0.5], [0.1]])
        q = m.estimate_q(v, k)
        assert q.shape == (2, 2)
        assert np.array_equal(q, q.T)

    def test_estimate_q_is_psd(self, rng):
        m = MehraAdaptiveQ(np.eye(2))
        v = rng.standard_normal((50, 2))
        k = rng.standard_normal((2, 2))
        assert np.linalg.eigvalsh(m.estimate_q(v, k)).min() > -1e-12

    def test_estimate_q_hand_computed(self):
        """C_hat = mean(nu nu^T) = 4 for a constant innovation of 2;
        K = [[1],[0]] -> Q_hat = [[4, 0], [0, 0]]."""
        m = MehraAdaptiveQ(np.eye(2))
        v = np.full((10, 1), 2.0)
        q = m.estimate_q(v, np.array([[1.0], [0.0]]))
        assert np.allclose(q, [[4.0, 0.0], [0.0, 0.0]])

    def test_gain_series_averaged(self, rng):
        m = MehraAdaptiveQ(np.eye(2))
        v = rng.standard_normal((20, 1))
        gains = np.repeat(np.array([[0.4], [0.2]])[None], 20, axis=0)
        q_series = m.estimate_q(v, gains)
        q_single = m.estimate_q(v, np.array([[0.4], [0.2]]))
        assert np.allclose(q_series, q_single)

    def test_scale_is_clipped(self, rng):
        m = MehraAdaptiveQ(1e-12 * np.eye(2), min_scale=0.5, max_scale=2.0)
        v = 1e6 * rng.standard_normal((20, 1))
        assert m.estimate_scale(v, np.array([[1.0], [0.0]])) == pytest.approx(2.0)

    def test_scale_low_clip(self):
        m = MehraAdaptiveQ(np.eye(2), min_scale=0.25, max_scale=4.0)
        v = np.full((20, 1), 1e-12)
        assert m.estimate_scale(v, np.array([[1.0], [0.0]])) == pytest.approx(0.25)

    def test_scale_recovers_a_known_inflation(self):
        """C_hat scaled by 9 scales tr(Q_hat) by 9, so lambda scales by 9."""
        m = MehraAdaptiveQ(np.eye(2), min_scale=1e-6, max_scale=1e6)
        v = np.full((20, 1), 1.0)
        k = np.array([[1.0], [0.0]])
        a = m.estimate_scale(v, k)
        b = m.estimate_scale(3.0 * v, k)
        assert b / a == pytest.approx(9.0)

    def test_non_square_q_nominal_raises(self):
        with pytest.raises(ValueError, match="square"):
            MehraAdaptiveQ(np.zeros((2, 3)))

    def test_zero_trace_raises(self):
        with pytest.raises(ValueError, match="positive trace"):
            MehraAdaptiveQ(np.zeros((2, 2)))

    def test_bad_clip_bounds_raise(self):
        with pytest.raises(ValueError, match="min_scale"):
            MehraAdaptiveQ(np.eye(2), min_scale=2.0, max_scale=1.0)

    def test_too_few_innovations_raise(self):
        m = MehraAdaptiveQ(np.eye(2))
        with pytest.raises(ValueError, match="at least 2"):
            m.estimate_q(np.zeros((1, 1)), np.array([[1.0], [0.0]]))

    def test_gain_dimension_mismatch_raises(self, rng):
        m = MehraAdaptiveQ(np.eye(2))
        with pytest.raises(ValueError, match="rows"):
            m.estimate_q(rng.standard_normal((10, 1)), np.ones((3, 1)))


class TestDataset:
    def test_shapes(self, small_dataset):
        x, y, idx = small_dataset
        assert x.ndim == 2 and x.shape[1] == N_FEATURES
        assert y.shape[0] == x.shape[0]
        assert idx.shape[0] == x.shape[0]

    def test_targets_inside_the_requested_range(self, small_dataset):
        _, y, _ = small_dataset
        assert float(np.min(y)) >= -1.5 - 1e-12
        assert float(np.max(y)) <= 1.5 + 1e-12

    def test_all_runs_contribute(self, small_dataset):
        _, _, idx = small_dataset
        assert np.unique(idx).size == 25

    def test_deterministic(self):
        a = generate_adaptive_dataset(n_runs=4, n_steps=200, seed=11)
        b = generate_adaptive_dataset(n_runs=4, n_steps=200, seed=11)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])

    def test_different_seed_gives_different_data(self):
        a = generate_adaptive_dataset(n_runs=4, n_steps=200, seed=11)
        b = generate_adaptive_dataset(n_runs=4, n_steps=200, seed=12)
        assert not np.array_equal(a[1], b[1])

    def test_features_all_finite(self, small_dataset):
        x, _, _ = small_dataset
        assert np.all(np.isfinite(x))

    def test_primary_feature_correlates_with_target(self, small_dataset):
        """log10(mean NIS) must rise with the true Q scale, or the model has nothing."""
        x, y, _ = small_dataset
        assert float(np.corrcoef(x[:, 0], y)[0, 1]) > 0.7

    def test_bad_n_runs_raises(self):
        with pytest.raises(ValueError, match="n_runs"):
            generate_adaptive_dataset(n_runs=0)

    def test_too_few_steps_raises(self):
        with pytest.raises(ValueError, match="n_steps"):
            generate_adaptive_dataset(n_runs=2, n_steps=10)

    def test_bad_range_raises(self):
        with pytest.raises(ValueError, match="log10_scale_range"):
            generate_adaptive_dataset(n_runs=2, n_steps=200, log10_scale_range=(1.0, 1.0))


class TestLearnedAdaptiveQ:
    def test_unfitted_flag(self):
        assert not LearnedAdaptiveQ().fitted

    def test_fitted_flag(self, fitted_model):
        assert fitted_model.fitted
        assert fitted_model.n_train > 0

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            LearnedAdaptiveQ().predict(np.zeros(N_FEATURES))

    def test_prediction_fields(self, fitted_model, small_dataset):
        x, _, _ = small_dataset
        p = fitted_model.predict(x[0])
        assert isinstance(p.log10_scale, float)
        assert p.log10_std >= 0.0
        assert 0.0 < p.confidence <= 1.0
        assert p.scale == pytest.approx(10.0**p.log10_scale, rel=1e-9)
        assert isinstance(p.extrapolating, bool)

    def test_in_domain_query_is_not_extrapolating(self, fitted_model, small_dataset):
        x, _, _ = small_dataset
        assert not fitted_model.predict(np.median(x, axis=0)).extrapolating

    def test_out_of_domain_flagged(self, fitted_model):
        assert fitted_model.predict(np.full(N_FEATURES, 1e6)).extrapolating

    def test_scale_is_clipped(self):
        x = np.random.default_rng(0).standard_normal((100, N_FEATURES))
        y = np.full(100, 10.0)
        m = LearnedAdaptiveQ(n_members=2, n_estimators=10, max_scale=8.0).fit(x, y)
        assert m.predict(x[0]).scale <= 8.0

    def test_predicts_the_training_signal(self, fitted_model, small_dataset):
        """On its own training data the model must at least correlate with the target."""
        x, y, _ = small_dataset
        mean, _ = fitted_model.predict_batch(x)
        assert float(np.corrcoef(mean, y)[0, 1]) > 0.85

    def test_predict_batch_shapes(self, fitted_model, small_dataset):
        x, _, _ = small_dataset
        mean, std = fitted_model.predict_batch(x[:20])
        assert mean.shape == (20,)
        assert std.shape == (20,)
        assert np.all(std >= 0.0)

    def test_batch_agrees_with_single(self, fitted_model, small_dataset):
        x, _, _ = small_dataset
        mean, std = fitted_model.predict_batch(x[:5])
        for i in range(5):
            p = fitted_model.predict(x[i])
            assert p.log10_scale == pytest.approx(float(mean[i]))
            # predict() uses ddof=1, predict_batch() uses ddof=1 as well
            assert p.log10_std == pytest.approx(float(std[i]), rel=1e-9)

    def test_deterministic_for_a_seed(self, small_dataset):
        x, y, _ = small_dataset
        a = LearnedAdaptiveQ(n_members=2, n_estimators=20, random_state=3).fit(x, y)
        b = LearnedAdaptiveQ(n_members=2, n_estimators=20, random_state=3).fit(x, y)
        assert a.predict(x[0]).log10_scale == pytest.approx(b.predict(x[0]).log10_scale)

    def test_single_member_rejected(self):
        with pytest.raises(ValueError, match="n_members"):
            LearnedAdaptiveQ(n_members=1)

    def test_bad_n_estimators_rejected(self):
        with pytest.raises(ValueError, match="n_estimators"):
            LearnedAdaptiveQ(n_estimators=0)

    def test_bad_clip_bounds_rejected(self):
        with pytest.raises(ValueError, match="min_scale"):
            LearnedAdaptiveQ(min_scale=10.0, max_scale=1.0)

    def test_wrong_feature_count_raises(self, small_dataset):
        x, y, _ = small_dataset
        with pytest.raises(ValueError, match="shape"):
            LearnedAdaptiveQ(n_members=2, n_estimators=5).fit(x[:, :3], y)

    def test_target_length_mismatch_raises(self, small_dataset):
        x, y, _ = small_dataset
        with pytest.raises(ValueError, match="elements"):
            LearnedAdaptiveQ(n_members=2, n_estimators=5).fit(x, y[:-1])

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 20"):
            LearnedAdaptiveQ(n_members=2, n_estimators=5).fit(
                np.zeros((5, N_FEATURES)), np.zeros(5)
            )

    def test_nonfinite_training_data_raises(self, small_dataset):
        x, y, _ = small_dataset
        bad = x.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            LearnedAdaptiveQ(n_members=2, n_estimators=5).fit(bad, y)

    def test_nonfinite_query_raises(self, fitted_model):
        with pytest.raises(ValueError, match="finite"):
            fitted_model.predict(np.full(N_FEATURES, np.nan))

    def test_wrong_query_length_raises(self, fitted_model):
        with pytest.raises(ValueError, match="elements"):
            fitted_model.predict(np.zeros(3))


class TestRunAdaptiveKf:
    def _setup(self, seed=0, u=1.0, n=200):
        f, q_nom = constant_velocity_cwna(1.0, 0.05)
        _, q_true = constant_velocity_cwna(1.0, 0.05 * 10.0**u)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        rng = np.random.default_rng(seed)
        truth, meas = simulate_linear_system(
            f, h, q_true, r, np.array([0.0, 1.0]), n, rng
        )
        return f, h, q_nom, r, truth, meas

    def test_fixed_keeps_scale_one(self):
        f, h, q_nom, r, _, meas = self._setup()
        res = run_adaptive_kf(
            f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.diag([100.0, 10.0]),
            measurements=meas, tuner="fixed",
        )
        assert np.allclose(res.scales, 1.0)
        assert np.all(np.isnan(res.confidences))

    def test_fixed_matches_a_plain_kf(self):
        f, h, q_nom, r, _, meas = self._setup()
        p0 = np.diag([100.0, 10.0])
        res = run_adaptive_kf(
            f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=p0,
            measurements=meas, tuner="fixed",
        )
        ref = KalmanFilter(f, h, q_nom, r, np.zeros(2), p0).run(meas)
        assert np.allclose(res.states, ref.x_post)
        assert np.allclose(res.covariances, ref.p_post)

    def test_mehra_adapts_upward_when_q_is_underestimated(self):
        f, h, q_nom, r, _, meas = self._setup(seed=1, u=1.5, n=400)
        res = run_adaptive_kf(
            f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.diag([100.0, 10.0]),
            measurements=meas, tuner="mehra",
        )
        assert res.scales[-1] > 1.0

    def test_learned_adapts_upward(self, fitted_model):
        f, h, q_nom, r, _, meas = self._setup(seed=2, u=1.5, n=400)
        res = run_adaptive_kf(
            f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.diag([100.0, 10.0]),
            measurements=meas, tuner="learned", model=fitted_model,
        )
        assert res.scales[-1] > 1.0
        assert np.any(np.isfinite(res.confidences))

    def test_learned_adapts_downward(self, fitted_model):
        f, h, q_nom, r, _, meas = self._setup(seed=3, u=-1.5, n=400)
        res = run_adaptive_kf(
            f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.diag([100.0, 10.0]),
            measurements=meas, tuner="learned", model=fitted_model,
        )
        assert res.scales[-1] < 1.0

    def test_scale_constant_before_the_first_window(self, fitted_model):
        f, h, q_nom, r, _, meas = self._setup(seed=4, u=1.0, n=200)
        res = run_adaptive_kf(
            f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.diag([100.0, 10.0]),
            measurements=meas, tuner="learned", model=fitted_model, window=40,
            update_every=20,
        )
        assert np.allclose(res.scales[:40], 1.0)

    def test_result_shapes(self):
        f, h, q_nom, r, _, meas = self._setup(n=120)
        res = run_adaptive_kf(
            f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.diag([100.0, 10.0]),
            measurements=meas, tuner="mehra",
        )
        assert res.states.shape == (120, 2)
        assert res.covariances.shape == (120, 2, 2)
        assert res.innovations.shape == (120, 1)
        assert res.scales.shape == (120,)

    def test_bad_tuner_raises(self):
        f, h, q_nom, r, _, meas = self._setup()
        with pytest.raises(ValueError, match="tuner"):
            run_adaptive_kf(
                f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.eye(2),
                measurements=meas, tuner="magic",
            )

    def test_learned_without_model_raises(self):
        f, h, q_nom, r, _, meas = self._setup()
        with pytest.raises(ValueError, match="fitted model"):
            run_adaptive_kf(
                f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.eye(2),
                measurements=meas, tuner="learned",
            )

    def test_learned_with_unfitted_model_raises(self):
        f, h, q_nom, r, _, meas = self._setup()
        with pytest.raises(RuntimeError, match="not fitted"):
            run_adaptive_kf(
                f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.eye(2),
                measurements=meas, tuner="learned", model=LearnedAdaptiveQ(),
            )

    def test_bad_window_raises(self):
        f, h, q_nom, r, _, meas = self._setup()
        with pytest.raises(ValueError, match="window"):
            run_adaptive_kf(
                f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.eye(2),
                measurements=meas, tuner="mehra", window=2,
            )

    def test_bad_cadence_raises(self):
        f, h, q_nom, r, _, meas = self._setup()
        with pytest.raises(ValueError, match="update_every"):
            run_adaptive_kf(
                f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.eye(2),
                measurements=meas, tuner="mehra", update_every=0,
            )

    def test_wrong_measurement_width_raises(self):
        f, h, q_nom, r, _, _ = self._setup()
        with pytest.raises(ValueError, match="columns"):
            run_adaptive_kf(
                f=f, h=h, q_nominal=q_nom, r=r, x0=np.zeros(2), p0=np.eye(2),
                measurements=np.zeros((50, 3)), tuner="fixed",
            )
