"""Unit tests for navbench.ukf — sigma points, the unscented transform, and the UKF."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import random_spd
from navbench import (
    CovarianceCollapseError,
    KalmanFilter,
    MerweSigmaPoints,
    UnscentedKalmanFilter,
    constant_velocity_2d,
    radar_measurement,
    simulate_linear_system,
    unscented_transform,
)


class TestMerweSigmaPoints:
    def test_num_points(self):
        assert MerweSigmaPoints(n=4).num_points == 9

    def test_lambda_default(self):
        # alpha = 1, kappa = 0 -> lambda = 1*(n+0) - n = 0
        assert MerweSigmaPoints(n=3).lambda_ == pytest.approx(0.0)

    def test_lambda_hand_computed(self):
        # alpha = 0.5, n = 2, kappa = 1 -> 0.25*(2+1) - 2 = -1.25
        assert MerweSigmaPoints(n=2, alpha=0.5, kappa=1.0).lambda_ == pytest.approx(-1.25)

    def test_weights_sum_to_one_for_mean(self):
        wm, _ = MerweSigmaPoints(n=4, alpha=0.5, beta=2.0, kappa=1.0).weights()
        assert float(np.sum(wm)) == pytest.approx(1.0)

    def test_covariance_weights_offset(self):
        pts = MerweSigmaPoints(n=3, alpha=0.5, beta=2.0, kappa=0.0)
        wm, wc = pts.weights()
        assert wc[0] - wm[0] == pytest.approx(1.0 - 0.25 + 2.0)
        assert np.allclose(wm[1:], wc[1:])

    def test_sigma_points_centred_on_mean(self, rng):
        pts = MerweSigmaPoints(n=3)
        x = rng.standard_normal(3)
        s = pts.sigma_points(x, random_spd(3, rng))
        assert np.allclose(s[0], x)
        assert np.allclose(np.mean(s[1:], axis=0), x, atol=1e-13)

    def test_sigma_points_shape(self, rng):
        s = MerweSigmaPoints(n=4).sigma_points(np.zeros(4), np.eye(4))
        assert s.shape == (9, 4)

    def test_sigma_points_reconstruct_mean_and_covariance(self, rng):
        n = 4
        pts = MerweSigmaPoints(n=n, alpha=1.0, beta=2.0, kappa=0.0)
        x = rng.standard_normal(n)
        p = random_spd(n, rng)
        wm, wc = pts.weights()
        s = pts.sigma_points(x, p)
        mean, cov = unscented_transform(s, wm, wc)
        assert np.allclose(mean, x, atol=1e-12)
        assert np.allclose(cov, p, atol=1e-10)

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5, np.nan])
    def test_bad_alpha_raises(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            MerweSigmaPoints(n=3, alpha=alpha)

    def test_bad_kappa_raises(self):
        with pytest.raises(ValueError, match="kappa"):
            MerweSigmaPoints(n=3, kappa=-4.0)

    @pytest.mark.parametrize("n", [0, -1])
    def test_bad_n_raises(self, n):
        with pytest.raises(ValueError, match="n must"):
            MerweSigmaPoints(n=n)

    def test_wrong_mean_size_raises(self):
        with pytest.raises(ValueError, match="elements"):
            MerweSigmaPoints(n=3).sigma_points(np.zeros(2), np.eye(3))

    def test_wrong_cov_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            MerweSigmaPoints(n=3).sigma_points(np.zeros(3), np.eye(2))

    def test_indefinite_covariance_raises_collapse(self):
        with pytest.raises(CovarianceCollapseError, match="positive definiteness"):
            MerweSigmaPoints(n=2).sigma_points(np.zeros(2), np.diag([1.0, -1.0]))

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            MerweSigmaPoints(n=2).sigma_points(np.array([np.nan, 0.0]), np.eye(2))


class TestUnscentedTransform:
    def test_exact_for_affine_map(self, rng):
        n = 3
        pts = MerweSigmaPoints(n=n, alpha=1.0, beta=2.0, kappa=0.0)
        wm, wc = pts.weights()
        x = rng.standard_normal(n)
        p = random_spd(n, rng)
        a = rng.standard_normal((2, n))
        b = rng.standard_normal(2)
        s = pts.sigma_points(x, p)
        mean, cov = unscented_transform((a @ s.T).T + b, wm, wc)
        assert np.allclose(mean, a @ x + b, atol=1e-11)
        assert np.allclose(cov, a @ p @ a.T, atol=1e-10)

    def test_noise_covariance_added(self, rng):
        pts = MerweSigmaPoints(n=2)
        wm, wc = pts.weights()
        s = pts.sigma_points(np.zeros(2), np.eye(2))
        _, c0 = unscented_transform(s, wm, wc)
        _, c1 = unscented_transform(s, wm, wc, noise_cov=3.0 * np.eye(2))
        assert np.allclose(c1 - c0, 3.0 * np.eye(2), atol=1e-12)

    def test_result_is_symmetric(self, rng):
        pts = MerweSigmaPoints(n=3)
        wm, wc = pts.weights()
        s = pts.sigma_points(rng.standard_normal(3), random_spd(3, rng))
        _, cov = unscented_transform(s**2, wm, wc)
        assert np.array_equal(cov, cov.T)

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="rows"):
            unscented_transform(np.zeros((5, 2)), np.ones(3), np.ones(3))

    def test_noise_shape_mismatch_raises(self):
        pts = MerweSigmaPoints(n=2)
        wm, wc = pts.weights()
        s = pts.sigma_points(np.zeros(2), np.eye(2))
        with pytest.raises(ValueError, match="noise_cov"):
            unscented_transform(s, wm, wc, noise_cov=np.eye(3))


class TestUkfReducesToKf:
    @pytest.mark.parametrize(
        "alpha,beta,kappa",
        [(1.0, 2.0, 0.0), (1.0, 0.0, 1.0), (0.5, 2.0, 1.0), (0.1, 2.0, 0.0)],
    )
    def test_matches_linear_kf(self, cv_model, rng, alpha, beta, kappa):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 100, rng)
        kf = KalmanFilter(f, h, q, r, x0, p0).run(z)
        ukf = UnscentedKalmanFilter(
            lambda x: f @ x, lambda x: h @ x, q, r, x0, p0,
            alpha=alpha, beta=beta, kappa=kappa,
        ).run(z)
        scale_x = max(1.0, float(np.max(np.abs(kf.x_post))))
        scale_p = float(np.max(np.abs(kf.p_post)))
        # Round-off entering the sigma points is amplified by ~1/alpha^2.
        tol = 1e-11 / alpha**2
        assert np.max(np.abs(kf.x_post - ukf.x_post)) / scale_x < tol
        assert np.max(np.abs(kf.p_post - ukf.p_post)) / scale_p < tol

    def test_gains_match(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 50, rng)
        kf = KalmanFilter(f, h, q, r, x0, p0).run(z)
        ukf = UnscentedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, x0, p0).run(z)
        assert np.allclose(kf.gain, ukf.gain, atol=1e-11)

    def test_nis_matches(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 50, rng)
        kf = KalmanFilter(f, h, q, r, x0, p0).run(z)
        ukf = UnscentedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, x0, p0).run(z)
        assert np.allclose(kf.nis, ukf.nis, rtol=1e-9)


class TestUkfNonlinear:
    def test_tracks_a_radar_target(self, rng):
        from navbench import simulate_radar_scenario

        f, q = constant_velocity_2d(1.0, 0.05)
        r = np.diag([400.0, 1e-4])
        truth, meas = simulate_radar_scenario(
            dt=1.0, n_steps=120, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
            x0=np.array([5000.0, -5.0, 5000.0, 3.0]), rng=rng,
        )
        ukf = UnscentedKalmanFilter(
            lambda x: f @ x, radar_measurement, q, r,
            np.array([truth[0, 0], 0.0, truth[0, 2], 0.0]),
            np.diag([400.0, 100.0, 400.0, 100.0]),
        )
        res = ukf.run(meas)
        err = np.hypot(truth[20:, 0] - res.x_post[20:, 0], truth[20:, 2] - res.x_post[20:, 2])
        assert float(np.sqrt(np.mean(err**2))) < 60.0

    def test_covariance_stays_positive_definite(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        ukf = UnscentedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, x0, p0)
        worst = np.inf
        for _ in range(300):
            ukf.predict()
            ukf.update(rng.standard_normal(1) * 3.0)
            worst = min(worst, float(np.linalg.eigvalsh(ukf.p).min()))
        assert worst > 0.0

    def test_nan_measurement_skipped(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 30, rng)
        z[9] = np.nan
        res = UnscentedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, x0, p0).run(z)
        assert not res.updated[9]
        assert np.isnan(res.nis[9])
        assert res.innovation_cov[9, 0, 0] > 0.0


class TestUkfValidation:
    def test_non_callable_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        with pytest.raises(TypeError, match="callable"):
            UnscentedKalmanFilter(f, lambda x: h @ x, q, r, x0, p0)

    def test_singular_r_raises(self, cv_model):
        f, q, h, _, x0, p0 = cv_model
        with pytest.raises(ValueError, match="positive definite"):
            UnscentedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, np.zeros((1, 1)), x0, p0)

    def test_wrong_q_shape_raises(self, cv_model):
        f, _, h, r, x0, p0 = cv_model
        with pytest.raises(ValueError, match="shape"):
            UnscentedKalmanFilter(lambda x: f @ x, lambda x: h @ x, np.eye(3), r, x0, p0)

    def test_wrong_z_size_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ukf = UnscentedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, x0, p0)
        with pytest.raises(ValueError, match="elements"):
            ukf.update([1.0, 2.0])

    def test_nonfinite_z_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ukf = UnscentedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, x0, p0)
        with pytest.raises(ValueError, match="finite"):
            ukf.update([np.inf])

    def test_bad_f_fun_output_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ukf = UnscentedKalmanFilter(lambda x: np.zeros(3), lambda x: h @ x, q, r, x0, p0)
        with pytest.raises(ValueError, match="f_fun must return"):
            ukf.predict()

    def test_bad_h_fun_output_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ukf = UnscentedKalmanFilter(lambda x: f @ x, lambda x: np.zeros(2), q, r, x0, p0)
        ukf.predict()
        with pytest.raises(ValueError, match="h_fun must return"):
            ukf.update([1.0])

    def test_wrong_measurement_width_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ukf = UnscentedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, x0, p0)
        with pytest.raises(ValueError, match="shape"):
            ukf.run(np.zeros((5, 4)))

    def test_collapsed_covariance_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        ukf = UnscentedKalmanFilter(lambda x: f @ x, lambda x: h @ x, q, r, x0, p0)
        ukf.p = np.diag([-1.0, -1.0])
        with pytest.raises(CovarianceCollapseError):
            ukf.predict()

    def test_nonlinear_measurement_used_in_run(self, rng):
        """Sanity: the radar h is genuinely nonlinear, so the UKF S differs from the EKF's."""
        f, q = constant_velocity_2d(1.0, 5.0)
        r = np.diag([3600.0, 0.1])
        x = np.array([300.0, -10.0, 80.0, -2.5])
        ukf = UnscentedKalmanFilter(
            lambda s: f @ s, radar_measurement, q, r, x, np.diag([300.0, 50.0, 300.0, 50.0])
        )
        ukf.predict()
        out = ukf.update(radar_measurement(x))
        s = np.asarray(out["innovation_cov"])
        assert s.shape == (2, 2)
        assert np.linalg.eigvalsh(s).min() > 0.0
