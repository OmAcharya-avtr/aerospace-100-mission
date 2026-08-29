"""Failure-mode tests (Level 3 requirement).

These exercise the ways a navigation filter actually breaks in service:
covariance collapse, sensor dropout, gross model mis-specification, and
divergence.  Each test asserts the failure is *detected and reported*, not
that it cannot happen.
"""

from __future__ import annotations

import numpy as np
import pytest

from navbench import (
    CovarianceCollapseError,
    ExtendedKalmanFilter,
    GpsModel,
    GyroModel,
    KalmanFilter,
    MerweSigmaPoints,
    MultiplicativeEKF,
    StarTrackerModel,
    UnscentedKalmanFilter,
    consistency_test,
    constant_velocity_2d,
    constant_velocity_cwna,
    covariance_health,
    ensemble_consistency,
    innovation_whiteness,
    nees,
    nis,
    quat_from_small_angle,
    quat_identity,
    quat_multiply,
    radar_jacobian,
    radar_measurement,
    score_run,
    simulate_linear_system,
    simulate_radar_scenario,
    steady_state_riccati,
)


class TestCovarianceCollapse:
    def test_cholesky_failure_is_reported_not_silently_regularised(self):
        with pytest.raises(CovarianceCollapseError, match="positive definiteness"):
            MerweSigmaPoints(n=3).sigma_points(np.zeros(3), np.diag([1.0, 1.0, -1e-9]))

    def test_kf_reports_indefinite_innovation_covariance(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        kf.p = np.diag([-1.0, -1.0])
        kf.r = np.array([[1e-30]])
        with pytest.raises(CovarianceCollapseError):
            kf.update([0.0])

    def test_covariance_health_flags_a_collapsed_matrix(self):
        h = covariance_health(np.diag([1.0, 1e-18]))
        assert h["condition"] > 1e15
        assert h["min_eigenvalue"] < 1e-15

    def test_zero_process_noise_and_perfect_measurements_drive_p_to_zero(self):
        """The classic collapse route: Q = 0 makes P shrink without bound until
        the filter ignores all further data. The filter does not crash, but the
        condition number explodes and the health check must show it."""
        f = np.array([[1.0]])
        h = np.array([[1.0]])
        kf = KalmanFilter(f, h, np.zeros((1, 1)), np.array([[1.0]]), np.zeros(1),
                          np.array([[100.0]]))
        for _ in range(5000):
            kf.predict()
            kf.update([0.0])
        assert float(kf.p[0, 0]) < 1e-3
        assert covariance_health(kf.p)["min_eigenvalue"] > 0.0

    def test_joseph_form_stays_psd_in_float32_ill_conditioned_geometry(self):
        """Two nearly parallel measurement rows (relative information 1e-6),
        R = 1e-8 I, float32 arithmetic, 500 updates with no re-symmetrisation.

        MEASURED RESULT, stated as found: in *this* configuration BOTH the
        Joseph form (min eigenvalue +2.77e-11) and the short form (+1.86e-11)
        remain positive definite. Ill conditioning alone was not enough to
        break the short form here, and this test does not pretend otherwise —
        it asserts only what it can: the Joseph form survives. The case where
        the two genuinely diverge is a sub-optimal gain (next test), which is
        an algebraic property rather than a round-off accident.
        """
        p = np.eye(2, dtype=np.float32)
        h = np.array([[1.0, 1.0], [1.0, 1.001]], dtype=np.float32)
        r = (1e-8 * np.eye(2)).astype(np.float32)
        for _ in range(500):
            s = h @ p @ h.T + r
            k = p @ h.T @ np.linalg.inv(s)
            a = np.eye(2, dtype=np.float32) - k @ h
            p = (a @ p @ a.T + k @ r @ k.T).astype(np.float32)
        assert np.all(np.isfinite(p))
        assert float(np.linalg.eigvalsh(0.5 * (p + p.T)).min()) > 0.0

    def test_short_form_loses_positive_definiteness_at_a_suboptimal_gain(self):
        """The Joseph form is valid for ANY gain; the short form only for the
        optimal one. Over-relaxing the gain by 1.5x makes (I - KH)P indefinite
        while the Joseph form stays positive definite. Fixed-gain (alpha-beta)
        implementations, gain scheduling and quantised coefficients all produce
        non-optimal gains routinely, so this is an operational failure mode."""
        p = np.array([[1.0, 0.2], [0.2, 0.5]])
        h = np.array([[1.0, 0.0]])
        r = np.array([[0.01]])
        k_opt = p @ h.T @ np.linalg.inv(h @ p @ h.T + r)
        k = 1.5 * k_opt
        a = np.eye(2) - k @ h
        joseph = a @ p @ a.T + k @ r @ k.T
        short = a @ p
        assert float(np.linalg.eigvalsh(0.5 * (joseph + joseph.T)).min()) > 0.0
        assert float(np.linalg.eigvalsh(0.5 * (short + short.T)).min()) < 0.0

    def test_mekf_collapse_is_raised(self):
        m = MultiplicativeEKF(sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity())
        m.p = -np.eye(6)
        with pytest.raises(CovarianceCollapseError):
            m.update_quaternion(quat_identity(), 1e-12)


class TestSensorDropout:
    def test_kf_prediction_only_across_a_gap(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 120, rng)
        z[40:80] = np.nan
        res = KalmanFilter(f, h, q, r, x0, p0).run(z)
        assert not np.any(res.updated[40:80])
        assert np.allclose(res.x_post[40:80], res.x_prior[40:80])

    def test_covariance_grows_monotonically_during_a_gap(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 120, rng)
        z[40:80] = np.nan
        res = KalmanFilter(f, h, q, r, x0, p0).run(z)
        tr = np.trace(res.p_post[40:80], axis1=1, axis2=2)
        assert np.all(np.diff(tr) > 0.0)

    def test_filter_recovers_after_the_gap(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        truth, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 200, rng)
        z_gap = z.copy()
        z_gap[60:100] = np.nan
        res = KalmanFilter(f, h, q, r, x0, p0).run(z_gap)
        err_after = np.abs(truth[150:, 0] - res.x_post[150:, 0])
        assert float(np.mean(err_after)) < 5.0

    def test_ekf_and_ukf_handle_dropout(self, rng):
        f, q = constant_velocity_2d(1.0, 0.05)
        r = np.diag([400.0, 1e-4])
        truth, meas = simulate_radar_scenario(
            dt=1.0, n_steps=120, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
            x0=np.array([5000.0, -5.0, 5000.0, 3.0]), rng=rng,
        )
        meas[50:70] = np.nan
        p0 = np.diag([400.0, 100.0, 400.0, 100.0])
        x0 = np.array([truth[0, 0], 0.0, truth[0, 2], 0.0])
        for flt in (
            ExtendedKalmanFilter(
                lambda x: f @ x, radar_measurement, q, r, x0, p0,
                f_jac=lambda x: f, h_jac=radar_jacobian,
            ),
            UnscentedKalmanFilter(lambda x: f @ x, radar_measurement, q, r, x0, p0),
        ):
            res = flt.run(meas)
            assert not np.any(res.updated[50:70])
            assert np.all(np.isfinite(res.x_post))

    def test_mekf_survives_star_tracker_dropout(self, short_attitude_truth, gyro_sigmas):
        sv, su = gyro_sigmas
        rng = np.random.default_rng(11)
        gyro = GyroModel(sigma_v=sv, sigma_u=su, dt=0.5)
        rates, _ = gyro.sample_series(short_attitude_truth.interval_rate(), rng)
        st = StarTrackerModel(sigma_rad=3e-5, reference_vectors=np.eye(3))
        q_meas = np.array(
            [st.sample_quaternion(q, rng) for q in short_attitude_truth.quat[1:]]
        )
        q_meas[80:140] = np.nan
        m = MultiplicativeEKF(
            sigma_v=sv, sigma_u=su, dt=0.5, quat0=short_attitude_truth.quat[0],
            p0=np.diag([0.05**2] * 3 + [2e-6**2] * 3),
        )
        res = m.run(rates, quat_meas=q_meas, sigma_rad=3e-5, measurement_every=1)
        assert not np.any(res.updated[80:140])
        assert np.all(np.isfinite(res.quat))
        assert np.max(np.abs(np.linalg.norm(res.quat, axis=1) - 1.0)) < 1e-13
        # Attitude uncertainty must grow while coasting on the gyro alone.
        var = res.covariance[:, 0, 0]
        assert var[139] > var[80]

    def test_gps_dropout_flag_propagates(self, rng):
        gps = GpsModel(sigma_pos=2.0, dropout_prob=1.0)
        out = gps.sample(np.zeros(3), rng)
        assert not out.valid
        assert np.all(np.isnan(out.position))

    def test_total_dropout_leaves_the_prior_untouched(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        z = np.full((50, 1), np.nan)
        res = KalmanFilter(f, h, q, r, x0, p0).run(z)
        assert not np.any(res.updated)
        assert np.allclose(res.x_post, 0.0)


class TestGrossMisspecification:
    def test_q_far_too_small_is_flagged_optimistic(self):
        f, q_true = constant_velocity_cwna(1.0, 0.05)
        _, q_bad = constant_velocity_cwna(1.0, 0.05 / 100.0)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        runs = np.zeros((25, 150))
        for i in range(25):
            rng = np.random.default_rng(6000 + i)
            truth, meas = simulate_linear_system(
                f, h, q_true, r, np.array([0.0, 1.0]), 150, rng
            )
            res = KalmanFilter(f, h, q_bad, r, np.zeros(2), np.diag([100.0, 10.0])).run(meas)
            runs[i] = nees(truth - res.x_post, res.p_post)
        avg, lo, hi = ensemble_consistency(runs[:, 20:], 2)
        assert float(np.mean(avg)) > hi

    def test_q_far_too_large_is_flagged_pessimistic(self):
        f, q_true = constant_velocity_cwna(1.0, 0.05)
        _, q_bad = constant_velocity_cwna(1.0, 0.05 * 100.0)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        runs = np.zeros((25, 150))
        for i in range(25):
            rng = np.random.default_rng(6100 + i)
            truth, meas = simulate_linear_system(
                f, h, q_true, r, np.array([0.0, 1.0]), 150, rng
            )
            res = KalmanFilter(f, h, q_bad, r, np.zeros(2), np.diag([100.0, 10.0])).run(meas)
            runs[i] = nees(truth - res.x_post, res.p_post)
        avg, lo, hi = ensemble_consistency(runs[:, 20:], 2)
        assert float(np.mean(avg)) < lo

    def test_r_understated_shows_up_in_nis(self):
        f, q = constant_velocity_cwna(1.0, 0.05)
        h = np.array([[1.0, 0.0]])
        r_true = np.array([[9.0]])
        r_bad = np.array([[0.09]])
        runs = np.zeros((25, 150))
        for i in range(25):
            rng = np.random.default_rng(6200 + i)
            _, meas = simulate_linear_system(
                f, h, q, r_true, np.array([0.0, 1.0]), 150, rng
            )
            res = KalmanFilter(f, h, q, r_bad, np.zeros(2), np.diag([100.0, 10.0])).run(meas)
            runs[i] = nis(res.innovation, res.innovation_cov)
        avg, lo, hi = ensemble_consistency(runs[:, 20:], 1)
        assert float(np.mean(avg)) > hi

    def test_wrong_dynamics_breaks_whiteness(self, rng):
        """The filter assumes constant velocity; the truth accelerates."""
        f, q = constant_velocity_cwna(1.0, 1e-6)
        h = np.array([[1.0, 0.0]])
        r = np.array([[1.0]])
        n = 300
        t = np.arange(n) * 1.0
        truth_pos = 0.5 * 0.02 * t**2
        meas = (truth_pos + rng.standard_normal(n))[:, None]
        res = KalmanFilter(f, h, q, r, np.zeros(2), np.diag([100.0, 10.0])).run(meas)
        assert not innovation_whiteness(res.innovation[50:], max_lag=10).passed

    def test_wrong_gyro_noise_spec_makes_the_mekf_optimistic(
        self, short_attitude_truth, gyro_sigmas
    ):
        sv, su = gyro_sigmas
        rng = np.random.default_rng(21)
        gyro = GyroModel(sigma_v=sv * 10.0, sigma_u=su, dt=0.5)
        rates, biases = gyro.sample_series(short_attitude_truth.interval_rate(), rng)
        st = StarTrackerModel(sigma_rad=3e-5, reference_vectors=np.eye(3))
        q_meas = np.array(
            [st.sample_quaternion(q, rng) for q in short_attitude_truth.quat[1:]]
        )
        m = MultiplicativeEKF(  # filter believes the ORIGINAL, too-small sigma_v
            sigma_v=sv, sigma_u=su, dt=0.5, quat0=short_attitude_truth.quat[0],
            p0=np.diag([0.05**2] * 3 + [2e-6**2] * 3),
        )
        res = m.run(rates, quat_meas=q_meas, sigma_rad=3e-5, measurement_every=4)
        err = res.error_state(short_attitude_truth.quat[1:], biases)
        test = consistency_test(
            nees(err[50:, :3], res.covariance[50:, :3, :3]), 3, independent=False
        )
        assert test.verdict == "optimistic"

    def test_measurement_unit_error_is_caught_by_nis(self, rng):
        """Bearing supplied in degrees instead of radians — a classic integration bug."""
        f, q = constant_velocity_2d(1.0, 0.05)
        r = np.diag([400.0, 1e-4])
        truth, meas = simulate_radar_scenario(
            dt=1.0, n_steps=120, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
            x0=np.array([5000.0, -5.0, 5000.0, 3.0]), rng=rng,
        )
        meas[:, 1] = np.degrees(meas[:, 1])
        p0 = np.diag([400.0, 100.0, 400.0, 100.0])
        x0 = np.array([truth[0, 0], 0.0, truth[0, 2], 0.0])
        res = ExtendedKalmanFilter(
            lambda x: f @ x, radar_measurement, q, r, x0, p0,
            f_jac=lambda x: f, h_jac=radar_jacobian,
        ).run(meas)
        assert consistency_test(res.nis, 2, independent=False).verdict == "optimistic"


class TestDivergence:
    def test_divergence_flag_fires_on_a_grossly_wrong_estimate(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        truth, meas = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 100, rng)
        res = KalmanFilter(f, h, q, r, x0, p0).run(meas)
        bad = res.x_post.copy()
        bad[-1, 0] += 1e4
        assert score_run("bad", truth, bad, res.p_post, burn_in=10).diverged

    def test_divergence_flag_quiet_for_a_healthy_filter(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        truth, meas = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 100, rng)
        res = KalmanFilter(f, h, q, r, x0, p0).run(meas)
        assert not score_run("ok", truth, res.x_post, res.p_post, burn_in=10).diverged

    def test_ekf_diverges_where_the_ukf_does_not_more_often(self):
        """Strongly nonlinear regime; the count is the measured comparison, and
        the test asserts only the ordering, not an absolute count."""
        f, q = constant_velocity_2d(1.0, 5.0)
        r = np.diag([3600.0, 0.1225])
        p0 = np.diag([300.0, 50.0, 300.0, 50.0])
        counts = {}
        for name in ("EKF", "UKF"):
            diverged = 0
            for i in range(20):
                rng = np.random.default_rng(7000 + i)
                truth, meas = simulate_radar_scenario(
                    dt=1.0, n_steps=60, q_psd=5.0, sigma_range=60.0, sigma_bearing=0.35,
                    x0=np.array([600.0, -20.0, 120.0, -4.0]), rng=rng,
                )
                x0 = np.array([truth[0, 0] + 10.0, 0.0, truth[0, 2] + 10.0, 0.0])
                flt = (
                    ExtendedKalmanFilter(
                        lambda x: f @ x, radar_measurement, q, r, x0, p0,
                        f_jac=lambda x: f, h_jac=radar_jacobian,
                    )
                    if name == "EKF"
                    else UnscentedKalmanFilter(
                        lambda x: f @ x, radar_measurement, q, r, x0, p0
                    )
                )
                res = flt.run(meas)
                if score_run(name, truth, res.x_post, res.p_post, burn_in=10).diverged:
                    diverged += 1
            counts[name] = diverged
        assert counts["UKF"] <= counts["EKF"]

    def test_undetectable_pair_fails_the_riccati_solve(self):
        """F = 2I with H observing nothing: P grows without bound, so the
        fixed-point iteration cannot converge and must say so."""
        f = np.array([[2.0, 0.0], [0.0, 2.0]])
        h = np.array([[0.0, 0.0]])
        q = np.eye(2)
        r = np.array([[1.0]])
        with pytest.raises(ValueError, match="did not converge"):
            steady_state_riccati(f, h, q, r, max_iter=200)


class TestNumericalGuards:
    def test_nan_measurement_never_silently_poisons_the_state(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        kf.predict()
        with pytest.raises(ValueError, match="finite"):
            kf.update([np.nan])
        assert np.all(np.isfinite(kf.x))

    def test_inf_measurement_rejected(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        with pytest.raises(ValueError, match="finite"):
            kf.update([np.inf])

    def test_quaternion_norm_never_drifts_in_a_long_mekf_run(self):
        m = MultiplicativeEKF(sigma_v=1e-5, sigma_u=1e-9, dt=0.05, quat0=quat_identity())
        worst = 0.0
        for k in range(20000):
            m.predict([0.03, -0.07, 0.11])
            if k % 20 == 0:
                m.update_quaternion(
                    quat_multiply(m.quat, quat_from_small_angle([1e-6, 0.0, 0.0])), 1e-5
                )
            worst = max(worst, abs(float(np.linalg.norm(m.quat)) - 1.0))
        assert worst < 1e-14

    def test_nees_refuses_a_singular_covariance(self):
        with pytest.raises(ValueError, match="positive definite"):
            nees([[1.0, 1.0]], np.zeros((1, 2, 2)))

    def test_zero_range_measurement_raises_rather_than_dividing_by_zero(self):
        with pytest.raises(ValueError, match="too small"):
            radar_measurement([0.0, 1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="too small"):
            radar_jacobian([0.0, 1.0, 0.0, 1.0])
