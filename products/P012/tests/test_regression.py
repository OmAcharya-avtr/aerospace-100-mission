"""Regression tests: pinned outputs from fixed seeds.

Every number below was produced by running this package on 2026-08-29 and is
committed so that a numerical change anywhere in the chain — truth generator,
sensor models, filters, feature extraction — shows up as a test failure rather
than as a silent drift.

If one of these fails, the correct response is to work out *why* the number
moved, not to re-pin it.
"""

from __future__ import annotations

import numpy as np
import pytest

from navbench import (
    ExtendedKalmanFilter,
    GyroModel,
    KalmanFilter,
    MultiplicativeEKF,
    StarTrackerModel,
    UnscentedKalmanFilter,
    arw_deg_per_sqrt_hour_to_si,
    attitude_trajectory,
    constant_velocity_2d,
    constant_velocity_cwna,
    generate_adaptive_dataset,
    nees,
    quat_from_euler_zyx,
    radar_jacobian,
    radar_measurement,
    rrw_deg_per_hour_1p5_to_si,
    simulate_linear_system,
    simulate_radar_scenario,
    steady_state_riccati,
)

REL = 1e-9


class TestPinnedRiccati:
    def test_cwna_steady_state(self):
        f, q = constant_velocity_cwna(1.0, 0.05)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        p, _, k, iters = steady_state_riccati(f, h, q, r, tol=1e-15)
        assert p[0, 0] == pytest.approx(4.2410415524549325, rel=REL)
        assert p[0, 1] == pytest.approx(0.8136658267512196, rel=REL)
        assert p[1, 1] == pytest.approx(0.28561322799978145, rel=REL)
        assert k[0, 0] == pytest.approx(0.3202951622539565, rel=REL)
        assert k[1, 0] == pytest.approx(0.06145028875015979, rel=REL)
        assert iters == 92


class TestPinnedLinearRun:
    @staticmethod
    @pytest.fixture(scope="class")
    def run():
        f, q = constant_velocity_cwna(1.0, 0.05)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        rng = np.random.default_rng(20260812)
        truth, meas = simulate_linear_system(
            f, h, q, r, np.array([0.0, 1.0]), 100, rng
        )
        res = KalmanFilter(f, h, q, r, np.zeros(2), np.diag([100.0, 10.0])).run(meas)
        return truth, meas, res

    def test_measurement_stream_pinned(self, run):
        _, meas, _ = run
        assert float(meas[0, 0]) == pytest.approx(2.316721540554809, rel=REL)
        assert float(meas[-1, 0]) == pytest.approx(180.06769645777436, rel=REL)

    def test_final_state_pinned(self, run):
        _, _, res = run
        assert res.x_post[-1, 0] == pytest.approx(177.25075329385677, rel=REL)
        assert res.x_post[-1, 1] == pytest.approx(3.7917225877923744, rel=REL)

    def test_final_covariance_pinned(self, run):
        _, _, res = run
        assert res.p_post[-1, 0, 0] == pytest.approx(2.8826564602856095, rel=REL)
        assert res.p_post[-1, 0, 1] == pytest.approx(0.5530525987514384, rel=REL)
        assert res.p_post[-1, 1, 1] == pytest.approx(0.23561322799978185, rel=REL)

    def test_mean_nis_pinned(self, run):
        _, _, res = run
        assert res.mean_nis() == pytest.approx(0.8368013056811614, rel=REL)

    def test_mean_nees_pinned(self, run):
        truth, _, res = run
        vals = nees(truth - res.x_post, res.p_post)
        assert float(np.mean(vals)) == pytest.approx(1.5916411378092683, rel=REL)

    def test_covariance_converges_to_the_steady_state(self, run):
        _, _, res = run
        f, q = constant_velocity_cwna(1.0, 0.05)
        _, p_post_inf, _, _ = steady_state_riccati(
            f, np.array([[1.0, 0.0]]), q, np.array([[9.0]]), tol=1e-15
        )
        assert np.allclose(res.p_post[-1], p_post_inf, rtol=1e-12)


class TestPinnedRadarRun:
    @staticmethod
    @pytest.fixture(scope="class")
    def runs():
        rng = np.random.default_rng(2026)
        truth, meas = simulate_radar_scenario(
            dt=1.0, n_steps=100, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
            x0=np.array([3000.0, -5.0, 3000.0, 3.0]), rng=rng,
        )
        f, q = constant_velocity_2d(1.0, 0.05)
        r = np.diag([400.0, 1e-4])
        p0 = np.diag([400.0, 100.0, 400.0, 100.0])
        x_init = np.array([truth[0, 0], 0.0, truth[0, 2], 0.0])
        ekf = ExtendedKalmanFilter(
            lambda x: f @ x, radar_measurement, q, r, x_init, p0,
            f_jac=lambda x: f, h_jac=radar_jacobian,
        ).run(meas)
        ukf = UnscentedKalmanFilter(
            lambda x: f @ x, radar_measurement, q, r, x_init, p0
        ).run(meas)
        return truth, ekf, ukf

    def test_ekf_final_state_pinned(self, runs):
        _, ekf, _ = runs
        assert np.allclose(
            ekf.x_post[-1],
            [2518.2879485428734, -4.980242378488801, 3278.639454876247, 2.417441041582308],
            rtol=REL,
        )

    def test_ukf_final_state_pinned(self, runs):
        _, _, ukf = runs
        assert np.allclose(
            ukf.x_post[-1],
            [2518.2738193363975, -4.980227019202023, 3278.6214908889947, 2.417425311356619],
            rtol=REL,
        )

    def test_mean_nis_pinned(self, runs):
        _, ekf, ukf = runs
        assert ekf.mean_nis() == pytest.approx(1.853739162606621, rel=REL)
        assert ukf.mean_nis() == pytest.approx(1.8538390368332207, rel=REL)

    def test_ekf_and_ukf_agree_in_this_near_linear_regime(self, runs):
        """Measured max relative state difference 2.65e-05 over 100 steps.
        Not machine precision: the measurement IS nonlinear, just weakly so at
        4 km range with a 10 mrad bearing sigma."""
        _, ekf, ukf = runs
        rel = np.max(np.abs(ekf.x_post - ukf.x_post)) / np.max(np.abs(ekf.x_post))
        assert rel == pytest.approx(2.6456706258562963e-05, rel=1e-6)


class TestPinnedAttitude:
    @staticmethod
    @pytest.fixture(scope="class")
    def truth():
        return attitude_trajectory(
            inertia=np.diag([10.0, 15.0, 20.0]),
            quat0=quat_from_euler_zyx(0.2, -0.1, 0.3),
            omega0=np.array([0.01, -0.02, 0.015]),
            dt=0.5, n_steps=200,
            torque_fn=lambda t, q, w: np.array([1e-5 * np.sin(0.01 * t), 0.0, 0.0]),
        )

    def test_final_quaternion_pinned(self, truth):
        assert np.allclose(
            truth.quat[-1],
            [0.07248670953848685, 0.48243185888060974, -0.4590247868419599,
             0.742496749852803],
            rtol=REL, atol=1e-15,
        )

    def test_final_rate_pinned(self, truth):
        assert np.allclose(
            truth.omega[-1],
            [0.02003096700649171, -0.0007318228305642697, 0.01935973045411701],
            rtol=REL, atol=1e-18,
        )

    def test_mekf_final_state_pinned(self, truth):
        sv = arw_deg_per_sqrt_hour_to_si(0.05)
        su = rrw_deg_per_hour_1p5_to_si(0.5)
        rng = np.random.default_rng(7)
        gyro = GyroModel(sigma_v=sv, sigma_u=su, dt=0.5, bias0=[1e-6, -2e-6, 3e-6])
        rates, biases = gyro.sample_series(truth.interval_rate(), rng)
        st = StarTrackerModel(sigma_rad=3e-5, reference_vectors=np.eye(3))
        q_meas = np.array([st.sample_quaternion(q, rng) for q in truth.quat[1:]])
        mekf = MultiplicativeEKF(
            sigma_v=sv, sigma_u=su, dt=0.5, quat0=truth.quat[0],
            p0=np.diag([0.05**2] * 3 + [2e-6**2] * 3),
        )
        res = mekf.run(rates, quat_meas=q_meas, sigma_rad=3e-5, measurement_every=4)
        assert np.allclose(
            res.quat[-1],
            [0.07248089004915613, 0.4824465732605741, -0.45901473097876905,
             0.7424939738956134],
            rtol=1e-8, atol=1e-14,
        )
        assert np.allclose(
            res.bias[-1],
            [8.195514343849122e-07, -2.255136964910542e-06, 1.8377925382481563e-06],
            rtol=1e-7, atol=1e-14,
        )
        err = res.error_state(truth.quat[1:], biases)
        assert float(np.sqrt(np.mean(np.sum(err[50:, :3] ** 2, axis=1)))) == pytest.approx(
            4.365122314603426e-05, rel=1e-7
        )


class TestPinnedSensors:
    def test_gyro_sample_pinned(self):
        rng = np.random.default_rng(99)
        g = GyroModel(sigma_v=1e-3, sigma_u=1e-5, dt=0.1)
        out = g.sample([0.1, 0.2, 0.3], rng)
        assert np.allclose(
            out.rate,
            [0.10026086989552749, 0.19853138002141948, 0.30015974265514667],
            rtol=REL,
        )

    def test_unit_conversions_pinned(self):
        assert arw_deg_per_sqrt_hour_to_si(0.05) == pytest.approx(
            1.4544410433286081e-05, rel=1e-14
        )
        assert rrw_deg_per_hour_1p5_to_si(0.5) == pytest.approx(
            4.040114009246134e-08, rel=1e-12
        )


class TestPinnedAdaptiveDataset:
    def test_shape_and_first_row_pinned(self):
        x, y, idx = generate_adaptive_dataset(n_runs=5, n_steps=300, seed=4242)
        assert x.shape == (60, 6)
        assert np.allclose(
            x[0],
            [0.056113625159554786, 0.056113625530116035, 0.2915841424770496,
             0.03658257208078934, 0.03658257208078934, 0.1],
            rtol=1e-8, atol=1e-14,
        )
        assert float(y[0]) == pytest.approx(0.8467544257696535, rel=REL)
        assert int(idx[0]) == 0

    def test_regeneration_is_identical(self):
        a = generate_adaptive_dataset(n_runs=3, n_steps=300, seed=4242)
        b = generate_adaptive_dataset(n_runs=3, n_steps=300, seed=4242)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])
