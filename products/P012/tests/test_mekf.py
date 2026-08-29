"""Unit tests for navbench.mekf — the multiplicative EKF for attitude."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from navbench import (
    CovarianceCollapseError,
    GyroModel,
    MultiplicativeEKF,
    StarTrackerModel,
    attitude_state_transition,
    dcm_from_quat,
    gyro_process_noise,
    nees,
    quat_conjugate,
    quat_from_small_angle,
    quat_identity,
    quat_multiply,
    quat_normalize,
    skew,
    small_angle_from_quat,
)


class TestStateTransition:
    def test_zero_rate_gives_identity_plus_coupling(self):
        phi = attitude_state_transition(np.zeros(3), 0.5)
        assert np.allclose(phi[:3, :3], np.eye(3))
        assert np.allclose(phi[:3, 3:], -0.5 * np.eye(3))
        assert np.allclose(phi[3:, 3:], np.eye(3))
        assert np.allclose(phi[3:, :3], 0.0)

    def test_zero_dt_is_identity(self):
        assert np.allclose(attitude_state_transition([0.1, 0.2, 0.3], 0.0), np.eye(6))

    def test_matches_matrix_exponential(self):
        for w in ([0.01, -0.02, 0.03], [0.5, 0.5, -0.5], [1e-9, 0.0, 0.0]):
            for dt in (0.01, 0.1, 1.0):
                fc = np.zeros((6, 6))
                fc[:3, :3] = -skew(w)
                fc[:3, 3:] = -np.eye(3)
                assert np.allclose(
                    attitude_state_transition(w, dt), expm(fc * dt), atol=1e-13
                )

    def test_series_and_closed_form_branches_agree(self):
        """Just either side of the |omega|*dt = 1e-8 branch switch."""
        w_small = np.array([1e-9, 0.0, 0.0])
        w_large = np.array([1.01e-8, 0.0, 0.0])
        a = attitude_state_transition(w_small, 1.0)
        b = attitude_state_transition(w_large, 1.0)
        assert np.max(np.abs(a - b)) < 1e-8

    def test_attitude_block_is_orthogonal(self):
        phi = attitude_state_transition([0.3, -0.4, 0.5], 2.0)
        r = phi[:3, :3]
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-13)

    def test_attitude_block_equals_transposed_dcm(self):
        """Phi_11 = exp(-[w x] dt) = R(dq(w dt))^T."""
        w = np.array([0.2, -0.1, 0.05])
        dt = 3.0
        assert np.allclose(
            attitude_state_transition(w, dt)[:3, :3],
            dcm_from_quat(quat_from_small_angle(w * dt)).T,
            atol=1e-13,
        )

    def test_nonfinite_rate_raises(self):
        with pytest.raises(ValueError, match="finite"):
            attitude_state_transition([np.nan, 0, 0], 1.0)

    def test_nonfinite_dt_raises(self):
        with pytest.raises(ValueError, match="finite"):
            attitude_state_transition([0, 0, 0], np.inf)


class TestProcessNoise:
    def test_farrenkopf_blocks(self):
        sv, su, dt = 1e-4, 1e-8, 0.5
        q = gyro_process_noise(sv, su, dt)
        assert q[0, 0] == pytest.approx(sv**2 * dt + su**2 * dt**3 / 3.0)
        assert q[0, 3] == pytest.approx(-su**2 * dt**2 / 2.0)
        assert q[3, 3] == pytest.approx(su**2 * dt)

    def test_symmetric(self):
        q = gyro_process_noise(1e-4, 1e-8, 0.5)
        assert np.array_equal(q, q.T)

    def test_positive_semi_definite(self):
        q = gyro_process_noise(1e-4, 1e-6, 1.0)
        assert np.linalg.eigvalsh(q).min() >= -1e-20

    def test_zero_sigmas_give_zero(self):
        assert np.allclose(gyro_process_noise(0.0, 0.0, 1.0), 0.0)

    def test_block_diagonal_in_axes(self):
        q = gyro_process_noise(1e-4, 1e-8, 0.5)
        assert np.allclose(q[:3, :3], q[0, 0] * np.eye(3))
        assert np.allclose(q[3:, 3:], q[3, 3] * np.eye(3))

    def test_scales_with_dt(self):
        a = gyro_process_noise(1e-4, 0.0, 1.0)
        b = gyro_process_noise(1e-4, 0.0, 2.0)
        assert b[0, 0] == pytest.approx(2.0 * a[0, 0])

    @pytest.mark.parametrize("kw", [
        {"sigma_v": -1.0, "sigma_u": 0.0, "dt": 1.0},
        {"sigma_v": 0.0, "sigma_u": -1.0, "dt": 1.0},
        {"sigma_v": 0.0, "sigma_u": 0.0, "dt": 0.0},
        {"sigma_v": np.nan, "sigma_u": 0.0, "dt": 1.0},
    ])
    def test_invalid_raises(self, kw):
        with pytest.raises(ValueError):
            gyro_process_noise(**kw)


class TestMekfConstruction:
    def test_defaults(self):
        m = MultiplicativeEKF(sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity())
        assert np.allclose(m.quat, quat_identity())
        assert np.allclose(m.bias, 0.0)
        assert m.p.shape == (6, 6)

    def test_quat0_is_normalised(self):
        m = MultiplicativeEKF(sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=[2.0, 0.0, 0.0, 0.0])
        assert np.linalg.norm(m.quat) == pytest.approx(1.0)

    def test_process_noise_property_scales(self):
        m = MultiplicativeEKF(
            sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity(), q_scale=4.0
        )
        assert np.allclose(m.process_noise, 4.0 * gyro_process_noise(1e-4, 1e-8, 0.5))

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
    def test_bad_dt_raises(self, bad):
        with pytest.raises(ValueError, match="dt"):
            MultiplicativeEKF(sigma_v=1e-4, sigma_u=1e-8, dt=bad, quat0=quat_identity())

    def test_bad_q_scale_raises(self):
        with pytest.raises(ValueError, match="q_scale"):
            MultiplicativeEKF(
                sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity(), q_scale=0.0
            )

    def test_bad_p0_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            MultiplicativeEKF(
                sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity(), p0=np.eye(3)
            )

    def test_asymmetric_p0_raises(self):
        p = np.eye(6)
        p[0, 1] = 1.0
        with pytest.raises(ValueError, match="symmetric"):
            MultiplicativeEKF(sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity(), p0=p)

    def test_indefinite_p0_raises(self):
        p = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
        with pytest.raises(ValueError, match="positive semi-definite"):
            MultiplicativeEKF(sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity(), p0=p)

    def test_nonfinite_bias0_raises(self):
        with pytest.raises(ValueError, match="finite"):
            MultiplicativeEKF(
                sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity(),
                bias0=[np.nan, 0, 0],
            )


class TestMekfPredict:
    def test_reference_propagates_with_bias_corrected_rate(self):
        m = MultiplicativeEKF(
            sigma_v=0.0, sigma_u=0.0, dt=1.0, quat0=quat_identity(), bias0=[0.1, 0.0, 0.0]
        )
        m.predict([0.1, 0.0, 0.0])
        assert np.allclose(m.quat, quat_identity())

    def test_reference_rotates_when_rate_nonzero(self):
        m = MultiplicativeEKF(sigma_v=0.0, sigma_u=0.0, dt=2.0, quat0=quat_identity())
        m.predict([0.0, 0.0, 0.1])
        assert np.allclose(m.quat, quat_from_small_angle([0.0, 0.0, 0.2]))

    def test_covariance_grows(self):
        m = MultiplicativeEKF(
            sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity(),
            p0=np.diag([1e-6] * 3 + [1e-12] * 3),
        )
        before = float(np.trace(m.p))
        m.predict(np.zeros(3))
        assert float(np.trace(m.p)) > before

    def test_covariance_stays_symmetric(self):
        m = MultiplicativeEKF(sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity())
        for _ in range(50):
            m.predict([0.01, -0.02, 0.03])
        assert np.array_equal(m.p, m.p.T)

    def test_quaternion_stays_unit_norm(self):
        m = MultiplicativeEKF(sigma_v=1e-4, sigma_u=1e-8, dt=0.1, quat0=quat_identity())
        for _ in range(5000):
            m.predict([0.03, -0.07, 0.11])
        assert np.linalg.norm(m.quat) == pytest.approx(1.0, abs=1e-14)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
    def test_bad_dt_override_raises(self, bad):
        m = MultiplicativeEKF(sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="dt"):
            m.predict(np.zeros(3), dt=bad)

    def test_nonfinite_rate_raises(self):
        m = MultiplicativeEKF(sigma_v=1e-4, sigma_u=1e-8, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="finite"):
            m.predict([np.nan, 0, 0])


class TestMekfQuaternionUpdate:
    def test_perfect_measurement_reduces_error(self):
        q_true = quat_normalize([0.9, 0.1, -0.2, 0.3])
        m = MultiplicativeEKF(
            sigma_v=1e-6, sigma_u=1e-10, dt=0.5, quat0=quat_identity(),
            p0=np.diag([1.0] * 3 + [1e-8] * 3),
        )
        before = np.linalg.norm(
            small_angle_from_quat(quat_multiply(quat_conjugate(m.quat), q_true))
        )
        m.update_quaternion(q_true, 1e-6)
        after = np.linalg.norm(
            small_angle_from_quat(quat_multiply(quat_conjugate(m.quat), q_true))
        )
        assert after < 0.01 * before

    def test_reset_folds_error_into_reference(self):
        m = MultiplicativeEKF(
            sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_normalize([0.9, 0.1, 0.2, 0.3]),
            p0=np.diag([0.05**2] * 3 + [1e-10] * 3),
        )
        q_before = m.quat.copy()
        q_meas = quat_normalize(
            quat_multiply(q_before, quat_from_small_angle([1e-3, -2e-3, 5e-4]))
        )
        out = m.update_quaternion(q_meas, 1e-4)
        dx = np.asarray(out["dx"])
        expected = quat_normalize(quat_multiply(q_before, quat_from_small_angle(dx[:3])))
        assert np.allclose(m.quat, expected, atol=1e-15)

    def test_reset_angle_reported(self):
        m = MultiplicativeEKF(
            sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity(),
            p0=np.diag([0.05**2] * 3 + [1e-10] * 3),
        )
        out = m.update_quaternion(quat_from_small_angle([1e-3, 0, 0]), 1e-5)
        assert float(out["reset_angle"]) == pytest.approx(
            float(np.linalg.norm(np.asarray(out["dx"])[:3]))
        )

    def test_second_identical_update_has_tiny_innovation(self):
        m = MultiplicativeEKF(
            sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity(),
            p0=np.diag([0.05**2] * 3 + [1e-10] * 3),
        )
        q_meas = quat_from_small_angle([1e-3, -2e-3, 5e-4])
        n1 = np.linalg.norm(np.asarray(m.update_quaternion(q_meas, 1e-4)["innovation"]))
        n2 = np.linalg.norm(np.asarray(m.update_quaternion(q_meas, 1e-4)["innovation"]))
        assert n2 < 0.2 * n1

    def test_quaternion_stays_unit_after_reset(self):
        m = MultiplicativeEKF(sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity())
        for _ in range(200):
            m.predict([0.01, 0.0, 0.0])
            m.update_quaternion(quat_from_small_angle([1e-4, 0.0, 0.0]), 1e-4)
        assert np.linalg.norm(m.quat) == pytest.approx(1.0, abs=1e-15)

    def test_covariance_shrinks(self):
        m = MultiplicativeEKF(
            sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity(),
            p0=np.diag([0.05**2] * 3 + [1e-10] * 3),
        )
        before = float(np.trace(m.p))
        m.update_quaternion(quat_identity(), 1e-4)
        assert float(np.trace(m.p)) < before

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
    def test_bad_sigma_raises(self, bad):
        m = MultiplicativeEKF(sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="sigma_rad"):
            m.update_quaternion(quat_identity(), bad)


class TestMekfVectorUpdate:
    def test_sensitivity_is_skew_of_predicted_vector(self):
        """H_att = [r_hat_b x]; verified by a first-order finite difference."""
        q = quat_normalize([0.9, 0.1, -0.2, 0.3])
        r_i = np.array([1.0, 0.0, 0.0])
        rot = dcm_from_quat(q).T
        pred = rot @ r_i
        eps = 1e-7
        for axis in range(3):
            a = np.zeros(3)
            a[axis] = eps
            q_pert = quat_multiply(q, quat_from_small_angle(a))
            actual = dcm_from_quat(q_pert).T @ r_i
            predicted = pred + skew(pred) @ a
            assert np.allclose(actual, predicted, atol=1e-12)

    def test_single_vector_reduces_perpendicular_error(self):
        q_true = quat_from_small_angle([0.0, 0.02, 0.0])
        m = MultiplicativeEKF(
            sigma_v=1e-6, sigma_u=1e-10, dt=0.5, quat0=quat_identity(),
            p0=np.diag([0.1**2] * 3 + [1e-12] * 3),
        )
        r_i = np.array([[1.0, 0.0, 0.0]])
        b = (dcm_from_quat(q_true).T @ r_i[0])[None, :]
        before = np.linalg.norm(
            small_angle_from_quat(quat_multiply(quat_conjugate(m.quat), q_true))
        )
        m.update_vectors(b, r_i, 1e-5)
        after = np.linalg.norm(
            small_angle_from_quat(quat_multiply(quat_conjugate(m.quat), q_true))
        )
        assert after < before

    def test_three_vectors_recover_attitude(self):
        q_true = quat_from_small_angle([0.01, -0.02, 0.015])
        m = MultiplicativeEKF(
            sigma_v=1e-6, sigma_u=1e-10, dt=0.5, quat0=quat_identity(),
            p0=np.diag([0.1**2] * 3 + [1e-12] * 3),
        )
        refs = np.eye(3)
        body = (dcm_from_quat(q_true).T @ refs.T).T
        m.update_vectors(body, refs, 1e-6)
        err = np.linalg.norm(
            small_angle_from_quat(quat_multiply(quat_conjugate(m.quat), q_true))
        )
        assert err < 1e-4

    def test_measurement_dimension_scales_with_vector_count(self):
        m = MultiplicativeEKF(sigma_v=1e-6, sigma_u=1e-10, dt=0.5, quat0=quat_identity())
        out = m.update_vectors(np.eye(3), np.eye(3), 1e-5)
        assert np.asarray(out["innovation"]).size == 9
        assert np.asarray(out["innovation_cov"]).shape == (9, 9)

    def test_shape_mismatch_raises(self):
        m = MultiplicativeEKF(sigma_v=1e-6, sigma_u=1e-10, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="shape"):
            m.update_vectors(np.eye(3), np.eye(2, 3), 1e-5)

    def test_empty_vectors_raise(self):
        m = MultiplicativeEKF(sigma_v=1e-6, sigma_u=1e-10, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="at least one"):
            m.update_vectors(np.zeros((0, 3)), np.zeros((0, 3)), 1e-5)

    def test_nonfinite_vectors_raise(self):
        m = MultiplicativeEKF(sigma_v=1e-6, sigma_u=1e-10, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="finite"):
            m.update_vectors([[np.nan, 0, 0]], [[1.0, 0, 0]], 1e-5)


class TestMekfRun:
    def _scenario(self, short_attitude_truth, gyro_sigmas, seed=0):
        sv, su = gyro_sigmas
        rng = np.random.default_rng(seed)
        gyro = GyroModel(sigma_v=sv, sigma_u=su, dt=0.5, bias0=2e-6 * rng.standard_normal(3))
        rates, biases = gyro.sample_series(short_attitude_truth.interval_rate(), rng)
        st = StarTrackerModel(sigma_rad=3e-5, reference_vectors=np.eye(3))
        q_meas = np.array(
            [st.sample_quaternion(q, rng) for q in short_attitude_truth.quat[1:]]
        )
        return rates, biases, q_meas

    def test_shapes(self, short_attitude_truth, gyro_sigmas):
        rates, biases, q_meas = self._scenario(short_attitude_truth, gyro_sigmas)
        sv, su = gyro_sigmas
        m = MultiplicativeEKF(sigma_v=sv, sigma_u=su, dt=0.5, quat0=short_attitude_truth.quat[0])
        res = m.run(rates, quat_meas=q_meas, sigma_rad=3e-5, measurement_every=4)
        n = rates.shape[0]
        assert res.quat.shape == (n, 4)
        assert res.bias.shape == (n, 3)
        assert res.covariance.shape == (n, 6, 6)
        assert res.innovation.shape == (n, 3)
        assert res.nis.shape == (n,)

    def test_updates_at_the_requested_cadence(self, short_attitude_truth, gyro_sigmas):
        rates, _, q_meas = self._scenario(short_attitude_truth, gyro_sigmas)
        sv, su = gyro_sigmas
        m = MultiplicativeEKF(sigma_v=sv, sigma_u=su, dt=0.5, quat0=short_attitude_truth.quat[0])
        res = m.run(rates, quat_meas=q_meas, sigma_rad=3e-5, measurement_every=5)
        idx = np.flatnonzero(res.updated)
        assert np.all((idx + 1) % 5 == 0)

    def test_attitude_converges(self, short_attitude_truth, gyro_sigmas):
        rates, biases, q_meas = self._scenario(short_attitude_truth, gyro_sigmas)
        sv, su = gyro_sigmas
        m = MultiplicativeEKF(
            sigma_v=sv, sigma_u=su, dt=0.5,
            quat0=quat_multiply(short_attitude_truth.quat[0], quat_from_small_angle([0.03, 0.02, -0.01])),
            p0=np.diag([0.05**2] * 3 + [2e-6**2] * 3),
        )
        res = m.run(rates, quat_meas=q_meas, sigma_rad=3e-5, measurement_every=4)
        err = res.attitude_error(short_attitude_truth.quat[1:])
        assert float(np.sqrt(np.mean(np.sum(err[100:] ** 2, axis=1)))) < 1e-4

    def test_error_state_shape_and_content(self, short_attitude_truth, gyro_sigmas):
        rates, biases, q_meas = self._scenario(short_attitude_truth, gyro_sigmas)
        sv, su = gyro_sigmas
        m = MultiplicativeEKF(sigma_v=sv, sigma_u=su, dt=0.5, quat0=short_attitude_truth.quat[0])
        res = m.run(rates, quat_meas=q_meas, sigma_rad=3e-5)
        err = res.error_state(short_attitude_truth.quat[1:], biases)
        assert err.shape == (rates.shape[0], 6)
        assert np.allclose(err[:, 3:], biases - res.bias)

    def test_nees_is_computable_and_finite(self, short_attitude_truth, gyro_sigmas):
        rates, biases, q_meas = self._scenario(short_attitude_truth, gyro_sigmas)
        sv, su = gyro_sigmas
        m = MultiplicativeEKF(
            sigma_v=sv, sigma_u=su, dt=0.5, quat0=short_attitude_truth.quat[0],
            p0=np.diag([0.05**2] * 3 + [2e-6**2] * 3),
        )
        res = m.run(rates, quat_meas=q_meas, sigma_rad=3e-5, measurement_every=4)
        err = res.error_state(short_attitude_truth.quat[1:], biases)
        vals = nees(err[50:], res.covariance[50:])
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= 0.0)

    def test_vector_mode_runs(self, short_attitude_truth, gyro_sigmas):
        sv, su = gyro_sigmas
        rng = np.random.default_rng(4)
        gyro = GyroModel(sigma_v=sv, sigma_u=su, dt=0.5)
        rates, _ = gyro.sample_series(short_attitude_truth.interval_rate(), rng)
        refs = np.eye(3)
        body = np.array(
            [(dcm_from_quat(q).T @ refs.T).T for q in short_attitude_truth.quat[1:]]
        )
        m = MultiplicativeEKF(sigma_v=sv, sigma_u=su, dt=0.5, quat0=short_attitude_truth.quat[0])
        res = m.run(rates, body_vectors=body, reference_vectors=refs, sigma_rad=1e-5)
        assert res.innovation.shape[1] == 9
        assert np.all(res.updated)

    def test_nan_quaternion_measurement_skipped(self, short_attitude_truth, gyro_sigmas):
        rates, _, q_meas = self._scenario(short_attitude_truth, gyro_sigmas)
        q_meas = q_meas.copy()
        q_meas[10] = np.nan
        sv, su = gyro_sigmas
        m = MultiplicativeEKF(sigma_v=sv, sigma_u=su, dt=0.5, quat0=short_attitude_truth.quat[0])
        res = m.run(rates, quat_meas=q_meas, sigma_rad=3e-5, measurement_every=1)
        assert not res.updated[10]
        assert np.isnan(res.nis[10])

    def test_requires_exactly_one_measurement_source(self, short_attitude_truth, gyro_sigmas):
        rates, _, q_meas = self._scenario(short_attitude_truth, gyro_sigmas)
        sv, su = gyro_sigmas
        m = MultiplicativeEKF(sigma_v=sv, sigma_u=su, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="exactly one"):
            m.run(rates)

    def test_bad_measurement_every_raises(self, short_attitude_truth, gyro_sigmas):
        rates, _, q_meas = self._scenario(short_attitude_truth, gyro_sigmas)
        sv, su = gyro_sigmas
        m = MultiplicativeEKF(sigma_v=sv, sigma_u=su, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="measurement_every"):
            m.run(rates, quat_meas=q_meas, measurement_every=0)

    def test_bad_omega_shape_raises(self):
        m = MultiplicativeEKF(sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="shape"):
            m.run(np.zeros((10, 4)), quat_meas=np.zeros((10, 4)))

    def test_bad_quat_meas_shape_raises(self):
        m = MultiplicativeEKF(sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="shape"):
            m.run(np.zeros((10, 3)), quat_meas=np.zeros((5, 4)))

    def test_vector_mode_requires_references(self):
        m = MultiplicativeEKF(sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity())
        with pytest.raises(ValueError, match="reference_vectors"):
            m.run(np.zeros((10, 3)), body_vectors=np.zeros((10, 2, 3)))

    def test_attitude_error_shape_mismatch_raises(self, short_attitude_truth, gyro_sigmas):
        rates, _, q_meas = self._scenario(short_attitude_truth, gyro_sigmas)
        sv, su = gyro_sigmas
        m = MultiplicativeEKF(sigma_v=sv, sigma_u=su, dt=0.5, quat0=quat_identity())
        res = m.run(rates, quat_meas=q_meas, sigma_rad=3e-5)
        with pytest.raises(ValueError, match="shape"):
            res.attitude_error(np.zeros((3, 4)))

    def test_collapsed_covariance_raises(self):
        m = MultiplicativeEKF(sigma_v=1e-5, sigma_u=1e-9, dt=0.5, quat0=quat_identity())
        m.p = -np.eye(6)
        with pytest.raises(CovarianceCollapseError):
            m._update_linear(np.zeros(3), np.hstack([np.eye(3), np.zeros((3, 3))]),
                             1e-30 * np.eye(3))
