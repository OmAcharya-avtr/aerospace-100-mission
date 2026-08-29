"""Unit tests for navbench.kf — linear Kalman filter, Joseph form, steady-state Riccati."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import random_spd
from navbench import (
    CovarianceCollapseError,
    KalmanFilter,
    constant_velocity_cwna,
    constant_velocity_dwna,
    covariance_health,
    joseph_update,
    random_walk,
    simulate_linear_system,
    steady_state_riccati,
    symmetrize,
)


class TestSymmetrize:
    def test_bit_exact_symmetry(self, rng):
        a = rng.standard_normal((5, 5))
        s = symmetrize(a)
        assert np.array_equal(s, s.T)

    def test_preserves_symmetric_input(self, rng):
        a = random_spd(4, rng)
        assert np.allclose(symmetrize(a), a)

    def test_averages_off_diagonal(self):
        a = np.array([[1.0, 3.0], [1.0, 2.0]])
        assert np.allclose(symmetrize(a), [[1.0, 2.0], [2.0, 2.0]])

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            symmetrize(np.zeros((2, 3)))


class TestJosephUpdate:
    def test_zero_gain_leaves_covariance(self, rng):
        p = random_spd(3, rng)
        h = np.array([[1.0, 0.0, 0.0]])
        out = joseph_update(p, np.zeros((3, 1)), h, np.array([[1.0]]))
        assert np.allclose(out, p)

    def test_result_is_symmetric(self, rng):
        p = random_spd(4, rng)
        h = rng.standard_normal((2, 4))
        r = random_spd(2, rng)
        k = rng.standard_normal((4, 2))
        out = joseph_update(p, k, h, r)
        assert np.array_equal(out, out.T)

    def test_agrees_with_short_form_at_optimal_gain(self, rng):
        p = random_spd(4, rng)
        h = rng.standard_normal((2, 4))
        r = random_spd(2, rng)
        s = h @ p @ h.T + r
        k = p @ h.T @ np.linalg.inv(s)
        short = (np.eye(4) - k @ h) @ p
        assert np.allclose(joseph_update(p, k, h, r), symmetrize(short), atol=1e-9)

    def test_stays_psd_at_suboptimal_gain(self, rng):
        """The short form loses positive definiteness here; Joseph does not."""
        p = np.array([[1.0, 0.2], [0.2, 0.5]])
        h = np.array([[1.0, 0.0]])
        r = np.array([[0.01]])
        s = h @ p @ h.T + r
        k_opt = p @ h.T @ np.linalg.inv(s)
        k = 1.5 * k_opt
        jos = joseph_update(p, k, h, r)
        short = (np.eye(2) - k @ h) @ p
        assert np.linalg.eigvalsh(jos).min() > 0.0
        assert np.linalg.eigvalsh(symmetrize(short)).min() < 0.0

    def test_shape_mismatch_raises(self, rng):
        p = random_spd(3, rng)
        with pytest.raises(ValueError, match="shape"):
            joseph_update(p, np.zeros((3, 2)), np.zeros((1, 3)), np.eye(1))

    def test_non_square_p_raises(self):
        with pytest.raises(ValueError, match="square"):
            joseph_update(np.zeros((2, 3)), np.zeros((2, 1)), np.zeros((1, 2)), np.eye(1))


class TestCovarianceHealth:
    def test_identity(self):
        h = covariance_health(np.eye(3))
        assert h["asymmetry"] == 0.0
        assert h["min_eigenvalue"] == pytest.approx(1.0)
        assert h["condition"] == pytest.approx(1.0)
        assert h["trace"] == pytest.approx(3.0)

    def test_detects_asymmetry(self):
        a = np.array([[1.0, 0.5], [0.0, 1.0]])
        assert covariance_health(a)["asymmetry"] == pytest.approx(0.5)

    def test_singular_gives_infinite_condition(self):
        assert covariance_health(np.diag([1.0, 0.0]))["condition"] == float("inf")

    def test_negative_eigenvalue_reported(self):
        assert covariance_health(np.diag([1.0, -2.0]))["min_eigenvalue"] == pytest.approx(-2.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            covariance_health(np.zeros((2, 3)))

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            covariance_health(np.array([[np.nan, 0.0], [0.0, 1.0]]))


class TestKalmanFilterConstruction:
    def test_dimensions(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        assert kf.n == 2
        assert kf.m == 1

    def test_asymmetric_q_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        bad = q.copy()
        bad[0, 1] += 1.0
        with pytest.raises(ValueError, match="symmetric"):
            KalmanFilter(f, h, bad, r, x0, p0)

    def test_singular_r_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        with pytest.raises(ValueError, match="positive definite"):
            KalmanFilter(f, h, q, np.zeros((1, 1)), x0, p0)

    def test_indefinite_p0_raises(self, cv_model):
        f, q, h, r, x0, _ = cv_model
        with pytest.raises(ValueError, match="positive semi-definite"):
            KalmanFilter(f, h, q, r, x0, np.diag([1.0, -1.0]))

    def test_wrong_f_shape_raises(self, cv_model):
        _, q, h, r, x0, p0 = cv_model
        with pytest.raises(ValueError, match="shape"):
            KalmanFilter(np.eye(3), h, q, r, x0, p0)

    def test_wrong_h_columns_raises(self, cv_model):
        f, q, _, r, x0, p0 = cv_model
        with pytest.raises(ValueError, match="shape"):
            KalmanFilter(f, np.array([[1.0, 0.0, 0.0]]), q, r, x0, p0)

    def test_empty_state_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            KalmanFilter(np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)),
                         np.zeros((0, 0)), np.zeros(0), np.zeros((0, 0)))

    def test_nonfinite_x0_raises(self, cv_model):
        f, q, h, r, _, p0 = cv_model
        with pytest.raises(ValueError, match="finite"):
            KalmanFilter(f, h, q, r, np.array([np.nan, 0.0]), p0)

    def test_b_row_mismatch_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        with pytest.raises(ValueError, match="rows"):
            KalmanFilter(f, h, q, r, x0, p0, b=np.ones((3, 1)))


class TestKalmanFilterRecursion:
    def test_predict_propagates_state(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, np.array([0.0, 2.0]), p0)
        x, _ = kf.predict()
        assert np.allclose(x, [2.0, 2.0])

    def test_predict_grows_covariance(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, np.eye(2) * 0.1)
        _, p = kf.predict()
        assert np.trace(p) > np.trace(np.eye(2) * 0.1)

    def test_control_input(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        b = np.array([[0.0], [1.0]])
        kf = KalmanFilter(f, h, q, r, np.zeros(2), p0, b=b)
        x, _ = kf.predict(u=[3.0])
        assert np.allclose(x, [0.0, 3.0])

    def test_control_without_b_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        with pytest.raises(ValueError, match="no b matrix"):
            kf.predict(u=[1.0])

    def test_wrong_u_size_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0, b=np.array([[0.0], [1.0]]))
        with pytest.raises(ValueError, match="elements"):
            kf.predict(u=[1.0, 2.0])

    def test_update_shrinks_covariance(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        kf.predict()
        before = np.trace(kf.p)
        kf.update([1.0])
        assert np.trace(kf.p) < before

    def test_update_returns_expected_keys(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        kf.predict()
        out = kf.update([1.0])
        assert set(out) == {"x", "p", "innovation", "innovation_cov", "gain", "nis"}

    def test_zero_measurement_noise_pins_the_state(self, cv_model):
        f, q, h, _, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, np.array([[1e-12]]), x0, p0)
        kf.predict()
        kf.update([5.0])
        assert kf.x[0] == pytest.approx(5.0, abs=1e-4)

    def test_nis_is_scalar_quadratic_form(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        kf.predict()
        out = kf.update([2.0])
        nu = np.asarray(out["innovation"])
        s = np.asarray(out["innovation_cov"])
        assert out["nis"] == pytest.approx(float(nu @ np.linalg.solve(s, nu)))

    def test_covariance_stays_symmetric_over_a_long_run(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        worst = 0.0
        for _ in range(2000):
            kf.predict()
            kf.update(rng.standard_normal(1) * 3.0)
            worst = max(worst, float(np.max(np.abs(kf.p - kf.p.T))))
        assert worst == 0.0

    def test_covariance_stays_positive_definite(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        worst = np.inf
        for _ in range(2000):
            kf.predict()
            kf.update(rng.standard_normal(1) * 3.0)
            worst = min(worst, float(np.linalg.eigvalsh(kf.p).min()))
        assert worst > 0.0

    def test_wrong_z_size_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        with pytest.raises(ValueError, match="elements"):
            kf.update([1.0, 2.0])

    def test_nonfinite_z_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        with pytest.raises(ValueError, match="finite"):
            kf.update([np.nan])

    def test_per_step_overrides(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        _, p1 = kf.predict(q=10.0 * q)
        kf2 = KalmanFilter(f, h, 10.0 * q, r, x0, p0)
        _, p2 = kf2.predict()
        assert np.allclose(p1, p2)


class TestBatchRun:
    def test_shapes(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 40, rng)
        res = KalmanFilter(f, h, q, r, x0, p0).run(z)
        assert res.x_prior.shape == (40, 2)
        assert res.p_post.shape == (40, 2, 2)
        assert res.innovation.shape == (40, 1)
        assert res.gain.shape == (40, 2, 1)
        assert res.nis.shape == (40,)
        assert res.n_steps == 40

    def test_all_updated_by_default(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 20, rng)
        res = KalmanFilter(f, h, q, r, x0, p0).run(z)
        assert bool(np.all(res.updated))

    def test_nan_measurement_is_skipped(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 20, rng)
        z[5] = np.nan
        res = KalmanFilter(f, h, q, r, x0, p0).run(z)
        assert not res.updated[5]
        assert np.isnan(res.nis[5])
        assert np.allclose(res.x_post[5], res.x_prior[5])

    def test_mask_skips(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 20, rng)
        mask = np.ones(20, dtype=bool)
        mask[3:6] = False
        res = KalmanFilter(f, h, q, r, x0, p0).run(z, mask=mask)
        assert not np.any(res.updated[3:6])

    def test_mean_nis_ignores_skipped(self, cv_model, rng):
        f, q, h, r, x0, p0 = cv_model
        _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 50, rng)
        z[10] = np.nan
        res = KalmanFilter(f, h, q, r, x0, p0).run(z)
        assert np.isfinite(res.mean_nis())

    def test_wrong_measurement_width_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        with pytest.raises(ValueError, match="shape"):
            KalmanFilter(f, h, q, r, x0, p0).run(np.zeros((10, 3)))

    def test_wrong_mask_length_raises(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        with pytest.raises(ValueError, match="elements"):
            KalmanFilter(f, h, q, r, x0, p0).run(np.zeros((10, 1)), mask=np.ones(5, dtype=bool))

    def test_mean_nis_close_to_dof_for_correct_filter(self, cv_model):
        f, q, h, r, _, p0 = cv_model
        nis_means = []
        for i in range(30):
            rng = np.random.default_rng(500 + i)
            _, z = simulate_linear_system(f, h, q, r, np.array([0.0, 1.0]), 300, rng)
            res = KalmanFilter(f, h, q, r, np.zeros(2), p0).run(z)
            nis_means.append(res.mean_nis())
        assert float(np.mean(nis_means)) == pytest.approx(1.0, abs=0.12)


class TestSteadyStateRiccati:
    def test_scalar_random_walk_golden_ratio(self):
        """q = r = 1 gives P^-_inf = phi = (1 + sqrt(5))/2, hand-solved."""
        f, h, q, r = random_walk(1.0, 1.0)
        p_prior, p_post, gain, _ = steady_state_riccati(f, h, q, r)
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        assert float(p_prior[0, 0]) == pytest.approx(phi, abs=1e-12)
        assert float(gain[0, 0]) == pytest.approx(phi / (phi + 1.0), abs=1e-12)
        assert float(p_post[0, 0]) == pytest.approx(phi - 1.0, abs=1e-12)

    @pytest.mark.parametrize("q,r", [(0.25, 4.0), (2.0, 0.5), (100.0, 0.01)])
    def test_scalar_closed_form(self, q, r):
        f, h, qm, rm = random_walk(q, r)
        p_prior, _, gain, _ = steady_state_riccati(f, h, qm, rm)
        p_hand = 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))
        assert float(p_prior[0, 0]) == pytest.approx(p_hand, rel=1e-12)
        assert float(gain[0, 0]) == pytest.approx(p_hand / (p_hand + r), rel=1e-12)

    def test_scalar_closed_form_low_snr_is_looser(self):
        """q/r = 1e-6: the fixed-point iteration converges linearly with a rate
        near 1, so the residual after the increment test is ~50x the increment.
        Measured 5.1e-12 relative at tol = 1e-14 and 1.1e-13 at tol = 1e-16
        (the floating-point floor). This is a property of the solver, documented
        in its docstring, not a defect in the closed form."""
        q, r = 1e-3, 1e3
        f, h, qm, rm = random_walk(q, r)
        p_hand = 0.5 * (q + np.sqrt(q * q + 4.0 * q * r))
        p_loose, _, _, it_loose = steady_state_riccati(f, h, qm, rm, tol=1e-14)
        p_tight, _, _, it_tight = steady_state_riccati(f, h, qm, rm, tol=1e-16)
        err_loose = abs(float(p_loose[0, 0]) - p_hand) / p_hand
        err_tight = abs(float(p_tight[0, 0]) - p_hand) / p_hand
        assert err_loose < 1e-11
        assert err_tight < 1e-12
        assert err_tight < err_loose
        assert it_tight > it_loose

    @pytest.mark.parametrize("dt,sa,sv", [(1.0, 0.1, 2.0), (0.5, 1.0, 0.5), (2.0, 0.02, 10.0)])
    def test_kalata_alpha_beta(self, dt, sa, sv):
        f, q = constant_velocity_dwna(dt, sa)
        h = np.array([[1.0, 0.0]])
        r = np.array([[sv**2]])
        _, _, gain, _ = steady_state_riccati(f, h, q, r)
        lam = sa * dt**2 / sv
        rho = (4.0 + lam - np.sqrt(8.0 * lam + lam * lam)) / 4.0
        alpha = 1.0 - rho**2
        beta = 2.0 * (2.0 - alpha) - 4.0 * np.sqrt(1.0 - alpha)
        assert np.allclose(gain.ravel(), [alpha, beta / dt], atol=1e-12)

    def test_matches_scipy_dare(self):
        from scipy.linalg import solve_discrete_are

        f, q = constant_velocity_cwna(1.0, 0.05)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        p_prior, _, _, _ = steady_state_riccati(f, h, q, r, tol=1e-15)
        assert np.allclose(p_prior, solve_discrete_are(f.T, h.T, q, r), atol=1e-10)

    def test_running_filter_converges_to_it(self, rng):
        f, q = constant_velocity_cwna(1.0, 0.05)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        _, p_post_inf, _, _ = steady_state_riccati(f, h, q, r, tol=1e-15)
        kf = KalmanFilter(f, h, q, r, np.zeros(2), np.diag([1e6, 1e4]))
        for _ in range(400):
            kf.predict()
            kf.update(rng.standard_normal(1) * 3.0)
        assert np.allclose(kf.p, p_post_inf, rtol=1e-11)

    def test_solution_is_symmetric_psd(self):
        f, q = constant_velocity_cwna(1.0, 0.05)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        p_prior, p_post, _, _ = steady_state_riccati(f, h, q, r)
        for p in (p_prior, p_post):
            assert np.array_equal(p, p.T)
            assert np.linalg.eigvalsh(p).min() > 0.0

    def test_posterior_below_prior(self):
        f, q = constant_velocity_cwna(1.0, 0.05)
        h = np.array([[1.0, 0.0]])
        r = np.array([[9.0]])
        p_prior, p_post, _, _ = steady_state_riccati(f, h, q, r)
        assert np.linalg.eigvalsh(p_prior - p_post).min() >= -1e-14

    def test_non_square_f_raises(self):
        with pytest.raises(ValueError, match="square"):
            steady_state_riccati(np.zeros((2, 3)), np.eye(1, 2), np.eye(2), np.eye(1))

    def test_singular_r_raises(self):
        f, h, q, _ = random_walk(1.0, 1.0)
        with pytest.raises(ValueError, match="positive definite"):
            steady_state_riccati(f, h, q, np.zeros((1, 1)))

    def test_bad_tol_raises(self):
        f, h, q, r = random_walk(1.0, 1.0)
        with pytest.raises(ValueError, match="tol"):
            steady_state_riccati(f, h, q, r, tol=0.0)

    def test_non_convergence_raises(self):
        f, h, q, r = random_walk(1.0, 1.0)
        with pytest.raises(ValueError, match="did not converge"):
            steady_state_riccati(f, h, q, r, tol=1e-30, max_iter=5)


class TestCovarianceCollapseError:
    def test_is_a_runtime_error(self):
        assert issubclass(CovarianceCollapseError, RuntimeError)

    def test_raised_on_indefinite_innovation_covariance(self, cv_model):
        f, q, h, r, x0, p0 = cv_model
        kf = KalmanFilter(f, h, q, r, x0, p0)
        kf.p = np.array([[-1.0, 0.0], [0.0, -1.0]])
        kf.r = np.array([[1e-30]])
        with pytest.raises(CovarianceCollapseError, match="positive definite"):
            kf.update([0.0])
