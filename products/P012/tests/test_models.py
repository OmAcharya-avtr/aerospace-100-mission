"""Unit tests for navbench.models — canonical dynamic and measurement models."""

from __future__ import annotations

import numpy as np
import pytest

from navbench import (
    constant_velocity_2d,
    constant_velocity_cwna,
    constant_velocity_dwna,
    radar_jacobian,
    radar_measurement,
    random_walk,
    simulate_linear_system,
    simulate_radar_scenario,
)


class TestRandomWalk:
    def test_matrices(self):
        f, h, q, r = random_walk(2.0, 3.0)
        assert np.allclose(f, [[1.0]])
        assert np.allclose(h, [[1.0]])
        assert np.allclose(q, [[2.0]])
        assert np.allclose(r, [[3.0]])

    @pytest.mark.parametrize("q,r", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (np.nan, 1.0)])
    def test_invalid_raises(self, q, r):
        with pytest.raises(ValueError):
            random_walk(q, r)


class TestConstantVelocityCwna:
    def test_transition_matrix(self):
        f, _ = constant_velocity_cwna(0.5, 1.0)
        assert np.allclose(f, [[1.0, 0.5], [0.0, 1.0]])

    def test_q_hand_computed(self):
        """q_psd = 2, T = 3: Q = 2*[[9, 4.5],[4.5, 3]]."""
        _, q = constant_velocity_cwna(3.0, 2.0)
        assert np.allclose(q, 2.0 * np.array([[27.0 / 3.0, 9.0 / 2.0], [9.0 / 2.0, 3.0]]))

    def test_q_is_symmetric_psd(self):
        _, q = constant_velocity_cwna(1.0, 0.05)
        assert np.allclose(q, q.T)
        assert np.linalg.eigvalsh(q).min() > 0.0

    def test_q_scales_linearly_with_psd(self):
        _, a = constant_velocity_cwna(1.0, 1.0)
        _, b = constant_velocity_cwna(1.0, 7.0)
        assert np.allclose(b, 7.0 * a)

    def test_zero_psd_allowed(self):
        _, q = constant_velocity_cwna(1.0, 0.0)
        assert np.allclose(q, 0.0)

    @pytest.mark.parametrize("dt", [0.0, -1.0, np.nan])
    def test_bad_dt_raises(self, dt):
        with pytest.raises(ValueError, match="dt"):
            constant_velocity_cwna(dt, 1.0)

    def test_negative_psd_raises(self):
        with pytest.raises(ValueError, match="q_psd"):
            constant_velocity_cwna(1.0, -1.0)


class TestConstantVelocityDwna:
    def test_q_hand_computed(self):
        """sigma_a = 2, T = 1: Q = 4*[[0.25, 0.5],[0.5, 1]]."""
        _, q = constant_velocity_dwna(1.0, 2.0)
        assert np.allclose(q, 4.0 * np.array([[0.25, 0.5], [0.5, 1.0]]))

    def test_q_is_rank_one(self):
        _, q = constant_velocity_dwna(1.0, 0.5)
        assert np.linalg.matrix_rank(q, tol=1e-14) == 1

    def test_q_is_psd(self):
        _, q = constant_velocity_dwna(0.5, 0.3)
        assert np.linalg.eigvalsh(q).min() > -1e-18

    def test_negative_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma_a"):
            constant_velocity_dwna(1.0, -1.0)


class TestConstantVelocity2d:
    def test_block_diagonal(self):
        f, q = constant_velocity_2d(0.5, 0.1)
        f1, q1 = constant_velocity_cwna(0.5, 0.1)
        assert np.allclose(f[:2, :2], f1)
        assert np.allclose(f[2:, 2:], f1)
        assert np.allclose(f[:2, 2:], 0.0)
        assert np.allclose(q[:2, 2:], 0.0)
        assert np.allclose(q[2:, 2:], q1)

    def test_shapes(self):
        f, q = constant_velocity_2d(1.0, 0.05)
        assert f.shape == (4, 4)
        assert q.shape == (4, 4)


class TestRadarMeasurement:
    def test_known_value(self):
        """x = 3, y = 4 -> range 5, bearing atan2(4,3)."""
        z = radar_measurement([3.0, 0.0, 4.0, 0.0])
        assert z[0] == pytest.approx(5.0)
        assert z[1] == pytest.approx(np.arctan2(4.0, 3.0))

    def test_on_axis(self):
        z = radar_measurement([10.0, 0.0, 0.0, 0.0])
        assert z == pytest.approx([10.0, 0.0])

    def test_bearing_wraps_correctly(self):
        z = radar_measurement([-1.0, 0.0, -1e-12, 0.0])
        assert abs(z[1]) > 3.0

    def test_velocity_does_not_affect_measurement(self):
        a = radar_measurement([3.0, 5.0, 4.0, -2.0])
        b = radar_measurement([3.0, -9.0, 4.0, 7.0])
        assert np.allclose(a, b)

    def test_zero_range_raises(self):
        with pytest.raises(ValueError, match="too small"):
            radar_measurement([0.0, 0.0, 0.0, 0.0])

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match="4 elements"):
            radar_measurement([1.0, 2.0])

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            radar_measurement([np.nan, 0.0, 1.0, 0.0])


class TestRadarJacobian:
    def test_hand_computed(self):
        """x = 3, y = 4, r = 5: dr/dx = 0.6, dr/dy = 0.8,
        dtheta/dx = -4/25 = -0.16, dtheta/dy = 3/25 = 0.12."""
        j = radar_jacobian([3.0, 0.0, 4.0, 0.0])
        assert np.allclose(j, [[0.6, 0.0, 0.8, 0.0], [-0.16, 0.0, 0.12, 0.0]])

    def test_shape(self):
        assert radar_jacobian([1.0, 0.0, 1.0, 0.0]).shape == (2, 4)

    def test_velocity_columns_zero(self, rng):
        j = radar_jacobian([100.0, 3.0, 200.0, -4.0])
        assert np.allclose(j[:, [1, 3]], 0.0)

    def test_matches_finite_difference(self, rng):
        x = np.array([1200.0, 2.0, -800.0, -1.0])
        eps = 1e-4
        for i in (0, 2):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            fd = (radar_measurement(xp) - radar_measurement(xm)) / (2 * eps)
            assert np.allclose(fd, radar_jacobian(x)[:, i], rtol=1e-6)

    def test_bearing_row_scales_as_inverse_range(self):
        a = radar_jacobian([100.0, 0.0, 100.0, 0.0])
        b = radar_jacobian([1000.0, 0.0, 1000.0, 0.0])
        assert np.max(np.abs(b[1])) == pytest.approx(np.max(np.abs(a[1])) / 10.0)

    def test_zero_range_raises(self):
        with pytest.raises(ValueError, match="too small"):
            radar_jacobian([0.0, 0.0, 0.0, 0.0])

    def test_wrong_size_raises(self):
        with pytest.raises(ValueError, match="4 elements"):
            radar_jacobian(np.zeros(3))


class TestSimulateLinearSystem:
    def test_shapes(self, cv_model, rng):
        f, q, h, r, _, _ = cv_model
        x, z = simulate_linear_system(f, h, q, r, np.zeros(2), 25, rng)
        assert x.shape == (25, 2)
        assert z.shape == (25, 1)

    def test_deterministic_for_a_seed(self, cv_model):
        f, q, h, r, _, _ = cv_model
        a = simulate_linear_system(f, h, q, r, np.zeros(2), 20, np.random.default_rng(5))
        b = simulate_linear_system(f, h, q, r, np.zeros(2), 20, np.random.default_rng(5))
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])

    def test_noise_free_is_deterministic_propagation(self, rng):
        f, _ = constant_velocity_cwna(1.0, 0.0)
        h = np.array([[1.0, 0.0]])
        x, z = simulate_linear_system(
            f, h, np.zeros((2, 2)), np.array([[1e-30]]), np.array([0.0, 2.0]), 5, rng
        )
        assert np.allclose(x[:, 0], [2.0, 4.0, 6.0, 8.0, 10.0])
        assert np.allclose(z[:, 0], x[:, 0], atol=1e-13)

    def test_singular_q_handled(self, rng):
        f, q = constant_velocity_dwna(1.0, 0.5)
        h = np.array([[1.0, 0.0]])
        x, _ = simulate_linear_system(f, h, q, np.array([[1.0]]), np.zeros(2), 50, rng)
        assert np.all(np.isfinite(x))

    def test_measurement_noise_statistics(self, rng):
        f = np.array([[1.0]])
        h = np.array([[1.0]])
        _, z = simulate_linear_system(
            f, h, np.zeros((1, 1)), np.array([[4.0]]), np.zeros(1), 40000, rng
        )
        assert float(np.std(z)) == pytest.approx(2.0, rel=0.03)

    def test_bad_shapes_raise(self, cv_model, rng):
        f, q, h, r, _, _ = cv_model
        with pytest.raises(ValueError, match="shape"):
            simulate_linear_system(np.eye(3), h, q, r, np.zeros(2), 5, rng)

    def test_bad_n_steps_raises(self, cv_model, rng):
        f, q, h, r, _, _ = cv_model
        with pytest.raises(ValueError, match="n_steps"):
            simulate_linear_system(f, h, q, r, np.zeros(2), 0, rng)

    def test_negative_definite_q_raises(self, cv_model, rng):
        f, _, h, r, _, _ = cv_model
        with pytest.raises(ValueError, match="positive semi-definite"):
            simulate_linear_system(f, h, -np.eye(2), r, np.zeros(2), 5, rng)


class TestSimulateRadarScenario:
    def test_shapes(self, rng):
        truth, meas = simulate_radar_scenario(
            dt=1.0, n_steps=30, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
            x0=np.array([1000.0, -1.0, 1000.0, 1.0]), rng=rng,
        )
        assert truth.shape == (30, 4)
        assert meas.shape == (30, 2)

    def test_measurements_near_true_polar(self, rng):
        truth, meas = simulate_radar_scenario(
            dt=1.0, n_steps=50, q_psd=0.0, sigma_range=1e-6, sigma_bearing=1e-9,
            x0=np.array([1000.0, -1.0, 1000.0, 1.0]), rng=rng,
        )
        for k in range(50):
            assert np.allclose(meas[k], radar_measurement(truth[k]), atol=1e-5)

    def test_deterministic_for_a_seed(self):
        kw = dict(dt=1.0, n_steps=20, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
                  x0=np.array([1000.0, -1.0, 1000.0, 1.0]))
        a = simulate_radar_scenario(rng=np.random.default_rng(9), **kw)
        b = simulate_radar_scenario(rng=np.random.default_rng(9), **kw)
        assert np.array_equal(a[1], b[1])

    @pytest.mark.parametrize("kw", [
        {"sigma_range": 0.0}, {"sigma_bearing": -1.0}, {"n_steps": 0},
    ])
    def test_invalid_raises(self, kw, rng):
        base = dict(dt=1.0, n_steps=10, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
                    x0=np.array([1000.0, -1.0, 1000.0, 1.0]), rng=rng)
        base.update(kw)
        with pytest.raises(ValueError):
            simulate_radar_scenario(**base)

    def test_wrong_x0_size_raises(self, rng):
        with pytest.raises(ValueError, match="4 elements"):
            simulate_radar_scenario(
                dt=1.0, n_steps=5, q_psd=0.05, sigma_range=20.0, sigma_bearing=0.01,
                x0=np.zeros(3), rng=rng,
            )
