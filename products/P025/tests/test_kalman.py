"""Kalman filter: known answers by hand, validation, and the two DARE routes."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdiscope.kalman import (
    KalmanFilter,
    KalmanState,
    steady_state_covariance,
    symmetrize,
)
from fdiscope.plant import PlantConfig, loop_matrices
from fdiscope.simulate import build_filter


def scalar_filter(q: float = 0.0, r: float = 1.0) -> KalmanFilter:
    return KalmanFilter(f=[[1.0]], h=[[1.0]], q=[[q]], r=[[r]])


class TestUpdateKnownAnswer:
    def test_scalar_update_by_hand(self):
        # F = H = 1, Q = 0, R = 1, prior x = 0, P = 1, measurement z = 2.
        #   y = 2 - 0                 = 2
        #   S = 1*1*1 + 1             = 2
        #   K = 1*1/2                 = 0.5
        #   x+ = 0 + 0.5*2            = 1
        #   P+ = (1-0.5)^2*1 + 0.5^2*1 = 0.25 + 0.25 = 0.5   (Joseph)
        #   NIS = 2 * (1/2) * 2       = 2
        kf = scalar_filter()
        res = kf.update(KalmanState(x=np.array([0.0]), p=np.array([[1.0]])), [2.0])
        assert np.isclose(res.innovation[0], 2.0)
        assert np.isclose(res.innovation_cov[0, 0], 2.0)
        assert np.isclose(res.gain[0, 0], 0.5)
        assert np.isclose(res.state.x[0], 1.0)
        assert np.isclose(res.state.p[0, 0], 0.5)
        assert np.isclose(res.nis, 2.0)

    def test_two_dimensional_identity_measurement(self):
        # H = I, P = 4I, R = 4I -> S = 8I, K = 0.5 I,
        # y = z - x = (1, -1) -> NIS = (1^2 + 1^2)/8 = 0.25
        kf = KalmanFilter(f=np.eye(2), h=np.eye(2), q=np.zeros((2, 2)), r=4.0 * np.eye(2))
        res = kf.update(KalmanState(x=np.zeros(2), p=4.0 * np.eye(2)), [1.0, -1.0])
        assert np.allclose(res.gain, 0.5 * np.eye(2))
        assert np.isclose(res.nis, 0.25)

    def test_zero_measurement_noise_gives_unit_gain(self):
        kf = KalmanFilter(f=[[1.0]], h=[[1.0]], q=[[0.0]], r=[[1e-14]])
        res = kf.update(KalmanState(x=np.array([0.0]), p=np.array([[1.0]])), [3.0])
        assert np.isclose(res.gain[0, 0], 1.0, atol=1e-10)
        assert np.isclose(res.state.x[0], 3.0, atol=1e-10)


class TestPredictKnownAnswer:
    def test_double_integrator_prediction(self):
        # F = [[1, 0.1], [0, 1]], G = [[0.005], [0.1]], x = (1, 2), u = 3:
        #   x- = (1 + 0.2 + 0.015, 2 + 0.3) = (1.215, 2.3)
        kf = KalmanFilter(
            f=[[1.0, 0.1], [0.0, 1.0]],
            h=np.eye(2),
            q=np.zeros((2, 2)),
            r=np.eye(2),
            g=[[0.005], [0.1]],
        )
        out = kf.predict(KalmanState(x=np.array([1.0, 2.0]), p=np.eye(2)), [3.0])
        assert np.allclose(out.x, [1.215, 2.3])

    def test_covariance_prediction_adds_q(self):
        kf = KalmanFilter(f=np.eye(2), h=np.eye(2), q=2.0 * np.eye(2), r=np.eye(2))
        out = kf.predict(KalmanState(x=np.zeros(2), p=np.eye(2)))
        assert np.allclose(out.p, 3.0 * np.eye(2))


class TestValidation:
    def test_rejects_non_square_f(self):
        with pytest.raises(ValueError, match="square"):
            KalmanFilter(f=[[1.0, 2.0]], h=[[1.0]], q=[[1.0]], r=[[1.0]])

    def test_rejects_h_with_wrong_state_dimension(self):
        with pytest.raises(ValueError, match="h must be"):
            KalmanFilter(f=np.eye(2), h=[[1.0, 2.0, 3.0]], q=np.eye(2), r=[[1.0]])

    def test_rejects_asymmetric_q(self):
        with pytest.raises(ValueError, match="q must be symmetric"):
            KalmanFilter(f=np.eye(2), h=np.eye(2), q=[[1.0, 2.0], [0.0, 1.0]], r=np.eye(2))

    def test_rejects_negative_definite_q(self):
        with pytest.raises(ValueError, match="positive semi-definite"):
            KalmanFilter(f=np.eye(2), h=np.eye(2), q=-np.eye(2), r=np.eye(2))

    def test_rejects_singular_r(self):
        with pytest.raises(ValueError, match="positive definite"):
            KalmanFilter(f=np.eye(2), h=np.eye(2), q=np.eye(2), r=np.zeros((2, 2)))

    def test_rejects_non_finite_entries(self):
        with pytest.raises(ValueError, match="finite"):
            KalmanFilter(f=[[np.inf]], h=[[1.0]], q=[[1.0]], r=[[1.0]])

    def test_predict_requires_u_when_g_present(self):
        kf = KalmanFilter(f=[[1.0]], h=[[1.0]], q=[[1.0]], r=[[1.0]], g=[[1.0]])
        with pytest.raises(ValueError, match="u is required"):
            kf.predict(KalmanState(x=np.zeros(1), p=np.eye(1)))

    def test_predict_rejects_u_without_g(self):
        with pytest.raises(ValueError, match="no control matrix"):
            scalar_filter().predict(KalmanState(x=np.zeros(1), p=np.eye(1)), [1.0])

    def test_update_rejects_wrong_measurement_length(self):
        with pytest.raises(ValueError, match="must have 1 elements"):
            scalar_filter().update(KalmanState(x=np.zeros(1), p=np.eye(1)), [1.0, 2.0])

    def test_rejects_g_with_wrong_row_count(self):
        with pytest.raises(ValueError, match="g must be"):
            KalmanFilter(f=np.eye(2), h=np.eye(2), q=np.eye(2), r=np.eye(2), g=[[1.0]])


class TestSteadyState:
    def test_dare_and_iteration_agree(self):
        kf = build_filter(loop_matrices(PlantConfig()))
        p_dare, s_dare = steady_state_covariance(kf, method="dare")
        p_iter, s_iter = steady_state_covariance(kf, method="iterate")
        assert np.allclose(p_dare, p_iter, rtol=1e-9)
        assert np.allclose(s_dare, s_iter, rtol=1e-11)

    def test_fixed_point_satisfies_the_riccati_equation(self):
        kf = build_filter(loop_matrices(PlantConfig()))
        p, s = steady_state_covariance(kf)
        k = p @ kf.h.T @ np.linalg.inv(s)
        ikh = np.eye(2) - k @ kf.h
        p_post = ikh @ p @ ikh.T + k @ kf.r @ k.T
        p_next = kf.f @ p_post @ kf.f.T + kf.q
        assert np.allclose(p_next, p, rtol=1e-9)

    def test_innovation_covariance_matches_definition(self):
        kf = build_filter(loop_matrices(PlantConfig()))
        p, s = steady_state_covariance(kf)
        assert np.allclose(s, kf.h @ p @ kf.h.T + kf.r)

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method must be"):
            steady_state_covariance(build_filter(loop_matrices(PlantConfig())), method="magic")

    def test_iteration_raises_if_it_cannot_converge(self):
        kf = build_filter(loop_matrices(PlantConfig()))
        with pytest.raises(RuntimeError, match="did not converge"):
            steady_state_covariance(kf, max_iter=2, tol=1e-16, method="iterate")


class TestProperties:
    def test_symmetrize_is_idempotent(self):
        a = np.array([[1.0, 2.0], [4.0, 8.0]])
        once = symmetrize(a)
        assert np.allclose(symmetrize(once), once)
        assert np.allclose(once, once.T)

    @settings(max_examples=50, deadline=None)
    @given(
        z=st.floats(-10.0, 10.0, allow_nan=False),
        r=st.floats(0.01, 10.0, allow_nan=False),
        p=st.floats(0.01, 10.0, allow_nan=False),
    )
    def test_posterior_variance_never_exceeds_the_prior(self, z, r, p):
        kf = scalar_filter(r=r)
        res = kf.update(KalmanState(x=np.array([0.0]), p=np.array([[p]])), [z])
        assert res.state.p[0, 0] <= p + 1e-12

    @settings(max_examples=50, deadline=None)
    @given(
        z=st.floats(-10.0, 10.0, allow_nan=False),
        r=st.floats(0.01, 10.0, allow_nan=False),
    )
    def test_nis_equals_squared_innovation_over_s(self, z, r):
        kf = scalar_filter(r=r)
        res = kf.update(KalmanState(x=np.array([0.0]), p=np.array([[1.0]])), [z])
        assert np.isclose(res.nis, z * z / (1.0 + r), rtol=1e-12)

    def test_state_copy_is_deep(self):
        state = KalmanState(x=np.zeros(2), p=np.eye(2))
        clone = state.copy()
        clone.x[0] = 5.0
        clone.p[0, 0] = 9.0
        assert state.x[0] == 0.0
        assert state.p[0, 0] == 1.0
