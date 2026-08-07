"""Tests for the Rauch-Tung-Striebel fixed-interval smoother."""

from __future__ import annotations

import numpy as np
import pytest

from estimkit import (
    KalmanFilter,
    UnscentedKalmanFilter,
    constant_velocity_cwna,
    min_eigenvalue,
    rts_smooth,
)


def make_track(seed: int = 2026, steps: int = 200):
    """Seeded constant-velocity truth and position measurements."""
    dt, q_psd, r_var = 1.0, 0.01, 4.0
    f, q = constant_velocity_cwna(dt, q_psd)
    h = np.array([[1.0, 0.0]])
    r = np.array([[r_var]])
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(q)
    x = np.array([0.0, 10.0])
    truth = np.empty((steps, 2))
    for k in range(steps):
        x = f @ x + chol @ rng.standard_normal(2)
        truth[k] = x
    z = truth[:, 0:1] + np.sqrt(r_var) * rng.standard_normal((steps, 1))
    return f, h, q, r, truth, z


def test_last_step_equals_filter():
    f, h, q, r, _, z = make_track()
    kf = KalmanFilter(f, h, q, r)
    res = kf.filter(np.zeros(2), np.diag([100.0, 100.0]), z)
    sm = rts_smooth(res)
    assert sm.x[-1] == pytest.approx(res.x_post[-1])
    assert sm.p[-1] == pytest.approx(res.p_post[-1])
    assert sm.gain[-1] == pytest.approx(np.zeros((2, 2)))


def test_smoother_reduces_rms_error():
    f, h, q, r, truth, z = make_track()
    kf = KalmanFilter(f, h, q, r)
    res = kf.filter(np.zeros(2), np.diag([100.0, 100.0]), z)
    sm = rts_smooth(res)
    rms_f = np.sqrt(np.mean((res.x_post - truth) ** 2, axis=0))
    rms_s = np.sqrt(np.mean((sm.x - truth) ** 2, axis=0))
    assert rms_s[0] < rms_f[0]
    assert rms_s[1] < rms_f[1]


def test_smoothed_covariance_never_exceeds_filtered_covariance():
    # Theory: P_{k|T} <= P^+_k, i.e. P^+_k - P_{k|T} is positive semi-definite.
    f, h, q, r, _, z = make_track(seed=4, steps=120)
    kf = KalmanFilter(f, h, q, r)
    res = kf.filter(np.zeros(2), np.diag([100.0, 100.0]), z)
    sm = rts_smooth(res)
    worst = min(min_eigenvalue(res.p_post[k] - sm.p[k]) for k in range(len(res)))
    assert worst >= -1e-12


def test_single_step_smoother_is_identity():
    f, h, q, r, _, z = make_track(steps=1)
    kf = KalmanFilter(f, h, q, r)
    res = kf.filter(np.zeros(2), np.eye(2) * 10.0, z)
    sm = rts_smooth(res)
    assert sm.x == pytest.approx(res.x_post)
    assert sm.p == pytest.approx(res.p_post)


def test_smoother_accepts_explicit_arrays_matching_filter_result():
    f, h, q, r, _, z = make_track(steps=60)
    kf = KalmanFilter(f, h, q, r)
    res = kf.filter(np.zeros(2), np.eye(2) * 10.0, z)
    a = rts_smooth(res)
    b = rts_smooth(
        x_prior=res.x_prior,
        p_prior=res.p_prior,
        x_post=res.x_post,
        p_post=res.p_post,
        transition=f,  # 2-D form broadcast over time
    )
    assert b.x == pytest.approx(a.x)
    assert b.p == pytest.approx(a.p)


def test_smoother_on_ukf_output_matches_kf_smoother_for_linear_model():
    f, h, q, r, _, z = make_track(steps=100)
    kf = KalmanFilter(f, h, q, r)
    ukf = UnscentedKalmanFilter(
        f=lambda x: f @ x, h=lambda x: h @ x,
        process_noise=q, measurement_noise=r, alpha=1.0, beta=2.0, kappa=0.0,
    )
    p0 = np.diag([100.0, 100.0])
    a = rts_smooth(kf.filter(np.zeros(2), p0, z))
    b = rts_smooth(ukf.filter(np.zeros(2), p0, z))
    assert np.max(np.abs(b.x - a.x)) / max(1.0, float(np.max(np.abs(a.x)))) < 1e-12
    assert np.max(np.abs(b.p - a.p)) < 1e-10


def test_smoothed_covariances_are_symmetric_and_psd():
    f, h, q, r, _, z = make_track(seed=8, steps=150)
    kf = KalmanFilter(f, h, q, r)
    sm = rts_smooth(kf.filter(np.zeros(2), np.diag([100.0, 100.0]), z))
    for p in sm.p:
        assert np.max(np.abs(p - p.T)) == 0.0
        assert min_eigenvalue(p) > 0.0


# --------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------- #
def test_missing_arguments_raise():
    with pytest.raises(ValueError, match="missing: x_prior"):
        rts_smooth(p_prior=np.zeros((2, 1, 1)), x_post=np.zeros((2, 1)),
                   p_post=np.zeros((2, 1, 1)), transition=np.eye(1))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="x_prior and x_post"):
        rts_smooth(x_prior=np.zeros((3, 2)), p_prior=np.zeros((3, 2, 2)),
                   x_post=np.zeros((3, 1)), p_post=np.zeros((3, 2, 2)),
                   transition=np.eye(2))


def test_bad_covariance_shape_raises():
    with pytest.raises(ValueError, match="p_prior and p_post must be"):
        rts_smooth(x_prior=np.zeros((3, 2)), p_prior=np.zeros((3, 3, 3)),
                   x_post=np.zeros((3, 2)), p_post=np.zeros((3, 2, 2)),
                   transition=np.eye(2))


def test_bad_transition_shape_raises():
    with pytest.raises(ValueError, match="transition must be"):
        rts_smooth(x_prior=np.zeros((3, 2)), p_prior=np.zeros((3, 2, 2)),
                   x_post=np.zeros((3, 2)), p_post=np.zeros((3, 2, 2)),
                   transition=np.eye(3))


def test_singular_predicted_covariance_raises():
    with pytest.raises(ValueError, match="collapsed"):
        rts_smooth(x_prior=np.zeros((2, 2)), p_prior=np.zeros((2, 2, 2)),
                   x_post=np.zeros((2, 2)), p_post=np.tile(np.eye(2), (2, 1, 1)),
                   transition=np.tile(np.eye(2), (2, 1, 1)))
