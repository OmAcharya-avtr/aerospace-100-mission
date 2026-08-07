"""Hypothesis property-based tests for the algebraic identities of the filters.

Three properties are exercised:

1. The Joseph-form covariance update keeps the covariance symmetric and
   positive semi-definite for arbitrary (including sub-optimal) gains.
2. A zero-measurement-noise update on an observable system collapses the
   state onto the measurement: H x^+ == z exactly.
3. The scaled unscented transform reproduces the mean and covariance of an
   affine map exactly, for any admissible (alpha, beta, kappa).
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from estimkit import (
    KalmanFilter,
    MerweSigmaPoints,
    is_positive_semidefinite,
    joseph_update,
    min_eigenvalue,
    unscented_transform,
)

SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_finite = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)


def _matrix(rows: int, cols: int) -> st.SearchStrategy[np.ndarray]:
    return st.lists(
        st.lists(_finite, min_size=cols, max_size=cols), min_size=rows, max_size=rows
    ).map(lambda v: np.asarray(v, dtype=float))


@st.composite
def psd_matrix(draw: st.DrawFn, n: int, floor: float = 0.1) -> np.ndarray:
    """A well-conditioned symmetric positive-definite n x n matrix."""
    a = draw(_matrix(n, n))
    return a @ a.T + floor * np.eye(n)


# --------------------------------------------------------------------- #
# Property 1 -- Joseph form keeps the covariance symmetric and PSD
# --------------------------------------------------------------------- #
@given(
    n=st.integers(min_value=1, max_value=4),
    m=st.integers(min_value=1, max_value=3),
    data=st.data(),
)
@SETTINGS
def test_joseph_update_keeps_covariance_symmetric_psd(n, m, data):
    p = data.draw(psd_matrix(n))
    r = data.draw(psd_matrix(m, floor=1e-3))
    h = data.draw(_matrix(m, n))
    k = data.draw(_matrix(n, m))  # arbitrary, deliberately not the optimal gain
    p_post = joseph_update(p, k, h, r)
    assert np.max(np.abs(p_post - p_post.T)) == 0.0
    # Tolerance scales with the magnitude of the result: the eigenvalue of a
    # sum of PSD congruences can only be negative at round-off level.
    tol = -1e-9 * max(1.0, float(np.max(np.abs(p_post))))
    assert is_positive_semidefinite(p_post, tol=tol), min_eigenvalue(p_post)


@given(
    n=st.integers(min_value=1, max_value=3),
    steps=st.integers(min_value=1, max_value=25),
    data=st.data(),
)
@SETTINGS
def test_filter_covariance_stays_symmetric_psd_over_a_run(n, steps, data):
    f = data.draw(_matrix(n, n))
    q = data.draw(psd_matrix(n, floor=1e-3))
    r = data.draw(psd_matrix(1, floor=1e-2))
    h = data.draw(_matrix(1, n))
    assume(np.max(np.abs(f)) > 0.0)
    p0 = data.draw(psd_matrix(n))
    kf = KalmanFilter(f, h, q, r)
    x = np.zeros(n)
    p = p0
    rng = np.random.default_rng(0)
    for _ in range(steps):
        x, p = kf.predict(x, p)
        res = kf.update(x, p, rng.standard_normal(1))
        x, p = res.x, res.p
        assert np.max(np.abs(p - p.T)) == 0.0
        assert min_eigenvalue(p) >= -1e-9 * max(1.0, float(np.max(np.abs(p))))


# --------------------------------------------------------------------- #
# Property 2 -- zero measurement noise collapses the state onto z
# --------------------------------------------------------------------- #
@given(n=st.integers(min_value=1, max_value=4), data=st.data())
@SETTINGS
def test_zero_measurement_noise_collapses_state_onto_measurement(n, data):
    """Fully observable (H square, invertible), R = 0 => H x^+ == z."""
    p = data.draw(psd_matrix(n, floor=0.5))
    h = data.draw(_matrix(n, n))
    assume(abs(np.linalg.det(h)) > 1e-2)
    assume(np.linalg.cond(h) < 1e4)
    x = data.draw(st.lists(_finite, min_size=n, max_size=n).map(np.asarray))
    z = data.draw(st.lists(_finite, min_size=n, max_size=n).map(np.asarray))

    kf = KalmanFilter(np.eye(n), h, np.eye(n) * 1e-6, np.zeros((n, n)))
    res = kf.update(x, p, z)
    scale = max(1.0, float(np.max(np.abs(z))), float(np.max(np.abs(h @ x))))
    assert np.allclose(h @ res.x, z, atol=1e-6 * scale, rtol=0.0)
    # With no measurement noise on a fully observable system the posterior
    # covariance must vanish.
    assert np.max(np.abs(res.p)) < 1e-6 * max(1.0, float(np.max(np.abs(p))))


@given(n=st.integers(min_value=2, max_value=4), data=st.data())
@SETTINGS
def test_zero_noise_partial_observation_matches_measured_subspace(n, data):
    """Observing only the first component with R = 0 pins that component."""
    p = data.draw(psd_matrix(n, floor=0.5))
    h = np.zeros((1, n))
    h[0, 0] = 1.0
    x = data.draw(st.lists(_finite, min_size=n, max_size=n).map(np.asarray))
    z = data.draw(_finite)
    kf = KalmanFilter(np.eye(n), h, np.eye(n) * 1e-6, np.zeros((1, 1)))
    res = kf.update(x, p, np.array([z]))
    assert abs(res.x[0] - z) < 1e-9 * max(1.0, abs(z))
    assert abs(res.p[0, 0]) < 1e-9 * max(1.0, float(np.max(np.abs(p))))


# --------------------------------------------------------------------- #
# Property 3 -- the unscented transform is exact for affine maps
# --------------------------------------------------------------------- #
@given(
    n=st.integers(min_value=1, max_value=4),
    d=st.integers(min_value=1, max_value=3),
    alpha=st.floats(min_value=1e-2, max_value=1.0, allow_nan=False),
    beta=st.floats(min_value=0.0, max_value=3.0, allow_nan=False),
    kappa=st.floats(min_value=-0.5, max_value=3.0, allow_nan=False),
    data=st.data(),
)
@SETTINGS
def test_unscented_transform_exact_for_affine_map(n, d, alpha, beta, kappa, data):
    assume(n + kappa > 1e-3)
    p = data.draw(psd_matrix(n, floor=0.25))
    mean = data.draw(st.lists(_finite, min_size=n, max_size=n).map(np.asarray))
    a = data.draw(_matrix(d, n))
    b = data.draw(st.lists(_finite, min_size=d, max_size=d).map(np.asarray))

    sp = MerweSigmaPoints(n, alpha=alpha, beta=beta, kappa=kappa)
    pts = sp.generate(mean, p)
    transformed = pts.points @ a.T + b
    m_ut, p_ut = unscented_transform(transformed, pts.wm, pts.wc)

    expected_mean = a @ mean + b
    expected_cov = a @ p @ a.T
    # Round-off in the scaled transform is amplified by ~1/alpha^2.
    amp = 1.0 / alpha**2
    tol_m = 1e-10 * amp * max(1.0, float(np.max(np.abs(expected_mean))))
    tol_c = 1e-10 * amp * max(1.0, float(np.max(np.abs(expected_cov))))
    assert np.allclose(m_ut, expected_mean, atol=tol_m, rtol=0.0)
    assert np.allclose(p_ut, expected_cov, atol=tol_c, rtol=0.0)


@given(
    n=st.integers(min_value=1, max_value=4),
    alpha=st.floats(min_value=1e-2, max_value=1.0, allow_nan=False),
    kappa=st.floats(min_value=-0.5, max_value=3.0, allow_nan=False),
    data=st.data(),
)
@SETTINGS
def test_sigma_points_reproduce_their_own_mean_and_covariance(n, alpha, kappa, data):
    assume(n + kappa > 1e-3)
    p = data.draw(psd_matrix(n, floor=0.25))
    mean = data.draw(st.lists(_finite, min_size=n, max_size=n).map(np.asarray))
    sp = MerweSigmaPoints(n, alpha=alpha, beta=2.0, kappa=kappa)
    pts = sp.generate(mean, p)
    assert abs(pts.wm.sum() - 1.0) < 1e-10 / alpha**2
    m_ut, p_ut = unscented_transform(pts.points, pts.wm, pts.wc)
    amp = 1.0 / alpha**2
    assert np.allclose(m_ut, mean, atol=1e-10 * amp * max(1.0, float(np.max(np.abs(mean)))))
    assert np.allclose(p_ut, p, atol=1e-10 * amp * max(1.0, float(np.max(np.abs(p)))))
