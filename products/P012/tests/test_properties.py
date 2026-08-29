"""Hypothesis property-based tests for algebraic identities.

Every property here is an identity that must hold for *all* admissible inputs,
not a numeric outcome for one scenario.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from navbench import (
    KalmanFilter,
    MerweSigmaPoints,
    attitude_state_transition,
    axis_angle_from_quat,
    chi2_bounds,
    dcm_from_quat,
    euler_zyx_from_quat,
    gyro_process_noise,
    joseph_update,
    nees,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_euler_zyx,
    quat_from_small_angle,
    quat_identity,
    quat_multiply,
    quat_normalize,
    skew,
    small_angle_from_quat,
    symmetrize,
    unscented_transform,
)

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
)

finite = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
angle = st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
vec3 = st.tuples(finite, finite, finite)
quat4 = st.tuples(finite, finite, finite, finite)
#: A rotation vector built from a direction and a magnitude, so that the
#: magnitude range is sampled directly rather than by rejection.
magnitude = st.floats(
    min_value=1e-9, max_value=np.pi - 1e-6, allow_nan=False, allow_infinity=False
)


def _rotvec(direction, mag):
    d = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        d, n = np.array([1.0, 0.0, 0.0]), 1.0
    return mag * d / n


class TestQuaternionProperties:
    @SETTINGS
    @given(quat4)
    def test_normalized_has_unit_norm(self, q):
        assume(np.linalg.norm(q) > 1e-6)
        assert abs(np.linalg.norm(quat_normalize(q)) - 1.0) < 1e-14

    @SETTINGS
    @given(quat4, quat4)
    def test_multiplication_is_associative(self, a, b):
        assume(np.linalg.norm(a) > 1e-3 and np.linalg.norm(b) > 1e-3)
        qa, qb = quat_normalize(a), quat_normalize(b)
        qc = quat_normalize([1.0, 0.3, -0.2, 0.5])
        left = quat_multiply(quat_multiply(qa, qb), qc)
        right = quat_multiply(qa, quat_multiply(qb, qc))
        assert np.allclose(left, right, atol=1e-13)

    @SETTINGS
    @given(quat4, quat4)
    def test_norm_is_multiplicative(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        assume(na > 1e-3 and nb > 1e-3)
        assert np.linalg.norm(quat_multiply(a, b)) == np.float64(
            np.linalg.norm(quat_multiply(a, b))
        )
        assert abs(np.linalg.norm(quat_multiply(a, b)) - na * nb) < 1e-9 * na * nb

    @SETTINGS
    @given(quat4)
    def test_dcm_is_orthogonal(self, q):
        assume(np.linalg.norm(q) > 1e-3)
        r = dcm_from_quat(q)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-13)
        assert abs(np.linalg.det(r) - 1.0) < 1e-13

    @SETTINGS
    @given(quat4, vec3)
    def test_rotation_preserves_norm(self, q, v):
        assume(np.linalg.norm(q) > 1e-3)
        rv = dcm_from_quat(q) @ np.asarray(v, dtype=float)
        assert abs(np.linalg.norm(rv) - np.linalg.norm(v)) < 1e-12 * max(
            1.0, float(np.linalg.norm(v))
        )

    @SETTINGS
    @given(quat4)
    def test_conjugate_undoes_rotation(self, q):
        assume(np.linalg.norm(q) > 1e-3)
        qn = quat_normalize(q)
        prod = quat_multiply(qn, quat_conjugate(qn))
        assert np.allclose(prod, quat_identity(), atol=1e-13)

    @SETTINGS
    @given(vec3, magnitude)
    def test_rotation_vector_round_trip(self, direction, mag):
        a = _rotvec(direction, mag)
        back = small_angle_from_quat(quat_from_small_angle(a))
        # RELATIVE tolerance: the absolute error scales with |a| across the
        # 9 decades sampled here.
        assert np.max(np.abs(back - a)) / mag < 1e-13

    @SETTINGS
    @given(vec3, magnitude)
    def test_axis_angle_round_trip(self, direction, mag):
        a = _rotvec(direction, mag)
        u = a / mag
        ax2, an2 = axis_angle_from_quat(quat_from_axis_angle(u, mag))
        assert np.max(np.abs(ax2 * an2 - u * mag)) / mag < 1e-13

    @SETTINGS
    @given(angle, st.floats(min_value=-1.2, max_value=1.2), angle)
    def test_euler_round_trip(self, yaw, pitch, roll):
        y, p, r = euler_zyx_from_quat(quat_from_euler_zyx(yaw, pitch, roll))
        assert abs(y - yaw) < 1e-10
        assert abs(p - pitch) < 1e-10
        assert abs(r - roll) < 1e-10

    @SETTINGS
    @given(vec3, vec3)
    def test_skew_is_cross_product(self, a, b):
        av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        assert np.allclose(skew(av) @ bv, np.cross(av, bv), atol=1e-13)

    @SETTINGS
    @given(vec3)
    def test_skew_squared_identity(self, v):
        """[v x]^2 = v v^T - |v|^2 I."""
        a = np.asarray(v, dtype=float)
        assert np.allclose(
            skew(a) @ skew(a), np.outer(a, a) - float(a @ a) * np.eye(3), atol=1e-12
        )


class TestCovarianceProperties:
    @SETTINGS
    @given(
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=1, max_value=3),
        st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_joseph_keeps_symmetry_and_psd_for_any_gain(self, n, m, seed):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((n, n))
        p = a @ a.T + n * np.eye(n)
        b = rng.standard_normal((m, m))
        r = b @ b.T + m * np.eye(m)
        h = rng.standard_normal((m, n))
        k = rng.standard_normal((n, m))  # deliberately NOT the optimal gain
        out = joseph_update(p, k, h, r)
        assert np.array_equal(out, out.T)
        assert float(np.linalg.eigvalsh(out).min()) > -1e-9 * float(np.max(np.abs(out)))

    @SETTINGS
    @given(st.integers(min_value=1, max_value=5), st.integers(min_value=0, max_value=2**31 - 1))
    def test_symmetrize_is_idempotent(self, n, seed):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((n, n))
        s = symmetrize(a)
        assert np.array_equal(symmetrize(s), s)

    @SETTINGS
    @given(st.integers(min_value=0, max_value=2**31 - 1))
    def test_kf_covariance_stays_psd_over_random_runs(self, seed):
        rng = np.random.default_rng(seed)
        dt = 1.0
        f = np.array([[1.0, dt], [0.0, 1.0]])
        h = np.array([[1.0, 0.0]])
        q = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * 0.05
        r = np.array([[9.0]])
        kf = KalmanFilter(f, h, q, r, np.zeros(2), np.diag([100.0, 10.0]))
        for _ in range(25):
            kf.predict()
            kf.update(rng.standard_normal(1) * 3.0)
            assert np.array_equal(kf.p, kf.p.T)
            assert float(np.linalg.eigvalsh(kf.p).min()) > 0.0

    @SETTINGS
    @given(positive, positive, st.floats(min_value=0.01, max_value=10.0))
    def test_gyro_process_noise_is_psd(self, sv, su, dt):
        q = gyro_process_noise(sv * 1e-4, su * 1e-6, dt)
        assert np.array_equal(q, q.T)
        assert float(np.linalg.eigvalsh(q).min()) > -1e-20


class TestUnscentedProperties:
    @SETTINGS
    @given(
        st.integers(min_value=1, max_value=4),
        st.floats(min_value=0.05, max_value=1.0),
        st.floats(min_value=0.0, max_value=3.0),
        st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_transform_is_exact_for_affine_maps(self, n, alpha, beta, seed):
        rng = np.random.default_rng(seed)
        pts = MerweSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=0.0)
        wm, wc = pts.weights()
        x = rng.standard_normal(n)
        a0 = rng.standard_normal((n, n))
        p = a0 @ a0.T + n * np.eye(n)
        amat = rng.standard_normal((2, n))
        b = rng.standard_normal(2)
        s = pts.sigma_points(x, p)
        mean, cov = unscented_transform((amat @ s.T).T + b, wm, wc)
        tol = 1e-9 / alpha**2
        scale = max(1.0, float(np.max(np.abs(amat @ x + b))))
        assert np.max(np.abs(mean - (amat @ x + b))) / scale < tol
        assert np.max(np.abs(cov - amat @ p @ amat.T)) / float(
            np.max(np.abs(amat @ p @ amat.T))
        ) < tol

    @SETTINGS
    @given(
        st.integers(min_value=1, max_value=5),
        st.floats(min_value=0.1, max_value=1.0),
        st.floats(min_value=-0.5, max_value=3.0),
    )
    def test_mean_weights_sum_to_one(self, n, alpha, kappa):
        assume(n + kappa > 0.0)
        wm, _ = MerweSigmaPoints(n=n, alpha=alpha, kappa=kappa).weights()
        assert abs(float(np.sum(wm)) - 1.0) < 1e-12


class TestAttitudeTransitionProperties:
    @SETTINGS
    @given(vec3, st.floats(min_value=0.001, max_value=5.0))
    def test_attitude_block_is_a_rotation(self, w, dt):
        phi = attitude_state_transition(np.asarray(w, dtype=float), dt)
        r = phi[:3, :3]
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
        assert np.allclose(phi[3:, 3:], np.eye(3))
        assert np.allclose(phi[3:, :3], 0.0)

    @SETTINGS
    @given(vec3, st.floats(min_value=0.001, max_value=2.0))
    def test_transition_composes_over_two_half_steps(self, w, dt):
        a = attitude_state_transition(np.asarray(w, dtype=float), dt)
        h = attitude_state_transition(np.asarray(w, dtype=float), dt / 2.0)
        assert np.allclose(a, h @ h, atol=1e-11)


class TestConsistencyProperties:
    @SETTINGS
    @given(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=200),
        st.floats(min_value=0.01, max_value=0.5),
    )
    def test_bounds_bracket_the_dof(self, dof, m, alpha):
        lo, hi = chi2_bounds(dof, m, alpha)
        assert lo < dof < hi

    @SETTINGS
    @given(st.integers(min_value=1, max_value=5), st.integers(min_value=0, max_value=2**31 - 1))
    def test_nees_is_non_negative_and_invariant_under_scaling(self, n, seed):
        rng = np.random.default_rng(seed)
        a = rng.standard_normal((n, n))
        p = a @ a.T + n * np.eye(n)
        e = rng.standard_normal(n)
        v1 = nees(e[None, :], p[None, ...])[0]
        c = 7.0
        v2 = nees((c * e)[None, :], (c * c * p)[None, ...])[0]
        assert v1 >= 0.0
        assert abs(v1 - v2) < 1e-9 * max(1.0, v1)
