"""Property-based tests for the algebraic identities the package rests on."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cmgsteer.arrays import CMGArray, pyramid_array, roof_array
from cmgsteer.singularity import (
    null_space_basis,
    singular_configuration,
    singularity_measure,
)
from cmgsteer.steering import (
    apply_rate_limit,
    pseudo_inverse_steer,
    sr_inverse_steer,
    sr_torque_error_closed_form,
)

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

angles = st.lists(
    st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
    min_size=4,
    max_size=4,
)
small_vectors = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=3,
    max_size=3,
)
signs = st.lists(st.sampled_from([-1.0, 1.0]), min_size=4, max_size=4)


def _rotation(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """A rotation matrix from three angles (ZYX), used only to move a whole array."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, -sg], [0.0, sg, cg]])
    return rz @ ry @ rx


class TestGeometryProperties:
    @SETTINGS
    @given(angles)
    def test_jacobian_columns_are_perpendicular_to_gimbal_axes(self, d):
        array = pyramid_array()
        a = array.jacobian(np.array(d))
        assert np.max(np.abs(np.sum(a.T * array.gimbal_axes, axis=1))) < 1e-14

    @SETTINGS
    @given(angles)
    def test_momentum_is_inside_the_capacity_ball(self, d):
        array = pyramid_array()
        assert np.linalg.norm(array.momentum(np.array(d))) <= 4.0 + 1e-12

    @SETTINGS
    @given(angles)
    def test_momentum_is_two_pi_periodic(self, d):
        array = pyramid_array()
        d = np.array(d)
        assert np.allclose(array.momentum(d), array.momentum(d + 2.0 * np.pi), atol=1e-9)

    @SETTINGS
    @given(angles)
    def test_measure_is_non_negative(self, d):
        array = pyramid_array()
        assert singularity_measure(array.jacobian(np.array(d))) >= 0.0

    @SETTINGS
    @given(
        angles,
        st.floats(min_value=-np.pi, max_value=np.pi),
        st.floats(min_value=-np.pi, max_value=np.pi),
        st.floats(min_value=-np.pi, max_value=np.pi),
    )
    def test_rotating_the_whole_array_rotates_the_momentum_and_preserves_the_measure(
        self, d, alpha, beta, gamma
    ):
        base = pyramid_array()
        rot = _rotation(alpha, beta, gamma)
        moved = CMGArray(
            base.gimbal_axes @ rot.T, base.ref_axes @ rot.T, base.rotor_momenta
        )
        d = np.array(d)
        assert np.allclose(moved.momentum(d), rot @ base.momentum(d), atol=1e-12)
        assert np.allclose(moved.jacobian(d), rot @ base.jacobian(d), atol=1e-12)
        moved_measure = singularity_measure(moved.jacobian(d))
        base_measure = singularity_measure(base.jacobian(d))
        assert abs(moved_measure - base_measure) < 1e-12

    @SETTINGS
    @given(small_vectors, signs)
    def test_analytic_singular_configurations_have_zero_measure(self, u, s):
        array = pyramid_array()
        u = np.array(u)
        if np.linalg.norm(u) < 1e-6:
            return
        try:
            d = singular_configuration(array, u, np.array(s))
        except ValueError:
            return
        assert singularity_measure(array.jacobian(d)) < 1e-12


class TestSteeringProperties:
    @SETTINGS
    @given(angles, small_vectors)
    def test_pseudo_inverse_is_exact_away_from_singularity(self, d, tau):
        array = pyramid_array()
        d = np.array(d)
        if singularity_measure(array.jacobian(d)) < 1e-3:
            return
        result = pseudo_inverse_steer(array, d, np.array(tau))
        assert result.torque_error_norm < 1e-9 * max(1.0, float(np.linalg.norm(tau)))

    @SETTINGS
    @given(
        angles,
        small_vectors,
        st.floats(min_value=1e-10, max_value=1.0, allow_nan=False),
    )
    def test_sr_error_matches_the_closed_form(self, d, tau, lam):
        array = pyramid_array()
        d = np.array(d)
        tau = np.array(tau)
        got = sr_inverse_steer(array, d, tau, lam=lam).torque_error
        want = sr_torque_error_closed_form(array.jacobian(d), tau, lam)
        assert np.max(np.abs(got - want)) < 1e-12

    @SETTINGS
    @given(
        angles,
        small_vectors,
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    def test_sr_error_never_exceeds_the_command(self, d, tau, lam):
        # each SVD weight lam / (sigma^2 + lam) is in [0, 1], so the error is a
        # contraction of the command
        array = pyramid_array()
        tau = np.array(tau)
        result = sr_inverse_steer(array, np.array(d), tau, lam=lam)
        assert result.torque_error_norm <= np.linalg.norm(tau) + 1e-12

    @SETTINGS
    @given(angles, small_vectors)
    def test_null_motion_never_changes_the_delivered_torque(self, d, tau):
        array = pyramid_array()
        d = np.array(d)
        basis = null_space_basis(array.jacobian(d))
        if basis.shape[1] == 0:
            return
        null = 0.3 * basis[:, 0]
        plain = pseudo_inverse_steer(array, d, np.array(tau))
        moved = pseudo_inverse_steer(array, d, np.array(tau), null_rates=null)
        assert np.max(np.abs(plain.achieved_torque - moved.achieved_torque)) < 1e-10

    @SETTINGS
    @given(
        st.lists(
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False),
            min_size=1,
            max_size=6,
        ),
        st.floats(min_value=1e-3, max_value=10.0, allow_nan=False),
        st.sampled_from(["clip", "scale"]),
    )
    def test_rate_limit_is_respected(self, rates, cap, mode):
        limited, _ = apply_rate_limit(rates, cap, mode=mode)
        assert np.max(np.abs(limited)) <= cap + 1e-12

    @SETTINGS
    @given(angles, small_vectors)
    def test_roof_array_torque_is_minus_jacobian_times_rates(self, d, tau):
        array = roof_array()
        d = np.array(d)
        result = sr_inverse_steer(array, d, np.array(tau), lam=1e-3)
        assert np.allclose(
            result.achieved_torque, -(array.jacobian(d) @ result.gimbal_rates), atol=1e-12
        )
