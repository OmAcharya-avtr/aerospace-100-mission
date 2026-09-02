"""Null-space basis, projector and the classical null-motion policies."""

import numpy as np
import pytest

from cmgsteer.arrays import pyramid_array
from cmgsteer.nullmotion import (
    GradientNullMotion,
    NoNullMotion,
    NullMotionPolicy,
    PreferredAngleNullMotion,
    null_motion_from_coefficients,
    null_projector,
    unit_null_vector,
)
from cmgsteer.singularity import manipulability_gradient, singularity_measure

TAU = np.array([0.10, -0.05, 0.20])
D = np.array([0.30, -0.50, 0.80, 0.20])


class TestUnitNullVector:
    def test_is_unit_and_in_the_null_space(self):
        array = pyramid_array()
        vec = unit_null_vector(array, D)
        assert np.linalg.norm(vec) == pytest.approx(1.0)
        assert np.max(np.abs(array.jacobian(D) @ vec)) < 1e-13

    def test_sign_is_fixed_by_the_gradient(self):
        array = pyramid_array()
        vec = unit_null_vector(array, D)
        assert float(vec @ manipulability_gradient(array, D)) >= 0.0

    def test_moving_along_it_raises_the_measure(self):
        array = pyramid_array()
        vec = unit_null_vector(array, D)
        base = singularity_measure(array.jacobian(D))
        up = singularity_measure(array.jacobian(D + 1e-3 * vec))
        down = singularity_measure(array.jacobian(D - 1e-3 * vec))
        assert up > base > down

    def test_raises_when_the_null_space_is_not_one_dimensional(self):
        array = pyramid_array()
        with pytest.raises(ValueError, match="null space has dimension 2"):
            unit_null_vector(array, np.full(4, np.pi / 2))

    def test_raises_for_three_free_gimbals(self):
        array = pyramid_array().with_locked([0])
        with pytest.raises(ValueError, match="null space has dimension 0"):
            unit_null_vector(array, D)

    def test_fallback_sign_convention_without_gradient_alignment(self):
        array = pyramid_array()
        vec = unit_null_vector(array, D, align_with_gradient=False)
        nonzero = np.flatnonzero(np.abs(vec) > 1e-12)
        assert vec[nonzero[0]] > 0.0


class TestProjector:
    def test_is_idempotent_and_symmetric(self):
        proj = null_projector(pyramid_array(), D)
        assert np.allclose(proj @ proj, proj, atol=1e-12)
        assert np.allclose(proj, proj.T, atol=1e-14)

    def test_projected_vectors_are_in_the_null_space(self):
        array = pyramid_array()
        proj = null_projector(array, D)
        rng = np.random.default_rng(12)
        for _ in range(10):
            v = rng.normal(size=4)
            assert np.max(np.abs(array.jacobian(D) @ (proj @ v))) < 1e-13

    def test_is_zero_when_there_is_no_null_space(self):
        array = pyramid_array().with_locked([2])
        assert np.max(np.abs(null_projector(array, D))) == 0.0

    def test_rank_equals_the_null_space_dimension(self):
        proj = null_projector(pyramid_array(), D)
        assert np.linalg.matrix_rank(proj, tol=1e-9) == 1


class TestCoefficients:
    def test_scales_the_unit_null_vector(self):
        array = pyramid_array()
        rates = null_motion_from_coefficients(array, D, [0.5], scale=0.4)
        assert np.allclose(rates, 0.2 * unit_null_vector(array, D))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="coefficients must have length 1"):
            null_motion_from_coefficients(pyramid_array(), D, [0.5, 0.5])


class TestPolicies:
    def test_base_class_is_abstract(self):
        with pytest.raises(NotImplementedError):
            NullMotionPolicy().rates(pyramid_array(), D, TAU)

    def test_no_null_motion_returns_zeros(self):
        policy = NoNullMotion()
        assert np.max(np.abs(policy.rates(pyramid_array(), D, TAU))) == 0.0
        assert policy.name == "none"
        policy.reset()

    def test_gradient_policy_lies_in_the_null_space(self):
        array = pyramid_array()
        rates = GradientNullMotion(gain=0.5).rates(array, D, TAU)
        assert np.max(np.abs(array.jacobian(D) @ rates)) < 1e-13

    def test_gradient_policy_increases_the_measure(self):
        array = pyramid_array()
        rates = GradientNullMotion(gain=1.0).rates(array, D, TAU)
        base = singularity_measure(array.jacobian(D))
        stepped = singularity_measure(array.jacobian(D + 1e-3 * rates))
        assert stepped > base

    def test_gradient_policy_respects_its_rate_cap(self):
        array = pyramid_array()
        rates = GradientNullMotion(gain=1e3, max_rate=0.25).rates(array, D, TAU)
        assert np.max(np.abs(rates)) == pytest.approx(0.25, rel=1e-12)
        assert np.max(np.abs(array.jacobian(D) @ rates)) < 1e-12

    def test_gradient_policy_is_zero_without_a_null_space(self):
        array = pyramid_array().with_locked([1])
        assert np.max(np.abs(GradientNullMotion().rates(array, D, TAU))) == 0.0

    def test_preferred_policy_moves_toward_the_preferred_set(self):
        array = pyramid_array()
        pref = np.zeros(4)
        rates = PreferredAngleNullMotion(preferred=pref, gain=1.0).rates(array, D, TAU)
        assert np.max(np.abs(array.jacobian(D) @ rates)) < 1e-13
        # the step reduces the wrapped distance to the preferred angles
        before = np.linalg.norm(np.arctan2(np.sin(D - pref), np.cos(D - pref)))
        after_d = D + 1e-3 * rates
        after = np.linalg.norm(np.arctan2(np.sin(after_d - pref), np.cos(after_d - pref)))
        assert after < before

    def test_preferred_policy_is_zero_at_the_preferred_set(self):
        array = pyramid_array()
        rates = PreferredAngleNullMotion(preferred=D, gain=1.0).rates(array, D, TAU)
        assert np.max(np.abs(rates)) < 1e-13

    def test_preferred_policy_handles_angle_wrapping(self):
        array = pyramid_array()
        pref = D + 2.0 * np.pi
        rates = PreferredAngleNullMotion(preferred=pref, gain=1.0).rates(array, D, TAU)
        assert np.max(np.abs(rates)) < 1e-12

    def test_preferred_policy_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="preferred must have length 4"):
            PreferredAngleNullMotion(preferred=np.zeros(2)).rates(pyramid_array(), D, TAU)

    def test_bad_cap_raises(self):
        with pytest.raises(ValueError, match="max_rate must be positive"):
            GradientNullMotion(gain=1.0, max_rate=-1.0).rates(pyramid_array(), D, TAU)
