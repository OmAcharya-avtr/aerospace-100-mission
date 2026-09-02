"""Singularity measure, analytic singular configurations, classification, surfaces."""

import numpy as np
import pytest

from cmgsteer.arrays import pyramid_array, roof_array
from cmgsteer.singularity import (
    classify_singularity,
    condition_number,
    fibonacci_directions,
    manipulability_gradient,
    min_singular_value,
    momentum_envelope,
    null_space_basis,
    singular_configuration,
    singular_direction,
    singular_surface,
    singularity_measure,
)

SIGN_SETS = [
    np.array([1.0, 1.0, 1.0, 1.0]),
    np.array([1.0, 1.0, 1.0, -1.0]),
    np.array([1.0, 1.0, -1.0, -1.0]),
    np.array([1.0, -1.0, 1.0, -1.0]),
    np.array([1.0, -1.0, -1.0, -1.0]),
]


class TestMeasureKnownAnswers:
    def test_measure_at_zero_is_hand_computable(self):
        # A A^T = diag(0.72, 0.72, 2.56) at delta = 0, so
        # m = sqrt(0.72 * 0.72 * 2.56) = 0.72 * 1.6 = 1.152 exactly.
        array = pyramid_array()
        assert singularity_measure(array.jacobian(np.zeros(4))) == pytest.approx(1.152, abs=1e-13)

    def test_measure_equals_sqrt_det_gram(self):
        array = pyramid_array()
        rng = np.random.default_rng(4)
        for _ in range(30):
            d = rng.uniform(-np.pi, np.pi, 4)
            a = array.jacobian(d)
            direct = float(np.sqrt(max(0.0, np.linalg.det(a @ a.T))))
            assert singularity_measure(a) == pytest.approx(direct, rel=1e-9, abs=1e-14)

    def test_measure_scales_as_the_cube_of_rotor_momentum(self):
        d = np.array([0.3, -0.4, 0.9, 0.1])
        m1 = singularity_measure(pyramid_array(rotor_momentum=1.0).jacobian(d))
        m2 = singularity_measure(pyramid_array(rotor_momentum=2.0).jacobian(d))
        assert m2 == pytest.approx(8.0 * m1, rel=1e-12)

    def test_min_singular_value_at_zero(self):
        # singular values are sqrt(0.72), sqrt(0.72), 1.6
        array = pyramid_array()
        assert min_singular_value(array.jacobian(np.zeros(4))) == pytest.approx(
            np.sqrt(0.72), rel=1e-12
        )

    def test_condition_number_at_zero(self):
        array = pyramid_array()
        assert condition_number(array.jacobian(np.zeros(4))) == pytest.approx(
            1.6 / np.sqrt(0.72), rel=1e-12
        )


class TestAnalyticSingularConfigurations:
    def test_all_ninety_degrees_is_singular(self):
        array = pyramid_array()
        assert singularity_measure(array.jacobian(np.full(4, np.pi / 2))) < 1e-14

    def test_construction_reproduces_the_ninety_degree_state(self):
        array = pyramid_array()
        d = singular_configuration(array, np.array([0.0, 0.0, 1.0]))
        assert np.allclose(d, np.pi / 2, atol=1e-12)

    def test_measure_vanishes_on_the_analytic_set(self):
        array = pyramid_array()
        rng = np.random.default_rng(5)
        worst = 0.0
        for _ in range(120):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            for signs in SIGN_SETS:
                d = singular_configuration(array, u, signs)
                worst = max(worst, singularity_measure(array.jacobian(d)))
        assert worst < 1e-13

    def test_singular_direction_annihilates_the_jacobian(self):
        array = pyramid_array()
        rng = np.random.default_rng(6)
        for _ in range(20):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            d = singular_configuration(array, u, SIGN_SETS[2])
            a = array.jacobian(d)
            assert np.max(np.abs(u @ a)) < 1e-13
            recovered = singular_direction(a)
            assert abs(abs(float(recovered @ u)) - 1.0) < 1e-9

    def test_direction_along_a_gimbal_axis_raises(self):
        array = pyramid_array()
        with pytest.raises(ValueError, match="parallel to gimbal axis"):
            singular_configuration(array, array.gimbal_axes[0])

    def test_bad_signs_raise(self):
        array = pyramid_array()
        with pytest.raises(ValueError, match=r"only \+1 and -1"):
            singular_configuration(array, [0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="signs must have length 4"):
            singular_configuration(array, [0.0, 0.0, 1.0], [1.0, 1.0])

    def test_zero_direction_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            singular_configuration(pyramid_array(), [0.0, 0.0, 0.0])

    def test_bad_direction_shape_raises(self):
        with pytest.raises(ValueError, match=r"shape \(3,\)"):
            singular_configuration(pyramid_array(), [0.0, 0.0])


class TestClassification:
    def test_z_saturation_is_external(self):
        array = pyramid_array()
        info = classify_singularity(array, np.full(4, np.pi / 2))
        assert info.singular
        assert info.kind == "external"
        assert np.allclose(info.signs, 1.0)
        assert np.allclose(np.abs(info.direction), [0.0, 0.0, 1.0], atol=1e-12)
        assert info.momentum[2] == pytest.approx(3.2, abs=1e-12)
        assert info.rank == 2

    def test_external_singularity_is_never_escapable(self):
        # All eps_i equal makes the second-order form definite, so no null
        # motion can raise h . u; saturation singularities are impassable.
        array = pyramid_array()
        rng = np.random.default_rng(7)
        for _ in range(15):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            d = singular_configuration(array, u)
            info = classify_singularity(array, d)
            assert info.kind == "external"
            assert info.passability == "elliptic"

    def test_mixed_signs_give_an_internal_singularity(self):
        array = pyramid_array()
        d = np.array([1.0, 1.0, 1.0, -1.0]) * np.pi / 2
        info = classify_singularity(array, d)
        assert info.singular
        assert info.kind == "internal"
        # Hand-computed: h = s1 + s2 + s3 - s4 = (0, -1.2, 1.6), |h| = 2.
        assert np.allclose(info.momentum, [0.0, -1.2, 1.6], atol=1e-12)
        assert np.linalg.norm(info.momentum) == pytest.approx(2.0, abs=1e-12)

    def test_internal_singularities_of_both_passabilities_exist(self):
        array = pyramid_array()
        rng = np.random.default_rng(8)
        seen = set()
        for _ in range(60):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            for signs in SIGN_SETS[1:]:
                info = classify_singularity(array, singular_configuration(array, u, signs))
                if info.kind == "internal":
                    seen.add(info.passability)
        assert "hyperbolic" in seen
        assert "elliptic" in seen

    def test_regular_configuration_is_not_singular(self):
        info = classify_singularity(pyramid_array(), np.zeros(4))
        assert not info.singular
        assert info.kind == "none"
        assert info.passability == "none"
        assert info.rank == 3

    def test_bad_tolerance_raises(self):
        with pytest.raises(ValueError, match=r"tol must lie in \(0, 1\)"):
            classify_singularity(pyramid_array(), np.zeros(4), tol=2.0)


class TestGradient:
    def test_matches_central_differences_away_from_singularity(self):
        array = pyramid_array()
        rng = np.random.default_rng(9)
        step = 1e-6
        worst = 0.0
        for _ in range(30):
            d = rng.uniform(-np.pi, np.pi, 4)
            analytic = manipulability_gradient(array, d)
            numeric = np.empty(4)
            for j in range(4):
                plus, minus = d.copy(), d.copy()
                plus[j] += step
                minus[j] -= step
                numeric[j] = (
                    singularity_measure(array.jacobian(plus))
                    - singularity_measure(array.jacobian(minus))
                ) / (2 * step)
            worst = max(worst, float(np.max(np.abs(analytic - numeric))))
        assert worst < 1e-8

    def test_finite_at_an_exact_singularity(self):
        array = pyramid_array()
        grad = manipulability_gradient(array, np.full(4, np.pi / 2))
        assert np.all(np.isfinite(grad))

    def test_gradient_is_zero_where_the_measure_is_stationary(self):
        # delta = 0 maximises m for the pyramid over the all-equal-angle family,
        # and by symmetry the full gradient vanishes there.
        array = pyramid_array()
        assert np.max(np.abs(manipulability_gradient(array, np.zeros(4)))) < 1e-13

    def test_locked_gimbal_shrinks_the_gradient(self):
        array = pyramid_array().with_locked([3])
        grad = manipulability_gradient(array, np.array([0.2, -0.3, 0.5, 0.1]))
        assert grad.shape == (3,)


class TestNullSpace:
    def test_null_space_is_one_dimensional_for_four_cmgs(self):
        array = pyramid_array()
        basis = null_space_basis(array.jacobian(np.array([0.2, -0.4, 0.6, 0.1])))
        assert basis.shape == (4, 1)

    def test_null_vectors_produce_no_momentum_rate(self):
        array = pyramid_array()
        d = np.array([0.2, -0.4, 0.6, 0.1])
        a = array.jacobian(d)
        basis = null_space_basis(a)
        assert np.max(np.abs(a @ basis)) < 1e-13

    def test_null_space_grows_at_a_singularity(self):
        array = pyramid_array()
        basis = null_space_basis(array.jacobian(np.full(4, np.pi / 2)))
        assert basis.shape[1] == 2

    def test_three_free_gimbals_have_no_null_space(self):
        array = pyramid_array().with_locked([0])
        basis = null_space_basis(array.jacobian(np.zeros(4)))
        assert basis.shape[1] == 0

    def test_under_actuated_jacobian_raises(self):
        with pytest.raises(ValueError, match="at least 3 free gimbals"):
            singularity_measure(np.zeros((3, 2)))

    def test_bad_jacobian_shape_raises(self):
        with pytest.raises(ValueError, match=r"shape \(3, n\)"):
            singularity_measure(np.zeros((4, 4)))


class TestSurfaces:
    def test_fibonacci_directions_are_unit(self):
        dirs = fibonacci_directions(200)
        assert dirs.shape == (200, 3)
        assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)

    def test_fibonacci_requires_a_positive_count(self):
        with pytest.raises(ValueError, match="n_points must be >= 1"):
            fibonacci_directions(0)

    def test_envelope_points_are_singular(self):
        array = pyramid_array()
        momenta, angles = momentum_envelope(array, n_points=200)
        measures = [singularity_measure(array.jacobian(d)) for d in angles]
        assert max(measures) < 1e-13
        assert momenta.shape[0] == angles.shape[0]

    def test_envelope_radius_brackets_the_known_z_saturation(self):
        array = pyramid_array()
        momenta, _ = momentum_envelope(array, n_points=800)
        radii = np.linalg.norm(momenta, axis=1)
        assert radii.max() <= array.total_momentum_capacity + 1e-12
        assert radii.max() >= 3.2 - 1e-2

    def test_internal_surface_lies_inside_the_envelope(self):
        array = pyramid_array()
        outer, _ = momentum_envelope(array, n_points=400)
        inner, _ = singular_surface(array, signs=SIGN_SETS[2], n_points=400)
        assert np.max(np.linalg.norm(inner, axis=1)) < np.max(np.linalg.norm(outer, axis=1))

    def test_surface_accepts_explicit_directions(self):
        array = pyramid_array()
        momenta, angles = singular_surface(array, directions=np.array([[0.0, 0.0, 1.0]]))
        assert momenta.shape == (1, 3)
        assert np.allclose(angles[0], np.pi / 2, atol=1e-12)

    def test_roof_array_envelope_is_computable(self):
        array = roof_array()
        momenta, angles = momentum_envelope(array, n_points=150)
        measures = [singularity_measure(array.jacobian(d)) for d in angles]
        assert max(measures) < 1e-13
        assert momenta.shape[0] > 100
