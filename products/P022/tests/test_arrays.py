"""Array geometry, momentum map and Jacobian."""

import numpy as np
import pytest

from cmgsteer.arrays import (
    STANDARD_PYRAMID_SKEW_DEG,
    CMGArray,
    general_array,
    pyramid_array,
    roof_array,
)


class TestPyramidGeometry:
    def test_standard_skew_angle_is_arctan_four_thirds(self):
        # arctan(4/3) is the 3-4-5 angle: sin = 0.8, cos = 0.6 exactly.
        assert STANDARD_PYRAMID_SKEW_DEG == pytest.approx(53.13010235415598, abs=1e-12)
        beta = np.radians(STANDARD_PYRAMID_SKEW_DEG)
        assert np.sin(beta) == pytest.approx(0.8, abs=1e-15)
        assert np.cos(beta) == pytest.approx(0.6, abs=1e-15)

    def test_gimbal_axes_are_unit_and_on_the_cone(self):
        array = pyramid_array()
        assert np.allclose(np.linalg.norm(array.gimbal_axes, axis=1), 1.0)
        # every gimbal axis makes the same angle with +z, cos = 0.6
        assert np.allclose(array.gimbal_axes[:, 2], 0.6, atol=1e-15)

    def test_reference_axes_perpendicular_to_gimbal_axes(self):
        array = pyramid_array()
        dots = np.sum(array.gimbal_axes * array.ref_axes, axis=1)
        assert np.max(np.abs(dots)) < 1e-15

    def test_transverse_axes_complete_a_right_handed_triad(self):
        array = pyramid_array()
        s = array.transverse_axes
        assert np.allclose(np.linalg.norm(s, axis=1), 1.0)
        # g x c = s and c x s = g
        assert np.allclose(np.cross(array.ref_axes, s), array.gimbal_axes, atol=1e-15)

    def test_capacity_is_the_sum_of_rotor_momenta(self):
        array = pyramid_array(rotor_momentum=2.5)
        assert array.total_momentum_capacity == pytest.approx(10.0)

    def test_names_and_counts(self):
        array = pyramid_array()
        assert array.n_cmgs == 4
        assert array.n_free == 4
        assert array.names == ("cmg1", "cmg2", "cmg3", "cmg4")

    def test_six_cmg_pyramid(self):
        array = pyramid_array(n_cmgs=6)
        assert array.n_cmgs == 6
        assert array.jacobian(np.zeros(6)).shape == (3, 6)


class TestMomentumMap:
    def test_matches_the_published_closed_form(self):
        # h for the four-CMG pyramid, as written in the SGCMG literature
        # (Wie 2008; Kurokawa 2007).  Checked here against the package's
        # general construction, which never uses this form.
        array = pyramid_array()
        beta = np.radians(STANDARD_PYRAMID_SKEW_DEG)
        cb, sb = np.cos(beta), np.sin(beta)
        rng = np.random.default_rng(0)
        for _ in range(25):
            d = rng.uniform(-np.pi, np.pi, 4)
            expected = np.array(
                [
                    -cb * np.sin(d[0]) - np.cos(d[1]) + cb * np.sin(d[2]) + np.cos(d[3]),
                    np.cos(d[0]) - cb * np.sin(d[1]) - np.cos(d[2]) + cb * np.sin(d[3]),
                    sb * np.sum(np.sin(d)),
                ]
            )
            assert np.allclose(array.momentum(d), expected, atol=1e-14)

    def test_momentum_at_zero_is_zero(self):
        array = pyramid_array()
        assert np.max(np.abs(array.momentum(np.zeros(4)))) < 1e-15

    def test_all_gimbals_at_ninety_degrees_gives_the_z_saturation_state(self):
        # Hand-computed: every rotor momentum is then the transverse axis
        # s_i = (-cos(b) cos(t_i), -cos(b) sin(t_i), sin(b)); the x and y parts
        # cancel over four equally spaced azimuths and h_z = 4 h0 sin(b)
        #                                                 = 4 * 0.8 = 3.2 N*m*s.
        array = pyramid_array()
        h = array.momentum(np.full(4, np.pi / 2))
        assert h[0] == pytest.approx(0.0, abs=1e-15)
        assert h[1] == pytest.approx(0.0, abs=1e-15)
        assert h[2] == pytest.approx(3.2, abs=1e-14)

    def test_momentum_never_exceeds_capacity(self):
        array = pyramid_array()
        rng = np.random.default_rng(1)
        d = rng.uniform(-10.0, 10.0, (200, 4))
        norms = np.linalg.norm([array.momentum(row) for row in d], axis=1)
        assert np.max(norms) <= array.total_momentum_capacity + 1e-12

    def test_rotor_directions_are_unit(self):
        array = pyramid_array()
        h = array.rotor_directions(np.array([0.3, -1.2, 2.0, 0.0]))
        assert np.allclose(np.linalg.norm(h, axis=1), 1.0)


class TestJacobian:
    def test_hand_computed_at_zero(self):
        # At delta = 0 every column is the transverse axis s_i:
        #   s_1 = (-0.6, 0, 0.8), s_2 = (0, -0.6, 0.8),
        #   s_3 = (+0.6, 0, 0.8), s_4 = (0, +0.6, 0.8).
        array = pyramid_array()
        expected = np.array(
            [[-0.6, 0.0, 0.6, 0.0], [0.0, -0.6, 0.0, 0.6], [0.8, 0.8, 0.8, 0.8]]
        )
        assert np.allclose(array.jacobian(np.zeros(4)), expected, atol=1e-15)

    def test_gram_matrix_at_zero_is_diagonal(self):
        # A A^T = diag(2 cos^2 b, 2 cos^2 b, 4 sin^2 b) = diag(0.72, 0.72, 2.56)
        array = pyramid_array()
        a = array.jacobian(np.zeros(4))
        assert np.allclose(a @ a.T, np.diag([0.72, 0.72, 2.56]), atol=1e-15)

    def test_columns_are_perpendicular_to_their_gimbal_axes(self):
        array = pyramid_array()
        d = np.array([0.2, -0.9, 1.7, 3.0])
        a = array.jacobian(d)
        dots = np.sum(a.T * array.gimbal_axes, axis=1)
        assert np.max(np.abs(dots)) < 1e-15

    def test_column_norms_equal_the_rotor_momenta(self):
        array = pyramid_array(rotor_momentum=3.0)
        a = array.jacobian(np.array([0.5, 1.0, -2.0, 0.1]))
        assert np.allclose(np.linalg.norm(a, axis=0), 3.0)

    def test_matches_central_differences(self):
        array = pyramid_array()
        rng = np.random.default_rng(2)
        step = 1e-6
        worst = 0.0
        for _ in range(20):
            d = rng.uniform(-np.pi, np.pi, 4)
            analytic = array.jacobian(d)
            numeric = np.empty_like(analytic)
            for j in range(4):
                plus, minus = d.copy(), d.copy()
                plus[j] += step
                minus[j] -= step
                numeric[:, j] = (array.momentum(plus) - array.momentum(minus)) / (2 * step)
            worst = max(worst, float(np.max(np.abs(analytic - numeric))))
        assert worst < 1e-9


class TestTorqueConvention:
    def test_torque_is_minus_jacobian_times_rates(self):
        array = pyramid_array()
        d = np.array([0.1, 0.2, -0.3, 0.4])
        rates = np.array([0.5, -0.2, 0.1, 0.3])
        assert np.allclose(array.torque(d, rates), -(array.jacobian(d) @ rates))

    def test_torque_accepts_full_length_rates_and_ignores_locked(self):
        array = pyramid_array().with_locked([1])
        d = np.zeros(4)
        full = np.array([1.0, 99.0, 2.0, 3.0])
        free = np.array([1.0, 2.0, 3.0])
        assert np.allclose(array.torque(d, full), array.torque(d, free))

    def test_wrong_rate_length_raises(self):
        array = pyramid_array()
        with pytest.raises(ValueError, match="gimbal_rates must have length"):
            array.torque(np.zeros(4), np.zeros(2))


class TestLocking:
    def test_locked_gimbal_removes_a_column(self):
        array = pyramid_array().with_locked([0])
        assert array.n_free == 3
        assert array.jacobian(np.zeros(4)).shape == (3, 3)
        assert array.jacobian(np.zeros(4), free_only=False).shape == (3, 4)

    def test_locked_gimbal_still_carries_momentum(self):
        array = pyramid_array()
        d = np.array([1.0, 0.0, 0.0, 0.0])
        assert np.allclose(array.momentum(d), array.with_locked([0]).momentum(d))

    def test_locking_everything_raises(self):
        with pytest.raises(ValueError, match="at least one must remain free"):
            pyramid_array().with_locked([0, 1, 2, 3])

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            pyramid_array().with_locked([9])

    def test_expand_rates_scatters_into_the_free_slots(self):
        array = pyramid_array().with_locked([2])
        out = array.expand_rates([1.0, 2.0, 3.0])
        assert np.allclose(out, [1.0, 2.0, 0.0, 3.0])


class TestRoofArray:
    def test_pairs_share_a_gimbal_axis(self):
        array = roof_array()
        assert np.allclose(array.gimbal_axes[0], array.gimbal_axes[1])
        assert np.allclose(array.gimbal_axes[2], array.gimbal_axes[3])
        assert not np.allclose(array.gimbal_axes[0], array.gimbal_axes[2])

    def test_zero_configuration_is_rank_deficient(self):
        # Both members of a pair have the same gimbal axis and the same
        # reference axis, so at delta = 0 their Jacobian columns coincide and
        # only two independent directions remain.  This is a property of the
        # roof geometry, not a defect.
        array = roof_array()
        assert np.linalg.matrix_rank(array.jacobian(np.zeros(4)), tol=1e-12) == 2

    def test_offset_configuration_has_full_rank(self):
        array = roof_array()
        d = np.array([0.4, -0.4, 0.4, -0.4])
        assert np.linalg.matrix_rank(array.jacobian(d), tol=1e-12) == 3

    def test_three_pairs(self):
        array = roof_array(n_pairs=3)
        assert array.n_cmgs == 6
        assert array.names[0] == "pair1a"


class TestGeneralArrayAndValidation:
    def test_general_array_picks_perpendicular_reference_axes(self):
        axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        array = general_array(axes)
        assert np.max(np.abs(np.sum(array.gimbal_axes * array.ref_axes, axis=1))) < 1e-15

    def test_general_array_normalises_input(self):
        array = general_array(np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]]))
        assert np.allclose(np.linalg.norm(array.gimbal_axes, axis=1), 1.0)

    def test_reference_axis_parallel_to_gimbal_axis_raises(self):
        with pytest.raises(ValueError, match="parallel to its gimbal axis"):
            CMGArray(
                np.eye(3),
                np.eye(3),
                np.ones(3),
            )

    def test_reference_axis_is_orthogonalised_not_rejected(self):
        # A reference axis that merely leans on the gimbal axis is projected.
        g = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        c = np.array([[1.0, 0.0, 0.5], [0.0, 0.3, 1.0], [0.2, 1.0, 0.0]])
        array = CMGArray(g, c, np.ones(3))
        assert np.max(np.abs(np.sum(array.gimbal_axes * array.ref_axes, axis=1))) < 1e-15

    def test_zero_length_axis_raises(self):
        with pytest.raises(ValueError, match="near-zero length"):
            general_array(np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))

    def test_negative_rotor_momentum_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            CMGArray(np.eye(3), np.roll(np.eye(3), 1, axis=1), np.array([1.0, -1.0, 1.0]))

    def test_too_few_cmgs_raises(self):
        with pytest.raises(ValueError, match="at least 2 CMGs"):
            general_array(np.array([[0.0, 0.0, 1.0]]))

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
            general_array(np.zeros((4, 2)))

    def test_wrong_delta_length_raises(self):
        with pytest.raises(ValueError, match="deltas must have length 4"):
            pyramid_array().momentum(np.zeros(3))

    def test_non_finite_deltas_raise(self):
        with pytest.raises(ValueError, match="finite"):
            pyramid_array().momentum(np.array([0.0, np.nan, 0.0, 0.0]))

    def test_bad_skew_angle_raises(self):
        with pytest.raises(ValueError, match=r"\(0, 90\)"):
            pyramid_array(skew_angle_deg=95.0)
        with pytest.raises(ValueError, match=r"\(0, 90\)"):
            roof_array(skew_angle_deg=0.0)

    def test_bad_rotor_momentum_raises(self):
        with pytest.raises(ValueError, match="positive"):
            pyramid_array(rotor_momentum=0.0)
        with pytest.raises(ValueError, match="positive"):
            roof_array(rotor_momentum=-1.0)

    def test_too_few_pyramid_cmgs_raises(self):
        with pytest.raises(ValueError, match="at least 3 CMGs"):
            pyramid_array(n_cmgs=2)

    def test_too_few_roof_pairs_raises(self):
        with pytest.raises(ValueError, match="at least 2 pairs"):
            roof_array(n_pairs=1)

    def test_names_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="names must have length"):
            CMGArray(np.eye(3), np.roll(np.eye(3), 1, axis=1), np.ones(3), names=("a", "b"))

    def test_summary_lists_every_cmg(self):
        text = pyramid_array().with_locked([1]).summary()
        assert "LOCKED" in text
        assert text.count("h0=") == 4

    def test_arrays_are_read_only(self):
        array = pyramid_array()
        with pytest.raises(ValueError):
            array.gimbal_axes[0, 0] = 5.0
