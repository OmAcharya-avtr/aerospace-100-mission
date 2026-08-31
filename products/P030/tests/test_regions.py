"""Unit and known-answer tests for keepout.regions."""

import numpy as np
import pytest

from keepout import (
    ExclusionCone,
    KeepOutSet,
    allowed_directions,
    allowed_fraction,
    allowed_mask,
    allowed_solid_angle,
    allowed_solid_angle_monte_carlo,
    cap_solid_angle,
    cap_union_solid_angle,
    rotation_matrix,
)
from keepout.geometry import spherical_to_unit

DEG = np.pi / 180.0
FULL = 4.0 * np.pi


def _pair(r1_deg=30.0, r2_deg=25.0, sep_deg=40.0):
    a = ExclusionCone(spherical_to_unit(0.0, 0.0), r1_deg * DEG, "A")
    b = ExclusionCone(spherical_to_unit(sep_deg * DEG, 0.0), r2_deg * DEG, "B")
    return KeepOutSet((a, b))


class TestAllowedMask:
    def test_single_cone(self):
        ks = KeepOutSet((ExclusionCone([0, 0, 1], 30 * DEG, "polar"),))
        dirs = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0]])
        assert list(allowed_mask(ks, dirs)) == [False, True, True]

    def test_scalar_direction_returns_1d(self):
        ks = KeepOutSet((ExclusionCone([0, 0, 1], 30 * DEG, "polar"),))
        assert allowed_mask(ks, [0, 0, 1]).shape == (1,)


class TestAllowedDirections:
    def test_all_returned_points_are_allowed(self):
        ks = _pair()
        pts = allowed_directions(ks, 2000)
        assert pts.shape[1] == 3
        assert np.all(allowed_mask(ks, pts))

    def test_empty_when_everything_is_blocked(self):
        ks = KeepOutSet((ExclusionCone([0, 0, 1], np.pi, "all"),))
        assert allowed_directions(ks, 500).shape[0] in (0, 1)


class TestAllowedSolidAngle:
    def test_empty_set_is_the_full_sphere(self):
        est = allowed_solid_angle(KeepOutSet(()))
        assert est.solid_angle == pytest.approx(FULL)
        assert est.standard_error == 0.0
        assert est.fraction == pytest.approx(1.0)

    def test_single_cap_matches_the_closed_form(self):
        for r_deg in (1.0, 15.0, 45.0, 90.0, 135.0, 179.0):
            ks = KeepOutSet((ExclusionCone([0.3, -0.5, 0.8], r_deg * DEG, "c"),))
            expected = FULL - cap_solid_angle(r_deg * DEG)
            assert allowed_solid_angle(ks).solid_angle == pytest.approx(expected, abs=1e-10)

    def test_two_cap_case_matches_the_closed_form(self):
        # The hand case: union of the two caps is 1.3107171092330592 sr, so the
        # allowed sky is 4 pi - 1.3107171092330592 = 11.255653505126114 sr.
        ks = _pair()
        expected = FULL - cap_union_solid_angle(30 * DEG, 25 * DEG, 40 * DEG)
        assert expected == pytest.approx(11.255653505126114, abs=1e-12)
        assert allowed_solid_angle(ks).solid_angle == pytest.approx(expected, abs=1e-11)

    def test_disjoint_caps_add(self):
        ks = KeepOutSet(
            (
                ExclusionCone([0, 0, 1], 20 * DEG, "n"),
                ExclusionCone([0, 0, -1], 20 * DEG, "s"),
            )
        )
        expected = FULL - 2 * cap_solid_angle(20 * DEG)
        assert allowed_solid_angle(ks).solid_angle == pytest.approx(expected, abs=1e-11)

    def test_nested_caps_count_once(self):
        ks = KeepOutSet(
            (
                ExclusionCone([0, 0, 1], 40 * DEG, "big"),
                ExclusionCone([0, 0, 1], 10 * DEG, "small"),
            )
        )
        expected = FULL - cap_solid_angle(40 * DEG)
        assert allowed_solid_angle(ks).solid_angle == pytest.approx(expected, abs=1e-11)

    def test_everything_blocked(self):
        ks = KeepOutSet((ExclusionCone([0, 0, 1], np.pi, "all"),))
        assert allowed_solid_angle(ks).solid_angle == pytest.approx(0.0, abs=1e-12)

    def test_rotation_invariance(self):
        ks = _pair()
        base = allowed_solid_angle(ks).solid_angle
        for seed_angle in (0.3, 1.7, 2.9):
            r = rotation_matrix([0.2, 0.9, -0.4], seed_angle)
            assert allowed_solid_angle(ks.rotated(r)).solid_angle == pytest.approx(
                base, abs=1e-10
            )

    def test_three_overlapping_caps_agree_with_monte_carlo(self):
        ks = KeepOutSet(
            (
                ExclusionCone([1, 0, 0], 50 * DEG, "a"),
                ExclusionCone([0.6, 0.8, 0], 45 * DEG, "b"),
                ExclusionCone([0.6, 0.3, 0.74], 40 * DEG, "c"),
            )
        )
        quad = allowed_solid_angle(ks).solid_angle
        mc = allowed_solid_angle_monte_carlo(ks, 400_000, seed=7)
        assert abs(quad - mc.solid_angle) < 4.0 * mc.standard_error

    def test_nodes_validation(self):
        with pytest.raises(ValueError, match="nodes_per_band"):
            allowed_solid_angle(_pair(), nodes_per_band=1)


class TestAllowedFraction:
    def test_matches_solid_angle(self):
        ks = _pair()
        assert allowed_fraction(ks) == pytest.approx(
            allowed_solid_angle(ks).solid_angle / FULL
        )

    def test_hand_case_fraction(self):
        # 11.255653505126114 / (4 pi) = 0.8956964465352194.
        assert allowed_fraction(_pair()) == pytest.approx(0.8956964465352194, abs=1e-12)


class TestMonteCarlo:
    def test_agrees_with_quadrature_within_error(self):
        ks = _pair()
        quad = allowed_solid_angle(ks).solid_angle
        mc = allowed_solid_angle_monte_carlo(ks, 200_000, seed=3)
        assert abs(mc.solid_angle - quad) < 4.0 * mc.standard_error
        assert mc.n_samples == 200_000

    def test_reproducible(self):
        ks = _pair()
        a = allowed_solid_angle_monte_carlo(ks, 5000, seed=11)
        b = allowed_solid_angle_monte_carlo(ks, 5000, seed=11)
        assert a.solid_angle == b.solid_angle

    def test_error_shrinks_with_sample_count(self):
        ks = _pair()
        small = allowed_solid_angle_monte_carlo(ks, 2_000, seed=1)
        large = allowed_solid_angle_monte_carlo(ks, 200_000, seed=1)
        assert large.standard_error < small.standard_error

    def test_fraction_standard_error(self):
        mc = allowed_solid_angle_monte_carlo(_pair(), 10_000, seed=2)
        assert mc.fraction_standard_error == pytest.approx(mc.standard_error / FULL)

    def test_sample_count_validation(self):
        with pytest.raises(ValueError, match="n_samples"):
            allowed_solid_angle_monte_carlo(_pair(), 0)
