"""Unit and known-answer tests for keepout.geometry."""

import numpy as np
import pytest

from keepout import geometry as g


class TestUnit:
    def test_normalises(self):
        assert np.allclose(g.unit([3.0, 4.0, 0.0]), [0.6, 0.8, 0.0])

    def test_stack(self):
        out = g.unit([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        assert out.shape == (2, 3)
        assert np.allclose(np.linalg.norm(out, axis=-1), 1.0)

    @pytest.mark.parametrize("bad", [[1.0, 2.0], [[1.0, 2.0]], 5.0])
    def test_shape_errors(self, bad):
        with pytest.raises(ValueError, match=r"expected shape"):
            g.unit(bad)

    def test_zero_vector_rejected(self):
        with pytest.raises(ValueError, match="norm"):
            g.unit([0.0, 0.0, 0.0])

    def test_nonfinite_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            g.unit([np.nan, 0.0, 1.0])


class TestAngularSeparation:
    def test_orthogonal(self):
        assert g.angular_separation([1, 0, 0], [0, 1, 0]) == pytest.approx(np.pi / 2)

    def test_antiparallel(self):
        assert g.angular_separation([1, 0, 0], [-1, 0, 0]) == pytest.approx(np.pi)

    def test_identical(self):
        assert g.angular_separation([0, 0, 1], [0, 0, 2]) == pytest.approx(0.0)

    def test_known_45_degrees(self):
        # (1,0,0) to (1,1,0)/sqrt(2): cos = 1/sqrt(2) -> 45 deg exactly.
        assert g.angular_separation([1, 0, 0], [1, 1, 0]) == pytest.approx(np.pi / 4)

    def test_small_angle_accuracy(self):
        # arccos(a.b) loses about half the digits here; atan2 does not.
        eps = 1e-9
        a = np.array([1.0, 0.0, 0.0])
        b = g.unit([1.0, eps, 0.0])
        assert g.angular_separation(a, b) == pytest.approx(eps, rel=1e-9)

    def test_broadcast(self):
        out = g.angular_separation([[1, 0, 0], [0, 1, 0]], [1, 0, 0])
        assert np.allclose(out, [0.0, np.pi / 2])


class TestRotation:
    def test_z_rotation(self):
        r = g.rotation_matrix([0, 0, 1], np.pi / 2)
        assert np.allclose(r @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-15)

    def test_orthogonal_det_one(self):
        r = g.rotation_matrix([1.0, 2.0, 3.0], 0.7)
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-14)
        assert np.linalg.det(r) == pytest.approx(1.0)

    def test_random_rotations_are_rotations(self):
        rs = g.random_rotations(50, seed=0)
        assert rs.shape == (50, 3, 3)
        for r in rs:
            assert np.allclose(r.T @ r, np.eye(3), atol=1e-13)
            assert np.linalg.det(r) == pytest.approx(1.0)

    def test_random_rotations_rejects_zero(self):
        with pytest.raises(ValueError, match="n must be"):
            g.random_rotations(0)


class TestSphericalCoordinates:
    def test_round_trip(self):
        ra = np.array([0.0, 1.0, 3.0, 6.0])
        dec = np.array([0.0, 0.5, -1.2, 1.5])
        v = g.spherical_to_unit(ra, dec)
        ra2, dec2 = g.unit_to_spherical(v)
        assert np.allclose(ra2, ra)
        assert np.allclose(dec2, dec)

    def test_pole(self):
        assert np.allclose(g.spherical_to_unit(0.0, np.pi / 2), [0, 0, 1], atol=1e-16)

    def test_out_of_range_declination(self):
        with pytest.raises(ValueError, match="declination"):
            g.spherical_to_unit(0.0, 2.0)


class TestCapAreas:
    def test_hemisphere(self):
        assert g.cap_solid_angle(np.pi / 2) == pytest.approx(2 * np.pi)

    def test_full_sphere(self):
        assert g.cap_solid_angle(np.pi) == pytest.approx(4 * np.pi)

    def test_sixty_degrees(self):
        # 2 pi (1 - cos 60 deg) = 2 pi (1 - 1/2) = pi exactly.
        assert g.cap_solid_angle(np.pi / 3) == pytest.approx(np.pi)

    def test_range_check(self):
        with pytest.raises(ValueError, match=r"\[0, pi\]"):
            g.cap_solid_angle(4.0)

    def test_disjoint_caps(self):
        assert g.cap_intersection_solid_angle(0.2, 0.3, 0.6) == 0.0

    def test_nested_caps(self):
        # r2 entirely inside r1 -> intersection is the smaller cap.
        expected = g.cap_solid_angle(0.2)
        assert g.cap_intersection_solid_angle(0.9, 0.2, 0.1) == pytest.approx(expected)

    def test_coincident_caps(self):
        expected = g.cap_solid_angle(0.4)
        assert g.cap_intersection_solid_angle(0.4, 0.4, 0.0) == pytest.approx(expected)

    def test_two_hemispheres_give_a_lune(self):
        # Two hemispheres whose poles are d apart intersect in a lune of
        # dihedral angle pi - d, and a lune of angle A has area 2 A.
        for d in (0.1, 0.7, 1.5, 3.0):
            assert g.cap_intersection_solid_angle(
                np.pi / 2, np.pi / 2, d
            ) == pytest.approx(2 * (np.pi - d))

    def test_hand_computed_two_cone_case(self):
        # Hand calculation. All angles below are exact double-precision values
        # printed by the arithmetic itself; every line is one elementary step.
        #   r1 = 30 deg, r2 = 25 deg, axis separation d = 40 deg.
        #   |r1 - r2| = 5 deg < 40 deg < 55 deg = r1 + r2, so the caps properly
        #   overlap and neither contains the other.
        #
        #   cos r1 = 0.8660254037844387   sin r1 = 0.5
        #   cos r2 = 0.9063077870366499   sin r2 = 0.42261826174069944
        #   cos d  = 0.7660444431189780   sin d  = 0.6427876096865393
        #
        #   cos a1 = (cos r2 - cos r1 cos d) / (sin r1 sin d)
        #     cos r1 cos d = 0.6634139481689384
        #     numerator    = 0.9063077870366499 - 0.6634139481689384
        #                  = 0.24289383886771154
        #     denominator  = 0.5 * 0.6427876096865393 = 0.32139380484326957
        #     cos a1 = 0.24289383886771154 / 0.32139380484326957 = 0.7557514650481854
        #     a1     = 0.713995443187023 rad = 40.90892548618917 deg
        #
        #   cos a2 = (cos r1 - cos r2 cos d) / (sin r2 sin d)
        #     cos r2 cos d = 0.6942720440148838
        #     numerator    = 0.8660254037844387 - 0.6942720440148838
        #                  = 0.17175335976955486
        #     denominator  = 0.42261826174069944 * 0.6427876096865393
        #                  = 0.2716537822741844
        #     cos a2 = 0.17175335976955486 / 0.2716537822741844 = 0.6322509421061604
        #     a2     = 0.8863412197824059 rad = 50.78361110200917 deg
        #
        #   cos g  = (cos d - cos r1 cos r2) / (sin r1 sin r2)
        #     cos r1 cos r2 = 0.7848855672213958
        #     numerator     = 0.7660444431189780 - 0.7848855672213958
        #                   = -0.018841124102417783
        #     denominator   = 0.5 * 0.42261826174069944 = 0.2113091308703497
        #     cos g = -0.018841124102417783 / 0.2113091308703497
        #           = -0.08916379535902733
        #     g     = 1.6600786915769774 rad = 95.11550268696070 deg
        #
        #   A = 2 (pi - g) - 2 a1 cos r1 - 2 a2 cos r2
        #     2 (pi - g)      = 2 * (3.141592653589793 - 1.6600786915769774)
        #                     = 2.9630279240256314
        #     2 a1 cos r1     = 2 * 0.713995443187023 * 0.8660254037844387
        #                     = 1.2366763839725818
        #     2 a2 cos r2     = 2 * 0.8863412197824059 * 0.9063077870366499
        #                     = 1.6065958989207145
        #     A = 2.9630279240256314 - 1.2366763839725818 - 1.6065958989207145
        #       = 0.11975564113233506 sr
        r1, r2, d = np.radians(30.0), np.radians(25.0), np.radians(40.0)
        assert g.cap_intersection_solid_angle(r1, r2, d) == pytest.approx(
            0.11975564113233506, abs=1e-14
        )
        # Union follows: A1 + A2 - A_int.
        #   A1 = 2 pi (1 - cos 30) = 0.8417872144769325 sr
        #   A2 = 2 pi (1 - cos 25) = 0.5886855358884618 sr
        #   union = 0.8417872144769325 + 0.5886855358884618 - 0.11975564113233506
        #         = 1.3107171092330592 sr
        assert g.cap_union_solid_angle(r1, r2, d) == pytest.approx(
            1.3107171092330592, abs=1e-14
        )

    def test_intersection_input_validation(self):
        with pytest.raises(ValueError, match="r1"):
            g.cap_intersection_solid_angle(-0.1, 0.2, 0.3)
        with pytest.raises(ValueError, match="separation"):
            g.cap_intersection_solid_angle(0.1, 0.2, 4.0)


class TestFibonacciSphere:
    def test_unit_norm_and_count(self):
        p = g.fibonacci_sphere(1000)
        assert p.shape == (1000, 3)
        assert np.allclose(np.linalg.norm(p, axis=1), 1.0)

    def test_centroid_near_origin(self):
        p = g.fibonacci_sphere(20000)
        assert np.linalg.norm(p.mean(axis=0)) < 1e-3

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="n must be"):
            g.fibonacci_sphere(0)
