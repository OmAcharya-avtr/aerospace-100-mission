"""Unit and known-answer tests for keepout.cones."""

import numpy as np
import pytest

from keepout import ExclusionCone, KeepOutSet, body_exclusion_cone, rotation_matrix
from keepout.geometry import spherical_to_unit

DEG = np.pi / 180.0


def _in_plane(angle_deg: float) -> np.ndarray:
    """Unit vector in the x-y plane at ``angle_deg`` from +x."""
    return spherical_to_unit(angle_deg * DEG, 0.0)


class TestExclusionCone:
    def test_axis_is_normalised(self):
        c = ExclusionCone([0.0, 0.0, 3.0], 0.5, "sun")
        assert np.allclose(c.axis, [0, 0, 1])

    def test_half_angle_degrees(self):
        assert ExclusionCone([1, 0, 0], np.pi / 6, "a").half_angle_deg == pytest.approx(30.0)

    def test_solid_angle(self):
        # 2 pi (1 - cos 60 deg) = pi.
        c = ExclusionCone([1, 0, 0], np.pi / 3, "a")
        assert c.solid_angle == pytest.approx(np.pi)

    def test_contains_and_margin(self):
        c = ExclusionCone([1, 0, 0], 30 * DEG, "sun")
        assert c.contains(_in_plane(10.0))
        assert not c.contains(_in_plane(50.0))
        assert c.margin(_in_plane(50.0)) == pytest.approx(20 * DEG)
        assert c.margin(_in_plane(10.0)) == pytest.approx(-20 * DEG)

    def test_boundary_is_allowed(self):
        c = ExclusionCone([1, 0, 0], 30 * DEG, "sun")
        assert not c.contains(_in_plane(30.0))

    def test_vectorised_contains(self):
        c = ExclusionCone([1, 0, 0], 30 * DEG, "sun")
        dirs = np.stack([_in_plane(a) for a in (0.0, 29.9, 30.1, 90.0)])
        assert list(c.contains(dirs)) == [True, True, False, False]

    def test_zero_half_angle(self):
        c = ExclusionCone([1, 0, 0], 0.0, "point")
        assert not c.contains([1, 0, 0])
        assert c.solid_angle == pytest.approx(0.0)

    def test_full_sphere_cone(self):
        c = ExclusionCone([1, 0, 0], np.pi, "everything")
        assert c.contains([-1, 0, 0]) is False  # separation == pi == half_angle
        assert c.contains([0, 1, 0])

    @pytest.mark.parametrize("bad", [-0.1, 4.0, np.nan])
    def test_bad_half_angle(self, bad):
        with pytest.raises(ValueError):
            ExclusionCone([1, 0, 0], bad, "a")

    def test_bad_name(self):
        with pytest.raises(ValueError, match="non-empty string"):
            ExclusionCone([1, 0, 0], 0.1, "")

    def test_rotated(self):
        c = ExclusionCone([1, 0, 0], 30 * DEG, "sun")
        r = rotation_matrix([0, 0, 1], np.pi / 2)
        assert np.allclose(c.rotated(r).axis, [0, 1, 0], atol=1e-15)

    def test_rotated_bad_shape(self):
        with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
            ExclusionCone([1, 0, 0], 0.1, "a").rotated(np.eye(2))


class TestBodyExclusionCone:
    def test_limb_reference_adds_angular_radius(self):
        c = body_exclusion_cone("earth", [0, 0, -1], 70.0 * DEG, 10.0 * DEG, "limb")
        assert c.half_angle_deg == pytest.approx(80.0)

    def test_center_reference_uses_the_angle_as_given(self):
        c = body_exclusion_cone("earth", [0, 0, -1], 70.0 * DEG, 80.0 * DEG, "center")
        assert c.half_angle_deg == pytest.approx(80.0)

    def test_clipped_at_pi(self):
        c = body_exclusion_cone("x", [1, 0, 0], 3.0, 3.0, "limb")
        assert c.half_angle == pytest.approx(np.pi)

    def test_bad_reference(self):
        with pytest.raises(ValueError, match="'limb' or 'center'"):
            body_exclusion_cone("x", [1, 0, 0], 0.1, 0.1, "edge")

    @pytest.mark.parametrize("kwargs", [{"body": -0.1}, {"excl": -0.1}])
    def test_negative_angles_rejected(self, kwargs):
        body = kwargs.get("body", 0.1)
        excl = kwargs.get("excl", 0.1)
        with pytest.raises(ValueError, match="must be >= 0"):
            body_exclusion_cone("x", [1, 0, 0], body, excl)


class TestKeepOutSet:
    """The hand case: cone A (axis +x, 30 deg) and cone B (axis at 40 deg, 25 deg).

    Every boresight below lies in the x-y plane, so the angle to axis A is the
    boresight's own azimuth and the angle to axis B is |azimuth - 40 deg|.
    Cone A covers azimuth (-30, 30); cone B covers (15, 65). The union is
    (-30, 65) and the overlap is (15, 30).
    """

    @pytest.fixture
    def pair(self):
        a = ExclusionCone(_in_plane(0.0), 30 * DEG, "A")
        b = ExclusionCone(_in_plane(40.0), 25 * DEG, "B")
        return KeepOutSet((a, b))

    def test_both_violated_in_the_overlap(self, pair):
        # 20 deg: 20 < 30 (inside A) and |40 - 20| = 20 < 25 (inside B).
        assert set(pair.violations(_in_plane(20.0))) == {"A", "B"}

    def test_only_b_violated(self, pair):
        # 35 deg: 35 > 30 (outside A) and |40 - 35| = 5 < 25 (inside B).
        assert pair.violations(_in_plane(35.0)) == ("B",)

    def test_only_a_violated(self, pair):
        # 5 deg: 5 < 30 (inside A) and |40 - 5| = 35 > 25 (outside B).
        assert pair.violations(_in_plane(5.0)) == ("A",)

    def test_allowed_outside_the_union(self, pair):
        # 70 deg: 70 > 30 and |40 - 70| = 30 > 25.
        assert pair.violations(_in_plane(70.0)) == ()
        assert pair.is_allowed(_in_plane(70.0))

    def test_violations_are_ordered_worst_first(self, pair):
        # 25 deg: margin to A = 25 - 30 = -5 deg, to B = 15 - 25 = -10 deg.
        # B is the deeper violation, so it comes first.
        assert pair.violations(_in_plane(25.0)) == ("B", "A")

    def test_margin_is_the_minimum(self, pair):
        # 70 deg: margin to A = 40 deg, to B = 5 deg -> worst case is 5 deg.
        assert pair.margin(_in_plane(70.0)) == pytest.approx(5 * DEG)

    def test_margins_shape(self, pair):
        dirs = np.stack([_in_plane(a) for a in (0.0, 40.0, 90.0)])
        m = pair.margins(dirs)
        assert m.shape == (3, 2)

    def test_names_and_length(self, pair):
        assert pair.names == ("A", "B")
        assert len(pair) == 2
        assert pair[0].name == "A"
        assert [c.name for c in pair] == ["A", "B"]

    def test_with_cone(self, pair):
        bigger = pair.with_cone(ExclusionCone([0, 0, 1], 0.2, "C"))
        assert bigger.names == ("A", "B", "C")
        assert pair.names == ("A", "B")

    def test_empty_set_allows_everything(self):
        empty = KeepOutSet(())
        assert empty.is_allowed([1, 0, 0])
        assert empty.margin([1, 0, 0]) == float("inf")
        assert empty.violations([1, 0, 0]) == ()
        assert empty.margins([1, 0, 0]).shape == (0,)

    def test_empty_set_vectorised(self):
        empty = KeepOutSet(())
        dirs = np.stack([_in_plane(a) for a in (0.0, 40.0)])
        assert list(empty.is_allowed(dirs)) == [True, True]
        assert np.all(np.isinf(empty.margin(dirs)))

    def test_type_check(self):
        with pytest.raises(TypeError, match="ExclusionCone"):
            KeepOutSet(("not a cone",))

    def test_violations_requires_a_single_direction(self, pair):
        with pytest.raises(ValueError, match="single direction"):
            pair.violations(np.stack([_in_plane(0.0), _in_plane(10.0)]))

    def test_rotated_set_gives_the_same_verdict(self, pair):
        r = rotation_matrix([0.3, -0.7, 0.2], 1.234)
        p = _in_plane(20.0)
        assert set(pair.rotated(r).violations(r @ p)) == set(pair.violations(p))
