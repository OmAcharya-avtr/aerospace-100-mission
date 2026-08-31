"""Property-based tests (Hypothesis).

The central invariant of this package: a keep-out verdict is a statement about
the *relative* geometry of a boresight and a set of cone axes. Rotating the
boresight and every cone axis by the same rotation cannot change which cones
are violated, by how much, or how much of the sky is left. If it did, the
answer would depend on the frame the caller happened to write the vectors in.
"""

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from keepout import (
    ExclusionCone,
    KeepOutSet,
    allowed_solid_angle,
    angular_separation,
    cap_intersection_solid_angle,
    cap_solid_angle,
    rotation_matrix,
    unit,
)

FINITE = dict(allow_nan=False, allow_infinity=False)
COMPONENT = st.floats(min_value=-5.0, max_value=5.0, **FINITE)
VECTOR = st.lists(COMPONENT, min_size=3, max_size=3)
HALF_ANGLE = st.floats(min_value=0.0, max_value=np.pi, **FINITE)
ANGLE = st.floats(min_value=-10.0, max_value=10.0, **FINITE)

SLOW = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
FAST = settings(max_examples=250, deadline=None)


def _ok(v):
    return np.linalg.norm(np.asarray(v, dtype=float)) > 1e-6


def _rot(axis, angle):
    return rotation_matrix(axis, angle)


class TestAngularSeparation:
    @given(a=VECTOR, b=VECTOR)
    @FAST
    def test_symmetric_and_bounded(self, a, b):
        assume(_ok(a) and _ok(b))
        s1 = angular_separation(a, b)
        s2 = angular_separation(b, a)
        assert abs(s1 - s2) < 1e-12
        assert -1e-12 <= s1 <= np.pi + 1e-12

    @given(a=VECTOR, b=VECTOR, axis=VECTOR, angle=ANGLE)
    @FAST
    def test_rotation_invariant(self, a, b, axis, angle):
        assume(_ok(a) and _ok(b) and _ok(axis))
        r = _rot(axis, angle)
        before = angular_separation(a, b)
        after = angular_separation(r @ np.asarray(a, float), r @ np.asarray(b, float))
        assert abs(before - after) < 1e-9

    @given(a=VECTOR, s=st.floats(min_value=1e-3, max_value=1e3, **FINITE))
    @FAST
    def test_scale_invariant(self, a, s):
        assume(_ok(a))
        assert angular_separation(a, np.asarray(a, float) * s) < 1e-12


class TestConeInvariance:
    @given(
        boresight=VECTOR,
        axis1=VECTOR,
        axis2=VECTOR,
        r1=HALF_ANGLE,
        r2=HALF_ANGLE,
        rot_axis=VECTOR,
        angle=ANGLE,
    )
    @FAST
    def test_violation_set_is_rotation_invariant(
        self, boresight, axis1, axis2, r1, r2, rot_axis, angle
    ):
        """Rotating the whole geometry cannot change which cones are violated.

        Restricted to boresights at least 1e-7 rad (0.02 arcsec) clear of every
        cone boundary. Rotating a unit vector perturbs it by of order 1e-16 rad,
        so a verdict taken closer to a boundary than that is not decidable in
        double precision and the discrete answer can legitimately flip; the
        continuous statement is
        :meth:`test_margins_are_rotation_invariant` below, which holds
        everywhere.
        """
        assume(_ok(boresight) and _ok(axis1) and _ok(axis2) and _ok(rot_axis))
        ks = KeepOutSet(
            (ExclusionCone(axis1, r1, "one"), ExclusionCone(axis2, r2, "two"))
        )
        b = unit(boresight)
        assume(np.min(np.abs(ks.margins(b))) > 1e-7)
        r = _rot(rot_axis, angle)
        assert set(ks.violations(b)) == set(ks.rotated(r).violations(r @ b))

    @given(
        boresight=VECTOR,
        axis1=VECTOR,
        axis2=VECTOR,
        r1=HALF_ANGLE,
        r2=HALF_ANGLE,
        rot_axis=VECTOR,
        angle=ANGLE,
    )
    @FAST
    def test_margins_are_rotation_invariant(
        self, boresight, axis1, axis2, r1, r2, rot_axis, angle
    ):
        assume(_ok(boresight) and _ok(axis1) and _ok(axis2) and _ok(rot_axis))
        ks = KeepOutSet(
            (ExclusionCone(axis1, r1, "one"), ExclusionCone(axis2, r2, "two"))
        )
        r = _rot(rot_axis, angle)
        b = unit(boresight)
        before = ks.margins(b)
        after = ks.rotated(r).margins(r @ b)
        assert np.allclose(before, after, atol=1e-9)

    @given(boresight=VECTOR, axis=VECTOR, r=HALF_ANGLE)
    @FAST
    def test_contains_iff_negative_margin(self, boresight, axis, r):
        assume(_ok(boresight) and _ok(axis))
        c = ExclusionCone(axis, r, "c")
        assert bool(c.contains(boresight)) == bool(c.margin(boresight) < 0.0)

    @given(boresight=VECTOR, axis=VECTOR, r=HALF_ANGLE)
    @FAST
    def test_margin_is_separation_minus_half_angle(self, boresight, axis, r):
        assume(_ok(boresight) and _ok(axis))
        c = ExclusionCone(axis, r, "c")
        sep = angular_separation(boresight, axis)
        assert abs(c.margin(boresight) - (sep - r)) < 1e-12

    @given(axis=VECTOR, r=HALF_ANGLE)
    @FAST
    def test_axis_is_always_violated_unless_the_cone_is_degenerate(self, axis, r):
        assume(_ok(axis))
        assume(r > 1e-9)
        c = ExclusionCone(axis, r, "c")
        assert c.contains(axis)


class TestCapAreaIdentities:
    @given(r1=HALF_ANGLE, r2=HALF_ANGLE, d=st.floats(min_value=0.0, max_value=np.pi, **FINITE))
    @FAST
    def test_intersection_is_bounded_by_both_caps(self, r1, r2, d):
        inter = cap_intersection_solid_angle(r1, r2, d)
        assert -1e-12 <= inter <= min(cap_solid_angle(r1), cap_solid_angle(r2)) + 1e-9

    @given(r1=HALF_ANGLE, r2=HALF_ANGLE, d=st.floats(min_value=0.0, max_value=np.pi, **FINITE))
    @FAST
    def test_union_never_exceeds_the_sphere(self, r1, r2, d):
        union = cap_solid_angle(r1) + cap_solid_angle(r2) - cap_intersection_solid_angle(
            r1, r2, d
        )
        assert -1e-12 <= union <= 4 * np.pi + 1e-9

    @given(r1=HALF_ANGLE, r2=HALF_ANGLE, d=st.floats(min_value=0.0, max_value=np.pi, **FINITE))
    @FAST
    def test_symmetric_in_the_two_caps(self, r1, r2, d):
        a = cap_intersection_solid_angle(r1, r2, d)
        b = cap_intersection_solid_angle(r2, r1, d)
        assert abs(a - b) < 1e-9


class TestAllowedRegionInvariance:
    @given(
        axis1=VECTOR,
        axis2=VECTOR,
        r1=st.floats(min_value=0.05, max_value=np.pi - 0.05, **FINITE),
        r2=st.floats(min_value=0.05, max_value=np.pi - 0.05, **FINITE),
        rot_axis=VECTOR,
        angle=ANGLE,
    )
    @SLOW
    def test_allowed_solid_angle_is_rotation_invariant(
        self, axis1, axis2, r1, r2, rot_axis, angle
    ):
        assume(_ok(axis1) and _ok(axis2) and _ok(rot_axis))
        ks = KeepOutSet(
            (ExclusionCone(axis1, r1, "one"), ExclusionCone(axis2, r2, "two"))
        )
        before = allowed_solid_angle(ks).solid_angle
        after = allowed_solid_angle(ks.rotated(_rot(rot_axis, angle))).solid_angle
        assert abs(before - after) < 1e-7

    @given(
        axis1=VECTOR,
        axis2=VECTOR,
        r1=st.floats(min_value=0.05, max_value=np.pi - 0.05, **FINITE),
        r2=st.floats(min_value=0.05, max_value=np.pi - 0.05, **FINITE),
    )
    @SLOW
    def test_band_quadrature_matches_the_two_cap_closed_form(self, axis1, axis2, r1, r2):
        assume(_ok(axis1) and _ok(axis2))
        a1, a2 = unit(axis1), unit(axis2)
        ks = KeepOutSet((ExclusionCone(a1, r1, "one"), ExclusionCone(a2, r2, "two")))
        d = angular_separation(a1, a2)
        closed = (
            4 * np.pi
            - cap_solid_angle(r1)
            - cap_solid_angle(r2)
            + cap_intersection_solid_angle(r1, r2, d)
        )
        assert abs(allowed_solid_angle(ks).solid_angle - closed) < 1e-7
