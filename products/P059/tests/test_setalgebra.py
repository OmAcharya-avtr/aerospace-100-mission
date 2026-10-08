"""Set-algebra unit tests, including the Minkowski/Pontryagin asymmetry."""

from __future__ import annotations

import numpy as np
import pytest

from invariantset import (
    EmptyPolytopeError,
    Polytope,
    intersect,
    is_subset,
    minkowski_sum,
    pontryagin_difference,
    support_gap,
)


def _triangle() -> Polytope:
    """{x1 >= 0, x2 >= 0, x1 + x2 <= 1}: vertices (0,0), (1,0), (0,1)."""
    return Polytope(
        np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]), np.array([0.0, 0.0, 1.0])
    )


class TestIntersection:
    def test_stacks_halfspaces(self):
        P = Polytope.unit_box(2)
        Q = Polytope.from_box([0.5, 0.0], [1.0, 1.0])
        R = intersect(P, Q)
        assert R.n_halfspaces == 8
        # x1 in [-1, 1] and x1 in [-0.5, 1.5] gives x1 in [-0.5, 1].
        assert R.support([1.0, 0.0]) == pytest.approx(1.0, abs=1e-9)
        assert R.support([-1.0, 0.0]) == pytest.approx(0.5, abs=1e-9)

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            intersect(Polytope.unit_box(2), Polytope.unit_box(3))

    def test_support_of_intersection_is_at_most_the_minimum(self):
        rng = np.random.default_rng(59101)
        P = Polytope.unit_box(2)
        Q = Polytope.from_box([0.4, -0.2], [0.8, 1.2])
        R = intersect(P, Q)
        for c in rng.normal(size=(40, 2)):
            assert R.support(c) <= min(P.support(c), Q.support(c)) + 1e-9


class TestPontryaginDifference:
    def test_boxes_subtract_half_widths(self):
        # box(r) (-) box(s) = box(r - s) exactly, elementwise.
        # r = (1.0, 2.0), s = (0.25, 0.5) gives half-widths (0.75, 1.5).
        P = Polytope.from_box([0.0, 0.0], [1.0, 2.0])
        Q = Polytope.from_box([0.0, 0.0], [0.25, 0.5])
        D = pontryagin_difference(P, Q)
        assert D.support([1.0, 0.0]) == pytest.approx(0.75, abs=1e-12)
        assert D.support([0.0, 1.0]) == pytest.approx(1.5, abs=1e-12)

    def test_triangle_minus_box_by_hand(self):
        # P = {x1>=0, x2>=0, x1+x2<=1}, Q = box of half-width 0.1.
        # Row -e1: rhs 0 - h_Q(-e1) = -0.1          -> x1 >= 0.1
        # Row -e2: rhs 0 - h_Q(-e2) = -0.1          -> x2 >= 0.1
        # Row (1,1): rhs 1 - h_Q((1,1)) = 1 - 0.2   -> x1 + x2 <= 0.8
        # That is a right triangle with legs 0.8 - 0.1 - 0.1 = 0.6,
        # so the area is 0.5 * 0.6 * 0.6 = 0.18.
        D = pontryagin_difference(_triangle(), Polytope.from_box([0.0, 0.0], [0.1, 0.1]))
        assert D.volume() == pytest.approx(0.18, abs=1e-12)

    def test_result_can_be_empty(self):
        # Eroding a box of half-width 0.1 by a box of half-width 0.5.
        D = pontryagin_difference(
            Polytope.from_box([0.0], [0.1]), Polytope.from_box([0.0], [0.5])
        )
        assert D.is_empty()

    def test_membership_characterisation(self):
        # x in P (-) Q  iff  x + Q subset P.  Checked pointwise.
        rng = np.random.default_rng(59102)
        P = _triangle()
        Q = Polytope.from_box([0.0, 0.0], [0.08, 0.05])
        D = pontryagin_difference(P, Q)
        VQ = Q.vertices()
        for x in rng.uniform(-0.2, 1.1, size=(300, 2)):
            inside = D.contains(x, tol=1e-12)
            shifted = all(P.contains(x + v, tol=1e-12) for v in VQ)
            assert inside == shifted

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            pontryagin_difference(Polytope.unit_box(2), Polytope.unit_box(1))


class TestMinkowskiSum:
    def test_boxes_add_half_widths(self):
        S = minkowski_sum(
            Polytope.from_box([0.0, 0.0], [1.0, 2.0]),
            Polytope.from_box([0.0, 0.0], [0.5, 0.25]),
        )
        assert S.support([1.0, 0.0]) == pytest.approx(1.5, abs=1e-9)
        assert S.support([0.0, 1.0]) == pytest.approx(2.25, abs=1e-9)

    def test_support_additivity(self):
        # h_{P (+) Q}(c) = h_P(c) + h_Q(c)  (Schneider 1993).
        rng = np.random.default_rng(59103)
        P = _triangle()
        Q = Polytope.from_box([0.1, -0.2], [0.3, 0.15])
        S = minkowski_sum(P, Q)
        for c in rng.normal(size=(40, 2)):
            assert S.support(c) == pytest.approx(P.support(c) + Q.support(c), abs=1e-9)

    def test_triangle_plus_box_area_by_hand(self):
        # area(A (+) B) = area(A) + area(B) + sum_i |e_i| h_B(n_i)
        # for a convex polygon A with edges e_i and outward unit normals n_i.
        # A = triangle with legs 0.6, B = box of half-width 0.1:
        #   area(A) = 0.18, area(B) = 0.04
        #   bottom edge: 0.6 * h_B((0,-1)) = 0.6 * 0.1       = 0.06
        #   left edge  : 0.6 * h_B((-1,0)) = 0.6 * 0.1       = 0.06
        #   hypotenuse : 0.6*sqrt(2) * h_B((1,1)/sqrt(2))
        #                = 0.848528 * 0.141421               = 0.12
        #   total = 0.18 + 0.04 + 0.06 + 0.06 + 0.12         = 0.46
        A = Polytope(
            np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]),
            np.array([-0.1, -0.1, 0.8]),
        )
        B = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
        assert A.volume() == pytest.approx(0.18, abs=1e-12)
        assert minkowski_sum(A, B).volume() == pytest.approx(0.46, abs=1e-9)

    def test_one_dimensional_sum(self):
        S = minkowski_sum(Polytope.from_bounds([-1.0], [2.0]), Polytope.from_bounds([0.0], [1.0]))
        assert S.support([1.0]) == pytest.approx(3.0, abs=1e-12)
        assert S.support([-1.0]) == pytest.approx(1.0, abs=1e-12)

    def test_empty_argument_raises(self):
        empty = Polytope(np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0]))
        with pytest.raises(EmptyPolytopeError):
            minkowski_sum(Polytope.from_bounds([-1.0], [1.0]), empty)

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            minkowski_sum(Polytope.unit_box(2), Polytope.unit_box(3))


class TestTheAsymmetry:
    """The Pontryagin difference is not an inverse of the Minkowski sum."""

    def test_erode_then_dilate_is_a_strict_subset(self):
        # Identity (I3): (P (-) Q) (+) Q subset P, strictly here.
        # area(P) = 0.5, area((P (-) Q) (+) Q) = 0.46 by the arithmetic in
        # test_triangle_plus_box_area_by_hand, so the deficit is 0.04,
        # which is 8.00 % of P.  The identity people assume is FALSE.
        P = _triangle()
        Q = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
        reopened = minkowski_sum(pontryagin_difference(P, Q), Q)
        contained, margin = is_subset(reopened, P)
        assert contained
        assert margin <= 1e-9
        assert P.volume() == pytest.approx(0.5, abs=1e-12)
        assert reopened.volume() == pytest.approx(0.46, abs=1e-9)
        deficit = P.volume() - reopened.volume()
        assert deficit == pytest.approx(0.04, abs=1e-9)
        assert deficit / P.volume() == pytest.approx(0.08, abs=1e-9)
        # And the assumed identity really does fail: P is NOT a subset.
        back, back_margin = is_subset(P, reopened)
        assert not back
        assert back_margin > 1e-3

    def test_dilate_then_erode_recovers_the_set(self):
        # Identity (I4): (P (+) Q) (-) Q = P for compact convex P, Q
        # (Kolmanovsky and Gilbert 1998).  This is the identity that holds.
        P = _triangle()
        Q = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
        closed = pontryagin_difference(minkowski_sum(P, Q), Q)
        forward, m1 = is_subset(closed, P)
        backward, m2 = is_subset(P, closed)
        assert forward and backward
        assert max(m1, m2) <= 1e-9
        assert closed.volume() == pytest.approx(P.volume(), abs=1e-9)

    def test_equality_in_i3_for_aligned_boxes(self):
        # The special case where (P (-) Q) (+) Q = P does hold: both aligned
        # boxes, with Q no larger than P in every axis.
        P = Polytope.from_box([0.0, 0.0], [1.0, 2.0])
        Q = Polytope.from_box([0.0, 0.0], [0.3, 0.4])
        reopened = minkowski_sum(pontryagin_difference(P, Q), Q)
        assert is_subset(reopened, P)[0]
        assert is_subset(P, reopened)[0]
        assert reopened.volume() == pytest.approx(P.volume(), abs=1e-9)


class TestSubsetAndGap:
    def test_is_subset_exact_and_reflexive(self):
        P = Polytope.unit_box(2)
        assert is_subset(P, P)[0]
        assert is_subset(Polytope.from_box([0.0, 0.0], [0.5, 0.5]), P)[0]
        contained, margin = is_subset(Polytope.from_box([0.0, 0.0], [1.5, 0.5]), P)
        assert not contained
        assert margin == pytest.approx(0.5, abs=1e-9)

    def test_empty_is_a_subset_of_everything(self):
        empty = Polytope(np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0]))
        contained, margin = is_subset(empty, Polytope.from_bounds([-1.0], [1.0]))
        assert contained
        assert margin == -np.inf

    def test_unbounded_candidate_is_not_a_subset(self):
        contained, margin = is_subset(Polytope([[1.0, 0.0]], [1.0]), Polytope.unit_box(2))
        assert not contained
        assert margin == np.inf

    def test_support_gap_signs(self):
        P = Polytope.unit_box(2)
        Q = Polytope.from_box([0.0, 0.0], [0.5, 0.5])
        gaps = support_gap(P, Q, P.A)
        assert np.all(gaps <= 0.0)
        assert gaps.shape == (4,)

    def test_is_subset_rejects_negative_tol(self):
        with pytest.raises(ValueError, match="non-negative"):
            is_subset(Polytope.unit_box(2), Polytope.unit_box(2), tol=-1.0)
