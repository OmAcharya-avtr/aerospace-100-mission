"""Unit tests for the H-representation polytope and its support function."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from invariantset import (
    EmptyPolytopeError,
    Polytope,
    UnboundedDirectionError,
    VertexEnumerationError,
)


def _support_by_lp(P: Polytope, c: np.ndarray) -> float:
    res = linprog(-np.asarray(c, float), A_ub=P.A, b_ub=P.b, bounds=(None, None), method="highs")
    assert res.status == 0
    return float(-res.fun)


class TestConstruction:
    def test_box_halfspace_count_and_dim(self):
        P = Polytope.from_box([0.0, 1.0], [2.0, 3.0])
        assert P.dim == 2
        assert P.n_halfspaces == 4
        assert P.is_box

    def test_from_bounds_matches_from_box(self):
        a = Polytope.from_bounds([-1.0, -2.0], [3.0, 4.0])
        b = Polytope.from_box([1.0, 1.0], [2.0, 3.0])
        assert np.allclose(a.A, b.A)
        assert np.allclose(a.b, b.b)

    def test_unit_box(self):
        P = Polytope.unit_box(3, 2.0)
        assert P.dim == 3 and P.n_halfspaces == 6
        # Vertex (2, 2, 2) is in, (2.1, 0, 0) is out.
        assert P.contains([2.0, 2.0, 2.0])
        assert not P.contains([2.1, 0.0, 0.0])

    def test_row_vector_A_is_promoted(self):
        P = Polytope([1.0, 1.0], [1.0])
        assert P.dim == 2 and P.n_halfspaces == 1

    def test_arrays_are_read_only(self):
        P = Polytope.unit_box(2)
        with pytest.raises(ValueError):
            P.A[0, 0] = 5.0
        with pytest.raises(ValueError):
            P.b[0] = 5.0

    def test_repr_names_the_kind(self):
        assert "box" in repr(Polytope.unit_box(2))
        assert "polytope" in repr(Polytope([[1.0, 0.0]], [1.0]))


class TestSupportFunction:
    def test_box_closed_form_equals_lp(self):
        # h_box(c) = c.m + |c|.r  (Rockafellar 1970).  Checked
        # against the generic linear programme on the same halfspaces.
        rng = np.random.default_rng(59001)
        P = Polytope.from_box([0.3, -0.7], [1.0, 2.0])
        generic = Polytope(P.A, P.b)  # same set, no box shortcut
        for c in rng.normal(size=(40, 2)):
            assert P.support(c) == pytest.approx(_support_by_lp(generic, c), abs=1e-10)

    def test_known_box_support_by_hand(self):
        # Box centre (1, 2), half-widths (0.5, 0.25), direction (2, -4):
        #   h = 2*1 + (-4)*2 + |2|*0.5 + |-4|*0.25 = 2 - 8 + 1 + 1 = -4
        P = Polytope.from_box([1.0, 2.0], [0.5, 0.25])
        assert P.support([2.0, -4.0]) == pytest.approx(-4.0, abs=1e-12)

    def test_support_equals_vertex_maximum(self):
        rng = np.random.default_rng(59002)
        P = Polytope(
            np.array([[1.0, 0.0], [-1.0, 0.5], [0.0, -1.0], [1.0, 1.0]]),
            np.array([1.0, 0.8, 1.0, 1.5]),
        )
        V = P.vertices()
        for c in rng.normal(size=(30, 2)):
            assert P.support(c) == pytest.approx(float(np.max(V @ c)), abs=1e-9)

    def test_support_many_matches_support(self):
        P = Polytope.from_box([0.0, 0.0], [1.0, 2.0])
        C = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert np.allclose(P.support_many(C), [P.support(c) for c in C])

    def test_unbounded_direction_raises(self):
        # The halfspace x1 <= 1 alone is bounded along +e1 only: h(e1) = 1,
        # and unbounded along -e1 and along both signs of e2.
        P = Polytope([[1.0, 0.0]], [1.0])
        assert P.support([1.0, 0.0]) == pytest.approx(1.0, abs=1e-9)
        for c in ([-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
            with pytest.raises(UnboundedDirectionError):
                P.support(c)

    def test_support_of_empty_polytope_raises(self):
        P = Polytope(np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0]))
        with pytest.raises(EmptyPolytopeError):
            P.support([1.0])


class TestPredicates:
    def test_is_empty(self):
        assert Polytope(np.array([[1.0], [-1.0]]), np.array([0.0, -1.0])).is_empty()
        assert not Polytope.unit_box(2).is_empty()

    def test_degenerate_row_with_negative_rhs_is_empty(self):
        P = Polytope(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([-1.0, 1.0]))
        assert P.is_empty()

    def test_degenerate_row_with_nonnegative_rhs_is_harmless(self):
        P = Polytope(np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]), np.array([3.0, 1.0, 1.0]))
        assert not P.is_empty()

    def test_is_bounded(self):
        assert Polytope.unit_box(2).is_bounded()
        assert Polytope(Polytope.unit_box(2).A, Polytope.unit_box(2).b).is_bounded()
        assert not Polytope([[1.0, 0.0]], [1.0]).is_bounded()

    def test_contains_tolerance(self):
        P = Polytope.from_box([0.0], [1.0])
        assert P.contains([1.0 + 1e-10])
        assert not P.contains([1.0 + 1e-3])


class TestRedundancy:
    def test_duplicate_rows_are_removed(self):
        A = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        b = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
        P = Polytope(A, b).remove_redundant()
        # x1 <= 2 is implied by x1 <= 1, so four rows survive.
        assert P.n_halfspaces == 4

    def test_removal_preserves_membership(self):
        rng = np.random.default_rng(59003)
        A = np.vstack([Polytope.unit_box(2).A, np.array([[1.0, 1.0], [1.0, 1.0]])])
        b = np.concatenate([Polytope.unit_box(2).b, [1.5, 3.0]])
        P = Polytope(A, b)
        R = P.remove_redundant()
        pts = rng.uniform(-1.5, 1.5, size=(400, 2))
        assert np.array_equal(
            [P.contains(p, tol=1e-12) for p in pts],
            [R.contains(p, tol=1e-12) for p in pts],
        )

    def test_degenerate_rows_dropped(self):
        A = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        b = np.array([5.0, 1.0, 1.0, 1.0, 1.0])
        assert Polytope(A, b).remove_redundant().n_halfspaces == 4

    def test_infeasible_degenerate_row_is_kept_as_the_witness(self):
        A = np.array([[0.0, 0.0], [1.0, 0.0]])
        b = np.array([-2.0, 1.0])
        R = Polytope(A, b).remove_redundant()
        assert R.n_halfspaces == 1
        assert R.is_empty()

    def test_large_tolerance_can_remove_a_real_facet(self):
        # x1 <= 1 and x1 <= 1.005: the second is redundant at any tolerance,
        # the first becomes "redundant" once tol >= 0.005, which enlarges the
        # set.  This is the mechanism section 4 of VALIDATION.md measures.
        A = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        b = np.array([1.0, 1.005, 1.0, 1.0, 1.0])
        tight = Polytope(A, b).remove_redundant(tol=1e-9)
        loose = Polytope(A, b).remove_redundant(tol=1e-2)
        assert tight.support([1.0, 0.0]) == pytest.approx(1.0, abs=1e-9)
        assert loose.support([1.0, 0.0]) == pytest.approx(1.005, abs=1e-9)

    def test_normalised_keeps_the_set(self):
        P = Polytope(np.array([[3.0, 4.0], [-1.0, 0.0], [0.0, -1.0]]), np.array([10.0, 0.0, 0.0]))
        N = P.normalised()
        assert np.allclose(np.linalg.norm(N.A, axis=1), 1.0)
        assert N.support([1.0, 0.0]) == pytest.approx(P.support([1.0, 0.0]), abs=1e-9)


class TestVerticesAndVolume:
    def test_unit_square_vertices(self):
        V = Polytope.unit_box(2).vertices()
        assert V.shape == (4, 2)
        assert {tuple(np.round(v, 9)) for v in V} == {
            (-1.0, -1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (1.0, 1.0),
        }

    def test_cube_vertices_and_volume(self):
        P = Polytope.unit_box(3, 0.5)
        assert P.vertices().shape == (8, 3)
        assert P.volume() == pytest.approx(1.0, abs=1e-9)

    def test_interval_volume(self):
        assert Polytope.from_bounds([-2.0], [3.0]).volume() == pytest.approx(5.0)

    def test_triangle_area(self):
        # Vertices (0,0), (1,0), (0,1): area 1/2 * 1 * 1 = 0.5
        P = Polytope(
            np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]), np.array([0.0, 0.0, 1.0])
        )
        assert P.volume() == pytest.approx(0.5, abs=1e-12)

    def test_flat_box_has_zero_volume(self):
        assert Polytope.from_box([0.0, 0.0], [1.0, 0.0]).volume() == pytest.approx(0.0)

    def test_unbounded_vertices_raises(self):
        with pytest.raises(VertexEnumerationError, match="unbounded"):
            Polytope([[1.0, 0.0]], [1.0]).vertices()

    def test_empty_vertices_raises(self):
        with pytest.raises(EmptyPolytopeError):
            Polytope(np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0])).vertices()

    def test_vertex_budget_is_enforced(self):
        P = Polytope.unit_box(2)
        with pytest.raises(VertexEnumerationError, match="max_bases"):
            P.vertices(max_bases=2)
