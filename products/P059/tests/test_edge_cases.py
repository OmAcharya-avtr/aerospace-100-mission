"""Edge cases: empty sets, unbounded directions, oversized disturbances."""

from __future__ import annotations

import numpy as np
import pytest

from invariantset import (
    CONVERGED,
    EMPTY,
    ITERATION_CAP,
    EmptyPolytopeError,
    Polytope,
    UnboundedDirectionError,
    VertexEnumerationError,
    is_subset,
    maximal_robust_invariant_set,
    minkowski_sum,
    pontryagin_difference,
    verify_robust_invariance,
)


class TestEmptySets:
    def test_empty_x_gives_empty_answer_at_k_1(self):
        X = Polytope(np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0]))
        W = Polytope.from_box([0.0], [0.1])
        res = maximal_robust_invariant_set(np.array([[0.5]]), X, W, max_iter=5)
        assert res.termination == EMPTY
        assert res.iterations == 1

    def test_single_point_x_is_invariant_only_without_disturbance(self):
        # X = {0}: a flat box.  With W = {0} the origin is invariant under
        # any A with A 0 = 0; with any non-zero W it is not.
        X = Polytope.from_box([0.0, 0.0], [0.0, 0.0])
        zero_w = Polytope.from_box([0.0, 0.0], [0.0, 0.0])
        res = maximal_robust_invariant_set(np.eye(2) * 0.5, X, zero_w, max_iter=5)
        assert res.termination == CONVERGED
        assert res.polytope.volume() == pytest.approx(0.0)

        small_w = Polytope.from_box([0.0, 0.0], [1e-6, 1e-6])
        res2 = maximal_robust_invariant_set(np.eye(2) * 0.5, X, small_w, max_iter=5)
        assert res2.termination == EMPTY

    def test_pontryagin_difference_into_the_empty_set(self):
        D = pontryagin_difference(
            Polytope.from_box([0.0, 0.0], [0.1, 0.1]),
            Polytope.from_box([0.0, 0.0], [1.0, 1.0]),
        )
        assert D.is_empty()
        with pytest.raises(EmptyPolytopeError):
            D.vertices()

    def test_empty_polytope_volume_is_zero(self):
        assert Polytope(np.array([[1.0], [-1.0]]), np.array([-1.0, -1.0])).volume() == 0.0


class TestUnboundedDirections:
    def test_unbounded_x_still_runs_the_recursion(self):
        # X = {x2 <= 1} only: unbounded in x1 and in -x2.  The recursion is
        # still well defined because it only ever evaluates h_W, not h_X.
        X = Polytope([[0.0, 1.0]], [1.0])
        W = Polytope.from_box([0.0, 0.0], [0.01, 0.01])
        A = np.diag([0.5, 0.5])
        res = maximal_robust_invariant_set(A, X, W, max_iter=10, track_geometry=True)
        assert res.termination == CONVERGED
        assert all(r.vertices is None for r in res.history)
        assert not res.polytope.is_bounded()

    def test_volume_of_an_unbounded_set_raises_through_vertices(self):
        with pytest.raises(VertexEnumerationError, match="unbounded"):
            Polytope([[0.0, 1.0]], [1.0]).volume()

    def test_unbounded_w_makes_the_pontryagin_difference_unusable(self):
        X = Polytope.unit_box(2)
        W = Polytope([[1.0, 0.0]], [0.1])
        with pytest.raises(UnboundedDirectionError):
            pontryagin_difference(X, W)

    def test_minkowski_sum_rejects_an_unbounded_argument(self):
        with pytest.raises(VertexEnumerationError, match="unbounded"):
            minkowski_sum(Polytope.unit_box(2), Polytope([[1.0, 0.0]], [1.0]))


class TestDisturbanceLargerThanTheConstraintSet:
    def test_w_bigger_than_x_is_empty(self):
        # |x| <= 0.1 but |w| <= 1.0: one step leaves X from anywhere.
        X = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
        W = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        res = maximal_robust_invariant_set(np.diag([0.5, 0.5]), X, W, max_iter=5)
        assert res.termination == EMPTY
        assert res.iterations == 1

    def test_w_equal_to_x_with_zero_dynamics_is_the_boundary(self):
        # A = 0 and W = X: successors are exactly W = X, so X is invariant
        # with zero margin.
        X = Polytope.from_box([0.0], [1.0])
        W = Polytope.from_box([0.0], [1.0])
        res = maximal_robust_invariant_set(np.zeros((1, 1)), X, W, max_iter=5)
        assert res.termination == CONVERGED
        invariant, margin = verify_robust_invariance(np.zeros((1, 1)), res.polytope, W)
        assert invariant
        assert margin == pytest.approx(0.0, abs=1e-15)

    def test_w_slightly_bigger_than_x_with_zero_dynamics_is_empty(self):
        X = Polytope.from_box([0.0], [1.0])
        W = Polytope.from_box([0.0], [1.0 + 1e-6])
        res = maximal_robust_invariant_set(np.zeros((1, 1)), X, W, max_iter=5)
        assert res.termination == EMPTY

    def test_zero_disturbance_reduces_to_the_undisturbed_problem(self):
        # With W = {0} and a stable A, X itself is invariant whenever
        # h_X(A^T c) <= b, which holds for A = 0.5 I and a centred box.
        X = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        W = Polytope.from_box([0.0, 0.0], [0.0, 0.0])
        res = maximal_robust_invariant_set(np.diag([0.5, 0.5]), X, W, max_iter=5)
        assert res.termination == CONVERGED
        assert res.iterations == 1
        assert res.polytope.volume() == pytest.approx(4.0, abs=1e-9)


class TestNonConvergence:
    def test_marginally_stable_system_does_not_converge(self):
        # Eigenvalues on the unit circle: A = R(0.3 rad), no damping.  Nothing
        # contracts, so the recursion shaves a sliver every iteration and
        # never terminates.
        A = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
        X = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        W = Polytope.from_box([0.0, 0.0], [0.01, 0.01])
        res = maximal_robust_invariant_set(A, X, W, max_iter=10, track_geometry=False)
        assert res.termination == ITERATION_CAP
        assert not res.converged
        assert "DID NOT CONVERGE" in res.report()
        assert "OUTER bound" in res.report()

    def test_unstable_system_hits_the_cap_or_empties(self):
        A = np.diag([1.2, 1.2])
        X = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        W = Polytope.from_box([0.0, 0.0], [0.05, 0.05])
        res = maximal_robust_invariant_set(A, X, W, max_iter=60, track_geometry=False)
        assert res.termination in (EMPTY, ITERATION_CAP)

    def test_cap_of_one_returns_the_first_iterate(self):
        A = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
        X = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        W = Polytope.from_box([0.0, 0.0], [0.01, 0.01])
        res = maximal_robust_invariant_set(A, X, W, max_iter=1, track_geometry=False)
        assert res.iterations == 1
        assert res.termination == ITERATION_CAP
        assert is_subset(res.polytope, X, tol=1e-9)[0]


class TestOneDimensional:
    def test_interval_algebra(self):
        P = Polytope.from_bounds([-1.0], [2.0])
        Q = Polytope.from_bounds([-0.25], [0.5])
        # P (+) Q = [-1 + -0.25, 2 + 0.5] = [-1.25, 2.5].
        # P (-) Q = {x : x + Q subset P}
        #         = {x : x - 0.25 >= -1  and  x + 0.5 <= 2}
        #         = [-0.75, 1.5].
        # Note the Pontryagin difference is NOT [-1 + 0.25, 2 - 0.5]
        # = [-0.75, 1.5] by coincidence here but is NOT an interval-width
        # subtraction in general; it is the erosion (4), and the first version
        # of this test asserted [-0.5, 1.75] and was wrong (VALIDATION.md
        # section 8, item 2).
        S = minkowski_sum(P, Q)
        D = pontryagin_difference(P, Q)
        assert S.support([1.0]) == pytest.approx(2.5, abs=1e-12)
        assert S.support([-1.0]) == pytest.approx(1.25, abs=1e-12)
        assert D.support([1.0]) == pytest.approx(1.5, abs=1e-12)
        assert D.support([-1.0]) == pytest.approx(0.75, abs=1e-12)

    def test_one_dimensional_vertices_are_the_endpoints(self):
        V = Polytope.from_bounds([-1.0], [2.0]).vertices()
        assert sorted(np.round(V.ravel(), 12)) == [-1.0, 2.0]
