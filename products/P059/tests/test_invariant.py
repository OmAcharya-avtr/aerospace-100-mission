"""Mechanics of the one-step-set recursion and its reporting."""

from __future__ import annotations

import numpy as np
import pytest

from invariantset import (
    CONVERGED,
    EMPTY,
    ITERATION_CAP,
    InvariantSetResult,
    Polytope,
    get_system,
    growth_table,
    is_subset,
    maximal_robust_invariant_set,
    pre_set,
    verify_robust_invariance,
)


class TestPreSet:
    def test_rows_are_c_times_a_with_shifted_rhs(self):
        # S = {|x| <= 1}^2, A = 0.5 I, W = box 0.1.
        # Row e1 becomes 0.5 x1 <= 1 - 0.1 = 0.9, i.e. x1 <= 1.8.
        S = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        W = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
        P = pre_set(np.diag([0.5, 0.5]), S, W)
        assert P.n_halfspaces == 4
        assert P.support([1.0, 0.0]) == pytest.approx(1.8, abs=1e-9)

    def test_singular_a_produces_degenerate_rows(self):
        S = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        W = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
        P = pre_set(np.array([[0.0, 1.0], [0.0, 0.0]]), S, W)
        norms = np.linalg.norm(P.A, axis=1)
        assert int(np.sum(norms < 1e-12)) == 2

    def test_pre_set_is_exactly_the_states_whose_successors_stay_in_s(self):
        rng = np.random.default_rng(59201)
        A = np.array([[0.9, 0.2], [-0.1, 0.8]])
        S = Polytope.from_box([0.0, 0.0], [1.0, 1.0])
        W = Polytope.from_box([0.0, 0.0], [0.05, 0.08])
        P = pre_set(A, S, W)
        VW = W.vertices()
        for x in rng.uniform(-1.6, 1.6, size=(300, 2)):
            in_pre = P.contains(x, tol=1e-12)
            stays = all(S.contains(A @ x + w, tol=1e-12) for w in VW)
            assert in_pre == stays

    def test_shape_validation(self):
        S = Polytope.unit_box(2)
        W = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
        with pytest.raises(ValueError, match="square"):
            pre_set(np.zeros((2, 3)), S, W)
        with pytest.raises(ValueError, match="dim"):
            pre_set(np.eye(3), S, W)
        with pytest.raises(ValueError, match="dim"):
            pre_set(np.eye(2), S, Polytope.unit_box(3))
        with pytest.raises(ValueError, match="non-finite"):
            pre_set(np.array([[np.nan, 0.0], [0.0, 1.0]]), S, W)


class TestConvergenceAndTermination:
    def test_converged_set_is_invariant_and_inside_x(self):
        s = get_system("attitude_loop")
        res = maximal_robust_invariant_set(s.A, s.X, s.W)
        assert res.converged
        assert is_subset(res.polytope, s.X, tol=1e-9)[0]
        assert verify_robust_invariance(s.A, res.polytope, s.W)[0]

    def test_iteration_cap_is_reported_not_raised(self):
        s = get_system("slow_pair")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=12, track_geometry=False)
        assert res.termination == ITERATION_CAP
        assert not res.converged
        assert res.iterations == 12
        assert res.polytope is not None
        assert "DID NOT CONVERGE" in res.report()

    def test_the_cap_result_is_only_an_outer_bound(self):
        # The returned polytope is a superset of S_inf and is itself NOT
        # robustly invariant, which is exactly why it must not be reported as
        # the answer.
        s = get_system("slow_pair")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=12, track_geometry=False)
        invariant, margin = verify_robust_invariance(s.A, res.polytope, s.W)
        assert not invariant
        assert margin > 0.0
        deeper = maximal_robust_invariant_set(
            s.A, s.X, s.W, max_iter=24, track_geometry=False
        )
        assert is_subset(deeper.polytope, res.polytope, tol=1e-9)[0]

    def test_more_iterations_never_grow_the_set(self):
        s = get_system("damped_rotation_2d")
        small = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=3, track_geometry=False)
        big = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=30, track_geometry=False)
        assert is_subset(big.polytope, small.polytope, tol=1e-9)[0]

    def test_empty_result_has_no_polytope(self):
        s = get_system("scalar_empty")
        res = maximal_robust_invariant_set(s.A, s.X, s.W)
        assert res.termination == EMPTY
        assert res.is_empty
        assert res.polytope is None
        assert not res.converged

    def test_already_invariant_x_terminates_immediately(self):
        s = get_system("decoupled_2d")
        res = maximal_robust_invariant_set(s.A, s.X, s.W)
        assert res.iterations == 1
        assert res.termination == CONVERGED

    def test_shrink_margin_decreases_monotonically(self):
        s = get_system("damped_rotation_2d")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=30, track_geometry=False)
        margins = [r.shrink_margin for r in res.history]
        for later, earlier in zip(margins[1:], margins[:-1], strict=True):
            assert later <= earlier + 1e-12

    def test_history_records_every_iteration(self):
        s = get_system("damped_rotation_2d")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=30, track_geometry=True)
        assert [r.k for r in res.history] == list(range(1, res.iterations + 1))

    def test_track_geometry_off_leaves_counts_unset(self):
        s = get_system("nilpotent_2d")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, track_geometry=False)
        assert all(r.vertices is None and r.volume is None for r in res.history)

    def test_vertex_budget_overflow_records_none_rather_than_raising(self):
        s = get_system("damped_rotation_2d")
        res = maximal_robust_invariant_set(
            s.A, s.X, s.W, max_iter=6, track_geometry=True, vertex_budget=1
        )
        assert all(r.vertices is None for r in res.history)
        assert res.iterations == 6


class TestVerifyRobustInvariance:
    def test_a_strict_subset_of_the_maximal_set_has_slack(self):
        s = get_system("decoupled_2d")
        shrunk = Polytope.from_box([0.0, 0.0], [0.5, 0.5])
        invariant, margin = verify_robust_invariance(s.A, shrunk, s.W)
        # Facet x1 <= 0.5: h(0.5 e1) + 0.1 - 0.5 = 0.25 + 0.1 - 0.5 = -0.15.
        assert invariant
        assert margin == pytest.approx(-0.15, abs=1e-12)

    def test_an_oversized_set_is_rejected(self):
        s = get_system("attitude_loop")
        invariant, margin = verify_robust_invariance(s.A, s.X.remove_redundant(), s.W)
        assert not invariant
        assert margin > 0.0

    def test_degenerate_rows_are_handled(self):
        S = Polytope(
            np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            np.array([1.0, 1.0, 1.0, 1.0]),
        )
        W = Polytope.from_box([0.0, 0.0], [0.1, 0.1])
        invariant, margin = verify_robust_invariance(
            np.array([[0.0, 1.0], [0.0, 0.0]]), S, W
        )
        # Rows e2 and -e2 map to the zero normal, so their margin is
        # h_W(c) - b = 0.1 - 1 = -0.9; rows +-e1 give 1 + 0.1 - 1 = 0.1 > 0.
        assert not invariant
        assert margin == pytest.approx(0.1, abs=1e-12)


class TestResultReporting:
    def test_report_contains_the_declared_criterion(self):
        s = get_system("nilpotent_2d")
        text = maximal_robust_invariant_set(s.A, s.X, s.W).report()
        assert "convergence tol" in text
        assert "redundancy tol" in text
        assert "iterations" in text

    def test_growth_table_ratios(self):
        s = get_system("damped_rotation_2d")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=30, track_geometry=True)
        rows = growth_table(res)
        assert rows[0].margin_ratio is None
        assert all(r.removed == r.halfspaces_raw - r.halfspaces for r in rows)
        assert all(r.volume_ratio <= 1.0 + 1e-9 for r in rows[1:] if r.volume_ratio)

    def test_empty_result_dataclass_defaults(self):
        res = InvariantSetResult(
            termination=EMPTY, converged=False, iterations=3, polytope=None
        )
        assert res.history == ()
        assert res.is_empty
