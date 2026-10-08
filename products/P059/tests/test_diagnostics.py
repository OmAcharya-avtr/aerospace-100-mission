"""Diagnostics: measured facet growth, stalling, and tolerance sensitivity."""

from __future__ import annotations

import numpy as np
import pytest

from invariantset import (
    CONVERGED,
    ITERATION_CAP,
    get_system,
    growth_table,
    maximal_robust_invariant_set,
    tolerance_sweep,
)


class TestMeasuredGrowth2D:
    """A box-constrained damped rotation in 2-D, growth measured not estimated.

    Measured on `damped_rotation_2d` (A = 0.995 R(0.2 rad), X and W boxes):
    the raw row count before redundancy removal is `8 k` because each
    iteration stacks `Omega_{k-1}` on `Pre(Omega_{k-1})`, and the kept facet
    count is `4 (k + 1)` until the recursion stops.  In two dimensions a
    bounded full-dimensional polytope has as many vertices as facets, so the
    vertex count follows the facet count exactly.
    """

    def test_raw_and_kept_counts_follow_the_measured_law(self):
        s = get_system("damped_rotation_2d")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=30, track_geometry=True)
        assert res.termination == CONVERGED
        assert res.iterations == 8
        rows = growth_table(res)
        for row in rows[:-1]:
            assert row.halfspaces_raw == 8 * row.k
            assert row.halfspaces == 4 * (row.k + 1)
            assert row.vertices == row.halfspaces
        assert rows[-1].halfspaces == 32
        assert rows[-1].vertices == 32

    def test_final_volume_is_pinned(self):
        s = get_system("damped_rotation_2d")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=30, track_geometry=True)
        assert res.polytope.volume() == pytest.approx(3.205016434, rel=1e-8)
        # X has area 4.0, so S_inf covers 80.13 % of the declared constraints.
        assert res.polytope.volume() / 4.0 == pytest.approx(0.801254108, rel=1e-7)


class TestMeasuredGrowth3D:
    """In three dimensions vertices grow at twice the facet rate.

    Measured on `damped_rotation_3d`: facets go 10, 14, 18, ... (+4 per
    iteration) while vertices go 16, 24, 32, ... (+8 per iteration).  The
    vertex-to-facet ratio therefore rises from 8/6 = 1.333 at `Omega_0` to
    56/30 = 1.867 at convergence, which is the growth figure this package
    reports rather than estimates.
    """

    def test_vertex_growth_is_twice_the_facet_growth(self):
        s = get_system("damped_rotation_3d")
        res = maximal_robust_invariant_set(
            s.A, s.X, s.W, max_iter=20, track_geometry=True, vertex_budget=500_000
        )
        assert res.termination == CONVERGED
        assert res.iterations == 7
        rows = growth_table(res)
        for row in rows[:-1]:
            assert row.halfspaces == 6 + 4 * row.k
            assert row.vertices == 8 + 8 * row.k
        assert rows[-1].halfspaces == 30
        assert rows[-1].vertices == 56
        assert rows[-1].vertices / rows[-1].halfspaces == pytest.approx(1.8667, abs=1e-4)


class TestStalling:
    def test_margin_ratio_identifies_the_stall(self):
        # slow_pair has a repeated pole at 0.999, so the shrink margin barely
        # moves: the measured ratio is 1.0 to 12 decimal places and the
        # recursion cannot finish inside any practical cap.
        s = get_system("slow_pair")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=10, track_geometry=False)
        assert res.termination == ITERATION_CAP
        ratios = [r.margin_ratio for r in growth_table(res) if r.margin_ratio is not None]
        assert ratios
        assert all(abs(r - 1.0) < 1e-9 for r in ratios)

    def test_attitude_loop_margins_decay_geometrically(self):
        s = get_system("attitude_loop")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=50, track_geometry=False)
        ratios = [
            r.margin_ratio
            for r in growth_table(res)
            if r.margin_ratio is not None and r.margin_ratio > 0.0
        ]
        assert ratios
        assert all(0.0 < r < 1.0 for r in ratios)


class TestToleranceSensitivity:
    """The documented case where a tolerance change alters the answer.

    Measured on `attitude_loop`.  The final iteration's shrink margin is
    3.6319877736e-03.  The redundancy tolerance must be strictly below that
    margin, otherwise the rows that `Pre` contributes are discarded as
    redundant and the recursion never contracts:

        redundancy_tol <= 3.0e-03  ->  converged, k = 5, 16 facets,
                                       area 0.538347284806 rad.rad/s
        redundancy_tol  = 4.0e-03  ->  iteration cap, never converges

    Bisection in `validation/validate_tolerance_sensitivity.py` puts the
    threshold between 3.631958008e-03 and 3.632019043e-03, bracketing that
    final margin.  One tolerance digit decides between an exact answer and no
    answer at all.
    """

    def test_small_tolerances_agree(self):
        s = get_system("attitude_loop")
        rows = tolerance_sweep(s.A, s.X, s.W, [0.0, 1e-12, 1e-9, 1e-6, 1e-3], max_iter=60)
        assert all(r.termination == CONVERGED for r in rows)
        assert {r.facets for r in rows} == {16}
        assert {r.iterations for r in rows} == {5}
        for r in rows:
            assert r.volume == pytest.approx(0.538347284806, rel=1e-9)
            assert abs(r.invariance_margin) <= 1e-12

    def test_four_millis_breaks_convergence(self):
        s = get_system("attitude_loop")
        rows = tolerance_sweep(s.A, s.X, s.W, [3e-3, 4e-3], max_iter=60)
        assert rows[0].termination == CONVERGED
        assert rows[0].facets == 16
        assert rows[1].termination == ITERATION_CAP
        assert rows[1].volume is None

    def test_the_threshold_brackets_the_final_shrink_margin(self):
        s = get_system("attitude_loop")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=50, track_geometry=False)
        final_margin = [r.shrink_margin for r in res.history if r.shrink_margin > 0][-1]
        assert final_margin == pytest.approx(3.6319877736e-03, rel=1e-9)
        below = maximal_robust_invariant_set(
            s.A, s.X, s.W, max_iter=60, redundancy_tol=0.99 * final_margin,
            track_geometry=False,
        )
        above = maximal_robust_invariant_set(
            s.A, s.X, s.W, max_iter=60, redundancy_tol=1.01 * final_margin,
            track_geometry=False,
        )
        assert below.termination == CONVERGED
        assert above.termination == ITERATION_CAP

    def test_a_very_loose_tolerance_returns_a_set_that_is_not_invariant(self):
        # redundancy_tol = 5e-2 with a 400-iteration cap returns a 6-facet set
        # of area 0.588122 rad.rad/s, 9.25 % larger than the true
        # 0.538347 rad.rad/s, and the independent check rejects it with a
        # facet margin of +4.158e-02.  A wrong answer, not a coarse one.
        from invariantset import verify_robust_invariance

        s = get_system("attitude_loop")
        res = maximal_robust_invariant_set(
            s.A, s.X, s.W, max_iter=400, redundancy_tol=5e-2, track_geometry=False
        )
        assert res.termination == ITERATION_CAP
        assert res.polytope.n_halfspaces == 6
        assert res.polytope.volume() == pytest.approx(0.588122297750, rel=1e-7)
        invariant, margin = verify_robust_invariance(s.A, res.polytope, s.W)
        assert not invariant
        assert margin == pytest.approx(4.1581559566e-02, rel=1e-6)
        excess = res.polytope.volume() / 0.538347284806 - 1.0
        assert excess == pytest.approx(0.0925, abs=5e-4)

    def test_convergence_tolerance_is_a_slack_on_the_guarantee(self):
        # A convergence tolerance of 5e-3 declares success at k = 4 with a
        # 14-facet set that violates (7) by +3.631988e-03 in b units.  The
        # tolerance buys an iteration and costs the guarantee.
        from invariantset import verify_robust_invariance

        s = get_system("attitude_loop")
        loose = maximal_robust_invariant_set(
            s.A, s.X, s.W, max_iter=50, convergence_tol=5e-3, track_geometry=False
        )
        assert loose.termination == CONVERGED
        assert loose.iterations == 4
        assert loose.polytope.n_halfspaces == 14
        invariant, margin = verify_robust_invariance(s.A, loose.polytope, s.W)
        assert not invariant
        assert margin == pytest.approx(3.631988e-03, rel=1e-5)

    def test_sweep_is_sorted_and_reference_relative(self):
        s = get_system("decoupled_2d")
        rows = tolerance_sweep(s.A, s.X, s.W, [1e-3, 1e-9, 1e-6], max_iter=10)
        assert [r.redundancy_tol for r in rows] == [1e-9, 1e-6, 1e-3]
        assert rows[0].volume_vs_reference == pytest.approx(1.0, abs=1e-12)
        assert all(r.volume == pytest.approx(4.0, abs=1e-9) for r in rows)


class TestGrowthTableShape:
    def test_ratios_are_none_on_the_first_row(self):
        s = get_system("nilpotent_2d")
        rows = growth_table(maximal_robust_invariant_set(s.A, s.X, s.W, track_geometry=True))
        assert rows[0].volume_ratio is None
        assert rows[0].margin_ratio is None

    def test_empty_history_gives_an_empty_table(self):
        from invariantset import EMPTY, InvariantSetResult

        assert growth_table(
            InvariantSetResult(termination=EMPTY, converged=False, iterations=0, polytope=None)
        ) == []

    def test_table_covers_every_history_record(self):
        s = get_system("damped_rotation_3d")
        res = maximal_robust_invariant_set(
            s.A, s.X, s.W, max_iter=20, track_geometry=True, vertex_budget=500_000
        )
        assert len(growth_table(res)) == len(res.history)
        assert all(np.isfinite(r.shrink_margin) for r in growth_table(res))
