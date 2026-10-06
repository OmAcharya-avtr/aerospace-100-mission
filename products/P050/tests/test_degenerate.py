"""Degenerate and adversarial instances, each handled explicitly.

The four cases the design calls out:

1. infeasible -- no rate meets the availability target;
2. ties -- two or more rates attain the same optimal goodput;
3. a constraint satisfied only by the lowest rate;
4. non-monotone goodput over the rate set.

plus the ones found while building: a single-entry table, a zero-availability
MODCOD, equal thresholds, an availability target just below the best
achievable, and a dwell fraction that makes mixing impossible.
"""

from __future__ import annotations

import numpy as np
import pytest

from coderateopt import (
    EmpiricalFade,
    InfeasibleProblem,
    LognormalFade,
    ModcodSet,
    RateProblem,
    compare_to_optimum,
    illustrative_modcod_table,
    select_rate,
    solve_closed_form_k2,
    solve_exhaustive,
    solve_milp,
)

ALL_SOLVERS = (solve_milp, solve_exhaustive, solve_closed_form_k2)


def _problem(table, fade, **kwargs) -> RateProblem:
    base = {"margin_db": 10.0, "availability_target": 0.5}
    base.update(kwargs)
    return RateProblem(modcods=table, fade=fade, **base)


# --- 1. infeasible -------------------------------------------------------


def test_infeasible_never_returns_a_least_bad_answer(quarter_fade, monotone_table):
    problem = _problem(monotone_table, quarter_fade, availability_target=0.999, max_entries=2)
    # m1 has A = 1.0 exactly, so 0.999 IS reachable; push the margin down
    # until nothing reaches it, and then every path must raise.
    hard = problem.with_margin(0.0)
    for solver in ALL_SOLVERS:
        with pytest.raises(InfeasibleProblem):
            solver(hard)
    with pytest.raises(InfeasibleProblem):
        select_rate(hard)


def test_infeasible_error_names_the_margin_that_would_fix_it():
    problem = RateProblem(
        modcods=illustrative_modcod_table(),
        fade=LognormalFade(0.4),
        margin_db=3.0,
        availability_target=0.999,
    )
    with pytest.raises(InfeasibleProblem) as info:
        select_rate(problem)
    text = str(info.value)
    assert "ook-r1/4" in text
    assert "dB from the current" in text
    assert info.value.mode == "per_interval"
    # The stated required margin must actually make the instance feasible.
    quantile = problem.fade.quantile_db(problem.availability_target)
    needed = float(problem.modcods.thresholds_db[0] - quantile)
    fixed = select_rate(problem.with_margin(needed + 1e-6))
    assert fixed.support == (0,)


def test_availability_target_just_below_the_best_achievable_is_feasible(quarter_fade):
    table = ModcodSet.from_rows([("a", 1.0, 6.0), ("b", 2.0, 8.0)])
    # At margin 10 dB: a -> level -4 -> A = 0.75, b -> level -2 -> A = 0.50.
    problem = _problem(table, quarter_fade, availability_target=0.75)
    solution = select_rate(problem)
    assert solution.support == (0,)
    assert solution.achieved_availability == pytest.approx(0.75)
    # One step above and it must raise.
    with pytest.raises(InfeasibleProblem):
        select_rate(problem.with_target(0.7500001))


# --- 2. ties -------------------------------------------------------------


def test_exact_tie_is_reported_and_resolved_deterministically(quarter_fade):
    # a: rate 2, thr 6 -> A = 0.75 -> goodput 1.5
    # b: rate 3, thr 8 -> A = 0.50 -> goodput 1.5   <- exact tie
    table = ModcodSet.from_rows([("a", 2.0, 6.0), ("b", 3.0, 8.0)])
    problem = _problem(table, quarter_fade, availability_target=0.4)
    solution = select_rate(problem)
    assert solution.expected_goodput == pytest.approx(1.5)
    assert solution.support == (0,)
    assert solution.tied_supports == ((1,),)
    assert not solution.is_unique()
    # The canonical choice is the lower-threshold entry, which is also the
    # higher-availability one -- a defensible tie-break, and a documented one.
    assert solution.achieved_availability == pytest.approx(0.75)


def test_three_way_tie(quarter_fade):
    table = ModcodSet.from_rows([("a", 1.5, 4.0), ("b", 2.0, 6.0), ("c", 3.0, 8.0)])
    # A = 1.00, 0.75, 0.50 -> goodput 1.5, 1.5, 1.5
    problem = _problem(table, quarter_fade, availability_target=0.4)
    solution = select_rate(problem)
    assert solution.expected_goodput == pytest.approx(1.5)
    assert solution.support == (0,)
    assert solution.tied_supports == ((1,), (2,))


def test_equal_thresholds_are_ordered_by_rate_and_the_better_one_wins(quarter_fade):
    table = ModcodSet.from_rows([("slow", 1.0, 6.0), ("fast", 2.0, 6.0)])
    assert table.names == ("fast", "slow")
    problem = _problem(table, quarter_fade, availability_target=0.7)
    solution = select_rate(problem)
    assert solution.support_names(table) == ("fast",)
    assert solution.expected_goodput == pytest.approx(1.5)
    assert solution.is_unique()


# --- 3. only the lowest rate works ---------------------------------------


def test_constraint_satisfied_only_by_the_lowest_rate(quarter_fade, monotone_table):
    problem = _problem(monotone_table, quarter_fade, availability_target=0.9, max_entries=4)
    solution = select_rate(problem)
    assert solution.support == (0,)
    assert solution.expected_goodput == pytest.approx(1.0)
    # The greedy rule agrees here, and the comparison must say so.
    greedy = next(c for c in compare_to_optimum(problem) if c.name == "highest_feasible_rate")
    assert greedy.matches_optimum


def test_single_entry_table_is_handled(quarter_fade):
    table = ModcodSet.from_rows([("only", 1.0, 4.0)])
    problem = _problem(table, quarter_fade, availability_target=0.9, max_entries=3)
    solution = select_rate(problem)
    assert solution.support == (0,)
    assert solution.time_fractions.tolist() == [1.0]
    assert solution.is_unique()
    assert problem.effective_k == 1


# --- 4. non-monotone goodput ---------------------------------------------


def test_non_monotone_goodput_over_the_rate_set(non_monotone_table, quarter_fade):
    problem = _problem(non_monotone_table, quarter_fade, availability_target=0.2)
    goodputs = problem.goodputs()
    # 1.0, 1.5, 2.0, 1.5 -- rises then falls, so the set is not monotone.
    assert np.argmax(goodputs) == 2
    assert goodputs[3] < goodputs[2]
    solution = select_rate(problem)
    assert solution.support == (2,)
    for solver in ALL_SOLVERS:
        assert solver(problem).expected_goodput == pytest.approx(2.0, abs=1e-9)


def test_strongly_non_monotone_table_still_finds_the_global_optimum():
    # Goodput deliberately zig-zags: a mid-table entry is best.
    fade = EmpiricalFade(np.array([-9.0, -7.0, -5.0, -3.0, -1.0]))
    # Survival: 1, 4/5, 3/5, 2/5, 1/5 at levels <=-9, -9..-7, -7..-5, -5..-3, -3..-1
    table = ModcodSet.from_rows(
        [
            ("a", 1.0, 1.0),  # level -9 -> A = 1.0 -> 1.0
            ("b", 1.0, 3.0),  # level -7 -> A = 0.8 -> 0.8
            ("c", 6.0, 5.0),  # level -5 -> A = 0.6 -> 3.6
            ("d", 2.0, 7.0),  # level -3 -> A = 0.4 -> 0.8
            ("e", 8.0, 9.0),  # level -1 -> A = 0.2 -> 1.6
        ]
    )
    problem = _problem(table, fade, availability_target=0.2)
    solution = select_rate(problem)
    assert solution.support_names(table) == ("c",)
    assert solution.expected_goodput == pytest.approx(3.6, abs=1e-12)
    # The greedy rule takes "e" (rate 8, A = 0.2 >= 0.2) at goodput 1.6:
    # relative loss 1 - 1.6/3.6 = 0.5555555555555556.
    greedy = next(c for c in compare_to_optimum(problem) if c.name == "highest_feasible_rate")
    assert greedy.heuristic_goodput == pytest.approx(1.6, abs=1e-12)
    assert greedy.relative_loss == pytest.approx(1.0 - 1.6 / 3.6, abs=1e-12)


def test_zero_availability_modcod_contributes_nothing_and_is_never_chosen(quarter_fade):
    table = ModcodSet.from_rows([("useful", 1.0, 4.0), ("hopeless", 1000.0, 20.0)])
    # hopeless: level 20-10 = 10 dB -> survival 0 -> goodput 0 despite rate 1000.
    problem = _problem(table, quarter_fade, availability_target=0.3, max_entries=2)
    assert problem.availabilities()[table.index("hopeless")] == 0.0
    assert problem.goodputs()[table.index("hopeless")] == 0.0
    solution = select_rate(problem)
    assert solution.support_names(table) == ("useful",)


def test_dwell_that_forbids_every_mix_falls_back_to_a_single_entry():
    problem = RateProblem(
        modcods=illustrative_modcod_table(),
        fade=LognormalFade(0.2),
        margin_db=12.0,
        availability_target=0.99,
        max_entries=4,
        mode="long_run",
        min_dwell_fraction=0.75,
    )
    assert problem.effective_k == 1
    solution = select_rate(problem)
    assert len(solution.support) == 1
    assert solution.worst_interval_availability >= 0.99 - 1e-9
