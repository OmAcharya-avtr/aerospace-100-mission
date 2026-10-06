"""Hand-solved instances. Every number below is derived in the comments.

The fade model is ``EmpiricalFade([-6, -4, -2, 0])``, whose survival function
takes the exact values

    P[fade >= x] = 1      for x <= -6
                 = 3/4    for -6 < x <= -4
                 = 1/2    for -4 < x <= -2
                 = 1/4    for -2 < x <=  0
                 = 0      for x >  0

so at a 10 dB margin the four MODCODs of ``monotone_table`` have

    m1 (rate 1, thr  4): level  4-10 = -6 -> A = 1.00, R*A = 1.00
    m2 (rate 2, thr  6): level  6-10 = -4 -> A = 0.75, R*A = 1.50
    m3 (rate 4, thr  8): level  8-10 = -2 -> A = 0.50, R*A = 2.00
    m4 (rate 8, thr 10): level 10-10 =  0 -> A = 0.25, R*A = 2.00

Canonical order is ascending threshold, so the indices are m1=0 ... m4=3.
"""

from __future__ import annotations

import pytest

from coderateopt import (
    InfeasibleProblem,
    ModcodSet,
    RateProblem,
    compare_to_optimum,
    highest_feasible_rate,
    select_rate,
    solve_closed_form_k2,
    solve_exhaustive,
    solve_milp,
)


def _problem(table, fade, **kwargs):
    base = {"margin_db": 10.0, "availability_target": 0.7}
    base.update(kwargs)
    return RateProblem(modcods=table, fade=fade, **base)


def test_per_interval_single_entry_known_answer(monotone_table, quarter_fade):
    # Target 0.7: allowed = {m1 (A=1.00), m2 (A=0.75)}; m3 and m4 are excluded
    # because 0.50 and 0.25 are below 0.70. Among the allowed, goodput is
    # 1.00 and 1.50, so m2 wins with expected goodput 1.5 bits/symbol and
    # achieved availability 0.75.
    problem = _problem(monotone_table, quarter_fade, availability_target=0.7)
    for method in ("milp", "exhaustive", "closed_form", "auto"):
        solution = select_rate(problem, method=method)
        assert solution.support == (1,)
        assert solution.expected_goodput == pytest.approx(1.5, abs=1e-12)
        assert solution.achieved_availability == pytest.approx(0.75, abs=1e-12)
        assert solution.worst_interval_availability == pytest.approx(0.75, abs=1e-12)


def test_per_interval_optimum_is_always_one_entry(monotone_table, quarter_fade):
    # Structural result: in per_interval mode a linear objective on the simplex
    # is maximised at a single vertex, so raising K cannot help. Checked here
    # for K = 1, 2, 3, 4 at target 0.7: the answer stays m2 at 1.5.
    for k in (1, 2, 3, 4):
        problem = _problem(monotone_table, quarter_fade, availability_target=0.7, max_entries=k)
        solution = select_rate(problem)
        assert solution.support == (1,)
        assert solution.expected_goodput == pytest.approx(1.5, abs=1e-12)


def test_tie_between_two_rates_is_reported(monotone_table, quarter_fade):
    # Target 0.25 admits all four. Goodputs are 1.0, 1.5, 2.0, 2.0, so m3 and
    # m4 are exactly tied at 2.0. The canonical choice is the lexicographically
    # smallest support, (2,) = m3, and (3,) = m4 is reported as the tie.
    problem = _problem(monotone_table, quarter_fade, availability_target=0.25)
    solution = select_rate(problem)
    assert solution.support == (2,)
    assert solution.expected_goodput == pytest.approx(2.0, abs=1e-12)
    assert solution.tied_supports == ((3,),)
    assert not solution.is_unique()


def test_long_run_two_entry_mix_known_answer_with_an_exact_tie(monotone_table, quarter_fade):
    # Target 0.6, long-run, K = 2. Singletons: m1 A=1.00 goodput 1.0;
    # m2 A=0.75 goodput 1.5; m3 and m4 fail the constraint on their own.
    # Pairs, with f the share of the *higher*-availability entry and the
    # availability constraint tight at the optimum of a linear objective:
    #   (m1, m3): 1.00 f + 0.50 (1-f) >= 0.60 -> f >= 0.2
    #             goodput 1.0 f + 2.0 (1-f) = 2.0 - f      -> f = 0.2 -> 1.80
    #   (m2, m3): 0.75 f + 0.50 (1-f) >= 0.60 -> f >= 0.4
    #             goodput 1.5 f + 2.0 (1-f) = 2.0 - 0.5 f  -> f = 0.4 -> 1.80
    #   (m2, m4): 0.75 f + 0.25 (1-f) >= 0.60 -> f >= 0.7
    #             goodput 1.5 f + 2.0 (1-f) = 2.0 - 0.5 f  -> f = 0.7 -> 1.65
    #   (m1, m4): 1.00 f + 0.25 (1-f) >= 0.60 -> f >= 7/15
    #             goodput 1.0 f + 2.0 (1-f) = 2.0 - f      -> f = 7/15 -> 1.5333
    #   (m3, m4): best availability 0.50 < 0.60, infeasible.
    # The maximum is 1.80, attained by BOTH (m1, m3) and (m2, m3). The
    # canonical support is therefore (0, 2) and (1, 2) is the reported tie.
    problem = _problem(
        monotone_table, quarter_fade, availability_target=0.6, max_entries=2, mode="long_run"
    )
    solution = select_rate(problem)
    assert solution.support == (0, 2)
    assert solution.expected_goodput == pytest.approx(1.8, abs=1e-12)
    assert solution.achieved_availability == pytest.approx(0.6, abs=1e-12)
    assert solution.worst_interval_availability == pytest.approx(0.5, abs=1e-12)
    assert solution.time_fractions[0] == pytest.approx(0.2, abs=1e-12)
    assert solution.time_fractions[2] == pytest.approx(0.8, abs=1e-12)
    assert solution.tied_supports == ((1, 2),)
    # All three solvers reach 1.80.
    for solver in (solve_milp, solve_exhaustive, solve_closed_form_k2):
        assert solver(problem).expected_goodput == pytest.approx(1.8, abs=1e-9)


def test_long_run_never_needs_more_than_two_entries(monotone_table, quarter_fade):
    # Structural result: two active constraints -> at most two non-zeros.
    for k in (2, 3, 4):
        problem = _problem(
            monotone_table, quarter_fade, availability_target=0.6, max_entries=k, mode="long_run"
        )
        solution = select_rate(problem)
        assert len(solution.support) <= 2
        assert solution.expected_goodput == pytest.approx(1.8, abs=1e-12)


def test_long_run_equals_per_interval_at_k_one(monotone_table, quarter_fade):
    # A single entry's long-run availability is its own A_m, so the two forms
    # of the constraint coincide exactly at K = 1. Target 0.7 -> m2, 1.5.
    kwargs = {"availability_target": 0.7, "max_entries": 1}
    a = select_rate(_problem(monotone_table, quarter_fade, mode="per_interval", **kwargs))
    b = select_rate(_problem(monotone_table, quarter_fade, mode="long_run", **kwargs))
    assert a.support == b.support == (1,)
    assert a.expected_goodput == pytest.approx(b.expected_goodput, abs=1e-12)


def test_modes_diverge_once_mixing_is_allowed(monotone_table, quarter_fade):
    # At target 0.6 and K = 2 the per-interval answer is m3 alone? No: m3 has
    # A = 0.50 < 0.60, so per-interval must fall back to m2 at goodput 1.5,
    # while long-run reaches 1.8 by mixing. The price of the extra 0.3
    # bits/symbol is a worst-interval availability of 0.50 instead of 0.75.
    kwargs = {"availability_target": 0.6, "max_entries": 2}
    strict = select_rate(_problem(monotone_table, quarter_fade, mode="per_interval", **kwargs))
    loose = select_rate(_problem(monotone_table, quarter_fade, mode="long_run", **kwargs))
    assert strict.expected_goodput == pytest.approx(1.5, abs=1e-12)
    assert loose.expected_goodput == pytest.approx(1.8, abs=1e-12)
    assert strict.worst_interval_availability == pytest.approx(0.75, abs=1e-12)
    assert loose.worst_interval_availability == pytest.approx(0.50, abs=1e-12)


def test_only_the_lowest_rate_satisfies_a_high_target(monotone_table, quarter_fade):
    # Target 0.9: only m1 (A = 1.00) qualifies, so the answer is the lowest
    # rate in the table at goodput 1.0 -- the degenerate case where the
    # constraint, not the objective, picks the MODCOD.
    problem = _problem(monotone_table, quarter_fade, availability_target=0.9, max_entries=3)
    solution = select_rate(problem)
    assert solution.support == (0,)
    assert solution.expected_goodput == pytest.approx(1.0, abs=1e-12)
    assert solution.is_unique()


def test_non_monotone_goodput_defeats_the_greedy_rule(non_monotone_table, quarter_fade):
    # non_monotone_table has goodputs 1.0, 1.5, 2.0, 1.5 at a 10 dB margin:
    # m4 is the highest *rate* (6.0) but only the third-best goodput, because
    # its availability is 0.25. At target 0.25 everything is allowed, so
    #   greedy "highest rate that meets the target" -> m4, goodput 6.0*0.25 = 1.5
    #   optimum                                     -> m3, goodput 4.0*0.50 = 2.0
    # relative loss = 1 - 1.5/2.0 = 0.25 exactly.
    problem = _problem(non_monotone_table, quarter_fade, availability_target=0.25)
    solution = select_rate(problem)
    assert solution.support == (2,)
    assert solution.expected_goodput == pytest.approx(2.0, abs=1e-12)
    assert highest_feasible_rate(problem) == 3
    greedy = next(c for c in compare_to_optimum(problem) if c.name == "highest_feasible_rate")
    assert greedy.heuristic_goodput == pytest.approx(1.5, abs=1e-12)
    assert greedy.relative_loss == pytest.approx(0.25, abs=1e-12)
    assert greedy.heuristic_meets_target
    assert not greedy.matches_optimum


def test_infeasible_instance_raises_with_the_diagnostic_numbers(quarter_fade):
    # Drop m1, so the best availability in the table is m2's 0.75. A target of
    # 0.9 is then unreachable and must raise, not return m2.
    table = ModcodSet.from_rows([("m2", 2.0, 6.0), ("m3", 4.0, 8.0), ("m4", 8.0, 10.0)])
    problem = _problem(table, quarter_fade, availability_target=0.9, max_entries=3)
    with pytest.raises(InfeasibleProblem) as info:
        select_rate(problem)
    error = info.value
    assert error.best_availability == pytest.approx(0.75, abs=1e-12)
    assert error.best_modcod == "m2"
    assert error.target == 0.9
    assert "unreachable" in str(error)
    assert "0.750000" in str(error)


def test_minimum_dwell_moves_then_destroys_the_mix(monotone_table, quarter_fade):
    # Unconstrained, the long-run optimum at target 0.6 is 0.2 of m1 with 0.8
    # of m3, goodput 1.8 (see the test above). Introduce a minimum dwell d and
    # the share of the higher-availability entry becomes max(required, d):
    #   (m1, m3): f >= 0.2, goodput 2.0 - 1.0 f
    #   (m2, m3): f >= 0.4, goodput 2.0 - 0.5 f
    # d = 0.25: (m1, m3) needs f = 0.25 -> 1.75; (m2, m3) keeps f = 0.4 -> 1.8.
    #           So the mix survives at 1.8 and the support moves to (m2, m3).
    tight = _problem(
        monotone_table,
        quarter_fade,
        availability_target=0.6,
        max_entries=2,
        mode="long_run",
        min_dwell_fraction=0.25,
    )
    solution = select_rate(tight)
    assert solution.support == (1, 2)
    assert solution.expected_goodput == pytest.approx(1.8, abs=1e-12)
    assert solution.time_fractions[1] == pytest.approx(0.4, abs=1e-12)

    # d = 0.45: (m1, m3) -> f = 0.45 -> 2.0 - 0.45 = 1.55
    #           (m2, m3) -> f = 0.45 -> 2.0 - 0.225 = 1.775
    # so goodput falls from 1.8 to 1.775, a loss of 0.025 bits/symbol, and the
    # mix is still worth having compared with m2 alone at 1.5.
    squeezed = _problem(
        monotone_table,
        quarter_fade,
        availability_target=0.6,
        max_entries=2,
        mode="long_run",
        min_dwell_fraction=0.45,
    )
    solution = select_rate(squeezed)
    assert solution.support == (1, 2)
    assert solution.expected_goodput == pytest.approx(1.775, abs=1e-12)
    assert solution.time_fractions[1] == pytest.approx(0.45, abs=1e-12)

    # d = 0.51: two entries cannot each hold 0.51 of unit time, so
    # effective_k collapses to floor(1/0.51) = 1 and the answer is the best
    # single entry, m2 at 1.5.
    collapsed_problem = _problem(
        monotone_table,
        quarter_fade,
        availability_target=0.6,
        max_entries=2,
        mode="long_run",
        min_dwell_fraction=0.51,
    )
    assert collapsed_problem.effective_k == 1
    collapsed = select_rate(collapsed_problem)
    assert collapsed.support == (1,)
    assert collapsed.expected_goodput == pytest.approx(1.5, abs=1e-12)
