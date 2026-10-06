"""The rules of thumb, and the gap between them and the optimum."""

from __future__ import annotations

import math

import numpy as np
import pytest

from coderateopt import (
    InfeasibleProblem,
    LognormalFade,
    RateProblem,
    compare_to_optimum,
    highest_feasible_rate,
    highest_rate_within_reserve,
    illustrative_modcod_table,
    select_rate,
)


def _problem(**overrides) -> RateProblem:
    kwargs = {
        "modcods": illustrative_modcod_table(),
        "fade": LognormalFade(0.2),
        "margin_db": 12.0,
        "availability_target": 0.99,
    }
    kwargs.update(overrides)
    return RateProblem(**kwargs)


def test_highest_feasible_rate_picks_the_top_rate_that_closes():
    problem = _problem()
    avail = problem.availabilities()
    index = highest_feasible_rate(problem)
    assert avail[index] >= problem.availability_target
    rates = problem.modcods.rates
    better = np.flatnonzero(
        (rates > rates[index]) & (avail >= problem.availability_target)
    )
    assert better.size == 0


def test_highest_feasible_rate_raises_when_nothing_closes():
    problem = _problem(margin_db=0.0, availability_target=0.9999)
    with pytest.raises(InfeasibleProblem, match="no MODCOD meets"):
        highest_feasible_rate(problem)


def test_highest_rate_within_reserve_uses_only_the_margin():
    # Margin 12 dB less a 3 dB reserve leaves 9 dB, so the fastest MODCOD with
    # a threshold at or below 9 dB is ook-r9/10 at 8.6 dB.
    problem = _problem()
    index = highest_rate_within_reserve(problem, 3.0)
    assert problem.modcods.names[index] == "ook-r9/10"
    assert problem.modcods.thresholds_db[index] <= 9.0 + 1e-12


def test_highest_rate_within_reserve_raises_when_nothing_fits():
    problem = _problem(margin_db=1.0)
    with pytest.raises(InfeasibleProblem, match="no MODCOD threshold fits"):
        highest_rate_within_reserve(problem, 0.0)


def test_comparison_reports_a_constraint_violation_as_a_negative_loss():
    # ook-r9/10 (A = 0.9474 at 12 dB, SI 0.2) beats the constrained optimum on
    # raw goodput precisely because it misses the 0.99 target. A negative
    # relative_loss must therefore come with heuristic_meets_target False.
    problem = _problem()
    rows = {row.name: row for row in compare_to_optimum(problem, reserve_db=3.0)}
    reserve = rows["highest_rate_within_reserve"]
    assert reserve.relative_loss < 0.0
    assert not reserve.heuristic_meets_target
    assert not reserve.matches_optimum


def test_comparison_reports_nan_when_a_heuristic_refuses():
    problem = _problem(margin_db=6.0, availability_target=0.9)
    rows = {row.name: row for row in compare_to_optimum(problem, reserve_db=20.0)}
    reserve = rows["highest_rate_within_reserve"]
    assert math.isnan(reserve.heuristic_goodput)
    assert math.isnan(reserve.relative_loss)
    assert not reserve.heuristic_meets_target


def test_greedy_can_match_the_optimum_and_says_so():
    # At a high target the constraint leaves one candidate, so the greedy rule
    # and the optimum must coincide. Reporting that is as important as
    # reporting a gap.
    problem = _problem(availability_target=0.9995)
    rows = {row.name: row for row in compare_to_optimum(problem)}
    greedy = rows["highest_feasible_rate"]
    assert greedy.matches_optimum
    assert greedy.relative_loss == pytest.approx(0.0, abs=1e-12)


def test_optimum_is_never_beaten_by_the_constraint_respecting_heuristic():
    problem = _problem(max_entries=2, mode="long_run")
    for target in (0.5, 0.8, 0.9, 0.95, 0.99):
        instance = problem.with_target(target)
        optimum = select_rate(instance).expected_goodput
        rows = {row.name: row for row in compare_to_optimum(instance)}
        greedy = rows["highest_feasible_rate"]
        assert greedy.heuristic_goodput <= optimum + 1e-9
        assert greedy.relative_loss >= -1e-9
        assert greedy.optimal_goodput == pytest.approx(optimum)
