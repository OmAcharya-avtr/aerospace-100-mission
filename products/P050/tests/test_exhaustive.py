"""Enumeration, the closed form, and their agreement with the MILP."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from coderateopt import (
    EmpiricalFade,
    InfeasibleProblem,
    LognormalFade,
    ModcodSet,
    RateProblem,
    count_subsets,
    enumerate_subsets,
    illustrative_modcod_table,
    restricted_lp,
    select_rate,
    solve_closed_form_k2,
    solve_exhaustive,
    solve_milp,
)


def test_count_subsets_matches_the_binomial_sum():
    # C(9,1) + C(9,2) = 9 + 36 = 45;  C(5,1)+C(5,2)+C(5,3) = 5+10+10 = 25
    assert count_subsets(9, 2) == 45
    assert count_subsets(5, 3) == 25
    assert count_subsets(4, 10) == 15  # K capped at M: 4+6+4+1
    assert count_subsets(1, 1) == 1


def test_enumerate_subsets_respects_the_allowed_mask():
    problem = RateProblem(
        modcods=illustrative_modcod_table(),
        fade=LognormalFade(0.2),
        margin_db=12.0,
        availability_target=0.99,
        max_entries=2,
        mode="per_interval",
    )
    allowed = set(np.flatnonzero(problem.allowed_mask()).tolist())
    subsets = enumerate_subsets(problem)
    assert all(set(s) <= allowed for s in subsets)
    assert len(subsets) == count_subsets(len(allowed), 2)


def test_restricted_lp_returns_none_when_the_subset_cannot_meet_the_target(
    monotone_table, quarter_fade
):
    problem = RateProblem(
        modcods=monotone_table,
        fade=quarter_fade,
        margin_db=10.0,
        availability_target=0.6,
        max_entries=2,
        mode="long_run",
    )
    # m3 (A=0.50) and m4 (A=0.25) cannot reach 0.60 in any mix.
    assert restricted_lp(problem, (2, 3)) is None
    # m1 alone (A=1.00) can.
    value, x = restricted_lp(problem, (0,))
    assert value == pytest.approx(1.0)
    assert x[0] == 1.0


def test_restricted_lp_rejects_a_subset_that_cannot_hold_the_dwell(
    monotone_table, quarter_fade
):
    problem = RateProblem(
        modcods=monotone_table,
        fade=quarter_fade,
        margin_db=10.0,
        availability_target=0.3,
        max_entries=3,
        mode="long_run",
        min_dwell_fraction=0.4,
    )
    assert restricted_lp(problem, (0, 1, 2)) is None  # 3 * 0.4 = 1.2 > 1


def test_closed_form_refuses_above_two_entries():
    problem = RateProblem(
        modcods=illustrative_modcod_table(),
        fade=LognormalFade(0.2),
        margin_db=12.0,
        availability_target=0.9,
        max_entries=3,
        mode="long_run",
    )
    with pytest.raises(ValueError, match="max_entries <= 2"):
        solve_closed_form_k2(problem)


def test_all_paths_refuse_an_infeasible_instance():
    problem = RateProblem(
        modcods=illustrative_modcod_table(),
        fade=LognormalFade(0.3),
        margin_db=-5.0,
        availability_target=0.99,
        max_entries=2,
        mode="long_run",
    )
    for solver in (solve_milp, solve_exhaustive, solve_closed_form_k2):
        with pytest.raises(InfeasibleProblem):
            solver(problem)


def _random_problem(data) -> RateProblem:
    n = data.draw(st.integers(min_value=2, max_value=6), label="n_modcods")
    rates = data.draw(
        st.lists(
            st.floats(min_value=0.1, max_value=8.0, allow_nan=False),
            min_size=n,
            max_size=n,
        ),
        label="rates",
    )
    thresholds = data.draw(
        st.lists(
            st.floats(min_value=-5.0, max_value=25.0, allow_nan=False),
            min_size=n,
            max_size=n,
            unique=True,
        ),
        label="thresholds",
    )
    table = ModcodSet.from_rows(
        [(f"m{i}", r, t) for i, (r, t) in enumerate(zip(rates, thresholds, strict=True))]
    )
    use_empirical = data.draw(st.booleans(), label="empirical")
    if use_empirical:
        count = data.draw(st.integers(min_value=4, max_value=40), label="samples")
        samples = data.draw(
            st.lists(
                st.floats(min_value=-20.0, max_value=5.0, allow_nan=False),
                min_size=count,
                max_size=count,
            ),
            label="sample_values",
        )
        fade = EmpiricalFade(np.asarray(samples))
    else:
        fade = LognormalFade(
            data.draw(st.floats(min_value=0.01, max_value=1.5), label="scintillation")
        )
    return RateProblem(
        modcods=table,
        fade=fade,
        margin_db=data.draw(st.floats(min_value=0.0, max_value=25.0), label="margin"),
        availability_target=data.draw(
            st.floats(min_value=0.05, max_value=0.995), label="target"
        ),
        max_entries=data.draw(st.integers(min_value=1, max_value=3), label="k"),
        mode=data.draw(st.sampled_from(["per_interval", "long_run"]), label="mode"),
        min_dwell_fraction=data.draw(
            st.sampled_from([0.0, 0.0, 0.05, 0.2, 0.4]), label="dwell"
        ),
    )


@settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(st.data())
def test_milp_and_enumeration_agree_on_random_instances(data):
    problem = _random_problem(data)
    try:
        by_milp = solve_milp(problem)
    except InfeasibleProblem:
        with pytest.raises(InfeasibleProblem):
            solve_exhaustive(problem)
        return
    by_enumeration = solve_exhaustive(problem)
    assert by_milp.expected_goodput == pytest.approx(
        by_enumeration.expected_goodput, rel=1e-7, abs=1e-9
    )


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(st.data())
def test_closed_form_agrees_wherever_it_applies(data):
    problem = _random_problem(data)
    if problem.effective_k > 2:
        return
    try:
        enumerated = solve_exhaustive(problem)
    except InfeasibleProblem:
        with pytest.raises(InfeasibleProblem):
            solve_closed_form_k2(problem)
        return
    closed = solve_closed_form_k2(problem)
    assert closed.expected_goodput == pytest.approx(
        enumerated.expected_goodput, rel=1e-9, abs=1e-12
    )


@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(st.data())
def test_solution_invariants_hold_on_random_instances(data):
    problem = _random_problem(data)
    try:
        solution = select_rate(problem)
    except InfeasibleProblem:
        return
    x = solution.time_fractions
    assert x.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(x >= -1e-12)
    assert len(solution.support) <= problem.effective_k
    assert all(x[i] >= problem.min_dwell_fraction - 1e-9 for i in solution.support)
    assert solution.achieved_availability >= problem.availability_target - 1e-9
    if problem.mode == "per_interval":
        # Structural result: one entry is always enough.
        assert len(solution.support) == 1
        assert solution.worst_interval_availability >= problem.availability_target - 1e-9
    if problem.mode == "long_run" and problem.min_dwell_fraction == 0.0:
        # Structural result: two active constraints, at most two non-zeros.
        assert len(solution.support) <= 2
