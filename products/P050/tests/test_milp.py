"""The MILP encoding itself: rows, bounds, options, failure modes."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, milp

from coderateopt import (
    InfeasibleProblem,
    LognormalFade,
    RateProblem,
    assert_feasible,
    build_milp,
    illustrative_modcod_table,
    solve_exhaustive,
    solve_milp,
)
from coderateopt.milp import MILP_OPTIONS


def _problem(**overrides) -> RateProblem:
    kwargs = {
        "modcods": illustrative_modcod_table(),
        "fade": LognormalFade(0.2),
        "margin_db": 12.0,
        "availability_target": 0.99,
        "max_entries": 2,
        "mode": "long_run",
    }
    kwargs.update(overrides)
    return RateProblem(**kwargs)


def test_encoding_shape_and_rows():
    problem = _problem()
    enc = build_milp(problem)
    m = problem.n_modcods
    # 1 simplex row + m linking rows + 1 cardinality row + 1 availability row
    assert enc.constraint_matrix.shape == (m + 3, 2 * m)
    assert enc.c.shape == (2 * m,)
    assert np.all(enc.c[m:] == 0.0)
    assert np.allclose(enc.c[:m], -problem.goodputs())
    assert enc.integrality.tolist() == [0.0] * m + [1.0] * m
    assert enc.row_labels[0] == "time_fractions_sum_to_one"
    assert enc.row_labels[-1] == "long_run_availability"
    assert "table_cardinality" in enc.row_labels


def test_per_interval_mode_drops_the_availability_row_and_fixes_bounds():
    problem = _problem(mode="per_interval")
    enc = build_milp(problem)
    assert "long_run_availability" not in enc.row_labels
    allowed = problem.allowed_mask()
    assert np.array_equal(enc.variable_upper[: problem.n_modcods], allowed.astype(float))
    assert np.array_equal(enc.variable_upper[problem.n_modcods :], allowed.astype(float))
    # Something must be excluded for this instance to be testing anything.
    assert not allowed.all()


def test_minimum_dwell_adds_one_row_per_modcod():
    problem = _problem(min_dwell_fraction=0.1)
    enc = build_milp(problem)
    dwell_rows = [label for label in enc.row_labels if label.startswith("min_dwell[")]
    assert len(dwell_rows) == problem.n_modcods
    row = enc.constraint_matrix[enc.row_labels.index("min_dwell[ook-r1/4]")]
    assert row[0] == 1.0
    assert row[problem.n_modcods] == pytest.approx(-0.1)


def test_cardinality_row_uses_effective_k():
    problem = _problem(max_entries=50)
    enc = build_milp(problem)
    assert enc.upper[enc.row_labels.index("table_cardinality")] == float(problem.n_modcods)


def test_solution_satisfies_every_constraint_it_was_given():
    problem = _problem(max_entries=3, min_dwell_fraction=0.05)
    solution = solve_milp(problem)
    x = solution.time_fractions
    assert x.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(x >= -1e-12)
    assert len(solution.support) <= problem.effective_k
    assert all(x[i] >= 0.05 - 1e-9 for i in solution.support)
    assert solution.achieved_availability >= problem.availability_target - 1e-9


def test_infeasible_is_detected_without_a_solver_call():
    problem = _problem(margin_db=0.0, availability_target=0.999999)
    with pytest.raises(InfeasibleProblem) as info:
        assert_feasible(problem)
    assert info.value.target == 0.999999
    assert "Raise the link margin" in str(info.value)
    with pytest.raises(InfeasibleProblem):
        solve_milp(problem)


def test_milp_result_is_not_canonical_but_exhaustive_is():
    problem = _problem()
    assert solve_milp(problem).canonical is False
    assert solve_exhaustive(problem).canonical is True
    assert solve_milp(problem).tied_supports == ()


def test_default_mip_rel_gap_returns_the_wrong_decision():
    """The documented instance behind MILP_OPTIONS, re-measured here.

    This is a regression test on scipy/HiGHS behaviour, not on this package.
    If a future HiGHS tightens its default gap the assertion on the *default*
    arm may stop holding; the assertion that matters -- that the package's
    own options reproduce the enumerated optimum -- is the second one.
    """
    problem = _problem(fade=LognormalFade(0.5281008782676444))
    enc = build_milp(problem)
    common = {
        "c": enc.c,
        "constraints": LinearConstraint(enc.constraint_matrix, enc.lower, enc.upper),
        "integrality": enc.integrality,
        "bounds": Bounds(enc.variable_lower, enc.variable_upper),
    }
    default_arm = milp(**common)
    tight_arm = milp(**common, options=dict(MILP_OPTIONS))
    enumerated = solve_exhaustive(problem).expected_goodput
    assert MILP_OPTIONS["mip_rel_gap"] == 0.0
    assert -float(tight_arm.fun) == pytest.approx(enumerated, abs=1e-9)
    assert -float(default_arm.fun) <= -float(tight_arm.fun) + 1e-12
    # The package path must match enumeration whatever the default does.
    assert solve_milp(problem).expected_goodput == pytest.approx(enumerated, abs=1e-9)


def test_goodput_is_non_increasing_in_the_availability_target():
    problem = _problem()
    previous = np.inf
    for target in (0.5, 0.8, 0.9, 0.95, 0.99, 0.995):
        value = solve_milp(problem.with_target(target)).expected_goodput
        assert value <= previous + 1e-9
        previous = value


def test_goodput_is_non_decreasing_in_the_margin():
    problem = _problem()
    previous = -np.inf
    for margin in (8.0, 10.0, 12.0, 14.0, 20.0):
        value = solve_milp(problem.with_margin(margin)).expected_goodput
        assert value >= previous - 1e-9
        previous = value


def test_goodput_is_non_decreasing_in_the_cardinality_limit():
    previous = -np.inf
    for k in (1, 2, 3, 4):
        value = solve_milp(_problem(max_entries=k)).expected_goodput
        assert value >= previous - 1e-9
        previous = value


def test_raw_highs_allocation_can_violate_the_availability_row():
    """The documented instance behind the polish step in solve_milp.

    HiGHS accepts an allocation that misses the availability target by about
    1.7e-8, inside its primal feasibility tolerance, and the goodput that
    buys is about 5e-7 relative. ``solve_milp`` re-solves the continuous part
    on the chosen support, so its own output must land exactly on the
    constraint. As with the MIP-gap test, the assertion that matters is the
    one about this package; the assertion about raw HiGHS may change with a
    future scipy.
    """
    problem = _problem(fade=LognormalFade(0.05), margin_db=15.5, availability_target=0.99)
    enc = build_milp(problem)
    raw = milp(
        c=enc.c,
        constraints=LinearConstraint(enc.constraint_matrix, enc.lower, enc.upper),
        integrality=enc.integrality,
        bounds=Bounds(enc.variable_lower, enc.variable_upper),
        options=dict(MILP_OPTIONS),
    )
    avail = problem.availabilities()
    raw_x = np.asarray(raw.x[: problem.n_modcods])
    raw_availability = float(avail @ raw_x)
    polished = solve_milp(problem)
    enumerated = solve_exhaustive(problem)
    # Whatever the raw solver does, the package's answer is exactly feasible
    # and matches enumeration to floating-point noise.
    assert polished.achieved_availability >= problem.availability_target - 1e-15
    assert polished.expected_goodput == pytest.approx(
        enumerated.expected_goodput, abs=1e-12
    )
    # And the raw allocation is never better than the true optimum by more
    # than its own feasibility slack allows.
    if raw_availability < problem.availability_target:
        assert -float(raw.fun) >= enumerated.expected_goodput - 1e-12


def test_polished_solution_is_exactly_feasible_on_a_grid():
    for margin in np.arange(9.0, 20.01, 0.5):
        for scintillation in (0.05, 0.1, 0.2, 0.4):
            problem = _problem(
                fade=LognormalFade(scintillation), margin_db=float(margin)
            )
            try:
                solution = solve_milp(problem)
            except InfeasibleProblem:
                continue
            assert solution.achieved_availability >= problem.availability_target - 1e-12
            assert solution.time_fractions.sum() == pytest.approx(1.0, abs=1e-12)
