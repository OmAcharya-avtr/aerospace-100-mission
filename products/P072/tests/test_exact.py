"""The exact solver: model structure, backends, enumeration, bounds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from downlinkpick.contacts import ContactPlan, ContactWindow, constant_rate_profile
from downlinkpick.dataset import generate_instance
from downlinkpick.exact import (
    BACKENDS,
    ENUMERATION_LIMIT,
    build_model,
    build_pulp_problem,
    enumerate_select,
    enumeration_size,
    exact_select,
    lp_relaxation_bound,
    model_arrays,
    pulp_available,
    solve_pulp,
    solve_scipy,
    write_lp,
)
from downlinkpick.greedy import greedy_select
from downlinkpick.inventory import DataProduct, Inventory
from downlinkpick.selection import SelectionProblem, check_invariants, sizes_from_ratios


def test_known_answer_on_the_tiny_problem(tiny_problem: SelectionProblem) -> None:
    # Hand calculation. The only candidate selections are: {cal, img} in w1 for
    # 0.8 * 13 = 10.4; {cal} in w2 for 5.0; {bulk} in either window for
    # 0.8 * 20 = 16.0 or 20.0; and {bulk in w2, cal in w1} for 20 + 4 = 24.0,
    # which fits because bulk is 2.16e7 bits against w2's 2e7 -- it does NOT
    # fit. bulk alone in w2 is 2.16e7 > 2e7, so bulk is unreachable. The
    # optimum is therefore {cal, img} in w1 at 10.4.
    selection = exact_select(tiny_problem)
    assert selection.status == "optimal"
    assert selection.selected_set == {"cal", "img"}
    assert selection.total_value == pytest.approx(10.4, rel=1e-12)
    assert check_invariants(tiny_problem, selection).ok


def test_bulk_is_unreachable_because_it_exceeds_every_window(
    tiny_problem: SelectionProblem,
) -> None:
    # 3e6 bytes at ratio 0.9 is 2.16e7 bits; each window carries 2.0e7.
    assert tiny_problem.sizes_bits["bulk"] == pytest.approx(2.16e7, rel=1e-15)
    assert tiny_problem.unreachable() == ("bulk",)


def test_model_structure(tiny_problem: SelectionProblem) -> None:
    model = build_model(tiny_problem)
    assert model.pairs == (
        ("cal", "w1"),
        ("cal", "w2"),
        ("img", "w1"),
        ("bulk", "w1"),
        ("bulk", "w2"),
    )
    assert model.n_columns == 5
    # Rows: 3 no-double-downlink + 2 capacity + 1 precedence (img in w1 only).
    assert model.n_rows == 6
    assert model.products == ("cal", "img", "bulk")
    assert model.windows == ("w1", "w2")
    assert model.column_of("img", "w1") == 2
    assert model.precedence_rows == ((2, (0,)),)


def test_capacity_scale_normalises_the_row(tiny_problem: SelectionProblem) -> None:
    model = build_model(tiny_problem)
    assert model.capacity_scale(0) == pytest.approx(2.0e7, rel=1e-15)
    _objective, matrix, upper = model_arrays(model)
    capacity_row = matrix[len(model.products)]
    assert upper[len(model.products)] == pytest.approx(1.0, rel=1e-15)
    # cal is 4e6 of w1's 2e7 bits, so its coefficient is 0.2.
    assert capacity_row[0] == pytest.approx(0.2, rel=1e-12)
    assert np.max(np.abs(capacity_row)) <= 1.5


def test_capacity_scale_handles_a_zero_capacity_window() -> None:
    inventory = Inventory(
        [DataProduct("p", 1.0e3, 0.0, 1000.0, declared_value=1.0, compressibility=1.0)]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 0.0))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    model = build_model(problem)
    assert model.capacity_scale(0) == 1.0
    assert exact_select(problem).assignments == ()


def test_model_arrays_shapes(instance_problem: SelectionProblem) -> None:
    model = build_model(instance_problem)
    objective, matrix, upper = model_arrays(model)
    assert objective.shape == (model.n_columns,)
    assert matrix.shape == (model.n_rows, model.n_columns)
    assert upper.shape == (model.n_rows,)


def test_enumeration_agrees_with_the_milp() -> None:
    for seed in range(1, 25):
        instance = generate_instance(
            seed, n_products=6, n_windows=2, capacity_fraction=0.4
        )
        values = {
            name: float(instance.inventory[name].declared_value or 0.0)
            for name in instance.inventory.names
        }
        problem = SelectionProblem(
            instance.inventory,
            instance.plan,
            values,
            sizes_from_ratios(instance.inventory, instance.true_ratios),
        )
        solved = exact_select(problem)
        enumerated = enumerate_select(problem)
        # A relative tolerance on both sides: the two sums accumulate the same
        # terms in different orders and agree to within rounding, not exactly.
        assert solved.total_value == pytest.approx(enumerated.total_value, rel=1e-9)
        assert enumerated.status == "optimal"


def test_enumeration_size_and_limit(tiny_problem: SelectionProblem) -> None:
    # cal has 2 feasible windows, img 1, bulk 2: (2+1)*(1+1)*(2+1) = 18.
    assert enumeration_size(tiny_problem) == 18
    with pytest.raises(ValueError, match="above the limit"):
        enumerate_select(tiny_problem, limit=5)
    assert ENUMERATION_LIMIT == 2_000_000


def test_enumeration_respects_precedence(tiny_problem: SelectionProblem) -> None:
    selection = enumerate_select(tiny_problem)
    assert check_invariants(tiny_problem, selection).ok


def test_greedy_never_beats_the_proved_optimum() -> None:
    for seed in range(1, 31):
        instance = generate_instance(seed, n_products=14, n_windows=3, capacity_fraction=0.4)
        values = {
            name: float(instance.inventory[name].declared_value or 0.0)
            for name in instance.inventory.names
        }
        problem = SelectionProblem(
            instance.inventory,
            instance.plan,
            values,
            sizes_from_ratios(instance.inventory, instance.true_ratios),
        )
        optimum = exact_select(problem).total_value
        heuristic = greedy_select(problem).total_value
        assert heuristic <= optimum * (1.0 + 1e-9) + 1e-12


def test_lp_relaxation_is_an_upper_bound(instance_problem: SelectionProblem) -> None:
    bound = lp_relaxation_bound(instance_problem)
    assert bound >= exact_select(instance_problem).total_value * (1.0 - 1e-12)


def test_lp_relaxation_of_an_empty_model() -> None:
    inventory = Inventory(
        [DataProduct("p", 1.0e3, 0.0, 10.0, declared_value=1.0, compressibility=1.0)]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e6))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    assert build_model(problem).n_columns == 0
    assert lp_relaxation_bound(problem) == 0.0
    assert exact_select(problem).total_value == 0.0
    assert solve_scipy(problem).status == "optimal"


def test_backends_enumerate_and_scipy_agree(tiny_problem: SelectionProblem) -> None:
    by_scipy = exact_select(tiny_problem, backend="scipy")
    by_enumeration = exact_select(tiny_problem, backend="enumerate")
    assert by_scipy.total_value == pytest.approx(by_enumeration.total_value, rel=1e-9)
    assert by_scipy.method == "exact-scipy"
    assert by_enumeration.method == "enumerate"
    assert "auto" in BACKENDS


def test_unknown_backend_raises(tiny_problem: SelectionProblem) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        exact_select(tiny_problem, backend="gurobi")


def test_pulp_backend_reports_its_own_absence(tiny_problem: SelectionProblem) -> None:
    # PuLP 4.1.0 does not bundle a solver and pulp.PULP_CBC_CMD no longer
    # exists, so in this environment the PuLP path must raise an actionable
    # message naming the SciPy alternative rather than failing obscurely. If a
    # solver IS present the same call must succeed and agree with SciPy.
    if pulp_available():
        through_pulp = solve_pulp(tiny_problem)
        assert through_pulp.total_value == pytest.approx(
            exact_select(tiny_problem, backend="scipy").total_value, rel=1e-9
        )
    else:
        with pytest.raises(RuntimeError, match="no available solver"):
            solve_pulp(tiny_problem)


def test_pulp_model_has_the_expected_rows(tiny_problem: SelectionProblem) -> None:
    import pulp

    lp = build_pulp_problem(tiny_problem)
    names = {constraint.name for constraint in lp.constraints()}
    assert "once_cal" in names
    assert "capacity_w1" in names
    assert any(name.startswith("precedence_") for name in names)
    assert lp.isMIP()
    assert lp.numVariables() == build_model(tiny_problem).n_columns
    assert all(constraint.sense == pulp.LpConstraintLE for constraint in lp.constraints())


def test_pulp_and_scipy_emissions_match(tiny_problem: SelectionProblem) -> None:
    model = build_model(tiny_problem)
    objective, matrix, upper = model_arrays(model)
    lp = build_pulp_problem(tiny_problem)
    position = {f"x_{index}": index for index in range(model.n_columns)}
    lp_objective = np.zeros(model.n_columns)
    for variable, coefficient in lp.objective.items():
        lp_objective[position[variable.name]] = float(coefficient)
    assert np.allclose(lp_objective, objective, rtol=0.0, atol=0.0)
    rows = []
    rhs = []
    for constraint in lp.constraints():
        row = np.zeros(model.n_columns)
        for variable, coefficient in constraint.items():
            row[position[variable.name]] = float(coefficient)
        rows.append(row)
        rhs.append(-float(constraint.constant))
    assert np.allclose(np.sort(np.vstack(rows), axis=0), np.sort(matrix, axis=0))
    assert np.allclose(np.sort(rhs), np.sort(upper))


def test_write_lp_round_trip(tiny_problem: SelectionProblem, tmp_path: Path) -> None:
    target = tmp_path / "model.lp"
    written = write_lp(tiny_problem, target)
    assert Path(written).exists()
    text = target.read_text()
    assert "Maximize" in text
    assert "capacity_w1" in text
    assert "Binaries" in text


def test_empty_model_returns_an_empty_selection() -> None:
    inventory = Inventory(
        [DataProduct("p", 1.0e3, 0.0, 10.0, declared_value=1.0, compressibility=1.0)]
    )
    plan = ContactPlan(
        [ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 1.0e6))]
    )
    problem = SelectionProblem.from_declared(inventory, plan)
    for backend in ("scipy", "enumerate", "auto"):
        selection = exact_select(problem, backend=backend)
        assert selection.assignments == ()
        assert selection.total_value == 0.0


def test_time_limit_is_passed_through(tiny_problem: SelectionProblem) -> None:
    selection = exact_select(tiny_problem, backend="scipy", time_limit_s=30.0)
    assert selection.total_value == pytest.approx(10.4, rel=1e-12)
