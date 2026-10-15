"""Baseline 2: the MILP, cross-checked against exhaustive enumeration."""

from __future__ import annotations

import pytest

from activityplan.exact import build_model, enumerate_optimum, solve_exact
from activityplan.feasibility import evaluate
from activityplan.greedy import plan_greedy
from activityplan.instances import INSTANCE_ORDER, RECOMMENDED_STEP_S, named_instance
from activityplan.model import (
    Activity,
    Contact,
    Problem,
    ResourceModel,
    SolarProfile,
)
from activityplan.timeline import Plan, RasterisedProblem


def _tiny(*, require_empty_recorder: bool = False) -> RasterisedProblem:
    problem = named_instance("tiny-enumerable")
    if require_empty_recorder:
        problem = Problem(
            activities=problem.activities,
            resources=problem.resources,
            horizon_s=problem.horizon_s,
            precedences=problem.precedences,
            exclusions=problem.exclusions,
            require_empty_recorder=True,
            name=problem.name,
        )
    return RasterisedProblem.build(problem, RECOMMENDED_STEP_S["tiny-enumerable"])


@pytest.mark.parametrize("require_empty_recorder", [False, True])
def test_milp_optimum_matches_exhaustive_enumeration(require_empty_recorder: bool) -> None:
    rasterised = _tiny(require_empty_recorder=require_empty_recorder)
    _, enumerated_value, examined = enumerate_optimum(rasterised)
    result = solve_exact(rasterised)
    assert examined > 1000
    assert result.proven_optimal
    assert result.value == pytest.approx(enumerated_value, abs=1e-9)


def test_enumeration_refuses_an_instance_that_is_too_large() -> None:
    rasterised = RasterisedProblem.build(named_instance("energy-bound"), 150.0)
    with pytest.raises(ValueError, match="max_candidates"):
        enumerate_optimum(rasterised, max_candidates=1000)


def test_milp_is_at_least_as_good_as_greedy_on_every_shipped_instance() -> None:
    for name in INSTANCE_ORDER:
        rasterised = RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name])
        greedy = plan_greedy(rasterised)
        exact = solve_exact(rasterised, time_limit_s=120.0)
        assert exact.value >= greedy.value - 1e-9, name
        assert exact.proven_optimal, name


def test_milp_plans_are_feasible_under_the_simulator() -> None:
    """The dominance lemma, end to end: the MILP's shunt and downlink choices
    are free continuous variables, and the simulator's fixed policies must still
    accept the plan it returns."""
    for name in INSTANCE_ORDER:
        rasterised = RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name])
        exact = solve_exact(rasterised, time_limit_s=120.0)
        report = evaluate(rasterised, exact.plan)
        assert report.feasible, (name, report.violations)


def test_model_size_is_as_documented() -> None:
    rasterised = _tiny()
    model = build_model(rasterised)
    # Four activities on a 12-step grid. Columns: one binary per admissible
    # start plus four continuous arrays of 12 (E, S, d, g) = 48.
    assert model.n_variables == model.n_binaries + 4 * 12
    assert model.n_steps == 12
    assert model.build_s >= 0.0
    assert len(model.pairs) == model.n_binaries


def test_primal_bound_is_added_as_a_row() -> None:
    rasterised = _tiny()
    without = build_model(rasterised)
    with_bound = build_model(rasterised, primal_bound=1.0)
    assert with_bound.n_constraints == without.n_constraints + 1


def test_an_unachievable_primal_bound_makes_the_model_infeasible() -> None:
    rasterised = _tiny()
    with pytest.raises(ValueError, match="no plan achieves"):
        solve_exact(rasterised, primal_bound=1000.0)


def test_milp_refuses_a_charge_efficiency_below_one() -> None:
    problem = named_instance("tiny-enumerable")
    resources = problem.resources
    lossy = ResourceModel(
        battery_capacity_wh=resources.battery_capacity_wh,
        max_depth_of_discharge=resources.max_depth_of_discharge,
        solar=resources.solar,
        recorder_capacity_mbit=resources.recorder_capacity_mbit,
        bus_load_w=resources.bus_load_w,
        contacts=resources.contacts,
        charge_efficiency=0.9,
    )
    rasterised = RasterisedProblem.build(
        Problem(
            activities=problem.activities,
            resources=lossy,
            horizon_s=problem.horizon_s,
            exclusions=problem.exclusions,
            name="lossy",
        ),
        100.0,
    )
    with pytest.raises(ValueError, match="charge_efficiency == 1.0"):
        build_model(rasterised)


def test_milp_names_the_empty_plan_violation_when_the_declaration_is_infeasible() -> None:
    # Dark, bus 36 W over 1000 s = 10 W*h against a usable 5 W*h: the empty plan
    # already breaks the depth-of-discharge floor, so there is nothing to solve
    # and the error says which constraint and by how much.
    resources = ResourceModel(
        battery_capacity_wh=10.0,
        max_depth_of_discharge=0.5,
        solar=SolarProfile(peak_power_w=0.0, orbit_period_s=1000.0),
        recorder_capacity_mbit=100.0,
        bus_load_w=36.0,
    )
    rasterised = RasterisedProblem.build(
        Problem(
            activities=(Activity.uniform("a", 100.0, 1.0),),
            resources=resources,
            horizon_s=1000.0,
            name="broken",
        ),
        100.0,
    )
    with pytest.raises(ValueError, match="does not admit even the empty plan"):
        solve_exact(rasterised)


def test_precedence_forces_the_predecessor_to_be_placed_too() -> None:
    resources = ResourceModel(
        battery_capacity_wh=100.0,
        max_depth_of_discharge=0.9,
        solar=SolarProfile(peak_power_w=0.0, orbit_period_s=1000.0),
        recorder_capacity_mbit=1000.0,
    )
    # "after" carries all the priority, but it cannot be placed unless "before"
    # is placed and finishes first.
    problem = Problem(
        activities=(
            Activity.uniform("before", 300.0, 0.0, priority=0.5),
            Activity.uniform("after", 300.0, 0.0, priority=9.0),
        ),
        resources=resources,
        horizon_s=1000.0,
        precedences=(("before", "after"),),
        name="prec",
    )
    rasterised = RasterisedProblem.build(problem, 100.0)
    result = solve_exact(rasterised)
    assert result.value == pytest.approx(9.5, rel=1e-12)
    assert result.plan.starts["before"] * 100.0 + 300.0 <= result.plan.starts["after"] * 100.0


def test_exclusion_forbids_overlap_in_the_milp() -> None:
    resources = ResourceModel(
        battery_capacity_wh=100.0,
        max_depth_of_discharge=0.9,
        solar=SolarProfile(peak_power_w=0.0, orbit_period_s=1000.0),
        recorder_capacity_mbit=1000.0,
    )
    problem = Problem(
        activities=(
            Activity.uniform("a", 600.0, 0.0, priority=1.0),
            Activity.uniform("b", 600.0, 0.0, priority=1.0),
        ),
        resources=resources,
        horizon_s=1000.0,
        exclusions=(("a", "b"),),
        name="excl",
    )
    rasterised = RasterisedProblem.build(problem, 100.0)
    result = solve_exact(rasterised)
    # Two 600 s activities cannot both fit in a 1000 s horizon without
    # overlapping, so exactly one is placed.
    assert result.value == pytest.approx(1.0, rel=1e-12)


def test_require_empty_recorder_bounds_the_last_step_content() -> None:
    resources = ResourceModel(
        battery_capacity_wh=100.0,
        max_depth_of_discharge=0.9,
        solar=SolarProfile(peak_power_w=0.0, orbit_period_s=1000.0),
        recorder_capacity_mbit=1000.0,
        contacts=(Contact(400.0, 600.0, 0.1, 0.0, "pass"),),
    )
    activities = (Activity.uniform("a", 100.0, 0.0, data_mbit=50.0, priority=1.0),)
    drained = Problem(
        activities=activities,
        resources=resources,
        horizon_s=1000.0,
        require_empty_recorder=True,
        name="drained",
    )
    rasterised = RasterisedProblem.build(drained, 100.0)
    # The contact can clear only 20 Mbit of the 50 Mbit the activity writes, so
    # with the drain requirement the activity cannot be placed at all.
    result = solve_exact(rasterised)
    assert result.value == pytest.approx(0.0, abs=1e-12)
    assert result.plan.starts == {}
    assert evaluate(rasterised, Plan({}, rasterised.grid)).feasible
