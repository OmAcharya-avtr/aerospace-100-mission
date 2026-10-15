"""Baseline 1: the greedy planner."""

from __future__ import annotations

import pytest

from activityplan.feasibility import evaluate
from activityplan.greedy import plan_greedy, topological_order
from activityplan.instances import INSTANCE_ORDER, RECOMMENDED_STEP_S, named_instance
from activityplan.model import (
    Activity,
    Problem,
    ResourceModel,
    SolarProfile,
)
from activityplan.timeline import RasterisedProblem

_DARK = SolarProfile(peak_power_w=0.0, orbit_period_s=1000.0)


def _rasterised(
    activities: tuple[Activity, ...],
    *,
    bus_load_w: float = 0.0,
    capacity_wh: float = 100.0,
    dod: float = 0.9,
    recorder_mbit: float = 1000.0,
    precedences: tuple[tuple[str, str], ...] = (),
    exclusions: tuple[tuple[str, str], ...] = (),
    horizon_s: float = 1000.0,
    step_s: float = 100.0,
) -> RasterisedProblem:
    resources = ResourceModel(
        battery_capacity_wh=capacity_wh,
        max_depth_of_discharge=dod,
        solar=_DARK,
        recorder_capacity_mbit=recorder_mbit,
        bus_load_w=bus_load_w,
    )
    problem = Problem(
        activities=activities,
        resources=resources,
        horizon_s=horizon_s,
        precedences=precedences,
        exclusions=exclusions,
        name="unit",
    )
    return RasterisedProblem.build(problem, step_s)


def test_topological_order_respects_precedence_then_priority() -> None:
    activities = (
        Activity.uniform("low", 100.0, 0.0, priority=1.0),
        Activity.uniform("high", 100.0, 0.0, priority=9.0),
        Activity.uniform("mid", 100.0, 0.0, priority=5.0),
    )
    # With no precedence the order is simply descending priority.
    plain = _rasterised(activities)
    assert topological_order(plain) == ("high", "mid", "low")
    # "low" must precede "high". Kahn's algorithm starts from the ready set
    # {low, mid} -- "high" has an incoming edge -- and takes the highest
    # priority of those, so "mid" goes first, then "low", which releases
    # "high". The highest-priority activity is therefore considered LAST, which
    # is exactly the cost of respecting precedence in a one-pass heuristic and
    # is pinned here because it is surprising.
    constrained = _rasterised(activities, precedences=(("low", "high"),))
    assert topological_order(constrained) == ("mid", "low", "high")


def test_greedy_places_at_the_earliest_feasible_start() -> None:
    activities = (Activity.uniform("a", 200.0, 0.0, earliest_start_s=300.0),)
    rasterised = _rasterised(activities)
    result = plan_greedy(rasterised)
    assert result.plan.starts == {"a": 3}
    assert result.simulations == 1


def test_greedy_respects_an_exclusion_by_moving_the_second_activity() -> None:
    activities = (
        Activity.uniform("a", 200.0, 0.0, priority=5.0),
        Activity.uniform("b", 200.0, 0.0, priority=1.0),
    )
    rasterised = _rasterised(activities, exclusions=(("a", "b"),))
    result = plan_greedy(rasterised)
    assert result.plan.starts == {"a": 0, "b": 2}
    assert not result.rejections


def test_greedy_drops_an_activity_and_records_the_reason() -> None:
    # Dark, bus 36 W, 10 steps: the fixed load alone costs 10 W*h of a 10 W*h
    # usable battery, so no activity of any size can be added.
    activities = (Activity.uniform("a", 100.0, 36.0, priority=2.0),)
    rasterised = _rasterised(activities, bus_load_w=36.0, capacity_wh=10.0, dod=1.0)
    result = plan_greedy(rasterised)
    assert result.plan.n_placed == 0
    assert len(result.rejections) == 1
    record = result.rejections[0]
    assert record.name == "a"
    assert record.candidates_tried == 10
    assert record.dominant_reason == "battery-depth-of-discharge"
    assert sum(count for _, count in record.binding_counts) == 10


def test_greedy_reports_a_structural_rejection_with_no_candidates() -> None:
    activities = (
        Activity.uniform("a", 150.0, 0.0, earliest_start_s=60.0, latest_finish_s=240.0),
    )
    rasterised = _rasterised(activities)
    result = plan_greedy(rasterised)
    assert result.rejections[0].candidates_tried == 0
    assert "no admissible start step" in result.rejections[0].dominant_reason


def test_greedy_is_deterministic() -> None:
    rasterised = RasterisedProblem.build(named_instance("energy-bound"), 150.0)
    first = plan_greedy(rasterised)
    second = plan_greedy(rasterised)
    assert first.plan.starts == second.plan.starts
    assert first.simulations == second.simulations


@pytest.mark.parametrize("name", INSTANCE_ORDER)
def test_greedy_output_is_feasible_on_every_shipped_instance(name: str) -> None:
    rasterised = RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name])
    result = plan_greedy(rasterised)
    report = evaluate(rasterised, result.plan)
    assert report.feasible, report.violations
    assert result.value == pytest.approx(report.value, rel=1e-12)
