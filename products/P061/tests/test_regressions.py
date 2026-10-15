"""Counterexamples Hypothesis found, pinned with literal inputs.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is lost
on a fresh clone. Every counterexample any property test in this repository has
produced is reproduced here with its literal inputs and a note saying what it
revealed.
"""

from __future__ import annotations

import pytest

from activityplan.feasibility import evaluate
from activityplan.greedy import plan_greedy
from activityplan.model import (
    Activity,
    PowerSegment,
    Problem,
    ResourceModel,
    SolarProfile,
)
from activityplan.timeline import Plan, RasterisedProblem


def test_greedy_returns_the_empty_plan_when_the_declaration_is_infeasible() -> None:
    """Hypothesis, 2026-10-15, against the unconditional claim that the greedy
    planner always returns a feasible plan.

    Literal falsifying inputs: ``peak_power_w = 20.0``,
    ``max_depth_of_discharge = 0.1``, ``bus_load_w = 21.0``, with the
    four-activity problem the property test builds. The array supplies far less
    than the bus draws, so over a 2400 s horizon the fixed load alone takes the
    battery through its 8 W*h usable energy. No plan is feasible, the greedy
    planner places nothing, and the report must say the declaration is
    infeasible rather than the planner having failed.
    """
    solar = SolarProfile(peak_power_w=20.0, orbit_period_s=1200.0, eclipse_fraction=0.3)
    resources = ResourceModel(
        battery_capacity_wh=80.0,
        max_depth_of_discharge=0.1,
        solar=solar,
        recorder_capacity_mbit=800.0,
        bus_load_w=21.0,
    )
    activities = tuple(
        Activity.uniform(f"a{index}", 200.0, 40.0, data_mbit=90.0, priority=1.0 + index)
        for index in range(4)
    )
    problem = Problem(
        activities=activities,
        resources=resources,
        horizon_s=2400.0,
        exclusions=(("a0", "a1"),),
        name="regression-infeasible-declaration",
    )
    rasterised = RasterisedProblem.build(problem, 100.0)
    empty = evaluate(rasterised, Plan({}, rasterised.grid))
    assert not empty.feasible
    assert empty.binding.name == "battery-depth-of-discharge"

    result = plan_greedy(rasterised)
    assert result.plan.n_placed == 0
    assert result.value == pytest.approx(0.0, abs=1e-15)
    report = evaluate(rasterised, result.plan)
    assert not report.feasible
    # Every activity was tried at every admissible start and every attempt was
    # refused for the same reason.
    assert len(result.rejections) == 4
    assert all(
        record.dominant_reason == "battery-depth-of-discharge" for record in result.rejections
    )


def test_energy_between_refuses_a_reversed_interval() -> None:
    """Hypothesis, 2026-10-15, against an additivity property whose split point
    was drawn independently of the duration.

    Literal falsifying inputs: a single segment ``PowerSegment(1.0, 0.0)`` and a
    split at ``3.0`` s, which is past the 1 s duration, so the second half of
    the additivity check asked for ``energy_between_wh(3.0, 2.0)``. The API
    refuses a reversed interval rather than returning zero, which is the
    behaviour pinned here; the property test now expresses the split as a
    fraction of the duration.
    """
    activity = Activity(name="a", power_profile=(PowerSegment(1.0, 0.0),))
    assert activity.energy_between_wh(3.0, 3.0) == pytest.approx(0.0, abs=1e-15)
    assert activity.energy_between_wh(0.0, 3.0) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError, match="must not be less than"):
        activity.energy_between_wh(3.0, 2.0)


def test_milp_and_enumeration_agree_once_downlink_volume_became_a_diagnostic() -> None:
    """The defect the exhaustive cross-check found during this build.

    An earlier draft treated the aggregate downlink volume -- total data
    produced against total declared contact capacity -- as a hard constraint.
    On this four-activity instance the MILP, which never had that constraint,
    returned 6.5 priority units while the enumeration, which evaluated it
    through the feasibility report, returned 5.0. The MILP was right: data left
    on the recorder at the horizon is not an infeasibility unless
    ``require_empty_recorder`` is declared. The constraint is now a diagnostic
    and the two agree.

    The literal instance is kept here because the disagreement was 1.5 priority
    units on an instance small enough to read by hand.
    """
    from activityplan.exact import enumerate_optimum, solve_exact
    from activityplan.model import Contact

    solar = SolarProfile(peak_power_w=90.0, orbit_period_s=1200.0, eclipse_fraction=0.35)
    resources = ResourceModel(
        battery_capacity_wh=20.0,
        max_depth_of_discharge=0.35,
        solar=solar,
        recorder_capacity_mbit=300.0,
        bus_load_w=30.0,
        contacts=(Contact(600.0, 800.0, 1.0, 8.0, "pass-1"),),
    )
    activities = (
        Activity.uniform("a", 200.0, 60.0, data_mbit=120.0, priority=3.0),
        Activity.uniform("b", 100.0, 90.0, data_mbit=80.0, priority=2.0),
        Activity.uniform("c", 200.0, 45.0, data_mbit=150.0, priority=1.5),
        Activity.uniform("d", 100.0, 70.0, data_mbit=60.0, priority=1.0),
    )
    problem = Problem(
        activities=activities,
        resources=resources,
        horizon_s=1200.0,
        exclusions=(("a", "b"),),
        name="regression-downlink-diagnostic",
    )
    rasterised = RasterisedProblem.build(problem, 100.0)
    _, enumerated, _ = enumerate_optimum(rasterised)
    exact = solve_exact(rasterised)
    assert enumerated == pytest.approx(6.5, abs=1e-9)
    assert exact.value == pytest.approx(6.5, abs=1e-9)
    report = evaluate(rasterised, exact.plan)
    # The diagnostic still records the 150 Mbit that cannot be downlinked.
    assert report.slack("downlink-volume").slack == pytest.approx(-150.0, rel=1e-12)
    assert not report.slack("downlink-volume").binding_candidate
    assert report.feasible


def test_require_empty_recorder_version_of_the_same_instance_agrees_too() -> None:
    """The same instance with ``require_empty_recorder`` set: both methods give
    5.0, which is what the earlier draft was accidentally computing."""
    from activityplan.exact import enumerate_optimum, solve_exact
    from activityplan.model import Contact

    solar = SolarProfile(peak_power_w=90.0, orbit_period_s=1200.0, eclipse_fraction=0.35)
    resources = ResourceModel(
        battery_capacity_wh=20.0,
        max_depth_of_discharge=0.35,
        solar=solar,
        recorder_capacity_mbit=300.0,
        bus_load_w=30.0,
        contacts=(Contact(600.0, 800.0, 1.0, 8.0, "pass-1"),),
    )
    activities = (
        Activity.uniform("a", 200.0, 60.0, data_mbit=120.0, priority=3.0),
        Activity.uniform("b", 100.0, 90.0, data_mbit=80.0, priority=2.0),
        Activity.uniform("c", 200.0, 45.0, data_mbit=150.0, priority=1.5),
        Activity.uniform("d", 100.0, 70.0, data_mbit=60.0, priority=1.0),
    )
    problem = Problem(
        activities=activities,
        resources=resources,
        horizon_s=1200.0,
        exclusions=(("a", "b"),),
        require_empty_recorder=True,
        name="regression-downlink-drained",
    )
    rasterised = RasterisedProblem.build(problem, 100.0)
    _, enumerated, _ = enumerate_optimum(rasterised)
    exact = solve_exact(rasterised)
    assert enumerated == pytest.approx(5.0, abs=1e-9)
    assert exact.value == pytest.approx(5.0, abs=1e-9)
