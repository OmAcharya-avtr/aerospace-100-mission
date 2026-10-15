"""Rendering. Returns strings, writes nothing, and uses no forbidden vocabulary."""

from __future__ import annotations

import pytest

from activityplan.feasibility import evaluate
from activityplan.greedy import plan_greedy
from activityplan.instances import INSTANCE_ORDER, RECOMMENDED_STEP_S, named_instance
from activityplan.report import (
    format_gap,
    render_declaration,
    render_feasibility,
    render_plan,
    render_rejections,
    render_table,
)
from activityplan.timeline import Plan, RasterisedProblem

_FORBIDDEN = (
    "flight-safe",
    "flight safe",
    "certified",
    "mission-ready",
    "mission ready",
    "production-ready",
    "production ready",
    "revolutionary",
    "cutting-edge",
    "powerful",
    "seamless",
    "game-changing",
    "!",
)


def _rasterised(name: str = "leo-imaging") -> RasterisedProblem:
    return RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name])


def test_render_table_sizes_columns_and_validates_rows() -> None:
    text = render_table(("a", "bbbb"), (("1", "2"), ("333", "4")))
    lines = text.splitlines()
    assert lines[0].startswith("a    bbbb")
    assert set(lines[1]) <= {"-", " "}
    assert lines[2].startswith("1    2")
    with pytest.raises(ValueError, match="at least one column"):
        render_table((), ())
    with pytest.raises(ValueError, match="cells but there are"):
        render_table(("a",), (("1", "2"),))


def test_render_declaration_reports_every_declared_quantity() -> None:
    rasterised = _rasterised()
    text = render_declaration(rasterised.problem, rasterised.grid.step_s)
    for fragment in (
        "instance",
        "horizon",
        "grid step",
        "battery",
        "recorder",
        "contacts",
        "energy generated",
        "W*h",
        "Mbit",
    ):
        assert fragment in text


def test_render_plan_lists_placed_and_unplaced_activities() -> None:
    rasterised = _rasterised()
    plan = plan_greedy(rasterised).plan
    text = render_plan(plan, rasterised)
    assert "placed" in text
    assert "not placed" in text
    for name in plan.placed:
        assert name in text


def test_render_plan_handles_the_empty_plan() -> None:
    rasterised = _rasterised()
    text = render_plan(Plan({}, rasterised.grid), rasterised)
    assert "placed   : 0 of" in text
    assert "not placed" in text


def test_render_feasibility_names_the_binding_constraint_first() -> None:
    rasterised = _rasterised()
    report = evaluate(rasterised, plan_greedy(rasterised).plan)
    text = render_feasibility(report, rasterised)
    assert "BINDING CONSTRAINT" in text
    assert report.binding.name in text
    assert "constraints, tightest normalised slack first" in text
    assert "diagnostics, never named as binding" in text
    assert "energy audit" in text
    assert "FEASIBLE" in text


def test_render_feasibility_marks_an_infeasible_plan() -> None:
    rasterised = _rasterised("energy-bound")
    everything = Plan(
        dict.fromkeys((activity.name for activity in rasterised.problem.activities), 0),
        rasterised.grid,
    )
    text = render_feasibility(evaluate(rasterised, everything), rasterised)
    assert "INFEASIBLE" in text
    assert "VIOLATED" in text
    assert "violations" in text


def test_render_feasibility_mentions_unplaceable_activities() -> None:
    from activityplan.model import Activity, Problem, ResourceModel, SolarProfile

    resources = ResourceModel(
        battery_capacity_wh=100.0,
        max_depth_of_discharge=0.9,
        solar=SolarProfile(peak_power_w=0.0, orbit_period_s=1000.0),
        recorder_capacity_mbit=100.0,
    )
    problem = Problem(
        activities=(
            Activity.uniform("a", 150.0, 0.0, earliest_start_s=60.0, latest_finish_s=240.0),
        ),
        resources=resources,
        horizon_s=1000.0,
        name="unplaceable",
    )
    rasterised = RasterisedProblem.build(problem, 100.0)
    text = render_feasibility(evaluate(rasterised, Plan({}, rasterised.grid)), rasterised)
    assert "unplaceable" in text


def test_render_rejections_explains_each_drop() -> None:
    rasterised = _rasterised("energy-bound")
    result = plan_greedy(rasterised)
    text = render_rejections(result)
    assert result.rejections
    for record in result.rejections:
        assert record.name in text


def test_render_rejections_handles_a_complete_plan() -> None:
    rasterised = _rasterised("window-bound")
    from activityplan.greedy import GreedyResult

    empty = GreedyResult(
        plan=Plan({}, rasterised.grid), order=(), simulations=0, rejections=(), value=0.0
    )
    assert render_rejections(empty) == "greedy planner placed every activity."


def test_format_gap_known_answers() -> None:
    assert format_gap(8.0, 10.0).startswith("20.00 %")
    assert format_gap(10.0, 10.0).startswith("0.00 %")
    assert "undefined" in format_gap(0.0, 0.0)


def test_no_forbidden_vocabulary_in_any_rendered_output() -> None:
    for name in INSTANCE_ORDER:
        rasterised = _rasterised(name)
        result = plan_greedy(rasterised)
        text = "\n".join(
            (
                render_declaration(rasterised.problem, rasterised.grid.step_s),
                render_plan(result.plan, rasterised),
                render_feasibility(evaluate(rasterised, result.plan), rasterised),
                render_rejections(result),
            )
        ).lower()
        for word in _FORBIDDEN:
            assert word not in text, (name, word)
