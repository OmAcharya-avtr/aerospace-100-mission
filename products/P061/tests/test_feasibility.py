"""The feasibility report: slacks, the binding constraint, and the tie-break."""

from __future__ import annotations

import pytest

from activityplan.feasibility import RESOURCE_CONSTRAINTS, ConstraintSlack, evaluate
from activityplan.model import (
    Activity,
    Contact,
    Problem,
    ResourceModel,
    SolarProfile,
)
from activityplan.timeline import Plan, RasterisedProblem

_DARK = SolarProfile(peak_power_w=0.0, orbit_period_s=1000.0)


def _rasterised(
    *,
    activities: tuple[Activity, ...],
    bus_load_w: float = 36.0,
    capacity_wh: float = 10.0,
    dod: float = 0.5,
    recorder_mbit: float = 100.0,
    contacts: tuple[Contact, ...] = (),
    precedences: tuple[tuple[str, str], ...] = (),
    exclusions: tuple[tuple[str, str], ...] = (),
    require_empty_recorder: bool = False,
    horizon_s: float = 1000.0,
    step_s: float = 100.0,
) -> RasterisedProblem:
    resources = ResourceModel(
        battery_capacity_wh=capacity_wh,
        max_depth_of_discharge=dod,
        solar=_DARK,
        recorder_capacity_mbit=recorder_mbit,
        bus_load_w=bus_load_w,
        contacts=contacts,
    )
    problem = Problem(
        activities=activities,
        resources=resources,
        horizon_s=horizon_s,
        precedences=precedences,
        exclusions=exclusions,
        require_empty_recorder=require_empty_recorder,
        name="unit",
    )
    return RasterisedProblem.build(problem, step_s)


def test_constraint_slack_normalisation_and_tolerance() -> None:
    entry = ConstraintSlack(name="x", slack=-2.0, unit="W*h", scale=4.0, where="here")
    assert entry.normalised == pytest.approx(-0.5, rel=1e-14)
    assert not entry.satisfied
    # A violation of 1e-12 of the scale is inside the declared 1e-9 tolerance.
    tiny = ConstraintSlack(name="x", slack=-4.0e-12, unit="W*h", scale=4.0, where="here")
    assert tiny.satisfied


def test_battery_slack_is_hand_calculable() -> None:
    # Dark, bus 36 W, 100 s steps, 5 steps: 5 W*h consumed from a 10 W*h
    # battery, so the minimum energy is 5 W*h against a floor of 5 W*h
    # (capacity 10, DoD 0.5). Slack 0, normalised 0 against a usable 5 W*h.
    rasterised = _rasterised(
        activities=(Activity.uniform("a", 100.0, 36.0),), horizon_s=500.0
    )
    report = evaluate(rasterised, Plan({}, rasterised.grid))
    battery = report.slack("battery-depth-of-discharge")
    assert battery.slack == pytest.approx(0.0, abs=1e-12)
    assert battery.scale == pytest.approx(5.0, rel=1e-14)
    assert report.feasible


def test_battery_violation_is_reported_as_a_violation() -> None:
    rasterised = _rasterised(
        activities=(Activity.uniform("a", 100.0, 36.0),), horizon_s=800.0
    )
    report = evaluate(rasterised, Plan({}, rasterised.grid))
    assert not report.feasible
    assert report.binding.name == "battery-depth-of-discharge"
    # 8 W*h consumed, minimum 2 W*h, floor 5 W*h, so the slack is -3 W*h.
    assert report.binding.slack == pytest.approx(-3.0, rel=1e-12)
    assert report.violations[0].name == "battery-depth-of-discharge"


def test_recorder_slack_is_hand_calculable() -> None:
    # 60 Mbit produced into a 100 Mbit recorder with no contact: slack 40 Mbit,
    # normalised 0.4.
    rasterised = _rasterised(
        activities=(Activity.uniform("a", 200.0, 0.0, data_mbit=60.0),),
        bus_load_w=0.0,
        capacity_wh=100.0,
    )
    report = evaluate(rasterised, Plan({"a": 0}, rasterised.grid))
    recorder = report.slack("recorder-capacity")
    assert recorder.slack == pytest.approx(40.0, rel=1e-13)
    assert recorder.normalised == pytest.approx(0.4, rel=1e-13)


def test_downlink_volume_is_a_diagnostic_not_a_constraint() -> None:
    # 400 Mbit produced against 100 Mbit of contact capacity: a deficit, but not
    # an infeasibility, because the data simply stays on the recorder.
    rasterised = _rasterised(
        activities=(Activity.uniform("a", 200.0, 0.0, data_mbit=400.0),),
        bus_load_w=0.0,
        capacity_wh=100.0,
        recorder_mbit=1000.0,
        contacts=(Contact(300.0, 500.0, 0.5, 0.0, "pass"),),
    )
    report = evaluate(rasterised, Plan({"a": 0}, rasterised.grid))
    volume = report.slack("downlink-volume")
    assert not volume.binding_candidate
    assert volume.slack == pytest.approx(-300.0, rel=1e-13)
    assert report.feasible
    assert volume not in report.candidates


def test_require_empty_recorder_turns_the_residual_into_a_constraint() -> None:
    rasterised = _rasterised(
        activities=(Activity.uniform("a", 200.0, 0.0, data_mbit=400.0),),
        bus_load_w=0.0,
        capacity_wh=100.0,
        recorder_mbit=1000.0,
        contacts=(Contact(300.0, 500.0, 0.5, 0.0, "pass"),),
        require_empty_recorder=True,
    )
    report = evaluate(rasterised, Plan({"a": 0}, rasterised.grid))
    drained = report.slack("recorder-drained")
    assert drained.binding_candidate
    assert drained.slack == pytest.approx(-300.0, rel=1e-13)
    assert not report.feasible


def test_require_empty_recorder_slack_is_exactly_zero_when_satisfied() -> None:
    rasterised = _rasterised(
        activities=(Activity.uniform("a", 200.0, 0.0, data_mbit=40.0),),
        bus_load_w=0.0,
        capacity_wh=100.0,
        contacts=(Contact(300.0, 900.0, 0.5, 0.0, "pass"),),
        require_empty_recorder=True,
    )
    report = evaluate(rasterised, Plan({"a": 0}, rasterised.grid))
    drained = report.slack("recorder-drained")
    # Zero, not negative zero: a sign bit on the headline slack of a satisfied
    # constraint reads as a violation to anyone skimming the report.
    assert drained.slack == 0.0
    assert str(drained.slack) == "0.0"
    assert report.feasible


def test_window_precedence_and_exclusion_slacks_are_hand_calculable() -> None:
    activities = (
        Activity.uniform("a", 200.0, 0.0, earliest_start_s=0.0, latest_finish_s=600.0),
        Activity.uniform("b", 100.0, 0.0, earliest_start_s=0.0, latest_finish_s=900.0),
    )
    rasterised = _rasterised(
        activities=activities,
        bus_load_w=0.0,
        capacity_wh=100.0,
        precedences=(("a", "b"),),
        exclusions=(("a", "b"),),
    )
    # a at step 1 runs [100, 300] s, b at step 5 runs [500, 600] s.
    report = evaluate(rasterised, Plan({"a": 1, "b": 5}, rasterised.grid))
    # window:a -- latest 600, finish 300, slack 300; float 600-0-200 = 400.
    assert report.slack("window:a").slack == pytest.approx(300.0, rel=1e-13)
    assert report.slack("window:a").normalised == pytest.approx(0.75, rel=1e-13)
    # precedence a->b -- b starts 500, a finishes 300, slack 200; scale is a's
    # duration, 200 s, so normalised 1.0.
    assert report.slack("precedence:a->b").slack == pytest.approx(200.0, rel=1e-13)
    assert report.slack("precedence:a->b").normalised == pytest.approx(1.0, rel=1e-13)
    # exclusion -- separation 500 - 300 = 200 s; scale min(200, 100) = 100.
    assert report.slack("exclusion:a|b").slack == pytest.approx(200.0, rel=1e-13)
    assert report.slack("exclusion:a|b").normalised == pytest.approx(2.0, rel=1e-13)
    assert report.feasible


def test_precedence_with_an_unplaced_predecessor_is_a_violation() -> None:
    activities = (
        Activity.uniform("a", 200.0, 0.0),
        Activity.uniform("b", 100.0, 0.0),
    )
    rasterised = _rasterised(
        activities=activities, bus_load_w=0.0, capacity_wh=100.0, precedences=(("a", "b"),)
    )
    report = evaluate(rasterised, Plan({"b": 5}, rasterised.grid))
    assert not report.feasible
    assert report.slack("precedence:a->b").slack == pytest.approx(-1000.0, rel=1e-13)
    assert "predecessor" in report.slack("precedence:a->b").where


def test_overlapping_excluded_pair_has_negative_separation() -> None:
    activities = (
        Activity.uniform("a", 300.0, 0.0),
        Activity.uniform("b", 300.0, 0.0),
    )
    rasterised = _rasterised(
        activities=activities, bus_load_w=0.0, capacity_wh=100.0, exclusions=(("a", "b"),)
    )
    report = evaluate(rasterised, Plan({"a": 0, "b": 1}, rasterised.grid))
    # a runs [0, 300], b runs [100, 400]; separation max(100-300, 0-400) = -200.
    assert report.slack("exclusion:a|b").slack == pytest.approx(-200.0, rel=1e-13)
    assert not report.feasible


def test_binding_constraint_breaks_ties_by_declared_evaluation_order() -> None:
    # The battery and the recorder are both exactly at their limits, so both
    # have a normalised slack of zero and the declared order -- battery first --
    # decides. Asserted because a silent change of order would move a headline.
    activities = (Activity.uniform("a", 200.0, 0.0, data_mbit=100.0),)
    rasterised = _rasterised(
        activities=activities, bus_load_w=36.0, capacity_wh=10.0, dod=0.5, horizon_s=500.0
    )
    report = evaluate(rasterised, Plan({"a": 0}, rasterised.grid))
    assert report.slack("battery-depth-of-discharge").normalised == pytest.approx(
        0.0, abs=1e-12
    )
    assert report.slack("recorder-capacity").normalised == pytest.approx(0.0, abs=1e-12)
    assert report.binding.name == "battery-depth-of-discharge"


def test_binding_resource_skips_time_constraints() -> None:
    activities = (Activity.uniform("a", 200.0, 0.0, data_mbit=10.0),)
    rasterised = _rasterised(activities=activities, bus_load_w=0.0, capacity_wh=100.0)
    report = evaluate(rasterised, Plan({"a": 8}, rasterised.grid))
    assert report.binding.name == "horizon"
    resource = report.binding_resource
    assert resource is not None
    assert resource.name in RESOURCE_CONSTRAINTS


def test_unplaceable_activities_are_reported_separately() -> None:
    activities = (
        Activity.uniform("a", 150.0, 0.0, earliest_start_s=60.0, latest_finish_s=240.0),
        Activity.uniform("b", 100.0, 0.0),
    )
    rasterised = _rasterised(activities=activities, bus_load_w=0.0, capacity_wh=100.0)
    report = evaluate(rasterised, Plan({"b": 0}, rasterised.grid))
    assert report.unplaceable == ("a",)
    assert report.unplaced == ("a",)


def test_report_accounting_of_value_and_forgone_priority() -> None:
    activities = (
        Activity.uniform("a", 100.0, 0.0, priority=2.0),
        Activity.uniform("b", 100.0, 0.0, priority=3.0),
    )
    rasterised = _rasterised(activities=activities, bus_load_w=0.0, capacity_wh=100.0)
    report = evaluate(rasterised, Plan({"a": 0}, rasterised.grid))
    assert report.value == pytest.approx(2.0, rel=1e-14)
    assert report.forgone_priority == pytest.approx(3.0, rel=1e-14)
    assert report.placed == ("a",)
    assert report.unplaced == ("b",)


def test_energy_audit_closes_and_matches_the_closed_form() -> None:
    rasterised = _rasterised(
        activities=(Activity.uniform("a", 200.0, 20.0, data_mbit=5.0),),
        bus_load_w=25.0,
        capacity_wh=40.0,
        contacts=(Contact(300.0, 500.0, 1.0, 10.0, "pass"),),
    )
    report = evaluate(rasterised, Plan({"a": 1}, rasterised.grid))
    audit = report.energy_audit()
    assert audit["residual"] == pytest.approx(0.0, abs=1e-9)
    assert audit["generated_steps"] == pytest.approx(audit["generated_closed_form"], abs=1e-9)
    assert audit["fixed_steps"] == pytest.approx(audit["fixed_closed_form"], rel=1e-10)


def test_slack_lookup_raises_for_an_unknown_constraint() -> None:
    rasterised = _rasterised(
        activities=(Activity.uniform("a", 100.0, 0.0),), bus_load_w=0.0, capacity_wh=100.0
    )
    report = evaluate(rasterised, Plan({}, rasterised.grid))
    with pytest.raises(KeyError, match="no constraint named"):
        report.slack("not-a-constraint")
