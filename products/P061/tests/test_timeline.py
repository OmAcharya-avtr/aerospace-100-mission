"""Discretisation: the grid, rasterisation exactness, and window tightening."""

from __future__ import annotations

import math

import numpy as np
import pytest

from activityplan.model import (
    Activity,
    Contact,
    PowerSegment,
    Problem,
    ResourceModel,
    SolarProfile,
)
from activityplan.timeline import Grid, Plan, RasterisedProblem


def _resources(**kwargs: object) -> ResourceModel:
    defaults: dict[str, object] = {
        "battery_capacity_wh": 50.0,
        "max_depth_of_discharge": 0.4,
        "solar": SolarProfile(peak_power_w=100.0, orbit_period_s=1200.0, eclipse_fraction=0.3),
        "recorder_capacity_mbit": 500.0,
        "bus_load_w": 20.0,
    }
    defaults.update(kwargs)
    return ResourceModel(**defaults)  # type: ignore[arg-type]


def test_grid_for_horizon_known_answer() -> None:
    grid = Grid.for_horizon(1200.0, 100.0)
    assert grid.n_steps == 12
    assert grid.horizon_s == pytest.approx(1200.0, rel=1e-14)
    assert grid.midpoints_s()[0] == pytest.approx(50.0, rel=1e-14)
    assert grid.edges_s()[-1] == pytest.approx(1200.0, rel=1e-14)
    assert grid.starts_s()[3] == pytest.approx(300.0, rel=1e-14)


def test_grid_refuses_an_off_grid_horizon_rather_than_rounding() -> None:
    with pytest.raises(ValueError, match="not an integer multiple"):
        Grid.for_horizon(1250.0, 100.0)


def test_grid_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError, match="step_s"):
        Grid(step_s=0.0, n_steps=5)
    with pytest.raises(ValueError, match="n_steps"):
        Grid(step_s=1.0, n_steps=0)


def test_grid_steps_for_rounds_up_but_tolerates_rounding() -> None:
    grid = Grid(step_s=100.0, n_steps=10)
    assert grid.steps_for(100.0) == 1
    assert grid.steps_for(101.0) == 2
    assert grid.steps_for(300.0) == 3
    # 3 * 100 computed as 0.1 * 3000 is 300.00000000000006 in binary floating
    # point; the tolerance must not turn that into four steps.
    assert grid.steps_for(0.1 * 3000.0) == 3
    with pytest.raises(ValueError, match="must be positive"):
        grid.steps_for(0.0)


def test_rasterised_activity_energy_sums_to_the_exact_activity_energy() -> None:
    activity = Activity(
        name="a", power_profile=(PowerSegment(130.0, 36.0), PowerSegment(90.0, 72.0))
    )
    problem = Problem(
        activities=(activity,), resources=_resources(), horizon_s=1200.0
    )
    rasterised = RasterisedProblem.build(problem, 100.0)
    # Duration 220 s on a 100 s grid occupies three steps, the last one only
    # 20 % occupied, and the per-step energies must still sum to the exact
    # activity energy: 130*36 + 90*72 = 4680 + 6480 = 11160 J = 3.1 W*h.
    assert int(rasterised.n_occupied[0]) == 3
    assert activity.energy_wh == pytest.approx(3.1, rel=1e-13)
    assert float(np.sum(rasterised.activity_energy_wh[0])) == pytest.approx(3.1, rel=1e-12)
    assert float(np.sum(rasterised.activity_data_mbit[0])) == pytest.approx(0.0, abs=1e-15)


def test_rasterised_data_sums_to_the_declared_volume() -> None:
    activity = Activity.uniform("a", 250.0, 10.0, data_mbit=75.0)
    problem = Problem(activities=(activity,), resources=_resources(), horizon_s=1200.0)
    rasterised = RasterisedProblem.build(problem, 100.0)
    assert float(np.sum(rasterised.activity_data_mbit[0])) == pytest.approx(75.0, rel=1e-13)
    assert rasterised.activity_data_mbit[0].size == 3


def test_rasterised_solar_steps_sum_to_the_whole_horizon_closed_form() -> None:
    problem = Problem(
        activities=(Activity.uniform("a", 100.0, 10.0),),
        resources=_resources(),
        horizon_s=1200.0,
    )
    rasterised = RasterisedProblem.build(problem, 100.0)
    whole = problem.resources.solar.energy_wh(0.0, 1200.0)
    assert float(math.fsum(rasterised.solar_wh.tolist())) == pytest.approx(whole, rel=1e-12)


def test_rasterised_fixed_load_and_downlink_capacity_known_answer() -> None:
    resources = _resources(
        bus_load_w=36.0,
        contacts=(Contact(300.0, 500.0, 2.0, 36.0, "pass"),),
    )
    problem = Problem(
        activities=(Activity.uniform("a", 100.0, 10.0),), resources=resources, horizon_s=1200.0
    )
    rasterised = RasterisedProblem.build(problem, 100.0)
    # Bus: 36 W over 100 s = 1 W*h in every step.
    assert rasterised.fixed_wh[0] == pytest.approx(1.0, rel=1e-13)
    # Steps 3 and 4 are entirely inside the contact, so each carries the bus
    # 1 W*h plus the transmitter 1 W*h.
    assert rasterised.fixed_wh[3] == pytest.approx(2.0, rel=1e-13)
    assert rasterised.fixed_wh[4] == pytest.approx(2.0, rel=1e-13)
    assert rasterised.fixed_wh[5] == pytest.approx(1.0, rel=1e-13)
    # Downlink: 2 Mbit/s for 100 s = 200 Mbit per contact step, 400 Mbit total.
    assert rasterised.downlink_cap_mbit[3] == pytest.approx(200.0, rel=1e-13)
    assert float(np.sum(rasterised.downlink_cap_mbit)) == pytest.approx(400.0, rel=1e-13)


def test_allowed_starts_tighten_the_window_inward() -> None:
    # Window [250, 760] s, duration 200 s, step 100 s. The earliest grid start
    # at or after 250 s is step 3 (300 s); the latest start whose finish is at
    # or before 760 s is step 5 (500 s, finishing 700 s).
    activity = Activity.uniform(
        "a", 200.0, 10.0, earliest_start_s=250.0, latest_finish_s=760.0
    )
    problem = Problem(activities=(activity,), resources=_resources(), horizon_s=1200.0)
    rasterised = RasterisedProblem.build(problem, 100.0)
    assert rasterised.allowed_starts[0].tolist() == [3, 4, 5]
    # Continuous float 760 - 250 - 200 = 310 s; discretised float
    # (5 - 3) * 100 = 200 s; 110 s lost to snapping.
    assert rasterised.window_tightening_s()[0] == pytest.approx(110.0, rel=1e-12)


def test_allowed_starts_can_be_empty_when_the_grid_is_too_coarse() -> None:
    activity = Activity.uniform(
        "a", 150.0, 10.0, earliest_start_s=60.0, latest_finish_s=240.0
    )
    problem = Problem(activities=(activity,), resources=_resources(), horizon_s=1200.0)
    rasterised = RasterisedProblem.build(problem, 100.0)
    # Earliest grid start is step 1 (100 s), which finishes at 250 s, past the
    # declared 240 s. No grid start works even though 60 s does in continuous
    # time; that is the discretisation and it is reported, not hidden.
    assert rasterised.allowed_starts[0].size == 0


def test_finish_uses_the_continuous_duration_not_the_rounded_occupancy() -> None:
    activity = Activity.uniform("a", 220.0, 10.0)
    problem = Problem(activities=(activity,), resources=_resources(), horizon_s=1200.0)
    rasterised = RasterisedProblem.build(problem, 100.0)
    assert int(rasterised.n_occupied[0]) == 3
    assert rasterised.finish_s(0, 2) == pytest.approx(420.0, rel=1e-13)


def test_plan_operations_are_immutable_and_value_is_the_priority_sum() -> None:
    grid = Grid(step_s=100.0, n_steps=10)
    problem = Problem(
        activities=(
            Activity.uniform("a", 100.0, 10.0, priority=2.0),
            Activity.uniform("b", 100.0, 10.0, priority=3.5),
        ),
        resources=_resources(),
        horizon_s=1000.0,
    )
    empty = Plan({}, grid)
    one = empty.with_activity("a", 3)
    two = one.with_activity("b", 7)
    assert empty.n_placed == 0
    assert one.placed == ("a",)
    assert two.placed == ("a", "b")
    assert two.start_s("b") == pytest.approx(700.0, rel=1e-14)
    assert two.value(problem) == pytest.approx(5.5, rel=1e-14)
    assert two.without_activity("a").placed == ("b",)


def test_plan_rejects_a_start_step_outside_the_grid() -> None:
    grid = Grid(step_s=100.0, n_steps=10)
    with pytest.raises(ValueError, match="outside the grid"):
        Plan({"a": 10}, grid)


def test_activity_load_raises_when_a_plan_runs_past_the_grid() -> None:
    activity = Activity.uniform("a", 300.0, 10.0)
    problem = Problem(activities=(activity,), resources=_resources(), horizon_s=1200.0)
    rasterised = RasterisedProblem.build(problem, 100.0)
    with pytest.raises(ValueError, match="runs past"):
        rasterised.activity_load_wh(Plan({"a": 11}, rasterised.grid))
    with pytest.raises(ValueError, match="runs past"):
        rasterised.activity_data_production_mbit(Plan({"a": 11}, rasterised.grid))


def test_midpoint_sampling_is_second_order_on_a_smooth_profile() -> None:
    # Eclipse fraction zero, so the profile is a single smooth half sine over
    # the period with no kink inside the horizon. The midpoint rule is then
    # second order and halving the step cuts the error by about four. Checked as
    # a band, not an equality, because the ratio is only asymptotically 4.
    smooth = SolarProfile(peak_power_w=100.0, orbit_period_s=1200.0, eclipse_fraction=0.0)
    problem = Problem(
        activities=(Activity.uniform("a", 100.0, 10.0),),
        resources=_resources(solar=smooth),
        horizon_s=1200.0,
    )
    exact = smooth.energy_wh(0.0, 1200.0)
    errors = []
    for step in (100.0, 50.0, 25.0):
        rasterised = RasterisedProblem.build(problem, step)
        sampled = float(math.fsum(rasterised.solar_step_wh_midpoint().tolist()))
        errors.append(abs(sampled - exact))
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5


def test_midpoint_sampling_does_not_converge_cleanly_across_the_eclipse_kink() -> None:
    # With an eclipse the profile has a derivative discontinuity at the end of
    # the sunlit arc, and the step containing it dominates the error, so the
    # observed ratio is not 4 and can exceed 1. This is why the package
    # integrates the solar profile in closed form per step rather than sampling
    # it, and the behaviour is pinned here so nobody "fixes" the closed form
    # into a sampler.
    kinked = SolarProfile(peak_power_w=100.0, orbit_period_s=1200.0, eclipse_fraction=0.3)
    problem = Problem(
        activities=(Activity.uniform("a", 100.0, 10.0),),
        resources=_resources(solar=kinked),
        horizon_s=1200.0,
    )
    exact = kinked.energy_wh(0.0, 1200.0)
    errors = []
    for step in (100.0, 50.0):
        rasterised = RasterisedProblem.build(problem, step)
        sampled = float(math.fsum(rasterised.solar_step_wh_midpoint().tolist()))
        errors.append(abs(sampled - exact))
    # Measured in this build: 0.003487 W*h at 100 s and 0.016464 W*h at 50 s,
    # so refining the step made the sampled estimate worse. Asserted as a
    # bound rather than as the literal values.
    assert errors[1] > errors[0]
    # Meanwhile the closed-form per-step array is exact to accumulation error.
    rasterised = RasterisedProblem.build(problem, 50.0)
    assert float(math.fsum(rasterised.solar_wh.tolist())) == pytest.approx(exact, rel=1e-12)
