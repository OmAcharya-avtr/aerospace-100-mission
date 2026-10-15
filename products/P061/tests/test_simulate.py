"""Resource simulation: hand-calculated recursions and the dominance lemma."""

from __future__ import annotations

import numpy as np
import pytest

from activityplan.model import (
    Activity,
    Contact,
    Problem,
    ResourceModel,
    SolarProfile,
)
from activityplan.simulate import simulate
from activityplan.timeline import Plan, RasterisedProblem

_DARK = SolarProfile(peak_power_w=0.0, orbit_period_s=1000.0)


def _rasterised(
    *,
    bus_load_w: float = 36.0,
    capacity_wh: float = 10.0,
    dod: float = 0.5,
    soc: float = 1.0,
    solar: SolarProfile = _DARK,
    recorder_mbit: float = 100.0,
    initial_recorder_mbit: float = 0.0,
    contacts: tuple[Contact, ...] = (),
    activities: tuple[Activity, ...] | None = None,
    charge_efficiency: float = 1.0,
    horizon_s: float = 1000.0,
    step_s: float = 100.0,
) -> RasterisedProblem:
    resources = ResourceModel(
        battery_capacity_wh=capacity_wh,
        max_depth_of_discharge=dod,
        solar=solar,
        recorder_capacity_mbit=recorder_mbit,
        initial_state_of_charge=soc,
        bus_load_w=bus_load_w,
        initial_recorder_mbit=initial_recorder_mbit,
        contacts=contacts,
        charge_efficiency=charge_efficiency,
    )
    if activities is None:
        activities = (Activity.uniform("a", 200.0, 36.0, data_mbit=60.0),)
    problem = Problem(activities=activities, resources=resources, horizon_s=horizon_s)
    return RasterisedProblem.build(problem, step_s)


def test_battery_discharges_by_the_hand_calculated_amount() -> None:
    # Dark, bus 36 W, 100 s steps: every step costs exactly 1 W*h. Starting at
    # 10 W*h the end-of-step energies are 9, 8, 7, ... with no activity placed.
    rasterised = _rasterised()
    trace = simulate(rasterised, Plan({}, rasterised.grid))
    assert trace.battery_wh[0] == pytest.approx(9.0, rel=1e-13)
    assert trace.battery_wh[1] == pytest.approx(8.0, rel=1e-13)
    assert trace.battery_wh[9] == pytest.approx(0.0, abs=1e-12)
    assert trace.min_battery_wh == pytest.approx(0.0, abs=1e-12)
    assert trace.min_battery_step == 9
    assert trace.shunted_wh == pytest.approx(0.0, abs=1e-15)


def test_battery_is_not_clamped_below_the_floor_or_below_zero() -> None:
    # 20 steps of 1 W*h from a 10 W*h battery must end at -10 W*h, because
    # clamping at the floor would hide the infeasibility this product reports.
    rasterised = _rasterised(horizon_s=2000.0)
    trace = simulate(rasterised, Plan({}, rasterised.grid))
    assert trace.battery_wh[-1] == pytest.approx(-10.0, rel=1e-12)


def test_activity_load_adds_to_the_fixed_load() -> None:
    # Activity: 36 W for 200 s = 1 W*h per step over two steps. Placed at step
    # 0 the first two steps cost 2 W*h each: 10 -> 8 -> 6 -> 5 -> ...
    rasterised = _rasterised()
    trace = simulate(rasterised, Plan({"a": 0}, rasterised.grid))
    assert trace.battery_wh[0] == pytest.approx(8.0, rel=1e-13)
    assert trace.battery_wh[1] == pytest.approx(6.0, rel=1e-13)
    assert trace.battery_wh[2] == pytest.approx(5.0, rel=1e-13)
    assert trace.activity_load_wh[0] == pytest.approx(1.0, rel=1e-13)
    assert trace.activity_load_wh[2] == pytest.approx(0.0, abs=1e-15)


def test_surplus_above_capacity_is_shunted_not_stored() -> None:
    # No load at all and a 100 W peak array: the battery starts full, so every
    # joule generated is shunted and the energy never exceeds the capacity.
    solar = SolarProfile(peak_power_w=100.0, orbit_period_s=1000.0)
    rasterised = _rasterised(bus_load_w=0.0, solar=solar)
    trace = simulate(rasterised, Plan({}, rasterised.grid))
    assert float(np.max(trace.battery_wh)) == pytest.approx(10.0, rel=1e-14)
    assert trace.shunted_wh == pytest.approx(trace.generated_wh, rel=1e-12)


def test_energy_balance_closes_for_a_full_trace() -> None:
    solar = SolarProfile(peak_power_w=80.0, orbit_period_s=1000.0, eclipse_fraction=0.35)
    rasterised = _rasterised(
        bus_load_w=25.0, capacity_wh=30.0, soc=0.8, solar=solar, contacts=(
            Contact(300.0, 500.0, 1.0, 10.0, "pass"),
        )
    )
    trace = simulate(rasterised, Plan({"a": 2}, rasterised.grid))
    assert trace.energy_residual_wh() == pytest.approx(0.0, abs=1e-9)
    assert trace.effective_generated_wh == pytest.approx(trace.generated_wh, rel=1e-12)


def test_charge_efficiency_reduces_only_the_surplus() -> None:
    solar = SolarProfile(peak_power_w=100.0, orbit_period_s=1000.0)
    lossless = _rasterised(
        bus_load_w=20.0, capacity_wh=100.0, dod=0.9, soc=0.2, solar=solar
    )
    lossy = _rasterised(
        bus_load_w=20.0,
        capacity_wh=100.0,
        dod=0.9,
        soc=0.2,
        solar=solar,
        charge_efficiency=0.9,
    )
    plan_a = Plan({}, lossless.grid)
    perfect = simulate(lossless, plan_a)
    reduced = simulate(lossy, Plan({}, lossy.grid))
    assert reduced.battery_wh[-1] < perfect.battery_wh[-1]
    assert reduced.energy_residual_wh() == pytest.approx(0.0, abs=1e-9)
    # The weighted generation figure is strictly below the raw one, because some
    # steps are in surplus.
    assert reduced.effective_generated_wh < reduced.generated_wh


def test_recorder_fills_then_drains_at_the_contact_rate() -> None:
    # Activity writes 60 Mbit over 200 s = 30 Mbit per 100 s step, placed at
    # step 0. A contact over [300, 500) s at 0.2 Mbit/s clears 20 Mbit per step.
    rasterised = _rasterised(
        bus_load_w=0.0, capacity_wh=100.0, contacts=(Contact(300.0, 500.0, 0.2, 0.0, "pass"),)
    )
    trace = simulate(rasterised, Plan({"a": 0}, rasterised.grid))
    assert trace.recorder_mbit[0] == pytest.approx(30.0, rel=1e-13)
    assert trace.recorder_mbit[1] == pytest.approx(60.0, rel=1e-13)
    assert trace.recorder_mbit[2] == pytest.approx(60.0, rel=1e-13)
    assert trace.recorder_mbit[3] == pytest.approx(40.0, rel=1e-13)
    assert trace.recorder_mbit[4] == pytest.approx(20.0, rel=1e-13)
    assert trace.recorder_mbit[5] == pytest.approx(20.0, rel=1e-13)
    assert trace.total_downlinked_mbit == pytest.approx(40.0, rel=1e-13)
    assert trace.total_produced_mbit == pytest.approx(60.0, rel=1e-13)
    assert trace.max_recorder_mbit == pytest.approx(60.0, rel=1e-13)
    assert trace.max_recorder_step == 1


def test_within_step_peak_is_recorded_separately_from_the_end_of_step_content() -> None:
    # In a step that both produces and downlinks, the pre-downlink content is
    # strictly above the end-of-step content, and both are reported.
    rasterised = _rasterised(
        bus_load_w=0.0, capacity_wh=100.0, contacts=(Contact(0.0, 200.0, 0.2, 0.0, "pass"),)
    )
    trace = simulate(rasterised, Plan({"a": 0}, rasterised.grid))
    assert trace.recorder_peak_mbit[0] == pytest.approx(30.0, rel=1e-13)
    assert trace.recorder_mbit[0] == pytest.approx(10.0, rel=1e-13)


def test_initial_recorder_content_is_carried() -> None:
    rasterised = _rasterised(bus_load_w=0.0, capacity_wh=100.0, initial_recorder_mbit=25.0)
    trace = simulate(rasterised, Plan({}, rasterised.grid))
    assert trace.recorder_mbit[0] == pytest.approx(25.0, rel=1e-13)


def test_clamp_policy_is_pointwise_maximal_in_the_battery_energy() -> None:
    """The dominance lemma the MILP relies on, checked numerically.

    The simulator shunts only when the battery is full. Any other admissible
    shunt schedule dissipates at least as much at least as early, so its energy
    trace can only be lower. Here the alternative is a deliberately wasteful
    policy that shunts a fixed 0.2 W*h in every surplus step, and the lemma is
    that the simulator's trace dominates it at every step.
    """
    solar = SolarProfile(peak_power_w=90.0, orbit_period_s=1000.0, eclipse_fraction=0.3)
    rasterised = _rasterised(bus_load_w=25.0, capacity_wh=40.0, soc=0.7, solar=solar)
    plan = Plan({"a": 3}, rasterised.grid)
    trace = simulate(rasterised, plan)

    capacity = rasterised.problem.resources.battery_capacity_wh
    energy = rasterised.problem.resources.initial_energy_wh
    activity_load = rasterised.activity_load_wh(plan)
    wasteful = np.empty(rasterised.n_steps, dtype=float)
    for k in range(rasterised.n_steps):
        surplus = rasterised.solar_wh[k] - rasterised.fixed_wh[k] - activity_load[k]
        extra = 0.2 if surplus > 0.0 else 0.0
        energy = min(capacity, energy + surplus - extra)
        wasteful[k] = energy
    assert np.all(trace.battery_wh >= wasteful - 1e-12)
    assert float(np.min(trace.battery_wh - wasteful)) > -1e-12
    assert float(np.max(trace.battery_wh - wasteful)) > 0.0


def test_maximal_drain_policy_is_pointwise_minimal_in_the_recorder_content() -> None:
    """The recorder half of the dominance lemma, against a lazy drain policy."""
    rasterised = _rasterised(
        bus_load_w=0.0,
        capacity_wh=100.0,
        contacts=(Contact(200.0, 600.0, 0.15, 0.0, "pass"),),
    )
    plan = Plan({"a": 0}, rasterised.grid)
    trace = simulate(rasterised, plan)
    produced = rasterised.activity_data_production_mbit(plan)
    stored = 0.0
    lazy = np.empty(rasterised.n_steps, dtype=float)
    for k in range(rasterised.n_steps):
        stored = stored + produced[k] - 0.5 * min(
            stored + produced[k], rasterised.downlink_cap_mbit[k]
        )
        lazy[k] = stored
    assert np.all(trace.recorder_mbit <= lazy + 1e-12)
    assert float(np.max(trace.recorder_mbit - lazy)) < 1e-12


def test_trace_rejects_a_plan_running_past_the_grid() -> None:
    rasterised = _rasterised()
    with pytest.raises(ValueError, match="runs past"):
        simulate(rasterised, Plan({"a": 9}, rasterised.grid))


def test_simulation_is_deterministic() -> None:
    rasterised = _rasterised()
    plan = Plan({"a": 1}, rasterised.grid)
    first = simulate(rasterised, plan)
    second = simulate(rasterised, plan)
    assert np.array_equal(first.battery_wh, second.battery_wh)
    assert np.array_equal(first.recorder_mbit, second.recorder_mbit)
