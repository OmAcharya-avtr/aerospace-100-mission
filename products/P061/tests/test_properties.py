"""Property-based tests over the algebraic identities the package relies on.

Every comparison here is to a tolerance. None of these properties is asserted
bit-exactly, because all of them run through an accumulation of floating-point
sums, and none of them asserts the equality of a boolean produced by a strict
threshold comparison.

Any falsifying example Hypothesis finds here must be pinned as a named
regression test with literal inputs in ``tests/test_regressions.py``, because
``.hypothesis/`` is gitignored and a counterexample found once is otherwise
lost on a fresh clone.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from activityplan.feasibility import evaluate
from activityplan.greedy import plan_greedy
from activityplan.model import (
    Activity,
    PowerSegment,
    Problem,
    ResourceModel,
    SolarProfile,
)
from activityplan.simulate import simulate
from activityplan.timeline import Plan, RasterisedProblem

_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

_powers = st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False)
_durations = st.floats(min_value=1.0, max_value=900.0, allow_nan=False, allow_infinity=False)


@st.composite
def _profile(draw: st.DrawFn) -> tuple[PowerSegment, ...]:
    count = draw(st.integers(min_value=1, max_value=4))
    return tuple(
        PowerSegment(draw(_durations), draw(_powers)) for _ in range(count)
    )


@_SETTINGS
@given(_profile())
def test_activity_energy_equals_the_sum_of_its_segments(profile: tuple[PowerSegment, ...]) -> None:
    activity = Activity(name="a", power_profile=profile)
    expected = math.fsum(segment.power_w * segment.duration_s / 3600.0 for segment in profile)
    assert activity.energy_wh == _approx(expected)


@_SETTINGS
@given(_profile(), st.floats(min_value=0.0, max_value=1.0))
def test_energy_between_is_additive_across_a_split(
    profile: tuple[PowerSegment, ...], split_fraction: float
) -> None:
    activity = Activity(name="a", power_profile=profile)
    # The split is expressed as a fraction of the duration. An absolute split
    # drawn independently of the duration can land beyond it, and the second
    # call then gets a reversed interval, which the API refuses by design; see
    # tests/test_regressions.py::test_energy_between_refuses_a_reversed_interval.
    split = split_fraction * activity.duration_s
    total = activity.energy_between_wh(0.0, activity.duration_s)
    halves = activity.energy_between_wh(0.0, split) + activity.energy_between_wh(
        split, activity.duration_s
    )
    assert halves == _approx(total)


@_SETTINGS
@given(
    st.floats(min_value=0.0, max_value=400.0),
    st.floats(min_value=100.0, max_value=4000.0),
    st.floats(min_value=0.0, max_value=0.9),
    st.floats(min_value=-5000.0, max_value=5000.0),
    st.floats(min_value=0.0, max_value=12_000.0),
    st.floats(min_value=0.0, max_value=12_000.0),
)
def test_solar_energy_is_additive_and_monotone(
    peak: float, period: float, fraction: float, phase: float, a: float, b: float
) -> None:
    solar = SolarProfile(
        peak_power_w=peak, orbit_period_s=period, eclipse_fraction=fraction, phase_s=phase
    )
    lo, hi = sorted((a, b))
    mid = 0.5 * (lo + hi)
    whole = solar.energy_wh(lo, hi)
    split = solar.energy_wh(lo, mid) + solar.energy_wh(mid, hi)
    assert split == _approx(whole, floor=1e-9)
    assert whole >= -1e-12
    # Monotone in the upper limit, because the integrand is non-negative.
    assert solar.energy_wh(lo, hi + 10.0) >= whole - 1e-9


@_SETTINGS
@given(
    st.floats(min_value=0.0, max_value=300.0),
    st.floats(min_value=0.0, max_value=0.8),
    st.floats(min_value=0.0, max_value=80.0),
    st.integers(min_value=1, max_value=6),
)
def test_simulated_energy_balance_closes(
    peak: float, fraction: float, bus_load_w: float, n_activities: int
) -> None:
    solar = SolarProfile(
        peak_power_w=peak, orbit_period_s=1200.0, eclipse_fraction=fraction, phase_s=137.0
    )
    resources = ResourceModel(
        battery_capacity_wh=60.0,
        max_depth_of_discharge=0.9,
        solar=solar,
        recorder_capacity_mbit=5000.0,
        bus_load_w=bus_load_w,
    )
    activities = tuple(
        Activity.uniform(f"a{index}", 200.0, 30.0 + 5.0 * index, data_mbit=20.0)
        for index in range(n_activities)
    )
    problem = Problem(activities=activities, resources=resources, horizon_s=2400.0)
    rasterised = RasterisedProblem.build(problem, 100.0)
    plan = Plan({f"a{index}": 2 * index for index in range(n_activities)}, rasterised.grid)
    trace = simulate(rasterised, plan)
    scale = max(1.0, abs(trace.generated_wh) + abs(trace.consumed_wh))
    assert abs(trace.energy_residual_wh()) <= 1e-9 * scale


@_SETTINGS
@given(
    st.floats(min_value=20.0, max_value=200.0),
    st.floats(min_value=0.1, max_value=0.8),
    st.floats(min_value=10.0, max_value=60.0),
)
def test_greedy_returns_a_plan_its_own_evaluator_calls_feasible(
    peak: float, dod: float, bus_load_w: float
) -> None:
    solar = SolarProfile(peak_power_w=peak, orbit_period_s=1200.0, eclipse_fraction=0.3)
    resources = ResourceModel(
        battery_capacity_wh=80.0,
        max_depth_of_discharge=dod,
        solar=solar,
        recorder_capacity_mbit=800.0,
        bus_load_w=bus_load_w,
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
    )
    rasterised = RasterisedProblem.build(problem, 100.0)
    empty = evaluate(rasterised, Plan({}, rasterised.grid))
    result = plan_greedy(rasterised)
    report = evaluate(rasterised, result.plan)
    # Re-evaluating the same plan with the same function reproduces the same
    # floats, so this is a consistency check on the planner, not a claim that a
    # threshold boolean is stable under a change of representation.
    #
    # The property is CONDITIONAL on the empty plan being feasible. Hypothesis
    # found peak=20 W, dod=0.1, bus=21 W, where the fixed load alone breaks the
    # depth-of-discharge floor: there the greedy planner correctly places
    # nothing and correctly returns an infeasible report, because the
    # declaration admits no plan at all. That case is pinned in
    # tests/test_regressions.py.
    if empty.feasible:
        assert report.feasible
    else:
        assert result.plan.n_placed == 0
        assert not report.feasible
    assert result.value == _approx(report.value)


@_SETTINGS
@given(st.lists(st.integers(min_value=0, max_value=5), min_size=0, max_size=4, unique=True))
def test_plan_value_is_monotone_in_the_placed_set(indices: list[int]) -> None:
    resources = ResourceModel(
        battery_capacity_wh=100.0,
        max_depth_of_discharge=0.9,
        solar=SolarProfile(peak_power_w=0.0, orbit_period_s=1000.0),
        recorder_capacity_mbit=1000.0,
    )
    activities = tuple(
        Activity.uniform(f"a{index}", 100.0, 0.0, priority=1.0 + index) for index in range(6)
    )
    problem = Problem(activities=activities, resources=resources, horizon_s=1000.0)
    rasterised = RasterisedProblem.build(problem, 100.0)
    plan = Plan({f"a{index}": index for index in indices}, rasterised.grid)
    assert plan.value(problem) >= 0.0
    for index in range(6):
        name = f"a{index}"
        bigger = plan.with_activity(name, 0)
        assert bigger.value(problem) >= plan.value(problem) - 1e-12


@_SETTINGS
@given(
    st.floats(min_value=1.0, max_value=600.0),
    st.floats(min_value=10.0, max_value=200.0),
    st.floats(min_value=10.0, max_value=300.0),
)
def test_rasterised_energy_and_data_sums_match_the_declaration(
    duration: float, power: float, data: float
) -> None:
    activity = Activity.uniform("a", duration, power, data_mbit=data)
    resources = ResourceModel(
        battery_capacity_wh=500.0,
        max_depth_of_discharge=0.9,
        solar=SolarProfile(peak_power_w=0.0, orbit_period_s=1200.0),
        recorder_capacity_mbit=5000.0,
    )
    problem = Problem(activities=(activity,), resources=resources, horizon_s=1200.0)
    rasterised = RasterisedProblem.build(problem, 100.0)
    assert float(np.sum(rasterised.activity_energy_wh[0])) == _approx(activity.energy_wh)
    assert float(np.sum(rasterised.activity_data_mbit[0])) == _approx(data)


def _approx(expected: float, *, floor: float = 1e-12):  # noqa: ANN202 - pytest helper
    import pytest

    return pytest.approx(expected, rel=1e-9, abs=floor)
