"""The shipped instances and the seeded family."""

from __future__ import annotations

import math

import numpy as np
import pytest

from activityplan.feasibility import evaluate
from activityplan.instances import (
    INSTANCE_ORDER,
    RECOMMENDED_STEP_S,
    _fixed_and_solar_steps,
    _worst_drawdown_wh,
    instance_family,
    named_instance,
    sample_instance_spec,
)
from activityplan.timeline import Plan, RasterisedProblem


@pytest.mark.parametrize("name", INSTANCE_ORDER)
def test_every_shipped_instance_admits_the_empty_plan(name: str) -> None:
    """The condition an earlier draft of energy-bound failed.

    A declaration whose fixed loads alone break the depth-of-discharge floor has
    no feasible plan at all, so every planner returns nothing and every
    comparison is vacuous. It is checked here for every instance because it is
    easy to reintroduce by changing a solar peak.
    """
    rasterised = RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name])
    report = evaluate(rasterised, Plan({}, rasterised.grid))
    assert report.feasible, (name, report.violations)


@pytest.mark.parametrize("name", INSTANCE_ORDER)
def test_every_shipped_instance_is_oversubscribed_or_structurally_tight(name: str) -> None:
    problem = named_instance(name)
    generated, fixed = problem.baseline_energy_balance_wh()
    available = generated - fixed + problem.resources.usable_wh
    energy_tight = problem.total_activity_energy_wh > available
    data_tight = (
        problem.total_data_mbit > problem.resources.recorder_capacity_mbit
        or problem.total_data_mbit > problem.resources.total_contact_capacity_mbit
    )
    structural = bool(problem.exclusions or problem.precedences)
    assert energy_tight or data_tight or structural, name


def test_named_instance_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown instance"):
        named_instance("not-an-instance")


def test_recommended_steps_divide_their_horizons() -> None:
    for name in INSTANCE_ORDER:
        problem = named_instance(name)
        ratio = problem.horizon_s / RECOMMENDED_STEP_S[name]
        assert abs(ratio - round(ratio)) < 1e-9, name


def test_tiny_instance_is_small_enough_to_enumerate() -> None:
    rasterised = RasterisedProblem.build(named_instance("tiny-enumerable"), 100.0)
    candidates = 1
    for index in range(rasterised.n_activities):
        candidates *= rasterised.allowed_starts[index].size + 1
    assert candidates < 100_000


def test_fixed_and_solar_helper_matches_the_rasteriser() -> None:
    """The sizing helper duplicates the rasteriser's arithmetic; they must agree.

    If they drift, the generator sizes resources against one model and the
    planners run against another, which is exactly the class of defect that
    produced a family of instances with no feasible plan.
    """
    spec = sample_instance_spec(4242)
    problem = spec.analytic_problem()
    rasterised = RasterisedProblem.build(problem, spec.step_s)
    fixed, solar = _fixed_and_solar_steps(
        spec.resources.solar,
        spec.resources.contacts,
        spec.resources.bus_load_w,
        spec.horizon_s,
        spec.step_s,
    )
    assert np.allclose(np.array(fixed), rasterised.fixed_wh, rtol=1e-12, atol=1e-15)
    assert np.allclose(np.array(solar), rasterised.solar_wh, rtol=1e-12, atol=1e-15)


def test_worst_drawdown_known_answer() -> None:
    # Deficits of +1, +1, -5, +2, +3, +1: the worst contiguous run is the last
    # three, summing to 6. The leading +1, +1 run is only 2.
    fixed = [1.0, 1.0, 0.0, 2.0, 3.0, 1.0]
    generated = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0]
    assert _worst_drawdown_wh(fixed, generated) == pytest.approx(6.0, rel=1e-14)
    # Never negative: a horizon in permanent surplus needs no battery.
    assert _worst_drawdown_wh([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0, abs=1e-15)


def test_sampled_instances_admit_the_empty_plan() -> None:
    for spec in instance_family(8, 7000):
        rasterised = RasterisedProblem.build(spec.analytic_problem(), spec.step_s)
        report = evaluate(rasterised, Plan({}, rasterised.grid))
        assert report.feasible, (spec.name, report.violations)


def test_sampled_instance_is_reproducible_from_its_seed() -> None:
    first = sample_instance_spec(333)
    second = sample_instance_spec(333)
    assert [r.name for r in first.requests] == [r.name for r in second.requests]
    assert first.resources.battery_capacity_wh == second.resources.battery_capacity_wh
    assert first.exclusions == second.exclusions
    assert first.precedences == second.precedences


def test_problem_from_builds_uniform_single_segment_activities() -> None:
    spec = sample_instance_spec(55, n_activities=4)
    durations = np.full(4, 200.0)
    energies = np.full(4, 3.0)
    problem = spec.problem_from(durations, energies, name_suffix="/test")
    assert problem.name.endswith("/test")
    for activity in problem.activities:
        assert len(activity.power_profile) == 1
        assert activity.duration_s == pytest.approx(200.0, rel=1e-14)
        assert activity.energy_wh == pytest.approx(3.0, rel=1e-13)


def test_problem_from_validates_the_estimate_lengths() -> None:
    spec = sample_instance_spec(55, n_activities=4)
    with pytest.raises(ValueError, match="expected 4 durations"):
        spec.problem_from(np.full(3, 200.0), np.full(3, 3.0))


def test_instance_family_and_sampler_validation() -> None:
    assert [spec.seed for spec in instance_family(3, 10)] == [10, 11, 12]
    with pytest.raises(ValueError, match="count"):
        instance_family(0, 10)
    with pytest.raises(ValueError, match="n_activities"):
        sample_instance_spec(1, n_activities=1)
    with pytest.raises(ValueError, match="n_orbits"):
        sample_instance_spec(1, n_orbits=0)
    with pytest.raises(ValueError, match="feasibility_margin"):
        sample_instance_spec(1, feasibility_margin=1.0)


def test_analytic_problem_matches_the_analytic_model() -> None:
    from activityplan.features import analytic_duration_s, analytic_energy_wh

    spec = sample_instance_spec(77, n_activities=5)
    problem = spec.analytic_problem()
    for request, activity in zip(spec.requests, problem.activities, strict=True):
        assert activity.duration_s == pytest.approx(analytic_duration_s(request), rel=1e-12)
        assert activity.energy_wh == pytest.approx(analytic_energy_wh(request), rel=1e-12)
    assert math.isfinite(problem.total_priority)
