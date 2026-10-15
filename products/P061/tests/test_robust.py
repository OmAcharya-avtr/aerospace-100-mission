"""Planning under predicted-duration uncertainty."""

from __future__ import annotations

import math

import numpy as np
import pytest

from activityplan.features import feature_matrix, realise, sample_requests
from activityplan.instances import sample_instance_spec
from activityplan.predictor import DurationEnergyPredictor
from activityplan.robust import (
    PLANNING_MODES,
    compare_modes,
    estimate_for_mode,
    realise_spec,
    score_realisation,
)
from activityplan.timeline import Plan, RasterisedProblem


def _predictor(seed: int = 61, n: int = 400) -> DurationEnergyPredictor:
    rng = np.random.default_rng(seed)
    requests = sample_requests(n, rng)
    realised = [realise(request, rng) for request in requests]
    return DurationEnergyPredictor(
        alpha=0.1, mode="normalised", n_estimators=40, random_state=seed
    ).fit(
        feature_matrix(list(requests)),
        np.array([item.duration_s for item in realised], dtype=float),
        np.array([item.energy_wh for item in realised], dtype=float),
    )


def test_realise_spec_is_reproducible_and_shaped() -> None:
    spec = sample_instance_spec(1234, n_activities=6)
    first = realise_spec(spec, np.random.default_rng(1))
    second = realise_spec(spec, np.random.default_rng(1))
    assert first[0].shape == (6,)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_oracle_mode_returns_the_truth_and_does_not_alias_it() -> None:
    spec = sample_instance_spec(1234, n_activities=5)
    truth = realise_spec(spec, np.random.default_rng(2))
    estimate = estimate_for_mode(spec, "oracle", truth=truth)
    assert np.array_equal(estimate.durations_s, truth[0])
    estimate.durations_s[0] = -1.0
    assert truth[0][0] > 0.0


def test_robust_mode_is_never_shorter_than_the_point_prediction() -> None:
    spec = sample_instance_spec(1234, n_activities=8)
    predictor = _predictor()
    point = estimate_for_mode(spec, "ml-point", predictor=predictor)
    robust = estimate_for_mode(spec, "ml-robust", predictor=predictor)
    assert np.all(robust.durations_s >= point.durations_s - 1e-9)
    assert np.all(robust.energies_wh >= point.energies_wh - 1e-9)
    assert float(np.sum(robust.durations_s)) > float(np.sum(point.durations_s))


def test_estimate_for_mode_validation() -> None:
    spec = sample_instance_spec(1234, n_activities=3)
    with pytest.raises(ValueError, match="mode must be one of"):
        estimate_for_mode(spec, "guess")
    with pytest.raises(ValueError, match="needs the realised truth"):
        estimate_for_mode(spec, "oracle")
    with pytest.raises(ValueError, match="needs a fitted predictor"):
        estimate_for_mode(spec, "ml-point")


def test_score_realisation_credits_a_feasible_plan_and_zeroes_a_violating_one() -> None:
    spec = sample_instance_spec(4242, n_activities=6)
    truth = realise_spec(spec, np.random.default_rng(3))
    problem = spec.problem_from(truth[0], truth[1])
    rasterised = RasterisedProblem.build(problem, spec.step_s)
    empty = score_realisation(spec, Plan({}, rasterised.grid), truth, "oracle")
    assert empty.feasible
    assert empty.realised_value == pytest.approx(0.0, abs=1e-15)
    assert empty.binding

    # Place every activity at step 0, which stacks every load on one step.
    stacked = Plan(dict.fromkeys((r.name for r in spec.requests), 0), rasterised.grid)
    crowded = score_realisation(spec, stacked, truth, "oracle")
    assert not crowded.feasible
    assert crowded.realised_value == pytest.approx(0.0, abs=1e-15)
    assert crowded.planned_value > 0.0


def test_score_realisation_reports_a_declaration_rejection_rather_than_raising() -> None:
    # A realised duration longer than the horizon makes the declaration itself
    # invalid; that is a genuine infeasibility and must be reported, not raised.
    spec = sample_instance_spec(4242, n_activities=4)
    truth = realise_spec(spec, np.random.default_rng(3))
    broken = (np.full(4, spec.horizon_s * 2.0), truth[1])
    rasterised = RasterisedProblem.build(spec.analytic_problem(), spec.step_s)
    outcome = score_realisation(spec, Plan({}, rasterised.grid), broken, "oracle")
    assert not outcome.feasible
    assert "do not admit the declaration" in outcome.reason
    assert outcome.binding == "declaration rejected the realised durations"
    assert math.isnan(outcome.planned_value)


def test_score_realisation_reports_a_grid_overrun_rather_than_raising() -> None:
    spec = sample_instance_spec(4242, n_activities=4)
    truth = realise_spec(spec, np.random.default_rng(3))
    rasterised = RasterisedProblem.build(spec.analytic_problem(), spec.step_s)
    last_step = rasterised.n_steps - 1
    plan = Plan({spec.requests[0].name: last_step}, rasterised.grid)
    long_truth = (truth[0].copy(), truth[1].copy())
    long_truth[0][0] = spec.step_s * 4.0
    outcome = score_realisation(spec, plan, long_truth, "oracle")
    assert not outcome.feasible
    assert (
        "past the grid" in outcome.reason
        or "do not admit the declaration" in outcome.reason
    )


def test_compare_modes_returns_one_outcome_per_mode_on_a_shared_realisation() -> None:
    spec = sample_instance_spec(9000)
    outcomes = compare_modes(
        spec, np.random.default_rng(1), predictor=_predictor(), planner="greedy"
    )
    assert tuple(outcome.mode for outcome in outcomes) == PLANNING_MODES
    for outcome in outcomes:
        assert outcome.n_placed >= 0
        assert outcome.reason


def test_compare_modes_rejects_an_unknown_planner() -> None:
    spec = sample_instance_spec(9000, n_activities=4)
    with pytest.raises(ValueError, match="planner must be"):
        compare_modes(
            spec, np.random.default_rng(1), predictor=_predictor(), planner="annealing"
        )


def test_oracle_plans_are_feasible_by_construction() -> None:
    """The oracle plans with the realised durations, so its plan is exactly the
    plan the greedy planner verified against those durations. Any infeasibility
    here would be a defect in the realisation path, not a modelling outcome."""
    predictor = _predictor()
    for offset in range(4):
        spec = sample_instance_spec(9000 + offset)
        outcomes = compare_modes(
            spec,
            np.random.default_rng(12_000 + offset),
            predictor=predictor,
            planner="greedy",
            modes=("oracle",),
        )
        assert outcomes[0].feasible, (spec.name, outcomes[0].reason)
