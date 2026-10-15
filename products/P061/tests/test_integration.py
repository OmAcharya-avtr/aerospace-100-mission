"""End-to-end: declaration, both baselines, the learned component, and the report."""

from __future__ import annotations

import numpy as np
import pytest

from activityplan.exact import solve_exact
from activityplan.feasibility import evaluate
from activityplan.features import feature_matrix, realise, sample_requests
from activityplan.greedy import plan_greedy
from activityplan.instances import (
    INSTANCE_ORDER,
    RECOMMENDED_STEP_S,
    instance_family,
    named_instance,
)
from activityplan.predictor import DurationEnergyPredictor
from activityplan.report import render_feasibility, render_plan
from activityplan.robust import compare_modes, estimate_for_mode
from activityplan.timeline import RasterisedProblem
from activityplan.warmstart import compare_warm_start, warm_start_from_plan


def test_full_pipeline_on_the_worked_example() -> None:
    rasterised = RasterisedProblem.build(
        named_instance("leo-imaging"), RECOMMENDED_STEP_S["leo-imaging"]
    )
    greedy = plan_greedy(rasterised)
    exact = solve_exact(rasterised, time_limit_s=120.0)
    assert exact.proven_optimal
    assert exact.value > greedy.value
    greedy_report = evaluate(rasterised, greedy.plan)
    exact_report = evaluate(rasterised, exact.plan)
    assert greedy_report.feasible
    assert exact_report.feasible
    assert exact_report.binding.name == "recorder-capacity"
    text = render_feasibility(exact_report, rasterised) + render_plan(exact.plan, rasterised)
    assert "recorder-capacity" in text
    assert exact_report.unplaced


def test_learned_predictor_feeds_a_warm_start_and_a_robust_plan() -> None:
    rng = np.random.default_rng(61)
    requests = sample_requests(400, rng)
    realised = [realise(request, rng) for request in requests]
    predictor = DurationEnergyPredictor(
        alpha=0.1, mode="normalised", n_estimators=40, random_state=61
    ).fit(
        feature_matrix(list(requests)),
        np.array([item.duration_s for item in realised], dtype=float),
        np.array([item.energy_wh for item in realised], dtype=float),
    )
    spec = instance_family(1, 9000, n_activities=8)[0]
    estimate = estimate_for_mode(spec, "ml-point", predictor=predictor)
    target = RasterisedProblem.build(
        spec.problem_from(estimate.durations_s, estimate.energies_wh, name_suffix="/ml"),
        spec.step_s,
    )
    candidate = plan_greedy(target).plan
    warm_start = warm_start_from_plan(target, candidate, source="greedy-ml")
    assert warm_start.verified
    comparison = compare_warm_start(target, warm_start, time_limit_s=120.0)
    assert comparison.optimum_preserved
    assert comparison.warm is not None

    outcomes = compare_modes(spec, np.random.default_rng(7), predictor=predictor)
    by_mode = {outcome.mode: outcome for outcome in outcomes}
    # The robust plan places no more than the point plan, because it books
    # longer durations and more energy for the same activities.
    assert by_mode["ml-robust"].n_placed <= by_mode["ml-point"].n_placed


@pytest.mark.parametrize("name", INSTANCE_ORDER)
def test_every_instance_yields_a_complete_report_from_both_planners(name: str) -> None:
    rasterised = RasterisedProblem.build(named_instance(name), RECOMMENDED_STEP_S[name])
    for plan in (
        plan_greedy(rasterised).plan,
        solve_exact(rasterised, time_limit_s=120.0).plan,
    ):
        report = evaluate(rasterised, plan)
        assert report.feasible, (name, report.violations)
        assert report.candidates
        assert report.binding in report.candidates
        audit = report.energy_audit()
        scale = max(1.0, abs(audit["generated_steps"]) + abs(audit["consumed_steps"]))
        assert abs(audit["residual"]) <= 1e-9 * scale
        text = render_feasibility(report, rasterised)
        assert report.binding.name in text


def test_a_finer_step_cannot_lower_the_discretised_optimum_here() -> None:
    """The discretisation bound, measured rather than asserted in general.

    Halving the step enlarges the set of expressible start times, so the
    discretised optimum cannot fall -- provided the finer grid still snaps every
    window at least as loosely, which it does because ceil and floor on a
    refined grid move inward by no more than before. Checked on the two
    instances whose step divides evenly in half.
    """
    for name in ("tiny-enumerable", "recorder-bound"):
        problem = named_instance(name)
        coarse = solve_exact(
            RasterisedProblem.build(problem, RECOMMENDED_STEP_S[name]), time_limit_s=120.0
        )
        fine = solve_exact(
            RasterisedProblem.build(problem, RECOMMENDED_STEP_S[name] / 2.0),
            time_limit_s=120.0,
        )
        assert fine.value >= coarse.value - 1e-9, name
