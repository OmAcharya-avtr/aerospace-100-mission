"""Warm-starting the exact solver: verification gate and effect."""

from __future__ import annotations

import numpy as np
import pytest

from activityplan.exact import solve_exact
from activityplan.greedy import plan_greedy
from activityplan.instances import RECOMMENDED_STEP_S, named_instance, sample_instance_spec
from activityplan.timeline import Plan, RasterisedProblem
from activityplan.warmstart import compare_warm_start, warm_start_from_plan


def _leo() -> RasterisedProblem:
    return RasterisedProblem.build(
        named_instance("leo-imaging"), RECOMMENDED_STEP_S["leo-imaging"]
    )


def test_a_feasible_candidate_becomes_a_bound() -> None:
    rasterised = _leo()
    candidate = plan_greedy(rasterised).plan
    warm_start = warm_start_from_plan(rasterised, candidate, source="greedy")
    assert warm_start.verified
    assert warm_start.bound == pytest.approx(candidate.value(rasterised.problem), rel=1e-12)
    assert "verified feasible" in warm_start.reason


def test_an_infeasible_candidate_is_refused_as_a_bound() -> None:
    """The dangerous direction: a plan for a different problem used as a bound
    on this one would cut off the optimum. The simulator gate stops it."""
    rasterised = _leo()
    stacked = Plan(
        dict.fromkeys((activity.name for activity in rasterised.problem.activities[:4]), 0),
        rasterised.grid,
    )
    warm_start = warm_start_from_plan(rasterised, stacked, source="stacked")
    assert not warm_start.verified
    assert warm_start.bound is None
    assert "rejected" in warm_start.reason
    assert warm_start.candidate_value > 0.0


def test_a_candidate_that_cannot_be_simulated_is_refused_not_raised() -> None:
    rasterised = _leo()
    last = rasterised.n_steps - 1
    overrun = Plan({"image-01": last}, rasterised.grid)
    warm_start = warm_start_from_plan(rasterised, overrun, source="overrun")
    assert not warm_start.verified
    assert "could not be simulated" in warm_start.reason


def test_warm_start_preserves_the_optimum() -> None:
    rasterised = _leo()
    candidate = plan_greedy(rasterised).plan
    warm_start = warm_start_from_plan(rasterised, candidate, source="greedy")
    comparison = compare_warm_start(rasterised, warm_start, time_limit_s=120.0)
    assert comparison.warm is not None
    assert comparison.optimum_preserved
    assert comparison.cold.value == pytest.approx(comparison.warm.value, abs=1e-9)
    assert comparison.node_ratio is not None
    assert comparison.time_ratio is not None


def test_no_warm_solve_runs_when_the_candidate_is_refused() -> None:
    rasterised = _leo()
    refused = warm_start_from_plan(
        rasterised,
        Plan(
            dict.fromkeys(
                (activity.name for activity in rasterised.problem.activities[:4]), 0
            ),
            rasterised.grid,
        ),
        source="stacked",
    )
    comparison = compare_warm_start(rasterised, refused, time_limit_s=60.0)
    assert comparison.warm is None
    assert comparison.optimum_preserved
    assert comparison.node_ratio is None
    assert comparison.time_ratio is None


def test_the_empty_plan_is_always_a_valid_bound_of_zero() -> None:
    rasterised = _leo()
    warm_start = warm_start_from_plan(rasterised, Plan({}, rasterised.grid), source="empty")
    assert warm_start.verified
    assert warm_start.bound == pytest.approx(0.0, abs=1e-15)
    comparison = compare_warm_start(rasterised, warm_start, time_limit_s=60.0)
    assert comparison.optimum_preserved


def test_warm_start_on_a_sampled_instance_with_presolve_off() -> None:
    # Six activities, not the family default of fourteen: with presolve off
    # HiGHS has to branch, and a fourteen-activity instance takes about twenty
    # seconds per solve. That cost is itself one of the warm-start study's
    # measurements and is reported in validation/validate_warmstart_output.txt;
    # here the instance is small so the test stays quick.
    spec = sample_instance_spec(9003, n_activities=6)
    rasterised = RasterisedProblem.build(spec.analytic_problem(), spec.step_s)
    candidate = plan_greedy(rasterised).plan
    warm_start = warm_start_from_plan(rasterised, candidate, source="greedy-analytic")
    comparison = compare_warm_start(
        rasterised, warm_start, presolve=False, time_limit_s=120.0
    )
    assert comparison.presolve is False
    assert comparison.warm is not None
    assert comparison.optimum_preserved
    # Both arms must agree with a plain solve of the same model.
    plain = solve_exact(rasterised, time_limit_s=120.0, presolve=False)
    assert plain.value == pytest.approx(comparison.cold.value, abs=1e-9)


def test_warm_start_result_fields_are_recorded() -> None:
    rasterised = _leo()
    candidate = plan_greedy(rasterised).plan
    warm_start = warm_start_from_plan(rasterised, candidate, source="label")
    assert warm_start.source == "label"
    comparison = compare_warm_start(rasterised, warm_start, time_limit_s=60.0)
    assert comparison.instance == "leo-imaging"
    assert comparison.warm is not None
    assert comparison.warm.primal_bound_applied == pytest.approx(
        warm_start.bound, rel=1e-12
    )
    assert comparison.cold.primal_bound_applied is None
    assert np.isfinite(comparison.cold.dual_bound)
