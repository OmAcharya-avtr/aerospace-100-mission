"""Planning under predicted-duration uncertainty, and what it costs.

The question this module answers
--------------------------------
An onboard planner does not know how long an activity will take. It plans with
an estimate, the activity then takes what it takes, and the plan either survives
or it does not. So the number that matters is not the predictor's RMSE: it is
the fraction of plans that are still feasible once the durations realise, and
the objective those surviving plans deliver.

Four estimators are compared on identical instances, which is the only fair
comparison available:

==============  ==============================================================
mode            durations and energies used to plan
==============  ==============================================================
``oracle``      the realised values. Not achievable; an upper bound.
``analytic``    the declared analytic model (:mod:`activityplan.features`).
``ml-point``    the learned point predictions.
``ml-robust``   the **upper** ends of the learned conformal intervals.
==============  ==============================================================

``ml-robust`` is the robust plan the spec calls for: planning against the upper
bound of a ``1 - alpha`` prediction interval makes the plan tolerant of an
overrun up to that bound, at the cost of fitting fewer activities. Both halves
of that trade are measured.

Scoring rule, stated because it is a choice
-------------------------------------------
A realised plan that violates **any** declared constraint scores zero, not
partial credit. The alternative -- crediting the activities that completed
before the violation -- needs a model of what the spacecraft does when a
constraint is breached mid-plan, and this package has no such model. Zero is
the conservative reading and it is applied identically to all four modes, so the
comparison between them is unaffected by the choice; only the absolute levels
are. This is restated in the README's Limitations.

A realisation can also fail *before* simulation, when a realised duration no
longer fits its activity's declared window or the horizon. That is a genuine
infeasibility and is reported with its own reason string rather than raised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .exact import solve_exact
from .feasibility import evaluate
from .features import (
    analytic_duration_s,
    analytic_energy_wh,
    feature_matrix,
    realise,
)
from .greedy import plan_greedy
from .instances import InstanceSpec
from .predictor import DurationEnergyPredictor
from .timeline import Plan, RasterisedProblem

__all__ = [
    "PLANNING_MODES",
    "Estimate",
    "RealisationOutcome",
    "compare_modes",
    "estimate_for_mode",
    "realise_spec",
    "score_realisation",
]

PLANNING_MODES: tuple[str, ...] = ("oracle", "analytic", "ml-point", "ml-robust")
"""The four estimators compared, in report order."""


@dataclass(frozen=True)
class Estimate:
    """Durations [s] and energies [W*h] an estimator supplies, in request order."""

    durations_s: np.ndarray
    energies_wh: np.ndarray
    mode: str


@dataclass(frozen=True)
class RealisationOutcome:
    """What happened when a plan met the realised durations.

    Attributes
    ----------
    mode
        Which estimator produced the plan.
    planned_value
        Objective the planner believed it had achieved, on its own estimates.
    realised_value
        Objective actually delivered: ``planned_value`` if the realised plan is
        feasible, otherwise ``0.0``.
    feasible
        Whether the realised plan satisfies every declared constraint.
    reason
        Why not, when it does not; the binding constraint when it does.
    n_placed
        Activities the plan placed.
    binding
        Name of the binding constraint of the realised plan, or the reason the
        realisation could not be simulated.
    """

    mode: str
    planned_value: float
    realised_value: float
    feasible: bool
    reason: str
    n_placed: int
    binding: str


def realise_spec(
    spec: InstanceSpec, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw realised ``(durations_s, energies_wh)`` for every request in ``spec``.

    One draw from :func:`activityplan.features.realise` per request, in request
    order, from the supplied seeded generator.
    """
    durations = np.empty(len(spec.requests), dtype=float)
    energies = np.empty(len(spec.requests), dtype=float)
    for position, request in enumerate(spec.requests):
        realised = realise(request, rng)
        durations[position] = realised.duration_s
        energies[position] = realised.energy_wh
    return durations, energies


def estimate_for_mode(
    spec: InstanceSpec,
    mode: str,
    *,
    predictor: DurationEnergyPredictor | None = None,
    truth: tuple[np.ndarray, np.ndarray] | None = None,
) -> Estimate:
    """Durations and energies for one planning mode.

    Parameters
    ----------
    spec
        The instance.
    mode
        One of :data:`PLANNING_MODES`.
    predictor
        Required for ``"ml-point"`` and ``"ml-robust"``.
    truth
        Required for ``"oracle"``: the realised ``(durations_s, energies_wh)``.

    Notes
    -----
    ``ml-robust`` uses the upper ends of both conformal intervals, floored at a
    small positive duration. Note that this is **not** a joint
    ``1 - alpha`` guarantee over the two targets: the union bound gives
    ``1 - 2*alpha`` for both holding simultaneously, and nothing in this package
    computes a tighter one.
    """
    if mode not in PLANNING_MODES:
        raise ValueError(f"mode must be one of {PLANNING_MODES}, got {mode!r}")
    if mode == "oracle":
        if truth is None:
            raise ValueError("mode 'oracle' needs the realised truth")
        return Estimate(durations_s=truth[0].copy(), energies_wh=truth[1].copy(), mode=mode)
    if mode == "analytic":
        durations = np.array([analytic_duration_s(r) for r in spec.requests], dtype=float)
        energies = np.array([analytic_energy_wh(r) for r in spec.requests], dtype=float)
        return Estimate(durations_s=durations, energies_wh=energies, mode=mode)
    if predictor is None:
        raise ValueError(f"mode {mode!r} needs a fitted predictor")
    features = feature_matrix(list(spec.requests))
    duration_interval = predictor.duration.predict(features)
    energy_interval = predictor.energy.predict(features)
    if mode == "ml-point":
        durations = np.maximum(1.0, duration_interval.point)
        energies = np.maximum(0.0, energy_interval.point)
    else:
        durations = np.maximum(1.0, duration_interval.upper)
        energies = np.maximum(0.0, energy_interval.upper)
    return Estimate(durations_s=durations, energies_wh=energies, mode=mode)


def _plan(spec: InstanceSpec, estimate: Estimate, planner: str, time_limit_s: float) -> Plan:
    problem = spec.problem_from(
        estimate.durations_s, estimate.energies_wh, name_suffix=f"/{estimate.mode}"
    )
    rasterised = RasterisedProblem.build(problem, spec.step_s)
    if planner == "greedy":
        return plan_greedy(rasterised).plan
    if planner == "exact":
        return solve_exact(rasterised, time_limit_s=time_limit_s).plan
    raise ValueError(f"planner must be 'greedy' or 'exact', got {planner!r}")


def score_realisation(
    spec: InstanceSpec,
    plan: Plan,
    truth: tuple[np.ndarray, np.ndarray],
    mode: str,
) -> RealisationOutcome:
    """Evaluate ``plan`` against the realised durations and energies.

    The plan's start steps are kept; only the activity durations and energies
    change. The grid is unchanged, so the step indices remain meaningful.
    """
    try:
        realised_problem = spec.problem_from(truth[0], truth[1], name_suffix=f"/realised-{mode}")
    except ValueError as error:
        return RealisationOutcome(
            mode=mode,
            planned_value=math.nan,
            realised_value=0.0,
            feasible=False,
            reason=f"realised durations do not admit the declaration: {error}",
            n_placed=plan.n_placed,
            binding="declaration rejected the realised durations",
        )
    rasterised = RasterisedProblem.build(realised_problem, spec.step_s)
    planned_value = plan.value(realised_problem)
    try:
        report = evaluate(rasterised, plan)
    except ValueError as error:
        return RealisationOutcome(
            mode=mode,
            planned_value=planned_value,
            realised_value=0.0,
            feasible=False,
            reason=f"realised plan runs past the grid: {error}",
            n_placed=plan.n_placed,
            binding="ran past the horizon",
        )
    if report.feasible:
        return RealisationOutcome(
            mode=mode,
            planned_value=planned_value,
            realised_value=planned_value,
            feasible=True,
            reason=(
                f"feasible; binding {report.binding.name} at "
                f"{report.binding.normalised:+.6f}"
            ),
            n_placed=plan.n_placed,
            binding=report.binding.name,
        )
    worst = report.violations[0]
    return RealisationOutcome(
        mode=mode,
        planned_value=planned_value,
        realised_value=0.0,
        feasible=False,
        reason=(
            f"{worst.name} violated by {-worst.slack:.6g} {worst.unit} at {worst.where}"
        ),
        n_placed=plan.n_placed,
        binding=worst.name,
    )


def compare_modes(
    spec: InstanceSpec,
    rng: np.random.Generator,
    *,
    predictor: DurationEnergyPredictor,
    planner: str = "greedy",
    modes: tuple[str, ...] = PLANNING_MODES,
    time_limit_s: float = 30.0,
) -> tuple[RealisationOutcome, ...]:
    """Plan ``spec`` under each mode, then score every plan on one shared realisation.

    One realisation is drawn from ``rng`` and used for all modes, so the modes
    see identical truth and the comparison is paired. Repeat over seeds to get a
    distribution; ``validation/validate_robust.py`` does.
    """
    truth = realise_spec(spec, rng)
    outcomes: list[RealisationOutcome] = []
    for mode in modes:
        estimate = estimate_for_mode(spec, mode, predictor=predictor, truth=truth)
        plan = _plan(spec, estimate, planner, time_limit_s)
        outcomes.append(score_realisation(spec, plan, truth, mode))
    return tuple(outcomes)
