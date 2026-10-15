"""The feasibility report: which constraint is binding, and the slack on the rest.

This module is the product's primary deliverable. A timeline is easy to draw and
hard to defend; what an operator has to be able to say is *which* declared
constraint is closest to being violated, by how much, and what the margin is on
everything else. Every constraint below is evaluated for every plan, in its own
physical unit and again normalised by a declared scale so that the smallest
normalised slack identifies the binding constraint.

Normalisation, stated because the choice determines the answer
--------------------------------------------------------------
There is no unit-free way to compare watt-hours with megabits, so each
constraint is normalised by the declared quantity that sets its own scale:

==============================  ==============================================
constraint                      normaliser
==============================  ==============================================
battery depth of discharge      usable energy ``capacity * dod`` [W*h]
recorder capacity               recorder capacity [Mbit]
recorder drained at the horizon recorder capacity [Mbit]
horizon                         horizon length [s]
activity time window            the activity's own window float [s], or its
                                duration when the float is zero
precedence separation           the predecessor's duration [s]
exclusion separation            the shorter of the two durations [s]
==============================  ==============================================

A normalised slack is therefore "fraction of the resource that is still
available", and the binding constraint is the one with the least of it. Any
reader who prefers a different normalisation gets a different binding
constraint, which is why the normaliser is printed in the report next to the
value. Two constraints can tie; the tie is broken by the declared evaluation
order, which is the order of the table above, and the report says so.

Diagnostics that are deliberately *not* candidates to be binding
----------------------------------------------------------------
The battery upper limit cannot be violated, because surplus is shunted
(:mod:`activityplan.simulate`), so its slack reaching zero means the battery is
full rather than that the plan is at a limit. The within-step recorder peak is
not a constraint either.

**Aggregate downlink volume is also a diagnostic, not a constraint.** Producing
more data than the declared contacts can carry is not an infeasibility: the
surplus simply stays on the recorder, and the physical limit is the recorder
capacity, which is evaluated step by step. It becomes a *necessary* condition
only when ``Problem.require_empty_recorder`` is set, and in that case the
constraint that is actually imposed is ``recorder-drained`` -- the end-of-horizon
content -- which is strictly stronger, so ``downlink-volume`` would add nothing
as a candidate. Treating it as a hard constraint was a defect in an earlier
draft of this module and it made the MILP optimum and the exhaustive
enumeration disagree by 1.5 priority units on a four-activity instance; the
cross-check in ``validation/validate_exact.py`` is what exposed it.

All three are reported with ``binding_candidate = False`` and are excluded from
the comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .simulate import Trace, simulate
from .timeline import Plan, RasterisedProblem

__all__ = ["ConstraintSlack", "FeasibilityReport", "evaluate"]

RESOURCE_CONSTRAINTS: tuple[str, ...] = (
    "battery-depth-of-discharge",
    "recorder-capacity",
    "recorder-drained",
)
"""The constraints that are limits on a physical resource rather than on time.

:attr:`FeasibilityReport.binding_resource` reports the tightest of these. It
exists because the objective gives no reward for finishing early, so an optimal
plan may place an activity arbitrarily late and leave a horizon or window slack
of one step. When that happens the ``horizon`` entry is genuinely the tightest
normalised slack and is genuinely binding -- extending the horizon would admit
more work -- but it is tie-breaking in the solver rather than a resource limit,
and a reader deciding what to change about the spacecraft wants both numbers.
The instance-family measurement in ``validation/validate_gap.py`` reports how
often each is named.
"""

_ABSOLUTE_TOLERANCE = 1e-9
"""Slack below ``-_ABSOLUTE_TOLERANCE * scale`` counts as a violation.

A strict ``slack >= 0`` test on a quantity accumulated over hundreds of
floating-point steps would turn a 1e-16 rounding into an infeasible verdict.
The tolerance is relative to each constraint's own declared scale and is
reported in the rendered output so that a reader can see what was allowed.
"""


@dataclass(frozen=True)
class ConstraintSlack:
    """Slack on one declared constraint.

    Attributes
    ----------
    name
        Identifier, e.g. ``"battery-depth-of-discharge"`` or
        ``"window:image-03"``.
    slack
        Remaining margin in ``unit``. Negative means violated.
    unit
        Physical unit of ``slack``, one of ``"W*h"``, ``"Mbit"``, ``"s"``.
    scale
        Declared normaliser, same unit as ``slack``, strictly positive.
    where
        Human-readable location of the tightest point, e.g. a step index and
        time, or an activity name.
    binding_candidate
        Whether this entry competes to be named the binding constraint.
        ``False`` for diagnostics that cannot be violated by construction.
    """

    name: str
    slack: float
    unit: str
    scale: float
    where: str
    binding_candidate: bool = True

    @property
    def normalised(self) -> float:
        """``slack / scale``, dimensionless. Negative means violated."""
        return self.slack / self.scale

    @property
    def satisfied(self) -> bool:
        """Whether the slack is non-negative to within the declared tolerance.

        The test is ``slack >= -1e-9 * scale``, not ``slack >= 0``.
        """
        return self.slack >= -_ABSOLUTE_TOLERANCE * self.scale


@dataclass(frozen=True)
class FeasibilityReport:
    """The auditable verdict on one plan.

    Attributes
    ----------
    plan, trace
        The plan evaluated and the resource trace it produced.
    slacks
        Every constraint, in declared evaluation order.
    placed, unplaced
        Activity names placed on the timeline and left off it.
    value
        Objective value: the sum of the placed activities' priorities.
    forgone_priority
        Sum of the unplaced activities' priorities.
    unplaceable
        Activities with no admissible start step at all, so no plan can place
        them. Reported separately because the cause is the declaration, not the
        planner.
    """

    plan: Plan
    trace: Trace
    slacks: tuple[ConstraintSlack, ...]
    placed: tuple[str, ...]
    unplaced: tuple[str, ...]
    value: float
    forgone_priority: float
    unplaceable: tuple[str, ...]

    @property
    def candidates(self) -> tuple[ConstraintSlack, ...]:
        """Constraints that compete to be binding, in declared order."""
        return tuple(entry for entry in self.slacks if entry.binding_candidate)

    @property
    def feasible(self) -> bool:
        """Whether every candidate constraint is satisfied within tolerance."""
        return all(entry.satisfied for entry in self.candidates)

    @property
    def violations(self) -> tuple[ConstraintSlack, ...]:
        """Candidate constraints that are not satisfied, tightest first."""
        bad = [entry for entry in self.candidates if not entry.satisfied]
        return tuple(sorted(bad, key=lambda e: e.normalised))

    @property
    def binding(self) -> ConstraintSlack:
        """The candidate constraint with the least normalised slack.

        For a feasible plan this is the constraint that stops the plan being
        improved in that direction. For an infeasible plan it is the worst
        violation. Ties are broken by declared evaluation order, which
        :func:`evaluate` fixes.
        """
        candidates = self.candidates
        if not candidates:
            raise ValueError("no candidate constraints were evaluated")
        best = candidates[0]
        for entry in candidates[1:]:
            if entry.normalised < best.normalised:
                best = entry
        return best

    @property
    def binding_resource(self) -> ConstraintSlack | None:
        """Tightest constraint among :data:`RESOURCE_CONSTRAINTS`, or ``None``.

        A secondary read of the binding constraint; see the note on
        :data:`RESOURCE_CONSTRAINTS` for why it is reported separately.
        """
        resources = [entry for entry in self.candidates if entry.name in RESOURCE_CONSTRAINTS]
        if not resources:
            return None
        best = resources[0]
        for entry in resources[1:]:
            if entry.normalised < best.normalised:
                best = entry
        return best

    def slack(self, name: str) -> ConstraintSlack:
        """The named constraint. Raises ``KeyError`` if it was not evaluated."""
        for entry in self.slacks:
            if entry.name == name:
                return entry
        raise KeyError(f"no constraint named {name!r} in this report")

    def energy_audit(self) -> dict[str, float]:
        """Closed-form against simulated energy accounting, all in W*h.

        Keys: ``generated_closed_form``, ``generated_steps``,
        ``fixed_closed_form``, ``consumed_steps``, ``activity_energy``,
        ``shunted``, ``battery_delta``, ``residual``. The residual is the
        closure of the battery recursion
        (:meth:`activityplan.simulate.Trace.energy_residual_wh`); the two
        ``generated`` entries compare the whole-horizon closed-form solar
        integral against the sum of the per-step closed-form integrals, which
        checks the orbit-accumulation arithmetic in
        :meth:`activityplan.model.SolarProfile.energy_wh` against its own
        per-step decomposition.
        """
        problem = self.trace
        generated_closed, fixed_closed = self._problem.baseline_energy_balance_wh()
        return {
            "generated_closed_form": generated_closed,
            "generated_steps": problem.generated_wh,
            "fixed_closed_form": fixed_closed,
            "fixed_steps": float(math.fsum(problem.fixed_load_wh.tolist())),
            "activity_energy": float(math.fsum(problem.activity_load_wh.tolist())),
            "consumed_steps": problem.consumed_wh,
            "shunted": problem.shunted_wh,
            "battery_delta": float(problem.battery_wh[-1]) - problem.initial_battery_wh,
            "residual": problem.energy_residual_wh(),
        }

    @property
    def _problem(self):  # noqa: ANN202 - internal accessor, typed via the dataclass
        return self.__dict__["_declared_problem"]


def _activity_interval(
    rasterised: RasterisedProblem, name: str, start_step: int
) -> tuple[float, float]:
    index = rasterised.problem.index(name)
    start_s = start_step * rasterised.grid.step_s
    return start_s, rasterised.finish_s(index, start_step)


def evaluate(rasterised: RasterisedProblem, plan: Plan) -> FeasibilityReport:
    """Simulate ``plan`` and evaluate every declared constraint on it.

    Parameters
    ----------
    rasterised
        The problem projected onto its grid.
    plan
        Start steps for the placed activities.

    Returns
    -------
    FeasibilityReport
        Slacks in declared evaluation order, the binding constraint, the
        objective value, and the energy audit.

    Notes
    -----
    The evaluation order is: battery depth of discharge, recorder capacity,
    downlink volume, horizon, activity windows (in declaration order),
    precedence pairs (in declaration order), exclusion pairs (in declaration
    order). Ties for the binding constraint are broken by that order.
    """
    problem = rasterised.problem
    resources = problem.resources
    trace = simulate(rasterised, plan)
    step = rasterised.grid.step_s

    entries: list[ConstraintSlack] = []

    floor = resources.floor_wh
    low_step = trace.min_battery_step
    entries.append(
        ConstraintSlack(
            name="battery-depth-of-discharge",
            slack=trace.min_battery_wh - floor,
            unit="W*h",
            scale=resources.usable_wh,
            where=(
                f"step {low_step} ending t={((low_step + 1) * step):.1f} s, "
                f"E={trace.min_battery_wh:.4f} W*h against floor {floor:.4f} W*h"
            ),
        )
    )
    high_step = int(trace.battery_wh.argmax())
    entries.append(
        ConstraintSlack(
            name="battery-upper-headroom",
            slack=resources.battery_capacity_wh - float(trace.battery_wh.max()),
            unit="W*h",
            scale=resources.battery_capacity_wh,
            where=(
                f"step {high_step}, E={float(trace.battery_wh.max()):.4f} W*h; "
                f"{trace.shunted_wh:.4f} W*h shunted over the horizon"
            ),
            binding_candidate=False,
        )
    )
    full_step = trace.max_recorder_step
    entries.append(
        ConstraintSlack(
            name="recorder-capacity",
            slack=resources.recorder_capacity_mbit - trace.max_recorder_mbit,
            unit="Mbit",
            scale=resources.recorder_capacity_mbit,
            where=(
                f"step {full_step} ending t={((full_step + 1) * step):.1f} s, "
                f"S={trace.max_recorder_mbit:.4f} Mbit of "
                f"{resources.recorder_capacity_mbit:.4f} Mbit"
            ),
        )
    )
    peak_step = int(trace.recorder_peak_mbit.argmax())
    entries.append(
        ConstraintSlack(
            name="recorder-within-step-peak",
            slack=resources.recorder_capacity_mbit - float(trace.recorder_peak_mbit.max()),
            unit="Mbit",
            scale=resources.recorder_capacity_mbit,
            where=(
                f"step {peak_step}, pre-downlink content "
                f"{float(trace.recorder_peak_mbit.max()):.4f} Mbit (diagnostic only)"
            ),
            binding_candidate=False,
        )
    )

    capacity = resources.total_contact_capacity_mbit
    demand = trace.total_produced_mbit + resources.initial_recorder_mbit
    entries.append(
        ConstraintSlack(
            name="downlink-volume",
            slack=capacity - demand,
            unit="Mbit",
            scale=max(capacity, 1e-12),
            where=(
                f"{demand:.4f} Mbit to downlink against {capacity:.4f} Mbit of "
                f"declared contact capacity over {len(resources.contacts)} contacts"
            ),
            binding_candidate=False,
        )
    )

    if problem.require_empty_recorder:
        # Degenerate by construction: for any plan that satisfies it the slack is
        # exactly zero, so when this entry is named binding it says only that the
        # option is switched on, not which resource limits the plan. Read the next
        # entry in the normalised-slack order for that. It remains a candidate
        # because a violation of it must be visible.
        residual = float(trace.recorder_mbit[-1])
        entries.append(
            ConstraintSlack(
                name="recorder-drained",
                slack=(-residual if residual > 0.0 else 0.0),
                unit="Mbit",
                scale=resources.recorder_capacity_mbit,
                where=(
                    f"{residual:.4f} Mbit still onboard at t={problem.horizon_s:.1f} s; "
                    "require_empty_recorder is set"
                ),
            )
        )

    last_finish = 0.0
    last_name = "none placed"
    for name, start_step in plan.starts.items():
        _, finish = _activity_interval(rasterised, name, start_step)
        if finish > last_finish:
            last_finish = finish
            last_name = name
    entries.append(
        ConstraintSlack(
            name="horizon",
            slack=problem.horizon_s - last_finish,
            unit="s",
            scale=problem.horizon_s,
            where=f"last finish t={last_finish:.1f} s ({last_name})",
        )
    )

    for activity in problem.activities:
        if activity.name not in plan.starts:
            continue
        start_s, finish_s = _activity_interval(
            rasterised, activity.name, plan.starts[activity.name]
        )
        latest = min(activity.latest_finish_s, problem.horizon_s)
        float_s = latest - activity.earliest_start_s - activity.duration_s
        entries.append(
            ConstraintSlack(
                name=f"window:{activity.name}",
                slack=latest - finish_s,
                unit="s",
                scale=float_s if float_s > 0.0 else activity.duration_s,
                where=(
                    f"placed [{start_s:.1f}, {finish_s:.1f}] s inside declared "
                    f"[{activity.earliest_start_s:.1f}, {latest:.1f}] s"
                ),
            )
        )

    for before, after in problem.precedences:
        if after not in plan.starts:
            continue
        if before not in plan.starts:
            entries.append(
                ConstraintSlack(
                    name=f"precedence:{before}->{after}",
                    slack=-problem.horizon_s,
                    unit="s",
                    scale=problem.activity(before).duration_s,
                    where=f"{after!r} is placed but its predecessor {before!r} is not",
                )
            )
            continue
        _, finish_before = _activity_interval(rasterised, before, plan.starts[before])
        start_after, _ = _activity_interval(rasterised, after, plan.starts[after])
        entries.append(
            ConstraintSlack(
                name=f"precedence:{before}->{after}",
                slack=start_after - finish_before,
                unit="s",
                scale=problem.activity(before).duration_s,
                where=(
                    f"{before} finishes t={finish_before:.1f} s, {after} starts "
                    f"t={start_after:.1f} s"
                ),
            )
        )

    for first, second in problem.exclusions:
        if first not in plan.starts or second not in plan.starts:
            continue
        a_start, a_finish = _activity_interval(rasterised, first, plan.starts[first])
        b_start, b_finish = _activity_interval(rasterised, second, plan.starts[second])
        separation = max(b_start - a_finish, a_start - b_finish)
        entries.append(
            ConstraintSlack(
                name=f"exclusion:{first}|{second}",
                slack=separation,
                unit="s",
                scale=min(
                    problem.activity(first).duration_s, problem.activity(second).duration_s
                ),
                where=(
                    f"{first} [{a_start:.1f}, {a_finish:.1f}] s, {second} "
                    f"[{b_start:.1f}, {b_finish:.1f}] s"
                ),
            )
        )

    placed = plan.placed
    unplaced = tuple(
        activity.name for activity in problem.activities if activity.name not in plan.starts
    )
    unplaceable = tuple(
        activity.name
        for position, activity in enumerate(problem.activities)
        if rasterised.allowed_starts[position].size == 0
    )
    report = FeasibilityReport(
        plan=plan,
        trace=trace,
        slacks=tuple(entries),
        placed=placed,
        unplaced=unplaced,
        value=plan.value(problem),
        forgone_priority=math.fsum(problem.activity(name).priority for name in unplaced),
        unplaceable=unplaceable,
    )
    object.__setattr__(report, "_declared_problem", problem)
    return report
