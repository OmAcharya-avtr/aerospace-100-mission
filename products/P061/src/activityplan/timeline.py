"""Discretisation of a declared problem onto a uniform step grid.

The discretisation is the single largest modelling decision in this package and
it is made in exactly one place, here, so that the greedy planner, the exact
MILP and the resource simulator cannot disagree about it. All three consume the
arrays built by :class:`RasterisedProblem`; none of them re-derives a quantity
from the continuous declaration.

The rules, stated in full
-------------------------
1. **The horizon must be an integer multiple of the step.** A non-integer ratio
   is refused with ``ValueError`` rather than rounded, because a silently
   shortened or lengthened horizon changes every slack in the feasibility
   report.
2. **Start times are restricted to grid points.** This is the discretisation.
   An activity whose optimal continuous start lies between two grid points
   cannot be expressed, so the discretised optimum is a lower bound on the
   continuous optimum for a maximisation problem. That bound is what this
   package reports; it never claims the continuous optimum.
3. **Occupancy is rounded up.** An activity of duration ``D`` occupies
   ``ceil(D / step)`` steps for the purpose of exclusion constraints. This is
   conservative: it can forbid an overlap that continuous time would allow, and
   can never permit one that continuous time forbids.
4. **Energy and data are exact, not sampled.** The energy an activity draws in
   a step is the closed-form integral of its piecewise-constant power profile
   over that step (:meth:`activityplan.model.Activity.energy_between_wh`), so
   the per-step energies sum to the activity's exact energy. Data production is
   apportioned by time overlap, so the per-step volumes sum to the declared
   ``data_mbit``. The consequence of rule 3 is that the final, partially
   occupied step of an off-grid activity carries a *reduced average* power:
   peak power within that step is understated by up to
   ``(1 - frac) * p_last`` where ``frac`` is the occupied fraction. This package
   constrains energy, not instantaneous power, so that understatement affects no
   constraint; it is recorded here because it would matter to anyone who added a
   peak-power constraint.
5. **Solar energy per step is the exact closed-form integral** over the step,
   not a sample. The midpoint-sampled alternative is also provided, as
   :meth:`Grid.solar_step_wh_midpoint`, solely so that
   ``validation/validate_energy.py`` can measure what sampling would cost.
6. **Windows are tightened, never loosened.** The earliest allowed start step is
   ``ceil(earliest_start_s / step)`` and the latest is
   ``floor((min(latest_finish_s, horizon_s) - duration_s) / step)``. Both move
   inward, so no plan this package emits violates a declared window, and some
   legal continuous plans are unreachable. The lost window float is reported by
   :meth:`RasterisedProblem.window_tightening_s`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import WH_PER_J, Problem

__all__ = ["Grid", "Plan", "RasterisedProblem"]

_GRID_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Grid:
    """A uniform time grid of ``n_steps`` steps of ``step_s`` seconds each.

    Step ``k`` covers ``[k * step_s, (k + 1) * step_s)`` seconds from the start
    of the horizon.
    """

    step_s: float
    n_steps: int

    def __post_init__(self) -> None:
        step = float(self.step_s)
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError(f"step_s must be finite and strictly positive, got {self.step_s!r}")
        count = int(self.n_steps)
        if count != self.n_steps or count <= 0:
            raise ValueError(f"n_steps must be a positive integer, got {self.n_steps!r}")
        object.__setattr__(self, "step_s", step)
        object.__setattr__(self, "n_steps", count)

    @classmethod
    def for_horizon(cls, horizon_s: float, step_s: float) -> Grid:
        """Grid covering ``horizon_s`` [s] exactly in steps of ``step_s`` [s].

        Raises ``ValueError`` if ``horizon_s / step_s`` is not an integer to
        within a relative tolerance of 1e-9. Rounding is refused, not applied.
        """
        ratio = float(horizon_s) / float(step_s)
        rounded = round(ratio)
        if rounded < 1 or abs(ratio - rounded) > _GRID_TOLERANCE * max(1.0, abs(ratio)):
            raise ValueError(
                f"horizon_s {horizon_s!r} is not an integer multiple of step_s "
                f"{step_s!r} (ratio {ratio!r}); choose a step that divides the horizon"
            )
        return cls(step_s=float(step_s), n_steps=int(rounded))

    @property
    def horizon_s(self) -> float:
        """Grid span [s] = ``step_s * n_steps``."""
        return self.step_s * self.n_steps

    def edges_s(self) -> np.ndarray:
        """Step boundaries [s], shape ``(n_steps + 1,)``."""
        return np.arange(self.n_steps + 1, dtype=float) * self.step_s

    def starts_s(self) -> np.ndarray:
        """Left edge of each step [s], shape ``(n_steps,)``."""
        return np.arange(self.n_steps, dtype=float) * self.step_s

    def midpoints_s(self) -> np.ndarray:
        """Midpoint of each step [s], shape ``(n_steps,)``."""
        return (np.arange(self.n_steps, dtype=float) + 0.5) * self.step_s

    def steps_for(self, duration_s: float) -> int:
        """Number of steps an activity of ``duration_s`` [s] occupies, rounded up."""
        if duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive, got {duration_s!r}")
        exact = duration_s / self.step_s
        rounded = round(exact)
        if abs(exact - rounded) <= _GRID_TOLERANCE * max(1.0, abs(exact)):
            return max(1, int(rounded))
        return int(math.ceil(exact))


@dataclass(frozen=True)
class Plan:
    """An assignment of start steps to a subset of the activities.

    ``starts`` maps activity name to the integer start step on ``grid``. An
    activity absent from the mapping is not placed; in an oversubscribed problem
    that is the normal case and not an error.
    """

    starts: dict[str, int]
    grid: Grid

    def __post_init__(self) -> None:
        cleaned = {str(name): int(step) for name, step in self.starts.items()}
        for name, step in cleaned.items():
            if step < 0 or step >= self.grid.n_steps:
                raise ValueError(
                    f"activity {name!r} start step {step} is outside the grid "
                    f"[0, {self.grid.n_steps})"
                )
        object.__setattr__(self, "starts", cleaned)

    @property
    def placed(self) -> tuple[str, ...]:
        """Names of the placed activities, ordered by start step then name."""
        return tuple(sorted(self.starts, key=lambda n: (self.starts[n], n)))

    @property
    def n_placed(self) -> int:
        """Number of placed activities."""
        return len(self.starts)

    def start_s(self, name: str) -> float:
        """Start time [s] of a placed activity."""
        return self.starts[name] * self.grid.step_s

    def with_activity(self, name: str, start_step: int) -> Plan:
        """A new plan with ``name`` placed at ``start_step``. Does not mutate."""
        updated = dict(self.starts)
        updated[name] = int(start_step)
        return Plan(updated, self.grid)

    def without_activity(self, name: str) -> Plan:
        """A new plan with ``name`` removed. Does not mutate."""
        updated = dict(self.starts)
        updated.pop(name, None)
        return Plan(updated, self.grid)

    def value(self, problem: Problem) -> float:
        """Objective value: sum of the priorities of the placed activities."""
        return math.fsum(problem.activity(name).priority for name in self.starts)


@dataclass(frozen=True)
class RasterisedProblem:
    """A problem projected onto a grid, with every per-step array precomputed.

    Attributes
    ----------
    problem, grid
        The declaration and the grid it was projected onto.
    activity_energy_wh
        One array per activity, length ``n_occupied[i]``, the exact energy [W*h]
        the activity draws in each step after its start.
    activity_data_mbit
        One array per activity, length ``n_occupied[i]``, data [Mbit] produced in
        each step after its start.
    n_occupied
        ``ceil(duration_s / step_s)`` per activity, shape ``(n_activities,)``.
    allowed_starts
        One integer array per activity of the start steps the declared window and
        the horizon permit. May be empty, in which case the activity can never be
        placed and the feasibility report says so.
    solar_wh
        Exact closed-form array energy [W*h] generated in each step.
    fixed_wh
        Energy [W*h] consumed in each step by loads that no plan can change: the
        bus load over the whole step plus any transmitter load over the part of
        the step inside a contact window.
    downlink_cap_mbit
        Maximum data [Mbit] downlinkable in each step, ``rate * overlap``.
    """

    problem: Problem
    grid: Grid
    activity_energy_wh: tuple[np.ndarray, ...]
    activity_data_mbit: tuple[np.ndarray, ...]
    n_occupied: np.ndarray
    allowed_starts: tuple[np.ndarray, ...]
    solar_wh: np.ndarray
    fixed_wh: np.ndarray
    downlink_cap_mbit: np.ndarray

    @classmethod
    def build(cls, problem: Problem, step_s: float) -> RasterisedProblem:
        """Project ``problem`` onto a grid of ``step_s`` [s] steps.

        Raises ``ValueError`` if the horizon is not an integer multiple of the
        step (see :meth:`Grid.for_horizon`).
        """
        grid = Grid.for_horizon(problem.horizon_s, step_s)
        step = grid.step_s
        n_steps = grid.n_steps

        energies: list[np.ndarray] = []
        volumes: list[np.ndarray] = []
        occupied: list[int] = []
        starts: list[np.ndarray] = []
        for activity in problem.activities:
            n_occ = grid.steps_for(activity.duration_s)
            occupied.append(n_occ)
            per_step_energy = np.empty(n_occ, dtype=float)
            per_step_data = np.empty(n_occ, dtype=float)
            duration = activity.duration_s
            rate = activity.data_rate_mbit_per_s
            for k in range(n_occ):
                lo = k * step
                hi = min((k + 1) * step, duration)
                per_step_energy[k] = activity.energy_between_wh(lo, hi)
                per_step_data[k] = rate * max(0.0, hi - lo)
            energies.append(per_step_energy)
            volumes.append(per_step_data)

            first = int(math.ceil(activity.earliest_start_s / step - _GRID_TOLERANCE))
            latest_finish = min(activity.latest_finish_s, problem.horizon_s)
            last = int(
                math.floor((latest_finish - duration) / step + _GRID_TOLERANCE)
            )
            last = min(last, n_steps - n_occ)
            first = max(first, 0)
            if last < first:
                starts.append(np.empty(0, dtype=int))
            else:
                starts.append(np.arange(first, last + 1, dtype=int))

        solar = np.array(
            [problem.resources.solar.energy_wh(k * step, (k + 1) * step) for k in range(n_steps)],
            dtype=float,
        )
        fixed = np.full(n_steps, problem.resources.bus_load_w * step * WH_PER_J, dtype=float)
        caps = np.zeros(n_steps, dtype=float)
        for contact in problem.resources.contacts:
            for k in range(n_steps):
                lo = k * step
                hi = (k + 1) * step
                overlap = min(hi, contact.end_s) - max(lo, contact.start_s)
                if overlap > 0.0:
                    fixed[k] += contact.power_w * overlap * WH_PER_J
                    caps[k] += contact.rate_mbit_per_s * overlap

        return cls(
            problem=problem,
            grid=grid,
            activity_energy_wh=tuple(energies),
            activity_data_mbit=tuple(volumes),
            n_occupied=np.array(occupied, dtype=int),
            allowed_starts=tuple(starts),
            solar_wh=solar,
            fixed_wh=fixed,
            downlink_cap_mbit=caps,
        )

    @property
    def n_steps(self) -> int:
        """Number of steps on the grid."""
        return self.grid.n_steps

    @property
    def n_activities(self) -> int:
        """Number of requested activities."""
        return self.problem.n_activities

    def occupancy(self, index: int, start_step: int) -> range:
        """Step indices occupied by activity ``index`` started at ``start_step``."""
        return range(start_step, start_step + int(self.n_occupied[index]))

    def finish_s(self, index: int, start_step: int) -> float:
        """Continuous finish time [s] = ``start_step * step_s + duration_s``.

        The *continuous* duration is used, not the rounded-up occupancy, because
        the declared window constraint is a statement about real time.
        """
        return start_step * self.grid.step_s + self.problem.activities[index].duration_s

    def window_tightening_s(self) -> tuple[float, ...]:
        """Window float lost to grid snapping, per activity [s].

        For each activity, the continuous window float
        ``min(latest_finish, horizon) - earliest_start - duration`` minus the
        discretised float ``(last_allowed - first_allowed) * step``. Zero when
        both window edges happen to land on grid points. Reported because rule 6
        of the module docstring tightens windows inward and the amount should be
        visible rather than inferred.
        """
        result: list[float] = []
        for position, activity in enumerate(self.problem.activities):
            latest_finish = min(activity.latest_finish_s, self.problem.horizon_s)
            continuous = latest_finish - activity.earliest_start_s - activity.duration_s
            allowed = self.allowed_starts[position]
            discrete = 0.0 if allowed.size == 0 else (allowed[-1] - allowed[0]) * self.grid.step_s
            result.append(max(0.0, continuous - discrete))
        return tuple(result)

    def solar_step_wh_midpoint(self) -> np.ndarray:
        """Midpoint-sampled solar energy per step [W*h], for validation only.

        ``p_in(t_mid) * step_s / 3600``. This is the second-order accurate
        midpoint rule; ``solar_wh`` is exact. The difference, and its observed
        convergence order under step halving, are measured in
        ``validation/validate_energy.py``. Never used by the planners.
        """
        solar = self.problem.resources.solar
        mids = self.grid.midpoints_s()
        sampled = np.array([solar.power_at(t) for t in mids], dtype=float)
        return sampled * self.grid.step_s * WH_PER_J

    def activity_load_wh(self, plan: Plan) -> np.ndarray:
        """Per-step energy [W*h] drawn by the placed activities of ``plan``."""
        load = np.zeros(self.n_steps, dtype=float)
        for name, start in plan.starts.items():
            index = self.problem.index(name)
            profile = self.activity_energy_wh[index]
            end = start + profile.size
            if end > self.n_steps:
                raise ValueError(
                    f"activity {name!r} started at step {start} occupies {profile.size} "
                    f"steps and runs past the {self.n_steps}-step grid"
                )
            load[start:end] += profile
        return load

    def activity_data_production_mbit(self, plan: Plan) -> np.ndarray:
        """Per-step data production [Mbit] by the placed activities of ``plan``."""
        produced = np.zeros(self.n_steps, dtype=float)
        for name, start in plan.starts.items():
            index = self.problem.index(name)
            profile = self.activity_data_mbit[index]
            end = start + profile.size
            if end > self.n_steps:
                raise ValueError(
                    f"activity {name!r} started at step {start} occupies {profile.size} "
                    f"steps and runs past the {self.n_steps}-step grid"
                )
            produced[start:end] += profile
        return produced
