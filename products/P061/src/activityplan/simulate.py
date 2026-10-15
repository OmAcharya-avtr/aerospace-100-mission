"""Explicit resource simulation of a plan on the discretised timeline.

One simulator, used by the greedy planner to test every candidate start, by the
feasibility report, and by the validation scripts to check that a plan the MILP
declared feasible really is. Nothing else in the package integrates a resource.

Battery model, stated in full
-----------------------------
Per step ``k`` of length ``dt`` [s], with all energies in watt-hours::

    surplus_k   = solar_k - load_k
    delta_k     = charge_efficiency * surplus_k   if surplus_k > 0
                = surplus_k                        otherwise
    raw_k       = E_{k-1} + delta_k
    shunt_k     = max(0, raw_k - capacity)
    E_k         = min(raw_k, capacity)

with ``E_{-1}`` the declared initial energy. Three properties of this recursion
matter and are stated rather than left to be discovered:

1. **The upper clamp is a shunt, not a loss of information.** Surplus array
   power above a full battery is dissipated, which is what a shunt regulator
   does (Patel 2004, chapter 6). ``shunt_k`` records how much.
2. **There is no lower clamp.** If ``E_k`` falls below the depth-of-discharge
   floor, or below zero, the simulation records it and the feasibility report
   calls the plan infeasible. Clamping at the floor would hide exactly the
   failure this product exists to report.
3. **The clamp policy is pointwise maximal in ``E``.** Shunting only when the
   battery is full is the least shunting any feasible policy can do, so for any
   other admissible shunt schedule the clamped trace has ``E_k`` at least as
   large at every step. Since the only battery constraint is a lower bound, a
   plan that is feasible under *some* shunt schedule is feasible under this one.
   That is what lets the MILP, which treats shunting as a free continuous
   decision, hand its plan to this simulator without the verdict changing. The
   property is asserted as a lemma in ``tests/test_simulate.py`` and measured
   over the whole instance family in ``validation/validate_gap.py``.

Recorder model
--------------
Per step, production is added first, then as much as the contact rate allows is
downlinked::

    S_pre_k = S_{k-1} + produced_k
    down_k  = min(S_pre_k, downlink_cap_k)
    S_k     = S_pre_k - down_k

The capacity constraint is imposed on the end-of-step content ``S_k``, because
the step is the declared accounting interval. The within-step peak ``S_pre_k``
is reported as a diagnostic and is not constrained; a user who needs the
instantaneous content bounded must choose a smaller step. Draining as fast as
the contact allows is pointwise minimal in ``S``, the mirror of property 3
above, so the same dominance argument applies to the recorder.

Validity range
--------------
No voltage, no internal resistance, no temperature, no self-discharge, no
battery ageing, no packet overhead, no retransmission, no downlink
acknowledgement latency. Valid as an energy and volume accounting over a horizon
of a few orbits; not valid as a power-system or link simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .timeline import Plan, RasterisedProblem

__all__ = ["Trace", "simulate"]


@dataclass(frozen=True)
class Trace:
    """Per-step resource trace of one plan. All arrays have length ``n_steps``.

    Attributes
    ----------
    solar_wh
        Exact closed-form array energy generated in each step [W*h].
    fixed_load_wh
        Energy consumed by bus and transmitter loads in each step [W*h].
    activity_load_wh
        Energy consumed by the placed activities in each step [W*h].
    battery_wh
        Battery energy at the *end* of each step [W*h]. May fall below the
        declared floor, and below zero; that is reported, not clamped.
    shunt_wh
        Surplus energy dissipated in each step because the battery was full
        [W*h].
    produced_mbit
        Data written to the recorder in each step [Mbit].
    downlinked_mbit
        Data downlinked in each step [Mbit].
    recorder_mbit
        Recorder content at the *end* of each step [Mbit].
    recorder_peak_mbit
        Within-step content after production and before downlink [Mbit]; a
        diagnostic, not a constrained quantity.
    """

    grid_step_s: float
    solar_wh: np.ndarray
    fixed_load_wh: np.ndarray
    activity_load_wh: np.ndarray
    battery_wh: np.ndarray
    shunt_wh: np.ndarray
    produced_mbit: np.ndarray
    downlinked_mbit: np.ndarray
    recorder_mbit: np.ndarray
    recorder_peak_mbit: np.ndarray
    initial_battery_wh: float
    initial_recorder_mbit: float
    effective_generated_wh: float

    @property
    def n_steps(self) -> int:
        """Number of steps in the trace."""
        return int(self.battery_wh.size)

    @property
    def total_load_wh(self) -> np.ndarray:
        """Fixed plus activity load per step [W*h]."""
        return self.fixed_load_wh + self.activity_load_wh

    @property
    def min_battery_wh(self) -> float:
        """Lowest end-of-step battery energy over the horizon [W*h]."""
        return float(self.battery_wh.min())

    @property
    def min_battery_step(self) -> int:
        """Index of the step at which the battery is lowest."""
        return int(np.argmin(self.battery_wh))

    @property
    def max_recorder_mbit(self) -> float:
        """Largest end-of-step recorder content over the horizon [Mbit]."""
        return float(self.recorder_mbit.max())

    @property
    def max_recorder_step(self) -> int:
        """Index of the step at which the recorder is fullest."""
        return int(np.argmax(self.recorder_mbit))

    @property
    def generated_wh(self) -> float:
        """Total array energy generated over the horizon [W*h]."""
        return float(math.fsum(self.solar_wh.tolist()))

    @property
    def consumed_wh(self) -> float:
        """Total energy consumed by all loads over the horizon [W*h]."""
        return float(math.fsum(self.total_load_wh.tolist()))

    @property
    def shunted_wh(self) -> float:
        """Total surplus energy dissipated over the horizon [W*h]."""
        return float(math.fsum(self.shunt_wh.tolist()))

    @property
    def total_downlinked_mbit(self) -> float:
        """Total data downlinked over the horizon [Mbit]."""
        return float(math.fsum(self.downlinked_mbit.tolist()))

    @property
    def total_produced_mbit(self) -> float:
        """Total data written to the recorder over the horizon [Mbit]."""
        return float(math.fsum(self.produced_mbit.tolist()))

    def energy_residual_wh(self) -> float:
        """Closure residual of the battery energy balance [W*h].

        ``(E_final - E_initial) - (effective_generated - consumed - shunted)``.
        Zero up to floating-point accumulation for any trace this module
        produces; it is returned so that ``validation/validate_energy.py`` can
        report the measured value rather than assert a property nobody checked.

        ``effective_generated_wh`` is the generation after the
        ``charge_efficiency`` weighting, which applies only to surplus steps.
        With ``charge_efficiency = 1`` it equals :attr:`generated_wh` and the
        residual is the plain ``generated - consumed - shunted`` closure.
        """
        delta = float(self.battery_wh[-1]) - self.initial_battery_wh
        flux = self.effective_generated_wh - self.consumed_wh - self.shunted_wh
        return delta - flux


def simulate(rasterised: RasterisedProblem, plan: Plan) -> Trace:
    """Simulate the declared resources under ``plan``.

    Parameters
    ----------
    rasterised
        The problem projected onto a grid by
        :meth:`activityplan.timeline.RasterisedProblem.build`.
    plan
        Start steps for the placed activities. Activities absent from the plan
        draw no power and produce no data.

    Returns
    -------
    Trace
        Per-step energy and data arrays. No constraint is evaluated here;
        :func:`activityplan.feasibility.evaluate` does that.

    Raises
    ------
    ValueError
        If a placed activity would run past the end of the grid.
    """
    resources = rasterised.problem.resources
    n_steps = rasterised.n_steps
    activity_load = rasterised.activity_load_wh(plan)
    produced = rasterised.activity_data_production_mbit(plan)
    solar = rasterised.solar_wh
    fixed = rasterised.fixed_wh
    caps = rasterised.downlink_cap_mbit

    capacity = resources.battery_capacity_wh
    efficiency = resources.charge_efficiency

    battery = np.empty(n_steps, dtype=float)
    shunt = np.zeros(n_steps, dtype=float)
    recorder = np.empty(n_steps, dtype=float)
    downlinked = np.empty(n_steps, dtype=float)
    peak = np.empty(n_steps, dtype=float)

    energy = resources.initial_energy_wh
    stored = resources.initial_recorder_mbit
    effective_generated = 0.0

    for k in range(n_steps):
        load = fixed[k] + activity_load[k]
        surplus = solar[k] - load
        if surplus > 0.0:
            delta = efficiency * surplus
            effective_generated += load + efficiency * surplus
        else:
            delta = surplus
            effective_generated += solar[k]
        raw = energy + delta
        if raw > capacity:
            shunt[k] = raw - capacity
            energy = capacity
        else:
            energy = raw
        battery[k] = energy

        pre = stored + produced[k]
        peak[k] = pre
        sent = min(pre, caps[k])
        downlinked[k] = sent
        stored = pre - sent
        recorder[k] = stored

    trace = Trace(
        grid_step_s=rasterised.grid.step_s,
        solar_wh=solar,
        fixed_load_wh=fixed,
        activity_load_wh=activity_load,
        battery_wh=battery,
        shunt_wh=shunt,
        produced_mbit=produced,
        downlinked_mbit=downlinked,
        recorder_mbit=recorder,
        recorder_peak_mbit=peak,
        initial_battery_wh=resources.initial_energy_wh,
        initial_recorder_mbit=resources.initial_recorder_mbit,
        effective_generated_wh=effective_generated,
    )
    return trace
