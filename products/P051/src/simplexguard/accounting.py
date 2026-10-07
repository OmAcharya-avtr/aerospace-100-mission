"""The assurance accounting. This is the deliverable.

A runtime guard that fires is easy to build and easy to oversell. The questions
a reviewer actually asks are quantitative, and this module answers exactly
those, from recorded episodes:

1. **Switch rate.** How often does authority change hands, per 1000 steps and
   per second of wall-clock plant time.
2. **Dwell-time distribution.** How long does each interval last, separately for
   baseline and performance authority. The mean hides the thing that matters:
   a guard that switches every other step is chattering, and chattering is
   visible only in the distribution.
3. **Fraction of the episode under baseline control.** The share of the mission
   spent on the conservative controller.
4. **Conservatism cost.** The tracking cost of the guarded run against the same
   performance controller run unguarded on the *same disturbance sequence*, and
   against the baseline alone, so the number sits between two measured extremes
   rather than against nothing.
5. **Constraint-violation count.** The number of recorded states outside the
   declared constraint polytope ``X``. Under the declared disturbance bound this
   must be zero. If it is not, the number is reported as it is; no tolerance is
   widened to make it zero.
6. **Invariant-set exits.** The number of recorded states outside ``S``. This is
   a strictly earlier symptom than a violation of ``X``, because ``S`` is a
   subset of ``X``, and it is the quantity that tells you the certificate has
   been lost rather than merely approached.

Nothing in this module is a safety argument. It is an audit of a finite number
of simulated episodes of one plant under one disturbance model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .guard import Mode
from .polytope import Polytope
from .simulate import Episode

__all__ = [
    "AssuranceReport",
    "DwellStatistics",
    "account",
    "dwell_statistics",
    "mode_runs",
]


def mode_runs(modes: np.ndarray) -> list[tuple[int, int, int]]:
    """Split a mode sequence into runs of equal value.

    Returns a list of ``(mode_value, start_index, length)`` covering the whole
    sequence in order. An empty input gives an empty list.
    """
    m = np.asarray(modes, dtype=int).ravel()
    if m.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(m)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [m.size]])
    return [(int(m[s]), int(s), int(e - s)) for s, e in zip(starts, ends, strict=True)]


@dataclass(frozen=True)
class DwellStatistics:
    """Dwell-time summary for one authority, in steps and in seconds."""

    n_intervals: int
    total_steps: int
    minimum: int
    median: float
    mean: float
    maximum: int
    fraction_of_length_one: float
    dt: float

    @property
    def mean_seconds(self) -> float:
        """Mean dwell in seconds."""
        return self.mean * self.dt

    def histogram(self, lengths: Sequence[int], max_bin: int = 10) -> dict[str, int]:
        """Counts of dwell lengths 1, 2, ... ``max_bin`` and a ``>max_bin`` bin."""
        arr = np.asarray(list(lengths), dtype=int)
        out = {str(k): int(np.sum(arr == k)) for k in range(1, int(max_bin) + 1)}
        out[f">{int(max_bin)}"] = int(np.sum(arr > int(max_bin)))
        return out


def dwell_statistics(lengths: Sequence[int], dt: float = 1.0) -> DwellStatistics:
    """Summarise a list of dwell lengths in steps."""
    arr = np.asarray(list(lengths), dtype=int)
    if arr.size == 0:
        return DwellStatistics(0, 0, 0, 0.0, 0.0, 0, 0.0, float(dt))
    if np.any(arr < 1):
        raise ValueError("dwell lengths must be at least 1 step")
    return DwellStatistics(
        n_intervals=int(arr.size),
        total_steps=int(arr.sum()),
        minimum=int(arr.min()),
        median=float(np.median(arr)),
        mean=float(arr.mean()),
        maximum=int(arr.max()),
        fraction_of_length_one=float(np.mean(arr == 1)),
        dt=float(dt),
    )


@dataclass(frozen=True)
class AssuranceReport:
    """The six accounting quantities, for one guarded episode with its pair runs."""

    n_steps: int
    dt: float
    # 1 switch rate
    n_switches: int
    switches_per_1000_steps: float
    switches_per_second: float
    # 2 dwell
    baseline_dwell: DwellStatistics
    performance_dwell: DwellStatistics
    baseline_dwell_lengths: tuple[int, ...]
    performance_dwell_lengths: tuple[int, ...]
    # 3 authority share
    baseline_fraction: float
    forced_by_dwell_steps: int
    # 4 conservatism
    guarded_cost: float
    unguarded_cost: float
    baseline_cost: float
    conservatism_cost_ratio: float
    conservatism_cost_absolute: float
    performance_gap_recovered: float
    # 5 violations
    guarded_violations: int
    unguarded_violations: int
    baseline_violations: int
    guarded_worst_residual: float
    unguarded_worst_residual: float
    # 6 invariant exits
    guarded_invariant_exits: int
    unguarded_invariant_exits: int
    # margins
    margin_min: float
    margin_median: float
    margin_at_switch_median: float
    cost_weights_state: tuple[float, ...]
    cost_weights_input: tuple[float, ...]

    def describe(self) -> str:
        """Fixed-width report, used verbatim by the CLI and validation scripts."""
        b, p = self.baseline_dwell, self.performance_dwell
        lines = [
            f"steps                            {self.n_steps}  "
            f"({self.n_steps * self.dt:.1f} s at dt = {self.dt:g} s)",
            "",
            "1  switch rate",
            f"   authority changes             {self.n_switches}",
            f"   per 1000 steps                {self.switches_per_1000_steps:.3f}",
            f"   per second                    {self.switches_per_second:.4f}",
            "",
            "2  dwell-time distribution (steps)",
            f"   baseline   n={b.n_intervals:<6d} min={b.minimum:<4d} "
            f"median={b.median:<7.2f} mean={b.mean:<8.3f} max={b.maximum:<5d} "
            f"len-1 fraction={b.fraction_of_length_one:.3f}",
            f"   performance n={p.n_intervals:<5d} min={p.minimum:<4d} "
            f"median={p.median:<7.2f} mean={p.mean:<8.3f} max={p.maximum:<5d} "
            f"len-1 fraction={p.fraction_of_length_one:.3f}",
            f"   baseline dwell histogram      "
            f"{b.histogram(self.baseline_dwell_lengths)}",
            "",
            "3  authority share",
            f"   fraction under baseline       {self.baseline_fraction:.6f}",
            f"   steps held by min-dwell       {self.forced_by_dwell_steps}",
            "",
            "4  conservatism cost "
            f"(weights state={list(self.cost_weights_state)}, "
            f"input={list(self.cost_weights_input)})",
            f"   guarded cost                  {self.guarded_cost:.6f}",
            f"   unguarded cost                {self.unguarded_cost:.6f}",
            f"   baseline-only cost            {self.baseline_cost:.6f}",
            f"   guarded / unguarded           {self.conservatism_cost_ratio:.6f}",
            f"   guarded - unguarded           {self.conservatism_cost_absolute:.6f}",
            f"   fraction of the baseline-to-unguarded gap recovered  "
            f"{self.performance_gap_recovered:.6f}",
            "",
            "5  constraint violations against the declared X",
            f"   guarded                       {self.guarded_violations}",
            f"   unguarded                     {self.unguarded_violations}",
            f"   baseline only                 {self.baseline_violations}",
            f"   guarded worst row residual    {self.guarded_worst_residual:.9e}",
            f"   unguarded worst row residual  {self.unguarded_worst_residual:.9e}",
            "",
            "6  invariant-set exits (certificate lost)",
            f"   guarded                       {self.guarded_invariant_exits}",
            f"   unguarded                     {self.unguarded_invariant_exits}",
            "",
            "   switching-condition margin",
            f"   minimum over the episode      {self.margin_min:.9e}",
            f"   median over the episode       {self.margin_median:.9e}",
            f"   median at the firing steps    {self.margin_at_switch_median:.9e}",
        ]
        return "\n".join(lines)


def account(
    guarded: Episode,
    unguarded: Episode,
    baseline_only: Episode,
    invariant_set: Polytope,
    dt: float | None = None,
) -> AssuranceReport:
    """Compute the six accounting quantities.

    Parameters
    ----------
    guarded, unguarded, baseline_only :
        Episodes produced by :mod:`simplexguard.simulate` on the **same**
        disturbance sequence. The function checks that the sequences match and
        raises if they do not, because an unpaired conservatism cost is not the
        quantity this report claims to compute.
    invariant_set :
        ``S``, for the exit count.
    dt :
        Sample interval in seconds; defaults to the guarded episode's plant.
    """
    if guarded.n_steps != unguarded.n_steps or guarded.n_steps != baseline_only.n_steps:
        raise ValueError("the three episodes must have the same length")
    if not np.allclose(guarded.disturbances, unguarded.disturbances):
        raise ValueError(
            "guarded and unguarded episodes were run on different disturbance "
            "sequences; the conservatism cost would not be a paired comparison"
        )
    if not np.allclose(guarded.disturbances, baseline_only.disturbances):
        raise ValueError("baseline-only episode was run on a different disturbance sequence")
    step = float(guarded.plant.dt if dt is None else dt)
    modes = guarded.modes
    n = guarded.n_steps
    n_switches = int(np.sum(modes[1:] != modes[:-1])) if n > 1 else 0
    runs = mode_runs(modes)
    base_lengths = [length for value, _, length in runs if value == Mode.BASELINE.value]
    perf_lengths = [length for value, _, length in runs if value == Mode.PERFORMANCE.value]
    cost_g, cost_u, cost_b = guarded.cost(), unguarded.cost(), baseline_only.cost()
    gap = cost_b - cost_u
    recovered = float((cost_b - cost_g) / gap) if abs(gap) > 0.0 else float("nan")
    fires = modes == Mode.BASELINE.value
    margin_at_fire = guarded.margins[fires]
    return AssuranceReport(
        n_steps=n,
        dt=step,
        n_switches=n_switches,
        switches_per_1000_steps=1000.0 * n_switches / n,
        switches_per_second=n_switches / (n * step),
        baseline_dwell=dwell_statistics(base_lengths, step),
        performance_dwell=dwell_statistics(perf_lengths, step),
        baseline_dwell_lengths=tuple(base_lengths),
        performance_dwell_lengths=tuple(perf_lengths),
        baseline_fraction=guarded.baseline_fraction,
        forced_by_dwell_steps=int(np.sum(guarded.forced_by_dwell)),
        guarded_cost=cost_g,
        unguarded_cost=cost_u,
        baseline_cost=cost_b,
        conservatism_cost_ratio=float(cost_g / cost_u) if cost_u > 0.0 else float("nan"),
        conservatism_cost_absolute=float(cost_g - cost_u),
        performance_gap_recovered=recovered,
        guarded_violations=int(guarded.constraint_violations().size),
        unguarded_violations=int(unguarded.constraint_violations().size),
        baseline_violations=int(baseline_only.constraint_violations().size),
        guarded_worst_residual=guarded.worst_constraint_residual(),
        unguarded_worst_residual=unguarded.worst_constraint_residual(),
        guarded_invariant_exits=int(guarded.invariant_exits(invariant_set).size),
        unguarded_invariant_exits=int(unguarded.invariant_exits(invariant_set).size),
        margin_min=float(np.min(guarded.margins)) if n else float("nan"),
        margin_median=float(np.median(guarded.margins)) if n else float("nan"),
        margin_at_switch_median=(
            float(np.median(margin_at_fire)) if margin_at_fire.size else float("nan")
        ),
        cost_weights_state=tuple(guarded.weights.state),
        cost_weights_input=tuple(guarded.weights.input),
    )
