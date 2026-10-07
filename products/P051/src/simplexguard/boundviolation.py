"""The deliberate bound-violation experiment.

Every safety statement in this package is conditional on the realised
disturbance lying inside the **declared** set ``W``. In use, that assumption is
the thing most likely to be wrong: a declared bound comes from an analysis, a
wind-tunnel campaign or an engineering judgement, and the plant does not read
the document. A guard whose assumption is false is therefore the realistic
failure mode, and this module measures it rather than asserting it away.

Method
------
Run the guarded architecture unchanged -- the invariant set, the eroded set and
the switching condition are all still computed from the **declared** ``W`` -- but
draw the realised disturbance from ``rho * W`` for a sweep of ``rho >= 1``. For
each ``rho`` and over several seeds, report:

* the number of recorded states outside the declared constraint set ``X``;
* the number of recorded states outside the invariant set ``S``, which is a
  strictly earlier symptom because ``S`` is a subset of ``X``;
* the worst constraint residual, in state units, so "how badly" has a number;
* the fraction of steps at which the realised disturbance was itself outside
  ``W``, which says how hard the assumption was actually broken;
* the baseline authority fraction and switch rate, which rise as the guard
  fights the larger disturbance.

What the numbers mean
--------------------
At ``rho = 1`` the guarantee applies and the violation count must be zero. Above
``rho = 1`` nothing is guaranteed, and the observed behaviour is **not** a
property of the Simplex architecture in general: it is a property of this plant,
this baseline, this invariant set and this disturbance sampler. The slack between
``S`` and ``X`` is an artefact of the invariant-set recursion, not a designed
safety margin, and the sweep measures how much of it there happens to be. Any
other plant will give other numbers.

The honest headline this experiment produces is that the **certificate** fails
before the **constraint** does: there is a band of ``rho`` in which the state
leaves ``S``, so the recursive invariance argument no longer holds and the guard
is operating outside its own theory, while ``X`` is still satisfied and an
observer watching only the constraints would see nothing wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .controllers import PerformanceController
from .guard import Mode, SimplexGuard
from .plant import Plant
from .polytope import Polytope
from .simulate import CostWeights, disturbance_sequence, simulate_guarded, square_wave_reference

__all__ = ["BoundSweep", "BoundViolationPoint", "bound_violation_sweep"]


@dataclass(frozen=True)
class BoundViolationPoint:
    """Aggregate over seeds at one disturbance scale ``rho``."""

    scale: float
    n_episodes: int
    n_steps: int
    recorded_states: int
    constraint_violations: int
    invariant_exits: int
    violation_rate: float
    exit_rate: float
    worst_constraint_residual: float
    worst_invariant_residual: float
    fraction_steps_outside_declared_W: float
    baseline_fraction: float
    switches_per_1000_steps: float
    episodes_with_any_violation: int
    episodes_with_any_exit: int

    def row(self) -> str:
        """One fixed-width report line."""
        return (
            f"rho={self.scale:6.3f}  X-viol={self.constraint_violations:<7d} "
            f"S-exit={self.invariant_exits:<7d} "
            f"viol/step={self.violation_rate:10.3e} exit/step={self.exit_rate:10.3e} "
            f"worst X resid={self.worst_constraint_residual:11.4e} "
            f"worst S resid={self.worst_invariant_residual:11.4e} "
            f"w outside W={self.fraction_steps_outside_declared_W:6.4f} "
            f"base frac={self.baseline_fraction:7.5f} "
            f"switch/1k={self.switches_per_1000_steps:7.2f} "
            f"eps w/viol={self.episodes_with_any_violation}/{self.n_episodes}"
        )


@dataclass(frozen=True)
class BoundSweep:
    """The sweep, plus the two thresholds it was run to find."""

    points: tuple[BoundViolationPoint, ...]
    first_scale_with_invariant_exit: float
    first_scale_with_constraint_violation: float
    disturbance_mode: str

    def threshold_text(self, value: float) -> str:
        """Human-readable rendering of a threshold scale, or "not observed"."""
        if np.isfinite(value):
            return f"rho = {value:g}"
        largest = max(p.scale for p in self.points)
        return f"not observed up to rho = {largest:g}"

    def describe(self) -> str:
        """Report block used verbatim by the validation script and the CLI."""
        lines = [f"disturbance sampler         {self.disturbance_mode}", ""]
        lines += [p.row() for p in self.points]
        lines += [
            "",
            "smallest scale at which the invariance certificate was lost (state left S)"
            f"   {self.threshold_text(self.first_scale_with_invariant_exit)}",
            "smallest scale at which a declared state constraint was broken            "
            f"   {self.threshold_text(self.first_scale_with_constraint_violation)}",
        ]
        return "\n".join(lines)


def bound_violation_sweep(
    plant: Plant,
    guard: SimplexGuard,
    performance: PerformanceController,
    invariant_set: Polytope,
    scales: Sequence[float] = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0),
    n_episodes: int = 6,
    n_steps: int = 1500,
    seed: int = 5102,
    reference_amplitude: float = 0.18,
    reference_period: int = 80,
    disturbance_mode: str = "uniform",
    reference_factory: Callable[[int], Callable[[int], np.ndarray]] | None = None,
) -> BoundSweep:
    """Sweep the realised disturbance scale and measure where the guarantee fails.

    The guard is **not** rebuilt for each scale: it keeps using the declared
    ``W``, which is the whole point. ``scales`` must start at or below 1 so the
    guaranteed case is in the table.
    """
    scale_list = [float(s) for s in scales]
    if not scale_list:
        raise ValueError("scales must be non-empty")
    if any(s < 0.0 or not np.isfinite(s) for s in scale_list):
        raise ValueError("scales must be finite and non-negative")
    if min(scale_list) > 1.0:
        raise ValueError(
            "the sweep must include a scale of at most 1.0, so that the guaranteed "
            "case appears in the table it is compared against"
        )
    if int(n_episodes) < 1:
        raise ValueError(f"n_episodes must be at least 1, got {n_episodes}")
    declared = plant.disturbance
    points: list[BoundViolationPoint] = []
    first_exit = float("inf")
    first_violation = float("inf")
    for scale in scale_list:
        tot_states = tot_viol = tot_exit = 0
        tot_outside = 0
        tot_switch = 0
        base_steps = 0
        worst_x = -np.inf
        worst_s = -np.inf
        eps_viol = eps_exit = 0
        for episode in range(int(n_episodes)):
            rng = np.random.default_rng(int(seed) + 7919 * episode)
            amp = float(reference_amplitude) * float(rng.uniform(0.75, 1.25))
            reference = (
                square_wave_reference(amp, int(reference_period), plant.n_states)
                if reference_factory is None
                else reference_factory(episode)
            )
            w = disturbance_sequence(plant, int(n_steps), rng, disturbance_mode, scale=scale)
            ep = simulate_guarded(
                plant, guard, performance, int(n_steps), reference, w, weights=CostWeights()
            )
            viol = ep.constraint_violations()
            exits = ep.invariant_exits(invariant_set)
            tot_states += ep.states.shape[0]
            tot_viol += int(viol.size)
            tot_exit += int(exits.size)
            eps_viol += int(viol.size > 0)
            eps_exit += int(exits.size > 0)
            tot_outside += int(np.sum(~declared.contains_many(w, tol=1e-12)))
            tot_switch += int(np.sum(ep.modes[1:] != ep.modes[:-1]))
            base_steps += int(np.sum(ep.modes == Mode.BASELINE.value))
            worst_x = max(worst_x, ep.worst_constraint_residual())
            worst_s = max(
                worst_s,
                float(np.max(ep.states @ invariant_set.A.T - invariant_set.b)),
            )
        total_steps = int(n_episodes) * int(n_steps)
        point = BoundViolationPoint(
            scale=scale,
            n_episodes=int(n_episodes),
            n_steps=int(n_steps),
            recorded_states=tot_states,
            constraint_violations=tot_viol,
            invariant_exits=tot_exit,
            violation_rate=tot_viol / tot_states,
            exit_rate=tot_exit / tot_states,
            worst_constraint_residual=float(worst_x),
            worst_invariant_residual=float(worst_s),
            fraction_steps_outside_declared_W=tot_outside / total_steps,
            baseline_fraction=base_steps / total_steps,
            switches_per_1000_steps=1000.0 * tot_switch / total_steps,
            episodes_with_any_violation=eps_viol,
            episodes_with_any_exit=eps_exit,
        )
        points.append(point)
        if tot_exit > 0 and scale < first_exit:
            first_exit = scale
        if tot_viol > 0 and scale < first_violation:
            first_violation = scale
    return BoundSweep(
        points=tuple(points),
        first_scale_with_invariant_exit=first_exit,
        first_scale_with_constraint_violation=first_violation,
        disturbance_mode=disturbance_mode,
    )
