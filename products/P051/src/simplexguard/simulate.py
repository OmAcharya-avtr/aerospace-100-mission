"""Episode simulation: guarded, unguarded and baseline-only, on the same noise.

Every comparison this package reports is **paired**. The disturbance sequence is
drawn once per episode and replayed to each architecture, so the conservatism
cost of the guard is a difference on a common realisation rather than a
difference of two independent Monte-Carlo means. Pairing is the reason the
conservatism numbers in the README have no error bars attached to the
difference: the difference is computed realisation by realisation.

Three architectures
-------------------
``guarded``
    The performance controller wrapped by :class:`simplexguard.guard.SimplexGuard`.
``unguarded``
    The performance controller alone. This is the reference the conservatism cost
    is measured against, and it is also the run that violates the declared
    constraints, which is the point.
``baseline``
    The conservative controller alone. The safety floor: it never leaves ``S``
    under the declared bound, and its tracking cost is the price of never
    trying.

Disturbance sampling
--------------------
The switching condition is worst-case over ``W``, so no distribution on ``W``
enters it. A simulation nevertheless has to draw something, and the choice
changes how often the worst case is approached:

``uniform``
    Uniform on the box ``W``. Realistic-looking, and almost never near a vertex
    in high dimension.
``vertex``
    Uniform over the ``2**n`` vertices of the box. This is the sampler that
    realises the worst case in each coordinate at every step, so it is the
    stress case for the accounting, and the one a reviewer should ask for.
``zero``
    No disturbance. Isolates the controllers from the noise.

All three are inside the declared bound, so all three must produce zero
constraint violations in a guarded run. ``validation/validate_accounting.py``
runs all three.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .controllers import BaselineController, PerformanceController
from .guard import GuardDecision, Mode, SimplexGuard
from .plant import Plant
from .polytope import Box, Polytope

__all__ = [
    "CostWeights",
    "Episode",
    "disturbance_sequence",
    "simulate_baseline",
    "simulate_guarded",
    "simulate_unguarded",
    "square_wave_reference",
]


def square_wave_reference(
    amplitude: float, period_steps: int, n_states: int = 2
) -> Callable[[int], np.ndarray]:
    """A square-wave reference on the first state coordinate, zero elsewhere.

    Parameters
    ----------
    amplitude :
        Reference magnitude in the units of the first state coordinate.
    period_steps :
        Steps per half period, so the sign flips every ``period_steps`` steps.
    n_states :
        Length of the returned reference vector.
    """
    if int(period_steps) < 1:
        raise ValueError(f"period_steps must be at least 1, got {period_steps}")
    if not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    if int(n_states) < 1:
        raise ValueError(f"n_states must be at least 1, got {n_states}")
    half = int(period_steps)
    n = int(n_states)
    amp = float(amplitude)

    def reference(step: int) -> np.ndarray:
        out = np.zeros(n)
        out[0] = amp if (int(step) // half) % 2 == 0 else -amp
        return out

    return reference


def disturbance_sequence(
    plant: Plant,
    n_steps: int,
    rng: np.random.Generator,
    mode: str = "uniform",
    scale: float = 1.0,
) -> np.ndarray:
    """Draw a disturbance sequence of shape ``(n_steps, n)``.

    Parameters
    ----------
    scale :
        Multiplier on the declared half-widths. ``scale <= 1`` is inside the
        declared bound and the guarantee applies; ``scale > 1`` is the deliberate
        bound violation of :mod:`simplexguard.boundviolation` and the guarantee
        does not apply.
    mode :
        ``"uniform"``, ``"vertex"`` or ``"zero"``; see the module docstring.
    """
    if not isinstance(plant.disturbance, Box):
        raise TypeError("disturbance_sequence requires a box disturbance set")
    if int(n_steps) < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(f"scale must be finite and non-negative, got {scale}")
    w_set = plant.disturbance
    radius = w_set.half_widths * float(scale)
    centre = w_set.centre
    n = int(n_steps)
    if mode == "zero":
        return np.zeros((n, plant.n_states))
    if mode == "uniform":
        return centre + rng.uniform(-1.0, 1.0, size=(n, plant.n_states)) * radius
    if mode == "vertex":
        signs = rng.integers(0, 2, size=(n, plant.n_states)) * 2.0 - 1.0
        return centre + signs * radius
    raise ValueError(f"unknown disturbance mode {mode!r}; expected uniform, vertex or zero")


@dataclass(frozen=True)
class CostWeights:
    """Diagonal weights of the tracking cost used for the conservatism measure.

    The stage cost is

        J = sum_k (x_k - r_k)^T diag(q) (x_k - r_k) + u_k^T diag(r_u) u_k,

    with ``q`` in inverse squared state units and ``r_u`` in inverse squared
    input units. The defaults weight the tracked coordinate heavily and the
    others lightly, so the number reported as "conservatism cost" is dominated
    by tracking error and not by control effort. The weights are an arbitrary
    choice of performance metric and are reported alongside every cost.
    """

    state: tuple[float, ...] = (10.0, 0.1)
    input: tuple[float, ...] = (0.01,)

    def evaluate(
        self, states: np.ndarray, inputs: np.ndarray, references: np.ndarray
    ) -> float:
        """Total cost over an episode; ``states`` excludes the terminal state."""
        q = np.asarray(self.state, dtype=float)
        r = np.asarray(self.input, dtype=float)
        err = np.asarray(states, dtype=float) - np.asarray(references, dtype=float)
        if err.shape[1] != q.shape[0]:
            raise ValueError(f"state weight has length {q.shape[0]}, states width {err.shape[1]}")
        u = np.atleast_2d(np.asarray(inputs, dtype=float))
        if u.shape[1] != r.shape[0]:
            raise ValueError(f"input weight has length {r.shape[0]}, inputs width {u.shape[1]}")
        return float(np.sum(err**2 @ q) + np.sum(u**2 @ r))


@dataclass(frozen=True)
class Episode:
    """One simulated run: a record, not a view of live state.

    ``states`` has ``n_steps + 1`` rows, because the terminal state is included
    so that a constraint violation caused by the final input is counted.
    """

    architecture: str
    plant: Plant
    states: np.ndarray
    inputs: np.ndarray
    references: np.ndarray
    disturbances: np.ndarray
    modes: np.ndarray
    margins: np.ndarray
    state_margins: np.ndarray
    certificate_lost: np.ndarray
    forced_by_dwell: np.ndarray
    proposed_inputs: np.ndarray
    weights: CostWeights = field(default_factory=CostWeights)

    @property
    def n_steps(self) -> int:
        """Number of control steps."""
        return int(self.inputs.shape[0])

    @property
    def baseline_fraction(self) -> float:
        """Fraction of steps under baseline authority, in ``[0, 1]``."""
        if self.n_steps == 0:
            return 0.0
        return float(np.mean(self.modes == Mode.BASELINE.value))

    def constraint_violations(self, tol: float = 0.0) -> np.ndarray:
        """Indices of recorded states, terminal one included, outside ``X``.

        Must be empty for a guarded run whose disturbances all lie inside the
        declared bound. If it is not empty, that is a defect in this package and
        not a tolerance to widen.
        """
        inside = self.plant.state_constraints.contains_many(self.states, tol=tol)
        return np.flatnonzero(~inside)

    def worst_constraint_residual(self) -> float:
        """Largest row residual ``max_i,k (G x_k - g)_i``; non-positive if inside."""
        return float(np.max(self.states @ self.plant.state_constraints.A.T
                            - self.plant.state_constraints.b))

    def invariant_exits(self, invariant: Polytope, tol: float = 0.0) -> np.ndarray:
        """Indices of recorded states outside the supplied invariant set ``S``.

        A non-empty result means the invariance certificate was lost, which can
        happen before any constraint of ``X`` is broken and is the earlier,
        sharper symptom of an under-declared disturbance bound.
        """
        inside = invariant.contains_many(self.states, tol=tol)
        return np.flatnonzero(~inside)

    def cost(self) -> float:
        """Tracking cost of this episode under :class:`CostWeights`."""
        return self.weights.evaluate(self.states[:-1], self.inputs, self.references)


def _finish(
    architecture: str,
    plant: Plant,
    states: list[np.ndarray],
    inputs: list[np.ndarray],
    references: list[np.ndarray],
    disturbances: np.ndarray,
    modes: list[int],
    margins: list[float],
    state_margins: list[float],
    certificate_lost: list[bool],
    forced: list[bool],
    proposed: list[np.ndarray],
    weights: CostWeights,
) -> Episode:
    return Episode(
        architecture=architecture,
        plant=plant,
        states=np.array(states, dtype=float),
        inputs=np.array(inputs, dtype=float),
        references=np.array(references, dtype=float),
        disturbances=np.asarray(disturbances, dtype=float),
        modes=np.array(modes, dtype=int),
        margins=np.array(margins, dtype=float),
        state_margins=np.array(state_margins, dtype=float),
        certificate_lost=np.array(certificate_lost, dtype=bool),
        forced_by_dwell=np.array(forced, dtype=bool),
        proposed_inputs=np.array(proposed, dtype=float),
        weights=weights,
    )


def _check_common(n_steps: int, x0: np.ndarray, plant: Plant) -> np.ndarray:
    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    v = np.asarray(x0, dtype=float).ravel()
    if v.shape[0] != plant.n_states:
        raise ValueError(f"x0 has length {v.shape[0]}, expected {plant.n_states}")
    if not np.all(np.isfinite(v)):
        raise ValueError("x0 contains non-finite entries")
    return v


def simulate_guarded(
    plant: Plant,
    guard: SimplexGuard,
    performance: PerformanceController,
    n_steps: int,
    reference: Callable[[int], np.ndarray],
    disturbances: np.ndarray,
    x0: np.ndarray | None = None,
    weights: CostWeights | None = None,
) -> Episode:
    """Run the guarded architecture. ``disturbances`` has shape ``(n_steps, n)``."""
    x = _check_common(n_steps, np.zeros(plant.n_states) if x0 is None else x0, plant)
    w = np.atleast_2d(np.asarray(disturbances, dtype=float))
    if w.shape != (int(n_steps), plant.n_states):
        raise ValueError(f"disturbances must have shape {(int(n_steps), plant.n_states)}")
    guard.reset()
    states, inputs, refs, modes = [x.copy()], [], [], []
    margins, state_margins, lost, forced, proposed = [], [], [], [], []
    for k in range(int(n_steps)):
        r = np.asarray(reference(k), dtype=float).ravel()
        u_perf = performance(x, r)
        decision: GuardDecision = guard.decide(x, u_perf)
        modes.append(decision.mode.value)
        margins.append(decision.margin)
        state_margins.append(decision.state_margin)
        lost.append(decision.certificate_lost)
        forced.append(decision.forced_by_dwell)
        proposed.append(np.atleast_1d(u_perf))
        inputs.append(np.atleast_1d(decision.applied_input))
        refs.append(r)
        x = plant.step(x, decision.applied_input, w[k])
        states.append(x.copy())
    return _finish(
        "guarded", plant, states, inputs, refs, w, modes, margins, state_margins,
        lost, forced, proposed, weights or CostWeights(),
    )


def simulate_unguarded(
    plant: Plant,
    performance: PerformanceController,
    n_steps: int,
    reference: Callable[[int], np.ndarray],
    disturbances: np.ndarray,
    x0: np.ndarray | None = None,
    weights: CostWeights | None = None,
) -> Episode:
    """Run the performance controller with no guard, on the same disturbances."""
    x = _check_common(n_steps, np.zeros(plant.n_states) if x0 is None else x0, plant)
    w = np.atleast_2d(np.asarray(disturbances, dtype=float))
    if w.shape != (int(n_steps), plant.n_states):
        raise ValueError(f"disturbances must have shape {(int(n_steps), plant.n_states)}")
    states, inputs, refs = [x.copy()], [], []
    for k in range(int(n_steps)):
        r = np.asarray(reference(k), dtype=float).ravel()
        u = performance(x, r)
        inputs.append(np.atleast_1d(u))
        refs.append(r)
        x = plant.step(x, u, w[k])
        states.append(x.copy())
    zeros = np.zeros(int(n_steps))
    return _finish(
        "unguarded", plant, states, inputs, refs, w,
        [Mode.PERFORMANCE.value] * int(n_steps), list(zeros), list(zeros),
        [False] * int(n_steps), [False] * int(n_steps),
        [np.atleast_1d(u) for u in inputs], weights or CostWeights(),
    )


def simulate_baseline(
    plant: Plant,
    baseline: BaselineController,
    n_steps: int,
    reference: Callable[[int], np.ndarray],
    disturbances: np.ndarray,
    x0: np.ndarray | None = None,
    weights: CostWeights | None = None,
) -> Episode:
    """Run the baseline controller alone, on the same disturbances.

    The baseline regulates to the origin and ignores the reference, so its cost
    against a non-zero reference is the price of never attempting the task. It
    is reported so that the conservatism cost of the guard can be read as a
    position between two extremes rather than as a penalty against nothing.
    """
    x = _check_common(n_steps, np.zeros(plant.n_states) if x0 is None else x0, plant)
    w = np.atleast_2d(np.asarray(disturbances, dtype=float))
    if w.shape != (int(n_steps), plant.n_states):
        raise ValueError(f"disturbances must have shape {(int(n_steps), plant.n_states)}")
    states, inputs, refs = [x.copy()], [], []
    for k in range(int(n_steps)):
        refs.append(np.asarray(reference(k), dtype=float).ravel())
        u = baseline(x)
        inputs.append(np.atleast_1d(u))
        x = plant.step(x, u, w[k])
        states.append(x.copy())
    ones = [Mode.BASELINE.value] * int(n_steps)
    zeros = np.zeros(int(n_steps))
    return _finish(
        "baseline", plant, states, inputs, refs, w, ones, list(zeros), list(zeros),
        [False] * int(n_steps), [False] * int(n_steps),
        [np.atleast_1d(u) for u in inputs], weights or CostWeights(),
    )
