"""Baseline 2: direct re-simulation. Slow, exact, and the reference.

What this module is for
-----------------------
Baseline 1 proves that most misses cannot have mattered. The ones it cannot
settle are settled here, by doing the expensive thing: run the declared
nonlinear plant twice, once with the miss and once without, and measure how
far apart the two trajectories get in the declared metric.

This is the reference every other number in the product is scored against. It
is not a model of the truth, it IS the truth for the declared plant, up to the
integrator's own error -- which is why the integrator is fixed-step and
declared rather than adaptive. An adaptive step would make the reference
trajectory depend on the perturbation, and then "the same scenario without the
miss" would not be the same scenario.

The cost, measured and not hand-waved
-------------------------------------
Every call counts its plant evaluations in :attr:`Resimulation.plant_calls`.
A re-simulation costs one right-hand-side evaluation per Runge-Kutta stage per
substep per remaining period: for the declared loop that is
``4 x 4 x steps_remaining``. The certificate costs two matrix exponentials and
``steps`` two-by-two matrix products, once per run, and then one logarithm per
miss. The ratio is reported in ``validation/validate_resimulation.py`` and in
the README, because "slow" is a measurement and not an adjective.

The optional early exit
-----------------------
:func:`peak_divergence` can stop early, when the linearised bound on the
remaining horizon has fallen below the peak already seen. That criterion rests
on the SAME linearisation as baseline 1, so using it would make the reference
depend on the thing the reference exists to check. It is therefore **off by
default**, and its agreement with the full-horizon reference is measured over
the whole miss population in ``validation/validate_resimulation.py`` rather
than assumed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .certificate import CertificateModel
from .plant import LoopModel
from .timeline import MissEvent, RunLog, miss_events

__all__ = [
    "CONSEQUENTIAL",
    "INCONSEQUENTIAL",
    "Resimulation",
    "Trajectory",
    "classify_miss",
    "peak_divergence",
    "resimulate_run",
    "simulate",
]

#: Verdict for a miss whose re-simulated divergence reached the tolerance.
CONSEQUENTIAL = "CONSEQUENTIAL"

#: Verdict for a miss whose re-simulated divergence stayed below tolerance.
INCONSEQUENTIAL = "INCONSEQUENTIAL"


@dataclass(frozen=True)
class Trajectory:
    """A simulated trajectory and what it cost.

    Attributes
    ----------
    states
        States at sample instants, rad and rad/s, shape ``(n + 1, 2)``, where
        index 0 is the state at ``start_index``.
    computed_rad
        Command computed by each iteration simulated, rad, shape ``(n,)``.
    saturated
        Whether each computed command hit the actuator limit, shape ``(n,)``.
    start_index
        Iteration index the trajectory starts at.
    plant_calls
        Right-hand-side evaluations used. The cost unit of this module.
    """

    states: np.ndarray
    computed_rad: np.ndarray
    saturated: np.ndarray
    start_index: int
    plant_calls: int

    @property
    def steps(self) -> int:
        """Number of iterations simulated."""
        return int(self.computed_rad.shape[0])


def _rk4(
    model: LoopModel, state: np.ndarray, control: float, duration_s: float, substeps: int
) -> tuple[np.ndarray, int]:
    """Classical fourth-order Runge-Kutta over ``duration_s`` at fixed step.

    The scheme is the standard one (Butcher 2016, *Numerical Methods for
    Ordinary Differential Equations*, 3rd edition, section 232); it is used
    rather than an exponential integrator because the plant is nonlinear, and
    at a fixed declared step rather than adaptively because the reference must
    not depend on the perturbation.
    """
    if duration_s <= 0.0:
        return np.asarray(state, dtype=float).copy(), 0
    derivative = model.plant.derivative_scalar
    array = np.asarray(state, dtype=float)
    alpha = float(array[0])
    rate = float(array[1])
    step = duration_s / substeps
    half = 0.5 * step
    sixth = step / 6.0
    for _ in range(substeps):
        first_a, first_r = derivative(alpha, rate, control)
        second_a, second_r = derivative(
            alpha + half * first_a, rate + half * first_r, control
        )
        third_a, third_r = derivative(
            alpha + half * second_a, rate + half * second_r, control
        )
        fourth_a, fourth_r = derivative(
            alpha + step * third_a, rate + step * third_r, control
        )
        alpha = alpha + sixth * (
            first_a + 2.0 * second_a + 2.0 * third_a + fourth_a
        )
        rate = rate + sixth * (
            first_r + 2.0 * second_r + 2.0 * third_r + fourth_r
        )
    return np.array([alpha, rate], dtype=float), 4 * substeps


def simulate(
    model: LoopModel,
    command_rad: np.ndarray,
    misses: Mapping[int, float] | None = None,
    *,
    start_index: int = 0,
    start_state: np.ndarray | None = None,
    prior_command_rad: float | None = None,
    substeps: int | None = None,
    stop_index: int | None = None,
) -> Trajectory:
    """Run the declared sampled loop, with the declared hold-last-value misses.

    Parameters
    ----------
    model
        The declared loop.
    command_rad
        Commanded incidence per iteration, rad, shape ``(steps,)``.
    misses
        Mapping from iteration index to overrun in seconds. An index outside
        ``[start_index, steps)`` or a non-positive overrun raises, because a
        silently ignored miss would make a counterfactual wrong in the one
        direction that matters.
    start_index
        First iteration to simulate. Used to re-simulate only the tail of a run
        from a reference state, which is what makes the per-miss study
        affordable.
    start_state
        State at ``start_index``, rad and rad/s. Defaults to the origin.
    prior_command_rad
        Command in force before ``start_index``, rad; needed when a miss occurs
        at ``start_index`` itself. Defaults to the command that iteration
        ``start_index`` would compute, which makes such a miss inject nothing.
    substeps
        Runge-Kutta substeps per period. Defaults to the sampling law's.
    stop_index
        Stop after this iteration index, exclusive. Used by the optional early
        exit of :func:`peak_divergence` so that stopping early actually costs
        less rather than merely reporting that it would. The miss model is
        evaluated against the FULL ``command_rad`` length, so truncating here
        does not change how a hold near the stop point behaves.

    Returns
    -------
    Trajectory
    """
    commands = np.asarray(command_rad, dtype=float)
    if commands.ndim != 1:
        raise ValueError(f"command_rad must be one-dimensional, got {commands.shape}")
    steps = int(commands.shape[0])
    if not 0 <= start_index < steps:
        raise ValueError(
            f"start_index must lie in [0, {steps}), got {start_index}"
        )
    period = model.sampling.period_s
    substep_count = model.sampling.substeps if substeps is None else int(substeps)
    if substep_count < 1:
        raise ValueError(f"substeps must be at least 1, got {substep_count}")
    schedule = dict(misses or {})
    for index, overrun in schedule.items():
        if not start_index <= index < steps:
            raise ValueError(
                f"miss index {index} is outside the simulated range "
                f"[{start_index}, {steps})"
            )
        if overrun <= 0.0:
            raise ValueError(
                f"miss at index {index} has a non-positive overrun {overrun}"
            )

    limit = steps if stop_index is None else int(stop_index)
    if not start_index < limit <= steps:
        raise ValueError(
            f"stop_index must lie in ({start_index}, {steps}], got {stop_index}"
        )
    simulated = limit - start_index
    states = np.empty((simulated + 1, 2), dtype=float)
    computed = np.full(simulated, np.nan, dtype=float)
    saturated = np.zeros(simulated, dtype=bool)
    current = (
        np.zeros(2, dtype=float)
        if start_state is None
        else np.asarray(start_state, dtype=float).copy()
    )
    states[0] = current
    calls = 0
    controller = model.controller
    plant = model.plant

    index = start_index
    previous = prior_command_rad
    while index < limit:
        local = index - start_index
        value, clipped = controller.command_with_saturation(
            plant, current, commands[index]
        )
        computed[local] = value
        saturated[local] = clipped
        overrun = schedule.get(index, 0.0)
        if overrun <= 0.0:
            current, used = _rk4(model, current, value, period, substep_count)
            calls += used
            states[local + 1] = current
            previous = value
            index += 1
            continue
        stale = value if previous is None else float(previous)
        hold_periods = int(np.floor(overrun / period + 1e-12))
        remainder = max(float(overrun - hold_periods * period), 0.0)
        # Whole periods inside the overrun: the loop still runs and still
        # computes a command, and that command is superseded.
        held = 0
        while held < hold_periods and index + held < limit:
            offset = index + held - start_index
            value, clipped = controller.command_with_saturation(
                plant, current, commands[index + held]
            )
            computed[offset] = value
            saturated[offset] = clipped
            current, used = _rk4(model, current, stale, period, substep_count)
            calls += used
            states[offset + 1] = current
            held += 1
        landing = index + held
        if landing >= limit:
            index = limit
            previous = stale
            continue
        offset = landing - start_index
        fresh, clipped = controller.command_with_saturation(
            plant, current, commands[landing]
        )
        computed[offset] = fresh
        saturated[offset] = clipped
        if remainder > 0.0 and held == hold_periods:
            current, used = _rk4(model, current, stale, remainder, substep_count)
            calls += used
            current, used = _rk4(
                model, current, fresh, period - remainder, substep_count
            )
            calls += used
        else:
            current, used = _rk4(model, current, fresh, period, substep_count)
            calls += used
        states[offset + 1] = current
        previous = fresh
        index = landing + 1
    return Trajectory(
        states=states,
        computed_rad=computed,
        saturated=saturated,
        start_index=start_index,
        plant_calls=calls,
    )


def reference_run(model: LoopModel, command_rad: np.ndarray, **kwargs: object) -> Trajectory:
    """The miss-free trajectory: :func:`simulate` with an empty miss set."""
    return simulate(model, command_rad, None, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Resimulation:
    """The re-simulated answer for one miss.

    Attributes
    ----------
    event
        The miss.
    peak_divergence
        The largest weighted-norm difference between the with-miss and the
        miss-free trajectory over the remaining sample instants, dimensionless
        in the declared metric.
    peak_index
        Iteration index at which the peak occurred.
    tolerance
        The declared tolerance it was compared against.
    verdict
        ``CONSEQUENTIAL`` or ``INCONSEQUENTIAL``.
    plant_calls
        Right-hand-side evaluations the re-simulation cost.
    steps_simulated
        Iterations re-simulated. Below ``event.steps_remaining`` only when the
        optional early exit was enabled.
    """

    event: MissEvent
    peak_divergence: float
    peak_index: int
    tolerance: float
    verdict: str
    plant_calls: int
    steps_simulated: int

    @property
    def consequential(self) -> bool:
        """True when the miss measurably changed the trajectory."""
        return self.verdict == CONSEQUENTIAL


def peak_divergence(
    model: LoopModel,
    command_rad: np.ndarray,
    reference_states: np.ndarray,
    event: MissEvent,
    *,
    constants: CertificateModel | None = None,
    early_exit: bool = False,
    substeps: int | None = None,
    chunk_steps: int = 16,
) -> tuple[float, int, int, int]:
    """Re-simulate one miss and return its peak divergence.

    Parameters
    ----------
    model
        The declared loop.
    command_rad
        The commanded incidence sequence, rad.
    reference_states
        The miss-free trajectory's states, shape ``(steps + 1, 2)``.
    event
        The miss to inject.
    constants
        Certificate constants, required only when ``early_exit`` is enabled.
    early_exit
        Stop once the linearised bound on the remaining horizon has fallen
        below the peak already seen. Off by default; see the module docstring.
    substeps
        Runge-Kutta substeps per period; defaults to the sampling law's.
    chunk_steps
        Iterations simulated per chunk when ``early_exit`` is enabled. Larger
        chunks overshoot the stopping point; smaller ones pay more fixed cost
        per chunk. Declared, not tuned.

    Returns
    -------
    (peak, peak_index, plant_calls, steps_simulated)
    """
    reference = np.asarray(reference_states, dtype=float)
    commands = np.asarray(command_rad, dtype=float)
    steps = int(commands.shape[0])
    if reference.shape != (steps + 1, 2):
        raise ValueError(
            f"reference_states must have shape ({steps + 1}, 2), got {reference.shape}"
        )
    if early_exit and constants is None:
        raise ValueError("early_exit requires the certificate constants")
    index = event.index
    start_state = reference[index]
    prior = event.stale_command_rad
    if not early_exit:
        trajectory = simulate(
            model,
            commands,
            {index: event.overrun_s},
            start_index=index,
            start_state=start_state,
            prior_command_rad=prior,
            substeps=substeps,
        )
        difference = trajectory.states - reference[index:]
        norms = model.metric.norm(difference)
        best = int(np.argmax(norms))
        return (
            float(norms[best]),
            index + best,
            trajectory.plant_calls,
            trajectory.steps,
        )

    assert constants is not None
    # Simulated in chunks and RESUMED, so that stopping early costs less rather
    # than merely reporting that it would. The criterion is: once the error at
    # a sample instant, amplified by the worst the linearised closed loop can
    # still do over the remaining horizon, cannot reach the peak already seen,
    # no later sample can beat that peak -- so the recorded peak is final. The
    # amplification is gamma_sup of equation (12), which is the SAME
    # linearisation baseline 1 rests on; that is why this is off by default.
    peak = 0.0
    best = 0
    calls = 0
    simulated = 0
    cursor = index
    state = start_state
    previous = prior
    while cursor < steps:
        stop = min(cursor + chunk_steps, steps)
        piece = simulate(
            model,
            commands,
            {index: event.overrun_s} if cursor == index else None,
            start_index=cursor,
            start_state=state,
            prior_command_rad=previous,
            substeps=substeps,
            stop_index=stop,
        )
        calls += piece.plant_calls
        norms = model.metric.norm(piece.states - reference[cursor : stop + 1])
        finished = False
        for offset, value in enumerate(norms):
            absolute = cursor + offset
            if value > peak:
                peak = float(value)
                best = absolute
            simulated = absolute - index
            remaining = steps - absolute
            if remaining <= 0:
                finished = True
                break
            gain = constants.gamma_sup(min(remaining, constants.horizon_steps))
            if peak > 0.0 and value * gain <= peak:
                finished = True
                break
        if finished:
            break
        state = piece.states[-1]
        previous = float(piece.computed_rad[-1])
        cursor = stop
    return peak, best, calls, simulated


def classify_miss(
    model: LoopModel,
    command_rad: np.ndarray,
    reference_states: np.ndarray,
    event: MissEvent,
    tolerance: float = 1.0,
    **kwargs: object,
) -> Resimulation:
    """Re-simulate one miss and give it a verdict against ``tolerance``."""
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    peak, where, calls, simulated = peak_divergence(
        model, command_rad, reference_states, event, **kwargs  # type: ignore[arg-type]
    )
    verdict = CONSEQUENTIAL if peak >= tolerance else INCONSEQUENTIAL
    return Resimulation(
        event=event,
        peak_divergence=peak,
        peak_index=where,
        tolerance=float(tolerance),
        verdict=verdict,
        plant_calls=calls,
        steps_simulated=simulated,
    )


def resimulate_run(
    log: RunLog,
    tolerance: float = 1.0,
    subset: tuple[int, ...] | None = None,
    **kwargs: object,
) -> tuple[Resimulation, ...]:
    """Re-simulate every miss in ``log``, or only the iterations in ``subset``.

    ``subset`` is how the certificate's residual set is handed to baseline 2:
    pass the iteration indices the certificate left undecided and only those
    are paid for.
    """
    events = miss_events(log)
    if subset is not None:
        wanted = set(subset)
        events = tuple(event for event in events if event.index in wanted)
    reference = np.asarray(log.states, dtype=float)
    return tuple(
        classify_miss(
            log.model, log.command_rad, reference, event, tolerance, **kwargs
        )
        for event in events
    )


def timed_resimulation(
    log: RunLog, tolerance: float = 1.0
) -> tuple[tuple[Resimulation, ...], float]:
    """Re-simulate every miss and return the results with the wall time in seconds.

    Wall time is unstable on shared hardware and is reported, never asserted.
    """
    started = time.perf_counter()
    results = resimulate_run(log, tolerance)
    return results, time.perf_counter() - started
