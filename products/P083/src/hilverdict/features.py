"""Cheap log features for the learned component, and nothing else.

Every feature below is read off the run log. None of them costs a simulation,
which is the whole premise of the learned component: it exists for the case
where re-simulation is unaffordable, so it may not use one.

Two of the features -- the overrun ``tau`` and the input step ``du`` -- are
*exactly* the two numbers the analytic certificate of
:mod:`hilverdict.certificate` is a function of. That is deliberate. The learned
model is given everything the proof uses and eleven further features besides,
so if it still loses, it loses on the merits and not because it was starved.
The comparison in ``validation/validate_learned_model.py`` rests on this.
"""

from __future__ import annotations

import numpy as np

from .timeline import MissEvent, RunLog, miss_events

__all__ = ["FEATURE_NAMES", "feature_matrix", "feature_vector"]

#: The declared feature order. Fixed, because a model fitted on one order and
#: scored on another is a silent defect that no test of the model's accuracy
#: can catch.
FEATURE_NAMES: tuple[str, ...] = (
    "overrun_periods",
    "overrun_s",
    "hold_periods",
    "input_step_rad",
    "abs_command_rad",
    "abs_alpha_rad",
    "abs_rate_rad_s",
    "tracking_error_rad",
    "remaining_fraction",
    "command_activity_rad",
    "misses_in_window",
    "consecutive_misses",
    "local_utilisation",
    "steps_to_switch",
)

#: Iterations looked back over for the windowed features. Declared.
WINDOW = 10

#: Iterations looked back over for the command-activity feature. Declared.
ACTIVITY_WINDOW = 5


def _steps_to_switch(command_rad: np.ndarray, index: int) -> int:
    """Iterations from ``index`` to the nearest change in the commanded value."""
    commands = np.asarray(command_rad, dtype=float)
    changes = np.nonzero(np.diff(commands) != 0.0)[0] + 1
    if changes.size == 0:
        return int(commands.shape[0])
    return int(np.min(np.abs(changes - index)))


def feature_vector(log: RunLog, event: MissEvent) -> np.ndarray:
    """The declared feature vector for one miss, in :data:`FEATURE_NAMES` order.

    Units are those of the underlying quantity: seconds, radians, radians per
    second, or dimensionless for the counts and fractions. No scaling or
    whitening is applied, because the learned model is a tree ensemble and a
    monotone rescaling of a feature cannot change a tree's splits.
    """
    index = event.index
    commands = np.asarray(log.computed_rad, dtype=float)
    durations = np.asarray(log.durations_s, dtype=float)
    states = np.asarray(log.states, dtype=float)
    reference = np.asarray(log.command_rad, dtype=float)
    missed = np.asarray(log.missed)
    period = log.model.sampling.period_s

    start = max(index - ACTIVITY_WINDOW, 0)
    if index - start >= 2:
        activity = float(np.mean(np.abs(np.diff(commands[start : index + 1]))))
    else:
        activity = 0.0
    window_start = max(index - WINDOW, 0)
    in_window = int(np.count_nonzero(missed[window_start:index]))
    consecutive = 0
    cursor = index - 1
    while cursor >= 0 and bool(missed[cursor]):
        consecutive += 1
        cursor -= 1
    local = (
        float(np.mean(durations[window_start : index + 1]) / period)
        if index >= window_start
        else float(durations[index] / period)
    )
    return np.array(
        [
            event.overrun_periods,
            event.overrun_s,
            float(event.hold_periods),
            event.input_step_rad,
            abs(float(commands[index])),
            abs(float(states[index, 0])),
            abs(float(states[index, 1])),
            abs(float(states[index, 0] - reference[index])),
            event.steps_remaining / log.steps,
            activity,
            float(in_window),
            float(consecutive),
            local,
            float(_steps_to_switch(reference, index)),
        ],
        dtype=float,
    )


def feature_matrix(
    log: RunLog, events: tuple[MissEvent, ...] | None = None
) -> tuple[np.ndarray, tuple[MissEvent, ...]]:
    """Feature matrix for every miss in ``log``.

    Returns ``(matrix, events)`` with ``matrix`` of shape
    ``(len(events), len(FEATURE_NAMES))``. An empty miss set gives a correctly
    shaped empty matrix rather than a scalar, so a caller stacking several runs
    never has to special-case a clean one.
    """
    selected = miss_events(log) if events is None else events
    if not selected:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float), ()
    rows = [feature_vector(log, event) for event in selected]
    return np.vstack(rows), tuple(selected)
