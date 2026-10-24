"""Static features for the risk ranker. Nothing here looks at the forward walk.

The ranker's purpose is triage of a long sequence for human review, so its
inputs must be things that are available before anybody has walked the
sequence: the declarations and the shape of the sequence. Every feature in
:data:`FEATURE_NAMES` is computed from the command dictionary, the declared
initial state and the sequence's command and time columns.

In particular there is no feature derived from
:func:`cmdguard.trace.forward_execute`. A feature that encoded the forward
walk's own conclusion would make the ranker's measured precision a statement
about label leakage rather than about commanding, and the leak would be
invisible in the score. The absence is asserted by test: ``test_features.py``
checks that the feature matrix of a case is unchanged when the case's
violations are changed by editing *only* the parts of the declaration the
features do not read.

Feature list
------------
Twenty features, in a fixed order, with units.

0  ``irreversible``                  1 if the command is flagged irreversible
1  ``n_preconditions``               count
2  ``n_unverified_preconditions``    count of preconditions the *declared
                                     initial state* leaves unverified
3  ``n_false_preconditions``         count of preconditions the declared
                                     initial state contradicts
4  ``power_draw_w``                  declared draw on ``power_w``, or 0
5  ``data_draw_mbps``                declared draw on ``data_rate_mbps``, or 0
6  ``draw_fraction_of_ceiling``      largest declared draw over its ceiling
7  ``hold_s``                        seconds
8  ``min_interval_s``                seconds
9  ``min_gap_after_s``               seconds
10 ``n_required_modes``              count; 0 means ungated
11 ``n_effects``                     count of predicates the command writes
12 ``n_effects_false``               count it writes false or unverified
13 ``sets_mode``                     1 if the command declares a mode effect
14 ``exclusion_group_size``          members of its exclusion group, 0 if none
15 ``position_fraction``             step index over (length - 1), in [0, 1]
16 ``dt_prev_s``                     seconds since the previous step
17 ``dt_next_s``                     seconds to the next step
18 ``n_occurrences``                 times this mnemonic appears in the sequence
19 ``neighbours_within_60s``         steps within 60 s either side

The last feature's window is 60 s and that is a declaration, not a tuning
result: it is the order of the longest declared ``min_gap_after_s`` in the
shipped dictionary.
"""

from __future__ import annotations

import numpy as np

from .model import Case, Verity

__all__ = ["FEATURE_NAMES", "NEIGHBOUR_WINDOW_S", "feature_matrix", "feature_frame_rows"]

FEATURE_NAMES: tuple[str, ...] = (
    "irreversible",
    "n_preconditions",
    "n_unverified_preconditions",
    "n_false_preconditions",
    "power_draw_w",
    "data_draw_mbps",
    "draw_fraction_of_ceiling",
    "hold_s",
    "min_interval_s",
    "min_gap_after_s",
    "n_required_modes",
    "n_effects",
    "n_effects_false",
    "sets_mode",
    "exclusion_group_size",
    "position_fraction",
    "dt_prev_s",
    "dt_next_s",
    "n_occurrences",
    "neighbours_within_60s",
)

#: Half-width, in seconds, of the neighbour-count window. See module docstring.
NEIGHBOUR_WINDOW_S = 60.0


def feature_matrix(case: Case) -> np.ndarray:
    """Static feature matrix for ``case``, shape ``(len(sequence), 20)``.

    Returns
    -------
    numpy.ndarray
        Float64 array. An undeclared command contributes a row of zeros except
        for its positional and timing features, which is the honest encoding:
        nothing about its draw or preconditions is known.
    """
    steps = case.sequence.steps
    n = len(steps)
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    if n == 0:
        return out

    times = np.array([float(s.t_s) for s in steps], dtype=np.float64)
    occurrences: dict[str, int] = {}
    for step in steps:
        occurrences[step.command] = occurrences.get(step.command, 0) + 1
    group_sizes = {
        group: len(case.dictionary.exclusion_members(group))
        for group in {
            spec.exclusion_group
            for spec in case.dictionary.commands.values()
            if spec.exclusion_group
        }
    }
    limits = case.dictionary.resource_limits

    for index, step in enumerate(steps):
        spec = case.dictionary.get(step.command)
        row = out[index]
        if spec is not None:
            row[0] = 1.0 if spec.irreversible else 0.0
            row[1] = float(len(spec.preconditions))
            unverified = 0
            contradicted = 0
            for name, required in spec.preconditions.items():
                declared = case.initial.verity(name)
                if declared is Verity.UNKNOWN:
                    unverified += 1
                elif declared is not required:
                    contradicted += 1
            row[2] = float(unverified)
            row[3] = float(contradicted)
            row[4] = float(spec.resource_draw.get("power_w", 0.0))
            row[5] = float(spec.resource_draw.get("data_rate_mbps", 0.0))
            fraction = 0.0
            for key in spec.draw_keys:
                ceiling = limits.limit_for(key)
                if ceiling:
                    fraction = max(fraction, float(spec.resource_draw[key]) / ceiling)
            row[6] = fraction
            row[7] = float(spec.hold_s)
            row[8] = float(spec.min_interval_s)
            row[9] = float(spec.min_gap_after_s)
            row[10] = float(len(spec.required_modes))
            row[11] = float(len(spec.effects))
            row[12] = float(
                sum(1 for v in spec.effects.values() if v is not Verity.TRUE)
            )
            row[13] = 1.0 if spec.mode_effect is not None else 0.0
            row[14] = float(group_sizes.get(spec.exclusion_group, 0))
        row[15] = float(index) / float(n - 1) if n > 1 else 0.0
        row[16] = float(times[index] - times[index - 1]) if index > 0 else 0.0
        row[17] = float(times[index + 1] - times[index]) if index + 1 < n else 0.0
        row[18] = float(occurrences[step.command])
        row[19] = float(
            np.count_nonzero(np.abs(times - times[index]) <= NEIGHBOUR_WINDOW_S) - 1
        )
    return out


def feature_frame_rows(case: Case) -> list[dict[str, float]]:
    """``feature_matrix`` as one dict per step, for printing and debugging."""
    matrix = feature_matrix(case)
    return [
        {name: float(value) for name, value in zip(FEATURE_NAMES, row, strict=True)}
        for row in matrix
    ]
