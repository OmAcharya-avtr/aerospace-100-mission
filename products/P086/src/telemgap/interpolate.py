"""Apply a :class:`~telemgap.policy.PolicyPlan`, and leave the refused gaps empty.

The function of this module is to be refusable. ``reconstruct`` places the
received samples on the nominal cadence grid and fills each gap interior
**only** where the policy permitted it, by the method the policy permitted.
Where the policy refused, the grid stays ``nan`` and the corresponding entries
of the ``refused`` mask are set, so a downstream consumer that wants to
interpolate anyway has to do it in its own code, under its own name, and not by
accident.

No part of this module decides anything. Every decision was made in
:mod:`telemgap.policy` and is carried in the plan.

Grid
----
The reconstruction grid runs from the earliest to the latest timestamp in the
record at the declared cadence:

    t_j = t_min + j * T,   j = 0 .. round((t_max - t_min) / T)          (10)

Received samples are placed at ``j = round((t_i - t_min) / T)``. Two samples
rounding to the same slot is a cadence anomaly the screens already reported; in
the grid the earlier one wins, and the count of such collisions is returned so
that it is never silent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policy import PolicyPlan
from .record import TelemetryRecord
from .screens import Gap, timeline_order


@dataclass
class Reconstruction:
    """A record placed on the cadence grid, with the refused gaps left empty.

    Attributes
    ----------
    times
        ``(m,)`` grid of equation (10), seconds.
    values
        ``(m, n_channels)`` engineering values. ``nan`` at every slot that was
        neither received nor filled.
    observed
        ``(m, n_channels)`` bool, true where a received sample was placed.
    synthesised
        ``(m, n_channels)`` bool, true where the policy permitted a fill and a
        value was written.
    refused
        ``(m, n_channels)`` bool, true where the policy refused and the slot
        was left ``nan``.
    method_used
        Per gap-decision method actually applied, keyed by
        ``(gap_index, channel)``.
    slot_collisions
        Number of received samples discarded because an earlier sample already
        occupied their grid slot.
    """

    times: np.ndarray
    values: np.ndarray
    observed: np.ndarray
    synthesised: np.ndarray
    refused: np.ndarray
    method_used: dict[tuple[int, int], str]
    slot_collisions: int

    @property
    def n_slots(self) -> int:
        """Number of grid slots."""
        return int(self.times.size)

    @property
    def fill_fraction(self) -> float:
        """Fraction of all channel-slots carrying a value, observed or synthesised."""
        total = self.values.size
        return float((self.observed | self.synthesised).sum() / total) if total else 0.0

    @property
    def refused_fraction(self) -> float:
        """Fraction of all channel-slots the policy refused to fill."""
        total = self.values.size
        return float(self.refused.sum() / total) if total else 0.0


def grid_for(record: TelemetryRecord) -> np.ndarray:
    """Equation (10): the nominal cadence grid spanning the record, seconds."""
    cadence = float(record.schema.cadence_s)
    start = float(np.min(record.timestamps))
    stop = float(np.max(record.timestamps))
    slots = int(np.rint((stop - start) / cadence))
    return start + cadence * np.arange(max(slots, 0) + 1, dtype=np.float64)


def _interior_slots(times: np.ndarray, gap: Gap, cadence: float) -> np.ndarray:
    """Grid slot indices strictly inside ``gap``, by a half-cadence margin."""
    margin = 0.5 * cadence
    return np.flatnonzero(
        (times > gap.start_time + margin) & (times < gap.end_time - margin)
    )


def reconstruct(
    record: TelemetryRecord,
    gaps: tuple[Gap, ...],
    plan: PolicyPlan,
) -> Reconstruction:
    """Place ``record`` on the cadence grid and fill only the permitted gaps.

    Parameters
    ----------
    record
        The record.
    gaps
        The gaps the plan was built for, in the same order, normally
        ``report.findings.gaps``.
    plan
        The :class:`~telemgap.policy.PolicyPlan`.

    Returns
    -------
    Reconstruction

    Raises
    ------
    ValueError
        If a decision in ``plan`` refers to a gap index outside ``gaps``, which
        would mean the plan and the gap list came from different screenings.
    """
    for decision in plan.decisions:
        if not 0 <= decision.gap_index < len(gaps):
            raise ValueError(
                f"plan decision refers to gap index {decision.gap_index} but only "
                f"{len(gaps)} gaps were given; the plan and the gap list disagree"
            )
    cadence = float(record.schema.cadence_s)
    times = grid_for(record)
    n_slots, n_channels = times.size, record.n_channels
    values = np.full((n_slots, n_channels), np.nan, dtype=np.float64)
    observed = np.zeros((n_slots, n_channels), dtype=bool)
    synthesised = np.zeros((n_slots, n_channels), dtype=bool)
    refused = np.zeros((n_slots, n_channels), dtype=bool)

    start = times[0] if n_slots else 0.0
    collisions = 0
    for index in timeline_order(record):
        slot = int(np.rint((record.timestamps[index] - start) / cadence))
        if not 0 <= slot < n_slots:
            continue
        if observed[slot].any():
            collisions += 1
            continue
        values[slot] = record.values[index]
        observed[slot] = True

    method_used: dict[tuple[int, int], str] = {}
    for decision in plan.decisions:
        gap = gaps[decision.gap_index]
        column = decision.channel
        interior = _interior_slots(times, gap, cadence)
        if interior.size == 0:
            method_used[(decision.gap_index, column)] = (
                "none" if decision.refused else decision.method
            )
            continue
        if decision.refused:
            refused[interior, column] = True
            method_used[(decision.gap_index, column)] = "none"
            continue
        left = float(record.values[gap.after_index, column])
        right = float(record.values[gap.before_index, column])
        if decision.method == "linear":
            fraction = (times[interior] - gap.start_time) / (gap.end_time - gap.start_time)
            filled = left + fraction * (right - left)
        else:
            filled = np.full(interior.size, left, dtype=np.float64)
        values[interior, column] = filled
        synthesised[interior, column] = True
        method_used[(decision.gap_index, column)] = decision.method

    return Reconstruction(
        times=times,
        values=values,
        observed=observed,
        synthesised=synthesised,
        refused=refused,
        method_used=method_used,
        slot_collisions=int(collisions),
    )


def nearest_indices(sorted_times: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the nearest entry of ``sorted_times`` for each entry of ``query``.

    Ties go to the lower index, which is arbitrary and stated rather than left
    to the reader to discover.
    """
    sorted_times = np.asarray(sorted_times, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if sorted_times.size == 0:
        raise ValueError("sorted_times must not be empty")
    upper = np.clip(np.searchsorted(sorted_times, query), 0, sorted_times.size - 1)
    lower = np.clip(upper - 1, 0, sorted_times.size - 1)
    take_lower = np.abs(query - sorted_times[lower]) <= np.abs(query - sorted_times[upper])
    return np.where(take_lower, lower, upper)


def measured_error(
    reconstruction: Reconstruction,
    truth_times: np.ndarray,
    truth_values: np.ndarray,
    column: int,
) -> tuple[float, int]:
    """Maximum absolute reconstruction error over the synthesised slots of one channel.

    Used by validation to compare the realised error against the bound the
    policy predicted. Each synthesised grid time is matched to the **nearest**
    truth time, not to the next one: the reconstruction grid is anchored on the
    first received timestamp, which carries the record's jitter, so a
    left-insertion lookup is off by one sample for every slot and would report
    a one-cadence difference of the signal as a reconstruction error.

    The caller is responsible for the truth series being on the same time base
    as the record. A record containing a clock reset is **not** on the same
    time base as the grid it was generated on, and this function cannot detect
    that; ``validation/validate_interpolation_policy.py`` therefore measures
    only on records generated with no clock reset, and says so.

    Returns
    -------
    (max_abs_error, n_compared)
        ``max_abs_error`` is ``nan`` when no slot was synthesised.
    """
    truth_times = np.asarray(truth_times, dtype=np.float64)
    truth_values = np.asarray(truth_values, dtype=np.float64)
    if truth_times.shape != truth_values.shape:
        raise ValueError(
            f"truth_times {truth_times.shape} and truth_values {truth_values.shape} "
            "must have the same shape"
        )
    slots = np.flatnonzero(reconstruction.synthesised[:, column])
    if slots.size == 0:
        return float("nan"), 0
    positions = nearest_indices(truth_times, reconstruction.times[slots])
    errors = np.abs(reconstruction.values[slots, column] - truth_values[positions])
    return float(np.max(errors)), int(slots.size)
