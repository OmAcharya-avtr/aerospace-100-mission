"""Contention: how much demand is pressing on each antenna, when.

The measure
-----------
Divide the horizon into bins of ``bin_s`` seconds. For an antenna ``s`` and a
bin ``b``:

* **supply** is the union of the visibility windows on ``s`` inside ``b``, in
  seconds. It must be a union and not a sum, because an antenna serves one
  spacecraft at a time: two overlapping windows supply the time they jointly
  cover exactly once.

* **demand** apportions each request's declared requirement across the
  opportunities it has. A request needing ``requested_seconds`` with
  ``effective_seconds`` of usable window across all antennas it is allowed to
  use contributes, to each bin, ``requested_seconds * overlap(window, bin) /
  effective_seconds``. A request with many opportunities therefore presses
  lightly on each of them, and a request with one opportunity presses its whole
  requirement onto that one. This is the property that makes the ratio
  informative: a station that is merely busy is not contended, a station that is
  the *only* option for several requests at once is.

* **ratio** is demand over supply, dimensionless. Above 1 the bin is
  over-subscribed on this apportionment.

"Usable" window is deliberate: a window shorter than a request's required
contact duration offers that request nothing and is excluded from its
``effective_seconds``. Without that exclusion a request surrounded by useless
three-second windows would appear to have ample opportunity.

What this is not
----------------
The ratio is an apportionment, not a feasibility test. It does not know about
turnaround times, spacecraft radio exclusion, or the combinatorial structure
that decides who actually loses. A bin with ratio 0.4 can still starve a request
whose only window sits inside it alongside a higher-priority one. That gap --
between a scalar pressure measure and the actual outcome of the assignment -- is
precisely the gap the learned conflict predictor in
:mod:`contactsched.predictors` is scored on closing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import SchedulingProblem


@dataclass(frozen=True)
class ContentionCell:
    """Demand and supply in one antenna-time bin.

    Attributes
    ----------
    station : str
        Antenna name.
    bin_start_s, bin_end_s : float
        Bin edges, seconds after the scenario epoch.
    supply_s : float
        Union window coverage inside the bin, seconds.
    demand_s : float
        Apportioned requirement inside the bin, seconds.
    requests : tuple of str
        Request ids with a usable window overlapping this cell, sorted.
    """

    station: str
    bin_start_s: float
    bin_end_s: float
    supply_s: float
    demand_s: float
    requests: tuple[str, ...]

    @property
    def ratio(self) -> float:
        """Demand over supply, dimensionless.

        Returns ``inf`` when there is demand and no supply, and 0.0 when there
        is neither. The infinity is the honest value: a request that needs this
        bin and finds no window in it has unbounded pressure there, and
        replacing that with a large finite number would hide it.
        """
        if self.supply_s <= 0.0:
            return float("inf") if self.demand_s > 0.0 else 0.0
        return float(self.demand_s / self.supply_s)


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection of two intervals, seconds; zero if disjoint."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def effective_seconds(problem: SchedulingProblem) -> dict[str, float]:
    """Usable window seconds per request, summed over allowed antennas.

    A window counts only if it is at least as long as the request's required
    contact duration.
    """
    out: dict[str, float] = {}
    for request in problem.requests:
        total = 0.0
        for window in problem.windows:
            if window.spacecraft != request.spacecraft:
                continue
            if not request.allows(window.station):
                continue
            if window.duration_s + 1e-9 < request.duration_s:
                continue
            total += window.duration_s
        out[request.request_id] = float(total)
    return out


def contention_map(
    problem: SchedulingProblem, horizon_s: float, bin_s: float = 1800.0
) -> list[ContentionCell]:
    """Demand, supply and ratio for every antenna-time bin.

    Parameters
    ----------
    problem : SchedulingProblem
        The instance.
    horizon_s : float
        Length of the scheduling horizon, seconds. Must be positive.
    bin_s : float
        Bin width, seconds. Must be positive. A bin far narrower than a typical
        pass makes the ratio noisy; a bin far wider hides the peak. The examples
        use 1800 s against passes of roughly 500 s and that choice is visible in
        ``screenshots/contention_map.png``.

    Returns
    -------
    list of ContentionCell
        Ordered by antenna then bin start.

    Raises
    ------
    ValueError
        On a non-positive horizon or bin width.
    """
    if float(horizon_s) <= 0.0:
        raise ValueError(f"horizon_s must be positive, got {horizon_s}")
    if float(bin_s) <= 0.0:
        raise ValueError(f"bin_s must be positive, got {bin_s}")
    edges = list(np.arange(0.0, float(horizon_s), float(bin_s))) + [float(horizon_s)]
    usable = effective_seconds(problem)
    cells: list[ContentionCell] = []
    for station in sorted({s.name for s in problem.stations}):
        station_windows = [w for w in problem.windows if w.station == station]
        for start, end in zip(edges[:-1], edges[1:], strict=True):
            intervals = [
                (max(w.rise_s, start), min(w.set_s, end))
                for w in station_windows
                if _overlap(w.rise_s, w.set_s, start, end) > 0.0
            ]
            merged: list[list[float]] = []
            for low, high in sorted(intervals):
                if merged and low <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], high)
                else:
                    merged.append([low, high])
            supply = float(sum(high - low for low, high in merged))
            demand = 0.0
            touching: set[str] = set()
            for request in problem.requests:
                if not request.allows(station):
                    continue
                total_usable = usable[request.request_id]
                if total_usable <= 0.0:
                    continue
                for window in station_windows:
                    if window.spacecraft != request.spacecraft:
                        continue
                    if window.duration_s + 1e-9 < request.duration_s:
                        continue
                    shared = _overlap(window.rise_s, window.set_s, start, end)
                    if shared <= 0.0:
                        continue
                    demand += request.requested_seconds * shared / total_usable
                    touching.add(request.request_id)
            cells.append(
                ContentionCell(
                    station=station,
                    bin_start_s=float(start),
                    bin_end_s=float(end),
                    supply_s=supply,
                    demand_s=float(demand),
                    requests=tuple(sorted(touching)),
                )
            )
    return cells


def request_contention(
    problem: SchedulingProblem, cells: list[ContentionCell]
) -> dict[str, dict[str, float]]:
    """Per-request contention summaries derived from the map.

    Returns
    -------
    dict
        ``{request_id: {"ratio_max": float, "ratio_weighted": float,
        "cells": float, "usable_seconds": float, "demand_over_usable": float}}``.

        ``ratio_max`` is the largest ratio over the cells the request touches
        and is the sharpest single pressure number. ``ratio_weighted`` is the
        mean of those ratios weighted by the request's own apportioned demand in
        each cell, which is smoother and less sensitive to one empty bin. Both
        are published as conflict-prediction baselines in
        :mod:`contactsched.predictors` because which of the two is stronger is a
        measurement, not an assumption, and it is reported in
        ``validation/validate_predictors_output.txt``.

        Infinite ratios are retained in ``ratio_max`` and excluded from
        ``ratio_weighted``, whose weights would otherwise be meaningless; the
        count of excluded cells is not reported per request and that is a stated
        limitation of this summary.
    """
    usable = effective_seconds(problem)
    totals: dict[str, dict[str, float]] = {}
    for request in problem.requests:
        ratios: list[float] = []
        weights: list[float] = []
        touched = 0
        for cell in cells:
            if request.request_id not in cell.requests:
                continue
            touched += 1
            ratios.append(cell.ratio)
            weights.append(cell.demand_s)
        finite = [(r, w) for r, w in zip(ratios, weights, strict=True) if np.isfinite(r)]
        weight_sum = sum(w for _r, w in finite)
        weighted = (
            float(sum(r * w for r, w in finite) / weight_sum)
            if weight_sum > 0.0
            else float(np.mean([r for r, _w in finite])) if finite else 0.0
        )
        usable_seconds = usable[request.request_id]
        totals[request.request_id] = {
            "ratio_max": float(max(ratios)) if ratios else 0.0,
            "ratio_weighted": weighted,
            "cells": float(touched),
            "usable_seconds": float(usable_seconds),
            "demand_over_usable": float(
                request.requested_seconds / usable_seconds
                if usable_seconds > 0.0
                else np.inf
                if request.requested_seconds > 0.0
                else 0.0
            ),
        }
    return totals
