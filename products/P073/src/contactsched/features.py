"""Features available *before* scheduling, for the conflict predictor.

The constraint that shapes this module is that every feature must be computable
without solving the assignment. A conflict predictor whose inputs include the
schedule is not a predictor, and it would be useless for the task it exists for:
telling a mission planner, at request-submission time, that a request is about
to be dropped so it can be renegotiated instead.

Everything here is therefore derived from the visibility windows, the declared
requests and the contention map. The most informative features are the ones that
describe *who else wants the same antenna at the same time*, and in particular
the ones restricted to strictly higher-priority competitors, because the greedy
and exact methods both resolve ties in their favour.

One feature deserves separate mention. ``capacity_deficit`` is
``min_contacts - standalone_capacity``, where the standalone capacity is the
largest number of mutually compatible candidates the request has with the entire
network to itself. When it is positive the request is unsatisfiable *for reasons
that have nothing to do with contention* -- it asked for more contacts than its
own passes can carry. Those cases are trivially predictable and they are a real
part of the label, so they are kept, and the measurement in
``validation/validate_predictors_output.txt`` reports the learned and baseline
scores both over all requests and over the subset with non-positive deficit,
which is the hard part of the problem. Reporting only the first would flatter
both models.

Units: seconds, dimensionless ratios, integer counts.
"""

from __future__ import annotations

import numpy as np

from .contention import contention_map, request_contention
from .exact import standalone_capacity
from .problem import SchedulingProblem

FEATURE_NAMES: tuple[str, ...] = (
    "priority",
    "weight",
    "duration_s",
    "min_contacts",
    "max_contacts",
    "n_candidates",
    "n_windows",
    "n_allowed_stations",
    "usable_seconds",
    "demand_over_usable",
    "ratio_max",
    "ratio_weighted",
    "standalone_capacity",
    "capacity_deficit",
    "window_slack",
    "mean_max_elevation_deg",
    "competitor_count",
    "competitor_overlap_s",
    "higher_priority_competitor_count",
    "higher_priority_overlap_s",
    "higher_priority_overlap_fraction",
    "instance_requests",
    "instance_demand_over_supply",
)
"""Ordered feature names. The order is the column order of the design matrix.

It is fixed here and nowhere else, so a model trained in one session and a
feature vector built in another cannot silently disagree about column order.
``tests/test_features.py`` asserts that the extractor returns exactly these keys.
"""

BASELINE_FEATURES: tuple[str, ...] = ("demand_over_usable", "ratio_weighted", "ratio_max")
"""The three single-number contention baselines the learned model is scored against.

Ordered strongest first as measured, not as guessed: ``demand_over_usable``
reaches 0.9230 area under the ROC curve on 725 held-out requests where the two
per-bin ratios the specification names reach 0.6228 and 0.6010. The numbers and
the method are in ``validation/validate_predictors_output.txt``."""


def request_features(
    problem: SchedulingProblem, horizon_s: float, bin_s: float = 1800.0
) -> dict[str, dict[str, float]]:
    """Pre-scheduling features for every request in a problem.

    Parameters
    ----------
    problem : SchedulingProblem
        The instance.
    horizon_s : float
        Scheduling horizon, seconds.
    bin_s : float
        Contention map bin width, seconds.

    Returns
    -------
    dict
        ``{request_id: {feature_name: value}}``, with keys exactly
        :data:`FEATURE_NAMES`.
    """
    cells = contention_map(problem, horizon_s, bin_s)
    contention = request_contention(problem, cells)
    total_supply = float(sum(c.supply_s for c in cells))
    total_demand = float(sum(r.requested_seconds for r in problem.requests))
    instance_ratio = total_demand / total_supply if total_supply > 0.0 else np.inf

    windows_by_request: dict[str, list] = {}
    for request in problem.requests:
        windows_by_request[request.request_id] = [
            w
            for w in problem.windows
            if w.spacecraft == request.spacecraft
            and request.allows(w.station)
            and w.duration_s + 1e-9 >= request.duration_s
        ]

    out: dict[str, dict[str, float]] = {}
    for request in problem.requests:
        own = windows_by_request[request.request_id]
        candidates = problem.candidates_of(request.request_id)
        competitor_ids: set[str] = set()
        higher_ids: set[str] = set()
        overlap_total = 0.0
        higher_overlap = 0.0
        for other in problem.requests:
            if other.request_id == request.request_id:
                continue
            shared = 0.0
            for mine in own:
                for theirs in windows_by_request[other.request_id]:
                    if theirs.station != mine.station:
                        continue
                    shared += max(
                        0.0, min(mine.set_s, theirs.set_s) - max(mine.rise_s, theirs.rise_s)
                    )
            if shared <= 0.0:
                continue
            competitor_ids.add(other.request_id)
            overlap_total += shared
            if other.priority < request.priority:
                higher_ids.add(other.request_id)
                higher_overlap += shared
        own_seconds = float(sum(w.duration_s for w in own))
        capacity = standalone_capacity(problem, request.request_id)
        out[request.request_id] = {
            "priority": float(request.priority),
            "weight": float(candidates[0].weight) if candidates else 0.0,
            "duration_s": float(request.duration_s),
            "min_contacts": float(request.min_contacts),
            "max_contacts": float(request.max_contacts),
            "n_candidates": float(len(candidates)),
            "n_windows": float(len(own)),
            "n_allowed_stations": float(
                len(problem.stations) if request.stations is None else len(request.stations)
            ),
            "usable_seconds": own_seconds,
            "demand_over_usable": float(contention[request.request_id]["demand_over_usable"]),
            "ratio_max": float(contention[request.request_id]["ratio_max"]),
            "ratio_weighted": float(contention[request.request_id]["ratio_weighted"]),
            "standalone_capacity": float(capacity),
            "capacity_deficit": float(int(request.min_contacts) - capacity),
            "window_slack": float(
                np.mean([w.duration_s / request.duration_s for w in own]) if own else 0.0
            ),
            "mean_max_elevation_deg": float(
                np.mean([w.max_elevation_deg for w in own]) if own else 0.0
            ),
            "competitor_count": float(len(competitor_ids)),
            "competitor_overlap_s": float(overlap_total),
            "higher_priority_competitor_count": float(len(higher_ids)),
            "higher_priority_overlap_s": float(higher_overlap),
            "higher_priority_overlap_fraction": float(
                higher_overlap / overlap_total if overlap_total > 0.0 else 0.0
            ),
            "instance_requests": float(len(problem.requests)),
            "instance_demand_over_supply": float(instance_ratio),
        }
    return out


def design_matrix(features: dict[str, dict[str, float]], order: list[str]) -> np.ndarray:
    """Stack per-request feature dictionaries into a matrix in a fixed order.

    Parameters
    ----------
    features : dict
        As returned by :func:`request_features`.
    order : list of str
        Request ids, giving the row order.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(order), len(FEATURE_NAMES))``, float. Infinite values,
        which :func:`request_features` can legitimately produce for a request
        with no usable window, are replaced by a finite sentinel of 1e6 because
        the tree ensembles used downstream split on comparisons and handle a
        large finite value identically to an infinity, while some scalers do
        not. The substitution is recorded here rather than hidden in the model.
    """
    rows = [[features[request_id][name] for name in FEATURE_NAMES] for request_id in order]
    matrix = np.asarray(rows, dtype=float)
    return np.where(np.isfinite(matrix), matrix, 1.0e6)
