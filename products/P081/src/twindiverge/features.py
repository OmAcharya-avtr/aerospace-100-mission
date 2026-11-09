"""Cheap features of a twin-versus-reality residual time series.

These are the input to the learned classifier of :mod:`twindiverge.learned`,
and the point of the list is what is NOT in it. The projections of the residual
onto the sensitivity columns -- ``a_j^T r`` -- are the sufficient statistics of
the linearised problem, and handing them to a classifier would make it a
re-implementation of baseline 2 with a decision tree on top, which would
measure nothing. Everything below is computed from the residual, the twin's own
trajectory and the logged current, with no sensitivity matrix and no
integration, so a classifier fitted on them is doing what the specification
asks: mapping a residual *signature* to a responsible parameter subset.

The sixteen declared features, on the noise-scaled rate residual ``r`` of
length ``n``:

====  ===========================================================
1-4   mean of ``r`` over each of four equal quarters of the record
5-8   least-squares slope of ``r`` over each quarter, per second
9     mean of ``r`` over the driven segment
10    mean of ``r`` over the coasting segment
11    root mean square of ``r``
12    largest absolute value of ``r``
13    position of that largest value, as a fraction of the record
14    Pearson correlation of ``r`` with the twin's rate
15    Pearson correlation of ``r`` with the square of the twin's rate
16    ratio of the driven-segment root mean square to the coasting one
====  ===========================================================

Features 14 and 15 are the ones that carry most of the viscous-versus-drag
information, and the fact that they are nearly perfectly correlated with each
other on this plant -- the rate excursion is small compared with the rate
itself -- is the same fact that makes the ``b`` and ``c_d`` directions
ill-conditioned in :mod:`twindiverge.identifiability`. The classifier inherits
the conditioning of the problem; it does not escape it.

A record with no driven segment or no coasting segment leaves features 9, 10
and 16 undefined. They are set to 0.0 and the substitution is reported by
:func:`feature_notes`, rather than being silently filled with a value that
would read as a measurement.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from .plant import Trajectory
from .telemetry import TelemetrySet

__all__ = ["FEATURE_NAMES", "extract", "extract_many", "feature_notes"]

#: Names of the sixteen declared features, in the column order of :func:`extract`.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "mean_q1",
    "mean_q2",
    "mean_q3",
    "mean_q4",
    "slope_q1",
    "slope_q2",
    "slope_q3",
    "slope_q4",
    "mean_driven",
    "mean_coast",
    "rms",
    "peak_abs",
    "peak_position",
    "corr_rate",
    "corr_rate_squared",
    "driven_coast_rms_ratio",
)


def _slope(times_s: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope of ``values`` against ``times_s``, per second.

    Returns 0.0 for a segment of fewer than two samples or of zero time span,
    because a slope is not defined there.
    """
    if values.size < 2:
        return 0.0
    centred_time = times_s - times_s.mean()
    denominator = float(centred_time @ centred_time)
    if denominator <= 0.0:
        return 0.0
    return float(centred_time @ (values - values.mean()) / denominator)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson correlation, or 0.0 when either series has no variance."""
    left = first - first.mean()
    right = second - second.mean()
    scale = float(np.linalg.norm(left) * np.linalg.norm(right))
    if scale <= 0.0:
        return 0.0
    return float(left @ right / scale)


def feature_notes(telemetry: TelemetrySet) -> tuple[str, ...]:
    """Features that are substituted rather than measured for this telemetry set."""
    campaign = telemetry.campaign
    notes = []
    if campaign.drive_duration_s <= 0.0:
        notes.append(
            "mean_driven and driven_coast_rms_ratio are set to 0.0: this campaign "
            "has no driven segment"
        )
    if campaign.drive_duration_s >= campaign.duration_s:
        notes.append(
            "mean_coast and driven_coast_rms_ratio are set to 0.0: this campaign "
            "has no coasting segment"
        )
    return tuple(notes)


def extract(
    telemetry: TelemetrySet, observed: np.ndarray, twin: Trajectory
) -> np.ndarray:
    """The sixteen declared features of one residual, shape ``(16,)``.

    Only the rate channel is used, whatever channels ``telemetry`` records: the
    learned component is specified as a residual-*time-series* classifier and
    the rate residual is the time series a twin produces. The torque channel's
    contribution to identifiability is baseline 1's business.
    """
    if "rate" not in telemetry.channels:
        raise ValueError(
            f"telemetry set {telemetry.name!r} records no rate channel, and the "
            "residual-signature features are defined on the rate residual"
        )
    array = np.asarray(observed, dtype=float)
    if array.shape != (telemetry.rows,):
        raise ValueError(
            f"telemetry set {telemetry.name!r} has {telemetry.rows} rows but the "
            f"observation has shape {array.shape}"
        )
    offset = telemetry.channels.index("rate") * telemetry.campaign.samples
    width = telemetry.campaign.samples
    residual = (array[offset : offset + width] - twin.rate_rad_s) / telemetry.sigmas["rate"]
    times = twin.times_s
    rate = twin.rate_rad_s

    edges = np.linspace(0, width, 5).astype(int)
    quarters = [slice(edges[index], edges[index + 1]) for index in range(4)]
    means = [float(residual[part].mean()) if residual[part].size else 0.0 for part in quarters]
    slopes = [_slope(times[part], residual[part]) for part in quarters]

    driven = times < telemetry.campaign.drive_duration_s - 1e-12
    coast = ~driven
    mean_driven = float(residual[driven].mean()) if driven.any() else 0.0
    mean_coast = float(residual[coast].mean()) if coast.any() else 0.0
    rms_driven = float(np.sqrt((residual[driven] ** 2).mean())) if driven.any() else 0.0
    rms_coast = float(np.sqrt((residual[coast] ** 2).mean())) if coast.any() else 0.0
    ratio = rms_driven / rms_coast if rms_coast > 0.0 else 0.0

    peak_index = int(np.argmax(np.abs(residual)))
    return np.array(
        [
            *means,
            *slopes,
            mean_driven,
            mean_coast,
            float(np.sqrt((residual**2).mean())),
            float(np.abs(residual).max()),
            peak_index / max(width - 1, 1),
            _correlation(residual, rate),
            _correlation(residual, rate * rate),
            ratio,
        ],
        dtype=float,
    )


def extract_many(
    telemetry: TelemetrySet, observations: np.ndarray, twin: Trajectory
) -> np.ndarray:
    """Features for a stack of observations, shape ``(cases, 16)``."""
    stack = np.atleast_2d(np.asarray(observations, dtype=float))
    return np.vstack([extract(telemetry, row, twin) for row in stack])
