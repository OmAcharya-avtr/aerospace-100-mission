"""Limit definition and limit-crossing detection for a degrading health metric.

A *limit* here is a single declared scalar threshold on one housekeeping health
metric, with a declared direction. Two directions are supported:

``"below"``
    The metric decreases toward a lower limit. Example: battery usable capacity
    in ampere-hours against an end-of-life capacity floor.
``"above"``
    The metric increases toward an upper limit. Example: reaction-wheel bearing
    drag torque in newton-metres against a declared drag-torque ceiling.

Units
-----
The metric and the limit carry whatever unit the caller declares in
:class:`LimitSpec.unit`; this module never converts. Times are in the caller's
time unit (the shipped generator uses days). All tolerances are in metric units.

The step-function problem
-------------------------
"Has this trajectory crossed the limit" is a boolean produced by a strict
comparison of two floating-point numbers. It is a step function of a rounded
quantity, so it is not reproducible across algebraically equivalent rearrangements
of the same arithmetic. :func:`crossing_status` therefore returns a three-valued
result with an explicit tolerance band, and :func:`has_crossed` -- which returns
a plain :class:`bool` -- documents that its value is only meaningful when the
margin is outside that band.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Direction = Literal["below", "above"]
CrossingStatus = Literal["crossed", "not_crossed", "within_tolerance"]

__all__ = [
    "CrossingStatus",
    "Direction",
    "LimitSpec",
    "crossing_status",
    "first_crossing_time",
    "has_crossed",
    "margin",
]


@dataclass(frozen=True)
class LimitSpec:
    """A declared scalar limit on one health metric.

    Parameters
    ----------
    value:
        The limit, in ``unit``.
    direction:
        ``"below"`` if the metric degrades downward toward ``value``;
        ``"above"`` if it degrades upward toward ``value``.
    unit:
        Free-text unit label for the metric and the limit. Not parsed.
    name:
        Free-text label for the metric.
    atol:
        Absolute tolerance band, in ``unit``, inside which the crossing boolean
        is declared indeterminate. Must be non-negative. Default 0.0 means the
        caller accepts a bare strict comparison and its reproducibility.
    """

    value: float
    direction: Direction = "below"
    unit: str = "dimensionless"
    name: str = "health_metric"
    atol: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError(f"limit value must be finite, got {self.value!r}")
        if self.direction not in ("below", "above"):
            raise ValueError(
                f"direction must be 'below' or 'above', got {self.direction!r}"
            )
        if not np.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError(f"atol must be finite and non-negative, got {self.atol!r}")


def margin(values: np.ndarray | float, limit: LimitSpec) -> np.ndarray:
    """Signed distance to the limit, positive while the metric is still inside it.

    For ``direction="below"`` this is ``value - limit``; for ``"above"`` it is
    ``limit - value``. Units: same as the metric. A margin of exactly zero is the
    crossing point of the declared limit.
    """
    arr = np.asarray(values, dtype=float)
    if limit.direction == "below":
        return arr - limit.value
    return limit.value - arr


def crossing_status(value: float, limit: LimitSpec) -> CrossingStatus:
    """Three-valued crossing test for a single metric value.

    Returns ``"within_tolerance"`` whenever ``abs(margin) <= limit.atol``, in
    which case the sign of the margin is not reproducible at this tolerance and
    no crossing claim is made. Outside the band the result is the ordinary
    comparison: ``"crossed"`` when the margin is negative.

    What is guaranteed: for ``abs(margin) > atol`` the result agrees with the
    exact-arithmetic answer provided the margin computation itself is accurate to
    better than ``atol``. Inside the band nothing is guaranteed and nothing is
    claimed.
    """
    m = float(margin(float(value), limit))
    if not np.isfinite(m):
        raise ValueError(f"margin is not finite (value={value!r}, limit={limit.value!r})")
    if abs(m) <= limit.atol:
        return "within_tolerance"
    return "crossed" if m < 0.0 else "not_crossed"


def has_crossed(value: float, limit: LimitSpec) -> bool:
    """Plain boolean crossing test: ``margin(value, limit) < 0``.

    This is a strict comparison on a floating-point quantity and is therefore a
    step function of a rounded number. Two algebraically identical routes to the
    same margin can land on opposite sides of zero. Use :func:`crossing_status`
    whenever the answer matters near the limit; this function exists for the case
    where the caller has already established that the margin is far from zero.
    """
    return bool(margin(float(value), limit) < 0.0)


def first_crossing_time(
    times: np.ndarray, values: np.ndarray, limit: LimitSpec
) -> float | None:
    """First time at which a sampled trajectory crosses ``limit``, or ``None``.

    The crossing is located by linear interpolation between the last sample with
    a non-negative margin and the first sample with a negative margin. For a
    smooth monotone trajectory sampled on a grid of step ``dt`` the interpolation
    error is ``O(dt**2)`` through the usual Taylor argument for linear
    interpolation of a twice-differentiable function; this is measured in
    ``validation/validate_crossing.py``.

    Parameters
    ----------
    times:
        Strictly increasing sample times, shape ``(n,)``.
    values:
        Metric values at ``times``, shape ``(n,)``.

    Returns
    -------
    float or None
        The interpolated crossing time, or ``None`` if no sample has a negative
        margin. ``None`` means *not observed to cross in this record*, which is a
        right-censoring statement, not a statement that the limit is never
        crossed.

    Raises
    ------
    ValueError
        If the inputs have different lengths, fewer than two samples, are not
        finite, or ``times`` is not strictly increasing.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.ndim != 1 or y.ndim != 1:
        raise ValueError("times and values must be one-dimensional")
    if t.shape != y.shape:
        raise ValueError(f"times and values must have the same shape, got {t.shape} and {y.shape}")
    if t.size < 2:
        raise ValueError(f"need at least two samples, got {t.size}")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("times and values must all be finite")
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("times must be strictly increasing")

    m = margin(y, limit)
    below = m < 0.0
    if not below.any():
        return None
    k = int(np.argmax(below))
    if k == 0:
        return float(t[0])
    m0, m1 = float(m[k - 1]), float(m[k])
    denom = m0 - m1
    if denom <= 0.0:
        return float(t[k])
    frac = m0 / denom
    return float(t[k - 1] + frac * (t[k] - t[k - 1]))
