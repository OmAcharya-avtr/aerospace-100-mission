"""Measuring whether an interval covers, and reporting where it does not.

Coverage here is always a *measured* fraction with a confidence interval on
it, never a nominal level restated. The confidence interval is Clopper-Pearson
(Clopper and Pearson 1934), the exact binomial interval, because the coverage
cells in the stratified table hold as few as a few dozen rows and a normal
approximation on 30 Bernoulli trials near 0.95 is not usable.

The one statistical point that governs every number in this module: the rows
are **grouped**. All the ages of one object share its features, its orbit and
its physics, so the effective sample size of a coverage cell is closer to the
number of objects in it than to the number of rows. Clopper-Pearson on the row
count therefore produces an interval that is too narrow, and this is stated
wherever such an interval is printed. :func:`coverage_by_group` gives the
object-level alternative -- one Bernoulli trial per object, the fraction of its
own ages that are covered thresholded at a half -- which is conservative in the
other direction. Both are reported and neither is presented as the answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval for a proportion.

    Parameters
    ----------
    successes, trials
        Counts; ``0 <= successes <= trials``.
    confidence
        Two-sided confidence level.

    Returns
    -------
    (low, high)
        ``(0.0, high)`` when ``successes == 0`` and ``(low, 1.0)`` when
        ``successes == trials``, which is the correct behaviour of the exact
        interval at the boundary rather than a degenerate zero-width interval.
    """
    if trials <= 0:
        raise ValueError(f"trials must be positive; got {trials}")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} must be in [0, {trials}]")
    alpha = 1.0 - confidence
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return low, high


@dataclass(frozen=True)
class IntervalMetrics:
    """Measured behaviour of one interval predictor on one set of rows.

    Attributes
    ----------
    n
        Rows.
    coverage
        Fraction with ``lower <= truth <= upper``.
    coverage_ci
        Clopper-Pearson interval on ``coverage``, treating rows as independent
        -- which they are not; see the module docstring.
    nominal
        The level the interval claimed.
    median_width_km, mean_width_km
        Width in km. Dominated by the largest drifts, so the log-ratio width
        below is the comparable number across altitude bands.
    median_log_width
        Median of ``log(upper / lower)``, a scale-free width.
    interval_score
        Mean Winkler interval score (Gneiting and Raftery 2007, eq. 43),
        ``(u - l) + (2/alpha)[(l - y)_+ + (y - u)_+]``, in km. Lower is
        better. This is the single number that trades width against coverage,
        and it is the one a reader should use to rank predictors; coverage and
        width separately can each be gamed.
    crossing_fraction
        Fraction of rows where ``lower > upper``, which unconstrained quantile
        regression permits and which this package reports rather than sorts
        away.
    """

    n: int
    coverage: float
    coverage_ci: tuple[float, float]
    nominal: float
    median_width_km: float
    mean_width_km: float
    median_log_width: float
    interval_score: float
    crossing_fraction: float


def interval_metrics(
    lower: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    *,
    nominal: float = 0.90,
    confidence: float = 0.95,
) -> IntervalMetrics:
    """Measure coverage, width and interval score of one predictor's intervals."""
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()
    y = np.asarray(truth, dtype=float).ravel()
    if not (low.shape == high.shape == y.shape):
        raise ValueError(f"shapes must match; got {low.shape}, {high.shape}, {y.shape}")
    usable = np.isfinite(low) & np.isfinite(high) & np.isfinite(y)
    low, high, y = low[usable], high[usable], y[usable]
    if low.size == 0:
        raise ValueError("no finite rows to measure")
    inside = (y >= np.minimum(low, high)) & (y <= np.maximum(low, high))
    successes = int(np.count_nonzero(inside))
    alpha = 1.0 - nominal
    width = np.maximum(high, low) - np.minimum(high, low)
    score = width + (2.0 / alpha) * (
        np.clip(np.minimum(low, high) - y, 0.0, None)
        + np.clip(y - np.maximum(low, high), 0.0, None)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_width = np.log(np.maximum(high, low) / np.minimum(high, low))
    return IntervalMetrics(
        n=int(low.size),
        coverage=successes / low.size,
        coverage_ci=clopper_pearson(successes, int(low.size), confidence),
        nominal=float(nominal),
        median_width_km=float(np.median(width)),
        mean_width_km=float(np.mean(width)),
        median_log_width=float(np.median(log_width[np.isfinite(log_width)])),
        interval_score=float(np.mean(score)),
        crossing_fraction=float(np.mean(low > high)),
    )


def coverage_by_group(
    lower: np.ndarray, upper: np.ndarray, truth: np.ndarray, groups: np.ndarray
) -> tuple[float, tuple[float, float], int]:
    """Object-level coverage: one trial per group, covered if over half its rows are.

    Returns ``(fraction, clopper_pearson_interval, n_groups)``. This is the
    conservative counterpart to row-level coverage and is reported beside it.
    """
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()
    y = np.asarray(truth, dtype=float).ravel()
    key = np.asarray(groups).ravel()
    inside = (y >= np.minimum(low, high)) & (y <= np.maximum(low, high))
    unique = np.unique(key)
    per_group = np.array([np.mean(inside[key == value]) for value in unique])
    successes = int(np.count_nonzero(per_group > 0.5))
    return (
        successes / unique.size,
        clopper_pearson(successes, int(unique.size)),
        int(unique.size),
    )


@dataclass(frozen=True)
class CoverageCell:
    """One cell of a stratified coverage table."""

    stratum: str
    n: int
    coverage: float
    coverage_ci: tuple[float, float]
    nominal: float
    median_log_width: float
    miscalibrated: bool

    @property
    def verdict(self) -> str:
        """One of ``"ok"``, ``"under"``, ``"over"`` or ``"thin"``.

        ``"ok"`` means the nominal level lies inside the measured coverage's
        confidence interval. ``"under"`` and ``"over"`` mean it lies below or
        above it, so the data are inconsistent with the claim. ``"thin"``
        means the cell was too small to be flagged either way, and is not a
        pass -- it is an absence of evidence, and the table prints it as such.
        """
        if self.miscalibrated:
            return "under" if self.coverage_ci[1] < self.nominal else "over"
        if not (self.coverage_ci[0] <= self.nominal <= self.coverage_ci[1]):
            return "thin"
        return "ok"


def coverage_table(
    lower: np.ndarray,
    upper: np.ndarray,
    truth: np.ndarray,
    strata: np.ndarray,
    *,
    nominal: float = 0.90,
    confidence: float = 0.95,
    minimum_cell: int = 20,
) -> list[CoverageCell]:
    """Coverage per stratum, with a cell flagged when nominal falls outside its interval.

    A cell is reported as miscalibrated when the nominal level is **outside**
    the Clopper-Pearson interval of the measured coverage -- that is, when the
    data are inconsistent with the claim at the stated confidence. A cell with
    fewer than ``minimum_cell`` rows is returned with ``n`` and its interval
    but is never flagged, because an interval that wide cannot be inconsistent
    with anything and flagging it would be noise.

    Cells are returned in the sorted order of the stratum labels, so the table
    is stable across runs and a diff of two committed outputs is readable.
    """
    low = np.asarray(lower, dtype=float).ravel()
    high = np.asarray(upper, dtype=float).ravel()
    y = np.asarray(truth, dtype=float).ravel()
    labels = np.asarray(strata).ravel()
    inside = (y >= np.minimum(low, high)) & (y <= np.maximum(low, high))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_width = np.log(np.maximum(high, low) / np.minimum(high, low))
    cells: list[CoverageCell] = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        n = int(np.count_nonzero(mask))
        successes = int(np.count_nonzero(inside[mask]))
        interval = clopper_pearson(successes, n, confidence)
        widths = log_width[mask]
        flagged = n >= minimum_cell and not (interval[0] <= nominal <= interval[1])
        cells.append(
            CoverageCell(
                stratum=str(label),
                n=n,
                coverage=successes / n,
                coverage_ci=interval,
                nominal=float(nominal),
                median_log_width=float(np.median(widths[np.isfinite(widths)]))
                if np.any(np.isfinite(widths))
                else float("nan"),
                miscalibrated=flagged,
            )
        )
    return cells


def altitude_stratum(perigee_altitude_km: np.ndarray) -> np.ndarray:
    """Label each row by a perigee-altitude band, as a sortable string."""
    altitude = np.asarray(perigee_altitude_km, dtype=float)
    edges = np.array([0.0, 400.0, 600.0, 800.0, 1100.0, np.inf])
    labels = [
        "alt 1 <400 km",
        "alt 2 400-600 km",
        "alt 3 600-800 km",
        "alt 4 800-1100 km",
        "alt 5 >1100 km",
    ]
    index = np.clip(np.searchsorted(edges, altitude, side="right") - 1, 0, len(labels) - 1)
    return np.array([labels[i] for i in index])


def age_stratum(epoch_age_days: np.ndarray) -> np.ndarray:
    """Label each row by an epoch-age band, as a sortable string."""
    age = np.asarray(epoch_age_days, dtype=float)
    edges = np.array([0.0, 1.0, 2.0, 4.0, np.inf])
    labels = ["age 1 <1 d", "age 2 1-2 d", "age 3 2-4 d", "age 4 >4 d"]
    index = np.clip(np.searchsorted(edges, age, side="right") - 1, 0, len(labels) - 1)
    return np.array([labels[i] for i in index])
