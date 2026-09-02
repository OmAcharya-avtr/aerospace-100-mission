"""Confusion matrices, binomial intervals and ROC curves.

Everything a benchmark table in this package prints comes from here, so the
same interval definition is used for every method.

Binomial intervals use the Wilson score interval (Wilson, *Journal of the
American Statistical Association*, 22(158), 1927, pp. 209-212) rather than the
normal approximation, because the false-alarm rates measured in this package
are small and the normal interval is badly wrong there -- it can even reach
below zero.  When a measured false-alarm rate has to be compared with its
design value, the honest question is whether the design value lies inside the
measured interval, and that is what :func:`wilson_interval` supports.

ROC curves are computed directly rather than through scikit-learn so that the
tie handling is explicit: thresholds are the sorted unique scores, and a
sample counts as detected when ``score >= threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm
from scipy.stats import t as student_t

__all__ = [
    "Interval",
    "wilson_interval",
    "mean_ci",
    "confusion_matrix",
    "ConfusionReport",
    "confusion_report",
    "RocCurve",
    "roc_curve",
]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided interval.

    Attributes
    ----------
    point, low, high : float
        Estimate and interval endpoints, in the units of the quantity.
    level : float
        Coverage level, e.g. 0.95.
    """

    point: float
    low: float
    high: float
    level: float = 0.95

    @property
    def half_width(self) -> float:
        """Half the interval width."""
        return 0.5 * (self.high - self.low)

    def contains(self, value: float) -> bool:
        """True if ``value`` lies in the closed interval."""
        return bool(self.low <= float(value) <= self.high)

    def __str__(self) -> str:
        return f"{self.point:.6g} [{self.low:.6g}, {self.high:.6g}]"


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes : int
        Number of successes (here: alarms), ``0 <= successes <= trials``.
    trials : int
        Number of independent trials, ``>= 1``.
    level : float
        Two-sided coverage, in ``(0, 1)``.

    Returns
    -------
    Interval
        Point estimate ``successes / trials`` with the Wilson endpoints.

    Examples
    --------
    >>> iv = wilson_interval(10, 1000)
    >>> round(iv.point, 4), round(iv.low, 5), round(iv.high, 5)
    (0.01, 0.00543, 0.01833)
    """
    n = int(trials)
    k = int(successes)
    if n < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    if not (0 <= k <= n):
        raise ValueError(f"successes must lie in [0, {n}], got {successes}")
    lv = float(level)
    if not (0.0 < lv < 1.0):
        raise ValueError(f"level must lie in (0, 1), got {level}")
    z = float(norm.isf(0.5 * (1.0 - lv)))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = 0.0 if k == 0 else max(0.0, float(centre - half))
    high = 1.0 if k == n else min(1.0, float(centre + half))
    # The endpoints at k = 0 and k = n are exactly 0 and 1 in exact
    # arithmetic; the special cases keep them exact rather than leaving a
    # 3e-18 residue that makes "low <= point <= high" fail.
    return Interval(point=p, low=low, high=high, level=lv)


def mean_ci(values: ArrayLike, level: float = 0.95) -> Interval:
    """Student-t interval on the mean of a sample.

    Parameters
    ----------
    values : array_like
        Finite sample, at least 2 elements.  Non-finite entries (an ``inf``
        detection delay, for instance) must be removed or replaced by the
        caller: silently dropping them would misreport the mean.
    level : float
        Two-sided coverage.

    Returns
    -------
    Interval
    """
    a = np.asarray(values, dtype=float).reshape(-1)
    if a.size < 2:
        raise ValueError(f"need at least 2 values, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError(
            "values must all be finite; censored (never-detected) cases must be "
            "handled explicitly, not averaged away"
        )
    lv = float(level)
    if not (0.0 < lv < 1.0):
        raise ValueError(f"level must lie in (0, 1), got {level}")
    mean = float(np.mean(a))
    sem = float(np.std(a, ddof=1) / np.sqrt(a.size))
    crit = float(student_t.isf(0.5 * (1.0 - lv), a.size - 1))
    return Interval(point=mean, low=mean - crit * sem, high=mean + crit * sem, level=lv)


def confusion_matrix(
    truth: ArrayLike, predicted: ArrayLike, n_classes: int
) -> NDArray[np.int64]:
    """Counts with truth on rows and prediction on columns.

    Parameters
    ----------
    truth, predicted : array_like of int
        Class indices in ``[0, n_classes)``, same length.
    n_classes : int
        Number of classes.

    Returns
    -------
    ndarray, shape (n_classes, n_classes), dtype int64
        ``M[i, j]`` is the number of samples of true class ``i`` predicted as
        class ``j``.
    """
    y = np.asarray(truth, dtype=int).reshape(-1)
    p = np.asarray(predicted, dtype=int).reshape(-1)
    c = int(n_classes)
    if y.size != p.size:
        raise ValueError(f"truth has {y.size} entries, predicted has {p.size}")
    if c < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")
    if y.size and (y.min() < 0 or y.max() >= c or p.min() < 0 or p.max() >= c):
        raise ValueError(f"class indices must lie in [0, {c})")
    m = np.zeros((c, c), dtype=np.int64)
    np.add.at(m, (y, p), 1)
    return m


@dataclass(frozen=True)
class ConfusionReport:
    """Per-class recall and precision alongside the full matrix."""

    matrix: NDArray[np.int64]
    labels: tuple[str, ...]
    recall: NDArray[np.float64] = field(repr=False)
    precision: NDArray[np.float64] = field(repr=False)
    accuracy: float = 0.0

    def to_text(self, width: int = 10) -> str:
        """Fixed-width rendering of the whole matrix, one row per true class."""
        head = " " * (width + 2) + "".join(f"{label[:width - 1]:>{width}}" for label in self.labels)
        lines = [head + f"{'recall':>{width}}"]
        for i, label in enumerate(self.labels):
            row = "".join(f"{int(v):>{width}}" for v in self.matrix[i])
            lines.append(
                f"{label[: width + 1]:<{width + 2}}" + row + f"{self.recall[i]:>{width}.4f}"
            )
        prec = "".join(f"{v:>{width}.4f}" for v in self.precision)
        lines.append(f"{'precision':<{width + 2}}" + prec)
        lines.append(f"overall accuracy = {self.accuracy:.6f}")
        return "\n".join(lines)


def confusion_report(
    truth: ArrayLike, predicted: ArrayLike, labels: tuple[str, ...]
) -> ConfusionReport:
    """Confusion matrix plus per-class recall and precision.

    Classes with no support get ``nan`` recall; classes never predicted get
    ``nan`` precision.  Neither is replaced by zero, because a zero would read
    as a measurement and it is not one.
    """
    m = confusion_matrix(truth, predicted, len(labels))
    support = m.sum(axis=1).astype(float)
    predicted_count = m.sum(axis=0).astype(float)
    diag = np.diag(m).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, diag / np.where(support > 0, support, 1.0), np.nan)
        precision = np.where(
            predicted_count > 0, diag / np.where(predicted_count > 0, predicted_count, 1.0), np.nan
        )
    total = float(m.sum())
    accuracy = float(diag.sum() / total) if total > 0 else float("nan")
    return ConfusionReport(
        matrix=m, labels=tuple(labels), recall=recall, precision=precision, accuracy=accuracy
    )


@dataclass(frozen=True)
class RocCurve:
    """A receiver operating characteristic.

    Attributes
    ----------
    fpr, tpr : ndarray, shape (K,)
        False- and true-positive rates, both ascending from ``(0, 0)`` to
        ``(1, 1)``.
    thresholds : ndarray, shape (K,)
        Score threshold for each point; the first is ``+inf``.
    auc : float
        Area under the curve by the trapezoidal rule.
    label : str
        Method name.
    """

    fpr: NDArray[np.float64]
    tpr: NDArray[np.float64]
    thresholds: NDArray[np.float64] = field(repr=False)
    auc: float = 0.0
    label: str = ""

    def tpr_at_fpr(self, target_fpr: float) -> float:
        """True-positive rate at the largest operating point with
        ``fpr <= target_fpr``, by linear interpolation in ``fpr``."""
        f = float(target_fpr)
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"target_fpr must lie in [0, 1], got {target_fpr}")
        return float(np.interp(f, self.fpr, self.tpr))


def roc_curve(
    scores_positive: ArrayLike, scores_negative: ArrayLike, label: str = ""
) -> RocCurve:
    """ROC from two score samples.

    Parameters
    ----------
    scores_positive : array_like
        Scores of faulted (positive) cases; higher means more fault-like.
    scores_negative : array_like
        Scores of fault-free (negative) cases.
    label : str
        Method name carried into the result.

    Returns
    -------
    RocCurve

    Notes
    -----
    A sample counts as detected when ``score >= threshold``, so tied scores
    are all detected or all missed together.  This matters here because the
    CUSUM statistic is exactly ``0`` for a large fraction of fault-free
    samples, and a tie-splitting convention would invent resolution that the
    detector does not have.
    """
    pos = np.asarray(scores_positive, dtype=float).reshape(-1)
    neg = np.asarray(scores_negative, dtype=float).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("both score samples must be non-empty")
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(neg))):
        raise ValueError("scores must be finite")
    thresholds = np.concatenate(([np.inf], np.unique(np.concatenate([pos, neg]))[::-1]))
    tpr = np.array([float(np.mean(pos >= t)) for t in thresholds])
    fpr = np.array([float(np.mean(neg >= t)) for t in thresholds])
    order = np.argsort(fpr, kind="stable")
    fpr_s, tpr_s = fpr[order], tpr[order]
    auc = float(np.trapezoid(tpr_s, fpr_s))
    return RocCurve(fpr=fpr, tpr=tpr, thresholds=thresholds, auc=auc, label=label)
