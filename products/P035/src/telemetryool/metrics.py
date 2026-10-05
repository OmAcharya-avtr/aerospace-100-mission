"""Confusion matrices in full, window-level ROC, and detection-delay statistics.

Three rules are enforced here rather than left to the caller.

**Confusion matrices are reported entire.** :class:`ConfusionMatrix` carries all
four counts and renders them as a labelled 2x2 table.  Derived scalars are
available as properties, but no single scalar is ever returned in place of the
matrix, because a single score hides the trade the operating point actually
made.

**The ROC is at the window level.** The unit of a false alarm in operations is
"this pass raised an alarm it should not have", not "this sample exceeded a
threshold", so the ROC sweeps the window trigger level of
:func:`telemetryool.calibration.window_trigger_level`.  The false-positive axis
is then exactly the quantity the detectors were matched on.

**Detection delay is always reported with the detection probability beside
it.** A detector that finds 20 % of events after 2 samples is not faster than
one that finds 95 % after 8; the mean delay alone cannot tell them apart.
:class:`DelayStats` keeps both, and the mean is explicitly conditional on
detection inside the window.

References
----------
Fawcett, T. (2006). "An introduction to ROC analysis." *Pattern Recognition
    Letters* 27(8), 861-874.  The trapezoidal AUC and the threshold-sweep
    construction used by :func:`window_roc`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .calibration import estimate_rate, window_trigger_level
from .runs import first_run_index

__all__ = ["ConfusionMatrix", "confusion_matrix", "RocCurve", "window_roc", "DelayStats",
           "delay_stats"]


@dataclass(frozen=True)
class ConfusionMatrix:
    """All four counts of a binary decision, plus the derived rates.

    Attributes
    ----------
    true_positive, false_positive, false_negative, true_negative
        Counts of windows.
    """

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def n_positive(self) -> int:
        """Windows that actually contained an anomaly."""
        return self.true_positive + self.false_negative

    @property
    def n_negative(self) -> int:
        """Windows that were nominal."""
        return self.false_positive + self.true_negative

    @property
    def total(self) -> int:
        """All windows."""
        return self.n_positive + self.n_negative

    @property
    def tpr(self) -> float:
        """True-positive rate (detection probability), or nan with no positives."""
        return self.true_positive / self.n_positive if self.n_positive else float("nan")

    @property
    def fpr(self) -> float:
        """False-positive rate (window false-alarm probability), or nan."""
        return self.false_positive / self.n_negative if self.n_negative else float("nan")

    @property
    def precision(self) -> float:
        """Positive predictive value, or nan if nothing was flagged."""
        flagged = self.true_positive + self.false_positive
        return self.true_positive / flagged if flagged else float("nan")

    @property
    def specificity(self) -> float:
        """True-negative rate, or nan with no negatives."""
        return self.true_negative / self.n_negative if self.n_negative else float("nan")

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall, or nan if either is undefined."""
        p, r = self.precision, self.tpr
        if not np.isfinite(p) or not np.isfinite(r) or (p + r) == 0:
            return float("nan")
        return 2.0 * p * r / (p + r)

    @property
    def accuracy(self) -> float:
        """Fraction of windows classified correctly.

        Reported for completeness and not used for ranking: with the class
        balance of a monitoring problem it is dominated by the negatives.
        """
        return (self.true_positive + self.true_negative) / self.total if self.total else float(
            "nan"
        )

    def table(self, indent: str = "") -> str:
        """The full 2x2 matrix as a labelled block of text."""
        rows = [
            f"{indent}                  predicted alarm   predicted nominal        total",
            f"{indent}actual anomaly  {self.true_positive:17d} {self.false_negative:19d} "
            f"{self.n_positive:12d}",
            f"{indent}actual nominal  {self.false_positive:17d} {self.true_negative:19d} "
            f"{self.n_negative:12d}",
            f"{indent}total           {self.true_positive + self.false_positive:17d} "
            f"{self.false_negative + self.true_negative:19d} {self.total:12d}",
        ]
        return "\n".join(rows)

    def summary(self) -> str:
        """One line of derived rates, each named."""
        return (
            f"tpr={self.tpr:.4f} fpr={self.fpr:.4f} precision={self.precision:.4f} "
            f"specificity={self.specificity:.4f} f1={self.f1:.4f} "
            f"accuracy={self.accuracy:.4f}"
        )


def confusion_matrix(
    alarmed_anomalous: NDArray[np.bool_], alarmed_nominal: NDArray[np.bool_]
) -> ConfusionMatrix:
    """Build a :class:`ConfusionMatrix` from two per-window alarm indicator arrays.

    Parameters
    ----------
    alarmed_anomalous
        Shape ``(n_positive,)``: did each anomalous window raise an alarm.
    alarmed_nominal
        Shape ``(n_negative,)``: did each nominal window raise an alarm.
    """
    pos = np.asarray(alarmed_anomalous, dtype=bool)
    neg = np.asarray(alarmed_nominal, dtype=bool)
    if pos.ndim != 1 or neg.ndim != 1:
        raise ValueError(
            f"both indicator arrays must be 1-D, got {pos.shape} and {neg.shape}"
        )
    tp = int(pos.sum())
    fp = int(neg.sum())
    return ConfusionMatrix(tp, fp, int(pos.size) - tp, int(neg.size) - fp)


@dataclass(frozen=True)
class RocCurve:
    """Window-level ROC.

    Attributes
    ----------
    thresholds
        Score thresholds, descending, with ``+inf`` prepended and ``-inf``
        appended so the curve runs from ``(0, 0)`` to ``(1, 1)``.
    fpr, tpr
        Same length as ``thresholds``.
    auc
        Trapezoidal area under the curve, dimensionless in [0, 1].
    """

    thresholds: NDArray[np.float64]
    fpr: NDArray[np.float64]
    tpr: NDArray[np.float64]
    auc: float

    def tpr_at_fpr(self, target_fpr: float) -> float:
        """TPR at the largest threshold whose FPR does not exceed ``target_fpr``.

        Returns ``nan`` if no point on the curve satisfies the constraint.
        """
        ok = self.fpr <= float(target_fpr)
        return float(self.tpr[ok].max()) if ok.any() else float("nan")


def window_roc(
    nominal_scores: NDArray[np.float64],
    anomalous_scores: NDArray[np.float64],
    persistence: int,
    max_points: int = 512,
) -> RocCurve:
    """Window-level ROC from score blocks, sweeping the window trigger level.

    Parameters
    ----------
    nominal_scores, anomalous_scores
        Shape ``(n_windows, window_length)``.
    persistence
        Debounce count used by the detector, >= 1.
    max_points
        Curve is thinned to at most this many thresholds, chosen as evenly
        spaced quantiles of the pooled trigger levels, so the stored curve is
        small without distorting the AUC (the AUC is computed on the thinned
        curve; with 512 points the trapezoidal error is below 1e-4 for the
        curves in this package).

    Returns
    -------
    RocCurve
    """
    w_neg = window_trigger_level(nominal_scores, persistence)
    w_pos = window_trigger_level(anomalous_scores, persistence)
    pooled = np.concatenate([w_neg, w_pos])
    qs = np.linspace(0.0, 1.0, min(max_points, pooled.size))
    cuts = np.unique(np.quantile(pooled, qs))
    # +inf pins the curve at (0, 0) and -inf at (1, 1), so the trapezoidal AUC
    # is taken over the whole unit square as the ROC convention requires.
    thresholds = np.concatenate([[np.inf], cuts[::-1], [-np.inf]])
    fpr = np.array([float((w_neg > t).mean()) for t in thresholds])
    tpr = np.array([float((w_pos > t).mean()) for t in thresholds])
    order = np.argsort(fpr, kind="stable")
    auc = float(np.trapezoid(tpr[order], fpr[order]))
    return RocCurve(thresholds, fpr, tpr, auc)


@dataclass(frozen=True)
class DelayStats:
    """Detection delay for an anomaly with a known onset.

    Attributes
    ----------
    onset
        Sample index at which the anomaly began.
    n_windows
        Anomalous windows evaluated.
    n_detected
        Windows alarming at or after ``onset`` within the window.
    n_early
        Windows alarming strictly *before* ``onset``.  These are false alarms
        that happen to occur in an anomalous window; they are excluded from the
        delay statistics and reported separately rather than counted as
        instant detections.
    detection_probability
        ``n_detected / n_windows``.
    detection_probability_se
        Binomial standard error of that estimate.
    mean_delay, median_delay, p10_delay, p90_delay
        Samples, **conditional on detection inside the window**.  ``nan`` when
        nothing was detected.
    delays
        The individual delays, samples, for the detected windows.
    """

    onset: int
    n_windows: int
    n_detected: int
    n_early: int
    detection_probability: float
    detection_probability_se: float
    mean_delay: float
    median_delay: float
    p10_delay: float
    p90_delay: float
    delays: NDArray[np.int64]

    def summary(self) -> str:
        """One line: detection probability with its SE, then the delay quantiles."""
        return (
            f"Pd={self.detection_probability:.4f}+/-{self.detection_probability_se:.4f} "
            f"delay mean={self.mean_delay:.2f} median={self.median_delay:.1f} "
            f"p10={self.p10_delay:.1f} p90={self.p90_delay:.1f} samples "
            f"(n_detected={self.n_detected}/{self.n_windows}, early={self.n_early})"
        )


def delay_stats(
    anomalous_scores: NDArray[np.float64],
    threshold: float,
    persistence: int,
    onset: int,
) -> DelayStats:
    """Detection delay at a fixed threshold for anomalous windows with known onset.

    Parameters
    ----------
    anomalous_scores
        Shape ``(n_windows, window_length)``.
    threshold
        Calibrated score threshold.
    persistence
        Debounce count, >= 1.
    onset
        Sample index at which the anomaly begins, in ``[0, window_length)``.

    Returns
    -------
    DelayStats

    Notes
    -----
    The delay is ``alarm_index - onset``, so a detector whose debounce count is
    ``p`` cannot report a delay below ``p - 1`` for an anomaly that only becomes
    visible at ``onset``.  Windows alarming before ``onset`` are counted in
    ``n_early`` and dropped from the delay statistics.
    """
    scores = np.asarray(anomalous_scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"anomalous_scores must be 2-D, got {scores.shape}")
    if not (0 <= int(onset) < scores.shape[1]):
        raise ValueError(
            f"onset {onset} is outside a window of length {scores.shape[1]}"
        )
    idx = first_run_index(scores > float(threshold), persistence)
    early = int(((idx >= 0) & (idx < int(onset))).sum())
    hit = idx >= int(onset)
    delays = (idx[hit] - int(onset)).astype(np.int64)
    n = int(scores.shape[0])
    n_det = int(hit.sum())
    est = estimate_rate(n_det, n)
    if n_det:
        mean = float(delays.mean())
        med = float(np.median(delays))
        p10 = float(np.percentile(delays, 10))
        p90 = float(np.percentile(delays, 90))
    else:
        mean = med = p10 = p90 = float("nan")
    return DelayStats(
        onset=int(onset),
        n_windows=n,
        n_detected=n_det,
        n_early=early,
        detection_probability=est.rate,
        detection_probability_se=est.standard_error,
        mean_delay=mean,
        median_delay=med,
        p10_delay=p10,
        p90_delay=p90,
        delays=delays,
    )
