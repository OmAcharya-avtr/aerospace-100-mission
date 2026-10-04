"""The three metrics that decide this, plus the intervals around them.

Metrics
-------
``false_alarm_rate``
    ``FP / (FP + TN)`` — of the windows with no overrun, the fraction flagged.
    This is the cost of a predictor that sheds load it did not need to shed.

``missed_overrun_rate``
    ``FN / (FN + TP)`` — of the windows that did contain an overrun, the
    fraction not flagged. Equal to ``1 - recall``.

``mean_lead_time``
    Over the true positives, the mean number of iterations between the flag
    and the first actual overrun inside the horizon. A flag raised at ``i``
    for an overrun at ``i + k`` has lead time ``k``, so the metric lies in
    ``[1, H]`` and is only meaningful for ``H > 1``.

``brier``
    Mean squared error of the probability output against the binary label
    (Brier, "Verification of Forecasts Expressed in Terms of Probability",
    *Monthly Weather Review* 78(1):1-3, 1950). This is what makes the
    uncertainty output falsifiable rather than decorative.

``roc_auc``
    Area under the ROC curve of the ranking score. Reported for the ranking
    quality independent of the chosen operating point.

Confidence intervals
--------------------
Rates are binomial proportions, so :func:`wilson_interval` gives the Wilson
score interval (Wilson, "Probable Inference, the Law of Succession, and
Statistical Inference", *Journal of the American Statistical Association*
22(158):209-212, 1927), which behaves near 0 and 1 where the normal
approximation does not. A difference between two predictors smaller than the
overlap of their intervals is not a result, and the comparison scripts say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from ..errors import ConfigurationError

__all__ = [
    "OverrunMetrics",
    "cutoff_for_flag_rate",
    "evaluate_predictor",
    "lead_time_stats",
    "wilson_interval",
]


def cutoff_for_flag_rate(scores: np.ndarray, target_rate: float) -> float:
    """Cut-off ``c`` whose achieved rate ``mean(scores > c)`` is nearest the target.

    A continuous score can hit any target rate; a discrete one cannot. The
    Markov baseline's probability takes exactly two values, so no cut-off
    gives it a 15 % flag rate and pretending otherwise would misreport it.
    This function searches the distinct score values and returns the cut-off
    whose **achieved** rate is closest to ``target_rate``, leaving the
    achieved rate visible in :attr:`OverrunMetrics.flag_rate`.

    Parameters
    ----------
    scores:
        Scores to calibrate on, finite.
    target_rate:
        Desired fraction of flagged rows in ``(0, 1)``.

    Returns
    -------
    float
        The chosen cut-off, in the units of ``scores``.
    """
    v = np.asarray(scores, dtype=np.float64).ravel()
    if v.size == 0:
        raise ConfigurationError("scores must be non-empty")
    if not np.all(np.isfinite(v)):
        raise ValueError("scores must all be finite")
    if not (0.0 < target_rate < 1.0):
        raise ConfigurationError(f"target_rate must be in (0, 1), got {target_rate!r}")
    levels = np.unique(v)
    cands = np.concatenate(
        ([np.nextafter(levels[0], -np.inf)], np.nextafter(levels, -np.inf), levels)
    )
    cands = np.unique(cands)
    rates = np.array([float(np.mean(v > c)) for c in cands], dtype=np.float64)
    return float(cands[int(np.argmin(np.abs(rates - target_rate)))])


def wilson_interval(successes: int, trials: int, *, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion [-].

    Parameters
    ----------
    successes:
        Count of successes, ``0 <= successes <= trials``.
    trials:
        Number of trials, >= 0. Returns ``(nan, nan)`` for zero trials.
    z:
        Normal quantile; 1.959964 for a two-sided 95 % interval.
    """
    if trials < 0 or successes < 0 or successes > trials:
        raise ConfigurationError(
            f"need 0 <= successes <= trials, got {successes!r} of {trials!r}"
        )
    if trials == 0:
        return (math.nan, math.nan)
    n = float(trials)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def lead_time_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    overrun: np.ndarray,
    *,
    horizon: int,
) -> dict[str, float]:
    """Lead time over the true positives, in iterations.

    Parameters
    ----------
    y_true:
        Boolean label per row: an overrun occurs within the horizon.
    y_pred:
        Boolean flag per row.
    overrun:
        Boolean per row, aligned with ``y_true``: whether *this* iteration
        itself overran. Row ``i`` of the dataset corresponds to iteration
        ``index[i]``, and ``overrun`` must be the per-row overrun indicator of
        iterations ``index[i] + 1 ... index[i] + horizon``; it is supplied as
        a ``(m, horizon)`` matrix.
    horizon:
        Look-ahead ``H``, >= 1.

    Returns
    -------
    dict
        ``mean_iters``, ``median_iters``, ``min_iters``, ``max_iters``,
        ``n_true_positive``. All ``nan`` when there are no true positives.
    """
    yt = np.asarray(y_true, dtype=bool).ravel()
    yp = np.asarray(y_pred, dtype=bool).ravel()
    ov = np.asarray(overrun, dtype=bool)
    if horizon < 1:
        raise ConfigurationError(f"horizon must be >= 1, got {horizon!r}")
    if ov.ndim != 2 or ov.shape != (yt.size, horizon):
        raise ConfigurationError(
            f"overrun must have shape ({yt.size}, {horizon}), got {ov.shape}"
        )
    if yt.size != yp.size:
        raise ConfigurationError(f"y_true {yt.size} and y_pred {yp.size} differ in length")
    tp = yt & yp
    if not np.any(tp):
        return {
            "mean_iters": math.nan,
            "median_iters": math.nan,
            "min_iters": math.nan,
            "max_iters": math.nan,
            "n_true_positive": 0.0,
        }
    first = np.argmax(ov[tp], axis=1) + 1
    return {
        "mean_iters": float(np.mean(first)),
        "median_iters": float(np.median(first)),
        "min_iters": float(np.min(first)),
        "max_iters": float(np.max(first)),
        "n_true_positive": float(first.size),
    }


@dataclass(frozen=True)
class OverrunMetrics:
    """All metrics for one predictor on one test set.

    Rates are dimensionless; lead times are in iterations. ``*_ci`` are
    two-sided 95 % Wilson intervals.
    """

    name: str
    n: int
    n_positive: int
    tp: int
    fp: int
    tn: int
    fn: int
    false_alarm_rate: float
    false_alarm_ci: tuple[float, float]
    missed_overrun_rate: float
    missed_overrun_ci: tuple[float, float]
    precision: float
    recall: float
    f1: float
    brier: float
    roc_auc: float
    mean_lead_iters: float
    median_lead_iters: float
    flag_rate: float

    def as_row(self) -> str:
        """One fixed-width table row."""
        return (
            f"{self.name:<22} {self.false_alarm_rate:>10.4f} {self.missed_overrun_rate:>10.4f} "
            f"{self.mean_lead_iters:>9.3f} {self.precision:>10.4f} {self.recall:>8.4f} "
            f"{self.f1:>8.4f} {self.brier:>9.5f} {self.roc_auc:>8.4f} {self.flag_rate:>9.4f}"
        )

    @staticmethod
    def header() -> str:
        """Header matching :meth:`as_row`."""
        return (
            f"{'predictor':<22} {'false-alarm':>10} {'missed':>10} {'lead(it)':>9} "
            f"{'precision':>10} {'recall':>8} {'F1':>8} {'Brier':>9} {'AUC':>8} "
            f"{'flag rate':>9}"
        )


def evaluate_predictor(
    predictor,
    x: np.ndarray,
    y: np.ndarray,
    overrun_window: np.ndarray,
    *,
    horizon: int,
    name: str | None = None,
) -> OverrunMetrics:
    """Score a fitted predictor on a test set.

    Parameters
    ----------
    predictor:
        Object with ``predict``, ``predict_proba`` and ``score``.
    x:
        ``(m, n_features)`` test features.
    y:
        ``(m,)`` boolean labels.
    overrun_window:
        ``(m, horizon)`` boolean matrix of the actual overruns inside each
        row's horizon, used for the lead time.
    horizon:
        Look-ahead ``H``.
    name:
        Override the predictor's ``name`` attribute.
    """
    yt = np.asarray(y, dtype=bool).ravel()
    pred = np.asarray(predictor.predict(x), dtype=bool).ravel()
    proba = np.asarray(predictor.predict_proba(x), dtype=np.float64).ravel()
    score = np.asarray(predictor.score(x), dtype=np.float64).ravel()
    if not (pred.size == proba.size == score.size == yt.size):
        raise ConfigurationError("predictor outputs and labels differ in length")
    tp = int(np.sum(yt & pred))
    fp = int(np.sum(~yt & pred))
    tn = int(np.sum(~yt & ~pred))
    fn = int(np.sum(yt & ~pred))
    far = fp / (fp + tn) if (fp + tn) else math.nan
    mor = fn / (fn + tp) if (fn + tp) else math.nan
    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    both_defined = not (math.isnan(precision) or math.isnan(recall))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if both_defined and (precision + recall) > 0.0
        else math.nan
    )
    brier = float(np.mean((proba - yt.astype(np.float64)) ** 2))
    auc = (
        float(roc_auc_score(yt, score))
        if 0 < int(yt.sum()) < yt.size
        else math.nan
    )
    lead = lead_time_stats(yt, pred, overrun_window, horizon=horizon)
    return OverrunMetrics(
        name=name or getattr(predictor, "name", type(predictor).__name__),
        n=int(yt.size),
        n_positive=int(yt.sum()),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        false_alarm_rate=far,
        false_alarm_ci=wilson_interval(fp, fp + tn),
        missed_overrun_rate=mor,
        missed_overrun_ci=wilson_interval(fn, fn + tp),
        precision=precision,
        recall=recall,
        f1=f1,
        brier=brier,
        roc_auc=auc,
        mean_lead_iters=lead["mean_iters"],
        median_lead_iters=lead["median_iters"],
        flag_rate=float(pred.mean()),
    )
