"""Calibration and rare-event metrics for probabilistic outage forecasts.

Why accuracy is not reported as a headline
------------------------------------------
At a 2.6 % base rate a forecaster that outputs "no outage" every time is
97.4 % accurate and completely useless. Accuracy, and any threshold-dependent
metric derived from it, is therefore reported here only next to the
all-negative reference so the reader can see how little it means. The numbers
that carry information at a low base rate are:

**Brier score** -- the mean squared error of the probability itself. Proper:
it is minimised only by the true conditional probability, so it cannot be
gamed by shading forecasts towards the base rate.

**Brier decomposition** (A. H. Murphy, "A new vector partition of the
probability score", *Journal of Applied Meteorology* 12(4):595-600, 1973) --
``BS = reliability - resolution + uncertainty``. ``reliability`` is the
calibration error, lower is better; ``resolution`` is how far the forecast
moves away from the base rate in the right direction, higher is better;
``uncertainty`` is ``base_rate * (1 - base_rate)``, a property of the problem
and not of the forecaster. A constant-rate forecaster has
``reliability = 0``, ``resolution = 0`` and ``BS = uncertainty`` exactly: it is
perfectly calibrated and has no skill. Being beaten on reliability is not the
same as being beaten; the resolution column is where skill shows up.

**Expected calibration error** -- the bin-count-weighted mean absolute gap
between forecast and observed frequency. Equal-count (quantile) bins by
default, because equal-width bins put almost every row of a rare-event problem
into the first bin.

**Average precision** -- the area under the precision-recall curve, which
unlike ROC AUC does not flatter a forecaster on an imbalanced problem.

Caveats that travel with the numbers
------------------------------------
The binned decomposition is exact only when each bin contains a single
distinct forecast value. With real bins there is a residual, and
:func:`brier_decomposition` returns it rather than quietly dropping it; a
large residual means the bins are too coarse. Separately, every bin frequency
is itself an estimate: a bin with 20 rows and 1 positive says almost nothing,
so :func:`reliability_curve` returns the counts and a Wilson interval and the
plotting code draws them.

Units
-----
Dimensionless probabilities and scores throughout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

__all__ = [
    "CalibrationReport",
    "ReliabilityCurve",
    "brier_decomposition",
    "brier_score",
    "evaluate_forecast",
    "expected_calibration_error",
    "reliability_curve",
    "wilson_interval",
]


def _check_pair(y: ArrayLike, p: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    yy = np.asarray(y, dtype=np.float64).ravel()
    pp = np.asarray(p, dtype=np.float64).ravel()
    if yy.size != pp.size:
        raise ValueError(f"y has {yy.size} elements but p has {pp.size}")
    if yy.size == 0:
        raise ValueError("y and p are empty")
    if not np.all(np.isin(yy, (0.0, 1.0))):
        raise ValueError("y must contain only 0 and 1")
    if not np.all(np.isfinite(pp)):
        raise ValueError("p contains non-finite values")
    if np.any(pp < 0.0) or np.any(pp > 1.0):
        raise ValueError(
            f"p must lie in [0, 1]; got min {pp.min()!r}, max {pp.max()!r}. "
            "A score that is not a probability cannot be scored for calibration."
        )
    return yy, pp


def brier_score(y: ArrayLike, p: ArrayLike) -> float:
    """Mean squared error of the forecast probability."""
    yy, pp = _check_pair(y, p)
    return float(np.mean((pp - yy) ** 2))


def wilson_interval(k: int, n: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because bins in a rare-event
    reliability curve routinely contain zero or one positive, where the normal
    interval is degenerate. Default ``z`` is the two-sided 95 % normal
    quantile.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if k < 0 or k > n:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}")
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class ReliabilityCurve:
    """Binned reliability diagram with per-bin uncertainty.

    Attributes
    ----------
    mean_forecast:
        Mean forecast probability in each bin.
    observed_frequency:
        Observed positive fraction in each bin.
    count:
        Rows in each bin.
    n_positive:
        Positives in each bin.
    lower, upper:
        Wilson 95 % interval on ``observed_frequency``.
    strategy, n_bins_requested:
        How the bins were made.
    """

    mean_forecast: NDArray[np.float64]
    observed_frequency: NDArray[np.float64]
    count: NDArray[np.int64]
    n_positive: NDArray[np.int64]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    strategy: str
    n_bins_requested: int

    def table(self) -> str:
        lines = [
            f"{'bin':>4s} {'n':>8s} {'pos':>6s} {'mean forecast':>14s} "
            f"{'observed':>10s} {'95% low':>10s} {'95% high':>10s}"
        ]
        for i in range(self.count.size):
            lines.append(
                f"{i:>4d} {self.count[i]:>8d} {self.n_positive[i]:>6d} "
                f"{self.mean_forecast[i]:>14.6f} {self.observed_frequency[i]:>10.6f} "
                f"{self.lower[i]:>10.6f} {self.upper[i]:>10.6f}"
            )
        return "\n".join(lines)


def _bin_edges(p: NDArray[np.float64], n_bins: int, strategy: str) -> NDArray[np.float64]:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy != "quantile":
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(p, q)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.asarray(np.unique(edges), dtype=np.float64)


def _assign(p: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.int64]:
    idx = np.digitize(p, edges[1:-1], right=True)
    return np.asarray(idx, dtype=np.int64)


def reliability_curve(
    y: ArrayLike, p: ArrayLike, *, n_bins: int = 10, strategy: str = "quantile"
) -> ReliabilityCurve:
    """Binned reliability diagram.

    Parameters
    ----------
    n_bins:
        Requested bins. With ``strategy='quantile'`` a forecaster that emits
        few distinct values (the constant-rate baseline emits one) collapses to
        fewer bins, which is reported in ``count.size`` rather than faked.
    strategy:
        ``"quantile"`` -- equal-count bins (default, correct for rare events).
        ``"uniform"`` -- equal-width bins on ``[0, 1]``.
    """
    yy, pp = _check_pair(y, p)
    nb = int(n_bins)
    if nb < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins!r}")
    edges = _bin_edges(pp, nb, strategy)
    if edges.size < 2:
        edges = np.array([-np.inf, np.inf])
    idx = _assign(pp, edges)
    n_actual = int(edges.size - 1)
    count = np.bincount(idx, minlength=n_actual).astype(np.int64)
    pos = np.bincount(idx, weights=yy, minlength=n_actual).astype(np.float64)
    sum_p = np.bincount(idx, weights=pp, minlength=n_actual).astype(np.float64)
    keep = count > 0
    count = count[keep]
    pos = pos[keep]
    sum_p = sum_p[keep]
    obs = pos / count
    mean_f = sum_p / count
    lo = np.empty(count.size, dtype=np.float64)
    hi = np.empty(count.size, dtype=np.float64)
    for i in range(count.size):
        lo[i], hi[i] = wilson_interval(int(round(pos[i])), int(count[i]))
    return ReliabilityCurve(
        mean_forecast=mean_f,
        observed_frequency=obs,
        count=count,
        n_positive=np.asarray(np.rint(pos), dtype=np.int64),
        lower=lo,
        upper=hi,
        strategy=strategy,
        n_bins_requested=nb,
    )


def expected_calibration_error(
    y: ArrayLike, p: ArrayLike, *, n_bins: int = 10, strategy: str = "quantile"
) -> tuple[float, float]:
    """``(ECE, MCE)``: count-weighted mean and maximum absolute calibration gap."""
    curve = reliability_curve(y, p, n_bins=n_bins, strategy=strategy)
    gap = np.abs(curve.mean_forecast - curve.observed_frequency)
    total = float(curve.count.sum())
    ece = float(np.sum(curve.count * gap) / total)
    mce = float(gap.max())
    return ece, mce


def brier_decomposition(
    y: ArrayLike, p: ArrayLike, *, n_bins: int = 10, strategy: str = "quantile"
) -> tuple[float, float, float, float, float]:
    """Murphy (1973) partition: ``(BS, reliability, resolution, uncertainty, residual)``.

    ``residual = BS - (reliability - resolution + uncertainty)`` is the binning
    error. It is exactly zero when every bin holds a single distinct forecast
    value, so a residual that is not small relative to ``BS`` means the bins are
    too coarse to support the decomposition.
    """
    yy, pp = _check_pair(y, p)
    curve = reliability_curve(yy, pp, n_bins=n_bins, strategy=strategy)
    n = float(yy.size)
    base = float(yy.mean())
    rel = float(np.sum(curve.count * (curve.mean_forecast - curve.observed_frequency) ** 2) / n)
    res = float(np.sum(curve.count * (curve.observed_frequency - base) ** 2) / n)
    unc = base * (1.0 - base)
    bs = brier_score(yy, pp)
    residual = bs - (rel - res + unc)
    return bs, rel, res, unc, residual


@dataclass(frozen=True)
class CalibrationReport:
    """Every metric for one forecaster on one held-out split."""

    name: str
    n: int
    n_positive: int
    base_rate: float
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    decomposition_residual: float
    ece: float
    mce: float
    log_loss: float
    roc_auc: float
    average_precision: float
    accuracy_at_half: float
    accuracy_all_negative: float
    brier_skill_score: float
    mean_forecast: float
    curve: ReliabilityCurve
    notes: str = ""

    def line(self) -> str:
        return (
            f"{self.name:<28s} {self.brier:>11.7f} {self.brier_skill_score:>9.4f} "
            f"{self.reliability:>11.3e} {self.resolution:>11.3e} {self.ece:>9.5f} "
            f"{self.mce:>9.5f} {self.log_loss:>9.5f} {self.roc_auc:>7.4f} "
            f"{self.average_precision:>8.4f} {self.mean_forecast:>9.5f}"
        )

    @staticmethod
    def header() -> str:
        return (
            f"{'forecaster':<28s} {'Brier':>11s} {'BSS':>9s} {'REL':>11s} {'RES':>11s} "
            f"{'ECE':>9s} {'MCE':>9s} {'logloss':>9s} {'AUC':>7s} {'AP':>8s} "
            f"{'mean p':>9s}"
        )


def evaluate_forecast(
    y: ArrayLike,
    p: ArrayLike,
    *,
    name: str,
    n_bins: int = 10,
    strategy: str = "quantile",
    reference_rate: float | None = None,
    notes: str = "",
) -> CalibrationReport:
    """Score one forecaster.

    Parameters
    ----------
    reference_rate:
        Base rate of the constant-rate reference forecaster, normally the
        **training** base rate. The Brier skill score is measured against a
        constant forecast at this rate, so it answers "did this beat the
        constant-rate baseline?" directly. Defaults to the base rate of ``y``
        itself, which is slightly generous to the baseline because it uses
        held-out information the baseline would not have had.
    """
    yy, pp = _check_pair(y, p)
    bs, rel, res, unc, residual = brier_decomposition(
        yy, pp, n_bins=n_bins, strategy=strategy
    )
    ece, mce = expected_calibration_error(yy, pp, n_bins=n_bins, strategy=strategy)
    base = float(yy.mean())
    ref = base if reference_rate is None else float(reference_rate)
    bs_ref = float(np.mean((ref - yy) ** 2))
    bss = 1.0 - bs / bs_ref if bs_ref > 0.0 else float("nan")
    eps = 1e-15
    clipped = np.clip(pp, eps, 1.0 - eps)
    ll = float(log_loss(yy, clipped, labels=[0.0, 1.0]))
    if 0 < int(yy.sum()) < yy.size:
        auc = float(roc_auc_score(yy, pp))
        ap = float(average_precision_score(yy, pp))
    else:
        auc = float("nan")
        ap = float("nan")
    return CalibrationReport(
        name=name,
        n=int(yy.size),
        n_positive=int(yy.sum()),
        base_rate=base,
        brier=bs,
        reliability=rel,
        resolution=res,
        uncertainty=unc,
        decomposition_residual=residual,
        ece=ece,
        mce=mce,
        log_loss=ll,
        roc_auc=auc,
        average_precision=ap,
        accuracy_at_half=float(np.mean((pp >= 0.5).astype(np.float64) == yy)),
        accuracy_all_negative=1.0 - base,
        brier_skill_score=bss,
        mean_forecast=float(pp.mean()),
        curve=reliability_curve(yy, pp, n_bins=n_bins, strategy=strategy),
        notes=notes,
    )
