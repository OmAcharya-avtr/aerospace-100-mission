"""Scoring rules, interval coverage and calibration for crossing-time forecasts.

Everything here takes a list of :class:`degradepredict.forecast.CrossingForecast`
and an array of realised remaining lives in days, and every split-by-censoring
function takes the event indicator as well. The split is not optional in the
reporting layer: a pooled coverage number over a population that is more than
half right-censored is the specific thing this product exists to avoid
reporting.

References
----------
- Koenker, R. and Bassett, G. (1978), "Regression quantiles", *Econometrica*
  46(1):33-50. The pinball (check) loss.
- Gneiting, T. and Raftery, A. E. (2007), "Strictly proper scoring rules,
  prediction, and estimation", *Journal of the American Statistical
  Association* 102(477):359-378. CRPS and its propriety.
- Laio, F. and Tamea, S. (2007), "Verification tools for probabilistic
  forecasts of continuous hydrological variables", *Hydrology and Earth System
  Sciences* 11(4):1267-1277. States the quantile form of CRPS used here:
  ``CRPS(F, y) = 2 * integral over alpha in (0,1) of the pinball loss at level
  alpha``.
- Gneiting, T., Balabdaoui, F. and Raftery, A. E. (2007), "Probabilistic
  forecasts, calibration and sharpness", *Journal of the Royal Statistical
  Society Series B* 69(2):243-268. Probabilistic calibration as uniformity of
  the probability integral transform.
- Graf, E., Schmoor, C., Sauerbrei, W. and Schumacher, M. (1999), *Statistics in
  Medicine* 18(17-18):2529-2545, for the IPCW weighting of a score under
  right censoring.

Units: all durations in days. Pinball loss and CRPS are therefore in days.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "CRPS_GRID",
    "calibration_curve",
    "coverage_by_censoring",
    "crps",
    "falsification_rate",
    "interval_scores",
    "ks_uniform",
    "mean_pinball",
    "pinball_loss",
    "pit_values",
]

#: Probability grid the CRPS integral is evaluated on: the 199-point midpoint
#: rule on (0, 1). On a uniform predictive distribution, whose CRPS has a closed
#: form, this grid reproduces it to a relative error of about 2e-6
#: (``tests/test_metrics.py::test_crps_of_a_uniform_predictive_matches_the_closed_form``),
#: and the grid dependence is measured again against a 999-point grid in
#: ``validation/validate_headtohead_output.txt``.
_CRPS_N = 199
CRPS_GRID = np.linspace(0.5 / _CRPS_N, 1.0 - 0.5 / _CRPS_N, _CRPS_N)


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, level: float) -> np.ndarray:
    """Pinball (check) loss at one quantile level, elementwise, in days.

    ``rho_a(u) = a * u`` for ``u >= 0`` and ``(a - 1) * u`` for ``u < 0``, with
    ``u = y_true - q_pred``. Minimised in expectation by the true ``level``
    quantile (Koenker and Bassett 1978).
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level!r}")
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(q_pred, dtype=float)
    if y.shape != q.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape}, q_pred {q.shape}")
    u = y - q
    return np.where(u >= 0.0, level * u, (level - 1.0) * u)


def _quantile_matrix(forecasts, levels: np.ndarray) -> np.ndarray:
    """Evaluate every forecast's quantile function on ``levels``: shape (n, m)."""
    rows = [np.asarray(f.quantile(levels), dtype=float).reshape(-1) for f in forecasts]
    return np.vstack(rows)


def mean_pinball(
    forecasts, y_true: np.ndarray, levels: np.ndarray | None = None
) -> np.ndarray:
    """Per-unit pinball loss averaged over ``levels``, days. Shape ``(n,)``.

    ``levels`` defaults to the nine-point grid
    ``0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95``-style spread used in the reports;
    pass an explicit grid to score on a different one. Lower is better.
    """
    lv = np.asarray(
        levels if levels is not None else (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
        dtype=float,
    )
    y = np.asarray(y_true, dtype=float)
    qm = _quantile_matrix(forecasts, lv)
    if qm.shape[0] != y.size:
        raise ValueError(f"{qm.shape[0]} forecasts for {y.size} realisations")
    losses = np.stack([pinball_loss(y, qm[:, j], float(lv[j])) for j in range(lv.size)])
    return losses.mean(axis=0)


def crps(forecasts, y_true: np.ndarray, grid: np.ndarray = CRPS_GRID) -> np.ndarray:
    """Continuous ranked probability score per unit, days. Shape ``(n,)``.

    Computed as ``2 * mean(pinball over the probability grid)``, the quantile
    representation of CRPS (Laio and Tamea 2007, equation 7). The default grid
    is the 199-point midpoint rule on (0, 1), so the returned value is a
    quadrature approximation rather than the closed-form CRPS of any particular
    predictive law. On a uniform predictive, where the closed form is known, the
    approximation is correct to about 2e-6 relative. The same grid is applied to
    every model being compared, and the grid dependence is remeasured against a
    999-point grid in the head-to-head validation output.
    """
    y = np.asarray(y_true, dtype=float)
    g = np.asarray(grid, dtype=float)
    qm = _quantile_matrix(forecasts, g)
    if qm.shape[0] != y.size:
        raise ValueError(f"{qm.shape[0]} forecasts for {y.size} realisations")
    losses = np.stack([pinball_loss(y, qm[:, j], float(g[j])) for j in range(g.size)])
    return 2.0 * losses.mean(axis=0)


def interval_scores(forecasts, level: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper endpoints of each forecast's central interval, days."""
    bounds = np.array([f.interval(level) for f in forecasts], dtype=float)
    return bounds[:, 0], bounds[:, 1]


def coverage_by_censoring(
    forecasts,
    remaining_true_days: np.ndarray,
    event: np.ndarray,
    level: float = 0.90,
) -> dict[str, float]:
    """Interval coverage reported **separately** for complete and censored units.

    Parameters
    ----------
    forecasts:
        One predictive distribution per unit.
    remaining_true_days:
        The true remaining life of every unit, including the censored ones.
        This is only available because the trajectories are generated: in the
        field the censored units' truth is, by definition, not yet known. That
        is the whole point -- the censored coverage number is the one a real
        deployment cannot measure without waiting, and it is the one that moves.
    event:
        ``True`` for units whose crossing was observed inside the follow-up.
    level:
        Nominal coverage of the central interval.

    Returns
    -------
    dict
        ``coverage_pooled``, ``coverage_complete``, ``coverage_censored``, the
        three counts, the mean interval widths, and ``nominal``. The pooled
        figure is returned so that reports can show how far it sits from both
        halves; it is not a summary of them.
    """
    y = np.asarray(remaining_true_days, dtype=float)
    e = np.asarray(event).astype(bool)
    if y.shape != e.shape:
        raise ValueError("remaining_true_days and event must have the same shape")
    lo, hi = interval_scores(forecasts, level)
    if lo.size != y.size:
        raise ValueError(f"{lo.size} forecasts for {y.size} realisations")
    inside = (y >= lo) & (y <= hi)
    width = hi - lo

    def _mean(mask: np.ndarray, arr: np.ndarray) -> float:
        return float(arr[mask].mean()) if mask.any() else float("nan")

    return {
        "nominal": float(level),
        "n": float(y.size),
        "n_complete": float(int(e.sum())),
        "n_censored": float(int((~e).sum())),
        "coverage_pooled": float(inside.mean()),
        "coverage_complete": _mean(e, inside),
        "coverage_censored": _mean(~e, inside),
        "mean_width_pooled_days": float(width.mean()),
        "mean_width_complete_days": _mean(e, width),
        "mean_width_censored_days": _mean(~e, width),
        "median_true_complete_days": float(np.median(y[e])) if e.any() else float("nan"),
        "median_true_censored_days": float(np.median(y[~e])) if (~e).any() else float("nan"),
    }


def falsification_rate(
    forecasts,
    censoring_time_days: np.ndarray,
    event: np.ndarray,
    level: float = 0.90,
) -> dict[str, float]:
    """The part of censored non-coverage a real deployment *can* measure.

    For a right-censored unit the only thing known at analysis time is that the
    remaining life exceeds the follow-up duration ``C``. An interval whose upper
    endpoint is below ``C`` is therefore **falsified by the censoring alone**,
    with no need to wait for the crossing. The falsification rate is a strict
    lower bound on the true non-coverage of the censored units and it is
    computable in the field, which is why it is reported next to the oracle
    coverage that is not.
    """
    c = np.asarray(censoring_time_days, dtype=float)
    e = np.asarray(event).astype(bool)
    _, hi = interval_scores(forecasts, level)
    cens = ~e
    if not cens.any():
        return {"n_censored": 0.0, "falsified_fraction": float("nan"),
                "oracle_noncoverage_bound": float("nan")}
    falsified = hi[cens] < c[cens]
    return {
        "n_censored": float(int(cens.sum())),
        "falsified_fraction": float(falsified.mean()),
        "oracle_noncoverage_bound": float(falsified.mean()),
    }


def pit_values(forecasts, remaining_true_days: np.ndarray) -> np.ndarray:
    """Probability integral transform ``F_i(y_i)`` per unit, in ``[0, 1]``.

    A probabilistically calibrated forecaster gives PIT values uniform on
    ``[0, 1]`` (Gneiting, Balabdaoui and Raftery 2007). For the learned model
    the CDF is the interpolated quantile function, so PIT values saturate at the
    outermost fitted levels: a realisation beyond the 0.95 quantile returns
    exactly 0.95. The saturation is visible as spikes at the ends of the
    calibration curve and is a property of the representation, not of the data.
    """
    y = np.asarray(remaining_true_days, dtype=float)
    out = np.array([float(f.cdf(float(yi))) for f, yi in zip(forecasts, y, strict=True)])
    return np.clip(out, 0.0, 1.0)


def calibration_curve(pit: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF of the PIT at ``n_bins`` equally spaced nominal levels.

    Returns ``(nominal, empirical)``: ``nominal[j]`` is a probability level and
    ``empirical[j]`` the fraction of PIT values at or below it. A calibrated
    forecaster lies on the diagonal.
    """
    p = np.asarray(pit, dtype=float)
    if p.size == 0:
        raise ValueError("no PIT values supplied")
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    nominal = np.linspace(1.0 / n_bins, 1.0, n_bins)
    empirical = np.array([float(np.mean(p <= q)) for q in nominal])
    return nominal, empirical


def ks_uniform(pit: np.ndarray) -> dict[str, float]:
    """One-sample Kolmogorov-Smirnov test of the PIT against ``Uniform(0, 1)``.

    The p-value is only interpretable when the forecasts are independent across
    units, which holds here because each trajectory is generated independently.
    It is **not** a calibration guarantee: failing to reject uniformity on a few
    hundred units is weak evidence.
    """
    p = np.asarray(pit, dtype=float)
    if p.size < 2:
        raise ValueError("need at least two PIT values")
    res = stats.kstest(p, "uniform")
    return {"ks_statistic": float(res.statistic), "ks_pvalue": float(res.pvalue),
            "n": float(p.size)}
