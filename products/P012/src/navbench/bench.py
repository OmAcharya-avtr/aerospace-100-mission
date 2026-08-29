"""The bench: run several estimators over one truth/measurement set and score them.

Scoring is deliberately two-sided.  RMS error answers "how close", the
chi-squared consistency statistics of :mod:`navbench.consistency` answer "does
the filter know how close it is".  A filter that wins on RMSE and fails NEES
is reported as failing NEES; the bench never collapses the two into a single
figure of merit.

DIVERGENCE.  A run is flagged divergent when the terminal NEES exceeds the
single-sample ``χ²_n`` 99.99 % quantile, i.e. when the error is grossly
outside what the filter's own covariance admits.  The threshold is a
convention, stated here and repeated in the README, not a theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .consistency import ConsistencyResult, consistency_test, nees, nis

__all__ = ["EstimatorScore", "score_run", "compare_scores", "DIVERGENCE_QUANTILE"]

#: Quantile of the single-sample chi-squared distribution above which the
#: terminal NEES is called divergent.
DIVERGENCE_QUANTILE = 0.9999


@dataclass(frozen=True)
class EstimatorScore:
    """Scores for one estimator on one run.

    Attributes
    ----------
    name : str
    rmse : ndarray, shape (n,)
        Per-component RMS error over the scored steps (state units).
    rmse_total : float
        RMS over all components and steps.
    nees_result, nis_result : ConsistencyResult
        Time-averaged consistency tests (``independent=False``: indicative).
    mean_nees, mean_nis : float
    diverged : bool
        Terminal NEES above the divergence threshold.
    n_steps : int
    """

    name: str
    rmse: NDArray[np.float64]
    rmse_total: float
    nees_result: ConsistencyResult
    nis_result: ConsistencyResult | None
    mean_nees: float
    mean_nis: float
    diverged: bool
    n_steps: int

    def summary(self) -> str:
        """Multi-line human-readable summary (no trailing newline)."""
        lines = [
            f"{self.name}: RMSE(total) = {self.rmse_total:.6g}, "
            f"per-component {np.array2string(self.rmse, precision=4)}",
            f"  {self.nees_result.summary()}",
        ]
        if self.nis_result is not None:
            lines.append(f"  {self.nis_result.summary()}")
        lines.append(f"  diverged: {self.diverged}")
        return "\n".join(lines)


def score_run(
    name: str,
    truth: ArrayLike,
    estimates: ArrayLike,
    covariances: ArrayLike,
    innovations: ArrayLike | None = None,
    innovation_covs: ArrayLike | None = None,
    burn_in: int = 0,
    alpha: float = 0.05,
) -> EstimatorScore:
    """Score one estimator run.

    Parameters
    ----------
    name : str
        Label for the estimator.
    truth : array_like, shape (N, n)
    estimates : array_like, shape (N, n)
    covariances : array_like, shape (N, n, n)
    innovations, innovation_covs : array_like, optional
        Shapes (N, m) and (N, m, m).  When omitted, only NEES is reported.
    burn_in : int
        Number of leading steps excluded from every statistic — the filter's
        transient from ``P₀`` is not a consistency failure.  Must be < N.
    alpha : float
        Two-sided significance for the chi-squared bounds.

    Returns
    -------
    EstimatorScore
    """
    x_true = np.atleast_2d(np.asarray(truth, dtype=float))
    x_est = np.atleast_2d(np.asarray(estimates, dtype=float))
    p = np.asarray(covariances, dtype=float)
    if x_true.shape != x_est.shape:
        raise ValueError(f"truth {x_true.shape} and estimates {x_est.shape} must match")
    n_steps, n = x_true.shape
    if p.shape != (n_steps, n, n):
        raise ValueError(f"covariances must have shape ({n_steps}, {n}, {n}), got {p.shape}")
    b = int(burn_in)
    if b < 0 or b >= n_steps:
        raise ValueError(f"burn_in must be in [0, {n_steps}), got {burn_in!r}")

    err = x_true[b:] - x_est[b:]
    rmse = np.sqrt(np.mean(err**2, axis=0))
    rmse_total = float(np.sqrt(np.mean(err**2)))
    nees_vals = nees(err, p[b:])
    nees_res = consistency_test(nees_vals, n, alpha, "NEES", independent=False)

    nis_res: ConsistencyResult | None = None
    mean_nis_val = float("nan")
    if innovations is not None and innovation_covs is not None:
        v = np.atleast_2d(np.asarray(innovations, dtype=float))[b:]
        s = np.asarray(innovation_covs, dtype=float)[b:]
        m = v.shape[1]
        nis_vals = nis(v, s)
        if np.any(np.isfinite(nis_vals)):
            nis_res = consistency_test(nis_vals, m, alpha, "NIS", independent=False)
            mean_nis_val = nis_res.mean

    thresh = float(stats.chi2.ppf(DIVERGENCE_QUANTILE, n))
    diverged = bool(nees_vals[-1] > thresh)
    return EstimatorScore(
        name=str(name),
        rmse=rmse,
        rmse_total=rmse_total,
        nees_result=nees_res,
        nis_result=nis_res,
        mean_nees=float(np.mean(nees_vals)),
        mean_nis=mean_nis_val,
        diverged=diverged,
        n_steps=int(n_steps - b),
    )


def compare_scores(scores: list[EstimatorScore]) -> str:
    """Render a fixed-width comparison table (no trailing newline).

    Columns: estimator, total RMSE, mean NEES with its bounds, mean NIS,
    consistency verdict, divergence flag.
    """
    if not scores:
        raise ValueError("scores must contain at least one entry")
    header = (
        f"{'estimator':<22}{'RMSE':>12}{'mean NEES':>12}{'NEES bounds':>24}"
        f"{'mean NIS':>11}{'verdict':>14}{'diverged':>10}"
    )
    lines = [header, "-" * len(header)]
    for s in scores:
        bounds = f"[{s.nees_result.lower:.3f}, {s.nees_result.upper:.3f}]"
        nis_txt = "n/a" if not np.isfinite(s.mean_nis) else f"{s.mean_nis:.4f}"
        lines.append(
            f"{s.name:<22}{s.rmse_total:>12.5g}{s.mean_nees:>12.4f}{bounds:>24}"
            f"{nis_txt:>11}{s.nees_result.verdict:>14}{str(s.diverged):>10}"
        )
    return "\n".join(lines)
