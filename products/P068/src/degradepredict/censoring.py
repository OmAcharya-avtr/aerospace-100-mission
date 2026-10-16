"""Right-censoring: Kaplan-Meier of the censoring distribution and IPCW weights.

Why this module exists
----------------------
Most units in a monitored fleet have not crossed their limit yet. Their
remaining life is known only to exceed the follow-up duration. Fitting a
forecaster to the units that *did* cross, and ignoring the rest, is a
complete-case analysis and it is biased: the units that crossed inside a fixed
follow-up window are the fast ones, so the fitted model under-predicts remaining
life. That bias is survivorship bias and it is the specific defect this product
exists to avoid.

What is implemented
-------------------
Inverse probability of censoring weighting (IPCW). The censoring distribution
``G(t) = P(C > t)`` is estimated by the Kaplan-Meier estimator applied with the
roles of event and censoring swapped, and each uncensored unit is weighted by
``1 / G(T_i-)``, the reciprocal of the probability that it would have been
followed long enough to be observed. Censored units then enter the fit through
the weights of the uncensored ones rather than being silently dropped.

References
----------
- Kaplan, E. L. and Meier, P. (1958), "Nonparametric estimation from incomplete
  observations", *Journal of the American Statistical Association*
  53(282):457-481. The product-limit estimator.
- Graf, E., Schmoor, C., Sauerbrei, W. and Schumacher, M. (1999), "Assessment
  and comparison of prognostic classification schemes for survival data",
  *Statistics in Medicine* 18(17-18):2529-2545. The inverse-probability-of-
  censoring weighting scheme used here, introduced there for the censored Brier
  score.
- Robins, J. M. and Rotnitzky, A. (1992), "Recovery of information and
  adjustment for dependent censoring using surrogate markers", in *AIDS
  Epidemiology: Methodological Issues*, Birkhauser, pp. 297-331. The general
  IPCW argument.

Assumptions
-----------
1. **Censoring is independent of the degradation state** given the covariates
   used. The shipped generator satisfies this by construction because the
   follow-up duration is drawn independently. A real fleet often does not: units
   are removed *because* they are degrading, and IPCW is then biased in a
   direction this package cannot measure.
2. ``G`` is estimated marginally, not conditionally on the features. A
   covariate-dependent censoring model is not implemented.
3. Weights are capped (see ``weight_cap``). The cap is a bias-variance trade
   with no optimality claim; it is a declared constant.

Not implemented: Buckley and James (1979), "Linear regression with censored
data", *Biometrika* 66(3):429-436, which imputes censored responses iteratively;
and Portnoy, S. (2003), "Censored regression quantiles", *Journal of the
American Statistical Association* 98(464):1001-1012, which is the principled
censored-quantile-regression estimator. Both are better answers than IPCW for
this problem and neither is here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["KaplanMeier", "censoring_summary", "ipcw_weights", "kaplan_meier"]


@dataclass(frozen=True)
class KaplanMeier:
    """Product-limit survival estimate on a grid of event times.

    Attributes
    ----------
    times:
        Strictly increasing event times, same unit as the input durations.
    survival:
        ``S(t)`` immediately **after** each event time, in ``[0, 1]``, decreasing.
    n_at_risk, n_events:
        Risk set size and event count at each time, for inspection.
    """

    times: np.ndarray
    survival: np.ndarray
    n_at_risk: np.ndarray
    n_events: np.ndarray

    def evaluate(self, t: np.ndarray | float, left_limit: bool = False) -> np.ndarray | float:
        """Evaluate ``S`` at ``t``. Scalar in, float out; array in, array out.

        ``left_limit=True`` returns ``S(t-)``, the value just before ``t``, which
        is what the IPCW weight needs so that a unit's own event does not enter
        its own weight.
        """
        q = np.atleast_1d(np.asarray(t, dtype=float))
        if self.times.size == 0:
            out = np.ones_like(q)
        else:
            side = "left" if left_limit else "right"
            idx = np.searchsorted(self.times, q, side=side) - 1
            out = np.asarray(
                np.where(idx < 0, 1.0, self.survival[np.clip(idx, 0, None)]), dtype=float
            )
        if np.isscalar(t) or np.asarray(t).ndim == 0:
            return float(out.reshape(-1)[0])
        return out


def kaplan_meier(durations: np.ndarray, events: np.ndarray) -> KaplanMeier:
    """Kaplan-Meier product-limit estimate of ``P(T > t)``.

    Parameters
    ----------
    durations:
        Observed durations ``min(T, C)``, strictly positive.
    events:
        Boolean, ``True`` where the duration is an event time and ``False``
        where it is a censoring time.

    Ties: events are taken to precede censorings at the same recorded time, the
    usual convention, so a unit censored at exactly an event time is still in
    that event's risk set.

    Raises
    ------
    ValueError
        On shape mismatch, empty input, or a non-positive or non-finite duration.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events).astype(bool)
    if d.ndim != 1 or d.shape != e.shape:
        raise ValueError("durations and events must be one-dimensional and the same length")
    if d.size == 0:
        raise ValueError("need at least one observation")
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("durations must be finite and strictly positive")

    order = np.lexsort((~e, d))  # within a tie, events (True) first
    d, e = d[order], e[order]
    unique_times = np.unique(d[e]) if e.any() else np.empty(0, dtype=float)
    surv, at_risk, n_ev = [], [], []
    s = 1.0
    for ti in unique_times:
        risk = int(np.sum(d >= ti))
        dead = int(np.sum((d == ti) & e))
        if risk > 0:
            s *= 1.0 - dead / risk
        surv.append(s)
        at_risk.append(risk)
        n_ev.append(dead)
    return KaplanMeier(
        times=unique_times,
        survival=np.asarray(surv, dtype=float),
        n_at_risk=np.asarray(at_risk, dtype=int),
        n_events=np.asarray(n_ev, dtype=int),
    )


def ipcw_weights(
    durations: np.ndarray,
    events: np.ndarray,
    *,
    weight_cap: float = 20.0,
    normalise: bool = True,
) -> np.ndarray:
    """Inverse-probability-of-censoring weights, one per observation.

    Censored observations get weight 0; uncensored observation ``i`` gets
    ``1 / G(T_i-)`` where ``G`` is the Kaplan-Meier estimate of the **censoring**
    distribution, obtained by swapping the roles of event and censoring.

    Parameters
    ----------
    durations, events:
        As for :func:`kaplan_meier`.
    weight_cap:
        Upper bound on an individual weight before normalisation. The estimate
        of ``G`` is unstable in the right tail where few units remain at risk,
        and an uncapped weight there can dominate the fit. The cap is a declared
        constant with no optimality claim; its effect is measured in
        ``validation/validate_headtohead.py``.
    normalise:
        Scale the non-zero weights to mean 1, which leaves the weighted least
        squares solution unchanged but keeps learning rates comparable with the
        unweighted fit.

    Returns
    -------
    numpy.ndarray
        Weights, shape ``(n,)``, non-negative, zero on censored rows.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events).astype(bool)
    if weight_cap <= 1.0:
        raise ValueError(f"weight_cap must exceed 1, got {weight_cap!r}")
    g = kaplan_meier(d, ~e)
    g_at = g.evaluate(d, left_limit=True)
    g_at = np.maximum(g_at, 1.0 / weight_cap)
    w = np.where(e, 1.0 / g_at, 0.0)
    w = np.minimum(w, weight_cap)
    if normalise and np.any(w > 0):
        w = w / w[w > 0].mean()
    return w


def censoring_summary(durations: np.ndarray, events: np.ndarray) -> dict[str, float]:
    """Counts and rates describing how censored a sample is."""
    e = np.asarray(events).astype(bool)
    d = np.asarray(durations, dtype=float)
    return {
        "n": float(e.size),
        "n_complete": float(int(e.sum())),
        "n_censored": float(int((~e).sum())),
        "censored_fraction": float((~e).mean()),
        "median_observed_duration_days": float(np.median(d)),
        "median_complete_duration_days": float(np.median(d[e])) if e.any() else float("nan"),
        "median_censoring_time_days": float(np.median(d[~e])) if (~e).any() else float("nan"),
    }
