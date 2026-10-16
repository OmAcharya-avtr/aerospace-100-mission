"""Head-to-head harness: three analytic baselines against the learned forecaster.

The comparison rules, fixed before any number was looked at:

1. Every model is scored on **the same** held-out trajectories, with the same
   forecast origins, the same limit and the same quantile grid.
2. Every model must produce a forecast for **every** test unit. A baseline that
   cannot fit -- because the fitted slope is not in the degrading direction, or
   the window already sits at the limit -- falls back to a declared,
   deliberately uninformative distribution (:data:`FALLBACK_QUANTILES`) and the
   fallback is counted and reported. Dropping the units a model fails on would
   let a model buy a better score by refusing the hard cases.
3. Coverage, pinball loss and CRPS are reported **separately for complete and
   right-censored** test units. The pooled number is printed next to them so a
   reader can see how far it sits from both, not as a summary.
4. If a baseline wins, that is the result. It is reported in the README, the
   model card and here, and it is not retuned away.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .baselines import DegradationFitError, exponential_baseline, linear_baseline
from .forecast import CrossingForecast, QuantileForecast
from .learned import DEFAULT_LEVELS, LearnedCrossingForecaster
from .limits import LimitSpec
from .metrics import (
    coverage_by_censoring,
    crps,
    falsification_rate,
    ks_uniform,
    mean_pinball,
    pit_values,
)
from .physics import powerlaw_fade_baseline

__all__ = [
    "BASELINE_FITTERS",
    "FALLBACK_QUANTILES",
    "ModelRun",
    "baseline_forecasts",
    "evaluate_forecasts",
    "run_head_to_head",
]

#: The declared fallback distribution, used when a baseline cannot fit a window.
#: Log-uniform over 1 to 10000 days: deliberately uninformative, so a model that
#: falls back often is penalised rather than excused. These are remaining-life
#: quantiles in days at :data:`degradepredict.learned.DEFAULT_LEVELS`.
FALLBACK_QUANTILES = np.exp(
    np.interp(
        np.asarray(DEFAULT_LEVELS, dtype=float),
        np.array([0.0, 1.0]),
        np.log(np.array([1.0, 10000.0])),
    )
)

BASELINE_FITTERS = {
    "linear": linear_baseline,
    "exponential": exponential_baseline,
    "powerlaw-fade": powerlaw_fade_baseline,
}


def _fallback(model: str) -> QuantileForecast:
    return QuantileForecast(
        levels=np.asarray(DEFAULT_LEVELS, dtype=float),
        values_days=FALLBACK_QUANTILES.copy(),
        model=f"{model}-fallback",
    )


@dataclass
class ModelRun:
    """One model's forecasts over a test set, with its cost and failure count."""

    name: str
    forecasts: list[CrossingForecast]
    n_fallback: int
    seconds: float
    extras: dict[str, float] = field(default_factory=dict)


def baseline_forecasts(
    trajectories, limit: LimitSpec, model: str, *, level: float = 0.90
) -> ModelRun:
    """Fit one analytic baseline to every trajectory's window.

    Parameters
    ----------
    trajectories:
        Test units.
    limit:
        Declared limit.
    model:
        Key into :data:`BASELINE_FITTERS`.
    level:
        Confidence level passed to the baseline; it sets Fieller's ``g`` but not
        the predictive distribution, which carries the full normal.
    """
    if model not in BASELINE_FITTERS:
        raise ValueError(f"unknown baseline {model!r}; have {sorted(BASELINE_FITTERS)}")
    fitter = BASELINE_FITTERS[model]
    out: list[CrossingForecast] = []
    n_fallback = 0
    g_values: list[float] = []
    start = time.perf_counter()
    for tr in trajectories:
        try:
            fit = fitter(tr.times_days, tr.values_percent, limit, level=level)
        except (DegradationFitError, ValueError):
            out.append(_fallback(model))
            n_fallback += 1
            continue
        out.append(fit.forecast)
        g_values.append(fit.fieller_g)
    seconds = time.perf_counter() - start
    extras = {
        "median_fieller_g": float(np.median(g_values)) if g_values else float("nan"),
        "fraction_g_ge_1": (
            float(np.mean(np.asarray(g_values) >= 1.0)) if g_values else float("nan")
        ),
    }
    return ModelRun(name=model, forecasts=out, n_fallback=n_fallback, seconds=seconds,
                    extras=extras)


def evaluate_forecasts(
    forecasts,
    trajectories,
    *,
    level: float = 0.90,
    crps_grid: np.ndarray | None = None,
) -> dict[str, float]:
    """Score one model's forecasts, split by censoring status.

    Returns a flat dict of floats. Keys ending ``_complete`` and ``_censored``
    are the split figures; keys ending ``_pooled`` mix the two populations and
    are present only so a report can show the gap.
    """
    y = np.array([tr.remaining_true_days for tr in trajectories], dtype=float)
    e = np.array([tr.event for tr in trajectories], dtype=bool)
    c = np.array([tr.followup_days for tr in trajectories], dtype=float)
    cov = coverage_by_censoring(forecasts, y, e, level=level)
    pin = mean_pinball(forecasts, y)
    cr = crps(forecasts, y) if crps_grid is None else crps(forecasts, y, crps_grid)
    pit = pit_values(forecasts, y)
    fals = falsification_rate(forecasts, c, e, level=level)

    def split(arr: np.ndarray, name: str) -> dict[str, float]:
        return {
            f"{name}_pooled": float(arr.mean()),
            f"{name}_complete": float(arr[e].mean()) if e.any() else float("nan"),
            f"{name}_censored": float(arr[~e].mean()) if (~e).any() else float("nan"),
        }

    out: dict[str, float] = dict(cov)
    out.update(split(pin, "pinball_days"))
    out.update(split(cr, "crps_days"))
    out["falsified_fraction_censored"] = fals["falsified_fraction"]
    out.update({f"pit_{k}": v for k, v in ks_uniform(pit).items()})
    out["pit_mean"] = float(pit.mean())
    out["pit_mean_complete"] = float(pit[e].mean()) if e.any() else float("nan")
    out["pit_mean_censored"] = float(pit[~e].mean()) if (~e).any() else float("nan")
    return out


def run_head_to_head(
    train,
    test,
    limit: LimitSpec,
    *,
    level: float = 0.90,
    include_unweighted: bool = True,
    forecaster_kwargs: dict | None = None,
) -> dict[str, dict]:
    """Fit every model and score it on the same test set.

    Returns
    -------
    dict
        ``{model_name: {"run": ModelRun, "scores": dict}}`` with the baselines
        first, then ``learned-ipcw`` and optionally ``learned-none``. Dict order
        is the reporting order.
    """
    results: dict[str, dict] = {}
    for name in ("linear", "exponential", "powerlaw-fade"):
        run = baseline_forecasts(test, limit, name, level=level)
        results[name] = {"run": run, "scores": evaluate_forecasts(run.forecasts, test,
                                                                  level=level)}
    kwargs = dict(forecaster_kwargs or {})
    variants = [("learned-ipcw", "ipcw")]
    if include_unweighted:
        variants.append(("learned-complete-case", "none"))
    for label, weighting in variants:
        start = time.perf_counter()
        model = LearnedCrossingForecaster(limit=limit, weighting=weighting, **kwargs).fit(train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        forecasts = model.predict_many(test)
        run = ModelRun(
            name=label,
            forecasts=list(forecasts),
            n_fallback=0,
            seconds=time.perf_counter() - start,
            extras={"fit_seconds": fit_seconds, "n_train": float(model.n_train_),
                    "n_train_used": float(model.n_train_used_)},
        )
        results[label] = {"run": run, "scores": evaluate_forecasts(forecasts, test, level=level)}
    return results
