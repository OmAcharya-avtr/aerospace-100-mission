"""Plain-text reports. No library module prints; these return strings."""

from __future__ import annotations

import numpy as np

from .baselines import BaselineFit

__all__ = ["format_fit", "format_head_to_head"]

_COLS = (
    ("coverage_complete", "cov cmpl"),
    ("coverage_censored", "cov cens"),
    ("coverage_pooled", "cov pool"),
    ("crps_days_complete", "CRPS cmpl"),
    ("crps_days_censored", "CRPS cens"),
    ("pinball_days_complete", "pin cmpl"),
    ("pinball_days_censored", "pin cens"),
)


def format_head_to_head(results: dict[str, dict], level: float = 0.90) -> str:
    """Render the output of :func:`degradepredict.benchmark.run_head_to_head`.

    Coverage columns are fractions against a nominal ``level``; CRPS and pinball
    are in days and lower is better. Complete and censored are always separate
    columns; the pooled column is printed so the gap is visible.
    """
    header = f"{'model':<22}" + "".join(f"{label:>11}" for _, label in _COLS) + f"{'fallback':>10}"
    lines = [
        f"nominal interval coverage: {level:.2f}  (complete and censored reported separately)",
        header,
        "-" * len(header),
    ]
    for name, payload in results.items():
        s = payload["scores"]
        row = f"{name:<22}" + "".join(f"{s[key]:>11.4f}" for key, _ in _COLS)
        row += f"{payload['run'].n_fallback:>10d}"
        lines.append(row)
    best_c = min(results, key=lambda k: results[k]["scores"]["crps_days_complete"])
    best_r = min(results, key=lambda k: results[k]["scores"]["crps_days_censored"])
    lines.append("")
    lines.append(f"lowest CRPS on complete trajectories  : {best_c}")
    lines.append(f"lowest CRPS on censored trajectories  : {best_r}")
    return "\n".join(lines)


def format_fit(fit: BaselineFit, unit: str = "percent of BOL") -> str:
    """One-paragraph summary of a single analytic fit, with units."""
    lo_f, hi_f = fit.fieller_interval_days
    lo_d, hi_d = fit.delta_interval_days
    bounded = "bounded" if np.isfinite(hi_f) else "UNBOUNDED (Fieller g >= 1)"
    scale = f"log({unit})" if fit.metric_link == "log" else unit
    return "\n".join(
        [
            (f"model              : {fit.model}  (metric link {fit.metric_link}, "
             f"time link t**{fit.z:g})"),
            f"samples            : {fit.n_samples}",
            f"slope              : {fit.slope:.6g} {scale} per day**{fit.z:g}",
            f"residual sd        : {fit.sigma:.6g} {scale}",
            f"crossing estimate  : {fit.crossing_days:.2f} d",
            f"Fieller {fit.level:.0%} interval : [{lo_f:.2f}, {hi_f:.2f}] d  {bounded}",
            f"delta  {fit.level:.0%} interval : [{lo_d:.2f}, {hi_d:.2f}] d",
            f"Fieller g          : {fit.fieller_g:.6g}",
        ]
    )
