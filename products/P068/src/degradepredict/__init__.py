"""degradepredict: calibrated limit-crossing forecasts for a degrading subsystem.

The question: a housekeeping health metric is drifting toward a declared limit.
When does it cross, and is the interval around that date calibrated -- including
for the units that have not crossed yet.

Three analytic baselines (linear, exponential, a cited power-law capacity-fade
shape) with inverse-regression prediction intervals, and a gradient-boosted
quantile forecaster that produces a distribution over the crossing time. Right
censoring is handled explicitly and every coverage and score is reported
separately for complete and right-censored trajectories.

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use.
"""

from __future__ import annotations

from .baselines import (
    BaselineFit,
    DegradationFitError,
    exponential_baseline,
    fit_transformed_ols,
    linear_baseline,
)
from .benchmark import ModelRun, baseline_forecasts, evaluate_forecasts, run_head_to_head
from .censoring import KaplanMeier, censoring_summary, ipcw_weights, kaplan_meier
from .features import FEATURE_NAMES, FEATURE_UNITS, feature_matrix, window_features
from .forecast import CrossingForecast, QuantileForecast, TransformedNormalForecast
from .generate import (
    DEFAULT_LIMIT,
    FAMILIES,
    GeneratorConfig,
    Trajectory,
    generate_population,
    mean_curve,
)
from .learned import DEFAULT_LEVELS, LearnedCrossingForecaster
from .limits import (
    LimitSpec,
    crossing_status,
    first_crossing_time,
    has_crossed,
    margin,
)
from .metrics import (
    calibration_curve,
    coverage_by_censoring,
    crps,
    falsification_rate,
    ks_uniform,
    mean_pinball,
    pinball_loss,
    pit_values,
)
from .physics import DEFAULT_FADE_EXPONENT, FADE_EXPONENT_GRID, powerlaw_fade_baseline
from .report import format_fit, format_head_to_head

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_FADE_EXPONENT",
    "DEFAULT_LEVELS",
    "DEFAULT_LIMIT",
    "FADE_EXPONENT_GRID",
    "FAMILIES",
    "FEATURE_NAMES",
    "FEATURE_UNITS",
    "BaselineFit",
    "CrossingForecast",
    "DegradationFitError",
    "GeneratorConfig",
    "KaplanMeier",
    "LearnedCrossingForecaster",
    "LimitSpec",
    "ModelRun",
    "QuantileForecast",
    "Trajectory",
    "TransformedNormalForecast",
    "__version__",
    "baseline_forecasts",
    "calibration_curve",
    "censoring_summary",
    "coverage_by_censoring",
    "crossing_status",
    "crps",
    "evaluate_forecasts",
    "exponential_baseline",
    "falsification_rate",
    "feature_matrix",
    "first_crossing_time",
    "fit_transformed_ols",
    "format_fit",
    "format_head_to_head",
    "generate_population",
    "has_crossed",
    "ipcw_weights",
    "kaplan_meier",
    "ks_uniform",
    "linear_baseline",
    "margin",
    "mean_curve",
    "mean_pinball",
    "pinball_loss",
    "pit_values",
    "powerlaw_fade_baseline",
    "run_head_to_head",
    "window_features",
]
