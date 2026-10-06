"""linkoutage -- outage and availability statistics for an optical link.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.

The package works on a **supplied** amplitude series exactly as readily as on
one it generates: every function in :mod:`linkoutage.fade`,
:mod:`linkoutage.distributions`, :mod:`linkoutage.markov`,
:mod:`linkoutage.features` and :mod:`linkoutage.predictors` takes an array,
not a model. :mod:`linkoutage.channel` is there to produce test series and to
supply the analytic comparands, and nothing else depends on it except the
analytic baseline.

Modules
-------
:mod:`linkoutage.fade`
    Down-crossings, level-crossing rate, fade durations with censoring,
    outage fraction, availability. Every definitional choice is explicit in
    :class:`linkoutage.fade.FadeDefinitions`.
:mod:`linkoutage.channel`
    Correlated lognormal fading model (AR(1) log-amplitude) and the exact
    discrete-time analytic crossing rate and mean fade duration of that model.
:mod:`linkoutage.distributions`
    Fade-duration distribution fitting with right censoring and a
    goodness-of-fit verdict on the memoryless hypothesis.
:mod:`linkoutage.markov`
    Two-state and N-state Markov and semi-Markov channel-state fitting, with
    dwell-time and Markov-order goodness-of-fit tests.
:mod:`linkoutage.features`
    Leakage-controlled features, labels and temporal splits for short-horizon
    outage prediction.
:mod:`linkoutage.predictors`
    Three baselines (constant base rate, analytic level-crossing rate,
    logistic regression) and a random forest with an ensemble-spread
    uncertainty output.
:mod:`linkoutage.calibration`
    Brier score and its Murphy decomposition, expected calibration error,
    reliability curves with Wilson intervals, average precision.
"""

from __future__ import annotations

from .calibration import (
    CalibrationReport,
    brier_decomposition,
    brier_score,
    evaluate_forecast,
    expected_calibration_error,
    reliability_curve,
)
from .channel import (
    LognormalSeries,
    amplitude_threshold_to_gaussian_level,
    analytic_level_crossing_rate,
    analytic_mean_fade_duration,
    analytic_outage_fraction,
    ar1_unit_variance,
    conditional_onset_probability,
    lognormal_amplitude_series,
    rho_from_tau,
    sigma_ln_i_from_si,
)
from .distributions import (
    DistributionFit,
    FitComparison,
    compare_fade_duration_models,
    exponential_mle_with_censoring,
    kaplan_meier_survival,
)
from .fade import (
    DEFAULT_DEFINITIONS,
    FadeDefinitions,
    FadeRuns,
    FadeStatistics,
    availability,
    down_crossing_indices,
    fade_durations,
    fade_runs,
    fade_statistics,
    level_crossing_rate,
    mean_fade_duration,
    outage_fraction,
    up_crossing_indices,
)
from .features import (
    FEATURE_NAMES,
    OutageDataset,
    build_outage_dataset,
    onset_labels,
    temporal_split,
)
from .markov import (
    MarkovFit,
    MarkovOrderTest,
    SemiMarkovFit,
    dwell_time_goodness_of_fit,
    fit_markov,
    fit_semi_markov,
    markov_order_test,
    state_sequence,
)
from .predictors import (
    AnalyticLcrPredictor,
    ConstantRatePredictor,
    LogisticBaseline,
    PlattCalibrated,
    RandomForestOutageClassifier,
    estimate_channel_parameters,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DEFINITIONS",
    "FEATURE_NAMES",
    "AnalyticLcrPredictor",
    "CalibrationReport",
    "ConstantRatePredictor",
    "DistributionFit",
    "FadeDefinitions",
    "FadeRuns",
    "FadeStatistics",
    "FitComparison",
    "LogisticBaseline",
    "LognormalSeries",
    "MarkovFit",
    "MarkovOrderTest",
    "OutageDataset",
    "PlattCalibrated",
    "RandomForestOutageClassifier",
    "SemiMarkovFit",
    "__version__",
    "amplitude_threshold_to_gaussian_level",
    "analytic_level_crossing_rate",
    "analytic_mean_fade_duration",
    "analytic_outage_fraction",
    "ar1_unit_variance",
    "availability",
    "brier_decomposition",
    "brier_score",
    "build_outage_dataset",
    "compare_fade_duration_models",
    "conditional_onset_probability",
    "down_crossing_indices",
    "dwell_time_goodness_of_fit",
    "estimate_channel_parameters",
    "evaluate_forecast",
    "expected_calibration_error",
    "exponential_mle_with_censoring",
    "fade_durations",
    "fade_runs",
    "fade_statistics",
    "fit_markov",
    "fit_semi_markov",
    "kaplan_meier_survival",
    "level_crossing_rate",
    "lognormal_amplitude_series",
    "markov_order_test",
    "mean_fade_duration",
    "onset_labels",
    "outage_fraction",
    "reliability_curve",
    "rho_from_tau",
    "sigma_ln_i_from_si",
    "state_sequence",
    "temporal_split",
    "up_crossing_indices",
]
