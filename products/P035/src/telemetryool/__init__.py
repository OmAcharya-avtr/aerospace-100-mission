"""telemetryool -- operational out-of-limit semantics and a matched-false-alarm-rate harness.

The package is organised baseline-first, which is also the order in which it was
built and the order in which its results should be read.

1. :mod:`telemetryool.limits` -- out-of-limit checking with the operational
   semantics: soft and hard limits, per-channel validity masks,
   persistence/debounce counts, mode-dependent limit tables.
2. :mod:`telemetryool.arl` -- designed thresholds.  Average run length and
   window false-alarm probability for CUSUM (Brook & Evans 1972 Markov chain and
   the Siegmund 1985 closed form), EWMA (Lucas & Saccucci 1990) and the
   persistence-counter limit check (exact).
3. :mod:`telemetryool.charts` -- the EWMA and CUSUM charts themselves.
4. :mod:`telemetryool.changepoint` -- offline mean-shift change-point detection
   with a Monte-Carlo-calibrated threshold.
5. :mod:`telemetryool.novelty` -- multivariate novelty models: the classical
   Hotelling ``T^2``/``Q`` baseline, a Gaussian-mixture density model and an
   Isolation Forest, each with a calibrated p-value and its finite-sample
   interval.
6. :mod:`telemetryool.calibration`, :mod:`telemetryool.metrics`,
   :mod:`telemetryool.harness` -- threshold calibration to a common window
   false-alarm probability, full confusion matrices, window-level ROC,
   detection-delay statistics, and the comparison harness that ties them
   together.

This software is research-grade.  It is not flight-qualified, not certified, and
not approved for operational aerospace use.
"""

from __future__ import annotations

from .arl import (
    SIEGMUND_B_OFFSET,
    ChartDesign,
    cusum_arl_markov,
    cusum_arl_siegmund,
    cusum_arl_siegmund_two_sided,
    cusum_window_false_alarm,
    design_cusum_h,
    design_ewma_L,
    design_ool_limit,
    ewma_arl_markov,
    ewma_sigma_z,
    ewma_window_false_alarm,
    ool_arl,
    ool_window_false_alarm,
)
from .calibration import (
    Calibration,
    RateEstimate,
    binomial_se,
    calibrate_threshold,
    clopper_pearson_interval,
    estimate_rate,
    wilson_interval,
    window_trigger_level,
    windows_for_precision,
)
from .changepoint import (
    ChangePointThreshold,
    SegmentStatistic,
    calibrate_change_point_threshold,
    detect_change_points,
    max_mean_shift_statistic,
)
from .charts import ChartRun, CusumChart, EwmaChart
from .detectors import (
    CusumDetector,
    EwmaDetector,
    LimitDetector,
    NoveltyDetector,
    WindowDetector,
    build_detector_suite,
    smoothed_features,
)
from .harness import (
    ComparisonResult,
    MethodResult,
    MethodScenarioResult,
    ScenarioSpec,
    run_comparison,
)
from .limits import (
    DEFAULT_MODE,
    AlarmState,
    BreachLevel,
    ChannelSpec,
    InvalidPolicy,
    LimitSet,
    OolChecker,
    OolSample,
    TelemetryMonitor,
    scale_limits,
)
from .metrics import (
    ConfusionMatrix,
    DelayStats,
    RocCurve,
    confusion_matrix,
    delay_stats,
    window_roc,
)
from .novelty import (
    GmmNovelty,
    HotellingT2Q,
    IsolationForestNovelty,
    NoveltyModel,
    PValue,
)
from .runs import any_run, first_run_index, run_counter_trace
from .synthetic import (
    ANOMALY_KINDS,
    Anomaly,
    NominalModel,
    apply_anomaly,
    equicorrelation,
    generate_anomalous,
    generate_nominal,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # arl
    "SIEGMUND_B_OFFSET",
    "ChartDesign",
    "cusum_arl_markov",
    "cusum_arl_siegmund",
    "cusum_arl_siegmund_two_sided",
    "cusum_window_false_alarm",
    "design_cusum_h",
    "design_ewma_L",
    "design_ool_limit",
    "ewma_arl_markov",
    "ewma_sigma_z",
    "ewma_window_false_alarm",
    "ool_arl",
    "ool_window_false_alarm",
    # calibration
    "Calibration",
    "RateEstimate",
    "binomial_se",
    "calibrate_threshold",
    "clopper_pearson_interval",
    "estimate_rate",
    "wilson_interval",
    "window_trigger_level",
    "windows_for_precision",
    # changepoint
    "ChangePointThreshold",
    "SegmentStatistic",
    "calibrate_change_point_threshold",
    "detect_change_points",
    "max_mean_shift_statistic",
    # charts
    "ChartRun",
    "CusumChart",
    "EwmaChart",
    # detectors
    "CusumDetector",
    "EwmaDetector",
    "LimitDetector",
    "NoveltyDetector",
    "WindowDetector",
    "build_detector_suite",
    "smoothed_features",
    # harness
    "ComparisonResult",
    "MethodResult",
    "MethodScenarioResult",
    "ScenarioSpec",
    "run_comparison",
    # limits
    "DEFAULT_MODE",
    "AlarmState",
    "BreachLevel",
    "ChannelSpec",
    "InvalidPolicy",
    "LimitSet",
    "OolChecker",
    "OolSample",
    "TelemetryMonitor",
    "scale_limits",
    # metrics
    "ConfusionMatrix",
    "DelayStats",
    "RocCurve",
    "confusion_matrix",
    "delay_stats",
    "window_roc",
    # novelty
    "GmmNovelty",
    "HotellingT2Q",
    "IsolationForestNovelty",
    "NoveltyModel",
    "PValue",
    # runs
    "any_run",
    "first_run_index",
    "run_counter_trace",
    # synthetic
    "ANOMALY_KINDS",
    "Anomaly",
    "NominalModel",
    "apply_anomaly",
    "equicorrelation",
    "generate_anomalous",
    "generate_nominal",
]
