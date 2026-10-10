"""telemdrift: streaming change detection on a univariate telemetry channel,
scored on detection delay against false-alarm rate.

This is a **benchmark harness**, not a streaming framework. It exists to answer
one question with measured numbers and error bars: given a telemetry channel and
a change you care about, how many samples after the change do you notice, at a
false-alarm rate you chose rather than inherited from a library default.

For production streaming use ``river`` (see README.md, Alternatives).

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

from .benchmark import (
    DETECTOR_LABELS,
    SCORED_CHANGES,
    STANDARD,
    BenchmarkConfig,
    analytic_factory,
    ar1_stream_fn,
    blind_fraction_from_scores,
    blind_fraction_table,
    calibrate_all_analytic,
    calibrate_learned_threshold,
    change_stream_fn,
    default_threshold_operating_points,
    detector_instance,
    learned_arl0,
    learned_arl1,
    measure_change_response,
    tradeoff_curve,
    transient_response,
)
from .detectors import (
    ADWIN,
    ANALYTIC_DETECTORS,
    CUSUM,
    EWMA,
    Detector,
    PageHinkley,
    WindowedKS,
    alarm_ratio_trace,
    first_alarm_at_or_after,
    ks_two_sample_statistic,
    make_detector,
)
from .features import FEATURE_NAMES, WINDOW, window_features, window_features_single
from .learned import (
    LearnedDetector,
    TrainingSet,
    build_training_set,
    score_stream,
    train_learned_detector,
)
from .scoring import (
    ARL0Result,
    ARL1Result,
    TradeoffPoint,
    blind_fraction,
    bootstrap_mean_ci,
    measure_arl0,
    measure_arl1,
    run_lengths_on_stream,
    wilson_interval,
)
from .streams import (
    CHANGE_TYPES,
    ChangeSpec,
    ar1_stationary,
    change_stream,
    stationary,
    transient_spike,
)
from .thresholds import CalibrationResult, calibrate_threshold

__version__ = "0.1.0"

__all__ = [
    "ADWIN",
    "ANALYTIC_DETECTORS",
    "ARL0Result",
    "ARL1Result",
    "CHANGE_TYPES",
    "CUSUM",
    "DETECTOR_LABELS",
    "EWMA",
    "FEATURE_NAMES",
    "SCORED_CHANGES",
    "STANDARD",
    "WINDOW",
    "BenchmarkConfig",
    "CalibrationResult",
    "ChangeSpec",
    "Detector",
    "LearnedDetector",
    "PageHinkley",
    "TradeoffPoint",
    "TrainingSet",
    "WindowedKS",
    "alarm_ratio_trace",
    "analytic_factory",
    "ar1_stationary",
    "ar1_stream_fn",
    "blind_fraction",
    "blind_fraction_from_scores",
    "blind_fraction_table",
    "bootstrap_mean_ci",
    "build_training_set",
    "calibrate_all_analytic",
    "calibrate_learned_threshold",
    "calibrate_threshold",
    "change_stream",
    "change_stream_fn",
    "default_threshold_operating_points",
    "detector_instance",
    "first_alarm_at_or_after",
    "ks_two_sample_statistic",
    "learned_arl0",
    "learned_arl1",
    "make_detector",
    "measure_arl0",
    "measure_arl1",
    "measure_change_response",
    "run_lengths_on_stream",
    "score_stream",
    "stationary",
    "tradeoff_curve",
    "train_learned_detector",
    "transient_response",
    "transient_spike",
    "wilson_interval",
    "window_features",
    "window_features_single",
    "__version__",
]
