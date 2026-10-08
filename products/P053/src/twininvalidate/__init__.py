"""twininvalidate: model-invalidation monitoring for a linear-Gaussian digital twin.

What this package answers: given a declared digital twin and a stream of its
residuals, **how many samples after the physical asset changes do you notice,
against how many false alarms you are prepared to accept.** The deliverable is
the detection-delay against false-alarm-rate curve (ARL1 against ARL0), per
change type, with the threshold-setting method declared.

What it does not answer: whether the asset is faulty or the twin is wrong. The
residual is driven by the mismatch between them and carries no information
about which side moved; :mod:`twininvalidate.ambiguity` demonstrates two
physically different worlds whose residual streams are algebraically identical.

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

from .ambiguity import PairedStreams, max_absolute_difference, paired_streams
from .arl import (
    ArlEstimate,
    CurvePoint,
    arl0_estimate,
    arl1_estimate,
    delay_after_onset,
    delay_curve,
    threshold_grid,
)
from .asset import (
    CHANGE_KINDS,
    AssetChange,
    StreamSpec,
    no_change,
    simulate_residuals,
    simulate_residuals_with,
)
from .classifier import (
    DriftClassifier,
    ReliabilityBin,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
)
from .datasets import (
    SCENARIO_LABELS,
    SCENARIOS,
    LabelledWindows,
    build_labelled_set,
    calibration_set,
    changed_streams,
    in_control_streams,
    training_set,
)
from .detectors import (
    BASELINES,
    DETECTORS,
    DetectorSpec,
    cusum_statistic,
    ewma_statistic,
    first_alarm,
    glr_statistic,
    statistic_path,
    variance_cusum_statistic,
)
from .features import DEFAULT_WINDOW, FEATURE_NAMES, N_FEATURES, window_features
from .monitor import InvalidationMonitor, MonitorVerdict
from .thresholds import (
    Calibration,
    ThresholdBracket,
    arl0_from_rate,
    bracket_threshold,
    calibrate_threshold,
    closed_form_threshold,
    ewma_lambda1_threshold,
    glr_window1_threshold,
    rate_from_arl0,
)
from .twin import (
    LinearGaussianTwin,
    SteadyStateFilter,
    attitude_channel,
    cwna_process_noise,
    reference_excitation,
    reference_twin,
    zoh_discretise,
)

__version__ = "0.1.0"

__all__ = [
    "ArlEstimate",
    "AssetChange",
    "BASELINES",
    "CHANGE_KINDS",
    "Calibration",
    "CurvePoint",
    "DEFAULT_WINDOW",
    "DETECTORS",
    "DetectorSpec",
    "DriftClassifier",
    "FEATURE_NAMES",
    "InvalidationMonitor",
    "LabelledWindows",
    "LinearGaussianTwin",
    "MonitorVerdict",
    "N_FEATURES",
    "PairedStreams",
    "ReliabilityBin",
    "SCENARIOS",
    "SCENARIO_LABELS",
    "SteadyStateFilter",
    "StreamSpec",
    "ThresholdBracket",
    "arl0_estimate",
    "arl0_from_rate",
    "arl1_estimate",
    "attitude_channel",
    "bracket_threshold",
    "brier_score",
    "build_labelled_set",
    "calibrate_threshold",
    "calibration_set",
    "changed_streams",
    "closed_form_threshold",
    "cusum_statistic",
    "cwna_process_noise",
    "delay_after_onset",
    "delay_curve",
    "ewma_lambda1_threshold",
    "ewma_statistic",
    "expected_calibration_error",
    "first_alarm",
    "glr_statistic",
    "glr_window1_threshold",
    "in_control_streams",
    "max_absolute_difference",
    "no_change",
    "paired_streams",
    "rate_from_arl0",
    "reference_excitation",
    "reference_twin",
    "reliability_diagram",
    "simulate_residuals",
    "simulate_residuals_with",
    "statistic_path",
    "threshold_grid",
    "training_set",
    "variance_cusum_statistic",
    "window_features",
    "zoh_discretise",
]
