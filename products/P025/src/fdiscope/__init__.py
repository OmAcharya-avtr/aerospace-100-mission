"""FDIScope -- residual-based fault detection and isolation for a GNC loop.

The package is organised as a pipeline, and each stage is usable on its own:

``plant`` -> ``kalman`` -> ``simulate`` -> ``residuals`` -> ``detectors``
-> ``isolation`` / ``classifier`` -> ``evaluate``

* :mod:`fdiscope.plant` -- single-axis attitude loop, ZOH-discretised.
* :mod:`fdiscope.kalman` -- Kalman filter written to expose the innovation.
* :mod:`fdiscope.faults` -- seven fault modes, sensor and actuator.
* :mod:`fdiscope.simulate` -- the closed loop with fault injection.
* :mod:`fdiscope.residuals` -- normalisation and filter-consistency checks.
* :mod:`fdiscope.analytic` -- threshold design and detection-delay expressions.
* :mod:`fdiscope.detectors` -- chi-squared and CUSUM tests.
* :mod:`fdiscope.isolation` -- classical GLR bank over fault signatures.
* :mod:`fdiscope.features`, :mod:`fdiscope.classifier` -- the learned model.
* :mod:`fdiscope.scenarios`, :mod:`fdiscope.evaluate`, :mod:`fdiscope.metrics`
  -- seeded scenarios and the benchmark harness.

Everything in this package is research-grade and SIMULATED.  It is not
flight-qualified, not certified, and not approved for operational aerospace
use.  See ``README.md``.
"""

from __future__ import annotations

from .analytic import (
    SIEGMUND_RHO,
    chi2_detection_power,
    chi2_false_alarm_rate,
    chi2_threshold,
    cusum_arl0_siegmund,
    cusum_delay_mean_path,
    cusum_delay_siegmund,
    cusum_delay_wald,
    cusum_kl_information,
    cusum_threshold_for_arl0,
    innovation_dc_gain,
    normalised_bias_signature,
    steady_state_gain,
    steady_state_innovation_mean,
)
from .classifier import ClassifierPrediction, FaultClassifier
from .detectors import (
    ChiSquaredDetector,
    CusumBank,
    CusumDetector,
    DetectorOutput,
    detection_delay,
    first_alarm_index,
)
from .evaluate import (
    BenchmarkConfig,
    IsolationOutcome,
    MethodResult,
    build_cusum_bank,
    build_default_bank,
    calibrate_all_thresholds,
    calibrate_threshold,
    class_labels,
    default_scenario_sets,
    default_signature_specs,
    design_thresholds,
    evaluate_detection,
    evaluate_isolation,
    harvest_training_rows,
    healthy_calibration_runs,
    method_names,
    run_scenarios,
    sequential_alarms,
    sequential_scores,
    window_scores,
)
from .faults import (
    ACTUATOR_FAULTS,
    FAULT_CLASSES,
    SENSOR_FAULTS,
    FaultSpec,
    FaultType,
    apply_actuator_fault,
    apply_sensor_fault,
    class_index,
)
from .features import N_FEATURES, feature_matrix, feature_names, window_features
from .isolation import (
    IsolationResult,
    SignatureBank,
    build_signature_bank,
    fault_signature,
    glr_statistics,
    isolate_window,
)
from .kalman import (
    KalmanFilter,
    KalmanState,
    UpdateResult,
    steady_state_covariance,
)
from .metrics import (
    ConfusionReport,
    Interval,
    RocCurve,
    confusion_matrix,
    confusion_report,
    mean_ci,
    roc_curve,
    wilson_interval,
)
from .plant import ControllerGains, LoopMatrices, PlantConfig, loop_matrices
from .residuals import (
    NisCheck,
    nis_consistency,
    nis_from_residual,
    normalise,
    whiteness,
)
from .scenarios import (
    DEFAULT_RANGES,
    MagnitudeRanges,
    Scenario,
    ScenarioSet,
    sample_scenario,
    sample_scenarios,
)
from .simulate import LoopConfig, LoopRun, build_filter, simulate_loop

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # plant and loop
    "PlantConfig",
    "ControllerGains",
    "LoopMatrices",
    "loop_matrices",
    "LoopConfig",
    "LoopRun",
    "build_filter",
    "simulate_loop",
    # estimator
    "KalmanFilter",
    "KalmanState",
    "UpdateResult",
    "steady_state_covariance",
    # faults
    "FaultType",
    "FaultSpec",
    "FAULT_CLASSES",
    "SENSOR_FAULTS",
    "ACTUATOR_FAULTS",
    "class_index",
    "apply_sensor_fault",
    "apply_actuator_fault",
    # residuals
    "normalise",
    "nis_from_residual",
    "NisCheck",
    "nis_consistency",
    "whiteness",
    # analytic design
    "SIEGMUND_RHO",
    "chi2_threshold",
    "chi2_false_alarm_rate",
    "chi2_detection_power",
    "cusum_kl_information",
    "cusum_delay_wald",
    "cusum_delay_siegmund",
    "cusum_delay_mean_path",
    "cusum_arl0_siegmund",
    "cusum_threshold_for_arl0",
    "steady_state_gain",
    "innovation_dc_gain",
    "steady_state_innovation_mean",
    "normalised_bias_signature",
    # detectors
    "DetectorOutput",
    "ChiSquaredDetector",
    "CusumDetector",
    "CusumBank",
    "first_alarm_index",
    "detection_delay",
    # isolation
    "SignatureBank",
    "IsolationResult",
    "fault_signature",
    "build_signature_bank",
    "glr_statistics",
    "isolate_window",
    # learned model
    "N_FEATURES",
    "feature_names",
    "window_features",
    "feature_matrix",
    "FaultClassifier",
    "ClassifierPrediction",
    # scenarios, benchmark, metrics
    "Scenario",
    "ScenarioSet",
    "MagnitudeRanges",
    "DEFAULT_RANGES",
    "sample_scenario",
    "sample_scenarios",
    "BenchmarkConfig",
    "MethodResult",
    "IsolationOutcome",
    "default_signature_specs",
    "build_cusum_bank",
    "build_default_bank",
    "default_scenario_sets",
    "design_thresholds",
    "run_scenarios",
    "harvest_training_rows",
    "healthy_calibration_runs",
    "calibrate_threshold",
    "calibrate_all_thresholds",
    "sequential_scores",
    "sequential_alarms",
    "evaluate_detection",
    "evaluate_isolation",
    "window_scores",
    "method_names",
    "class_labels",
    "Interval",
    "wilson_interval",
    "mean_ci",
    "confusion_matrix",
    "ConfusionReport",
    "confusion_report",
    "RocCurve",
    "roc_curve",
]
