"""calibaudit: audit a binary probabilistic forecast.

Decompose the Brier score, quantify the calibration estimator's own bias,
recalibrate against the raw forecast as baseline, and measure whether that
helped on held-out data.

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

from .binning import BINNING_STRATEGIES, assign_bins, bin_edges
from .decomposition import BrierDecomposition, binned_decomposition, murphy_decomposition
from .ece import (
    DebiasedECE,
    ECEBiasCurve,
    ECEBiasRow,
    calibration_gaps,
    debiased_ece,
    ece_bias_curve,
    expected_calibration_error,
    maximum_calibration_error,
    null_ece_distribution,
)
from .recalibration import (
    METHOD_NAMES,
    IsotonicCalibration,
    MethodResult,
    PlattScaling,
    RawForecast,
    RecalibrationAudit,
    SampleSizeRow,
    SampleSizeSweep,
    get_method,
    recalibration_audit,
    sample_size_sweep,
)
from .reliability import (
    BandCoverage,
    ReliabilityCurve,
    band_coverage,
    bootstrap_reliability,
    reliability_curve,
)
from .scores import (
    base_rate_forecast,
    brier_score,
    brier_skill_score,
    check_forecasts,
    constant_forecast,
    log_score,
    log_skill_score,
    skill_score,
)
from .synthetic import (
    SPEC_NAMES,
    AnalyticTruth,
    ForecastSample,
    ForecastSpec,
    analytic_truth,
    calibration_map,
    distortion,
    get_spec,
    sample_forecast,
    spec_names,
)

__version__ = "0.1.0"

__all__ = [
    "AnalyticTruth",
    "BINNING_STRATEGIES",
    "BandCoverage",
    "BrierDecomposition",
    "DebiasedECE",
    "ECEBiasCurve",
    "ECEBiasRow",
    "ForecastSample",
    "ForecastSpec",
    "IsotonicCalibration",
    "METHOD_NAMES",
    "MethodResult",
    "PlattScaling",
    "RawForecast",
    "RecalibrationAudit",
    "ReliabilityCurve",
    "SPEC_NAMES",
    "SampleSizeRow",
    "SampleSizeSweep",
    "__version__",
    "analytic_truth",
    "assign_bins",
    "band_coverage",
    "base_rate_forecast",
    "bin_edges",
    "binned_decomposition",
    "bootstrap_reliability",
    "brier_score",
    "brier_skill_score",
    "calibration_gaps",
    "calibration_map",
    "check_forecasts",
    "constant_forecast",
    "debiased_ece",
    "distortion",
    "ece_bias_curve",
    "expected_calibration_error",
    "get_method",
    "get_spec",
    "log_score",
    "log_skill_score",
    "maximum_calibration_error",
    "murphy_decomposition",
    "null_ece_distribution",
    "recalibration_audit",
    "reliability_curve",
    "sample_forecast",
    "sample_size_sweep",
    "skill_score",
    "spec_names",
]
