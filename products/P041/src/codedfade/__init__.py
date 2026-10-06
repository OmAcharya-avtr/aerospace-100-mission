"""codedfade: coded free-space-optical link performance over temporally correlated fading.

Interleaver depth is the design variable. The channel is generated as a
filtered-Gaussian sample path so that fade duration is an emergent property of the
model, the codes are implemented in-package so the Monte Carlo loop has no external
dependency, and the hardware abstraction layer carries the device contract that a
level-4 claim would eventually be measured against.

This software is research-grade. It is **not flight-qualified, not certified, and
not approved for operational aerospace use.**
"""

from __future__ import annotations

from .channel import (
    GAMMA_GAMMA_SI_PEAK,
    ChannelConfig,
    autocorrelation,
    correlated_gaussian,
    gamma_gamma_parameters_from_si,
    gamma_gamma_scintillation_index,
    generate_amplitude,
    generate_irradiance,
    measured_correlation_time,
    rytov_to_gamma_gamma,
)
from .convolutional import ConvolutionalCode, ViterbiResult
from .fade import (
    FadeStatistics,
    exponential_exceedance,
    fade_runs,
    fade_statistics,
    lognormal_standard_level,
    markov_crossing_rate,
    markov_mean_fade_duration,
    required_interleaver_depth,
    rice_crossing_rate_gauss_kernel,
    rice_mean_fade_duration_gauss_kernel,
)
from .gf import PRIMITIVE_POLYNOMIALS, GF2m
from .hal import (
    ROUND_TRIP_SYMBOL_LIMIT,
    CheckResult,
    DeviceModemBackend,
    ModemBackend,
    ModemConfig,
    ModemSession,
    PreflightReport,
    RunCapture,
    RunMode,
    SimulatedModemBackend,
    environment_record,
    preflight_checks,
    run_preflight,
)
from .interleave import (
    BlockInterleaver,
    ConvolutionalInterleaver,
    InterleaverCost,
    burst_dispersion,
)
from .link import (
    CodedLink,
    LinkResult,
    conditional_bit_error_probability,
    q_function,
    uncoded_bit_error_rate,
)
from .predictor import (
    FEATURE_NAMES,
    WINDOW,
    CrossingDataset,
    FadeExceedancePredictor,
    Prediction,
    brier_score,
    build_dataset,
    crossing_features,
    empirical_exceedance_baseline,
    expected_calibration_error,
    log_loss_score,
    reliability_table,
    roc_auc,
)
from .reedsolomon import (
    ReedSolomon,
    RSDecodeResult,
    bits_to_symbols,
    symbols_to_bits,
)

__version__ = "0.1.0"

__all__ = [
    "BlockInterleaver",
    "ChannelConfig",
    "CheckResult",
    "CodedLink",
    "ConvolutionalCode",
    "ConvolutionalInterleaver",
    "CrossingDataset",
    "DeviceModemBackend",
    "FEATURE_NAMES",
    "FadeExceedancePredictor",
    "GAMMA_GAMMA_SI_PEAK",
    "FadeStatistics",
    "GF2m",
    "InterleaverCost",
    "LinkResult",
    "ModemBackend",
    "ModemConfig",
    "ModemSession",
    "PRIMITIVE_POLYNOMIALS",
    "ROUND_TRIP_SYMBOL_LIMIT",
    "Prediction",
    "PreflightReport",
    "RSDecodeResult",
    "ReedSolomon",
    "RunCapture",
    "RunMode",
    "SimulatedModemBackend",
    "ViterbiResult",
    "WINDOW",
    "__version__",
    "autocorrelation",
    "bits_to_symbols",
    "brier_score",
    "build_dataset",
    "burst_dispersion",
    "conditional_bit_error_probability",
    "correlated_gaussian",
    "crossing_features",
    "empirical_exceedance_baseline",
    "environment_record",
    "expected_calibration_error",
    "exponential_exceedance",
    "fade_runs",
    "fade_statistics",
    "gamma_gamma_parameters_from_si",
    "gamma_gamma_scintillation_index",
    "generate_amplitude",
    "generate_irradiance",
    "log_loss_score",
    "lognormal_standard_level",
    "markov_crossing_rate",
    "markov_mean_fade_duration",
    "measured_correlation_time",
    "preflight_checks",
    "q_function",
    "reliability_table",
    "required_interleaver_depth",
    "rice_crossing_rate_gauss_kernel",
    "rice_mean_fade_duration_gauss_kernel",
    "roc_auc",
    "run_preflight",
    "rytov_to_gamma_gamma",
    "symbols_to_bits",
    "uncoded_bit_error_rate",
]
