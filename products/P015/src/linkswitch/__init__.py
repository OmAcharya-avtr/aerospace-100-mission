"""linkswitch -- hybrid RF-optical link switching policies.

A dual-channel link model (lognormal optical scintillation fading, rain-faded
RF) with three switching policies -- fixed threshold, hysteresis, and a learned
outage predictor with a confidence output -- and a seeded Monte Carlo harness
that scores them on delivered throughput, outage time and switch count.

**The fading models are simulated, not measured.** Nothing in this package has
been validated against field data. Research-grade software: not flight
qualified, not certified, not approved for operational aerospace use.
"""

from .analytic import (
    expected_throughput_fixed_threshold,
    optical_outage_probability,
    optimal_fixed_threshold_db,
)
from .channels import (
    DB_PER_NEPER,
    OpticalChannelParams,
    RainChannelParams,
    ar1_rho,
    fresnel_crossing_time,
    lognormal_sigma_ln,
    rain_specific_attenuation_db_per_km,
    sigma_i2_rytov_plane,
    simulate_ar1_unit,
    simulate_optical_margin_db,
    simulate_rain_attenuation_db,
)
from .evaluate import (
    ConfidenceInterval,
    PolicyRun,
    mean_ci,
    paired_diff_ci,
    run_monte_carlo,
    summarise,
)
from .learned import (
    FEATURE_NAMES,
    LearnedSwitchPolicy,
    OutagePredictor,
    TelemetryFeatureConfig,
    make_features,
    make_labels,
)
from .metrics import PolicyMetrics, evaluate_selection
from .policies import (
    AlwaysOpticalPolicy,
    AlwaysRfPolicy,
    ClairvoyantPolicy,
    FixedThresholdPolicy,
    HysteresisPolicy,
    Policy,
    shift_causal,
)
from .reference import (
    FIGURE_SEEDS,
    HORIZON_SEEDS,
    TEST_SEEDS,
    TRAIN_SEEDS,
    TUNE_SEEDS,
    scenario_a_stationary,
    scenario_b_operational,
)
from .scenario import HybridLinkScenario, LinkTrace, simulate_trace

__version__ = "0.1.0"

__all__ = [
    "DB_PER_NEPER",
    "FEATURE_NAMES",
    "FIGURE_SEEDS",
    "HORIZON_SEEDS",
    "TEST_SEEDS",
    "TRAIN_SEEDS",
    "TUNE_SEEDS",
    "AlwaysOpticalPolicy",
    "AlwaysRfPolicy",
    "ClairvoyantPolicy",
    "ConfidenceInterval",
    "FixedThresholdPolicy",
    "HybridLinkScenario",
    "HysteresisPolicy",
    "LearnedSwitchPolicy",
    "LinkTrace",
    "OpticalChannelParams",
    "OutagePredictor",
    "Policy",
    "PolicyMetrics",
    "PolicyRun",
    "RainChannelParams",
    "TelemetryFeatureConfig",
    "__version__",
    "ar1_rho",
    "evaluate_selection",
    "expected_throughput_fixed_threshold",
    "fresnel_crossing_time",
    "lognormal_sigma_ln",
    "make_features",
    "make_labels",
    "mean_ci",
    "optical_outage_probability",
    "optimal_fixed_threshold_db",
    "paired_diff_ci",
    "rain_specific_attenuation_db_per_km",
    "run_monte_carlo",
    "scenario_a_stationary",
    "scenario_b_operational",
    "shift_causal",
    "sigma_i2_rytov_plane",
    "simulate_ar1_unit",
    "simulate_optical_margin_db",
    "simulate_rain_attenuation_db",
    "simulate_trace",
    "summarise",
]
