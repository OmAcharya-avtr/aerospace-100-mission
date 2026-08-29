"""LinkSwitch — hybrid RF/FSO link switching policies on simulated dual-channel
telemetry.

Research-grade software. Not flight-qualified, not certified, not approved
for operational aerospace use. All fading and rain data used here are
SIMULATED; no field-measured turbulence or link data is used anywhere in
this package.
"""

from .analytic import (
    OptimalThresholdResult,
    crossing_probability,
    expected_throughput_analytic,
    optimal_threshold_analytic,
    optimal_threshold_grid,
    p_rf_available_estimate,
)
from .learn import OutagePredictor, train_outage_predictor
from .metrics import Aggregate, aggregate_runs, compare_policies, mean_ci
from .optical import OpticalParams
from .policies import FixedThresholdPolicy, HysteresisPolicy, LearnedPolicy
from .rf import RFParams
from .scenario import ScenarioConfig, SwitchCost, Telemetry, generate_telemetry
from .simulate import RunMetrics, run_monte_carlo, simulate_policy

__version__ = "0.1.0"

__all__ = [
    "OpticalParams",
    "RFParams",
    "SwitchCost",
    "ScenarioConfig",
    "Telemetry",
    "generate_telemetry",
    "FixedThresholdPolicy",
    "HysteresisPolicy",
    "LearnedPolicy",
    "OutagePredictor",
    "train_outage_predictor",
    "RunMetrics",
    "simulate_policy",
    "run_monte_carlo",
    "Aggregate",
    "mean_ci",
    "aggregate_runs",
    "compare_policies",
    "OptimalThresholdResult",
    "optimal_threshold_analytic",
    "optimal_threshold_grid",
    "crossing_probability",
    "expected_throughput_analytic",
    "p_rf_available_estimate",
    "__version__",
]
