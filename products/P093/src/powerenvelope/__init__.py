"""powerenvelope -- energy per useful decision, and the factor the usual unit misses.

An on-board autonomy pipeline is quoted in **energy per inference**. A mission
acts on **decisions it accepted**. Two things sit between the two numbers and
both cost energy: inferences whose output fails the gate, and the platform
power drawn while a duty-cycled pipeline sleeps.

The headline output is the factor by which energy per inference understates the
measured energy per useful decision, decomposed into the rejection factor, the
duty-cycle factor and the nominal-versus-measured active-power error, with the
residual reported rather than asserted away. The factor is reported whatever its
value, including a value near 1, which would say the quoted unit is adequate for
this pipeline on this board.

The measurement model every number here depends on is stated in
:mod:`powerenvelope.trace` and, verbatim, in
:data:`powerenvelope.report.MEASUREMENT_MODEL`.

:mod:`powerenvelope.synthetic` is deliberately NOT re-exported here. It writes a
synthetic campaign, and keeping it off the top-level surface means a
``from powerenvelope import *`` cannot pull a trace generator into the namespace
of an analysis of measured data. Import it by its module path, which makes every
use of it visible at the import line.

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use.
"""

from __future__ import annotations

from .baselines import (
    CONFIDENCE,
    RAMP_SEPARABLE_MIN_S,
    RAMP_WINDOW_S,
    VARIANTS,
    BetweenRunSpread,
    DutyCycleModel,
    EnergyPerInference,
    FloorPower,
    FloorRegression,
    GroundTruth,
    RailAccounting,
    RampEnergy,
    RunOrderRow,
    active_window_summary,
    between_run_spread,
    duty_cycle_model,
    energy_per_inference,
    energy_per_useful_decision_from_cycle_mj,
    floor_occupancy_regression,
    floor_power,
    floor_variants,
    ground_truth,
    mean_active_power_mw,
    rail_accounting_curve,
    ramp_energy,
    run_order_rows,
    sleep_power_mw,
    two_term_cycle_energy_mj,
)
from .bounds import (
    EPS,
    Bracket,
    duty_cycle_lower_bracket,
    extreme_value_bracket,
    headline_lower_bracket,
    riemann_bracket,
    slack_from_conditioning,
)
from .decomposition import Decomposition, decompose
from .energy import (
    GAP_TOLERANCE_S,
    SAMPLE_PERIOD_S,
    Interstitial,
    PhaseOccupancy,
    WindowEnergy,
    active_windows,
    cell_energy,
    clip_series,
    declared_active_energy,
    declared_phases,
    declared_sleep_energy,
    energy_over_windows,
    interstitial_samples,
    merge_intervals,
    phase_occupancy,
    run_bounds,
    run_energy,
    sleep_energy,
    trapezoid_energy_mj,
    whole_run_energy,
)
from .predictor import (
    EXCLUDED_DESCRIPTOR_CLASSES,
    LICENSED_DESCRIPTORS,
    ActivePowerPredictor,
    AnalyticActivePowerModel,
    ConstantPowerPredictor,
    DescriptorLeakError,
    LinearActivePowerModel,
    Prediction,
    Score,
    assert_descriptors_licensed,
    score,
    target_quantisation_sd,
)
from .provenance import Origin, Provenance
from .report import MEASUREMENT_MODEL
from .schema import (
    CURRENT_LSB_MA,
    DECISION_COLUMNS,
    MONITOR_COLUMNS,
    NOMINAL_RAIL_MV,
    POWER_COLUMNS,
    REJECT_REASONS,
    SchemaError,
)
from .trace import (
    CYCLE_S,
    FLOOR_CELL,
    PERMISSIVE_CELL,
    PLANNED_CELLS,
    REFERENCE_CELL,
    REFERENCE_REPEATS,
    Campaign,
    Cell,
    DecisionTrace,
    MonitorTrace,
    PowerTrace,
    TraceDataAbsentError,
    campaign_is_complete,
    load_campaign,
    load_decisions,
    load_monitor,
    load_power,
)
from .windows import (
    FEATURE_NAMES,
    MIN_ACTIVE_S,
    WINDOW_S,
    WindowSet,
    extract_windows,
    grouped_splits,
    pipeline_free_monitor_mask,
)

__version__ = "0.1.0"

__all__ = [
    "CONFIDENCE",
    "CURRENT_LSB_MA",
    "CYCLE_S",
    "DECISION_COLUMNS",
    "EPS",
    "EXCLUDED_DESCRIPTOR_CLASSES",
    "FEATURE_NAMES",
    "FLOOR_CELL",
    "GAP_TOLERANCE_S",
    "LICENSED_DESCRIPTORS",
    "MEASUREMENT_MODEL",
    "MIN_ACTIVE_S",
    "MONITOR_COLUMNS",
    "NOMINAL_RAIL_MV",
    "PERMISSIVE_CELL",
    "PLANNED_CELLS",
    "POWER_COLUMNS",
    "RAMP_SEPARABLE_MIN_S",
    "RAMP_WINDOW_S",
    "REFERENCE_CELL",
    "REFERENCE_REPEATS",
    "REJECT_REASONS",
    "SAMPLE_PERIOD_S",
    "VARIANTS",
    "WINDOW_S",
    "ActivePowerPredictor",
    "AnalyticActivePowerModel",
    "BetweenRunSpread",
    "Bracket",
    "Campaign",
    "Cell",
    "ConstantPowerPredictor",
    "DecisionTrace",
    "Decomposition",
    "DescriptorLeakError",
    "DutyCycleModel",
    "EnergyPerInference",
    "FloorPower",
    "FloorRegression",
    "GroundTruth",
    "LinearActivePowerModel",
    "Interstitial",
    "MonitorTrace",
    "Origin",
    "PhaseOccupancy",
    "PowerTrace",
    "Prediction",
    "Provenance",
    "RailAccounting",
    "RunOrderRow",
    "RampEnergy",
    "SchemaError",
    "Score",
    "TraceDataAbsentError",
    "WindowEnergy",
    "WindowSet",
    "__version__",
    "active_window_summary",
    "active_windows",
    "assert_descriptors_licensed",
    "between_run_spread",
    "campaign_is_complete",
    "cell_energy",
    "clip_series",
    "declared_active_energy",
    "declared_phases",
    "declared_sleep_energy",
    "decompose",
    "duty_cycle_lower_bracket",
    "duty_cycle_model",
    "energy_over_windows",
    "energy_per_inference",
    "energy_per_useful_decision_from_cycle_mj",
    "extract_windows",
    "extreme_value_bracket",
    "floor_occupancy_regression",
    "floor_power",
    "floor_variants",
    "ground_truth",
    "grouped_splits",
    "headline_lower_bracket",
    "interstitial_samples",
    "load_campaign",
    "load_decisions",
    "load_monitor",
    "load_power",
    "mean_active_power_mw",
    "merge_intervals",
    "phase_occupancy",
    "pipeline_free_monitor_mask",
    "rail_accounting_curve",
    "ramp_energy",
    "riemann_bracket",
    "run_bounds",
    "run_energy",
    "run_order_rows",
    "score",
    "slack_from_conditioning",
    "sleep_energy",
    "sleep_power_mw",
    "target_quantisation_sd",
    "trapezoid_energy_mj",
    "two_term_cycle_energy_mj",
    "whole_run_energy",
]
