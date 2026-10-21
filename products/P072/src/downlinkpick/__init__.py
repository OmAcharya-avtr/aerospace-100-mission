"""downlinkpick: choose what to downlink, and say what the choice cost.

Takes a recorded-data inventory and a contact plan and returns a selection, the
marginal value of the next unit of contact time, and the identity of what was
left behind with the reason it was left.

Two selectors are implemented and both are exposed. :func:`greedy_select` is
value-density ordering with window packing -- the operational default, and the
baseline the exact solver is scored against. :func:`exact_select` solves the
multiple-knapsack MILP to proved optimality on instances small enough to solve.
Which one you should use is a question this package answers with a measured gap
distribution rather than with an opinion; see README.md.

The learned components are a value model for products whose declared value is
missing or stale, and a compressibility predictor feeding the size estimate.
Both are benchmarked against the baselines they have to beat in
``validation/``, and the benchmark result -- including where a baseline wins --
is reported in README.md and MODEL_CARD.md.

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use.
"""

from __future__ import annotations

from .benchmark import (
    ArmResult,
    GapStatistics,
    RatioArmResult,
    benchmark_ratio_sources,
    benchmark_value_sources,
    gap_statistics,
)
from .cases import Case, load_case
from .compress import (
    CompressibilityModel,
    RatioPrediction,
    entropy_floor_ratio,
    entropy_floor_ratios,
    global_mean_ratios,
    instrument_constant_ratios,
    instrument_constant_table,
)
from .contacts import (
    ContactPlan,
    ContactWindow,
    check_slack_bits,
    constant_rate_profile,
    feasibility_slack_bits,
    rate_profile_from_elevation,
)
from .dataset import (
    COMPRESSION_FEATURES,
    INSTRUMENTS,
    VALUE_FEATURES,
    GeneratedInstance,
    TrainingTable,
    generate_instance,
    generate_plan,
    generate_training_table,
    instance_family,
)
from .exact import (
    MilpModel,
    build_model,
    build_pulp_problem,
    enumerate_select,
    exact_select,
    lp_relaxation_bound,
    pulp_available,
    write_lp,
)
from .greedy import dantzig_bound, density_order, greedy_select
from .inventory import DataProduct, Inventory
from .marginal import (
    LeftBehind,
    SweepPoint,
    WindowMarginal,
    best_window_to_extend,
    capacity_sweep,
    left_behind_report,
    marginal_value_of_contact,
    value_left_behind,
)
from .report import (
    format_full_report,
    format_invariants,
    format_left_behind,
    format_marginals,
    format_plan,
    format_selection,
)
from .selection import (
    Assignment,
    InvariantReport,
    Selection,
    SelectionProblem,
    check_invariants,
    sizes_from_ratios,
)
from .stability import StabilityPoint, jaccard, perturb_values, stability_curve
from .valuemodel import (
    SizeAgeHeuristic,
    ValueModel,
    ValuePrediction,
    declared_values,
    fit_size_age_heuristic,
    hybrid_values,
    size_age_values,
    spearman,
)

__version__ = "0.1.0"

__all__ = [
    "COMPRESSION_FEATURES",
    "INSTRUMENTS",
    "VALUE_FEATURES",
    "ArmResult",
    "Assignment",
    "Case",
    "CompressibilityModel",
    "ContactPlan",
    "ContactWindow",
    "DataProduct",
    "GapStatistics",
    "GeneratedInstance",
    "Inventory",
    "InvariantReport",
    "LeftBehind",
    "MilpModel",
    "RatioArmResult",
    "RatioPrediction",
    "Selection",
    "SelectionProblem",
    "SizeAgeHeuristic",
    "StabilityPoint",
    "SweepPoint",
    "TrainingTable",
    "ValueModel",
    "ValuePrediction",
    "WindowMarginal",
    "__version__",
    "benchmark_ratio_sources",
    "benchmark_value_sources",
    "best_window_to_extend",
    "build_model",
    "build_pulp_problem",
    "capacity_sweep",
    "check_invariants",
    "check_slack_bits",
    "constant_rate_profile",
    "dantzig_bound",
    "declared_values",
    "density_order",
    "entropy_floor_ratio",
    "entropy_floor_ratios",
    "enumerate_select",
    "exact_select",
    "feasibility_slack_bits",
    "fit_size_age_heuristic",
    "format_full_report",
    "format_invariants",
    "format_left_behind",
    "format_marginals",
    "format_plan",
    "format_selection",
    "gap_statistics",
    "generate_instance",
    "generate_plan",
    "generate_training_table",
    "global_mean_ratios",
    "greedy_select",
    "hybrid_values",
    "instance_family",
    "instrument_constant_ratios",
    "instrument_constant_table",
    "jaccard",
    "left_behind_report",
    "load_case",
    "lp_relaxation_bound",
    "marginal_value_of_contact",
    "perturb_values",
    "pulp_available",
    "rate_profile_from_elevation",
    "size_age_values",
    "sizes_from_ratios",
    "spearman",
    "stability_curve",
    "value_left_behind",
    "write_lp",
]
