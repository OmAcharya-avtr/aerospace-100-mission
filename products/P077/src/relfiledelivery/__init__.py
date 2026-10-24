"""relfiledelivery -- what a file transfer costs over an intermittent link.

A measurement harness, not a protocol. It drives three retransmission
disciplines against a seeded model of a declared link -- outage durations,
residual loss, forward/return asymmetry, contact windows -- and reports
delivery-time quantiles with the tail stated separately, retransmission
overhead, and per-contact completion probability.

**This is not a CFDP implementation.** ``cfdp-py`` and ``spacepackets``
implement the CCSDS file delivery protocol and its packet formats; use them for
the protocol. What they do not answer is the operator's question: given this
link, how long will this file take, how much overhead will it cost, and what is
the spread. That question is what this package measures.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

from .analytic import (
    AnalyticDelivery,
    RoundsExpectation,
    analytic_delivery,
    analytic_delivery_time_quantile,
    clean_pass_seconds,
    expected_rounds,
    expected_rounds_detail,
    expected_total_transmissions,
    expected_transmissions_per_pdu,
    extended_delivery_time_quantile,
    extended_expected_delivery_time,
    rounds_cdf,
    rounds_quantile,
    transmissions_quantile,
    turnaround_seconds,
)
from .benchmark import BenchmarkRow, calibration_table, render_table, run_benchmark, score
from .dataset import (
    FEATURE_COLUMNS,
    REGIMES,
    TARGET_COLUMN,
    Scenario,
    ScenarioRanges,
    feature_matrix,
    fit_split,
    generate_scenario,
    generate_table,
    targets,
)
from .link import (
    ContactSchedule,
    ContactWindow,
    LinkConditions,
    OutageModel,
    OutageTrace,
    uniform_windows,
)
from .modes import ALL_MODES, TransferMode, parse_mode
from .montecarlo import (
    DEFAULT_QUANTILES,
    DeliveryDistribution,
    completion_by_contact,
    run_ensemble,
    standard_error_of_mean,
    summarise,
)
from .payload import (
    CompletionInvariantError,
    FileTransferSpec,
    digest,
    reassemble,
    source_payload,
)
from .predictors import (
    AnalyticPredictor,
    DeliveryPredictor,
    LearnedQuantilePredictor,
    MeanLookupPredictor,
    build_predictors,
)
from .simulate import (
    TransferResult,
    accounting_discrepancy,
    default_transmission_cap,
    independent_transmission_count,
    simulate_transfer,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_MODES",
    "DEFAULT_QUANTILES",
    "FEATURE_COLUMNS",
    "REGIMES",
    "TARGET_COLUMN",
    "AnalyticDelivery",
    "AnalyticPredictor",
    "BenchmarkRow",
    "CompletionInvariantError",
    "ContactSchedule",
    "ContactWindow",
    "DeliveryDistribution",
    "DeliveryPredictor",
    "FileTransferSpec",
    "LearnedQuantilePredictor",
    "LinkConditions",
    "MeanLookupPredictor",
    "OutageModel",
    "OutageTrace",
    "RoundsExpectation",
    "Scenario",
    "ScenarioRanges",
    "TransferMode",
    "TransferResult",
    "__version__",
    "accounting_discrepancy",
    "analytic_delivery",
    "analytic_delivery_time_quantile",
    "build_predictors",
    "calibration_table",
    "clean_pass_seconds",
    "completion_by_contact",
    "default_transmission_cap",
    "digest",
    "expected_rounds",
    "expected_rounds_detail",
    "expected_total_transmissions",
    "expected_transmissions_per_pdu",
    "extended_delivery_time_quantile",
    "extended_expected_delivery_time",
    "feature_matrix",
    "fit_split",
    "generate_scenario",
    "generate_table",
    "independent_transmission_count",
    "parse_mode",
    "reassemble",
    "render_table",
    "rounds_cdf",
    "rounds_quantile",
    "run_benchmark",
    "run_ensemble",
    "score",
    "simulate_transfer",
    "source_payload",
    "standard_error_of_mean",
    "summarise",
    "targets",
    "transmissions_quantile",
    "turnaround_seconds",
    "uniform_windows",
]
