"""hiltimingdiff -- per-stage timing divergence between simulation and hardware.

One six-stage PAT (pointing, acquisition, tracking) frame pipeline is executed
in two modes and compared stage by stage:

* **simulation mode**, measured in this container by
  :func:`hiltimingdiff.simulate.measure_simulation`, which imports the
  committed ``data/pat_pipeline.py`` and times it with the committed
  ``data/staged_exec.py``;
* **hardware mode**, read from the committed campaign in ``data/traces/``, run
  on a Jetson Orin Nano by ``data/campaign.py``.

The same file runs in both modes, so a per-stage timing difference is a
property of the platform and not of two implementations. The md5 of every
committed board program is asserted before a frame is timed
(:func:`hiltimingdiff.sources.verify_source_identity`).

The headline is the divergence, not a by-product of it: per-stage ratio
``R[s]``, the spread ``max(R)/min(R)``, the rank disagreement between the
declared and measured stage orderings, the least and most trustworthy stages,
and how far a single measured scaling factor transfers to a stage nobody put on
the board.

There is no learned component. The registry declares ``ai_enabled: false``, and
a model fitted to the same measurements it would be scored against would add a
second thing to trust and no honest information.

Every hardware figure here is a figure at an unpinned, ``schedutil``-governed
clock in ``MAXN_SUPER``, on a board that drifted 7.52 per cent while the
campaign ran. Both facts are measured, reported with the headline, and
quantified: see :mod:`hiltimingdiff.frequency` and :mod:`hiltimingdiff.drift`.

Research-grade. Not flight-qualified, not certified, and not approved for
operational aerospace use.
"""

from __future__ import annotations

from .baselines import (
    HOLDOUT_STAGES,
    PREREGISTERED_SPREAD,
    PREREGISTERED_TRANSFER_ERROR,
    TRAIN_STAGES,
    Baseline1,
    Baseline2,
    Baseline3,
    baseline_1,
    baseline_2,
    baseline_3,
)
from .bootstrap import (
    BOOTSTRAP_SEED,
    CONFIDENCE,
    RESAMPLES,
    RESAMPLING_UNIT,
    Interval,
    bootstrap_ratio_of_medians,
    bootstrap_statistic,
    percentile_interval,
    unit_resamples,
)
from .divergence import (
    Divergence,
    PassUnits,
    divergence,
    kendall_tau_b,
    ordering,
    pass_units,
    rank_permutation,
)
from .drift import (
    Confounding,
    DriftComparison,
    DriftFit,
    LinearFit,
    calibration_blocks,
    cell_drift_factors,
    confounding,
    drift_adjusted_units,
    drift_comparison,
    fit_drift,
    linear_fit,
    midranks,
    spearman_rho,
)
from .frequency import (
    FrequencyCovariation,
    StageCovariation,
    frequency_covariation,
    pair_frequency_with_passes,
)
from .gate import GateCensus, gate_census
from .report import (
    MEASUREMENT_MODEL,
    RESEARCH_GRADE,
    SCOPE_LINE,
    Headline,
    Verdict,
    build_headline,
    headline_lines,
    spread_verdict,
    transfer_verdict,
)
from .simulate import (
    DEFAULT_FRAMES,
    DEFAULT_PASSES,
    SimulationResult,
    calibrate_ns_per_op,
    measure_simulation,
)
from .sources import (
    SOURCE_MD5,
    TIMED_SOURCES,
    SourceIdentityError,
    declared_ops,
    executor_module,
    md5_of_file,
    measured_md5,
    pipeline_module,
    reject_reasons,
    source_identity_failures,
    stage_names,
    verify_source_identity,
)
from .traces import (
    CALIBRATION_BLOCKS,
    GATE_BRANCHES,
    REQUIRED_CELLS,
    REQUIRED_FILES,
    Campaign,
    TraceDataAbsentError,
    campaign_is_complete,
    hardware_pass_table,
    load_campaign,
    traces_label,
)

__version__ = "0.1.0"

__all__ = [
    "BOOTSTRAP_SEED",
    "CALIBRATION_BLOCKS",
    "CONFIDENCE",
    "DEFAULT_FRAMES",
    "DEFAULT_PASSES",
    "GATE_BRANCHES",
    "HOLDOUT_STAGES",
    "MEASUREMENT_MODEL",
    "PREREGISTERED_SPREAD",
    "PREREGISTERED_TRANSFER_ERROR",
    "REQUIRED_CELLS",
    "REQUIRED_FILES",
    "RESAMPLES",
    "RESAMPLING_UNIT",
    "RESEARCH_GRADE",
    "SCOPE_LINE",
    "SOURCE_MD5",
    "TIMED_SOURCES",
    "TRAIN_STAGES",
    "Baseline1",
    "Baseline2",
    "Baseline3",
    "Campaign",
    "Confounding",
    "Divergence",
    "DriftComparison",
    "DriftFit",
    "FrequencyCovariation",
    "GateCensus",
    "Headline",
    "Interval",
    "LinearFit",
    "PassUnits",
    "SimulationResult",
    "SourceIdentityError",
    "StageCovariation",
    "TraceDataAbsentError",
    "Verdict",
    "__version__",
    "baseline_1",
    "baseline_2",
    "baseline_3",
    "bootstrap_ratio_of_medians",
    "bootstrap_statistic",
    "build_headline",
    "calibrate_ns_per_op",
    "calibration_blocks",
    "campaign_is_complete",
    "cell_drift_factors",
    "confounding",
    "declared_ops",
    "divergence",
    "drift_adjusted_units",
    "drift_comparison",
    "executor_module",
    "fit_drift",
    "frequency_covariation",
    "gate_census",
    "hardware_pass_table",
    "headline_lines",
    "kendall_tau_b",
    "linear_fit",
    "load_campaign",
    "md5_of_file",
    "midranks",
    "measure_simulation",
    "measured_md5",
    "ordering",
    "pair_frequency_with_passes",
    "pass_units",
    "percentile_interval",
    "pipeline_module",
    "rank_permutation",
    "reject_reasons",
    "source_identity_failures",
    "spearman_rho",
    "spread_verdict",
    "stage_names",
    "traces_label",
    "transfer_verdict",
    "unit_resamples",
    "verify_source_identity",
]
