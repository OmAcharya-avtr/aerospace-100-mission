"""deadlinerisk -- how far the mean-plus-margin rule is wrong about deadline misses.

The headline output is the factor by which the mean-plus-margin rule understates
the counted deadline-miss rate on a measured hardware run, with a moving-block
bootstrap interval that accounts for serial dependence between consecutive
deadlines.

The measurement model every number here depends on is stated in
:mod:`deadlinerisk.trace`.

This software is research-grade. It is not flight-qualified, not certified, and
not approved for operational aerospace use.
"""

from __future__ import annotations

from .accounting import DeadlineAccount, account, autocorrelation, miss_indicator
from .baselines import (
    GpdFit,
    MeanMarginResult,
    ThresholdChoice,
    choose_threshold,
    counted_rate,
    gpd_tail,
    mean_plus_margin,
    threshold_stability,
)
from .bootstrap import (
    BlockChoice,
    RateInterval,
    UnderstatementFactor,
    block_length_from_autocorrelation,
    counted_rate_interval,
    moving_block_bootstrap,
    moving_block_index,
    percentile_interval,
    understatement_factor,
)
from .bounds import (
    PairBracket,
    bonferroni_pair_bracket,
    cantelli_slack,
    cantelli_upper_bound,
    markov_slack,
    markov_upper_bound,
)
from .predictor import (
    ConstantRatePredictor,
    ContentionMissPredictor,
    Prediction,
    Score,
    score,
    target_noise_sd,
)
from .provenance import Origin, Provenance
from .schema import JOB_COLUMNS, MONITOR_COLUMNS, SchemaError
from .speed_scaling import (
    BOARD_CORES,
    available_fraction,
    predicted_miss_probability,
    predicted_response,
)
from .trace import (
    Campaign,
    JobTrace,
    MonitorTrace,
    TraceDataAbsentError,
    campaign_is_complete,
    load_campaign,
    load_job_trace,
    load_monitor_trace,
)
from .windows import (
    FEATURE_NAMES,
    WINDOW_S,
    WindowSet,
    extract_windows,
    grouped_splits,
    time_split,
)

__version__ = "0.1.0"

__all__ = [
    "BOARD_CORES",
    "FEATURE_NAMES",
    "JOB_COLUMNS",
    "MONITOR_COLUMNS",
    "WINDOW_S",
    "BlockChoice",
    "Campaign",
    "ConstantRatePredictor",
    "ContentionMissPredictor",
    "DeadlineAccount",
    "GpdFit",
    "JobTrace",
    "MeanMarginResult",
    "MonitorTrace",
    "Origin",
    "PairBracket",
    "Prediction",
    "Provenance",
    "RateInterval",
    "SchemaError",
    "Score",
    "ThresholdChoice",
    "TraceDataAbsentError",
    "UnderstatementFactor",
    "WindowSet",
    "__version__",
    "account",
    "autocorrelation",
    "available_fraction",
    "block_length_from_autocorrelation",
    "bonferroni_pair_bracket",
    "campaign_is_complete",
    "cantelli_slack",
    "cantelli_upper_bound",
    "choose_threshold",
    "counted_rate",
    "counted_rate_interval",
    "extract_windows",
    "gpd_tail",
    "grouped_splits",
    "load_campaign",
    "load_job_trace",
    "load_monitor_trace",
    "markov_slack",
    "markov_upper_bound",
    "mean_plus_margin",
    "miss_indicator",
    "moving_block_bootstrap",
    "moving_block_index",
    "percentile_interval",
    "predicted_miss_probability",
    "predicted_response",
    "score",
    "target_noise_sd",
    "threshold_stability",
    "time_split",
    "understatement_factor",
]
