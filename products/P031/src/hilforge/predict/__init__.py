"""Deadline-overrun prediction: two deterministic baselines, then a learned model.

Order matters here. The two baselines are implemented first and the learned
model is measured against them on the same held-out traces, on lead time,
false-alarm rate and missed-overrun rate. See ``MODEL_CARD.md`` for the
measured outcome, including where a baseline wins.
"""

from __future__ import annotations

from .baselines import FixedThresholdPredictor, QueueingOverrunPredictor
from .data import TraceConfig, generate_trace, split_trace
from .features import FEATURE_NAMES, build_dataset, make_features
from .learned import LearnedOverrunPredictor
from .metrics import (
    OverrunMetrics,
    evaluate_predictor,
    lead_time_stats,
    wilson_interval,
)

__all__ = [
    "FEATURE_NAMES",
    "FixedThresholdPredictor",
    "LearnedOverrunPredictor",
    "OverrunMetrics",
    "QueueingOverrunPredictor",
    "TraceConfig",
    "build_dataset",
    "evaluate_predictor",
    "generate_trace",
    "lead_time_stats",
    "make_features",
    "split_trace",
    "wilson_interval",
]
