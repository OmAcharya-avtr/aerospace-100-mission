"""latencynet -- end-to-end latency prediction for staged embedded pipelines.

What this package is
--------------------
Given per-stage profiling of a serial pipeline -- a short aligned trace of
each stage's latency -- predict the *end-to-end tail*: p99 and p99.9, which is
what a deadline cares about. Three predictors, all producing prediction
intervals rather than point estimates, compared on pipelines none of them has
seen:

1. :class:`~latencynet.predictors.AnalyticTailPredictor` -- the analytic
   sum-of-stages model. Means add exactly; variances add exactly when stages
   are uncorrelated; a Fenton-Wilkinson lognormal moment match turns the two
   moments into a quantile.
2. :class:`~latencynet.linear.LinearTailPredictor` -- ordinary least squares
   on thirteen probe features, with the exact Student-t prediction interval.
3. :class:`~latencynet.learned.LearnedTailPredictor` -- gradient-boosted
   trees with quantile-loss interval heads.

Any of the three can be wrapped by
:class:`~latencynet.conformal.ConformalPredictor` for a split-conformal
interval with a distribution-free marginal coverage guarantee.

What this package is not
------------------------
It is not a profiler and it does not measure anything. Every latency it
validates against is *injected* from a declared distribution with a fixed
seed, because the only host available is a shared single-core container whose
wall-clock timings move by factors of several between runs; a measured target
would turn every reported error into a statement about host load. The one
place a measured number appears is the cross-check against P033 EdgeInfer,
where it is labelled volatile with its repeat count and method.

Research-grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

from .analytic import (
    AnalyticPrediction,
    SumOfStagesModel,
    fenton_wilkinson_quantile,
    sum_of_stages_mean,
    sum_of_stages_variance,
)
from .compare import (
    ComparisonResult,
    ModelScore,
    compare_models,
    default_models,
    format_comparison,
)
from .conformal import ConformalPredictor, conformal_radius, coverage_quantisation
from .dataset import (
    PipelineDataset,
    PipelineRecord,
    build_dataset,
    feature_matrix,
    generate_record,
    log_target,
)
from .features import (
    DEPENDENCE_FEATURE_INDICES,
    FEATURE_NAMES,
    N_FEATURES,
    ProbeSummary,
    summarise_probe_trace,
)
from .learned import BoostingHyperparameters, LearnedTailPredictor
from .linear import (
    LinearTailPredictor,
    OLSFit,
    ols_fit,
    ols_predict,
    ols_prediction_interval,
)
from .metrics import (
    AccuracyResult,
    CoverageResult,
    interval_coverage,
    log_accuracy,
    paired_difference_test,
    wilson_interval,
)
from .pipeline import (
    PipelineSpec,
    StageSpec,
    make_lognormal_pipeline,
    sample_stage_latencies,
    sample_total_latency,
)
from .predictors import AnalyticTailPredictor, IntervalPrediction, TailPredictor
from .tails import (
    ConvergenceResult,
    lognormal_quantile,
    lognormal_quantile_se,
    lognormal_tail_convergence,
    quantile,
    quantile_min_samples,
)
from .units import ms_to_s, ns_to_s, s_to_ms, s_to_ns, s_to_us, us_to_s

__version__ = "0.1.0"

__all__ = [
    "DEPENDENCE_FEATURE_INDICES",
    "FEATURE_NAMES",
    "N_FEATURES",
    "AccuracyResult",
    "AnalyticPrediction",
    "AnalyticTailPredictor",
    "BoostingHyperparameters",
    "ComparisonResult",
    "ConformalPredictor",
    "ConvergenceResult",
    "CoverageResult",
    "IntervalPrediction",
    "LearnedTailPredictor",
    "LinearTailPredictor",
    "ModelScore",
    "OLSFit",
    "PipelineDataset",
    "PipelineRecord",
    "PipelineSpec",
    "ProbeSummary",
    "StageSpec",
    "SumOfStagesModel",
    "TailPredictor",
    "__version__",
    "build_dataset",
    "compare_models",
    "conformal_radius",
    "coverage_quantisation",
    "default_models",
    "feature_matrix",
    "fenton_wilkinson_quantile",
    "format_comparison",
    "generate_record",
    "interval_coverage",
    "log_accuracy",
    "log_target",
    "lognormal_quantile",
    "lognormal_quantile_se",
    "lognormal_tail_convergence",
    "make_lognormal_pipeline",
    "ms_to_s",
    "ns_to_s",
    "ols_fit",
    "ols_predict",
    "ols_prediction_interval",
    "paired_difference_test",
    "quantile",
    "quantile_min_samples",
    "s_to_ms",
    "s_to_ns",
    "s_to_us",
    "sample_stage_latencies",
    "sample_total_latency",
    "sum_of_stages_mean",
    "sum_of_stages_variance",
    "summarise_probe_trace",
    "us_to_s",
    "wilson_interval",
]
