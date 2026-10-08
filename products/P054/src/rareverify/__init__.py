"""rareverify: planning and executing a Monte-Carlo rare-event verification campaign.

The package answers one question end to end: given a requirement whose
violation is rare, how many simulation runs does a campaign need, how should
those runs be spent, and what interval may be quoted afterwards.

What is estimated
-----------------
Every probability this package produces is the probability that **the simulated
model** violates the requirement.  It is not the probability that a vehicle
fails.  The gap between the two is model error, input-distribution error and
requirement-specification error, none of which is estimated here and none of
which any confidence interval in this package covers.  That conflation is the
single most common misquotation of a number like these.

Layout
------
:mod:`rareverify.intervals`
    Clopper-Pearson and Wilson intervals, the zero-failure closed forms, and
    exact coverage computation.
:mod:`rareverify.planner`
    Sample-size planning for demonstration and for estimation.
:mod:`rareverify.limitstates`
    Limit states with reference probabilities: linear Gaussian and lognormal
    ratio (closed form), rippled (Gauss-Hermite quadrature).
:mod:`rareverify.montecarlo`
    The crude estimator and its cost arithmetic.
:mod:`rareverify.tilting`
    Importance sampling in the declared mean-shift family, design-point
    search, and the deliberately bad tilts.
:mod:`rareverify.subset`
    Subset simulation (Au and Beck 2001).
:mod:`rareverify.surrogate`
    The learned Gaussian-process surrogate of the limit state and the two ways
    it can be used, one biased and one not.
:mod:`rareverify.benchmark`
    Replication-based measurement of variance reduction.
:mod:`rareverify.knownanswer`
    Known-answer checks with tolerances in counting-noise floors.
:mod:`rareverify.campaign`
    Plan, run, and quote a valid interval.

Not flight-qualified, not certified, not approved for operational aerospace
use.
"""

from __future__ import annotations

from .benchmark import (
    ReplicationSummary,
    VarianceReduction,
    replicate,
    summarise,
    variance_reduction,
)
from .campaign import CampaignReport, run_campaign
from .estimate import (
    EstimateInterval,
    RareEventEstimate,
    binomial_interval,
    bootstrap_interval,
    counting_noise_floor,
    interval_for,
)
from .intervals import (
    ProportionInterval,
    clopper_pearson,
    exact_coverage,
    proportion_interval,
    rule_of_three_upper,
    wilson,
    zero_failure_upper,
)
from .knownanswer import KnownAnswerCheck, check_against_reference
from .limitstates import (
    LimitState,
    LinearGaussianLimitState,
    LognormalRatioLimitState,
    RippledLimitState,
    limit_state_from_name,
)
from .montecarlo import crude_monte_carlo, required_samples_for_cov
from .planner import (
    CampaignPlan,
    detection_probability,
    expected_violations,
    plan_campaign,
    probability_of_zero_violations,
    samples_for_relative_width,
    samples_for_zero_failure_bound,
)
from .subset import SubsetLevel, subset_simulation
from .surrogate import (
    SurrogateFit,
    SurrogateLimitState,
    fit_surrogate,
    radial_design,
    surrogate_design_point,
    surrogate_guided_importance_sampling,
    surrogate_probability,
)
from .tilting import (
    MeanShiftTilt,
    analytic_mean_shift,
    find_design_point,
    find_design_point_radial,
    importance_sampling,
    oracle_mean_shift,
    orthogonal_mean_shift,
    scaled_mean_shift,
)

__version__ = "0.1.0"

__all__ = [
    "CampaignPlan",
    "CampaignReport",
    "EstimateInterval",
    "KnownAnswerCheck",
    "LimitState",
    "LinearGaussianLimitState",
    "LognormalRatioLimitState",
    "MeanShiftTilt",
    "ProportionInterval",
    "RareEventEstimate",
    "ReplicationSummary",
    "RippledLimitState",
    "SubsetLevel",
    "SurrogateFit",
    "SurrogateLimitState",
    "VarianceReduction",
    "__version__",
    "analytic_mean_shift",
    "binomial_interval",
    "bootstrap_interval",
    "check_against_reference",
    "clopper_pearson",
    "counting_noise_floor",
    "crude_monte_carlo",
    "detection_probability",
    "exact_coverage",
    "expected_violations",
    "find_design_point",
    "find_design_point_radial",
    "fit_surrogate",
    "importance_sampling",
    "interval_for",
    "limit_state_from_name",
    "oracle_mean_shift",
    "orthogonal_mean_shift",
    "plan_campaign",
    "probability_of_zero_violations",
    "proportion_interval",
    "radial_design",
    "replicate",
    "required_samples_for_cov",
    "rule_of_three_upper",
    "run_campaign",
    "samples_for_relative_width",
    "samples_for_zero_failure_bound",
    "scaled_mean_shift",
    "subset_simulation",
    "summarise",
    "surrogate_design_point",
    "surrogate_guided_importance_sampling",
    "surrogate_probability",
    "variance_reduction",
    "wilson",
    "zero_failure_upper",
]
