"""End-to-end campaign: plan, run, bound.

A campaign ties the three parts together in the order a real one happens:
plan the sample size from a target probability and a confidence requirement,
run an estimator, then quote the interval that is *valid for that estimator*.
The last step is where campaigns go wrong, so :func:`run_campaign` records the
estimator kind alongside the interval and
:attr:`CampaignReport.interval_is_binomial` makes it explicit.

The plan and the verdict must use the **same** bound.  The first version of
this module sized the campaign from the one-sided Clopper-Pearson limit and
then judged the result against the upper end of the two-sided interval, which
is a different and wider number: at ``n = 29956`` and ``k = 0`` the one-sided
95 % limit is 9.99994e-5 while the two-sided upper limit is 1.23136e-4, so a
campaign sized to demonstrate ``p <= 1e-4`` reported that it had failed to do
so.  :attr:`CampaignReport.one_sided_upper` now carries the quantity the
verdict is based on, the two-sided interval is still reported for
completeness, and the mismatch is recorded in validation/VALIDATION.md.

The probability a campaign reports is the failure probability **of the model it
simulated**.  Nothing in this package estimates the probability of a failure of
any real vehicle, and no interval here accounts for model error, for the
difference between the simulated input distribution and reality, or for the
possibility that the requirement was written down wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .estimate import EstimateInterval, RareEventEstimate, interval_for
from .intervals import proportion_interval, zero_failure_upper
from .limitstates import LimitState
from .montecarlo import crude_monte_carlo
from .planner import CampaignPlan, plan_campaign
from .tilting import MeanShiftTilt, analytic_mean_shift, importance_sampling

__all__ = ["CampaignReport", "run_campaign"]


@dataclass(frozen=True)
class CampaignReport:
    """The result of a planned and executed campaign.

    Attributes
    ----------
    plan
        The plan the campaign was sized from.
    estimate
        The estimator result.
    interval
        The interval valid for that estimator.
    interval_is_binomial
        True only when a Clopper-Pearson or Wilson interval was used, which is
        only valid for the crude estimator.
    one_sided_upper
        The one-sided upper confidence limit at the campaign's confidence
        level: the Clopper-Pearson or Wilson one-sided limit for a crude run,
        the one-sided central-limit bound for a weighted estimator.  This is
        the number the verdict uses and the number a campaign should quote.
    zero_failure_bound
        The closed-form one-sided bound for the crude run, present only when
        the crude estimator observed no failure; ``None`` otherwise.  Equal to
        :attr:`one_sided_upper` in that case.
    meets_target
        Whether :attr:`one_sided_upper` is at or below the plan's target
        probability.
    """

    plan: CampaignPlan
    estimate: RareEventEstimate
    interval: EstimateInterval
    interval_is_binomial: bool
    one_sided_upper: float
    zero_failure_bound: float | None
    meets_target: bool

    def describe(self) -> str:
        """Multi-line summary. Contains no I/O."""
        lines = [self.plan.describe(), self.estimate.describe(), self.interval.describe()]
        lines.append(
            f"one-sided upper confidence limit used for the verdict: "
            f"{self.one_sided_upper:.6e}"
        )
        if self.zero_failure_bound is not None:
            lines.append(
                f"zero-failure one-sided upper bound: {self.zero_failure_bound:.6e}"
            )
        lines.append(
            "target met by interval upper limit: "
            f"{'yes' if self.meets_target else 'no'}"
        )
        lines.append(
            "this is the failure probability of the simulated model, not of any "
            "real vehicle"
        )
        return "\n".join(lines)


def run_campaign(
    limit_state: LimitState,
    p_target: float,
    confidence: float = 0.95,
    estimator: str = "crude",
    n_samples: int | None = None,
    tilt: MeanShiftTilt | None = None,
    interval_method: str = "clopper-pearson",
    rng: np.random.Generator | None = None,
) -> CampaignReport:
    """Plan a campaign, run it, and quote a valid interval.

    Parameters
    ----------
    limit_state
        The limit state to verify against.
    p_target
        Target violation probability to bound, in ``(0, 1)``.
    confidence
        Confidence level in ``(0, 1)``.
    estimator
        ``"crude"`` or ``"importance-sampling"``.  Importance sampling uses the
        analytic design-point tilt unless ``tilt`` is supplied.
    n_samples
        Override the planned sample size.  When omitted, the demonstration
        sample size from the plan is used.
    tilt
        Mean-shift tilt for the importance-sampling estimator.
    interval_method
        Binomial interval method for the crude estimator.
    rng
        Random generator.

    Returns
    -------
    CampaignReport
    """
    plan = plan_campaign(p_target, confidence=confidence, method=interval_method)
    n = plan.n_demonstration if n_samples is None else int(n_samples)
    if n < 1:
        raise ValueError(f"n_samples must be at least 1, got {n}")
    generator = np.random.default_rng(0) if rng is None else rng
    if estimator == "crude":
        estimate = crude_monte_carlo(limit_state, n, rng=generator)
    elif estimator == "importance-sampling":
        chosen = analytic_mean_shift(limit_state) if tilt is None else tilt
        estimate = importance_sampling(limit_state, chosen, n, rng=generator)
    else:
        raise ValueError(
            f"estimator must be 'crude' or 'importance-sampling', got {estimator!r}"
        )
    interval = interval_for(
        estimate, confidence=confidence, binomial_method=interval_method
    )
    bound = (
        zero_failure_upper(n, confidence=confidence, method=interval_method, side="upper")
        if estimate.is_binomial and estimate.n_failures == 0
        else None
    )
    if estimate.is_binomial:
        one_sided = proportion_interval(
            estimate.n_failures,
            n,
            confidence=confidence,
            method=interval_method,
            side="upper",
        ).upper
    else:
        z = float(stats.norm.isf(1.0 - confidence))
        one_sided = min(1.0, estimate.estimate + z * estimate.standard_error)
    return CampaignReport(
        plan=plan,
        estimate=estimate,
        interval=interval,
        interval_is_binomial=estimate.is_binomial,
        one_sided_upper=float(one_sided),
        zero_failure_bound=bound,
        meets_target=bool(one_sided <= p_target),
    )
