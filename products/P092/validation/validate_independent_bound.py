"""Bound checks: the answer inside a bound derived from different mathematics.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are not the same check, and the second catches what the
first cannot. Three bounds are checked here, none of them used by any estimator
in this product:

1. **Markov.** ``P(R > D) <= E[R]/D`` for a non-negative response ``R``. The
   left-hand side is a COUNTED rate, so this is a counted number against a
   first-moment bound.
2. **Cantelli**, the one-sided Chebyshev inequality.
   ``P(R - mean >= t) <= s^2/(s^2 + t^2)``. The mean-plus-margin rule's Gaussian
   tail must lie at or below it for any mean and variance.
3. **Bonferroni bracket.** ``max(p1, p2) <= P(miss in either of a consecutive
   pair) <= p1 + p2``, from the monotonicity of probability and the union bound.
   Exact in exact arithmetic, so a violation is a defect and not a tolerance
   question.

Every slack follows the CONDITIONING of its own bound expression. None is
floored at an absolute constant: a floor makes a narrow comparison vacuous,
which is how an earlier product of this mission admitted the very expression its
check existed to catch. The slacks are printed beside the quantities so a reader
can see they are not doing the work.

INPUT ORIGIN: the committed SYNTHETIC generator, plus the MEASURED campaign when
its traces are on disk. Every line says which.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from deadlinerisk.accounting import account, miss_indicator  # noqa: E402
from deadlinerisk.baselines import mean_plus_margin  # noqa: E402
from deadlinerisk.bounds import (  # noqa: E402
    bonferroni_pair_bracket,
    cantelli_slack,
    cantelli_upper_bound,
    markov_slack,
    markov_upper_bound,
)
from deadlinerisk.trace import TraceDataAbsentError  # noqa: E402

EPS = float(np.finfo(np.float64).eps)


def _bounds_section(out, campaign, label: str) -> None:
    out.rule(f"{label}: Markov and Cantelli")
    out(campaign.provenance.banner())
    out("")
    out(
        f"  {'run':<12}{'counted_exec':>14}{'markov':>13}{'m_slack':>11}"
        f"{'gauss_P':>13}{'cantelli':>13}{'c_slack':>11}"
    )
    for run in sorted(campaign.jobs):
        jobs = campaign.jobs[run]
        totals = account(jobs)
        margin = mean_plus_margin(jobs.response_s, jobs.deadline_s)
        mean = float(jobs.response_s.mean())
        markov = markov_upper_bound(mean, jobs.deadline_s)
        m_slack = markov_slack(mean, jobs.deadline_s)
        cantelli = cantelli_upper_bound(margin.mean_s, margin.sd_s, jobs.deadline_s)
        c_slack = cantelli_slack(margin.mean_s, margin.sd_s, jobs.deadline_s)
        out(
            f"  {run:<12}{totals.executed_miss_rate:>14.6e}{markov:>13.6e}"
            f"{m_slack:>11.2e}{margin.probability:>13.6e}{cantelli:>13.6e}"
            f"{c_slack:>11.2e}"
        )
    out("")
    for run in sorted(campaign.jobs):
        jobs = campaign.jobs[run]
        totals = account(jobs)
        margin = mean_plus_margin(jobs.response_s, jobs.deadline_s)
        mean = float(jobs.response_s.mean())
        markov = markov_upper_bound(mean, jobs.deadline_s)
        out.check(
            f"{label}/{run}: counted executed-job miss rate <= Markov bound",
            totals.executed_miss_rate <= markov + markov_slack(mean, jobs.deadline_s),
            f"{totals.executed_miss_rate:.6e} <= {markov:.6e}",
        )
        cantelli = cantelli_upper_bound(margin.mean_s, margin.sd_s, jobs.deadline_s)
        slack = cantelli_slack(margin.mean_s, margin.sd_s, jobs.deadline_s)
        out.check(
            f"{label}/{run}: Gaussian mean-plus-margin tail <= Cantelli bound",
            margin.probability <= cantelli + slack,
            f"{margin.probability:.6e} <= {cantelli:.6e} (slack {slack:.2e}, "
            f"{'not ' if slack < abs(cantelli - margin.probability) else ''}"
            "load-bearing)",
        )

    out.rule(f"{label}: Bonferroni bracket on the consecutive-pair miss rate")
    out(
        f"  {'run':<12}{'pairs':>10}{'p_first':>13}{'p_second':>13}"
        f"{'lower':>13}{'observed':>13}{'upper':>13}{'slack':>11}"
    )
    for run in sorted(campaign.jobs):
        bracket = bonferroni_pair_bracket(miss_indicator(campaign.jobs[run]))
        out(
            f"  {run:<12}{bracket.n_pairs:>10}{bracket.p_first:>13.6e}"
            f"{bracket.p_second:>13.6e}{bracket.lower:>13.6e}"
            f"{bracket.observed:>13.6e}{bracket.upper:>13.6e}{bracket.slack:>11.2e}"
        )
    out("")
    for run in sorted(campaign.jobs):
        bracket = bonferroni_pair_bracket(miss_indicator(campaign.jobs[run]))
        out.check(
            f"{label}/{run}: pair miss rate inside the Bonferroni bracket",
            bracket.inside,
            f"[{bracket.lower:.6e}, {bracket.upper:.6e}] contains "
            f"{bracket.observed:.6e}",
        )


def main() -> int:
    out = _bootstrap.Tee("validate_independent_bound_output.txt")
    out.rule("INDEPENDENT BOUNDS -- mathematics the estimators do not use")
    out("INPUT ORIGIN: SYNTHETIC generator, plus the MEASURED campaign when present.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    out.rule("0  the bounds, and why each slack is what it is")
    out(
        "  Markov slack      : 4*eps*bound. The bound is a quotient of two positive "
        "quantities, so its relative condition number is 1 in each argument and the "
        "only amplification is the division. The slack vanishes with the bound."
    )
    out(
        "  Cantelli slack    : 8*eps*((|D|+|mean|)/|D-mean|)*bound. The ill-conditioned "
        "step is the subtraction D - mean; the slack is its relative condition number "
        "times the bound's own magnitude. No floor."
    )
    out(
        "  Bonferroni slack  : 4*eps*max(upper, observed, lower). Every quantity is a "
        "sum of 0/1 over a common denominator, so each carries one rounding of the "
        "division. The slack is zero on an all-zero indicator."
    )
    out(f"  double-precision eps: {EPS:.6e}")

    out.rule("1  the two function-level inequalities, independent of any data")
    for k in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 12.0):
        tail = float(stats.norm.sf(k))
        bound = 1.0 / (1.0 + k * k)
        out.check(
            f"norm.sf({k}) <= 1/(1 + {k}^2)",
            tail <= bound,
            f"{tail:.6e} <= {bound:.6e}",
        )

    out.rule("2  the slack is NOT doing the work -- a floor would be vacuous")
    tiny_bound = cantelli_upper_bound(0.0, 1e-14, 1.0)
    tiny_slack = cantelli_slack(0.0, 1e-14, 1.0)
    out(f"  sd = 1e-14, D - mean = 1: bound {tiny_bound:.6e}, slack {tiny_slack:.6e}")
    out.check(
        "a 1e-28 bound gets a slack far below it, so a narrow comparison stays narrow",
        tiny_slack < 1e-3 * tiny_bound,
        f"slack/bound = {tiny_slack / tiny_bound:.3e}",
    )
    out(
        "  An absolute floor of 1.8e-15, which an earlier product of this mission used, "
        "would exceed this bound by thirteen orders of magnitude and admit anything."
    )

    out.rule("3  SYNTHETIC campaign")
    _bounds_section(out, _bootstrap.synthetic_campaign(), "SYNTHETIC")

    out.rule("4  MEASURED campaign")
    try:
        measured = _bootstrap.require_measured_campaign()
    except TraceDataAbsentError as error:
        out("  MEASURED CAMPAIGN NOT AVAILABLE. The refusal text follows verbatim:")
        for line in str(error).splitlines():
            out(f"    {line}")
        out.unmeasurable_check(
            "independent bounds on the MEASURED campaign",
            "the measured traces were not on disk, or the campaign had not written "
            "CAMPAIGN_COMPLETE, when this script ran. Nothing was checked and "
            "nothing was substituted.",
        )
    else:
        _bounds_section(out, measured, "MEASURED")

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
