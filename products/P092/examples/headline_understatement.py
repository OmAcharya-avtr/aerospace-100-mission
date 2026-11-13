"""The headline: the three baselines against the counted rate, with the interval.

Draws, from the MEASURED campaign:

- left: the survival function of the measured response times on the ground-truth
  run, with the deadline marked, the Gaussian mean-plus-margin tail and the GPD
  peaks-over-threshold tail drawn over it, and the counted miss rate as a point
  with its moving-block bootstrap interval;
- right: the understatement factor per run, with the ground-truth run's
  bootstrap interval.

Writes ``screenshots/headline_understatement.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from scipy import stats  # noqa: E402

from deadlinerisk.accounting import account  # noqa: E402
from deadlinerisk.baselines import (  # noqa: E402
    choose_threshold,
    mean_plus_margin,
    threshold_stability,
)
from deadlinerisk.bootstrap import counted_rate_interval, understatement_factor  # noqa: E402
from deadlinerisk.trace import TraceDataAbsentError  # noqa: E402

RESAMPLES = 2000
SEED = 920092


def main() -> int:
    try:
        campaign = _common.measured_campaign()
    except TraceDataAbsentError as error:
        print(f"REFUSED: {error}")
        return 1

    jobs = campaign.jobs["gt_long"]
    deadline = jobs.deadline_s
    totals = account(jobs)
    margin = mean_plus_margin(jobs.response_s, deadline)
    counted = counted_rate_interval(jobs, RESAMPLES, SEED)
    fits = threshold_stability(jobs.response_s, deadline)
    evt = choose_threshold(fits) if fits else None

    handle, save = _common.figure("headline_understatement")
    left = handle.add_subplot(1, 2, 1)
    ordered = np.sort(jobs.response_s)
    survival = 1.0 - np.arange(1, ordered.size + 1) / ordered.size
    positive = survival > 0.0
    left.semilogy(
        ordered[positive] * 1e3, survival[positive], lw=1.4,
        label=f"measured empirical survival ({ordered.size} executed jobs)",
    )
    grid = np.linspace(float(ordered.min()), max(float(ordered.max()), deadline * 1.05), 400)
    left.semilogy(
        grid * 1e3,
        stats.norm.sf((grid - margin.mean_s) / margin.sd_s),
        lw=1.2, ls="--",
        label=f"baseline 1 Gaussian mean+k*sd (k at D = {margin.k_to_deadline:.2f})",
    )
    if evt is not None:
        chosen = evt.chosen
        tail_grid = grid[grid > chosen.threshold_s]
        if tail_grid.size:
            argument = 1.0 + chosen.shape * (tail_grid - chosen.threshold_s) / chosen.scale
            valid = argument > 0.0
            left.semilogy(
                tail_grid[valid] * 1e3,
                chosen.zeta * argument[valid] ** (-1.0 / chosen.shape),
                lw=1.2, ls=":",
                label=f"baseline 2 GPD at u = {chosen.threshold_s * 1e3:.3f} ms, "
                f"xi = {chosen.shape:.3f}",
            )
    left.axvline(deadline * 1e3, color="black", lw=1.0)
    left.annotate(
        f"deadline D = {deadline * 1e3:.2f} ms",
        xy=(deadline * 1e3, left.get_ylim()[0] * 4.0),
        xytext=(-6, 0), textcoords="offset points",
        rotation=90, ha="right", va="bottom", fontsize=8,
    )
    left.errorbar(
        [deadline * 1e3], [counted.rate],
        yerr=[[max(counted.rate - counted.low, 0.0)], [max(counted.high - counted.rate, 0.0)]],
        fmt="o", ms=6, capsize=4, color="crimson",
        label=f"baseline 3 counted rate over ALL releases\n"
        f"{totals.misses} misses / {totals.releases} releases, "
        f"block {counted.block.block_length}",
    )
    left.set_xlabel("response time (ms)")
    left.set_ylabel("P(response > x)")
    left.set_title("gt_long: measured tail against the two model tails")
    left.grid(alpha=0.3, which="both")
    left.legend(fontsize=7, loc="lower left")

    right = handle.add_subplot(1, 2, 2)
    labels: list[str] = []
    factors: list[float] = []
    for run in sorted(campaign.jobs):
        run_jobs = campaign.jobs[run]
        run_margin = mean_plus_margin(run_jobs.response_s, run_jobs.deadline_s)
        if run_margin.probability <= 0.0:
            continue
        labels.append(run)
        factors.append(account(run_jobs).miss_rate / run_margin.probability)
    positions = np.arange(len(labels))
    right.bar(positions, factors, color="steelblue", width=0.62)
    right.axhline(1.0, color="black", lw=1.0)
    right.annotate(
        "factor 1: the rule is exactly right",
        xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
        xytext=(0, 4), textcoords="offset points", fontsize=8,
    )
    try:
        headline = understatement_factor(jobs, RESAMPLES, SEED)
    except ValueError:
        headline = None
    if headline is not None and "gt_long" in labels:
        index = labels.index("gt_long")
        right.errorbar(
            [index], [headline.factor],
            yerr=[
                [max(headline.factor - headline.low, 0.0)],
                [max(headline.high - headline.factor, 0.0)],
            ],
            fmt="none", ecolor="crimson", capsize=5, lw=1.6,
        )
    right.set_yscale("log")
    right.set_xticks(positions)
    right.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    right.set_ylabel("counted rate / mean-plus-margin P  (log scale)")
    right.set_title("understatement factor per run; interval on gt_long")
    right.grid(alpha=0.3, axis="y", which="both")

    path = save()
    print(f"period T                 {campaign.period_s * 1e3:.3f} ms")
    print(f"deadline D               {campaign.deadline_s * 1e3:.3f} ms")
    print(f"gt_long releases         {totals.releases}")
    print(f"gt_long misses           {totals.misses}")
    print(f"gt_long dropped          {totals.dropped}")
    print(f"counted miss rate        {totals.miss_rate:.6e}")
    print(f"k with mean + k*sd = D   {margin.k_to_deadline:.6f}")
    print(f"mean-plus-margin P       {margin.probability:.6e}")
    if evt is not None:
        print(f"GPD P at chosen u        {evt.chosen.probability:.6e}")
    if headline is not None:
        print(f"understatement factor    {headline.factor:.6e}")
        print(f"95% interval             [{headline.low:.6e}, {headline.high:.6e}]")
        print(f"interval contains 1      {headline.interval_contains_one}")
    print(f"figure written to        {path}")
    print(
        "What to notice: where the dashed Gaussian line sits relative to the solid "
        "measured survival at the deadline. The vertical distance between them at "
        "x = D is the understatement factor."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
