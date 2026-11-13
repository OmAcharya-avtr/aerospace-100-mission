"""THE HEADLINE, measured: how far the mean-plus-margin rule is wrong on this board.

The reported factor is

    counted miss rate on the ground-truth run / mean-plus-margin probability

with a 95 per cent moving-block bootstrap interval. The factor is reported
whatever its value. If the interval contains 1, the rule is indistinguishable
from adequate on this board under this payload and that is the headline.

Also measured here, because the scope is part of the result: the per-second
foreign CPU occupancy on every run. The board was SHARED throughout. Foreign
occupancy is measured contamination, not assumed absence, and the measured
figures are printed whatever they are.

INPUT ORIGIN: the MEASURED hardware campaign in ``data/traces/`` and nothing
else. There is no synthetic fallback: if the traces are absent or the campaign
has not written CAMPAIGN_COMPLETE, this script refuses and exits non-zero.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from deadlinerisk.accounting import account, miss_indicator  # noqa: E402
from deadlinerisk.baselines import mean_plus_margin  # noqa: E402
from deadlinerisk.bootstrap import (  # noqa: E402
    block_length_from_autocorrelation,
    counted_rate_interval,
    understatement_factor,
)
from deadlinerisk.report import campaign_lines  # noqa: E402
from deadlinerisk.trace import TraceDataAbsentError  # noqa: E402


def main() -> int:
    out = _bootstrap.Tee("validate_measured_campaign_output.txt")
    out.rule("MEASURED CAMPAIGN -- the headline understatement factor")
    out("INPUT ORIGIN: the MEASURED hardware campaign only. No synthetic fallback.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(_bootstrap.MEASUREMENT_MODEL)

    try:
        campaign = _bootstrap.require_measured_campaign()
    except TraceDataAbsentError as error:
        out.rule("REFUSED -- no measured campaign")
        for line in str(error).splitlines():
            out(f"  {line}")
        out.check(
            "the MEASURED campaign is on disk and complete",
            False,
            "nothing was reported and nothing was substituted",
        )
        out("")
        out(f"raw output written to {out.save()}")
        return 1

    out.rule("1  campaign identity")
    for line in campaign_lines(campaign):
        out(f"  {line}")

    out.rule("2  the declared ground-truth-level rule and the probe table it read")
    out(f"  declared target miss rate : {campaign.manifest.get('gt_target_rate')}")
    out(f"  declared usable band      : {campaign.manifest.get('gt_rate_band')}")
    out(f"  chosen level              : {campaign.manifest.get('gt_level')}")
    out(f"  reason                    : {campaign.manifest.get('gt_reason')}")
    out("")
    out(f"  {'n_cpu':>6}{'releases':>10}{'misses':>9}{'miss_rate':>14}{'median_exec_ms':>16}")
    for probe in campaign.manifest.get("probes", []):
        out(
            f"  {probe.get('n_cpu_stress', -1):>6}{probe.get('releases', -1):>10}"
            f"{probe.get('misses', -1):>9}{probe.get('miss_rate', float('nan')):>14.6f}"
            f"{probe.get('median_exec_s', float('nan')) * 1e3:>16.4f}"
        )
    out("")
    out("  calibration history, including the attempts that missed:")
    out(f"  {'attempt':>8}{'w':>10}{'median_exec_ms':>16}{'frac_of_D':>12}")
    for entry in campaign.manifest.get("calibration", []):
        out(
            f"  {entry.get('attempt', -1):>8}{entry.get('w', -1):>10}"
            f"{entry.get('median_exec_s', float('nan')) * 1e3:>16.4f}"
            f"{entry.get('frac_of_deadline', float('nan')):>12.4f}"
        )

    out.rule("3  deadline accounting, every run")
    out(
        f"  {'run':<12}{'rows':>8}{'drops':>8}{'releases':>10}{'misses':>8}"
        f"{'counted_rate':>14}{'exec_only_rate':>16}"
    )
    for run in sorted(campaign.jobs):
        totals = account(campaign.jobs[run])
        out(
            f"  {run:<12}{totals.executed:>8}{totals.dropped:>8}{totals.releases:>10}"
            f"{totals.misses:>8}{totals.miss_rate:>14.6e}"
            f"{totals.executed_miss_rate:>16.6e}"
        )
    for run in sorted(campaign.jobs):
        jobs = campaign.jobs[run]
        totals = account(jobs)
        out.check(
            f"{run}: releases = sum(1 + dropped_before)",
            totals.releases == int((1 + jobs.dropped_before).sum()),
            f"{totals.releases}",
        )
        out.check(
            f"{run}: misses = sum(missed + dropped_before)",
            totals.misses == int((jobs.missed + jobs.dropped_before).sum()),
            f"{totals.misses}",
        )

    run = _bootstrap.GROUND_TRUTH_RUN
    jobs = campaign.jobs[run]
    totals = account(jobs)
    indicator = miss_indicator(jobs)

    out.rule(f"4  baseline 1 -- the mean-plus-margin rule on {run}")
    margin = mean_plus_margin(jobs.response_s, jobs.deadline_s)
    out(f"  samples (executed jobs) : {margin.n}")
    out(f"  mean response           : {margin.mean_s * 1e3:.6f} ms")
    out(f"  sd response             : {margin.sd_s * 1e3:.6f} ms")
    out(f"  k with mean + k*sd = D  : {margin.k_to_deadline:.6f}")
    out(f"  Gaussian P(resp > D)    : {margin.probability:.6e}")
    out(
        "  The rule is computed from the response times of the EXECUTED jobs, because "
        "a dropped release has no response time. It is compared below against the "
        "counted rate over ALL releases, which is the comparison the practice makes."
    )

    out.rule(f"5  baseline 3 -- the counted rate on {run}, with its block selection")
    choice = block_length_from_autocorrelation(indicator.astype(np.float64))
    out(f"  releases in the indicator : {choice.n}")
    out(f"  misses in the indicator   : {int(indicator.sum())}")
    out(f"  white-noise band 2/sqrt(n): {choice.band:.6e}")
    out("  autocorrelation, lags 0..5: " + ", ".join(f"{v:.6f}" for v in choice.rho_head))
    out(f"  cut-off lag L             : {choice.lag_cut}")
    out(f"  integrated tau            : {choice.tau_integrated:.6f}")
    out(f"  block length (releases)   : {choice.block_length}"
        + ("  [capped at n//20]" if choice.capped else ""))
    out("")
    out(f"  declared rule: {choice.rule}")
    interval = counted_rate_interval(jobs, _bootstrap.RESAMPLES, _bootstrap.SEED_ANALYSIS)
    out("")
    out(f"  counted miss rate         : {interval.rate:.6e}")
    out(
        f"  {int(interval.confidence * 100)}% moving-block interval : "
        f"[{interval.low:.6e}, {interval.high:.6e}]"
    )
    out(f"  resamples / seed          : {interval.resamples} / {interval.seed}")
    out.check(
        "the counted rate lies inside its own bootstrap interval",
        interval.low <= interval.rate <= interval.high,
        f"{interval.low:.6e} <= {interval.rate:.6e} <= {interval.high:.6e}",
    )
    out.check(
        "the indicator sums to the reconstructed miss count",
        int(indicator.sum()) == totals.misses,
        f"{int(indicator.sum())} vs {totals.misses}",
    )

    out.rule("6  THE HEADLINE")
    try:
        headline = understatement_factor(
            jobs, _bootstrap.RESAMPLES, _bootstrap.SEED_ANALYSIS
        )
    except ValueError as error:
        out("  THE FACTOR IS NOT A FINITE NUMBER on this campaign:")
        for line in str(error).splitlines():
            out(f"    {line}")
        out(f"  counted miss rate        : {totals.miss_rate:.6e}")
        out(f"  mean-plus-margin P       : {margin.probability:.6e}")
        out.unmeasurable_check(
            "the understatement FACTOR as a ratio",
            "the mean-plus-margin probability underflowed to 0.0 at this k, so the "
            "ratio is not representable. The two rates are reported separately "
            "above; no substitute number was invented.",
        )
    else:
        out(f"  counted miss rate        : {headline.counted_rate:.6e}")
        out(f"  mean-plus-margin P       : {headline.margin_probability:.6e}")
        out(f"  UNDERSTATEMENT FACTOR    : {headline.factor:.6e}")
        out(
            f"  {int(headline.confidence * 100)}% moving-block interval: "
            f"[{headline.low:.6e}, {headline.high:.6e}]"
        )
        out(f"  block length (releases)  : {headline.block.block_length}")
        out(f"  block length (rows)      : {headline.block_rows}")
        out(
            f"  non-finite replicates    : {headline.n_degenerate} of "
            f"{headline.resamples}"
        )
        out(f"  interval contains 1      : {headline.interval_contains_one}")
        out("")
        if headline.interval_contains_one:
            out(
                "  VERDICT: the interval CONTAINS 1. On this board, under this payload, "
                "at this period and this contention level, the mean-plus-margin rule is "
                "INDISTINGUISHABLE FROM ADEQUATE. That is the headline and it is "
                "published as the headline."
            )
        elif headline.low > 1.0:
            out(
                "  VERDICT: the interval EXCLUDES 1 from above. The rule UNDERSTATES "
                "the counted miss rate by the factor above on this board, under this "
                "payload, at this period and this contention level."
            )
        else:
            out(
                "  VERDICT: the interval EXCLUDES 1 from below. The rule OVERSTATES "
                "the counted miss rate on this board, under this payload, at this "
                "period and this contention level."
            )
        out.check(
            "the point factor lies inside its own bootstrap interval",
            headline.low <= headline.factor <= headline.high,
            f"{headline.low:.6e} <= {headline.factor:.6e} <= {headline.high:.6e}",
        )
        out.check(
            "no bootstrap replicate was non-finite",
            headline.n_degenerate == 0,
            f"{headline.n_degenerate} of {headline.resamples} replicates had a "
            "mean-plus-margin probability of exactly zero",
        )

    out.rule("7  MEASURED foreign CPU occupancy -- the board was shared")
    out(
        "  Per-second foreign occupancy from /proc/stat jiffy deltas, as a percentage "
        "of total CPU capacity. This is measured contamination; nothing here assumes "
        "an idle board."
    )
    out("")
    out(
        f"  {'run':<12}{'samples':>9}{'mean_%':>9}{'median_%':>10}{'p95_%':>9}"
        f"{'max_%':>9}{'our_task_%':>12}{'stress_%':>10}"
    )
    worst = 0.0
    for name in sorted(campaign.monitors):
        monitor = campaign.monitors[name]
        if len(monitor) == 0:
            continue
        foreign = monitor.foreign_pct
        worst = max(worst, float(foreign.max()))
        out(
            f"  {name:<12}{len(monitor):>9}{float(foreign.mean()):>9.3f}"
            f"{float(np.median(foreign)):>10.3f}"
            f"{float(np.percentile(foreign, 95)):>9.3f}{float(foreign.max()):>9.3f}"
            f"{float(monitor.task_pct.mean()):>12.3f}"
            f"{float(monitor.stress_pct.mean()):>10.3f}"
        )
    out("")
    out(f"  worst single-second foreign occupancy across the campaign: {worst:.3f} %")
    out(
        "  The campaign cannot attribute any part of the measured miss rate to foreign "
        "load rather than to the declared stressors, because the foreign load was not "
        "controlled. The figures above are the contamination, reported."
    )

    out.rule("8  thermal and clock state, measured")
    out(
        f"  {'run':<12}{'tj_mean_C':>11}{'tj_max_C':>10}{'clk_mean_kHz':>14}"
        f"{'clk_min_kHz':>13}{'load1_mean':>12}"
    )
    for name in sorted(campaign.monitors):
        monitor = campaign.monitors[name]
        if len(monitor) == 0:
            continue
        out(
            f"  {name:<12}{float(monitor.tj_degC.mean()):>11.2f}"
            f"{float(monitor.tj_degC.max()):>10.2f}"
            f"{float(monitor.clk_mean_khz.mean()):>14.1f}"
            f"{float(monitor.clk_min_khz.min()):>13.0f}"
            f"{float(monitor.load1.mean()):>12.3f}"
        )

    out.rule("9  scope, which is part of the result")
    for line in (
        "one power mode, MAXN_SUPER. `sudo -n true` returns non-zero on this board, so "
        "nvpmodel and jetson_clocks are unreachable unattended and every figure here is "
        "a MAXN_SUPER figure.",
        "one ambient, never instrumented or varied. Held-out ambient is STRUCTURALLY "
        "UNMEASURABLE from this campaign.",
        "one CPython CPU-only payload. Nothing here supports a statement about a "
        "compiled payload, a GPU payload, or a different period.",
        "one period and one implicit deadline.",
        "one board, with its own heatsink and fan.",
        "the board was SHARED throughout; section 7 reports the measured foreign "
        "occupancy rather than assuming absence.",
    ):
        out(f"  - {line}")
    out.unmeasurable_check(
        "held-out ambient temperature",
        "the campaign ran at one uninstrumented ambient. There is no second ambient to "
        "hold out, so this is not a check that failed; it is a check that cannot exist "
        "in this campaign.",
    )
    out.unmeasurable_check(
        "a second power mode",
        "`sudo -n true` returns non-zero on this board, so nvpmodel cannot be driven "
        "unattended. Every figure is a MAXN_SUPER figure and no comparison across "
        "power modes was possible.",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
