"""The worked example quoted in README.md, run so the quoted output is real.

Every line this script prints is pasted into README.md under "Quick start" and
"A worked example". The mission's build guide makes re-running it the LAST step
before hand-back, because two products of this mission caught their own wrong
README numbers that way.

INPUT ORIGIN: the MEASURED hardware campaign in ``data/traces/``. The README's
worked example is required to use measured data, so this script refuses rather
than substituting the synthetic generator.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from deadlinerisk import (  # noqa: E402
    account,
    choose_threshold,
    counted_rate_interval,
    mean_plus_margin,
    threshold_stability,
    understatement_factor,
)
from deadlinerisk.trace import TraceDataAbsentError  # noqa: E402


def main() -> int:
    out = _bootstrap.Tee("worked_example_output.txt")
    out.rule("WORKED EXAMPLE -- the code block in README.md, executed")
    out("INPUT ORIGIN: the MEASURED hardware campaign only.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    try:
        campaign = _bootstrap.require_measured_campaign()
    except TraceDataAbsentError as error:
        out.rule("REFUSED -- no measured campaign")
        for line in str(error).splitlines():
            out(f"  {line}")
        out.check(
            "the MEASURED campaign is on disk and complete",
            False,
            "the README's worked example uses measured data and nothing was substituted",
        )
        out("")
        out(f"raw output written to {out.save()}")
        return 1

    out.rule("the example, line by line")
    out("")
    out("    from deadlinerisk import (")
    out("        account, choose_threshold, counted_rate_interval,")
    out("        mean_plus_margin, threshold_stability, understatement_factor,")
    out("    )")
    out("    from deadlinerisk.provenance import Origin")
    out("    from deadlinerisk.trace import load_campaign")
    out("")
    out('    campaign = load_campaign("data/traces", origin=Origin.MEASURED)')
    out('    jobs = campaign.jobs["gt_long"]')
    out("    totals = account(jobs)")
    out("    margin = mean_plus_margin(jobs.response_s, jobs.deadline_s)")
    out("    counted = counted_rate_interval(jobs, resamples=4000, seed=920092)")
    out("    headline = understatement_factor(jobs, resamples=4000, seed=920092)")
    out("    evt = choose_threshold(")
    out("        threshold_stability(jobs.response_s, jobs.deadline_s)")
    out("    )")
    out("")
    out.rule("printed output")

    jobs = campaign.jobs[_bootstrap.GROUND_TRUTH_RUN]
    totals = account(jobs)
    margin = mean_plus_margin(jobs.response_s, jobs.deadline_s)
    counted = counted_rate_interval(
        jobs, _bootstrap.RESAMPLES, _bootstrap.SEED_ANALYSIS
    )
    evt = choose_threshold(threshold_stability(jobs.response_s, jobs.deadline_s))

    out(f"period T                 {campaign.period_s * 1e3:.3f} ms")
    out(f"deadline D (implicit)    {campaign.deadline_s * 1e3:.3f} ms")
    out(f"executed rows            {totals.executed}")
    out(f"dropped releases         {totals.dropped}")
    out(f"releases                 {totals.releases}")
    out(f"misses                   {totals.misses}")
    out(f"counted miss rate        {totals.miss_rate:.6e}")
    out(f"mean response            {margin.mean_s * 1e3:.6f} ms")
    out(f"sd response              {margin.sd_s * 1e3:.6f} ms")
    out(f"k with mean + k*sd = D   {margin.k_to_deadline:.6f}")
    out(f"mean-plus-margin P       {margin.probability:.6e}")
    out(f"block length (releases)  {counted.block.block_length}")
    out(
        f"counted 95% interval     [{counted.low:.6e}, {counted.high:.6e}]"
    )
    out(f"EVT threshold u          {evt.chosen.threshold_s * 1e3:.6f} ms")
    out(f"EVT shape xi             {evt.chosen.shape:.6f}")
    out(f"EVT P(resp > D)          {evt.chosen.probability:.6e}")
    try:
        headline = understatement_factor(
            jobs, _bootstrap.RESAMPLES, _bootstrap.SEED_ANALYSIS
        )
    except ValueError as error:
        out("understatement factor    NOT A FINITE NUMBER on this campaign:")
        for line in str(error).splitlines():
            out(f"                         {line}")
        out.unmeasurable_check(
            "the understatement factor in the worked example",
            "the mean-plus-margin probability underflowed to zero, so the ratio is "
            "not representable. The two rates are printed above.",
        )
    else:
        out(f"understatement factor    {headline.factor:.6e}")
        out(
            f"factor 95% interval      [{headline.low:.6e}, {headline.high:.6e}]"
        )
        out(f"interval contains 1      {headline.interval_contains_one}")
        out.check(
            "the factor lies inside its own interval",
            headline.low <= headline.factor <= headline.high,
            f"[{headline.low:.6e}, {headline.high:.6e}]",
        )
    out.check(
        "the counted rate lies inside its own interval",
        counted.low <= counted.rate <= counted.high,
        f"[{counted.low:.6e}, {counted.high:.6e}]",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
