"""The worked example quoted in README.md, executed so the output cannot drift.

Input: SYNTHETIC fixtures, both declared regimes. This is the example a reader
runs first, and it exists in two regimes on purpose: a reader must be able to see
what the product reports when the board does NOT throttle as well as when it
does, because which of those the real board does is a measured question and not
an assumption.

Every line printed under "README BLOCK" below is pasted into README.md verbatim.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from thermalbudget import (  # noqa: E402
    VALIDATION_LEVEL,
    __version__,
    burst_baseline,
    equilibrium_test,
    overstatement,
    rc_baseline,
    sustained_truth,
)
from thermalbudget.provenance import meet  # noqa: E402


def main() -> int:
    out = _bootstrap.Tee("worked_example_output.txt")
    out.rule("WORKED EXAMPLE -- the README's first run, in both declared regimes")
    out(f"thermalbudget {__version__}")
    out(f"VALIDATION_LEVEL = {VALIDATION_LEVEL}")
    out("INPUT ORIGIN: SYNTHETIC fixtures. Not a measurement of any board. The measured")
    out("figures are in validate_measured_campaign_output.txt, which refuses to run")
    out("without the campaign's traces.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    for regime in ("no_throttle", "throttled"):
        out.rule(f"REGIME {regime}")
        campaign = _bootstrap.synthetic_campaign(regime)
        out(f"runs loaded: {len(campaign)}; duty cycles {list(campaign.duty_cycles)}")

        out("")
        out("README BLOCK")
        out("-" * 78)
        verdict = equilibrium_test(campaign["soak_1.00_r1"])
        out(f"equilibrium  soak_1.00_r1  {verdict.status}")
        out(f"             slope {verdict.slope_c_per_s:+.6f} degC/s over the final "
            f"{verdict.window_s:g} s")
        out(f"             criterion {verdict.threshold_c_per_s:g} degC/s, "
            f"{verdict.n_samples} finite samples")

        base1 = burst_baseline(
            campaign.bursts(),
            seed=_bootstrap.SEED_ANALYSIS,
            n_resamples=_bootstrap.RESAMPLES,
        )
        base3 = sustained_truth(
            campaign.soaks(1.0),
            window_s=_bootstrap.WINDOW_S,
            threshold_c_per_s=_bootstrap.THRESHOLD_C_PER_S,
            seed=_bootstrap.SEED_ANALYSIS,
            n_resamples=_bootstrap.RESAMPLES,
        )
        base2 = rc_baseline(campaign.soaks(1.0), campaign.cooldowns(), campaign.bursts())
        head = overstatement(
            base1.throughput_groups,
            base3.throughput_groups,
            seed=_bootstrap.SEED_ANALYSIS,
            n_resamples=_bootstrap.RESAMPLES,
            origin=meet(base1.origin, base3.origin),
        )
        out(f"baseline 1   burst      {base1.throughput.point:9.2f} "
            f"[{base1.throughput.ci_low:.2f}, {base1.throughput.ci_high:.2f}] iters/s")
        out(f"baseline 2   predicted  {base2.predicted_throughput_iters_per_s:9.2f} iters/s"
            + ("   [clock relation DEGENERATE: reduces to baseline 1]"
               if base2.reduces_to_baseline_1 else
               f"   [clock {base2.predicted_clock_khz:.0f} kHz at "
               f"{base2.predicted_die_c:.2f} degC]"))
        out(f"             tau rise   {base2.tau_rise_mean_s:9.2f} s      "
            f"tau cooldown {base2.tau_cool_mean_s:.2f} s      ratio "
            f"{base2.tau_ratio:.3f}")
        out(f"baseline 3   sustained  {base3.throughput.point:9.2f} "
            f"[{base3.throughput.ci_low:.2f}, {base3.throughput.ci_high:.2f}] iters/s "
            f"over {len(base3.included)} equilibrated run(s)")
        out(f"             steady die {base3.die_c.point:9.4f} "
            f"[{base3.die_c.ci_low:.4f}, {base3.die_c.ci_high:.4f}] degC")
        out(f"HEADLINE     sustained/burst {head.sustained_fraction.point:.6f} "
            f"[{head.sustained_fraction.ci_low:.6f}, "
            f"{head.sustained_fraction.ci_high:.6f}]")
        out(f"             overstatement   {100.0 * head.overstatement.point:+.3f} per cent "
            f"[{100.0 * head.overstatement.ci_low:+.3f}, "
            f"{100.0 * head.overstatement.ci_high:+.3f}] per cent")
        out(f"             verdict         {head.verdict}")
        out(f"             origin          {head.origin}")
        out("-" * 78)

        out.check(
            f"the {regime} regime produced a finite headline with an ordered interval",
            head.overstatement.ci_low <= head.overstatement.point <= head.overstatement.ci_high,
            f"verdict {head.verdict}",
        )
        out.check(
            f"the {regime} headline is labelled SYNTHETIC",
            str(head.origin) == "SYNTHETIC",
            "a figure from a fixture may not be reported as a measurement",
        )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
