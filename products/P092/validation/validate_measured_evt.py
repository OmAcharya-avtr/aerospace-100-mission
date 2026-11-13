"""MEASURED baseline 2: the GPD peaks-over-threshold comparison and its diagnostic.

The threshold-choice diagnostic is part of the deliverable, not an optional
extra. The shape ``xi`` and the modified scale ``beta* = beta - xi*u`` are
reported over the whole declared candidate grid, and the chosen threshold is
produced by a rule stated in code (``deadlinerisk.baselines.THRESHOLD_RULE``)
rather than asserted after looking at a plot. ``examples/evt_threshold_stability.py``
draws the same table as the committed figure ``screenshots/evt_threshold_stability.png``.

The three baselines are then compared against each other on the ground-truth
run: the Gaussian mean-plus-margin probability, the GPD tail probability at the
chosen threshold, and the counted rate with its moving-block interval.

Reference: Coles 2001, "An Introduction to Statistical Modeling of Extreme
Values", Springer, chapter 4, for the peaks-over-threshold model, the tail
formula and the modified-scale diagnostic (eq. 4.10).

INPUT ORIGIN: the MEASURED hardware campaign in ``data/traces/`` and nothing
else. No synthetic fallback.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from deadlinerisk.accounting import account  # noqa: E402
from deadlinerisk.baselines import (  # noqa: E402
    MIN_EXCEEDANCES,
    THRESHOLD_QUANTILES,
    choose_threshold,
    mean_plus_margin,
    threshold_stability,
)
from deadlinerisk.bootstrap import counted_rate_interval  # noqa: E402
from deadlinerisk.trace import TraceDataAbsentError  # noqa: E402


def main() -> int:
    out = _bootstrap.Tee("validate_measured_evt_output.txt")
    out.rule("MEASURED EVT -- peaks over threshold, with the stability diagnostic")
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

    run = _bootstrap.GROUND_TRUTH_RUN
    jobs = campaign.jobs[run]
    totals = account(jobs)

    out.rule("1  the declared candidate grid")
    out(f"  candidate quantiles     : {list(THRESHOLD_QUANTILES)}")
    out(f"  minimum exceedances     : {MIN_EXCEEDANCES}")
    out(f"  response samples on {run}: {len(jobs)}")

    out.rule("2  THE THRESHOLD-STABILITY DIAGNOSTIC, measured")
    fits = threshold_stability(jobs.response_s, jobs.deadline_s)
    out(f"  admissible candidates   : {len(fits)} of {len(THRESHOLD_QUANTILES)}")
    if not fits:
        out(
            "  NO candidate threshold left enough exceedances. The GPD baseline cannot "
            "be fitted on this run."
        )
        out.unmeasurable_check(
            "the GPD tail estimate on the ground-truth run",
            f"every candidate threshold on the declared grid left fewer than "
            f"{MIN_EXCEEDANCES} exceedances, so no fit was attempted. This is a "
            "property of the measured response distribution, not a failure of the "
            "fit.",
        )
    else:
        out("")
        out(
            f"  {'quantile':>9}{'u_ms':>12}{'n_exc':>8}{'xi':>12}{'beta_ms':>12}"
            f"{'mod_beta_ms':>14}{'P(resp>D)':>14}{'endpoint<D':>12}"
        )
        for fit in fits:
            out(
                f"  {fit.quantile:>9.4f}{fit.threshold_s * 1e3:>12.6f}"
                f"{fit.n_exceedances:>8}{fit.shape:>12.6f}{fit.scale * 1e3:>12.6f}"
                f"{fit.modified_scale * 1e3:>14.6f}{fit.probability:>14.6e}"
                f"{str(fit.endpoint_below_deadline):>12}"
            )
        choice = choose_threshold(fits)
        out("")
        out("  declared choice rule:")
        for chunk in choice.rule.split("; "):
            out(f"    {chunk}")
        out("")
        out(
            f"  {'quantile':>9}{'forward-window shape sd':>26}"
        )
        for fit, spread in zip(fits, choice.shape_spread, strict=True):
            marker = "  <- chosen" if fit is choice.chosen else ""
            rendered = "inf (window too short)" if not np.isfinite(spread) else f"{spread:.6f}"
            out(f"  {fit.quantile:>9.4f}{rendered:>26}{marker}")
        out("")
        out(f"  chosen threshold u      : {choice.chosen.threshold_s * 1e3:.6f} ms "
            f"(quantile {choice.chosen.quantile:.4f})")
        out(f"  exceedances at u        : {choice.chosen.n_exceedances}")
        out(f"  exceedance rate zeta    : {choice.chosen.zeta:.6e}")
        out(f"  GPD shape xi            : {choice.chosen.shape:.6f}")
        out(f"  GPD scale beta          : {choice.chosen.scale * 1e3:.6f} ms")
        out(f"  modified scale beta*    : {choice.chosen.modified_scale * 1e3:.6f} ms")
        out(f"  GPD P(resp > D)         : {choice.chosen.probability:.6e}")
        out(f"  endpoint below D        : {choice.chosen.endpoint_below_deadline}")
        out.check(
            "the chosen candidate minimises the declared forward-window shape spread",
            float(np.nanmin(np.asarray(choice.shape_spread)))
            == float(
                np.asarray(choice.shape_spread)[
                    [fit.threshold_s for fit in fits].index(choice.chosen.threshold_s)
                ]
            ),
            "the choice is made by code, not by eye",
        )
        out.check(
            "the chosen threshold leaves at least the declared minimum exceedances",
            choice.chosen.n_exceedances >= MIN_EXCEEDANCES,
            f"{choice.chosen.n_exceedances} >= {MIN_EXCEEDANCES}",
        )
        out.check(
            "the modified scale is beta - xi*u at every candidate",
            all(
                abs(fit.modified_scale - (fit.scale - fit.shape * fit.threshold_s))
                <= 8.0 * np.finfo(np.float64).eps * max(abs(fit.scale), 1e-300)
                for fit in fits
            ),
            "Coles 2001 eq. 4.10",
        )
        if choice.chosen.endpoint_below_deadline:
            out(
                "  NOTE: the fitted shape is negative and the fitted finite upper "
                "endpoint u - beta/xi lies BELOW the deadline, so the GPD model states "
                "that the deadline cannot be exceeded and its probability is exactly "
                "zero. That is a MODEL statement, not a measurement, and it is "
                "reported as one: the counted rate below is not zero."
            )

    out.rule("3  the three baselines side by side, on the ground-truth run")
    margin = mean_plus_margin(jobs.response_s, jobs.deadline_s)
    interval = counted_rate_interval(jobs, _bootstrap.RESAMPLES, _bootstrap.SEED_ANALYSIS)
    gpd_probability = choose_threshold(fits).chosen.probability if fits else float("nan")
    # Column widths carry a space of their own: a value exactly as wide as its
    # field prints with no padding and runs into its neighbour. Measured on this
    # campaign, 'none -- the rule gives a point' is 30 characters and
    # 7.939543e-165 is 13, and both collided in the first committed output.
    out(f"  {'baseline':<42}{'P(miss)':>15}{'interval':>32}")
    out(
        f"  {'1 mean-plus-margin (Gaussian)':<42}{margin.probability:>15.6e}"
        f"{'none -- the rule gives a point':>32}"
    )
    out(
        f"  {'2 GPD peaks-over-threshold':<42}{gpd_probability:>15.6e}"
        f"{'none reported here':>32}"
    )
    out(
        f"  {'3 counted rate over all releases':<42}{interval.rate:>15.6e}"
        f"{f'[{interval.low:.4e}, {interval.high:.4e}]':>32}"
    )
    out("")
    out(f"  k with mean + k*sd = D  : {margin.k_to_deadline:.6f}")
    out(f"  counted misses          : {totals.misses} of {totals.releases} releases")
    out(
        f"  dropped releases        : {totals.dropped}  (each one a miss by the "
        "measurement model)"
    )
    out("")
    for name, probability in (
        ("baseline 1 (mean-plus-margin)", margin.probability),
        ("baseline 2 (GPD)", gpd_probability),
    ):
        if not np.isfinite(probability):
            out(f"  {name}: not finite, no interval comparison made")
            continue
        inside = interval.low <= probability <= interval.high
        out(
            f"  {name}: {'INSIDE' if inside else 'OUTSIDE'} the counted rate's "
            f"{int(interval.confidence * 100)}% moving-block interval"
        )
    out(
        "  A baseline inside the counted rate's interval is not distinguishable from "
        "the ground truth by this campaign. A baseline outside it is wrong by more than "
        "the campaign's own sampling variability, in the direction shown."
    )

    out.rule("4  the same comparison on every mix")
    out(
        f"  {'run':<12}{'releases':>10}{'counted':>15}{'gauss_P':>15}{'gpd_P':>15}"
        f"{'k':>9}{'n_adm':>7}"
    )
    for name in sorted(campaign.jobs):
        mix = campaign.jobs[name]
        mix_totals = account(mix)
        mix_margin = mean_plus_margin(mix.response_s, mix.deadline_s)
        mix_fits = threshold_stability(mix.response_s, mix.deadline_s)
        mix_gpd = choose_threshold(mix_fits).chosen.probability if mix_fits else float("nan")
        out(
            f"  {name:<12}{mix_totals.releases:>10}{mix_totals.miss_rate:>15.6e}"
            f"{mix_margin.probability:>15.6e}{mix_gpd:>15.6e}"
            f"{mix_margin.k_to_deadline:>9.3f}{len(mix_fits):>7}"
        )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
