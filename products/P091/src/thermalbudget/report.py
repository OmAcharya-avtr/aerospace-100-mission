"""Assembling the four results into one analysis, and rendering it as text.

:func:`analyse` runs the whole chain in the specification's build order --
equilibrium test, baseline 1, baseline 2, baseline 3, then the learned model --
and returns every intermediate alongside the headline, so nothing in the report
is recomputed from a different input than the number beside it.

:func:`render` writes it out. Three rules govern the rendering and are the
reason this module exists at all rather than the scripts formatting their own
output:

1. **Every figure carries its origin in the same place**, at the head of the
   report and again on the headline line. A report assembled from synthetic
   fixtures says ``SYNTHETIC`` in both places and the words "not a measurement
   of anything" in the header.
2. **The structurally unmeasurable axis is printed as a section, not as a
   footnote.** Held-out ambient appears in the report with its reason every
   time, whether or not a learned model was fitted.
3. **A refusal prints as a line of the report.** A run excluded by the
   equilibrium test, a relaxation fit that would not converge, a clock relation
   that is not identifiable: each is a named line, because the report's value is
   that a reader can see what did not happen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from . import bootstrap as bs
from . import limits
from .baselines import (
    BurstBaseline,
    Overstatement,
    RCBaseline,
    SustainedTruth,
    burst_baseline,
    overstatement,
    rc_baseline,
    sustained_truth,
)
from .equilibrium import DEFAULT_SLOPE_THRESHOLD_C_PER_S, DEFAULT_WINDOW_S, equilibrium_test
from .predictor import (
    HeldOutDutyResult,
    ModelVerdict,
    NotEnoughRunsError,
    build_dataset,
    compare_against_rc,
    leave_one_duty_out,
)
from .provenance import Origin, label, meet


@dataclass(frozen=True)
class CampaignAnalysis:
    """Everything the product computes from one campaign, in build order.

    Attributes
    ----------
    campaign:
        The loaded campaign.
    equilibrium:
        Verdict for every run in the campaign, including idle and cooldown runs,
        which are expected to be flat and to pass; a cooldown that FAILS is
        informative and is therefore tested too.
    baseline1, baseline2, baseline3:
        The three baselines.
    headline:
        The cold-burst overstatement with its interval.
    learned, learned_forest:
        Held-out-duty results for the two models, or ``None`` when the learned
        step could not run.
    verdict, verdict_forest:
        Comparison of each learned margin against baseline 3's run-to-run
        interval.
    notes:
        Named refusals and anything a reader needs.
    origin:
        Meet over everything above.
    """

    campaign: Any
    equilibrium: dict
    baseline1: BurstBaseline
    baseline2: RCBaseline
    baseline3: SustainedTruth
    headline: Overstatement
    learned: HeldOutDutyResult | None
    learned_forest: HeldOutDutyResult | None
    verdict: ModelVerdict | None
    verdict_forest: ModelVerdict | None
    notes: tuple[str, ...]
    origin: Origin


def analyse(
    campaign,
    *,
    window_s: float = DEFAULT_WINDOW_S,
    threshold_c_per_s: float = DEFAULT_SLOPE_THRESHOLD_C_PER_S,
    n_resamples: int = bs.DEFAULT_RESAMPLES,
    run_learned: bool = True,
) -> CampaignAnalysis:
    """Run the whole chain over ``campaign``.

    Parameters
    ----------
    campaign:
        A :class:`~thermalbudget.trace.Campaign`.
    window_s, threshold_c_per_s:
        The equilibrium criterion.
    n_resamples:
        Bootstrap resamples.
    run_learned:
        False skips build step 5, for a report of the classical baselines alone.

    Raises
    ------
    ValueError
        If the campaign lacks the runs a baseline needs. The message names the
        runs, and nothing is substituted for an absent one.
    """
    notes: list[str] = []
    verdicts = {
        trace.label: equilibrium_test(
            trace, window_s=window_s, threshold_c_per_s=threshold_c_per_s
        )
        for trace in campaign
    }
    bursts = campaign.bursts()
    soaks_full = campaign.soaks(1.0)
    if not bursts:
        raise ValueError("the campaign has no burst_* run; baseline 1 cannot be computed")
    if not soaks_full:
        raise ValueError("the campaign has no duty-1.0 soak_* run; baseline 3 cannot be computed")

    base1 = burst_baseline(bursts, seed=910091, n_resamples=n_resamples)
    base2 = rc_baseline(soaks_full, campaign.cooldowns(), bursts)
    base3 = sustained_truth(
        soaks_full,
        window_s=window_s,
        threshold_c_per_s=threshold_c_per_s,
        n_resamples=n_resamples,
    )
    notes.extend(base2.notes)
    for excluded_label, reason in base3.excluded:
        notes.append(f"baseline 3 excluded {excluded_label}: {reason}")

    head = overstatement(
        base1.throughput_groups,
        base3.throughput_groups,
        n_resamples=n_resamples,
        origin=meet(base1.origin, base3.origin),
    )

    learned = forest = None
    verdict = verdict_forest = None
    if run_learned:
        try:
            dataset = build_dataset(
                campaign.soaks(),
                equilibrium_window_s=window_s,
                threshold_c_per_s=threshold_c_per_s,
            )
            for excluded_label, reason in dataset.excluded:
                notes.append(f"learned dataset excluded {excluded_label}: {reason}")
            learned = leave_one_duty_out(dataset, campaign.soaks(), kind="ridge")
            verdict = compare_against_rc(learned, base3.die_c)
            forest = leave_one_duty_out(dataset, campaign.soaks(), kind="forest")
            verdict_forest = compare_against_rc(forest, base3.die_c)
        except (NotEnoughRunsError, ValueError) as error:
            notes.append(f"the learned step did not run: {error}")
    else:
        notes.append("the learned step was skipped by request; baselines only")

    return CampaignAnalysis(
        campaign=campaign,
        equilibrium=verdicts,
        baseline1=base1,
        baseline2=base2,
        baseline3=base3,
        headline=head,
        learned=learned,
        learned_forest=forest,
        verdict=verdict,
        verdict_forest=verdict_forest,
        notes=tuple(notes),
        origin=meet(base1.origin, base2.origin, base3.origin),
    )


def _interval(result: bs.BootstrapResult, unit: str, digits: int = 2) -> str:
    return (
        f"{result.point:.{digits}f} {unit} "
        f"[{result.ci_low:.{digits}f}, {result.ci_high:.{digits}f}] "
        f"({100.0 * result.confidence:g} per cent, {result.n_resamples} resamples)"
    )


def render(analysis: CampaignAnalysis) -> str:
    """Render ``analysis`` as plain text, with every origin label in place."""
    out: list[str] = []
    write = out.append
    origin = analysis.origin
    campaign = analysis.campaign

    write("=" * 78)
    write("thermalbudget -- cold-burst overstatement report")
    write("=" * 78)
    write(f"origin           : {origin} -- {label(origin)}")
    write(f"trace directory  : {campaign.directory.name}/")
    write(f"board (manifest) : {campaign.board_model}")
    write(
        f"declared target  : {limits.BOARD}  "
        f"[{'MATCHES' if campaign.board_matches_declared_target() else 'DOES NOT MATCH'}]"
    )
    write(f"campaign mode    : {limits.POWER_MODE} only (declared limit 1)")
    write(f"runs loaded      : {len(campaign)}")
    idle = campaign.idle_floor_c()
    write(
        "idle floor       : "
        + (f"{idle:.3f} degC (ambient proxy, declared limit 2)" if math.isfinite(idle) else
           "not available: no idle_pre run")
    )

    write("")
    write("-" * 78)
    write("STEP 1  equilibrium test (tested, not assumed)")
    write("-" * 78)
    write(f"criterion: |slope| <= {analysis.baseline3.threshold_c_per_s:g} degC/s over the "
          f"final {analysis.baseline3.window_s:g} s")
    for run_label in sorted(analysis.equilibrium):
        verdict = analysis.equilibrium[run_label]
        slope = (
            f"{verdict.slope_c_per_s:+.6f}"
            if math.isfinite(verdict.slope_c_per_s)
            else "    n/a"
        )
        write(
            f"  {run_label:<16s} {verdict.status:<21s} slope {slope} degC/s  "
            f"n={verdict.n_samples:<4d} missing={verdict.n_missing:<3d} "
            f"mean={verdict.mean_die_c:.2f} degC"
        )

    write("")
    write("-" * 78)
    write("STEP 2  baseline 1 -- cold-burst extrapolation (the thing being tested)")
    write("-" * 78)
    write(f"burst throughput : {_interval(analysis.baseline1.throughput, 'iters/s')}")
    point, low, high = analysis.baseline1.t_interval
    write(
        f"independent check: Student-t over per-run means {point:.2f} [{low:.2f}, {high:.2f}] "
        "iters/s (different mathematics, reported beside the bootstrap)"
    )
    write(f"runs             : {analysis.baseline1.n_runs}, first "
          f"{analysis.baseline1.skip_s:g} s dropped from each")
    write(f"mean die temp    : {analysis.baseline1.mean_die_c:.2f} degC during the bursts")
    write(
        f"run-level draws  : {analysis.baseline1.throughput.n_distinct_group_draws} distinct "
        "multisets; a two-run cluster bootstrap is granular and is not a substitute for "
        "more runs"
    )
    write("baseline 1 predicts sustained throughput = burst throughput = "
          f"{analysis.baseline1.prediction_iters_per_s:.2f} iters/s")

    write("")
    write("-" * 78)
    write("STEP 3  baseline 2 -- first-order thermal RC network, closed form")
    write("-" * 78)
    base2 = analysis.baseline2
    for run_label, tau in sorted(base2.tau_rise_s.items()):
        fit = base2.rise_fits[run_label]
        write(
            f"  rise     {run_label:<16s} tau {tau:8.2f} s  Tinf {fit.asymptote_c:7.3f} degC  "
            f"T0 {fit.initial_c:7.3f} degC  rms {fit.rms_residual_c:.4f} degC  "
            f"rho {fit.residual_autocorrelation:+.3f}  span {fit.span_in_tau:.2f} tau"
        )
    if base2.tau_cool_s:
        for run_label, tau in sorted(base2.tau_cool_s.items()):
            fit = base2.cooldown_fits[run_label]
            write(
                f"  cooldown {run_label:<16s} tau {tau:8.2f} s  Tamb {fit.asymptote_c:7.3f} "
                f"degC  T0 {fit.initial_c:7.3f} degC  rms {fit.rms_residual_c:.4f} degC"
            )
    else:
        write("  cooldown: NO usable cooldown fit; the independent tau is NOT AVAILABLE")
    write(
        f"  tau side by side: rise mean {base2.tau_rise_mean_s:.2f} s, cooldown mean "
        f"{base2.tau_cool_mean_s:.2f} s, ratio cooldown/rise {base2.tau_ratio:.3f} "
        "(1.000 for a true single-pole system; neither value is preferred over the other)"
    )
    for estimate in base2.independent_tau:
        write(
            f"  independent tau [{estimate.method}] {estimate.tau_s:.2f} s "
            f"+/- {estimate.stderr_s:.2f} s, conditioning {estimate.conditioning:.4g}, "
            f"n={estimate.n}"
        )
        write(f"      {estimate.note}")
    write(f"  clock relation : {'IDENTIFIABLE' if base2.relation.identifiable else 'DEGENERATE'}")
    write(f"      {base2.relation.reason}")
    write(
        f"  predicted      : die {base2.predicted_die_c:.3f} degC "
        f"(+/- {base2.predicted_die_spread_c:.3f} half-range across runs), clock "
        f"{base2.predicted_clock_khz:.0f} kHz"
        + ("  [DEGENERATE: the unthrottled clock]" if base2.clock_degenerate else "")
        + ("  [EXTRAPOLATING outside the measured temperature range]"
           if base2.clock_extrapolating else "")
    )
    write(
        f"  anchor         : {base2.anchor_throughput_iters_per_s:.2f} iters/s at "
        f"{base2.anchor_clock_khz:.0f} kHz, from the burst runs (NOT from the soak "
        "equilibrium, which would be circular)"
    )
    write(
        f"  exponent       : {'measured' if base2.exponent.identifiable else 'NOT IDENTIFIABLE'}"
        f" -- {base2.exponent.reason}"
    )
    write(
        "baseline 2 predicts sustained throughput = "
        f"{base2.predicted_throughput_iters_per_s:.2f} iters/s"
        + ("  [reduces to baseline 1]" if base2.reduces_to_baseline_1 else "")
    )

    write("")
    write("-" * 78)
    write("STEP 4  baseline 3 -- ground truth, directly measured")
    write("-" * 78)
    base3 = analysis.baseline3
    write(f"included runs    : {list(base3.included)}")
    if base3.excluded:
        for excluded_label, reason in base3.excluded:
            write(f"EXCLUDED         : {excluded_label} -- {reason}")
    else:
        write("EXCLUDED         : none")
    write(f"sustained        : {_interval(base3.throughput, 'iters/s')}")
    write(f"steady die temp  : {_interval(base3.die_c, 'degC', digits=3)}")
    point, low, high = base3.t_interval_throughput
    write(
        f"independent check: Student-t over per-run means {point:.2f} [{low:.2f}, {high:.2f}] "
        "iters/s"
    )
    write("within-run intervals (the other level of the spread):")
    for run_label, result in sorted(base3.within_run_throughput.items()):
        write(
            f"  {run_label:<16s} {result.point:.2f} [{result.ci_low:.2f}, {result.ci_high:.2f}] "
            f"iters/s over n={result.n_samples} equilibrium samples"
        )
    write(
        f"run-level draws  : {base3.throughput.n_distinct_group_draws} distinct multisets "
        f"over {base3.throughput.n_groups} runs"
    )

    write("")
    write("=" * 78)
    write("HEADLINE  cold-burst overstatement")
    write("=" * 78)
    head = analysis.headline
    write(f"origin           : {head.origin}")
    write(f"burst            : {head.burst_iters_per_s:.2f} iters/s")
    write(f"sustained        : {head.sustained_iters_per_s:.2f} iters/s")
    write(
        f"sustained/burst  : {head.sustained_fraction.point:.6f} "
        f"[{head.sustained_fraction.ci_low:.6f}, {head.sustained_fraction.ci_high:.6f}]"
    )
    write(
        f"overstatement    : {100.0 * head.overstatement.point:+.3f} per cent "
        f"[{100.0 * head.overstatement.ci_low:+.3f}, "
        f"{100.0 * head.overstatement.ci_high:+.3f}] per cent"
    )
    write(f"verdict          : {head.verdict}")
    write(head.statement)

    write("")
    write("-" * 78)
    write("STEP 5  learned thermal-state predictor, against baseline 2 on the same input")
    write("-" * 78)
    if analysis.learned is None:
        write("the learned step produced no result; see the notes below")
    else:
        for result, verdict, name in (
            (analysis.learned, analysis.verdict, "ridge (primary)"),
            (analysis.learned_forest, analysis.verdict_forest, "forest (comparison)"),
        ):
            if result is None or verdict is None:
                continue
            write(f"model {name}")
            write(
                f"  held-out duty pooled MAE: learned {result.learned_mae_c:.4f} degC, "
                f"RC on the same warm-up window {result.rc_mae_c:.4f} degC, margin "
                f"{result.margin_c:+.4f} degC"
            )
            write(f"  two-sigma coverage      : {result.coverage:.3f}")
            for split in result.splits:
                write(
                    f"    {split.held_out:<12s} n={split.n_rows:<3d} learned "
                    f"{split.learned_mae_c:.4f} degC   RC {split.rc_mae_c:.4f} degC   "
                    f"margin {split.margin_c:+.4f} degC"
                )
            for duty, reason in result.skipped:
                write(f"    SKIPPED duty {duty:g}: {reason}")
            write(f"  verdict                 : {verdict.verdict}")
            write(f"  {verdict.statement}")

    write("")
    write("-" * 78)
    write("STRUCTURALLY UNMEASURABLE in this campaign")
    write("-" * 78)
    for claim in limits.structurally_unmeasurable_axes():
        write(f"  {claim.key}: {claim.claim}")
        write(f"      reason: {claim.reason}")
    write(
        "  This axis is not approximated by the held-out duty-cycle split and is not "
        "dropped. It has no score."
    )

    write("")
    write("-" * 78)
    write("NOTES AND REFUSALS")
    write("-" * 78)
    if analysis.notes:
        for note in analysis.notes:
            write(f"  - {note}")
    else:
        write("  none")

    write("")
    write("-" * 78)
    write("UNMEASURED, BY NAME")
    write("-" * 78)
    for claim in limits.UNMEASURED_BY_NAME:
        write(f"  {claim.key:<18s} {claim.claim}")
    write("")
    write(
        "Research-grade. This software is not flight-qualified, not certified and not "
        "approved for operational aerospace use."
    )
    return "\n".join(out) + "\n"
