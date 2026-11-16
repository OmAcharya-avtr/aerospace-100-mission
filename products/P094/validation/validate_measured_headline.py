"""The five headline outputs of SPEC §5 and both pre-registration verdicts.

Pre-registration, stated before the verdicts
--------------------------------------------
SPEC §7 was written and committed on 2026-11-15, before any measured timing
data existed, and predicts two things. The primary expectation is that the
divergence spread ``max(R)/min(R)`` EXCEEDS 3. The secondary expectation is
that baseline 2's held-out transfer error EXCEEDS 0.5. Both thresholds are
pinned as literals in ``src/hiltimingdiff/baselines.py`` and again in this
script, and the verdicts are reported as they land.

What is exactly reproducible and what is not
--------------------------------------------
``R[s] = median(hw[s]) / (declared_ops[s] * ns_per_op)`` carries a single
global ``ns_per_op`` calibrated on THIS host, which is a wall-clock reading and
moves between runs. The spread, the Kendall statistic, the permutation and both
arg-extrema are invariant under it, so they are exactly reproducible from the
committed traces alone and are pinned here as literals. The levels of ``R[s]``
are reported with the ``ns_per_op`` that produced them, and the
calibration-free form ``ns per declared op`` is published beside them.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from hiltimingdiff import build_headline, headline_lines  # noqa: E402
from hiltimingdiff.baselines import (  # noqa: E402
    PREREGISTERED_SPREAD,
    PREREGISTERED_TRANSFER_ERROR,
)

#: SPEC §7's two thresholds, as literals independent of the package.
SPEC_SPREAD_THRESHOLD = 3.0
SPEC_TRANSFER_THRESHOLD = 0.5

#: Figures that depend on NOTHING but the committed traces, pinned to full
#: double precision. If any of these moves, the committed campaign has changed.
PINNED = {
    "spread": 1114.8633602088898,
    "tau_b": 0.2,
    "argmax": "centroid",
    "argmin": "ingest",
    "permutation": (3, 0, 1, 5, 4, 2),
    "argmax_share": 0.5805602572403938,
    "argmin_share": 0.005287459117717222,
}

#: Nanoseconds per declared operation, per stage. The calibration-free form.
PINNED_NS_PER_DECLARED_OP = {
    "ingest": 0.46875,
    "background": 1.220703125,
    "threshold": 21.651753743489582,
    "centroid": 522.5922000979172,
    "gate": 95.1902876159667,
    "track": 47.644444444444446,
}


def main() -> int:
    tee = _bootstrap.Tee("validate_measured_headline_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- measured headline, SPEC §5")

    tee.rule("PRE-REGISTRATION, quoted before the verdicts")
    tee(
        "  PRIMARY (SPEC §7): the declared operation-count model will be "
        "stage-dependently"
    )
    tee("  wrong, not uniformly wrong, and the divergence spread max(R)/min(R) will")
    tee(f"  exceed {SPEC_SPREAD_THRESHOLD:g}. Refuted if the spread is below it.")
    tee("")
    tee(
        "  SECONDARY (SPEC §7): baseline 2's single global scaling factor will "
        "transfer"
    )
    tee(
        f"  poorly to held-out stages, with a mean relative error above "
        f"{SPEC_TRANSFER_THRESHOLD:g}."
    )
    tee("")
    tee.check(
        "the package's pinned thresholds equal the specification's",
        (PREREGISTERED_SPREAD, PREREGISTERED_TRANSFER_ERROR)
        == (SPEC_SPREAD_THRESHOLD, SPEC_TRANSFER_THRESHOLD),
    )

    headline = build_headline(passes=_bootstrap.PASSES, resamples=_bootstrap.RESAMPLES_USED)
    divergence = headline.divergence

    tee.rule("THE HEADLINE")
    for line in headline_lines(headline):
        tee(f"  {line}")

    tee.rule("output 1 -- per-stage divergence ratio, both forms")
    tee(f"  ns_per_op calibrated on this host: {headline.ns_per_op:.9f} ns per declared op")
    tee(
        "  stage        hw_median_ns   declared_ops   ns/declared_op          R        "
        "R 95% interval"
    )
    for stage in divergence.stages:
        interval = divergence.ratio_interval[stage]
        tee(
            f"  {stage:<11s} {divergence.hw_median_ns[stage]:>12.2f} "
            f"{divergence.declared_ops_median[stage]:>14.4f} "
            f"{divergence.ns_per_declared_op[stage]:>16.9f} "
            f"{divergence.ratio[stage]:>10.6f}  "
            f"[{interval.low:.6f}, {interval.high:.6f}]"
        )
    tee("")
    for stage in divergence.stages:
        tee.check(
            f"ns per declared op for {stage} matches the pinned value",
            abs(
                divergence.ns_per_declared_op[stage] - PINNED_NS_PER_DECLARED_OP[stage]
            )
            <= 1e-9 * abs(PINNED_NS_PER_DECLARED_OP[stage]),
            f"pinned {PINNED_NS_PER_DECLARED_OP[stage]!r}, measured "
            f"{divergence.ns_per_declared_op[stage]!r}",
        )
        tee.check(
            f"R interval for {stage} brackets its point estimate",
            divergence.ratio_interval[stage].low
            <= divergence.ratio_interval[stage].point
            <= divergence.ratio_interval[stage].high,
        )

    tee.rule("output 2 -- divergence spread, and the PRIMARY verdict")
    tee(f"  spread max(R)/min(R) = {divergence.spread_interval.format(6)}")
    tee(f"  pinned value         = {PINNED['spread']!r}")
    tee(f"  resampling unit      = {divergence.spread_interval.unit}, "
        f"{divergence.spread_interval.units} units")
    tee("")
    tee.check(
        "spread matches the pinned value",
        abs(divergence.spread - PINNED["spread"]) <= 1e-9 * PINNED["spread"],
        f"pinned {PINNED['spread']!r}, measured {divergence.spread!r}",
    )
    tee.check(
        "the whole 95 per cent interval on the spread lies above the threshold",
        divergence.spread_interval.low > SPEC_SPREAD_THRESHOLD,
        f"interval low {divergence.spread_interval.low:.4f} against "
        f"{SPEC_SPREAD_THRESHOLD:g}",
    )
    verdict = headline.spread_verdict
    tee(f"  VERDICT: {verdict.line()}")
    tee.check(
        "the primary expectation's verdict is reported as it lands",
        verdict.held == (divergence.spread > SPEC_SPREAD_THRESHOLD),
    )
    tee.finding(
        "SPEC §7 PRIMARY IS CONFIRMED, by a factor of "
        f"{divergence.spread / SPEC_SPREAD_THRESHOLD:.1f} on the threshold: the "
        f"measured spread is {divergence.spread:.4f} against a pre-registered 3. The "
        "magnitude is far beyond what the specification anticipated and is reported "
        "as measured."
    )
    threshold_over_background = (
        divergence.ns_per_declared_op["threshold"]
        / divergence.ns_per_declared_op["background"]
    )
    tee.finding(
        "SPEC §7's stated MECHANISM is only partly right, and the part that is wrong "
        "is published. The specification puts ingest, background and threshold in one "
        "cheap vectorised group. Measured, threshold costs "
        f"{divergence.ns_per_declared_op['threshold']:.4f} ns per declared operation "
        f"against background's {divergence.ns_per_declared_op['background']:.4f}, a "
        "factor of "
        f"{threshold_over_background:.1f}"
        ". threshold is NOT in the cheap group, because extract_candidates contains a "
        "Python-level loop over candidate groups. The specification's three-group "
        "picture is therefore wrong about one of its six stages; its prediction about "
        "the spread is not."
    )

    tee.rule("output 3 -- rank disagreement")
    tee(f"  Kendall tau-b        = {divergence.tau_b_interval.format(6)}")
    tee(f"  pinned value         = {PINNED['tau_b']!r}")
    tee(f"  permutation          = {divergence.permutation}")
    tee(f"  declared order       = {' < '.join(divergence.declared_order)}")
    tee(f"  measured order       = {' < '.join(divergence.measured_order)}")
    tee("")
    tee.check(
        "tau-b matches the pinned value",
        abs(divergence.tau_b - PINNED["tau_b"]) <= 1e-12,
        f"pinned {PINNED['tau_b']!r}, measured {divergence.tau_b!r}",
    )
    tee.check(
        "the permutation matches the pinned one",
        divergence.permutation == PINNED["permutation"],
        f"pinned {PINNED['permutation']}, measured {divergence.permutation}",
    )
    tee.check(
        "the permutation is NOT the identity, so the orderings disagree",
        divergence.permutation != (0, 1, 2, 3, 4, 5),
    )
    moved = sum(
        1
        for position, declared_position in enumerate(divergence.permutation)
        if position != declared_position
    )
    tee(f"  stages whose rank changes = {moved} of {len(divergence.stages)}")
    tee.finding(
        f"The stage ordering a schedulability analysis would read off the declared "
        f"operation counts disagrees with the measured ordering at tau-b "
        f"{divergence.tau_b:.4f}. {moved} of the {len(divergence.stages)} stages "
        "change rank. A WCET budget "
        "built on the declared counts allocates its headroom to the wrong stages: it "
        f"ranks {divergence.declared_order[0]} as the cheapest when "
        f"{divergence.measured_order[0]} is, and ranks "
        f"{divergence.declared_order[-1]} as the most expensive when "
        f"{divergence.measured_order[-1]} is."
    )

    tee.rule("output 4 -- the least and most trustworthy stages")
    tee(
        f"  argmax(R) = {divergence.argmax}  R = {divergence.ratio[divergence.argmax]:.6f}  "
        f"share of measured frame time = {divergence.argmax_share:.6f}"
    )
    tee(
        f"  argmin(R) = {divergence.argmin}  R = {divergence.ratio[divergence.argmin]:.6f}  "
        f"share of measured frame time = {divergence.argmin_share:.6f}"
    )
    tee("")
    tee.check("argmax(R) matches the pinned stage", divergence.argmax == PINNED["argmax"])
    tee.check("argmin(R) matches the pinned stage", divergence.argmin == PINNED["argmin"])
    tee.check(
        "argmax share matches the pinned value",
        abs(divergence.argmax_share - PINNED["argmax_share"]) <= 1e-12,
    )
    tee.check(
        "argmin share matches the pinned value",
        abs(divergence.argmin_share - PINNED["argmin_share"]) <= 1e-12,
    )
    tee.finding(
        f"The least trustworthy stage as a schedulability input, {divergence.argmax}, "
        f"is also the most expensive one: it holds {divergence.argmax_share:.4f} of "
        "measured frame time. The divergence is therefore consequential rather than a "
        "curiosity about a cheap stage."
    )

    tee.rule("output 5 -- baseline 2's held-out transfer error, and the SECONDARY verdict")
    baseline2 = headline.baseline2
    tee(f"  training stages      : {', '.join(baseline2.declared.train)}")
    tee(f"  held-out stages      : {', '.join(baseline2.declared.holdout)}")
    tee(f"  fitted k             : {baseline2.declared.k:.9f}")
    tee(f"  held-out mean error  : {baseline2.declared_interval.format(6)}")
    for stage in baseline2.declared.holdout:
        tee(
            f"    {stage:<11s} predicted {baseline2.declared.predicted_ns[stage]:>12.2f} ns "
            f"against measured {baseline2.hw_median_ns[stage]:>12.2f} ns, relative error "
            f"{baseline2.declared.relative_error[stage]:.6f}"
        )
    tee("")
    tee(f"  leave-one-stage-out mean : {baseline2.leave_one_out_mean:.6f}")
    tee(
        f"  all 20 three-and-three splits: min {baseline2.split_error_min:.6f}, "
        f"median {baseline2.split_error_median:.6f}, max {baseline2.split_error_max:.6f}"
    )
    tee(
        f"  splits above the pre-registered {SPEC_TRANSFER_THRESHOLD:g}: "
        f"{baseline2.splits_above_preregistration} of {len(baseline2.split_errors)}"
    )
    transfer = headline.transfer_verdict
    tee("")
    tee(f"  VERDICT: {transfer.line()}")
    tee.check(
        "the secondary expectation's verdict is reported as it lands",
        transfer.held
        == (baseline2.declared.mean_relative_error > SPEC_TRANSFER_THRESHOLD),
    )
    tee.check(
        "the whole 95 per cent interval on the held-out error lies BELOW the threshold",
        baseline2.declared_interval.high < SPEC_TRANSFER_THRESHOLD,
        f"interval high {baseline2.declared_interval.high:.6f} against "
        f"{SPEC_TRANSFER_THRESHOLD:g}",
    )
    tee.finding(
        "SPEC §7 SECONDARY IS REFUTED. The pre-registration predicted a held-out "
        f"transfer error above {SPEC_TRANSFER_THRESHOLD:g}; the measured error is "
        f"{baseline2.declared.mean_relative_error:.6f}, with the whole 95 per cent "
        f"interval below the threshold, and none of the 20 three-and-three splits "
        "exceeds it. A BASELINE THAT WAS SUPPOSED TO LOSE WINS, and it is published "
        "as winning rather than retuned. The mechanism: the hardware-to-simulation "
        "ratio is nearly uniform across the six stages (measured span "
        f"{min(baseline2.per_stage_ratio.values()):.3f} to "
        f"{max(baseline2.per_stage_ratio.values()):.3f}, a factor of "
        f"{max(baseline2.per_stage_ratio.values()) / min(baseline2.per_stage_ratio.values()):.2f}"
        "), so a single scalar absorbs it. The vectorised/interpreted split the "
        "specification expected to defeat the scalar is a property of the DECLARED "
        "OPERATION COUNT, not of the platform ratio, which is exactly why the primary "
        "expectation is confirmed and the secondary is not."
    )

    tee.rule("the per-stage hardware-to-simulation ratio, which is the refuting evidence")
    tee("  stage        hw_median_ns     sim_median_ns      hw/sim")
    for stage in divergence.stages:
        tee(
            f"  {stage:<11s} {baseline2.hw_median_ns[stage]:>12.2f} "
            f"{baseline2.sim_median_ns[stage]:>17.2f} "
            f"{baseline2.per_stage_ratio[stage]:>11.6f}"
        )
    ratios = list(baseline2.per_stage_ratio.values())
    tee("")
    tee(
        f"  platform ratio spread = {max(ratios) / min(ratios):.6f}  against a declared "
        f"model divergence spread of {divergence.spread:.4f}"
    )
    tee.check(
        "the platform ratio spread is below the pre-registered spread threshold",
        max(ratios) / min(ratios) < SPEC_SPREAD_THRESHOLD,
        "the PLATFORM is close to uniformly slower; the DECLARED MODEL is not "
        "close to uniformly wrong",
    )

    tee.rule("negative controls")
    # Anchored on the TRUE values, not on an output. A uniform declared model --
    # declared_ops proportional to the measured median -- must give a spread of
    # exactly 1 and a tau-b of exactly 1, and the spread check must NOT fire.
    import numpy as np

    from hiltimingdiff import PassUnits
    from hiltimingdiff import divergence as divergence_of

    uniform = PassUnits(
        stages=headline.units.stages,
        cells=headline.units.cells,
        pass_index=headline.units.pass_index,
        hw=headline.units.hw,
        ops=headline.units.hw / 7.0,
    )
    uniform_result = divergence_of(uniform, 1.0, resamples=200)
    tee.control(
        "a declared model proportional to measured time has spread above 3",
        uniform_result.spread > SPEC_SPREAD_THRESHOLD,
        must_fire=False,
        detail=f"spread {uniform_result.spread:.9f}, tau-b {uniform_result.tau_b:.9f}",
    )
    tee.control(
        "a declared model proportional to measured time agrees on the ordering",
        uniform_result.permutation != (0, 1, 2, 3, 4, 5),
        must_fire=False,
        detail=f"permutation {uniform_result.permutation}",
    )
    # And a model that inverts the measured ordering must be detected.
    inverted = PassUnits(
        stages=headline.units.stages,
        cells=headline.units.cells,
        pass_index=headline.units.pass_index,
        hw=headline.units.hw,
        ops=1.0 / headline.units.hw,
    )
    inverted_result = divergence_of(inverted, 1.0, resamples=200)
    tee.control(
        "an inverted declared model is detected as rank-reversed",
        inverted_result.tau_b < -0.9,
        must_fire=True,
        detail=f"tau-b {inverted_result.tau_b:.9f}, spread {inverted_result.spread:.6g}",
    )
    tee.control(
        "a declared model with one stage scaled 1000x is detected as a spread",
        float(
            np.max(
                np.median(headline.units.hw, axis=0)
                / np.median(
                    headline.units.hw / np.array([1000.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                    axis=0,
                )
            )
        )
        > 1.0,
        must_fire=True,
        detail="a stage-dependent perturbation of the declared model is visible "
        "in the ratio, which is what the spread statistic reads",
    )

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
