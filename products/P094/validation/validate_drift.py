"""Board drift, fitted over eleven blocks, and how much of each R[s] could be it.

The board got 7.52 per cent slower while the campaign ran. P093 measured its
floor once, at the start, under the highest foreign load of its session, and a
published bound was falsified by exactly this effect. This script does three
things about it.

1. Fits the drift over elapsed campaign time across all eleven calibration
   blocks, and reports how badly a straight line describes it.
2. Reports, for every stage ratio R[s], the RAW MEASURED figure as the headline
   and a DRIFT-ADJUSTED variant beside it, each labelled.
3. Checks the randomisation claim of SPEC §8 rather than repeating it:
   regresses each cell's median total frame time on ``order_index`` and on
   ``W`` separately, and reports the correlation between ``order_index`` and
   ``W`` itself. That last number does NOT support the claim as strongly as
   SPEC §8 states it, and the measurement is published as it is.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from hiltimingdiff import (  # noqa: E402
    calibrate_ns_per_op,
    cell_drift_factors,
    confounding,
    drift_comparison,
    load_campaign,
    measure_simulation,
    pass_units,
)
from hiltimingdiff.drift import spearman_rho  # noqa: E402
from hiltimingdiff.traces import CALIBRATION_BLOCKS  # noqa: E402

#: Pinned: the drift ratio ``meta.json`` recorded, to full double precision.
PINNED_DRIFT_RATIO = 1.0752408447278534

#: Pinned: the straight-line fit over the eleven blocks.
PINNED_FITTED_RATIO = 1.0682256407994783
PINNED_FIT_R_SQUARED = 0.7268610650663789

#: Pinned: the confounding check.
PINNED_ORDER_R_SQUARED = 0.23312847578756135
PINNED_WIDTH_R_SQUARED = 0.9824296874606212
PINNED_ORDER_WIDTH_PEARSON = 0.5011614417507341
#: Spearman's rho on TIE-AVERAGED MIDRANKS, cross-checked against
#: ``scipy.stats.spearmanr``. The superseded value is below, kept as a literal
#: so a regression to the ordinal-rank form is caught rather than merely
#: differing from a number nobody remembers.
PINNED_ORDER_WIDTH_SPEARMAN = 0.5809475019311126

#: The WRONG value this product published before 2026-11-16, from
#: ``argsort(argsort(x))`` over cells in alphabetical order. The same ordinal
#: form returns 0.85 over the same cells in execution order, which is what makes
#: it not a rank statistic at all.
SUPERSEDED_ORDINAL_SPEARMAN = 0.5333333333333333
SUPERSEDED_ORDINAL_SPEARMAN_IN_EXECUTION_ORDER = 0.85


def main() -> int:
    tee = _bootstrap.Tee("validate_drift_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- board drift, raw against adjusted")

    campaign = load_campaign()
    units = pass_units(campaign)
    simulation = measure_simulation(campaign, passes=_bootstrap.PASSES)
    ns_per_op = calibrate_ns_per_op(simulation)
    comparison = drift_comparison(
        campaign, units, ns_per_op, resamples=_bootstrap.RESAMPLES_USED
    )
    drift = comparison.drift

    tee.rule("the calibration series: ELEVEN blocks, not the ten SPEC §8 states")
    tee(
        "  The reference is a fixed 160x160 float64 matmul, data-independent, sharing "
        "no code with"
    )
    tee(
        "  the pipeline. The campaign program emits an explicit leading block, one "
        "before each of"
    )
    tee(
        "  the nine cells, and one after the last: eleven. SPEC §8 says ten. The "
        "discrepancy is"
    )
    tee("  reported, not reconciled by dropping a block.")
    tee("")
    tee(drift.blocks.to_string(index=False))
    tee("")
    tee.check(
        "the committed calibration table holds exactly the eleven pinned blocks",
        tuple(drift.blocks["which"]) == CALIBRATION_BLOCKS,
    )
    tee.check(
        "every block holds 2000 repetitions",
        set(drift.blocks["reps"].unique()) == {2000},
    )
    tee.finding(
        "SPEC §8 and the dispatch brief both say TEN calibration blocks. The committed "
        "calibration.csv holds ELEVEN (floor_start, floor_00..floor_08, floor_end), "
        "22000 rows at 2000 repetitions each. The specification's count is one low. "
        "Nothing in the analysis depends on the count, and the discrepancy is recorded "
        "rather than silently corrected."
    )

    tee.rule("the fit, and how badly a straight line describes this drift")
    tee(f"  start median (floor_start) : {drift.start_median_ns:.1f} ns")
    tee(f"  end median (floor_end)     : {drift.end_median_ns:.1f} ns")
    tee(f"  bracket ratio end/start    : {drift.bracket_ratio:.16f}")
    tee(f"  meta.json drift ratio      : {drift.meta_ratio:.16f}")
    tee(f"  campaign elapsed           : {drift.campaign_elapsed_s:.4f} s")
    tee(f"  fitted slope               : {drift.fit.slope:.9f} ns per second")
    tee(f"  fitted intercept           : {drift.fit.intercept:.6f} ns")
    tee(f"  fitted ratio over the run  : {drift.fitted_ratio:.16f}")
    tee(f"  straight-line r_squared    : {drift.fit.r_squared:.9f}")
    tee(f"  adjacent steps increasing  : {drift.monotonic_blocks} of 10")
    tee(f"  largest single block step  : {drift.max_block_step_ratio:.9f}")
    tee("")
    tee.check(
        "the bracket ratio matches meta.json to full precision",
        abs(drift.bracket_ratio - drift.meta_ratio) <= 1e-15 * drift.meta_ratio,
    )
    tee.check(
        "the bracket ratio matches the pinned value",
        abs(drift.bracket_ratio - PINNED_DRIFT_RATIO) <= 1e-15 * PINNED_DRIFT_RATIO,
    )
    tee.check(
        "the fitted ratio matches the pinned value",
        abs(drift.fitted_ratio - PINNED_FITTED_RATIO) <= 1e-9 * PINNED_FITTED_RATIO,
    )
    tee.check(
        "the straight-line r_squared matches the pinned value",
        abs(drift.fit.r_squared - PINNED_FIT_R_SQUARED) <= 1e-9,
    )
    tee.finding(
        "The drift is STEP-LIKE, not linear, and the linear fit is published with that "
        f"stated. A straight line explains {drift.fit.r_squared:.4f} of the "
        "calibration reference's variation; only "
        f"{drift.monotonic_blocks} of the 10 adjacent block-to-block steps are "
        f"increases; and the largest single step, {drift.max_block_step_ratio:.4f}, is "
        f"{(drift.max_block_step_ratio - 1.0) / (drift.bracket_ratio - 1.0):.0%} of the "
        "whole campaign's drift. Fitting a line is therefore a smoothing of a staircase "
        f"and under-reads the bracket: {drift.fitted_ratio:.4f} fitted against "
        f"{drift.bracket_ratio:.4f} measured end to start. Both are reported."
    )

    tee.rule("per-cell fitted drift factors, normalised to 1.0 at the campaign start")
    factors = cell_drift_factors(campaign, drift)
    for cell in sorted(factors, key=campaign.order_index):
        tee(
            f"  order {campaign.order_index(cell)}  W={campaign.cell_width(cell):>3d}  "
            f"{cell:<18s} factor {factors[cell]:.9f}"
        )
    tee("")
    tee.check(
        "every factor exceeds 1.0, so every cell is adjusted downwards",
        min(factors.values()) > 1.0,
    )
    tee.check(
        "the factors increase with execution order",
        [factors[cell] for cell in sorted(factors, key=campaign.order_index)]
        == sorted(factors.values()),
    )

    tee.rule("RAW MEASURED (the headline) beside the DRIFT-ADJUSTED variant")
    tee(comparison.frame().to_string(index=False))
    tee("")
    tee(f"  raw measured spread       : {comparison.raw.spread:.9f}   <-- HEADLINE")
    tee(f"  drift-adjusted spread     : {comparison.adjusted.spread:.9f}")
    tee(f"  relative shift in spread  : {comparison.spread_relative_shift:+.9f}")
    tee(f"  largest per-stage shift   : {comparison.max_abs_ratio_shift:.9f}")
    tee("")
    tee.check(
        "no stage's R moves by more than 5 per cent under the adjustment",
        comparison.max_abs_ratio_shift < 0.05,
        f"largest shift {comparison.max_abs_ratio_shift:.6f}",
    )
    tee.check(
        "the spread stays far above the pre-registered 3 under the adjustment",
        comparison.adjusted.spread > 3.0,
        f"adjusted spread {comparison.adjusted.spread:.4f}",
    )
    tee.check(
        "the permutation is unchanged by the adjustment",
        comparison.adjusted.permutation == comparison.raw.permutation,
        f"{comparison.raw.permutation}",
    )
    tee.finding(
        "HOW MUCH OF EACH R[s] COULD BE DRIFT: at most "
        f"{comparison.max_abs_ratio_shift:.4f} in relative terms, and that is an upper "
        "bound on the whole adjustment rather than on the uncertainty. The adjustment "
        "is a PER-CELL multiplicative factor, so it is stage-independent by "
        "construction: it can move the level of an R[s] and it cannot move the ratio "
        "between two stages of the same cell. Drift therefore cannot explain the "
        f"measured spread of {comparison.raw.spread:.2f}, and the adjusted spread is "
        f"{comparison.adjusted.spread:.2f} -- slightly LARGER, because removing a "
        "per-cell factor re-weights which cells dominate the pooled medians."
    )

    tee.rule("structural check: the adjustment cannot change a within-cell stage ratio")
    from hiltimingdiff import drift_adjusted_units

    adjusted_units = drift_adjusted_units(units, factors)
    raw_pairs = units.hw[:, 3] / units.hw[:, 0]
    adjusted_pairs = adjusted_units.hw[:, 3] / adjusted_units.hw[:, 0]
    worst = float(np.max(np.abs(adjusted_pairs / raw_pairs - 1.0)))
    tee(f"  largest relative change in the centroid/ingest ratio: {worst:.3e}")
    tee.check(
        "the within-cell centroid/ingest ratio is unchanged to rounding",
        worst < 1e-12,
        "agreeing to within rounding, which is what is guaranteed; exact in exact "
        "arithmetic",
    )

    tee.rule("CONFOUNDING: is elapsed time confounded with frame dimension")
    confound = confounding(campaign, units)
    tee("  cell median total frame time, regressed SEPARATELY on each predictor:")
    tee(
        f"    on order_index : slope {confound.on_order.slope:>12.3f} ns/position, "
        f"r_squared {confound.on_order.r_squared:.9f}"
    )
    tee(
        f"    on W           : slope {confound.on_width.slope:>12.3f} ns/pixel,    "
        f"r_squared {confound.on_width.r_squared:.9f}"
    )
    tee("")
    tee(f"    corr(order_index, W) Pearson  : {confound.order_width_correlation:.16f}")
    tee(
        f"    corr(order_index, W) Spearman : {confound.order_width_spearman:.16f}  "
        "(tie-averaged midranks)"
    )
    tee("")
    tee("  cell          order  W   median total ns")
    for index in range(len(confound.cells)):
        tee(
            f"  {confound.cells[index]:<18s} {int(confound.order_index[index]):>2d} "
            f"{int(confound.width[index]):>3d}  {confound.median_total_ns[index]:>15.2f}"
        )
    tee("")
    tee.check(
        "frame dimension explains almost all between-cell variance",
        confound.on_width.r_squared > 0.95,
        f"r_squared {confound.on_width.r_squared:.6f}",
    )
    tee.check(
        "execution order explains far less than frame dimension",
        confound.on_order.r_squared < confound.on_width.r_squared,
        f"{confound.on_order.r_squared:.6f} against {confound.on_width.r_squared:.6f}",
    )
    tee.check(
        "the confounding figures match their pinned values",
        abs(confound.on_order.r_squared - PINNED_ORDER_R_SQUARED) <= 1e-9
        and abs(confound.on_width.r_squared - PINNED_WIDTH_R_SQUARED) <= 1e-9
        and abs(confound.order_width_correlation - PINNED_ORDER_WIDTH_PEARSON) <= 1e-12
        and abs(confound.order_width_spearman - PINNED_ORDER_WIDTH_SPEARMAN) <= 1e-15,
    )
    tee.rule("a DEFECT in this product's own Spearman, found and corrected")
    tee(
        "  Found by the coordinator's independent check on 2026-11-16, not by this "
        "build."
    )
    tee("")
    tee(
        f"    published before the fix : {SUPERSEDED_ORDINAL_SPEARMAN!r}   WRONG"
    )
    tee(f"    corrected value         : {confound.order_width_spearman!r}")
    tee("")
    tee(
        "  MECHANISM. The original implementation ranked with "
        "numpy.argsort(numpy.argsort(x)),"
    )
    tee(
        "  which is an ORDINAL rank: it breaks ties by position in the array. W takes "
        "three"
    )
    tee(
        "  values over the nine cells -- two 48s, five 64s and two 96s -- so the five "
        "64s"
    )
    tee(
        "  received five distinct ranks instead of five copies of their midrank. "
        "Spearman's"
    )
    tee(
        "  rho is defined on midranks, so the published figure was a Pearson "
        "correlation on"
    )
    tee("  an arbitrary ordering and not Spearman's rho.")
    tee("")
    tee(
        "  It was not even well defined. The ordinal form is not invariant under a "
        "permutation"
    )
    tee(
        f"  of the rows: on this campaign it returns "
        f"{SUPERSEDED_ORDINAL_SPEARMAN_IN_EXECUTION_ORDER!r} with the cells in "
        "EXECUTION order"
    )
    tee(
        f"  and {SUPERSEDED_ORDINAL_SPEARMAN!r} with them in the alphabetical order "
        "this"
    )
    tee(
        "  function's groupby happens to produce, from identical data. The midrank "
        "form"
    )
    tee(f"  returns {confound.order_width_spearman!r} either way.")
    tee("")
    tee(
        "  The Pearson figure was unaffected and is unchanged, which is what made the "
        "defect"
    )
    tee("  diagnosable: the disagreement was isolated to the rank treatment.")
    tee("")
    # Reproduce the defect here, from first principles, so the correction is a
    # measurement in this output rather than a claim about one.
    ordinal_alphabetical = float(
        np.corrcoef(
            np.argsort(np.argsort(confound.order_index)).astype(np.float64),
            np.argsort(np.argsort(confound.width)).astype(np.float64),
        )[0, 1]
    )
    execution = np.argsort(confound.order_index, kind="stable")
    ordinal_execution = float(
        np.corrcoef(
            np.argsort(np.argsort(confound.order_index[execution])).astype(np.float64),
            np.argsort(np.argsort(confound.width[execution])).astype(np.float64),
        )[0, 1]
    )
    tee.check(
        "the superseded ordinal form is reproduced exactly, in both row orders",
        abs(ordinal_alphabetical - SUPERSEDED_ORDINAL_SPEARMAN) <= 1e-15
        and abs(ordinal_execution - SUPERSEDED_ORDINAL_SPEARMAN_IN_EXECUTION_ORDER)
        <= 1e-15,
        f"alphabetical {ordinal_alphabetical!r}, execution {ordinal_execution!r}",
    )
    tee.check(
        "the corrected midrank form IS invariant under that same reordering",
        abs(
            spearman_rho(confound.order_index, confound.width)
            - spearman_rho(confound.order_index[execution], confound.width[execution])
        )
        <= 1e-15,
        "a rank statistic must not depend on the order of the observations",
    )
    tee.check(
        "the corrected Spearman matches the pinned value",
        abs(confound.order_width_spearman - PINNED_ORDER_WIDTH_SPEARMAN) <= 1e-15,
        f"pinned {PINNED_ORDER_WIDTH_SPEARMAN!r}, measured "
        f"{confound.order_width_spearman!r}",
    )
    tee.check(
        "the corrected Spearman is FURTHER from the < 0.2 the SPEC §8 claim needs",
        confound.order_width_spearman > SUPERSEDED_ORDINAL_SPEARMAN,
        "the finding is unchanged in direction and STRENGTHENED in magnitude by the "
        "correction; nothing was retuned to recover the old number",
    )
    tee.finding(
        "A DEFECT IN THIS PRODUCT, found by the coordinator's independent check and "
        f"corrected. The published Spearman rho was {SUPERSEDED_ORDINAL_SPEARMAN!r}, "
        "computed with ordinal ranks from argsort(argsort(x)) on a series with five "
        f"tied values. The correct tie-averaged-midrank value is "
        f"{confound.order_width_spearman!r}, which agrees with scipy.stats.spearmanr "
        "to 1e-12. The Pearson figure was unaffected. THE CONCLUSION IS UNCHANGED IN "
        "DIRECTION AND STRENGTHENED IN MAGNITUDE: the corrected rho is further from "
        "the < 0.2 that SPEC §8's decorrelation claim needs, not closer to it."
    )

    tee.rule("the falsified claim, restated on the corrected figures")
    tee.falsified(
        "order_index and W are uncorrelated, as SPEC §8's randomisation intended",
        f"measured Pearson {confound.order_width_correlation:.16f} and Spearman "
        f"{confound.order_width_spearman:.16f} (midranks), against the < 0.2 the "
        "claim needs",
        abs(confound.order_width_correlation - PINNED_ORDER_WIDTH_PEARSON) <= 1e-12
        and abs(confound.order_width_spearman - PINNED_ORDER_WIDTH_SPEARMAN) <= 1e-15,
        f"pinned Pearson {PINNED_ORDER_WIDTH_PEARSON!r} and Spearman "
        f"{PINNED_ORDER_WIDTH_SPEARMAN!r}; this script exits 0 while the measurement "
        "is exactly these two numbers and non-zero the moment either moves",
    )
    tee.finding(
        "A CHECK THAT FAILS, published as failing. SPEC §8 randomised the cell order "
        "so that elapsed time would not be confounded with frame dimension, and the "
        "brief states that drift is therefore NOT confounded with frame dimension. "
        f"Measured, corr(order_index, W) is {confound.order_width_correlation:.4f} "
        f"Pearson and {confound.order_width_spearman:.4f} Spearman on midranks. That "
        "is not near "
        "zero. With nine cells and one draw under seed 20261115, the randomisation "
        "removed P093's SYSTEMATIC ascending-dimension ordering but left a substantial "
        "positive association in this particular draw. The claim as stated is "
        "therefore only partly supported, and the honest version is: dimension "
        f"dominates the between-cell variance ({confound.on_width.r_squared:.4f} "
        f"against order's {confound.on_order.r_squared:.4f}), and the drift's total "
        f"magnitude of {drift.bracket_ratio:.4f} is three orders of magnitude too "
        "small to produce the measured spread, so the headline does not rest on the "
        "decorrelation -- but the decorrelation itself was not achieved."
    )

    tee.rule("negative controls")
    # Anchored on the TRUE per-cell factors. An adjustment by the INVERSE factors
    # must move R in the opposite direction by a comparable amount.
    inverse = {cell: 1.0 / value for cell, value in factors.items()}
    inverse_units = drift_adjusted_units(units, inverse)
    from hiltimingdiff import divergence as divergence_of

    inverse_result = divergence_of(inverse_units, ns_per_op, resamples=200)
    tee.control(
        "adjusting by the INVERSE drift factors raises every R",
        all(
            inverse_result.ratio[stage] > comparison.raw.ratio[stage]
            for stage in units.stages
        ),
        must_fire=True,
        detail="an adjustment in the wrong direction is visible in every stage",
    )
    unit_factors = dict.fromkeys(factors, 1.0)
    unit_result = divergence_of(
        drift_adjusted_units(units, unit_factors), ns_per_op, resamples=200
    )
    tee.control(
        "adjusting by a factor of exactly 1.0 changes the spread",
        abs(unit_result.spread - comparison.raw.spread) > 1e-9,
        must_fire=False,
        detail=f"spread {unit_result.spread:.9f} against raw {comparison.raw.spread:.9f}",
    )
    # A control on the confounding regression: a deliberately order-ordered
    # campaign must show a much higher order r_squared.
    sorted_by_width = np.argsort(confound.width, kind="stable").astype(np.float64)
    from hiltimingdiff import linear_fit

    contrived = linear_fit(
        sorted_by_width,
        confound.median_total_ns[np.argsort(confound.width, kind="stable")],
        x_label="width_sorted_position",
        y_label="cell_median_total_ns",
    )
    tee.control(
        "an ascending-dimension ordering shows a much stronger order association",
        contrived.r_squared > confound.on_order.r_squared + 0.3,
        must_fire=True,
        detail=f"contrived r_squared {contrived.r_squared:.6f} against the measured "
        f"randomised order's {confound.on_order.r_squared:.6f}; this is what P093's "
        "ordering looked like",
    )

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
