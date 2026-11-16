"""Example 3: the board drift and the unpinned clock, both measured.

Two things this product must handle honestly, side by side. The left panel is
the drift reference across all eleven calibration blocks with the straight-line
fit through it, so a reader can see that the drift is step-like rather than
linear. The right panel is the raw measured divergence ratio beside the
drift-adjusted variant, with the raw figure as the headline, and the measured
frequency co-variation printed beneath.

Writes ``screenshots/drift_and_clock.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from hiltimingdiff import (  # noqa: E402
    calibrate_ns_per_op,
    confounding,
    drift_comparison,
    frequency_covariation,
    load_campaign,
    measure_simulation,
    pass_units,
)

PASSES = 4


def main() -> int:
    tee = _common.Tee("drift_and_clock")
    _common.scope_header(
        tee, "P094 HilTimingDiff -- example 3: board drift and the unpinned clock"
    )

    campaign = load_campaign()
    units = pass_units(campaign)
    simulation = measure_simulation(campaign, passes=PASSES)
    ns_per_op = calibrate_ns_per_op(simulation)
    comparison = drift_comparison(campaign, units, ns_per_op, resamples=2000)
    drift = comparison.drift
    covariation = frequency_covariation(campaign, units)
    confound = confounding(campaign, units)

    tee("BOARD DRIFT, fitted over eleven calibration blocks")
    tee(drift.blocks.to_string(index=False))
    tee("")
    tee(f"  bracket ratio end/start : {drift.bracket_ratio:.16f}")
    tee(f"  fitted ratio            : {drift.fitted_ratio:.16f}")
    tee(f"  straight-line r_squared : {drift.fit.r_squared:.9f}")
    tee(f"  adjacent steps rising   : {drift.monotonic_blocks} of 10")
    tee(f"  largest single step     : {drift.max_block_step_ratio:.9f}")
    tee("")

    tee("RAW MEASURED (the headline) beside the DRIFT-ADJUSTED variant")
    tee(comparison.frame().to_string(index=False))
    tee(f"  raw measured spread   : {comparison.raw.spread:.6f}   <-- HEADLINE")
    tee(f"  drift-adjusted spread : {comparison.adjusted.spread:.6f}")
    tee(f"  relative shift        : {comparison.spread_relative_shift:+.6f}")
    tee("")

    tee("THE UNPINNED CLOCK, measured co-variation")
    tee(f"  governor {covariation.governor}, power mode {covariation.power_mode}")
    tee(f"  passwordless sudo available: {covariation.sudo_passwordless}")
    tee(
        f"  per-core frequency {covariation.frequency_min_khz:.0f} to "
        f"{covariation.frequency_max_khz:.0f} kHz"
    )
    tee("  stage        pooled r2   within-cell r2")
    for stage in covariation.stages:
        item = covariation.per_stage[stage]
        tee(
            f"  {stage:<11s} {item.pooled_variance_share:>10.6f} "
            f"{item.within_cell_variance_share:>16.6f}"
        )
    tee("")
    tee("CONFOUNDING CHECK on the randomised cell order")
    tee(
        f"  cell median total on order_index : r_squared "
        f"{confound.on_order.r_squared:.6f}"
    )
    tee(f"  cell median total on W           : r_squared {confound.on_width.r_squared:.6f}")
    tee(
        f"  corr(order_index, W)             : Pearson "
        f"{confound.order_width_correlation:.6f}, Spearman "
        f"{confound.order_width_spearman:.6f}"
    )
    tee(
        "  The Pearson figure is NOT near zero. SPEC §8's randomisation removed the "
        "systematic"
    )
    tee(
        "  ascending-dimension ordering of P093; it did not make order and dimension "
        "orthogonal"
    )
    tee("  in this draw, and that is published rather than asserted away.")

    pyplot = _common.style()
    figure, axes = pyplot.subplots(1, 2, figsize=(12.6, 6.0))

    left = axes[0]
    elapsed = drift.blocks["elapsed_s"].to_numpy(dtype=np.float64)
    medians = drift.blocks["median_ns"].to_numpy(dtype=np.float64)
    left.plot(
        elapsed,
        medians,
        marker="o",
        color=_common.HARDWARE,
        linewidth=1.6,
        markersize=6,
        label="calibration block median (160x160 float64 matmul)",
    )
    grid = np.linspace(0.0, float(drift.campaign_elapsed_s), 100)
    left.plot(
        grid,
        drift.fit.predict(grid),
        color=_common.DECLARED,
        linewidth=1.8,
        linestyle="--",
        label=f"straight-line fit, r_squared {drift.fit.r_squared:.4f}",
    )
    for index, label in enumerate(drift.blocks["which"]):
        if label in ("floor_start", "floor_end"):
            left.annotate(
                label,
                (elapsed[index], medians[index]),
                xytext=(14 if label == "floor_start" else -14, 6),
                textcoords="offset points",
                ha="left" if label == "floor_start" else "right",
                fontsize=9,
                color=_common.NEUTRAL,
            )
    left.set_xlabel("elapsed campaign time (s)")
    left.set_ylabel("calibration reference median (ns)")
    left.set_title(
        "Board drift, eleven blocks, fitted not bracketed\n"
        f"end/start = {drift.bracket_ratio:.6f}, fitted = {drift.fitted_ratio:.6f}, "
        f"{drift.monotonic_blocks} of 10 steps rising"
    )
    left.legend(loc="lower right", fontsize=8.5)

    right = axes[1]
    stages = list(comparison.raw.stages)
    positions = np.arange(len(stages))
    raw = np.array([comparison.raw.ratio[stage] for stage in stages])
    adjusted = np.array([comparison.adjusted.ratio[stage] for stage in stages])
    width = 0.38
    right.bar(
        positions - width / 2,
        raw,
        width,
        label="raw measured (headline)",
        color=_common.HARDWARE,
        edgecolor=_common.INK,
        linewidth=0.6,
    )
    right.bar(
        positions + width / 2,
        adjusted,
        width,
        label="drift-adjusted variant",
        color=_common.SIMULATION,
        edgecolor=_common.INK,
        linewidth=0.6,
    )
    right.set_yscale("log")
    right.set_xticks(positions)
    right.set_xticklabels(stages, rotation=20, ha="right")
    right.set_ylabel("R[s] = measured / declared (log scale)")
    shifts = comparison.ratio_shift()
    right.set_title(
        "How much of each R[s] could be drift\n"
        f"largest shift {comparison.max_abs_ratio_shift:.4f}; spread "
        f"{comparison.raw.spread:.0f} raw, {comparison.adjusted.spread:.0f} adjusted"
    )
    for index, stage in enumerate(stages):
        right.annotate(
            f"{shifts[stage]:+.3f}",
            (positions[index], max(raw[index], adjusted[index])),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=_common.NEUTRAL,
        )
    right.legend(loc="upper left", fontsize=8.5)

    figure.suptitle(
        "P094 HilTimingDiff -- the two things that could be measurement artefacts, "
        "measured",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "UNPINNED schedutil clock in MAXN_SUPER; passwordless sudo unavailable on the "
        "board. Both bar series carry the same host-calibrated ns_per_op, so their "
        "HEIGHTS move between runs\nwhile the relative shift printed above each pair "
        "does not.\nMeasured frequency co-variation: at most "
        f"{covariation.max_pooled_share:.6f} of per-stage variance pooled and "
        f"{covariation.max_within_cell_share:.6f} within-cell, over 54 paired points. "
        "Research-grade, not flight-qualified.",
        ha="center",
        fontsize=8.5,
        color=_common.NEUTRAL,
    )
    figure.tight_layout(rect=(0.0, 0.095, 1.0, 0.94))
    target = _common.figure_path("drift_and_clock.png")
    figure.savefig(target)
    pyplot.close(figure)

    tee("")
    tee(f"figure written: {_common.relative(target)}")
    path = tee.save()
    print(f"output written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
