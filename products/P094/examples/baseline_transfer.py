"""Example 2: the three baselines, and where a single scaling factor transfers.

Prints the three SPEC §4 baselines in specification order and draws two panels:
the declared timing model against measured time stage by stage, and baseline
2's held-out predictions against ground truth with the distribution of all
twenty three-and-three splits behind them.

Writes ``screenshots/baseline_transfer.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from hiltimingdiff import (  # noqa: E402
    baseline_1,
    baseline_2,
    baseline_3,
    calibrate_ns_per_op,
    load_campaign,
    measure_simulation,
    pass_units,
    transfer_verdict,
)
from hiltimingdiff.baselines import (  # noqa: E402
    HOLDOUT_STAGES,
    PREREGISTERED_TRANSFER_ERROR,
    TRAIN_STAGES,
)

PASSES = 4


def main() -> int:
    tee = _common.Tee("baseline_transfer")
    _common.scope_header(tee, "P094 HilTimingDiff -- example 2: the three baselines")

    campaign = load_campaign()
    units = pass_units(campaign)
    simulation = measure_simulation(campaign, passes=PASSES)
    ns_per_op = calibrate_ns_per_op(simulation)

    first = baseline_1(units, ns_per_op)
    second = baseline_2(units, simulation, resamples=2000)
    third = baseline_3(units, resamples=2000)

    tee("BASELINE 1 -- the declared timing model, the object being refuted")
    tee(f"  ns_per_op calibrated once on this host : {ns_per_op:.9f} ns per declared op")
    tee("  stage         declared_ns      hw_median_ns   signed rel err")
    for stage in first.stages:
        tee(
            f"  {stage:<11s} {first.declared_ns[stage]:>14.2f} "
            f"{first.hw_median_ns[stage]:>17.2f} "
            f"{first.signed_relative_error[stage]:>+16.6f}"
        )
    tee(f"  mean relative error over stages : {first.mean_relative_error:.6f}")
    tee("")

    tee("BASELINE 2 -- one scalar fitted on training stages, scored on held-out stages")
    tee(f"  training : {', '.join(TRAIN_STAGES)}")
    tee(f"  held out : {', '.join(HOLDOUT_STAGES)}")
    tee(f"  k        : {second.declared.k:.6f}")
    tee(f"  held-out mean relative error : {second.declared_interval.format(6)}")
    for stage in HOLDOUT_STAGES:
        tee(
            f"    {stage:<11s} predicted {second.declared.predicted_ns[stage]:>12.2f} ns "
            f"against measured {second.hw_median_ns[stage]:>12.2f} ns, error "
            f"{second.declared.relative_error[stage]:.6f}"
        )
    tee(f"  leave-one-stage-out mean : {second.leave_one_out_mean:.6f}")
    tee(
        f"  all 20 splits            : min {second.split_error_min:.6f}, median "
        f"{second.split_error_median:.6f}, max {second.split_error_max:.6f}"
    )
    tee(
        f"  splits above the pre-registered {PREREGISTERED_TRANSFER_ERROR:g} : "
        f"{second.splits_above_preregistration} of {len(second.split_errors)}"
    )
    tee(f"  {transfer_verdict(second).line()}")
    tee("")

    tee(f"BASELINE 3 -- ground truth, {third.unit}-unit {third.confidence:.0%} interval")
    tee("  stage         median_ns            95% interval      relative width")
    for stage in third.stages:
        interval = third.interval[stage]
        tee(
            f"  {stage:<11s} {third.median_ns[stage]:>13.2f}  "
            f"[{interval.low:>12.2f}, {interval.high:>12.2f}] {interval.relative_width:>14.6f}"
        )
    tee("")
    tee("The hardware-to-simulation ratio, which is why baseline 2 transfers:")
    for stage in units.stages:
        tee(f"  {stage:<11s} {second.per_stage_ratio[stage]:.6f}")
    ratios = list(second.per_stage_ratio.values())
    declared_ratios = [
        first.hw_median_ns[stage] / first.declared_ns[stage] for stage in first.stages
    ]
    tee(
        f"  spread {max(ratios) / min(ratios):.6f}, against a declared-model divergence "
        f"spread of {max(declared_ratios) / min(declared_ratios):.4f}"
    )

    pyplot = _common.style()
    figure, axes = pyplot.subplots(1, 2, figsize=(12.6, 6.0))

    stages = list(units.stages)
    positions = np.arange(len(stages))
    declared = np.array([first.declared_ns[stage] for stage in stages])
    measured = np.array([first.hw_median_ns[stage] for stage in stages])
    simulated = np.array([second.sim_median_ns[stage] for stage in stages])

    left = axes[0]
    width = 0.27
    left.bar(
        positions - width,
        declared,
        width,
        label="declared (operation count x ns_per_op)",
        color=_common.DECLARED,
        edgecolor=_common.INK,
        linewidth=0.6,
    )
    left.bar(
        positions,
        simulated,
        width,
        label="simulation mode, measured here",
        color=_common.SIMULATION,
        edgecolor=_common.INK,
        linewidth=0.6,
    )
    left.bar(
        positions + width,
        measured,
        width,
        label="hardware mode, measured on the board",
        color=_common.HARDWARE,
        edgecolor=_common.INK,
        linewidth=0.6,
    )
    left.set_yscale("log")
    left.set_xticks(positions)
    left.set_xticklabels(stages, rotation=20, ha="right")
    left.set_ylabel("nanoseconds per frame (log scale)")
    left.set_title(
        "Baseline 1 against ground truth\n"
        f"mean relative error {first.mean_relative_error:.3f}, worst "
        f"{first.worst_stage} at {first.max_relative_error:.3f}"
    )
    left.legend(loc="upper left", fontsize=8.5)

    right = axes[1]
    errors = np.asarray(second.split_errors, dtype=np.float64)
    right.hist(
        errors,
        bins=10,
        color=_common.NEUTRAL,
        edgecolor=_common.INK,
        linewidth=0.6,
        alpha=0.55,
        label="all 20 three-and-three splits",
    )
    right.axvline(
        second.declared.mean_relative_error,
        color=_common.HARDWARE,
        linewidth=2.2,
        label=f"declared split = {second.declared.mean_relative_error:.4f}",
    )
    right.axvline(
        second.leave_one_out_mean,
        color=_common.SIMULATION,
        linewidth=2.0,
        linestyle="--",
        label=f"leave-one-stage-out = {second.leave_one_out_mean:.4f}",
    )
    right.axvline(
        PREREGISTERED_TRANSFER_ERROR,
        color=_common.DECLARED,
        linewidth=2.2,
        linestyle=":",
        label=f"SPEC §7 pre-registered > {PREREGISTERED_TRANSFER_ERROR:g}",
    )
    right.set_xlabel("mean relative error on the held-out stages")
    right.set_ylabel("splits")
    # The word on the figure is the verdict the measurement in THIS run lands
    # on, not a literal. The held-out error is a wall-clock reading on a shared
    # two-core host, and a hard-coded "REFUTES" would be a caption that stopped
    # describing its own figure the moment the measurement crossed 0.5.
    verdict_word = (
        "REFUTES"
        if second.declared.mean_relative_error <= PREREGISTERED_TRANSFER_ERROR
        else "CONFIRMS"
    )
    right.set_title(
        f"Baseline 2 transfer, and the pre-registration it {verdict_word}\n"
        f"{second.splits_above_preregistration} of {len(second.split_errors)} splits "
        f"exceed the pre-registered threshold"
    )
    right.legend(loc="upper right", fontsize=8.5)

    figure.suptitle(
        "P094 HilTimingDiff -- the three baselines of SPEC §4, classical work first",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Hardware mode: Jetson Orin Nano, 810000 timed frames, UNPINNED schedutil "
        "clock in MAXN_SUPER.\nThe BLUE bars are calibration-free and reproduce "
        "exactly; the AMBER bars are wall-clock readings taken on this host now, and "
        "the RED bars\ncarry the host-calibrated ns_per_op. Both move between runs. "
        "Research-grade, not flight-qualified.",
        ha="center",
        fontsize=8.5,
        color=_common.NEUTRAL,
    )
    figure.tight_layout(rect=(0.0, 0.075, 1.0, 0.94))
    target = _common.figure_path("baseline_transfer.png")
    figure.savefig(target)
    pyplot.close(figure)

    tee("")
    tee(f"figure written: {_common.relative(target)}")
    path = tee.save()
    print(f"output written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
