"""Example 1: the five headline outputs, and the figure behind them.

Runs the whole product: loads the committed hardware campaign, measures
simulation mode here with the committed instrument, and prints the five SPEC §5
outputs with their pass-unit intervals and both pre-registration verdicts.

Writes ``screenshots/headline_divergence.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _common  # noqa: E402

ROOT = _common.bootstrap()

from hiltimingdiff import build_headline, headline_lines  # noqa: E402

#: Simulation-mode passes. Four passes over nine cells at the board's own
#: 1500-frame stack is 54000 timed frames and about twelve seconds on two cores.
PASSES = 4


def main() -> int:
    tee = _common.Tee("headline_divergence")
    _common.scope_header(tee, "P094 HilTimingDiff -- example 1: the headline divergence")

    headline = build_headline(passes=PASSES, resamples=2000)
    divergence = headline.divergence

    for line in headline_lines(headline):
        tee(line)
    tee("")
    tee(
        f"simulation mode measured here: {headline.simulation.machine}, "
        f"{headline.simulation.passes} passes x {headline.simulation.frames} frames x 9 "
        f"cells = {headline.simulation.n_timed_frames} timed frames in "
        f"{headline.simulation.elapsed_s:.1f} s"
    )
    tee(
        f"hardware mode read from {headline.campaign.label}/: "
        f"{headline.n_hardware_frames} timed frames on the board"
    )

    tee("")
    tee("The calibration-free form, which depends on nothing but the committed traces:")
    tee(
        "  stage        ns per declared op         95% interval (same units)   "
        "hw median ns   declared ops"
    )
    for stage in divergence.stages:
        interval = divergence.ns_per_declared_op_interval[stage]
        tee(
            f"  {stage:<11s} {divergence.ns_per_declared_op[stage]:>18.6f} "
            f"[{interval.low:>12.6f}, {interval.high:>12.6f}] "
            f"{divergence.hw_median_ns[stage]:>14.2f} "
            f"{divergence.declared_ops_median[stage]:>14.4f}"
        )

    pyplot = _common.style()
    figure, axes = pyplot.subplots(1, 2, figsize=(12.6, 6.0))

    stages = list(divergence.stages)
    positions = np.arange(len(stages))
    # The point estimate is CALIBRATION-FREE, so its error bar must be too. The
    # interval is taken in ns-per-declared-op units directly and is NOT
    # recovered from the R interval by multiplying back by ns_per_op: that round
    # trip is exact only in exact arithmetic, and it blocked this product at the
    # release gate. `ingest`'s interval high EQUALS its point estimate, because
    # its per-pass medians are coarsely quantised, so the round trip landed one
    # ulp below the point for some values of ns_per_op and matplotlib refused a
    # negative `xerr`. Mixing a quantity that does not move with bounds that do
    # was the defect; the fix is to never rescale one into the other.
    per_op = np.array([divergence.ns_per_declared_op[stage] for stage in stages])
    low = np.array(
        [divergence.ns_per_declared_op_interval[stage].low for stage in stages]
    )
    high = np.array(
        [divergence.ns_per_declared_op_interval[stage].high for stage in stages]
    )

    # Fail loudly and informatively here rather than inside matplotlib. A
    # zero-width bar is legitimate -- `ingest`'s high equals its point exactly --
    # but a negative one never is, and it is never clipped away.
    for index, stage in enumerate(stages):
        if not (low[index] <= per_op[index] <= high[index]):
            raise ValueError(
                f"interval does not bracket the point estimate for stage {stage!r}: "
                f"low={low[index]!r} point={per_op[index]!r} high={high[index]!r}. "
                "This is a real inconsistency and is not clipped: an error bar "
                "drawn after clipping would not mean what the axis label says."
            )

    left = axes[0]
    left.barh(
        positions,
        per_op,
        color=_common.HARDWARE,
        edgecolor=_common.INK,
        linewidth=0.7,
        height=0.6,
    )
    left.errorbar(
        per_op,
        positions,
        xerr=[per_op - low, high - per_op],
        fmt="none",
        ecolor=_common.INK,
        elinewidth=1.1,
        capsize=4,
    )
    left.set_xscale("log")
    left.set_yticks(positions)
    left.set_yticklabels(stages)
    left.invert_yaxis()
    left.set_xlabel("measured nanoseconds per DECLARED operation (log scale)")
    left.set_title(
        "Per-stage divergence, calibration-free form\n"
        f"spread max/min = {divergence.spread:.1f}  "
        f"[{divergence.spread_interval.low:.1f}, {divergence.spread_interval.high:.1f}]"
        f"   argmax = {divergence.argmax}, argmin = {divergence.argmin}"
    )
    for index, value in enumerate(per_op):
        left.annotate(
            f"{value:.3g}",
            (value, positions[index]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9.5,
        )
    left.axvline(
        float(np.min(per_op)),
        color=_common.NEUTRAL,
        linestyle=":",
        linewidth=1.0,
    )

    right = axes[1]
    declared_rank = {stage: rank for rank, stage in enumerate(divergence.declared_order)}
    measured_rank = {stage: rank for rank, stage in enumerate(divergence.measured_order)}
    for stage in stages:
        right.plot(
            [0, 1],
            [declared_rank[stage], measured_rank[stage]],
            marker="o",
            color=_common.DECLARED
            if declared_rank[stage] != measured_rank[stage]
            else _common.NEUTRAL,
            linewidth=1.8 if declared_rank[stage] != measured_rank[stage] else 1.0,
            markersize=6,
        )
        right.annotate(
            stage,
            (0, declared_rank[stage]),
            xytext=(-8, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9.5,
        )
        right.annotate(
            stage,
            (1, measured_rank[stage]),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9.5,
        )
    right.set_xlim(-0.45, 1.45)
    right.set_xticks([0, 1])
    right.set_xticklabels(
        ["declared operation count\n(cheapest at the top)", "measured on the board"]
    )
    right.set_yticks(range(len(stages)))
    right.set_yticklabels([f"rank {index}" for index in range(len(stages))])
    right.invert_yaxis()
    right.grid(axis="x", visible=False)
    right.set_title(
        "Rank disagreement\n"
        f"Kendall tau-b = {divergence.tau_b:.4f}   permutation = {divergence.permutation}"
    )

    figure.suptitle(
        "P094 HilTimingDiff -- simulation-to-hardware per-stage timing divergence",
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.012,
        "Hardware mode: Jetson Orin Nano, 810000 timed frames, UNPINNED schedutil "
        "clock in MAXN_SUPER, board drift 1.0752 over the campaign. "
        "Research-grade, not flight-qualified.",
        ha="center",
        fontsize=8.5,
        color=_common.NEUTRAL,
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.94))
    target = _common.figure_path("headline_divergence.png")
    figure.savefig(target)
    pyplot.close(figure)

    tee("")
    tee(f"figure written: {_common.relative(target)}")
    path = tee.save()
    print(f"output written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
