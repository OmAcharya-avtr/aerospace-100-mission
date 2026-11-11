"""The harness's own cost, and how much it perturbs what it measures.

Left: the per-repeat cost breakdown, with the independently measured sandwich
the in-bracket overhead has to sit inside. Right: the perturbation ratio of a
declared reference payload as the payload gets cheaper, which is where a
harness stops being an instrument.

The right panel is measured over FIVE trials per payload size and plotted as a
median with a min-to-max band, because a single trial of it on a shared
two-core host is dominated by noise above about twenty microseconds -- single
trials have come back both above and below 1.0 at the same payload size. The
band is the honest content of the figure: where it straddles 1.0, the
perturbation is not resolvable here; where the median is well above 1.0, it
is.

EVERY NUMBER IN THIS FIGURE IS A PROPERTY OF THE HOST IT WAS PRODUCED ON AND
NOT OF AN NVIDIA JETSON ORIN NANO. The declared target was not reachable. The
host's architecture is printed in the figure, because the target is aarch64.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import platform  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import LIVE, NEUTRAL, REFUSED, SIMULATED, bootstrap, save, style  # noqa: E402

ROOT = bootstrap()

from boardprobe import (  # noqa: E402
    ORIN_NANO,
    LiveSource,
    benchmark,
    characterise_overhead,
    reference_payload,
)

WORK_SIZES = (50, 100, 250, 500, 1000, 2000, 4000, 8000)

#: Trials per payload size. The ratio is noisy on a shared two-core host, so
#: the median and the min-to-max band are reported rather than one trial.
TRIALS = 5


def main() -> int:
    style()
    clock = LiveSource(ORIN_NANO)
    report = characterise_overhead(clock, repeats=400)

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.1))

    names = ["one sealed\nReading", "overhead inside\nthe bracket",
             "one clock\nread pair"]
    values = [
        report.reading_cost_s * 1e6,
        report.in_bracket_overhead_s * 1e6,
        report.clock_pair_cost_s * 1e6,
    ]
    bars = left.bar(names, values, color=[SIMULATED, LIVE, NEUTRAL], width=0.6)
    left.bar_label(bars, fmt="%.2f", fontsize=9)
    left.axhspan(values[0], values[2], color=LIVE, alpha=0.08)
    left.set_ylabel("microseconds per repeat")
    left.set_title(
        "the in-bracket overhead inside its independently\nmeasured sandwich"
    )
    left.set_ylim(0, max(values) * 1.3)

    medians = []
    lows = []
    highs = []
    costs = []
    for work in WORK_SIZES:
        payload = reference_payload(work)
        trial_ratios = []
        trial_costs = []
        for _ in range(TRIALS):
            with_sampling = benchmark(
                payload, f"work={work}", clock, repeats=120, warmup=5
            )
            baseline_before = clock.read("monotonic_clock")
            for _ in range(120):
                payload()
            baseline_after = clock.read("monotonic_clock")
            without = (baseline_after.value - baseline_before.value) / 120.0
            trial_costs.append(without * 1e6)
            trial_ratios.append(
                with_sampling.mean_duration_s / without if without > 0 else np.nan
            )
        medians.append(float(np.median(trial_ratios)))
        lows.append(float(np.min(trial_ratios)))
        highs.append(float(np.max(trial_ratios)))
        costs.append(float(np.median(trial_costs)))
    right.fill_between(costs, lows, highs, color=LIVE, alpha=0.15,
                       label=f"min to max over {TRIALS} trials")
    right.plot(costs, medians, marker="o", color=LIVE, linewidth=1.6,
               label="median")
    right.axhline(1.0, color=NEUTRAL, linestyle=":", linewidth=1.0,
                  label="no perturbation")
    right.axhline(1.1, color=REFUSED, linestyle="--", linewidth=1.0,
                  label="10 per cent perturbation")
    right.set_xscale("log")
    right.set_xlabel("payload cost without sampling [us]")
    right.set_ylabel("measured / unsampled")
    right.set_title(
        f"the harness stops being an instrument\nbelow a few microseconds "
        f"({TRIALS} trials per point)"
    )
    right.legend(fontsize=7.5)

    figure.suptitle(
        f"boardprobe harness overhead on {platform.machine()} {platform.system()} "
        "-- NOT an NVIDIA Jetson Orin Nano measurement "
        "(Level 3, hardware-pending)",
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    path = save(figure, "harness_overhead")
    print(f"written: {path}")
    print(f"host: {platform.machine()} {platform.system()} (target is aarch64)")
    print(f"one sealed Reading      [us]: {values[0]:.3f}")
    print(f"in-bracket overhead     [us]: {values[1]:.3f}")
    print(f"one clock read pair     [us]: {values[2]:.3f}")
    print(f"trials per payload size     : {TRIALS}")
    print(f"payload costs, median   [us]: {[round(c, 2) for c in costs]}")
    print(f"perturbation ratio, median  : {[round(r, 3) for r in medians]}")
    print(f"perturbation ratio, min     : {[round(r, 3) for r in lows]}")
    print(f"perturbation ratio, max     : {[round(r, 3) for r in highs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
