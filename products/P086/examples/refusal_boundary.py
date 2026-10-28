"""The refuse region in the plane of gap length against declared tolerance."""

from __future__ import annotations

import numpy as np

from _common import PERMIT, REFUSE, bootstrap, save, style, tolerances

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from telemgap import dataset, policy  # noqa: E402
from telemgap.screens import Gap  # noqa: E402

SLOT_GRID = np.arange(1, 13)
TOLERANCE_DECADES = np.logspace(-4, 0, 40)


def main() -> int:
    style()
    generated = dataset.generate(samples=4000, seed=515, n_clock_resets=0)
    record = generated.record
    curvatures = policy.channel_curvatures(record)
    declared = tolerances()

    figure, axes = plt.subplots(2, 2, figsize=(9.8, 6.4))
    colours = ListedColormap([REFUSE, PERMIT])
    for column, (channel, axis) in enumerate(
        zip(record.schema.channels, axes.ravel(), strict=True)
    ):
        estimate = curvatures[channel.name]
        decisions = np.zeros((TOLERANCE_DECADES.size, SLOT_GRID.size))
        for row, tolerance in enumerate(TOLERANCE_DECADES):
            for col, slots in enumerate(SLOT_GRID):
                duration = (slots + 1) * record.schema.cadence_s
                gap = Gap(
                    after_index=0,
                    before_index=1,
                    start_time=0.0,
                    end_time=duration,
                    duration_s=duration,
                    unobserved_s=duration - record.schema.cadence_s,
                    missing_frames_from_time=int(slots),
                    missing_frames_from_counter=int(slots),
                    agrees=True,
                )
                decision = policy.decide_gap(
                    gap=gap,
                    gap_index=0,
                    channel_index=column,
                    channel_name=channel.name,
                    unit_tolerance=float(tolerance),
                    curvature=estimate,
                    slew=float(channel.max_slew),
                    cadence_s=float(record.schema.cadence_s),
                )
                decisions[row, col] = 0.0 if decision.refused else 1.0
        axis.pcolormesh(
            SLOT_GRID,
            TOLERANCE_DECADES,
            decisions,
            cmap=colours,
            vmin=0.0,
            vmax=1.0,
            shading="nearest",
        )
        axis.set_yscale("log")
        axis.axhline(
            declared[channel.name],
            color="#111111",
            linewidth=1.1,
            linestyle="--",
        )
        axis.axvline(policy.DEFAULT_MAX_GAP_SLOTS, color="#111111", linewidth=0.8, linestyle=":")
        axis.set_xlabel("gap length [cadence intervals]")
        axis.set_ylabel(f"declared tolerance [{channel.unit}]")
        axis.grid(False)
        axis.set_title(
            f"{channel.name}: M2 {estimate.value:.3g}/s^2 ({estimate.source}), "
            f"{policy.DEFAULT_COVERAGE_K:g} sigma = "
            f"{policy.DEFAULT_COVERAGE_K * estimate.sigma:.3g}\n"
            f"measurable {estimate.measurable}, resolvable {estimate.resolvable}",
            fontsize=8,
        )
    figure.suptitle(
        "telemgap.policy: green is permitted, red is refused. The dashed line is this "
        "repository's declared tolerance\nand the dotted line is the Nyquist gate. At the "
        "declared tolerance the two channels whose curvature is estimated are refused at "
        "every gap length,\nfor two different stated reasons: an estimate below its own "
        "noise floor, and content above the estimator's bandwidth.",
        fontsize=10,
    )
    figure.subplots_adjust(hspace=0.55, wspace=0.28)
    path = save(figure, "refusal_boundary")
    print(f"written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
