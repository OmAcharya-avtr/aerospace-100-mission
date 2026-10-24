"""The closed-form baseline scored against the iterated reference over a family.

Writes ``screenshots/baseline_vs_reference.png``. Two panels: the distribution
of the closed-loop offset error and the open-loop transmit-direction error over
a seeded family of crosslinks, and the same errors against range.
"""

from __future__ import annotations

import numpy as np

from _common import SERIES, bootstrap, plt, save, style_axes

bootstrap()

from pointahead.angle import score_closed_form  # noqa: E402
from pointahead.budget import direction_noise_floor_rad  # noqa: E402
from pointahead.scenario import scenario_family  # noqa: E402

SEED = 760076
COUNT = 300


def main() -> int:
    family = scenario_family(SEED, COUNT)
    ranges = []
    offsets = []
    transmits = []
    floors = []
    angles = []
    for link in family:
        score = score_closed_form(link)
        ranges.append(link.range_m / 1.0e3)
        offsets.append(score.offset_difference_rad)
        transmits.append(score.transmit_direction_rad)
        floors.append(direction_noise_floor_rad(link))
        angles.append(score.reference_rad)
    ranges_array = np.array(ranges)
    offsets_array = np.array(offsets)
    transmits_array = np.array(transmits)
    floors_array = np.array(floors)
    angles_array = np.array(angles)

    print(f"seed                    : {SEED}")
    print(f"crosslinks              : {COUNT}")
    print(f"ranges                  : {ranges_array.min():.1f} to "
          f"{ranges_array.max():.1f} km")
    print(f"point-ahead angles      : {angles_array.min() * 1.0e6:.6f} to "
          f"{angles_array.max() * 1.0e6:.6f} urad")
    print()
    print(f"{'quantity':<40s} {'median':>14s} {'p95':>14s} {'max':>14s}")
    print("-" * 86)
    for label, array in (
        ("closed-loop offset error / rad", offsets_array),
        ("open-loop transmit error / rad", transmits_array),
        ("float64 direction resolution / rad", floors_array),
    ):
        print(
            f"{label:<40s} {float(np.median(array)):>14.6e} "
            f"{float(np.percentile(array, 95)):>14.6e} {float(array.max()):>14.6e}"
        )
    below = int(np.sum(offsets_array <= floors_array))
    print()
    print(f"closed-loop errors at or below the float64 floor: {below} of {COUNT}")
    print(f"worst closed-loop error as a fraction of the angle: "
          f"{float((offsets_array / angles_array).max()):.6e}")
    print(f"worst open-loop error as a fraction of the angle  : "
          f"{float((transmits_array / angles_array).max()):.6e}")

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    bins = np.logspace(-15.0, -8.5, 40)
    left.hist(offsets_array, bins=bins, color=SERIES[0], alpha=0.85, label="closed loop (offset)")
    left.hist(
        transmits_array, bins=bins, color=SERIES[1], alpha=0.75, label="open loop (transmit)"
    )
    left.set_xscale("log")
    style_axes(
        left,
        f"Baseline against reference over {COUNT} seeded crosslinks",
        "disagreement / rad (log scale)",
        "crosslinks",
    )
    left.legend(fontsize=8, loc="upper center", framealpha=0.95)

    right.scatter(
        ranges_array, offsets_array, s=12, color=SERIES[0], alpha=0.8, label="closed loop"
    )
    right.scatter(
        ranges_array, transmits_array, s=12, color=SERIES[1], alpha=0.6, label="open loop"
    )
    right.scatter(
        ranges_array,
        floors_array,
        s=8,
        color="#666666",
        alpha=0.6,
        marker="x",
        label="float64 direction resolution",
    )
    right.set_yscale("log")
    style_axes(right, "Disagreement against range", "range / km", "disagreement / rad")
    right.legend(fontsize=8, loc="lower right", framealpha=0.95)

    print()
    print(f"written: {save(figure, 'baseline_vs_reference.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
