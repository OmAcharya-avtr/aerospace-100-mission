"""What planning against a prediction interval buys, and what it breaks.

Left: realised feasibility rate and realised objective for the four planning
modes, against the unachievable oracle.
Middle: why the duration-robust plan is not the safest one. Booking the upper
end of the duration interval makes the plan assume a LOW data production rate,
because the data volume is declared and the rate is volume over duration. The
realised rate is then higher than planned, and the recorder overflows.
Right: the binding constraint of each mode's failures.

Run: ``python examples/robust_tradeoff.py``
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from _common import ANALYTIC, EXACT, GREEDY, LEARNED, ROBUST, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from activityplan.features import feature_matrix, realise, sample_requests  # noqa: E402
from activityplan.instances import instance_family  # noqa: E402
from activityplan.predictor import DurationEnergyPredictor  # noqa: E402
from activityplan.robust import (  # noqa: E402
    PLANNING_MODES,
    compare_modes,
    estimate_for_mode,
    realise_spec,
)

SEED = 61
N_TRAIN = 1000
ALPHA = 0.10
FAMILY_SEED = 9000
FAMILY_SIZE = 12
REALISATIONS = 3
COLOURS = {
    "oracle": EXACT,
    "analytic": ANALYTIC,
    "ml-point": LEARNED,
    "ml-robust": ROBUST,
}


def main() -> int:
    style()
    rng = np.random.default_rng(SEED)
    requests = sample_requests(N_TRAIN, rng)
    realised = [realise(request, rng) for request in requests]
    predictor = DurationEnergyPredictor(
        alpha=ALPHA, mode="normalised", n_estimators=160, random_state=SEED
    ).fit(
        feature_matrix(list(requests)),
        np.array([item.duration_s for item in realised], dtype=float),
        np.array([item.energy_wh for item in realised], dtype=float),
    )

    feasible = dict.fromkeys(PLANNING_MODES, 0)
    objective = dict.fromkeys(PLANNING_MODES, 0.0)
    placed = dict.fromkeys(PLANNING_MODES, 0.0)
    reasons: dict[str, Counter[str]] = {mode: Counter() for mode in PLANNING_MODES}
    plans = 0
    for spec in instance_family(FAMILY_SIZE, FAMILY_SEED):
        for repeat in range(REALISATIONS):
            outcomes = compare_modes(
                spec,
                np.random.default_rng(spec.seed * 100 + repeat),
                predictor=predictor,
                planner="greedy",
            )
            plans += 1
            for outcome in outcomes:
                feasible[outcome.mode] += int(outcome.feasible)
                objective[outcome.mode] += outcome.realised_value
                placed[outcome.mode] += outcome.n_placed
                if not outcome.feasible:
                    reasons[outcome.mode][outcome.binding] += 1

    rate_ratio: dict[str, float] = {}
    underbooked: dict[str, float] = {}
    for mode in PLANNING_MODES:
        planned_rates: list[float] = []
        realised_rates: list[float] = []
        for spec in instance_family(FAMILY_SIZE, FAMILY_SEED):
            truth = realise_spec(spec, np.random.default_rng(spec.seed + 500))
            estimate = estimate_for_mode(spec, mode, predictor=predictor, truth=truth)
            volumes = np.array([request.data_mbit for request in spec.requests], dtype=float)
            planned_rates.extend((volumes / estimate.durations_s).tolist())
            realised_rates.extend((volumes / truth[0]).tolist())
        planned = np.array(planned_rates)
        actual = np.array(realised_rates)
        rate_ratio[mode] = float(np.mean(actual / planned))
        underbooked[mode] = float(np.mean(actual > planned))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

    ax = axes[0]
    positions = np.arange(len(PLANNING_MODES))
    rates = [feasible[mode] / plans for mode in PLANNING_MODES]
    bars = ax.bar(positions, rates, color=[COLOURS[mode] for mode in PLANNING_MODES], width=0.6)
    for bar, mode in zip(bars, PLANNING_MODES, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{bar.get_height():.3f}\n{objective[mode] / plans:.2f} units",
            ha="center",
            fontsize=7.5,
            color=COLOURS[mode],
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(PLANNING_MODES, rotation=12)
    ax.set_ylabel("realised feasibility rate")
    ax.set_ylim(0.0, 1.15)
    ax.set_title(
        f"{plans} plans per mode ({FAMILY_SIZE} instances x {REALISATIONS} realisations).\n"
        "A plan violating any constraint scores zero, not partial credit.\n"
        "The second line is the mean realised objective."
    )

    ax = axes[1]
    bars = ax.bar(
        positions,
        [rate_ratio[mode] for mode in PLANNING_MODES],
        color=[COLOURS[mode] for mode in PLANNING_MODES],
        width=0.6,
    )
    for bar, mode in zip(bars, PLANNING_MODES, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{bar.get_height():.4f}\n{underbooked[mode]:.0%} underbooked",
            ha="center",
            fontsize=7.5,
            color=COLOURS[mode],
        )
    ax.axhline(1.0, color=EXACT, linestyle="--", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(PLANNING_MODES, rotation=12)
    ax.set_ylabel("realised data rate / planned data rate")
    ax.set_ylim(0.9, 1.3)
    ax.set_title(
        "the mechanism. Data volume is DECLARED, so the rate is\n"
        "volume over duration. Booking a LONGER duration books a\n"
        "LOWER rate, and the recorder then fills faster than planned."
    )

    ax = axes[2]
    labels = sorted({name for counter in reasons.values() for name in counter})
    bottom = np.zeros(len(PLANNING_MODES))
    palette = [GREEDY, LEARNED, ROBUST, ANALYTIC, EXACT]
    for index, label in enumerate(labels):
        heights = np.array([reasons[mode][label] for mode in PLANNING_MODES], dtype=float)
        ax.bar(
            positions,
            heights,
            bottom=bottom,
            color=palette[index % len(palette)],
            width=0.6,
            label=label,
        )
        bottom += heights
    ax.set_xticks(positions)
    ax.set_xticklabels(PLANNING_MODES, rotation=12)
    ax.set_ylabel("infeasible plans")
    ax.set_title(
        "why the failures failed. The analytic mode breaks the\n"
        "battery by underbooking duration; the robust mode breaks\n"
        "the recorder by underbooking rate. Opposite mechanisms."
    )
    ax.legend(fontsize=6.5, loc="upper left")

    fig.suptitle(
        "activityplan: a duration-robust plan is not a resource-robust plan",
        fontsize=11,
    )
    fig.subplots_adjust(top=0.80, wspace=0.3)
    path = save(fig, "robust_tradeoff.png")
    print(f"wrote {path.relative_to(add_src_to_path())}")
    print("")
    print(f"plans per mode : {plans}")
    print(
        f"{'mode':11s} {'feasible':>9s} {'rate':>7s} {'mean realised':>14s} "
        f"{'mean placed':>12s} {'rate ratio':>11s} {'underbooked':>12s}"
    )
    for mode in PLANNING_MODES:
        print(
            f"{mode:11s} {feasible[mode]:9d} {feasible[mode] / plans:7.4f} "
            f"{objective[mode] / plans:14.3f} {placed[mode] / plans:12.3f} "
            f"{rate_ratio[mode]:11.4f} {underbooked[mode]:12.4f}"
        )
    print("")
    for mode in PLANNING_MODES:
        parts = ", ".join(f"{name} x{count}" for name, count in reasons[mode].most_common())
        print(f"{mode:11s} failures: {parts or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
