"""Selection stability under perturbed value estimates.

What to notice: both quantities are needed. The Jaccard similarity of the
selected set falls faster than the value retained, because a selection that
swaps two products of similar value loses membership and keeps value. Reporting
only the Jaccard overstates the harm; reporting only the value hides the churn
an operator has to re-check by hand. The horizontal line is the mean gap the
greedy baseline leaves on the same family: above the noise level where the two
cross, the optimality bought by the exact solver is smaller than the value it
loses to uncertainty in the value table.
"""

from __future__ import annotations

import numpy as np

from _common import EXACT, GREEDY, LEFT_BEHIND, ORACLE, bootstrap, save, style

bootstrap()

from downlinkpick.dataset import generate_instance  # noqa: E402
from downlinkpick.exact import exact_select  # noqa: E402
from downlinkpick.greedy import greedy_select  # noqa: E402
from downlinkpick.selection import SelectionProblem, sizes_from_ratios  # noqa: E402
from downlinkpick.stability import stability_curve  # noqa: E402

SIGMAS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
SEEDS = tuple(range(1, 13))
N_TRIALS = 16


def _problem(instance) -> SelectionProblem:
    values = {
        name: float(instance.inventory[name].declared_value or 0.0)
        for name in instance.inventory.names
    }
    return SelectionProblem(
        instance.inventory,
        instance.plan,
        values,
        sizes_from_ratios(instance.inventory, instance.true_ratios),
    )


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    pooled: dict[str, dict[str, list[list[float]]]] = {
        "greedy": {"j": [], "v": []},
        "exact": {"j": [], "v": []},
    }
    gaps: list[float] = []
    for seed in SEEDS:
        problem = _problem(
            generate_instance(seed, n_products=20, n_windows=3, capacity_fraction=0.45)
        )
        optimum = exact_select(problem).total_value
        gaps.append(
            (optimum - greedy_select(problem).total_value) / optimum if optimum > 0 else 0.0
        )
        for label, selector in (("greedy", greedy_select), ("exact", exact_select)):
            curve = stability_curve(
                problem, SIGMAS, selector=selector, n_trials=N_TRIALS, seed=1000 + seed
            )
            pooled[label]["j"].append([point.mean_jaccard for point in curve])
            pooled[label]["v"].append([point.mean_value_retained for point in curve])

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.2))
    figure.subplots_adjust(wspace=0.26, bottom=0.17, top=0.88)

    sigmas = np.asarray(SIGMAS)
    for label, colour in (("greedy", GREEDY), ("exact", EXACT)):
        similarity = np.asarray(pooled[label]["j"])
        retained = np.asarray(pooled[label]["v"])
        left.plot(sigmas, similarity.mean(axis=0), color=colour, linewidth=1.8, marker="o",
                  markersize=3.5, label=f"{label}, Jaccard")
        left.fill_between(
            sigmas,
            np.quantile(similarity, 0.05, axis=0),
            np.quantile(similarity, 0.95, axis=0),
            color=colour,
            alpha=0.15,
        )
        right.plot(sigmas, 1.0 - retained.mean(axis=0), color=colour, linewidth=1.8,
                   marker="o", markersize=3.5, label=f"{label}, value lost")

    left.set_xlabel("relative value noise, sigma of log-normal perturbation")
    left.set_ylabel("mean Jaccard similarity of the selected set")
    left.set_title(f"membership stability, {len(SEEDS)} instances x {N_TRIALS} trials")
    left.set_ylim(top=1.005)
    left.legend(fontsize=8, loc="lower left")

    gap_mean = float(np.mean(gaps))
    right.axhline(
        gap_mean,
        color=ORACLE,
        linewidth=1.2,
        linestyle="--",
        label=f"mean greedy gap {gap_mean:.4f}",
    )
    right.set_xlabel("relative value noise, sigma")
    right.set_ylabel("fraction of the unperturbed value lost")
    right.set_title("value cost of noise, against the gap greedy leaves")
    right.set_yscale("symlog", linthresh=1e-5)
    right.legend(fontsize=8, loc="upper left")

    figure.text(
        0.5,
        0.01,
        "greedy's loss can be negative -- its unperturbed ordering is not optimal, so "
        "noise sometimes improves it; the exact solver's cannot",
        ha="center",
        fontsize=7,
        color=LEFT_BEHIND,
    )
    return save(figure, "stability_curve")


if __name__ == "__main__":
    print(main())
