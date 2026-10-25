"""The distribution of the greedy optimality gap over a seeded instance family.

What to notice: the baseline ties the proven optimum on nearly half the
instances, and the tail is what matters. A mean gap hides both facts, which is
why the whole distribution is plotted and the mean is only annotated.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import EXACT, GREEDY, bootstrap, save, style

bootstrap()

from contactsched.benchmark import optimality_gap  # noqa: E402
from contactsched.dataset import generate_instance  # noqa: E402
from contactsched.exact import solve_exact  # noqa: E402
from contactsched.greedy import schedule_greedy  # noqa: E402

SEEDS = range(0, 160)
HORIZON_S = 43200.0


def gaps_for(mode: str, order: str, problems) -> list[float]:
    """Optimality gaps of one greedy variant against the exact optimum."""
    values = []
    for problem in problems:
        greedy = schedule_greedy(problem, mode, order)
        exact = solve_exact(problem, mode)
        values.append(optimality_gap(greedy, exact))
    return values


def main() -> None:
    style()
    problems = []
    for seed in SEEDS:
        instance = generate_instance(seed, horizon_s=HORIZON_S)
        if instance.problem.candidates:
            problems.append(instance.problem)

    series = {
        "priority order, per contact": gaps_for("weighted_contacts", "priority", problems),
        "demand order, per contact": gaps_for("weighted_contacts", "demand", problems),
        "priority order, per second": gaps_for("weighted_seconds", "priority", problems),
    }

    figure, (ecdf, box) = plt.subplots(
        1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [3, 2]}
    )
    colours = (GREEDY, EXACT, "#7b4fa8")
    for (label, values), colour in zip(series.items(), colours, strict=True):
        ordered = np.sort(np.asarray(values))
        fraction = np.arange(1, len(ordered) + 1) / len(ordered)
        ties = float(np.mean(ordered <= 1e-12))
        ecdf.step(
            ordered,
            fraction,
            where="post",
            color=colour,
            lw=1.6,
            label=f"{label}: median {np.median(ordered):.4f}, ties {ties:.3f}",
        )
    ecdf.set_xlabel("optimality gap, (exact - greedy) / exact")
    ecdf.set_ylabel("fraction of instances at or below")
    ecdf.set_title(f"Greedy optimality gap over {len(problems)} instances")
    ecdf.set_ylim(0.0, 1.02)
    ecdf.legend(fontsize=7, loc="lower right")

    box.boxplot(
        list(series.values()),
        tick_labels=["prio\nper contact", "demand\nper contact", "prio\nper second"],
        showmeans=True,
        widths=0.55,
    )
    box.set_ylabel("optimality gap")
    box.set_title("The tail is the result, not the mean")

    figure.tight_layout()
    path = save(figure, "greedy_optimality_gap")

    for label, values in series.items():
        array = np.asarray(values)
        print(
            f"{label:<30} n {len(array):3d}  mean {array.mean():.6f}  "
            f"median {np.median(array):.6f}  p90 {np.quantile(array, 0.9):.6f}  "
            f"max {array.max():.6f}  ties {float(np.mean(array <= 1e-12)):.6f}"
        )
    print(f"figure written: {path}")


if __name__ == "__main__":
    main()
