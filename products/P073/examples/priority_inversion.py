"""Priority inversion: what the priority scheme promised against what it delivered.

What to notice: every point below the diagonal order is a request served worse
than something less urgent. The inversions here are not caused by the weights --
the solver maximised exactly what it was asked to -- but by geometry and by
requests declaring needs their own passes cannot carry.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import EXACT, GREEDY, STARVED, bootstrap, save, style

bootstrap()

from contactsched.dataset import generate_instance  # noqa: E402
from contactsched.exact import solve_exact  # noqa: E402
from contactsched.fairness import analyse  # noqa: E402
from contactsched.greedy import schedule_greedy  # noqa: E402

SEED = 7
HORIZON_S = 43200.0
FAMILY = range(0, 60)


def main() -> None:
    style()
    instance = generate_instance(SEED, horizon_s=HORIZON_S)
    problem = instance.problem
    greedy_report = analyse(problem, schedule_greedy(problem))
    exact_report = analyse(problem, solve_exact(problem))

    figure, (scatter, hist, delivered) = plt.subplots(1, 3, figsize=(13.0, 4.2))

    rng = np.random.default_rng(1)
    for report, colour, label in (
        (greedy_report, GREEDY, "greedy"),
        (exact_report, EXACT, "exact"),
    ):
        points = [(o.priority, o.satisfaction) for o in report.outcomes if o.has_demand]
        jitter = rng.uniform(-0.12, 0.12, size=len(points))
        scatter.scatter(
            [p + j for (p, _s), j in zip(points, jitter, strict=True)],
            [s for _p, s in points],
            color=colour,
            s=34,
            alpha=0.85,
            label=f"{label} ({report.inversion_count} inversions of "
            f"{report.comparable_pairs} pairs)",
        )
    for report, colour in ((greedy_report, GREEDY), (exact_report, EXACT)):
        for inversion in report.inversions:
            scatter.plot(
                [inversion.higher_priority, inversion.lower_priority],
                [inversion.higher_satisfaction, inversion.lower_satisfaction],
                color=colour,
                lw=0.7,
                alpha=0.45,
            )
    scatter.set_xlabel("declared priority (1 is most urgent)")
    scatter.set_ylabel("delivered satisfaction")
    scatter.set_title(f"Promise against delivery, seed {SEED}")
    scatter.set_ylim(-0.06, 1.1)
    scatter.legend(fontsize=7, loc="lower right")

    greedy_rates = []
    exact_rates = []
    for seed in FAMILY:
        family_instance = generate_instance(seed, horizon_s=HORIZON_S)
        if not family_instance.problem.candidates:
            continue
        greedy_rates.append(
            analyse(
                family_instance.problem, schedule_greedy(family_instance.problem)
            ).inversion_rate
        )
        exact_rates.append(
            analyse(
                family_instance.problem, solve_exact(family_instance.problem)
            ).inversion_rate
        )
    bins = np.linspace(0.0, max(0.001, max(greedy_rates + exact_rates)), 16)
    hist.hist(greedy_rates, bins=bins, color=GREEDY, alpha=0.65, label="greedy")
    hist.hist(exact_rates, bins=bins, color=EXACT, alpha=0.65, label="exact")
    hist.set_xlabel("inversion rate (inversions / comparable pairs)")
    hist.set_ylabel("instances")
    hist.set_title(f"Inversion rate over {len(greedy_rates)} instances")
    hist.legend(fontsize=8)

    priorities = sorted(
        {o.priority for o in exact_report.outcomes if o.has_demand}
    )
    width = 0.36
    positions = np.arange(len(priorities))
    for offset, (report, colour, label) in enumerate(
        ((greedy_report, GREEDY, "greedy"), (exact_report, EXACT, "exact"))
    ):
        values = [report.delivered_by_priority.get(p, 0.0) for p in priorities]
        delivered.bar(
            positions + (offset - 0.5) * width,
            values,
            width=width,
            color=colour,
            label=label,
        )
    # The promise is that delivered satisfaction does not RISE with the priority
    # number, so the reference is the running maximum of what the most urgent
    # class received and not a flat line at 1.0. A bar above this staircase is a
    # class served better than a more urgent one.
    exact_values = [exact_report.delivered_by_priority.get(p, 0.0) for p in priorities]
    staircase = np.minimum.accumulate(exact_values)
    delivered.step(
        positions,
        staircase,
        where="mid",
        color=STARVED,
        ls="--",
        lw=1.2,
        label="ceiling the promise implies (exact)",
    )
    delivered.set_xticks(positions)
    delivered.set_xticklabels([f"p{p}" for p in priorities])
    delivered.set_ylim(0.0, 1.15)
    delivered.set_ylabel("mean delivered satisfaction")
    delivered.set_title("Delivered satisfaction by priority class")
    delivered.legend(fontsize=7)

    figure.tight_layout()
    path = save(figure, "priority_inversion")

    for label, report in (("greedy", greedy_report), ("exact", exact_report)):
        print(
            f"{label:<8} inversions {report.inversion_count:3d} of "
            f"{report.comparable_pairs:3d} pairs, rate {report.inversion_rate:.6f}, "
            f"worst gap {report.worst_inversion_gap:.6f}, "
            f"aggregate promise held {report.priority_monotone}"
        )
        for inversion in report.inversions:
            print(
                f"    {inversion.higher_request} (p{inversion.higher_priority}, "
                f"{inversion.higher_satisfaction:.4f}) below {inversion.lower_request} "
                f"(p{inversion.lower_priority}, {inversion.lower_satisfaction:.4f}), "
                f"gap {inversion.gap:.4f}"
            )
    print(f"family instances          : {len(greedy_rates)}")
    print(f"mean inversion rate greedy: {np.mean(greedy_rates):.6f}")
    print(f"mean inversion rate exact : {np.mean(exact_rates):.6f}")
    print(f"figure written            : {path}")


if __name__ == "__main__":
    main()
