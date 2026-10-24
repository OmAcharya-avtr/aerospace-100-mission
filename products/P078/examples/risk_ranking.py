"""What the ranker actually hands a reviewer, on one sequence.

The top panel is the per-step risk with the forest's per-tree dispersion drawn
as an error bar, and the steps that really are involved in a violation marked.
The bottom panel is what a reviewer with a budget of five commands would read
under each ordering, and which of those five were worth reading.

The dispersion is drawn because the model exposes it, not because it is
calibrated. It is the spread of an ensemble, and the calibration measurement in
``validation/validate_ranker_output.txt`` shows the risk is not a probability.
"""

from __future__ import annotations

import _common
from _common import FLAG, LEARNED, RANDOM, REFUSE, RESOURCE, save, style

_common.bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cmdguard.dataset import build_corpus, generate_clean_case  # noqa: E402
from cmdguard.defects import inject  # noqa: E402
from cmdguard.ranker import (  # noqa: E402
    RiskRanker,
    involved_mask,
    irreversibility_ordering,
    precision_at_k,
    random_ordering,
    resource_draw_ordering,
)
from cmdguard.trace import forward_execute  # noqa: E402

TRAIN_SEEDS = range(0, 40)
PROBE_SEED = 100_003
PROBE_FAMILY = "swap_enabler"
K = 5


def main() -> None:
    style()
    train = build_corpus(TRAIN_SEEDS, include_clean=False)
    ranker = RiskRanker().fit([(entry.case, entry.report) for entry in train])
    assert ranker.metrics is not None

    case, record = inject(generate_clean_case(PROBE_SEED), PROBE_FAMILY, PROBE_SEED)
    report = forward_execute(case)
    scores = ranker.score(case)
    labels = involved_mask(report, len(case.sequence))
    steps = case.sequence.steps

    figure = plt.figure(figsize=(12.0, 7.6))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.62)

    axis = figure.add_subplot(grid[0])
    positions = np.arange(len(steps))
    colours = [REFUSE if involved else LEARNED for involved in labels]
    axis.bar(positions, scores.risk, color=colours, width=0.7)
    axis.errorbar(positions, scores.risk, yerr=scores.spread, fmt="none",
                  ecolor="#333333", capsize=2, linewidth=0.8)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{index} {step.command}" for index, step in enumerate(steps)],
        rotation=60,
        ha="right",
        fontsize=6.2,
    )
    axis.set_ylabel("learned risk (forest mean)")
    axis.set_ylim(0.0, 1.05)
    axis.set_title(
        f"per-step risk on one defective sequence ({record.family}, seed {PROBE_SEED})\n"
        f"dark red: the {int(labels.sum())} steps a reviewer has to read to understand "
        f"the {report.n_violations} violation(s). error bar: per-tree dispersion, not a "
        "calibrated interval",
        fontsize=9.5,
    )

    axis = figure.add_subplot(grid[1])
    orderings = (
        ("learned ranker", scores.order(), LEARNED),
        ("irreversibility flag", irreversibility_ordering(case), FLAG),
        ("resource draw", resource_draw_ordering(case), RESOURCE),
        ("random control", random_ordering(case, 4711), RANDOM),
    )
    axis.set_yticks(range(len(orderings)))
    axis.set_yticklabels([label for label, _, _ in orderings], fontsize=8)
    axis.set_xticks(positions)
    axis.set_xticklabels([str(index) for index in positions], fontsize=6.5)
    axis.set_xlabel("step index")
    # Room on the right for the per-ordering precision label, which otherwise
    # lands on top of the markers at the end of the sequence.
    axis.set_xlim(-0.8, len(steps) + 7.0)
    axis.set_ylim(-0.8, len(orderings) - 0.2)
    axis.grid(axis="x")
    for row, (_label, order, colour) in enumerate(orderings):
        chosen = list(order)[:K]
        precision = precision_at_k(order, labels, K)
        for rank, index in enumerate(chosen, start=1):
            hit = bool(labels[index])
            axis.plot([index], [row], marker="o" if hit else "x", markersize=9,
                      color=REFUSE if hit else colour,
                      markeredgewidth=1.6, linestyle="")
            axis.annotate(str(rank), (index, row), textcoords="offset points",
                          xytext=(0, 9), ha="center", fontsize=6.0, color="#333333")
        axis.annotate(
            f"precision at {K} = {precision:.3f}" if precision == precision else "undefined",
            (len(steps) + 0.4, row),
            ha="left",
            va="center",
            fontsize=7.5,
            color=colour,
        )
    for index in np.flatnonzero(labels):
        axis.axvline(index, color=REFUSE, linewidth=0.8, alpha=0.35)
    axis.set_title(
        f"the first {K} commands each ordering would put in front of a reviewer\n"
        "filled circle: the step really is involved in a violation. cross: it is not. "
        "the vertical lines are the involved steps",
        fontsize=9.5,
    )

    print(f"wrote {save(figure, 'risk_ranking')}")
    print(f"case                 : {case.name}")
    print(f"defect injected      : {record.render()}")
    print(f"violations           : {report.n_violations} {report.kind_counts()}")
    print(f"steps involved       : {report.involved_steps()}")
    for label, order, _ in orderings:
        print(f"{label:<22s} top {K}: {list(order)[:K]}  "
              f"precision {precision_at_k(order, labels, K):.6f}")
    print(f"training rows        : {ranker.metrics.n_rows}, positive fraction "
          f"{ranker.metrics.positive_fraction:.6f}")


if __name__ == "__main__":
    main()
