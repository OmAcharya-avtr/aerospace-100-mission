"""What the per-command static lint misses, per defect family, measured.

Two panels. The first is the miss rate by family at both lint strengths, with
the forward walk's violation count on each bar so the reader can see the
denominator. The second is the other half of the story and the one that decides
whether anybody keeps running the tool: the lint emits far more findings than
there are violations, because it re-evaluates every command against the
declared initial state.
"""

from __future__ import annotations

import _common
from _common import FORWARD, LINT, REFUSE, save, style

_common.bootstrap()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cmdguard.benchmark import lint_miss_rate  # noqa: E402
from cmdguard.dataset import build_corpus  # noqa: E402
from cmdguard.lint import LintStrength  # noqa: E402

N_SEEDS = 60


def main() -> None:
    style()
    corpus = build_corpus(range(N_SEEDS), include_clean=True)
    weak = lint_miss_rate(corpus, LintStrength.PER_COMMAND)
    strong = lint_miss_rate(corpus, LintStrength.WITH_TIMING)
    families = [name for name in sorted(weak.families) if weak.families[name].n_truth > 0]
    positions = np.arange(len(families))

    figure = plt.figure(figsize=(11.0, 7.0))
    grid = figure.add_gridspec(2, 1, height_ratios=[1.3, 1.0], hspace=0.55)

    axis = figure.add_subplot(grid[0])
    weak_rates = [weak.families[name].miss_rate for name in families]
    strong_rates = [strong.families[name].miss_rate for name in families]
    axis.bar(positions - 0.19, weak_rates, width=0.36, color=LINT,
             label="per-command lint")
    axis.bar(positions + 0.19, strong_rates, width=0.36, color=FORWARD,
             label="per-command lint plus a timing pass")
    axis.axhline(weak.total.miss_rate, color=LINT, linestyle="--", linewidth=1.0)
    axis.axhline(strong.total.miss_rate, color=FORWARD, linestyle="--", linewidth=1.0)
    axis.annotate(f"all families {weak.total.miss_rate:.3f}",
                  (len(families) - 0.5, weak.total.miss_rate), fontsize=7,
                  color=LINT, va="bottom", ha="right")
    axis.annotate(f"all families {strong.total.miss_rate:.3f}",
                  (len(families) - 0.5, strong.total.miss_rate), fontsize=7,
                  color=FORWARD, va="top", ha="right")
    for index, name in enumerate(families):
        axis.annotate(f"n={weak.families[name].n_truth}", (index, 0.02),
                      ha="center", fontsize=6.5, color="#333333")
    axis.set_xticks(positions)
    axis.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    axis.set_ylabel("fraction of violations missed")
    axis.set_ylim(0.0, 1.12)
    axis.legend(loc="upper left", fontsize=8)
    axis.set_title(
        f"static-lint miss rate against forward execution\n"
        f"{N_SEEDS} seeds x {len(families)} defect families; n is the violation count "
        "in that family, and undeclared_typo is the control",
        fontsize=10,
    )

    axis = figure.add_subplot(grid[1])
    findings = [strong.families[name].n_lint_findings for name in families]
    truth = [strong.families[name].n_truth for name in families]
    axis.bar(positions - 0.19, truth, width=0.36, color=FORWARD,
             label="violations found by forward execution")
    axis.bar(positions + 0.19, findings, width=0.36, color=REFUSE,
             label="findings emitted by the lint")
    axis.set_xticks(positions)
    axis.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    axis.set_ylabel("count")
    axis.set_yscale("log")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2,
                fontsize=8)
    axis.set_title(
        f"the other half: the lint emits {strong.total.n_lint_findings} findings against "
        f"{strong.total.n_truth} violations,\n"
        f"of which {strong.total.n_lint_false_positive} are not confirmed by the forward "
        f"walk ({strong.total.false_positive_rate:.3f} of all findings). log scale",
        fontsize=10,
    )

    print(f"wrote {save(figure, 'lint_miss_rate')}")
    print(f"per_command  miss rate {weak.total.miss_rate:.6f} "
          f"({weak.total.n_missed}/{weak.total.n_truth}), "
          f"ordering-only {weak.total.ordering_miss_rate:.6f}")
    print(f"with_timing  miss rate {strong.total.miss_rate:.6f} "
          f"({strong.total.n_missed}/{strong.total.n_truth}), "
          f"ordering-only {strong.total.ordering_miss_rate:.6f}")
    print(f"with_timing  false-positive rate {strong.total.false_positive_rate:.6f} "
          f"({strong.total.n_lint_false_positive}/{strong.total.n_lint_findings})")


if __name__ == "__main__":
    main()
