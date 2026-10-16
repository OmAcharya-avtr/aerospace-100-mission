"""Interval coverage, complete against right-censored, for every model.

What to notice: the pooled bar sits between the two halves and describes
neither. For the learned forecaster the complete and censored bars differ by
more than forty points of coverage, and the pooled number is the one a report
would have quoted.
"""

from __future__ import annotations

import numpy as np

from _common import CENSORED, COMPLETE, NOMINAL, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from degradepredict.benchmark import run_head_to_head  # noqa: E402
from degradepredict.generate import DEFAULT_LIMIT, generate_population  # noqa: E402

MODELS = ("linear", "exponential", "powerlaw-fade", "learned-ipcw", "learned-complete-case")
LEVEL = 0.90


def main() -> None:
    style()
    train = generate_population(1200, seed=20261016)
    test = generate_population(400, seed=20261017)
    results = run_head_to_head(train, test, DEFAULT_LIMIT, level=LEVEL)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8),
                             gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes[0]
    x = np.arange(len(MODELS))
    width = 0.26
    complete = [results[m]["scores"]["coverage_complete"] for m in MODELS]
    censored = [results[m]["scores"]["coverage_censored"] for m in MODELS]
    pooled = [results[m]["scores"]["coverage_pooled"] for m in MODELS]
    ax.bar(x - width, complete, width, color=COMPLETE, label="complete trajectories")
    ax.bar(x, censored, width, color=CENSORED, label="right-censored trajectories")
    ax.bar(x + width, pooled, width, color="#9a9a9a", label="pooled (describes neither)")
    ax.axhline(LEVEL, color=NOMINAL, lw=1.4, ls="--", label=f"nominal {LEVEL:.2f}")
    for xi, (c, r) in enumerate(zip(complete, censored, strict=True)):
        ax.annotate(f"{c:.2f}", (xi - width, c), ha="center", va="bottom", fontsize=7.5)
        ax.annotate(f"{r:.2f}", (xi, r), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=18, ha="right", fontsize=8)
    ax.set_ylabel("interval coverage")
    ax.set_ylim(0.0, 1.18)
    ax.set_title(
        f"90 % crossing-time interval coverage, {int(results['linear']['scores']['n'])} "
        f"held-out units "
        f"({int(results['linear']['scores']['n_censored'])} censored)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper center", ncol=2)

    bx = axes[1]
    truth = np.array([tr.remaining_true_days for tr in test])
    event = np.array([tr.event for tr in test])
    bins = np.linspace(0.0, np.percentile(truth, 99), 32)
    bx.hist(truth[event], bins=bins, color=COMPLETE, alpha=0.75, label="complete")
    bx.hist(truth[~event], bins=bins, color=CENSORED, alpha=0.75, label="right-censored")
    bx.set_xlabel("true remaining life [d]")
    bx.set_ylabel("units")
    bx.set_title(
        f"the censored units are the long-lived ones\n"
        f"median {np.median(truth[event]):.0f} d complete against "
        f"{np.median(truth[~event]):.0f} d censored",
        fontsize=10,
    )
    bx.legend(fontsize=8)
    fig.tight_layout()
    print(f"wrote {save(fig, 'coverage_split.png')}")


if __name__ == "__main__":
    main()
