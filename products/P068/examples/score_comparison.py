"""CRPS head to head, split by censoring and broken down by true shape family.

What to notice: the left panel has two winners. The learned forecaster has the
lowest CRPS on complete trajectories and the exponential baseline has the
lowest on right-censored ones, which are the majority. The right panel says
why the baselines look bad pooled: each is close to the truth on the shape it
was written for and far off everywhere else.
"""

from __future__ import annotations

import numpy as np

from _common import MODEL_COLOURS, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from degradepredict.benchmark import baseline_forecasts, run_head_to_head  # noqa: E402
from degradepredict.generate import DEFAULT_LIMIT, generate_population  # noqa: E402
from degradepredict.metrics import crps  # noqa: E402

MODELS = ("linear", "exponential", "powerlaw-fade", "learned-ipcw", "learned-complete-case")


def main() -> None:
    style()
    train = generate_population(1200, seed=20261016)
    test = generate_population(400, seed=20261017)
    truth = np.array([tr.remaining_true_days for tr in test])
    family = np.array([tr.family for tr in test])
    results = run_head_to_head(train, test, DEFAULT_LIMIT)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax = axes[0]
    x = np.arange(len(MODELS))
    comp = [results[m]["scores"]["crps_days_complete"] for m in MODELS]
    cens = [results[m]["scores"]["crps_days_censored"] for m in MODELS]
    ax.bar(x - 0.19, comp, 0.38, color="#1f6fb4", label="complete")
    ax.bar(x + 0.19, cens, 0.38, color="#d9700b", label="right-censored")
    for xi, (c, r) in enumerate(zip(comp, cens, strict=True)):
        ax.annotate(f"{c:.0f}", (xi - 0.19, c), ha="center", va="bottom", fontsize=7.5)
        ax.annotate(f"{r:.0f}", (xi + 0.19, r), ha="center", va="bottom", fontsize=7.5)
    best_c = MODELS[int(np.argmin(comp))]
    best_r = MODELS[int(np.argmin(cens))]
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=22, ha="right", fontsize=7.5)
    ax.set_ylabel("CRPS [d], log scale, lower is better")
    ax.set_title(
        f"two winners: {best_c} on complete, {best_r} on censored",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    bx = axes[1]
    families = ("linear", "exponential", "powerlaw")
    width = 0.26
    pos = np.arange(len(families))
    for i, name in enumerate(("linear", "exponential", "powerlaw-fade")):
        run = baseline_forecasts(test, DEFAULT_LIMIT, name)
        scores = crps(run.forecasts, truth)
        vals = [float(scores[family == f].mean()) for f in families]
        bx.bar(pos + (i - 1) * width, vals, width, color=MODEL_COLOURS[name], label=name)
    bx.set_yscale("log")
    bx.set_xticks(pos)
    bx.set_xticklabels([f"true shape: {f}" for f in families], fontsize=8)
    bx.set_ylabel("CRPS [d], log scale")
    bx.set_title("each baseline is good only on the shape it was written for", fontsize=10)
    bx.legend(fontsize=8)
    fig.suptitle(
        "Continuous ranked probability score of the crossing-time forecast, "
        "1200 train / 400 test, seeds 20261016 and 20261017",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    print(f"wrote {save(fig, 'score_comparison.png')}")


if __name__ == "__main__":
    main()
