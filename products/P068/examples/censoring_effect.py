"""What censoring does: the Kaplan-Meier curves, the weights, and the bias.

What to notice: the left panel shows that more than half the units are still
inside their limit when the record ends, so a complete-case fit sees only the
fast ones. The middle panel is the IPCW weight each observed unit carries to
compensate. The right panel is what that compensation is worth -- it moves the
censored scores in the right direction and does not close the gap.
"""

from __future__ import annotations

import numpy as np

from _common import CENSORED, COMPLETE, LEARNED, LEARNED_ALT, add_src_to_path, save, style

add_src_to_path()

import matplotlib.pyplot as plt  # noqa: E402

from degradepredict.benchmark import evaluate_forecasts  # noqa: E402
from degradepredict.censoring import ipcw_weights, kaplan_meier  # noqa: E402
from degradepredict.generate import DEFAULT_LIMIT, generate_population  # noqa: E402
from degradepredict.learned import LearnedCrossingForecaster  # noqa: E402


def main() -> None:
    style()
    train = generate_population(1200, seed=20261016)
    test = generate_population(400, seed=20261017)
    durations = np.array([tr.remaining_observed_days for tr in train])
    events = np.array([tr.event for tr in train])
    truth = np.array([tr.remaining_true_days for tr in train])

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4))

    ax = axes[0]
    km_t = kaplan_meier(durations, events)
    km_c = kaplan_meier(durations, ~events)
    grid = np.linspace(1.0, float(durations.max()), 500)
    ax.step(km_t.times, km_t.survival, where="post", color=COMPLETE,
            label="S(t): not yet crossed")
    ax.step(km_c.times, km_c.survival, where="post", color=CENSORED,
            label="G(t): still being followed")
    ax.plot(np.sort(truth), 1.0 - np.arange(truth.size) / truth.size, color="#1a1a1a",
            lw=1.0, ls="--", label="true S(t), known only here")
    ax.set_xlim(0, float(np.percentile(durations, 99)))
    ax.set_xlabel("remaining life from the forecast origin [d]")
    ax.set_ylabel("survival")
    ax.set_title(
        f"Kaplan-Meier on {len(train)} training units\n"
        f"{int((~events).sum())} of {len(train)} right-censored "
        f"({(~events).mean():.1%})",
        fontsize=9.5,
    )
    ax.legend(fontsize=7.5)
    _ = grid

    bx = axes[1]
    w = ipcw_weights(durations, events, normalise=False)
    bx.scatter(durations[events], w[events], s=6, color=COMPLETE, label="observed crossing")
    bx.scatter(durations[~events], w[~events], s=6, color=CENSORED,
               label="censored (weight 0)")
    bx.set_xlabel("observed duration [d]")
    bx.set_ylabel("IPCW weight 1 / G(T-)")
    bx.set_title("late events carry more weight\nbecause they survived more censoring",
                 fontsize=9.5)
    bx.legend(fontsize=7.5)

    cx = axes[2]
    rows = []
    for label, weighting, colour in (("complete-case", "none", LEARNED_ALT),
                                     ("IPCW", "ipcw", LEARNED)):
        model = LearnedCrossingForecaster(limit=DEFAULT_LIMIT, weighting=weighting).fit(train)
        scores = evaluate_forecasts(model.predict_many(test), test, level=0.90)
        rows.append((label, colour, scores))
    keys = ("coverage_complete", "coverage_censored")
    pos = np.arange(len(keys))
    for i, (label, colour, scores) in enumerate(rows):
        vals = [scores[k] for k in keys]
        cx.bar(pos + (i - 0.5) * 0.36, vals, 0.36, color=colour, label=label)
        for p, v in zip(pos, vals, strict=True):
            cx.annotate(f"{v:.3f}", (p + (i - 0.5) * 0.36, v), ha="center", va="bottom",
                        fontsize=7.5)
    cx.axhline(0.90, color="#d9a40b", lw=1.3, ls="--", label="nominal 0.90")
    cx.set_xticks(pos)
    cx.set_xticklabels(["complete", "right-censored"], fontsize=8)
    cx.set_ylim(0, 1.0)
    cx.set_ylabel("interval coverage")
    cx.set_title("what the weighting buys", fontsize=9.5)
    cx.legend(fontsize=7.5)
    fig.suptitle(
        "Right censoring: the estimator, the weights, and the residual bias they "
        "do not remove",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    print(f"wrote {save(fig, 'censoring_effect.png')}")


if __name__ == "__main__":
    main()
