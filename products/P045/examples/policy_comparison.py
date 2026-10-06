"""The learned redundancy policy against its three baselines, both fade regimes.

Left panel: cost per frame on the reporting seeds, lower is better, with
standard errors across seeds.  The cost is elapsed symbol times plus the drop
penalty charged for a frame still undecoded after the last round; it is the
quantity the learned policy is trained to minimise.

Right panel: goodput on the same runs, higher is better.

Every free parameter of every policy was chosen on the tune seeds, which are
disjoint from both the fit seeds the learned model trained on and the report
seeds plotted here.  Without that, the comparison is rigged for the learned
model, because it is the only policy that would have had a fitting pass.

What to notice: the tuned fixed schedule wins in both regimes.  The learned
policy beats the analytic fixed schedule comfortably -- because computing the
optimum at the stationary *mixture* error probability is not the same as
optimising the expected cost of a two-state channel -- and loses to a two-
parameter grid search.  That is the published result.  It is not retuned and it
is not removed.

Writes ../screenshots/policy_comparison.png.  Runtime: about 200 s on one core.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)

from arqlonghaul.datasets import SEED_SPLIT, long_burst_env, short_burst_env  # noqa: E402
from arqlonghaul.policy import (  # noqa: E402
    LearnedRedundancyPolicy,
    analytic_fixed,
    collect_transitions,
    evaluate,
    tune_escalating,
    tune_fixed,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "screenshots", "policy_comparison.png")

FIT_SEEDS = SEED_SPLIT.fit[:100]
TUNE_SEEDS = SEED_SPLIT.tune[:30]
REPORT_SEEDS = SEED_SPLIT.report[:200]
FRAMES = 120
ORDER = ("analytic-fixed", "escalating", "learned-forest", "tuned-fixed")


def main() -> None:
    envs = (long_burst_env(), short_burst_env())
    results: dict[str, dict[str, dict[str, float]]] = {}
    for env in envs:
        print(f"=== {env.name}")
        policies = {}
        policies["analytic-fixed"] = analytic_fixed(env)[0]
        policies["tuned-fixed"] = tune_fixed(env, TUNE_SEEDS, FRAMES)[0]
        policies["escalating"] = tune_escalating(env, TUNE_SEEDS, FRAMES)[0]
        tr = collect_transitions(env, FIT_SEEDS, FRAMES, rng_seed=7)
        learned = LearnedRedundancyPolicy(
            len(env.actions), n_estimators=140, max_depth=10, min_samples_leaf=20
        )
        learned.fit(tr["features"], tr["actions"], tr["cost"])
        policies["learned-forest"] = learned
        env_res = {}
        for name in ORDER:
            r = evaluate(env, policies[name], REPORT_SEEDS, FRAMES)
            env_res[name] = r
            print(
                f"  {name:<16} cost {r['cost_per_frame']:8.2f} "
                f"+-{r['cost_stderr']:.2f}  goodput {r['goodput']:.5f} "
                f"+-{r['goodput_stderr']:.5f}  residual {r['residual_fer']:.3e}"
            )
        results[env.name] = env_res

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    width = 0.36
    x = np.arange(len(ORDER))
    colours = ("#7a7a7a", "#c98a1c")
    for i, (env_name, env_res) in enumerate(results.items()):
        ax.bar(
            x + (i - 0.5) * width,
            [env_res[n]["cost_per_frame"] for n in ORDER],
            width,
            yerr=[env_res[n]["cost_stderr"] for n in ORDER],
            capsize=3,
            color=colours[i],
            label=env_name,
        )
        ax2.bar(
            x + (i - 0.5) * width,
            [env_res[n]["goodput"] for n in ORDER],
            width,
            yerr=[env_res[n]["goodput_stderr"] for n in ORDER],
            capsize=3,
            color=colours[i],
            label=env_name,
        )
    for axis, label, title in (
        (ax, "cost per frame (symbol times, lower better)",
         "Surrogate cost: elapsed symbols plus drop penalty"),
        (ax2, "goodput (information symbols per symbol time)",
         "Goodput on the reporting seeds"),
    ):
        axis.set_xticks(x)
        axis.set_xticklabels(ORDER, rotation=20, ha="right", fontsize=8)
        axis.set_ylabel(label)
        axis.set_title(title)
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=8)
    ax2.set_ylim(0.0, max(
        r["goodput"] for env_res in results.values() for r in env_res.values()
    ) * 1.25)
    fig.suptitle(
        f"Learned redundancy policy against three baselines, "
        f"{len(REPORT_SEEDS)} report seeds x {FRAMES} frames; "
        "every free parameter tuned on disjoint seeds",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote screenshots/{os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
