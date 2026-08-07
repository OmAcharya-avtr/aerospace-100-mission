"""Example 3 - reacquisition-time comparison: learned policy vs both baselines.

Generates ``screenshots/ex03_reacq_comparison.png``: a bar chart of mean
time-to-reacquire with 95 % confidence intervals, the survival curves of the
reacquisition time, and the action mix chosen by each policy.

Run from the product root:  python examples/ex03_reacq_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackbench.reacq import (  # noqa: E402
    ACTIONS,
    AlwaysFullPolicy,
    AlwaysLocalPolicy,
    ReacqConfig,
    ReacqEnv,
    evaluate_policy,
    train_q_learning,
)

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "ex03_reacq_comparison.png"
TRAIN_EPISODES = 20_000
EVAL_EPISODES = 2_000
EVAL_SEED = 999


def episode_times(policy, cfg: ReacqConfig, n: int, seed: int) -> np.ndarray:
    """Per-episode reacquisition times [s] under common random numbers."""
    env = ReacqEnv(cfg)
    out = np.zeros(n)
    for i in range(n):
        s = env.reset(seed=seed + i)
        while True:
            s, _, done, _ = env.step(policy.act(s))
            if done:
                out[i] = min(env.t, cfg.max_time)
                break
    return out


def main() -> int:
    """Build the figure and save it."""
    cfg = ReacqConfig()
    learned = train_q_learning(cfg, episodes=TRAIN_EPISODES, seed=12345)
    policies = {
        "always-full\n(baseline)": AlwaysFullPolicy(),
        "always-local\n(baseline)": AlwaysLocalPolicy(),
        "Q-learning\n(learned)": learned,
    }
    stats = {k: evaluate_policy(p, cfg, EVAL_EPISODES, EVAL_SEED)
             for k, p in policies.items()}
    times = {k: episode_times(p, cfg, EVAL_EPISODES, EVAL_SEED)
             for k, p in policies.items()}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#8c8c8c", "#4c72b0", "#c44e52"]

    ax = axes[0]
    names = list(stats)
    means = [stats[k]["mean_time_s"] for k in names]
    lo = [stats[k]["mean_time_s"] - stats[k]["ci_low_s"] for k in names]
    hi = [stats[k]["ci_high_s"] - stats[k]["mean_time_s"] for k in names]
    bars = ax.bar(names, means, yerr=[lo, hi], capsize=6, color=colors, alpha=0.9)
    for b, k in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.35,
                f"{stats[k]['mean_time_s']:.2f} s\nsuccess "
                f"{stats[k]['success_rate']:.1%}",
                ha="center", fontsize=8)
    ax.set_ylabel("mean time to reacquire [s]")
    ax.set_ylim(0, max(means) * 1.35)
    ax.set_title(f"Mean time-to-reacquire, {EVAL_EPISODES} seeded episodes\n"
                 "error bars = 95 % CI (normal approximation)", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1]
    for (k, t), c in zip(times.items(), colors):
        ts = np.sort(t)
        surv = 1.0 - np.arange(1, ts.size + 1) / ts.size
        ax.step(ts, surv, where="post", color=c, lw=1.4,
                label=k.replace("\n", " "))
    ax.set_xscale("log")
    ax.set_xlabel("time to reacquire [s]")
    ax.set_ylabel("P(time > t)")
    ax.set_title("Survival curve of the reacquisition time\n"
                 f"(right edge = {cfg.max_time:.0f} s timeout)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    width = 0.25
    x = np.arange(len(ACTIONS))
    for i, (k, c) in enumerate(zip(names, colors)):
        mix = stats[k]["action_mix"]
        total = max(sum(mix.values()), 1)
        ax.bar(x + (i - 1) * width, [mix[a] / total for a in ACTIONS],
               width, color=c, alpha=0.9, label=k.replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(ACTIONS)
    ax.set_ylabel("fraction of attempts")
    ax.set_title("Action mix\n(the learned policy prefers expanding rings)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("TrackBench - reacquisition policy benchmark "
                 f"(tabular Q-learning, {TRAIN_EPISODES} training episodes, seed 12345)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    for k in names:
        s = stats[k]
        sys.stdout.write(
            f"{k.replace(chr(10), ' '):>26}: mean {s['mean_time_s']:.4f} s "
            f"[{s['ci_low_s']:.4f}, {s['ci_high_s']:.4f}], median "
            f"{s['median_time_s']:.4f} s, success {s['success_rate']:.3f}\n"
        )
    sys.stdout.write(f"saved {OUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
