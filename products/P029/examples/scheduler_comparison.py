"""Fixed-threshold against learned desaturation scheduling, on one held-out episode.

Run: ``python3 scheduler_comparison.py``   (about one minute on two cores)
Writes ``../screenshots/scheduler_comparison.png``.

A model is trained here so the example is self-contained: 44 fitting episodes and 18
knob-tuning episodes, against the 60 plus 25 used for the numbers in
``validation/learned_vs_fixed_ci.py`` and ``MODEL_CARD.md``, and 30 held-out episodes
against 80 there. Treat the figure as an illustration of the behaviour and the validation
script as the evidence; with a smaller fitting set than this the learned scheduler loses
to the baseline on combined cost, which is itself worth knowing.
"""

from __future__ import annotations

import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from momentummgr import (  # noqa: E402
    evaluate_policy,
    rollout,
    sample_episode,
    tune_fixed_threshold,
)
from momentummgr.learned import train_scheduler  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "screenshots"
OUT.mkdir(exist_ok=True)

t0 = time.time()
fit = [sample_episode(s) for s in range(1000, 1044)]
tune = [sample_episode(s) for s in range(2000, 2018)]
held = [sample_episode(s) for s in range(5000, 5030)]
baseline, base_cost, _ = tune_fixed_threshold(fit + tune)
scheduler, diag, _, _ = train_scheduler(fit, tune, fallback=baseline)
print(f"baseline: on = {baseline.on_fraction:.2f}, off = {baseline.off_fraction:.2f}")
print(f"learned:  threshold = {diag['decision_threshold']:.2f}, "
      f"deferral band = {diag['min_confidence']:.2f}, rows = {int(diag['n_rows'])}")

base_m = evaluate_policy(baseline, held)
learn_m = evaluate_policy(scheduler, held)
print(f"\n30 held-out episodes  {'baseline':>12}{'learned':>12}")
print("-" * 56)
for name, label in (("duty_fraction", "magnetorquer duty"),
                    ("near_saturation_fraction", "time near saturation"),
                    ("max_h_fraction", "peak |h| / envelope"),
                    ("cost", "combined cost")):
    b = float(np.mean([getattr(m, name) for m in base_m]))
    ln = float(np.mean([getattr(m, name) for m in learn_m]))
    print(f"{label:<32}{b:>12.5f}{ln:>12.5f}")

episode = held[3]
roll_b = rollout(episode, baseline.decider())
roll_l = rollout(episode, scheduler.decider())
hours = roll_b.time_s / 3600.0
env = episode.envelope_nms

fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), sharex=True,
                         gridspec_kw={"height_ratios": [2.0, 1.2, 1.2]})

ax = axes[0]
ax.plot(hours, np.linalg.norm(roll_b.h_history_nms, axis=1) / env, lw=1.6,
        label=f"fixed threshold, duty {roll_b.metrics.duty_fraction:.3f}")
ax.plot(hours, np.linalg.norm(roll_l.h_history_nms, axis=1) / env, lw=1.6,
        label=f"learned, duty {roll_l.metrics.duty_fraction:.3f}")
ax.axhline(1.0, color="crimson", ls="--", lw=1.4, label="wheel envelope")
ax.axhline(0.8, color="darkorange", ls=":", lw=1.4, label="near-saturation level")
ax.set_ylabel(r"$|h_{wheel}| \, / \, h_{env}$")
ax.set_title(f"Episode seed {episode.seed}: {episode.n_windows} windows of "
             f"{episode.window_s:.0f} s, envelope {env * 1e3:.1f} mN m s")
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(hours[:-1], np.linalg.norm(roll_b.dipole_history_am2, axis=1), lw=1.2,
        label="fixed threshold")
ax.plot(hours[:-1], np.linalg.norm(roll_l.dipole_history_am2, axis=1), lw=1.2,
        label="learned")
ax.axhline(episode.max_dipole_am2, color="0.4", ls="--", lw=1.0, label="dipole limit")
ax.set_ylabel(r"$|m|$ [A m$^2$]")
ax.legend(fontsize=9, ncol=3)
ax.grid(alpha=0.3)

ax = axes[2]
window_hours = (np.arange(episode.n_windows) + 0.5) * episode.window_s / 3600.0
merit = roll_l.features[:, 7]
ax.plot(window_hours, merit, color="0.35", lw=1.2,
        label=r"dumping figure of merit $|B|\sin\theta$ (scaled)")
ax.scatter(window_hours[roll_l.actions], merit[roll_l.actions], s=34, color="tab:orange",
           zorder=3, label="learned: dump")
ax.scatter(window_hours[roll_b.actions], np.full(int(roll_b.actions.sum()), -0.05), s=26,
           marker="s", color="tab:blue", zorder=3, label="fixed threshold: dump")
ax.set_xlabel("time [h]")
ax.set_ylabel("merit / actions")
ax.legend(fontsize=9, ncol=3)
ax.grid(alpha=0.3)

fig.suptitle("momentummgr: desaturation scheduling, fixed threshold against a learned "
             "scheduler", fontsize=12)
fig.tight_layout()
path = OUT / "scheduler_comparison.png"
fig.savefig(path, dpi=130)
print("\nSmaller fitting set than the validation run, so these means are illustrative.")
print("The evidence, with confidence intervals, is validation/learned_vs_fixed_ci.py.")
print(f"\nwrote {path}")
print(f"wall time {time.time() - t0:.1f} s")
