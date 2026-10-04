"""Example 1: utilization bounds and where a real task set falls against them.

Produces ../screenshots/schedulability_bounds.png with two panels:

  left   the Liu & Layland rate-monotonic bound n(2^(1/n)-1) for n = 1..20,
         its ln 2 asymptote, and the EDF bound U = 1, with three task sets
         plotted at their own (n, U) so the gap between "sufficient" and
         "exact" is visible as area on the page;
  right  the same three sets' exact response times from RTA, as a fraction of
         their deadlines, which is the number the bounds were standing in for.

Run from products/P036/:  python examples/schedulability_bounds.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtclock.schedulability import (  # noqa: E402
    response_time_analysis,
    rm_utilization_bound,
    rm_utilization_test,
)
from rtclock.taskset import PeriodicTask, TaskSet  # noqa: E402

SETS = {
    "standard\n(7,3) (12,3) (20,5)": TaskSet(
        [PeriodicTask("t1", 7.0, 3.0), PeriodicTask("t2", 12.0, 3.0), PeriodicTask("t3", 20.0, 5.0)]
    ).rate_monotonic(),
    "harmonic U=1\n(4,1) (8,2) (16,8)": TaskSet(
        [PeriodicTask("a", 4.0, 1.0), PeriodicTask("b", 8.0, 2.0), PeriodicTask("c", 16.0, 8.0)]
    ).rate_monotonic(),
    "light\n(10,1) (20,2) (50,5)": TaskSet(
        [
            PeriodicTask("p", 10.0, 1.0),
            PeriodicTask("q", 20.0, 2.0),
            PeriodicTask("r", 50.0, 5.0),
        ]
    ).rate_monotonic(),
}

ns = list(range(1, 21))
bounds = [rm_utilization_bound(n) for n in ns]
ln2 = math.log(2.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

ax1.plot(ns, bounds, "o-", color="#1f4e79", lw=1.6, ms=4,
         label=r"RM bound $n(2^{1/n}-1)$  [Liu & Layland 1973]")
ax1.axhline(1.0, color="#2e7d32", ls="-", lw=1.4,
            label=r"EDF bound $U \leq 1$ (exact, $D=T$)")
ax1.axhline(ln2, color="#b71c1c", ls="--", lw=1.2,
            label=rf"$\ln 2 = {ln2:.6f}$ (limit as $n \to \infty$)")
ax1.fill_between(ns, bounds, 1.0, color="#ffe0b2", alpha=0.7,
                 label="bound inconclusive: run RTA")

markers = ["s", "^", "D"]
colors = ["#c62828", "#6a1b9a", "#00695c"]
for (_label, ts), m, c in zip(SETS.items(), markers, colors, strict=True):
    ax1.plot(len(ts), ts.total_utilization, m, color=c, ms=10, mew=1.2, mec="k", zorder=5)
    ax1.annotate(
        f"U = {ts.total_utilization:.4f}",
        (len(ts), ts.total_utilization),
        textcoords="offset points",
        xytext=(12, -3),
        fontsize=8.5,
        color=c,
    )

ax1.set_xlabel("number of tasks $n$")
ax1.set_ylabel("total utilization $U = \\sum C_i / T_i$")
ax1.set_title("Utilization bounds: sufficient (RM) vs exact (EDF)")
ax1.set_xlim(0.5, 20.5)
ax1.set_ylim(0.22, 1.10)
ax1.set_xticks([1, 2, 3, 5, 10, 15, 20])
ax1.grid(alpha=0.3, lw=0.5)
ax1.legend(fontsize=8, loc="lower left", framealpha=0.95)

labels: list[str] = []
offsets = [-0.26, 0.0, 0.26]
for (label, ts), c, off in zip(SETS.items(), colors, offsets, strict=True):
    rts = response_time_analysis(ts)
    fractions = [rt.response_s / rt.deadline_s for rt in rts]
    xs = [i + off for i in range(len(fractions))]
    ax2.bar(xs, fractions, width=0.24, color=c, edgecolor="k", lw=0.6,
            label=f"{label.splitlines()[0]}  (RM test: "
                  f"{'pass' if rm_utilization_test(ts).schedulable else 'inconclusive'})")
    for x, f in zip(xs, fractions, strict=True):
        ax2.text(x, f + 0.02, f"{f:.3f}", ha="center", fontsize=7.5, rotation=90)
    labels = [rt.name for rt in rts]

ax2.axhline(1.0, color="k", ls="--", lw=1.3)
ax2.text(2.35, 1.015, "deadline", ha="right", fontsize=8.5)
ax2.set_xticks(range(3))
ax2.set_xticklabels(["highest priority", "middle", "lowest priority"])
ax2.set_ylabel("$R_i / D_i$  (exact response-time analysis)")
ax2.set_title("What the bounds stood in for: exact $R/D$ per task")
ax2.set_ylim(0.0, 1.22)
ax2.grid(axis="y", alpha=0.3, lw=0.5)
ax2.legend(fontsize=8, loc="upper left", framealpha=0.95)

fig.suptitle(
    "rtclock: the RM bound rejects two of these three sets; RTA shows all three meet "
    "their deadlines",
    fontsize=10.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.955))
out = Path(__file__).resolve().parents[1] / "screenshots" / "schedulability_bounds.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
print()
print(f"RM bound n=1..10: {[f'{rm_utilization_bound(n):.6f}' for n in range(1, 11)]}")
print(f"ln 2            : {ln2:.12f}")
for label, ts in SETS.items():
    rts = response_time_analysis(ts)
    print(f"{label.replace(chr(10), ' '):<34} U = {ts.total_utilization:.6f}  "
          f"RM test {'pass' if rm_utilization_test(ts).schedulable else 'inconclusive':<12} "
          f"R/D = {[f'{rt.response_s / rt.deadline_s:.4f}' for rt in rts]}")
