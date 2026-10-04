"""Example 2: watching the response-time recurrence converge, and blocking bite.

Produces ../screenshots/response_time_convergence.png with two panels:

  left   the iterate sequence R^(0), R^(1), ... for each task of the standard
         set, with the deadline drawn, so the monotone convergence of the
         Audsley et al. recurrence is visible rather than asserted;
  right  the same set with a priority-ceiling blocking term swept from 0 to
         1.5 s, showing the response time of the blocked task crossing its
         deadline -- the arithmetic that makes a shared resource a timing
         problem.

Run from products/P036/:  python examples/response_time_convergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rtclock.schedulability import response_time_analysis  # noqa: E402
from rtclock.taskset import PeriodicTask, TaskSet  # noqa: E402


def standard() -> TaskSet:
    return TaskSet(
        [
            PeriodicTask("t1", 7.0, 3.0),
            PeriodicTask("t2", 12.0, 3.0),
            PeriodicTask("t3", 20.0, 5.0),
        ]
    ).rate_monotonic()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))

colors = {"t1": "#1f4e79", "t2": "#ef6c00", "t3": "#c62828"}
rts = response_time_analysis(standard())
for rt in rts:
    ks = list(range(len(rt.iterates)))
    ax1.plot(ks, rt.iterates, "o-", color=colors[rt.name], lw=1.6, ms=6,
             label=f"{rt.name}: R = {rt.response_s:g} s in {rt.iterations} iterations")
    ax1.axhline(rt.deadline_s, color=colors[rt.name], ls=":", lw=1.1, alpha=0.8)
    ax1.annotate(f"$D_{{{rt.name[-1]}}}$ = {rt.deadline_s:g}",
                 (len(rt.iterates) - 0.9, rt.deadline_s),
                 textcoords="offset points", xytext=(0, 4), fontsize=8,
                 color=colors[rt.name])
ax1.set_xlabel("iteration $m$ of $R^{(m+1)} = C + B + \\sum \\lceil R^{(m)}/T_j \\rceil C_j$")
ax1.set_ylabel("iterate $R^{(m)}$  [s]")
ax1.set_title("Monotone convergence of the RTA recurrence\n"
              "standard set: t1(7,3) t2(12,3) t3(20,5), U = 0.9286")
ax1.grid(alpha=0.3, lw=0.5)
ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
ax1.set_ylim(0, 23)

blocks = np.linspace(0.0, 7.0, 281)
series: dict[str, list[float]] = {"t1": [], "t2": [], "t3": []}
for b in blocks:
    # Blocking applied to t2 only, as a priority-ceiling term from a
    # lower-priority task sharing a semaphore whose ceiling reaches t2.
    # t2 is used rather than t3 because t3 already sits exactly on its
    # deadline at B = 0, so any blocking at all breaks it and the staircase
    # the ceiling function produces would not be visible.
    res = {r.name: r for r in response_time_analysis(standard(), blocking_s={"t2": float(b)})}
    for name in series:
        series[name].append(res[name].response_s)

for name, vals in series.items():
    ax2.plot(blocks, vals, "-", color=colors[name], lw=2.0, label=f"{name}  $R$")
ax2.axhline(12.0, color="#ef6c00", ls="--", lw=1.3)
ax2.text(6.9, 12.3, "$D_2$ = 12 s", ha="right", fontsize=9, color="#ef6c00")

crossing = next((b for b, r in zip(blocks, series["t2"], strict=True) if r > 12.0), None)
if crossing is not None:
    ax2.axvline(crossing, color="#6a1b9a", ls="-.", lw=1.3)
    ax2.annotate(f"t2 misses from\n$B_2$ = {crossing:.3f} s",
                 (crossing, 4.0), textcoords="offset points", xytext=(8, 0),
                 fontsize=8.5, color="#6a1b9a")

ax2.set_xlabel("priority-ceiling blocking term $B_2$  [s]   (Sha, Rajkumar & Lehoczky 1990)")
ax2.set_ylabel("worst-case response time $R$  [s]")
ax2.set_title("A shared resource is a timing problem\n"
              "$R_2$ against $B_2$; the staircase is the $\\lceil \\cdot \\rceil$ term")
ax2.grid(alpha=0.3, lw=0.5)
ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
ax2.set_ylim(0, 23)

fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "screenshots" / "response_time_convergence.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
print()
for rt in rts:
    print(f"{rt.name}: iterates {[f'{x:g}' for x in rt.iterates]}  "
          f"R = {rt.response_s:g} s  D = {rt.deadline_s:g} s  "
          f"met = {rt.meets_deadline}")
print()
print(f"smallest swept B_2 at which t2 misses its deadline: {crossing:.6f} s")
for b in (0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 5.0, 7.0):
    res = {r.name: r for r in response_time_analysis(standard(), blocking_s={"t2": b})}
    print(f"  B_2 = {b:.3f} s -> R_2 = {res['t2'].response_s:6.3f} s  "
          f"met = {res['t2'].meets_deadline}")
