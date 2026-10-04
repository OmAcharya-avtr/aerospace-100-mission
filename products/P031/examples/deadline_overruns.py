"""Show the two overrun definitions diverging on one injected trace.

Produces ``screenshots/deadline_overruns.png``: the per-iteration durations
against the deadline, the completion times against the release-time deadlines,
the lateness, and a bar comparison of the two counts. The point of the figure
is that "how many overruns did the loop have" has two answers and they differ
by a factor on a loaded trace.

Run: ``python examples/deadline_overruns.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hilforge.backends import make_backend_pair
from hilforge.loop import HilLoop, LoopConfig
from hilforge.predict import TraceConfig, generate_trace
from hilforge.timing import PeriodSpec, overrun_report

PERIOD = 0.010
N = 320
SEED = 51515
OUT = ROOT / "screenshots" / "deadline_overruns.png"


def main() -> int:
    cfg = TraceConfig.preset("bursty", n_iterations=N + 200)
    trace = generate_trace(cfg, seed=SEED)
    durations = trace.totals_s[200 : 200 + N]
    account = overrun_report(durations, PERIOD)

    sim, _ = make_backend_pair(seed=SEED, sample_dt_s=PERIOD)
    record = HilLoop(
        sim,
        LoopConfig(
            period=PeriodSpec(period_s=PERIOD, cascade_limit=0),
            n_iterations=N,
            injected_durations_s=tuple(durations),
        ),
    ).run()

    print(f"period                   : {PERIOD:.6e} s")
    print(f"iterations               : {N}")
    print(f"mean duration            : {float(durations.mean()):.6e} s")
    print(f"utilisation E[d]/T       : {float(durations.mean()) / PERIOD:.6f}")
    print(f"direct overruns          : {account.direct_count} "
          f"({account.direct_rate:.4f} per iteration)")
    print(f"cascade overruns         : {account.cascade_count} "
          f"({account.cascade_rate:.4f} per iteration)")
    print(f"max consecutive cascade  : {account.max_consecutive_cascade} "
          f"(starting at index {account.first_cascade_run_start})")
    print(f"worst lateness           : {float(np.max(account.lateness_s)):.6e} s")
    print(f"loop agrees with the account: "
          f"{record.overruns.as_dict() == account.as_dict()}")
    print(f"figure                   : {OUT}")

    idx = np.arange(N)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
    fig.suptitle(
        "HilForge: direct and cascade deadline overruns on one injected trace",
        fontsize=12,
    )

    ax = axes[0, 0]
    ax.plot(idx, durations * 1e3, lw=0.9, color="#44618c")
    ax.axhline(PERIOD * 1e3, color="k", ls="--", lw=1.2, label="deadline D = T")
    direct = np.asarray(account.direct_indices, dtype=int)
    if direct.size:
        ax.plot(direct, durations[direct] * 1e3, "o", ms=3.0, color="#b3412c",
                label=f"direct overrun ({direct.size})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("duration [ms]")
    ax.set_title("iteration duration vs the deadline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    releases = idx * PERIOD
    deadlines = releases + PERIOD
    ax.plot(idx, (np.asarray(account.completion_s) - releases) * 1e3, lw=0.9,
            color="#2f6f4e", label="completion − release")
    ax.plot(idx, (deadlines - releases) * 1e3, "k--", lw=1.2, label="deadline − release")
    cascade = np.asarray(account.cascade_indices, dtype=int)
    if cascade.size:
        ax.plot(
            cascade,
            (np.asarray(account.completion_s)[cascade] - releases[cascade]) * 1e3,
            "o", ms=3.0, color="#b3412c",
            label=f"cascade overrun ({cascade.size})",
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel("time since release [ms]")
    ax.set_title("completion on the release-time timeline:\nlateness carries forward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    lateness = np.asarray(account.lateness_s) * 1e3
    ax.fill_between(idx, 0, np.maximum(lateness, 0), color="#b3412c", alpha=0.75,
                    label="late")
    ax.fill_between(idx, np.minimum(lateness, 0), 0, color="#44618c", alpha=0.6,
                    label="early")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("lateness [ms]")
    ax.set_title("lateness c[i] − D[i]: positive is a missed deadline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    names = ["direct\nd[i] > D", "cascade\nc[i] > D[i]"]
    values = [account.direct_count, account.cascade_count]
    bars = ax.bar(names, values, color=["#44618c", "#b3412c"])
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f" {value}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("overruns in 320 iterations")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title(
        f"the same run, two definitions\n"
        f"ratio cascade/direct = {values[1] / max(values[0], 1):.2f}"
    )
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
