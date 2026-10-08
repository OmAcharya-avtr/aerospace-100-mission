"""What a counterexample looks like: the trace that violates, beside one that does not.

For two instances, the surrogate-guided search is run until it finds a
violation, and the violating trace is plotted against the trace at the box
centre, which satisfies. The requirement's bound and its time window are drawn
on, so the reader can see where and by how much the requirement failed.

Saves ``../screenshots/counterexample_trace.png``.

A counterexample is a fact about the simulator at one setting. It is not a
statement about any vehicle, and the simulator is a synthetic benchmark whose
parameter values were not identified from one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from _bootstrap import add_src_to_path  # noqa: E402

ROOT = add_src_to_path()

from falsifyloop.instances import instance  # noqa: E402
from falsifyloop.search import surrogate_guided  # noqa: E402
from falsifyloop.systems import LoopInput  # noqa: E402

SHOWN = (
    ("attitude-envelope", "theta", 17.0, "attitude |theta| [deg]", (0.0, 2.0)),
    ("rate-envelope", "q", 75.0, "attitude rate |q| [deg/s]", (0.0, 2.0)),
)
BUDGET = 400
SEED = 5


def main() -> int:
    fig, axes = plt.subplots(2, len(SHOWN), figsize=(7.0 * len(SHOWN), 8.4), sharex=True)
    report = []
    for column, (identifier, signal, bound, ylabel, window) in enumerate(SHOWN):
        inst = instance(identifier)
        found = surrogate_guided(inst, BUDGET, SEED)
        if not found.found:
            raise RuntimeError(f"no counterexample found for {identifier} within {BUDGET}")
        bad = inst.simulate(found.best_vector)
        good = inst.simulate(inst.centre())

        top = axes[0, column]
        for trace, label, colour in (
            (good, "box centre (satisfies)", "#1f77b4"),
            (bad, f"counterexample (sim {found.first_violation})", "#d62728"),
        ):
            top.plot(trace.times, np.abs(trace.signal(signal)), color=colour, lw=1.8, label=label)
        top.axhline(bound, color="black", ls="--", lw=1.3, label=f"bound {bound:g}")
        top.axvspan(*window, color="#999999", alpha=0.12)
        top.set_ylabel(ylabel)
        top.legend(fontsize=9, frameon=False, loc="upper right")
        top.grid(alpha=0.25)
        top.set_title(
            f"{identifier} [{inst.tier}]\n"
            f"{inst.requirement}  -  robustness {found.best_robustness:+.4f}",
            fontsize=10,
        )

        bottom = axes[1, column]
        for trace, label, colour in (
            (good, "box centre", "#1f77b4"),
            (bad, "counterexample", "#d62728"),
        ):
            bottom.plot(
                trace.times, trace.signal("delta"), color=colour, lw=1.6, label=f"{label} delta"
            )
            bottom.plot(
                trace.times,
                trace.signal("cmd"),
                color=colour,
                lw=1.0,
                ls=":",
                label=f"{label} command",
            )
        bottom.axhline(20.0, color="black", ls="--", lw=1.0)
        bottom.axhline(-20.0, color="black", ls="--", lw=1.0)
        bottom.set_xlabel("time [s]")
        bottom.set_ylabel("actuator deflection [deg]")
        bottom.legend(fontsize=8, frameon=False, ncol=2)
        bottom.grid(alpha=0.25)
        bottom.set_title(
            "Actuator: solid is the deflection, dotted the command. "
            "Dashed lines are the 20 deg limit.",
            fontsize=9.5,
        )

        peak = float(np.max(np.abs(bad.signal(signal))))
        report.append((identifier, found.first_violation, peak, bound, found.best_vector))

    fig.suptitle(
        "A counterexample, and a setting that satisfies, on the same axes. "
        "The simulator is a synthetic benchmark: no parameter in it was identified "
        "from any aircraft.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = ROOT / "screenshots" / "counterexample_trace.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print(f"wrote {out.relative_to(ROOT)}")
    for identifier, where, peak, bound, vector in report:
        print("")
        print(f"{identifier}: violation found at simulation {where}")
        print(f"  peak magnitude {peak:.4f} against the bound {bound:g}")
        for name, value in zip(LoopInput.FIELDS, vector, strict=True):
            print(f"  {name:<16s} {value:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
