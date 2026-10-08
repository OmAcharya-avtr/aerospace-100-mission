"""The requirement language itself: robustness as a signal over time.

For one trace, the robustness of each sub-formula is plotted as a function of
time alongside the signals it reads, and the Boolean semantics is overlaid as a
shaded band. The two must agree in sign at every sample, and the figure is where
that agreement becomes visible rather than only tested.

Saves ``../screenshots/requirement_monitor.png``.
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

from falsifyloop.requirements import (  # noqa: E402
    Abs,
    Always,
    Eventually,
    Predicate,
    Signal,
    robustness,
    satisfies,
)
from falsifyloop.systems import LoopInput, simulate  # noqa: E402

# A setting that overshoots and rings: large step, high gain, low rate feedback,
# slow actuator, and a gust near the closed-loop natural frequency.
SETTING = LoopInput(5.5, 1.85, 0.28, 2.6, 12.0, 0.85)


def main() -> int:
    trace = simulate(SETTING)
    error = Abs(Signal("error"))
    formulas = [
        ("|error| <= 3 deg (pointwise)", Predicate(error, "<=", 3.0, scale=3.0)),
        ("always[1.2,2] |error| <= 3", Always(Predicate(error, "<=", 3.0, scale=3.0), 1.2, 2.0)),
        (
            "eventually[0,1] always[0,0.5] |error| <= 3",
            Eventually(Always(Predicate(error, "<=", 3.0, scale=3.0), 0.0, 0.5), 0.0, 1.0),
        ),
    ]

    fig, axes = plt.subplots(len(formulas) + 1, 1, figsize=(11, 10.5), sharex=True)

    top = axes[0]
    top.plot(trace.times, trace.signal("theta"), color="#1f77b4", lw=1.8, label="theta [deg]")
    top.axhline(
        SETTING.step_amplitude, color="#555555", ls=":", lw=1.2, label="commanded step"
    )
    top.plot(
        trace.times, np.abs(trace.signal("error")), color="#d62728", lw=1.4, label="|error| [deg]"
    )
    top.axhline(3.0, color="black", ls="--", lw=1.2, label="3 deg band")
    top.set_ylabel("deg")
    top.legend(fontsize=9, frameon=False, ncol=4)
    top.grid(alpha=0.25)
    top.set_title(
        "One simulated trace. Below: the robustness of three requirements over it, "
        "as a signal.",
        fontsize=11,
    )

    for ax, (label, formula) in zip(axes[1:], formulas, strict=True):
        rho = formula.rho(trace)
        sat = formula.sat(trace)
        finite = np.isfinite(rho)
        ax.plot(trace.times[finite], rho[finite], color="#111111", lw=1.8)
        ax.axhline(0.0, color="#d62728", lw=1.2, ls="--")
        ax.fill_between(
            trace.times,
            0.0,
            1.0,
            where=~sat,
            color="#d62728",
            alpha=0.12,
            linewidth=0,
            transform=ax.get_xaxis_transform(),
        )
        verdict = "violated" if robustness(formula, trace) < 0.0 else "not violated"
        ax.set_ylabel("robustness")
        ax.set_title(
            f"{label}   -   at t = 0: rho = {robustness(formula, trace):+.4f}, "
            f"Boolean semantics says {satisfies(formula, trace)} ({verdict})",
            fontsize=10,
        )
        ax.grid(alpha=0.25)
        if not finite.all():
            ax.text(
                0.99,
                0.05,
                f"{int((~finite).sum())} samples have infinite robustness (empty window) "
                "and are not drawn",
                transform=ax.transAxes,
                ha="right",
                fontsize=8,
                color="#555555",
            )

    axes[-1].set_xlabel("time [s]")
    fig.suptitle(
        "Red shading marks the samples the Boolean semantics calls unsatisfied.\n"
        "The black curve is below zero over exactly those samples: that agreement, "
        "with no tolerance band, is the property this package is built on.",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = ROOT / "screenshots" / "requirement_monitor.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print(f"wrote {out.relative_to(ROOT)}")
    print("")
    for label, formula in formulas:
        rho = formula.rho(trace)
        sat = formula.sat(trace)
        agree = bool(np.array_equal(rho >= 0.0, sat))
        print(
            f"{label:<46s} rho(0) = {robustness(formula, trace):+.6f}  "
            f"satisfied = {str(satisfies(formula, trace)):<5s}  "
            f"elementwise sign agreement = {agree}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
