"""What the two rules of thumb actually cost, across the margin range.

Writes ``../screenshots/heuristic_gap.png``.

Left panel: relative goodput loss of each rule against the constrained
optimum, as a function of link margin, at three scintillation indices. The
"fastest MODCOD that meets the target" rule is almost always right on a
well-behaved MODCOD table -- that is the honest result, and this plot is where
it shows. The fixed-reserve rule is not, and its losses are large.

Right panel: the thing the fixed-reserve rule gets wrong that goodput does not
capture. For each margin it plots the *unavailability* actually achieved by the
rule's choice -- `1 - A` on a log scale, because that is where the interesting
range is -- against the unavailability the target allows. Points **above** the
line are link designs that do not meet the availability they were specified to
meet, which is a different and worse failure than losing a few per cent of
throughput.

Runtime: about 50 s on one contended core.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coderateopt import (  # noqa: E402
    InfeasibleProblem,
    LognormalFade,
    RateProblem,
    compare_to_optimum,
    highest_rate_within_reserve,
    illustrative_modcod_table,
)

TABLE = illustrative_modcod_table()
MARGINS = np.arange(6.0, 24.01, 0.25)
SCINTILLATIONS = (0.05, 0.2, 0.6)
TARGET = 0.99
RESERVE_DB = 3.0
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "heuristic_gap.png"

figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
colours = ("#1b4965", "#bc4b51", "#5f8d4e")

summary: dict[str, list[float]] = {"greedy": [], "reserve": []}
violations = 0
evaluated = 0

for scintillation, colour in zip(SCINTILLATIONS, colours, strict=True):
    greedy_loss: list[float] = []
    reserve_loss: list[float] = []
    reserve_availability: list[float] = []
    for margin in MARGINS:
        problem = RateProblem(
            modcods=TABLE,
            fade=LognormalFade(scintillation),
            margin_db=float(margin),
            availability_target=TARGET,
            max_entries=1,
            mode="per_interval",
        )
        try:
            rows = {row.name: row for row in compare_to_optimum(problem, reserve_db=RESERVE_DB)}
        except InfeasibleProblem:
            greedy_loss.append(np.nan)
            reserve_loss.append(np.nan)
            reserve_availability.append(np.nan)
            continue
        evaluated += 1
        greedy = rows["highest_feasible_rate"]
        reserve = rows["highest_rate_within_reserve"]
        greedy_loss.append(greedy.relative_loss if greedy.heuristic_meets_target else np.nan)
        reserve_loss.append(reserve.relative_loss if reserve.heuristic_meets_target else np.nan)
        summary["greedy"].append(greedy_loss[-1])
        summary["reserve"].append(reserve_loss[-1])
        try:
            index = highest_rate_within_reserve(problem, RESERVE_DB)
        except InfeasibleProblem:
            reserve_availability.append(np.nan)
            continue
        achieved = float(problem.availabilities()[index])
        reserve_availability.append(max(1.0 - achieved, 1e-12))
        if achieved < TARGET:
            violations += 1

    axes[0].plot(
        MARGINS,
        greedy_loss,
        color=colour,
        linewidth=1.6,
        label=f"greedy, scintillation {scintillation}",
    )
    axes[0].plot(
        MARGINS,
        reserve_loss,
        color=colour,
        linewidth=1.2,
        linestyle="--",
        label=f"{RESERVE_DB:.0f} dB reserve, scintillation {scintillation}",
    )
    axes[1].plot(
        MARGINS,
        reserve_availability,
        color=colour,
        linewidth=1.6,
        label=f"scintillation {scintillation}",
    )

axes[0].set_xlabel("clear-sky link margin, dB")
axes[0].set_ylabel("relative goodput loss against the optimum")
axes[0].set_title(
    f"cost of the rules of thumb, target {TARGET}\n"
    "gaps are where the rule refused or the instance was infeasible"
)
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper right", fontsize=7.5)

axes[1].axhline(
    1.0 - TARGET, color="black", linewidth=1.2, label=f"target {TARGET} ({1.0 - TARGET:g})"
)
axes[1].set_xlabel("clear-sky link margin, dB")
axes[1].set_ylabel("unavailability 1 - A of the fixed-reserve rule's choice")
axes[1].set_yscale("log")
axes[1].set_ylim(1e-10, 1.0)
axes[1].set_title(
    f"the {RESERVE_DB:.0f} dB fixed-reserve rule against the availability it was given\n"
    "above the line the link does not meet its specification"
)
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend(loc="lower left", fontsize=8)

figure.tight_layout()
figure.savefig(OUT, dpi=140)
print(f"wrote {OUT.parent.name}/{OUT.name}")
print(f"grid points evaluated: {evaluated}")
for name, values in summary.items():
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size:
        print(
            f"{name:<8s} constraint-respecting points {finite.size}  mean loss "
            f"{finite.mean():.6f}  max loss {finite.max():.6f}"
        )
print(
    f"fixed-reserve choices below the {TARGET} target: {violations} of {evaluated} "
    f"({violations / max(evaluated, 1):.4f})"
)
