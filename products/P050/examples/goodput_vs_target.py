"""Optimal goodput against the availability target, and what the two readings
of "availability" cost each other.

Writes ``../screenshots/goodput_vs_target.png``.

Left panel: the achievable goodput as the availability target is raised, for
three scintillation indices. The curves are staircases because the MODCOD set
is discrete -- the goodput holds flat while one MODCOD still clears the
target, then steps down. The step edges are the only places a link designer's
decision changes, and they are not where a continuous approximation would put
them.

Right panel: the goodput the long-run reading buys over the per-interval
reading at K = 2, against the worst-interval availability it gives up to get
it. Points at zero gain are where the two readings pick the same answer, and
there the shortfall can be negative because the chosen MODCOD clears the
target with room to spare. The script asserts the claim that matters: no
point has a positive goodput gain at a non-positive shortfall.

Runtime: about 60 s on one contended core.
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
    illustrative_modcod_table,
    select_rate,
)

TABLE = illustrative_modcod_table()
MARGIN_DB = 14.0
SCINTILLATIONS = (0.05, 0.2, 0.6)
TARGETS = 1.0 - np.logspace(-4.0, -0.7, 140)
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "goodput_vs_target.png"


def solve(target: float, mode: str, k: int):
    problem = RateProblem(
        modcods=TABLE,
        fade=LognormalFade(scintillation),
        margin_db=MARGIN_DB,
        availability_target=float(target),
        max_entries=k,
        mode=mode,
    )
    try:
        return problem, select_rate(problem)
    except InfeasibleProblem:
        return problem, None


figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

for scintillation, colour in zip(SCINTILLATIONS, ("#1b4965", "#bc4b51", "#5f8d4e"), strict=True):
    goodputs = []
    for target in TARGETS:
        _, solution = solve(target, "per_interval", 1)
        goodputs.append(solution.expected_goodput if solution is not None else np.nan)
    axes[0].step(
        1.0 - TARGETS,
        goodputs,
        where="post",
        color=colour,
        label=f"scintillation index {scintillation}",
    )

axes[0].set_xscale("log")
axes[0].invert_xaxis()
axes[0].set_xlabel("unavailability allowed, 1 - target (log scale)")
axes[0].set_ylabel("optimal expected goodput, bits/symbol")
axes[0].set_title(
    f"single-MODCOD optimum, margin {MARGIN_DB:.1f} dB\n"
    "flat stretches are where the decision does not change"
)
axes[0].grid(True, which="both", alpha=0.3)
axes[0].legend(loc="lower left", fontsize=9)

gains = []
shortfalls = []
labels = []
for scintillation in SCINTILLATIONS:
    for target in (0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995):
        _, strict = solve(target, "per_interval", 2)
        _, loose = solve(target, "long_run", 2)
        if strict is None or loose is None:
            continue
        gains.append(loose.expected_goodput / strict.expected_goodput - 1.0)
        shortfalls.append(target - loose.worst_interval_availability)
        labels.append(scintillation)

labels_array = np.asarray(labels)
for scintillation, colour in zip(SCINTILLATIONS, ("#1b4965", "#bc4b51", "#5f8d4e"), strict=True):
    mask = labels_array == scintillation
    axes[1].scatter(
        np.asarray(shortfalls)[mask],
        np.asarray(gains)[mask],
        s=44,
        color=colour,
        edgecolor="black",
        linewidth=0.5,
        label=f"scintillation index {scintillation}",
    )

axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[1].axvline(0.0, color="black", linewidth=0.8)
axes[1].set_xlabel("worst-interval availability shortfall against the target")
axes[1].set_ylabel("relative goodput gain from the long-run reading")
axes[1].set_title(
    "long_run against per_interval at K = 2\n"
    "points with zero gain are where the two readings agree"
)
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="center right", fontsize=9)

figure.tight_layout()
figure.savefig(OUT, dpi=140)
print(f"wrote {OUT.parent.name}/{OUT.name}")
print(f"targets swept: {TARGETS.size} from {TARGETS.min():.6f} to {TARGETS.max():.6f}")
print(f"mode-comparison points: {len(gains)}")
if gains:
    print(f"goodput gain  min {min(gains):.6f}  max {max(gains):.6f}")
    print(f"availability shortfall  min {min(shortfalls):.6f}  max {max(shortfalls):.6f}")
    free_lunches = [
        (g, s) for g, s in zip(gains, shortfalls, strict=True) if g > 1e-12 and s <= 1e-12
    ]
    print(f"points with a positive gain at a non-positive shortfall: {len(free_lunches)}")
    if free_lunches:
        raise SystemExit(f"unexpected free lunch: {free_lunches}")
