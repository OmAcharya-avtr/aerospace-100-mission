"""Where the decision is fragile and where it is not.

Writes ``../screenshots/sensitivity_map.png``.

Left panel: the chosen MODCOD over a grid of link margin and scintillation
index, at a fixed availability target. The boundaries between regions are the
only places the answer changes; the white region is where no MODCOD in the
table meets the target at all.

Right panel: the *relative width* of the stability interval in the
scintillation index at each margin -- how far the scintillation estimate can
be wrong before the answer changes, in total. Ringed points are the margins
flagged knife-edge: a 10 % error in **one** direction changes the selected
MODCOD. Note that the two are not the same test, and the plot shows why -- a
margin can have a wide total interval and still be knife-edge, because the
nominal point sits near one end of it. The same decision can be flat at one
margin and knife-edge at another a decibel away, which is the point of
plotting it.

Runtime: about 90 s on one contended core.
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
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from coderateopt import (  # noqa: E402
    InfeasibleProblem,
    LognormalFade,
    RateProblem,
    illustrative_modcod_table,
    scintillation_sensitivity,
    select_rate,
)

TABLE = illustrative_modcod_table()
TARGET = 0.999
MARGINS = np.arange(6.0, 24.01, 0.5)
SCINTILLATIONS = np.linspace(0.02, 1.0, 50)
KNIFE_EDGE_AT = 0.10
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "sensitivity_map.png"


def problem(margin_db: float, scintillation: float) -> RateProblem:
    return RateProblem(
        modcods=TABLE,
        fade=LognormalFade(float(scintillation)),
        margin_db=float(margin_db),
        availability_target=TARGET,
        max_entries=1,
        mode="per_interval",
    )


grid = np.full((SCINTILLATIONS.size, MARGINS.size), np.nan)
for column, margin in enumerate(MARGINS):
    for row, scintillation in enumerate(SCINTILLATIONS):
        try:
            solution = select_rate(problem(margin, scintillation))
        except InfeasibleProblem:
            continue
        grid[row, column] = solution.support[0]

figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

used = sorted({int(v) for v in np.unique(grid[~np.isnan(grid)])})
palette = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(used)))
cmap = ListedColormap(palette)
cmap.set_bad("white")
remapped = np.full_like(grid, np.nan)
for new_index, old_index in enumerate(used):
    remapped[grid == old_index] = new_index
norm = BoundaryNorm(np.arange(-0.5, len(used) + 0.5, 1.0), cmap.N)

mesh = axes[0].pcolormesh(
    MARGINS, SCINTILLATIONS, remapped, cmap=cmap, norm=norm, shading="nearest"
)
bar = figure.colorbar(mesh, ax=axes[0], ticks=np.arange(len(used)))
bar.ax.set_yticklabels([TABLE.names[i] for i in used], fontsize=8)
axes[0].set_xlabel("clear-sky link margin, dB")
axes[0].set_ylabel("scintillation index")
axes[0].set_title(
    f"selected MODCOD, availability target {TARGET}\n"
    "white: no MODCOD in the table meets the target"
)

widths = []
knife = []
for margin in MARGINS:
    base = problem(margin, 0.2)
    try:
        report = scintillation_sensitivity(base, grid_points=40)
    except (InfeasibleProblem, ValueError):
        widths.append(np.nan)
        knife.append(False)
        continue
    if report.signature is None:
        widths.append(np.nan)
        knife.append(False)
        continue
    widths.append(report.relative_width)
    knife.append(report.is_knife_edge)

widths_array = np.asarray(widths, dtype=float)
knife_array = np.asarray(knife, dtype=bool)
axes[1].plot(MARGINS, widths_array, color="#1b4965", marker="o", markersize=3.5, linewidth=1.2)
axes[1].scatter(
    MARGINS[knife_array],
    widths_array[knife_array],
    s=70,
    facecolor="none",
    edgecolor="#bc4b51",
    linewidth=1.6,
    label=f"knife-edge at +/-{KNIFE_EDGE_AT:.0%}",
)
axes[1].set_yscale("log")
axes[1].set_xlabel("clear-sky link margin, dB")
axes[1].set_ylabel("relative width of the stability interval")
axes[1].set_title(
    "how wrong the scintillation estimate may be\n"
    f"nominal scintillation index 0.2, target {TARGET}; ringed points are knife-edge"
)
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend(loc="upper left", fontsize=9)

figure.tight_layout()
figure.savefig(OUT, dpi=140)
print(f"wrote {OUT.parent.name}/{OUT.name}")
print(f"grid: {MARGINS.size} margins x {SCINTILLATIONS.size} scintillation indices")
print(f"MODCODs that appear as the optimum: {[TABLE.names[i] for i in used]}")
print(f"infeasible grid cells: {int(np.isnan(grid).sum())} of {grid.size}")
finite = widths_array[np.isfinite(widths_array)]
print(
    f"relative stability width: min {finite.min():.6f}  median {np.median(finite):.6f}  "
    f"max {finite.max():.6f}"
)
print(f"margins flagged knife-edge at +/-{KNIFE_EDGE_AT:.0%}: "
      f"{int(knife_array.sum())} of {MARGINS.size}")
