"""What each telemetry set lets you say, and what it refuses.

What to notice: the left panel is the verdict of baseline 1 for every parameter
on every declared telemetry set -- REFUSED means structurally unidentifiable,
WEAK means identifiable in principle and not from this record, IDENTIFIABLE
means both. Three rate-only sets refuse all five parameters. The right panel is
the loading of each identifiable combination on each parameter for the reference
set: those four combinations are what is reported in place of the five refused
parameters, and together they span exactly the space of the four ratios
k_t/J, T_c/J, b/J and c_d/J.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from _common import ACCENT, REFUSED, WARN, bootstrap, save, style  # noqa: E402

ROOT = bootstrap()

from twindiverge.identifiability import VERDICTS, analyse  # noqa: E402
from twindiverge.plant import NOMINAL, PARAMETER_NAMES  # noqa: E402
from twindiverge.telemetry import TELEMETRY_SETS  # noqa: E402


def main() -> int:
    style()
    names = list(TELEMETRY_SETS)
    grid = np.zeros((len(names), 5), dtype=int)
    reports = {}
    for row, name in enumerate(names):
        report = analyse(TELEMETRY_SETS[name], NOMINAL)
        reports[name] = report
        for column, verdict in enumerate(report.verdict):
            grid[row, column] = VERDICTS.index(verdict)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colours = ListedColormap([REFUSED, WARN, ACCENT])
    axes[0].imshow(grid, cmap=colours, vmin=0, vmax=2, aspect="auto")
    axes[0].set_xticks(range(5), PARAMETER_NAMES)
    axes[0].set_yticks(range(len(names)), names, fontsize=7)
    for row in range(len(names)):
        for column in range(5):
            axes[0].text(
                column, row, VERDICTS[grid[row, column]][0],
                ha="center", va="center", color="white", fontsize=8,
            )
    axes[0].set_title("baseline 1's verdict per parameter (R / W / I)")
    axes[0].grid(False)
    axes[0].legend(
        handles=[
            Patch(facecolor=REFUSED, label="REFUSED (structurally unidentifiable)"),
            Patch(facecolor=WARN, label="WEAK (practically unidentifiable)"),
            Patch(facecolor=ACCENT, label="IDENTIFIABLE"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=7, ncols=1,
    )

    report = reports["rate_full"]
    loadings = report.identifiable_basis
    image = axes[1].imshow(
        loadings, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto"
    )
    axes[1].set_xticks(range(5), PARAMETER_NAMES)
    axes[1].set_yticks(
        range(report.rank),
        [
            f"combination {index + 1}\nbound {report.direction_crb_relative[index]:.2e}"
            for index in range(report.rank)
        ],
        fontsize=7,
    )
    for row in range(report.rank):
        for column in range(5):
            axes[1].text(
                column, row, f"{loadings[row, column]:+.2f}",
                ha="center", va="center", fontsize=7,
            )
    axes[1].set_title(
        "rate_full: the four identifiable combinations, equation (11)"
    )
    axes[1].grid(False)
    figure.colorbar(image, ax=axes[1], label="loading in logarithmic parameters")
    figure.suptitle(
        "twindiverge: refusal is an output, and the combinations are what replaces it",
        fontsize=11,
    )
    path = save(figure, "attribution_map.png")
    print(f"written: {path}")
    for name in names:
        report = reports[name]
        print(
            f"  {name:<20s} rank {report.rank} verdicts "
            f"{' '.join(verdict[0] for verdict in report.verdict)} "
            f"refused {len(report.refused)} weak {len(report.weak)}"
        )
    print("  rate_full identifiable combinations:")
    for label, bound in reports["rate_full"].combinations():
        print(f"    {label}  bound {bound:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
