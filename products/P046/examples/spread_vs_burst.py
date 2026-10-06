"""Minimum spread against burst dispersion on a fixed array: the two disagree.

Sweeps the helical read step on a fixed 16 x 16 array, so pair memory and pair
latency are identical for every point, and plots minimum spread and the largest
fully dispersed burst together. Writes ``../screenshots/spread_vs_burst.png``.

What to notice: the two metrics are not measuring the same thing. The step that
maximises minimum spread gives a fully dispersed burst of 15; the step that
maximises burst dispersion gives 127, with a minimum spread of 4. A designer who
ranks interleavers on minimum spread, which is the metric a bare permutation
function invites, picks the worse of the two for a fading link and pays nothing
less for it.

Runtime: about 20 s on one core.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from interleavekit import HelicalInterleaver  # noqa: E402
from interleavekit.metrics import (  # noqa: E402
    dispersion,
    max_burst_fully_dispersed,
    minimum_spread,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "spread_vs_burst.png")

ROWS = 16
COLUMNS = 16
STEPS = list(range(0, 17))


def main() -> None:
    spreads = []
    bursts = []
    disps = []
    for step in STEPS:
        il = HelicalInterleaver(ROWS, COLUMNS, step)
        pi = il.permutation()
        spreads.append(minimum_spread(pi))
        bursts.append(max_burst_fully_dispersed(il.position_of_input()))
        disps.append(dispersion(pi))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.0))

    ax.plot(STEPS, spreads, marker="o", color="#4878a8", linewidth=1.8, label="minimum spread")
    ax.set_xlabel("helical read step (rows per column)")
    ax.set_ylabel("minimum spread (index units)", color="#4878a8")
    ax.tick_params(axis="y", labelcolor="#4878a8")
    ax.grid(True, alpha=0.3)

    axb = ax.twinx()
    axb.plot(
        STEPS, bursts, marker="s", color="#a84848", linewidth=1.8,
        label="largest fully dispersed burst",
    )
    axb.set_ylabel("largest fully dispersed burst (symbols)", color="#a84848")
    axb.tick_params(axis="y", labelcolor="#a84848")

    best_spread = int(np.argmax(spreads))
    best_burst = int(np.argmax(bursts))
    ax.axvline(STEPS[best_spread], color="#4878a8", linestyle=":", alpha=0.7)
    ax.axvline(STEPS[best_burst], color="#a84848", linestyle=":", alpha=0.7)
    ax.set_title(
        f"{ROWS} x {COLUMNS} helical array, identical memory at every step\n"
        f"best by spread: step {STEPS[best_spread]} (burst {bursts[best_spread]}) | "
        f"best by burst: step {STEPS[best_burst]} (spread {spreads[best_burst]})"
    )
    lines = ax.get_lines()[:1] + axb.get_lines()[:1]
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="center left")

    sc = ax2.scatter(spreads, bursts, c=STEPS, cmap="viridis", s=60, zorder=3)
    for step, sp, bu in zip(STEPS, spreads, bursts, strict=True):
        ax2.annotate(str(step), (sp, bu), fontsize=7, xytext=(4, 3),
                     textcoords="offset points")
    ax2.set_xlabel("minimum spread (index units)")
    ax2.set_ylabel("largest fully dispersed burst (symbols)")
    ax2.set_title("If the two metrics agreed, this would be a rising line")
    ax2.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax2, label="helical read step")

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    print(f"{'step':>5} {'min spread':>11} {'dispersion':>11} {'burst':>6}")
    for step, sp, di, bu in zip(STEPS, spreads, disps, bursts, strict=True):
        print(f"{step:>5} {sp:>11} {di:>11.6f} {bu:>6}")
    print(
        f"best by minimum spread: step {STEPS[best_spread]} -> burst {bursts[best_spread]}; "
        f"best by burst: step {STEPS[best_burst]} -> spread {spreads[best_burst]}"
    )
    print(f"wrote screenshots/{os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
