"""Worst-case surviving run against channel burst length, for all four constructions.

All four are sized to the same pair memory of 512 symbols, so the curves compare
constructions at equal cost, not at equal parameters. Writes
``../screenshots/burst_dispersion_curves.png``.

What to notice: every curve is flat at 1 up to its own largest fully dispersed
burst and then climbs. The flat section is the whole design margin, and it differs
by a factor of 15.9 between the best and worst candidate at identical memory (127
symbols for the helical read at step 8, 8 for the S-random permutation). The
no-interleaving diagonal is what a burst does with no interleaver at all.

Runtime: about 10 s on one core.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from interleavekit import (  # noqa: E402
    BlockInterleaver,
    ConvolutionalInterleaver,
    HelicalInterleaver,
    SRandomInterleaver,
)
from interleavekit.metrics import burst_dispersion_profile, max_burst_fully_dispersed  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "burst_dispersion_curves.png")

MAX_BURST = 200


def main() -> None:
    lengths = np.arange(1, MAX_BURST + 1)
    series = []

    for il, label in [
        (BlockInterleaver(16, 16), "block, depth 16 x span 16"),
        (BlockInterleaver(32, 8), "block, depth 32 x span 8"),
        (HelicalInterleaver(16, 16, 8), "helical, 16 x 16, step 8"),
        (SRandomInterleaver(256, 8, seed=0), "S-random, N 256, spread 8"),
    ]:
        pos = il.position_of_input()
        profile = burst_dispersion_profile(pos, lengths)
        series.append((label, profile, max_burst_fully_dispersed(pos),
                       il.cost().pair_memory_symbols))

    bank = ConvolutionalInterleaver(registers=16, slope=1)
    n = bank.max_delay_symbols + MAX_BURST + 400
    bank_pos = bank.transmitted_position(np.arange(n))
    window = bank.steady_state_range(n)
    series.append((
        "convolutional, 16 registers, slope 1",
        burst_dispersion_profile(bank_pos, lengths, window),
        max_burst_fully_dispersed(bank_pos, window),
        bank.cost().pair_memory_symbols,
    ))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    ax.plot(lengths, lengths, color="0.6", linestyle=":", label="no interleaving")
    for label, profile, limit, mem in series:
        ax.plot(lengths, profile, linewidth=1.8, label=f"{label} ({mem} sym)")
        ax.plot([limit], [1], marker="o", markersize=5, color=ax.lines[-1].get_color())
    ax.set_xlabel("channel burst length (transmitted symbols)")
    ax.set_ylabel("worst-case surviving run after de-interleaving (symbols)")
    ax.set_title("Burst dispersion, all candidates at 512-symbol pair memory\n"
                 "except the bank, labelled with its own")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")

    names = [s[0] for s in series]
    limits = [s[2] for s in series]
    mems = [s[3] for s in series]
    order = np.argsort(limits)
    ypos = np.arange(len(order))
    ax2.barh(ypos, [limits[i] for i in order], color="#4878a8")
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f"{names[i]}\n{mems[i]} sym pair memory" for i in order], fontsize=7.5)
    ax2.set_xlabel("largest fully dispersed burst (symbols)")
    ax2.set_title("Design margin per candidate")
    ax2.grid(True, axis="x", alpha=0.3)
    for k, i in enumerate(order):
        ax2.text(limits[i] + 1.5, k, str(limits[i]), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    print("largest fully dispersed burst, by candidate:")
    for label, _profile, limit, mem in series:
        print(f"  {label:<40} {limit:>5} symbols, pair memory {mem} symbols")
    print(f"wrote screenshots/{os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
