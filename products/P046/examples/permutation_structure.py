"""The permutation of each construction, drawn, with the burst window that defeats it.

Each panel plots the points (transmitted position, source index) for one
construction at N = 256, and shades the narrowest channel burst window that
catches two adjacent source symbols -- the window whose width is the construction's
largest fully dispersed burst plus one. Writes
``../screenshots/permutation_structure.png``.

What to notice: the block interleaver's points lie on a few steep lines, which is
why a burst that is wider than one line spacing catches two adjacent source
symbols. The helical read at step 8 fans the points out so that no narrow vertical
strip contains two adjacent source indices, which is exactly what its larger
dispersed burst means geometrically. The S-random panel has no visible structure
and no large dispersed burst either: a permutation can be thoroughly scrambled and
still put two adjacent source symbols close together.

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
from interleavekit.metrics import max_burst_fully_dispersed, minimum_spread  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "permutation_structure.png")

N = 256


def worst_window(pos: np.ndarray) -> tuple[int, int, int]:
    """Transmitted window that catches an adjacent source pair, and its width."""
    gaps = np.abs(np.diff(pos))
    j = int(np.argmin(gaps))
    lo = int(min(pos[j], pos[j + 1]))
    hi = int(max(pos[j], pos[j + 1]))
    return lo, hi, hi - lo + 1


def main() -> None:
    panels = []
    for il, label in [
        (BlockInterleaver(16, 16), "block, 16 x 16"),
        (BlockInterleaver(32, 8), "block, 32 x 8"),
        (HelicalInterleaver(16, 16, 8), "helical, 16 x 16, step 8"),
        (SRandomInterleaver(N, 8, seed=0), "S-random, spread 8"),
    ]:
        pos = il.position_of_input()
        panels.append((
            label, np.arange(pos.size), pos, minimum_spread(il.permutation()),
            max_burst_fully_dispersed(pos), il.cost().pair_memory_symbols,
        ))

    bank = ConvolutionalInterleaver(registers=16, slope=1)
    nb = bank.max_delay_symbols + N
    bank_pos = bank.transmitted_position(np.arange(nb))
    window = bank.steady_state_range(nb)
    keep = (bank_pos >= window[0]) & (bank_pos < window[1])
    panels.append((
        "convolutional, 16 registers, slope 1",
        np.flatnonzero(keep), bank_pos[keep], None,
        max_burst_fully_dispersed(bank_pos, window), bank.cost().pair_memory_symbols,
    ))

    fig, axes = plt.subplots(1, 5, figsize=(19.5, 4.4))
    for ax, (label, src, pos, spread, burst, mem) in zip(axes, panels, strict=True):
        ax.scatter(pos, src, s=3.0, color="#30506e")
        lo, hi, width = worst_window(np.asarray(pos))
        ax.axvspan(lo, hi, color="#a84848", alpha=0.25)
        ax.set_xlabel("transmitted position")
        if ax is axes[0]:
            ax.set_ylabel("source index")
        spread_txt = "n/a" if spread is None else str(spread)
        ax.set_title(
            f"{label}\nmin spread {spread_txt}, burst {burst}\n"
            f"pair memory {mem} sym, worst window {width} sym",
            fontsize=8.5,
        )
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    print(f"{'construction':<40} {'min spread':>11} {'burst':>6} {'pair mem':>9}")
    for label, _src, _pos, spread, burst, mem in panels:
        spread_txt = "n/a" if spread is None else str(spread)
        print(f"{label:<40} {spread_txt:>11} {burst:>6} {mem:>9}")
    print(f"wrote screenshots/{os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
