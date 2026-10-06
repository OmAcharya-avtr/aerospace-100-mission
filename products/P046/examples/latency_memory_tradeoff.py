"""Latency and memory paid per symbol of dispersed burst, block against shift-register bank.

For a range of target burst lengths, searches each family for the cheapest
parameter set that **measurably** fully disperses that burst, and plots what it
costs. Writes ``../screenshots/latency_memory_tradeoff.png``.

What to notice: both families scale linearly in the target burst, and the lines are
parallel, so the ratio between them is roughly constant rather than closing. Under
full-block buffering for the block interleaver and the exact shift-register delays
for the bank, the bank costs between 3.3 and 3.9 times less pair memory across the
range -- larger than the factor of two usually quoted, and the README says why.

Runtime: about 45 s on one core.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from interleavekit.cost import cheapest_for_burst  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "screenshots", "latency_memory_tradeoff.png")

SYMBOL_RATE_HZ = 1.0e6
TARGETS = [4, 8, 12, 16, 24, 32, 48, 64]


def main() -> None:
    families = ("block", "helical", "convolutional")
    memory = {f: [] for f in families}
    latency_ms = {f: [] for f in families}
    achieved = {f: [] for f in families}

    for target in TARGETS:
        best = cheapest_for_burst(
            target,
            symbol_rate_hz=SYMBOL_RATE_HZ,
            families=families,
            max_block_symbols=1024,
            max_registers=48,
            max_slope=48,
        )
        for family in families:
            row = best[family]
            memory[family].append(np.nan if row is None else row.pair_memory_symbols)
            latency_ms[family].append(np.nan if row is None else row.pair_latency_ms)
            achieved[family].append(0 if row is None else row.max_burst_fully_dispersed)

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.8))
    labels = {
        "block": "block (full-block buffering)",
        "helical": "helical (full-block buffering)",
        "convolutional": "convolutional (shift-register bank)",
    }
    for family in families:
        ax.plot(TARGETS, memory[family], marker="o", linewidth=1.8, label=labels[family])
        ax2.plot(TARGETS, latency_ms[family], marker="o", linewidth=1.8, label=labels[family])

    ax.set_xlabel("target burst to fully disperse (symbols)")
    ax.set_ylabel("pair memory (symbols)")
    ax.set_title("Memory paid for the target")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax2.set_xlabel("target burst to fully disperse (symbols)")
    ax2.set_ylabel(f"pair latency (ms at {SYMBOL_RATE_HZ:.3g} sym/s)")
    ax2.set_title("End-to-end latency paid for the target")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ratio = np.asarray(memory["block"], dtype=float) / np.asarray(
        memory["convolutional"], dtype=float
    )
    ax3.plot(TARGETS, ratio, marker="s", color="#a84848", linewidth=1.8)
    ax3.axhline(2.0, color="0.5", linestyle=":", label="the factor of 2 often quoted")
    ax3.set_xlabel("target burst to fully disperse (symbols)")
    ax3.set_ylabel("block pair memory / bank pair memory")
    ax3.set_title("Measured ratio under these two cost models")
    ax3.set_ylim(0, max(4.5, float(np.nanmax(ratio)) * 1.1))
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    plt.close(fig)

    print(f"{'target':>7} {'block mem':>10} {'bank mem':>9} {'ratio':>7} {'bank burst':>11}")
    for k, target in enumerate(TARGETS):
        print(
            f"{target:>7} {memory['block'][k]:>10.0f} {memory['convolutional'][k]:>9.0f} "
            f"{ratio[k]:>7.3f} {achieved['convolutional'][k]:>11}"
        )
    print(f"memory ratio range: {np.nanmin(ratio):.3f} to {np.nanmax(ratio):.3f}")
    print(f"wrote screenshots/{os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
