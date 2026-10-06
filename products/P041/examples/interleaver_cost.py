"""The trade a designer actually makes: coding gain against latency and memory.

Writes ``../screenshots/interleaver_cost.png``.

Left: latency and memory against depth for the block and the convolutional
construction, with the fade correlation length marked. Right: the burst-dispersion
rule, equation (22), showing how many errors a burst of a given length deposits in
one codeword at each depth, against the code's correction radius.

Runtime: under 10 s on one core.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from codedfade.interleave import (  # noqa: E402
    BlockInterleaver,
    ConvolutionalInterleaver,
    burst_dispersion,
)
from codedfade.reedsolomon import ReedSolomon  # noqa: E402

FS = 1.0e6
LC = 200.0


def main() -> None:
    code = ReedSolomon(31, 21, 5)
    depths = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    block = [BlockInterleaver(int(d), code.n).cost(FS, code.m) for d in depths]
    conv = [
        ConvolutionalInterleaver(int(d), 1).cost(FS, code.m) for d in depths
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    ax = axes[0]
    ax.plot(
        depths,
        [c.latency_ms for c in block],
        marker="o",
        color="#1f77b4",
        label="block, latency ms (span 31)",
    )
    ax.plot(
        depths,
        [c.latency_ms for c in conv],
        marker="s",
        color="#9467bd",
        label="convolutional B=D, M=1, latency ms",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("depth D (block) or branches B (convolutional)")
    ax.set_ylabel("end-to-end latency, ms at 1 Mbaud")
    ax.axvline(LC, color="0.5", ls=":", lw=1.2)
    ax.annotate(
        f"Lc = {LC:.0f} symbols\n(tau = 200 us)",
        xy=(LC, 0.3),
        xytext=(260, 0.06),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
    )
    ax2 = ax.twinx()
    ax2.plot(
        depths,
        [c.memory_bytes for c in block],
        marker="^",
        ls="--",
        color="#d62728",
        label="block, memory B",
    )
    ax2.plot(
        depths,
        [c.memory_bytes for c in conv],
        marker="v",
        ls="--",
        color="#ff7f0e",
        label="convolutional, memory B",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("total storage both ends, bytes")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7.5, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Depth is bought with latency and memory, both linear in D")

    ax = axes[1]
    bursts = np.unique(np.round(np.logspace(0, 3.7, 40)).astype(int))
    for d, colour in zip(
        (1, 8, 64, 256, 1024), ["#333333", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
        strict=True,
    ):
        ax.plot(
            bursts,
            [burst_dispersion(d, int(b)) for b in bursts],
            label=f"D = {d}",
            color=colour,
        )
    ax.axhline(
        code.t,
        color="0.3",
        ls="--",
        lw=1.4,
        label=f"t = {code.t}: at or below this the codeword survives",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("burst length L, consecutive channel symbols")
    ax.set_ylabel("worst-case errors in one codeword, ceil(L/D)")
    ax.set_title("Equation (22): the depth needed is L/t, not L")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join("..", "screenshots", "interleaver_cost.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {os.path.basename(out)} to the screenshots directory")


if __name__ == "__main__":
    main()
