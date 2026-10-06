"""Window sizing: the closed-form knee, and the window a sender actually needs.

Left panel: selective-repeat goodput against send window, in multiples of N,
for three frame error rates.  Solid curves are the simulated state machine;
dashed curves are the closed form ``(1-p) min(1, W/N)``, whose knee is at
W = N.  The gap between them is head-of-line blocking: a real sender cannot
advance its window base past an unacknowledged frame, so a frame under
retransmission holds its sequence number and the window fills behind it.  The
closed form does not model that and is an upper bound everywhere above W = 1.

Right panel: the same curves on a byte axis, for the GEO preset, which is what
a buffer-sizing decision is actually made in.  The vertical line is the
textbook recommendation of one bandwidth-delay product plus one frame.

What to notice: at W = N, the textbook window, the simulated sender achieves
between 36 and 73 per cent of the ideal depending on the frame error rate.  The
window needed to come within one per cent of the ideal is two to six times N.
On the GEO preset that is the difference between 67 kB and 400 kB of send
buffer, which is a procurement decision, not a rounding error.

Writes ../screenshots/window_sizing.png.  Runtime: about 70 s on one core.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
)

from arqlonghaul import closedform as cf  # noqa: E402
from arqlonghaul.channel import IndependentFrameChannel  # noqa: E402
from arqlonghaul.link import preset  # noqa: E402
from arqlonghaul.protocols import simulate_selective_repeat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "screenshots", "window_sizing.png")

N = 60
SLOTS = 400_000
MULTIPLES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
FERS = (0.01, 0.05, 0.20)


def main() -> None:
    rng = np.random.default_rng(45_045_8)
    link = preset("geo")
    frame_bytes = link.frame_bits / 8.0
    print(f"GEO preset: N = {link.slots_per_cycle:.2f} slots, "
          f"frame = {frame_bytes:.0f} bytes, "
          f"BDP = {link.bdp_bytes:.0f} bytes")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    colours = ("#1f3b73", "#b3321c", "#1a7a4c")
    for fer, colour in zip(FERS, colours, strict=True):
        errors = IndependentFrameChannel(fer).errors(SLOTS, rng)
        sim = []
        closed = []
        for mult in MULTIPLES:
            w = max(1, int(round(mult * N)))
            sim.append(simulate_selective_repeat(errors, N, w).goodput)
            closed.append(cf.sr_throughput(fer, N, w))
            print(f"fer {fer:5.2f}  W/N {mult:5.2f}  W {w:5d}  "
                  f"simulated {sim[-1]:.5f}  closed form {closed[-1]:.5f}")
        ax.plot(MULTIPLES, sim, "o-", color=colour, label=f"simulated, p = {fer:g}")
        ax.plot(MULTIPLES, closed, "--", color=colour, lw=1.1,
                label=f"closed form, p = {fer:g}")
        ax2.plot(
            [m * N * frame_bytes / 1000.0 for m in MULTIPLES],
            sim,
            "o-",
            color=colour,
            label=f"simulated, p = {fer:g}",
        )
    ax.axvline(1.0, color="k", ls=":", lw=1.2)
    ax.annotate(
        "textbook window\n(one BDP + one frame)",
        xy=(1.0, 0.15),
        xytext=(1.6, 0.12),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    ax.set_xlabel("send window / N")
    ax.set_ylabel("goodput (frames delivered per slot)")
    ax.set_title(f"Selective repeat against window size, N = {N} slots")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax2.axvline(
        link.min_continuous_window * frame_bytes / 1000.0,
        color="k",
        ls=":",
        lw=1.2,
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("send buffer (kB), GEO preset, 1115-octet frames")
    ax2.set_ylabel("goodput (frames delivered per slot)")
    ax2.set_title("The same thing as a buffer-sizing decision\n"
                  "dotted: one bandwidth-delay product plus one frame")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote screenshots/{os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
