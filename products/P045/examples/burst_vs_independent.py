"""The central figure: goodput against burst length at a fixed marginal error rate.

Left panel: measured goodput of the three protocol state machines on a
Gilbert-Elliott channel whose mean frame error rate is held at 5 per cent while
the mean burst length is swept from 1 (memoryless) to 200 frames.  The dashed
lines are the independent-error closed forms, which do not depend on burst
length and are therefore flat.  The vertical distance between a solid curve and
its dashed line is the error a practitioner makes by sizing the link with the
textbook formula.

Right panel: the same thing as a percentage, which is the number worth quoting.

What to notice: go-back-N rises by a factor of nearly four as the errors bunch
together, because one burst costs one go-back where the same error mass spread
out would cost many.  Window-limited selective repeat rises for the same
reason.  Stop-and-wait does not move at all, because its throughput is linear
in the marginal error rate and correlation does not change a marginal.  Nothing
here is made worse by bursts, and nothing here is predicted by the formula.

Writes ../screenshots/burst_vs_independent.png.  Runtime: about 60 s on one core.
"""

from __future__ import annotations

import math
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
from arqlonghaul.channel import GilbertElliottChannel, IndependentFrameChannel  # noqa: E402
from arqlonghaul.protocols import (  # noqa: E402
    simulate_go_back_n,
    simulate_selective_repeat,
    simulate_stop_and_wait,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "screenshots", "burst_vs_independent.png")

FER = 0.05
N = 60
SLOTS = 500_000
BURSTS = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, 200.0)


def main() -> None:
    rng = np.random.default_rng(45_045_7)
    w_min = math.ceil(N)
    w_big = 40 * N
    closed = {
        "stop-and-wait": cf.sw_throughput(FER, N),
        f"go-back-N, W={w_min}": cf.gbn_throughput(FER, N, w_min),
        f"selective repeat, W={w_min}": cf.sr_throughput(FER, N, w_min),
        f"selective repeat, W={w_big}": cf.sr_throughput(FER, N, w_big),
    }
    measured: dict[str, list[float]] = {k: [] for k in closed}
    for burst in BURSTS:
        if burst <= 1.0:
            errors = IndependentFrameChannel(FER).errors(SLOTS, rng)
        else:
            errors = GilbertElliottChannel.from_mean_and_burst(FER, burst).errors(
                SLOTS, rng
            )
        measured["stop-and-wait"].append(simulate_stop_and_wait(errors, N).goodput)
        measured[f"go-back-N, W={w_min}"].append(
            simulate_go_back_n(errors, N, w_min).goodput
        )
        measured[f"selective repeat, W={w_min}"].append(
            simulate_selective_repeat(errors, N, w_min).goodput
        )
        measured[f"selective repeat, W={w_big}"].append(
            simulate_selective_repeat(errors, N, w_big).goodput
        )
        print(
            f"burst {burst:6.1f}  "
            + "  ".join(f"{k.split(',')[0][:4]}={v[-1]:.4f}" for k, v in measured.items())
        )

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    colours = ("#1f3b73", "#b3321c", "#1a7a4c", "#7a4ca8")
    for (label, series), colour in zip(measured.items(), colours, strict=True):
        ax.plot(BURSTS, series, "o-", color=colour, label=f"{label} (simulated)")
        ax.axhline(
            closed[label], color=colour, ls="--", lw=1.1, alpha=0.8,
        )
        pct = [100 * (closed[label] / v - 1) for v in series]
        ax2.plot(BURSTS, pct, "o-", color=colour, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("mean burst length (frames)")
    ax.set_ylabel("goodput (frames delivered per slot)")
    ax.set_title(
        f"Goodput at a fixed {FER:.0%} marginal frame error rate, N = {N} slots\n"
        "dashed: independent-error closed form (burst-independent)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center left")
    ax2.set_xscale("log")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_xlabel("mean burst length (frames)")
    ax2.set_ylabel("closed form / measured - 1 (%)")
    ax2.set_title(
        "Error of the independent-error formula\n"
        "negative: the formula predicts less goodput than the protocol achieves"
    )
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote screenshots/{os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
