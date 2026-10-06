"""HARQ: when is retransmitting cheaper than sending redundancy up front?

Left panel: goodput against the feedback latency D, in channel symbol times,
for two strategies with the same incremental-redundancy machinery behind them.
The high-rate strategy (R = 0.90) sends a short first transmission and expects
to retransmit.  The low-rate strategy (R = 0.50) pays for redundancy before it
knows whether it is needed.  They cross at a definite value of D, marked.

Right panel: the throughput-optimal first-transmission rate against D, found by
grid search over the exact nested dynamic program.  It falls monotonically: the
longer the round trip, the more redundancy it is worth paying for in advance.
The vertical markers are the actual values of D for the link presets in this
repository, taking BPSK so one channel symbol carries one bit.

What to notice: the crossover sits at tens of symbol times, and every real
space link in the right panel is four to six orders of magnitude past it.  For
a single HARQ process with no pipelining, the answer on a space link is always
to pay up front.  That conclusion rests on the no-pipelining assumption: a
sender running enough parallel HARQ processes to fill the round trip hides D,
and then retransmission wins again.

Writes ../screenshots/harq_crossover.png.  Runtime: about 40 s on one core.
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

from arqlonghaul.harq import crossover_rtt, optimal_first_rate  # noqa: E402
from arqlonghaul.link import PRESETS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, os.pardir, "screenshots", "harq_crossover.png")

K = 200
ALPHA = 0.5
ESN0_DB = 1.0
DELTA = 40
MAX_ROUNDS = 4


def main() -> None:
    cx = crossover_rtt(
        k=K,
        esn0_db=ESN0_DB,
        delta=DELTA,
        retransmit_rate=0.90,
        upfront_rate=0.50,
        max_rounds=MAX_ROUNDS,
        alpha=ALPHA,
    )
    print(f"k = {K}, Es/N0 = {ESN0_DB} dB, alpha = {ALPHA}, delta = {DELTA}, "
          f"M = {MAX_ROUNDS}")
    print(f"crossover D = {cx['crossover_d']:.2f} symbol times")

    d_grid = np.unique(np.round(np.logspace(0, 6, 28)))
    rates = []
    for d in d_grid:
        rate, n1, res = optimal_first_rate(K, float(d), ESN0_DB, DELTA, MAX_ROUNDS, ALPHA)
        rates.append(rate)
        print(f"D {d:10.0f}  optimal R1 {rate:.4f}  n1 {n1:4d}  goodput {res.goodput:.6f}")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax.loglog(cx["d"], cx["goodput_retransmit"], "-", color="#b3321c",
              label="R = 0.90 first transmission, retransmit with IR")
    ax.loglog(cx["d"], cx["goodput_upfront"], "-", color="#1f3b73",
              label="R = 0.50 first transmission, redundancy up front")
    if np.isfinite(cx["crossover_d"]):
        ax.axvline(cx["crossover_d"], color="k", ls=":", lw=1.2)
        ax.annotate(
            f"crossover\nD = {cx['crossover_d']:.0f} symbols",
            xy=(cx["crossover_d"], float(np.interp(cx["crossover_d"], cx["d"],
                                                   cx["goodput_upfront"]))),
            xytext=(cx["crossover_d"] * 3, 0.05),
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.8},
        )
    ax.set_xlabel("feedback latency D (channel symbol times)")
    ax.set_ylabel("goodput (information symbols per symbol time)")
    ax.set_title("Retransmit versus redundancy up front\n"
                 "one HARQ process, no pipelining")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax2.semilogx(d_grid, rates, "o-", color="#1a7a4c")
    for name, link in PRESETS.items():
        ax2.axvline(link.bdp_bits, color="#555555", ls=":", lw=1.0)
        ax2.text(
            link.bdp_bits,
            max(rates) - 0.01,
            f" {name}",
            rotation=90,
            fontsize=7,
            va="top",
        )
    ax2.set_xlabel("feedback latency D (channel symbol times)")
    ax2.set_ylabel("throughput-optimal first-transmission code rate")
    ax2.set_title("Optimal first rate against round-trip cost\n"
                  "dotted: D for each link preset (BPSK, one bit per symbol)")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote screenshots/{os.path.basename(OUT)}")


if __name__ == "__main__":
    main()
