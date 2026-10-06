"""PPM symbol error probability and photons per bit against the signal level.

Writes ``../screenshots/ppm_error_and_rate.png``.

What to notice. Left: with no background the symbol error probability is
exp(-n_s)(M-1)/M, so every order collapses onto one exponential and the order only
shifts it by (M-1)/M --- there is no diversity gain from a larger M in error
probability. Adding background (dashed) raises the floor, and the larger the order
the more slots there are for a false count to land in. Right: where the order does
pay is photons per bit. Each curve has a minimum at a finite n_s and approaches
1/log2(M) only as n_s goes to zero, at which point almost every symbol is erased.
The useful operating point is the knee, not the limit.

Runtime: about 15 s on one core.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from photoncount.capacity import (  # noqa: E402
    erasure_channel_capacity,
    hard_decision_capacity,
    minimum_photons_per_bit,
    photons_per_bit,
)
from photoncount.ppm import PPMConfig, symbol_error_probability  # noqa: E402

ORDERS = (4, 16, 64, 256)
COLOURS = ("#1b4965", "#5fa8d3", "#c1121f", "#f2a541")
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "ppm_error_and_rate.png"


def main() -> int:
    signal = np.logspace(-1, 1.6, 90)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    report = []
    for order, colour in zip(ORDERS, COLOURS, strict=True):
        clean = [symbol_error_probability(PPMConfig(order, float(s)))["error"] for s in signal]
        noisy = [
            symbol_error_probability(PPMConfig(order, float(s), 0.05))["error"] for s in signal
        ]
        ax_left.semilogy(signal, clean, color=colour, lw=1.9, label=f"M = {order}, $n_0=0$")
        ax_left.semilogy(signal, noisy, color=colour, lw=1.3, ls="--",
                         label=f"M = {order}, $n_0=0.05$")
        report.append((order, clean, noisy))
    ax_left.set_xscale("log")
    ax_left.set_xlabel("signal counts per pulsed slot $n_s$")
    ax_left.set_ylabel("symbol error probability")
    ax_left.set_title("Exact symbol error probability (P4)")
    ax_left.set_ylim(1e-6, 1.2)
    ax_left.legend(fontsize=7, ncol=2)
    ax_left.grid(alpha=0.3, which="both")

    minima = []
    for order, colour in zip(ORDERS, COLOURS, strict=True):
        ppb_clean = []
        ppb_hard = []
        for s in signal:
            cfg = PPMConfig(order, float(s))
            ppb_clean.append(
                photons_per_bit(cfg, erasure_channel_capacity(cfg)["bits_per_symbol"])
            )
            cfg_n = PPMConfig(order, float(s), 0.05)
            cap = hard_decision_capacity(cfg_n)["bits_per_symbol"]
            ppb_hard.append(photons_per_bit(cfg_n, cap) if cap > 0 else np.nan)
        ax_right.loglog(signal, ppb_clean, color=colour, lw=1.9, label=f"M = {order}, $n_0=0$")
        ax_right.loglog(signal, ppb_hard, color=colour, lw=1.3, ls="--",
                        label=f"M = {order}, $n_0=0.05$ hard")
        limit = minimum_photons_per_bit(order)
        ax_right.axhline(limit, color=colour, ls=":", lw=1.0)
        idx = int(np.nanargmin(ppb_hard))
        minima.append((order, float(signal[idx]), float(ppb_hard[idx]), limit))
    ax_right.set_xlabel("signal counts per pulsed slot $n_s$")
    ax_right.set_ylabel("photons per bit (detected signal counts)")
    ax_right.set_title("Photons per bit; dotted lines are the $1/\\log_2 M$ limits")
    ax_right.legend(fontsize=7, ncol=2)
    ax_right.grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)

    print("ppm_error_and_rate.py")
    print(f"   wrote screenshots/{OUT.name}")
    print("   background-free symbol error probability at n_s = 3:")
    for order, clean, _ in report:
        idx = int(np.argmin(np.abs(signal - 3.0)))
        print(f"      M = {order:3d}: {clean[idx]:.6f}  "
              f"(closed form exp(-n_s)(M-1)/M = "
              f"{float(np.exp(-signal[idx]) * (order - 1) / order):.6f})")
    print("   best photons per bit with n_0 = 0.05, hard decisions:")
    print(f"      {'M':>5} {'n_s at minimum':>16} {'photons/bit':>13} {'1/log2(M)':>11}")
    for order, s, ppb, limit in minima:
        print(f"      {order:5d} {s:16.4f} {ppb:13.5f} {limit:11.5f}")
    print("   the minimum is well above the limit: the limit costs an erasure")
    print("   probability approaching 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
