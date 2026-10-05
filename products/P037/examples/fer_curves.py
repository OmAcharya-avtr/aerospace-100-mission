"""Frame error rate against Eb/N0 for the three links, with error bars.

Produces ../screenshots/fer_curves.png: the analytic uncoded curve
(Eq. (11)), the semi-analytic RS(255,223) curve (Eqs. (7)/(8)), and measured
Monte Carlo points for all three links with their binomial standard errors.
The convolutional link is Monte Carlo only; no analytic curve is drawn for
it because none is claimed.
"""

from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "../src")

from framesync.channel import bpsk_ber  # noqa: E402
from framesync.fer import (  # noqa: E402
    measure_conv_fer,
    measure_rs_fer,
    measure_uncoded_fer,
    uncoded_fer,
)
from framesync.frames import FrameGeometry  # noqa: E402
from framesync.rs import RS_RATE, rs_frame_error_rate  # noqa: E402

SEED = 20261005
GEOM = FrameGeometry(data_octets=1115, fecf=True, asm_bits=32)
CONV_INFO_BITS = 512


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    fine = np.linspace(2.0, 13.0, 220)
    unc_curve = uncoded_fer(fine, GEOM.frame_bits)
    rs_curve = rs_frame_error_rate(bpsk_ber(fine, RS_RATE), 5)

    unc_pts = [measure_uncoded_fer(x, GEOM, 1500, rng) for x in (8.0, 9.0, 10.0, 11.0)]
    rs_pts = [measure_rs_fer(x, n, rng, interleave=5)
              for x, n in ((5.25, 120), (5.5, 150), (5.75, 200), (6.0, 250))]
    conv_pts = [measure_conv_fer(x, CONV_INFO_BITS, n, rng, soft=True)
                for x, n in ((2.0, 250), (3.0, 250), (4.0, 300))]

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.semilogy(fine, unc_curve, "-", color="#1f3b73", lw=1.8,
                label="uncoded, Eq. (11), n = 8936 bits")
    ax.semilogy(fine, rs_curve, "-", color="#9b2226", lw=1.8,
                label="RS(255,223) I=5, Eqs. (7)/(8)")

    for pts, colour, marker, label in (
        (unc_pts, "#1f3b73", "o", "uncoded measured"),
        (rs_pts, "#9b2226", "s", "RS(255,223) measured"),
        (conv_pts, "#2a7f62", "^", "conv (171,133) r=1/2 soft, measured"),
    ):
        x = [p.ebn0_db for p in pts]
        y = [max(p.fer, 1e-6) for p in pts]
        err = [p.stderr for p in pts]
        ax.errorbar(x, y, yerr=err, fmt=marker, color=colour, capsize=3,
                    ms=6, ls="none", label=label)

    ax.set_xlabel("Eb/N0 per information bit (dB)")
    ax.set_ylabel("frame error rate")
    ax.set_title("CCSDS telemetry frame error rate, BPSK over AWGN\n"
                 "error bars are binomial standard errors")
    ax.set_ylim(1e-6, 2.0)
    ax.set_xlim(2.0, 13.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    out = "../screenshots/fer_curves.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print("frame error rate against Eb/N0 (framesync example 1)")
    print("-" * 72)
    for pts in (unc_pts, rs_pts, conv_pts):
        for p in pts:
            print("  " + p.as_row())
    print(f"\nconvolutional frames carry {CONV_INFO_BITS} information bits, so the")
    print("three links are NOT the same frame length; the figure is a link")
    print("comparison, not a like-for-like frame-length comparison.")
    print(f"\nsaved {out} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
