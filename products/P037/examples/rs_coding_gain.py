"""RS(255,223) coding gain: the two curves and the Eb/N0 separation at 1e-5.

Produces ../screenshots/rs_coding_gain.png: post-decoding bit error rate of
the RS link, Eqs. (9)/(10), against the uncoded bit error rate of Eq. (1),
with the Eb/N0 separation at 1e-5 marked. Both axes are in Eb/N0 per
information bit, so the 0.58 dB rate loss of the coded link is already
inside the curves.

The figure also marks the error-amplification region where Eq. (10) rises
above the channel bit error rate: below about Eb/N0 = 4.5 dB the expected
symbol-error count approaches E = 16 and a hard-decision RS decoder stops
helping. That is a real property of the code, not an artefact.
"""

from __future__ import annotations

import math
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "../src")

from framesync.channel import bpsk_ber  # noqa: E402
from framesync.fer import coding_gain_db  # noqa: E402
from framesync.rs import RS_E, RS_RATE, rs_output_bit_error_rate  # noqa: E402

TARGET = 1e-5


def main() -> int:
    t0 = time.time()
    snr = np.linspace(2.0, 12.0, 400)
    unc = bpsk_ber(snr)
    p_chan = bpsk_ber(snr, RS_RATE)
    rs_out = rs_output_bit_error_rate(p_chan)

    g_ber = coding_gain_db(TARGET, metric="ber")
    g_fer = coding_gain_db(TARGET, metric="fer")
    g_inf = 10.0 * math.log10(RS_RATE * (RS_E + 1))

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.semilogy(snr, unc, "-", color="#1f3b73", lw=1.8,
                label="uncoded BPSK, Eq. (1)")
    ax.semilogy(snr, p_chan, "--", color="#6b7280", lw=1.2,
                label="RS channel bit error rate (R = 223/255)")
    ax.semilogy(snr, rs_out, "-", color="#9b2226", lw=1.8,
                label="RS(255,223) post-decoding, Eqs. (9)/(10)")
    ax.axhline(TARGET, color="grey", ls=":", lw=1.0)
    for x, colour in ((g_ber["ebn0_rs_db"], "#9b2226"),
                      (g_ber["ebn0_uncoded_db"], "#1f3b73")):
        ax.plot([x], [TARGET], "o", color=colour, ms=7)
        ax.axvline(x, color=colour, ls=":", lw=0.9)
    ax.annotate(
        f"gain {g_ber['gain_db']:.2f} dB at BER {TARGET:g}",
        xy=((g_ber["ebn0_rs_db"] + g_ber["ebn0_uncoded_db"]) / 2, TARGET * 2.2),
        ha="center", fontsize=9,
    )
    ax.set_xlabel("Eb/N0 per information bit (dB)")
    ax.set_ylabel("bit error rate")
    ax.set_title("RS(255,223) coding gain over an uncoded BPSK/AWGN link\n"
                 f"asymptotic bound 10 log10(R(E+1)) = {g_inf:.2f} dB")
    ax.set_ylim(1e-12, 1.0)
    ax.set_xlim(2.0, 12.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    out = "../screenshots/rs_coding_gain.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    print("RS(255,223) coding gain (framesync example 4)")
    print("-" * 72)
    print(f"code rate R = 223/255 = {RS_RATE:.9f} ({10 * math.log10(RS_RATE):.4f} dB "
          f"of rate loss)")
    for name, g in (("BER", g_ber), ("FER", g_fer)):
        print(f"{name} = {TARGET:g}: uncoded {g['ebn0_uncoded_db']:7.4f} dB, "
              f"RS {g['ebn0_rs_db']:7.4f} dB, gain {g['gain_db']:6.4f} dB")
    print(f"asymptotic coding gain bound 10 log10(R(E+1)) = {g_inf:.4f} dB")
    print("measured gain is well below the asymptote, as expected for a high-rate")
    print("hard-decision block code at a moderate error rate")
    crossover = snr[np.argmax(rs_out < p_chan)]
    print(f"\nEq. (10) crosses below the channel bit error rate at about "
          f"{crossover:.2f} dB;")
    print("below that the hard-decision decoder amplifies errors rather than")
    print("correcting them (see README Limitations)")
    print(f"\nsaved {out} ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
