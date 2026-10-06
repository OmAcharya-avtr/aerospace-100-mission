"""The central figure: post-decoding FER against interleaver depth, by fade length.

Writes ``../screenshots/depth_knee.png``.

The knee sits where the depth passes the fade correlation length. Three correlation
times are plotted against the same code, the same SNR and the same channel seed, so
the three curves differ only in how long the fades last. The x axis is ``D/Lc``, the
dimensionless quantity that actually governs the result, with the raw depth on a
second axis.

Runtime: about 90 s on one core.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from codedfade.channel import ChannelConfig  # noqa: E402
from codedfade.link import CodedLink, uncoded_bit_error_rate  # noqa: E402
from codedfade.reedsolomon import ReedSolomon  # noqa: E402

SI = 0.6
FS = 1.0e6
SNR_DB = 14.0
CODEWORDS = 1024
DEPTHS = [1, 4, 16, 64, 256, 1024, 4096]
TAUS = [2.0e-5, 2.0e-4, 1.0e-3]


def main() -> None:
    code = ReedSolomon(31, 21, 5)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    colours = ["#1f77b4", "#d62728", "#2ca02c"]

    for colour, tau in zip(colours, TAUS, strict=True):
        cfg = ChannelConfig(SI, tau, FS, "lognormal", "exp", seed=7)
        lc = cfg.samples_per_correlation_time
        results = CodedLink(code, cfg, SNR_DB).depth_sweep(DEPTHS, CODEWORDS)
        ratio = np.array([r.depth / lc for r in results])
        fer = np.array([r.frame_error_rate for r in results])
        se = np.array([r.frame_error_rate_standard_error for r in results])
        ber = np.array([r.post_bit_error_rate for r in results])
        lat = np.array([r.latency_ms for r in results])
        label = f"tau = {tau * 1e6:.0f} us (Lc = {lc:.0f} sym)"
        axes[0].errorbar(
            ratio, np.maximum(fer, 1e-5), yerr=se, marker="o", color=colour, label=label
        )
        axes[1].plot(
            np.array(DEPTHS), np.maximum(ber, 1e-6), marker="s", color=colour, label=label
        )
        axes[2].plot(lat, np.maximum(fer, 1e-5), marker="^", color=colour, label=label)

    uncoded = uncoded_bit_error_rate(
        ChannelConfig(SI, 2.0e-4, FS, seed=7), SNR_DB, 400_000
    )
    axes[1].axhline(
        uncoded, color="0.3", ls="--", label=f"uncoded BER = {uncoded:.2e}"
    )

    axes[0].axvline(1.0, color="0.5", ls=":", lw=1.2)
    axes[0].annotate(
        "D = Lc",
        xy=(1.0, 2e-4),
        xytext=(1.6, 4e-4),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
    )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("interleaver depth / fade correlation length, D / Lc")
    axes[0].set_ylabel("frame error rate (one RS(31,21) codeword)")
    axes[0].set_title("The knee is at D/Lc ~ 1, not at a depth")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("interleaver depth D, symbols")
    axes[1].set_ylabel("post-decoding bit error rate")
    axes[1].set_title("Same data against raw depth: the curves do not line up")

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("end-to-end interleaving latency, ms at 1 Mbaud")
    axes[2].set_ylabel("frame error rate")
    axes[2].set_title("What the depth costs: FER against latency")

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"RS(31,21) over GF(2^5), OOK, mean SNR {SNR_DB:.0f} dB, lognormal SI {SI}, "
        f"{CODEWORDS} codewords per point, channel seed 7",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join("..", "screenshots", "depth_knee.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {os.path.basename(out)} to the screenshots directory")


if __name__ == "__main__":
    main()
