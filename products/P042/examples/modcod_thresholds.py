"""Example 2: where the MODCOD thresholds come from.

Writes ``../screenshots/modcod_thresholds.png``.

Three panels:

1. Monte Carlo uncoded BER through the real modulator and detector, against the
   closed forms. Notice BPSK and QPSK sit on their exact curves, while 8PSK and
   16QAM depart from theirs below about 10 dB -- those closed forms are
   nearest-neighbour high-SNR approximations and are plotted dashed for that
   reason. The thresholds are taken from the *measured* points, never the curves.
2. Post-decoding BER per MODCOD, from the measured channel BER through the exact
   Reed-Solomon bounded-distance accounting. Notice how steep the waterfalls are:
   RS(255,127) with t = 64 goes from 1e-2 to 1e-18 in under a decibel, which is
   why a threshold is a meaningful idea at all here, and why a stale channel
   estimate is so expensive -- half a decibel of prediction error moves you from
   working to not working.
3. The ladder: required SNR against spectral efficiency, with the Monte Carlo
   threshold uncertainty as error bars. Notice the error bars are small (worst
   0.05 dB) and the steps are 1.1 to 3.6 dB apart, so the granularity of the
   ladder, not the threshold measurement, is what limits a perfect policy.

Runtime: about 25 s on one core.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from acmpilot.modcod import measure_thresholds  # noqa: E402
from acmpilot.modulation import theoretical_ber  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "screenshots" / "modcod_thresholds.png"
OUT_REL = "screenshots/modcod_thresholds.png"
EXACT = {"BPSK", "QPSK"}


def main() -> int:
    table, curves = measure_thresholds(seed=20261006)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    ax = axes[0]
    for i, name in enumerate(sorted(curves)):
        cur = curves[name]
        colour = f"C{i}"
        mask = cur["ber"] > 0
        ax.semilogy(
            cur["snr_db"][mask], cur["ber"][mask], "o", ms=3, color=colour,
            label=f"{name} measured",
        )
        fine = np.linspace(cur["snr_db"][0], cur["snr_db"][-1], 240)
        style = "-" if name in EXACT else "--"
        ax.semilogy(
            fine, theoretical_ber(name, fine), style, lw=1.2, color=colour,
            label=f"{name} closed form ({'exact' if name in EXACT else 'high-SNR approx'})",
        )
    ax.set_xlabel("Es/N0, dB")
    ax.set_ylabel("uncoded bit error rate")
    ax.set_ylim(1e-4, 1.0)
    ax.set_title("Uncoded BER: Monte Carlo against closed form")
    ax.legend(fontsize=6.5, ncol=1)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for i, mode in enumerate(table.modcods):
        cur = curves[mode.modulation]
        post = mode.code.output_bit_error_rate(cur["ber"])
        mask = post > 1e-14
        ax.semilogy(
            cur["snr_db"][mask], post[mask], lw=1.4, color=f"C{i}",
            label=f"{mode.name} (eta {mode.spectral_efficiency:.3f})",
        )
    ax.axhline(
        table.target_ber, color="k", ls="--", lw=1.0,
        label=f"target BER {table.target_ber:g}",
    )
    for thr in table.thresholds_db:
        ax.axvline(thr, color="0.75", lw=0.6, zorder=0)
    ax.set_xlabel("Es/N0, dB")
    ax.set_ylabel("post-decoding bit error rate")
    ax.set_ylim(1e-12, 1.0)
    ax.set_xlim(0.0, 20.0)
    ax.set_title("Post-decoding BER and the measured thresholds")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    eta = table.spectral_efficiencies
    ax.errorbar(
        table.thresholds_db, eta, xerr=table.threshold_sigma_db, fmt="o-",
        capsize=3, lw=1.3, ms=5,
    )
    for mode, thr, e in zip(table.modcods, table.thresholds_db, eta, strict=True):
        ax.annotate(
            mode.name, (thr, e), textcoords="offset points", xytext=(6, -9), fontsize=6.5
        )
    steps = np.diff(table.thresholds_db)
    ax.set_xlabel("required SNR per symbol, dB (error bars: Monte Carlo 1 sigma)")
    ax.set_ylabel("spectral efficiency, bit/symbol")
    ax.set_title(
        f"The ladder: steps {steps.min():.2f} to {steps.max():.2f} dB, "
        f"worst threshold sigma {np.nanmax(table.threshold_sigma_db):.3f} dB"
    )
    ax.grid(alpha=0.3)

    fig.suptitle(
        "acmpilot: MODCOD thresholds measured in this package, not taken from a "
        "standard. DVB-S2 numbers are several dB better at equal efficiency.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=120)
    plt.close(fig)
    for mode, thr, sig in zip(
        table.modcods, table.thresholds_db, table.threshold_sigma_db, strict=True
    ):
        print(f"{mode.name:22s} eta={mode.spectral_efficiency:6.3f} thr={thr:7.3f} dB +/-{sig:.3f}")
    print(f"ladder steps, dB: {np.round(steps, 3).tolist()}")
    print(f"written: {OUT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
