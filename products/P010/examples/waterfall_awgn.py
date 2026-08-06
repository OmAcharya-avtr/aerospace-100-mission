"""BER waterfall: analytic vs Monte Carlo for OOK, BPSK and 4/16-PPM over AWGN.

Saves ../screenshots/waterfall_awgn.png. Runtime ~20 s on 2 CPU cores.
Run from the examples/ directory:  python3 waterfall_awgn.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from berbench import analytic_ber, mc_ber, n_bits_for_target  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "waterfall_awgn.png"

CASES = [
    ("ook", {}, "OOK (optimal threshold)"),
    ("bpsk", {}, "BPSK"),
    ("ppm", {"M": 4}, "4-PPM (exact)"),
    ("ppm", {"M": 16}, "16-PPM (exact)"),
]
SNR = np.arange(0.0, 14.5, 0.5)  # analytic grid, dB
SNR_MC = np.arange(0.0, 13.0, 2.0)  # MC points, dB
N_CAP = 20_000_000  # per-point bit cap keeps total runtime ~20 s


def main() -> None:
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for mod, kw, label in CASES:
        ana = analytic_ber(mod, SNR, **kw)
        (ln,) = ax.semilogy(SNR, np.maximum(ana.ber, 1e-16), label=f"{label} analytic")
        ana_mc = analytic_ber(mod, SNR_MC, **kw).ber
        # size each MC point for >= ~150 expected errors, capped for runtime
        keep = ana_mc > 150 / N_CAP
        n = int(min(N_CAP, n_bits_for_target(float(ana_mc[keep].min()), 150)))
        mc = mc_ber(mod, SNR_MC[keep], n=n, seed=2026, **kw)
        yerr = np.vstack([mc.ber - mc.ci_low, mc.ci_high - mc.ber])
        ax.errorbar(mc.snr_db, mc.ber, yerr=yerr, fmt="o", ms=5, mfc="none",
                    color=ln.get_color(), label=f"{label} MC (95% CI)")
        print(f"{label}: n={n} bits/point, errors={mc.n_errors.tolist()}, "
              f"{mc.elapsed_s:.1f} s")
    # 16-PPM union bound for comparison
    ub = analytic_ber("ppm", SNR, M=16, ppm_method="union")
    ax.semilogy(SNR, np.maximum(ub.ber, 1e-16), ":", color="gray",
                label="16-PPM union bound")
    ax.set_xlabel(r"$E_b/N_0$ [dB]")
    ax.set_ylabel("Bit error ratio")
    ax.set_ylim(1e-8, 1.0)
    ax.set_xlim(SNR[0], SNR[-1])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("BER over AWGN: analytic vs Monte Carlo (berbench 0.1.0)")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT} ({time.perf_counter() - t0:.1f} s total)")


if __name__ == "__main__":
    main()
