"""OOK over lognormal fading (weak-turbulence FSO) vs AWGN.

Shows sigma_I^2 = 0.1 and 0.3 with the adaptive (CSI) threshold, plus the
fixed no-CSI threshold at sigma_I^2 = 0.3 exhibiting its BER floor.
Monte Carlo points overlay the Gauss-Hermite analytic averages.

Saves ../screenshots/ook_lognormal.png. Runtime ~15 s.
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

OUT = Path(__file__).resolve().parents[1] / "screenshots" / "ook_lognormal.png"

SNR = np.arange(0.0, 24.5, 0.5)
SNR_MC = np.arange(0.0, 23.0, 4.0)
N_CAP = 20_000_000

CASES = [
    ({"channel": "awgn"}, "AWGN (no fading)", "-"),
    ({"channel": "lognormal", "sigma_i2": 0.1}, r"lognormal $\sigma_I^2$=0.1", "-"),
    ({"channel": "lognormal", "sigma_i2": 0.3}, r"lognormal $\sigma_I^2$=0.3", "-"),
    ({"channel": "lognormal", "sigma_i2": 0.3, "threshold": 0.5},
     r"$\sigma_I^2$=0.3, fixed threshold (no CSI)", "--"),
]


def main() -> None:
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for kw, label, style in CASES:
        ana = analytic_ber("ook", SNR, **kw)
        (ln,) = ax.semilogy(SNR, np.maximum(ana.ber, 1e-16), style, label=f"{label} analytic")
        ana_mc = analytic_ber("ook", SNR_MC, **kw).ber
        keep = ana_mc > 150 / N_CAP
        n = int(min(N_CAP, n_bits_for_target(float(ana_mc[keep].min()), 150)))
        mc = mc_ber("ook", SNR_MC[keep], n=n, seed=7, **kw)
        yerr = np.vstack([mc.ber - mc.ci_low, mc.ci_high - mc.ber])
        ax.errorbar(mc.snr_db, mc.ber, yerr=yerr, fmt="o", ms=5, mfc="none",
                    color=ln.get_color(), label=f"{label} MC")
        print(f"{label}: n={n}, errors={mc.n_errors.tolist()}, {mc.elapsed_s:.1f} s")
    ax.set_xlabel(r"$E_b/N_0$ [dB]")
    ax.set_ylabel("Bit error ratio")
    ax.set_ylim(1e-8, 1.0)
    ax.set_xlim(SNR[0], SNR[-1])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("OOK over lognormal fading (weak turbulence) vs AWGN")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT} ({time.perf_counter() - t0:.1f} s total)")


if __name__ == "__main__":
    main()
