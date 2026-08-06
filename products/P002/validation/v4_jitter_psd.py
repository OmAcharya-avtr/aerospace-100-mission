"""V4 - disturbance-synthesis PSD verification.

Claim under test
----------------
``dynamics.synthesize_jitter`` produces a series whose measured one-sided PSD
matches the target PSD within tolerance, and whose variance matches the
integral of the target PSD over [0, fs/2].

Method
------
1. Synthesise N = 2^18 samples at fs = 2000 Hz for three PSD shapes.
2. Estimate the PSD by Welch (Hann window, 50 % overlap, nperseg = 8192) and
   average the ratio measured/target in decade bands.
3. Compare sample variance against the quadrature integral of the target PSD
   (Parseval check).
4. Verify that averaging M independent realisations reduces the scatter of
   the ratio as 1/sqrt(M), confirming the estimator is unbiased rather than
   accidentally matched.

Chi-square note: a Welch estimate with K averaged segments has a relative
standard deviation of about 1/sqrt(K) per bin. With N = 2^18 and
nperseg = 8192 there are K = 63 segments, so ~12.6 % per-bin scatter is
expected; band medians over many bins are far tighter.

Run: python validation/v4_jitter_psd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackforge.dynamics import JitterPSD, synthesize_jitter, welch_psd  # noqa: E402

FS = 2000.0
N = 2**18
NPERSEG = 8192


def band_report(f: np.ndarray, meas: np.ndarray, target: np.ndarray) -> list[tuple]:
    """Median measured/target ratio in decade bands."""
    rows = []
    for lo, hi in ((0.5, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, 900.0)):
        m = (f >= lo) & (f < hi)
        if m.sum() < 4:
            continue
        ratio = meas[m] / target[m]
        rows.append((lo, hi, int(m.sum()), float(np.median(ratio)),
                     float(np.mean(ratio)), float(np.std(ratio))))
    return rows


def main() -> int:
    """Run the PSD validation and print tables."""
    print("V4 - synthesised jitter PSD vs target PSD")
    print(f"fs = {FS} Hz, N = {N} samples ({N / FS:.1f} s), "
          f"Welch nperseg = {NPERSEG} (K = {2 * N // NPERSEG - 1} segments)")
    print(f"expected per-bin relative scatter ~ 1/sqrt(K) = "
          f"{1 / np.sqrt(2 * N / NPERSEG - 1):.3f}")
    print()
    cases = {
        "flat->f^-2 (S0=1e-12, fc=3 Hz)": JitterPSD(1e-12, 3.0, 2.0),
        "flat->f^-4 (S0=4e-12, fc=10 Hz)": JitterPSD(4e-12, 10.0, 4.0),
        "wideband  (S0=2e-11, fc=200 Hz)": JitterPSD(2e-11, 200.0, 2.0),
    }
    worst_band = 0.0
    worst_var = 0.0
    for i, (name, psd) in enumerate(cases.items()):
        x = synthesize_jitter(psd, N, FS, np.random.default_rng(1000 + i))
        f, p = welch_psd(x, FS, nperseg=NPERSEG)
        tgt = psd(f)
        print(f"case: {name}")
        print(f"   {'band [Hz]':>16} {'bins':>6} {'median':>9} {'mean':>9} {'std':>9}")
        for lo, hi, nb, med, mean, sd in band_report(f, p, tgt):
            worst_band = max(worst_band, abs(med - 1.0))
            print(f"   {f'{lo:g} - {hi:g}':>16} {nb:6d} {med:9.4f} "
                  f"{mean:9.4f} {sd:9.4f}")
        var_meas = float(np.var(x))
        var_ana = psd.variance(FS / 2.0)
        dev = (var_meas - var_ana) / var_ana
        worst_var = max(worst_var, abs(dev))
        print(f"   variance: measured {var_meas:.6e} rad^2, "
              f"target integral {var_ana:.6e} rad^2, deviation {dev:+.3%}")
        print(f"   RMS: measured {np.sqrt(var_meas):.4e} rad, "
              f"target {np.sqrt(var_ana):.4e} rad")
        print()

    print("ensemble convergence of the band-median ratio (case 1, 1-10 Hz band)")
    psd = JitterPSD(1e-12, 3.0, 2.0)
    print(f"   {'M realisations':>16} {'mean ratio':>12} {'std of ratio':>14}")
    for m_reps in (1, 4, 16):
        ratios = []
        for k in range(m_reps):
            x = synthesize_jitter(psd, 2**15, FS, np.random.default_rng(5000 + k))
            f, p = welch_psd(x, FS, nperseg=4096)
            band = (f >= 1.0) & (f < 10.0)
            ratios.append(np.mean(p[band] / psd(f)[band]))
        print(f"   {m_reps:16d} {np.mean(ratios):12.4f} "
              f"{(np.std(ratios) if m_reps > 1 else float('nan')):14.4f}")
    print()

    tol_band, tol_var = 0.10, 0.10
    ok = worst_band < tol_band and worst_var < tol_var
    print(f"PASS criteria: worst |band median - 1| < {tol_band:.0%} -> {worst_band:.3%}")
    print(f"               worst |variance deviation| < {tol_var:.0%} -> {worst_var:.3%}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
