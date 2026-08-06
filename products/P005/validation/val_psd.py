"""Validation: Welch PSD known-answer cases.

Case A — pure sinusoid: x(t) = A sin(2 pi f0 t). Analytic answer: all
power at f0, total integrated one-sided PSD = A^2 / 2, band RMS around
f0 = A / sqrt(2) (Bendat & Piersol 2010, "Random Data", 4th ed., ch. 5).

Case B — white Gaussian noise, variance sigma^2, sample rate fs.
Analytic answer: flat one-sided PSD at sigma^2 / (fs/2), and
integral of the PSD = sigma^2 (Parseval).

Rerun:  python validation/val_psd.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jitterscope import band_rms, cumulative_rms, psd  # noqa: E402

FS = 1000.0
lines: list[str] = []


def log(msg: str = "") -> None:
    lines.append(msg)


log("=== jitterscope PSD known-answer validation ===")
log(f"numpy {np.__version__}; fs = {FS} Hz")
log()

# --- Case A: pure sinusoid -------------------------------------------------
A, F0 = 2.0e-6, 80.0
t = np.arange(0, 20.0, 1 / FS)
x = A * np.sin(2 * np.pi * F0 * t)
f, pxx = psd(x, FS, nperseg=4096)
df = f[1] - f[0]
f_peak = f[int(np.argmax(pxx))]
power = float(np.trapezoid(pxx, f))
rms_band = float(band_rms((f, pxx), [(70.0, 90.0)])[0])
rms_far = float(band_rms((f, pxx), [(200.0, 400.0)])[0])

log("Case A: pure sinusoid A = 2.0e-6 rad, f0 = 80.000 Hz, 20 s record")
log(f"  Welch: nperseg = 4096, hann, 50% overlap, resolution df = {df:.4f} Hz")
log(f"  peak frequency        : measured {f_peak:.4f} Hz   expected {F0:.4f} Hz   "
    f"|err| {abs(f_peak - F0):.4f} Hz ({abs(f_peak - F0) / df:.3f} bins)")
log(f"  integrated power      : measured {power:.6e} rad^2   expected {A**2 / 2:.6e} rad^2   "
    f"rel err {abs(power - A**2 / 2) / (A**2 / 2):.3e}")
log(f"  band RMS 70-90 Hz     : measured {rms_band:.6e} rad   expected "
    f"{A / np.sqrt(2):.6e} rad   rel err {abs(rms_band - A / np.sqrt(2)) / (A / np.sqrt(2)):.3e}")
log(f"  band RMS 200-400 Hz   : measured {rms_far:.6e} rad   expected ~0 "
    f"(leakage floor; ratio to tone band {rms_far / rms_band:.3e})")
log(f"  PASS criteria: peak within 1 bin, power rel err < 1e-2  ->  "
    f"{'PASS' if abs(f_peak - F0) <= df and abs(power - A**2 / 2) / (A**2 / 2) < 1e-2 else 'FAIL'}")
log()

# --- Case B: white noise ---------------------------------------------------
SIGMA = 1.0e-6
rng = np.random.default_rng(20260806)
xw = rng.normal(0.0, SIGMA, 200_000)
fw, pw = psd(xw, FS, nperseg=4096)
expected_level = SIGMA**2 / (FS / 2)
median_level = float(np.median(pw[1:-1]))
mean_level = float(np.mean(pw[1:-1]))
integ = float(np.trapezoid(pw, fw))
sample_var = float(np.var(xw))
_, cum = cumulative_rms((fw, pw))

log("Case B: white Gaussian noise sigma = 1.0e-6 rad, 200000 samples (200 s), seed 20260806")
log(f"  Welch: nperseg = 4096, hann, 50% overlap, {2 * 200_000 // 4096 - 1} averaged segments")
log(f"  PSD level (median)    : measured {median_level:.6e} rad^2/Hz   expected "
    f"{expected_level:.6e}   rel err {abs(median_level - expected_level) / expected_level:.3e}")
log(f"  PSD level (mean)      : measured {mean_level:.6e} rad^2/Hz   rel err "
    f"{abs(mean_level - expected_level) / expected_level:.3e}")
log(f"  flatness (std/mean over bins): {float(np.std(pw[1:-1]) / mean_level):.4f} "
    f"(chi^2_2k estimator: expected ~{np.sqrt(1 / (2 * 200_000 // 4096 - 1)):.4f})")
log(f"  integral of PSD       : measured {integ:.6e} rad^2   sample variance "
    f"{sample_var:.6e} rad^2   rel err {abs(integ - sample_var) / sample_var:.3e}")
log(f"  cumulative RMS (total): {cum[-1]:.6e} rad   sample std {np.std(xw):.6e} rad")
ok_b = (abs(median_level - expected_level) / expected_level < 0.05
        and abs(integ - sample_var) / sample_var < 0.05)
log(f"  PASS criteria: level and Parseval rel err < 5e-2  ->  {'PASS' if ok_b else 'FAIL'}")
log()

out = Path(__file__).resolve().parent / "val_psd_output.txt"
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"saved {out}")
