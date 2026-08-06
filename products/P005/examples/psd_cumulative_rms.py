"""Example: Welch PSD + band RMS table + cumulative RMS plot.

Generates 60 s of nominal synthetic pointing telemetry (reaction-wheel
tones at 45/90/135 Hz over colored + white noise), estimates the PSD,
integrates band-limited RMS jitter, and saves a two-panel figure to
../screenshots/psd_cumulative_rms.png. Also prints the band RMS table
and the implied Gaussian-beam pointing loss.

Run from examples/:  python psd_cumulative_rms.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jitterscope import band_rms, cumulative_rms, generate_telemetry, pointing_loss_avg, psd

FS = 1000.0
BANDS = [(0.5, 10.0), (10.0, 40.0), (40.0, 100.0), (100.0, 250.0), (250.0, 500.0)]

t, x, _ = generate_telemetry(duration_s=60.0, fs=FS, seed=2026)
f, pxx = psd(x, FS, nperseg=4096)
fc, cum = cumulative_rms((f, pxx))
rms = band_rms((f, pxx), BANDS)

print("Band-limited RMS jitter (sigma = sqrt(int PSD df)):")
print(f"{'band [Hz]':>18} {'RMS [urad]':>12}")
for (lo, hi), r in zip(BANDS, rms):
    print(f"{lo:8.1f}-{hi:8.1f} {r * 1e6:12.4f}")
sigma_total = cum[-1]
print(f"{'total (Parseval)':>18} {sigma_total * 1e6:12.4f}")

theta_div = 10e-6  # 1/e^2 half-angle divergence [rad]
loss = pointing_loss_avg(sigma_total / np.sqrt(2), theta_div)
print(f"\nGaussian-beam average pointing loss for theta_div = {theta_div * 1e6:.0f} urad: "
      f"{loss:.4f} ({-10 * np.log10(loss):.2f} dB)")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ax1.loglog(f[1:], pxx[1:] * 1e12, lw=0.8, color="tab:blue")
ax1.set_ylabel("PSD [urad$^2$/Hz]")
ax1.set_title("jitterscope — Welch PSD (hann, nperseg=4096) and cumulative RMS jitter")
for k in (1, 2, 3):
    ax1.axvline(45 * k, color="tab:red", ls=":", lw=0.8)
ax1.annotate("wheel harmonics 45/90/135 Hz", xy=(48, ax1.get_ylim()[1] * 0.3),
             color="tab:red", fontsize=8)
ax1.grid(True, which="both", alpha=0.3)

ax2.semilogx(fc[1:], cum[1:] * 1e6, lw=1.2, color="tab:green")
ax2.set_xlabel("frequency [Hz]")
ax2.set_ylabel("cumulative RMS [urad]")
ax2.axhline(sigma_total * 1e6, color="k", ls="--", lw=0.8,
            label=f"total RMS = {sigma_total * 1e6:.3f} urad")
ax2.legend(loc="lower right")
ax2.grid(True, which="both", alpha=0.3)

out = Path(__file__).resolve().parents[1] / "screenshots" / "psd_cumulative_rms.png"
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
