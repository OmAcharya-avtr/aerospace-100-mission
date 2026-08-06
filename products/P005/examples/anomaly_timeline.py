"""Example: anomaly-score timeline with injected faults marked.

Fits both the band z-score baseline and the MLP autoencoder-equivalent
NominalModel on 60 s of nominal telemetry, then scores a 60 s test
record containing three injected faults (new tone at 20 s, band-energy
shift at 35 s, transients from 48 s). Saves
../screenshots/anomaly_timeline.png.

Run from examples/:  python anomaly_timeline.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jitterscope import BandZScoreBaseline, FeatureExtractor, NominalModel, detect
from jitterscope import generate_telemetry

FS = 1000.0
FAULTS = [
    {"kind": "new_tone", "t_start": 20.0, "freq_hz": 137.0, "rms": 1.0e-6},
    {"kind": "band_shift", "t_start": 35.0, "f_lo": 200.0, "f_hi": 300.0, "factor": 8.0},
    {"kind": "transient", "t_start": 48.0, "rate_hz": 1.0, "amp": 8e-6, "decay_s": 0.05},
]

_, x_nom, _ = generate_telemetry(60.0, FS, seed=2026)
t, x_test, mask = generate_telemetry(60.0, FS, seed=777, faults=FAULTS)

ext = FeatureExtractor(fs=FS)
feats, _ = ext.transform(x_nom)
mlp = NominalModel(seed=0).fit(feats)
base = BandZScoreBaseline().fit(feats)

res_mlp = detect(x_test, model=mlp, extractor=ext)
res_base = detect(x_test, model=base, extractor=ext)
print(f"MLP:      {res_mlp.n_anomalous}/{res_mlp.scores.size} windows flagged, "
      f"threshold {res_mlp.threshold:.4g}")
print(f"baseline: {res_base.n_anomalous}/{res_base.scores.size} windows flagged, "
      f"threshold {res_base.threshold:.4g}")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(t, x_test * 1e6, lw=0.3, color="0.4")
axes[0].set_ylabel("telemetry [urad]")
axes[0].set_title("jitterscope — anomaly-score timeline with injected faults")

for ax, res, name in (
    (axes[1], res_base, "baseline band z-score (max |z|)"),
    (axes[2], res_mlp, "MLP autoencoder (reconstruction MSE)"),
):
    ax.semilogy(res.window_centers_s, res.scores, lw=1.0, color="tab:blue")
    ax.axhline(res.threshold, color="k", ls="--", lw=0.9,
               label=f"threshold (q0.995 nominal) = {res.threshold:.3g}")
    flagged = res.flags
    ax.plot(res.window_centers_s[flagged], res.scores[flagged], "r.", ms=5,
            label=f"flagged ({int(flagged.sum())})")
    ax.set_ylabel(name.split(" (")[0] + "\nscore")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

for ax in axes:
    for f0 in (20.0, 35.0, 48.0):
        ax.axvline(f0, color="tab:orange", ls=":", lw=1.0)
axes[0].annotate("new tone", xy=(20.3, axes[0].get_ylim()[1] * 0.8), fontsize=8,
                 color="tab:orange")
axes[0].annotate("band shift", xy=(35.3, axes[0].get_ylim()[1] * 0.8), fontsize=8,
                 color="tab:orange")
axes[0].annotate("transients", xy=(48.3, axes[0].get_ylim()[1] * 0.8), fontsize=8,
                 color="tab:orange")
axes[2].set_xlabel("time [s]")

out = Path(__file__).resolve().parents[1] / "screenshots" / "anomaly_timeline.png"
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"saved {out}")
