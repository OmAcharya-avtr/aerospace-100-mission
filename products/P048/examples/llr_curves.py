"""Exact, max-log and known-CSI LLRs against the received sample.

Writes ``../screenshots/llr_curves.png``.

The point of the left panel is that the exact LLR of a fading channel is not
affine in ``y``: the known-CSI LLR is a straight line, and marginalising over
the fading bends it and flattens its tails, because a large ``|y|`` is
explained either by a bit or by a deep fade. The right panel is the max-log
error, which is one-signed and of order one nat over the whole range.

Runtime: under 10 s.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from softdecode.channel import (  # noqa: E402
    GammaGammaFading,
    LognormalFading,
    amplitude_quadrature,
)
from softdecode.detection import DetectionModel  # noqa: E402
from softdecode.llr import (  # noqa: E402
    llr_ook_known_csi,
    llr_ook_marginal,
    llr_ook_maxlog,
)

det = DetectionModel(1.0)
amplitude = det.ook_amplitude(10.0)
y = np.linspace(-0.6 * amplitude, 1.8 * amplitude, 601)

models = [
    ("lognormal $\\sigma_I^2=0.05$", LognormalFading(0.05), "tab:blue"),
    ("lognormal $\\sigma_I^2=0.30$", LognormalFading(0.30), "tab:orange"),
    ("lognormal $\\sigma_I^2=1.00$", LognormalFading(1.00), "tab:green"),
    ("gamma-gamma (4, 2)", GammaGammaFading(4.0, 2.0), "tab:red"),
]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

axes[0].plot(
    y,
    llr_ook_known_csi(y, amplitude, 1.0, det.sigma),
    "k--",
    lw=1.6,
    label="known CSI, $h=1$ (affine)",
)
for label, model, colour in models:
    hq, wq = amplitude_quadrature(model)
    axes[0].plot(y, llr_ook_marginal(y, amplitude, hq, wq, det.sigma), color=colour,
                 lw=1.8, label=f"exact, {label}")
axes[0].axvline(amplitude / 2, color="grey", lw=0.8, ls=":")
axes[0].annotate("known-CSI threshold\n$y = ah/2$", (amplitude / 2, -45),
                 xytext=(amplitude / 2 + 0.6, -42), fontsize=8, color="grey")
axes[0].axhline(0.0, color="grey", lw=0.8)
axes[0].set_xlabel("received sample $y$ (noise units, $\\sigma=1$)")
axes[0].set_ylabel("LLR  $\\log P(0)/P(1)$")
axes[0].set_title(f"Exact OOK LLR, $E_b/N_0$ = 10 dB, $a$ = {amplitude:.3f}")
axes[0].set_ylim(-60, 25)
axes[0].legend(fontsize=8, loc="upper right")
axes[0].grid(alpha=0.3)

for label, model, colour in models:
    hq, wq = amplitude_quadrature(model)
    exact = llr_ook_marginal(y, amplitude, hq, wq, det.sigma)
    maxlog = llr_ook_maxlog(y, amplitude, hq, wq, det.sigma)
    axes[1].plot(y, maxlog - exact, color=colour, lw=1.8, label=label)
axes[1].axhline(0.0, color="grey", lw=0.8)
axes[1].axhline(np.log(2.0), color="k", lw=0.8, ls=":")
axes[1].annotate("$\\log 2$", (y[5], np.log(2.0) + 0.05), fontsize=8)
axes[1].set_xlabel("received sample $y$ (noise units, $\\sigma=1$)")
axes[1].set_ylabel("max-log minus exact (nats)")
axes[1].set_title("Max-log error: one-signed, order one nat")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../screenshots/llr_curves.png", dpi=130)
print("wrote ../screenshots/llr_curves.png")
for label, model, _ in models:
    hq, wq = amplitude_quadrature(model)
    exact = llr_ook_marginal(y, amplitude, hq, wq, det.sigma)
    maxlog = llr_ook_maxlog(y, amplitude, hq, wq, det.sigma)
    print(f"  {label:<30} nodes {hq.size:>4}  max-log error "
          f"min {np.min(maxlog - exact):+.6f}  max {np.max(maxlog - exact):+.6f}")
