"""The learned LLR corrector against every analytic alternative.

Writes ``../screenshots/corrector_benchmark.png``.

Left: decoded bit error rate at Eb/N0 = 8 dB for three mismatch regimes. The
dashed line is the true-CSI bound, which nothing can cross. Note the two
honest results: the single-scalar rescaling, which is the obvious non-learned
fix, recovers almost nothing when the estimate is biased high, because a
scalar cannot move the decision threshold; and the **analytic** CSI-aware LLR
beats the learned corrector in all three regimes, so the learned model is not
the right answer here -- marginalising over the CSI error is.

Right: the LLR transfer characteristic. The learned corrector is plotted
against the plug-in LLR it was given, with the analytic CSI-aware LLR for
comparison. The learned curve bends the over-confident plug-in values back
towards zero and shifts the zero crossing, which is what the scalar cannot do.

This example uses smaller splits than ``validation/validate_corrector.py``
so that it fits the compute budget; the published numbers come from the
validation script.

Runtime: about 110 s on one shared core.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from softdecode.channel import LognormalFading  # noqa: E402
from softdecode.corrector import (  # noqa: E402
    LlrCorrector,
    build_features,
    tune_clip,
    tune_scale_and_clip,
)
from softdecode.csi import MultiplicativeCsiError, StaleCsiError  # noqa: E402
from softdecode.datasets import make_split, make_training_set  # noqa: E402
from softdecode.ldpc import make_regular_ldpc  # noqa: E402
from softdecode.llr import clip_llr  # noqa: E402
from softdecode.simulate import decode_ber, demap_ook  # noqa: E402

TRAIN_BLOCKS = 220
TUNE_BLOCKS = 300
REPORT_BLOCKS = 700
EBN0_DB = 8.0

code = make_regular_ldpc()
fading = LognormalFading(0.3)
cases = [
    ("bias +2 dB", MultiplicativeCsiError(2.0, 1.0)),
    ("bias -2 dB", MultiplicativeCsiError(-2.0, 1.0)),
    ("stale $\\rho$=0.8", StaleCsiError(0.8)),
]
methods = [
    "known_csi",
    "csi_aware_exact",
    "csi_aware_maxlog",
    "plugin",
    "plugin_scaled",
    "plugin_scaled_clipped",
    "learned",
]
results = {m: [] for m in methods}
transfer = None

for label, error in cases:
    train = make_training_set(
        code, fading, error, blocks_per_point=TRAIN_BLOCKS, ebn0_db=(4.0, 6.0, 8.0, 10.0)
    )
    corrector = LlrCorrector(n_estimators=24, max_depth=12, min_samples_leaf=150)
    corrector.fit(train.features, train.bits)
    tune = make_split(code, EBN0_DB, fading, error, TUNE_BLOCKS, "tune")
    tuned = tune_scale_and_clip(code, tune, fading, error)
    tune_features = build_features(tune, demap_ook(tune, "plugin", fading, error))
    tune_clip(code, tune, corrector, tune_features)

    report = make_split(code, EBN0_DB, fading, error, REPORT_BLOCKS, "report")
    plugin = demap_ook(report, "plugin", fading, error)
    features = build_features(report, plugin)
    learned = corrector.predict(features).reshape(report.codeword.shape)
    llrs = {
        "known_csi": demap_ook(report, "known_csi", fading, error),
        "csi_aware_exact": demap_ook(report, "csi_aware_exact", fading, error),
        "csi_aware_maxlog": demap_ook(report, "csi_aware_maxlog", fading, error),
        "plugin": plugin,
        "plugin_scaled": plugin * tuned["plugin_scaled"].scale,
        "plugin_scaled_clipped": clip_llr(
            plugin * tuned["plugin_scaled_clipped"].scale,
            tuned["plugin_scaled_clipped"].clip,
        ),
        "learned": learned,
    }
    print(f"\n{label}: tuned scale {tuned['plugin_scaled'].scale:.3f}, "
          f"joint ({tuned['plugin_scaled_clipped'].scale:.3f}, "
          f"{tuned['plugin_scaled_clipped'].clip:.2f}), "
          f"learned clip {corrector.clip:.2f}")
    for name in methods:
        rate = decode_ber(code, report, llrs[name]).rate
        results[name].append(rate)
        print(f"  {name:<24} BER {rate:.6e}")
    if label == "bias +2 dB":
        transfer = (
            plugin.ravel(),
            learned.ravel(),
            llrs["csi_aware_exact"].ravel(),
            plugin.ravel() * tuned["plugin_scaled"].scale,
        )

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
x = np.arange(len(cases))
width = 0.12
colours = ["black", "tab:blue", "tab:cyan", "tab:red", "tab:orange", "tab:brown", "tab:green"]
for i, (name, colour) in enumerate(zip(methods, colours, strict=True)):
    axes[0].bar(x + (i - 3) * width, results[name], width, label=name, color=colour,
                alpha=0.85)
axes[0].set_yscale("log")
axes[0].set_xticks(x)
axes[0].set_xticklabels([c[0] for c in cases])
axes[0].set_ylabel("decoded bit error rate")
axes[0].set_title(f"Mismatched CSI at $E_b/N_0$ = {EBN0_DB:.0f} dB, "
                  f"{REPORT_BLOCKS} report blocks")
axes[0].legend(fontsize=7.5, ncol=2)
axes[0].grid(alpha=0.3, axis="y", which="both")

plug, learn, aware, scaled = transfer
order = np.argsort(plug)
step = max(1, order.size // 4000)
sel = order[::step]
axes[1].plot(plug[sel], plug[sel], "k:", lw=1.2, label="plug-in (identity)")
axes[1].plot(plug[sel], scaled[sel], color="tab:orange", lw=1.6,
             label="plug-in $\\times\\ \\alpha$ (tuned scalar)")
axes[1].scatter(plug[sel], aware[sel], s=2, color="tab:blue", alpha=0.35,
                label="analytic CSI-aware")
axes[1].scatter(plug[sel], learn[sel], s=2, color="tab:green", alpha=0.35,
                label="learned corrector")
axes[1].axhline(0.0, color="grey", lw=0.8)
axes[1].axvline(0.0, color="grey", lw=0.8)
axes[1].set_xlim(-60, 60)
axes[1].set_ylim(-20, 20)
axes[1].set_xlabel("plug-in LLR given to the corrector")
axes[1].set_ylabel("corrected LLR")
axes[1].set_title("Transfer characteristic, bias +2 dB")
axes[1].legend(fontsize=8, markerscale=4)
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("../screenshots/corrector_benchmark.png", dpi=130)
print("\nwrote ../screenshots/corrector_benchmark.png")
