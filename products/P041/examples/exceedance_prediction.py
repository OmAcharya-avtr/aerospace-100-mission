"""The AI result, including the part where the analytic closure is wrong.

Writes ``../screenshots/exceedance_prediction.png``.

Left: the measured fade-duration survival function against the exponential closure
of the level-crossing result, equation (16). The closure has the right mean by
construction and the wrong shape, which is the headline finding. Middle: the
reliability diagram for the learned model against the two state-blind baselines.
Right: the uncertainty output, split by whether the model was right or wrong.

Runtime: about 110 s on one core.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from codedfade.fade import (  # noqa: E402
    exponential_exceedance,
    lognormal_standard_level,
    markov_mean_fade_duration,
)
from codedfade.predictor import (  # noqa: E402
    FadeExceedancePredictor,
    brier_score,
    build_dataset,
    empirical_exceedance_baseline,
    reliability_table,
    roc_auc,
)

SI, TAU, FS, THRESHOLD, TARGET = 0.6, 2.0e-4, 1.0e6, 0.6, 14
SAMPLES = 150_000


def main() -> None:
    train = build_dataset(list(range(16)), SAMPLES, THRESHOLD, TARGET, SI, TAU, FS)
    test = build_dataset(list(range(100, 110)), SAMPLES, THRESHOLD, TARGET, SI, TAU, FS)
    model = FadeExceedancePredictor(160, 8, 0).fit(train.features, train.labels)
    out = model.predict(test.features)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    ax = axes[0]
    durations = np.sort(test.durations_samples)
    survival = 1.0 - np.arange(durations.size) / durations.size
    ax.step(durations, survival, where="post", color="#1f77b4", lw=1.6,
            label="measured survival, held-out seeds")
    u = lognormal_standard_level(THRESHOLD, SI)
    mfd = markov_mean_fade_duration(u, TAU, FS) * FS
    grid = np.linspace(1, durations.max(), 400)
    ax.plot(
        grid,
        exponential_exceedance(grid, mfd),
        color="#d62728",
        ls="--",
        lw=1.6,
        label=f"eq (16): exp(-t/MFD), MFD = {mfd:.1f} sym",
    )
    ax.axvline(TARGET, color="0.4", ls=":", lw=1.2)
    predicted = float(exponential_exceedance(float(TARGET), mfd))
    observed = float(test.labels.mean())
    ax.annotate(
        f"at t = {TARGET} sym:\npredicted {predicted:.3f}\nobserved {observed:.3f}\n"
        f"ratio {predicted / observed:.2f}",
        xy=(TARGET, observed),
        xytext=(TARGET * 3.0, 0.33),
        fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fade duration, symbols")
    ax.set_ylabel("P(T > t)")
    ax.set_title("The exponential closure has the right mean\nand the wrong shape")
    ax.legend(fontsize=8)

    ax = axes[1]
    mean_p, obs, count = reliability_table(out.probability, test.labels, bins=10)
    sel = count > 0
    ax.plot([0, 1], [0, 1], color="0.5", ls=":", lw=1.2, label="perfect calibration")
    ax.plot(
        mean_p[sel], obs[sel], marker="o", color="#2ca02c", lw=1.6,
        label=f"learned (Brier {brier_score(out.probability, test.labels):.4f}, "
              f"AUC {roc_auc(out.probability, test.labels):.3f})",
    )
    emp = empirical_exceedance_baseline(train.labels)
    ax.plot(
        [emp], [observed], marker="s", ms=9, color="#ff7f0e", ls="none",
        label=f"empirical constant (Brier "
              f"{brier_score(np.full(test.labels.size, emp), test.labels):.4f})",
    )
    ax.plot(
        [predicted], [observed], marker="D", ms=9, color="#d62728", ls="none",
        label=f"eq (16) constant (Brier "
              f"{brier_score(test.analytic_configured, test.labels):.4f})",
    )
    ax.set_xlabel("predicted P(T > t_target)")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability on held-out seeds")
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[2]
    wrong = np.abs(out.probability - test.labels) > 0.5
    bins = np.linspace(0, float(out.uncertainty.max()), 30)
    ax.hist(
        out.uncertainty[~wrong], bins=bins, color="#2ca02c", alpha=0.65, density=True,
        label=f"model right (n={int((~wrong).sum())}), "
              f"mean sigma {out.uncertainty[~wrong].mean():.4f}",
    )
    ax.hist(
        out.uncertainty[wrong], bins=bins, color="#d62728", alpha=0.65, density=True,
        label=f"model wrong (n={int(wrong.sum())}), "
              f"mean sigma {out.uncertainty[wrong].mean():.4f}",
    )
    ax.set_xlabel("per-tree standard deviation of the predicted probability")
    ax.set_ylabel("density")
    ax.set_title("The uncertainty output, split by outcome")
    ax.legend(fontsize=8)

    for a in axes:
        a.grid(True, alpha=0.3)
    fig.suptitle(
        f"fade-duration exceedance, lognormal SI {SI}, tau {TAU * 1e6:.0f} us, "
        f"threshold {THRESHOLD}, target {TARGET} symbols; "
        f"{train.features.shape[0]} train / {test.features.shape[0]} held-out events",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = os.path.join("..", "screenshots", "exceedance_prediction.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {os.path.basename(out_path)} to the screenshots directory")


if __name__ == "__main__":
    main()
