#!/usr/bin/env python3
"""The three-way comparison, in both dependence regimes, as a figure.

Left panel: stages independent. Right panel: stages correlated. Each point is
one held-out pipeline; the x axis is its true p99 from a large reference
sample and the y axis is a model's prediction. The diagonal is perfect.

What to look at: in the correlated panel the analytic sum-of-stages points sit
systematically below the diagonal, because dropping the covariance terms
understates the total variance and so under-predicts the tail. Under-prediction
is the unsafe direction for a deadline.

Writes ../screenshots/model_comparison.png. Runtime about 12 s.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latencynet.dataset import build_dataset, log_target
from latencynet.learned import LearnedTailPredictor
from latencynet.linear import LinearTailPredictor
from latencynet.metrics import log_accuracy
from latencynet.predictors import AnalyticTailPredictor

P = 0.99
N_TRAIN = 120
N_CALIBRATION = 40
N_TEST = 60
N_REFERENCE = 15_000
N_REFERENCE_TEST = 60_000
SEED = 20260402
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "model_comparison.png"

STYLES = (
    ("analytic_sum_indep", "tab:blue", "o"),
    ("linear_ols", "tab:green", "s"),
    ("learned_gbt", "tab:red", "^"),
)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=False)
    for ax, regime in zip(axes, ("independent", "correlated"), strict=True):
        dataset = build_dataset(
            regime,
            n_train=N_TRAIN,
            n_calibration=N_CALIBRATION,
            n_test=N_TEST,
            seed=SEED,
            n_reference=N_REFERENCE,
            n_reference_test=N_REFERENCE_TEST,
        )
        truth = log_target(dataset.test, P)
        models = {
            "analytic_sum_indep": AnalyticTailPredictor().fit(dataset.train, P),
            "linear_ols": LinearTailPredictor().fit(dataset.train, P),
            "learned_gbt": LearnedTailPredictor().fit(dataset.train, P),
        }
        truth_us = np.exp(truth) * 1.0e6
        print(f"\nregime={regime}  p={P}  held-out pipelines={len(dataset.test)}")
        for name, colour, marker in STYLES:
            pred = models[name].predict_log(dataset.test)
            acc = log_accuracy(pred, truth)
            ax.scatter(
                truth_us,
                np.exp(pred) * 1.0e6,
                s=18,
                alpha=0.7,
                c=colour,
                marker=marker,
                label=f"{name}: mean |dln q| {acc.mean_abs_log_error:.4f}, "
                      f"bias {acc.bias_log:+.4f}",
            )
            print(
                f"  {name:<22}mean |dln q| {acc.mean_abs_log_error:.5f}   "
                f"median {acc.median_abs_log_error:.5f}   bias {acc.bias_log:+.5f}   "
                f"mean relative error {acc.mean_relative_error * 100:.2f} %"
            )
        lo = float(truth_us.min()) * 0.75
        hi = float(truth_us.max()) * 1.25
        ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", label="perfect")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"true p{P * 100:g} from {N_REFERENCE_TEST} reference passes (us)")
        ax.set_ylabel(f"predicted p{P * 100:g} (us)")
        ax.set_title(f"{regime} stages")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"latencynet: held-out p{P * 100:g} prediction, two baselines and the learned model",
        fontsize=11,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
