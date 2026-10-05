#!/usr/bin/env python3
"""Measured coverage against nominal, for native and conformal intervals.

An interval that claims 90 % and delivers 60 % is worse than no interval,
because it is believed. This example measures, on held-out pipelines, what
fraction each model's interval actually contains, at seven nominal levels,
and plots it against the diagonal with binomial error bars.

What to look at: the analytic model's native interval sits far below the
diagonal. It propagates probe sampling uncertainty only -- it has no term for
the Fenton-Wilkinson approximation error and none for stage dependence -- so
it is over-confident by construction, and this figure is what that looks like.
The split-conformal intervals track the diagonal.

Writes ../screenshots/interval_calibration.png. Runtime about 12 s.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latencynet.conformal import ConformalPredictor, coverage_quantisation
from latencynet.dataset import build_dataset, log_target
from latencynet.linear import LinearTailPredictor
from latencynet.metrics import interval_coverage
from latencynet.predictors import AnalyticTailPredictor

P = 0.99
REGIME = "correlated"
N_TRAIN = 120
N_CALIBRATION = 60
N_TEST = 150
N_REFERENCE = 15_000
N_REFERENCE_TEST = 60_000
SEED = 20260402
LEVELS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
OUT = Path(__file__).resolve().parents[1] / "screenshots" / "interval_calibration.png"


def main() -> None:
    dataset = build_dataset(
        REGIME,
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
    }
    series: dict[str, tuple[list[float], list[float], list[float]]] = {}
    print(f"regime={REGIME}  p={P}  held-out pipelines={len(dataset.test)}  "
          f"calibration={len(dataset.calibration)}")
    print(f"conformal coverage is quantised in steps of 1/(m+1) = "
          f"{coverage_quantisation(len(dataset.calibration)):.4f}")
    for name, model in models.items():
        for kind in ("native", "conformal"):
            measured: list[float] = []
            errors: list[float] = []
            widths: list[float] = []
            for level in LEVELS:
                if kind == "native":
                    pred = model.predict_log_interval(dataset.test, level=level)
                else:
                    wrapper = ConformalPredictor.from_fitted(model, P, level=level)
                    wrapper.calibrate(dataset.calibration)
                    pred = wrapper.predict_log_interval(dataset.test)
                cov = interval_coverage(pred, truth)
                measured.append(cov.measured)
                errors.append(cov.standard_error)
                widths.append(cov.mean_log_width)
            series[f"{name} {kind}"] = (measured, errors, widths)
            print(f"\n  {name} {kind}")
            for level, m, e, w in zip(LEVELS, measured, errors, widths, strict=True):
                print(
                    f"    nominal {level:.2f}   measured {m:.3f} +/- {e:.3f}   "
                    f"mean log width {w:.4f}"
                )

    fig, (ax_cov, ax_width) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax_cov.plot([0.25, 1.0], [0.25, 1.0], color="black", lw=1.0, ls="--", label="perfect")
    styles = {
        "analytic_sum_indep native": ("tab:blue", "o", "-"),
        "analytic_sum_indep conformal": ("tab:blue", "o", ":"),
        "linear_ols native": ("tab:green", "s", "-"),
        "linear_ols conformal": ("tab:green", "s", ":"),
    }
    for label, (measured, errors, widths) in series.items():
        colour, marker, ls = styles[label]
        ax_cov.errorbar(
            LEVELS, measured, yerr=errors, color=colour, marker=marker, ls=ls,
            capsize=3, ms=5, lw=1.2, label=label,
        )
        ax_width.plot(LEVELS, widths, color=colour, marker=marker, ls=ls, lw=1.2, label=label)
    ax_cov.set_xlabel("nominal coverage")
    ax_cov.set_ylabel("measured coverage on held-out pipelines")
    ax_cov.set_title(f"calibration, p{P * 100:g}, {REGIME} stages, n={len(dataset.test)}")
    ax_cov.set_xlim(0.25, 1.0)
    ax_cov.set_ylim(0.0, 1.02)
    ax_cov.legend(fontsize=7.5, loc="upper left")
    ax_cov.grid(alpha=0.3)

    ax_width.set_xlabel("nominal coverage")
    ax_width.set_ylabel("mean interval width in log space")
    ax_width.set_title("cost of coverage: a log width of 0.2 is a factor 1.22 end to end")
    ax_width.legend(fontsize=7.5, loc="upper left")
    ax_width.grid(alpha=0.3)

    fig.suptitle("latencynet: prediction-interval calibration and width", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print(f"\nwrote {OUT}")
    print(
        "Note: np.mean of the measured coverages is not a summary of anything; "
        "read the rows."
    )


if __name__ == "__main__":
    main()
