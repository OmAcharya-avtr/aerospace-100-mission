"""Example: designed vs measured window false-alarm probability, and what breaks it.

Left panel: the design value and an independent Monte-Carlo measurement for the
limit check, EWMA and CUSUM, with 95 % Wilson intervals.  The intervals are the
point of the plot; a bar chart without them would be a claim rather than a
measurement.

Right panel: the same thresholds applied to serially correlated nominal data.
The designs assume independent samples, so this is where they stop delivering.

Writes ``../screenshots/far_design_vs_measured.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from telemetryool.arl import (  # noqa: E402
    cusum_window_false_alarm,
    design_cusum_h,
    design_ewma_L,
    design_ool_limit,
    ewma_window_false_alarm,
    ool_window_false_alarm,
)
from telemetryool.calibration import estimate_rate  # noqa: E402
from telemetryool.charts import CusumChart, EwmaChart  # noqa: E402
from telemetryool.runs import any_run  # noqa: E402
from telemetryool.synthetic import NominalModel, generate_nominal  # noqa: E402

ALPHA = 0.05
WINDOW = 100
WINDOWS = 20_000
RHOS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
PERSISTENCE = 3


def nominal(rho: float, seed: int) -> np.ndarray:
    return generate_nominal(
        NominalModel(1, rho_time=rho), WINDOWS, WINDOW, np.random.default_rng(seed)
    )[:, :, 0]


def main() -> int:
    cusum_design = design_cusum_h(ALPHA, 0.5, WINDOW)
    ewma_design = design_ewma_L(ALPHA, 0.2, WINDOW)
    ool_design = design_ool_limit(ALPHA, PERSISTENCE, WINDOW)
    p_exceed = 2.0 * (1.0 - float(norm.cdf(ool_design.threshold)))

    ewma_chart = EwmaChart(lam=0.2, limit_mult=ewma_design.threshold)
    cusum_chart = CusumChart(k=0.5, h=cusum_design.threshold)
    methods = {
        f"limit check\npersistence={PERSISTENCE}\nL={ool_design.threshold:.3f}": (
            lambda block: any_run(np.abs(block) > ool_design.threshold, PERSISTENCE),
            ool_window_false_alarm(p_exceed, PERSISTENCE, WINDOW),
        ),
        f"EWMA\nlam=0.2\nL={ewma_design.threshold:.3f}": (
            lambda block: any_run(ewma_chart.breach_mask(block), 1),
            ewma_window_false_alarm(0.2, ewma_design.threshold, WINDOW),
        ),
        f"CUSUM\nk=0.5\nh={cusum_design.threshold:.3f}": (
            lambda block: any_run(cusum_chart.breach_mask(block), 1),
            cusum_window_false_alarm(0.5, cusum_design.threshold, WINDOW),
        ),
    }

    print(f"target alpha_W = {ALPHA}, W = {WINDOW} samples, {WINDOWS} windows per cell")
    print()
    print(f"{'method':14s} {'design':>10s} {'measured':>10s} {'SE':>9s} "
          f"{'z':>7s} {'95% Wilson':>24s}")
    iid_block = nominal(0.0, 777)
    measured, lows, highs, designs, labels = [], [], [], [], []
    for label, (alarm_fn, design) in methods.items():
        flags = alarm_fn(iid_block)
        est = estimate_rate(int(flags.sum()), int(flags.size))
        measured.append(est.rate)
        lows.append(est.rate - est.wilson_low)
        highs.append(est.wilson_high - est.rate)
        designs.append(design)
        labels.append(label)
        short = label.split("\n")[0]
        print(f"{short:14s} {design:10.6f} {est.rate:10.6f} {est.standard_error:9.6f} "
              f"{est.z_against(design):7.2f} [{est.wilson_low:.6f}, {est.wilson_high:.6f}]")

    print()
    print("Serially correlated nominal data, same thresholds:")
    print(f"{'rho':>6s} " + " ".join(f"{label.split(chr(10))[0]:>14s}" for label in labels))
    curves = {label: [] for label in labels}
    for rho in RHOS:
        block = nominal(rho, 888)
        row = []
        for label, (alarm_fn, _design) in methods.items():
            flags = alarm_fn(block)
            rate = float(flags.mean())
            curves[label].append(rate)
            row.append(rate)
        print(f"{rho:6.2f} " + " ".join(f"{v:14.5f}" for v in row))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    x = np.arange(len(labels))
    ax.bar(x - 0.19, designs, width=0.36, color="#90a4ae", label="design (analytic)")
    ax.bar(x + 0.19, measured, width=0.36, color="#1565c0",
           yerr=[lows, highs], capsize=5, label="measured (20000 windows, 95 % Wilson)")
    ax.axhline(ALPHA, color="#c62828", linestyle="--", linewidth=1.2,
               label=f"target alpha_W = {ALPHA}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("window false-alarm probability")
    ax.set_ylim(0.0, 0.065)
    ax.set_title(f"Design vs measurement under the design hypothesis\n(iid normal, W = {WINDOW})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for label, values in curves.items():
        ax.plot(RHOS, values, "o-", label=label.replace("\n", " "), linewidth=1.6)
    ax.axhline(ALPHA, color="#c62828", linestyle="--", linewidth=1.2,
               label=f"design alpha_W = {ALPHA}")
    ax.set_xlabel("AR(1) coefficient rho of the nominal channel")
    ax.set_ylabel("measured window false-alarm probability")
    ax.set_yscale("log")
    ax.set_title("Same thresholds on serially correlated telemetry\n"
                 "(the design hypothesis is independence)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, which="both")

    out = Path(__file__).resolve().parent.parent / "screenshots"
    out.mkdir(exist_ok=True)
    path = out / "far_design_vs_measured.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
