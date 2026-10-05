"""Example: detection delay and detection probability vs shift size, at matched FAR.

Every method is calibrated to the same window false-alarm probability on
independent nominal data, then run against step anomalies of increasing size.
Two panels, because either alone is misleading: a method can have a short mean
delay simply because it only detects the easy cases, so the delay panel is only
readable next to the detection-probability panel.

Writes ``../screenshots/detection_delay_curves.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from telemetryool.arl import cusum_arl_markov, design_cusum_h  # noqa: E402
from telemetryool.detectors import build_detector_suite  # noqa: E402
from telemetryool.harness import ScenarioSpec, run_comparison  # noqa: E402
from telemetryool.synthetic import Anomaly, NominalModel, equicorrelation  # noqa: E402

ALPHA = 0.05
WINDOW = 100
CHANNELS = 4
ONSET = 40
SHIFTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
COLOURS = ["#616161", "#f9a825", "#c62828", "#2e7d32", "#1565c0", "#6a1b9a"]


def main() -> int:
    model = NominalModel(CHANNELS, correlation=equicorrelation(CHANNELS, 0.6))
    scenarios = [
        ScenarioSpec(f"step_{shift}", Anomaly("step", ONSET, shift), 1200)
        for shift in SHIFTS
    ]
    result = run_comparison(
        build_detector_suite(persistence=1),
        model,
        scenarios,
        target_alpha_w=ALPHA,
        window_length=WINDOW,
        n_train_windows=1000,
        n_cal_windows=4000,
        n_measure_windows=6000,
        seed=424242,
    )
    print(f"target alpha_W = {ALPHA}, W = {WINDOW}, onset at sample {ONSET}, "
          f"{CHANNELS} channels, equicorrelation 0.6")
    print(f"1200 windows per shift size; binomial SE on Pd near 0.5 is "
          f"{0.5 / np.sqrt(1200):.4f}")
    print(f"harness wall time {result.elapsed_s:.1f} s")
    print()
    print("delivered window false-alarm probability per method (independent 6000 windows):")
    for method in result.methods:
        print(f"  {method.name:22s} {method.false_alarm.rate:.5f} "
              f"+/- {method.false_alarm.standard_error:.5f}")
    print()
    header = f"{'shift':>7s} " + " ".join(f"{m.name:>22s}" for m in result.methods)
    print("Detection probability:")
    print(header)
    for shift, sc in zip(SHIFTS, scenarios, strict=True):
        print(
            f"{shift:7.2f} "
            + " ".join(
                f"{m.per_scenario[sc.name].delay.detection_probability:22.4f}"
                for m in result.methods
            )
        )
    print()
    print("Mean detection delay in samples, conditional on detection:")
    print(header)
    for shift, sc in zip(SHIFTS, scenarios, strict=True):
        print(
            f"{shift:7.2f} "
            + " ".join(
                f"{m.per_scenario[sc.name].delay.mean_delay:22.2f}"
                for m in result.methods
            )
        )

    design = design_cusum_h(ALPHA, 0.5, WINDOW)
    analytic = [cusum_arl_markov(s, 0.5, design.threshold, 800) for s in SHIFTS]
    print()
    print("Analytic single-channel CUSUM ARL1 (Brook-Evans), for reference:")
    print(f"{'shift':>7s} {'ARL1':>10s}")
    for shift, value in zip(SHIFTS, analytic, strict=True):
        print(f"{shift:7.2f} {value:10.3f}")
    print("The measured curve sits below this because the chart is warm at the onset and")
    print("because the harness monitors four channels through one alarm bus, so its")
    print("calibrated threshold differs from the single-channel design value.")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    ax = axes[0]
    for colour, method in zip(COLOURS, result.methods, strict=True):
        pd_values = [
            method.per_scenario[sc.name].delay.detection_probability for sc in scenarios
        ]
        errs = [
            method.per_scenario[sc.name].delay.detection_probability_se for sc in scenarios
        ]
        ax.errorbar(SHIFTS, pd_values, yerr=errs, fmt="o-", color=colour,
                    linewidth=1.5, markersize=4, capsize=3, label=method.name)
    ax.axhline(ALPHA, color="#c62828", linestyle="--", linewidth=1.0,
               label=f"matched alpha_W = {ALPHA}")
    ax.set_xlabel("step size [sigma]")
    ax.set_ylabel("detection probability within the window")
    ax.set_title(f"Detection probability at matched false-alarm rate\n"
                 f"(onset at sample {ONSET} of {WINDOW})")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    for colour, method in zip(COLOURS, result.methods, strict=True):
        delays, errs, shown = [], [], []
        for shift, sc in zip(SHIFTS, scenarios, strict=True):
            stats = method.per_scenario[sc.name].delay
            # A mean delay from a handful of detections is noise; suppress it
            # rather than drawing a line through it.
            if stats.detection_probability < 0.2 or stats.n_detected < 30:
                continue
            shown.append(shift)
            delays.append(stats.mean_delay)
            errs.append(
                float(stats.delays.std(ddof=1) / np.sqrt(stats.n_detected))
                if stats.n_detected > 1
                else 0.0
            )
        if shown:
            ax.errorbar(shown, delays, yerr=errs, fmt="o-", color=colour,
                        linewidth=1.5, markersize=4, capsize=3, label=method.name)
    ax.plot(SHIFTS, analytic, "k:", linewidth=1.4,
            label="CUSUM ARL1, single channel (Brook-Evans)")
    ax.set_xlabel("step size [sigma]")
    ax.set_ylabel("mean detection delay [samples]")
    ax.set_yscale("log")
    ax.set_title("Mean delay conditional on detection\n"
                 "(points with detection probability below 0.2 omitted)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, which="both")

    out = Path(__file__).resolve().parent.parent / "screenshots"
    out.mkdir(exist_ok=True)
    path = out / "detection_delay_curves.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
