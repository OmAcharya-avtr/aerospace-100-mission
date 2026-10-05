"""Example: window-level ROC curves for every method, with the matched operating point.

One panel per anomaly scenario.  Every curve is the window-level ROC against the
same nominal measurement set, so the horizontal axis is the quantity the methods
were matched on.  The marker on each curve is the calibrated operating point:
this is where the detection-delay table is read, and it is at the same
false-alarm rate for every method by construction.

Writes ``../screenshots/roc_matched_far.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from telemetryool.detectors import build_detector_suite  # noqa: E402
from telemetryool.harness import ScenarioSpec, run_comparison  # noqa: E402
from telemetryool.synthetic import Anomaly, NominalModel, equicorrelation  # noqa: E402

ALPHA = 0.05
WINDOW = 100
CHANNELS = 4
ONSET = 40
COLOURS = ["#616161", "#f9a825", "#c62828", "#2e7d32", "#1565c0", "#6a1b9a"]


def main() -> int:
    model = NominalModel(CHANNELS, correlation=equicorrelation(CHANNELS, 0.6))
    scenarios = [
        ScenarioSpec("step 1.0 sigma, channel 0", Anomaly("step", ONSET, 1.0), 1500),
        ScenarioSpec("stuck channel 0", Anomaly("stuck", ONSET), 1500),
        ScenarioSpec(
            "decorrelation, channels 0-1",
            Anomaly("decorrelate", ONSET, channels=[0, 1]),
            1500,
        ),
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
        seed=20261005,
    )
    print(f"target alpha_W = {ALPHA}, W = {WINDOW}, {CHANNELS} channels, "
          "equicorrelation 0.6")
    print("calibration 4000 windows, nominal measurement 6000 windows, "
          "1500 windows per scenario")
    print(f"harness wall time {result.elapsed_s:.1f} s")
    print()
    print(result.far_table())
    print()
    print(f"{'scenario':30s} " + " ".join(f"{m.name:>20s}" for m in result.methods))
    for sc in scenarios:
        print(
            f"{sc.name:30s} "
            + " ".join(
                f"{m.per_scenario[sc.name].roc.auc:20.4f}" for m in result.methods
            )
        )
    print("(window-level ROC area under the curve)")

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5.0 * len(scenarios), 5.0))
    for ax, sc in zip(axes, scenarios, strict=True):
        for colour, method in zip(COLOURS, result.methods, strict=True):
            res = method.per_scenario[sc.name]
            ax.plot(res.roc.fpr, res.roc.tpr, color=colour, linewidth=1.5,
                    label=f"{method.name}  AUC={res.roc.auc:.3f}")
            ax.plot(
                [method.false_alarm.rate],
                [res.confusion.tpr],
                "o",
                color=colour,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
        ax.plot([0, 1], [0, 1], ":", color="#999999", linewidth=1.0, label="chance")
        ax.axvline(ALPHA, color="#c62828", linestyle="--", linewidth=1.0,
                   label=f"matched alpha_W = {ALPHA}")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("window false-alarm probability")
        ax.set_ylabel("detection probability")
        ax.set_title(sc.name)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Window-level ROC, every method against the same nominal set; "
        "markers are the matched operating point",
        fontsize=12,
    )
    out = Path(__file__).resolve().parent.parent / "screenshots"
    out.mkdir(exist_ok=True)
    path = out / "roc_matched_far.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
