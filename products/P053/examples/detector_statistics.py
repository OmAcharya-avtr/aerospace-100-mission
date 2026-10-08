"""Statistic paths of the three baselines against their calibrated thresholds.

Saves ``../screenshots/detector_statistics.png``. What to notice: on the
parameter step the CUSUM climbs linearly once the change arrives, because it
accumulates without bound; the windowed GLR plateaus, because its window
truncates the accumulation. On the noise-variance change none of the three has
any drift at all and they cross only on tail excursions, which is the whole
reason that change takes three to four times as long to detect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
from _bootstrap import add_src_to_path  # noqa: E402

ROOT = add_src_to_path()

from twininvalidate import (  # noqa: E402
    SCENARIO_LABELS,
    SCENARIOS,
    AssetChange,
    DetectorSpec,
    StreamSpec,
    calibrate_threshold,
    in_control_streams,
    simulate_residuals,
)

ONSET = 400
N_SAMPLES = 1600
DT = 0.05
TARGET = 1000.0
NAMES = ("cusum", "ewma", "glr")
SHOWN = ("parameter_step", "noise_variance")


def main() -> int:
    bank = in_control_streams(n_runs=200, n_samples=2000, seed=53001)
    thresholds = {
        n: calibrate_threshold(DetectorSpec(n), bank, TARGET).threshold for n in NAMES
    }
    fig, axes = plt.subplots(len(NAMES), len(SHOWN), figsize=(11, 8.5), sharex=True)
    t = np.arange(N_SAMPLES) * DT
    for col, scenario in enumerate(SHOWN):
        change = SCENARIOS[scenario]
        shifted = AssetChange(change.kind, ONSET, change.magnitude, change.ramp_samples)
        z = simulate_residuals(
            StreamSpec(change=shifted, n_runs=40, n_samples=N_SAMPLES, seed=53501)
        )
        for row, name in enumerate(NAMES):
            spec = DetectorSpec(name)
            stat = spec.statistic(z)
            ax = axes[row, col]
            for i in range(6):
                ax.plot(t, stat[i], lw=0.7, color="0.6")
            ax.plot(t, stat.mean(axis=0), lw=1.8, color="C0", label="mean of 40 runs")
            ax.axhline(
                thresholds[name],
                color="C3",
                ls="-",
                lw=1.3,
                label=f"threshold {thresholds[name]:.2f} (ARL0 = {TARGET:.0f})",
            )
            ax.axvline(ONSET * DT, color="C2", ls="--", lw=1.1, label="change onset")
            ax.set_ylim(0, max(3.0 * thresholds[name], float(stat.mean(axis=0).max()) * 1.4))
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel(f"{spec.label()}\nstatistic")
            if row == 0:
                ax.set_title(SCENARIO_LABELS[scenario], fontsize=10)
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time (s), 20 Hz sampling")
    fig.suptitle(
        "Detector statistics against thresholds calibrated to a declared ARL0 of "
        f"{TARGET:.0f} samples\n"
        "twininvalidate, seed 53501; thresholds set on in-control data only",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / "screenshots" / "detector_statistics.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    for name in NAMES:
        print(f"  {DetectorSpec(name).label():<24}threshold {thresholds[name]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
