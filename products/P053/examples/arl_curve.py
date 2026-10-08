"""The headline figure: detection delay against false-alarm rate, per change type.

Saves ``../screenshots/arl_curve.png``. What to notice: every curve rises to
the right, so there is no threshold that buys both a longer time between false
alarms and a shorter delay. The CUSUM curve lies below the other two
everywhere. The noise-variance panel sits three to four times higher than the
parameter-step panel at the same false-alarm rate, and the variance-CUSUM
oracle -- which is handed the post-change residual variance -- is the highest
curve in that panel, not the lowest.
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
    DetectorSpec,
    changed_streams,
    delay_curve,
    in_control_streams,
    threshold_grid,
)

N_RUNS = 250
N_SAMPLES = 3000
NAMES = ("cusum", "ewma", "glr", "varcusum")
STYLES = {
    "cusum": dict(color="C0", marker="o"),
    "ewma": dict(color="C1", marker="s"),
    "glr": dict(color="C2", marker="^"),
    "varcusum": dict(color="C3", marker="x", ls="--"),
}


def main() -> int:
    bank = in_control_streams(n_runs=N_RUNS, n_samples=N_SAMPLES, seed=53001)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=True)
    summary: list[str] = []
    for ax, scenario in zip(axes, SCENARIOS, strict=True):
        oc = changed_streams(scenario, n_runs=N_RUNS, n_samples=N_SAMPLES)
        for name in NAMES:
            spec = DetectorSpec(name)
            grid = threshold_grid(spec.statistic(bank), n_points=12)
            curve = delay_curve(spec, bank, oc, grid)
            arl0 = np.array([p.arl0.value for p in curve])
            arl1 = np.array([p.arl1.value for p in curve])
            det = np.array([p.arl1.detection_fraction for p in curve])
            keep = np.isfinite(arl0) & np.isfinite(arl1) & (det > 0.95) & (arl0 > 2.0)
            ax.plot(arl0[keep], arl1[keep], lw=1.4, ms=4, label=spec.label(), **STYLES[name])
            if keep.any():
                summary.append(
                    f"  {scenario:<16}{spec.label():<32}"
                    f"ARL0 {arl0[keep][-1]:>8.0f} -> delay {arl1[keep][-1]:>7.1f}"
                )
        ax.axvline(1000.0, color="0.4", ls=":", lw=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("in-control ARL0 (samples)\nlonger between false alarms ->")
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("detection delay, zero-state ARL1 (samples)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    for ax in axes:
        ax.text(
            1000.0,
            ax.get_ylim()[0] * 1.15,
            " declared target\n ARL0 = 1000",
            fontsize=7,
            color="0.3",
            ha="left",
            va="bottom",
        )
    fig.suptitle(
        "Detection delay against false-alarm rate, per change type\n"
        f"twininvalidate, {N_RUNS} runs x {N_SAMPLES} samples per point; thresholds set "
        "on in-control data only; points with censored runs removed",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT / "screenshots" / "arl_curve.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    print("rightmost uncensored point of each curve:")
    for line in summary:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
