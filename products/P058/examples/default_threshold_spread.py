"""Measure ARL0 at each detector's own default threshold and plot the spread.

This is the figure behind the claim that comparing detectors at default
thresholds compares operating points rather than detectors.

Writes ../screenshots/default_threshold_spread.png.
"""

from __future__ import annotations

from pathlib import Path

from telemdrift.benchmark import (
    DETECTOR_LABELS,
    STANDARD,
    default_threshold_operating_points,
)
from telemdrift.detectors import ANALYTIC_DETECTORS, make_detector
from telemdrift.plotting import plot_operating_point_spread

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "screenshots" / "default_threshold_spread.png"


def main() -> None:
    res = default_threshold_operating_points(STANDARD)
    labels, arl0s, sems, texts = [], [], [], []
    print(f"{len(STANDARD.eval_seeds)} seeds x {STANDARD.eval_length} stationary samples "
          f"= {len(STANDARD.eval_seeds) * STANDARD.eval_length} per detector")
    print()
    print("detector        default           ARL0        SEM   rel.SEM   runs")
    for key in ANALYTIC_DETECTORS:
        r = res[key]
        det = make_detector(key)
        labels.append(DETECTOR_LABELS[key])
        arl0s.append(r.arl0)
        sems.append(0.0 if r.sem != r.sem else r.sem)
        texts.append(f"{det.threshold_name}={det.threshold:.3g}")
        print(f"{DETECTOR_LABELS[key]:15s} {det.threshold_name}={det.threshold:<13.4g} "
              f"{r.arl0:9.1f} {r.sem:10.1f} {100 * r.relative_sem:7.2f}% {r.n_runs:6d}")
    print()
    print(f"spread (widest / narrowest default) = {max(arl0s) / min(arl0s):.1f}x")
    path = plot_operating_point_spread(labels, arl0s, sems, texts, OUT,
                                       STANDARD.target_arl0)
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
