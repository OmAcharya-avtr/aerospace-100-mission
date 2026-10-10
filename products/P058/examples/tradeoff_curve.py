"""The delay-versus-false-alarm trade-off curve for the five analytic detectors.

This is the figure that makes the detectors comparable. A single (ARL0, ARL1)
pair compares nothing; the curve is the detector. A detector is better than
another only where its curve lies lower (shorter delay) at the same ARL0.

Reduced budget compared with validation/validate_tradeoff.py so the example runs
in well under a minute on two cores; the shapes agree and the error bars are
wider. The published numbers come from the validation script, not from here.

Writes ../screenshots/tradeoff_curve.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from telemdrift.benchmark import (
    DETECTOR_LABELS,
    STANDARD,
    BenchmarkConfig,
    tradeoff_curve,
)
from telemdrift.detectors import ANALYTIC_DETECTORS, make_detector
from telemdrift.plotting import plot_tradeoff
from telemdrift.streams import ChangeSpec

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "screenshots" / "tradeoff_curve.png"

SPEC = ChangeSpec("mean_step", 1.0)
CONFIG = BenchmarkConfig(sweep_seeds=(58_301, 58_302), sweep_length=20_000)
REPLICATES = 100

#: Threshold sweeps, each spanning roughly two decades of ARL0. Chosen from the
#: measured default-threshold ARL0 of each detector, not from its target.
SWEEPS = {
    "cusum": [3.0, 4.0, 5.0, 6.0, 7.0],
    "page_hinkley": [10.0, 20.0, 30.0, 45.0, 65.0],
    "ewma": [2.2, 2.6, 2.85, 3.1, 3.4],
    "ks": [0.11, 0.125, 0.14, 0.16, 0.185],
    "adwin": [1e-3, 1e-5, 1e-7, 1e-9, 1e-11],
}


def main() -> None:
    print(f"change: {SPEC.kind} magnitude {SPEC.magnitude:g}")
    print(f"ARL0 budget per point: {len(CONFIG.sweep_seeds)} x {CONFIG.sweep_length} "
          f"samples; ARL1 replicates per point: {REPLICATES}")
    curves = {}
    for key in ANALYTIC_DETECTORS:
        label = DETECTOR_LABELS[key]
        pts = tradeoff_curve(key, SWEEPS[key], SPEC, CONFIG, replicates=REPLICATES)
        curves[label] = [(p.arl0.arl0, p.arl1.arl1, p.arl1.sem) for p in pts]
        name = make_detector(key).threshold_name
        print(f"\n{label}")
        print(f"  {name:>12s}     ARL0   ARL0 SEM     ARL1  ARL1 SEM  censored")
        for p in pts:
            print(f"  {p.threshold:12.6g} {p.arl0.arl0:8.1f} {p.arl0.sem:10.1f} "
                  f"{p.arl1.arl1:8.1f} {p.arl1.sem:9.1f} "
                  f"{100 * p.arl1.censored_rate:8.1f}%")
    path = plot_tradeoff(
        curves, OUT,
        f"Detection delay against false-alarm rate, {SPEC.kind} +{SPEC.magnitude:g} sigma",
        target_arl0=STANDARD.target_arl0,
    )
    print(f"\nwrote {path.relative_to(REPO)} "
          f"({np.round(Path(path).stat().st_size / 1024, 1)} kB)")


if __name__ == "__main__":
    main()
