"""Reliability diagram with bootstrap bands for a known-overconfident forecast.

Run: ``python examples/reliability_diagram.py``
Writes ``screenshots/reliability_diagram.png``.
"""

from __future__ import annotations

import numpy as np
from _common import save

from calibaudit import bootstrap_reliability, calibration_map, get_spec, sample_forecast
from calibaudit.plotting import plot_reliability

SPEC = "overconfident"
N = 4000
BINS = 12
SEED = 56


def main() -> None:
    spec = get_spec(SPEC)
    s = sample_forecast(spec, N, seed=SEED)
    curve = bootstrap_reliability(
        s.forecasts, s.outcomes, n_bins=BINS, n_bootstrap=1000, level=0.9, seed=SEED + 1
    )
    grid = np.linspace(0.005, 0.995, 400)
    fig = plot_reliability(
        curve,
        title=f"{spec.name}: sharpened logits (T = {spec.param})",
        truth_x=grid,
        truth_y=calibration_map(spec, grid),
    )
    save(fig, "reliability_diagram")
    print(curve.table())
    excluded = int(np.sum(curve.diagonal_excluded()))
    print(f"bins whose 90 % band excludes the mean forecast: {excluded} of {curve.n_occupied}")


if __name__ == "__main__":
    main()
