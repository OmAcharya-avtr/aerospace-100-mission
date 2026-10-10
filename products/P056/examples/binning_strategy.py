"""Equal-width against equal-mass binning, on a rare-event forecast.

Run: ``python examples/binning_strategy.py``
Writes ``screenshots/binning_strategy.png``.
"""

from __future__ import annotations

from _common import save

from calibaudit import ece_bias_curve, get_spec
from calibaudit.plotting import plot_ece_bias_curve

BINS_GRID = (5, 10, 20, 50)
SAMPLES_GRID = (200, 1000, 5000)
REPLICATES = 60
SEED = 56


def main() -> None:
    spec = get_spec("calibrated_rare")
    curves = {}
    for strategy in ("equal_width", "equal_mass"):
        curves[strategy] = ece_bias_curve(
            spec,
            n_bins_grid=BINS_GRID,
            n_samples_grid=SAMPLES_GRID,
            strategy=strategy,
            n_replicates=REPLICATES,
            seed=SEED,
        )
    fig = plot_ece_bias_curve(
        curves["equal_mass"],
        title="ECE bias on a base-rate-0.1 calibrated forecaster (equal-mass bins)",
    )
    save(fig, "binning_strategy")
    for strategy, curve in curves.items():
        print(f"=== {strategy} ===")
        print(curve.table())
        slope, intercept, r2 = curve.power_law_fit()
        print(
            f"power-law fit  bias = exp({intercept:.6f}) (B/n)^{slope:.6f}, R^2 = {r2:.6f}"
        )
        print()
    width = curves["equal_width"]
    mass = curves["equal_mass"]
    print(f"{'bins':>5} {'n':>7} {'bias_equal_width':>17} {'bias_equal_mass':>16} {'ratio':>8}")
    for a, b in zip(width.rows, mass.rows, strict=True):
        print(
            f"{a.n_bins:>5d} {a.n_samples:>7d} {a.bias:>17.6f} {b.bias:>16.6f} "
            f"{a.bias / b.bias:>8.3f}"
        )


if __name__ == "__main__":
    main()
