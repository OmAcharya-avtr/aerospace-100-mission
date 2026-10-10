"""The ECE estimator's bias on a forecast whose population ECE is exactly zero.

Run: ``python examples/ece_bias_curve.py``
Writes ``screenshots/ece_bias_curve.png``.
"""

from __future__ import annotations

from _common import save

from calibaudit import ece_bias_curve, get_spec
from calibaudit.plotting import plot_ece_bias_curve

BINS_GRID = (5, 10, 20, 50, 100)
SAMPLES_GRID = (200, 1000, 5000, 20000)
REPLICATES = 60
SEED = 56


def main() -> None:
    spec = get_spec("calibrated")
    curve = ece_bias_curve(
        spec,
        n_bins_grid=BINS_GRID,
        n_samples_grid=SAMPLES_GRID,
        strategy="equal_width",
        n_replicates=REPLICATES,
        seed=SEED,
    )
    fig = plot_ece_bias_curve(curve, title="ECE of a perfectly calibrated forecaster")
    save(fig, "ece_bias_curve")
    print(curve.table())
    slope, intercept, r2 = curve.power_law_fit()
    print()
    print(
        f"power-law fit  bias = exp({intercept:.6f}) (B/n)^{slope:.6f}, "
        f"R^2 = {r2:.6f}   [theory: exponent 0.5]"
    )
    print(f"population ECE of spec '{spec.name}' is exactly 0; every digit above is bias.")


if __name__ == "__main__":
    main()
