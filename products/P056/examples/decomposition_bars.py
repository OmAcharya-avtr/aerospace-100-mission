"""Brier decomposition of six forecasters, and what a three-term report drops.

Run: ``python examples/decomposition_bars.py``
Writes ``screenshots/decomposition_bars.png``.
"""

from __future__ import annotations

from _common import save

from calibaudit import SPEC_NAMES, binned_decomposition, get_spec, sample_forecast
from calibaudit.plotting import plot_decomposition_bars

N = 8000
BINS = 10
SEED = 56


def main() -> None:
    decompositions = {}
    for name in SPEC_NAMES:
        s = sample_forecast(get_spec(name), N, seed=SEED)
        decompositions[name] = binned_decomposition(
            s.forecasts, s.outcomes, n_bins=BINS, strategy="equal_width"
        )
    fig = plot_decomposition_bars(
        decompositions, title=f"n = {N}, {BINS} equal-width bins"
    )
    save(fig, "decomposition_bars")
    head = (
        f"{'spec':>19} {'BS':>10} {'REL':>10} {'RES':>10} {'UNC':>10} "
        f"{'WBV':>10} {'WBC':>11} {'3-term resid':>13} {'identity':>11}"
    )
    print(head)
    print("-" * len(head))
    for name, d in decompositions.items():
        print(
            f"{name:>19} {d.brier:>10.6f} {d.reliability:>10.6f} {d.resolution:>10.6f} "
            f"{d.uncertainty:>10.6f} {d.within_bin_variance:>10.6f} "
            f"{d.within_bin_covariance:>11.6f} {d.three_term_residual:>+13.6f} "
            f"{d.identity_residual:>+11.2e}"
        )


if __name__ == "__main__":
    main()
