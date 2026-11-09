"""The Cramer-Rao bound against the measured scatter of repeated fits.

What to notice: each point is one identifiable direction. The horizontal axis is
the bound, computed from the Fisher information of equation (9); the vertical
axis is the standard deviation measured over independent noise realisations of
the same truth, using an estimator that never sees the bound. Everything must
lie on or above the diagonal. The linear estimator of equation (12) sits on it,
which says the bound is tight as well as true; the nonlinear global fit sits
above it, because on a problem this ill-conditioned the global optimum wanders
along a curved valley and is biased, and the plain Cramer-Rao inequality bounds
unbiased estimators only.

The 200 s campaigns are used. On the 20 s campaigns the nonlinear fit falls
BELOW the bound in its worst direction -- measured ratios 0.118 and 0.351 --
because the bound there is 21 relative units while the plant's declared validity
window will not let any parameter move that far: the estimator is constrained,
so it is not the unbiased estimator the inequality is about. That is recorded
here rather than drawn, because a figure of it would read as a violation.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import ACCENT, ANALYTIC, LEARNED, REFUSED, bootstrap, save, style  # noqa: E402

ROOT = bootstrap()

from twindiverge.identifiability import analyse  # noqa: E402
from twindiverge.pipeline import montecarlo_directions  # noqa: E402
from twindiverge.plant import NOMINAL, simulate_with_sensitivities  # noqa: E402
from twindiverge.telemetry import (  # noqa: E402
    TELEMETRY_SETS,
    design_matrix,
    measure,
    residual_vector,
)

#: Realisations for the linear estimator, which is a projection and costs
#: nothing, and for the nonlinear fit, which costs one optimisation each.
LINEAR_SEEDS = tuple(range(70000, 70048))
NONLINEAR_SEEDS = tuple(range(70000, 70008))
SETS = ("rate_full", "rate_torque_full")
NONLINEAR_SETS = ("rate_full",)
MARKERS = {"rate_full": "s", "rate_torque_full": "o"}


def linear_amplitudes(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Equation (13) per realisation, and the bound, for one telemetry set."""
    telemetry = TELEMETRY_SETS[name]
    report = analyse(telemetry, NOMINAL)
    twin = simulate_with_sensitivities(NOMINAL, telemetry.campaign)
    matrix, _ = design_matrix(telemetry, NOMINAL, trajectory=twin)
    left, values, _ = np.linalg.svd(matrix, full_matrices=False)
    rows = []
    for seed in LINEAR_SEEDS:
        observed, _ = measure(telemetry, NOMINAL, seed=seed)
        residual = residual_vector(telemetry, observed, twin)
        rows.append((left[:, : report.rank].T @ residual) / values[: report.rank])
    return np.vstack(rows), 1.0 / values[: report.rank]


def main() -> int:
    style()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    printed = []
    for name in SETS:
        telemetry = TELEMETRY_SETS[name]
        linear, bound = linear_amplitudes(name)
        series = [(f"linear, eq (12), n={len(LINEAR_SEEDS)}", linear, ACCENT)]
        if name in NONLINEAR_SETS:
            nonlinear, _ = montecarlo_directions(
                telemetry, NOMINAL, NONLINEAR_SEEDS
            )
            series.append(
                (f"nonlinear fit, n={len(NONLINEAR_SEEDS)}", nonlinear, LEARNED)
            )
        for label, estimates, colour in series:
            spread = estimates.std(axis=0, ddof=1)
            axes[0].loglog(
                bound, spread, MARKERS[name], color=colour, markersize=7,
                markerfacecolor="none",
                label=f"{name}, {label}",
            )
            for index in range(bound.size):
                printed.append(
                    (name, label, index + 1, float(bound[index]),
                     float(spread[index]), float(spread[index] / bound[index]))
                )
            axes[1].plot(
                np.arange(1, bound.size + 1), spread / bound, MARKERS[name],
                color=colour, markersize=7, markerfacecolor="none",
                label=f"{name}, {label}",
            )
    limits = (1e-6, 1e2)
    axes[0].plot(limits, limits, color=REFUSED, linestyle="--", linewidth=1.0,
                 label="the bound (equality)")
    axes[0].set_xlim(*limits)
    axes[0].set_ylim(*limits)
    axes[0].set_xlabel("Cramer-Rao bound, relative standard error")
    axes[0].set_ylabel("measured standard deviation")
    axes[0].set_title("every point must lie on or above the dashed line")
    axes[0].legend(fontsize=6, loc="upper left")
    axes[1].axhline(1.0, color=REFUSED, linestyle="--", linewidth=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("identifiable direction, best determined first")
    axes[1].set_ylabel("measured standard deviation divided by the bound")
    axes[1].set_title("how much room the bound had")
    axes[1].legend(fontsize=6, loc="upper right")
    figure.suptitle(
        "twindiverge: the bound against independent noise realisations",
        fontsize=11,
    )
    _ = ANALYTIC
    path = save(figure, "crb_vs_empirical.png")
    print(f"written: {path}")
    for name, label, index, bound_value, spread, ratio in printed:
        print(
            f"  {name:<18s} {label:<16s} direction {index}: bound {bound_value:.6e} "
            f"measured {spread:.6e} ratio {ratio:.6f}"
        )
    worst = min(row[5] for row in printed)
    print(f"  least ratio over all points : {worst:.6f} "
          f"(a ratio below 1 is allowed only within sampling error)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
