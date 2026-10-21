"""Monte Carlo convergence against the analytic value, with exact intervals.

What to notice: the Clopper-Pearson interval shrinks as the inverse square root
of the sample count and brackets the Foster-Estes value at every sample size
shown. The sample count needed for a useful interval scales as the inverse of
Pc, which is why Monte Carlo is a reference for this product and not a method.
"""

from __future__ import annotations

import warnings

import numpy as np

from _common import FOSTER, MONTE_CARLO, REFERENCE, bootstrap, save, style

bootstrap()

from conjprob.cases import load_case  # noqa: E402
from conjprob.dilution import DilutionWarning  # noqa: E402
from conjprob.montecarlo import pc_monte_carlo  # noqa: E402
from conjprob.report import assess_conjunction  # noqa: E402

SAMPLE_COUNTS = (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000)
SEED = 20261021


def main() -> None:
    style()
    import matplotlib.pyplot as plt

    case = load_case("examples/cases/leo_crossing.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DilutionWarning)
        report = assess_conjunction(case.primary, case.secondary, case.radius_m)
    reference = report.analytic["foster"]
    geometry = report.geometry

    estimates = []
    for count in SAMPLE_COUNTS:
        estimates.append(
            pc_monte_carlo(
                geometry.combined_covariance_m2,
                geometry.miss_vector_m,
                geometry.relative_velocity_mps,
                case.radius_m,
                n_samples=count,
                seed=SEED,
            )
        )

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    counts = np.array(SAMPLE_COUNTS, dtype=float)
    centres = np.array([estimate.pc for estimate in estimates])
    lows = np.array([estimate.ci_low for estimate in estimates])
    highs = np.array([estimate.ci_high for estimate in estimates])
    axes[0].errorbar(
        counts,
        centres,
        yerr=np.vstack((centres - lows, highs - centres)),
        fmt="o",
        color=MONTE_CARLO,
        markersize=4,
        capsize=3,
        label="Monte Carlo, 95 per cent Clopper-Pearson",
    )
    axes[0].axhline(reference, color=FOSTER, linewidth=1.4, label="Foster-Estes")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("samples")
    axes[0].set_ylabel("collision probability")
    axes[0].set_title(f"{case.name}: Pc = {reference:.6e}")
    axes[0].legend(loc="upper right")

    widths = highs - lows
    axes[1].loglog(counts, widths, marker="o", color=MONTE_CARLO, label="interval width")
    axes[1].loglog(
        counts,
        widths[0] * np.sqrt(counts[0] / counts),
        color=REFERENCE,
        linestyle="--",
        linewidth=1.0,
        label="n to the minus one half",
    )
    axes[1].set_xlabel("samples")
    axes[1].set_ylabel("95 per cent interval width")
    slope = np.polyfit(np.log(counts), np.log(widths), 1)[0]
    axes[1].set_title(f"measured log-log slope {slope:.4f}")
    axes[1].legend(loc="lower left")
    print(f"wrote {save(figure, 'monte_carlo_convergence.png')}")
    for count, estimate in zip(SAMPLE_COUNTS, estimates, strict=True):
        inside = estimate.ci_low <= reference <= estimate.ci_high
        print(
            f"n = {count:>9d}  hits {estimate.hits:>7d}  pc {estimate.pc:.6e}  "
            f"CI [{estimate.ci_low:.6e}, {estimate.ci_high:.6e}]  "
            f"contains Foster: {inside}"
        )
    print(f"measured interval-width slope against sample count: {slope:.6f}")


if __name__ == "__main__":
    main()
