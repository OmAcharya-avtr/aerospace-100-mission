"""The closed form against the simulation, in and out of its own assumptions.

Produces ``screenshots/analytic_vs_simulation.png``. Left: the no-outage limit,
where the closed form is exact and the simulated mean must sit on the identity
line to within sampling error, with the error bars drawn. Right: the same
comparison once outages and contact windows are declared, where the closed form
is an extension and the scatter off the line is the measured size of its error.
"""

from __future__ import annotations

from _common import ANALYTIC, MONTE_CARLO, bootstrap, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from relfiledelivery import (  # noqa: E402
    TransferMode,
    analytic_delivery,
    extended_expected_delivery_time,
    generate_scenario,
    run_ensemble,
    standard_error_of_mean,
)

RUNS = 240


def main() -> str:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))

    for axis, regime, title in (
        (axes[0], "no_outage", "no outages: the closed form is exact"),
        (axes[1], "correlated_outage", "outages declared: the closed form is an extension"),
    ):
        predicted = []
        measured = []
        errors = []
        for seed in range(60):
            scenario = generate_scenario(seed, regime)
            mode = TransferMode.DEFERRED_NAK
            distribution = run_ensemble(scenario.spec, scenario.link, mode, RUNS, 5150)
            if distribution.completed < 2:
                continue
            if regime == "no_outage":
                value = analytic_delivery(
                    scenario.spec, scenario.link, mode
                ).expected_delivery_time_s
            else:
                value = extended_expected_delivery_time(scenario.spec, scenario.link, mode)
            predicted.append(value)
            measured.append(distribution.mean_s)
            errors.append(standard_error_of_mean(distribution.delivery_times_s))

        predicted_array = np.array(predicted)
        measured_array = np.array(measured)
        axis.errorbar(
            predicted_array,
            measured_array,
            yerr=np.array(errors),
            fmt="o",
            markersize=3.5,
            color=MONTE_CARLO,
            elinewidth=0.9,
            capsize=2,
            label=f"{len(predicted)} scenarios, {RUNS} runs each",
        )
        lo = min(predicted_array.min(), measured_array.min()) * 0.8
        hi = max(predicted_array.max(), measured_array.max()) * 1.25
        axis.plot([lo, hi], [lo, hi], color=ANALYTIC, linewidth=1.3, label="identity")
        ratio = measured_array / predicted_array
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(lo, hi)
        axis.set_ylim(lo, hi)
        axis.set_xlabel("closed-form prediction, s")
        axis.set_ylabel("simulated mean delivery time, s")
        axis.set_title(
            f"{title}\nmeasured/predicted: median {np.median(ratio):.3f}, "
            f"q0.05 {np.quantile(ratio, 0.05):.3f}, q0.95 {np.quantile(ratio, 0.95):.3f}",
            fontsize=9,
        )
        axis.legend(fontsize=7, loc="upper left")

    figure.suptitle(
        "Baseline 1 against baseline 2: exact where it applies, measured where it does not",
        fontsize=10,
    )
    figure.subplots_adjust(wspace=0.3, top=0.84)
    return save(figure, "analytic_vs_simulation")


if __name__ == "__main__":
    print(main())
