"""Recovery of a known misalignment, against the Cramer-Rao bound of equation (12).

Left: the measured error on the observable subspace against the bound, one point
per record, over the swept grid. Middle: the standardised score of equation (24)
against a standard normal. Right: the bound and the measured error as functions
of record length, on a fixed well-excited configuration.
"""

from __future__ import annotations

from _common import BATCH, BOUND, MAP, PROFILE_COLOURS, bootstrap, save, style

bootstrap()
style()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aligncal import batch, benchmark, dataset, observability  # noqa: E402

PROFILES = ("single_axis", "near_single_axis", "two_axis", "three_axis")
NOISES = (5.0e-5, 2.0e-4, 8.0e-4)
LENGTHS = (150, 600, 2400)
SEEDS = 12
PRIOR_SIGMA_RAD = 1.0e-3


def main() -> str:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    figure.subplots_adjust(wspace=0.32)

    left = axes[0]
    scores: list[float] = []
    for profile in PROFILES:
        bounds = []
        errors = []
        for noise in NOISES:
            for length in LENGTHS:
                for index in range(SEEDS):
                    generated = dataset.generate(
                        profile=profile,
                        samples=length,
                        noise_sigma=noise,
                        theta_max_rad=3.0e-3,
                        seed=31420261 + 1000 * index + length,
                    )
                    verdict = observability.assess(generated.record)
                    sensor = verdict["secondary_1"]
                    solution = batch.solve(generated.record, verdict, "secondary_1")
                    error = solution.theta_rad - generated.truth["secondary_1"]
                    values = np.array([d.eigenvalue for d in sensor.degrees_of_freedom])
                    mask = sensor.observable_mask
                    bounds.append(float(np.sqrt(np.sum(1.0 / values[mask]))))
                    errors.append(float(np.linalg.norm(sensor.project_observable(error))))
                    scores.extend(
                        float(z) for z in benchmark.crb_z_scores(error, sensor) if np.isfinite(z)
                    )
        left.scatter(
            np.array(bounds) * 1e3,
            np.array(errors) * 1e3,
            s=6,
            alpha=0.5,
            color=PROFILE_COLOURS[profile],
            label=profile,
        )
    limits = np.array([2.0e-3, 2.0e2])
    left.plot(limits, limits, color=BOUND, linestyle="--", linewidth=1.2, label="error = bound")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("Cramer-Rao one sigma on the retained subspace, mrad")
    left.set_ylabel("measured error on that subspace, mrad")
    left.set_title("the batch solution tracks its own bound\nover four decades")
    left.legend(fontsize=7, loc="upper left")

    middle = axes[1]
    array = np.array(scores)
    middle.hist(
        array, bins=40, density=True, color=BATCH, alpha=0.75,
        label=f"measured, {array.size} scores",
    )
    grid = np.linspace(-4.0, 4.0, 400)
    middle.plot(
        grid,
        np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi),
        color=BOUND,
        linewidth=1.5,
        label="standard normal",
    )
    middle.set_xlabel("standardised error, equation (24)")
    middle.set_ylabel("density")
    middle.set_title(
        f"mean {float(np.mean(array)):+.4f}, std {float(np.std(array)):.4f}\n"
        f"worst |z| {float(np.max(np.abs(array))):.4f}"
    )
    middle.legend(fontsize=8)

    right = axes[2]
    lengths = (150, 300, 600, 1200, 2400, 4800)
    bound_curve = []
    batch_curve = []
    map_curve = []
    for length in lengths:
        bounds = []
        plain = []
        regularised = []
        sensors = []
        for index in range(30):
            generated = dataset.generate(
                profile="three_axis",
                samples=length,
                amplitude=0.05,
                noise_sigma=2.0e-4,
                theta_max_rad=3.0e-3,
                seed=777000 + index,
            )
            verdict = observability.assess(generated.record)
            sensor = verdict["secondary_1"]
            sensors.append(sensor)
            truth = generated.truth["secondary_1"]
            values = np.array([d.eigenvalue for d in sensor.degrees_of_freedom])
            bounds.append(float(np.sqrt(np.sum(1.0 / values[sensor.observable_mask]))))
            plain.append(batch.solve(generated.record, verdict, "secondary_1").theta_rad - truth)
            regularised.append(
                batch.solve_map(
                    generated.record, verdict, "secondary_1", prior_sigma_rad=PRIOR_SIGMA_RAD
                ).theta_rad
                - truth
            )
        bound_curve.append(float(np.median(bounds)) * 1e3)
        batch_curve.append(
            benchmark.score("b", np.array(plain), sensors).rmse_observable_mrad
        )
        map_curve.append(benchmark.score("m", np.array(regularised), sensors).rmse_observable_mrad)
    right.loglog(lengths, bound_curve, color=BOUND, linestyle="--", marker="o", markersize=4,
                 label="Cramer-Rao one sigma, eq (12)")
    right.loglog(lengths, batch_curve, color=BATCH, marker="s", markersize=4,
                 label="batch, eq (14)-(16)")
    right.loglog(lengths, map_curve, color=MAP, marker="^", markersize=4,
                 label="batch MAP, eq (18)")
    right.set_xticks(list(lengths))
    right.set_xticklabels([str(value) for value in lengths], fontsize=8)
    right.minorticks_off()
    right.set_xlabel("record length, samples")
    right.set_ylabel("RMSE on the observable subspace, mrad")
    right.set_title("fixed three_axis excitation, 2e-4 rad/s noise\n30 records per point")
    right.legend(fontsize=8)

    figure.suptitle(
        "Recovery of a known misalignment against the Cramer-Rao bound. "
        "The regularised solution of (18) is biased and is allowed below the bound.",
        fontsize=11,
    )
    figure.subplots_adjust(top=0.84)
    return save(figure, "recovery_vs_crb")


if __name__ == "__main__":
    print(main())
