"""The learned surrogate against the analytic importance-sampling baseline.

Writes ../screenshots/surrogate_benchmark.png.

Left: the rough limit state in standard normal space, with the analytic
smooth-part design point, the true design point and the one the surrogate
found. Right: the measured variance of each estimator at an equal budget of
true limit-state evaluations, on a smooth instance where the surrogate cannot
help and a rough one where it can.

Run from the product directory:

    python examples/surrogate_benchmark.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

from rareverify.benchmark import replicate, summarise  # noqa: E402
from rareverify.limitstates import (  # noqa: E402
    LinearGaussianLimitState,
    RippledLimitState,
)
from rareverify.montecarlo import crude_monte_carlo  # noqa: E402
from rareverify.surrogate import (  # noqa: E402
    fit_surrogate,
    surrogate_design_point,
    surrogate_guided_importance_sampling,
)
from rareverify.tilting import (  # noqa: E402
    analytic_mean_shift,
    importance_sampling,
    oracle_mean_shift,
)

warnings.simplefilter("ignore", ConvergenceWarning)

OUTPUT = (
    Path(__file__).resolve().parent.parent / "screenshots" / "surrogate_benchmark.png"
)
BUDGET = 60_000
REPLICATIONS_CHEAP = 30
REPLICATIONS_SURROGATE = 10
N_TRAIN = 100


def main() -> None:
    smooth = LinearGaussianLimitState(beta=3.719, dimension=2)
    rough = RippledLimitState(beta=5.5, amplitude=2.5, frequency=2.0)

    rng = np.random.default_rng(606)
    fit = fit_surrogate(rough, n_train=N_TRAIN, rng=rng)
    design = surrogate_design_point(fit, rng=rng)
    oracle_tilt = oracle_mean_shift(rough, rng=np.random.default_rng(3))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    grid = np.linspace(-6.5, 6.5, 320)
    mesh_x, mesh_y = np.meshgrid(grid, grid)
    points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    true_g = rough.g(points).reshape(mesh_x.shape)
    surrogate_g = np.asarray(fit.predict(points)).reshape(mesh_x.shape)

    axes[0].contourf(
        mesh_x, mesh_y, (true_g <= 0).astype(float), levels=[0.5, 1.5],
        colors=["tab:red"], alpha=0.18,
    )
    axes[0].contour(mesh_x, mesh_y, true_g, levels=[0.0], colors="tab:red", linewidths=1.8)
    axes[0].contour(
        mesh_x, mesh_y, surrogate_g, levels=[0.0], colors="tab:blue",
        linewidths=1.2, linestyles="--",
    )
    axes[0].scatter(
        fit.x_train[:, 0], fit.x_train[:, 1], s=6, color="grey", alpha=0.6,
        label=f"{N_TRAIN} training evaluations",
    )
    analytic_point = rough.design_point()
    axes[0].plot(
        *analytic_point, marker="X", markersize=11, color="black", linestyle="none",
        label=f"analytic smooth design point, |x| = {np.linalg.norm(analytic_point):.3f}",
    )
    axes[0].plot(
        *oracle_tilt.theta, marker="*", markersize=15, color="tab:green", linestyle="none",
        label=f"true design point, |x| = {oracle_tilt.norm:.3f}",
    )
    axes[0].plot(
        *design.point, marker="o", markersize=8, color="tab:blue", linestyle="none",
        label=f"surrogate design point, |x| = {design.beta:.3f}",
    )
    circle = plt.Circle((0, 0), 1.0, fill=False, color="grey", linewidth=0.6)
    axes[0].add_patch(circle)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x0 (standard normal)")
    axes[0].set_ylabel("x1 (standard normal)")
    axes[0].set_title(
        "Rough limit state: red is the failure region,\n"
        "blue dashed is what the surrogate learned"
    )
    axes[0].legend(loc="lower left", fontsize=7.5)
    axes[0].grid(alpha=0.25)

    labels = []
    groups = {"crude": [], "analytic-IS": [], "surrogate-IS": [], "oracle-IS": []}
    for name, state in (("smooth", smooth), ("rough", rough)):
        reference = state.analytic_probability()
        labels.append(f"{name}\np = {reference:.3e}")
        groups["crude"].append(
            summarise(
                "crude",
                replicate(
                    lambda rng_, s=state: crude_monte_carlo(s, BUDGET, rng=rng_),
                    REPLICATIONS_CHEAP,
                    seed=701,
                ),
                reference,
            ).empirical_cov
        )
        groups["analytic-IS"].append(
            summarise(
                "analytic-IS",
                replicate(
                    lambda rng_, s=state: importance_sampling(
                        s, analytic_mean_shift(s), BUDGET, rng=rng_
                    ),
                    REPLICATIONS_CHEAP,
                    seed=702,
                ),
                reference,
            ).empirical_cov
        )
        local_oracle = oracle_mean_shift(state, rng=np.random.default_rng(3))
        groups["oracle-IS"].append(
            summarise(
                "oracle-IS",
                replicate(
                    lambda rng_, s=state, t=local_oracle: importance_sampling(
                        s, t, BUDGET, rng=rng_
                    ),
                    REPLICATIONS_CHEAP,
                    seed=703,
                ),
                reference,
            ).empirical_cov
        )

        def surrogate_run(rng_, s=state):
            local = fit_surrogate(s, n_train=N_TRAIN, rng=rng_)
            estimate, _ = surrogate_guided_importance_sampling(
                s, local, BUDGET - N_TRAIN, rng=rng_
            )
            return estimate

        groups["surrogate-IS"].append(
            summarise(
                "surrogate-IS",
                replicate(surrogate_run, REPLICATIONS_SURROGATE, seed=704),
                reference,
            ).empirical_cov
        )

    positions = np.arange(len(labels))
    width = 0.2
    for index, (name, values) in enumerate(groups.items()):
        axes[1].bar(
            positions + (index - 1.5) * width, values, width=width, label=name
        )
        for x, value in zip(positions + (index - 1.5) * width, values, strict=True):
            axes[1].text(x, value * 1.12, f"{value:.4f}", ha="center", fontsize=6.5)
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("measured coefficient of variation")
    axes[1].set_title(
        f"Equal budget of {BUDGET} TRUE evaluations\n"
        f"(the surrogate's {N_TRAIN} training runs are counted against it)"
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y", which="both")
    axes[1].text(
        -0.42,
        0.0055,
        "On the smooth instance analytic-IS, surrogate-IS and oracle-IS use the\n"
        "same tilt to six decimals, so the differences between those three bars\n"
        "are replication noise, not method quality.",
        fontsize=7,
    )

    figure.suptitle(
        "rareverify: the surrogate cannot beat an exact free design point on a "
        "smooth limit state, and does beat the smooth-part one when the "
        "boundary is not smooth",
        fontsize=9,
    )
    figure.tight_layout()
    OUTPUT.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT, dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    main()
