"""The three correlation models against the generator's own correlation function.

The fitted exponential of equation (23), the per-regime form of equation (24)
and the learned model of equation (35), all plotted against the implied
correlations they were fitted to and against the generator's truth.
Writes ``screenshots/correlation_models.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    EMPIRICAL,
    FITTED,
    LEARNED,
    NEUTRAL,
    bootstrap,
    save,
    style,
)

ROOT = bootstrap()

import matplotlib.pyplot as plt  # noqa: E402

from sitediverse.field import (  # noqa: E402
    RECORD_SPECS,
    REGIME_COUNT,
    generate_record,
    true_correlation_at,
)
from sitediverse.geometry import along_cross_matrices  # noqa: E402
from sitediverse.study import run_study  # noqa: E402


def main() -> int:
    style()
    record = generate_record(RECORD_SPECS["anisotropic"])
    study = run_study(record, "anisotropic", margin_replicates=100, margin_block=64)
    table = study.table
    separation = table.separation_km()
    along, cross = along_cross_matrices(record.sites)

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    train = table.mask(0)
    axes[0].scatter(
        separation[train], table.target[train], s=10, color=NEUTRAL,
        alpha=0.5, label="implied correlation, equation (11), training rows",
    )
    grid = np.linspace(50.0, 7400.0, 400)
    axes[0].plot(
        grid, study.exponential.predict(grid), color=FITTED, linewidth=1.8,
        label=f"fitted exponential, L = {study.exponential.length_km:.0f} km",
    )
    for regime in range(REGIME_COUNT):
        axes[0].plot(
            grid,
            study.per_regime.models[regime].predict(grid),
            color=FITTED,
            linewidth=0.9,
            linestyle=":",
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("great-circle separation, km")
    axes[0].set_ylabel("latent correlation")
    axes[0].set_title("baseline 2: isotropic in separation alone")
    axes[0].legend(loc="upper right")

    truth = np.array(
        [
            float(
                true_correlation_at(
                    record.spec,
                    int(table.regime[row]),
                    along[int(table.site_i[row]), int(table.site_j[row])],
                    cross[int(table.site_i[row]), int(table.site_j[row])],
                )
            )
            for row in range(table.rows)
        ]
    )
    sites_cell = table.mask(1)
    axes[1].plot([-0.1, 1.0], [-0.1, 1.0], color=NEUTRAL, linestyle="--",
                 linewidth=1.0)
    axes[1].scatter(
        truth[sites_cell],
        study.predictions["exponential"][sites_cell],
        s=14, color=FITTED, marker="s", label="fitted exponential",
    )
    axes[1].scatter(
        truth[sites_cell],
        study.predictions["learned"][sites_cell],
        s=14, color=LEARNED, marker="o", label="learned, equation (35)",
    )
    axes[1].scatter(
        truth[sites_cell],
        table.target[sites_cell],
        s=14, color=EMPIRICAL, marker="x",
        label="implied, equation (11)",
    )
    axes[1].set_xlabel("generator's true correlation, equation (4)")
    axes[1].set_ylabel("predicted correlation")
    axes[1].set_title("held-out SITES: against the truth")
    axes[1].legend(loc="upper left")

    path = save(figure, "correlation_models.png")
    print(f"written: {path}")
    print(f"  training rows                  : {int(train.sum())}")
    print(f"  pooled fit rho_0, L            : {study.exponential.nugget:.6f}, "
          f"{study.exponential.length_km:.2f} km")
    for regime, model in enumerate(study.per_regime.models):
        print(f"    regime {regime}: rho_0 {model.nugget:.6f}, L "
              f"{model.length_km:.2f} km, true L_along "
              f"{record.spec.along_km[regime]:.0f}, L_cross "
              f"{record.spec.cross_km[regime]:.0f}")
    for name in ("exponential", "per_regime", "learned"):
        error = float(
            np.mean(np.abs(study.predictions[name][sites_cell] - truth[sites_cell]))
        )
        print(f"  held-out SITES MAE vs truth, {name:12s}: {error:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
