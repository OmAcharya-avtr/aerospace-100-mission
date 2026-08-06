"""Example: sigma_I^2 vs Rytov variance sweep — simulation, theory, surrogate.

Sweeps Cn^2 at fixed lambda = 1550 nm, L = 2000 m (point-like receiver),
runs the split-step simulator at each point, overlays weak-fluctuation
Rytov theory and the MLP-ensemble surrogate (trained on the committed
campaign dataset) with its uncertainty band.

Saves ../screenshots/sweep_sigma_i2.png. Runtime: ~30 s.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from scintinet import SimParams, Surrogate, rytov_variance, simulate_scintillation  # noqa: E402

LAM = 1.55e-6  # m
LENGTH = 2000.0  # m
POINT_D = 0.002  # m, ~1 grid pixel (point-like receiver)
CN2_SWEEP = np.logspace(-16.2, -14.9, 7)  # m^(-2/3), weak regime


def main() -> None:
    # Simulation sweep (seeded).
    sim_vals = []
    for i, cn2 in enumerate(CN2_SWEEP):
        p = SimParams(
            cn2=float(cn2), wavelength=LAM, path_length=LENGTH,
            grid_size=256, grid_width=0.5, n_screens=8, n_realizations=8,
        )
        sim_vals.append(simulate_scintillation(p, seed=7000 + i).sigma_i2_point)
    sim_vals = np.array(sim_vals)

    # Theory.
    sigma_r2 = np.array([rytov_variance(c, LAM, LENGTH) for c in CN2_SWEEP])

    # Surrogate trained on the committed campaign dataset.
    rows = list(csv.DictReader((HERE.parent / "validation" / "dataset.csv").open()))
    x_train = np.array(
        [
            [float(r["cn2"]), float(r["path_length_m"]),
             float(r["wavelength_m"]), float(r["aperture_d_m"])]
            for r in rows
        ]
    )
    y_train = np.array([float(r["sigma_i2_sim"]) for r in rows])
    surrogate = Surrogate(n_members=5, hidden_layer_sizes=(32, 32), random_state=0)
    surrogate.fit(x_train, y_train)
    x_query = np.column_stack(
        [CN2_SWEEP, np.full_like(CN2_SWEEP, LENGTH),
         np.full_like(CN2_SWEEP, LAM), np.full_like(CN2_SWEEP, POINT_D)]
    )
    mu, sd = surrogate.predict(x_query, return_std=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(sigma_r2, sigma_r2, "k-", label=r"Rytov theory $\sigma_I^2=\sigma_R^2$")
    ax.loglog(sigma_r2, sim_vals, "o", color="tab:blue",
              label="split-step simulation (256$^2$, 8 screens)")
    ax.loglog(sigma_r2, mu, "s--", color="tab:orange", label="MLP surrogate (5-ensemble)")
    ax.fill_between(sigma_r2, np.maximum(mu - 2 * sd, 1e-6), mu + 2 * sd,
                    color="tab:orange", alpha=0.25, label=r"surrogate $\pm 2\sigma$ ensemble")
    ax.set_xlabel(r"Rytov variance $\sigma_R^2$ (plane wave)")
    ax.set_ylabel(r"scintillation index $\sigma_I^2$")
    ax.set_title(
        f"Weak-fluctuation scintillation, $\\lambda$={LAM * 1e9:.0f} nm, L={LENGTH:.0f} m"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = HERE.parent / "screenshots" / "sweep_sigma_i2.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
