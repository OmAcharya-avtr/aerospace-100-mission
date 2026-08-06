"""Reduced-scale simulation campaign: generate the sigma_I^2 dataset.

Runs the split-step simulator over a small grid of (Cn^2, L, lambda, D)
points in the weak-fluctuation regime and writes validation/dataset.csv.
Deterministic: point i uses seed BASE_SEED + i. Reduced-scale for the
2-CPU / <3-minute compute budget (256^2 grid, 8 screens, 8 realizations);
see DATASET_CARD.md for what a full campaign would require.

Run from the product root or validation/:
    python validation/run_campaign.py
"""

import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from scintinet import SimParams, scintillation_index_weak, simulate_scintillation  # noqa: E402

BASE_SEED = 2026
CN2_VALUES = (1e-16, 3.16e-16, 1e-15)          # m^(-2/3)
LENGTHS = (1000.0, 2000.0, 3000.0)             # m
WAVELENGTHS = (8.5e-7, 1.55e-6)                # m
APERTURES = (0.002, 0.05, 0.1)                 # m; 0.002 ~ 1 pixel (point-like)
GRID_SIZE = 256
GRID_WIDTH = 0.5                                # m
N_SCREENS = 8
N_REALIZATIONS = 8


def main() -> None:
    t0 = time.time()
    rows = []
    idx = 0
    combos = [
        (cn2, ell, lam)
        for cn2 in CN2_VALUES
        for ell in LENGTHS
        for lam in WAVELENGTHS
    ]
    for cn2, ell, lam in combos:
        params = SimParams(
            cn2=cn2, wavelength=lam, path_length=ell,
            aperture_diameters=APERTURES,
            grid_size=GRID_SIZE, grid_width=GRID_WIDTH,
            n_screens=N_SCREENS, n_realizations=N_REALIZATIONS,
        )
        result = simulate_scintillation(params, seed=BASE_SEED + idx)
        for dia in APERTURES:
            rows.append(
                {
                    "cn2": cn2,
                    "path_length_m": ell,
                    "wavelength_m": lam,
                    "aperture_d_m": dia,
                    "sigma_i2_sim": result.sigma_i2_aperture[dia],
                    "sigma_i2_sim_point": result.sigma_i2_point,
                    "sigma_i2_rytov": scintillation_index_weak(
                        cn2, lam, ell, aperture_diameter=dia
                    ),
                    "seed": BASE_SEED + idx,
                }
            )
        idx += 1

    out = HERE / "dataset.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    log = HERE / "campaign_log.txt"
    log.write_text(
        f"scintinet 0.1.0 reduced simulation campaign\n"
        f"grid {GRID_SIZE}^2, width {GRID_WIDTH} m, {N_SCREENS} screens, "
        f"{N_REALIZATIONS} realizations/point\n"
        f"{len(combos)} simulation points x {len(APERTURES)} apertures = "
        f"{len(rows)} dataset rows\n"
        f"base seed {BASE_SEED} (point i uses seed {BASE_SEED}+i)\n"
        f"wall time: {elapsed:.1f} s\n"
    )
    print(f"wrote {out} ({len(rows)} rows) in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
