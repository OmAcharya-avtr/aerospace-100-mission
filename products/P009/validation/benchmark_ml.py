"""ML-vs-baseline benchmark and error analysis on held-out synthetic data.

Trains the FogCast gradient-boosting model on the 70 % training split of the seeded
synthetic dataset (n=6000, seed=42) and evaluates MAE/RMSE (dB/km) of the ML model,
the Kim baseline, and the Kruse baseline on the 15 % held-out test split, plus the
empirical coverage of the nominal 90 % prediction interval, and a per-regime error
breakdown. Writes benchmark_results.md next to this script.

NOTE: the ground truth is the SYNTHETIC generative process (Kim + perturbations),
so these metrics measure fidelity to that process, not to field measurements.
"""

import time
from pathlib import Path

import numpy as np

from fogcast import (
    FogCastModel,
    generate_dataset,
    kim_attenuation_db_km,
    kruse_attenuation_db_km,
    split_indices,
)

N_SAMPLES = 6000
SEED = 42

REGIMES = [
    ("dense fog (V <= 0.5 km)", 0.05, 0.5),
    ("fog (0.5 < V <= 1 km)", 0.5, 1.0),
    ("haze (1 < V <= 6 km)", 1.0, 6.0),
    ("clear (V > 6 km)", 6.0, 50.0),
]


def mae_rmse(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    err = pred - y
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def main() -> None:
    data = generate_dataset(n_samples=N_SAMPLES, seed=SEED)
    idx_train, idx_val, idx_test = split_indices(N_SAMPLES, seed=SEED)

    t0 = time.perf_counter()
    model = FogCastModel.train_default(n_samples=N_SAMPLES, seed=SEED)
    train_s = time.perf_counter() - t0

    v = data["visibility_km"][idx_test]
    lam = data["wavelength_nm"][idx_test]
    rh = data["rh_percent"][idx_test]
    y = data["attenuation_db_km"][idx_test]

    point, lo, hi = model.predict(v, lam, rh)
    kim = np.asarray(kim_attenuation_db_km(v, lam))
    kruse = np.asarray(kruse_attenuation_db_km(v, lam))
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    width_med = float(np.median(hi - lo))

    lines = []
    lines.append("# FogCast benchmark — ML vs Kim vs Kruse on held-out synthetic test data")
    lines.append("")
    lines.append(f"Dataset: n={N_SAMPLES}, seed={SEED}; split 70/15/15 "
                 f"(train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}).")
    lines.append(f"Training time: {train_s:.1f} s (3x GradientBoostingRegressor, "
                 "300 estimators, depth 3, lr 0.05, 2 CPU cores).")
    lines.append("")
    lines.append("Ground truth = synthetic generative process (Kim + perturbations); "
                 "metrics measure fidelity to that process, NOT to field measurements.")
    lines.append("")
    lines.append("## Overall (test split, dB/km)")
    lines.append("")
    lines.append("| Predictor | MAE (dB/km) | RMSE (dB/km) |")
    lines.append("|---|---|---|")
    for name, pred in [("ML (GBR)", point), ("Kim baseline", kim), ("Kruse baseline", kruse)]:
        mae, rmse = mae_rmse(pred, y)
        lines.append(f"| {name} | {mae:.3f} | {rmse:.3f} |")
    lines.append("")
    lines.append(f"90 % prediction-interval empirical coverage: **{coverage:.3f}** "
                 "(nominal 0.90; tolerance band 0.85-0.95). "
                 f"Median interval width: {width_med:.3f} dB/km.")
    lines.append("")
    lines.append("## Per-regime MAE (dB/km)")
    lines.append("")
    lines.append("| Regime | n | ML | Kim | Kruse |")
    lines.append("|---|---|---|---|---|")
    for label, v_lo, v_hi in REGIMES:
        mask = (v > v_lo) & (v <= v_hi)
        n = int(mask.sum())
        m_ml = float(np.mean(np.abs(point[mask] - y[mask])))
        m_kim = float(np.mean(np.abs(kim[mask] - y[mask])))
        m_kru = float(np.mean(np.abs(kruse[mask] - y[mask])))
        lines.append(f"| {label} | {n} | {m_ml:.3f} | {m_kim:.3f} | {m_kru:.3f} |")
    lines.append("")
    lines.append("Interpretation: the ML model's edge over the Kim baseline comes from "
                 "learning the synthetic humidity effect and averaging the exponent noise; "
                 "the Kruse baseline is worst in fog because its q(V) branch overestimates "
                 "the long-wavelength advantage there (the documented Kim-vs-Kruse "
                 "disagreement).")

    out = Path(__file__).resolve().parent / "benchmark_results.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
