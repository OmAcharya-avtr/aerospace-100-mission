"""Train the fade-probability surrogate and benchmark it honestly.

Steps (all seeded, deterministic):
1. Load data/surrogate_dataset.npz (run scripts/generate_dataset.py first;
   this script generates it automatically if absent).
2. 80/20 train/test split (seed 123).
3. Fit the 5-member GradientBoosting ensemble.
4. Benchmark on held-out Monte Carlo truth vs the analytic lognormal
   scintillation-only baseline, overall and split by jitter regime.
5. Save models/surrogate.joblib and validation/surrogate_benchmark.txt.

Usage: python scripts/train_surrogate.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beamtwin.stats import analytic_fade_probability_lognormal  # noqa: E402
from beamtwin.surrogate import P_FLOOR, FadeSurrogate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED = 123


def _analytic_log10(x_row: np.ndarray) -> float:
    """Analytic scintillation-only baseline in log10 space for a feature row.

    Reconstructs sigma_ln from log10(range) and log10(cn2) at 1550 nm (the
    dataset's fixed wavelength) via the plane-wave Rytov variance.
    """
    range_m = 10.0 ** x_row[0]
    cn2 = 10.0 ** x_row[1]
    margin_db = x_row[4]
    k = 2.0 * math.pi / 1550e-9
    s_r2 = 1.23 * cn2 * k ** (7.0 / 6.0) * range_m ** (11.0 / 6.0)
    sigma_ln = math.sqrt(math.log1p(s_r2))
    p = analytic_fade_probability_lognormal(margin_db, sigma_ln)
    return math.log10(max(p, P_FLOOR))


def main() -> None:
    data_file = ROOT / "data" / "surrogate_dataset.npz"
    if not data_file.exists():
        print("dataset missing; generating (seed 42)...")
        import subprocess

        subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_dataset.py")], check=True)
    payload = np.load(data_file)
    x, y = payload["X"], payload["y"]

    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(x))
    n_test = len(x) // 5
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_te, y_te = x[test_idx], y[test_idx]

    t0 = time.perf_counter()
    surrogate = FadeSurrogate(n_members=5, random_state=7).fit(x_tr, y_tr)
    train_s = time.perf_counter() - t0

    mean_log, std_log = surrogate.predict_log10(x_te)
    baseline_log = np.array([_analytic_log10(row) for row in x_te])

    def mae(pred: np.ndarray, truth: np.ndarray) -> float:
        return float(np.mean(np.abs(pred - truth)))

    def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
        return float(np.sqrt(np.mean((pred - truth) ** 2)))

    jitter = x_te[:, 2]  # jitter_ratio feature
    low_j = jitter < 0.05
    hi_j = jitter >= 0.05

    t0 = time.perf_counter()
    for _ in range(20):
        surrogate.predict_log10(x_te)
    pred_per_s = 20 * len(x_te) / (time.perf_counter() - t0)

    # Coverage of the +/-2-ensemble-std band (honest check, expected < 95 %:
    # ensemble spread is not a calibrated interval).
    cover = float(np.mean(np.abs(mean_log - y_te) <= 2.0 * std_log))

    lines = [
        "Surrogate benchmark — held-out Monte Carlo truth (log10 P_fade, floor 1e-4)",
        f"dataset: {len(x)} scenarios, train {len(x_tr)} / test {len(x_te)} (split seed {SPLIT_SEED})",
        f"training time: {train_s:.1f} s (5-member GradientBoosting ensemble)",
        "",
        f"{'subset':<28}{'n':>6}  {'MAE surr':>9}  {'MAE base':>9}  {'RMSE surr':>10}  {'RMSE base':>10}",
        f"{'all test':<28}{len(x_te):>6}  {mae(mean_log, y_te):>9.3f}  "
        f"{mae(baseline_log, y_te):>9.3f}  {rmse(mean_log, y_te):>10.3f}  "
        f"{rmse(baseline_log, y_te):>10.3f}",
        f"{'low jitter (ratio < 0.05)':<28}{int(low_j.sum()):>6}  "
        f"{mae(mean_log[low_j], y_te[low_j]):>9.3f}  "
        f"{mae(baseline_log[low_j], y_te[low_j]):>9.3f}  "
        f"{rmse(mean_log[low_j], y_te[low_j]):>10.3f}  "
        f"{rmse(baseline_log[low_j], y_te[low_j]):>10.3f}",
        f"{'high jitter (ratio >= 0.05)':<28}{int(hi_j.sum()):>6}  "
        f"{mae(mean_log[hi_j], y_te[hi_j]):>9.3f}  "
        f"{mae(baseline_log[hi_j], y_te[hi_j]):>9.3f}  "
        f"{rmse(mean_log[hi_j], y_te[hi_j]):>10.3f}  "
        f"{rmse(baseline_log[hi_j], y_te[hi_j]):>10.3f}",
        "",
        f"ensemble +/-2 std coverage of truth: {cover:.1%} (spread estimate, not calibrated)",
        f"surrogate throughput: {pred_per_s:,.0f} predictions/s "
        "(vs ~1 Monte Carlo query of 1e5 samples in ~5 ms + Python overhead)",
        "",
        "Baseline = closed-form lognormal, scintillation-only: exact when jitter -> 0,",
        "biased low when jitter is significant. The surrogate's value is the combined",
        "jitter+scintillation regime; in the scintillation-only limit the analytic",
        "baseline is the reference and the surrogate must only match it.",
    ]
    text = "\n".join(lines)
    print(text)
    (ROOT / "models").mkdir(exist_ok=True)
    surrogate.save(ROOT / "models" / "surrogate.joblib")
    (ROOT / "validation").mkdir(exist_ok=True)
    (ROOT / "validation" / "surrogate_benchmark.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nmodel saved to {ROOT / 'models' / 'surrogate.joblib'}")


if __name__ == "__main__":
    main()
