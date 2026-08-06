"""Benchmark: MLP-ensemble surrogate vs Rytov analytic baseline.

Both models are evaluated on the same held-out simulation points from
validation/dataset.csv (75/25 shuffle split, seed 0). Writes
validation/benchmark_results.txt. Honest expectation: with ~40 training
points in-regime, the analytic baseline may win; report whatever is
measured.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from scintinet import Surrogate, rytov_baseline  # noqa: E402

SPLIT_SEED = 0
TEST_FRACTION = 0.25


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    rel = np.abs(pred - truth) / truth
    return {
        "rmse_log10": float(np.sqrt(np.mean((np.log10(pred) - np.log10(truth)) ** 2))),
        "median_rel_err": float(np.median(rel)),
        "max_rel_err": float(np.max(rel)),
    }


def main() -> None:
    rows = list(csv.DictReader((HERE / "dataset.csv").open()))
    x = np.array(
        [
            [float(r["cn2"]), float(r["path_length_m"]),
             float(r["wavelength_m"]), float(r["aperture_d_m"])]
            for r in rows
        ]
    )
    y = np.array([float(r["sigma_i2_sim"]) for r in rows])

    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(y))
    n_test = int(round(TEST_FRACTION * len(y)))
    test_idx, train_idx = order[:n_test], order[n_test:]

    t0 = time.time()
    surrogate = Surrogate(n_members=5, hidden_layer_sizes=(32, 32), random_state=0)
    surrogate.fit(x[train_idx], y[train_idx])
    train_s = time.time() - t0

    pred_s, std_s = surrogate.predict(x[test_idx], return_std=True)
    pred_b = rytov_baseline(x[test_idx])
    m_s = metrics(pred_s, y[test_idx])
    m_b = metrics(pred_b, y[test_idx])

    t0 = time.time()
    for _ in range(20):
        surrogate.predict(x[test_idx])
    pred_ms = (time.time() - t0) / 20 / len(test_idx) * 1e3

    lines = [
        "Surrogate vs Rytov analytic baseline on held-out simulation points",
        f"dataset: {len(y)} rows; train {len(train_idx)}, test {len(test_idx)} "
        f"(shuffle split, seed {SPLIT_SEED})",
        f"surrogate: 5-member MLP ensemble (32,32), lbfgs; fit time {train_s:.1f} s",
        "",
        f"{'model':<18}{'RMSE(log10)':>12}{'median|rel|':>13}{'max|rel|':>11}",
        f"{'MLP surrogate':<18}{m_s['rmse_log10']:>12.4f}"
        f"{m_s['median_rel_err']:>13.4f}{m_s['max_rel_err']:>11.4f}",
        f"{'Rytov baseline':<18}{m_b['rmse_log10']:>12.4f}"
        f"{m_b['median_rel_err']:>13.4f}{m_b['max_rel_err']:>11.4f}",
        "",
        f"winner on RMSE(log10): "
        f"{'surrogate' if m_s['rmse_log10'] < m_b['rmse_log10'] else 'Rytov baseline'}",
        f"mean ensemble std on test set: {float(np.mean(std_s)):.4e} "
        f"(mean prediction {float(np.mean(pred_s)):.4e})",
        f"surrogate prediction cost: {pred_ms:.3f} ms/point "
        "(vs ~1-10 s/point for the split-step simulation)",
        "",
        "Note: the baseline is exact Rytov theory evaluated in its own validity",
        "regime, so the surrogate is not expected to beat it here; the surrogate's",
        "value is speed vs full simulation and extensibility outside analytic",
        "validity (see MODEL_CARD.md).",
    ]
    (HERE / "benchmark_results.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
