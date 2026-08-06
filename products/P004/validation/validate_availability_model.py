"""Validation 3: availability model vs climatology baseline, with calibration.

Trains the classical baseline (climatological monthly prior) and the ML model
(bagged gradient boosting) on the SAME seeded synthetic training split and
evaluates both on the SAME held-out test split.  Reports Brier score, log
loss, ROC AUC, expected calibration error (ECE, 10 equal-width bins) and
writes a calibration (reliability) diagram to
validation/calibration_curve.png.

All data is synthetic (see DATASET_CARD.md).  The model is not certified for
operational flight use.

Run: python validation/validate_availability_model.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from passplanner import ClimatologyBaselineModel, PassSuccessModel, generate_dataset  # noqa: E402

TRAIN_N = 8000
TEST_N = 4000
TRAIN_SEED = 20260301
TEST_SEED = 20260302
MODEL_SEED = 7
N_BINS = 10


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS):
    """ECE with equal-width bins; returns (ece, bin_centres, bin_freq, bin_pred, counts)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    centres, freq, pred, counts = [], [], [], []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        counts.append(n)
        if n == 0:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            freq.append(np.nan)
            pred.append(np.nan)
            continue
        f = float(y[m].mean())
        q = float(p[m].mean())
        ece += (n / len(y)) * abs(f - q)
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        freq.append(f)
        pred.append(q)
    return ece, np.array(centres), np.array(freq), np.array(pred), np.array(counts)


def main() -> int:
    lines = []
    w = lines.append
    w("PassPlanner validation 3 -- availability model vs climatology baseline")
    w(f"run: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    w("data: SYNTHETIC (passplanner.synthdata); no real meteorological observations")
    w(f"train: n={TRAIN_N} seed={TRAIN_SEED}   test: n={TEST_N} seed={TEST_SEED} "
      f"(disjoint seeds -> independent draws from the same generative process)")
    w("")

    train = generate_dataset(TRAIN_N, seed=TRAIN_SEED)
    test = generate_dataset(TEST_N, seed=TEST_SEED)

    baseline = ClimatologyBaselineModel().fit(train.x, train.y)
    t_a = time.perf_counter()
    model = PassSuccessModel(n_members=5, seed=MODEL_SEED).fit(train.x, train.y)
    train_time = time.perf_counter() - t_a

    p_base = baseline.predict_proba(test.x)
    p_model, sigma = model.predict_with_uncertainty(test.x)

    # A third reference: the exact generative probability (the achievable floor).
    p_oracle = test.p_true

    w(f"training time (5 GBM members, 2 CPU cores): {train_time:.1f} s")
    w("")
    w(f"  {'model':<28} {'Brier':>9} {'log loss':>10} {'ROC AUC':>9} {'ECE':>8}")
    rows = []
    for name, p in (("climatology baseline", p_base),
                    ("ML (bagged GBM)", p_model),
                    ("oracle p_true (floor)", p_oracle)):
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        brier = brier_score_loss(test.y, p)
        ll = log_loss(test.y, pc)
        auc = roc_auc_score(test.y, p)
        ece = expected_calibration_error(test.y, p)[0]
        rows.append((name, brier, ll, auc, ece))
        w(f"  {name:<28} {brier:>9.4f} {ll:>10.4f} {auc:>9.4f} {ece:>8.4f}")
    w("")
    brier_base, brier_model = rows[0][1], rows[1][1]
    improve = 100.0 * (brier_base - brier_model) / brier_base
    w(f"Brier improvement of ML over baseline: {improve:.2f} % "
      f"({brier_base:.4f} -> {brier_model:.4f})")
    w(f"Brier of the oracle (irreducible):     {rows[2][1]:.4f}")
    w("")
    w("uncertainty output (ensemble std of member probabilities):")
    w(f"  mean {sigma.mean():.4f}, median {np.median(sigma):.4f}, "
      f"p95 {np.percentile(sigma, 95):.4f}, max {sigma.max():.4f}")
    corr = float(np.corrcoef(sigma, np.abs(p_model - test.p_true))[0, 1])
    w(f"  correlation(sigma, |p_pred - p_true|) = {corr:.4f} "
      "(positive => larger spread flags larger error)")
    w("")

    # Calibration table + figure.
    ece_m, centres, freq_m, pred_m, counts = expected_calibration_error(test.y, p_model)
    _ece_b, _c, freq_b, pred_b, _cb = expected_calibration_error(test.y, p_base)
    w(f"calibration table, ML model ({N_BINS} equal-width bins):")
    w(f"  {'bin':>12} {'n':>6} {'mean pred':>10} {'observed':>10} {'|diff|':>8}")
    for i in range(N_BINS):
        if counts[i] == 0:
            continue
        w(f"  {centres[i] - 0.05:>5.2f}-{centres[i] + 0.05:<6.2f} {counts[i]:>6d} "
          f"{pred_m[i]:>10.4f} {freq_m[i]:>10.4f} {abs(pred_m[i] - freq_m[i]):>8.4f}")
    w("")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(pred_b[~np.isnan(pred_b)], freq_b[~np.isnan(freq_b)], "o-",
            color="#888888", label=f"climatology baseline (ECE {rows[0][4]:.3f})")
    ax.plot(pred_m[~np.isnan(pred_m)], freq_m[~np.isnan(freq_m)], "s-",
            color="#1f77b4", label=f"ML bagged GBM (ECE {ece_m:.3f})")
    ax.set_xlabel("predicted pass-success probability")
    ax.set_ylabel("observed success frequency")
    ax.set_title("Calibration (reliability) diagram\nheld-out synthetic test set "
                 f"(n = {TEST_N})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.hist(p_model, bins=20, range=(0, 1), color="#1f77b4", alpha=0.75,
            label="ML predictions")
    ax.hist(p_base, bins=20, range=(0, 1), histtype="step", color="#888888", lw=1.5,
            label="baseline predictions")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("count")
    ax.set_title("Prediction distribution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("PassPlanner availability model -- SYNTHETIC DATA, not certified "
                 "for operational flight use", fontsize=9)
    fig.tight_layout()
    out_png = Path(__file__).parent / "calibration_curve.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    w(f"calibration figure written: {out_png.name}")

    ok = brier_model < brier_base and ece_m < 0.05
    w("")
    w(f"acceptance: ML Brier < baseline Brier  -> {brier_model < brier_base}")
    w(f"acceptance: ML ECE < 0.05              -> {ece_m < 0.05} (ECE = {ece_m:.4f})")
    w(f"VERDICT: {'PASS' if ok else 'FAIL'}")
    text = "\n".join(lines)
    print(text)
    (Path(__file__).parent / "validate_availability_model_output.txt").write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
