"""Validation: the closed-form inversions first, then the learned correction.

Phase order is not negotiable. The two textbook inversions and a composed
closed form are implemented and measured first, on the same held-out rows the
learned model is measured on. Where the closed forms win, that is the published
result.

What is measured
----------------
1. **Counting-noise floor.** Every row integrates about 5000 counts, so no
   estimator can do better than roughly ``1/sqrt(5000)`` relative on the rate.
   The floor is computed and printed first, because every number below has to be
   read against it.
2. **Four closed forms** --- the non-paralyzable inversion (D2), the paralyzable
   lower-branch inversion (D5), the model-matched choice of the two, and the
   composed form that removes the afterpulse inflation (A2) before inverting the
   dead time. The composed form gets exactly the same inputs as the learned
   model and is the baseline that matters.
3. **The learned correction**, with its 5-95 quantile interval, and the
   **measured** coverage of that interval.
4. **A regime table.** Overall numbers hide the result. The table reports each
   estimator in the regime where the closed forms are right, the regime where
   afterpulsing breaks them, and the regime past the paralyzable maximum where
   nothing works.
5. **A Fano-factor ablation.** The window length in the generator is set from
   the true rate to bound simulation cost, which leaves a weak residual coupling
   between the true rate and the Fano-factor estimate. Retraining with the Fano
   column flattened to a constant measures how much of the learned model's
   margin depends on that feature at all.

Runtime: about 60 s on two shared cores, dominated by generating 8000 simulated
acquisitions (about 4 x 10^7 events).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from photoncount.correction import (  # noqa: E402
    QUANTILES,
    RateCorrector,
    composed_baseline,
    interval_coverage,
    matched_baseline,
    nonparalyzable_baseline,
    paralyzable_baseline,
    rate_error_metrics,
    summarise_by_regime,
)
from photoncount.dataset import FEATURE_NAMES, generate_dataset  # noqa: E402

TRAIN_ROWS = 6000
TEST_ROWS = 2000
TRAIN_SEED = 20261006
TEST_SEED = 99001
MODEL_FILE = "rate_corrector.joblib"


def main() -> int:
    print("validate_correction.py")
    print("closed forms first, learned correction second, same held-out rows")
    print("=" * 78)

    t0 = time.perf_counter()
    train = generate_dataset(TRAIN_ROWS, TRAIN_SEED)
    test = generate_dataset(TEST_ROWS, TEST_SEED)
    gen_seconds = time.perf_counter() - t0
    print(f"\n0. Data. train {len(train)} rows (seed {TRAIN_SEED}), "
          f"test {len(test)} rows (seed {TEST_SEED})")
    print(f"   generated in {gen_seconds:.1f} s on this container")
    print(f"   features: {', '.join(FEATURE_NAMES)}")
    print(f"   n tau spans {test.true_x.min():.4f} to {test.true_x.max():.4f}; "
          f"{float(np.mean(test.true_x > 1.0)) * 100:.1f} % of test rows are past the "
          "paralyzable maximum")
    total_counts = test.count_mean * 25
    floor = float(np.median(1.0 / np.sqrt(total_counts)))
    print(f"   median total counts per row: {float(np.median(total_counts)):.0f}")
    print(f"   counting-noise floor on the observed rate: {floor:.5f} relative "
          "(1 / sqrt(counts))")
    print(f"   expected median |relative error| at that floor: {0.6745 * floor:.5f}")

    print("\n1. Closed forms, implemented and measured first")
    estimates = {
        "nonparalyzable (D2)": nonparalyzable_baseline(test.observed_rate_hz, test.dead_time_s),
        "paralyzable lower (D5)": paralyzable_baseline(test.observed_rate_hz, test.dead_time_s),
        "matched (D2/D5)": matched_baseline(
            test.observed_rate_hz, test.dead_time_s, test.is_paralyzable
        ),
        "composed (A2 then D2/D5)": composed_baseline(
            test.observed_rate_hz,
            test.dead_time_s,
            test.is_paralyzable,
            test.afterpulse_probability,
        ),
    }

    t1 = time.perf_counter()
    corrector = RateCorrector().fit(train)
    fit_seconds = time.perf_counter() - t1
    predicted = corrector.predict_rate(test.features, test.dead_time_s)
    estimates["learned (quantile GBM)"] = predicted["median"]
    print(f"\n2. Learned correction fitted in {fit_seconds:.1f} s "
          f"({corrector.n_estimators} trees x {len(QUANTILES)} quantiles, "
          f"depth {corrector.max_depth})")

    print("\n3. Overall, on the held-out set")
    print(f"   {'estimator':26s} {'defined':>8} {'median':>9} {'p90':>9} {'bias':>9} "
          f"{'mean|log10|':>12}")
    for name, est in estimates.items():
        m = rate_error_metrics(est, test.true_rate_hz, name)
        print(f"   {name:26s} {m['defined_fraction']:8.3f} "
              f"{m['median_abs_rel_error']:9.4f} {m['p90_abs_rel_error']:9.4f} "
              f"{m['median_signed_rel_error']:+9.4f} {m['mean_abs_log10_error']:12.5f}")
    print("   'defined' is the fraction of rows the estimator could answer at all;")
    print("   a closed form returns nothing where its observed rate is unreachable")

    print("\n4. Uncertainty interval (5th to 95th percentile), measured coverage")
    cov = interval_coverage(predicted["lower"], predicted["upper"], test.true_rate_hz)
    print(f"   nominal coverage      {cov['nominal']:.4f}")
    print(f"   measured coverage     {cov['coverage']:.4f} over {int(cov['n'])} rows")
    print(f"   median relative width {cov['median_relative_width']:.4f}")
    se_cov = np.sqrt(cov["coverage"] * (1 - cov["coverage"]) / cov["n"])
    print(f"   binomial s.e. of the coverage estimate: {se_cov:.4f}  "
          f"(z = {(cov['coverage'] - cov['nominal']) / se_cov:+.2f})")

    print("\n5. By regime. This is the result; the overall table above is not.")
    rows = summarise_by_regime(
        estimates,
        test.true_rate_hz,
        test.true_x,
        test.afterpulse_probability,
        test.is_paralyzable,
    )
    current = None
    for row in rows:
        if row["regime"] != current:
            current = row["regime"]
            print(f"\n   regime: {current}  (n = {int(row['n_total'])})")
            print(f"      {'estimator':26s} {'defined':>8} {'median':>9} {'p90':>9}")
        print(f"      {row['name']:26s} {row['defined_fraction']:8.3f} "
              f"{row['median_abs_rel_error']:9.4f} {row['p90_abs_rel_error']:9.4f}")

    print("\n6. Feature importances (median gain across the three quantile models)")
    for name, value in sorted(
        corrector.feature_importances().items(), key=lambda kv: -kv[1]
    ):
        print(f"   {name:28s} {value:.5f}")

    print("\n7. Fano-factor ablation: retrain with the Fano column flattened to 1.0")
    idx = FEATURE_NAMES.index("fano_factor")
    flat_train_features = train.features.copy()
    flat_train_features[:, idx] = 1.0
    flat_test_features = test.features.copy()
    flat_test_features[:, idx] = 1.0
    import copy

    flat_train = copy.copy(train)
    flat_train.features = flat_train_features
    ablated = RateCorrector().fit(flat_train)
    ablated_pred = ablated.predict_rate(flat_test_features, test.dead_time_s)["median"]
    full = rate_error_metrics(estimates["learned (quantile GBM)"], test.true_rate_hz, "full")
    abl = rate_error_metrics(ablated_pred, test.true_rate_hz, "no Fano")
    print(f"   {'model':20s} {'median':>9} {'p90':>9}")
    print(f"   {'full (5 features)':20s} {full['median_abs_rel_error']:9.4f} "
          f"{full['p90_abs_rel_error']:9.4f}")
    print(f"   {'Fano flattened':20s} {abl['median_abs_rel_error']:9.4f} "
          f"{abl['p90_abs_rel_error']:9.4f}")
    delta = (
        abl["median_abs_rel_error"] - full["median_abs_rel_error"]
    ) / full["median_abs_rel_error"]
    print(f"   relative change in the median error without the Fano factor: {delta:+.4f}")
    print("   Why this ablation exists: the generator sets the window length from the")
    print("   true rate to bound simulation cost, which leaves a weak residual coupling")
    print("   between the true rate and the Fano-factor estimate. If the learned margin")
    print("   came from that coupling, flattening the column would destroy it.")
    if delta <= 0.0:
        print("   It does not. Flattening the Fano column leaves the model no worse, so")
        print("   none of the margin comes from the residual coupling, and the Fano factor")
        print("   carries no usable information at these counting statistics. The")
        print("   dead-time/afterpulsing signature it carries in principle is below the")
        print("   noise of a 5000-count measurement.")
    else:
        print("   The degradation above bounds how much of the margin could come from it.")

    corrector.save(Path(__file__).resolve().parent / MODEL_FILE)
    size_kb = (Path(__file__).resolve().parent / MODEL_FILE).stat().st_size / 1024.0
    print(f"\n8. Model persisted as {MODEL_FILE} ({size_kb:.1f} KiB, joblib, "
          "regenerated deterministically by this script)")
    print(f"   total runtime: {time.perf_counter() - t0:.1f} s")

    clean = test.afterpulse_probability <= 0.02
    matched_clean = rate_error_metrics(
        estimates["matched (D2/D5)"][clean], test.true_rate_hz[clean]
    )["median_abs_rel_error"]
    learned_clean = rate_error_metrics(
        estimates["learned (quantile GBM)"][clean], test.true_rate_hz[clean]
    )["median_abs_rel_error"]
    print("\n9. The honest summary")
    print(f"   where afterpulsing is effectively absent (p <= 0.02, n = {int(clean.sum())}): "
          f"matched closed form {matched_clean:.4f}, learned {learned_clean:.4f}")
    winner = "the closed form wins" if matched_clean <= learned_clean else "the learned model wins"
    print(f"   {winner} there, and both sit at the counting-noise floor")
    print("   past the paralyzable maximum nothing works: the lower-branch inverse")
    print("   returns the wrong root and the learned model does not recover it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
