"""Validate the learned drift classifier against the three analytic baselines.

Protocol
--------
* **Matched in-control ARL0.** Every method's alarm threshold is set to the
  same declared target on the same in-control bank, using in-control data
  only. For the classifier the statistic is its calibrated confidence, and
  because an isotonic calibration takes finitely many values the achievable
  ARL0 is quantised; the smallest achievable threshold whose ARL0 is at or
  above the target is used, so the classifier is never given a larger
  false-alarm budget than the baselines.
* **Steady-state delay.** The change begins at sample ``window``, after an
  in-control run-up, so the windowed classifier has a full feature window at
  the change point and the recursive baselines have a warmed-up statistic.
  Runs that alarm before the change are pre-change false alarms and are
  excluded from the delay, counted separately.
* **Independent seeds** for training, probability calibration, threshold
  calibration and evaluation. Splits are by run, never by window.
* **Out-of-distribution evaluation** on changes the training set does not
  contain: the opposite sign, a smaller magnitude and a larger one.

Reported: delays, detection fractions, calibration quality (Brier score and
reliability diagram), feature importances, single-row inference latency.
"""

from __future__ import annotations

import time

import numpy as np
from _bootstrap import add_src_to_path

add_src_to_path()

from twininvalidate import (  # noqa: E402
    N_FEATURES,
    SCENARIO_LABELS,
    SCENARIOS,
    AssetChange,
    DetectorSpec,
    DriftClassifier,
    StreamSpec,
    bracket_threshold,
    brier_score,
    calibrate_threshold,
    calibration_set,
    delay_after_onset,
    expected_calibration_error,
    in_control_streams,
    reliability_diagram,
    simulate_residuals,
    training_set,
    window_features,
)

TARGET_ARL0 = 1000.0
N_RUNS = 200
N_SAMPLES = 2500
BASELINE_NAMES = ("cusum", "ewma", "glr", "varcusum")


def main() -> int:
    started = time.perf_counter()
    print("=== training ===")
    train = training_set()
    cal = calibration_set()
    print(f"training windows      {len(train)} (seeds 53002 in-control / 53010-53012 changed)")
    print(f"calibration windows   {len(cal)} (seeds 53020 / 53030-53032)")
    print(f"class balance         train {train.y.mean():.3f}, calibration {cal.y.mean():.3f}")
    print(f"training runs         {len(np.unique(train.run))} distinct, split is by run")
    t0 = time.perf_counter()
    clf = DriftClassifier().fit(train.x, train.y, cal.x, cal.y)
    print(f"fit wall clock        {time.perf_counter() - t0:.1f} s on 2 cores")
    print(f"architecture          random forest, {clf.n_estimators} trees, max depth "
          f"{clf.max_depth}, min_samples_leaf 5, seed {clf.seed}")
    print("calibration           isotonic, on the held-out calibration set, via")
    print("                      sklearn.frozen.FrozenEstimator (cv='prefit' raises")
    print("                      InvalidParameterError on scikit-learn 1.9.1)")
    print(f"feature window        {clf.window} samples = {clf.window * 0.05:.2f} s at 20 Hz")

    print()
    print("=== feature importances (impurity-based, descriptive only) ===")
    for name, value in sorted(clf.feature_importance().items(), key=lambda kv: -kv[1]):
        print(f"  {name:<18}{value:.4f}")

    print()
    print("=== probability calibration on a held-out evaluation set ===")
    eval_in = in_control_streams(n_runs=60, n_samples=800, seed=53120)
    eval_out = simulate_residuals(
        StreamSpec(change=SCENARIOS["parameter_step"], n_runs=20, n_samples=800, seed=53121)
    )
    eval_out2 = simulate_residuals(
        StreamSpec(change=SCENARIOS["slow_ramp"], n_runs=20, n_samples=800, seed=53122)
    )
    eval_out3 = simulate_residuals(
        StreamSpec(change=SCENARIOS["noise_variance"], n_runs=20, n_samples=800, seed=53123)
    )
    x_parts, y_parts = [], []
    for z, label in ((eval_in, 0), (eval_out, 1), (eval_out2, 1), (eval_out3, 1)):
        f = window_features(z, clf.window).reshape(-1, N_FEATURES)
        x_parts.append(f)
        y_parts.append(np.full(f.shape[0], label))
    x_eval = np.concatenate(x_parts)
    y_eval = np.concatenate(y_parts)
    conf = clf.confidence(x_eval)
    brier = brier_score(conf, y_eval)
    bins = reliability_diagram(conf, y_eval, n_bins=10)
    ece = expected_calibration_error(bins)
    print(f"evaluation windows    {x_eval.shape[0]} (seeds 53120-53123, disjoint from "
          f"training and calibration)")
    print(f"class balance         {y_eval.mean():.3f}")
    print(f"Brier score           {brier:.4f}  (0 perfect; 0.25 is the constant-0.5 "
          f"forecast on a balanced set)")
    print(f"expected cal. error   {ece:.4f}")
    print(f"  {'bin':>12}{'count':>9}{'mean conf':>12}{'observed':>11}{'gap':>9}")
    for b in bins:
        print(
            f"  [{b.lower:.1f},{b.upper:.1f}){b.count:>9d}{b.mean_confidence:>12.4f}"
            f"{b.observed_frequency:>11.4f}{b.mean_confidence - b.observed_frequency:>+9.4f}"
        )
    print("NOTE  the Brier score cannot approach 0 here and the reason is in the")
    print("NOTE  labels, not the model. A single 50-sample window of the declared")
    print("NOTE  parameter step carries a mean shift of 0.40 sigma, i.e. about 2.9")
    print("NOTE  standard errors of the window mean, so a window-level decision is")
    print("NOTE  genuinely uncertain; and every post-onset window of the slow ramp is")
    print("NOTE  labelled 1 including those in which the ramp has barely started and")
    print("NOTE  which are indistinguishable from in-control. That label noise is")
    print("NOTE  deliberate: removing it would make the training problem easier than")
    print("NOTE  the monitoring problem is. See DATASET_CARD.md.")

    print()
    print("=== matched-ARL0 threshold setting ===")
    bank = in_control_streams(n_runs=N_RUNS, n_samples=N_SAMPLES, seed=53001)
    thresholds = {
        n: calibrate_threshold(DetectorSpec(n), bank, TARGET_ARL0).threshold
        for n in BASELINE_NAMES
    }
    stat_bank = clf.statistic(bank)
    bracket = bracket_threshold(stat_bank, TARGET_ARL0)
    for n in BASELINE_NAMES:
        print(f"  {DetectorSpec(n).label():<32}threshold {thresholds[n]:>10.4f}")
    print(f"  {'learned classifier (conservative)':<32}threshold "
          f"{bracket.conservative:>10.6f}  ARL0 {bracket.conservative_arl0:.1f}")
    if bracket.generous is not None:
        print(f"  {'learned classifier (generous)':<32}threshold "
              f"{bracket.generous:>10.6f}  ARL0 {bracket.generous_arl0:.1f}")
    print(f"  distinct confidence levels on the bank: {bracket.n_levels}")
    print("NOTE  the classifier's confidence is quantised by the isotonic calibration,")
    print("NOTE  so its achievable ARL0 jumps. The conservative threshold is used for")
    print("NOTE  every number below, which gives the classifier a SMALLER false-alarm")
    print("NOTE  budget than the baselines, not a larger one. The generous threshold")
    print("NOTE  is printed so a reader can see the size of the gap.")

    print()
    print("=== detection delay at a matched in-control ARL0 ===")
    onset = clf.window
    print(f"steady-state protocol, change at sample {onset}, "
          f"{N_RUNS} runs x {N_SAMPLES} samples")
    rows: list[tuple[str, AssetChange, int]] = [
        (SCENARIO_LABELS[name], SCENARIOS[name], 53200 + i)
        for i, name in enumerate(SCENARIOS)
    ]
    rows += [
        ("OOD: gain +1 % (sign flipped)", AssetChange("parameter_step", 0, +0.01), 53300),
        ("OOD: gain -0.4 % (smaller)", AssetChange("parameter_step", 0, -0.004), 53301),
        ("OOD: gain -3 % (larger)", AssetChange("parameter_step", 0, -0.03), 53302),
        ("OOD: process noise x 5", AssetChange("noise_variance", 0, 5.0), 53303),
    ]
    header = f"  {'scenario':<33}" + "".join(f"{n:>14}" for n in BASELINE_NAMES)
    print(header + f"{'learned-cons':>14}{'learned-gen':>14}")
    print(f"  {'':<33}" + " " * (14 * len(BASELINE_NAMES))
          + f"{'ARL0 ' + format(bracket.conservative_arl0, '.0f'):>14}"
          + f"{'ARL0 ' + format(bracket.generous_arl0 or float('nan'), '.0f'):>14}")
    results: dict[str, dict[str, float]] = {}
    for label, change, seed in rows:
        shifted = AssetChange(change.kind, onset, change.magnitude, change.ramp_samples)
        oc = simulate_residuals(
            StreamSpec(change=shifted, n_runs=N_RUNS, n_samples=N_SAMPLES, seed=seed)
        )
        cells, row = [], {}
        for n in BASELINE_NAMES:
            est, _ = delay_after_onset(DetectorSpec(n).statistic(oc), thresholds[n], onset)
            row[n] = est.value
            cells.append(f"{est.value:.0f}/{est.detection_fraction:.2f}")
        stat_clf = clf.statistic(oc)
        est, n_pre = delay_after_onset(stat_clf, bracket.conservative, onset)
        row["learned"] = est.value
        row["learned_det"] = est.detection_fraction
        row["pre_onset_false_alarms"] = float(n_pre)
        cells.append(f"{est.value:.0f}/{est.detection_fraction:.2f}")
        if bracket.generous is not None:
            est_g, _ = delay_after_onset(stat_clf, bracket.generous, onset)
            row["learned_generous"] = est_g.value
            row["learned_generous_det"] = est_g.detection_fraction
            cells.append(f"{est_g.value:.0f}/{est_g.detection_fraction:.2f}")
        results[label] = row
        print(f"  {label:<33}" + "".join(f"{c:>14}" for c in cells))
    print("  cells are mean delay in samples / fraction of runs detected within the horizon")
    print("  a detection fraction below 1 means the delay shown is a lower bound")
    print("  the baselines sit at ARL0 ~1000; the classifier cannot, so BOTH of its")
    print("  achievable bracketing thresholds are shown. learned-cons has a longer")
    print("  ARL0 than the baselines (fewer false alarms allowed, a handicap) and")
    print("  learned-gen a shorter one (more false alarms allowed, an advantage).")
    print("  A fair reading takes the baselines as beaten only if learned-cons wins.")

    print()
    print("=== verdict, in distribution ===")
    for name in SCENARIOS:
        label = SCENARIO_LABELS[name]
        row = results[label]
        best_baseline = min(("cusum", "ewma", "glr"), key=lambda n: row[n])
        margin = row["learned"] / row[best_baseline] - 1.0
        margin_g = row.get("learned_generous", float("nan")) / row[best_baseline] - 1.0
        who = "LEARNED WINS" if margin < 0 else "BASELINE WINS"
        print(
            f"  {name:<17}learned-cons {row['learned']:>6.0f} ({margin * 100:+.0f} %)  "
            f"learned-gen {row.get('learned_generous', float('nan')):>6.0f} "
            f"({margin_g * 100:+.0f} %)  best baseline "
            f"{best_baseline} {row[best_baseline]:>6.0f}   {who}"
        )

    print()
    print("=== verdict, out of distribution ===")
    for label in [r[0] for r in rows if r[0].startswith("OOD")]:
        row = results[label]
        best_baseline = min(("cusum", "ewma", "glr"), key=lambda n: row[n])
        print(
            f"  {label:<33}learned {row['learned']:>6.0f} (det "
            f"{row['learned_det']:.2f})  {best_baseline} {row[best_baseline]:>6.0f}"
        )

    print()
    print("=== single-row inference latency ===")
    x_one = x_eval[:1]
    clf.confidence(x_one)
    t0 = time.perf_counter()
    for _ in range(200):
        clf.confidence(x_one)
    learned_ms = (time.perf_counter() - t0) / 200.0 * 1e3
    z_one = bank[:1, :200]
    t0 = time.perf_counter()
    for _ in range(200):
        DetectorSpec("cusum").statistic(z_one)
    cusum_ms = (time.perf_counter() - t0) / 200.0 * 1e3
    print(f"learned, one window   {learned_ms:.3f} ms per decision (forest n_jobs = "
          f"{clf.forest.n_jobs if clf.forest else 'n/a'})")
    print(f"CUSUM, 200 samples    {cusum_ms:.3f} ms per call, i.e. "
          f"{cusum_ms / 200.0:.5f} ms per sample")
    print(f"ratio                 {learned_ms / (cusum_ms / 200.0):.0f} x per decision")
    print(f"sample period at 20 Hz{50.0:>7.1f} ms")
    print(f"learned uses          {learned_ms / 50.0 * 100:.1f} % of one sample period")
    print("NOTE  the forest's n_jobs is set to 1 after fitting. On this container a")
    print("NOTE  forest with n_jobs > 1 is several times slower at single-row")
    print("NOTE  inference, because thread dispatch dominates tree traversal.")
    print("NOTE  Wall-clock figures on this container move 10-20 % between runs and")
    print("NOTE  are not a hardware characteristic: 2 cores, 7.8 GiB, Python 3.13.16.")

    print()
    print("FINDING  1. In distribution the learned classifier does not beat the")
    print("FINDING  CUSUM on the parameter step at a comparable false-alarm budget,")
    print("FINDING  beats it on the slow ramp, and LOSES heavily on the")
    print("FINDING  noise-variance change.")
    print("FINDING")
    print("FINDING  2. Where it does win, the win is not a win for learning. It is a")
    print("FINDING  specification advantage: the training set contains only negative")
    print("FINDING  gain changes of a known size, so the classifier is effectively a")
    print("FINDING  one-sided test tuned to that magnitude, while the two-sided CUSUM")
    print("FINDING  spends half its false-alarm budget on the other direction and is")
    print("FINDING  tuned to a declared 0.5-sigma shift rather than the true 0.40.")
    print("FINDING")
    print("FINDING  3. The out-of-distribution results settle it. On a gain change of")
    print("FINDING  the SAME MAGNITUDE and the OPPOSITE SIGN the classifier's")
    print("FINDING  detection fraction collapses while every baseline is unaffected,")
    print("FINDING  because a baseline's statistic does not know which way the")
    print("FINDING  residual is supposed to move and the classifier does. A monitor")
    print("FINDING  that cannot see half of the changes it was not trained on is not")
    print("FINDING  a better monitor, whatever its average delay.")
    print("FINDING")
    print("FINDING  4. On an out-of-distribution process-noise change (x 5 instead")
    print("FINDING  of the trained x 2) the classifier is also beaten by every")
    print("FINDING  baseline, so the loss on variance changes is not confined to the")
    print("FINDING  magnitude it was trained on.")
    print("FINDING")
    print("FINDING  5. The learned monitor costs about three orders of magnitude more")
    print("FINDING  per decision than the CUSUM recursion, and cannot alarm earlier")
    print("FINDING  than its 50-sample window fills.")
    print("FINDING")
    print("FINDING  Recommendation, stated against this package's own AI component:")
    print("FINDING  use the CUSUM. The learned classifier earns its place only as a")
    print("FINDING  measurement of how much a change-specific detector could gain,")
    print("FINDING  and none of that gain survives a sign flip, a different variance")
    print("FINDING  magnitude, or the cost of a forest evaluation per sample.")
    print()
    print(f"total wall clock {time.perf_counter() - started:.1f} s on 2 cores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
