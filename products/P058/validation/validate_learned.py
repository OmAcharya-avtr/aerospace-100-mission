"""The learned detector's own numbers: the ones MODEL_CARD.md reports.

Window-level classification metrics on a held-out split, the reliability of the
score the model offers as a confidence output, feature importances, the training
compute, and the determinism of the whole pipeline.

The stream-level comparison against the analytic detectors at equal ARL0 is in
validate_change_types.py and validate_tradeoff.py. That is the comparison that
decides whether the learned detector is useful; this script describes the model.
"""

from __future__ import annotations

import time

import numpy as np
from _harness import run

TRAIN_SEEDS = range(58_001, 58_009)
TEST_SEEDS = range(58_051, 58_057)


def body(report) -> None:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    from telemdrift.benchmark import (
        STANDARD,
        calibrate_learned_threshold,
    )
    from telemdrift.features import FEATURE_NAMES
    from telemdrift.learned import (
        TRAIN_CHANGES,
        LearnedDetector,
        build_training_set,
        score_stream,
        train_learned_detector,
    )
    from telemdrift.streams import stationary

    cfg = STANDARD

    report.section("0. Problem, baseline and split")
    print("  Task        : given a trailing window of 50 standardised telemetry")
    print("                samples, score the proposition 'a change occurred in or")
    print("                just before this window'.")
    print("  Baseline    : the five analytic detectors, implemented and calibrated")
    print("                FIRST. The comparison is at equal measured ARL0 and is in")
    print("                validate_change_types.py section 6.")
    print("  Split       : by SEED, not by row. Training seeds "
          f"{list(TRAIN_SEEDS)};")
    print(f"                test seeds {list(TEST_SEEDS)}. A row split would leak:")
    print("                consecutive windows overlap by 49 of 50 samples, so")
    print("                neighbouring rows are nearly identical and a random row")
    print("                split would put near-copies on both sides. This is the")
    print("                single most important methodological choice here.")
    print("  Labels      : 1 when the window's right edge is in [change, change+50).")
    print(f"  Positives   : from {len(TRAIN_CHANGES)} declared change specs: "
          f"{TRAIN_CHANGES}")
    print("  Negatives   : stationary streams only. Transients are HELD OUT, so the")
    print("                transient experiment is fair to all six detectors.")

    report.section("1. Training set, test set and compute")
    t0 = time.perf_counter()
    train = build_training_set(seeds=TRAIN_SEEDS)
    build_s = time.perf_counter() - t0
    test = build_training_set(seeds=TEST_SEEDS)
    print(f"  training : {train.summary()}")
    print(f"  test     : {test.summary()}")
    print(f"  generation wall clock: {build_s:.2f} s for the training set")
    t0 = time.perf_counter()
    model = train_learned_detector(train, n_estimators=100)
    fit_s = time.perf_counter() - t0
    print("  fit: RandomForestClassifier(n_estimators=100, max_depth=8, "
          "class_weight='balanced_subsample', n_jobs=1, random_state=58000)")
    print(f"  fit wall clock: {fit_s:.1f} s on two cores "
          "(moves 10-40 % between runs)")
    report.check("training completes well inside the three-minute batch budget",
                 fit_s < 180.0, f"{fit_s:.1f} s")
    report.check("the training and test seed sets are disjoint",
                 not (set(TRAIN_SEEDS) & set(TEST_SEEDS)))
    report.check("the model is single-threaded for inference", model.n_jobs == 1)

    report.section("2. Window-level metrics on the held-out seeds")
    p_test = model.predict_proba(test.features)[:, 1]
    p_train = model.predict_proba(train.features)[:, 1]
    rows = []
    for label, y, p in (("train", train.labels, p_train), ("held-out", test.labels, p_test)):
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        base = float(y.mean())
        rows.append((label, auc, ap, brier, base, y.size))
        print(f"  {label:9s} n={y.size:7d}  positive rate {100 * base:5.2f} %  "
              f"ROC-AUC {auc:.4f}  average precision {ap:.4f}  Brier {brier:.5f}")
    print()
    print("  Average precision is the metric to read here, not ROC-AUC: the positive")
    print("  class is under 4 % of windows and ROC-AUC flatters a classifier on an")
    print("  imbalanced problem. The no-skill average precision equals the positive")
    print(f"  rate, {100 * rows[1][4]:.2f} %, so the held-out figure of "
          f"{rows[1][2]:.4f} is")
    print(f"  {rows[1][2] / rows[1][4]:.1f}x no-skill.")
    gap = rows[0][1] - rows[1][1]
    ap_gap = rows[0][2] - rows[1][2]
    print()
    print(f"  train-minus-held-out ROC-AUC gap           : {gap:+.4f}")
    print(f"  train-minus-held-out average-precision gap : {ap_gap:+.4f}")
    report.check("held-out average precision beats the no-skill rate by more than 3x",
                 rows[1][2] / rows[1][4] > 3.0,
                 f"{rows[1][2]:.4f} against no-skill {rows[1][4]:.4f}")
    print()
    print("  THE NEXT CHECK FAILS, AND THE FAILURE IS THE FINDING.")
    print("  The expectation declared before the measurement was a train-to-held-out")
    print("  ROC-AUC gap under 0.05, which is what a model that had learned the")
    print("  change structure rather than the training seeds would show. The measured")
    print("  gap is larger. The model overfits at the window level.")
    report.check("the train-to-held-out ROC-AUC gap is under 0.05 "
                 "(DECLARED EXPECTATION, MEASURED TO FAIL)",
                 abs(gap) < 0.05, f"measured {gap:+.4f}, expected < 0.05")
    print()
    print("  Why, and what follows from it:")
    print("  * The split is by seed, so a held-out stream shares no samples with any")
    print("    training stream. A depth-8 forest with 100 trees over 78000 windows")
    print("    has enough capacity to fit the particular noise realisations of eight")
    print("    seeds, and ROC-AUC on the training rows reflects that.")
    print("  * The honest number is therefore the HELD-OUT one, and it is the only")
    print("    one quoted anywhere outside this paragraph. A reader who sees ROC-AUC")
    print(f"    {rows[0][1]:.3f} quoted for this model is being shown the training")
    print("    figure.")
    print("  * No retuning was performed to close the gap. Reducing depth or trees")
    print("    until the gap closed would produce a prettier number for the same")
    print("    stream-level result, which is already a loss at equal ARL0, and this")
    print("    portfolio's policy is that an honest negative is not retuned away.")
    report.check("the measured overfitting gap is in the documented band 0.08 to 0.16",
                 0.08 < gap < 0.16, f"{gap:+.4f}")
    report.finding(
        f"THE LEARNED MODEL OVERFITS AT THE WINDOW LEVEL: train ROC-AUC "
        f"{rows[0][1]:.3f} against held-out {rows[1][1]:.3f}, a gap of {gap:+.3f} "
        f"against a pre-declared expectation of under 0.05; average precision "
        f"{rows[0][2]:.3f} against {rows[1][2]:.3f}. The split is by seed so this is "
        "not leakage, it is capacity. Not retuned: the stream-level result is already "
        "a loss at equal ARL0 and closing the gap would only change the window-level "
        "number."
    )
    report.finding(
        f"the learned detector's window-level held-out average precision is "
        f"{rows[1][2]:.3f} against a no-skill rate of {rows[1][4]:.3f} "
        f"({rows[1][2] / rows[1][4]:.1f}x), ROC-AUC {rows[1][1]:.3f}. These are "
        "respectable window-level numbers and they do NOT translate into a shorter "
        "detection delay at equal ARL0: see validate_change_types.py section 6."
    )

    report.section("3. Is the score a usable confidence output")
    print("  The model exposes the forest's vote fraction as its confidence output.")
    print("  This section measures whether that number means what it looks like.")
    print()
    print("  bin            n   mean score   observed frequency   gap")
    edges = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    worst_gap = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (p_test >= lo) & (p_test < hi if hi < 1.0 else p_test <= hi)
        if mask.sum() < 30:
            print(f"  [{lo:.1f},{hi:.1f})  {int(mask.sum()):8d}   "
                  "(fewer than 30 windows, not reported)")
            continue
        mean_p = float(p_test[mask].mean())
        obs = float(test.labels[mask].mean())
        gap_b = obs - mean_p
        worst_gap = max(worst_gap, abs(gap_b))
        print(f"  [{lo:.1f},{hi:.1f})  {int(mask.sum()):8d}   {mean_p:10.4f}   "
              f"{obs:18.4f}   {gap_b:+.4f}")
    print()
    print(f"  worst absolute reliability gap over the populated bins: {worst_gap:.4f}")
    print()
    print("  The score is NOT claimed to be calibrated and this is the evidence. A")
    print("  forest vote fraction on a class-weighted, heavily imbalanced problem is")
    print("  an ordering, not a probability. It is used in this package only as a")
    print("  monotone score whose threshold is moved until the measured ARL0 matches")
    print("  a target, which needs no calibration at all. A reader who wants a")
    print("  calibrated probability should recalibrate it; a sibling product in this")
    print("  portfolio (P056 CalibAudit) is the one that does that, and it is not")
    print("  imported here.")
    report.finding(
        f"the learned detector's confidence output is MISCALIBRATED: the worst gap "
        f"between mean predicted score and observed frequency over the populated "
        f"bins is {worst_gap:.3f}. The package uses the score only as a monotone "
        "ordering whose threshold is set by measured ARL0, which does not require "
        "calibration, and the README does not claim a probability."
    )

    report.section("4. Feature importances, and what they say about the model")
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    print("  feature                 importance")
    for i in order:
        print(f"  {FEATURE_NAMES[i]:22s} {imp[i]:10.4f}")
    print()
    top = FEATURE_NAMES[order[0]]
    print(f"  The dominant feature is '{top}'.")
    print("  The two scale features were included as a pair so the model could in")
    print("  principle distinguish a transient (range moves, IQR does not) from a")
    print("  variance step (both move):")
    for name in ("iqr", "range"):
        i = FEATURE_NAMES.index(name)
        print(f"    {name:8s} importance {imp[i]:.4f}")
    print("  validate_transient.py measures whether it actually does. It fires on")
    print("  94 % of transients, so whatever those features carry, it is not enough.")

    report.section("5. Threshold, operating point and determinism")
    learned = calibrate_learned_threshold(model, cfg)
    print(f"  {learned.summary()}")
    print()
    print("  Determinism: the whole pipeline is seeded. Rebuilding the training set,")
    print("  refitting and recalibrating must give the same threshold.")
    train2 = build_training_set(seeds=TRAIN_SEEDS)
    model2 = train_learned_detector(train2, n_estimators=100)
    same_features = np.array_equal(train.features, train2.features)
    same_scores = np.array_equal(
        model.predict_proba(test.features)[:, 1],
        model2.predict_proba(test.features)[:, 1],
    )
    print(f"  training features identical : {same_features}")
    print(f"  held-out scores identical   : {same_scores}")
    report.check("the training set regenerates bit-for-bit", same_features)
    report.check("the fitted model reproduces identical scores", same_scores)

    report.section("6. The batch score path gives identical results to the online one")
    print("  Every learned ARL figure in this repository comes from the batch path,")
    print("  because the online path costs milliseconds per sample. The two must")
    print("  agree exactly or those figures describe a different detector.")
    x = stationary(4_000, 58_960)
    det = LearnedDetector(model, 0.9, cfg.window)
    det.reset()
    online = []
    for v in x:
        det.update(v)
        online.append(det.last_score)
    batch = score_stream(model, x, cfg.window)
    diff = float(np.max(np.abs(np.asarray(online)[cfg.window - 1:]
                               - batch[cfg.window - 1:])))
    print(f"  4000 samples, worst absolute score difference: {diff:.3e}")
    report.check("the batch and online score paths are bit-identical", diff == 0.0,
                 f"worst {diff:.3e}")

    det_a = LearnedDetector(model, 0.5, cfg.window)
    det_a.reset()
    online_alarms = [i for i, v in enumerate(x) if det_a.update(v)]
    batch_alarms = LearnedDetector(model, 0.5, cfg.window).alarms_from_scores(batch)
    print(f"  alarm indices identical: {online_alarms == batch_alarms.tolist()} "
          f"({len(online_alarms)} alarms)")
    report.check("the batch and online alarm sequences are identical",
                 online_alarms == batch_alarms.tolist())

    report.section("7. What this model is not")
    print("  It is not certified for operational flight use.")
    print("  It is trained entirely on synthetic i.i.d. Gaussian streams with")
    print("  injected changes of four declared kinds. It has never seen real")
    print("  spacecraft telemetry, and validate_robustness.py section 1 shows what")
    print("  autocorrelation alone does to the operating point of every detector")
    print("  here, learned or analytic.")
    print("  It loses to EWMA and CUSUM at equal ARL0 on three of the four change")
    print("  types it was trained on, and ties on the fourth.")
    print("  Its confidence output is not calibrated (section 3).")
    print("  Its online cost is four orders of magnitude above the analytic")
    print("  recursions with this API on this container (validate_cost.py s.3).")


if __name__ == "__main__":
    raise SystemExit(run("validate_learned",
                         "telemdrift 0.1.0 - the learned detector's own numbers",
                         body))
